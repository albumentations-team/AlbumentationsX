"""Functional implementations for image mixing operations.

This module provides utility functions for blending and combining images,
such as copy-and-paste operations with masking.
"""

import random
from collections.abc import Sequence
from typing import Any, Literal, TypedDict, cast
from warnings import warn

import cv2
import numpy as np
from typing_extensions import NotRequired

import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.augmentations.crops.transforms import Crop
from albumentations.augmentations.geometric.resize import LongestMaxSize, SmallestMaxSize
from albumentations.core.bbox_utils import BboxProcessor, denormalize_bboxes, normalize_bboxes
from albumentations.core.composition import (
    _BBOX_INSTANCE_ID,
    _KP_INSTANCE_ID,
    Compose,
)
from albumentations.core.keypoints_utils import KeypointsProcessor
from albumentations.core.type_definitions import (
    NUM_BBOXES_COLUMNS_IN_ALBUMENTATIONS,
    NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS,
)


# Type definition for a processed mosaic item
class ProcessedMosaicItem(TypedDict):
    """Preprocessed Mosaic grid item: cell RGB image, optional semantic mask, optional (N,H,W) instance masks, bboxes, keypoints."""

    image: np.ndarray
    mask: NotRequired[np.ndarray | None]
    masks: NotRequired[np.ndarray | None]  # (N, H, W) instance masks aligned with item image
    bboxes: NotRequired[np.ndarray | None]
    keypoints: NotRequired[np.ndarray | None]


def copy_and_paste_blend(
    base_image: np.ndarray,
    overlay_image: np.ndarray,
    overlay_mask: np.ndarray,
    offset: tuple[int, int],
) -> np.ndarray:
    """Copy overlay pixels onto the base image where mask > 0, at the given (y, x) offset.
    Same shape as base_image; overlay and mask must match.

    This function copies pixels from the overlay image to the base image only where
    the mask has non-zero values. The overlay is placed at the specified offset
    from the top-left corner of the base image.

    Args:
        base_image (np.ndarray): The destination image that will be modified.
        overlay_image (np.ndarray): The source image containing pixels to copy.
        overlay_mask (np.ndarray): Binary mask indicating which pixels to copy from the overlay.
            Pixels are copied where mask > 0.
        offset (tuple[int, int]): The (y, x) offset specifying where to place the
            top-left corner of the overlay relative to the base image.

    Returns:
        np.ndarray: The blended image with the overlay applied to the base image.

    """
    y_offset, x_offset = offset

    blended_image = base_image.copy()
    mask_indices = np.where(overlay_mask > 0)
    blended_image[mask_indices[0] + y_offset, mask_indices[1] + x_offset] = overlay_image[
        mask_indices[0],
        mask_indices[1],
    ]
    return blended_image


def _soft_blend_clip_high(base_image: np.ndarray, donor_image: np.ndarray) -> float:
    """Choose the upper clip for soft alpha blends: 255 for uint8 inputs, and for floats either 1.0 or 255.0 based on
    observed max channel values.
    """
    if base_image.dtype == np.uint8:
        return 255.0
    max_val = max(float(np.max(base_image)), float(np.max(donor_image)))
    return 255.0 if max_val > 1.0 + 1e-3 else 1.0


def blend_images_using_alpha(
    base_image: np.ndarray,
    donor_image: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Blend donor pixels onto base image using a float alpha mask, doing a hard copy where alpha == 1 and linear blend elsewhere.

    Args:
        base_image (np.ndarray): Target image (H, W, C).
        donor_image (np.ndarray): Source image, same shape as base_image.
        alpha (np.ndarray): Float mask (H, W) in [0, 1]. 1 = full donor, 0 = full base.

    Returns:
        np.ndarray: Blended image, same shape and dtype as base_image.

    """
    img_dtype = base_image.dtype

    is_hard = np.all((alpha == 0) | (alpha == 1))
    if is_hard:
        result = base_image.copy()
        paste_mask = alpha > 0
        result[paste_mask] = donor_image[paste_mask]
        return result

    alpha_3d = alpha[..., np.newaxis].astype(np.float32)
    blended = donor_image.astype(np.float32) * alpha_3d + base_image.astype(np.float32) * (1.0 - alpha_3d)
    clip_high = _soft_blend_clip_high(base_image, donor_image)
    return np.clip(blended, 0, clip_high).astype(img_dtype)


def create_copy_paste_alpha(
    instance_masks: np.ndarray,
    blend_mode: str,
    blend_sigma: float,
) -> np.ndarray:
    """Create a float alpha mask from the union of selected instance masks, with optional Gaussian blur for soft edges at boundaries.

    Args:
        instance_masks (np.ndarray): (K, H, W) binary masks of instances to paste.
        blend_mode (str): "hard" for binary alpha, "gaussian" for soft edges.
        blend_sigma (float): Sigma for gaussian blur (only used when blend_mode="gaussian").

    Returns:
        np.ndarray: Float alpha mask (H, W) in [0, 1].

    """
    alpha = np.any(instance_masks > 0, axis=0).astype(np.float32)

    if blend_mode == "gaussian" and blend_sigma > 0:
        kernel_size = int(np.ceil(blend_sigma * 6)) | 1
        alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), blend_sigma)
        np.clip(alpha, 0, 1, out=alpha)
        # Drop numerical halo so low-opacity tails outside the true mask do not count as paste/occlusion.
        alpha[alpha < 1e-3] = 0.0

    return alpha


def compute_instance_visibility(
    existing_masks: np.ndarray,
    paste_mask: np.ndarray,
) -> np.ndarray:
    """For each existing instance mask, compute the fraction of original foreground pixels that remain outside the
    pasted binary union for CopyAndPaste survivor logic.

    Args:
        existing_masks (np.ndarray): (N, H, W) binary masks of existing instances.
        paste_mask (np.ndarray): (H, W) binary mask of the pasted instance union (opaque region).

    Returns:
        np.ndarray: (N,) array of visibility ratios in [0, 1]. 1.0 = fully visible, 0.0 = fully occluded.

    """
    opaque_region = paste_mask > 0

    original_areas = np.sum(existing_masks > 0, axis=(1, 2)).astype(np.float64)
    occluded_areas = np.sum((existing_masks > 0) & opaque_region[np.newaxis], axis=(1, 2)).astype(np.float64)

    remaining_areas = original_areas - occluded_areas

    safe_areas = np.where(original_areas > 0, original_areas, 1.0)
    return np.where(original_areas > 0, remaining_areas / safe_areas, 1.0)


def calculate_mosaic_center_point(
    grid_yx: tuple[int, int],
    cell_shape: tuple[int, int],
    target_size: tuple[int, int],
    center_range: tuple[float, float],
    py_random: random.Random,
) -> tuple[int, int]:
    """Compute mosaic crop center by sampling in the valid zone so target_size crop overlaps
    all grid cells. center_range and py_random control proportional sampling.

    Ensures the center point allows a crop of target_size to overlap
    all grid cells, applying randomness based on center_range proportionally
    within the valid region where the center can lie.

    Args:
        grid_yx (tuple[int, int]): The (rows, cols) of the mosaic grid.
        cell_shape (tuple[int, int]): Shape of each cell in the mosaic grid.
        target_size (tuple[int, int]): The final output (height, width).
        center_range (tuple[float, float]): Range [0.0-1.0] for sampling center proportionally
                                            within the valid zone.
        py_random (random.Random): Random state instance.

    Returns:
        tuple[int, int]: The calculated (x, y) center point relative to the
                         top-left of the conceptual large grid.

    """
    rows, cols = grid_yx
    cell_h, cell_w = cell_shape
    target_h, target_w = target_size

    large_grid_h = rows * cell_h
    large_grid_w = cols * cell_w

    # Define valid center range bounds (inclusive)
    # The center must be far enough from edges so the crop window fits
    min_cx = target_w // 2
    max_cx = large_grid_w - (target_w + 1) // 2
    min_cy = target_h // 2
    max_cy = large_grid_h - (target_h + 1) // 2

    # Calculate valid range dimensions (size of the safe zone)
    valid_w = max_cx - min_cx + 1
    valid_h = max_cy - min_cy + 1

    # Sample relative position within the valid range using center_range
    rel_x = py_random.uniform(*center_range)
    rel_y = py_random.uniform(*center_range)

    # Calculate center coordinates by scaling relative position within valid range
    # Add the minimum bound to shift the range start
    center_x = min_cx + int(valid_w * rel_x)
    center_y = min_cy + int(valid_h * rel_y)

    # Ensure the result is strictly within the calculated bounds after int conversion
    # (This clip is mostly a safety measure, shouldn't be needed with correct int conversion)
    center_x = max(min_cx, min(center_x, max_cx))
    center_y = max(min_cy, min(center_y, max_cy))

    return center_x, center_y


def calculate_cell_placements(
    grid_yx: tuple[int, int],
    cell_shape: tuple[int, int],
    target_size: tuple[int, int],
    center_xy: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """Compute cell placements by clipping grid lines to the crop window. Returns list of
    (x_min, y_min, x_max, y_max) per cell on the output canvas.

    Args:
        grid_yx (tuple[int, int]): The (rows, cols) of the mosaic grid.
        cell_shape (tuple[int, int]): Shape of each cell in the mosaic grid.
        target_size (tuple[int, int]): The final output (height, width).
        center_xy (tuple[int, int]): The calculated (x, y) center of the final crop window,
                                        relative to the top-left of the conceptual large grid.

    Returns:
        list[tuple[int, int, int, int]]:
            A list containing placement coordinates `(x_min, y_min, x_max, y_max)`
            for each resulting cell part on the final output canvas.

    """
    rows, cols = grid_yx
    cell_h, cell_w = cell_shape
    target_h, target_w = target_size
    center_x, center_y = center_xy

    # 1. Generate grid line coordinates using arange for the large grid
    y_coords_large = np.arange(rows + 1) * cell_h
    x_coords_large = np.arange(cols + 1) * cell_w

    # 2. Calculate Crop Window boundaries
    crop_x_min = center_x - target_w // 2
    crop_y_min = center_y - target_h // 2
    crop_x_max = crop_x_min + target_w
    crop_y_max = crop_y_min + target_h

    def _clip_coords(coords: np.ndarray, min_val: int, max_val: int) -> np.ndarray:
        clipped_coords = np.clip(coords, min_val, max_val)
        # Subtract min_val to convert absolute clipped coordinates
        # into coordinates relative to the crop window's origin (min_val becomes 0).
        return np.unique(clipped_coords) - min_val

    y_coords_clipped = _clip_coords(y_coords_large, crop_y_min, crop_y_max)
    x_coords_clipped = _clip_coords(x_coords_large, crop_x_min, crop_x_max)

    # 4. Form all cell coordinates efficiently
    num_x_intervals = len(x_coords_clipped) - 1
    num_y_intervals = len(y_coords_clipped) - 1
    result = []

    for y_idx in range(num_y_intervals):
        y_min = y_coords_clipped[y_idx]
        y_max = y_coords_clipped[y_idx + 1]
        for x_idx in range(num_x_intervals):
            x_min = x_coords_clipped[x_idx]
            x_max = x_coords_clipped[x_idx + 1]
            result.append((int(x_min), int(y_min), int(x_max), int(y_max)))

    return result


def _check_data_compatibility(
    primary_data: np.ndarray | None,
    item_data: np.ndarray | None,
    data_key: Literal["image", "mask"],
) -> tuple[bool, str | None]:  # Returns (is_compatible, error_message)
    """Check if item_data dimensions and channels match primary_data. Returns (ok, error_msg);
    used to validate mosaic/mixup additional items.
    """
    # 1. Check if item has the required data (image is always required)
    if item_data is None:
        if data_key == "image":
            return False, "Item is missing required key 'image'"
        # Mask is optional, missing is compatible
        return True, None

    # 2. If item data exists, check against primary data (if primary data exists)
    if primary_data is None:  # No primary data to compare against
        return True, None

    # Both primary and item data exist, compare them
    primary_ndim = primary_data.ndim
    item_ndim = item_data.ndim

    # Special handling for masks: allow 2D masks from metadata to be compatible with 3D primary masks with 1 channel
    # Primary is always 3D after Compose preprocessing, but metadata items might have 2D masks
    if data_key == "mask" and item_ndim == 2 and primary_data.shape[-1] == 1:
        return True, None

    if primary_ndim != item_ndim:
        return False, (
            f"Item '{data_key}' has {item_ndim} dimensions, but primary has {primary_ndim}. "
            f"Primary shape: {primary_data.shape}, Item shape: {item_data.shape}"
        )

    primary_channels = primary_data.shape[-1]
    item_channels = item_data.shape[-1]
    if primary_channels != item_channels:
        return False, (
            f"Item '{data_key}' has {item_channels} channels, but primary has {primary_channels}. "
            f"Primary shape: {primary_data.shape}, Item shape: {item_data.shape}"
        )

    # Dimensions match (either both 2D or both 3D with same channels)
    return True, None


def filter_valid_metadata(
    metadata_input: Sequence[dict[str, Any]] | None,
    metadata_key_name: str,
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Filter metadata dicts to those compatible with primary data (image/mask dimensions and
    channels). Uses _check_data_compatibility; warns and skips invalid items.
    """
    if not isinstance(metadata_input, Sequence):
        warn(
            f"Metadata under key '{metadata_key_name}' is not a Sequence (e.g., list or tuple). "
            f"Returning empty list for additional items.",
            UserWarning,
            stacklevel=3,
        )
        return []

    valid_items = []
    primary_image = data.get("image")
    primary_mask = data.get("mask")

    for i, item in enumerate(metadata_input):
        if not isinstance(item, dict):
            warn(
                f"Item at index {i} in '{metadata_key_name}' is not a dict and will be skipped.",
                UserWarning,
                stacklevel=4,
            )
            continue

        item_is_valid = True  # Assume valid initially
        for target_key, primary_target_data in [
            ("image", primary_image),
            ("mask", primary_mask),
        ]:
            item_target_data = item.get(target_key)

            is_compatible, error_msg = _check_data_compatibility(
                primary_target_data,
                item_target_data,
                cast("Literal['image', 'mask']", target_key),
            )

            if not is_compatible:
                msg = (
                    f"Item at index {i} in '{metadata_key_name}' skipped due "
                    f"to incompatibility in '{target_key}': {error_msg}"
                )
                warn(msg, UserWarning, stacklevel=4)
                item_is_valid = False
                break  # Stop checking other targets for this item

        if item_is_valid:
            valid_items.append(item)

    return valid_items


def assign_items_to_grid_cells(
    num_items: int,
    cell_placements: list[tuple[int, int, int, int]],
    py_random: random.Random,
) -> dict[tuple[int, int, int, int], int]:
    """Assign primary (index 0) to largest-area placement; remaining items randomly to others.
    Returns mapping from (x1,y1,x2,y2) to item index.

    Assigns the primary item (index 0) to the placement with the largest area,
    and assigns the remaining items (indices 1 to num_items-1) randomly to the
    remaining placements.

    Args:
        num_items (int): The total number of items to assign (primary + additional + replicas).
        cell_placements (list[tuple[int, int, int, int]]): List of placement
                                coords (x1, y1, x2, y2) for cells to be filled.
        py_random (random.Random): Random state instance.

    Returns:
        dict[tuple[int, int, int, int], int]: Dict mapping placement coords (x1, y1, x2, y2)
                                            to assigned item index.

    """
    if not cell_placements:
        return {}

    # Find the placement tuple with the largest area for primary assignment
    primary_placement = max(
        cell_placements,
        key=lambda coords: (coords[2] - coords[0]) * (coords[3] - coords[1]),
    )

    placement_to_item_index: dict[tuple[int, int, int, int], int] = {
        primary_placement: 0,
    }

    # Use list comprehension for potentially better performance
    remaining_placements = [coords for coords in cell_placements if coords != primary_placement]

    # Indices for additional/replicated items start from 1
    remaining_item_indices = list(range(1, num_items))
    py_random.shuffle(remaining_item_indices)

    num_to_assign = min(len(remaining_placements), len(remaining_item_indices))
    for i in range(num_to_assign):
        placement_to_item_index[remaining_placements[i]] = remaining_item_indices[i]

    return placement_to_item_index


def _ensure_mosaic_bbox_binding_fields(item: dict[str, Any], required_labels: Sequence[str]) -> None:
    if not required_labels:
        return
    arr = item["bboxes"]
    n_rows = len(arr) if isinstance(arr, np.ndarray) else len(np.asarray(arr))
    if _BBOX_INSTANCE_ID in required_labels and _BBOX_INSTANCE_ID not in item:
        item[_BBOX_INSTANCE_ID] = list(range(n_rows))
    for field in required_labels:
        if field.startswith("_ibl_bbox_") and field not in item:
            user_key = field.removeprefix("_ibl_bbox_")
            if user_key in item:
                item[field] = item[user_key]


def _ensure_mosaic_keypoint_binding_fields(item: dict[str, Any], required_labels: Sequence[str]) -> None:
    if not required_labels:
        return
    kps = np.asarray(item["keypoints"])
    n_kp = kps.shape[0]
    if _KP_INSTANCE_ID in required_labels and _KP_INSTANCE_ID not in item:
        item[_KP_INSTANCE_ID] = [0] * n_kp
    for field in required_labels:
        if field.startswith("_ibl_kp_") and field not in item:
            user_key = field.removeprefix("_ibl_kp_")
            if user_key in item:
                item[field] = item[user_key]


def _validate_mosaic_item_label_fields(
    item: dict[str, Any],
    required_labels: Sequence[str],
    data_key: str,
    params_cls_name: str,
) -> None:
    missing = [field for field in required_labels if field not in item]
    if not missing:
        return
    raise ValueError(
        f"Item contains '{data_key}' but is missing required label fields: {missing}. "
        f"Ensure all label fields declared in {params_cls_name} ({required_labels}) are present "
        f"in the item dictionary when '{data_key}' is present.",
    )


def _preprocess_item_annotations(
    item: dict[str, Any],
    processor: BboxProcessor | KeypointsProcessor | None,
    data_key: Literal["bboxes", "keypoints"],
) -> np.ndarray | None:
    """Preprocess bboxes or keypoints for one item with given processor. Returns processed
    array or original if no processor; validates label fields.
    """
    original_data = item.get(data_key)

    if not (processor and data_key in item and item.get(data_key) is not None):
        return original_data

    required_labels = processor.params.label_fields or []
    if data_key == "bboxes":
        _ensure_mosaic_bbox_binding_fields(item, required_labels)
    else:
        _ensure_mosaic_keypoint_binding_fields(item, required_labels)

    _validate_mosaic_item_label_fields(item, required_labels, data_key, type(processor.params).__name__)

    temp_data: dict[str, Any] = {"image": item["image"], data_key: item[data_key]}
    for field in required_labels:
        if field in item:
            temp_data[field] = item[field]

    processor.preprocess(temp_data)
    return temp_data.get(data_key)


def preprocess_copy_paste_annotations(
    item: dict[str, Any],
    processor: BboxProcessor | KeypointsProcessor | None,
    data_key: Literal["bboxes", "keypoints"],
) -> np.ndarray | None:
    """Preprocess bboxes or keypoints for a single donor item. Delegates to internal
    annotation preprocessing with proper processor label encoding.
    """
    return _preprocess_item_annotations(item, processor, data_key)


def unpack_label_wrappers(item: dict[str, Any]) -> dict[str, Any]:
    """Unpack `bbox_labels` and `keypoint_labels` wrapper dicts into top-level label fields so processors can find them directly.

    Both Mosaic and CopyAndPaste store per-item label values under `bbox_labels` and
    `keypoint_labels` (dicts mapping label-field-name → value). This helper flattens
    them so that `_preprocess_item_annotations` can find the fields at the top level.

    Raises:
        TypeError: If `bbox_labels` or `keypoint_labels` is present but is not a dict
            (e.g. a bare list of labels). Both wrapper keys are reserved by the
            Mosaic/CopyAndPaste metadata format and must map label-field name to its
            list of values, e.g. `{"class_id": [3, 7]}` — not `[3, 7]` directly.

    """
    if "bbox_labels" not in item and "keypoint_labels" not in item:
        return item
    unpacked = {k: v for k, v in item.items() if k not in ("bbox_labels", "keypoint_labels")}
    for wrapper_key in ("bbox_labels", "keypoint_labels"):
        if wrapper_key not in item:
            continue
        labels = item[wrapper_key]
        if labels is None:
            continue
        if not isinstance(labels, dict):
            raise TypeError(
                f"Mosaic/CopyAndPaste metadata: `{wrapper_key}` must be a dict mapping "
                f"label-field name to its values (e.g. `{{'class_id': [3, 7]}}`), got "
                f"{type(labels).__name__}. The keys `bbox_labels` and `keypoint_labels` "
                "are reserved wrapper keys in per-item metadata; if you intended to pass "
                f"a bare list of labels, wrap it in a dict keyed by the label_field name "
                f"declared in BboxParams/KeypointParams.label_fields, e.g. "
                f"`{wrapper_key}={{'<your_label_field>': <values>}}`.",
            )
        unpacked.update(labels)
    return unpacked


def preprocess_selected_mosaic_items(
    selected_raw_items: list[dict[str, Any]],
    bbox_processor: BboxProcessor | None,  # Allow None
    keypoint_processor: KeypointsProcessor | None,  # Allow None
) -> list[ProcessedMosaicItem]:
    """Preprocess bboxes and keypoints per item via processors; update encoders. Returns list
    of ProcessedMosaicItem (image, mask, preprocessed bboxes/keypoints).

    Iterates through items, preprocesses annotations individually using processors
    (updating label encoders), and returns a list of dicts with original image/mask
    and the corresponding preprocessed bboxes/keypoints.
    """
    if not selected_raw_items:
        return []

    result_data_items: list[ProcessedMosaicItem] = []

    for item in selected_raw_items:
        flat_item = unpack_label_wrappers(item)
        processed_bboxes = _preprocess_item_annotations(flat_item, bbox_processor, "bboxes")
        processed_keypoints = _preprocess_item_annotations(flat_item, keypoint_processor, "keypoints")

        # Construct the final processed item dict
        processed_item_dict: ProcessedMosaicItem = {
            "image": item["image"],
            "mask": item.get("mask"),
            "bboxes": processed_bboxes,  # Already np.ndarray or None
            "keypoints": processed_keypoints,  # Already np.ndarray or None
        }
        inst_masks = flat_item.get("masks")
        if inst_masks is not None:
            processed_item_dict["masks"] = np.copy(np.asarray(inst_masks))
        result_data_items.append(processed_item_dict)

    return result_data_items


def get_opposite_crop_coords(
    cell_size: tuple[int, int],
    crop_size: tuple[int, int],
    cell_position: Literal["top_left", "top_right", "center", "bottom_left", "bottom_right"],
) -> tuple[int, int, int, int]:
    """Compute (x_min, y_min, x_max, y_max) for crop of crop_size in cell_size, opposite
    cell_position (e.g. top_left → bottom-right). Raises if crop larger than cell.

    Given a cell of `cell_size`, this function determines the top-left (x_min, y_min)
    and bottom-right (x_max, y_max) coordinates for a crop of `crop_size`, such
    that the crop is located in the corner or center opposite to `cell_position`.

    For example, if `cell_position` is "top_left", the crop coordinates will
    correspond to the bottom-right region of the cell.

    Args:
        cell_size (tuple[int, int]): The (height, width) of the cell from which to crop.
        crop_size (tuple[int, int]): The (height, width) of the desired crop.
        cell_position (Literal['top_left', 'top_right', 'center', 'bottom_left', 'bottom_right']): The reference
            position within the cell. The crop will be taken from the opposite position.

    Returns:
        tuple[int, int, int, int]: (x_min, y_min, x_max, y_max) representing the crop coordinates.

    Raises:
        ValueError: If crop_size is larger than cell_size in either dimension.

    """
    cell_h, cell_w = cell_size
    crop_h, crop_w = crop_size

    if crop_h > cell_h or crop_w > cell_w:
        raise ValueError(f"Crop size {crop_size} cannot be larger than cell size {cell_size}")

    # Determine top-left corner (x_min, y_min) based on the OPPOSITE position
    if cell_position == "top_left":  # Crop from bottom_right
        x_min = cell_w - crop_w
        y_min = cell_h - crop_h
    elif cell_position == "top_right":  # Crop from bottom_left
        x_min = 0
        y_min = cell_h - crop_h
    elif cell_position == "bottom_left":  # Crop from top_right
        x_min = cell_w - crop_w
        y_min = 0
    elif cell_position == "bottom_right":  # Crop from top_left
        x_min = 0
        y_min = 0
    elif cell_position == "center":  # Crop from center
        x_min = (cell_w - crop_w) // 2
        y_min = (cell_h - crop_h) // 2
    else:
        # Should be unreachable due to Literal type hint, but good practice
        raise ValueError(f"Invalid cell_position: {cell_position}")

    # Calculate bottom-right corner
    x_max = x_min + crop_w
    y_max = y_min + crop_h

    return x_min, y_min, x_max, y_max


def _mosaic_cell_geometry_compose(
    cell_shape: tuple[int, int],
    target_shape: tuple[int, int],
    fill: float | tuple[float, ...],
    fill_mask: float | tuple[float, ...],
    fit_mode: Literal["cover", "contain"],
    interpolation: int,
    mask_interpolation: int,
    cell_position: Literal["top_left", "top_right", "center", "bottom_left", "bottom_right"],
    *,
    with_bbox_params: bool,
    with_keypoint_params: bool,
) -> Compose:
    """Construct Albumentations Compose per Mosaic grid cell so RGB, mask, and stacked instance masks share identical resize/crop."""
    compose_kwargs: dict[str, Any] = {"p": 1.0}
    if with_bbox_params:
        compose_kwargs["bbox_params"] = {"coord_format": "albumentations"}
    if with_keypoint_params:
        compose_kwargs["keypoint_params"] = {"coord_format": "xy"}

    crop_coords = get_opposite_crop_coords(cell_shape, target_shape, cell_position)

    if fit_mode == "cover":
        return Compose(
            [
                SmallestMaxSize(
                    max_size_hw=cell_shape,
                    interpolation=interpolation,
                    mask_interpolation=mask_interpolation,
                    p=1.0,
                ),
                Crop(
                    x_min=crop_coords[0],
                    y_min=crop_coords[1],
                    x_max=crop_coords[2],
                    y_max=crop_coords[3],
                ),
            ],
            **compose_kwargs,
        )
    if fit_mode == "contain":
        return Compose(
            [
                LongestMaxSize(
                    max_size_hw=cell_shape,
                    interpolation=interpolation,
                    mask_interpolation=mask_interpolation,
                    p=1.0,
                ),
                Crop(
                    x_min=crop_coords[0],
                    y_min=crop_coords[1],
                    x_max=crop_coords[2],
                    y_max=crop_coords[3],
                    pad_if_needed=True,
                    fill=fill,
                    fill_mask=fill_mask,
                    p=1.0,
                ),
            ],
            **compose_kwargs,
        )
    raise ValueError(f"Invalid fit_mode: {fit_mode}. Must be 'cover' or 'contain'.")


def process_cell_geometry(
    cell_shape: tuple[int, int],
    item: ProcessedMosaicItem,
    target_shape: tuple[int, int],
    fill: float | tuple[float, ...],
    fill_mask: float | tuple[float, ...],
    fit_mode: Literal["cover", "contain"],
    interpolation: int,
    mask_interpolation: int,
    cell_position: Literal["top_left", "top_right", "center", "bottom_left", "bottom_right"],
) -> ProcessedMosaicItem:
    """Pad and/or crop one item to target_shape. PadIfNeeded and Crop with fit_mode and
    cell_position; returns ProcessedMosaicItem (image, mask, bboxes, keypoints).

    Uses a Compose pipeline with PadIfNeeded and Crop to ensure the output
    matches the target cell dimensions exactly, handling both padding and cropping cases.

    Args:
        cell_shape (tuple[int, int]): Shape of the cell.
        item (ProcessedMosaicItem): The preprocessed mosaic item dictionary.
        target_shape (tuple[int, int]): Target shape of the cell.
        fill (float | tuple[float, ...]): Fill value for image padding.
        fill_mask (float | tuple[float, ...]): Fill value for mask padding.
        fit_mode (Literal['cover', 'contain']): Fit mode for the mosaic.
        interpolation (int): Interpolation method for image.
        mask_interpolation (int): Interpolation method for mask.
        cell_position (Literal['top_left', 'top_right', 'center', 'bottom_left', 'bottom_right']): Position
        of the cell.

    Returns: (ProcessedMosaicItem): Dictionary containing the geometrically processed image,
        mask, bboxes, and keypoints, fitting the target dimensions.

    """
    geom_pipeline = _mosaic_cell_geometry_compose(
        cell_shape,
        target_shape,
        fill,
        fill_mask,
        fit_mode,
        interpolation,
        mask_interpolation,
        cell_position,
        with_bbox_params=item.get("bboxes") is not None,
        with_keypoint_params=item.get("keypoints") is not None,
    )

    # Prepare input data for the pipeline
    geom_input: dict[str, Any] = {"image": item["image"]}
    item_mask = item.get("mask")
    if item_mask is not None:
        geom_input["mask"] = item_mask
    item_bboxes = item.get("bboxes")
    if item_bboxes is not None:
        geom_input["bboxes"] = item_bboxes
    item_keypoints = item.get("keypoints")
    if item_keypoints is not None:
        geom_input["keypoints"] = item_keypoints

    # Apply the pipeline (`force_apply` explicit so **geom_input cannot bind to it under mypy)
    processed_item = geom_pipeline(force_apply=False, **geom_input)

    result: ProcessedMosaicItem = {
        "image": processed_item["image"],
        "mask": processed_item.get("mask"),
        "bboxes": processed_item.get("bboxes"),
        "keypoints": processed_item.get("keypoints"),
    }

    raw_masks = item.get("masks")
    if raw_masks is not None and isinstance(raw_masks, np.ndarray) and raw_masks.size > 0:
        m = raw_masks
        if m.ndim == 4 and m.shape[-1] == 1:
            m = np.squeeze(m, axis=-1)
        if m.ndim == 3 and m.shape[0] > 0:
            geom_masks_only = _mosaic_cell_geometry_compose(
                cell_shape,
                target_shape,
                fill,
                fill_mask,
                fit_mode,
                interpolation,
                mask_interpolation,
                cell_position,
                with_bbox_params=False,
                with_keypoint_params=False,
            )
            ref_dtype = item["image"].dtype
            planes: list[np.ndarray] = []
            for i in range(m.shape[0]):
                layer = m[i]
                img3 = np.stack([layer, layer, layer], axis=-1)
                if img3.dtype != ref_dtype:
                    img3 = img3.astype(ref_dtype, copy=False)
                out_img = geom_masks_only(force_apply=False, image=img3)["image"]
                planes.append(out_img[..., 0])
            result["masks"] = np.stack(planes, axis=0)

    return result


def shift_cell_coordinates(
    processed_item_geom: ProcessedMosaicItem,
    placement_coords: tuple[int, int, int, int],
) -> ProcessedMosaicItem:
    """Shift bbox and keypoint coords by placement offset onto final canvas. Returns
    ProcessedMosaicItem with image, mask, shifted bboxes and keypoints.

    Args:
        processed_item_geom (ProcessedMosaicItem): The output from process_cell_geometry.
        placement_coords (tuple[int, int, int, int]): The (x1, y1, x2, y2) placement on the final canvas.

    Returns: (ProcessedMosaicItem): A dictionary with keys 'bboxes' and 'keypoints', containing the shifted
        numpy arrays (potentially empty).

    """
    tgt_x1, tgt_y1, _, _ = placement_coords

    shifted_bboxes = None
    shifted_keypoints = None

    bboxes_geom = processed_item_geom.get("bboxes")
    if bboxes_geom is not None and np.asarray(bboxes_geom).size > 0:
        bboxes_geom_arr = np.asarray(bboxes_geom)  # Ensure it's an array
        bbox_shift_vector = np.array([tgt_x1, tgt_y1, tgt_x1, tgt_y1], dtype=np.int32)
        shifted_bboxes = fgeometric.shift_bboxes(bboxes_geom_arr, bbox_shift_vector)

    keypoints_geom = processed_item_geom.get("keypoints")
    if keypoints_geom is not None and np.asarray(keypoints_geom).size > 0:
        keypoints_geom_arr = np.asarray(keypoints_geom)  # Ensure it's an array
        kp_shift_vector = np.array([tgt_x1, tgt_y1, 0], dtype=keypoints_geom_arr.dtype)
        shifted_keypoints = fgeometric.shift_keypoints(keypoints_geom_arr, kp_shift_vector)

    return {
        "bboxes": shifted_bboxes,
        "keypoints": shifted_keypoints,
        "image": processed_item_geom["image"],
        "mask": processed_item_geom.get("mask"),
    }


def assemble_mosaic_from_processed_cells(
    processed_cells: dict[tuple[int, int, int, int], dict[str, Any]],
    target_shape: tuple[int, ...],  # Use full canvas shape (H, W) or (H, W, C)
    dtype: np.dtype,
    data_key: Literal["image", "mask"],
    fill: float | tuple[float, ...] | None,  # Value for image fill or mask fill
) -> np.ndarray:
    """Build mosaic: fill canvas with fill, paste each cell segment at its placement.
    data_key 'image' or 'mask'; handles multi-channel masks. Returns canvas array.

    Initializes the canvas with the fill value and overwrites with processed segments.
    Handles potentially multi-channel masks.
    Addresses potential broadcasting errors if mask segments have unexpected dimensions.
    Assumes input data is valid and correctly sized.

    Args:
        processed_cells (dict[tuple[int, int, int, int], dict[str, Any]]): Dictionary mapping
            placement coords to processed cell data.
        target_shape (tuple[int, ...]): The target shape of the output canvas (e.g., (H, W) or (H, W, C)).
        dtype (np.dtype): NumPy dtype for the canvas.
        data_key (Literal['image', 'mask']): Specifies whether to assemble 'image' or 'mask'.
        fill (float | tuple[float, ...] | None): Value used to initialize the canvas (image fill or mask fill).
              Should be a float/int or a tuple matching the number of channels.
              If None, defaults to 0.

    Returns:
        np.ndarray: The assembled mosaic canvas.

    """
    # Use 0 as default fill if None is provided
    actual_fill = fill if fill is not None else 0

    # Convert fill to numpy array to handle broadcasting in np.full
    fill_value = np.array(actual_fill, dtype=dtype)
    # Initialize canvas with the fill value.
    # If fill_value shape is incompatible with target_shape, np.full will raise ValueError.
    canvas = np.full(target_shape, fill_value=fill_value, dtype=dtype)

    # Iterate and paste segments onto the pre-filled canvas
    for placement_coords, cell_data in processed_cells.items():
        segment = cell_data.get(data_key)

        # If segment exists, paste it over the filled background
        if segment is not None:
            tgt_x1, tgt_y1, tgt_x2, tgt_y2 = placement_coords

            # Handle dimension mismatch for masks:
            # If canvas is 3D but segment is 2D, expand segment
            if data_key == "mask" and len(target_shape) == 3 and segment.ndim == 2:
                segment = np.expand_dims(segment, axis=-1)

            canvas[tgt_y1:tgt_y2, tgt_x1:tgt_x2] = segment

    return canvas


def assemble_mosaic_instance_masks_stack(
    processed_cells: dict[tuple[int, int, int, int], dict[str, Any]],
    canvas_hw: tuple[int, int],
    dtype: np.dtype,
    fill: float | tuple[float, ...] | None,
) -> np.ndarray:
    """One full-canvas mask per instance; iteration order over `processed_cells` matches
    `Mosaic.apply_to_bboxes` / `apply_to_masks` (dict insertion order).
    """
    canvas_h, canvas_w = canvas_hw
    actual_fill = fill if fill is not None else 0
    fill_value = np.array(actual_fill, dtype=dtype)

    layers: list[np.ndarray] = []
    for placement_coords, cell_data in processed_cells.items():
        stack = cell_data.get("masks")
        if stack is None or not isinstance(stack, np.ndarray) or stack.size == 0:
            continue
        if stack.ndim == 4 and stack.shape[-1] == 1:
            stack = np.squeeze(stack, axis=-1)
        if stack.ndim != 3:
            continue
        tgt_x1, tgt_y1, tgt_x2, tgt_y2 = placement_coords
        for i in range(stack.shape[0]):
            canvas = np.full((canvas_h, canvas_w), fill_value=fill_value, dtype=dtype)
            segment = stack[i]
            if segment.ndim == 3 and segment.shape[-1] == 1:
                segment = segment[..., 0]
            canvas[tgt_y1:tgt_y2, tgt_x1:tgt_x2] = segment
            layers.append(canvas)

    if not layers:
        return np.empty((0, canvas_h, canvas_w), dtype=dtype)
    return np.stack(layers, axis=0)


def _mosaic_cell_local_instance_ids(
    bb: np.ndarray | None,
    kp: np.ndarray | None,
    need_bbox_remap: bool,
    need_kp_remap: bool,
    n_bf: int,
    n_kf: int,
    bbox_id_idx: int,
    kp_id_idx: int,
) -> set[int]:
    local_ids: set[int] = set()
    if need_bbox_remap and bb is not None and np.asarray(bb).size > 0 and n_bf > 0:
        bb_np = np.asarray(bb)
        n_geo_bb = bb_np.shape[1] - n_bf
        id_col_bb = n_geo_bb + bbox_id_idx
        for row in range(bb_np.shape[0]):
            local_ids.add(int(bb_np[row, id_col_bb]))
    if need_kp_remap and kp is not None and np.asarray(kp).size > 0 and n_kf > 0:
        kp_np = np.asarray(kp)
        n_geo_kp = kp_np.shape[1] - n_kf
        id_col_kp = n_geo_kp + kp_id_idx
        for row in range(kp_np.shape[0]):
            local_ids.add(int(kp_np[row, id_col_kp]))
    return local_ids


def _remap_mosaic_bboxes_column(
    bb_arr: np.ndarray,
    need_bbox_remap: bool,
    n_bf: int,
    bbox_id_idx: int,
    local_to_global: dict[int, int],
) -> None:
    if not (need_bbox_remap and n_bf > 0):
        return
    n_geo_bb = bb_arr.shape[1] - n_bf
    id_col_bb = n_geo_bb + bbox_id_idx
    for row in range(bb_arr.shape[0]):
        lid = int(bb_arr[row, id_col_bb])
        bb_arr[row, id_col_bb] = float(local_to_global[lid])


def _remap_mosaic_keypoints_column(
    kp_arr: np.ndarray,
    need_kp_remap: bool,
    n_kf: int,
    kp_id_idx: int,
    local_to_global: dict[int, int],
) -> None:
    if not (need_kp_remap and n_kf > 0):
        return
    n_geo_kp = kp_arr.shape[1] - n_kf
    id_col_kp = n_geo_kp + kp_id_idx
    for row in range(kp_arr.shape[0]):
        lid = int(kp_arr[row, id_col_kp])
        if lid in local_to_global:
            kp_arr[row, id_col_kp] = float(local_to_global[lid])


def _remap_one_mosaic_cell_instance_ids(
    cell: ProcessedMosaicItem,
    need_bbox_remap: bool,
    need_kp_remap: bool,
    n_bf: int,
    n_kf: int,
    bbox_id_idx: int,
    kp_id_idx: int,
    global_next: int,
) -> tuple[ProcessedMosaicItem, int]:
    cell_out: dict[str, Any] = {"image": cell["image"]}
    if "mask" in cell:
        cell_out["mask"] = cell["mask"]
    if "masks" in cell:
        cell_out["masks"] = cell["masks"]

    bb = cell.get("bboxes")
    kp = cell.get("keypoints")
    local_ids = _mosaic_cell_local_instance_ids(
        bb,
        kp,
        need_bbox_remap,
        need_kp_remap,
        n_bf,
        n_kf,
        bbox_id_idx,
        kp_id_idx,
    )
    local_to_global = {lid: global_next + idx for idx, lid in enumerate(sorted(local_ids))}
    next_global = global_next + len(local_to_global)

    if bb is not None and np.asarray(bb).size > 0:
        bb_arr = np.asarray(bb, dtype=np.float32, copy=True)
        _remap_mosaic_bboxes_column(bb_arr, need_bbox_remap, n_bf, bbox_id_idx, local_to_global)
        cell_out["bboxes"] = bb_arr
    else:
        cell_out["bboxes"] = cell.get(
            "bboxes",
            np.empty((0, NUM_BBOXES_COLUMNS_IN_ALBUMENTATIONS), dtype=np.float32),
        )

    if kp is not None and np.asarray(kp).size > 0:
        kp_arr = np.asarray(kp, dtype=np.float32, copy=True)
        _remap_mosaic_keypoints_column(kp_arr, need_kp_remap, n_kf, kp_id_idx, local_to_global)
        cell_out["keypoints"] = kp_arr
    else:
        cell_out["keypoints"] = cell.get(
            "keypoints",
            np.empty((0, NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS), dtype=np.float32),
        )

    return cast("ProcessedMosaicItem", cell_out), next_global


def remap_mosaic_instance_label_ids(
    processed_cells: dict[tuple[int, int, int, int], ProcessedMosaicItem],
    bbox_processor: BboxProcessor | None,
    keypoint_processor: KeypointsProcessor | None,
) -> dict[tuple[int, int, int, int], ProcessedMosaicItem]:
    """Assign globally unique instance id column values per mosaic cell so repack does not
    merge distinct instances that reused local ids (0..n-1) in different cells.
    """
    bbox_fields = bbox_processor.params.label_fields if bbox_processor else None
    kp_fields = keypoint_processor.params.label_fields if keypoint_processor else None

    need_bbox_remap = bool(bbox_fields and _BBOX_INSTANCE_ID in bbox_fields)
    need_kp_remap = bool(kp_fields and _KP_INSTANCE_ID in kp_fields)
    if not need_bbox_remap and not need_kp_remap:
        return processed_cells

    n_bf = len(bbox_fields) if bbox_fields else 0
    bbox_id_idx = bbox_fields.index(_BBOX_INSTANCE_ID) if bbox_fields and need_bbox_remap else -1

    n_kf = len(kp_fields) if kp_fields else 0
    kp_id_idx = kp_fields.index(_KP_INSTANCE_ID) if kp_fields and need_kp_remap else -1

    global_next = 0
    new_cells: dict[tuple[int, int, int, int], ProcessedMosaicItem] = {}

    for placement, cell in processed_cells.items():
        cell_out, global_next = _remap_one_mosaic_cell_instance_ids(
            cell,
            need_bbox_remap,
            need_kp_remap,
            n_bf,
            n_kf,
            bbox_id_idx,
            kp_id_idx,
            global_next,
        )
        new_cells[placement] = cell_out

    return new_cells


def process_all_mosaic_geometries(
    canvas_shape: tuple[int, int],
    cell_shape: tuple[int, int],
    placement_to_item_index: dict[tuple[int, int, int, int], int],
    final_items_for_grid: list[ProcessedMosaicItem],
    fill: float | tuple[float, ...],
    fill_mask: float | tuple[float, ...],
    fit_mode: Literal["cover", "contain"],
    interpolation: Literal[
        cv2.INTER_NEAREST,
        cv2.INTER_NEAREST_EXACT,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4,
        cv2.INTER_LINEAR_EXACT,
    ],
    mask_interpolation: Literal[
        cv2.INTER_NEAREST,
        cv2.INTER_NEAREST_EXACT,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4,
        cv2.INTER_LINEAR_EXACT,
    ],
) -> dict[tuple[int, int, int, int], ProcessedMosaicItem]:
    """Crop/pad every assigned cell via process_cell_geometry. Returns placement ->
    ProcessedMosaicItem (bbox/keypoint coords not yet shifted).

    Iterates through assigned placements, applies geometric transforms via process_cell_geometry,
    and returns a dictionary mapping final placement coordinates to the processed item data.
    The bbox/keypoint coordinates in the returned dict are *not* shifted yet.

    Args:
        canvas_shape (tuple[int, int]): The shape of the canvas.
        cell_shape (tuple[int, int]): Shape of each cell in the mosaic grid.
        placement_to_item_index (dict[tuple[int, int, int, int], int]): Mapping from placement
            coordinates (x1, y1, x2, y2) to assigned item index.
        final_items_for_grid (list[ProcessedMosaicItem]): List of all preprocessed items available.
        fill (float | tuple[float, ...]): Fill value for image padding.
        fill_mask (float | tuple[float, ...]): Fill value for mask padding.
        fit_mode (Literal['cover', 'contain']): Fit mode for the mosaic.
        interpolation (Literal[cv2.INTER_NEAREST, cv2.INTER_NEAREST_EXACT, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT]): Interpolation for image.
        mask_interpolation (Literal[cv2.INTER_NEAREST, cv2.INTER_NEAREST_EXACT, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT]): Interpolation for mask.

    Returns:
        dict[tuple[int, int, int, int], ProcessedMosaicItem]: Dictionary mapping final placement
        coordinates (x1, y1, x2, y2) to the geometrically processed item data (image, mask, un-shifted bboxes/kps).

    """
    processed_cells_geom: dict[tuple[int, int, int, int], ProcessedMosaicItem] = {}

    # Iterate directly over placements and their assigned item indices
    for placement_coords, item_idx in placement_to_item_index.items():
        item = final_items_for_grid[item_idx]
        tgt_x1, tgt_y1, tgt_x2, tgt_y2 = placement_coords
        target_h = tgt_y2 - tgt_y1
        target_w = tgt_x2 - tgt_x1

        cell_position = get_cell_relative_position(placement_coords, canvas_shape)

        # Apply geometric processing (crop/pad)
        processed_cells_geom[placement_coords] = process_cell_geometry(
            cell_shape=cell_shape,
            item=item,
            target_shape=(target_h, target_w),
            fill=fill,
            fill_mask=fill_mask,
            fit_mode=fit_mode,
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            cell_position=cell_position,
        )

    return processed_cells_geom


def get_cell_relative_position(
    placement_coords: tuple[int, int, int, int],
    target_shape: tuple[int, int],
) -> Literal["top_left", "top_right", "center", "bottom_left", "bottom_right"]:
    """Return cell quadrant relative to canvas center: top_left, top_right, bottom_left,
    bottom_right, or center. For mosaic crop positioning.

    Compares the cell center to the canvas center and returns its quadrant
    or "center" if it lies on or very close to a central axis.

    Args:
        placement_coords (tuple[int, int, int, int]): The (x_min, y_min, x_max, y_max) coordinates
            of the cell.
        target_shape (tuple[int, int]): The (height, width) of the overall target canvas.

    Returns:
        Literal['top_left', 'top_right', 'center', 'bottom_left', 'bottom_right']:
            The position of the cell relative to the center of the target canvas.

    """
    target_h, target_w = target_shape
    x1, y1, x2, y2 = placement_coords

    canvas_center_x = target_w / 2.0
    canvas_center_y = target_h / 2.0

    cell_center_x = (x1 + x2) / 2.0
    cell_center_y = (y1 + y2) / 2.0

    # Determine vertical position
    if cell_center_y < canvas_center_y:
        v_pos = "top"
    elif cell_center_y > canvas_center_y:
        v_pos = "bottom"
    else:  # Exactly on the horizontal center line
        v_pos = "center"

    # Determine horizontal position
    if cell_center_x < canvas_center_x:
        h_pos = "left"
    elif cell_center_x > canvas_center_x:
        h_pos = "right"
    else:  # Exactly on the vertical center line
        h_pos = "center"

    # Map positions to the final string
    position_map: dict[tuple[str, str], str] = {
        ("top", "left"): "top_left",
        ("top", "right"): "top_right",
        ("bottom", "left"): "bottom_left",
        ("bottom", "right"): "bottom_right",
    }

    # Default to "center" if the combination is not in the map
    # (which happens if either v_pos or h_pos is "center")
    return cast(
        "Literal['top_left', 'top_right', 'center', 'bottom_left', 'bottom_right']",
        position_map.get((v_pos, h_pos), "center"),
    )


def shift_all_coordinates(
    processed_cells_geom: dict[tuple[int, int, int, int], ProcessedMosaicItem],
    canvas_shape: tuple[int, int],
) -> dict[tuple[int, int, int, int], ProcessedMosaicItem]:  # Return type matches input, but values are updated
    """Shift bbox and keypoint coordinates in each cell to final canvas positions. Same keys as
    input; values are ProcessedMosaicItem with shifted bboxes/keypoints.

    Iterates through the processed cells (keyed by placement coords), applies coordinate
    shifting to bboxes/keypoints, and returns a new dictionary with the same keys
    but updated ProcessedMosaicItem values containing the *shifted* coordinates.

    Args:
        processed_cells_geom (dict[tuple[int, int, int, int], ProcessedMosaicItem]):
             Output from process_all_mosaic_geometries (keyed by placement coords).
        canvas_shape (tuple[int, int]): The shape of the canvas.

    Returns:
        dict[tuple[int, int, int, int], ProcessedMosaicItem]: Final dictionary mapping
        placement coords (x1, y1, x2, y2) to processed cell data with shifted coordinates.

    """
    final_processed_cells: dict[tuple[int, int, int, int], ProcessedMosaicItem] = {}
    canvas_h, canvas_w = canvas_shape

    for placement_coords, cell_data_geom in processed_cells_geom.items():
        tgt_x1, tgt_y1 = placement_coords[:2]

        cell_width = placement_coords[2] - placement_coords[0]
        cell_height = placement_coords[3] - placement_coords[1]

        # Extract geometrically processed bboxes/keypoints
        bboxes_geom = cell_data_geom.get("bboxes")
        keypoints_geom = cell_data_geom.get("keypoints")

        final_cell_data: ProcessedMosaicItem = {
            "image": cell_data_geom["image"],
            "mask": cell_data_geom.get("mask"),
        }
        if "masks" in cell_data_geom:
            final_cell_data["masks"] = cell_data_geom["masks"]

        # Perform shifting if data exists
        if bboxes_geom is not None and bboxes_geom.size > 0:
            bboxes_geom_arr = np.asarray(bboxes_geom)
            bbox_denoramlized = denormalize_bboxes(bboxes_geom_arr, (cell_height, cell_width))
            bbox_shift_vector = np.array([tgt_x1, tgt_y1, tgt_x1, tgt_y1], dtype=np.float32)

            shifted_bboxes_denormalized = fgeometric.shift_bboxes(bbox_denoramlized, bbox_shift_vector)
            shifted_bboxes = normalize_bboxes(shifted_bboxes_denormalized, (canvas_h, canvas_w))
            final_cell_data["bboxes"] = shifted_bboxes
        else:
            final_cell_data["bboxes"] = np.empty((0, NUM_BBOXES_COLUMNS_IN_ALBUMENTATIONS))

        if keypoints_geom is not None and keypoints_geom.size > 0:
            keypoints_geom_arr = np.asarray(keypoints_geom)

            # Ensure shift vector matches keypoint dtype (usually float)
            kp_shift_vector = np.array([tgt_x1, tgt_y1, 0], dtype=keypoints_geom_arr.dtype)

            shifted_keypoints = fgeometric.shift_keypoints(keypoints_geom_arr, kp_shift_vector)

            final_cell_data["keypoints"] = shifted_keypoints
        else:
            final_cell_data["keypoints"] = np.empty((0, NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS))

        final_processed_cells[placement_coords] = final_cell_data

    return final_processed_cells
