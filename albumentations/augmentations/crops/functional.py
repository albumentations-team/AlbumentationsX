"""Functional implementations of image cropping operations.

This module provides utility functions for performing various cropping operations on images,
bounding boxes, and keypoints. It includes functions to calculate crop coordinates, crop images,
and handle the corresponding transformations for bounding boxes and keypoints to maintain
consistency between different data types during cropping operations.
"""

from collections.abc import Sequence
from typing import Any, Literal

import cv2
import numpy as np
from albucore import maybe_process_in_chunks, preserve_channel_dim

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.utils import handle_empty_array
from albumentations.core.bbox_utils import (
    BBOX_OBB_MIN_COLUMNS,
    denormalize_bboxes,
    normalize_bbox_angles_decorator,
    normalize_bboxes,
    obb_to_polygons,
    polygons_to_obb,
)
from albumentations.core.type_definitions import ImageType

__all__ = [
    "crop",
    "crop_and_pad",
    "crop_and_pad_bboxes",
    "crop_and_pad_keypoints",
    "crop_bboxes_by_coords",
    "crop_keypoints_by_coords",
    "get_center_crop_coords",
    "get_crop_coords",
    "pad_along_axes",
    "volume_crop_yx",
    "volumes_crop_yx",
]


def get_crop_coords(
    image_shape: tuple[int, int],
    crop_shape: tuple[int, int],
    h_start: float,
    w_start: float,
) -> tuple[int, int, int, int]:
    """Get crop coordinates.

    This function gets the crop coordinates.

    Args:
        image_shape (tuple[int, int]): Original image shape.
        crop_shape (tuple[int, int]): Crop shape.
        h_start (float): Start height.
        w_start (float): Start width.

    Returns:
        tuple[int, int, int, int]: Crop coordinates.

    """
    # h_start is [0, 1) and should map to [0, (height - crop_height)]  (note inclusive)
    # This is conceptually equivalent to mapping onto `range(0, (height - crop_height + 1))`
    # See: https://github.com/albumentations-team/albumentations/pull/1080
    # We want range for coordinated to be [0, image_size], right side is included

    height, width = image_shape[:2]

    # Clip crop dimensions to image dimensions
    crop_height = min(crop_shape[0], height)
    crop_width = min(crop_shape[1], width)

    y_min = int((height - crop_height + 1) * h_start)
    y_max = y_min + crop_height
    x_min = int((width - crop_width + 1) * w_start)
    x_max = x_min + crop_width
    return x_min, y_min, x_max, y_max


def _process_clipped_obb_boxes(
    clipped_indices: np.ndarray,
    polygons_clipped: np.ndarray,
    crop_width: int,
    crop_height: int,
    extras: np.ndarray | None,
) -> np.ndarray:
    """Process OBB boxes that were clipped by crop boundaries.

    Refits OBBs using cv2.minAreaRect for boxes with clipped corners.
    """
    clipped_polys = polygons_clipped[clipped_indices]

    # Normalize clipped polygons
    clipped_polys_norm = clipped_polys.copy()
    clipped_polys_norm[..., 0] /= crop_width
    clipped_polys_norm[..., 1] /= crop_height

    # Fit new OBBs from clipped polygons
    clipped_extras = extras[clipped_indices] if extras is not None else None
    return polygons_to_obb(clipped_polys_norm, extra_fields=clipped_extras)


def _process_unclipped_obb_boxes(
    unclipped_indices: np.ndarray,
    bboxes: np.ndarray,
    image_shape: tuple[int, int],
    crop_coords: tuple[int, int, int, int],
    crop_width: int,
    crop_height: int,
    extras: np.ndarray | None,
) -> np.ndarray:
    """Process OBB boxes that were fully inside crop (vectorized center shift).

    Preserves original angle and dimensions, only adjusts center position.
    Returns transformed bboxes.
    """
    crop_x_min, crop_y_min = crop_coords[:2]
    unclipped_bboxes = bboxes[unclipped_indices]

    # Extract OBB parameters (vectorized)
    x_min, y_min, x_max, y_max = (
        unclipped_bboxes[:, 0],
        unclipped_bboxes[:, 1],
        unclipped_bboxes[:, 2],
        unclipped_bboxes[:, 3],
    )
    angles = unclipped_bboxes[:, 4]

    # Calculate original centers in pixels (vectorized)
    center_x_px = (x_min * image_shape[1] + x_max * image_shape[1]) / 2
    center_y_px = (y_min * image_shape[0] + y_max * image_shape[0]) / 2

    # Adjust centers to crop region (vectorized)
    new_center_x_px = center_x_px - crop_x_min
    new_center_y_px = center_y_px - crop_y_min

    # Calculate normalized dimensions (vectorized)
    width_norm = x_max - x_min
    height_norm = y_max - y_min

    # Normalize new centers (vectorized)
    new_center_x_norm = new_center_x_px / crop_width
    new_center_y_norm = new_center_y_px / crop_height

    # Calculate new bounds in normalized coords relative to crop (vectorized)
    new_x_min = new_center_x_norm - width_norm * image_shape[1] / crop_width / 2
    new_x_max = new_center_x_norm + width_norm * image_shape[1] / crop_width / 2
    new_y_min = new_center_y_norm - height_norm * image_shape[0] / crop_height / 2
    new_y_max = new_center_y_norm + height_norm * image_shape[0] / crop_height / 2

    # Build new bboxes with same angles (vectorized)
    num_bboxes = len(unclipped_indices)
    num_cols = 5 if extras is None else 5 + extras.shape[1]
    result = np.empty((num_bboxes, num_cols), dtype=bboxes.dtype)

    result[:, 0] = new_x_min
    result[:, 1] = new_y_min
    result[:, 2] = new_x_max
    result[:, 3] = new_y_max
    result[:, 4] = angles

    # Copy extra fields if present
    if extras is not None:
        result[:, 5:] = extras[unclipped_indices]

    return result


@normalize_bbox_angles_decorator()
@handle_empty_array("bboxes")
def crop_bboxes_by_coords_obb(
    bboxes: np.ndarray,
    crop_coords: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Crop oriented bounding boxes using vectorized polygon-based method.

    This function handles OBB cropping by:
    1. Converting OBB to 4 corner polygons
    2. Cropping polygon corners by the crop region
    3. Detecting which boxes had corners clipped (vectorized)
    4. For unclipped boxes: vectorized center shift (fast path, preserves angle)
    5. For clipped boxes: fitting new OBB via cv2.minAreaRect (batch processing)

    Args:
        bboxes (np.ndarray): Array of OBB with shape (N, 5+) where each row is
                             [x_min, y_min, x_max, y_max, angle, ...] in normalized coordinates.
        crop_coords (tuple[int, int, int, int]): Crop coordinates (x_min, y_min, x_max, y_max)
                                                 in absolute pixel values.
        image_shape (tuple[int, int]): Original image shape (height, width).

    Returns:
        np.ndarray: Array of cropped OBB in normalized coordinates relative to crop region.

    Note:
        This implements the industry-standard polygon-based approach (PyTorch Torchvision).
        Fully vectorized for performance - processes all boxes simultaneously.
        Boxes fully inside the crop use fast vectorized center shift.
        Boxes partially cropped get batch-refitted via cv2.minAreaRect.

    """
    # Extract extra fields if present (e.g., class labels)
    extras = bboxes[:, 5:] if bboxes.shape[1] > BBOX_OBB_MIN_COLUMNS else None

    # Step 1: Convert OBB to polygons (4 corners each)
    polygons = obb_to_polygons(bboxes)  # Shape: (N, 4, 2) in normalized coords

    # Step 2: Denormalize polygons to pixel coordinates
    polygons_px = polygons.copy()
    polygons_px[..., 0] *= image_shape[1]  # x coords
    polygons_px[..., 1] *= image_shape[0]  # y coords

    # Step 3: Subtract crop offset
    crop_x_min, crop_y_min, crop_x_max, crop_y_max = crop_coords
    polygons_cropped = polygons_px.copy()
    polygons_cropped[..., 0] -= crop_x_min
    polygons_cropped[..., 1] -= crop_y_min

    # Step 4: Clip to crop boundaries
    crop_height = crop_y_max - crop_y_min
    crop_width = crop_x_max - crop_x_min
    polygons_clipped = np.clip(
        polygons_cropped,
        [0, 0],
        [crop_width, crop_height],
    )

    # Step 5: Check which boxes had corners changed (were clipped)
    eps = 1e-6
    clipped_mask = np.any(np.abs(polygons_cropped - polygons_clipped) > eps, axis=(1, 2))  # Shape: (N,)

    # Step 6: Filter out boxes that are completely outside crop (all corners collapsed)
    # Vectorized check: compute per-polygon extents
    x_coords = polygons_clipped[:, :, 0]  # Shape: (N, 4)
    y_coords = polygons_clipped[:, :, 1]  # Shape: (N, 4)
    x_range = x_coords.max(axis=1) - x_coords.min(axis=1)  # Shape: (N,)
    y_range = y_coords.max(axis=1) - y_coords.min(axis=1)  # Shape: (N,)

    # Keep boxes with non-negligible extent; truly collapsed polygons are removed
    # Use epsilon to detect degenerate polygons, not arbitrary pixel threshold
    valid_mask = (x_range > eps) & (y_range > eps)

    # Filter to valid boxes only
    if not np.any(valid_mask):
        # All boxes filtered out
        empty_shape = (0, bboxes.shape[1]) if len(bboxes.shape) > 1 else (0, BBOX_OBB_MIN_COLUMNS)
        return np.empty(empty_shape, dtype=np.float32)

    # Keep only valid indices
    valid_indices = np.where(valid_mask)[0]
    filtered_bboxes = bboxes[valid_indices]
    filtered_polygons_clipped = polygons_clipped[valid_indices]
    filtered_clipped_mask = clipped_mask[valid_indices]
    filtered_extras = extras[valid_indices] if extras is not None else None

    # Step 7: Process clipped and unclipped boxes separately (vectorized)
    result_bboxes = np.empty((len(filtered_bboxes), filtered_bboxes.shape[1]), dtype=np.float32)

    # Handle clipped boxes (need refitting)
    if np.any(filtered_clipped_mask):
        clipped_indices = np.where(filtered_clipped_mask)[0]
        new_obbs = _process_clipped_obb_boxes(
            clipped_indices,
            filtered_polygons_clipped,
            crop_width,
            crop_height,
            filtered_extras,
        )
        result_bboxes[clipped_indices] = new_obbs

    # Handle unclipped boxes (vectorized center shift)
    if np.any(~filtered_clipped_mask):
        unclipped_indices = np.where(~filtered_clipped_mask)[0]
        result_bboxes[unclipped_indices] = _process_unclipped_obb_boxes(
            unclipped_indices,
            filtered_bboxes,
            image_shape,
            crop_coords,
            crop_width,
            crop_height,
            filtered_extras,
        )

    return result_bboxes


def crop_bboxes_by_coords(
    bboxes: np.ndarray,
    crop_coords: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    bbox_type: Literal["obb", "hbb"],
) -> np.ndarray:
    """Crop bounding boxes based on given crop coordinates.

    This function adjusts bounding boxes to fit within a cropped image.
    Supports both HBB (axis-aligned) and OBB (oriented) bounding boxes.

    Args:
        bboxes (np.ndarray): Array of normalized bounding boxes (Albumentations format) with shape (N, 4+)
                             where each row is [x_min, y_min, x_max, y_max, ...] for HBB or
                             [x_min, y_min, x_max, y_max, angle, ...] for OBB.
        crop_coords (tuple[int, int, int, int]): Crop coordinates (x_min, y_min, x_max, y_max)
                                                 in absolute pixel values.
        image_shape (tuple[int, int]): Original image shape (height, width).
        bbox_type (Literal["obb", "hbb"]): Type of bounding box - "hbb" or "obb". Must be explicitly provided.

    Returns:
        np.ndarray: Array of cropped bounding boxes in normalized coordinates (Albumentations format).

    Note:
        Bounding boxes that fall completely outside the crop area will be removed.
        Bounding boxes that partially overlap with the crop area will be adjusted to fit within it.
        For OBB, uses polygon-based cropping with cv2.minAreaRect for partially clipped boxes.

    """
    if not bboxes.size:
        return bboxes

    # Process as OBB only when bbox_type == "obb"
    # Otherwise, use the HBB path (supports HBB with extra columns like class labels)
    is_obb = bbox_type == "obb"

    if is_obb:
        return crop_bboxes_by_coords_obb(bboxes, crop_coords, image_shape)

    # HBB path - convert normalized to absolute, crop, then normalize back
    cropped_bboxes = denormalize_bboxes(bboxes.copy().astype(np.float32), image_shape)

    x_min, y_min = crop_coords[:2]

    # Subtract crop coordinates
    cropped_bboxes[:, [0, 2]] -= x_min
    cropped_bboxes[:, [1, 3]] -= y_min

    # Calculate crop shape
    crop_height = crop_coords[3] - crop_coords[1]
    crop_width = crop_coords[2] - crop_coords[0]
    crop_shape = (crop_height, crop_width)

    return normalize_bboxes(cropped_bboxes, crop_shape)


@handle_empty_array("keypoints")
def crop_keypoints_by_coords(
    keypoints: np.ndarray,
    crop_coords: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop keypoints using the provided coordinates of bottom-left and top-right corners in pixels.

    Args:
        keypoints (np.ndarray): An array of keypoints with shape (N, 4+) where each row is (x, y, angle, scale, ...).
        crop_coords (tuple): Crop box coords (x1, y1, x2, y2).

    Returns:
        np.ndarray: An array of cropped keypoints with the same shape as the input.

    """
    x1, y1 = crop_coords[:2]

    cropped_keypoints = keypoints.copy()
    cropped_keypoints[:, 0] -= x1  # Adjust x coordinates
    cropped_keypoints[:, 1] -= y1  # Adjust y coordinates

    return cropped_keypoints


def get_center_crop_coords(image_shape: tuple[int, int], crop_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Get center crop coordinates.

    This function gets the center crop coordinates.

    Args:
        image_shape (tuple[int, int]): Original image shape.
        crop_shape (tuple[int, int]): Crop shape.

    Returns:
        tuple[int, int, int, int]: Center crop coordinates.

    """
    height, width = image_shape[:2]
    crop_height, crop_width = crop_shape[:2]

    y_min = (height - crop_height) // 2
    y_max = y_min + crop_height
    x_min = (width - crop_width) // 2
    x_max = x_min + crop_width
    return x_min, y_min, x_max, y_max


def crop(img: ImageType, x_min: int, y_min: int, x_max: int, y_max: int) -> ImageType:
    """Crop an image.

    This function crops an image.

    Args:
        img (np.ndarray): Input image.
        x_min (int): Minimum x coordinate.
        y_min (int): Minimum y coordinate.
        x_max (int): Maximum x coordinate.
        y_max (int): Maximum y coordinate.

    Returns:
        np.ndarray: Cropped image.

    """
    height, width = img.shape[:2]
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max. But we got"
            f" (x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max})",
        )

    if x_min < 0 or x_max > width or y_min < 0 or y_max > height:
        raise ValueError(
            "Values for crop should be non negative and equal or smaller than image sizes"
            f"(x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max}, "
            f"height = {height}, width = {width})",
        )

    return img[y_min:y_max, x_min:x_max]


@preserve_channel_dim
def crop_and_pad(
    img: ImageType,
    crop_params: tuple[int, int, int, int] | None,
    pad_params: tuple[int, int, int, int] | None,
    pad_value: tuple[float, ...] | float | None,
    image_shape: tuple[int, int],
    interpolation: int,
    pad_mode: int,
    keep_size: bool,
) -> ImageType:
    """Crop and pad an image.

    This function crops and pads an image.

    Args:
        img (np.ndarray): Input image.
        crop_params (tuple[int, int, int, int] | None): Crop parameters.
        pad_params (tuple[int, int, int, int] | None): Pad parameters.
        pad_value (tuple[float, ...] | float | None): Pad value.
        image_shape (tuple[int, int]): Original image shape.
        interpolation (int): Interpolation method.
        pad_mode (int): Pad mode.
        keep_size (bool): Whether to keep the original size.

    Returns:
        np.ndarray: Cropped and padded image.

    """
    if crop_params is not None and any(i != 0 for i in crop_params):
        img = crop(img, *crop_params)
    if pad_params is not None and any(i != 0 for i in pad_params):
        img = fgeometric.pad_with_params(
            img,
            pad_params[0],
            pad_params[1],
            pad_params[2],
            pad_params[3],
            border_mode=pad_mode,
            value=pad_value,
        )

    if keep_size:
        rows, cols = image_shape[:2]
        resize_fn = maybe_process_in_chunks(cv2.resize, dsize=(cols, rows), interpolation=interpolation)
        return resize_fn(img)

    return img


def crop_and_pad_bboxes(
    bboxes: np.ndarray,
    crop_params: tuple[int, int, int, int] | None,
    pad_params: tuple[int, int, int, int] | None,
    image_shape: tuple[int, int],
    result_shape: tuple[int, int],
    bbox_type: Literal["obb", "hbb"],
) -> np.ndarray:
    """Crop and pad bounding boxes.

    This function crops and pads bounding boxes. Supports both HBB and OBB.

    Args:
        bboxes (np.ndarray): Array of bounding boxes (HBB or OBB).
        crop_params (tuple[int, int, int, int] | None): Crop parameters.
        pad_params (tuple[int, int, int, int] | None): Pad parameters.
        image_shape (tuple[int, int]): Original image shape.
        result_shape (tuple[int, int]): Result image shape.
        bbox_type (Literal["obb", "hbb"]): Type of bounding box - "hbb" or "obb". Must be explicitly provided.

    Returns:
        np.ndarray: Array of cropped and padded bounding boxes.

    """
    if len(bboxes) == 0:
        return bboxes

    # Only process as OBB if explicitly bbox_type='obb'
    is_obb = bbox_type == "obb"

    if is_obb and crop_params is not None:
        # For OBB with crop, use specialized OBB crop function that clips polygons
        extras = bboxes[:, 5:] if bboxes.shape[1] > BBOX_OBB_MIN_COLUMNS else None

        # Convert to polygons
        polygons = obb_to_polygons(bboxes)  # normalized coords

        # Denormalize to pixels
        polygons_px = polygons.copy()
        polygons_px[..., 0] *= image_shape[1]
        polygons_px[..., 1] *= image_shape[0]

        # Apply crop: shift to crop-relative coordinates and clip to crop rectangle
        crop_x, crop_y, crop_x_max, crop_y_max = crop_params
        polygons_px[..., 0] -= crop_x
        polygons_px[..., 1] -= crop_y

        # Clip polygons to crop boundaries
        crop_width = crop_x_max - crop_x
        crop_height = crop_y_max - crop_y
        polygons_px[..., 0] = np.clip(polygons_px[..., 0], 0, crop_width)
        polygons_px[..., 1] = np.clip(polygons_px[..., 1], 0, crop_height)

        # Apply pad if needed
        if pad_params is not None:
            top, _, left, _ = pad_params
            polygons_px[..., 0] += left
            polygons_px[..., 1] += top

        # Normalize to result shape
        result_width = result_shape[1]
        result_height = result_shape[0]
        polygons_px[..., 0] /= result_width
        polygons_px[..., 1] /= result_height

        # Convert back to OBB
        return polygons_to_obb(polygons_px, extra_fields=extras)

    # HBB path - original logic
    # Denormalize bboxes
    denormalized_bboxes = denormalize_bboxes(bboxes, image_shape)

    if crop_params is not None:
        crop_x, crop_y = crop_params[:2]
        # Subtract crop values from x and y coordinates
        denormalized_bboxes[:, [0, 2]] -= crop_x
        denormalized_bboxes[:, [1, 3]] -= crop_y

    if pad_params is not None:
        top, _, left, _ = pad_params
        # Add pad values to x and y coordinates
        denormalized_bboxes[:, [0, 2]] += left
        denormalized_bboxes[:, [1, 3]] += top

    # Normalize bboxes to the result shape
    return normalize_bboxes(denormalized_bboxes, result_shape)


@handle_empty_array("keypoints")
def crop_and_pad_keypoints(
    keypoints: np.ndarray,
    crop_params: tuple[int, int, int, int] | None = None,
    pad_params: tuple[int, int, int, int] | None = None,
    image_shape: tuple[int, int] = (0, 0),
    result_shape: tuple[int, int] = (0, 0),
    keep_size: bool = False,
) -> np.ndarray:
    """Crop and pad multiple keypoints simultaneously.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (N, 4+) where each row is (x, y, angle, scale, ...).
        crop_params (Sequence[int], optional): Crop parameters [crop_x1, crop_y1, ...].
        pad_params (Sequence[int], optional): Pad parameters [top, bottom, left, right].
        image_shape (Tuple[int, int]): Original image shape (rows, cols).
        result_shape (Tuple[int, int]): Result image shape (rows, cols).
        keep_size (bool): Whether to keep the original size.

    Returns:
        np.ndarray: Array of transformed keypoints with the same shape as input.

    """
    transformed_keypoints = keypoints.copy()

    if crop_params is not None:
        crop_x1, crop_y1 = crop_params[:2]
        transformed_keypoints[:, 0] -= crop_x1
        transformed_keypoints[:, 1] -= crop_y1

    if pad_params is not None:
        top, _, left, _ = pad_params
        transformed_keypoints[:, 0] += left
        transformed_keypoints[:, 1] += top

    rows, cols = image_shape[:2]
    result_rows, result_cols = result_shape[:2]

    if keep_size and (result_cols != cols or result_rows != rows):
        scale_x = cols / result_cols
        scale_y = rows / result_rows
        return fgeometric.keypoints_scale(transformed_keypoints, scale_x, scale_y)

    return transformed_keypoints


def volume_crop_yx(
    volume: ImageType,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> ImageType:
    """Crop a single volume along Y (height) and X (width) axes only.

    Args:
        volume (np.ndarray): Input volume with shape (D, H, W) or (D, H, W, C).
        x_min (int): Minimum width coordinate.
        y_min (int): Minimum height coordinate.
        x_max (int): Maximum width coordinate.
        y_max (int): Maximum height coordinate.

    Returns:
        np.ndarray: Cropped volume (D, H_new, W_new, [C]).

    Raises:
        ValueError: If crop coordinates are invalid.

    """
    _, height, width = volume.shape[:3]
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "Crop coordinates must satisfy min < max. Got: "
            f"(x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max})",
        )

    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        raise ValueError(
            "Crop coordinates must be within image dimensions (H, W). Got: "
            f"(x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}) "
            f"for volume shape {volume.shape[:3]}",
        )

    # Crop along H (axis 1) and W (axis 2)
    return volume[:, y_min:y_max, x_min:x_max]


def volumes_crop_yx(
    volumes: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> np.ndarray:
    """Crop a batch of volumes along Y (height) and X (width) axes only.

    Args:
        volumes (np.ndarray): Input batch of volumes with shape (B, D, H, W) or (B, D, H, W, C).
        x_min (int): Minimum width coordinate.
        y_min (int): Minimum height coordinate.
        x_max (int): Maximum width coordinate.
        y_max (int): Maximum height coordinate.

    Returns:
        np.ndarray: Cropped batch of volumes (B, D, H_new, W_new, [C]).

    Raises:
        ValueError: If crop coordinates are invalid or volumes shape is incorrect.

    """
    if not 4 <= volumes.ndim <= 5:
        raise ValueError(f"Input volumes should have 4 or 5 dimensions, got {volumes.ndim}")

    depth, height, width = volumes.shape[1:4]
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "Crop coordinates must satisfy min < max. Got: "
            f"(x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max})",
        )

    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        raise ValueError(
            "Crop coordinates must be within image dimensions (H, W). Got: "
            f"(x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}) "
            f"for volume shape {(depth, height, width)}",
        )

    # Crop along H (axis 2) and W (axis 3)
    return volumes[:, :, y_min:y_max, x_min:x_max]


def pad_along_axes(
    arr: np.ndarray,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    h_axis: int,
    w_axis: int,
    border_mode: int,
    pad_value: float | Sequence[float] = 0,
) -> np.ndarray:
    """Pad an array along specified height (H) and width (W) axes using np.pad.

    Args:
        arr (np.ndarray): Input array.
        pad_top (int): Padding added to the top (start of H axis).
        pad_bottom (int): Padding added to the bottom (end of H axis).
        pad_left (int): Padding added to the left (start of W axis).
        pad_right (int): Padding added to the right (end of W axis).
        h_axis (int): Index of the height axis (Y).
        w_axis (int): Index of the width axis (X).
        border_mode (int): OpenCV border mode.
        pad_value (float | Sequence[float]): Value for constant padding.

    Returns:
        np.ndarray: Padded array.

    Raises:
        ValueError: If border_mode is unsupported or axis indices are out of bounds.

    """
    ndim = arr.ndim
    if not (0 <= h_axis < ndim and 0 <= w_axis < ndim):
        raise ValueError(f"Axis indices {h_axis}, {w_axis} are out of bounds for array with {ndim} dimensions.")
    if h_axis == w_axis:
        raise ValueError(f"Height axis {h_axis} and width axis {w_axis} cannot be the same.")

    mode_map = {
        cv2.BORDER_CONSTANT: "constant",
        cv2.BORDER_REPLICATE: "edge",
        cv2.BORDER_REFLECT: "reflect",
        cv2.BORDER_REFLECT_101: "symmetric",
        cv2.BORDER_WRAP: "wrap",
    }
    if border_mode not in mode_map:
        raise ValueError(f"Unsupported border_mode: {border_mode}")
    np_mode = mode_map[border_mode]

    pad_width = [(0, 0)] * ndim  # Initialize padding for all dimensions
    pad_width[h_axis] = (pad_top, pad_bottom)
    pad_width[w_axis] = (pad_left, pad_right)

    # Initialize kwargs with mode
    kwargs: dict[str, Any] = {"mode": np_mode}
    # Add constant_values only if mode is constant
    if np_mode == "constant":
        kwargs["constant_values"] = pad_value

    return np.pad(arr, pad_width, **kwargs)
