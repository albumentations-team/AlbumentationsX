"""Transforms that combine multiple images and their associated annotations.

This module contains transformations that take multiple input sources (e.g., a primary image
and additional images provided via metadata) and combine them into a single output.
Examples include overlaying elements (`OverlayElements`) or creating complex compositions
like `Mosaic`.
"""

import random
from collections.abc import Sequence
from copy import deepcopy
from typing import Annotated, Any, Literal, cast

import cv2
import numpy as np
from pydantic import AfterValidator, model_validator
from typing_extensions import Self

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.mixing import functional as fmixing
from albumentations.core.bbox_utils import (
    BboxProcessor,
    check_bboxes,
    convert_bboxes_from_albumentations,
    convert_bboxes_to_albumentations,
    denormalize_bboxes,
    filter_bboxes,
)
from albumentations.core.keypoints_utils import KeypointsProcessor
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.type_definitions import LENGTH_RAW_BBOX, ImageType, Targets

__all__ = ["CopyAndPaste", "Mosaic", "OverlayElements"]


class OverlayElements(DualTransform):
    """Apply overlay images/masks onto an input image (e.g. stickers, logos). Optional bboxes
    and masks for placement. Uses metadata_key.

    Args:
        metadata_key (str): Additional target key for metadata. Default `overlay_metadata`.
        p (float): Probability of applying the transformation. Default: 0.5.

    Possible Metadata Fields:
        - image (ImageType): The overlay image to be applied. This is a required field.
        - bbox (list[int]): The bounding box specifying the region where the overlay should be applied. It should
                            contain four floats: [y_min, x_min, y_max, x_max]. If `label_id` is provided, it should
                            be appended as the fifth element in the bbox. BBox should be in Albumentations format,
                            that is the same as normalized Pascal VOC format
                            [x_min / width, y_min / height, x_max / width, y_max / height]
        - mask (np.ndarray): An optional mask that defines the non-rectangular region of the overlay image. If not
                             provided, the entire overlay image is used.
        - mask_id (int): An optional identifier for the mask. If provided, the regions specified by the mask will
                         be labeled with this identifier in the output mask.

    Targets:
        image, mask

    Image types:
        uint8, float32

    References:
        doc-augmentation: https://github.com/danaaubakirova/doc-augmentation

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Prepare primary data (base image and mask)
        >>> image = np.zeros((300, 300, 3), dtype=np.uint8)
        >>> mask = np.zeros((300, 300), dtype=np.uint8)
        >>>
        >>> # 1. Create a simple overlay image (a red square)
        >>> overlay_image1 = np.zeros((50, 50, 3), dtype=np.uint8)
        >>> overlay_image1[:, :, 0] = 255  # Red color
        >>>
        >>> # 2. Create another overlay with a mask (a blue circle with transparency)
        >>> overlay_image2 = np.zeros((80, 80, 3), dtype=np.uint8)
        >>> overlay_image2[:, :, 2] = 255  # Blue color
        >>> overlay_mask2 = np.zeros((80, 80), dtype=np.uint8)
        >>> # Create a circular mask
        >>> center = (40, 40)
        >>> radius = 30
        >>> for i in range(80):
        ...     for j in range(80):
        ...         if (i - center[0])**2 + (j - center[1])**2 < radius**2:
        ...             overlay_mask2[i, j] = 255
        >>>
        >>> # 3. Create an overlay with both bbox and mask_id
        >>> overlay_image3 = np.zeros((60, 120, 3), dtype=np.uint8)
        >>> overlay_image3[:, :, 1] = 255  # Green color
        >>> # Create a rectangular mask with rounded corners
        >>> overlay_mask3 = np.zeros((60, 120), dtype=np.uint8)
        >>> cv2.rectangle(overlay_mask3, (10, 10), (110, 50), 255, -1)
        >>>
        >>> # Create the metadata list - each item is a dictionary with overlay information
        >>> overlay_metadata = [
        ...     {
        ...         'image': overlay_image1,
        ...         # No bbox provided - will be placed randomly
        ...     },
        ...     {
        ...         'image': overlay_image2,
        ...         'bbox': [0.6, 0.1, 0.9, 0.4],  # Normalized coordinates [x_min, y_min, x_max, y_max]
        ...         'mask': overlay_mask2,
        ...         'mask_id': 1  # This overlay will update the mask with id 1
        ...     },
        ...     {
        ...         'image': overlay_image3,
        ...         'bbox': [0.1, 0.7, 0.5, 0.9],  # Bottom left placement
        ...         'mask': overlay_mask3,
        ...         'mask_id': 2  # This overlay will update the mask with id 2
        ...     }
        ... ]
        >>>
        >>> # Create the transform
        >>> transform = A.Compose([
        ...     A.OverlayElements(p=1.0),
        ... ])
        >>>
        >>> # Apply the transform
        >>> result = transform(
        ...     image=image,
        ...     mask=mask,
        ...     overlay_metadata=overlay_metadata  # Pass metadata using the default key
        ... )
        >>>
        >>> # Get results with overlays applied
        >>> result_image = result['image']  # Image with the three overlays applied
        >>> result_mask = result['mask']    # Mask with regions labeled using the mask_id values
        >>>
        >>> # Let's verify the mask contains the specified mask_id values
        >>> has_mask_id_1 = np.any(result_mask == 1)  # Should be True
        >>> has_mask_id_2 = np.any(result_mask == 2)  # Should be True

    """

    _targets = (Targets.IMAGE, Targets.MASK)

    class InitSchema(BaseTransformInitSchema):
        metadata_key: str

    def __init__(
        self,
        metadata_key: str = "overlay_metadata",
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.metadata_key = metadata_key

    @property
    def targets_as_params(self) -> list[str]:
        return [self.metadata_key]

    @staticmethod
    def preprocess_metadata(
        metadata: dict[str, Any],
        img_shape: tuple[int, int],
        random_state: random.Random,
    ) -> dict[str, Any]:
        overlay_image = metadata["image"]
        overlay_height, overlay_width = overlay_image.shape[:2]
        image_height, image_width = img_shape[:2]

        if "bbox" in metadata:
            bbox = metadata["bbox"]
            bbox_np = np.array([bbox])
            check_bboxes(bbox_np)
            denormalized_bbox = denormalize_bboxes(bbox_np, img_shape[:2])[0]

            x_min, y_min, x_max, y_max = (int(x) for x in denormalized_bbox[:4])

            if "mask" in metadata:
                mask = metadata["mask"]
                mask = fgeometric.resize(mask, (y_max - y_min, x_max - x_min), cv2.INTER_NEAREST)
            else:
                mask = np.ones((y_max - y_min, x_max - x_min), dtype=np.uint8)

            overlay_image = fgeometric.resize(overlay_image, (y_max - y_min, x_max - x_min), cv2.INTER_AREA)
            offset = (y_min, x_min)

            if len(bbox) == LENGTH_RAW_BBOX and "bbox_id" in metadata:
                bbox = [x_min, y_min, x_max, y_max, metadata["bbox_id"]]
            else:
                bbox = (x_min, y_min, x_max, y_max, *bbox[4:])
        else:
            if image_height < overlay_height or image_width < overlay_width:
                overlay_image = fgeometric.resize(overlay_image, (image_height, image_width), cv2.INTER_AREA)
                overlay_height, overlay_width = overlay_image.shape[:2]

            mask = metadata["mask"] if "mask" in metadata else np.ones_like(overlay_image, dtype=np.uint8)

            max_x_offset = image_width - overlay_width
            max_y_offset = image_height - overlay_height

            offset_x = random_state.randint(0, max_x_offset)
            offset_y = random_state.randint(0, max_y_offset)

            offset = (offset_y, offset_x)

            bbox = [
                offset_x,
                offset_y,
                offset_x + overlay_width,
                offset_y + overlay_height,
            ]

            if "bbox_id" in metadata:
                bbox = [*bbox, metadata["bbox_id"]]

        result = {
            "overlay_image": overlay_image,
            "overlay_mask": mask,
            "offset": offset,
            "bbox": bbox,
        }

        if "mask_id" in metadata:
            result["mask_id"] = metadata["mask_id"]

        return result

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        metadata = data[self.metadata_key]
        img_shape = params["shape"]

        if isinstance(metadata, list):
            overlay_data = [self.preprocess_metadata(md, img_shape, self.py_random) for md in metadata]
        else:
            overlay_data = [self.preprocess_metadata(metadata, img_shape, self.py_random)]

        return {
            "overlay_data": overlay_data,
        }

    def apply(
        self,
        img: ImageType,
        overlay_data: list[dict[str, Any]],
        **params: Any,
    ) -> ImageType:
        for data in overlay_data:
            overlay_image = data["overlay_image"]
            overlay_mask = data["overlay_mask"]
            offset = data["offset"]
            img = fmixing.copy_and_paste_blend(img, overlay_image, overlay_mask, offset=offset)
        return img

    def apply_to_mask(
        self,
        mask: ImageType,
        overlay_data: list[dict[str, Any]],
        **params: Any,
    ) -> ImageType:
        for data in overlay_data:
            if "mask_id" in data and data["mask_id"] is not None:
                overlay_mask = data["overlay_mask"]
                offset = data["offset"]
                mask_id = data["mask_id"]

                y_min, x_min = offset
                y_max = y_min + overlay_mask.shape[0]
                x_max = x_min + overlay_mask.shape[1]

                mask_section = mask[y_min:y_max, x_min:x_max]
                mask_section[overlay_mask > 0] = mask_id

        return mask


class CopyAndPaste(DualTransform):
    """Paste object instances onto the primary image, updating all annotations (instance masks,
    bboxes, keypoints). Designed for instance segmentation training.

    The user provides a list of object dicts via the metadata key — one dict per object to paste.
    Every object in the list is pasted. Each object is resized from its source image dimensions
    to the target image dimensions before pasting. Existing instances that become sufficiently
    occluded by pasted objects are removed from annotations.

    Note:
        Most Copy-Paste implementations (e.g. detectron2) accept a single donor image with all
        its instance masks and internally sample a random subset of instances to paste, coupling
        donor selection, instance sampling, and pasting into one opaque step. This implementation
        separates those concerns: donor selection and instance selection are done by the user
        externally, and the transform pastes every object in the provided list. One extra line of
        code outside the transform enables deterministic control, class-balanced pasting,
        hard-example mining, and curriculum strategies. The metadata format is `list[dict]`
        (one dict per object), consistent with `Mosaic`.

    Args:
        min_visibility_after_paste (float): Minimum mask area ratio (area_after / area_before) for
            an existing instance to survive after occlusion by pasted objects. Instances whose
            remaining visible area falls below this threshold are removed from masks and bboxes.
            Default: 0.05.
        blend_mode (Literal["hard", "gaussian"]): How to blend pasted pixels. "hard" does direct
            pixel copy (paper default). "gaussian" applies gaussian blur to the alpha mask for
            soft edges at instance boundaries. Default: "hard".
        blend_sigma_range (tuple[float, float]): Sigma range for gaussian blur when
            blend_mode="gaussian". Ignored when blend_mode="hard". Default: (1.0, 3.0).
        metadata_key (str): Key in the Compose call data dict containing the list of object
            dictionaries to paste. Default: "copy_paste_metadata".
        p (float): Probability of applying the transform. Default: 0.5.

    Metadata Format:
        The value at `metadata_key` must be a list of dicts. Each dict represents one object
        to paste and must contain:
            - image (np.ndarray): Source image (H, W, C) containing the object. Required.
            - mask (np.ndarray): Binary **instance** mask (H, W) for this object in the source
              image: pixels to paste from `image`. Required. This is not the same as the
              pipeline `mask` target below (semantic segmentation); entries with no positive
              pixels after resize to the target size are skipped.
            - semantic_mask (np.ndarray): Optional semantic label map (H, W), same shape as
              `image`, aligned pixel-wise. When provided and the pipeline passes a `mask`
              target, pasted pixels copy class ids from this map into the output semantic mask
              inside the paste footprint (see `apply_to_mask`).
            - bbox (np.ndarray | list): Bounding box of the object in the **same coordinate
              format** as `BboxParams.coord_format` declared in `Compose` (e.g. pascal_voc,
              yolo, coco, albumentations). Optional — if absent, a tight box is derived from the
              instance `mask` and converted to that format.
            - keypoints (np.ndarray): Keypoints for the object in the **same format** as
              `KeypointParams.coord_format` declared in `Compose`. Optional.
            - bbox_labels (dict[str, Any]): Label values for the object's bbox, keyed by the
              label field names declared in `BboxParams.label_fields`. Supports multiple
              label fields. E.g. `{"class_id": 3, "is_crowd": 0}`.
            - keypoint_labels (dict[str, Any]): Label values for the object's keypoints, keyed
              by the label field names declared in `KeypointParams.label_fields`. A list
              value is accepted when the object has multiple keypoints.
              E.g. `{"joint_name": "left_eye"}` or `{"visibility": [2, 2]}`.

    Targets:
        image, mask, bboxes, keypoints

    Keypoints vs instance masks:
        When the pipeline supplies instance masks as `masks` (N, H, W) and
        `paste_surviving_indices` is computed from them, primary keypoints are filtered only if
        `keypoints.shape[0]` equals N (one row per instance, same order as stacked masks).
        Otherwise existing keypoints are left unchanged and pasted keypoints are still appended.

    Image types:
        uint8, float32

    Supported bboxes:
        hbb

    Reference:
        Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation: https://arxiv.org/abs/2012.07177

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Primary data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> instance_masks = np.zeros((2, 100, 100), dtype=np.uint8)
        >>> instance_masks[0, 10:30, 10:30] = 1
        >>> instance_masks[1, 50:80, 50:80] = 1
        >>> bboxes = np.array([[10, 10, 30, 30], [50, 50, 80, 80]], dtype=np.float32)
        >>> class_labels = [1, 2]
        >>>
        >>> # User selects which objects to paste (e.g. from another dataset sample)
        >>> donor_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> obj_mask = np.zeros((100, 100), dtype=np.uint8)
        >>> obj_mask[40:60, 40:60] = 1
        >>>
        >>> transform = A.Compose([
        ...     A.CopyAndPaste(
        ...         min_visibility_after_paste=0.05,
        ...         p=1.0,
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['class_labels']))
        >>>
        >>> result = transform(
        ...     image=image,
        ...     masks=instance_masks,
        ...     bboxes=bboxes,
        ...     class_labels=class_labels,
        ...     copy_paste_metadata=[
        ...         {
        ...             'image': donor_image,
        ...             'mask': obj_mask,
        ...             'bbox': [40, 40, 60, 60],
        ...             'bbox_labels': {'class_labels': 3},
        ...         },
        ...     ],
        ... )
        >>> result_image = result['image']
        >>> result_masks = result['masks']         # (N_surviving + K, H, W)
        >>> result_bboxes = result['bboxes']       # Updated bboxes
        >>> result_labels = result['class_labels'] # Updated labels

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        min_visibility_after_paste: float
        blend_mode: Literal["hard", "gaussian"]
        blend_sigma_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        metadata_key: str

    def __init__(
        self,
        min_visibility_after_paste: float = 0.05,
        blend_mode: Literal["hard", "gaussian"] = "hard",
        blend_sigma_range: tuple[float, float] = (1.0, 3.0),
        metadata_key: str = "copy_paste_metadata",
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.min_visibility_after_paste = min_visibility_after_paste
        self.blend_mode = blend_mode
        self.blend_sigma_range = blend_sigma_range
        self.metadata_key = metadata_key

    @property
    def targets_as_params(self) -> list[str]:
        return [self.metadata_key]

    @staticmethod
    def _instance_masks_to_3d(masks: Any) -> np.ndarray | None:
        """Normalize masks to (N, H, W) for CopyAndPaste visibility from a stacked ndarray (4D ok) or a sequence of
        per-instance (H, W) arrays.
        """
        if masks is None:
            return None
        if isinstance(masks, np.ndarray):
            if masks.size == 0:
                return None
            return masks.squeeze(-1) if masks.ndim == 4 else masks
        if isinstance(masks, Sequence) and not isinstance(masks, (str, bytes, np.ndarray)):
            if len(masks) == 0:
                return None
            return np.stack([np.asarray(m) for m in masks], axis=0)
        return None

    def _compute_surviving_indices(
        self,
        data: dict[str, Any],
        paste_union_mask: np.ndarray,
    ) -> tuple[np.ndarray | None, int | None]:
        """Return surviving indices and n_instances from stacked masks vs paste visibility,
        or (None, None) without instance masks.

        Compares each instance mask to the opaque paste footprint for visibility ratios.
        n_instances is the stacked `masks` axis length and matches keypoint row count when filtering survivors.
        """
        masks_3d = self._instance_masks_to_3d(data.get("masks"))
        if masks_3d is None:
            return None, None

        n_instances = int(masks_3d.shape[0])
        visibility = fmixing.compute_instance_visibility(masks_3d, paste_union_mask)
        surviving = np.where(visibility >= self.min_visibility_after_paste)[0]
        return surviving, n_instances

    @staticmethod
    def _resize_mask_to_target(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        if mask.shape[0] != target_shape[0] or mask.shape[1] != target_shape[1]:
            return fgeometric.resize(mask, target_shape, cv2.INTER_NEAREST)
        return mask

    @staticmethod
    def _derive_bbox_from_mask(mask: np.ndarray, bbox_processor: BboxProcessor) -> np.ndarray:
        """Derive a tight HBB from a binary mask in BboxParams.coord_format via internal Pascal VOC pixels; OBB appends
        angle zero.
        """
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        height, width = mask.shape[:2]
        if not np.any(rows):
            pascal_px = np.zeros((1, 4), dtype=np.float32)
        else:
            row_indices = np.where(rows)[0]
            col_indices = np.where(cols)[0]
            pascal_px = np.array(
                [
                    [
                        float(col_indices[0]),
                        float(row_indices[0]),
                        float(col_indices[-1] + 1),
                        float(row_indices[-1] + 1),
                    ],
                ],
                dtype=np.float32,
            )

        bbox_type = bbox_processor.params.bbox_type
        if bbox_type == "obb" and pascal_px.shape[1] == 4:
            pascal_px = np.column_stack([pascal_px, np.zeros(1, dtype=np.float32)])

        alb = convert_bboxes_to_albumentations(
            pascal_px,
            "pascal_voc",
            (height, width),
            bbox_type,
            check_validity=False,
        )
        coord_format = bbox_processor.params.coord_format
        if coord_format == "albumentations":
            return alb.reshape(-1)

        converted = convert_bboxes_from_albumentations(
            alb,
            coord_format,
            (height, width),
            bbox_type,
            check_validity=False,
        )
        return converted.reshape(-1)

    @staticmethod
    def _keypoint_label_values_for_item(
        val: Any,
        num_keypoints: int,
        field: str,
        item_idx: int,
    ) -> list[Any]:
        if isinstance(val, np.ndarray):
            field_values = np.asarray(val).reshape(-1).tolist()
        elif isinstance(val, list):
            field_values = val
        else:
            field_values = [val] * num_keypoints
        if len(field_values) != num_keypoints:
            raise ValueError(
                f"CopyAndPaste: keypoint label field '{field}' must have one value per keypoint "
                f"for pasted object at index {item_idx}; got {len(field_values)} for "
                f"{num_keypoints} keypoints.",
            )
        return field_values

    def _prepare_pasted_bboxes(
        self,
        items: list[dict[str, Any]],
        pasted_masks: np.ndarray,
        target_image: np.ndarray,
        instance_ids: list[int],
    ) -> np.ndarray | None:
        """Build a preprocessed bounding-box array from all pasted object items, encoding extra
        label fields through the bbox processor.

        Label values are read from `bbox_labels` in each item — a dict mapping
        label field name to the scalar value for that object, e.g.
        `{"class_id": 3, "is_crowd": 0}`. With instance binding, internal `_ibl_bbox_*` fields
        map to user keys; `_bbox_instance_id` is taken from `instance_ids`.
        """
        bbox_processor = cast("BboxProcessor", self.get_processor("bboxes"))
        label_fields = (bbox_processor.params.label_fields or []) if bbox_processor else []

        all_bboxes: list[np.ndarray] = []
        all_labels: dict[str, list[Any]] = {field: [] for field in label_fields}

        for idx, item in enumerate(items):
            bbox = (
                np.asarray(item["bbox"], dtype=np.float32).ravel()
                if "bbox" in item
                else self._derive_bbox_from_mask(pasted_masks[idx], bbox_processor)
            )
            all_bboxes.append(bbox)

            item_labels: dict[str, Any] = item.get("bbox_labels", {})
            for field in label_fields:
                if field == "_bbox_instance_id":
                    all_labels[field].append(instance_ids[idx])
                    continue
                if field.startswith("_ibl_bbox_"):
                    user_key = field.removeprefix("_ibl_bbox_")
                    if user_key not in item_labels:
                        raise ValueError(
                            f"CopyAndPaste: missing bbox label field '{user_key}' for pasted object at index {idx}. "
                            "Provide `bbox_labels` with every field declared in BboxParams.label_fields.",
                        )
                    all_labels[field].append(item_labels[user_key])
                    continue
                if field not in item_labels:
                    raise ValueError(
                        f"CopyAndPaste: missing bbox label field '{field}' for pasted object at index {idx}. "
                        "Provide `bbox_labels` with every field declared in BboxParams.label_fields.",
                    )
                all_labels[field].append(item_labels[field])

        num_bboxes = len(all_bboxes)
        for field_name, field_values in all_labels.items():
            if len(field_values) != num_bboxes:
                raise ValueError(
                    f"CopyAndPaste: label field '{field_name}' has {len(field_values)} values for "
                    f"{num_bboxes} pasted bboxes; expected one label per bbox.",
                )

        donor_item: dict[str, Any] = {
            "image": target_image,
            "bboxes": np.array(all_bboxes, dtype=np.float32),
        }
        for field in label_fields:
            donor_item[field] = all_labels[field]

        return fmixing.preprocess_copy_paste_annotations(donor_item, bbox_processor, "bboxes")

    def _collect_labels_for_one_pasted_keypoint_item(
        self,
        item_idx: int,
        item: dict[str, Any],
        kp_label_fields: Sequence[str],
        instance_ids: list[int],
        num_keypoints: int,
        all_labels: dict[str, list[Any]],
    ) -> None:
        item_labels: dict[str, Any] = item.get("keypoint_labels", {})
        for field in kp_label_fields:
            if field == "_kp_instance_id":
                all_labels[field].extend([instance_ids[item_idx]] * num_keypoints)
                continue
            if field.startswith("_ibl_kp_"):
                user_key = field.removeprefix("_ibl_kp_")
                if user_key not in item_labels:
                    raise ValueError(
                        f"CopyAndPaste: missing keypoint label field '{user_key}' for pasted object at "
                        f"index {item_idx}. Provide `keypoint_labels` with every field declared in "
                        "KeypointParams.label_fields.",
                    )
                val = item_labels[user_key]
                field_values = self._keypoint_label_values_for_item(val, num_keypoints, user_key, item_idx)
                all_labels[field].extend(field_values)
                continue
            if field not in item_labels:
                raise ValueError(
                    f"CopyAndPaste: missing keypoint label field '{field}' for pasted object at "
                    f"index {item_idx}. Provide `keypoint_labels` with every field declared in "
                    "KeypointParams.label_fields.",
                )
            val = item_labels[field]
            field_values = self._keypoint_label_values_for_item(val, num_keypoints, field, item_idx)
            all_labels[field].extend(field_values)

    def _prepare_pasted_keypoints(
        self,
        items: list[dict[str, Any]],
        target_image: np.ndarray,
        instance_ids: list[int],
    ) -> np.ndarray | None:
        """Build a preprocessed keypoints array from all pasted object items, encoding label
        fields through the keypoint processor.

        Label values are read from `keypoint_labels` in each item — a dict mapping
        label field name to scalar or list of values for that object, e.g.
        `{"joint_name": "left_eye"}` or `{"visibility": [2, 2]}`. With instance binding,
        `_kp_instance_id` is replicated per keypoint row from `instance_ids`.
        """
        keypoint_processor = cast("KeypointsProcessor", self.get_processor("keypoints"))
        kp_label_fields = (keypoint_processor.params.label_fields or []) if keypoint_processor else []

        all_kps: list[np.ndarray] = []
        all_labels: dict[str, list[Any]] = {field: [] for field in kp_label_fields}

        for item_idx, item in enumerate(items):
            if "keypoints" not in item:
                continue
            raw = np.asarray(item["keypoints"], dtype=np.float32)
            if raw.ndim == 1:
                raw = raw[np.newaxis]
            num_keypoints = raw.shape[0]
            all_kps.append(raw)
            self._collect_labels_for_one_pasted_keypoint_item(
                item_idx,
                item,
                kp_label_fields,
                instance_ids,
                num_keypoints,
                all_labels,
            )

        if not all_kps:
            return None

        concatenated_keypoints = np.concatenate(all_kps, axis=0)
        total_keypoints = concatenated_keypoints.shape[0]

        for field in kp_label_fields:
            if len(all_labels[field]) != total_keypoints:
                raise ValueError(
                    f"CopyAndPaste: keypoint label field '{field}' has {len(all_labels[field])} values "
                    f"for {total_keypoints} concatenated keypoints.",
                )

        donor_item: dict[str, Any] = {
            "image": target_image,
            "keypoints": concatenated_keypoints,
        }
        for field in kp_label_fields:
            donor_item[field] = all_labels[field]

        return fmixing.preprocess_copy_paste_annotations(donor_item, keypoint_processor, "keypoints")

    def _gather_valid_copy_paste_items(
        self,
        data: dict[str, Any],
        target_shape: tuple[int, int],
    ) -> tuple[list[dict[str, Any]], list[np.ndarray], np.ndarray] | None:
        metadata = data.get(self.metadata_key)
        if not isinstance(metadata, list) or not metadata:
            return None

        valid_items: list[dict[str, Any]] = []
        pasted_masks_list: list[np.ndarray] = []
        composite_image = data["image"].copy()

        for item in metadata:
            if not isinstance(item, dict) or "image" not in item or "mask" not in item:
                continue
            src_mask = self._resize_mask_to_target(item["mask"], target_shape)
            if not np.any(src_mask > 0):
                continue
            src_image = item["image"]
            if src_image.shape[0] != target_shape[0] or src_image.shape[1] != target_shape[1]:
                src_image = fgeometric.resize(src_image, target_shape, cv2.INTER_AREA)
            valid_items.append(item)
            pasted_masks_list.append(src_mask)
            mask_bool = src_mask > 0
            composite_image[mask_bool] = src_image[mask_bool]

        if not valid_items:
            return None
        return valid_items, pasted_masks_list, composite_image

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        target_shape = params["shape"][:2]
        gathered = self._gather_valid_copy_paste_items(data, target_shape)
        if gathered is None:
            return self._no_op_params()

        valid_items, pasted_masks_list, composite_image = gathered
        pasted_masks = np.stack(pasted_masks_list, axis=0)
        paste_union_mask = np.any(pasted_masks > 0, axis=0)

        blend_sigma = self.py_random.uniform(*self.blend_sigma_range)
        alpha = fmixing.create_copy_paste_alpha(pasted_masks, self.blend_mode, blend_sigma)

        surviving_indices, paste_primary_instance_count = self._compute_surviving_indices(data, paste_union_mask)

        if surviving_indices is not None and surviving_indices.size > 0:
            next_paste_instance_id = int(np.max(surviving_indices)) + 1
        else:
            next_paste_instance_id = 0
        paste_instance_ids = [next_paste_instance_id + k for k in range(len(valid_items))]

        pasted_bboxes = (
            self._prepare_pasted_bboxes(valid_items, pasted_masks, composite_image, paste_instance_ids)
            if "bboxes" in data
            else None
        )

        pasted_keypoints = (
            self._prepare_pasted_keypoints(valid_items, composite_image, paste_instance_ids)
            if "keypoints" in data
            else None
        )

        donor_mask = None
        for item in valid_items:
            if "semantic_mask" in item:
                item_mask = self._resize_mask_to_target(item["semantic_mask"], target_shape)
                if donor_mask is None:
                    donor_mask = np.zeros(target_shape, dtype=item_mask.dtype)
                paste_region = (
                    item["mask"]
                    if item["mask"].shape == target_shape
                    else self._resize_mask_to_target(
                        item["mask"],
                        target_shape,
                    )
                )
                donor_mask[paste_region > 0] = item_mask[paste_region > 0]

        return {
            "paste_donor_image": composite_image,
            "paste_alpha": alpha,
            "paste_instance_masks": pasted_masks,
            "paste_surviving_indices": surviving_indices,
            "paste_primary_instance_count": paste_primary_instance_count,
            "paste_bboxes": pasted_bboxes,
            "paste_keypoints": pasted_keypoints,
            "paste_donor_mask": donor_mask,
        }

    @staticmethod
    def _no_op_params() -> dict[str, Any]:
        return {
            "paste_donor_image": None,
            "paste_alpha": None,
            "paste_instance_masks": None,
            "paste_surviving_indices": None,
            "paste_primary_instance_count": None,
            "paste_bboxes": None,
            "paste_keypoints": None,
            "paste_donor_mask": None,
        }

    def apply(
        self,
        img: ImageType,
        paste_donor_image: np.ndarray | None,
        paste_alpha: np.ndarray | None,
        **params: Any,
    ) -> ImageType:
        if paste_donor_image is None or paste_alpha is None:
            return img
        return fmixing.blend_images_using_alpha(img, paste_donor_image, paste_alpha)

    def apply_to_mask(
        self,
        mask: ImageType,
        paste_alpha: np.ndarray | None,
        paste_donor_mask: np.ndarray | None,
        **params: Any,
    ) -> ImageType:
        if paste_alpha is None:
            return mask
        if paste_donor_mask is not None:
            result = mask.copy()
            paste_instance_masks = params.get("paste_instance_masks")
            if paste_instance_masks is not None:
                paste_region = np.any(paste_instance_masks > 0, axis=0)
            else:
                paste_region = paste_alpha > 0
            donor = paste_donor_mask
            if result.ndim > donor.ndim:
                donor = donor[..., np.newaxis]
            if result.ndim > paste_region.ndim:
                paste_region = paste_region[..., np.newaxis]
            result[paste_region] = donor[paste_region]
            return result
        return mask

    def apply_to_masks(
        self,
        masks: ImageType,
        paste_alpha: np.ndarray | None,
        paste_instance_masks: np.ndarray | None,
        paste_surviving_indices: np.ndarray | None,
        **params: Any,
    ) -> ImageType:
        if paste_alpha is None or paste_instance_masks is None:
            return masks

        if isinstance(masks, (list, tuple)):
            if len(masks) == 0:
                return paste_instance_masks
            masks = np.stack([np.asarray(m) for m in masks], axis=0)

        pasted = paste_instance_masks
        if masks.ndim == 4 and pasted.ndim == 3:
            pasted = pasted[..., np.newaxis]

        if masks.size == 0:
            return pasted

        paste_region = np.any(paste_instance_masks > 0, axis=0)

        surviving = masks[paste_surviving_indices].copy() if paste_surviving_indices is not None else masks.copy()

        if surviving.ndim == 4:
            paste_region_4d = paste_region[np.newaxis, :, :, np.newaxis]
            np.putmask(surviving, np.broadcast_to(paste_region_4d, surviving.shape), 0)
        else:
            paste_region_3d = paste_region[np.newaxis, :, :]
            np.putmask(surviving, np.broadcast_to(paste_region_3d, surviving.shape), 0)

        return np.concatenate([surviving, pasted], axis=0)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        paste_surviving_indices: np.ndarray | None,
        paste_bboxes: np.ndarray | None,
        paste_alpha: np.ndarray | None,
        **params: Any,
    ) -> np.ndarray:
        if paste_alpha is None:
            return bboxes

        bbox_processor = cast("BboxProcessor", self.get_processor("bboxes"))
        bbox_label_fields = bbox_processor.params.label_fields or []

        if paste_surviving_indices is not None and bboxes.size > 0:
            if "_bbox_instance_id" in bbox_label_fields:
                n_lf = len(bbox_label_fields)
                id_col = bboxes.shape[1] - n_lf + bbox_label_fields.index("_bbox_instance_id")
                inst_col = bboxes[:, id_col].astype(np.int64, copy=False)
                keep = np.isin(inst_col, paste_surviving_indices)
                surviving_bboxes = bboxes[keep]
            else:
                surviving_bboxes = bboxes[paste_surviving_indices]
        else:
            surviving_bboxes = bboxes

        if paste_bboxes is not None and paste_bboxes.size > 0:
            if surviving_bboxes.size == 0:
                return paste_bboxes
            return np.concatenate([surviving_bboxes, paste_bboxes], axis=0)

        return surviving_bboxes

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        paste_alpha: np.ndarray | None,
        paste_keypoints: np.ndarray | None,
        **params: Any,
    ) -> np.ndarray:
        if paste_alpha is None:
            return keypoints

        paste_surviving_indices = params.get("paste_surviving_indices")
        paste_primary_instance_count = params.get("paste_primary_instance_count")
        keypoint_processor = cast("KeypointsProcessor", self.get_processor("keypoints"))
        kp_label_fields = keypoint_processor.params.label_fields or []

        surviving_keypoints = keypoints
        if paste_surviving_indices is not None and keypoints.size > 0:
            if "_kp_instance_id" in kp_label_fields:
                n_kf = len(kp_label_fields)
                id_col = keypoints.shape[1] - n_kf + kp_label_fields.index("_kp_instance_id")
                inst_col = keypoints[:, id_col].astype(np.int64, copy=False)
                keep = np.isin(inst_col, paste_surviving_indices)
                surviving_keypoints = keypoints[keep]
            else:
                aligned = (
                    paste_primary_instance_count is not None and keypoints.shape[0] == paste_primary_instance_count
                )
                if aligned:
                    survivor_idx = np.asarray(paste_surviving_indices)
                    if survivor_idx.size == 0:
                        surviving_keypoints = keypoints[:0]
                    elif int(survivor_idx.max()) < keypoints.shape[0] and int(survivor_idx.min()) >= 0:
                        surviving_keypoints = keypoints[survivor_idx]

        if paste_keypoints is not None and paste_keypoints.size > 0:
            if surviving_keypoints.size == 0:
                return paste_keypoints
            return np.concatenate([surviving_keypoints, paste_keypoints], axis=0)

        return surviving_keypoints


class Mosaic(DualTransform):
    """Combine multiple images and annotations into one image via a mosaic grid. Uses metadata
    for additional images; common in object detection training.

    Mosaic creates a grid of images by placing the primary image and additional images from metadata
    into cells of a larger canvas, then crops a region to produce the final output. This is commonly
    used in object detection training to increase data diversity and help models learn to detect
    objects at different scales and contexts.

    The transform takes a primary input image (and its annotations) and combines it with
    additional images/annotations provided via metadata. It calculates the geometry for
    a mosaic grid, selects additional items, preprocesses annotations consistently
    (handling label encoding updates), applies geometric transformations, and assembles
    the final output.

    Args:
        grid_yx (tuple[int, int]): The number of rows (y) and columns (x) in the mosaic grid.
            Determines the maximum number of images involved (grid_yx[0] * grid_yx[1]).
            Default: (2, 2).
        target_size (tuple[int, int]): The desired output (height, width) for the final mosaic image.
            after cropping the mosaic grid.
        cell_shape (tuple[int, int]): cell shape of each cell in the mosaic grid.
        fit_mode (Literal['cover', 'contain']): How to fit images into mosaic cells.
            - "cover": Scale image to fill the entire cell, potentially cropping parts.
            - "contain": Scale image to fit entirely within the cell, potentially adding padding.
            Default: "cover".
        metadata_key (str): Key in the input dictionary specifying the list of additional data dictionaries
            for the mosaic. Each dictionary in the list should represent one potential additional item.
            Expected keys: 'image' (required, np.ndarray), and optionally 'mask' (np.ndarray),
            'masks' (np.ndarray, stacked instance masks), 'bboxes' (np.ndarray), 'keypoints' (np.ndarray),
            and label fields supplied via the `bbox_labels` and `keypoint_labels` wrapper dicts
            (see Metadata Format below). Default: "mosaic_metadata".
        center_range (tuple[float, float]): Range [0.0-1.0] to sample the center point of the mosaic view
            relative to the valid central region of the conceptual large grid. This affects which parts
            of the assembled grid are visible in the final crop. Default: (0.3, 0.7).
        interpolation (int): OpenCV interpolation flag used for resizing images during geometric processing.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (int): OpenCV interpolation flag used for resizing masks during geometric processing.
            Default: cv2.INTER_NEAREST.
        fill (tuple[float, ...] | float): Value used for padding images if needed during geometric processing.
            Default: 0.
        fill_mask (tuple[float, ...] | float): Value used for padding masks if needed during geometric processing.
            Default: 0.
        p (float): Probability of applying the transform. Default: 0.5.

    Workflow (`get_params_dependent_on_data`):
        1. Calculate Geometry & Visible Cells: Determine which grid cells are visible in the final
           `target_size` crop and their placement coordinates on the output canvas.
        2. Validate Raw Additional Metadata: Filter the list provided via `metadata_key`,
           keeping only valid items (dicts with an 'image' key).
        3. Select Subset of Raw Additional Metadata: Choose a subset of the valid raw items based
           on the number of visible cells requiring additional data.
        4. Preprocess Selected Raw Additional Items: Preprocess bboxes/keypoints for the *selected*
           additional items *only*. This uses shared processors from `Compose`, updating their
           internal state (e.g., `LabelEncoder`) based on labels in these selected items.
        5. Prepare Primary Data: Extract preprocessed primary data fields from the input `data` dictionary
            into a `primary` dictionary.
        6. Determine & Perform Replication: If fewer additional items were selected than needed,
           replicate the preprocessed primary data as required.
        7. Combine Final Items: Create the list of all preprocessed items (primary, selected additional,
           replicated primary) that will be used.
        8. Assign Items to VISIBLE Grid Cells
        9. Process Geometry & Shift Coordinates: For each assigned item:
            a. Apply geometric transforms to image/mask based on `fit_mode`:
               - "cover": Resize to smallest dimension covering the cell, then crop to cell size
               - "contain": Resize to largest dimension fitting in the cell, then pad to cell size
            b. Apply geometric shift to the *preprocessed* bboxes/keypoints based on cell placement.
       10. Return Parameters: Return the processed cell data (image, mask, shifted bboxes, shifted kps)
           keyed by placement coordinates.

    Label Handling:
        - The transform relies on `bbox_processor` and `keypoint_processor` provided by `Compose`.
        - `Compose.preprocess` initially fits the processors' `LabelEncoder` on the primary data.
        - This transform (`Mosaic`) preprocesses the *selected* additional raw items using the same
          processors. If new labels are found, the shared `LabelEncoder` state is updated via its
          `update` method.
        - `Compose.postprocess` uses the final updated encoder state to decode all labels present
          in the mosaic output for the current `Compose` call.
        - The encoder state is transient per `Compose` call.

    Note:
        If fewer additional images are provided than needed to fill the grid, the primary image
        will be replicated to fill the remaining cells. For example, with a 2x2 grid, if only
        one additional image is provided, the mosaic will contain the primary image in two cells
        and the additional image in one cell, with one visible cell selected from these three.
        Stacked instance masks on the `masks` key (N, H, W) are transformed via `apply_to_masks` like
        other DualTransforms; `_targets` only lists `Targets` enum values (no `Targets.MASKS`).

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Supported bboxes:
        hbb, obb

    Reference:
        YOLOv4: Optimal Speed and Accuracy of Object Detection: https://arxiv.org/pdf/2004.10934

    Metadata Format:
        Each dict in the metadata list represents one additional image and must contain:
            - image (np.ndarray): Additional image. Required.
            - mask (np.ndarray): Semantic mask for the additional image. Optional.
            - masks (np.ndarray): Stacked instance masks (N, H, W) for the additional image.
              Optional; same geometry as image. Use with instance_binding / pipeline masks target.
            - bboxes (np.ndarray): Bounding boxes in the **same coordinate format** as
              `BboxParams.coord_format` declared in `Compose`. Optional.
            - keypoints (np.ndarray): Keypoints in the **same format** as
              `KeypointParams.coord_format` declared in `Compose`. Optional.
            - bbox_labels (dict[str, Any]): Label lists for bboxes, keyed by label field name
              as declared in `BboxParams.label_fields`. Each value must be a list with one
              entry per bbox. E.g. `{"class_id": [3, 7], "is_crowd": [0, 1]}`.
            - keypoint_labels (dict[str, Any]): Label lists for keypoints, keyed by label
              field name as declared in `KeypointParams.label_fields`. Each value must be a
              list with one entry per keypoint. E.g. `{"joint_name": ["left_eye", "nose"]}`.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Prepare primary data
        >>> primary_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> primary_mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> primary_bboxes = np.array([[10, 10, 40, 40], [50, 50, 90, 90]], dtype=np.float32)
        >>> primary_bbox_classes = [1, 2]
        >>> primary_keypoints = np.array([[25, 25], [75, 75]], dtype=np.float32)
        >>> primary_keypoint_classes = ['eye', 'nose']
        >>>
        >>> # Prepare additional images for mosaic.
        >>> # bbox_labels and keypoint_labels are dicts mapping field name -> list of values.
        >>> mosaic_metadata = [
        ...     {
        ...         'image': np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
        ...         'mask': np.random.randint(0, 2, (100, 100), dtype=np.uint8),
        ...         'bboxes': np.array([[20, 20, 60, 60]], dtype=np.float32),
        ...         'bbox_labels': {'bbox_classes': [3]},
        ...         'keypoints': np.array([[40, 40]], dtype=np.float32),
        ...         'keypoint_labels': {'keypoint_classes': ['mouth']},
        ...     },
        ...     {
        ...         'image': np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
        ...         'mask': np.random.randint(0, 2, (100, 100), dtype=np.uint8),
        ...         'bboxes': np.array([[30, 30, 70, 70]], dtype=np.float32),
        ...         'bbox_labels': {'bbox_classes': [4]},
        ...         'keypoints': np.array([[50, 50], [65, 65]], dtype=np.float32),
        ...         'keypoint_labels': {'keypoint_classes': ['eye', 'eye']},
        ...     },
        ... ]
        >>>
        >>> transform = A.Compose([
        ...     A.Mosaic(
        ...         grid_yx=(2, 2),
        ...         target_size=(200, 200),
        ...         cell_shape=(120, 120),
        ...         center_range=(0.4, 0.6),
        ...         fit_mode="cover",
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_classes']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_classes']))
        >>>
        >>> transformed = transform(
        ...     image=primary_image,
        ...     mask=primary_mask,
        ...     bboxes=primary_bboxes,
        ...     bbox_classes=primary_bbox_classes,
        ...     keypoints=primary_keypoints,
        ...     keypoint_classes=primary_keypoint_classes,
        ...     mosaic_metadata=mosaic_metadata,
        ... )
        >>>
        >>> mosaic_image = transformed['image']
        >>> mosaic_bboxes = transformed['bboxes']
        >>> mosaic_bbox_classes = transformed['bbox_classes']
        >>> mosaic_keypoint_classes = transformed['keypoint_classes']

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})

    class InitSchema(BaseTransformInitSchema):
        grid_yx: tuple[int, int]
        target_size: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
        ]
        cell_shape: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
        ]
        metadata_key: str
        center_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float
        fit_mode: Literal["cover", "contain"]

        @model_validator(mode="after")
        def _check_cell_shape(self) -> Self:
            if (
                self.cell_shape[0] * self.grid_yx[0] < self.target_size[0]
                or self.cell_shape[1] * self.grid_yx[1] < self.target_size[1]
            ):
                raise ValueError("Target size should be smaller than cell_shape * grid_yx")
            return self

    def __init__(
        self,
        grid_yx: tuple[int, int] = (2, 2),
        target_size: tuple[int, int] = (512, 512),
        cell_shape: tuple[int, int] = (512, 512),
        center_range: tuple[float, float] = (0.3, 0.7),
        fit_mode: Literal["cover", "contain"] = "cover",
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_LINEAR,
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_NEAREST,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        metadata_key: str = "mosaic_metadata",
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.grid_yx = grid_yx
        self.target_size = target_size

        self.metadata_key = metadata_key
        self.center_range = center_range
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.fill = fill
        self.fill_mask = fill_mask
        self.fit_mode = fit_mode
        self.cell_shape = cell_shape

    @property
    def targets_as_params(self) -> list[str]:
        """Return list of target keys passed as params (e.g. to get_params_dependent_on_data).
        For Mosaic/FMix: metadata key for preprocessed mosaic/mix.

        Returns:
            list[str]: List containing the metadata key name

        """
        return [self.metadata_key]

    def _calculate_geometry(self, data: dict[str, Any]) -> list[tuple[int, int, int, int]]:
        # Step 1: Calculate Geometry & Cell Placements
        center_xy = fmixing.calculate_mosaic_center_point(
            grid_yx=self.grid_yx,
            cell_shape=self.cell_shape,
            target_size=self.target_size,
            center_range=self.center_range,
            py_random=self.py_random,
        )

        self.applied_config = {
            "center_range": center_xy,
        }

        return fmixing.calculate_cell_placements(
            grid_yx=self.grid_yx,
            cell_shape=self.cell_shape,
            target_size=self.target_size,
            center_xy=center_xy,
        )

    def _select_additional_items(self, data: dict[str, Any], num_additional_needed: int) -> list[dict[str, Any]]:
        valid_items = fmixing.filter_valid_metadata(data.get(self.metadata_key), self.metadata_key, data)
        if len(valid_items) > num_additional_needed:
            return self.py_random.sample(valid_items, num_additional_needed)
        return valid_items

    def _preprocess_additional_items(
        self,
        additional_items: list[dict[str, Any]],
        data: dict[str, Any],
    ) -> list[fmixing.ProcessedMosaicItem]:
        if "bboxes" in data or "keypoints" in data:
            bbox_processor = cast("BboxProcessor", self.get_processor("bboxes"))
            keypoint_processor = cast("KeypointsProcessor", self.get_processor("keypoints"))
            return fmixing.preprocess_selected_mosaic_items(additional_items, bbox_processor, keypoint_processor)
        if "masks" in data:
            out: list[fmixing.ProcessedMosaicItem] = []
            for item in additional_items:
                if not isinstance(item, dict) or "image" not in item:
                    continue
                flat_item = fmixing.unpack_label_wrappers(item)
                entry: fmixing.ProcessedMosaicItem = {"image": item["image"]}
                if flat_item.get("mask") is not None:
                    entry["mask"] = flat_item["mask"]
                if flat_item.get("masks") is not None:
                    entry["masks"] = np.copy(np.asarray(flat_item["masks"]))
                out.append(entry)
            return out
        return cast("list[fmixing.ProcessedMosaicItem]", list(additional_items))

    def _prepare_final_items(
        self,
        primary: fmixing.ProcessedMosaicItem,
        additional_items: list[fmixing.ProcessedMosaicItem],
        num_needed: int,
    ) -> list[fmixing.ProcessedMosaicItem]:
        num_replications = max(0, num_needed - len(additional_items))
        replicated = [deepcopy(primary) for _ in range(num_replications)]
        return [primary, *additional_items, *replicated]

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        cell_placements = self._calculate_geometry(data)

        num_cells = len(cell_placements)
        num_additional_needed = max(0, num_cells - 1)

        additional_items = self._select_additional_items(data, num_additional_needed)

        preprocessed_additional = self._preprocess_additional_items(additional_items, data)

        primary = self.get_primary_data(data)
        final_items = self._prepare_final_items(primary, preprocessed_additional, num_additional_needed)

        placement_to_item_index = fmixing.assign_items_to_grid_cells(
            num_items=len(final_items),
            cell_placements=cell_placements,
            py_random=self.py_random,
        )

        processed_cells = fmixing.process_all_mosaic_geometries(
            canvas_shape=self.target_size,
            cell_shape=self.cell_shape,
            placement_to_item_index=placement_to_item_index,
            final_items_for_grid=final_items,
            fill=self.fill,
            fill_mask=self.fill_mask if self.fill_mask is not None else self.fill,
            fit_mode=self.fit_mode,
            interpolation=self.interpolation,
            mask_interpolation=self.mask_interpolation,
        )

        if "bboxes" in data or "keypoints" in data or "masks" in data:
            processed_cells = fmixing.shift_all_coordinates(processed_cells, canvas_shape=self.target_size)
            bbox_proc = self.get_processor("bboxes")
            kp_proc = self.get_processor("keypoints")
            processed_cells = fmixing.remap_mosaic_instance_label_ids(
                processed_cells,
                bbox_proc if isinstance(bbox_proc, BboxProcessor) else None,
                kp_proc if isinstance(kp_proc, KeypointsProcessor) else None,
            )

        result = {"processed_cells": processed_cells, "target_shape": self._get_target_shape(data["image"].shape)}
        if "mask" in data:
            result["target_mask_shape"] = self._get_target_shape(data["mask"].shape)
        if "masks" in data:
            ms = data["masks"].shape
            # Stacked instance masks are (N, H, W); do not treat N as spatial dim.
            if len(ms) >= 3:
                result["target_masks_shape"] = (int(ms[0]), self.target_size[0], self.target_size[1])
            else:
                result["target_masks_shape"] = tuple(self._get_target_shape(ms))
        return result

    @staticmethod
    def get_primary_data(data: dict[str, Any]) -> fmixing.ProcessedMosaicItem:
        """Return a copy of the primary item from data so the original is not mutated. Call from
        Mosaic/FMix to build composed image from primary plus patches.

        Args:
            data (dict[str, Any]): Dictionary containing the primary data.

        Returns:
            fmixing.ProcessedMosaicItem: A copy of the primary data.

        """
        mask = data.get("mask")
        if mask is not None:
            mask = mask.copy()
        bboxes = data.get("bboxes")
        if bboxes is not None:
            bboxes = bboxes.copy()
        keypoints = data.get("keypoints")
        if keypoints is not None:
            keypoints = keypoints.copy()
        masks = data.get("masks")
        if masks is not None:
            masks = masks.copy()
        primary: fmixing.ProcessedMosaicItem = {
            "image": data["image"],
            "mask": mask,
            "bboxes": bboxes,
            "keypoints": keypoints,
        }
        if masks is not None:
            primary["masks"] = masks
        return primary

    def _get_target_shape(self, np_shape: tuple[int, ...]) -> list[int]:
        target_shape = list(np_shape)
        target_shape[0] = self.target_size[0]
        target_shape[1] = self.target_size[1]
        return target_shape

    def apply(
        self,
        img: ImageType,
        processed_cells: dict[tuple[int, int, int, int], dict[str, Any]],
        target_shape: tuple[int, int],
        **params: Any,
    ) -> ImageType:
        return fmixing.assemble_mosaic_from_processed_cells(
            processed_cells=processed_cells,
            target_shape=target_shape,
            dtype=img.dtype,
            data_key="image",
            fill=self.fill,
        )

    def apply_to_mask(
        self,
        mask: ImageType,
        processed_cells: dict[tuple[int, int, int, int], dict[str, Any]],
        target_mask_shape: tuple[int, int],
        **params: Any,
    ) -> ImageType:
        return fmixing.assemble_mosaic_from_processed_cells(
            processed_cells=processed_cells,
            target_shape=target_mask_shape,
            dtype=mask.dtype,
            data_key="mask",
            fill=self.fill_mask,
        )

    def apply_to_masks(
        self,
        masks: ImageType,
        processed_cells: dict[tuple[int, int, int, int], dict[str, Any]],
        target_masks_shape: tuple[int, ...],
        **params: Any,
    ) -> ImageType:
        canvas_hw = (int(target_masks_shape[1]), int(target_masks_shape[2]))
        return fmixing.assemble_mosaic_instance_masks_stack(
            processed_cells=processed_cells,
            canvas_hw=canvas_hw,
            dtype=masks.dtype,
            fill=self.fill_mask,
        )

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,  # Original bboxes - ignored
        processed_cells: dict[tuple[int, int, int, int], dict[str, Any]],
        **params: Any,
    ) -> np.ndarray:
        all_shifted_bboxes = []

        for cell_data in processed_cells.values():
            shifted_bboxes = cell_data.get("bboxes")
            if shifted_bboxes is not None and np.asarray(shifted_bboxes).size > 0:
                all_shifted_bboxes.append(shifted_bboxes)

        bbox_processor = cast("BboxProcessor", self.get_processor("bboxes"))

        if not all_shifted_bboxes:
            # Preserve correct column count for empty result
            if bbox_processor.params.bbox_type == "obb":
                num_cols = max(bboxes.shape[1] if bboxes.ndim > 1 else 5, 5)
            else:
                num_cols = max(bboxes.shape[1] if bboxes.ndim > 1 else 4, 4)
            return np.empty((0, num_cols), dtype=bboxes.dtype)

        # Concatenate (these are absolute pixel coordinates)
        combined_bboxes = np.concatenate(all_shifted_bboxes, axis=0)

        # Apply filtering using processor parameters
        return filter_bboxes(
            combined_bboxes,
            self.target_size,
            bbox_processor.params.bbox_type,
            min_area=bbox_processor.params.min_area,
            min_visibility=bbox_processor.params.min_visibility,
            min_width=bbox_processor.params.min_width,
            min_height=bbox_processor.params.min_height,
            max_accept_ratio=bbox_processor.params.max_accept_ratio,
            clip_after_transform=bbox_processor.params.clip_after_transform,
        )

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,  # Original keypoints - ignored
        processed_cells: dict[tuple[int, int, int, int], dict[str, Any]],
        **params: Any,
    ) -> np.ndarray:
        all_shifted_keypoints = []

        for cell_data in processed_cells.values():
            shifted_keypoints = cell_data.get("keypoints")
            if shifted_keypoints is not None and np.asarray(shifted_keypoints).size > 0:
                all_shifted_keypoints.append(shifted_keypoints)

        if not all_shifted_keypoints:
            return np.empty((0, keypoints.shape[1]), dtype=keypoints.dtype)

        combined_keypoints = np.concatenate(all_shifted_keypoints, axis=0)

        keypoint_processor = self.get_processor("keypoints")
        kp_fields = (
            keypoint_processor.params.label_fields
            if isinstance(keypoint_processor, KeypointsProcessor) and keypoint_processor.params.label_fields
            else []
        )
        if "_kp_instance_id" in kp_fields:
            return combined_keypoints

        target_h, target_w = self.target_size
        valid_indices = (
            (combined_keypoints[:, 0] >= 0)
            & (combined_keypoints[:, 0] < target_w)
            & (combined_keypoints[:, 1] >= 0)
            & (combined_keypoints[:, 1] < target_h)
        )

        return combined_keypoints[valid_indices]
