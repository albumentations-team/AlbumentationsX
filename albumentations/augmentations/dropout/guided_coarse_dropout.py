"""Guided coarse dropout constrained to a caller-supplied binary region."""

from typing import Annotated, Any, cast

import numpy as np
from pydantic import AfterValidator, Field

import albumentations.augmentations.dropout.functional as fdropout
from albumentations.augmentations.dropout.transforms import BaseDropout, BaseDropoutInitSchema, DropoutFillValue
from albumentations.core.bbox_utils import BboxProcessor, denormalize_bboxes, normalize_bboxes
from albumentations.core.invocation import SamplingContext
from albumentations.core.keypoints_utils import KeypointsProcessor
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.type_definitions import ImageType, Targets

__all__ = ["GuidedCoarseDropout"]


class GuidedCoarseDropout(BaseDropout):
    """Apply coarse dropout within a caller-supplied binary region while preserving selected bounding
    boxes and filtering annotations by the actual dropout mask.

    The caller supplies a two-dimensional binary region as top-level metadata. A pixel value
    of True or 1 permits dropout; False or 0 leaves the pixel unchanged.
    Hole centers are sampled uniformly over the entire eligible region, after subtracting
    protected bounding boxes and their margins.

    Args:
        region_key (str): Top-level key containing the binary (H, W) dropout region.
            Default: "dropout_region".
        protected_bbox_labels (list[str | int | float] | None): Labels of boxes to protect.
            String labels use the configured bbox label encoder. Default: None.
        protection_margin (float): Relative expansion applied to every protected box side.
            A margin m expands horizontally by m * box_width and vertically by
            m * box_height. Default: 0.0.
        num_holes_range (tuple[int, int]): Inclusive number of holes sampled per image.
            Default: (1, 1).
        hole_height_range (tuple[float, float]): Hole-height fraction of the full image height.
            Default: (0.05, 0.20).
        hole_width_range (tuple[float, float]): Hole-width fraction of the full image width.
            Default: (0.05, 0.20).
        fill (float | tuple[float, ...] | str): Value used for dropped image pixels.
            Default: 0.
        fill_mask (float | tuple[float, ...] | None): Value used for dropped mask pixels.
            None leaves masks unchanged. Default: None.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Supported bboxes:
        hbb

    Examples:
        >>> import albumentations as A
        >>> import numpy as np
        >>> image = np.full((100, 100, 3), 255, dtype=np.uint8)
        >>> dropout_region = np.zeros((100, 100), dtype=np.uint8)
        >>> dropout_region[20:80, 20:80] = 1
        >>> transform = A.Compose(
        ...     [A.GuidedCoarseDropout(fill=0, p=1.0)],
        ...     bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
        ... )
        >>> result = transform(
        ...     image=image,
        ...     dropout_region=dropout_region,
        ...     bboxes=[[40, 40, 60, 60]],
        ...     labels=["person"],
        ... )

    Notes:
        - The region is aligned metadata and is returned unchanged. Place this transform before
          geometry changes unless the caller has already aligned the region to their coordinates.
        - If no eligible pixels remain, the transform is a no-op.
        - fill="random_uniform" samples one value per original hole, including holes whose
          rectangular footprint is clipped by the eligible region.

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)
    _supported_bbox_types: frozenset[str] = frozenset({"hbb"})

    class InitSchema(BaseDropoutInitSchema):
        region_key: str
        protected_bbox_labels: list[str | int | float] | None
        protection_margin: float = Field(ge=0)
        num_holes_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
            AfterValidator(nondecreasing),
        ]
        hole_height_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        hole_width_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]

    def __init__(
        self,
        region_key: str = "dropout_region",
        protected_bbox_labels: list[str | int | float] | None = None,
        protection_margin: float = 0.0,
        num_holes_range: tuple[int, int] = (1, 1),
        hole_height_range: tuple[float, float] = (0.05, 0.20),
        hole_width_range: tuple[float, float] = (0.05, 0.20),
        fill: DropoutFillValue = 0,
        fill_mask: tuple[float, ...] | float | None = None,
        p: float = 0.5,
    ):
        super().__init__(fill=fill, fill_mask=fill_mask, p=p)
        self.region_key = region_key
        self.protected_bbox_labels = protected_bbox_labels
        self.protection_margin = protection_margin
        self.num_holes_range = num_holes_range
        self.hole_height_range = hole_height_range
        self.hole_width_range = hole_width_range

    @property
    def targets_as_params(self) -> list[str]:
        return [self.region_key]

    def _get_dropout_region(self, data: dict[str, Any], image_shape: tuple[int, int]) -> np.ndarray:
        region = data[self.region_key]
        if not isinstance(region, np.ndarray) or region.ndim != 2 or region.shape != image_shape:
            msg = f"{self.region_key} must be a 2-D array with shape {image_shape}"
            raise ValueError(msg)
        valid_dtype = (
            np.issubdtype(region.dtype, np.bool_)
            or np.issubdtype(region.dtype, np.integer)
            or np.issubdtype(
                region.dtype,
                np.floating,
            )
        )
        if not valid_dtype or not np.all(np.isfinite(region)) or not np.all((region == 0) | (region == 1)):
            msg = f"{self.region_key} must be binary with values 0 or 1"
            raise ValueError(msg)
        return region.astype(bool, copy=False)

    def _get_protected_bboxes(
        self,
        data: dict[str, Any],
        image_shape: tuple[int, int],
    ) -> np.ndarray | None:
        if self.protected_bbox_labels is None:
            return None
        bboxes = data.get("bboxes")
        if bboxes is None or len(bboxes) == 0:
            return None
        bbox_processor = cast("BboxProcessor | None", self.get_processor("bboxes"))
        if bbox_processor is None:
            return None
        if all(isinstance(label, (int, float)) for label in self.protected_bbox_labels):
            protected_labels = np.asarray(self.protected_bbox_labels)
        else:
            label_fields = bbox_processor.params.label_fields
            if label_fields is None:
                raise ValueError("BboxParams.label_fields is required for string protected_bbox_labels")
            metadata = bbox_processor.label_manager.metadata["bboxes"][label_fields[0]]
            if metadata.encoder is None:
                raise ValueError(f"No encoder found for label field {label_fields[0]}")
            try:
                protected_labels = metadata.encoder.transform(self.protected_bbox_labels)
            except KeyError:
                return None
        selected = bboxes[np.isin(bboxes[:, 4], protected_labels), :4]
        return denormalize_bboxes(selected, image_shape) if len(selected) else None

    def _build_protected_mask(self, bboxes: np.ndarray | None, image_shape: tuple[int, int]) -> np.ndarray:
        height, width = image_shape
        protected_mask = np.zeros(image_shape, dtype=bool)
        if bboxes is None:
            return protected_mask
        for x1, y1, x2, y2 in bboxes:
            margin_x = self.protection_margin * (x2 - x1)
            margin_y = self.protection_margin * (y2 - y1)
            left = max(0, int(np.floor(x1 - margin_x)))
            top = max(0, int(np.floor(y1 - margin_y)))
            right = min(width, int(np.ceil(x2 + margin_x)))
            bottom = min(height, int(np.ceil(y2 + margin_y)))
            protected_mask[top:bottom, left:right] = True
        return protected_mask

    @staticmethod
    def _sample_centers_uniform(
        eligible_mask: np.ndarray,
        num_holes: int,
        random_generator: np.random.Generator,
    ) -> np.ndarray:
        ys, xs = np.nonzero(eligible_mask)
        indices = random_generator.integers(0, len(ys), size=num_holes)
        return np.column_stack((xs[indices], ys[indices])).astype(np.int32)

    @staticmethod
    def _centers_to_holes(
        centers: np.ndarray,
        hole_heights: np.ndarray,
        hole_widths: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        height, width = image_shape
        left = np.clip(centers[:, 0] - hole_widths // 2, 0, width)
        top = np.clip(centers[:, 1] - hole_heights // 2, 0, height)
        right = np.clip(left + hole_widths, 0, width)
        bottom = np.clip(top + hole_heights, 0, height)
        return np.column_stack((left, top, right, bottom)).astype(np.int32)

    @staticmethod
    def _rasterize_holes(holes: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(image_shape, dtype=bool)
        for left, top, right, bottom in holes:
            mask[top:bottom, left:right] = True
        return mask

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        sampling: SamplingContext,
    ) -> dict[str, Any]:
        image_shape = params["shape"][:2]
        height, width = image_shape
        region = self._get_dropout_region(data, image_shape)
        protected_mask = self._build_protected_mask(self._get_protected_bboxes(data, image_shape), image_shape)
        eligible_mask = region & ~protected_mask
        if not eligible_mask.any():
            return {"holes": np.empty((0, 4), dtype=np.int32), "dropout_mask": None, "seed": 0}

        num_holes = sampling.py_random.randint(*self.num_holes_range)
        centers = self._sample_centers_uniform(eligible_mask, num_holes, sampling.random_generator)
        hole_heights = np.maximum(
            1,
            (height * sampling.random_generator.uniform(*self.hole_height_range, size=num_holes)).astype(np.int32),
        )
        hole_widths = np.maximum(
            1,
            (width * sampling.random_generator.uniform(*self.hole_width_range, size=num_holes)).astype(np.int32),
        )
        holes = self._centers_to_holes(centers, hole_heights, hole_widths, image_shape)
        dropout_mask = self._rasterize_holes(holes, image_shape) & eligible_mask
        return {
            "holes": holes,
            "dropout_mask": dropout_mask if dropout_mask.any() else None,
            "seed": int(sampling.random_generator.integers(0, 2**32 - 1)),
        }

    def apply(
        self,
        img: ImageType,
        holes: np.ndarray,
        seed: int,
        **params: Any,
    ) -> ImageType:
        dropout_mask = params.get("dropout_mask")
        if dropout_mask is None:
            return img
        self._validate_fill_channel_count(img, {2, 3})
        return fdropout.fill_masked_holes(img, holes, dropout_mask, self.fill, np.random.default_rng(seed))

    def apply_to_mask(
        self,
        mask: ImageType,
        holes: np.ndarray,
        seed: int,
        **params: Any,
    ) -> ImageType:
        dropout_mask = params.get("dropout_mask")
        if dropout_mask is None or self.fill_mask is None:
            return mask
        return fdropout.fill_masked_pixels(mask, dropout_mask, self.fill_mask)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        holes: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        dropout_mask = params.get("dropout_mask")
        if dropout_mask is None or len(bboxes) == 0:
            return bboxes
        processor = cast("BboxProcessor | None", self.get_processor("bboxes"))
        if processor is None:
            return bboxes
        image_shape = params["shape"][:2]
        result = fdropout.mask_dropout_bboxes(
            denormalize_bboxes(bboxes, image_shape),
            dropout_mask,
            image_shape,
            processor.params.min_area,
            processor.params.min_visibility,
        )
        return normalize_bboxes(result, image_shape)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        holes: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        dropout_mask = params.get("dropout_mask")
        if dropout_mask is None or len(keypoints) == 0:
            return keypoints
        processor = cast("KeypointsProcessor | None", self.get_processor("keypoints"))
        if processor is None or not processor.params.remove_invisible:
            return keypoints
        return fdropout.mask_dropout_keypoints(keypoints, dropout_mask)
