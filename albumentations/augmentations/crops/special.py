"""Specialized crop transforms."""

from typing import Annotated, Any

from typing_extensions import Self

from ._transforms_shared import (
    ALL_TARGETS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    AfterValidator,
    BaseTransformInitSchema,
    Field,
    ImageType,
    check_range_bounds,
    cv2,
    fcrops,
    model_validator,
    nondecreasing,
    np,
    reduce_sum,
)
from .base import (
    BaseCrop,
)


class CropNonEmptyMaskIfExists(BaseCrop):
    """Crop a region containing non-empty mask pixels; if mask empty or missing, fall back to
    random crop. Good for segmentation to focus on labeled regions.

    This transform attempts to crop a region containing a mask (non-zero pixels). If the mask is empty or not provided,
    it falls back to a random crop. This is particularly useful for segmentation tasks where you want to focus on
    regions of interest defined by the mask.

    Args:
        height (int): Vertical size of crop in pixels. Must be > 0.
        width (int): Horizontal size of crop in pixels. Must be > 0.
        ignore_values (list of int, optional): Values to ignore in mask, `0` values are always ignored.
            For example, if background value is 5, set `ignore_values=[5]` to ignore it. Default: None.
        ignore_channels (list of int, optional): Channels to ignore in mask.
            For example, if background is the first channel, set `ignore_channels=[0]` to ignore it. Default: None.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32


    Supported bboxes:
        hbb, obb
    Note:
        - If a mask is provided, the transform will try to crop an area containing non-zero (or non-ignored) pixels.
        - If no suitable area is found in the mask or no mask is provided, it will perform a random crop.
        - The crop size (height, width) must not exceed the original image dimensions.
        - Bounding boxes and keypoints are also cropped along with the image and mask.

    Raises:
        ValueError: If the specified crop size is larger than the input image dimensions.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[25:75, 25:75] = 1  # Create a non-empty region in the mask
        >>> transform = A.Compose([
        ...     A.CropNonEmptyMaskIfExists(height=50, width=50, p=1.0),
        ... ])
        >>> transformed = transform(image=image, mask=mask)
        >>> transformed_image = transformed['image']
        >>> transformed_mask = transformed['mask']
        # The resulting crop will likely include part of the non-zero region in the mask

    Raises:
        ValueError: If the specified crop size is larger than the input image dimensions.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> # Create a mask with non-empty region in the center
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[25:75, 25:75] = 1  # Create a non-empty region in the mask
        >>>
        >>> # Create bounding boxes and keypoints in the mask region
        >>> bboxes = np.array([
        ...     [20, 20, 60, 60],     # Box overlapping with non-empty region
        ...     [30, 30, 70, 70],     # Box mostly inside non-empty region
        ... ], dtype=np.float32)
        >>> bbox_labels = ['cat', 'dog']
        >>>
        >>> # Add some keypoints inside mask region
        >>> keypoints = np.array([
        ...     [40, 40],             # Inside non-empty region
        ...     [60, 60],             # At edge of non-empty region
        ...     [90, 90]              # Outside non-empty region
        ... ], dtype=np.float32)
        >>> keypoint_labels = ['eye', 'nose', 'ear']
        >>>
        >>> # Define transform that will crop around the non-empty mask region
        >>> transform = A.Compose([
        ...     A.CropNonEmptyMaskIfExists(
        ...         height=50,
        ...         width=50,
        ...         ignore_values=None,
        ...         ignore_channels=None,
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(
        ...     format='pascal_voc',
        ...     label_fields=['bbox_labels']
        ... ), keypoint_params=A.KeypointParams(
        ...     format='xy',
        ...     label_fields=['keypoint_labels']
        ... ))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> transformed_image = transformed['image']  # 50x50 image centered on mask region
        >>> transformed_mask = transformed['mask']    # 50x50 mask showing part of non-empty region
        >>> transformed_bboxes = transformed['bboxes']  # Bounding boxes adjusted to new coordinates
        >>> transformed_bbox_labels = transformed['bbox_labels']  # Labels preserved for visible boxes
        >>> transformed_keypoints = transformed['keypoints']  # Keypoints adjusted to new coordinates
        >>> transformed_keypoint_labels = transformed['keypoint_labels']  # Labels for visible keypoints

    """

    class InitSchema(BaseCrop.InitSchema):
        ignore_values: list[int] | None
        ignore_channels: list[int] | None
        height: Annotated[int, Field(ge=1)]
        width: Annotated[int, Field(ge=1)]

    def __init__(
        self,
        height: int,
        width: int,
        ignore_values: list[int] | None = None,
        ignore_channels: list[int] | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)

        self.height = height
        self.width = width
        self.ignore_values = ignore_values
        self.ignore_channels = ignore_channels

    def _preprocess_mask(self, mask: ImageType) -> ImageType:
        mask_height, mask_width = mask.shape[:2]

        if self.ignore_values is not None:
            ignore_values_np = np.array(self.ignore_values)
            mask = np.where(np.isin(mask, ignore_values_np), 0, mask)

        if mask.ndim == NUM_MULTI_CHANNEL_DIMENSIONS and self.ignore_channels is not None:
            target_channels = np.array([ch for ch in range(mask.shape[-1]) if ch not in self.ignore_channels])
            mask = np.take(mask, target_channels, axis=-1)

        if self.height > mask_height or self.width > mask_width:
            raise ValueError(
                f"Crop size ({self.height},{self.width}) is larger than image ({mask_height},{mask_width})",
            )

        return mask

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        if "mask" in data:
            mask = self._preprocess_mask(data["mask"])
        elif "masks" in data and len(data["masks"]):
            masks = data["masks"]
            mask = self._preprocess_mask(np.copy(masks[0]))
            for m in masks[1:]:
                mask |= self._preprocess_mask(m)
        else:
            msg = "Can not find mask for CropNonEmptyMaskIfExists"
            raise RuntimeError(msg)

        mask_height, mask_width = mask.shape[:2]

        if mask.any():
            # Find non-zero regions in mask
            mask_sum = reduce_sum(mask, axis=-1) if mask.ndim == NUM_MULTI_CHANNEL_DIMENSIONS else mask
            non_zero_xy = cv2.findNonZero((mask_sum > 0).astype(np.uint8))
            non_zero_yx = non_zero_xy[:, 0, ::-1]
            y, x = self.py_random.choice(non_zero_yx)

            # Calculate crop coordinates centered around chosen point
            x_min = x - self.py_random.randint(0, self.width - 1)
            y_min = y - self.py_random.randint(0, self.height - 1)
            x_min = np.clip(x_min, 0, mask_width - self.width)
            y_min = np.clip(y_min, 0, mask_height - self.height)
        else:
            # Random crop if no non-zero regions
            x_min = self.py_random.randint(0, mask_width - self.width)
            y_min = self.py_random.randint(0, mask_height - self.height)

        x_max = x_min + self.width
        y_max = y_min + self.height

        return {"crop_coords": (x_min, y_min, x_max, y_max)}


class RandomCropNearBBox(BaseCrop):
    """Crop around a reference bbox (cropping_bbox_key) with random shift (max_part_shift).
    Use when you have a region of interest to augment.

    Args:
        max_part_shift (tuple[float, float]): Range (min, max) for shift in `height` and `width`
            dimensions relative to `cropping_bbox` dimension. Default (0, 0.3).
        cropping_bbox_key (str): Additional target key for cropping box. Default `cropping_bbox`.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32


    Supported bboxes:
        hbb, obb
    Examples:
        >>> aug = Compose([RandomCropNearBBox(max_part_shift=(0.1, 0.5), cropping_bbox_key='test_bbox')],
        >>>              bbox_params=BboxParams("pascal_voc"))
        >>> result = aug(image=image, bboxes=bboxes, test_bbox=[0, 5, 10, 20])

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        max_part_shift: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        cropping_bbox_key: str

    def __init__(
        self,
        max_part_shift: tuple[float, float] = (0, 0.3),
        cropping_bbox_key: str = "cropping_bbox",
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.max_part_shift = max_part_shift
        self.cropping_bbox_key = cropping_bbox_key

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, tuple[float, ...]]:
        bbox = data[self.cropping_bbox_key]

        image_shape = params["shape"][:2]

        bbox = self._clip_bbox(bbox, image_shape)

        h_max_shift = round((bbox[3] - bbox[1]) * self.max_part_shift[0])
        w_max_shift = round((bbox[2] - bbox[0]) * self.max_part_shift[1])

        x_min = bbox[0] - self.py_random.randint(-w_max_shift, w_max_shift)
        x_max = bbox[2] + self.py_random.randint(-w_max_shift, w_max_shift)

        y_min = bbox[1] - self.py_random.randint(-h_max_shift, h_max_shift)
        y_max = bbox[3] + self.py_random.randint(-h_max_shift, h_max_shift)

        crop_coords = self._clip_bbox((x_min, y_min, x_max, y_max), image_shape)

        if crop_coords[0] == crop_coords[2] or crop_coords[1] == crop_coords[3]:
            crop_shape = (bbox[3] - bbox[1], bbox[2] - bbox[0])
            crop_coords = fcrops.get_center_crop_coords(image_shape, crop_shape)

        return {"crop_coords": crop_coords}

    @property
    def targets_as_params(self) -> list[str]:
        return [self.cropping_bbox_key]


class RandomCropFromBorders(BaseCrop):
    """Randomly remove a strip from each border (crop_left/right/top/bottom). No resize;
    output smaller. Good for trimming variable borders or slight zoom.

    This transform randomly crops parts of the input (image, mask, bounding boxes, or keypoints)
    from each of its borders. The amount of cropping is specified as a fraction of the input's
    dimensions for each side independently.

    Args:
        crop_left (float): The maximum fraction of width to crop from the left side.
            Must be in the range [0.0, 1.0]. Default: 0.1
        crop_right (float): The maximum fraction of width to crop from the right side.
            Must be in the range [0.0, 1.0]. Default: 0.1
        crop_top (float): The maximum fraction of height to crop from the top.
            Must be in the range [0.0, 1.0]. Default: 0.1
        crop_bottom (float): The maximum fraction of height to crop from the bottom.
            Must be in the range [0.0, 1.0]. Default: 0.1
        p (float): Probability of applying the transform. Default: 1.0

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32


    Supported bboxes:
        hbb, obb
    Note:
        - The actual amount of cropping for each side is randomly chosen between 0 and
          the specified maximum for each application of the transform.
        - The sum of crop_left and crop_right must not exceed 1.0, and the sum of
          crop_top and crop_bottom must not exceed 1.0. Otherwise, a ValueError will be raised.
        - This transform does not resize the input after cropping, so the output dimensions
          will be smaller than the input dimensions.
        - Bounding boxes that end up fully outside the cropped area will be removed.
        - Keypoints that end up outside the cropped area will be removed.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Define transform with crop fractions for each border
        >>> transform = A.Compose([
        ...     A.RandomCropFromBorders(
        ...         crop_left=0.1,     # Max 10% crop from left
        ...         crop_right=0.2,    # Max 20% crop from right
        ...         crop_top=0.15,     # Max 15% crop from top
        ...         crop_bottom=0.05,  # Max 5% crop from bottom
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply transform
        >>> result = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Access transformed data
        >>> transformed_image = result['image']  # Reduced size image with borders cropped
        >>> transformed_mask = result['mask']    # Reduced size mask with borders cropped
        >>> transformed_bboxes = result['bboxes']  # Bounding boxes adjusted to new dimensions
        >>> transformed_bbox_labels = result['bbox_labels']  # Bounding box labels after crop
        >>> transformed_keypoints = result['keypoints']  # Keypoints adjusted to new dimensions
        >>> transformed_keypoint_labels = result['keypoint_labels']  # Keypoint labels after crop
        >>>
        >>> # The resulting output shapes will be smaller, with dimensions reduced by
        >>> # the random crop amounts from each side (within the specified maximums)
        >>> print(f"Original image shape: (100, 100, 3)")
        >>> print(f"Transformed image shape: {transformed_image.shape}")  # e.g., (85, 75, 3)

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        crop_left: float = Field(
            ge=0.0,
            le=1.0,
        )
        crop_right: float = Field(
            ge=0.0,
            le=1.0,
        )
        crop_top: float = Field(
            ge=0.0,
            le=1.0,
        )
        crop_bottom: float = Field(
            ge=0.0,
            le=1.0,
        )

        @model_validator(mode="after")
        def _validate_crop_values(self) -> Self:
            if self.crop_left + self.crop_right > 1.0:
                msg = "The sum of crop_left and crop_right must be <= 1."
                raise ValueError(msg)
            if self.crop_top + self.crop_bottom > 1.0:
                msg = "The sum of crop_top and crop_bottom must be <= 1."
                raise ValueError(msg)
            return self

    def __init__(
        self,
        crop_left: float = 0.1,
        crop_right: float = 0.1,
        crop_top: float = 0.1,
        crop_bottom: float = 0.1,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, tuple[int, int, int, int]]:
        height, width = params["shape"][:2]

        x_min = self.py_random.randint(0, int(self.crop_left * width))
        x_max = self.py_random.randint(max(x_min + 1, int((1 - self.crop_right) * width)), width)

        y_min = self.py_random.randint(0, int(self.crop_top * height))
        y_max = self.py_random.randint(max(y_min + 1, int((1 - self.crop_bottom) * height)), height)

        crop_coords = x_min, y_min, x_max, y_max

        self.applied_config = {
            "crop_left": x_min / width if width > 0 else 0.0,
            "crop_right": 1.0 - x_max / width if width > 0 else 0.0,
            "crop_top": y_min / height if height > 0 else 0.0,
            "crop_bottom": 1.0 - y_max / height if height > 0 else 0.0,
        }

        return {"crop_coords": crop_coords}


__all__ = [
    "CropNonEmptyMaskIfExists",
    "RandomCropFromBorders",
    "RandomCropNearBBox",
]
