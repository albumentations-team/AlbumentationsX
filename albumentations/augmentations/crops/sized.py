"""Random sized crop transforms."""

from typing import Annotated, Any, Literal

from ._transforms_shared import (
    ALL_TARGETS,
    AfterValidator,
    BaseTransformInitSchema,
    Field,
    check_range_bounds,
    cv2,
    fcrops,
    math,
    nondecreasing,
)
from .base import (
    _BaseRandomSizedCrop,
)


class RandomSizedCrop(_BaseRandomSizedCrop):
    """Random crop with height in min_max_height and aspect ratio (w2h_ratio), then resize to
    size. Scale and aspect variation with fixed output size.

    This transform first crops a random portion of the input and then resizes it to a specified size.
    The size of the random crop is controlled by the 'min_max_height' parameter.

    Args:
        min_max_height (tuple[int, int]): Minimum and maximum height of the crop in pixels.
        size (tuple[int, int]): Target size for the output image, i.e. (height, width) after crop and resize.
        w2h_ratio (float): Aspect ratio (width/height) of crop. Default: 1.0
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        area_for_downscale (Literal[None, "image", "image_mask"]): Controls automatic use of INTER_AREA interpolation
            for downscaling. Options:
            - None: No automatic interpolation selection, always use the specified interpolation method
            - "image": Use INTER_AREA when downscaling images, retain specified interpolation for upscaling and masks
            - "image_mask": Use INTER_AREA when downscaling both images and masks
            Default: None.
        p (float): Probability of applying the transform. Default: 1.0

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32


    Supported bboxes:
        hbb, obb
    Note:
        - The crop size is randomly selected for each execution within the range specified by 'min_max_height'.
        - The aspect ratio of the crop is determined by the 'w2h_ratio' parameter.
        - After cropping, the result is resized to the specified 'size'.
        - Bounding boxes that end up fully outside the cropped area will be removed.
        - Keypoints that end up outside the cropped area will be removed.
        - This transform differs from RandomResizedCrop in that it allows more control over the crop size
          through the 'min_max_height' parameter, rather than using a scale parameter.
        - When area_for_downscale is set, INTER_AREA interpolation will be used automatically for
          downscaling (when the crop is larger than the target size), which provides better quality for size reduction.

    Mathematical Details:
        1. A random crop height h is sampled from the range [min_max_height[0], min_max_height[1]].
        2. The crop width w is calculated as: w = h * w2h_ratio
        3. A random location for the crop is selected within the input image.
        4. The image is cropped to the size (h, w).
        5. The crop is then resized to the specified 'size'.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Define transform with parameters as tuples
        >>> transform = A.Compose([
        ...     A.RandomSizedCrop(
        ...         min_max_height=(50, 80),
        ...         size=(64, 64),
        ...         w2h_ratio=1.0,
        ...         interpolation=cv2.INTER_LINEAR,
        ...         mask_interpolation=cv2.INTER_NEAREST,
        ...         area_for_downscale="image",  # Use INTER_AREA for image downscaling
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
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
        >>> transformed_image = transformed['image']       # Shape: (64, 64, 3)
        >>> transformed_mask = transformed['mask']         # Shape: (64, 64)
        >>> transformed_bboxes = transformed['bboxes']     # Bounding boxes adjusted to new crop and size
        >>> transformed_bbox_labels = transformed['bbox_labels']  # Labels for the preserved bboxes
        >>> transformed_keypoints = transformed['keypoints']      # Keypoints adjusted to new crop and size
        >>> transformed_keypoint_labels = transformed['keypoint_labels']  # Labels for the preserved keypoints

    """

    _targets = ALL_TARGETS
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})

    class InitSchema(BaseTransformInitSchema):
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
        min_max_height: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
            AfterValidator(nondecreasing),
        ]
        w2h_ratio: Annotated[float, Field(gt=0)]
        size: Annotated[tuple[int, int], AfterValidator(check_range_bounds(1, None))]
        area_for_downscale: Literal["image", "image_mask"] | None

    def __init__(
        self,
        min_max_height: tuple[int, int],
        size: tuple[int, int],
        w2h_ratio: float = 1.0,
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
        area_for_downscale: Literal["image", "image_mask"] | None = None,
        p: float = 1.0,
    ):
        super().__init__(
            size=size,
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            area_for_downscale=area_for_downscale,
            p=p,
        )
        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, tuple[int, int, int, int]]:
        image_shape = params["shape"][:2]

        crop_height = self.py_random.randint(*self.min_max_height)
        crop_width = int(crop_height * self.w2h_ratio)

        crop_shape = (crop_height, crop_width)

        h_start = self.py_random.random()
        w_start = self.py_random.random()

        crop_coords = fcrops.get_crop_coords(image_shape, crop_shape, h_start, w_start)

        self.applied_config = {
            "min_max_height": (crop_height, crop_height),
        }

        return {"crop_coords": crop_coords}


class RandomResizedCrop(_BaseRandomSizedCrop):
    """Random crop with scale and ratio ranges (torchvision-style), then resize to size.
    Standard for training on varying resolutions; scale and ratio control crop.

    This transform first crops a random portion of the input image (or mask, bounding boxes, keypoints)
    and then resizes the crop to a specified size. It's particularly useful for training neural networks
    on images of varying sizes and aspect ratios.

    Args:
        size (tuple[int, int]): Target size for the output image, i.e. (height, width) after crop and resize.
        scale (tuple[float, float]): Range of the random size of the crop relative to the input size.
            For example, (0.08, 1.0) means the crop size will be between 8% and 100% of the input size.
            Default: (0.08, 1.0)
        ratio (tuple[float, float]): Range of aspect ratios of the random crop.
            For example, (0.75, 1.3333) allows crop aspect ratios from 3:4 to 4:3.
            Default: (0.75, 1.3333333333333333)
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR
        mask_interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST
        area_for_downscale (Literal[None, "image", "image_mask"]): Controls automatic use of INTER_AREA interpolation
            for downscaling. Options:
            - None: No automatic interpolation selection, always use the specified interpolation method
            - "image": Use INTER_AREA when downscaling images, retain specified interpolation for upscaling and masks
            - "image_mask": Use INTER_AREA when downscaling both images and masks
            Default: None.
        p (float): Probability of applying the transform. Default: 1.0

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32


    Supported bboxes:
        hbb, obb
    Note:
        - This transform attempts to crop a random area with an aspect ratio and relative size
          specified by 'ratio' and 'scale' parameters. If it fails to find a suitable crop after
          10 attempts, it will return a crop from the center of the image.
        - The crop's aspect ratio is defined as width / height.
        - Bounding boxes that end up fully outside the cropped area will be removed.
        - Keypoints that end up outside the cropped area will be removed.
        - After cropping, the result is resized to the specified size.
        - When area_for_downscale is set, INTER_AREA interpolation will be used automatically for
          downscaling (when the crop is larger than the target size), which provides better quality for size reduction.

    Mathematical Details:
        1. A target area A is sampled from the range [scale[0] * input_area, scale[1] * input_area].
        2. A target aspect ratio r is sampled from the range [ratio[0], ratio[1]].
        3. The crop width and height are computed as:
           w = sqrt(A * r)
           h = sqrt(A / r)
        4. If w and h are within the input image dimensions, the crop is accepted.
           Otherwise, steps 1-3 are repeated (up to 10 times).
        5. If no valid crop is found after 10 attempts, a centered crop is taken.
        6. The crop is then resized to the specified size.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Define transform with parameters as tuples
        >>> transform = A.Compose([
        ...     A.RandomResizedCrop(
        ...         size=(64, 64),
        ...         scale=(0.5, 0.9),  # Crop size will be 50-90% of original image
        ...         ratio=(0.75, 1.33),  # Aspect ratio will vary from 3:4 to 4:3
        ...         interpolation=cv2.INTER_LINEAR,
        ...         mask_interpolation=cv2.INTER_NEAREST,
        ...         area_for_downscale="image",  # Use INTER_AREA for image downscaling
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
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
        >>> transformed_image = transformed['image']       # Shape: (64, 64, 3)
        >>> transformed_mask = transformed['mask']         # Shape: (64, 64)
        >>> transformed_bboxes = transformed['bboxes']     # Bounding boxes adjusted to new crop and size
        >>> transformed_bbox_labels = transformed['bbox_labels']  # Labels for the preserved bboxes
        >>> transformed_keypoints = transformed['keypoints']      # Keypoints adjusted to new crop and size
        >>> transformed_keypoint_labels = transformed['keypoint_labels']  # Labels for the preserved keypoints

    """

    _targets = ALL_TARGETS
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})

    class InitSchema(BaseTransformInitSchema):
        scale: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, 1)), AfterValidator(nondecreasing)]
        ratio: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        size: Annotated[tuple[int, int], AfterValidator(check_range_bounds(1, None))]
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
        area_for_downscale: Literal["image", "image_mask"] | None

    def __init__(
        self,
        size: tuple[int, int],
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (0.75, 1.3333333333333333),
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
        area_for_downscale: Literal["image", "image_mask"] | None = None,
        p: float = 1.0,
    ):
        super().__init__(
            size=size,
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            area_for_downscale=area_for_downscale,
            p=p,
        )
        self.scale = scale
        self.ratio = ratio

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, tuple[int, int, int, int]]:
        image_shape = params["shape"][:2]
        image_height, image_width = image_shape

        area = image_height * image_width

        # Pre-compute constants to avoid repeated calculations
        scale_min_area = self.scale[0] * area
        scale_max_area = self.scale[1] * area
        log_ratio_min = math.log(self.ratio[0])
        log_ratio_max = math.log(self.ratio[1])

        for _ in range(10):
            target_area = self.py_random.uniform(scale_min_area, scale_max_area)
            aspect_ratio = math.exp(self.py_random.uniform(log_ratio_min, log_ratio_max))

            width = round(math.sqrt(target_area * aspect_ratio))
            height = round(math.sqrt(target_area / aspect_ratio))

            if 0 < width <= image_width and 0 < height <= image_height:
                h_start = self.py_random.random()
                w_start = self.py_random.random()
                crop_coords = fcrops.get_crop_coords(image_shape, (height, width), h_start, w_start)
                sampled_scale = target_area / area
                self.applied_config = {
                    "scale": (sampled_scale, sampled_scale),
                    "ratio": (aspect_ratio, aspect_ratio),
                }
                return {"crop_coords": crop_coords}

        # Fallback to central crop - use proper function
        in_ratio = image_width / image_height
        if in_ratio < self.ratio[0]:
            width = image_width
            height = round(image_width / self.ratio[0])
        elif in_ratio > self.ratio[1]:
            height = image_height
            width = round(height * self.ratio[1])
        else:  # whole image
            width = image_width
            height = image_height

        crop_coords = fcrops.get_center_crop_coords(image_shape, (height, width))
        fallback_scale = (width * height) / area
        fallback_ratio = width / height if height > 0 else 1.0
        self.applied_config = {
            "scale": (fallback_scale, fallback_scale),
            "ratio": (fallback_ratio, fallback_ratio),
        }
        return {"crop_coords": crop_coords}


__all__ = [
    "RandomResizedCrop",
    "RandomSizedCrop",
]
