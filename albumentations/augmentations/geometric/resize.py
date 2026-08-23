"""Transforms for resizing images and associated data.

This module provides transform classes for resizing operations, including uniform resizing,
scaling with aspect ratio preservation, and size-constrained transformations.
"""

from collections.abc import Sequence
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import Field, model_validator
from typing_extensions import Self

from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes
from albumentations.core.invocation import SamplingContext
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.type_definitions import (
    ALL_TARGETS,
    CV2_INTER_LINEAR,
    CV2_INTER_NEAREST,
    FullInterpolationType,
    ImageType,
)

from . import functional as fgeometric

__all__ = ["LetterBox", "LongestMaxSize", "RandomScale", "Resize", "SmallestMaxSize"]


class RandomScale(DualTransform):
    """Resize by a random scale factor (scale_range). Output size differs from input; all
    targets scaled together. Useful for scale augmentation without cropping.

    Args:
        scale_range (tuple[float, float] or dict[Literal["x", "y"], tuple[float, float]]): Scaling factor range.
            - If a tuple `(low, high)`: A single scale factor is sampled per image from
              `(1 + low, 1 + high)` and applied uniformly to both width and height.
            - If a dict with keys `"x"` and `"y"`: Each entry must be a `(low, high)`
              tuple. Scale factors are sampled independently per axis. Both keys are required.
            Default: `(-0.1, 0.1)`.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        area_for_downscale (Literal[None, "image", "image_mask"]): Controls automatic use of INTER_AREA interpolation
            for downscaling. Options:
            - None: No automatic interpolation selection, always use the specified interpolation method
            - "image": Use INTER_AREA when downscaling images, retain specified interpolation for upscaling and masks
            - "image_mask": Use INTER_AREA when downscaling both images and masks
            Default: None.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Supported bboxes:
        hbb, obb

    Note:
        - The output image size is different from the input image size.
        - Scale factor is sampled independently per image side (width and height).
        - Bounding box coordinates are scaled accordingly.
        - Keypoint coordinates are scaled accordingly.
        - When area_for_downscale is set, INTER_AREA interpolation will be used automatically for
          downscaling (scale < 1.0), which provides better quality for size reduction.
        - When `scale_range` is a tuple, the same scale factor is applied to both width and height
          (uniform scaling). When it is a dict, `scale_x` and `scale_y` are sampled independently
          (anisotropic scaling).

    Mathematical formulation:
        Let (W, H) be the original image dimensions and (W', H') be the output dimensions.

        **Uniform** (tuple `scale_range`):
            The scale factor s is sampled from [1 + low, 1 + high].
            Then, W' = W * s and H' = H * s.

        **Anisotropic** (dict `scale_range`):
            scale_x is sampled from [1 + scale_range["x"][0], 1 + scale_range["x"][1]].
            scale_y is sampled from [1 + scale_range["y"][0], 1 + scale_range["y"][1]].
            Then, W' = W * scale_x and H' = H * scale_y.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create sample data for demonstration
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> # Add some shapes to visualize scaling effects
        >>> cv2.rectangle(image, (25, 25), (75, 75), (255, 0, 0), -1)  # Red square
        >>> cv2.circle(image, (50, 50), 10, (0, 255, 0), -1)  # Green circle
        >>>
        >>> # Create a mask for segmentation
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[25:75, 25:75] = 1  # Mask covering the red square
        >>>
        >>> # Create bounding boxes and keypoints
        >>> bboxes = np.array([[25, 25, 75, 75]])  # Box around the red square
        >>> bbox_labels = [1]
        >>> keypoints = np.array([[50, 50]])  # Center of circle
        >>> keypoint_labels = [0]
        >>>
        >>> # Example 1: Uniform scaling with tuple scale_range
        >>> transform = A.Compose([
        ...     A.RandomScale(
        ...         scale_range=(-0.3, 0.5),     # Uniform: scale between 0.7x and 1.5x
        ...         interpolation=cv2.INTER_LINEAR,
        ...         mask_interpolation=cv2.INTER_NEAREST,
        ...         area_for_downscale="image",  # Use INTER_AREA for image downscaling
        ...         p=1.0                         # Always apply
        ...     )
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform to all targets
        >>> result = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed results
        >>> scaled_image = result['image']        # Dimensions will be between 70-150 pixels
        >>> scaled_mask = result['mask']          # Mask scaled proportionally to image
        >>> scaled_bboxes = result['bboxes']      # Bounding boxes adjusted to new dimensions
        >>> scaled_bbox_labels = result['bbox_labels']  # Labels remain unchanged
        >>> scaled_keypoints = result['keypoints']      # Keypoints adjusted to new dimensions
        >>> scaled_keypoint_labels = result['keypoint_labels']  # Labels remain unchanged
        >>>
        >>> # Example 2: Anisotropic scaling with dict scale_range
        >>> transform2 = A.Compose([
        ...     A.RandomScale(
        ...         scale_range={"x": (-0.20, 0.30), "y": (-0.10, 0.15)},  # Independent axes
        ...         interpolation=cv2.INTER_LINEAR,
        ...         p=1.0
        ...     )
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> result2 = transform2(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>> # Width and height will scale independently

    """

    _targets = ALL_TARGETS
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})

    class InitSchema(BaseTransformInitSchema):
        scale_range: tuple[float, float] | dict[Literal["x", "y"], tuple[float, float]]
        area_for_downscale: Literal["image", "image_mask"] | None
        interpolation: FullInterpolationType
        mask_interpolation: FullInterpolationType

        @staticmethod
        def _validate_range(low: float, high: float) -> None:
            if not np.isfinite(low) or not np.isfinite(high):
                raise ValueError(f"scale_range bounds must be finite. Got ({low}, {high})")
            if low > high:
                raise ValueError(f"scale_range low must be <= high. Got ({low}, {high})")
            if low <= -1.0:
                raise ValueError(f"scale_range low must be > -1 to avoid non-positive dimensions. Got {low}")

        @model_validator(mode="after")
        def _validate_scale_range(self) -> Self:
            if isinstance(self.scale_range, dict):
                if set(self.scale_range) != {"x", "y"}:
                    raise ValueError(
                        f"scale_range dict must contain exactly 'x' and 'y' keys. Got keys: {set(self.scale_range)}",
                    )
                for rng in self.scale_range.values():
                    self._validate_range(*rng)
            else:
                self._validate_range(*self.scale_range)
            return self

    def __init__(
        self,
        scale_range: tuple[float, float] | dict[Literal["x", "y"], tuple[float, float]] = (-0.1, 0.1),
        interpolation: FullInterpolationType = CV2_INTER_LINEAR,
        mask_interpolation: FullInterpolationType = CV2_INTER_NEAREST,
        area_for_downscale: Literal["image", "image_mask"] | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.scale_range = scale_range
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.area_for_downscale = area_for_downscale

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        sampling: SamplingContext,
    ) -> dict[str, float]:
        if isinstance(self.scale_range, dict):
            scale_x = sampling.py_random.uniform(*self.scale_range["x"]) + 1.0
            scale_y = sampling.py_random.uniform(*self.scale_range["y"]) + 1.0
            sampling.applied_overrides["scale_range"] = {"x": scale_x - 1.0, "y": scale_y - 1.0}
        else:
            scale = sampling.py_random.uniform(*self.scale_range) + 1.0
            scale_x = scale_y = scale
            sampling.applied_overrides["scale_range"] = scale - 1.0
        return {"scale_x": scale_x, "scale_y": scale_y}

    def apply(
        self,
        img: ImageType,
        scale_x: float,
        scale_y: float,
        **params: Any,
    ) -> ImageType:
        is_downscale = scale_x < 1.0 or scale_y < 1.0
        interpolation = self.interpolation
        if self.area_for_downscale in ["image", "image_mask"] and is_downscale:
            interpolation = cv2.INTER_AREA

        result = fgeometric.scale_xy(img, scale_x, scale_y, interpolation)
        return fgeometric.clip_if_interpolation_can_overshoot(result, interpolation)

    def apply_to_mask(
        self,
        mask: ImageType,
        scale_x: float,
        scale_y: float,
        **params: Any,
    ) -> ImageType:
        is_downscale = scale_x < 1.0 or scale_y < 1.0
        interpolation = self.mask_interpolation
        if self.area_for_downscale == "image_mask" and is_downscale:
            interpolation = cv2.INTER_AREA

        return fgeometric.scale_xy(mask, scale_x, scale_y, interpolation)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        scale_x: float,
        scale_y: float,
        **params: Any,
    ) -> np.ndarray:
        height, width = params["shape"][:2]
        new_height = max(1, round(height * scale_y))
        new_width = max(1, round(width * scale_x))
        return fgeometric.resize_bboxes(
            bboxes,
            image_shape=(height, width),
            output_shape=(new_height, new_width),
            bbox_type=params["bbox_type"],
        )

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        scale_x: float,
        scale_y: float,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.keypoints_scale(keypoints, scale_x, scale_y)


class BaseMaxSizeTransform(DualTransform):
    """Resize so longest or smallest side meets a maximum; aspect ratio fixed. Use
    LongestMaxSize or SmallestMaxSize; max_size or max_size_hw sets the constraint.

    This class provides common functionality for derived transforms like LongestMaxSize and
    SmallestMaxSize that resize images based on size constraints while preserving aspect ratio.

    Args:
        max_size (int, Sequence[int], optional): Maximum size constraint. The specific interpretation
            depends on the derived class. Default: None.
        max_size_hw (tuple[int | None, int | None], optional): Maximum (height, width) constraints.
            Either max_size or max_size_hw must be specified, but not both. Default: None.
        interpolation (OpenCV flag): Flag for the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): Flag for the mask interpolation algorithm.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        area_for_downscale (Literal[None, "image", "image_mask"]): Controls automatic use of INTER_AREA interpolation
            for downscaling. Options:
            - None: No automatic interpolation selection, always use the specified interpolation method
            - "image": Use INTER_AREA when downscaling images, retain specified interpolation for upscaling and masks
            - "image_mask": Use INTER_AREA when downscaling both images and masks
            Default: None.
        p (float): Probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - This is a base class that should be extended by concrete resize transforms.
        - The scaling calculation is implemented in derived classes.
        - Aspect ratio is preserved by applying the same scale factor to both dimensions.
        - When area_for_downscale is set, INTER_AREA interpolation will be used automatically for
          downscaling (scale < 1.0), which provides better quality for size reduction.

    Examples:
        >>> import numpy as np
        >>> from albumentations.augmentations.geometric.resize import BaseMaxSizeTransform
        >>> import cv2
        >>>
        >>> # Example of creating a custom transform that extends BaseMaxSizeTransform
        >>> class CustomMaxSize(BaseMaxSizeTransform):
        ...     def sample_parameters(self, params, data, sampling):
        ...         img_h, img_w = params["shape"][:2]
        ...         # Calculate scale factor - here we scale to make the image area constant
        ...         target_area = 300 * 300  # Target area of 300x300
        ...         current_area = img_h * img_w
        ...         scale = np.sqrt(target_area / current_area)
        ...         return {"scale": scale}
        >>>
        >>> # Prepare sample data
        >>> image = np.zeros((100, 200, 3), dtype=np.uint8)
        >>> # Add a rectangle to visualize the effect
        >>> cv2.rectangle(image, (50, 20), (150, 80), (255, 0, 0), -1)
        >>>
        >>> # Create a mask
        >>> mask = np.zeros((100, 200), dtype=np.uint8)
        >>> mask[20:80, 50:150] = 1
        >>>
        >>> # Create bounding boxes and keypoints
        >>> bboxes = np.array([[50, 20, 150, 80]])
        >>> bbox_labels = [1]
        >>> keypoints = np.array([[100, 50]])
        >>> keypoint_labels = [0]
        >>>
        >>> # Apply the custom transform
        >>> transform = A.Compose([
        ...     CustomMaxSize(
        ...         max_size=None,
        ...         max_size_hw=(None, None),  # Not used in our custom implementation
        ...         interpolation=cv2.INTER_LINEAR,
        ...         mask_interpolation=cv2.INTER_NEAREST,
        ...         area_for_downscale="image",  # Use INTER_AREA when downscaling images
        ...         p=1.0
        ...     )
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> result = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get results
        >>> transformed_image = result['image']  # Shape will be approximately (122, 245, 3)
        >>> transformed_mask = result['mask']    # Shape will be approximately (122, 245)
        >>> transformed_bboxes = result['bboxes']  # Bounding boxes are scale invariant
        >>> transformed_keypoints = result['keypoints']  # Keypoints scaled proportionally
        >>> transformed_bbox_labels = result['bbox_labels']  # Labels remain unchanged
        >>> transformed_keypoint_labels = result['keypoint_labels']  # Labels remain unchanged

    """

    _targets = ALL_TARGETS
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})

    class InitSchema(BaseTransformInitSchema):
        max_size: int | Sequence[int] | None
        max_size_hw: tuple[int | None, int | None] | None
        area_for_downscale: Literal["image", "image_mask"] | None
        interpolation: FullInterpolationType
        mask_interpolation: FullInterpolationType

        @model_validator(mode="after")
        def _validate_size_parameters(self) -> Self:
            if self.max_size is None and self.max_size_hw is None:
                raise ValueError("Either max_size or max_size_hw must be specified")
            if self.max_size is not None and self.max_size_hw is not None:
                raise ValueError("Only one of max_size or max_size_hw should be specified")
            return self

    def __init__(
        self,
        max_size: int | Sequence[int] | None = None,
        max_size_hw: tuple[int | None, int | None] | None = None,
        interpolation: FullInterpolationType = CV2_INTER_LINEAR,
        mask_interpolation: FullInterpolationType = CV2_INTER_NEAREST,
        area_for_downscale: Literal["image", "image_mask"] | None = None,
        p: float = 1,
    ):
        super().__init__(p=p)
        self.max_size = max_size
        self.max_size_hw = max_size_hw
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.area_for_downscale = area_for_downscale

    def apply(
        self,
        img: ImageType,
        scale: float,
        **params: Any,
    ) -> ImageType:
        height, width = img.shape[:2]
        new_height, new_width = max(1, round(height * scale)), max(1, round(width * scale))

        interpolation = self.interpolation
        if self.area_for_downscale in ["image", "image_mask"] and scale < 1.0:
            interpolation = cv2.INTER_AREA

        result = fgeometric.resize(img, (new_height, new_width), interpolation=interpolation)
        return fgeometric.clip_if_interpolation_can_overshoot(result, interpolation)

    def apply_to_mask(
        self,
        mask: ImageType,
        scale: float,
        **params: Any,
    ) -> ImageType:
        height, width = mask.shape[:2]
        new_height, new_width = max(1, round(height * scale)), max(1, round(width * scale))

        interpolation = self.mask_interpolation
        if self.area_for_downscale == "image_mask" and scale < 1.0:
            interpolation = cv2.INTER_AREA

        return fgeometric.resize(mask, (new_height, new_width), interpolation=interpolation)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        # Bounding box coordinates are scale invariant
        return bboxes

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        scale: float,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.keypoints_scale(keypoints, scale, scale)


class LongestMaxSize(BaseMaxSizeTransform):
    """Rescale an image so that the longest side is equal to max_size or sides meet max_size_hw constraints,
        keeping the aspect ratio.

    Args:
        max_size (int, Sequence[int], optional): Maximum size of the longest side after the transformation.
            When using a list or tuple, the max size will be randomly selected from the values provided. Default: None.
        max_size_hw (tuple[int | None, int | None], optional): Maximum (height, width) constraints. Supports:
            - (height, width): Both dimensions must fit within these bounds
            - (height, None): Only height is constrained, width scales proportionally
            - (None, width): Only width is constrained, height scales proportionally
            If specified, max_size must be None. Default: None.
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        area_for_downscale (Literal[None, "image", "image_mask"]): Controls automatic use of INTER_AREA interpolation
            for downscaling. Options:
            - None: No automatic interpolation selection, always use the specified interpolation method
            - "image": Use INTER_AREA when downscaling images, retain specified interpolation for upscaling and masks
            - "image_mask": Use INTER_AREA when downscaling both images and masks
            Default: None.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Supported bboxes:
        hbb, obb

    Note:
        - This transform scales images based on their longest side:
            * If the longest side is **smaller** than max_size: the image will be **upscaled** (scale > 1.0)
            * If the longest side is **equal** to max_size: the image will **not be resized** (scale = 1.0)
            * If the longest side is **larger** than max_size: the image will be **downscaled** (scale < 1.0)
        - This transform will not crop the image. The resulting image may be smaller than specified in both dimensions.
        - For non-square images, both sides will be scaled proportionally to maintain the aspect ratio.
        - Bounding boxes and keypoints are scaled accordingly.
        - When area_for_downscale is set, INTER_AREA will be used for downscaling, providing better quality.

    Mathematical Details:
        Let (W, H) be the original width and height of the image.

        When using max_size:
            1. The scaling factor s is calculated as:
               s = max_size / max(W, H)
            2. The new dimensions (W', H') are:
               W' = W * s
               H' = H * s

        When using max_size_hw=(H_target, W_target):
            1. For both dimensions specified:
               s = min(H_target/H, W_target/W)
               This ensures both dimensions fit within the specified bounds.

            2. For height only (W_target=None):
               s = H_target/H
               Width will scale proportionally.

            3. For width only (H_target=None):
               s = W_target/W
               Height will scale proportionally.

            4. The new dimensions (W', H') are:
               W' = W * s
               H' = H * s

    Examples:
        >>> import albumentations as A
        >>> import cv2
        >>> # Using max_size
        >>> transform1 = A.LongestMaxSize(max_size=1024, area_for_downscale="image")
        >>> # Input image (1500, 800) -> Output (1024, 546)
        >>>
        >>> # Using max_size_hw with both dimensions
        >>> transform2 = A.LongestMaxSize(max_size_hw=(800, 1024), area_for_downscale="image_mask")
        >>> # Input (1500, 800) -> Output (800, 427)
        >>> # Input (800, 1500) -> Output (546, 1024)
        >>>
        >>> # Using max_size_hw with only height
        >>> transform3 = A.LongestMaxSize(max_size_hw=(800, None))
        >>> # Input (1500, 800) -> Output (800, 427)
        >>>
        >>> # Common use case with padding
        >>> transform4 = A.Compose([
        ...     A.LongestMaxSize(max_size=1024, area_for_downscale="image"),
        ...     A.PadIfNeeded(min_height=1024, min_width=1024),
        ... ])

    """

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        sampling: SamplingContext,
    ) -> dict[str, Any]:
        img_h, img_w = params["shape"][:2]

        if self.max_size is not None:
            if isinstance(self.max_size, (list, tuple)):
                max_size = sampling.py_random.choice(self.max_size)
            else:
                max_size = self.max_size
            sampling.applied_overrides["max_size"] = max_size
            scale = max_size / max(img_h, img_w)
        elif self.max_size_hw is not None:
            max_h, max_w = self.max_size_hw
            if max_h is not None and max_w is not None:
                h_scale = max_h / img_h
                w_scale = max_w / img_w
                scale = min(h_scale, w_scale)
            elif max_h is not None:
                scale = max_h / img_h
            else:
                if max_w is None:
                    raise RuntimeError("max_w must be initialized when max_h is not set")
                scale = max_w / img_w
        else:
            raise RuntimeError("Either max_size or max_size_hw must be set")

        return {"scale": scale}


class SmallestMaxSize(BaseMaxSizeTransform):
    """Rescale an image so that minimum side is equal to max_size or sides meet max_size_hw constraints,
    keeping the aspect ratio.

    Args:
        max_size (int, list of int, optional): Maximum size of smallest side of the image after the transformation.
            When using a list, max size will be randomly selected from the values in the list. Default: None.
        max_size_hw (tuple[int | None, int | None], optional): Maximum (height, width) constraints. Supports:
            - (height, width): Both dimensions must be at least these values
            - (height, None): Only height is constrained, width scales proportionally
            - (None, width): Only width is constrained, height scales proportionally
            If specified, max_size must be None. Default: None.
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        area_for_downscale (Literal[None, "image", "image_mask"]): Controls automatic use of INTER_AREA interpolation
            for downscaling. Options:
            - None: No automatic interpolation selection, always use the specified interpolation method
            - "image": Use INTER_AREA when downscaling images, retain specified interpolation for upscaling and masks
            - "image_mask": Use INTER_AREA when downscaling both images and masks
            Default: None.
        p (float): Probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Supported bboxes:
        hbb, obb

    Note:
        - This transform scales images based on their smallest side:
            * If the smallest side is **smaller** than max_size: the image will be **upscaled** (scale > 1.0)
            * If the smallest side is **equal** to max_size: the image will **not be resized** (scale = 1.0)
            * If the smallest side is **larger** than max_size: the image will be **downscaled** (scale < 1.0)
        - This transform will not crop the image. The resulting image may be larger than specified in both dimensions.
        - For non-square images, both sides will be scaled proportionally to maintain the aspect ratio.
        - Bounding boxes and keypoints are scaled accordingly.
        - When area_for_downscale is set, INTER_AREA will be used for downscaling, providing better quality.

    Mathematical Details:
        Let (W, H) be the original width and height of the image.

        When using max_size:
            1. The scaling factor s is calculated as:
               s = max_size / min(W, H)
            2. The new dimensions (W', H') are:
               W' = W * s
               H' = H * s

        When using max_size_hw=(H_target, W_target):
            1. For both dimensions specified:
               s = max(H_target/H, W_target/W)
               This ensures both dimensions are at least as large as specified.

            2. For height only (W_target=None):
               s = H_target/H
               Width will scale proportionally.

            3. For width only (H_target=None):
               s = W_target/W
               Height will scale proportionally.

            4. The new dimensions (W', H') are:
               W' = W * s
               H' = H * s

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> # Using max_size
        >>> transform1 = A.SmallestMaxSize(max_size=120, area_for_downscale="image")
        >>> # Input image (100, 150) -> Output (120, 180)
        >>>
        >>> # Using max_size_hw with both dimensions
        >>> transform2 = A.SmallestMaxSize(max_size_hw=(100, 200), area_for_downscale="image_mask")
        >>> # Input (80, 160) -> Output (100, 200)
        >>> # Input (160, 80) -> Output (400, 200)
        >>>
        >>> # Using max_size_hw with only height
        >>> transform3 = A.SmallestMaxSize(max_size_hw=(100, None))
        >>> # Input (80, 160) -> Output (100, 200)

    """

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        sampling: SamplingContext,
    ) -> dict[str, Any]:
        img_h, img_w = params["shape"][:2]

        if self.max_size is not None:
            if isinstance(self.max_size, (list, tuple)):
                max_size = sampling.py_random.choice(self.max_size)
            else:
                max_size = self.max_size
            sampling.applied_overrides["max_size"] = max_size
            scale = max_size / min(img_h, img_w)
        elif self.max_size_hw is not None:
            max_h, max_w = self.max_size_hw
            if max_h is not None and max_w is not None:
                h_scale = max_h / img_h
                w_scale = max_w / img_w
                scale = max(h_scale, w_scale)
            elif max_h is not None:
                scale = max_h / img_h
            else:
                if max_w is None:
                    raise RuntimeError("max_w must be initialized when max_h is not set")
                scale = max_w / img_w
        else:
            raise RuntimeError("Either max_size or max_size_hw must be set")

        return {"scale": scale}


class Resize(DualTransform):
    """Resize to given height and width. Params: height, width, interpolation, area_for_downscale.
    Supports image, mask, bboxes, keypoints.

    Args:
        height (int): desired height of the output.
        width (int): desired width of the output.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        area_for_downscale (Literal[None, "image", "image_mask"]): Controls automatic use of INTER_AREA interpolation
            for downscaling. Options:
            - None: No automatic interpolation selection, always use the specified interpolation method
            - "image": Use INTER_AREA when downscaling images, retain specified interpolation for upscaling and masks
            - "image_mask": Use INTER_AREA when downscaling both images and masks
            Default: None.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Supported bboxes:
        hbb, obb

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create sample data for demonstration
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> # Add some shapes to visualize resize effects
        >>> cv2.rectangle(image, (25, 25), (75, 75), (255, 0, 0), -1)  # Red square
        >>> cv2.circle(image, (50, 50), 10, (0, 255, 0), -1)  # Green circle
        >>>
        >>> # Create a mask for segmentation
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[25:75, 25:75] = 1  # Mask covering the red square
        >>>
        >>> # Create bounding boxes and keypoints
        >>> bboxes = np.array([[25, 25, 75, 75]])  # Box around the red square
        >>> bbox_labels = [1]
        >>> keypoints = np.array([[50, 50]])  # Center of circle
        >>> keypoint_labels = [0]
        >>>
        >>> # Resize all data to 224x224 (common input size for many CNNs)
        >>> transform = A.Compose([
        ...     A.Resize(
        ...         height=224,
        ...         width=224,
        ...         interpolation=cv2.INTER_LINEAR,
        ...         mask_interpolation=cv2.INTER_NEAREST,
        ...         area_for_downscale="image",  # Use INTER_AREA when downscaling images
        ...         p=1.0
        ...     )
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform to all targets
        >>> result = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed results
        >>> resized_image = result['image']        # Shape will be (224, 224, 3)
        >>> resized_mask = result['mask']          # Shape will be (224, 224)
        >>> resized_bboxes = result['bboxes']      # Bounding boxes scaled to new dimensions
        >>> resized_bbox_labels = result['bbox_labels']  # Labels remain unchanged
        >>> resized_keypoints = result['keypoints']      # Keypoints scaled to new dimensions
        >>> resized_keypoint_labels = result['keypoint_labels']  # Labels remain unchanged
        >>>
        >>> # Note: When resizing from 100x100 to 224x224:
        >>> # - The red square will be scaled from (25-75) to approximately (56-168)
        >>> # - The keypoint at (50, 50) will move to approximately (112, 112)
        >>> # - All spatial relationships are preserved but coordinates are scaled

    """

    _targets = ALL_TARGETS
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})

    class InitSchema(BaseTransformInitSchema):
        height: int = Field(ge=1)
        width: int = Field(ge=1)
        area_for_downscale: Literal["image", "image_mask"] | None
        interpolation: FullInterpolationType
        mask_interpolation: FullInterpolationType

    def __init__(
        self,
        height: int,
        width: int,
        interpolation: FullInterpolationType = CV2_INTER_LINEAR,
        mask_interpolation: FullInterpolationType = CV2_INTER_NEAREST,
        area_for_downscale: Literal["image", "image_mask"] | None = None,
        p: float = 1,
    ):
        super().__init__(p=p)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.area_for_downscale = area_for_downscale

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        height, width = img.shape[:2]
        is_downscale = (self.height < height) or (self.width < width)

        interpolation = self.interpolation
        if self.area_for_downscale in ["image", "image_mask"] and is_downscale:
            interpolation = cv2.INTER_AREA

        result = fgeometric.resize(img, (self.height, self.width), interpolation=interpolation)
        return fgeometric.clip_if_interpolation_can_overshoot(result, interpolation)

    def apply_to_mask(self, mask: ImageType, **params: Any) -> ImageType:
        height, width = mask.shape[:2]
        is_downscale = (self.height < height) or (self.width < width)

        interpolation = self.mask_interpolation
        if self.area_for_downscale == "image_mask" and is_downscale:
            interpolation = cv2.INTER_AREA

        return fgeometric.resize(mask, (self.height, self.width), interpolation=interpolation)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.resize_bboxes(
            bboxes,
            image_shape=params["shape"][:2],
            output_shape=(self.height, self.width),
            bbox_type=params["bbox_type"],
        )

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        height, width = params["shape"][:2]
        scale_x = self.width / width
        scale_y = self.height / height
        return fgeometric.keypoints_scale(keypoints, scale_x, scale_y)


class LetterBox(DualTransform):
    """Scale image to fit a target canvas preserving aspect ratio, then pad to exact canvas size:
        YOLO letterbox, equivalent to LongestMaxSize + PadIfNeeded.

    The image is downscaled or upscaled so its longest side fits the target, then constant-color padding
    fills the remaining area. All targets (masks, bboxes, keypoints) are adjusted accordingly.

    Args:
        size (tuple[int, int]): Target `(height, width)` of the output canvas.
        interpolation (OpenCV flag): Interpolation method used when resizing the image.
            Default: `cv2.INTER_LINEAR`.
        mask_interpolation (OpenCV flag): Interpolation method used when resizing masks.
            Default: `cv2.INTER_NEAREST`.
        fill (tuple[float, ...] | float): Constant pixel value for image padding.
            Default: `0`. For YOLO-style padding, use `114` with uint8 images or `114 / 255` with float32 images.
        fill_mask (tuple[float, ...] | float): Constant pixel value for mask padding.
            Default: `0`.
        position (Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"]):
            Where to place the resized image on the canvas. Default: `"center"`.
        p (float): Probability of applying the transform. Default: `1.0`.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Supported bboxes:
        hbb, obb

    Note:
        - The output size is always exactly `(height, width)`.
        - Images smaller than the target are upscaled; images larger are downscaled.
        - Bounding boxes and keypoints are adjusted for both the resize and padding steps.
        - YOLO uses `fill=114` for uint8 images. Use `fill=114 / 255` for normalized float32 images.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>> image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (480, 640), dtype=np.uint8)
        >>> bboxes = np.array([[100, 80, 300, 200]], dtype=np.float32)
        >>> bbox_labels = [1]
        >>> keypoints = np.array([[200, 150]], dtype=np.float32)
        >>> keypoint_labels = [0]
        >>>
        >>> transform = A.Compose([
        ...     A.LetterBox(size=(640, 640), fill=114, fill_mask=0, p=1.0)
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> result = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels,
        ... )
        >>> result['image'].shape
        (640, 640, 3)

    """

    _targets = ALL_TARGETS
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})

    class InitSchema(BaseTransformInitSchema):
        size: tuple[int, int]
        interpolation: FullInterpolationType
        mask_interpolation: FullInterpolationType
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float
        position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"]

    def __init__(
        self,
        size: tuple[int, int],
        interpolation: FullInterpolationType = CV2_INTER_LINEAR,
        mask_interpolation: FullInterpolationType = CV2_INTER_NEAREST,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"] = "center",
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.size = size
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.fill = fill
        self.fill_mask = fill_mask
        self.position = position

    def apply(
        self,
        img: ImageType,
        new_height: int,
        new_width: int,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> ImageType:
        resized = fgeometric.resize(img, (new_height, new_width), interpolation=self.interpolation)
        result = fgeometric.pad_with_params(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=cv2.BORDER_CONSTANT,
            value=self.fill,
        )
        return fgeometric.clip_if_interpolation_can_overshoot(result, self.interpolation)

    def apply_to_mask(
        self,
        mask: ImageType,
        new_height: int,
        new_width: int,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> ImageType:
        resized = fgeometric.resize(
            mask,
            (new_height, new_width),
            interpolation=self.mask_interpolation,
        )
        return fgeometric.pad_with_params(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=cv2.BORDER_CONSTANT,
            value=self.fill_mask,
        )

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        new_height: int,
        new_width: int,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> np.ndarray:
        # Bboxes are normalized [0,1] w.r.t. original image dimensions.
        # Uniform-scale resize keeps normalized coords unchanged.
        # Padding shifts them: denormalize in resized space, shift, renormalize in padded space.
        target_h, target_w = self.size

        bboxes_abs = denormalize_bboxes(bboxes, (new_height, new_width))
        padded_abs = fgeometric.pad_bboxes(
            bboxes_abs,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            image_shape=(new_height, new_width),
        )
        return normalize_bboxes(padded_abs, (target_h, target_w))

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        scale: float,
        new_height: int,
        new_width: int,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> np.ndarray:
        scaled = fgeometric.keypoints_scale(keypoints, scale, scale)
        return fgeometric.pad_keypoints(
            scaled,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            image_shape=(new_height, new_width),
        )

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        sampling: SamplingContext,
    ) -> dict[str, Any]:
        img_h, img_w = params["shape"][:2]
        target_h, target_w = self.size

        scale = min(target_h / img_h, target_w / img_w)
        new_h, new_w = max(1, round(img_h * scale)), max(1, round(img_w * scale))

        pad_h = target_h - new_h
        pad_w = target_w - new_w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        pad_top, pad_bottom, pad_left, pad_right = fgeometric.adjust_padding_by_position(
            h_top=pad_top,
            h_bottom=pad_bottom,
            w_left=pad_left,
            w_right=pad_right,
            position=self.position,
            py_random=sampling.py_random,
        )

        return {
            "scale": scale,
            "new_height": new_h,
            "new_width": new_w,
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right,
        }
