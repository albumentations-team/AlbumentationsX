"""Basic crop and crop-pad transforms."""

from collections.abc import Sequence
from typing import Annotated, Any, Literal, cast

from typing_extensions import Self

from ._transforms_shared import (
    ALL_TARGETS,
    CV2_BORDER_CONSTANT,
    CV2_INTER_LINEAR,
    CV2_INTER_NEAREST,
    PAIR,
    BaseTransformInitSchema,
    BorderModeType,
    DualTransform,
    Field,
    FullInterpolationType,
    ImageType,
    PercentType,
    PxType,
    fcrops,
    fgeometric,
    model_validator,
    np,
)
from .base import (
    BaseCropAndPad,
    CropSizeError,
)


class RandomCrop(BaseCropAndPad):
    """Crop a random region of fixed height and width. Optional pad when crop exceeds
    image. All targets cropped together. Common for fixed-resolution training.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        pad_if_needed (bool): Whether to pad if crop size exceeds image size. Default: False.
        border_mode (OpenCV flag): OpenCV border mode used for padding. Default: cv2.BORDER_CONSTANT.
        fill (tuple[float, ...] | float): Padding value for images if border_mode is
            cv2.BORDER_CONSTANT. Default: 0.
        fill_mask (tuple[float, ...] | float): Padding value for masks if border_mode is
            cv2.BORDER_CONSTANT. Default: 0.
        pad_position (Literal['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'random']):
            Position of padding. Default: 'center'.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32


    Supported bboxes:
        hbb, obb
    Note:
        If pad_if_needed is True and crop size exceeds image dimensions, the image will be padded
        before applying the random crop.

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
        >>> # Example 1: Basic random crop
        >>> transform = A.Compose([
        ...     A.RandomCrop(height=64, width=64),
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
        >>> transformed_image = transformed['image']  # Will be 64x64
        >>> transformed_mask = transformed['mask']    # Will be 64x64
        >>> transformed_bboxes = transformed['bboxes']  # Bounding boxes adjusted to the cropped area
        >>> transformed_bbox_labels = transformed['bbox_labels']  # Labels for boxes that remain after cropping
        >>> transformed_keypoints = transformed['keypoints']  # Keypoints adjusted to the cropped area
        >>> transformed_keypoint_labels = transformed['keypoint_labels']  # Labels for keypoints that remain
        >>>
        >>> # Example 2: Random crop with padding when needed
        >>> # This is useful when you want to crop to a size larger than some images
        >>> transform_padded = A.Compose([
        ...     A.RandomCrop(
        ...         height=120,  # Larger than original image height
        ...         width=120,   # Larger than original image width
        ...         pad_if_needed=True,
        ...         border_mode=cv2.BORDER_CONSTANT,
        ...         fill=0,      # Black padding for image
        ...         fill_mask=0  # Zero padding for mask
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the padded transform
        >>> padded_transformed = transform_padded(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # The result will be 120x120 with padding
        >>> padded_image = padded_transformed['image']
        >>> padded_mask = padded_transformed['mask']
        >>> padded_bboxes = padded_transformed['bboxes']  # Coordinates adjusted to the new dimensions

    """

    class InitSchema(BaseCropAndPad.InitSchema):
        height: Annotated[int, Field(ge=1)]
        width: Annotated[int, Field(ge=1)]
        border_mode: BorderModeType

        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

    def __init__(
        self,
        height: int,
        width: int,
        pad_if_needed: bool = False,
        pad_position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"] = "center",
        border_mode: BorderModeType = CV2_BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0.0,
        fill_mask: tuple[float, ...] | float = 0.0,
        p: float = 1.0,
    ):
        super().__init__(
            pad_if_needed=pad_if_needed,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
            pad_position=pad_position,
            p=p,
        )
        self.height = height
        self.width = width

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        image_shape = params["shape"][:2]
        image_height, image_width = image_shape

        if not self.pad_if_needed and (self.height > image_height or self.width > image_width):
            raise CropSizeError(
                f"Crop size (height, width) exceeds image dimensions (height, width):"
                f" {(self.height, self.width)} vs {image_shape[:2]}",
            )

        # Get padding params first if needed
        pad_params = self._get_pad_params(image_shape, (self.height, self.width))

        # If padding is needed, adjust the image shape for crop calculation
        if pad_params is not None:
            pad_top = pad_params["pad_top"]
            pad_bottom = pad_params["pad_bottom"]
            pad_left = pad_params["pad_left"]
            pad_right = pad_params["pad_right"]

            padded_height = image_height + pad_top + pad_bottom
            padded_width = image_width + pad_left + pad_right
            padded_shape = (padded_height, padded_width)

            # Get random crop coordinates based on padded dimensions
            h_start = self.py_random.random()
            w_start = self.py_random.random()
            crop_coords = fcrops.get_crop_coords(padded_shape, (self.height, self.width), h_start, w_start)
        else:
            # Get random crop coordinates based on original dimensions
            h_start = self.py_random.random()
            w_start = self.py_random.random()
            crop_coords = fcrops.get_crop_coords(image_shape, (self.height, self.width), h_start, w_start)

        return {
            "crop_coords": crop_coords,
            "pad_params": pad_params,
        }


class CenterCrop(BaseCropAndPad):
    """Crop the center region of fixed height and width. Optional pad when crop exceeds
    image. All targets share the same center window. Good for center-focused data.

    This transform crops the center of the input image, mask, bounding boxes, and keypoints to the specified dimensions.
    It's useful when you want to focus on the central region of the input, discarding peripheral information.

    Args:
        height (int): The height of the crop. Must be greater than 0.
        width (int): The width of the crop. Must be greater than 0.
        pad_if_needed (bool): Whether to pad if crop size exceeds image size. Default: False.
        border_mode (OpenCV flag): OpenCV border mode used for padding. Default: cv2.BORDER_CONSTANT.
        fill (tuple[float, ...] | float): Padding value for images if border_mode is
            cv2.BORDER_CONSTANT. Default: 0.
        fill_mask (tuple[float, ...] | float): Padding value for masks if border_mode is
            cv2.BORDER_CONSTANT. Default: 0.
        pad_position (Literal['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'random']):
            Position of padding. Default: 'center'.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32


    Supported bboxes:
        hbb, obb
    Note:
        - If pad_if_needed is False and crop size exceeds image dimensions, it will raise a CropSizeError.
        - If pad_if_needed is True and crop size exceeds image dimensions, the image will be padded.
        - For bounding boxes and keypoints, coordinates are adjusted appropriately for both padding and cropping.

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
        >>> # Example 1: Basic center crop without padding
        >>> transform = A.Compose([
        ...     A.CenterCrop(height=64, width=64),
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
        >>> transformed_image = transformed['image']  # Will be 64x64
        >>> transformed_mask = transformed['mask']    # Will be 64x64
        >>> transformed_bboxes = transformed['bboxes']  # Bounding boxes adjusted to the cropped area
        >>> transformed_bbox_labels = transformed['bbox_labels']  # Labels for boxes that remain after cropping
        >>> transformed_keypoints = transformed['keypoints']  # Keypoints adjusted to the cropped area
        >>> transformed_keypoint_labels = transformed['keypoint_labels']  # Labels for keypoints that remain
        >>>
        >>> # Example 2: Center crop with padding when needed
        >>> transform_padded = A.Compose([
        ...     A.CenterCrop(
        ...         height=120,  # Larger than original image height
        ...         width=120,   # Larger than original image width
        ...         pad_if_needed=True,
        ...         border_mode=cv2.BORDER_CONSTANT,
        ...         fill=0,      # Black padding for image
        ...         fill_mask=0  # Zero padding for mask
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the padded transform
        >>> padded_transformed = transform_padded(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # The result will be 120x120 with padding
        >>> padded_image = padded_transformed['image']
        >>> padded_mask = padded_transformed['mask']
        >>> padded_bboxes = padded_transformed['bboxes']  # Coordinates adjusted to the new dimensions
        >>> padded_keypoints = padded_transformed['keypoints']  # Coordinates adjusted to the new dimensions

    """

    class InitSchema(BaseCropAndPad.InitSchema):
        height: Annotated[int, Field(ge=1)]
        width: Annotated[int, Field(ge=1)]
        border_mode: BorderModeType

        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

    def __init__(
        self,
        height: int,
        width: int,
        pad_if_needed: bool = False,
        pad_position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"] = "center",
        border_mode: BorderModeType = CV2_BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0.0,
        fill_mask: tuple[float, ...] | float = 0.0,
        p: float = 1.0,
    ):
        super().__init__(
            pad_if_needed=pad_if_needed,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
            pad_position=pad_position,
            p=p,
        )
        self.height = height
        self.width = width

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        image_shape = params["shape"][:2]
        image_height, image_width = image_shape

        if not self.pad_if_needed and (self.height > image_height or self.width > image_width):
            raise CropSizeError(
                f"Crop size (height, width) exceeds image dimensions (height, width):"
                f" {(self.height, self.width)} vs {image_shape[:2]}",
            )

        # Get padding params first if needed
        pad_params = self._get_pad_params(image_shape, (self.height, self.width))

        # If padding is needed, adjust the image shape for crop calculation
        if pad_params is not None:
            pad_top = pad_params["pad_top"]
            pad_bottom = pad_params["pad_bottom"]
            pad_left = pad_params["pad_left"]
            pad_right = pad_params["pad_right"]

            padded_height = image_height + pad_top + pad_bottom
            padded_width = image_width + pad_left + pad_right
            padded_shape = (padded_height, padded_width)

            # Get crop coordinates based on padded dimensions
            crop_coords = fcrops.get_center_crop_coords(padded_shape, (self.height, self.width))
        else:
            # Get crop coordinates based on original dimensions
            crop_coords = fcrops.get_center_crop_coords(image_shape, (self.height, self.width))

        return {
            "crop_coords": crop_coords,
            "pad_params": pad_params,
        }


class Crop(BaseCropAndPad):
    """Crop a fixed region by (x_min, y_min, x_max, y_max). Deterministic; optional pad when
    region exceeds image. Use for fixed ROI or sliding-window pipelines.

    This transform crops a rectangular region from the input image, mask, bounding boxes, and keypoints
    based on specified coordinates. It's useful when you want to extract a specific area of interest
    from your inputs.

    Args:
        x_min (int): Minimum x-coordinate of the crop region (left edge). Must be >= 0. Default: 0.
        y_min (int): Minimum y-coordinate of the crop region (top edge). Must be >= 0. Default: 0.
        x_max (int): Maximum x-coordinate of the crop region (right edge). Must be > x_min. Default: 1024.
        y_max (int): Maximum y-coordinate of the crop region (bottom edge). Must be > y_min. Default: 1024.
        pad_if_needed (bool): Whether to pad if crop coordinates exceed image dimensions. Default: False.
        border_mode (OpenCV flag): OpenCV border mode used for padding. Default: cv2.BORDER_CONSTANT.
        fill (tuple[float, ...] | float): Padding value if border_mode is cv2.BORDER_CONSTANT. Default: 0.
        fill_mask (tuple[float, ...] | float): Padding value for masks. Default: 0.
        pad_position (Literal['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'random']):
            Position of padding. Default: 'center'.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32


    Supported bboxes:
        hbb, obb
    Note:
        - The crop coordinates are applied as follows: x_min <= x < x_max and y_min <= y < y_max.
        - If pad_if_needed is False and crop region extends beyond image boundaries, it will be clipped.
        - If pad_if_needed is True, image will be padded to accommodate the full crop region.
        - For bounding boxes and keypoints, coordinates are adjusted appropriately for both padding and cropping.

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
        >>> # Example 1: Basic crop with fixed coordinates
        >>> transform = A.Compose([
        ...     A.Crop(x_min=20, y_min=20, x_max=80, y_max=80),
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
        >>> transformed_image = transformed['image']  # Will be 60x60 - cropped from (20,20) to (80,80)
        >>> transformed_mask = transformed['mask']    # Will be 60x60
        >>> transformed_bboxes = transformed['bboxes']  # Bounding boxes adjusted to the cropped area
        >>> transformed_bbox_labels = transformed['bbox_labels']  # Labels for boxes that remain after cropping
        >>>
        >>> # Example 2: Crop with padding when the crop region extends beyond image dimensions
        >>> transform_padded = A.Compose([
        ...     A.Crop(
        ...         x_min=50, y_min=50, x_max=150, y_max=150,  # Extends beyond the 100x100 image
        ...         pad_if_needed=True,
        ...         border_mode=cv2.BORDER_CONSTANT,
        ...         fill=0,      # Black padding for image
        ...         fill_mask=0  # Zero padding for mask
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the padded transform
        >>> padded_transformed = transform_padded(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # The result will be 100x100 (50:150, 50:150) with padding on right and bottom
        >>> padded_image = padded_transformed['image']  # 100x100 with 50 pixels of original + 50 pixels of padding
        >>> padded_mask = padded_transformed['mask']
        >>> padded_bboxes = padded_transformed['bboxes']  # Coordinates adjusted to the cropped and padded area
        >>>
        >>> # Example 3: Crop with reflection padding and custom position
        >>> transform_reflect = A.Compose([
        ...     A.Crop(
        ...         x_min=-20, y_min=-20, x_max=80, y_max=80,  # Negative coordinates (outside image)
        ...         pad_if_needed=True,
        ...         border_mode=cv2.BORDER_REFLECT_101,  # Reflect image for padding
        ...         pad_position="top_left"  # Apply padding at top-left
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']))
        >>>
        >>> # The resulting crop will use reflection padding for the negative coordinates
        >>> reflect_result = transform_reflect(
        ...     image=image,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels
        ... )

    """

    class InitSchema(BaseCropAndPad.InitSchema):
        x_min: Annotated[int, Field(ge=0)]
        y_min: Annotated[int, Field(ge=0)]
        x_max: Annotated[int, Field(gt=0)]
        y_max: Annotated[int, Field(gt=0)]
        border_mode: BorderModeType

        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

        @model_validator(mode="after")
        def _validate_coordinates(self) -> Self:
            if not self.x_min < self.x_max:
                msg = "x_max must be greater than x_min"
                raise ValueError(msg)
            if not self.y_min < self.y_max:
                msg = "y_max must be greater than y_min"
                raise ValueError(msg)

            return self

    def __init__(
        self,
        x_min: int = 0,
        y_min: int = 0,
        x_max: int = 1024,
        y_max: int = 1024,
        pad_if_needed: bool = False,
        pad_position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"] = "center",
        border_mode: BorderModeType = CV2_BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        p: float = 1.0,
    ):
        super().__init__(
            pad_if_needed=pad_if_needed,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
            pad_position=pad_position,
            p=p,
        )
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    # New helper function for computing minimum padding
    def _compute_min_padding(self, image_height: int, image_width: int) -> tuple[int, int, int, int]:
        pad_top = 0
        pad_bottom = max(0, self.y_max - image_height)
        pad_left = 0
        pad_right = max(0, self.x_max - image_width)
        return pad_top, pad_bottom, pad_left, pad_right

    # New helper function for distributing and adjusting padding
    def _compute_adjusted_padding(self, pad_top: int, pad_bottom: int, pad_left: int, pad_right: int) -> dict[str, int]:
        delta_h = pad_top + pad_bottom
        delta_w = pad_left + pad_right
        pad_top_dist = delta_h // 2
        pad_bottom_dist = delta_h - pad_top_dist
        pad_left_dist = delta_w // 2
        pad_right_dist = delta_w - pad_left_dist

        (pad_top_adj, pad_bottom_adj, pad_left_adj, pad_right_adj) = fgeometric.adjust_padding_by_position(
            h_top=pad_top_dist,
            h_bottom=pad_bottom_dist,
            w_left=pad_left_dist,
            w_right=pad_right_dist,
            position=self.pad_position,
            py_random=self.py_random,
        )

        final_top = max(pad_top_adj, pad_top)
        final_bottom = max(pad_bottom_adj, pad_bottom)
        final_left = max(pad_left_adj, pad_left)
        final_right = max(pad_right_adj, pad_right)

        return {
            "pad_top": final_top,
            "pad_bottom": final_bottom,
            "pad_left": final_left,
            "pad_right": final_right,
        }

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        image_shape = params["shape"][:2]
        image_height, image_width = image_shape

        if not self.pad_if_needed:
            return {"crop_coords": (self.x_min, self.y_min, self.x_max, self.y_max), "pad_params": None}

        pad_top, pad_bottom, pad_left, pad_right = self._compute_min_padding(image_height, image_width)
        pad_params = None

        if any([pad_top, pad_bottom, pad_left, pad_right]):
            pad_params = self._compute_adjusted_padding(pad_top, pad_bottom, pad_left, pad_right)

        return {"crop_coords": (self.x_min, self.y_min, self.x_max, self.y_max), "pad_params": pad_params}


class CropAndPad(DualTransform):
    """Crop or pad each side by pixels (px) or fractions (percent). Positive pad, negative crop.
    Per-side control via tuples. Good for letterboxing or trimming.

    This transform allows for simultaneous cropping and padding of images. Cropping removes pixels from the sides
    (i.e., extracts a subimage), while padding adds pixels to the sides (e.g., black pixels). The amount of
    cropping/padding can be specified either in absolute pixels or as a fraction of the image size.

    Args:
        px (int, tuple of int, tuple of tuples of int, or None):
            The number of pixels to crop (negative values) or pad (positive values) on each side of the image.
            Either this or the parameter `percent` may be set, not both at the same time.
            - If int: crop/pad all sides by this value.
            - If tuple of 2 ints: crop/pad by (top/bottom, left/right).
            - If tuple of 4 ints: crop/pad by (top, right, bottom, left).
            - Each int can also be a tuple of 2 ints for a range.
            Default: None.

        percent (float, tuple of float, tuple of tuples of float, or None):
            The fraction of the image size to crop (negative values) or pad (positive values) on each side.
            Either this or the parameter `px` may be set, not both at the same time.
            - If float: crop/pad all sides by this fraction.
            - If tuple of 2 floats: crop/pad by (top/bottom, left/right) fractions.
            - If tuple of 4 floats: crop/pad by (top, right, bottom, left) fractions.
            - Each float can also be a tuple of 2 floats for a range.
            Default: None.

        border_mode (int):
            OpenCV border mode used for padding. Default: cv2.BORDER_CONSTANT.

        fill (tuple[float, ...] | float):
            The constant value to use for padding if border_mode is cv2.BORDER_CONSTANT.
            Default: 0.

        fill_mask (tuple[float, ...] | float):
            Same as fill but used for mask padding. Default: 0.

        keep_size (bool):
            If True, the output image will be resized to the input image size after cropping/padding.
            Default: True.

        sample_independently (bool):
            If True and ranges are used for px/percent, sample a value for each side independently.
            If False, sample one value and use it for all sides. Default: True.

        interpolation (int):
            OpenCV interpolation flag used for resizing if keep_size is True.
            Default: cv2.INTER_LINEAR.

        mask_interpolation (int):
            OpenCV interpolation flag used for resizing if keep_size is True.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.

        p (float):
            Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32


    Supported bboxes:
        hbb, obb
    Note:
        - This transform will never crop images below a height or width of 1.
        - When using pixel values (px), the image will be cropped/padded by exactly that many pixels.
        - When using percentages (percent), the amount of crop/pad will be calculated based on the image size.
        - Bounding boxes that end up fully outside the image after cropping will be removed.
        - Keypoints that end up outside the image after cropping will be removed.

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
        >>> # Example 1: Using px parameter with specific values for each side
        >>> # Crop 10px from top, pad 20px on right, pad 30px on bottom, crop 40px from left
        >>> transform_px = A.Compose([
        ...     A.CropAndPad(
        ...         px=(-10, 20, 30, -40),  # (top, right, bottom, left)
        ...         border_mode=cv2.BORDER_CONSTANT,
        ...         fill=128,  # Gray padding color
        ...         fill_mask=0,
        ...         keep_size=False,  # Don't resize back to original dimensions
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> result_px = transform_px(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data with px parameters
        >>> transformed_image_px = result_px['image']  # Shape will be different from original
        >>> transformed_mask_px = result_px['mask']
        >>> transformed_bboxes_px = result_px['bboxes']  # Adjusted to new dimensions
        >>> transformed_bbox_labels_px = result_px['bbox_labels']  # Bounding box labels after crop
        >>> transformed_keypoints_px = result_px['keypoints']  # Adjusted to new dimensions
        >>> transformed_keypoint_labels_px = result_px['keypoint_labels']  # Keypoint labels after crop
        >>>
        >>> # Example 2: Using percent parameter as a single value
        >>> # This will pad all sides by 10% of image dimensions
        >>> transform_percent = A.Compose([
        ...     A.CropAndPad(
        ...         percent=0.1,  # Pad all sides by 10%
        ...         border_mode=cv2.BORDER_REFLECT,  # Use reflection padding
        ...         keep_size=True,  # Resize back to original dimensions
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> result_percent = transform_percent(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data with percent parameters
        >>> # Since keep_size=True, image dimensions remain the same (100x100)
        >>> transformed_image_pct = result_percent['image']
        >>> transformed_mask_pct = result_percent['mask']
        >>> transformed_bboxes_pct = result_percent['bboxes']
        >>> transformed_bbox_labels_pct = result_percent['bbox_labels']
        >>> transformed_keypoints_pct = result_percent['keypoints']
        >>> transformed_keypoint_labels_pct = result_percent['keypoint_labels']
        >>>
        >>> # Example 3: Random padding within a range
        >>> # Pad top and bottom by 5-15%, left and right by 10-20%
        >>> transform_random = A.Compose([
        ...     A.CropAndPad(
        ...         percent=[(0.05, 0.15), (0.1, 0.2), (0.05, 0.15), (0.1, 0.2)],  # (top, right, bottom, left)
        ...         sample_independently=True,  # Sample each side independently
        ...         border_mode=cv2.BORDER_CONSTANT,
        ...         fill=0,  # Black padding
        ...         keep_size=False,
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Result dimensions will vary based on the random padding values chosen

    """

    _targets = ALL_TARGETS
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})

    class InitSchema(BaseTransformInitSchema):
        px: PxType | None
        percent: PercentType | None
        keep_size: bool
        sample_independently: bool
        interpolation: FullInterpolationType
        mask_interpolation: FullInterpolationType
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float
        border_mode: BorderModeType

        @model_validator(mode="after")
        def _check_px_percent(self) -> Self:
            if self.px is None and self.percent is None:
                msg = "Both px and percent parameters cannot be None simultaneously."
                raise ValueError(msg)
            if self.px is not None and self.percent is not None:
                msg = "Only px or percent may be set!"
                raise ValueError(msg)

            return self

    def __init__(
        self,
        px: PxType | None = None,
        percent: PercentType | None = None,
        keep_size: bool = True,
        sample_independently: bool = True,
        interpolation: FullInterpolationType = CV2_INTER_LINEAR,
        mask_interpolation: FullInterpolationType = CV2_INTER_NEAREST,
        border_mode: BorderModeType = CV2_BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        p: float = 1.0,
    ):
        super().__init__(p=p)

        self.px = px
        self.percent = percent

        self.border_mode = border_mode
        self.fill = fill
        self.fill_mask = fill_mask

        self.keep_size = keep_size
        self.sample_independently = sample_independently

        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation

    def apply(
        self,
        img: ImageType,
        crop_params: tuple[int, int, int, int] | None,
        pad_params: tuple[int, int, int, int] | None,
        fill: tuple[float, ...] | float,
        **params: Any,
    ) -> ImageType:
        return fcrops.crop_and_pad(
            img,
            crop_params,
            pad_params,
            fill,
            params["shape"][:2],
            self.interpolation,
            self.border_mode,
            self.keep_size,
        )

    def apply_to_mask(
        self,
        mask: ImageType,
        crop_params: tuple[int, int, int, int] | None,
        pad_params: tuple[int, int, int, int] | None,
        fill_mask: tuple[float, ...] | float,
        **params: Any,
    ) -> ImageType:
        return fcrops.crop_and_pad(
            mask,
            crop_params,
            pad_params,
            fill_mask,
            params["shape"][:2],
            self.mask_interpolation,
            self.border_mode,
            self.keep_size,
        )

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        crop_params: tuple[int, int, int, int] | None,
        pad_params: tuple[int, int, int, int] | None,
        result_shape: tuple[int, int],
        **params: Any,
    ) -> np.ndarray:
        return fcrops.crop_and_pad_bboxes(bboxes, crop_params, pad_params, params["shape"][:2], result_shape)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        crop_params: tuple[int, int, int, int] | None,
        pad_params: tuple[int, int, int, int] | None,
        result_shape: tuple[int, int],
        **params: Any,
    ) -> np.ndarray:
        return fcrops.crop_and_pad_keypoints(
            keypoints,
            crop_params,
            pad_params,
            params["shape"][:2],
            result_shape,
            self.keep_size,
        )

    @staticmethod
    def __prevent_zero(val1: int, val2: int, max_val: int) -> tuple[int, int]:
        regain = abs(max_val) + 1
        regain1 = regain // 2
        regain2 = regain // 2
        if regain1 + regain2 < regain:
            regain1 += 1

        if regain1 > val1:
            diff = regain1 - val1
            regain1 = val1
            regain2 += diff
        elif regain2 > val2:
            diff = regain2 - val2
            regain2 = val2
            regain1 += diff

        return val1 - regain1, val2 - regain2

    @staticmethod
    def _prevent_zero(crop_params: list[int], height: int, width: int) -> list[int]:
        top, right, bottom, left = crop_params

        remaining_height = height - (top + bottom)
        remaining_width = width - (left + right)

        if remaining_height < 1:
            top, bottom = CropAndPad.__prevent_zero(top, bottom, height)
        if remaining_width < 1:
            left, right = CropAndPad.__prevent_zero(left, right, width)

        return [max(top, 0), max(right, 0), max(bottom, 0), max(left, 0)]

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        height, width = params["shape"][:2]
        percent_params: list[float] | None = None

        if self.px is not None:
            new_params = self._get_px_params()
        else:
            percent_params = self._get_percent_params()
            new_params = [
                int(percent_params[0] * height),
                int(percent_params[1] * width),
                int(percent_params[2] * height),
                int(percent_params[3] * width),
            ]

        pad_params = [max(i, 0) for i in new_params]

        crop_params = self._prevent_zero([-min(i, 0) for i in new_params], height, width)

        top, right, bottom, left = crop_params
        crop_params = [left, top, width - right, height - bottom]
        result_rows = crop_params[3] - crop_params[1]
        result_cols = crop_params[2] - crop_params[0]
        if result_cols == width and result_rows == height:
            crop_params = []

        top, right, bottom, left = pad_params
        pad_params = [top, bottom, left, right]
        if any(pad_params):
            result_rows += top + bottom
            result_cols += left + right
        else:
            pad_params = []

        sampled_fill = None if pad_params is None else self._get_pad_value(self.fill)
        sampled_fill_mask = (
            None if pad_params is None else self._get_pad_value(cast("tuple[float, ...] | float", self.fill_mask))
        )

        applied_config: dict[str, Any] = {}
        if self.px is not None:
            applied_config["px"] = tuple(new_params)
        else:
            if percent_params is None:
                raise RuntimeError("percent_params must be initialized when px is not set")
            applied_config["percent"] = tuple(percent_params)
        if sampled_fill is not None:
            applied_config["fill"] = sampled_fill
        if sampled_fill_mask is not None:
            applied_config["fill_mask"] = sampled_fill_mask
        self.applied_config = applied_config

        return {
            "crop_params": tuple(crop_params) if crop_params else None,
            "pad_params": tuple(pad_params) if pad_params else None,
            "fill": sampled_fill,
            "fill_mask": sampled_fill_mask,
            "result_shape": (result_rows, result_cols),
        }

    def _get_px_params(self) -> list[int]:
        if self.px is None:
            msg = "px is not set"
            raise ValueError(msg)

        if isinstance(self.px, int):
            return [self.px] * 4
        if len(self.px) == PAIR:
            if self.sample_independently:
                return [self.py_random.randrange(*self.px) for _ in range(4)]
            px = self.py_random.randrange(*self.px)
            return [px] * 4
        if isinstance(self.px[0], int):
            return list(cast("tuple[int, int, int, int]", self.px))
        # len(self.px[0]) == PAIR case - each element is a range tuple
        return [self.py_random.randrange(*cast("tuple[int, int]", i)) for i in self.px]

    def _get_percent_params(self) -> list[float]:
        if self.percent is None:
            msg = "percent is not set"
            raise ValueError(msg)

        if isinstance(self.percent, float):
            params = [self.percent] * 4
        elif len(self.percent) == PAIR:
            if self.sample_independently:
                params = [self.py_random.uniform(*self.percent) for _ in range(4)]
            else:
                px = self.py_random.uniform(*self.percent)
                params = [px] * 4
        elif isinstance(self.percent[0], (int, float)):
            params = list(cast("tuple[float, float, float, float]", self.percent))
        else:
            # len(self.percent[0]) == PAIR case - each element is a range tuple
            params = [self.py_random.uniform(*cast("tuple[float, float]", i)) for i in self.percent]

        return params  # params = [top, right, bottom, left]

    def _get_pad_value(
        self,
        fill: Sequence[float] | float,
    ) -> int | float:
        if isinstance(fill, (list, tuple)):
            if len(fill) == PAIR:
                a, b = fill
                if isinstance(a, int) and isinstance(b, int):
                    return self.py_random.randint(a, b)
                return self.py_random.uniform(a, b)
            return self.py_random.choice(fill)

        if isinstance(fill, (int, float)):
            return fill

        msg = "fill should be a number or list, or tuple of two numbers."
        raise ValueError(msg)


__all__ = [
    "CenterCrop",
    "Crop",
    "CropAndPad",
    "RandomCrop",
]
