"""Base crop transform classes and shared crop schemas."""

from typing import Annotated, Any, ClassVar, Literal, cast

from ._transforms_shared import (
    ALL_TARGETS,
    CV2_INTER_LINEAR,
    CV2_INTER_NEAREST,
    AfterValidator,
    BaseTransformInitSchema,
    BorderModeType,
    DualTransform,
    FullInterpolationType,
    ImageType,
    StackedMasks4D,
    VolumeType,
    check_range_bounds,
    cv2,
    denormalize_bboxes,
    fcrops,
    fgeometric,
    normalize_bboxes,
    np,
)


class CropSizeError(Exception):
    """Raised when requested crop dimensions are incompatible with image size, required padding, or generated crop
    coordinate constraints.

    Used by crop transforms to fail early before generating invalid crop coordinates.
    """


class BaseCrop(DualTransform):
    """Abstract base for crop-only transforms. Subclasses return crop_coords from
    get_params_dependent_on_data. All targets cropped consistently.

    This abstract class provides the foundation for all cropping transformations.
    It handles cropping of different data types including images, masks, bounding boxes,
    keypoints, and volumes while keeping their spatial relationships intact.

    Child classes must implement the `get_params_dependent_on_data` method to determine
    crop coordinates based on transform-specific logic. This method should return a dictionary
    containing at least a 'crop_coords' key with a tuple value (x_min, y_min, x_max, y_max).

    Args:
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        This class is not meant to be used directly. Instead, use or create derived
        transforms that implement the specific cropping behavior required.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> from albumentations.augmentations.crops.transforms import BaseCrop
        >>>
        >>> # Example of a custom crop transform that inherits from BaseCrop
        >>> class CustomCenterCrop(BaseCrop):
        ...     '''A simple custom center crop with configurable size'''
        ...     def __init__(self, crop_height, crop_width, p=1.0):
        ...         super().__init__(p=p)
        ...         self.crop_height = crop_height
        ...         self.crop_width = crop_width
        ...
        ...     def get_params_dependent_on_data(self, params, data):
        ...         '''Calculate crop coordinates based on center of image'''
        ...         image_height, image_width = params["shape"][:2]
        ...
        ...         # Calculate center crop coordinates
        ...         x_min = max(0, (image_width - self.crop_width) // 2)
        ...         y_min = max(0, (image_height - self.crop_height) // 2)
        ...         x_max = min(image_width, x_min + self.crop_width)
        ...         y_max = min(image_height, y_min + self.crop_height)
        ...
        ...         return {"crop_coords": (x_min, y_min, x_max, y_max)}
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Use the custom transform in a pipeline
        >>> transform = A.Compose(
        ...     [CustomCenterCrop(crop_height=80, crop_width=80)],
        ...     bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...     keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels'])
        ... )
        >>>
        >>> # Apply the transform to data
        >>> result = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> transformed_image = result['image']  # Will be 80x80
        >>> transformed_mask = result['mask']    # Will be 80x80
        >>> transformed_bboxes = result['bboxes']  # Bounding boxes adjusted to the cropped area
        >>> transformed_bbox_labels = result['bbox_labels']  # Labels for bboxes that remain after cropping
        >>> transformed_keypoints = result['keypoints']  # Keypoints adjusted to the cropped area
        >>> transformed_keypoint_labels = result['keypoint_labels']  # Labels for keypoints that remain after cropping

    """

    _targets = ALL_TARGETS
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})

    def apply(
        self,
        img: ImageType,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> ImageType:
        return fcrops.crop(img, x_min=crop_coords[0], y_min=crop_coords[1], x_max=crop_coords[2], y_max=crop_coords[3])

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        return fcrops.crop_bboxes_by_coords(bboxes, crop_coords, params["shape"][:2])

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        return fcrops.crop_keypoints_by_coords(keypoints, crop_coords)

    def apply_to_mask(
        self,
        mask: ImageType,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> ImageType:
        if mask.size == 0:
            # Return empty array with cropped dimensions
            # Assume mask shape is (H, W, C)
            crop_height = crop_coords[3] - crop_coords[1]
            crop_width = crop_coords[2] - crop_coords[0]
            return cast("ImageType", np.empty((crop_height, crop_width, mask.shape[2]), dtype=mask.dtype))
        return self.apply(mask, crop_coords, **params)

    def apply_to_masks(
        self,
        masks: StackedMasks4D,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> StackedMasks4D:
        if masks.size == 0:
            crop_height = crop_coords[3] - crop_coords[1]
            crop_width = crop_coords[2] - crop_coords[0]
            return StackedMasks4D(np.empty((0, crop_height, crop_width, masks.shape[3]), dtype=masks.dtype))
        return StackedMasks4D(self.apply_to_images(masks, crop_coords, **params))

    def apply_to_images(
        self,
        images: ImageType,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> ImageType:
        return fcrops.volume_crop_yx(images, crop_coords[0], crop_coords[1], crop_coords[2], crop_coords[3])

    def apply_to_volume(
        self,
        volume: VolumeType,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> VolumeType:
        return fcrops.volume_crop_yx(volume, crop_coords[0], crop_coords[1], crop_coords[2], crop_coords[3])

    def apply_to_volumes(
        self,
        volumes: VolumeType,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> VolumeType:
        return fcrops.volumes_crop_yx(volumes, crop_coords[0], crop_coords[1], crop_coords[2], crop_coords[3])

    def apply_to_mask3d(
        self,
        mask3d: VolumeType,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> VolumeType:
        if mask3d.size == 0:
            # Return empty array with cropped dimensions
            # Assume mask3d shape is (D, H, W, C)
            crop_height = crop_coords[3] - crop_coords[1]
            crop_width = crop_coords[2] - crop_coords[0]
            return cast(
                "VolumeType",
                np.empty((mask3d.shape[0], crop_height, crop_width, mask3d.shape[3]), dtype=mask3d.dtype),
            )
        return self.apply_to_images(mask3d, crop_coords, **params)

    def apply_to_masks3d(
        self,
        masks3d: VolumeType,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> VolumeType:
        if masks3d.size == 0:
            # Return empty array with cropped dimensions
            # Assume masks3d shape is (N, D, H, W, C)
            crop_height = crop_coords[3] - crop_coords[1]
            crop_width = crop_coords[2] - crop_coords[0]
            return cast(
                "VolumeType",
                np.empty((0, masks3d.shape[1], crop_height, crop_width, masks3d.shape[4]), dtype=masks3d.dtype),
            )
        return self.apply_to_volumes(masks3d, crop_coords, **params)

    @staticmethod
    def _clip_bbox(bbox: tuple[int, int, int, int], image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        height, width = image_shape[:2]
        x_min, y_min, x_max, y_max = bbox
        x_min = np.clip(x_min, 0, width)
        y_min = np.clip(y_min, 0, height)

        x_max = np.clip(x_max, x_min, width)
        y_max = np.clip(y_max, y_min, height)
        return x_min, y_min, x_max, y_max


class BaseCropAndPad(BaseCrop):
    """Abstract base for crop+pad transforms (e.g. fixed size). Adds pad_if_needed,
    border_mode, fill, pad_position to BaseCrop. Subclasses define crop and pad logic.

    This abstract class extends BaseCrop by adding padding capabilities. It's the foundation
    for transforms that may need to both crop parts of the input and add padding, such as when
    converting inputs to a specific target size. The class handles the complexities of applying
    these operations to different data types (images, masks, bounding boxes, keypoints) while
    maintaining their spatial relationships.

    Child classes must implement the `get_params_dependent_on_data` method to determine
    crop coordinates and padding parameters based on transform-specific logic.

    Args:
        pad_if_needed (bool): Whether to pad the input if the crop size exceeds input dimensions.
        border_mode (int): OpenCV border mode used for padding.
        fill (tuple[float, ...] | float): Value to fill the padded area if border_mode is BORDER_CONSTANT.
            For multi-channel images, this can be a tuple with a value for each channel.
        fill_mask (tuple[float, ...] | float): Value to fill the padded area in masks.
        pad_position (Literal['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'random']):
            Position of padding when pad_if_needed is True.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        This class is not meant to be used directly. Instead, use or create derived
        transforms that implement the specific cropping and padding behavior required.

    Examples:
        >>> import numpy as np
        >>> import cv2
        >>> import albumentations as A
        >>> from albumentations.augmentations.crops.transforms import BaseCropAndPad
        >>>
        >>> # Example of a custom transform that inherits from BaseCropAndPad
        >>> # This transform crops to a fixed size, padding if needed to maintain dimensions
        >>> class CustomFixedSizeCrop(BaseCropAndPad):
        ...     '''A custom fixed-size crop that pads if needed to maintain output size'''
        ...     def __init__(
        ...         self,
        ...         height=224,
        ...         width=224,
        ...         offset_x=0,  # Offset for crop position
        ...         offset_y=0,  # Offset for crop position
        ...         pad_if_needed=True,
        ...         border_mode=cv2.BORDER_CONSTANT,
        ...         fill=0,
        ...         fill_mask=0,
        ...         pad_position="center",
        ...         p=1.0,
        ...     ):
        ...         super().__init__(
        ...             pad_if_needed=pad_if_needed,
        ...             border_mode=border_mode,
        ...             fill=fill,
        ...             fill_mask=fill_mask,
        ...             pad_position=pad_position,
        ...             p=p,
        ...         )
        ...         self.height = height
        ...         self.width = width
        ...         self.offset_x = offset_x
        ...         self.offset_y = offset_y
        ...
        ...     def get_params_dependent_on_data(self, params, data):
        ...         '''Calculate crop coordinates and padding if needed'''
        ...         image_shape = params["shape"][:2]
        ...         image_height, image_width = image_shape
        ...
        ...         # Calculate crop coordinates with offsets
        ...         x_min = self.offset_x
        ...         y_min = self.offset_y
        ...         x_max = min(x_min + self.width, image_width)
        ...         y_max = min(y_min + self.height, image_height)
        ...
        ...         # Get padding params if needed
        ...         pad_params = self._get_pad_params(
        ...             image_shape,
        ...             (self.height, self.width)
        ...         ) if self.pad_if_needed else None
        ...
        ...         return {
        ...             "crop_coords": (x_min, y_min, x_max, y_max),
        ...             "pad_params": pad_params,
        ...         }
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Use the custom transform in a pipeline
        >>> # This will create a 224x224 crop with padding as needed
        >>> transform = A.Compose(
        ...     [CustomFixedSizeCrop(
        ...         height=224,
        ...         width=224,
        ...         offset_x=20,
        ...         offset_y=10,
        ...         fill=127,  # Gray color for padding
        ...         fill_mask=0
        ...     )],
        ...     bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...     keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform to data
        >>> result = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> transformed_image = result['image']  # Will be 224x224 with padding
        >>> transformed_mask = result['mask']    # Will be 224x224 with padding
        >>> transformed_bboxes = result['bboxes']  # Bounding boxes adjusted to the cropped and padded area
        >>> transformed_bbox_labels = result['bbox_labels']  # Bounding box labels after crop
        >>> transformed_keypoints = result['keypoints']  # Keypoints adjusted to the cropped and padded area
        >>> transformed_keypoint_labels = result['keypoint_labels']  # Keypoint labels after crop

    """

    class InitSchema(BaseTransformInitSchema):
        pad_if_needed: bool
        border_mode: BorderModeType
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float
        pad_position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"]

    def __init__(
        self,
        pad_if_needed: bool,
        border_mode: BorderModeType,
        fill: tuple[float, ...] | float,
        fill_mask: tuple[float, ...] | float,
        pad_position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"],
        p: float,
    ):
        super().__init__(p=p)
        self.pad_if_needed = pad_if_needed
        self.border_mode = border_mode
        self.fill = fill
        self.fill_mask = fill_mask
        self.pad_position = pad_position

    def _get_pad_params(self, image_shape: tuple[int, int], target_shape: tuple[int, int]) -> dict[str, Any] | None:
        """Compute pad amounts (top, right, bottom, left) and position so image reaches
        target_shape. Returns None if no padding needed or pad_if_needed is False.
        """
        if not self.pad_if_needed:
            return None

        h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = fgeometric.get_padding_params(
            image_shape=image_shape,
            min_height=target_shape[0],
            min_width=target_shape[1],
            pad_height_divisor=None,
            pad_width_divisor=None,
        )

        if h_pad_top == h_pad_bottom == w_pad_left == w_pad_right == 0:
            return None

        h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = fgeometric.adjust_padding_by_position(
            h_top=h_pad_top,
            h_bottom=h_pad_bottom,
            w_left=w_pad_left,
            w_right=w_pad_right,
            position=self.pad_position,
            py_random=self.py_random,
        )

        return {
            "pad_top": h_pad_top,
            "pad_bottom": h_pad_bottom,
            "pad_left": w_pad_left,
            "pad_right": w_pad_right,
        }

    def apply(
        self,
        img: ImageType,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> ImageType:
        pad_params = params.get("pad_params")
        if pad_params is not None:
            img = fgeometric.pad_with_params(
                img,
                pad_params["pad_top"],
                pad_params["pad_bottom"],
                pad_params["pad_left"],
                pad_params["pad_right"],
                border_mode=self.border_mode,
                value=self.fill,
            )
        return BaseCrop.apply(self, img, crop_coords, **params)

    def apply_to_mask(
        self,
        mask: ImageType,
        crop_coords: Any,
        **params: Any,
    ) -> ImageType:
        pad_params = params.get("pad_params")
        if pad_params is not None:
            mask = fgeometric.pad_with_params(
                mask,
                pad_params["pad_top"],
                pad_params["pad_bottom"],
                pad_params["pad_left"],
                pad_params["pad_right"],
                border_mode=self.border_mode,
                value=self.fill_mask,
            )
        # Note' that super().apply would apply the padding twice as it is looped to this.apply
        return BaseCrop.apply(self, mask, crop_coords=crop_coords, **params)

    def apply_to_images(
        self,
        images: ImageType,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> ImageType:
        pad_params = params.get("pad_params")
        if pad_params is not None:
            images = fcrops.pad_along_axes(
                images,
                pad_params["pad_top"],
                pad_params["pad_bottom"],
                pad_params["pad_left"],
                pad_params["pad_right"],
                h_axis=1,
                w_axis=2,
                border_mode=self.border_mode,
                pad_value=self.fill,
            )
        return BaseCrop.apply_to_images(self, images, crop_coords, **params)

    def apply_to_volumes(
        self,
        volumes: VolumeType,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> VolumeType:
        pad_params = params.get("pad_params")
        if pad_params is not None:
            volumes = fcrops.pad_along_axes(
                volumes,
                pad_params["pad_top"],
                pad_params["pad_bottom"],
                pad_params["pad_left"],
                pad_params["pad_right"],
                h_axis=2,
                w_axis=3,
                border_mode=self.border_mode,
                pad_value=self.fill,
            )
        return BaseCrop.apply_to_volumes(self, volumes, crop_coords, **params)

    def apply_to_mask3d(
        self,
        mask3d: VolumeType,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> VolumeType:
        return self.apply_to_images(mask3d, crop_coords, **params)

    def apply_to_masks3d(
        self,
        masks3d: VolumeType,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> VolumeType:
        return self.apply_to_volumes(masks3d, crop_coords, **params)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        pad_params = params.get("pad_params")
        image_shape = params["shape"][:2]

        if pad_params is not None:
            # First denormalize bboxes to absolute coordinates
            bboxes_np = denormalize_bboxes(bboxes, image_shape)

            # Apply padding to bboxes (already works with absolute coordinates)
            bboxes_np = fgeometric.pad_bboxes(
                bboxes_np,
                pad_params["pad_top"],
                pad_params["pad_bottom"],
                pad_params["pad_left"],
                pad_params["pad_right"],
                self.border_mode,
                image_shape=image_shape,
            )

            # Update shape to padded dimensions
            padded_height = image_shape[0] + pad_params["pad_top"] + pad_params["pad_bottom"]
            padded_width = image_shape[1] + pad_params["pad_left"] + pad_params["pad_right"]
            padded_shape = (padded_height, padded_width)

            bboxes_np = normalize_bboxes(bboxes_np, padded_shape)

            params["shape"] = padded_shape

            return BaseCrop.apply_to_bboxes(self, bboxes_np, crop_coords, **params)

        # If no padding, use original function behavior
        return BaseCrop.apply_to_bboxes(self, bboxes, crop_coords, **params)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        pad_params = params.get("pad_params")
        image_shape = params["shape"][:2]

        if pad_params is not None:
            # Calculate padded dimensions
            padded_height = image_shape[0] + pad_params["pad_top"] + pad_params["pad_bottom"]
            padded_width = image_shape[1] + pad_params["pad_left"] + pad_params["pad_right"]

            # First apply padding to keypoints using original image shape
            keypoints = fgeometric.pad_keypoints(
                keypoints,
                pad_params["pad_top"],
                pad_params["pad_bottom"],
                pad_params["pad_left"],
                pad_params["pad_right"],
                self.border_mode,
                image_shape=image_shape,
            )

            # Update image shape for subsequent crop operation
            params = {**params, "shape": (padded_height, padded_width)}

        return BaseCrop.apply_to_keypoints(self, keypoints, crop_coords, **params)


class BaseRandomSizedCropInitSchema(BaseTransformInitSchema):
    """Shared validation schema for random sized crop transforms that sample source crop windows before resizing
    them to the final user-requested output dimensions.

    Keeps common size validation in one place for both random sized crop variants.
    """

    size: Annotated[tuple[int, int], AfterValidator(check_range_bounds(1, None))]


class _BaseRandomSizedCropInitSchema(BaseRandomSizedCropInitSchema):
    interpolation: FullInterpolationType
    mask_interpolation: FullInterpolationType
    area_for_downscale: Literal["image", "image_mask"] | None


class _BaseRandomSizedCrop(DualTransform):
    """Abstract base for random crop then resize to fixed size. Subclasses pick crop region;
    output shape (height, width). Bboxes and keypoints scaled with the crop.

    This abstract class provides the foundation for RandomSizedCrop and RandomResizedCrop transforms.
    It handles cropping and resizing for different data types (image, mask, bboxes, keypoints) while
    maintaining their spatial relationships.

    Child classes must implement the `get_params_dependent_on_data` method to determine how the
    crop coordinates are selected according to transform-specific parameters and logic.

    Args:
        size (tuple[int, int]): Target size (height, width) after cropping and resizing.
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm
            for image resizing. Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): Flag that is used to specify the interpolation
            algorithm for mask resizing. Default: cv2.INTER_NEAREST.
        area_for_downscale (Literal[None, "image", "image_mask"]): Controls automatic use of INTER_AREA interpolation
            for downscaling. Options:
            - None: No automatic interpolation selection, always use the specified interpolation method
            - "image": Use INTER_AREA when downscaling images, retain specified interpolation for upscaling and masks
            - "image_mask": Use INTER_AREA when downscaling both images and masks
            Default: None.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        This class is not meant to be used directly. Instead, use derived transforms
        like RandomSizedCrop or RandomResizedCrop that implement specific crop selection
        strategies.
        When area_for_downscale is set, INTER_AREA interpolation will be used automatically for
        downscaling (when the crop is larger than the target size), which provides better quality for size reduction.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Example of a custom transform that inherits from _BaseRandomSizedCrop
        >>> class CustomRandomCrop(_BaseRandomSizedCrop):
        ...     def __init__(
        ...         self,
        ...         size=(224, 224),
        ...         custom_parameter=0.5,
        ...         interpolation=cv2.INTER_LINEAR,
        ...         mask_interpolation=cv2.INTER_NEAREST,
        ...         area_for_downscale="image",
        ...         p=1.0
        ...     ):
        ...         super().__init__(
        ...             size=size,
        ...             interpolation=interpolation,
        ...             mask_interpolation=mask_interpolation,
        ...             area_for_downscale=area_for_downscale,
        ...             p=p,
        ...         )
        ...         self.custom_parameter = custom_parameter
        ...
        ...     def get_params_dependent_on_data(self, params, data):
        ...         # Custom logic to select crop coordinates
        ...         image_height, image_width = params["shape"][:2]
        ...
        ...         # Simple example: calculate crop size based on custom_parameter
        ...         crop_height = int(image_height * self.custom_parameter)
        ...         crop_width = int(image_width * self.custom_parameter)
        ...
        ...         # Random position
        ...         y1 = self.py_random.randint(0, image_height - crop_height + 1)
        ...         x1 = self.py_random.randint(0, image_width - crop_width + 1)
        ...         y2 = y1 + crop_height
        ...         x2 = x1 + crop_width
        ...
        ...         return {"crop_coords": (x1, y1, x2, y2)}
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Create a pipeline with our custom transform
        >>> transform = A.Compose(
        ...     [CustomRandomCrop(size=(64, 64), custom_parameter=0.6, area_for_downscale="image")],
        ...     bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...     keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels'])
        ... )
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
        >>> transformed_bboxes = transformed['bboxes']  # Bounding boxes adjusted to new dimensions
        >>> transformed_bbox_labels = transformed['bbox_labels']  # Labels for bboxes that remain after cropping
        >>> transformed_keypoints = transformed['keypoints']  # Keypoints adjusted to new dimensions
        >>> transformed_keypoint_labels = transformed['keypoint_labels']  # Labels for keypoints that remain

    """

    InitSchema: ClassVar[type[BaseTransformInitSchema]] = _BaseRandomSizedCropInitSchema

    def __init__(
        self,
        size: tuple[int, int],
        interpolation: FullInterpolationType = CV2_INTER_LINEAR,
        mask_interpolation: FullInterpolationType = CV2_INTER_NEAREST,
        area_for_downscale: Literal["image", "image_mask"] | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.size = size
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.area_for_downscale = area_for_downscale

    def _get_interpolation_for_resize(self, crop_shape: tuple[int, int], target_type: str) -> int:
        """Choose OpenCV interpolation for resizing crop to self.size. INTER_AREA when
        downscaling if area_for_downscale set; else image or mask interpolation.

        Args:
            crop_shape (tuple[int, int]): Shape of the crop (height, width)
            target_type (str): Either "image" or "mask" to determine base interpolation

        Returns:
            int: OpenCV interpolation flag.

        """
        crop_height, crop_width = crop_shape
        target_height, target_width = self.size

        # Determine if this is downscaling
        is_downscale = (crop_height > target_height) or (crop_width > target_width)

        # Use INTER_AREA for downscaling if configured
        if (is_downscale and (target_type == "image" and self.area_for_downscale in ["image", "image_mask"])) or (
            target_type == "mask" and self.area_for_downscale == "image_mask"
        ):
            return cv2.INTER_AREA
        # Get base interpolation
        return self.interpolation if target_type == "image" else self.mask_interpolation

    def apply(
        self,
        img: ImageType,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> ImageType:
        crop = fcrops.crop(img, *crop_coords)
        interpolation = self._get_interpolation_for_resize(cast("tuple[int, int]", crop.shape[:2]), "image")
        return fgeometric.resize(crop, self.size, interpolation)

    def apply_to_mask(
        self,
        mask: ImageType,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> ImageType:
        crop = fcrops.crop(mask, *crop_coords)
        interpolation = self._get_interpolation_for_resize(cast("tuple[int, int]", crop.shape[:2]), "mask")
        return fgeometric.resize(crop, self.size, interpolation)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        return fcrops.crop_bboxes_by_coords(bboxes, crop_coords, params["shape"])

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        # First, crop the keypoints
        cropped_keypoints = fcrops.crop_keypoints_by_coords(keypoints, crop_coords)

        # Calculate the dimensions of the crop
        crop_height = crop_coords[3] - crop_coords[1]
        crop_width = crop_coords[2] - crop_coords[0]

        # Calculate scaling factors
        scale_x = self.size[1] / crop_width
        scale_y = self.size[0] / crop_height

        # Scale the cropped keypoints
        return fgeometric.keypoints_scale(cropped_keypoints, scale_x, scale_y)

    def apply_to_images(
        self,
        images: ImageType,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> ImageType:
        # First crop the volume using volume_crop_yx (reduces data size)
        crop = fcrops.volume_crop_yx(images, *crop_coords)

        # Get interpolation method based on crop dimensions
        interpolation = self._get_interpolation_for_resize(cast("tuple[int, int]", crop.shape[1:3]), "image")

        # Then resize the smaller cropped volume using the selected interpolation
        result = np.empty((images.shape[0], self.size[0], self.size[1], crop.shape[-1]), dtype=crop.dtype)
        for i in range(images.shape[0]):
            result[i] = fgeometric.resize(crop[i], self.size, interpolation)
        return cast("ImageType", result)

    def apply_to_mask3d(
        self,
        mask3d: VolumeType,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> VolumeType:
        return self.apply_to_images(mask3d, crop_coords, **params)


__all__ = [
    "BaseCrop",
    "BaseCropAndPad",
    "BaseRandomSizedCropInitSchema",
    "CropSizeError",
    "_BaseRandomSizedCrop",
]
