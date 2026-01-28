"""Geometric transformations for flip and symmetry operations.

This module contains transforms that apply various flip and symmetry operations
to images and other target types. These transforms modify the geometric arrangement
of the input data while preserving the pixel values themselves.

Available transforms:
- VerticalFlip: Flips the input upside down (around the x-axis)
- HorizontalFlip: Flips the input left to right (around the y-axis)
- Transpose: Swaps rows and columns (flips around the main diagonal)
- D4: Applies one of eight possible square symmetry transformations (dihedral group D4)
- SquareSymmetry: Alias for D4 with a more intuitive name

These transforms are particularly useful for:
- Data augmentation to improve model generalization
- Addressing orientation biases in training data
- Working with data that doesn't have a natural orientation (e.g., satellite imagery)
- Exploiting symmetries in the problem domain

All transforms support various target types including images, masks, bounding boxes,
keypoints, volumes, and 3D masks, ensuring consistent transformation across
different data modalities.
"""

from typing import Any, Literal

import numpy as np
from albucore import hflip, vflip

from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    DualTransform,
)
from albumentations.core.type_definitions import (
    ALL_TARGETS,
    ImageType,
    VolumeType,
    d4_group_elements,
)

from . import functional as fgeometric

__all__ = [
    "D4",
    "HorizontalFlip",
    "SquareSymmetry",
    "Transpose",
    "VerticalFlip",
]


class VerticalFlip(DualTransform):
    """Flip the input vertically around the x-axis.

    Args:
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32


    Supported bboxes:
        hbb, obb
    Note:
        - This transform flips the image upside down. The top of the image becomes the bottom and vice versa.
        - The dimensions of the image remain unchanged.
        - For multi-channel images (like RGB), each channel is flipped independently.
        - Bounding boxes are adjusted to match their new positions in the flipped image.
        - Keypoints are moved to their new positions in the flipped image.

    Mathematical Details:
        1. For an input image I of shape (H, W, C), the output O is:
           O[i, j, k] = I[H-1-i, j, k] for all i in [0, H-1], j in [0, W-1], k in [0, C-1]
        2. For bounding boxes with coordinates (x_min, y_min, x_max, y_max):
           new_bbox = (x_min, H-y_max, x_max, H-y_min)
        3. For keypoints with coordinates (x, y):
           new_keypoint = (x, H-y)
        where H is the height of the image.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.array([
        ...     [[1, 2, 3], [4, 5, 6]],
        ...     [[7, 8, 9], [10, 11, 12]]
        ... ])
        >>> transform = A.VerticalFlip(p=1.0)
        >>> result = transform(image=image)
        >>> flipped_image = result['image']
        >>> print(flipped_image)
        [[[ 7  8  9]
          [10 11 12]]
         [[ 1  2  3]
          [ 4  5  6]]]
        # The original image is flipped vertically, with rows reversed

    """

    _targets = ALL_TARGETS
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        return vflip(img)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        bbox_type = params.get("bbox_type", "hbb")
        return fgeometric.bboxes_vflip(bboxes, bbox_type=bbox_type)

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.keypoints_vflip(keypoints, params["shape"][0])

    def apply_to_mask(self, mask: ImageType, **params: Any) -> ImageType:
        if mask.size == 0:
            # Assume mask shape is (H, W, C) - return empty array with same shape
            return mask
        return self.apply(mask, **params)

    def apply_to_masks(self, masks: ImageType, **params: Any) -> ImageType:
        if masks.size == 0:
            # Assume masks shape is (N, H, W, C) - return empty array with same shape
            return masks
        return self.apply_to_images(masks, **params)

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return fgeometric.vflip_images(images)

    def apply_to_volumes(self, volumes: VolumeType, **params: Any) -> VolumeType:
        return fgeometric.vflip_volumes(volumes)

    def apply_to_mask3d(self, mask3d: VolumeType, **params: Any) -> VolumeType:
        if mask3d.size == 0:
            # Assume mask3d shape is (D, H, W, C) - return empty array with same shape
            return mask3d
        return self.apply_to_images(mask3d, **params)

    def apply_to_masks3d(self, masks3d: VolumeType, **params: Any) -> VolumeType:
        if masks3d.size == 0:
            # Assume masks3d shape is (N, D, H, W, C) - return empty array with same shape
            return masks3d
        return self.apply_to_volumes(masks3d, **params)


class HorizontalFlip(DualTransform):
    """Flip the input horizontally around the y-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32


    Supported bboxes:
        hbb, obb
    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Prepare sample data
        >>> image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        >>> mask = np.array([[1, 0], [0, 1]])
        >>> bboxes = np.array([[0.1, 0.5, 0.3, 0.9]])  # [x_min, y_min, x_max, y_max] format
        >>> keypoints = np.array([[0.1, 0.5], [0.9, 0.5]])  # [x, y] format
        >>>
        >>> # Create a transform with horizontal flip
        >>> transform = A.Compose([
        ...     A.HorizontalFlip(p=1.0)  # Always apply for this example
        ... ], bbox_params=A.BboxParams(format='yolo', label_fields=[]),
        ...    keypoint_params=A.KeypointParams(format='normalized'))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
        >>>
        >>> # Get the transformed data
        >>> flipped_image = transformed["image"]  # Image flipped horizontally
        >>> flipped_mask = transformed["mask"]    # Mask flipped horizontally
        >>> flipped_bboxes = transformed["bboxes"]  # BBox coordinates adjusted for horizontal flip
        >>> flipped_keypoints = transformed["keypoints"]  # Keypoint x-coordinates flipped

    """

    _targets = ALL_TARGETS
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        return hflip(img)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        bbox_type = params.get("bbox_type", "hbb")
        return fgeometric.bboxes_hflip(bboxes, bbox_type=bbox_type)

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.keypoints_hflip(keypoints, params["shape"][1])

    def apply_to_mask(self, mask: ImageType, **params: Any) -> ImageType:
        if mask.size == 0:
            # Assume mask shape is (H, W, C) - return empty array with same shape
            return mask
        return self.apply(mask, **params)

    def apply_to_masks(self, masks: ImageType, **params: Any) -> ImageType:
        if masks.size == 0:
            # Assume masks shape is (N, H, W, C) - return empty array with same shape
            return masks
        return self.apply_to_images(masks, **params)

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return fgeometric.hflip_images(images)

    def apply_to_volumes(self, volumes: VolumeType, **params: Any) -> VolumeType:
        return fgeometric.hflip_volumes(volumes)

    def apply_to_mask3d(self, mask3d: VolumeType, **params: Any) -> VolumeType:
        if mask3d.size == 0:
            # Assume mask3d shape is (D, H, W, C) - return empty array with same shape
            return mask3d
        return self.apply_to_images(mask3d, **params)

    def apply_to_masks3d(self, masks3d: VolumeType, **params: Any) -> VolumeType:
        if masks3d.size == 0:
            # Assume masks3d shape is (N, D, H, W, C) - return empty array with same shape
            return masks3d
        return self.apply_to_volumes(masks3d, **params)


class Transpose(DualTransform):
    """Transpose the input by swapping its rows and columns.

    This transform flips the image over its main diagonal, effectively switching its width and height.
    It's equivalent to a 90-degree rotation followed by a horizontal flip.

    Args:
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Supported bboxes:
        hbb, obb

    Note:
        - The dimensions of the output will be swapped compared to the input. For example,
          an input image of shape (100, 200, 3) will result in an output of shape (200, 100, 3).
        - This transform is its own inverse. Applying it twice will return the original input.
        - For multi-channel images (like RGB), the channels are preserved in their original order.
        - Bounding boxes will have their coordinates adjusted to match the new image dimensions.
        - Keypoints will have their x and y coordinates swapped.

    Mathematical Details:
        1. For an input image I of shape (H, W, C), the output O is:
           O[i, j, k] = I[j, i, k] for all i in [0, W-1], j in [0, H-1], k in [0, C-1]
        2. For bounding boxes with coordinates (x_min, y_min, x_max, y_max):
           new_bbox = (y_min, x_min, y_max, x_max)
        3. For keypoints with coordinates (x, y):
           new_keypoint = (y, x)

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.array([
        ...     [[1, 2, 3], [4, 5, 6]],
        ...     [[7, 8, 9], [10, 11, 12]]
        ... ])
        >>> transform = A.Transpose(p=1.0)
        >>> result = transform(image=image)
        >>> transposed_image = result['image']
        >>> print(transposed_image)
        [[[ 1  2  3]
          [ 7  8  9]]
         [[ 4  5  6]
          [10 11 12]]]
        # The original 2x2x3 image is now 2x2x3, with rows and columns swapped

    """

    _targets = ALL_TARGETS
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        return fgeometric.transpose(img)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        bbox_type = params.get("bbox_type", "hbb")
        return fgeometric.bboxes_transpose(bboxes, bbox_type=bbox_type)

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.keypoints_transpose(keypoints)

    def apply_to_mask(self, mask: ImageType, **params: Any) -> ImageType:
        if mask.size == 0:
            # Transpose swaps H and W
            # Assume mask shape is (H, W, C) -> (W, H, C)
            return np.empty((mask.shape[1], mask.shape[0], mask.shape[2]), dtype=mask.dtype)
        return self.apply(mask, **params)

    def apply_to_masks(self, masks: ImageType, **params: Any) -> ImageType:
        if masks.size == 0:
            # Transpose swaps H and W
            # Assume masks shape is (N, H, W, C) -> (N, W, H, C)
            return np.empty((0, masks.shape[2], masks.shape[1], masks.shape[3]), dtype=masks.dtype)
        return self.apply_to_images(masks, **params)

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return fgeometric.transpose_images(images)

    def apply_to_volumes(self, volumes: VolumeType, **params: Any) -> VolumeType:
        return fgeometric.transpose_volumes(volumes)

    def apply_to_mask3d(self, mask3d: VolumeType, **params: Any) -> VolumeType:
        if mask3d.size == 0:
            # Transpose swaps H and W
            # Assume mask3d shape is (D, H, W, C) -> (D, W, H, C)
            return np.empty((mask3d.shape[0], mask3d.shape[2], mask3d.shape[1], mask3d.shape[3]), dtype=mask3d.dtype)
        return self.apply_to_images(mask3d, **params)

    def apply_to_masks3d(self, masks3d: VolumeType, **params: Any) -> VolumeType:
        if masks3d.size == 0:
            # Transpose swaps H and W
            # Assume masks3d shape is (N, D, H, W, C) -> (N, D, W, H, C)
            return np.empty(
                (0, masks3d.shape[1], masks3d.shape[3], masks3d.shape[2], masks3d.shape[4]),
                dtype=masks3d.dtype,
            )
        return self.apply_to_volumes(masks3d, **params)


class D4(DualTransform):
    """Applies one of the eight possible D4 dihedral group transformations to a square-shaped input,
    maintaining the square shape. These transformations correspond to the symmetries of a square,
    including rotations and reflections.

    The D4 group transformations include:
    - 'e' (identity): No transformation is applied.
    - 'r90' (rotation by 90 degrees counterclockwise)
    - 'r180' (rotation by 180 degrees)
    - 'r270' (rotation by 270 degrees counterclockwise)
    - 'v' (reflection across the vertical midline)
    - 'hvt' (reflection across the anti-diagonal)
    - 'h' (reflection across the horizontal midline)
    - 't' (reflection across the main diagonal)

    Even if the probability (`p`) of applying the transform is set to 1, the identity transformation
    'e' may still occur, which means the input will remain unchanged in one out of eight cases.

    Args:
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Supported bboxes:
        hbb, obb

    Note:
        - This transform is particularly useful for augmenting data that does not have a clear orientation,
          such as top-view satellite or drone imagery, or certain types of medical images.
        - The input image should be square-shaped for optimal results. Non-square inputs may lead to
          unexpected behavior or distortions.
        - When applied to bounding boxes or keypoints, their coordinates will be adjusted according
          to the selected transformation.
        - This transform preserves the aspect ratio and size of the input.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.Compose([
        ...     A.D4(p=1.0),
        ... ])
        >>> transformed = transform(image=image)
        >>> transformed_image = transformed['image']
        # The resulting image will be one of the 8 possible D4 transformations of the input

    """

    _targets = ALL_TARGETS
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})

    class InitSchema(BaseTransformInitSchema):
        pass

    def __init__(
        self,
        p: float = 1,
    ):
        super().__init__(p=p)

    def apply(
        self,
        img: ImageType,
        group_element: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"],
        **params: Any,
    ) -> ImageType:
        return fgeometric.d4(img, group_element)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        group_element: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"],
        **params: Any,
    ) -> np.ndarray:
        bbox_type = params.get("bbox_type", "hbb")
        return fgeometric.bboxes_d4(bboxes, group_element, bbox_type=bbox_type)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        group_element: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"],
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.keypoints_d4(keypoints, group_element, params["shape"])

    def apply_to_mask(
        self,
        mask: ImageType,
        group_element: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"],
        **params: Any,
    ) -> ImageType:
        if mask.size == 0:
            # Group elements that transpose dimensions: "r90", "r270", "t", "hvt"
            # Assume mask shape is (H, W, C)
            if group_element in {"r90", "r270", "t", "hvt"}:
                # These swap H and W: (H, W, C) -> (W, H, C)
                return np.empty((mask.shape[1], mask.shape[0], mask.shape[2]), dtype=mask.dtype)
            # Other elements preserve dimensions: "e", "r180", "v", "h"
            return mask
        return self.apply(mask, group_element)

    def apply_to_masks(
        self,
        masks: ImageType,
        group_element: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"],
        **params: Any,
    ) -> ImageType:
        if masks.size == 0:
            # Group elements that transpose dimensions: "r90", "r270", "t", "hvt"
            # Assume masks shape is (N, H, W, C)
            if group_element in {"r90", "r270", "t", "hvt"}:
                # These swap H and W: (N, H, W, C) -> (N, W, H, C)
                return np.empty((0, masks.shape[2], masks.shape[1], masks.shape[3]), dtype=masks.dtype)
            # Other elements preserve dimensions: "e", "r180", "v", "h"
            return masks
        return self.apply_to_images(masks, group_element)

    def apply_to_images(
        self,
        images: ImageType,
        group_element: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"],
        **params: Any,
    ) -> ImageType:
        return fgeometric.d4_images(images, group_element)

    def apply_to_volumes(
        self,
        volumes: VolumeType,
        group_element: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"],
        **params: Any,
    ) -> VolumeType:
        return fgeometric.d4_images(volumes, group_element)

    def apply_to_mask3d(
        self,
        mask3d: VolumeType,
        group_element: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"],
        **params: Any,
    ) -> VolumeType:
        if mask3d.size == 0:
            # Group elements that transpose dimensions: "r90", "r270", "t", "hvt"
            # Assume mask3d shape is (D, H, W, C)
            if group_element in {"r90", "r270", "t", "hvt"}:
                # These swap H and W: (D, H, W, C) -> (D, W, H, C)
                return np.empty(
                    (mask3d.shape[0], mask3d.shape[2], mask3d.shape[1], mask3d.shape[3]),
                    dtype=mask3d.dtype,
                )
            # Other elements preserve dimensions: "e", "r180", "v", "h"
            return mask3d
        return self.apply_to_images(mask3d, group_element)

    def apply_to_masks3d(
        self,
        masks3d: VolumeType,
        group_element: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"],
        **params: Any,
    ) -> VolumeType:
        if masks3d.size == 0:
            # Group elements that transpose dimensions: "r90", "r270", "t", "hvt"
            # Assume masks3d shape is (N, D, H, W, C)
            if group_element in {"r90", "r270", "t", "hvt"}:
                # These swap H and W: (N, D, H, W, C) -> (N, D, W, H, C)
                return np.empty(
                    (0, masks3d.shape[1], masks3d.shape[3], masks3d.shape[2], masks3d.shape[4]),
                    dtype=masks3d.dtype,
                )
            # Other elements preserve dimensions: "e", "r180", "v", "h"
            return masks3d
        return self.apply_to_volumes(masks3d, group_element)

    def get_params(self) -> dict[str, Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]]:
        return {
            "group_element": self.random_generator.choice(d4_group_elements),
        }


class SquareSymmetry(D4):
    """Applies one of the eight possible square symmetry transformations to a square-shaped input.
    This is an alias for D4 transform with a more intuitive name for those not familiar with group theory.

    The square symmetry transformations include:
    - Identity: No transformation is applied
    - 90° rotation: Rotate 90 degrees counterclockwise
    - 180° rotation: Rotate 180 degrees
    - 270° rotation: Rotate 270 degrees counterclockwise
    - Vertical flip: Mirror across vertical axis
    - Anti-diagonal flip: Mirror across anti-diagonal
    - Horizontal flip: Mirror across horizontal axis
    - Main diagonal flip: Mirror across main diagonal

    Args:
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Supported bboxes:
        hbb, obb

    Note:
        - This transform is particularly useful for augmenting data that does not have a clear orientation,
          such as top-view satellite or drone imagery, or certain types of medical images.
        - The input image should be square-shaped for optimal results. Non-square inputs may lead to
          unexpected behavior or distortions.
        - When applied to bounding boxes or keypoints, their coordinates will be adjusted according
          to the selected transformation.
        - This transform preserves the aspect ratio and size of the input.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.Compose([
        ...     A.SquareSymmetry(p=1.0),
        ... ])
        >>> transformed = transform(image=image)
        >>> transformed_image = transformed['image']
        # The resulting image will be one of the 8 possible square symmetry transformations of the input

    """
