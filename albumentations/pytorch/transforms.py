"""Module containing PyTorch-specific transforms for Albumentations.
This module provides transforms that convert NumPy arrays to PyTorch tensors in
the appropriate format. It handles both 2D image data and 3D volumetric data,
ensuring that the tensor dimensions are correctly arranged according to PyTorch's
expected format (channels first). These transforms are typically used as the final
step in an augmentation pipeline before feeding data to a PyTorch model.
"""

from typing import Any, ClassVar

import numpy as np
import torch
from albucore.tensor_bridge import numpy_to_tensor_layout, numpy_to_tensor_storage
from numpy.typing import NDArray

from albumentations.core.transforms_interface import BaseTransformInitSchema, BasicTransform
from albumentations.core.type_definitions import (
    MONO_CHANNEL_DIMENSIONS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    NUM_VOLUME_DIMENSIONS,
    StackedMasks4D,
    Targets,
    VolumeType,
)

__all__ = ["ToTensor3D", "ToTensorV2"]


class BaseTensorTransform(BasicTransform):
    """Mark terminal transforms that accept NumPy targets and return Tensors."""

    _is_tensor_terminal = True


def _vector_field_to_tensor(field: np.ndarray, dimensions: int) -> torch.Tensor:
    """Move vector components before their spatial axes and preserve field dtype."""
    if field.ndim != dimensions + 1:
        raise TypeError(f"Expected a {dimensions}D spatial vector field with a component axis, got {field.ndim}D")
    return numpy_to_tensor_layout(field, "CHW" if dimensions == 2 else "CDHW", contiguous=True)


class ToTensorV2(BaseTensorTransform):
    """Converts images/masks to PyTorch Tensors, inheriting from BasicTransform.
    For images:
        Converts `HWC` format to PyTorch `CHW` format

    Args:
        transpose_mask (bool): If True, transposes 3D input mask dimensions from `[height, width, num_channels]` to
            `[num_channels, height, width]`.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, images, mask, masks, gradient, displacement, vector_field, normal

    Image types:
        uint8, float16, float32

    Examples:
        >>> transform = ToTensorV2(transpose_mask=True)

    """

    _targets = (
        Targets.IMAGE,
        Targets.IMAGES,
        Targets.MASK,
        Targets.MASKS,
        Targets.GRADIENT,
        Targets.DISPLACEMENT,
        Targets.VECTOR_FIELD,
        Targets.NORMAL,
    )
    _target_apply_methods: ClassVar[dict[str, str]] = {
        "gradient": "_to_tensor_gradient",
        "displacement": "_to_tensor_displacement",
        "vector_field": "_to_tensor_vector_field",
        "normal": "_to_tensor_normal",
    }

    class InitSchema(BaseTransformInitSchema):
        transpose_mask: bool

    def __init__(self, transpose_mask: bool = False, p: float = 1.0):
        super().__init__(p=p)
        self.transpose_mask = transpose_mask

    def apply(
        self, img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32], *args: Any, **params: Any
    ) -> torch.Tensor:
        if img.ndim not in {MONO_CHANNEL_DIMENSIONS, NUM_MULTI_CHANNEL_DIMENSIONS}:
            msg = "Albumentations only supports images in HW or HWC format"
            raise ValueError(msg)

        if img.ndim == MONO_CHANNEL_DIMENSIONS:
            return numpy_to_tensor_storage(img[np.newaxis, ...], contiguous=True)
        return numpy_to_tensor_layout(img, "CHW", contiguous=True)

    def apply_to_mask(
        self, mask: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32], **params: Any
    ) -> torch.Tensor:
        if self.transpose_mask and mask.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
            return numpy_to_tensor_layout(mask, "CHW", contiguous=True)
        return numpy_to_tensor_storage(mask, contiguous=True)

    def apply_to_masks(self, masks: StackedMasks4D, **params: Any) -> torch.Tensor:
        if self.transpose_mask and masks.ndim == NUM_VOLUME_DIMENSIONS:
            return numpy_to_tensor_layout(masks, "NCHW", contiguous=True)
        return numpy_to_tensor_storage(masks, contiguous=True)

    def apply_to_images(
        self, images: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32], *args: Any, **params: Any
    ) -> torch.Tensor:
        return numpy_to_tensor_layout(images, "NCHW", contiguous=True)

    def _to_tensor_gradient(self, gradient: np.ndarray, **params: Any) -> torch.Tensor:
        return _vector_field_to_tensor(gradient, 2)

    def _to_tensor_displacement(self, displacement: np.ndarray, **params: Any) -> torch.Tensor:
        return _vector_field_to_tensor(displacement, 2)

    def _to_tensor_vector_field(self, vector_field: np.ndarray, **params: Any) -> torch.Tensor:
        return _vector_field_to_tensor(vector_field, 2)

    def _to_tensor_normal(self, normal: np.ndarray, **params: Any) -> torch.Tensor:
        return _vector_field_to_tensor(normal, 2)


class ToTensor3D(BaseTensorTransform):
    """Convert 3D volume data and masks to PyTorch tensors (D,H,W,C or D,H,W -> C,D,H,W).
    For 3D medical imaging pipelines; p=1.0 by default.

    This transform is designed for 3D medical imaging data. It converts numpy arrays
    to PyTorch tensors and ensures consistent channel positioning.

    For all inputs (volume data and masks):
        - Input:  (D, H, W, C) or (D, H, W) - depth, height, width, [channels]
        - Output: (C, D, H, W) - channels first format for PyTorch
                 For single-channel input, adds C=1 dimension

    Note:
        This transform always moves channels to first position as this is
        the standard PyTorch format. For masks that need to stay in DHWC format,
        use a different transform or handle the transposition after this transform.

    Args:
        p (float): Probability of applying the transform. Default: 1.0

    Targets:
        volume, volumes, mask3d, masks3d, gradient3d, displacement3d, vector_field3d, normal3d

    Image types:
        uint8, float16, float32

    Examples:
        >>> transform = ToTensor3D(p=1.0)

    """

    _targets = (
        Targets.VOLUME,
        Targets.VOLUMES,
        Targets.MASK3D,
        Targets.MASKS3D,
        Targets.GRADIENT3D,
        Targets.DISPLACEMENT3D,
        Targets.VECTOR_FIELD3D,
        Targets.NORMAL3D,
    )
    _target_apply_methods: ClassVar[dict[str, str]] = {
        "gradient3d": "_to_tensor_gradient3d",
        "displacement3d": "_to_tensor_displacement3d",
        "vector_field3d": "_to_tensor_vector_field3d",
        "normal3d": "_to_tensor_normal3d",
    }

    def __init__(self, p: float = 1.0):
        super().__init__(p=p)

    def apply_to_volume(self, volume: VolumeType, *args: Any, **params: Any) -> torch.Tensor:
        if volume.ndim == NUM_VOLUME_DIMENSIONS:  # D,H,W,C
            return numpy_to_tensor_layout(volume, "CDHW", contiguous=True)
        if volume.ndim == NUM_VOLUME_DIMENSIONS - 1:  # D,H,W
            return numpy_to_tensor_storage(volume[np.newaxis, ...], contiguous=True)
        raise TypeError(f"volume must be 3D or 4D array (D,H,W) or (D,H,W,C), got {volume.ndim}D array")

    def apply_to_mask3d(self, mask3d: VolumeType, **params: Any) -> torch.Tensor:
        return self.apply_to_volume(mask3d, **params)

    def apply_to_volumes(self, volumes: np.ndarray, **params: Any) -> torch.Tensor:
        if volumes.ndim == NUM_VOLUME_DIMENSIONS + 1:  # N,D,H,W,C
            return numpy_to_tensor_layout(volumes, "NCDHW", contiguous=True)
        if volumes.ndim == NUM_VOLUME_DIMENSIONS:  # N,D,H,W
            return numpy_to_tensor_storage(volumes[:, np.newaxis, ...], contiguous=True)
        raise TypeError(f"volumes must be a 4D or 5D array (N,D,H,W) or (N,D,H,W,C), got {volumes.ndim}D array")

    def apply_to_masks3d(self, masks3d: np.ndarray, **params: Any) -> torch.Tensor:
        return self.apply_to_volumes(masks3d, **params)

    def _to_tensor_gradient3d(self, gradient3d: np.ndarray, **params: Any) -> torch.Tensor:
        return _vector_field_to_tensor(gradient3d, 3)

    def _to_tensor_displacement3d(self, displacement3d: np.ndarray, **params: Any) -> torch.Tensor:
        return _vector_field_to_tensor(displacement3d, 3)

    def _to_tensor_vector_field3d(self, vector_field3d: np.ndarray, **params: Any) -> torch.Tensor:
        return _vector_field_to_tensor(vector_field3d, 3)

    def _to_tensor_normal3d(self, normal3d: np.ndarray, **params: Any) -> torch.Tensor:
        return _vector_field_to_tensor(normal3d, 3)
