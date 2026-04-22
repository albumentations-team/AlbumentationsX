"""Module containing PyTorch-specific transforms for Albumentations.
This module provides transforms that convert NumPy arrays to PyTorch tensors in
the appropriate format. It handles both 2D image data and 3D volumetric data,
ensuring that the tensor dimensions are correctly arranged according to PyTorch's
expected format (channels first). These transforms are typically used as the final
step in an augmentation pipeline before feeding data to a PyTorch model.
"""

from typing import Any

import numpy as np
import torch

from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.type_definitions import (
    MONO_CHANNEL_DIMENSIONS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    NUM_VOLUME_DIMENSIONS,
    ImageType,
    StackedMasks4D,
    Targets,
    VolumeType,
)

__all__ = ["ToTensor3D", "ToTensorV2"]


class ToTensorV2(BasicTransform):
    """Converts images/masks to PyTorch Tensors, inheriting from BasicTransform.
    For images:
        Converts `HWC` format to PyTorch `CHW` format

    Args:
        transpose_mask (bool): If True, transposes 3D input mask dimensions from `[height, width, num_channels]` to
            `[num_channels, height, width]`.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask

    """

    _targets = (Targets.IMAGE, Targets.MASK)

    def __init__(self, transpose_mask: bool = False, p: float = 1.0):
        super().__init__(p=p)
        self.transpose_mask = transpose_mask

    @property
    def targets(self) -> dict[str, Any]:
        """Mapping of target name to function (image, images, mask, masks). Compose uses this
        to dispatch apply vs apply_to_images/apply_to_mask/apply_to_masks.

        Returns:
            dict[str, Any]: Dictionary mapping target names to corresponding transform functions.

        """
        return {
            "image": self.apply,
            "images": self.apply_to_images,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
        }

    def apply(self, img: ImageType, **params: Any) -> torch.Tensor:
        if img.ndim not in {MONO_CHANNEL_DIMENSIONS, NUM_MULTI_CHANNEL_DIMENSIONS}:
            msg = "Albumentations only supports images in HW or HWC format"
            raise ValueError(msg)

        if img.ndim == MONO_CHANNEL_DIMENSIONS:
            img = np.expand_dims(img, 2)

        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))

    def apply_to_mask(self, mask: ImageType, **params: Any) -> torch.Tensor:
        if self.transpose_mask and mask.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
            mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(mask))

    def apply_to_masks(self, masks: StackedMasks4D, **params: Any) -> torch.Tensor:
        arr: np.ndarray = masks
        if self.transpose_mask and arr.ndim == NUM_VOLUME_DIMENSIONS:  # (N, H, W, C)
            arr = np.transpose(arr, (0, 3, 1, 2))  # -> (N, C, H, W)
        return torch.from_numpy(np.ascontiguousarray(arr))

    def apply_to_images(self, images: ImageType, **params: Any) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2)))  # -> (N,C,H,W)


class ToTensor3D(BasicTransform):
    """Convert 3D volumes and masks to PyTorch tensors (D,H,W,C or D,H,W -> C,D,H,W).
    For 3D medical imaging pipelines; p=1.0 by default.

    This transform is designed for 3D medical imaging data. It converts numpy arrays
    to PyTorch tensors and ensures consistent channel positioning.

    For all inputs (volumes and masks):
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
        volume, mask3d

    """

    _targets = (Targets.VOLUME, Targets.MASK3D)

    def __init__(self, p: float = 1.0):
        super().__init__(p=p)

    @property
    def targets(self) -> dict[str, Any]:
        """Return mapping of target name to target function (volume, mask3d). Compose uses
        this to dispatch apply_to_volume vs apply_to_mask3d.

        Returns:
            dict[str, Any]: Dictionary mapping target names to corresponding transform functions

        """
        return {
            "volume": self.apply_to_volume,
            "mask3d": self.apply_to_mask3d,
        }

    def apply_to_volume(self, volume: VolumeType, **params: Any) -> torch.Tensor:
        if volume.ndim == NUM_VOLUME_DIMENSIONS:  # D,H,W,C
            return torch.from_numpy(np.ascontiguousarray(volume.transpose(3, 0, 1, 2)))
        if volume.ndim == NUM_VOLUME_DIMENSIONS - 1:  # D,H,W
            return torch.from_numpy(np.ascontiguousarray(volume[np.newaxis, ...]))
        raise ValueError(f"Expected 3D or 4D array (D,H,W) or (D,H,W,C), got {volume.ndim}D array")

    def apply_to_mask3d(self, mask3d: VolumeType, **params: Any) -> torch.Tensor:
        return self.apply_to_volume(mask3d, **params)
