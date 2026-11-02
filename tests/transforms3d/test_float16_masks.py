import cv2
import numpy as np
import pytest
import torch

import albumentations as A
from albumentations.core.tensor import numpy_to_tensor_spatial, tensor_to_numpy_spatial


@pytest.mark.parametrize("operation", ["affine", "elastic"])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("explicit_channels", [False, True])
@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR])
def test_resampling_preserves_fractional_float16_masks(
    operation: str,
    container: str,
    batched: bool,
    explicit_channels: bool,
    interpolation: int,
) -> None:
    volume = np.full((5, 7, 5, 1), 0.5, dtype=np.float16)
    shape = (5, 7, 5, 3) if explicit_channels else (5, 7, 5)
    if operation == "affine":
        mask = np.resize(np.array([-0.25, 0.25, 0.75, 2.25], dtype=np.float16), shape)
        transform = A.Affine3D(
            translate_percent_range={"x": (0.2, 0.2), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            interpolation=interpolation,
            mask_interpolation=interpolation,
            fill_mask=0.75,
            p=1,
        )
        expected = np.full_like(mask, 0.75)
        expected[:, :, 1:] = mask[:, :, :-1]
    else:
        mask = np.full(shape, 0.25, dtype=np.float16)
        transform = A.ElasticTransform3D(
            displacement_range=(0.05, 0.05),
            control_grid_shape=(7, 7),
            interpolation=interpolation,
            mask_interpolation=interpolation,
            border_mode=cv2.BORDER_REPLICATE,
            p=1,
        )
        expected = mask.copy()
    volume_target, mask_target = ("volumes", "masks3d") if batched else ("volume", "mask3d")
    if batched:
        volume = np.stack([volume, volume])
        mask = np.stack([mask, mask])
        expected = np.stack([expected, expected])
    original = mask.copy()
    volume_value = numpy_to_tensor_spatial(volume, volume_target) if container == "tensor" else volume
    if container == "tensor":
        mask_value = numpy_to_tensor_spatial(mask, mask_target) if explicit_channels else torch.from_numpy(mask)
    else:
        mask_value = mask
    result = A.Compose([transform], seed=137, strict=True)(**{volume_target: volume_value, mask_target: mask_value})
    actual = result[mask_target]
    if isinstance(actual, torch.Tensor):
        actual = tensor_to_numpy_spatial(actual, mask_target) if explicit_channels else actual.numpy()
    assert actual.dtype == np.float16
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(mask, original)
