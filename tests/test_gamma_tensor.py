"""Gamma preserves values and ownership while using the native Tensor image route."""

import numpy as np
import pytest
import torch

import albumentations as A
from albumentations.augmentations.pixel import functional as fpixel
from albumentations.core import transforms_interface


def _expected(image, gamma):
    if image.dtype == np.uint8:
        table = ((np.arange(256, dtype=np.float64) / 255) ** gamma * 255).astype(np.uint8)
        return table[image]
    return np.power(image.astype(np.float32), gamma).astype(image.dtype)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 9])
@pytest.mark.parametrize("contiguous", [False, True])
def test_gamma_functional_tensor_values(dtype, channels, contiguous):
    original = np.arange(channels * 5 * 7, dtype=np.float32).reshape(channels, 5, 7)
    original = original.astype(dtype) if dtype == np.uint8 else (original / original.size).astype(dtype)
    image = torch.from_numpy(original.copy())
    if not contiguous:
        image = image.transpose(-2, -1).contiguous().transpose(-2, -1)
    before = image.clone()

    result = fpixel.gamma_transform(image, 1.4)

    expected = _expected(original, 1.4)
    assert result.dtype == image.dtype
    assert result.shape == image.shape
    assert not result.is_inference()
    if dtype == np.float16:
        np.testing.assert_array_max_ulp(result.numpy(), expected, maxulp=1)
    elif dtype == np.uint8:
        np.testing.assert_array_equal(result.numpy(), expected)
    else:
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-6)
    result.zero_()
    torch.testing.assert_close(image, before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("contiguous", [False, True])
def test_gamma_compose_uses_native_tensor_image(dtype, contiguous, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Gamma entered the NumPy fallback")

    monkeypatch.setattr(transforms_interface, "tensor_to_numpy_spatial", fail)
    original = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)
    original = original.astype(dtype) if dtype == np.uint8 else (original / original.size).astype(dtype)
    image = torch.from_numpy(original.copy())
    if not contiguous:
        image = image.transpose(-2, -1).contiguous().transpose(-2, -1)
    before = image.clone()
    pipeline = A.Compose([A.RandomGamma(gamma_range=(140, 140), p=1)], strict=True, seed=137)

    result = pipeline(image=image)["image"]

    expected = torch.from_numpy(_expected(original, 1.4))
    torch.testing.assert_close(result, expected, rtol=0 if dtype == np.uint8 else 1e-5, atol=1e-6)
    result.zero_()
    torch.testing.assert_close(image, before, rtol=0, atol=0)
