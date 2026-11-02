import numpy as np
import pytest

import albumentations.augmentations.pixel._functional_illumination as fillumination
import albumentations.augmentations.pixel.functional as fpixel


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 5])
@pytest.mark.parametrize(("brightness", "contrast"), [(0.19, -0.23), (0.0, 0.23), (0.19, 0.0)])
@pytest.mark.parametrize("readonly", [False, True])
def test_plasma_batch_matches_stepwise_storage_rounding(dtype, channels, brightness, contrast, readonly):
    rng = np.random.default_rng(137)
    images = rng.uniform(0.0625, 0.875, (8, 9, 11, channels)).astype(dtype)
    original = images.copy()
    images.flags.writeable = not readonly
    pattern = rng.random((9, 11), dtype=np.float32)
    channel_pattern = pattern[..., None]
    weights = np.add(np.multiply(channel_pattern, np.float32(contrast)), np.float32(1))
    offsets = np.subtract(np.float32(1), weights)
    expected = []
    for image in original:
        work = image
        if brightness:
            work = np.add(work.astype(np.float32), np.multiply(channel_pattern, np.float32(brightness))).astype(dtype)
        if contrast:
            image_mean = np.mean(work, dtype=np.float64)
            work = np.multiply(work.astype(np.float32), weights).astype(dtype)
            wide_offsets = np.multiply(image_mean, offsets, dtype=np.float64)
            work = np.add(work.astype(np.float32), wide_offsets.astype(np.float32)).astype(dtype)
        expected.append(np.clip(work, 0, 1))

    result = fpixel.apply_plasma_brightness_contrast_batch(images, brightness, contrast, pattern)

    assert result.dtype == dtype
    assert result.shape == images.shape
    assert result.flags.c_contiguous and result.flags.writeable
    assert not np.shares_memory(result, images)
    np.testing.assert_array_equal(images, original)
    np.testing.assert_allclose(result, np.stack(expected), rtol=1e-3 if dtype == np.float16 else 1e-6, atol=1e-7)


def test_plasma_half_preserves_overflow_for_unbounded_image_means():
    values = np.array([60000, -60000, -60000], dtype=np.float16)
    images = np.broadcast_to(values[None, None, :, None], (8, 1, 3, 1)).copy()
    original = images.copy()

    with np.errstate(over="raise", invalid="raise"), pytest.raises(FloatingPointError, match="overflow"):
        fpixel.apply_plasma_brightness_contrast_batch(images, 0, 0.09, np.ones((1, 3), dtype=np.float32))

    np.testing.assert_array_equal(images, original)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_plasma_preserves_invalid_error_for_infinite_means(dtype):
    images = np.full((8, 2, 3, 1), np.inf, dtype=dtype)
    original = images.copy()

    with np.errstate(invalid="raise"), pytest.raises(FloatingPointError, match="invalid"):
        fpixel.apply_plasma_brightness_contrast_batch(images, 0, 0.5, np.ones((2, 3), dtype=np.float32))

    np.testing.assert_array_equal(images, original)


def test_plasma_preserves_wide_tiny_offsets_under_underflow_raise(monkeypatch):
    values = np.array([0.5, -0.5, 1e-37], dtype=np.float32)
    images = np.broadcast_to(values[None, None, :, None], (8, 1, 3, 1)).copy()
    original = images.copy()
    pattern = np.array([[1, 0.5, 0]], dtype=np.float32)

    monkeypatch.setattr(fillumination, "mean", lambda image: np.float64(1e-38))

    with np.errstate(under="raise", over="raise", invalid="raise"):
        result = fpixel.apply_plasma_brightness_contrast_batch(images, 0, 0.5, pattern)

    expected = np.broadcast_to(np.array([0.75, 0, 1e-37], dtype=np.float32)[None, None, :, None], images.shape)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(images, original)
