"""Exact byte-input behavior of the public conversion transform."""

import numpy as np
import pytest

import albumentations as A


@pytest.mark.parametrize("channels", [1, 3, 4, 5, 9])
@pytest.mark.parametrize("max_value", [None, 255, 255.0, 128.0, 17.3, 0.5])
@pytest.mark.parametrize("target", ["image", "images"])
def test_from_float_byte_scalar_promotion(channels, max_value, target):
    values = np.array([0, 1, 2, 127, 128, 254, 255], dtype=np.uint8)
    image = np.broadcast_to(values[None, :, None], (3, 7, channels)).copy()
    source = image if target == "image" else np.stack([image, image])
    original = source.copy()
    if max_value is None and channels <= 4:
        mapped = [0, 255, 254, 129, 128, 2, 1]
    elif max_value is None or max_value == 255:
        mapped = [0, 255, 255, 255, 255, 255, 255]
    elif max_value == 128:
        mapped = [0, 128, 255, 255, 255, 255, 255]
    elif max_value == 17.3:
        mapped = [0, 17, 35, 255, 255, 255, 255]
    else:
        mapped = [0, 0, 1, 64, 64, 127, 128]
    expected = np.broadcast_to(np.array(mapped, dtype=np.uint8)[None, :, None], image.shape)
    if target == "images":
        expected = np.stack([expected, expected])
    pipeline = A.Compose([A.FromFloat(max_value=max_value, p=1)], strict=True, seed=137)

    actual = pipeline(**{target: source})[target]

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, original)
    assert actual.dtype == np.uint8
    assert actual.flags.writeable


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("width", [3, 7, 13])
@pytest.mark.parametrize("max_value", [None, 255.0, 17.3, 0.5])
def test_from_float_omitted_channel_matches_explicit_channel_and_compose(dtype, width, max_value):
    values = np.resize(np.array([0, 1, 2, 127, 128, 254, 255], dtype=np.uint8), (3, width))
    image = values if dtype == np.uint8 else (values.astype(np.float32) / 255).astype(dtype)
    original = image.copy()
    direct = A.FromFloat(max_value=max_value, p=1)(image=image)["image"]
    explicit = A.FromFloat(max_value=max_value, p=1)(image=image[..., None])["image"][..., 0]
    composed = A.Compose([A.FromFloat(max_value=max_value, p=1)], strict=True)(image=image)["image"]

    np.testing.assert_array_equal(direct, explicit)
    np.testing.assert_array_equal(direct, composed)
    np.testing.assert_array_equal(image, original)
    assert direct.shape == image.shape
    assert direct.dtype == np.uint8
