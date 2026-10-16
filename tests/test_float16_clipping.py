import cv2
import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.geometric import functional as fgeometric


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("target", ["image", "images", "volume"])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    ("transform_type", "kwargs", "expected_values"),
    [
        (
            A.RandomBrightnessContrast,
            {"brightness_range": (0.5, 0.5), "contrast_range": (0, 0), "brightness_by_max": True},
            [0.5, 0.75, 1, 1],
        ),
        (
            A.RandomBrightnessContrast,
            {"brightness_range": (0, 0), "contrast_range": (0.5, 0.5), "brightness_by_max": True},
            [0, 0.375, 1, 1],
        ),
        (
            A.RandomBrightnessContrast,
            {"brightness_range": (0, 0), "contrast_range": (0.5, 0.5), "brightness_by_max": False},
            [0, 0.125, 0.875, 1],
        ),
        (
            A.ColorJitter,
            {"brightness_range": (1.5, 1.5), "contrast_range": (1, 1), "saturation_range": (1, 1), "hue_range": (0, 0)},
            [0, 0.375, 1, 1],
        ),
        (
            A.ColorJitter,
            {"brightness_range": (1, 1), "contrast_range": (1.5, 1.5), "saturation_range": (1, 1), "hue_range": (0, 0)},
            [0, 0.125, 0.875, 1],
        ),
    ],
    ids=["add-brightness", "scale-contrast", "mean-contrast", "jitter-brightness", "jitter-contrast"],
)
def test_photometric_float_clipping(dtype, target, strided, transform_type, kwargs, expected_values):
    image = np.tile(np.array([0, 0.25, 0.75, 1], dtype=dtype), (8, 2))[..., None]
    expected = np.tile(np.array(expected_values, dtype=dtype), (8, 2))[..., None]
    if target != "image":
        image = np.stack((image, image))
        expected = np.stack((expected, expected))
    if strided:
        image = np.repeat(image, 2, axis=-2)[..., ::2, :]
        assert not image.flags.c_contiguous
    image.setflags(write=False)
    original = image.copy()
    transform = A.Compose([transform_type(**kwargs, p=1)], strict=True, seed=137)

    result = transform(**{target: image})[target]

    assert result.dtype == dtype
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("target", ["image", "images", "volume"])
@pytest.mark.parametrize("elementwise", [False, True])
def test_multiplicative_noise_float_clipping(dtype, target, elementwise):
    image = np.tile(np.array([0, 0.25, 0.75, 1], dtype=dtype), (8, 2))[..., None]
    expected = np.tile(np.array([0, 0.375, 1, 1], dtype=dtype), (8, 2))[..., None]
    if target != "image":
        image = np.stack((image, image))
        expected = np.stack((expected, expected))
    original = image.copy()
    transform = A.Compose([A.MultiplicativeNoise(multiplier=(1.5, 1.5), elementwise=elementwise, p=1)], seed=137)

    result = transform(**{target: image})[target]

    assert result.dtype == dtype
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_exposure_matching_image_float_clipping(dtype):
    image = np.array([0, 0.25, 0.75, 1], dtype=dtype).reshape(2, 2, 1)
    original = image.copy()
    transform = A.Compose([A.ExposureMatching(target_mean_range=(0.75, 0.75), p=1)], seed=137)

    result = transform(image=image)["image"]

    assert result.dtype == dtype
    np.testing.assert_array_equal(result.ravel(), np.array([0, 0.375, 1, 1], dtype=dtype))
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("target", ["image", "images", "volume"])
def test_film_grain_float_clipping(dtype, target):
    shape = (8, 8, 1) if target == "image" else (2, 8, 8, 1)
    image = np.full(shape, 0.5, dtype=dtype)
    image.setflags(write=False)
    transform = A.Compose([A.FilmGrain(intensity_range=(1, 1), grain_size_range=(1, 1), p=1)], seed=137)

    result = transform(**{target: image})[target]

    assert result.dtype == dtype
    assert result.shape == image.shape
    assert result.min() == 0
    assert result.max() == 1
    np.testing.assert_array_equal(image, np.full(shape, 0.5, dtype=dtype))


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("target", ["image", "images", "volume"])
@pytest.mark.parametrize(("mode", "effect_type"), [("corner", "darken"), ("gaussian", "brighten")])
@pytest.mark.parametrize("channels", [1, 3, 5])
def test_illumination_float_clipping(dtype, target, mode, effect_type, channels):
    shape = (8, 8, channels) if target == "image" else (2, 8, 8, channels)
    image = np.ones(shape, dtype=dtype)
    image.setflags(write=False)
    transform = A.Compose(
        [
            A.Illumination(mode=mode, effect_type=effect_type, intensity_range=(0.2, 0.2), p=1),
        ],
        seed=137,
    )

    result = transform(**{target: image})[target]

    assert result.dtype == dtype
    np.testing.assert_array_equal(result, np.ones(shape, dtype=dtype))
    np.testing.assert_array_equal(image, np.ones(shape, dtype=dtype))


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("target", ["image", "images", "volume"])
@pytest.mark.parametrize("interpolation", [cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
def test_resize_float_clipping_preserves_mask_range(dtype, target, interpolation):
    image = np.zeros((8, 8, 1), dtype=dtype)
    image[:, 4:] = 1
    if target != "image":
        image = np.stack((image, image))
    original = image.copy()
    mask_target = {"image": "mask", "images": "masks", "volume": "mask3d"}[target]
    mask = np.full(image.shape[:-1], 137, dtype=np.float32)
    transform = A.Compose([A.Resize(19, 19, interpolation=interpolation, p=1)], seed=137)

    result = transform(**{target: image, mask_target: mask})

    assert result[target].dtype == dtype
    assert result[target].min() == 0
    assert result[target].max() == 1
    np.testing.assert_array_equal(result[mask_target], np.full(result[target].shape[:-1], 137, dtype=np.float32))
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("target", ["image", "images", "volume"])
@pytest.mark.parametrize("interpolation", [cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
def test_same_size_resize_preserves_readonly_input(dtype, target, interpolation):
    shape = (8, 8, 1) if target == "image" else (2, 8, 8, 1)
    image = np.full(shape, 0.5 if dtype != np.uint8 else 137, dtype=dtype)
    original = image.copy()
    image.setflags(write=False)
    transform = A.Compose([A.Resize(8, 8, interpolation=interpolation, p=1)], seed=137)

    result = transform(**{target: image})[target]

    assert result.dtype == image.dtype
    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("dtype", [np.dtype(np.float16), np.dtype(np.float16).newbyteorder("S")])
@pytest.mark.parametrize("interpolation", [cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
@pytest.mark.parametrize("readonly", [False, True])
def test_interpolated_half_clipping_preserves_byte_order(dtype, interpolation, readonly):
    image = np.array([-0.25, 0, 0.25, 1, 1.25], dtype=dtype).reshape(1, 5, 1)
    original = image.copy()
    if readonly:
        image.setflags(write=False)

    result = fgeometric.clip_if_interpolation_can_overshoot(image, interpolation)

    assert result.dtype == dtype
    np.testing.assert_array_equal(result.ravel(), np.array([0, 0, 0.25, 1, 1], dtype=dtype))
    if readonly:
        np.testing.assert_array_equal(image, original)
    else:
        assert result is image


@pytest.mark.parametrize("dtype", [np.dtype(np.float16), np.dtype(np.float16).newbyteorder("S")])
@pytest.mark.parametrize("interpolation", [cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
def test_interpolation_clipping_leaves_shared_views(dtype, interpolation):
    image = np.empty((8, 8, 1), dtype=dtype)
    image[..., 0] = np.tile([0, 0.25, 0.75, 1], (8, 2))
    original = image.copy()
    image.setflags(write=False)
    view = image.transpose(1, 0, 2)

    result = fgeometric.clip_if_interpolation_can_overshoot(view, interpolation, source=image)

    assert result is view
    np.testing.assert_array_equal(result, original.transpose(1, 0, 2))
    np.testing.assert_array_equal(image, original)
