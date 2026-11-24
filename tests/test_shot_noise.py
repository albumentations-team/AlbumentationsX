import numpy as np
import pytest

import albumentations as A
import albumentations.augmentations.pixel.functional as fpixel
from albumentations.core.tensor import numpy_to_tensor_spatial, tensor_to_numpy_spatial


@pytest.mark.parametrize("target", ["image", "images", "volume", "volumes"])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("channels", [1, 3])
def test_shot_noise_compose_supports_all_targets_and_dtypes(target, container, dtype, channels):
    shapes = {
        "image": (31, 29, channels),
        "images": (2, 31, 29, channels),
        "volume": (3, 31, 29, channels),
        "volumes": (2, 3, 31, 29, channels),
    }
    rng = np.random.default_rng(137)
    if dtype is np.uint8:
        image = rng.integers(0, 256, shapes[target], dtype=np.uint8)
    else:
        half_image = rng.random(shapes[target], dtype=np.float32).astype(np.float16)
        image = half_image if dtype is np.float16 else half_image.astype(dtype)
    original = image.copy()

    def make_source(value):
        source = value.copy()
        if container == "tensor":
            return numpy_to_tensor_spatial(source, target)
        source.flags.writeable = False
        return source

    def run(value):
        source = make_source(value)
        pipeline = A.Compose([A.ShotNoise(scale_range=(0.2, 0.2), p=1.0)], seed=137, strict=True)
        return pipeline(**{target: source})[target], source

    result, source = run(image)
    actual = tensor_to_numpy_spatial(result, target) if container == "tensor" else result
    preserved = tensor_to_numpy_spatial(source, target) if container == "tensor" else source

    assert type(result) is type(source)
    assert actual.shape == image.shape
    assert actual.dtype == dtype
    assert np.isfinite(actual).all()
    if dtype is np.uint8:
        assert np.all((actual >= 0) & (actual <= 255))
    else:
        assert np.all((actual >= 0) & (actual <= 1))
    np.testing.assert_array_equal(preserved, original)

    if dtype is np.float16:
        float32_result, _ = run(image.astype(np.float32))
        expected = tensor_to_numpy_spatial(float32_result, target) if container == "tensor" else float32_result
        np.testing.assert_array_equal(actual, expected.astype(np.float16))


@pytest.mark.parametrize("scale", [0.05, 0.3, 1.0])
@pytest.mark.parametrize("level", [0, 17, 128, 255])
def test_shot_noise_uint8_matches_poisson_output_distribution(scale, level):
    rng = np.random.default_rng(137)
    num_samples = 200_000
    image = np.full((1, num_samples, 1), level, dtype=np.uint8)

    actual = fpixel.shot_noise(image, scale, rng).ravel()

    expected_counts = rng.poisson((level / 255) ** 2.2 / scale + 1e-6, num_samples)
    expected = np.rint(255 * np.clip(expected_counts * scale, 0, 1) ** (1 / 2.2)).astype(np.uint8)
    actual_pmf = np.bincount(actual, minlength=256) / num_samples
    expected_pmf = np.bincount(expected, minlength=256) / num_samples
    assert 0.5 * np.abs(actual_pmf - expected_pmf).sum() < 0.01
