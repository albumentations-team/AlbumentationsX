import numpy as np
import pytest
from albucore import from_float, to_float

import albumentations as A
from albumentations.augmentations.medical import functional as fmedical


def test_rician_noise_has_the_expected_low_signal_noise_floor() -> None:
    std = 0.1
    image = np.zeros((256, 256, 1), dtype=np.float32)

    result = A.Compose([A.RicianNoise(std_range=(std, std), p=1.0)], seed=137)(image=image)["image"]

    np.testing.assert_allclose(result.mean(), std * np.sqrt(np.pi / 2), rtol=0.025)


def test_rician_noise_per_channel_mode_controls_field_sharing() -> None:
    image = np.zeros((64, 64, 3), dtype=np.float32)

    shared = A.Compose([A.RicianNoise(std_range=(0.1, 0.1), p=1.0)], seed=137)(image=image)["image"]
    per_channel = A.Compose(
        [A.RicianNoise(std_range=(0.1, 0.1), per_channel=True, p=1.0)],
        seed=137,
    )(image=image)["image"]

    np.testing.assert_array_equal(shared[..., 0], shared[..., 1])
    assert not np.array_equal(per_channel[..., 0], per_channel[..., 1])


@pytest.mark.parametrize(
    "volume",
    [
        np.full((1, 9, 13, 1), 0.4, dtype=np.float32),
        np.asfortranarray(np.full((3, 7, 11, 2), 0.4, dtype=np.float32)),
    ],
    ids=("single-slice", "noncontiguous"),
)
def test_rician_noise_zero_std_is_an_exact_volume_identity(volume: np.ndarray) -> None:
    result = A.Compose([A.RicianNoise(std_range=(0.0, 0.0), p=1.0)], seed=137)(volume=volume)["volume"]

    np.testing.assert_array_equal(result, volume)


@pytest.mark.parametrize("shape", [(63, 65, 3), (7, 233, 239, 3), (5, 457, 459, 1)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "fortran"])
def test_rician_reconstruction_preserves_sampled_fields_and_replay(shape, dtype, per_channel, layout) -> None:
    source = np.random.default_rng(137).integers(0, 256, (*shape[:-2], shape[-2] * 2, shape[-1]), dtype=np.uint8)
    if dtype == np.float32:
        source = to_float(source)
    image = source[..., ::2, :]
    if layout == "contiguous":
        image = image.copy()
    elif layout == "fortran":
        image = np.asfortranarray(image)
    original = image.copy()
    image.setflags(write=False)
    target = "image" if len(shape) == 3 else "volume"
    pipeline = A.ReplayCompose([A.RicianNoise(std_range=(0.1, 0.1), per_channel=per_channel, p=1)], strict=True)
    pipeline.set_random_seed(137)

    result = pipeline(**{target: image})

    record = result["replay"]["transforms"][0]["params"]
    fields = record["target_params"][0]["params"]
    real_noise = fields["real_noise"].copy()
    imaginary_noise = fields["imaginary_noise"].copy()
    signal = to_float(image) if dtype == np.uint8 else image
    expected = np.clip(np.sqrt(np.square(signal + real_noise) + np.square(imaginary_noise)), 0, 1)
    if dtype == np.uint8:
        expected = from_float(expected, target_dtype=np.uint8)
    np.testing.assert_allclose(result[target], expected, atol=1 if dtype == np.uint8 else 2e-7, rtol=0)

    replayed = A.ReplayCompose.replay(result["replay"], **{target: image})

    np.testing.assert_array_equal(replayed[target], result[target])
    np.testing.assert_array_equal(fields["real_noise"], real_noise)
    np.testing.assert_array_equal(fields["imaginary_noise"], imaginary_noise)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize(
    ("shape", "layout"),
    [
        ((1023, 1025, 1), "contiguous"),
        ((1024, 1024, 1), "contiguous"),
        ((1024, 1024, 1), "readonly"),
        ((1024, 1024, 1), "negative-stride"),
    ],
)
def test_rician_backend_boundary_and_noise_layouts_match_reference(shape, layout) -> None:
    generator = np.random.default_rng(137)
    image = generator.random(shape, dtype=np.float32)
    real_noise = generator.standard_normal(shape, dtype=np.float32) * 0.1
    imaginary_noise = generator.standard_normal(shape, dtype=np.float32) * 0.1
    if layout == "readonly":
        imaginary_noise.setflags(write=False)
    elif layout == "negative-stride":
        imaginary_noise = imaginary_noise[::-1]
    originals = [array.copy() for array in (image, real_noise, imaginary_noise)]
    expected = np.clip(np.sqrt(np.square(image + real_noise) + np.square(imaginary_noise)), 0, 1)

    result = fmedical.rician_noise(image, real_noise, imaginary_noise)

    np.testing.assert_allclose(result, expected, atol=2e-7, rtol=0)
    for array, original in zip((image, real_noise, imaginary_noise), originals, strict=True):
        np.testing.assert_array_equal(array, original)
    assert not np.shares_memory(result, image)
