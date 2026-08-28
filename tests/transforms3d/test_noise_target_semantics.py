import numpy as np
import pytest

import albumentations as A

NOISE_TRANSFORM_FACTORIES = [
    pytest.param(
        lambda: A.GaussNoise(std_range=(0.1, 0.1), mean_range=(0.0, 0.0), p=1.0),
        id="gauss-noise",
    ),
    pytest.param(
        lambda: A.ISONoise(color_shift_range=(0.1, 0.1), intensity_range=(0.1, 0.1), p=1.0),
        id="iso-noise",
    ),
    pytest.param(
        lambda: A.MultiplicativeNoise(multiplier=(0.8, 1.2), elementwise=True, p=1.0),
        id="multiplicative-noise",
    ),
    pytest.param(lambda: A.ShotNoise(scale_range=(0.2, 0.2), p=1.0), id="shot-noise"),
    pytest.param(
        lambda: A.AdditiveNoise(
            noise_type="gaussian",
            spatial_mode="per_pixel",
            noise_params={"mean_range": (0.0, 0.0), "std_range": (0.1, 0.1)},
            p=1.0,
        ),
        id="additive-noise",
    ),
    pytest.param(
        lambda: A.SaltAndPepper(amount_range=(0.2, 0.2), salt_vs_pepper_range=(0.5, 0.5), p=1.0),
        id="salt-and-pepper",
    ),
    pytest.param(
        lambda: A.FilmGrain(intensity_range=(0.2, 0.2), grain_size_range=(1, 1), p=1.0),
        id="film-grain",
    ),
    pytest.param(lambda: A.RicianNoise(std_range=(0.1, 0.1), p=1.0), id="rician-noise"),
]


@pytest.mark.parametrize("transform_factory", NOISE_TRANSFORM_FACTORIES)
def test_noise_volume_samples_a_full_depth_field(transform_factory):
    image = np.random.default_rng(137).random((11, 13, 3), dtype=np.float32)
    volume = np.stack([image, image], axis=0)

    result = A.Compose([transform_factory()], seed=137)(volume=volume)["volume"]

    assert not np.array_equal(result[0], result[1])


@pytest.mark.parametrize("transform_factory", NOISE_TRANSFORM_FACTORIES)
def test_noise_targets_keep_2d_batch_behavior_and_use_a_distinct_volume_field(transform_factory):
    image = np.random.default_rng(137).random((11, 13, 3), dtype=np.float32)
    volume = np.stack([image, image], axis=0)
    result = A.Compose([transform_factory()], seed=137)(image=image, images=volume, volume=volume)

    np.testing.assert_allclose(result["image"], result["images"][0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result["images"][0], result["images"][1], rtol=1e-6, atol=1e-6)
    assert not np.array_equal(result["volume"][0], result["volume"][1])


def test_additive_noise_patch_mode_extrudes_its_2d_patch_program_through_depth():
    volume = np.full((2, 11, 13, 1), 0.5, dtype=np.float32)
    transform = A.Compose(
        [
            A.AdditiveNoise(
                noise_type="uniform",
                spatial_mode="patch",
                noise_params={"ranges": [(0.1, 0.1)]},
                patch_count_range=(1, 1),
                patch_height_range=(0.5, 0.5),
                patch_width_range=(0.5, 0.5),
                p=1.0,
            ),
        ],
        seed=137,
    )

    result = transform(volume=volume)["volume"]

    np.testing.assert_array_equal(result[0], result[1])
