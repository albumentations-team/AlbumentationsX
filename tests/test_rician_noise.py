import numpy as np
import pytest

import albumentations as A


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
