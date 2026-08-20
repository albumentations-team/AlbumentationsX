"""Tests for the RicianNoise transform.

Covers: basic application, identity at zero std, shape/dtype preservation,
seeded reproducibility, volume support, per-channel behaviour, mask
invariance, serialization, edge cases (constant/zero signal, D=1,
non-contiguous, arbitrary channels), and the Rician noise-floor property.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import albumentations as A


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def uint8_image():
    return np.random.RandomState(0).randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def float32_image():
    return np.random.RandomState(0).rand(64, 64, 3).astype(np.float32)


@pytest.fixture
def grayscale_image():
    return np.random.RandomState(0).randint(0, 256, (64, 64), dtype=np.uint8)


@pytest.fixture
def volume():
    return np.random.RandomState(0).randint(0, 256, (16, 32, 32), dtype=np.uint8)


@pytest.fixture
def volume_with_channels():
    return np.random.RandomState(0).randint(0, 256, (8, 32, 32, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Basic application
# ---------------------------------------------------------------------------
class TestBasicApplication:
    def test_output_differs_from_input(self, uint8_image):
        t = A.RicianNoise(std_range=(0.1, 0.2), p=1.0)
        result = t(image=uint8_image)["image"]
        assert not np.array_equal(result, uint8_image)

    def test_uint8_shape_dtype_preserved(self, uint8_image):
        t = A.RicianNoise(std_range=(0.1, 0.2), p=1.0)
        result = t(image=uint8_image)["image"]
        assert result.shape == uint8_image.shape
        assert result.dtype == np.uint8

    def test_float32_shape_dtype_preserved(self, float32_image):
        t = A.RicianNoise(std_range=(0.1, 0.2), p=1.0)
        result = t(image=float32_image)["image"]
        assert result.shape == float32_image.shape
        assert result.dtype == np.float32

    def test_float32_range(self, float32_image):
        t = A.RicianNoise(std_range=(0.1, 0.3), p=1.0)
        result = t(image=float32_image)["image"]
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_uint8_range(self, uint8_image):
        t = A.RicianNoise(std_range=(0.1, 0.3), p=1.0)
        result = t(image=uint8_image)["image"]
        assert result.min() >= 0
        assert result.max() <= 255


# ---------------------------------------------------------------------------
# Identity at zero std
# ---------------------------------------------------------------------------
class TestZeroStd:
    def test_uint8_identity(self, uint8_image):
        t = A.RicianNoise(std_range=(0.0, 0.0), p=1.0)
        result = t(image=uint8_image)["image"]
        np.testing.assert_array_equal(result, uint8_image)

    def test_float32_identity(self, float32_image):
        t = A.RicianNoise(std_range=(0.0, 0.0), p=1.0)
        result = t(image=float32_image)["image"]
        np.testing.assert_array_equal(result, float32_image)

    def test_volume_identity(self, volume):
        t = A.RicianNoise(std_range=(0.0, 0.0), p=1.0)
        result = t(volume=volume)["volume"]
        np.testing.assert_array_equal(result, volume)


# ---------------------------------------------------------------------------
# Seeded reproducibility
# ---------------------------------------------------------------------------
class TestReproducibility:
    def test_same_seed_same_output(self, uint8_image):
        t = A.RicianNoise(std_range=(0.1, 0.2), p=1.0)

        t.set_random_seed(123)
        r1 = t(image=uint8_image)["image"]

        t.set_random_seed(123)
        r2 = t(image=uint8_image)["image"]

        np.testing.assert_array_equal(r1, r2)

    def test_different_seed_different_output(self, uint8_image):
        t = A.RicianNoise(std_range=(0.1, 0.2), p=1.0)

        t.set_random_seed(1)
        r1 = t(image=uint8_image)["image"]

        t.set_random_seed(2)
        r2 = t(image=uint8_image)["image"]

        assert not np.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Volume support
# ---------------------------------------------------------------------------
class TestVolume:
    def test_volume_dhw(self, volume):
        t = A.RicianNoise(std_range=(0.05, 0.10), p=1.0)
        result = t(volume=volume)["volume"]
        assert result.shape == volume.shape
        assert result.dtype == volume.dtype

    def test_volume_dhwc(self, volume_with_channels):
        t = A.RicianNoise(std_range=(0.05, 0.10), p=1.0)
        result = t(volume=volume_with_channels)["volume"]
        assert result.shape == volume_with_channels.shape
        assert result.dtype == volume_with_channels.dtype

    def test_volume_d1(self):
        vol = np.random.randint(0, 256, (1, 32, 32), dtype=np.uint8)
        t = A.RicianNoise(std_range=(0.05, 0.10), p=1.0)
        result = t(volume=vol)["volume"]
        assert result.shape == vol.shape

    def test_volume_differs_from_input(self, volume):
        t = A.RicianNoise(std_range=(0.1, 0.2), p=1.0)
        result = t(volume=volume)["volume"]
        assert not np.array_equal(result, volume)


# ---------------------------------------------------------------------------
# Per-channel vs shared std
# ---------------------------------------------------------------------------
class TestPerChannel:
    def test_per_channel_shape_preserved(self, uint8_image):
        t = A.RicianNoise(std_range=(0.05, 0.15), per_channel=True, p=1.0)
        result = t(image=uint8_image)["image"]
        assert result.shape == uint8_image.shape
        assert result.dtype == uint8_image.dtype

    def test_per_channel_volume(self, volume_with_channels):
        t = A.RicianNoise(std_range=(0.05, 0.15), per_channel=True, p=1.0)
        result = t(volume=volume_with_channels)["volume"]
        assert result.shape == volume_with_channels.shape

    def test_shared_vs_per_channel_differ(self, uint8_image):
        seed = 42
        t_shared = A.RicianNoise(std_range=(0.1, 0.2), per_channel=False, p=1.0)
        t_per_ch = A.RicianNoise(std_range=(0.1, 0.2), per_channel=True, p=1.0)

        t_shared.set_random_seed(seed)
        r_shared = t_shared(image=uint8_image)["image"]

        t_per_ch.set_random_seed(seed)
        r_per_ch = t_per_ch(image=uint8_image)["image"]

        # They should generally differ (different noise application strategy)
        assert not np.array_equal(r_shared, r_per_ch)


# ---------------------------------------------------------------------------
# Masks unchanged
# ---------------------------------------------------------------------------
class TestMaskUnchanged:
    def test_mask_not_modified(self, uint8_image):
        mask = np.random.randint(0, 2, (64, 64), dtype=np.uint8)
        t = A.RicianNoise(std_range=(0.1, 0.2), p=1.0)
        result = t(image=uint8_image, mask=mask)
        np.testing.assert_array_equal(result["mask"], mask)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
class TestSerialization:
    def test_json_round_trip(self):
        t = A.RicianNoise(std_range=(0.03, 0.12), per_channel=True, p=0.7)
        d = t.to_dict()
        json_str = json.dumps(d)
        t2 = A.from_dict(json.loads(json_str))
        assert t2.std_range == t.std_range
        assert t2.per_channel == t.per_channel
        assert t2.p == t.p

    def test_repr(self):
        t = A.RicianNoise(std_range=(0.05, 0.15), p=1.0)
        r = repr(t)
        assert "RicianNoise" in r


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_constant_signal(self):
        """Constant signal should still produce valid output."""
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        t = A.RicianNoise(std_range=(0.1, 0.2), p=1.0)
        result = t(image=img)["image"]
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_zero_signal(self):
        """Zero signal shows the Rician noise floor (positive bias)."""
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        t = A.RicianNoise(std_range=(0.1, 0.2), p=1.0)
        result = t(image=img)["image"]
        assert result.shape == img.shape
        # Rician noise on zero signal produces a positive noise floor
        assert result.mean() > 0

    def test_non_contiguous_input(self, uint8_image):
        nc = np.asfortranarray(uint8_image)
        assert not nc.flags.c_contiguous
        t = A.RicianNoise(std_range=(0.1, 0.2), p=1.0)
        result = t(image=nc)["image"]
        assert result.shape == uint8_image.shape

    def test_arbitrary_channel_count(self):
        """Should work with any number of channels."""
        for num_c in [1, 2, 4, 7]:
            img = np.random.randint(0, 256, (32, 32, num_c), dtype=np.uint8)
            t = A.RicianNoise(std_range=(0.05, 0.1), p=1.0)
            result = t(image=img)["image"]
            assert result.shape == img.shape

    def test_arbitrary_channel_per_channel(self):
        for num_c in [1, 2, 5]:
            img = np.random.randint(0, 256, (32, 32, num_c), dtype=np.uint8)
            t = A.RicianNoise(std_range=(0.05, 0.1), per_channel=True, p=1.0)
            result = t(image=img)["image"]
            assert result.shape == img.shape

    def test_grayscale_hw(self, grayscale_image):
        t = A.RicianNoise(std_range=(0.05, 0.15), p=1.0)
        result = t(image=grayscale_image)["image"]
        assert result.shape == grayscale_image.shape
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_negative_std_raises(self):
        with pytest.raises(Exception):
            A.RicianNoise(std_range=(-0.1, 0.2))

    def test_decreasing_range_raises(self):
        with pytest.raises(Exception):
            A.RicianNoise(std_range=(0.5, 0.1))

    def test_std_above_one_raises(self):
        with pytest.raises(Exception):
            A.RicianNoise(std_range=(0.5, 1.5))


# ---------------------------------------------------------------------------
# Rician noise properties
# ---------------------------------------------------------------------------
class TestRicianProperties:
    def test_rician_bias_at_low_snr(self):
        """At zero signal, Rician noise produces a positive noise floor.

        The expected magnitude is std * sqrt(pi/2) ≈ std * 1.2533.
        """
        img = np.zeros((256, 256), dtype=np.float32)
        std = 0.1
        t = A.RicianNoise(std_range=(std, std), p=1.0)
        t.set_random_seed(0)
        result = t(image=img)["image"]

        expected_mean = std * np.sqrt(np.pi / 2)
        assert abs(result.mean() - expected_mean) < 0.02, (
            f"Mean {result.mean():.4f} should be close to {expected_mean:.4f}"
        )

    def test_noise_approximately_gaussian_at_high_snr(self):
        """At high SNR, Rician noise ≈ Gaussian noise (additive)."""
        signal = 0.8
        std = 0.02
        img = np.full((256, 256), signal, dtype=np.float32)
        t = A.RicianNoise(std_range=(std, std), p=1.0)
        t.set_random_seed(0)
        result = t(image=img)["image"]

        noise = result - signal
        # At high SNR, the mean of the noise should be close to 0
        # and std should be close to the input std
        assert abs(noise.mean()) < 0.01
        assert abs(noise.std() - std) < 0.01


# ---------------------------------------------------------------------------
# Compose integration
# ---------------------------------------------------------------------------
class TestComposeIntegration:
    def test_compose_image(self, uint8_image):
        pipe = A.Compose([A.RicianNoise(std_range=(0.05, 0.1), p=1.0)])
        result = pipe(image=uint8_image)
        assert "image" in result
        assert result["image"].shape == uint8_image.shape

    def test_compose_volume(self, volume):
        pipe = A.Compose([A.RicianNoise(std_range=(0.05, 0.1), p=1.0)])
        result = pipe(volume=volume)
        assert "volume" in result
        assert result["volume"].shape == volume.shape
