"""Functional and transform-level contracts for KSpaceSpikeNoise (k-space MRI spike artifacts).

The functional kernel injects real amplitudes at sampled Fourier bins and their conjugate
mirrors, guaranteeing a Hermitian half-spectrum and a real reconstruction via `irfftn`.
Closed-form reference values in the phantom tests were derived analytically and validated
against scipy's FFT semantics on 8x8 flat fields: a distinct conjugate pair produces a
cosine of amplitude `2a / N**2`, while a self-conjugate bin (DC or Nyquist) injects
`a / N**2` once.
"""

from __future__ import annotations

import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.pixel import functional as fpixel


def _flat(h: int, w: int, value: float, channels: int = 1) -> np.ndarray:
    return np.full((h, w, channels), np.float32(value), dtype=np.float32)


def _cos_rows(n: int, freq: int) -> np.ndarray:
    """Cosine of frequency `freq` along axis 0 (rows) of an n x n grid."""
    return np.cos(2 * np.pi * freq * np.arange(n) / n)[:, None]


def _cos_cols(n: int, freq: int) -> np.ndarray:
    """Cosine of frequency `freq` along axis 1 (columns) of an n x n grid."""
    return np.cos(2 * np.pi * freq * np.arange(n) / n)[None, :]


def test_k_space_spike_injects_known_frequency_peak() -> None:
    img = _flat(8, 8, 0.5)
    a = 0.25 * np.abs(np.fft.rfftn(img, axes=(0, 1))).max()  # 0.25 * 32 = 8.0

    out = fpixel.k_space_spike(img, np.array([[1, 0]]), 0.25)

    diff = np.fft.rfftn(out, axes=(0, 1)) - np.fft.rfftn(img, axes=(0, 1))
    # The injected energy lands exactly at the sampled bin and its conjugate mirror.
    np.testing.assert_allclose(diff[1, 0, 0], a, atol=1e-4)
    np.testing.assert_allclose(diff[7, 0, 0], a, atol=1e-4)
    residual = np.abs(diff).copy()
    residual[1, 0, 0] = 0
    residual[7, 0, 0] = 0
    assert residual.max() < 1e-3
    assert np.isfinite(out).all()


def test_k_space_spike_dc_self_conjugate_single_injection() -> None:
    img = _flat(8, 8, 0.5)
    # max|F| of a flat 0.5 field is 8 * 8 * 0.5 = 32; a = 0.25 * 32 = 8.
    out = fpixel.k_space_spike(img, np.array([[0, 0]]), 0.25)

    np.testing.assert_allclose(out, 0.5 + 8 / 64, atol=1e-6)


def test_k_space_spike_nyquist_self_conjugate_single_injection() -> None:
    img = _flat(8, 8, 0.5)
    rows = np.cos(np.pi * np.arange(8))
    expected = 0.5 + (8 / 64) * rows[:, None] * rows[None, :]

    out = fpixel.k_space_spike(img, np.array([[4, 4]]), 0.25)

    np.testing.assert_allclose(out[:, :, 0], expected, atol=1e-6)


def test_k_space_spike_boundary_fold_collision() -> None:
    # k = (0, 7) mirrors to (0, 1); both fold onto the single halved bin (0, 1),
    # and irfftn's halved-axis doubling yields the full pair amplitude 2a / N^2.
    img = _flat(8, 8, 0.5)
    expected = np.broadcast_to(0.5 + (2 * 8 / 64) * _cos_cols(8, 1), (8, 8))

    out = fpixel.k_space_spike(img, np.array([[0, 7]]), 0.25)

    np.testing.assert_allclose(out[:, :, 0], expected, atol=1e-6)


def test_k_space_spike_preserves_odd_shape_and_is_finite() -> None:
    img = _flat(9, 7, 0.5, channels=2)

    out = fpixel.k_space_spike(img, np.array([[4, 3], [1, 6]]), 0.25)

    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0
    assert out.max() <= 1


def test_k_space_spike_d1_volume_is_finite() -> None:
    # D=1: the depth axis has size 1, so the depth coordinate must be 0.
    volume = np.full((1, 9, 7, 1), np.float32(0.5), dtype=np.float32)

    out = fpixel.k_space_spike(volume, np.array([[0, 4, 3]]), 0.25)

    assert out.shape == (1, 9, 7, 1)
    assert np.isfinite(out).all()


def test_k_space_spike_zero_intensity_is_exact_identity_float32() -> None:
    img = _flat(17, 13, 0.5, channels=3)

    out = fpixel.k_space_spike(img, np.array([[1, 0]]), 0.0)

    np.testing.assert_array_equal(out, img)


def test_k_space_spike_per_channel_amplitude_scales_per_channel_max() -> None:
    channels = [0.5, 0.25, 0.75]
    img = np.concatenate([_flat(8, 8, c) for c in channels], axis=-1).astype(np.float32)
    spikes = np.array([[[1, 0]], [[0, 1]], [[4, 4]]])
    intensity = 0.25

    out = fpixel.k_space_spike(img, spikes, intensity)

    rows = np.cos(2 * np.pi * np.arange(8) / 8)[:, None]
    nyq = np.cos(np.pi * np.arange(8))
    # Per-channel spectrum maxima are 64c; per-channel amplitudes are 0.25 * 64c = 16c.
    expected_ch0 = np.broadcast_to(channels[0] + (2 * 16 * channels[0] / 64) * rows, (8, 8))
    expected_ch1 = np.broadcast_to(channels[1] + (2 * 16 * channels[1] / 64) * rows.T, (8, 8))
    expected_ch2 = channels[2] + (16 * channels[2] / 64) * nyq[:, None] * nyq[None, :]
    np.testing.assert_allclose(out[..., 0], expected_ch0, atol=1e-6)
    np.testing.assert_allclose(out[..., 1], expected_ch1, atol=1e-6)
    np.testing.assert_allclose(out[..., 2], expected_ch2, atol=1e-6)


def test_k_space_spike_reconstruction_is_real_and_hermitian() -> None:
    img = _flat(8, 8, 0.5, channels=3)
    out = fpixel.k_space_spike(img, np.array([[1, 0]]), 0.25)

    assert np.isrealobj(out)
    spectrum = np.fft.rfftn(out, axes=(0, 1))
    # The pair (1, 0) <-> (7, 0) carries equal magnitude (Hermitian symmetry).
    np.testing.assert_allclose(np.abs(spectrum[1, 0]), np.abs(spectrum[7, 0]), atol=1e-4)


def test_k_space_spike_shared_spikes_give_identical_channels() -> None:
    img = _flat(8, 8, 0.5, channels=3)

    out = fpixel.k_space_spike(img, np.array([[1, 0]]), 0.25)

    np.testing.assert_array_equal(out[..., 0], out[..., 1])
    np.testing.assert_array_equal(out[..., 1], out[..., 2])


def test_k_space_spike_3d_volume_plane_wave_along_depth() -> None:
    # A 3D spike at (1, 0, 0) produces stripes along the depth axis.
    depth, h, w = 4, 8, 8
    exact = np.full((depth, h, w, 1), np.float32(0.5), dtype=np.float32)
    max_amplitude = np.abs(np.fft.rfftn(exact, axes=(0, 1, 2))).max()  # 0.5 * 4*8*8 = 128
    a = 0.25 * max_amplitude

    out = fpixel.k_space_spike(exact, np.array([[1, 0, 0]]), 0.25)

    expected = 0.5 + (2 * a / (depth * h * w)) * np.cos(2 * np.pi * np.arange(depth) / depth)
    np.testing.assert_allclose(out[:, 0, 0, 0], expected, atol=1e-5)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 5])
def test_k_space_spike_noise_preserves_shape_dtype_and_range(dtype: np.dtype, channels: int) -> None:
    rng = np.random.default_rng(137)
    image = (
        rng.integers(0, 256, (23, 19, channels), dtype=np.uint8)
        if dtype == np.uint8
        else rng.random((23, 19, channels), dtype=np.float32)
    )
    transform = A.Compose(
        [
            A.KSpaceSpikeNoise(
                num_spikes_range=(2, 4),
                intensity_range=(0.2, 0.4),
                per_channel=channels != 1,
                p=1.0,
            ),
        ],
        seed=137,
    )

    result = transform(image=image)["image"]

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert np.isfinite(result).all()
    if dtype == np.uint8:
        assert result.min() >= 0
        assert result.max() <= 255
    else:
        assert result.min() >= 0
        assert result.max() <= 1


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_zero_intensity_is_an_exact_identity(dtype: np.dtype) -> None:
    rng = np.random.default_rng(137)
    image = (
        rng.integers(0, 256, (17, 13, 3), dtype=np.uint8)
        if dtype == np.uint8
        else rng.random((17, 13, 3), dtype=np.float32)
    )
    transform = A.Compose(
        [A.KSpaceSpikeNoise(intensity_range=(0.0, 0.0), per_channel=True, p=1.0)],
        seed=137,
    )

    result = transform(image=image)["image"]

    np.testing.assert_array_equal(result, image)


def test_zero_spikes_is_an_exact_identity() -> None:
    image = np.random.default_rng(137).random((17, 13, 3), dtype=np.float32)
    transform = A.Compose(
        [A.KSpaceSpikeNoise(num_spikes_range=(0, 0), p=1.0)],
        seed=137,
    )

    result = transform(image=image)["image"]

    np.testing.assert_array_equal(result, image)


def test_shared_and_per_channel_spikes_have_distinct_channel_semantics() -> None:
    image = np.zeros((31, 29, 3), dtype=np.float32)
    image[15, 14] = 1.0

    shared = A.Compose(
        [A.KSpaceSpikeNoise(num_spikes_range=(2, 2), intensity_range=(0.25, 0.25), p=1.0)],
        seed=137,
    )(image=image)["image"]
    per_channel = A.Compose(
        [
            A.KSpaceSpikeNoise(
                num_spikes_range=(2, 2),
                intensity_range=(0.25, 0.25),
                per_channel=True,
                p=1.0,
            ),
        ],
        seed=137,
    )(image=image)["image"]

    np.testing.assert_array_equal(shared[..., 0], shared[..., 1])
    assert not np.array_equal(per_channel[..., 0], per_channel[..., 1])


def test_batch_and_volume_reuse_one_spike_realization() -> None:
    image = np.random.default_rng(137).random((13, 17, 3), dtype=np.float32)
    images = np.stack([image, image], axis=0)
    volume = np.stack([image, image], axis=0)
    transform = A.Compose(
        [A.KSpaceSpikeNoise(num_spikes_range=(2, 2), intensity_range=(0.2, 0.2), p=1.0)],
        seed=137,
    )

    result = transform(images=images, volume=volume)

    # Batch items share one spike realization. SciPy may round their batched FFTs differently
    # at float32 precision; a 3D volume spike is a plane wave, so depth slices legitimately
    # differ from each other but stay finite and in range.
    np.testing.assert_allclose(result["images"][0], result["images"][1], rtol=1e-6, atol=2e-7)
    assert result["volume"].shape == volume.shape
    assert np.isfinite(result["volume"]).all()


def test_image_and_volume_targets_receive_rank_appropriate_spikes() -> None:
    image = np.random.default_rng(137).random((13, 17, 3), dtype=np.float32)
    volume = np.random.default_rng(137).random((3, 13, 17, 3), dtype=np.float32)
    transform = A.Compose(
        [A.KSpaceSpikeNoise(num_spikes_range=(3, 3), intensity_range=(0.2, 0.2), per_channel=True, p=1.0)],
        seed=137,
    )

    result = transform(image=image, volume=volume)

    assert result["image"].shape == (13, 17, 3)
    assert result["volume"].shape == (3, 13, 17, 3)
    assert np.isfinite(result["image"]).all()
    assert np.isfinite(result["volume"]).all()


def test_per_channel_handles_mismatched_target_channel_counts() -> None:
    image = np.zeros((11, 13, 3), dtype=np.float32)
    volume = np.zeros((2, 11, 13, 5), dtype=np.float32)
    transform = A.Compose(
        [
            A.KSpaceSpikeNoise(
                num_spikes_range=(1, 1),
                intensity_range=(0.1, 0.1),
                per_channel=True,
                p=1.0,
            ),
        ],
        seed=137,
    )

    result = transform(image=image, volume=volume)

    assert result["image"].shape == (11, 13, 3)
    assert result["volume"].shape == (2, 11, 13, 5)


def test_d1_volume_via_compose_is_finite() -> None:
    volume = np.random.default_rng(137).random((1, 16, 16, 1), dtype=np.float32)
    transform = A.Compose(
        [A.KSpaceSpikeNoise(num_spikes_range=(2, 2), intensity_range=(0.2, 0.2), p=1.0)],
        seed=137,
    )

    result = transform(volume=volume)["volume"]

    assert result.shape == (1, 16, 16, 1)
    assert np.isfinite(result).all()


def test_k_space_spike_noise_is_seed_reproducible() -> None:
    image = np.random.default_rng(137).random((19, 23, 3), dtype=np.float32)
    transform_kwargs = {
        "num_spikes_range": (2, 4),
        "intensity_range": (0.1, 0.3),
        "per_channel": True,
        "p": 1.0,
    }

    first = A.Compose([A.KSpaceSpikeNoise(**transform_kwargs)], seed=137)(image=image)["image"]
    second = A.Compose([A.KSpaceSpikeNoise(**transform_kwargs)], seed=137)(image=image)["image"]

    np.testing.assert_array_equal(first, second)


def test_k_space_spike_noise_replay_reuses_realized_spikes() -> None:
    image = np.random.default_rng(137).random((19, 23, 3), dtype=np.float32)
    pipeline = A.ReplayCompose(
        [
            A.KSpaceSpikeNoise(
                num_spikes_range=(3, 3),
                intensity_range=(0.2, 0.2),
                per_channel=True,
                p=1.0,
            ),
        ],
        seed=137,
    )

    original = pipeline(image=image)
    replayed = A.ReplayCompose.replay(original["replay"], image=image)

    np.testing.assert_array_equal(replayed["image"], original["image"])


def test_k_space_spike_noise_records_constructor_valid_realized_ranges() -> None:
    image = np.random.default_rng(137).random((19, 23, 3), dtype=np.float32)
    transform = A.Compose(
        [
            A.KSpaceSpikeNoise(
                num_spikes_range=(2, 6),
                intensity_range=(0.1, 0.3),
                per_channel=True,
                p=1.0,
            ),
        ],
        save_applied_params=True,
        seed=137,
    )

    result = transform(image=image)
    applied = result["applied_transforms"][0][1]

    assert applied["num_spikes_range"] in {2, 3, 4, 5, 6}
    assert 0.1 <= applied["intensity_range"] <= 0.3
    assert applied["per_channel"] is True


def test_k_space_spike_noise_serialization_roundtrip_is_runnable() -> None:
    image = np.random.default_rng(137).random((19, 23, 3), dtype=np.float32)
    transform = A.KSpaceSpikeNoise(num_spikes_range=(2, 4), intensity_range=(0.1, 0.3), p=1.0)

    restored = A.from_dict(A.to_dict(transform))

    result = A.Compose([restored], seed=137)(image=image)["image"]
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert np.isfinite(result).all()


def test_extreme_intensity_stays_finite_and_clipped() -> None:
    image = np.full((17, 19, 5), 0.5, dtype=np.float32)
    transform = A.Compose(
        [
            A.KSpaceSpikeNoise(
                num_spikes_range=(3, 3),
                intensity_range=(100.0, 100.0),
                per_channel=True,
                p=1.0,
            ),
        ],
        seed=137,
    )

    result = transform(image=image)["image"]

    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert result.min() >= 0
    assert result.max() <= 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_spikes_range": (3, 1)}, "less than the second"),
        ({"num_spikes_range": (-1, 2)}, "must be >= 0"),
        ({"intensity_range": (0.5, 0.1)}, "less than the second"),
        ({"intensity_range": (-0.5, 0.5)}, "must be >= 0"),
    ],
)
def test_k_space_spike_noise_rejects_invalid_ranges(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        A.KSpaceSpikeNoise(**kwargs)
