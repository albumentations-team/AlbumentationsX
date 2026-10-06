"""Independent Fourier and Dirichlet-kernel references for rectangular k-space truncation."""

from itertools import pairwise

import numpy as np
import pytest
import torch

import albumentations as A


def _full_fft_reference(data: np.ndarray, retained_fraction: float) -> np.ndarray:
    signal = data.astype(np.float64)
    if data.dtype == np.uint8:
        signal /= 255
    axes = tuple(range(data.ndim - 1))
    retained = np.ones(data.shape[:-1], dtype=bool)
    for axis in axes:
        length = data.shape[axis]
        frequencies = np.rint(np.fft.fftfreq(length) * length).astype(np.int64)
        shape = [1] * len(axes)
        shape[axis] = length
        retained &= (np.abs(frequencies) <= int(retained_fraction * length / 2)).reshape(shape)
    spectrum = np.fft.fftn(signal, axes=axes) * retained[..., None]
    reconstructed = np.fft.ifftn(spectrum, axes=axes)
    assert np.abs(reconstructed.imag).max() < 1e-12
    return np.clip(reconstructed.real, 0, 1)


@pytest.mark.parametrize("shape", [(8, 12, 1), (7, 11, 5), (8, 12, 16, 3), (7, 9, 11, 5), (1, 9, 13, 1)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("fraction", [0.0, 0.35, 0.8])
def test_matches_full_spatial_fft_without_mutating_strided_input(
    shape: tuple[int, ...], dtype: type[np.generic], fraction: float
) -> None:
    raw_shape = (shape[0], shape[1] * 2, *shape[2:])
    raw = np.random.default_rng(137).integers(0, 256, raw_shape, dtype=np.uint8)
    data = raw if dtype == np.uint8 else raw.astype(np.float32) / 255
    data = data[:, ::2]
    original = data.copy()
    data.setflags(write=False)
    target = "image" if len(shape) == 3 else "volume"
    result = A.Compose([A.GibbsRinging(retained_fraction_range=(fraction, fraction), p=1)], seed=137, strict=True)(
        **{target: data}
    )[target]
    expected = _full_fft_reference(data, fraction)
    normalized = result.astype(np.float32) / 255 if dtype == np.uint8 else result
    np.testing.assert_allclose(normalized, expected, atol=0.5 / 255 + 1e-6 if dtype == np.uint8 else 2e-6)
    assert result.dtype == dtype
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize(
    "length,frequency,fraction,kept", [(8, 2, 0.5, True), (8, 2, 0.49, False), (8, 4, 0.99, False), (9, 4, 8 / 9, True)]
)
def test_signed_frequency_cutoff_and_nyquist_parity(
    axis: int, length: int, frequency: int, fraction: float, kept: bool
) -> None:
    shape = [5, 7, 11, 1]
    shape[axis] = length
    coordinate_shape = [1] * 4
    coordinate_shape[axis] = length
    signal = 0.4 + 0.1 * np.cos(2 * np.pi * frequency * np.arange(length) / length)
    volume = np.broadcast_to(signal.reshape(coordinate_shape), shape).astype(np.float32)
    result = A.Compose([A.GibbsRinging(retained_fraction_range=(fraction, fraction), p=1)], strict=True)(volume=volume)[
        "volume"
    ]
    np.testing.assert_allclose(result, volume if kept else 0.4, atol=1e-7)


def test_depth_step_rings_like_the_dirichlet_kernel() -> None:
    length, cutoff = 64, 8
    signal = np.full(length, 0.25, dtype=np.float64)
    signal[16:48] = 0.75
    offsets = np.arange(1, length)
    weights = np.empty(length)
    weights[0] = (2 * cutoff + 1) / length
    weights[1:] = np.sin((2 * cutoff + 1) * np.pi * offsets / length) / (length * np.sin(np.pi * offsets / length))
    expected = sum(weight * np.roll(signal, index) for index, weight in enumerate(weights))
    volume = np.broadcast_to(signal[:, None, None, None], (length, 5, 7, 1)).astype(np.float32)
    result = A.Compose([A.GibbsRinging(retained_fraction_range=(0.25, 0.25), p=1)], strict=True)(volume=volume)[
        "volume"
    ]
    np.testing.assert_allclose(result[:, 2, 3, 0], expected, atol=1e-7)
    assert result.min() < 0.25
    assert result.max() > 0.75


def test_lower_retention_removes_more_signal_energy() -> None:
    positions = np.arange(32)
    signal = 0.5 + sum(0.1 * np.cos(2 * np.pi * frequency * positions / 32) for frequency in (1, 3, 7))
    volume = np.broadcast_to(signal[:, None, None, None], (32, 5, 7, 1)).astype(np.float32)
    errors = []
    for fraction in (0.0, 0.125, 0.25, 0.5):
        result = A.Compose([A.GibbsRinging(retained_fraction_range=(fraction, fraction), p=1)], strict=True)(
            volume=volume
        )["volume"]
        errors.append(np.square(result - volume).mean())
    assert all(left > right for left, right in pairwise(errors))
    np.testing.assert_allclose(errors, [0.015, 0.01, 0.005, 0], atol=1e-7)


@pytest.mark.parametrize("shape", [(7, 9, 5), (7, 9, 11, 5), (1, 1, 1, 3)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_full_retention_is_bitwise_identity(shape: tuple[int, ...], dtype: type[np.generic]) -> None:
    raw = np.random.default_rng(137).integers(0, 256, shape, dtype=np.uint8)
    data = raw if dtype == np.uint8 else raw.astype(np.float32) / 255
    target = "image" if len(shape) == 3 else "volume"
    result = A.Compose([A.GibbsRinging(retained_fraction_range=(1, 1), p=1)], strict=True)(**{target: data})[target]
    np.testing.assert_array_equal(result, data)


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_constant_channels_and_singleton_depth_match_image(fraction: float) -> None:
    image = np.broadcast_to(np.array([0.0, 0.25, 0.75], dtype=np.float32), (9, 13, 3)).copy()
    pipeline = A.Compose([A.GibbsRinging(retained_fraction_range=(fraction, fraction), p=1)], strict=True)
    result = pipeline(image=image, volume=image[None])
    np.testing.assert_allclose(result["image"], image, atol=1e-7)
    np.testing.assert_allclose(result["volume"][0], result["image"], atol=1e-7)


@pytest.mark.parametrize("target,shape", [("image", (9, 13)), ("volume", (7, 9, 13))])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_compose_normalizes_channel_free_inputs_before_spatial_fft(
    target: str, shape: tuple[int, ...], dtype: type[np.generic]
) -> None:
    raw = np.random.default_rng(137).integers(0, 256, shape, dtype=np.uint8)
    data = raw if dtype == np.uint8 else raw.astype(np.float32) / 255
    result = A.Compose([A.GibbsRinging(retained_fraction_range=(0.5, 0.5), p=1)], strict=True)(**{target: data})[target]
    expected = _full_fft_reference(data[..., None], 0.5)[..., 0]
    normalized = result.astype(np.float32) / 255 if dtype == np.uint8 else result
    np.testing.assert_allclose(normalized, expected, atol=0.5 / 255 + 1e-6 if dtype == np.uint8 else 2e-6)
    assert result.shape == data.shape


def test_direct_image_and_seeded_replay_preserve_annotations() -> None:
    image = np.random.default_rng(137).random((9, 13, 5), dtype=np.float32)
    direct = A.GibbsRinging(retained_fraction_range=(0.5, 0.5), p=1)(image=image)["image"]
    np.testing.assert_allclose(direct, _full_fft_reference(image, 0.5), atol=2e-6)
    inputs = {"image": image, "other": image.copy(), "mask": np.zeros((9, 13), dtype=np.uint8), "keypoints": [[3, 2]]}
    pipeline = A.ReplayCompose(
        [A.GibbsRinging(p=1)],
        additional_targets={"other": "image"},
        keypoint_params=A.KeypointParams(coord_format="xy"),
        strict=True,
    )
    pipeline.set_random_seed(137)
    result = pipeline(**inputs)
    replay = A.ReplayCompose.replay(result["replay"], **inputs)
    np.testing.assert_array_equal(result["image"], result["other"])
    np.testing.assert_array_equal(result["mask"], inputs["mask"])
    np.testing.assert_array_equal(result["keypoints"], inputs["keypoints"])
    for key in inputs:
        np.testing.assert_array_equal(result[key], replay[key])
    seeded = A.Compose([A.GibbsRinging(p=1)], seed=137, strict=True)(image=image)["image"]
    np.testing.assert_array_equal(seeded, result["image"])


def test_mixed_image_volume_aliases_share_cutoff_and_preserve_volume_labels() -> None:
    rng = np.random.default_rng(137)
    inputs = {
        "image": rng.integers(0, 256, (9, 13, 1), dtype=np.uint8),
        "volume": rng.random((7, 9, 13, 3), dtype=np.float32),
        "other": rng.integers(0, 256, (7, 9, 13, 5), dtype=np.uint8),
        "mask3d": rng.integers(0, 4, (7, 9, 13), dtype=np.uint8),
        "keypoints": [[3, 2, 1]],
    }
    pipeline = A.ReplayCompose(
        [A.GibbsRinging(p=1)],
        additional_targets={"other": "volume"},
        keypoint_params=A.KeypointParams(coord_format="xyz"),
        strict=True,
    )
    pipeline.set_random_seed(137)
    result = pipeline(**inputs)
    fraction = result["replay"]["transforms"][0]["params"]["params"]["retained_fraction"]
    for target in ("image", "volume", "other"):
        normalized = result[target].astype(np.float32) / 255 if inputs[target].dtype == np.uint8 else result[target]
        tolerance = 0.5 / 255 + 1e-6 if inputs[target].dtype == np.uint8 else 2e-6
        np.testing.assert_allclose(normalized, _full_fft_reference(inputs[target], fraction), atol=tolerance)
    np.testing.assert_array_equal(result["mask3d"], inputs["mask3d"])
    np.testing.assert_array_equal(result["keypoints"], inputs["keypoints"])


@pytest.mark.parametrize("target,shape", [("images", (9, 13, 5)), ("volumes", (7, 9, 13, 5))])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_collection_and_tensor_routes_keep_items_and_channels_independent(
    target: str, shape: tuple[int, ...], dtype: type[np.generic]
) -> None:
    raw = np.random.default_rng(137).integers(0, 102, (*shape[:-1], 1), dtype=np.uint8)
    data = raw if dtype == np.uint8 else raw.astype(np.float32) / 255
    item = np.repeat(data, shape[-1], axis=-1)
    batch = np.stack([item, np.zeros_like(item)])
    tensor = torch.from_numpy(batch).movedim(-1, 1)
    pipeline = A.Compose(
        [A.GibbsRinging(retained_fraction_range=(0.5, 0.5), p=1)],
        additional_targets={"other": target},
        seed=137,
        strict=True,
    )
    result = pipeline(**{target: batch, "other": tensor})
    np.testing.assert_array_equal(result["other"].movedim(1, -1).numpy(), result[target])
    normalized = result[target][0].astype(np.float32) / 255 if dtype == np.uint8 else result[target][0]
    np.testing.assert_allclose(
        normalized, _full_fft_reference(item, 0.5), atol=0.5 / 255 + 1e-6 if dtype == np.uint8 else 2e-6
    )
    np.testing.assert_array_equal(result[target][1], 0)
    np.testing.assert_allclose(result[target][0, ..., 0], result[target][0, ..., -1], atol=1e-7)
    np.testing.assert_array_equal(batch[0], item)


@pytest.mark.parametrize("bounds", [(-0.1, 0.5), (0.5, 1.1), (0.8, 0.2), (0, float("nan")), (0, float("inf"))])
def test_invalid_retained_fraction_is_rejected(bounds: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        A.GibbsRinging(retained_fraction_range=bounds, strict=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input_is_rejected(value: float) -> None:
    image = np.full((9, 13, 1), value, dtype=np.float32)
    with pytest.raises(ValueError, match="finite inputs"):
        A.Compose([A.GibbsRinging(p=1)], strict=True)(image=image)


@pytest.mark.parametrize("dtype", [np.complex64, np.float64, np.int16])
def test_unsupported_dtype_is_rejected(dtype: type[np.generic]) -> None:
    with pytest.raises(ValueError, match="real inputs"):
        A.Compose([A.GibbsRinging(p=1)], strict=True)(image=np.ones((9, 13, 1), dtype=dtype))
