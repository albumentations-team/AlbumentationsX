"""Full-FFT and closed-form references for periodic MRI ghosting."""

import numpy as np
import pytest
import torch

import albumentations as A
from albumentations.augmentations.transforms3d import functional as f3d


def _full_fft_reference(volume: np.ndarray, ghosts: int, intensity: float, axis: int, restore: float) -> np.ndarray:
    signal = volume.astype(np.float64)
    if volume.dtype == np.uint8:
        signal /= 255
    length = signal.shape[axis]
    frequencies = np.rint(np.fft.fftfreq(length) * length).astype(np.int64)
    affected = (frequencies % ghosts == 0) & (np.abs(frequencies) > int(restore * length / 2))
    shape = [1] * signal.ndim
    shape[axis] = length
    spectrum = np.fft.fftn(signal, axes=(0, 1, 2)) * (1 - intensity * affected).reshape(shape)
    reconstructed = np.fft.ifftn(spectrum, axes=(0, 1, 2))
    assert np.abs(reconstructed.imag).max() < 1e-12
    return np.clip(np.abs(reconstructed), 0, 1)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("shape", [(8, 12, 16, 1), (7, 9, 11, 5)])
@pytest.mark.parametrize("ghosts", [2, 3])
@pytest.mark.parametrize("restore", [0.0, 0.3])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_public_ghosting_matches_full_3d_fft_reference(
    axis: int,
    shape: tuple[int, ...],
    ghosts: int,
    restore: float,
    dtype: type[np.generic],
) -> None:
    raw = np.random.default_rng(137).integers(0, 256, (shape[0], shape[1] * 2, shape[2], shape[3]), dtype=np.uint8)
    data = raw if dtype == np.uint8 else raw.astype(np.float32) / 255
    data = data[:, ::2]
    original = data.copy()
    data.setflags(write=False)
    pipeline = A.Compose(
        [
            A.GhostingArtifact(
                num_ghosts_range=(ghosts, ghosts),
                intensity_range=(0.6, 0.6),
                restore_range=(restore, restore),
                axis=axis,
                p=1,
            ),
        ],
        seed=137,
        strict=True,
    )
    result = pipeline(volume=data)["volume"]
    expected = _full_fft_reference(data, ghosts, 0.6, axis, restore)
    actual = result.astype(np.float32) / 255 if dtype == np.uint8 else result
    np.testing.assert_allclose(actual, expected, atol=0.5 / 255 + 1e-6 if dtype == np.uint8 else 2e-6)
    assert result.dtype == dtype
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_impulse_ghost_spacing_matches_circular_shift_formula(axis: int) -> None:
    volume = np.zeros((12, 16, 20, 1), dtype=np.float32)
    volume[1, 2, 3, 0] = 0.5
    ghosts, intensity = 4, 0.6
    spacing = volume.shape[axis] // ghosts
    comb = sum(np.roll(volume, index * spacing, axis=axis) for index in range(ghosts)) / ghosts
    expected = np.abs(volume - intensity * comb + intensity * volume.mean(axis=axis, keepdims=True))
    result = f3d.ghosting_artifact(volume, ghosts, intensity, axis, 0)
    np.testing.assert_allclose(result, expected, atol=1e-7)
    position = [1, 2, 3, 0]
    position[axis] += spacing
    assert result[tuple(position)] > 0.5 * intensity / volume.shape[axis]


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("policy", ["zero-strength", "restore-all", "singleton-axis", "no-affected-bin"])
def test_identity_limits_are_exact(dtype: type[np.generic], policy: str) -> None:
    shape = (1, 9, 13, 3) if policy == "singleton-axis" else (5, 9, 13, 3)
    raw = np.random.default_rng(137).integers(0, 256, shape, dtype=np.uint8)
    volume = raw if dtype == np.uint8 else raw.astype(np.float32) / 255
    kwargs = {"intensity_range": (0, 0)} if policy == "zero-strength" else {}
    if policy == "restore-all":
        kwargs["restore_range"] = (1, 1)
    if policy == "singleton-axis":
        kwargs["axis"] = 0
    if policy == "no-affected-bin":
        kwargs["num_ghosts_range"] = (100, 100)
    result = A.Compose([A.GhostingArtifact(**kwargs, p=1)], seed=137, strict=True)(volume=volume)["volume"]
    np.testing.assert_array_equal(result, volume)


def test_constant_volume_and_zero_signal_are_preserved() -> None:
    volume = np.full((7, 9, 13, 2), 0.25, dtype=np.float32)
    volume[..., 1] = 0
    result = A.Compose([A.GhostingArtifact(intensity_range=(1, 1), p=1)], seed=137, strict=True)(volume=volume)[
        "volume"
    ]
    np.testing.assert_allclose(result[..., 0], 0.25, atol=1e-7)
    np.testing.assert_array_equal(result[..., 1], 0)


def test_replay_and_annotations_share_one_realized_acquisition() -> None:
    volume = np.random.default_rng(137).random((7, 9, 13, 3), dtype=np.float32)
    mask = np.random.default_rng(137).integers(0, 4, volume.shape[:3], dtype=np.uint8)
    keypoints = np.array([[3.0, 2.0, 1.0]], dtype=np.float32)
    inputs = {"volume": volume, "other": volume.copy(), "mask3d": mask, "keypoints": keypoints}
    pipeline = A.ReplayCompose(
        [A.GhostingArtifact(p=1)],
        additional_targets={"other": "volume"},
        keypoint_params=A.KeypointParams(coord_format="xyz"),
        strict=True,
    )
    pipeline.set_random_seed(137)
    result = pipeline(**inputs)
    replay = A.ReplayCompose.replay(result["replay"], **inputs)
    np.testing.assert_array_equal(result["volume"], result["other"])
    np.testing.assert_array_equal(result["mask3d"], mask)
    np.testing.assert_array_equal(result["keypoints"], keypoints)
    for key in inputs:
        np.testing.assert_array_equal(result[key], replay[key])


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_collection_channels_and_tensor_bridge_match_scalar_volumes(dtype: type[np.generic]) -> None:
    raw = np.random.default_rng(137).integers(0, 102, (5, 9, 13, 1), dtype=np.uint8)
    signal = raw if dtype == np.uint8 else raw.astype(np.float32) / 255
    volume = np.repeat(signal, 5, axis=-1)
    volumes = np.stack([volume, np.zeros_like(volume)])
    transforms = [A.Compose([A.GhostingArtifact(p=1)], seed=137, strict=True) for _ in range(3)]
    tensor = torch.from_numpy(volumes).permute(0, 4, 1, 2, 3)
    expected = transforms[0](volume=volume)["volume"]
    result = transforms[1](volumes=volumes)["volumes"]
    tensor_result = transforms[2](volumes=tensor)["volumes"].permute(0, 2, 3, 4, 1).numpy()
    np.testing.assert_array_equal(result[0], expected)
    np.testing.assert_array_equal(result[1], 0)
    np.testing.assert_allclose(result[0, ..., 0], result[0, ..., 4], atol=1e-7)
    np.testing.assert_array_equal(tensor_result, result)
    np.testing.assert_array_equal(volumes[0], volume)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_ghosts_range": (1, 3)},
        {"num_ghosts_range": (4, 2)},
        {"intensity_range": (-0.1, 0.5)},
        {"intensity_range": (0.5, 1.1)},
        {"restore_range": (0, float("nan"))},
        {"restore_range": (0, float("inf"))},
        {"axis": 3},
    ],
)
def test_invalid_policy_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        A.GhostingArtifact(**kwargs, strict=True)


@pytest.mark.parametrize("dtype", [np.complex64, np.float64, np.int16])
def test_unsupported_dtype_is_rejected(dtype: type[np.generic]) -> None:
    with pytest.raises(ValueError, match="real magnitude volumes"):
        A.Compose([A.GhostingArtifact(p=1)], strict=True)(volume=np.ones((5, 9, 13, 1), dtype=dtype))


@pytest.mark.parametrize("restore", [0.0, 0.5])
def test_central_band_preserves_selected_low_frequency(restore: float) -> None:
    cosine = np.cos(2 * np.pi * 4 * np.arange(16) / 16)
    volume = np.broadcast_to((0.25 + 0.1 * cosine).astype(np.float32), (3, 5, 16))[..., None].copy()
    amplitude = 0.1 if restore == 0.5 else 0.1 * 0.4
    expected = np.broadcast_to(0.25 + amplitude * cosine, volume.shape[:3])[..., None]
    result = f3d.ghosting_artifact(volume, 4, 0.6, 2, restore)
    np.testing.assert_allclose(result, expected, atol=1e-7)
