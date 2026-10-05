"""Independent reconstruction and public target contracts for MRI acquisition motion."""

import cv2
import numpy as np
import pytest
import torch
from scipy import ndimage

import albumentations as A
from albumentations.augmentations.transforms3d import functional as f3d


def _reference_motion(
    volume: np.ndarray,
    matrices: np.ndarray,
    boundaries: tuple[int, ...],
    axis: int,
    interpolation: int,
) -> np.ndarray:
    signal = volume.astype(np.float64)
    if volume.dtype == np.uint8:
        signal /= 255
    states = [signal]
    for matrix in matrices:
        inverse = np.linalg.inv(matrix.astype(np.float64))
        moved = np.stack(
            [
                ndimage.affine_transform(
                    signal[..., channel],
                    inverse[:3, :3][::-1, ::-1],
                    offset=inverse[:3, 3][::-1],
                    order=interpolation,
                    mode="grid-constant",
                    cval=0,
                    prefilter=False,
                )
                for channel in range(signal.shape[-1])
            ],
            axis=-1,
        )
        states.append(moved)
    spectra = [np.fft.fftshift(np.fft.fftn(state, axes=(0, 1, 2)), axes=(0, 1, 2)) for state in states]
    combined = np.empty_like(spectra[0])
    for spectrum, begin, end in zip(spectra, (0, *boundaries), (*boundaries, volume.shape[axis]), strict=True):
        index = [slice(None)] * volume.ndim
        index[axis] = slice(begin, end)
        combined[tuple(index)] = spectrum[tuple(index)]
    return np.clip(np.abs(np.fft.ifftn(np.fft.ifftshift(combined, axes=(0, 1, 2)), axes=(0, 1, 2))), 0, 1)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("shape", [(6, 8, 10, 1), (7, 9, 11, 3)])
@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR])
@pytest.mark.parametrize("num_events", [1, 2])
def test_motion_matches_independent_resampling_and_fft(
    axis: int,
    shape: tuple[int, ...],
    interpolation: int,
    num_events: int,
) -> None:
    volume = np.random.default_rng(137).random(shape, dtype=np.float32) * 0.4
    first = np.eye(4, dtype=np.float32)
    first[:3, 3] = (0.75, -0.25, 0.375)
    angle = np.deg2rad(13)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    second = np.eye(4, dtype=np.float32)
    second[:3, :3] = rotation
    centre = (np.array(shape[:3][::-1]) - 1) / 2
    second[:3, 3] = centre - rotation @ centre
    matrices = np.stack([first, second])[:num_events]
    boundaries = (1, shape[axis] // 2 + 1)[:num_events]

    result = f3d.motion_artifact(volume, matrices, boundaries, axis, interpolation)

    expected = _reference_motion(volume, matrices, boundaries, axis, interpolation)
    np.testing.assert_allclose(result, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("num_events", [0, 2])
def test_zero_motion_is_exact_identity(dtype: type[np.generic], num_events: int) -> None:
    source = np.random.default_rng(137).integers(0, 256, (5, 7, 9, 3), dtype=np.uint8)
    volume = source if dtype == np.uint8 else source.astype(np.float32) / 255
    original = volume.copy()
    volume.setflags(write=False)
    pipeline = A.Compose(
        [
            A.MotionArtifact(
                num_events_range=(num_events, num_events), rotate_range=(0, 0), translate_percent_range=(0, 0), p=1
            )
        ],
        strict=True,
    )

    result = pipeline(volume=volume)["volume"]

    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(volume, original)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [1, 5])
def test_public_strided_volume_matches_reference_and_preserves_input(dtype: type[np.generic], channels: int) -> None:
    source = np.random.default_rng(137).integers(0, 256, (5, 14, 9, channels), dtype=np.uint8)
    volume = source if dtype == np.uint8 else source.astype(np.float32) / 255
    volume = volume[:, ::2, :, :]
    original = volume.copy()
    volume.setflags(write=False)
    pipeline = A.ReplayCompose([A.MotionArtifact(num_events_range=(1, 3), p=1)], strict=True)
    pipeline.set_random_seed(137)

    result = pipeline(volume=volume)

    params = result["replay"]["transforms"][0]["params"]["params"]
    expected = _reference_motion(volume, params["matrices"], params["boundaries"], 2, cv2.INTER_LINEAR)
    actual = result["volume"].astype(np.float32)
    if dtype == np.uint8:
        actual /= 255
    np.testing.assert_allclose(actual, expected, atol=1 / 255 if dtype == np.uint8 else 3e-6)
    np.testing.assert_array_equal(volume, original)


def test_channels_and_volume_collection_share_motion_without_cross_talk() -> None:
    signal = np.random.default_rng(137).random((5, 7, 9, 1), dtype=np.float32) * 0.2
    volume = np.concatenate([signal, signal * 0.5, np.zeros_like(signal)], axis=-1)
    volumes = np.stack([volume, volume * 0.5])
    original = volumes.copy()
    pipeline = A.Compose([A.MotionArtifact(p=1)], seed=137, strict=True)

    result = pipeline(volumes=volumes)["volumes"]

    assert not np.allclose(result[0, ..., 0:1], signal)
    np.testing.assert_allclose(result[0, ..., 1], result[0, ..., 0] * 0.5, atol=1e-6)
    np.testing.assert_array_equal(result[..., 2], 0)
    np.testing.assert_allclose(result[1], result[0] * 0.5, atol=1e-6)
    np.testing.assert_array_equal(volumes, original)


def test_replay_and_seed_preserve_all_motion_states_and_annotations() -> None:
    volume = np.random.default_rng(137).random((5, 7, 9, 2), dtype=np.float32)
    mask = np.random.default_rng(138).integers(0, 4, (5, 7, 9), dtype=np.uint8)
    keypoints = np.array([[4.0, 3.0, 2.0]], dtype=np.float32)
    inputs = {"volume": volume, "other_volume": volume.copy(), "mask3d": mask, "keypoints": keypoints}
    pipelines = [
        A.ReplayCompose(
            [A.MotionArtifact(num_events_range=(2, 3), p=1)],
            additional_targets={"other_volume": "volume"},
            keypoint_params=A.KeypointParams(coord_format="xyz"),
            strict=True,
        )
        for _ in range(2)
    ]
    for pipeline in pipelines:
        pipeline.set_random_seed(137)
    first, second = (pipeline(**inputs) for pipeline in pipelines)
    replayed = A.ReplayCompose.replay(first["replay"], **inputs)
    params = first["replay"]["transforms"][0]["params"]["params"]

    assert len(params["boundaries"]) in (2, 3)
    assert list(params["boundaries"]) == sorted(set(params["boundaries"]))
    assert 0 < params["boundaries"][0] < params["boundaries"][-1] < volume.shape[2]
    rotations = params["matrices"][:, :3, :3]
    np.testing.assert_allclose(
        rotations @ rotations.swapaxes(1, 2), np.broadcast_to(np.eye(3), rotations.shape), atol=1e-6
    )
    np.testing.assert_allclose(np.linalg.det(rotations), 1, atol=1e-6)
    for key in inputs:
        np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first[key], replayed[key])
    np.testing.assert_array_equal(first["volume"], first["other_volume"])
    np.testing.assert_array_equal(first["mask3d"], mask)
    np.testing.assert_array_equal(first["keypoints"], keypoints)


def test_channel_free_volume_matches_explicit_channel() -> None:
    volume = np.random.default_rng(137).random((5, 7, 9), dtype=np.float32)
    channel_free = A.Compose([A.MotionArtifact(p=1)], seed=137, strict=True)
    explicit_channel = A.Compose([A.MotionArtifact(p=1)], seed=137, strict=True)

    result = channel_free(volume=volume)["volume"]
    expected = explicit_channel(volume=volume[..., None])["volume"][..., 0]

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("target", ["volume", "volumes"])
def test_tensor_volume_uses_spatial_axes_only(target: str) -> None:
    volume = np.random.default_rng(137).random((5, 7, 9, 3), dtype=np.float32)
    data = volume if target == "volume" else np.stack([volume, volume * 0.5])
    tensor_axes = (3, 0, 1, 2) if target == "volume" else (0, 4, 1, 2, 3)
    numpy_axes = (1, 2, 3, 0) if target == "volume" else (0, 2, 3, 4, 1)
    tensor = torch.from_numpy(data).permute(*tensor_axes)
    original = tensor.clone()
    numpy_pipeline = A.Compose([A.MotionArtifact(p=1)], seed=137, strict=True)
    tensor_pipeline = A.Compose([A.MotionArtifact(p=1)], seed=137, strict=True)

    expected = numpy_pipeline(**{target: data})[target]
    result = tensor_pipeline(**{target: tensor})[target]

    np.testing.assert_array_equal(result.permute(*numpy_axes).numpy(), expected)
    torch.testing.assert_close(tensor, original, rtol=0, atol=0)


def test_stationary_event_after_motion_retains_its_original_spectrum_segment() -> None:
    volume = np.random.default_rng(137).random((5, 7, 9, 1), dtype=np.float32)
    moved = np.eye(4, dtype=np.float32)
    moved[0, 3] = 1
    matrices = np.stack([moved, np.eye(4, dtype=np.float32)])

    result = f3d.motion_artifact(volume, matrices, (2, 5), 2, cv2.INTER_LINEAR)

    expected = _reference_motion(volume, matrices, (2, 5), 2, cv2.INTER_LINEAR)
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_extreme_motion_clips_finite_output(dtype: type[np.generic]) -> None:
    volume = np.ones((5, 7, 9, 2), dtype=dtype)
    if dtype == np.uint8:
        volume *= 255
    pipeline = A.Compose(
        [A.MotionArtifact(rotate_range=(180, 180), translate_percent_range=(1, 1), p=1)],
        seed=137,
        strict=True,
    )

    result = pipeline(volume=volume)["volume"]

    assert np.isfinite(result).all()
    assert result.dtype == dtype
    assert 0 <= result.min() <= result.max() <= (255 if dtype == np.uint8 else 1)


@pytest.mark.parametrize("shape", [(1, 7, 9, 1), (5, 1, 9, 1), (5, 7, 1, 1)])
def test_singleton_spatial_axis_is_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="at least two voxels"):
        A.Compose([A.MotionArtifact(p=1)], strict=True)(volume=np.zeros(shape, dtype=np.float32))


def test_event_count_must_fit_acquisition_axis() -> None:
    pipeline = A.Compose([A.MotionArtifact(num_events_range=(3, 5), axis=0, p=1)], strict=True)
    with pytest.raises(ValueError, match="acquisition-axis length"):
        pipeline(volume=np.zeros((5, 7, 9, 1), dtype=np.float32))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.float64, np.int16])
def test_unsupported_magnitude_dtype_is_rejected(dtype: type[np.generic]) -> None:
    pipeline = A.Compose([A.MotionArtifact(p=1)], strict=True)
    with pytest.raises(ValueError, match="real magnitude volumes"):
        pipeline(volume=np.ones((5, 7, 9, 1), dtype=dtype))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_events_range": (-1, 2)},
        {"num_events_range": (3, 1)},
        {"rotate_range": (0, float("inf"))},
        {"translate_percent_range": (float("nan"), 0)},
        {"axis": 3},
        {"interpolation": cv2.INTER_CUBIC},
    ],
)
def test_invalid_motion_policy_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        A.MotionArtifact(**kwargs, strict=True)
