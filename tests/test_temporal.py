import json

import numpy as np
import pytest
import torch

import albumentations as A
from albumentations.augmentations.other import temporal_functional as ftemporal


def make_indexed_video(num_frames: int, *, dtype: np.dtype = np.dtype(np.uint8)) -> np.ndarray:
    frame_values = np.arange(num_frames, dtype=dtype)
    return np.broadcast_to(frame_values[:, None, None, None], (num_frames, 5, 7, 3)).copy()


@pytest.mark.parametrize(
    ("input_frames", "output_frames", "expected_indices"),
    [
        (6, 4, [0, 1, 3, 5]),
        (3, 5, [0, 0, 1, 1, 2]),
        (1, 4, [0, 0, 0, 0]),
    ],
)
def test_uniform_temporal_subsample_numpy_values(
    input_frames: int,
    output_frames: int,
    expected_indices: list[int],
) -> None:
    video = make_indexed_video(input_frames)
    source = video.copy()
    transform = A.Compose([A.UniformTemporalSubsample(num_frames=output_frames)], strict=True)

    result = transform(images=video)["images"]

    np.testing.assert_array_equal(result, source[expected_indices])
    np.testing.assert_array_equal(video, source)
    assert result.shape == (output_frames, 5, 7, 3)
    assert result.dtype == video.dtype


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_uniform_temporal_subsample_tensor_noncontiguous(dtype: torch.dtype) -> None:
    video = torch.arange(6 * 3 * 7 * 5).to(dtype).reshape(6, 3, 7, 5).transpose(-1, -2)
    assert video.shape == (6, 3, 5, 7)
    assert not video.is_contiguous()
    source = video.clone()
    transform = A.Compose([A.UniformTemporalSubsample(num_frames=4)], strict=True)

    result = transform(images=video)["images"]

    torch.testing.assert_close(result, source.index_select(0, torch.tensor([0, 1, 3, 5])))
    torch.testing.assert_close(video, source)
    assert result.dtype == dtype


def test_uniform_temporal_subsample_aligns_images_aliases() -> None:
    video = make_indexed_video(5)
    optical_flow = video + 20
    transform = A.Compose(
        [A.UniformTemporalSubsample(num_frames=3)],
        additional_targets={"optical_flow": "images"},
        strict=True,
    )

    result = transform(images=video, optical_flow=optical_flow)

    np.testing.assert_array_equal(result["images"], video[[0, 2, 4]])
    np.testing.assert_array_equal(result["optical_flow"], optical_flow[[0, 2, 4]])


def test_uniform_temporal_subsample_rejects_misaligned_aliases() -> None:
    transform = A.Compose(
        [A.UniformTemporalSubsample(num_frames=3)],
        additional_targets={"optical_flow": "images"},
        strict=True,
    )

    with pytest.raises(ValueError, match="same number of frames"):
        transform(images=make_indexed_video(5), optical_flow=make_indexed_video(4))


@pytest.mark.parametrize(
    "data",
    [
        {"images": make_indexed_video(4), "masks": np.zeros((4, 5, 7), dtype=np.uint8)},
        {"image": np.zeros((5, 7, 3), dtype=np.uint8)},
    ],
)
def test_uniform_temporal_subsample_rejects_unsupported_targets(data: dict[str, np.ndarray]) -> None:
    transform = A.Compose([A.UniformTemporalSubsample(num_frames=2)], strict=True)

    with pytest.raises(ValueError, match="supports only `images`"):
        transform(**data)


def test_uniform_temporal_subsample_rejects_empty_video() -> None:
    transform = A.Compose([A.UniformTemporalSubsample(num_frames=2)], strict=True)

    with pytest.raises(ValueError, match="at least one frame"):
        transform(images=np.empty((0, 5, 7, 3), dtype=np.uint8))


def test_uniform_temporal_subsample_functional_owner() -> None:
    video = make_indexed_video(5)

    result = ftemporal.uniform_temporal_subsample(video, (0, 2, 4))

    np.testing.assert_array_equal(result, video[[0, 2, 4]])


@pytest.mark.parametrize("num_frames", [0, -1])
def test_uniform_temporal_subsample_rejects_invalid_num_frames(num_frames: int) -> None:
    with pytest.raises(ValueError):
        A.UniformTemporalSubsample(num_frames=num_frames)


def test_uniform_temporal_subsample_replay_and_serialization() -> None:
    video = make_indexed_video(5)
    replay_transform = A.ReplayCompose([A.UniformTemporalSubsample(num_frames=3)], p=1)

    result = replay_transform(images=video)
    replay = json.loads(json.dumps(result["replay"]))
    replayed = A.ReplayCompose.replay(replay, images=video + 10)
    restored = A.from_dict(A.to_dict(A.UniformTemporalSubsample(num_frames=3, p=1)))

    np.testing.assert_array_equal(replayed["images"], (video + 10)[[0, 2, 4]])
    np.testing.assert_array_equal(restored(images=video)["images"], video[[0, 2, 4]])
