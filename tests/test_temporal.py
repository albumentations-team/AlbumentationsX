import json
from copy import deepcopy

import numpy as np
import pytest
import torch

import albumentations as A


def make_indexed_video(num_frames: int) -> np.ndarray:
    return np.broadcast_to(np.arange(num_frames, dtype=np.uint8)[:, None, None, None], (num_frames, 5, 7, 3)).copy()


@pytest.mark.parametrize(
    ("input_frames", "output_frames", "expected_indices"),
    [
        (6, 4, [0, 1, 3, 5]),
        (3, 5, [0, 0, 1, 1, 2]),
        (1, 4, [0, 0, 0, 0]),
        (6, 1, [0]),
        (6, 2, [0, 5]),
        (3, 3, [0, 1, 2]),
    ],
)
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_uniform_temporal_subsample_values(input_frames, output_frames, expected_indices, dtype) -> None:
    video = make_indexed_video(input_frames).astype(dtype)
    source = video.copy()
    transform = A.Compose([A.UniformTemporalSubsample(num_frames=output_frames)], strict=True)

    result = transform(images=video)["images"]

    np.testing.assert_array_equal(result, source[expected_indices])
    np.testing.assert_array_equal(video, source)
    assert result.dtype == video.dtype
    assert not np.shares_memory(result, video)


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_uniform_temporal_subsample_selects_bound_masks_and_aliases(tensor, dtype) -> None:
    video = make_indexed_video(6).astype(dtype)[:, :, ::2]
    masks = video[..., 0]
    if tensor:
        video = torch.from_numpy(video).permute(0, 3, 1, 2)
        masks = torch.from_numpy(masks)
    source = deepcopy({"images": video, "flow": video + 10, "masks": masks, "depth": masks + 20})
    transform = A.Compose(
        [A.UniformTemporalSubsample(num_frames=4)],
        additional_targets={"flow": "images", "depth": "masks"},
        frame_binding=["images", "masks"],
        strict=True,
    )

    result = transform(**deepcopy(source))

    for name, value in source.items():
        expected = value[[0, 1, 3, 5]]
        if tensor:
            torch.testing.assert_close(result[name], expected)
        else:
            np.testing.assert_array_equal(result[name], expected)


def test_uniform_temporal_subsample_selects_instances_before_spatial_transform() -> None:
    video = make_indexed_video(3)
    instance = {
        "bbox": [1, 1, 4, 4],
        "mask": np.arange(35, dtype=np.uint8).reshape(5, 7),
        "keypoints": [[2, 3]],
        "bbox_labels": {"label": "cat"},
        "keypoint_labels": {"point": ["nose"]},
    }
    annotations = [{"instances": [instance]}, {}, {"instances": [deepcopy(instance), deepcopy(instance)]}]
    source = deepcopy(annotations)
    transform = A.Compose(
        [A.UniformTemporalSubsample(num_frames=5), A.HorizontalFlip(p=1)],
        frame_binding=["images", "frame_annotations"],
        instance_binding=["bboxes", "masks", "keypoints"],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["label"]),
        keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["point"]),
        strict=True,
    )

    result = transform(images=video, frame_annotations=annotations)

    np.testing.assert_array_equal(result["images"], video[[0, 0, 1, 1, 2], :, ::-1])
    frames = result["frame_annotations"]
    assert [len(frame["instances"]) for frame in frames] == [1, 1, 0, 0, 2]
    for frame in (frames[0], frames[1], frames[4]):
        for item in frame["instances"]:
            np.testing.assert_allclose(item["bbox"], [3, 1, 6, 4])
            np.testing.assert_array_equal(item["mask"], source[0]["instances"][0]["mask"][:, ::-1])
            np.testing.assert_allclose(item["keypoints"], [[4, 3]])
            assert item["bbox_labels"] == {"label": "cat"}
            assert item["keypoint_labels"] == {"point": ["nose"]}
    frames[0]["instances"][0]["mask"][0, 0] = 137
    assert frames[1]["instances"][0]["mask"][0, 0] == 6
    np.testing.assert_array_equal(annotations[0]["instances"][0]["mask"], source[0]["instances"][0]["mask"])


def test_uniform_temporal_subsample_replay_across_layouts_and_spatial_sizes() -> None:
    video = make_indexed_video(6)
    transform = A.ReplayCompose([A.UniformTemporalSubsample(num_frames=4)])
    recorded = transform(images=video)
    replay = json.loads(json.dumps(recorded["replay"], allow_nan=False))
    other = torch.arange(6 * 5 * 9 * 11, dtype=torch.float32).reshape(6, 5, 9, 11).transpose(-1, -2)

    result = A.ReplayCompose.replay(replay, images=other)["images"]

    torch.testing.assert_close(result, other[[0, 1, 3, 5]])
    sampled = replay["transforms"][0]["params"]["params"]
    assert sampled["frame_indices"] == [0, 1, 3, 5]
    assert {"frame_ids", "instance_ids", "instance_frame_ids", "source_frame_ids", "frame_id_column"}.isdisjoint(
        sampled
    )


def test_uniform_temporal_subsample_nested_with_bound_masks() -> None:
    video = make_indexed_video(6)
    masks = video[..., 0]
    transform = A.Compose(
        [A.OneOf([A.Compose([A.UniformTemporalSubsample(num_frames=4)], telemetry=False)], p=1), A.HorizontalFlip(p=1)],
        frame_binding=["images", "masks"],
        telemetry=False,
    )

    result = transform(images=video, masks=masks)

    np.testing.assert_array_equal(result["images"], video[[0, 1, 3, 5], :, ::-1])
    np.testing.assert_array_equal(result["masks"], masks[[0, 1, 3, 5], :, ::-1])


def test_uniform_temporal_subsample_preserves_empty_frames_with_configured_labels() -> None:
    video = make_indexed_video(2)
    transform = A.Compose(
        [A.UniformTemporalSubsample(num_frames=3)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["category"]),
        keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["point"]),
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )

    result = transform(images=video, frame_annotations=[{}, {}])

    np.testing.assert_array_equal(result["images"], video[[0, 0, 1]])
    assert result["frame_annotations"] == [{}, {}, {}]


@pytest.mark.parametrize("num_frames", [0, -1])
def test_uniform_temporal_subsample_rejects_invalid_num_frames(num_frames) -> None:
    with pytest.raises(ValueError):
        A.UniformTemporalSubsample(num_frames=num_frames)


def test_uniform_temporal_subsample_requires_images() -> None:
    transform = A.Compose([A.UniformTemporalSubsample(num_frames=2)])
    with pytest.raises(ValueError, match="requires an image-like target"):
        transform(image=np.zeros((5, 7, 3), dtype=np.uint8))


def test_uniform_temporal_subsample_requires_binding_for_annotations() -> None:
    transform = A.Compose([A.UniformTemporalSubsample(num_frames=2)])
    with pytest.raises(ValueError, match="requires Compose\\(frame_binding"):
        transform(images=make_indexed_video(4), masks=np.zeros((4, 5, 7), dtype=np.uint8))


@pytest.mark.parametrize("num_frames", [1, 2])
def test_uniform_temporal_subsample_rejects_empty_video(num_frames) -> None:
    transform = A.Compose([A.UniformTemporalSubsample(num_frames=num_frames)])
    with pytest.raises(ValueError, match="frame_indices"):
        transform(images=np.empty((0, 5, 7, 3), dtype=np.uint8))


def test_uniform_temporal_subsample_skip_preserves_inputs() -> None:
    video = make_indexed_video(6)[..., 0]
    user_data = {"source": "clip"}
    skipped = A.Compose([A.UniformTemporalSubsample(num_frames=4, p=0)])

    result = skipped(images=video, user_data=user_data)

    np.testing.assert_array_equal(result["images"], video)
    assert result["user_data"] is user_data


def test_uniform_temporal_subsample_preserves_channel_free_numpy_layout() -> None:
    video = make_indexed_video(6)[..., 0]
    video.setflags(write=False)
    transform = A.Compose([A.UniformTemporalSubsample(num_frames=4)])

    result = transform(images=video)["images"]

    assert result.shape == (4, 5, 7)
    np.testing.assert_array_equal(result, video[[0, 1, 3, 5]])
