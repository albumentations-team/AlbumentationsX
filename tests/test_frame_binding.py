import pickle
from copy import deepcopy

import numpy as np
import pytest
import torch

import albumentations as A
from albumentations.core.transform_params import SampledParams
from albumentations.pytorch import ToTensorV2
from tests.helpers.contract_assertions import assert_contract_values_equal


def _clip(frame_count: int = 3, height: int = 8, width: int = 12) -> np.ndarray:
    return np.arange(frame_count * height * width, dtype=np.uint8).reshape(frame_count, height, width, 1)


def test_image_and_mask_binding_validates_one_mask_per_frame() -> None:
    images = _clip()
    masks = np.arange(3 * 8 * 12, dtype=np.uint8).reshape(3, 8, 12)
    compose = A.Compose(
        [A.HorizontalFlip(p=1)],
        frame_binding=["images", "masks"],
        strict=True,
        telemetry=False,
    )

    result = compose(images=images, masks=masks)

    np.testing.assert_array_equal(result["images"], images[:, :, ::-1])
    np.testing.assert_array_equal(result["masks"], masks[:, :, ::-1])


def test_image_and_mask_binding_processes_tensor_batches() -> None:
    images = torch.arange(3 * 1 * 8 * 12, dtype=torch.float32).reshape(3, 1, 8, 12)
    masks = torch.arange(3 * 8 * 12, dtype=torch.uint8).reshape(3, 8, 12)
    compose = A.Compose(
        [A.HorizontalFlip(p=1)],
        frame_binding=["images", "masks"],
        telemetry=False,
    )

    result = compose(images=images, masks=masks)

    assert torch.equal(result["images"], images.flip(-1))
    assert torch.equal(result["masks"], masks.flip(-1))


def test_image_and_mask_binding_rejects_different_frame_counts() -> None:
    compose = A.Compose(
        [A.NoOp(p=1)],
        frame_binding=["images", "masks"],
        telemetry=False,
    )

    with pytest.raises(ValueError, match=r"same length.*got 3 and 2"):
        compose(images=_clip(), masks=np.zeros((2, 8, 12), dtype=np.uint8))


@pytest.mark.parametrize("with_masks", [True, False])
def test_frame_binding_validates_additional_collection_lengths(with_masks: bool) -> None:
    compose = A.Compose(
        [A.NoOp(p=1)],
        additional_targets={"images_copy": "images"},
        frame_binding=["images", "masks"],
        telemetry=False,
    )

    data = {"images": _clip(), "images_copy": _clip(frame_count=2)}
    if with_masks:
        data["masks"] = np.zeros((3, 8, 12), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"images_copy.*same length.*got 2 and 3"):
        compose(**data)


@pytest.mark.parametrize("target_name", ["bboxes", "keypoints"])
def test_image_and_mask_binding_rejects_unbound_frame_annotations(target_name: str) -> None:
    compose = A.Compose(
        [A.NoOp(p=1)],
        frame_binding=["images", "masks"],
        telemetry=False,
    )
    data = {
        "images": _clip(),
        "masks": np.zeros((3, 8, 12), dtype=np.uint8),
        target_name: [],
    }

    with pytest.raises(ValueError, match=target_name):
        compose(**data)


def test_instance_binding_with_masks_per_frame_requires_frame_dictionaries() -> None:
    with pytest.raises(ValueError, match="frame_annotations"):
        A.Compose(
            [A.NoOp(p=1)],
            bbox_params=A.BboxParams(coord_format="pascal_voc"),
            instance_binding=["masks", "bboxes"],
            frame_binding=["images", "masks"],
            telemetry=False,
        )


def test_frame_annotation_binding_processes_each_frame_and_its_labels() -> None:
    images = _clip(frame_count=2, height=16, width=20)
    mask = np.arange(16 * 20, dtype=np.uint8).reshape(16, 20)
    annotations = [
        {
            "mask": mask,
            "bboxes": [[2, 3, 10, 12]],
            "class_id": ["cat"],
            "keypoints": [[4, 5]],
            "point_id": ["left"],
        },
        {},
    ]
    compose = A.Compose(
        [A.HorizontalFlip(p=1)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_id"]),
        keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["point_id"]),
        frame_binding=["images", "frame_annotations"],
        strict=True,
        telemetry=False,
    )

    result = compose(images=images, frame_annotations=annotations)

    frame = result["frame_annotations"][0]
    np.testing.assert_array_equal(frame["mask"], mask[:, ::-1])
    np.testing.assert_allclose(frame["bboxes"], [[10, 3, 18, 12]])
    assert frame["class_id"] == ["cat"]
    np.testing.assert_allclose(frame["keypoints"], [[15, 5]])
    assert frame["point_id"] == ["left"]
    assert result["frame_annotations"][1] == {}


def test_frame_labels_keep_their_own_encoding_and_container_type() -> None:
    annotations = [
        {"bboxes": [[1, 2, 4, 5]], "label": [137]},
        {"bboxes": [[1, 2, 4, 5]], "label": np.array(["cat"])},
    ]
    compose = A.Compose(
        [A.HorizontalFlip(p=1)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["label"]),
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )

    result = compose(images=_clip(frame_count=2), frame_annotations=annotations)

    assert result["frame_annotations"][0]["label"] == [137]
    np.testing.assert_array_equal(result["frame_annotations"][1]["label"], annotations[1]["label"])
    assert result["frame_annotations"][1]["label"].dtype == annotations[1]["label"].dtype


def test_keypoint_label_mapping_uses_each_frame_encoder() -> None:
    annotations = [
        {"keypoints": [[1, 2]], "label": ["left"]},
        {"keypoints": [[2, 3], [4, 3]], "label": ["a", "left"]},
    ]
    compose = A.Compose(
        [A.HorizontalFlip(p=1)],
        keypoint_params=A.KeypointParams(
            coord_format="xy",
            label_fields=["label"],
            label_mapping={"HorizontalFlip": {"label": {"left": "right"}}},
        ),
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )

    result = compose(images=_clip(frame_count=2), frame_annotations=annotations)

    assert result["frame_annotations"][0]["label"] == ["right"]
    assert result["frame_annotations"][1]["label"] == ["a", "right"]
    np.testing.assert_allclose(result["frame_annotations"][0]["keypoints"], [[10, 2]])
    np.testing.assert_allclose(result["frame_annotations"][1]["keypoints"], [[9, 3], [7, 3]])


@pytest.mark.parametrize(("nested", "serialize"), [(False, False), (True, False), (False, True)])
def test_frame_annotation_binding_processes_masks_without_processors(nested: bool, serialize: bool) -> None:
    images = _clip(frame_count=2)
    masks = [np.arange(8 * 12, dtype=np.uint8).reshape(8, 12) + index for index in range(2)]
    annotations = [{"mask": mask} for mask in masks]
    transform = A.HorizontalFlip(p=1)
    compose = A.Compose(
        [A.Compose([transform], telemetry=False) if nested else transform],
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )
    if serialize:
        compose = pickle.loads(pickle.dumps(compose))  # noqa: S301 - round-trip of a locally created Compose

    result = compose(images=images, frame_annotations=annotations)

    np.testing.assert_array_equal(result["images"], images[:, :, ::-1])
    for frame_index, frame in enumerate(result["frame_annotations"]):
        np.testing.assert_array_equal(frame["mask"], masks[frame_index][:, ::-1])
        assert annotations[frame_index]["mask"] is masks[frame_index]


def test_frame_binding_combines_with_instance_binding_and_empty_frames() -> None:
    images = _clip(frame_count=2, height=16, width=20)
    annotations = [
        {
            "instances": [
                {
                    "mask": np.ones((16, 20), dtype=np.uint8),
                    "bbox": [2, 3, 10, 12],
                    "keypoints": np.array([[4, 5]], dtype=np.float32),
                },
            ],
        },
        {},
    ]
    compose = A.Compose(
        [A.Compose([A.HorizontalFlip(p=1)], telemetry=False)],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        keypoint_params=A.KeypointParams(coord_format="xy"),
        instance_binding=["masks", "bboxes", "keypoints"],
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )

    result = compose(images=images, frame_annotations=annotations)

    frames = result["frame_annotations"]
    assert [len(frame["instances"]) for frame in frames] == [1, 0]
    np.testing.assert_allclose(frames[0]["instances"][0]["bbox"], [10, 3, 18, 12])
    np.testing.assert_allclose(frames[0]["instances"][0]["keypoints"], [[15, 5]])
    assert frames[0]["instances"][0]["mask"].shape == (16, 20)


@pytest.mark.parametrize("nested", [False, True])
def test_frame_instance_filtering_does_not_change_other_frames(nested: bool) -> None:
    images = _clip(frame_count=2, height=16, width=20)
    outside_mask = np.zeros((16, 20), dtype=np.uint8)
    outside_mask[:2, :2] = 1
    inside_mask = np.zeros((16, 20), dtype=np.uint8)
    inside_mask[6:10, 10:14] = 1
    annotations = [
        {"instances": [{"mask": outside_mask, "bbox": [0, 0, 2, 2]}]},
        {"instances": [{"mask": inside_mask, "bbox": [10, 6, 14, 10]}]},
    ]
    transform = A.CenterCrop(height=8, width=10, p=1)
    compose = A.Compose(
        [A.Compose([transform], telemetry=False) if nested else transform],
        bbox_params=A.BboxParams(coord_format="pascal_voc", min_visibility=0.1),
        instance_binding=["masks", "bboxes"],
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )

    result = compose(images=images, frame_annotations=annotations)

    frames = result["frame_annotations"]
    assert [len(frame["instances"]) for frame in frames] == [0, 1]
    np.testing.assert_array_equal(frames[1]["instances"][0]["mask"], inside_mask[4:12, 5:15])
    np.testing.assert_allclose(frames[1]["instances"][0]["bbox"], [5, 2, 9, 6])


@pytest.mark.parametrize("check_each_transform", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_frame_instance_visibility_filter_is_frame_local(check_each_transform: bool, nested: bool) -> None:
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[6:10, 10:14] = 1
    annotations = [
        {"instances": [{"mask": mask, "bbox": [10, 6, 14, 10], "bbox_labels": {"label": "keep"}}]},
        {
            "instances": [
                {"mask": mask, "bbox": [0, 0, 14, 10], "bbox_labels": {"label": "drop"}},
                {"mask": mask, "bbox": [10, 6, 14, 10], "bbox_labels": {"label": "keep too"}},
            ]
        },
    ]
    transform = A.CenterCrop(height=8, width=10, p=1)
    compose = A.Compose(
        [A.Compose([transform], telemetry=False) if nested else transform],
        bbox_params=A.BboxParams(
            coord_format="pascal_voc",
            min_visibility=0.5,
            label_fields=["label"],
            check_each_transform=check_each_transform,
        ),
        instance_binding=["masks", "bboxes"],
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )

    result = compose(images=_clip(frame_count=2, height=16, width=20), frame_annotations=annotations)

    frames = result["frame_annotations"]
    assert [len(frame["instances"]) for frame in frames] == [1, 1]
    assert frames[0]["instances"][0]["bbox_labels"] == {"label": "keep"}
    np.testing.assert_array_equal(frames[0]["instances"][0]["mask"], mask[4:12, 5:15])
    np.testing.assert_allclose(frames[0]["instances"][0]["bbox"], [5, 2, 9, 6])
    assert frames[1]["instances"][0]["bbox_labels"] == {"label": "keep too"}
    np.testing.assert_array_equal(frames[1]["instances"][0]["mask"], mask[4:12, 5:15])
    np.testing.assert_allclose(frames[1]["instances"][0]["bbox"], [5, 2, 9, 6])


@pytest.mark.parametrize("mask_channels", [None, 1, 3])
@pytest.mark.parametrize("tensor_images", [False, True])
def test_frame_tensor_annotations_preserve_layout_and_type(mask_channels: int | None, tensor_images: bool) -> None:
    mask_shape = (8, 12) if mask_channels is None else (mask_channels, 8, 12)
    mask = torch.arange(np.prod(mask_shape), dtype=torch.float32).reshape(mask_shape)
    annotations = [
        {
            "mask": mask,
            "bboxes": torch.tensor([[1, 2, 4, 5]], dtype=torch.float32),
            "keypoints": torch.tensor([[1, 2]], dtype=torch.float32),
            "label": ["cat"],
        },
        {},
    ]
    images = _clip(frame_count=2)
    if tensor_images:
        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    compose = A.Compose(
        [A.Resize(height=4, width=6, p=1)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["label"]),
        keypoint_params=A.KeypointParams(coord_format="xy"),
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )

    result = compose(images=images, frame_annotations=annotations)

    frame = result["frame_annotations"][0]
    assert isinstance(frame["mask"], torch.Tensor)
    expected_shape = (4, 6) if mask_channels is None else (mask_channels, 4, 6)
    assert frame["mask"].shape == expected_shape
    torch.testing.assert_close(frame["mask"], mask[..., ::2, ::2])
    torch.testing.assert_close(frame["bboxes"], torch.tensor([[0.5, 1, 2, 2.5]], dtype=torch.float32))
    torch.testing.assert_close(frame["keypoints"], torch.tensor([[0.25, 0.75]], dtype=torch.float32))
    assert frame["label"] == ["cat"]
    assert result["frame_annotations"][1] == {}


@pytest.mark.parametrize("p", [0, 1])
@pytest.mark.parametrize(
    "invalid_tensor",
    [
        torch.zeros((8, 12), dtype=torch.float32, requires_grad=True),
        torch.zeros((8, 12), dtype=torch.float64),
        torch.zeros((8, 12), device="meta"),
    ],
)
def test_frame_tensor_validation_precedes_root_probability(p: float, invalid_tensor: torch.Tensor) -> None:
    compose = A.Compose(
        [A.NoOp(p=1)],
        p=p,
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )

    with pytest.raises((ValueError, TypeError), match=r"frame_annotations\[0\].mask"):
        compose(images=_clip(frame_count=1), frame_annotations=[{"mask": invalid_tensor}])


def test_frame_tensor_targets_reject_numpy_to_tensor_terminal() -> None:
    compose = A.Compose(
        [ToTensorV2(p=1)],
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )

    with pytest.raises(TypeError, match="accept NumPy input only"):
        compose(images=_clip(frame_count=1), frame_annotations=[{"mask": torch.zeros((8, 12))}])


def test_frame_tensor_mask_uses_native_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    original = A.HorizontalFlip.apply_to_mask

    def apply_to_mask(self: A.HorizontalFlip, mask: torch.Tensor, **params: object) -> torch.Tensor:
        assert isinstance(mask, torch.Tensor)
        return original(self, mask, **params)

    monkeypatch.setattr(A.HorizontalFlip, "apply_to_mask", apply_to_mask)
    mask = torch.arange(8 * 12, dtype=torch.float32).reshape(8, 12)
    compose = A.Compose(
        [A.HorizontalFlip(p=1)],
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )

    result = compose(images=_clip(frame_count=1), frame_annotations=[{"mask": mask}])

    torch.testing.assert_close(result["frame_annotations"][0]["mask"], mask.flip(-1))


def test_frame_tensor_restoration_follows_selection_and_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    masks = [torch.arange(8 * 12, dtype=torch.float32).reshape(8, 12), torch.ones((1, 8, 12))]
    annotations = [{"mask": mask, "mask_copy": mask.clone()} for mask in masks]
    transform = A.UniformTemporalSubsample(num_frames=3)
    monkeypatch.setattr(
        A.UniformTemporalSubsample,
        "sample_parameters",
        lambda *args, **kwargs: SampledParams(params={"frame_indices": (1, 0, 1)}),
    )
    compose = A.ReplayCompose(
        [transform, A.HorizontalFlip(p=1)],
        additional_targets={"mask_copy": "mask"},
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )
    images = _clip(frame_count=2)

    result = compose(images=images, frame_annotations=annotations)
    replayed = A.ReplayCompose.replay(result["replay"], images=images, frame_annotations=annotations)

    for output in (result, replayed):
        for frame, index in zip(output["frame_annotations"], (1, 0, 1), strict=True):
            torch.testing.assert_close(frame["mask"], masks[index].flip(-1))
            torch.testing.assert_close(frame["mask_copy"], masks[index].flip(-1))
    result["frame_annotations"][0]["mask"][0, 0, 0] = 137
    assert result["frame_annotations"][2]["mask"][0, 0, 0] == 1
    assert masks[1][0, 0, 0] == 1


def test_bbox_safe_crop_samples_from_annotations_across_frames() -> None:
    images = _clip(frame_count=2, height=16, width=20)
    annotations = [{"bboxes": [[1, 2, 4, 5]]}, {"bboxes": [[16, 11, 19, 15]]}]
    compose = A.Compose(
        [A.BBoxSafeRandomCrop(erosion_rate=0, p=1)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", min_visibility=1),
        frame_binding=["images", "frame_annotations"],
        seed=137,
        telemetry=False,
    )

    result = compose(images=images, frame_annotations=annotations)

    assert [len(frame["bboxes"]) for frame in result["frame_annotations"]] == [1, 1]
    for frame, original in zip(result["frame_annotations"], annotations, strict=True):
        bbox = frame["bboxes"][0]
        np.testing.assert_allclose(
            bbox[2:] - bbox[:2],
            [
                original["bboxes"][0][2] - original["bboxes"][0][0],
                original["bboxes"][0][3] - original["bboxes"][0][1],
            ],
            atol=1e-5,
        )


@pytest.mark.parametrize("tensor_input", [False, True])
def test_sampled_frame_indices_select_bound_images_and_annotations(
    monkeypatch: pytest.MonkeyPatch,
    tensor_input: bool,
) -> None:
    images = _clip()
    masks = np.arange(3 * 8 * 12, dtype=np.uint8).reshape(3, 8, 12)
    if tensor_input:
        images = torch.from_numpy(np.moveaxis(images, -1, 1))
        masks = torch.from_numpy(masks)
    monkeypatch.setattr(
        A.UniformTemporalSubsample,
        "sample_parameters",
        lambda *args, **kwargs: SampledParams(params={"frame_indices": (2, 0, 2)}),
    )
    compose = A.ReplayCompose(
        [A.UniformTemporalSubsample(num_frames=3), A.HorizontalFlip(p=1)],
        frame_binding=["images", "masks"],
        strict=True,
        telemetry=False,
    )

    result = compose(images=images, masks=masks)
    replayed = A.ReplayCompose.replay(result["replay"], images=images, masks=masks)

    expected_images = images[[2, 0, 2]].flip(-1) if tensor_input else images[[2, 0, 2], :, ::-1]
    expected_masks = masks[[2, 0, 2]].flip(-1) if tensor_input else masks[[2, 0, 2], :, ::-1]
    assert_contract_values_equal(result["images"], expected_images)
    assert_contract_values_equal(result["masks"], expected_masks)
    assert_contract_values_equal(replayed["images"], expected_images)
    assert_contract_values_equal(replayed["masks"], expected_masks)


def test_sampled_frame_indices_select_aliases_of_bound_collections() -> None:
    images = _clip()
    masks = np.arange(3 * 8 * 12, dtype=np.uint8).reshape(3, 8, 12)
    images_copy = images + 5
    masks_copy = masks + 7
    compose = A.Compose(
        [A.UniformTemporalSubsample(num_frames=2)],
        additional_targets={"images_copy": "images", "masks_copy": "masks"},
        frame_binding=["images", "masks"],
        telemetry=False,
    )
    result = compose(images=images, masks=masks, images_copy=images_copy, masks_copy=masks_copy)

    np.testing.assert_array_equal(result["images"], images[[0, 2]])
    np.testing.assert_array_equal(result["masks"], masks[[0, 2]])
    np.testing.assert_array_equal(result["images_copy"], images_copy[[0, 2]])
    np.testing.assert_array_equal(result["masks_copy"], masks_copy[[0, 2]])


def test_frame_indices_require_binding_for_annotation_aliases() -> None:
    transform = A.UniformTemporalSubsample(num_frames=1)
    transform.add_targets({"boxes": "bboxes"})

    with pytest.raises(ValueError, match=r"requires Compose\(frame_binding"):
        transform.apply_with_params(
            SampledParams(params={"frame_indices": (0,)}),
            images=_clip(),
            boxes=np.zeros((1, 4), dtype=np.float32),
        )


def test_sampled_frame_indices_copy_repeated_frame_annotations(monkeypatch: pytest.MonkeyPatch) -> None:
    images = _clip()
    masks = [np.full((8, 12), index, dtype=np.uint8) for index in range(3)]
    annotations = [{"mask": mask} for mask in masks]
    monkeypatch.setattr(
        A.UniformTemporalSubsample,
        "sample_parameters",
        lambda *args, **kwargs: SampledParams(params={"frame_indices": (2, 0, 2)}),
    )
    compose = A.Compose(
        [A.UniformTemporalSubsample(num_frames=3)],
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )
    result = compose(images=images, frame_annotations=annotations)

    assert [int(frame["mask"][0, 0]) for frame in result["frame_annotations"]] == [2, 0, 2]
    assert result["frame_annotations"][0] is not result["frame_annotations"][2]
    assert not np.shares_memory(result["frame_annotations"][0]["mask"], result["frame_annotations"][2]["mask"])
    result["frame_annotations"][0]["mask"][0, 0] = 137
    assert result["frame_annotations"][2]["mask"][0, 0] == 2
    assert masks[2][0, 0] == 2


def test_frame_annotations_require_explicit_binding() -> None:
    compose = A.Compose([A.NoOp(p=1)], telemetry=False)

    with pytest.raises(ValueError, match=r"requires Compose\(frame_binding"):
        compose(images=_clip(), frame_annotations=[{}, {}, {}])


@pytest.mark.parametrize(
    "frame_binding",
    [("images", "masks"), ("images", "frame_annotations")],
)
def test_frame_binding_is_serialized_by_replay_compose(frame_binding: tuple[str, str]) -> None:
    compose = A.ReplayCompose(
        [A.NoOp(p=1)],
        frame_binding=frame_binding,
        telemetry=False,
    )

    assert compose.to_dict_private()["frame_binding"] == list(frame_binding)


def test_replay_compose_replays_frame_annotations() -> None:
    images = _clip(frame_count=2, height=16, width=20)
    annotations = [
        {
            "mask": np.arange(16 * 20, dtype=np.uint8).reshape(16, 20),
            "bboxes": [[2, 3, 10, 12]],
            "class_id": ["cat"],
            "keypoints": [[4, 5]],
            "point_id": ["left"],
        },
        {},
    ]
    inputs = {"images": images, "frame_annotations": annotations}
    compose = A.ReplayCompose(
        [A.HorizontalFlip(p=1)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_id"]),
        keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["point_id"]),
        frame_binding=["images", "frame_annotations"],
        seed=137,
        strict=True,
        telemetry=False,
    )

    recorded = compose(**deepcopy(inputs))
    replayed = A.ReplayCompose.replay(recorded["replay"], **deepcopy(inputs))

    assert_contract_values_equal(replayed["images"], recorded["images"])
    assert_contract_values_equal(replayed["frame_annotations"], recorded["frame_annotations"])


def test_temporal_handlers_receive_flat_masks_and_rebased_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    annotations = [
        {
            "instances": [
                {"mask": np.full((8, 12), 1, dtype=np.uint8), "keypoints": [[1, 2], [3, 4]]},
                {"mask": np.full((8, 12), 2, dtype=np.uint8), "keypoints": []},
            ]
        },
        {"instances": []},
        {"instances": [{"mask": np.full((8, 12), 3, dtype=np.uint8), "keypoints": [[5, 6]]}]},
    ]
    incoming = []
    original = A.UniformTemporalSubsample.apply_to_masks

    def apply_to_masks(self, masks, frame_indices, **params):
        incoming.append((masks.shape, params["frame_ids"].copy(), params["instance_ids"].copy()))
        return original(self, masks, frame_indices, **params)

    monkeypatch.setattr(A.UniformTemporalSubsample, "apply_to_masks", apply_to_masks)
    monkeypatch.setattr(
        A.UniformTemporalSubsample,
        "sample_parameters",
        lambda self, *args, **kwargs: SampledParams(
            params={"frame_indices": (2, 0, 2, 1, 0) if self.num_frames == 5 else (4, 2, 0)},
        ),
    )
    compose = A.Compose(
        [A.UniformTemporalSubsample(num_frames=5), A.HorizontalFlip(p=1), A.UniformTemporalSubsample(num_frames=3)],
        keypoint_params=A.KeypointParams(coord_format="xy"),
        instance_binding=["masks", "keypoints"],
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )
    result = compose(images=_clip(), frame_annotations=annotations)

    np.testing.assert_array_equal(incoming[0][1], [0, 0, 2])
    np.testing.assert_array_equal(incoming[0][2], [0, 1, 2])
    np.testing.assert_array_equal(incoming[1][1], [0, 1, 1, 2, 4, 4])
    np.testing.assert_array_equal(incoming[1][2], np.arange(6))
    assert incoming[0][0] == (3, 8, 12, 1)
    frames = result["frame_annotations"]
    assert [len(frame["instances"]) for frame in frames] == [2, 1, 1]
    np.testing.assert_array_equal(result["images"], _clip()[[0, 2, 2], :, ::-1])
    np.testing.assert_allclose(frames[0]["instances"][0]["keypoints"], [[10, 2], [8, 4]])
    assert frames[0]["instances"][1]["keypoints"].shape == (0, 2)
    np.testing.assert_allclose(frames[1]["instances"][0]["keypoints"], [[6, 6]])
    np.testing.assert_allclose(frames[2]["instances"][0]["keypoints"], [[6, 6]])
    frames[1]["instances"][0]["mask"][0, 0] = 137
    assert frames[2]["instances"][0]["mask"][0, 0] == 3


def test_frame_keypoint_mapping_does_not_swap_rows_between_frames() -> None:
    compose = A.Compose(
        [A.HorizontalFlip(p=1)],
        keypoint_params=A.KeypointParams(
            coord_format="xy",
            label_fields=["label"],
            label_mapping={"HorizontalFlip": {"label": {"left": "right", "right": "left"}}},
        ),
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )
    result = compose(
        images=_clip(frame_count=2),
        frame_annotations=[
            {"keypoints": [[1, 2]], "label": ["left"]},
            {"keypoints": [[5, 6]], "label": ["right"]},
        ],
    )
    first, second = result["frame_annotations"]
    np.testing.assert_allclose(first["keypoints"], [[10, 2]])
    np.testing.assert_allclose(second["keypoints"], [[6, 6]])
    assert first["label"] == ["right"]
    assert second["label"] == ["left"]


def test_frame_annotation_aliases_preserve_extra_columns() -> None:
    compose = A.Compose(
        [A.UniformTemporalSubsample(num_frames=5), A.HorizontalFlip(p=1)],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        keypoint_params=A.KeypointParams(coord_format="xy"),
        additional_targets={"boxes": "bboxes", "points": "keypoints"},
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )
    result = compose(
        images=_clip(),
        frame_annotations=[
            {"boxes": [[1, 2, 4, 5, 137]], "points": [[1, 2, 42]]},
            {},
            {"boxes": [[2, 3, 5, 6, 138]], "points": [[5, 6, 43]]},
        ],
    )
    frames = result["frame_annotations"]
    assert frames[2] == frames[3] == {}
    np.testing.assert_allclose(frames[0]["boxes"], [[8, 2, 11, 5, 137]])
    np.testing.assert_allclose(frames[1]["points"], [[10, 2, 42]])
    np.testing.assert_allclose(frames[4]["boxes"], [[7, 3, 10, 6, 138]])
    np.testing.assert_allclose(frames[4]["points"], [[6, 6, 43]])


def test_frame_binding_allows_video_without_annotations() -> None:
    compose = A.Compose(
        [A.UniformTemporalSubsample(num_frames=2)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["label"]),
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )
    result = compose(images=_clip())

    assert set(result) == {"images"}
    np.testing.assert_array_equal(result["images"], _clip()[[0, 2]])


def test_frame_instances_require_instance_binding() -> None:
    compose = A.Compose([A.NoOp()], frame_binding=["images", "frame_annotations"], telemetry=False)
    with pytest.raises(ValueError, match="requires Compose\\(instance_binding"):
        compose(images=_clip(frame_count=1), frame_annotations=[{"instances": []}])


@pytest.mark.parametrize("binding", [["bboxes", "keypoints"], ["mask", "bboxes"]])
@pytest.mark.parametrize("target", ["mask", "masks", "bboxes", "keypoints"])
@pytest.mark.parametrize("alias", [False, True])
def test_frame_instance_binding_rejects_targets_outside_instances(binding, target, alias) -> None:
    name = "annotation" if alias else target
    compose = A.Compose(
        [A.HorizontalFlip(p=1)],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        keypoint_params=A.KeypointParams(coord_format="xy"),
        instance_binding=binding,
        additional_targets={name: target} if alias else None,
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )
    values = {
        "mask": np.zeros((8, 12), dtype=np.uint8),
        "masks": np.zeros((1, 8, 12), dtype=np.uint8),
        "bboxes": [[1, 2, 4, 5]],
        "keypoints": [[1, 2]],
    }
    annotations = {"instances": [], name: values[target]}

    with pytest.raises(ValueError, match="Put bound frame objects in an `instances` list"):
        compose(images=_clip(frame_count=1), frame_annotations=[annotations])


def test_frame_annotation_aliases_share_configured_labels() -> None:
    compose = A.Compose(
        [A.UniformTemporalSubsample(num_frames=3), A.HorizontalFlip(p=1)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["label"]),
        additional_targets={"boxes": "bboxes"},
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )
    result = compose(
        images=_clip(frame_count=2),
        frame_annotations=[
            {"bboxes": [[1, 2, 4, 5]], "boxes": [[2, 3, 5, 6]], "label": ["cat"]},
            {},
        ],
    )
    for frame in result["frame_annotations"][:2]:
        np.testing.assert_allclose(frame["bboxes"], [[8, 2, 11, 5]])
        np.testing.assert_allclose(frame["boxes"], [[7, 3, 10, 6]])
        assert frame["label"] == ["cat"]
    assert result["frame_annotations"][2] == {}


def test_frame_binding_sampler_sees_only_declared_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    original = A.RandomBrightnessContrast.sample_parameters

    def sample_parameters(self, params, data, targets, sampling):
        seen.extend((view.name, view.canonical_type) for view in targets.ordered)
        return original(self, params, data, targets, sampling)

    monkeypatch.setattr(A.RandomBrightnessContrast, "sample_parameters", sample_parameters)
    masks = np.arange(2 * 8 * 12, dtype=np.uint8).reshape(2, 8, 12)
    compose = A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0, p=1)],
        frame_binding=["images", "frame_annotations"],
        telemetry=False,
    )
    result = compose(images=_clip(frame_count=2), frame_annotations=[{"mask": mask} for mask in masks])

    assert seen == [("images", "images")]
    for frame, mask in zip(result["frame_annotations"], masks, strict=True):
        np.testing.assert_array_equal(frame["mask"], mask)
