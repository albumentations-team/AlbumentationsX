from typing import Any

import numpy as np
import pytest
import torch
from pydantic import field_validator

import albumentations as A


class _CheckedNoOp(A.NoOp):
    class InitSchema(A.NoOp.InitSchema):
        marker: int

        @field_validator("marker")
        @classmethod
        def check_marker(cls, value: int) -> int:
            if value < 0:
                raise TypeError(f"marker must be nonnegative, got {value}")
            return value

    def __init__(self, marker: int = 137, p: float = 0.5):
        super().__init__(p=p)
        self.marker = marker


@pytest.mark.parametrize("strict", [False, True])
def test_custom_schema_error_cannot_replace_constructor_arguments(strict: bool) -> None:
    with pytest.raises(TypeError, match=r"marker must be nonnegative, got -1"):
        _CheckedNoOp(marker=-1, p=0.9, strict=strict)


@pytest.mark.parametrize("strict", [False, True])
def test_invalid_declared_transform_value_raises_in_both_modes(strict: bool) -> None:
    with pytest.raises(ValueError, match=r"(?s)\bp\b.*less than or equal to 1.*1\.5"):
        A.Affine(p=1.5, strict=strict)


def test_permissive_unknown_argument_preserves_valid_arguments() -> None:
    with pytest.warns(UserWarning, match=r"typo.*Blur"):
        transform = A.Blur(blur_range=(3, 5), p=0.9, typo=137)

    assert (transform.blur_range, transform.p) == ((3, 5), 0.9)


def test_wrong_declared_transform_argument_kind() -> None:
    with pytest.raises(TypeError, match=r"(?s)\bp\b.*valid number.*input_type=dict"):
        A.Affine(p={})


def test_duplicate_constructor_argument_is_not_silently_overwritten() -> None:
    with pytest.raises(TypeError, match=r"multiple values for argument 'blur_range'.*constructor signature"):
        A.Blur((3, 5), blur_range=(5, 7))


@pytest.mark.parametrize("strict", [False, True])
def test_unknown_alias_target_fails_at_compose_construction(strict: bool) -> None:
    with pytest.raises(ValueError, match=r"additional_targets.*right.*imgae.*image"):
        A.Compose([A.HorizontalFlip()], additional_targets={"right": "imgae"}, strict=strict)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"p": float("nan")}, ValueError, r"p must be in \[0, 1\], got nan"),
        ({"p": "always"}, TypeError, r"p must be a real number.*got str"),
        ({"bbox_params": {}}, ValueError, r"bbox_params must include coord_format.*pascal_voc"),
        ({"keypoint_params": []}, TypeError, r"keypoint_params must be a dict or KeypointParams, got list"),
        ({"additional_targets": []}, TypeError, r"additional_targets must be a mapping, got list"),
    ],
)
def test_invalid_compose_configuration(kwargs: dict[str, Any], error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        A.Compose([A.HorizontalFlip()], **kwargs)


def test_alias_requires_canonical_source_even_on_skip() -> None:
    pipeline = A.Compose([A.HorizontalFlip()], p=0, additional_targets={"right": "image"})

    with pytest.raises(ValueError, match=r"right.*requires canonical target 'image'"):
        pipeline(right=np.zeros((5, 7, 3), dtype=np.uint8))


def test_alias_can_reference_a_custom_target_in_one_branch() -> None:
    class HeatmapFlip(A.HorizontalFlip):
        @property
        def targets(self) -> dict[str, Any]:
            return {**super().targets, "heatmap": self.apply}

    pipeline = A.Compose(
        [A.Sequential([A.NoOp()]), HeatmapFlip(p=1)],
        additional_targets={"aux": "heatmap"},
        strict=True,
    )
    image = np.arange(35, dtype=np.uint8).reshape(5, 7, 1)

    np.testing.assert_array_equal(pipeline(image=image, aux=image)["aux"], image[:, ::-1])


def test_alias_cannot_reference_transform_metadata() -> None:
    with pytest.raises(ValueError, match=r"additional_targets.*aux.*unknown target 'refs'"):
        A.Compose([A.HistogramMatching(metadata_key="refs")], additional_targets={"aux": "refs"})


@pytest.mark.parametrize("use_tensor", [False, True], ids=["numpy", "cpu-tensor"])
@pytest.mark.parametrize("strict", [False, True])
def test_incompatible_shapes_fail_even_when_compose_would_skip(use_tensor: bool, strict: bool) -> None:
    image = torch.zeros((3, 5, 7)) if use_tensor else np.zeros((5, 7, 3), dtype=np.uint8)
    mask = torch.zeros((4, 7)) if use_tensor else np.zeros((4, 7), dtype=np.uint8)
    pipeline = A.Compose([A.HorizontalFlip()], p=0, strict=strict)

    with pytest.raises(ValueError, match=r"image.*\(5, 7\).*mask.*\(4, 7\)"):
        pipeline(image=image, mask=mask)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"image": "frame.png"}, r"image.*NumPy array or torch.Tensor.*str"),
        ({"image": np.zeros((5,), dtype=np.uint8)}, r"image.*\(H, W\).*\(5,\)"),
        ({"image": torch.zeros((5,))}, r"image.*\(H, W\).*\(5,\)"),
    ],
    ids=["container", "numpy-rank", "tensor-rank"],
)
def test_invalid_target_kind_or_rank_fails_when_compose_would_skip(data: dict[str, Any], message: str) -> None:
    pipeline = A.Compose([A.HorizontalFlip()], p=0)

    with pytest.raises(TypeError, match=message):
        pipeline(**data)


@pytest.mark.parametrize("trace", [False, True], ids=["call", "trace"])
def test_validation_error_preserves_next_seeded_augmentation(trace: bool) -> None:
    def pipeline() -> A.Compose:
        return A.Compose([A.HorizontalFlip(p=1)], p=0.5, seed=139)

    image = np.arange(35, dtype=np.uint8).reshape(5, 7, 1)
    mask = np.zeros((4, 7), dtype=np.uint8)
    failed = pipeline()
    invoke = failed.run_with_trace if trace else failed
    with pytest.raises(ValueError, match=r"image.*\(5, 7\).*mask.*\(4, 7\)"):
        invoke(image=image, mask=mask)

    np.testing.assert_array_equal(failed(image=image)["image"], pipeline()(image=image)["image"])


@pytest.mark.parametrize(
    ("annotations", "error", "message"),
    [
        ({"bboxes": [[0, 0, 2, 2]]}, ValueError, r"bboxes requires label_fields.*class_ids.*Pass these fields"),
        ({"bboxes": [[0, 0, 2, 2]], "class_ids": []}, ValueError, r"lengths of bboxes and class_ids.*"),
        ({"bboxes": [], "class_ids": [1]}, ValueError, r"lengths of bboxes and class_ids.*Got 0 and 1"),
        (
            {"bboxes": [[0, 0, np.nan, 2]], "class_ids": [1]},
            ValueError,
            r"bboxes must have finite coordinates.*nan at bboxes\[0, 2\].*Replace",
        ),
        ({"bboxes": np.zeros((4,)), "class_ids": [1]}, TypeError, r"bboxes must have shape.*got \(4,\)"),
        ({"bboxes": [[0, 0, 2]], "class_ids": [1]}, ValueError, r"bboxes needs at least 4.*got shape \(1, 3\)"),
    ],
)
def test_bbox_validation_runs_before_rng(
    annotations: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    pipeline = A.Compose(
        [A.HorizontalFlip(p=1)],
        p=0.5,
        seed=139,
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_ids"]),
    )
    image = np.arange(35, dtype=np.uint8).reshape(5, 7, 1)
    with pytest.raises(error, match=message):
        pipeline(image=image, **annotations)

    result = pipeline(image=image, bboxes=[], class_ids=[])
    np.testing.assert_array_equal(result["image"], image[:, ::-1])
    assert result["bboxes"].shape == (0, 4)
    assert result["class_ids"] == []


def test_nonfinite_tensor_keypoints_rejected_even_when_compose_would_skip() -> None:
    pipeline = A.Compose(
        [A.HorizontalFlip()],
        p=0,
        keypoint_params=A.KeypointParams(coord_format="xy", remove_invisible=False),
    )
    with pytest.raises(ValueError, match=r"keypoints must have finite coordinates.*inf at keypoints\[0, 0\]"):
        pipeline(image=torch.zeros((3, 5, 7)), keypoints=torch.tensor([[torch.inf, 0.0]]))


def test_skipped_bbox_pipeline_returns_unprocessed_targets() -> None:
    pipeline = A.Compose(
        [A.HorizontalFlip(p=1)],
        p=0,
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_ids"]),
    )
    boxes = [[0, 0, 2, 2]]
    class_ids = ["left"]
    image = np.zeros((5, 7, 3), dtype=np.uint8)

    result = pipeline(image=image, bboxes=boxes, class_ids=class_ids)

    assert result["image"] is image
    assert result["bboxes"] is boxes
    assert result["class_ids"] is class_ids


def test_skipped_bbox_pipeline_does_not_copy_strided_image(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = A.Compose([A.NoOp()], p=0, bbox_params=A.BboxParams(coord_format="pascal_voc"))
    image = np.zeros((5, 7, 3), dtype=np.uint8)[:, ::-1]
    ascontiguousarray = np.ascontiguousarray

    def check_copy(array: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        assert array is not image
        return ascontiguousarray(array, *args, **kwargs)

    monkeypatch.setattr(np, "ascontiguousarray", check_copy)

    assert pipeline(image=image, bboxes=[[0, 0, 2, 2]])["image"] is image


def test_frame_ids_cannot_fill_missing_bbox_coordinates() -> None:
    pipeline = A.Compose(
        [A.HorizontalFlip()],
        p=0,
        frame_binding=["images", "frame_annotations"],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
    )
    with pytest.raises(ValueError, match=r"frame_annotations\[1\]\.bboxes needs at least 4.*\(1, 3\)"):
        pipeline(images=np.zeros((2, 5, 7, 3), dtype=np.uint8), frame_annotations=[{}, {"bboxes": [[0, 0, 2]]}])


@pytest.mark.parametrize(
    ("payload", "error", "message"),
    [
        ({}, ValueError, r"replay is missing required fields.*applied.*Use a complete"),
        (
            {"__class_fullname__": "MissingTransform", "applied": True, "params": {}},
            ValueError,
            r"replay references unknown transform 'MissingTransform'.*Import",
        ),
        (
            {"__class_fullname__": "ReplayCompose", "applied": True, "params": []},
            TypeError,
            r"replay.params must be a dict or None, got list",
        ),
        (
            {"__class_fullname__": "HorizontalFlip", "applied": True, "params": None},
            ValueError,
            r"replay.params is required for an applied transform.*original replay record",
        ),
    ],
)
def test_malformed_replay_payload(payload: dict[str, Any], error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        A.ReplayCompose.replay(payload, image=np.zeros((5, 7, 3), dtype=np.uint8))


def test_deserialization_names_missing_lambda_transform() -> None:
    payload = A.to_dict(A.Lambda(name="custom_flip"))

    with pytest.raises(ValueError, match=r"Pass the non-serializable transform 'custom_flip'.*nonserializable"):
        A.from_dict(payload, nonserializable={})


def test_internal_preprocessing_requires_an_invocation() -> None:
    pipeline = A.Compose([])

    with pytest.raises(RuntimeError, match="Compose execution state requires an active invocation"):
        pipeline.preprocess({"image": np.zeros((5, 7), dtype=np.uint8)})
