"""Portable-constructor contracts for every public composition type."""

import copy
import inspect
import io
from collections.abc import Iterator

import cv2
import pytest

import albumentations as A
from albumentations.core.composition import BaseCompose
from albumentations.core.keypoints_utils import KeypointParams


def _iter_public_composition_types(base_class: type[BaseCompose]) -> Iterator[type[BaseCompose]]:
    for subclass in base_class.__subclasses__():
        if subclass.__module__.startswith("albumentations."):
            yield subclass
            yield from _iter_public_composition_types(subclass)


def _portable_constructor_parameters(composition_type: type[BaseCompose]) -> set[str]:
    aliases = {"first", "second"} if composition_type is A.OneOrOther else set()
    return {
        name
        for name, parameter in inspect.signature(composition_type.__init__).parameters.items()
        if name not in {"self", *aliases}
        and parameter.kind in {parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY}
    }


def _non_default_witness(parameter_name: str) -> object:
    if parameter_name == "transforms":
        return [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]
    if parameter_name == "bbox_params":
        return A.BboxParams(
            "pascal_voc",
            label_fields=["bbox_labels"],
            min_area=1.0,
            min_visibility=0.1,
            min_width=1.0,
            min_height=1.0,
            check_each_transform=False,
            filter_invalid_bboxes=True,
            max_accept_ratio=3.0,
            clip_bboxes_on_input=True,
            clip_after_transform=False,
        )
    if parameter_name == "keypoint_params":
        return A.KeypointParams(
            "xy",
            label_fields=["keypoint_labels"],
            remove_invisible=False,
            angle_in_degrees=False,
            check_each_transform=False,
            label_mapping={"HorizontalFlip": {"keypoint_labels": {0: 1, 1: 0}}},
        )
    if parameter_name == "additional_targets":
        return {"image2": "image"}
    if parameter_name == "p":
        return 0.73
    if parameter_name == "is_check_shapes":
        return False
    if parameter_name == "save_key":
        return "captured_replay"
    if parameter_name == "seed":
        return 137
    if parameter_name == "instance_binding":
        return ("mask", "bboxes")
    if parameter_name == "semantic_mask_label_mappings":
        return {"HorizontalFlip": {0: 1, 1: 0}}
    if parameter_name == "strict":
        return True
    if parameter_name == "mask_interpolation":
        return cv2.INTER_LINEAR
    if parameter_name == "save_applied_params":
        return True
    if parameter_name == "telemetry":
        return False
    if parameter_name == "strict_instance_invariant":
        return False
    if parameter_name == "n":
        return 2
    if parameter_name == "replace":
        return True
    if parameter_name == "channels":
        return (1,)
    msg = f"Add a non-default portable-constructor witness for '{parameter_name}'."
    raise AssertionError(msg)


COMPOSITION_TYPES = tuple(sorted(_iter_public_composition_types(BaseCompose), key=lambda cls: cls.__name__))


@pytest.mark.parametrize("composition_type", COMPOSITION_TYPES, ids=lambda cls: cls.__name__)
def test_composition_portable_constructor_witnesses_roundtrip_all_formats(
    composition_type: type[BaseCompose],
) -> None:
    parameter_names = _portable_constructor_parameters(composition_type)
    init_kwargs = {name: _non_default_witness(name) for name in parameter_names}

    original = composition_type(**copy.deepcopy(init_kwargs))
    serialized = A.to_dict(original)
    missing_fields = parameter_names.difference(serialized["transform"])
    assert not missing_fields, (
        f"{composition_type.__name__} does not serialize portable constructor fields: {sorted(missing_fields)}"
    )
    restored = A.from_dict(serialized)

    assert A.to_dict(restored) == serialized

    for data_format in ("json", "yaml"):
        buffer = io.StringIO()
        A.save(original, buffer, data_format=data_format)
        buffer.seek(0)
        reloaded = A.load(buffer, data_format=data_format)
        assert A.to_dict(reloaded) == serialized


@pytest.mark.parametrize(
    "params",
    [
        A.BboxParams(
            "pascal_voc",
            label_fields=["bbox_labels"],
            min_area=1.0,
            min_visibility=0.1,
            min_width=1.0,
            min_height=1.0,
            check_each_transform=False,
            filter_invalid_bboxes=True,
            max_accept_ratio=3.0,
            clip_bboxes_on_input=True,
            clip_after_transform=False,
        ),
        A.KeypointParams(
            "xy",
            label_fields=["keypoint_labels"],
            remove_invisible=False,
            angle_in_degrees=False,
            check_each_transform=False,
            label_mapping={"HorizontalFlip": {"keypoint_labels": {0: 1, 1: 0}}},
        ),
    ],
    ids=("bbox", "keypoints"),
)
def test_processor_params_serialize_every_public_constructor_field(
    params: A.BboxParams | KeypointParams,
) -> None:
    parameter_names = {
        name
        for name, parameter in inspect.signature(type(params).__init__).parameters.items()
        if name != "self" and parameter.kind in {parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY}
    }

    assert set(params.to_dict_private()) == parameter_names


def test_processor_params_roundtrip_through_compose_json_and_yaml() -> None:
    original = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=_non_default_witness("bbox_params"),
        keypoint_params=_non_default_witness("keypoint_params"),
        telemetry=False,
    )
    expected = A.to_dict(original)

    for data_format in ("json", "yaml"):
        buffer = io.StringIO()
        A.save(original, buffer, data_format=data_format)
        buffer.seek(0)
        assert A.to_dict(A.load(buffer, data_format=data_format)) == expected
