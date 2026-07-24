"""Completeness contracts for the transform-case registry."""

import inspect
import json
from pathlib import Path
from typing import Literal, get_args, get_origin, get_type_hints

import numpy as np
import pytest

import albumentations as A
from tests.helpers.applied_config import ReplayProfile
from tests.helpers.transform_cases import (
    PRIMARY_TRANSFORM_CONTRACT_CASES,
    TRANSFORM_CASES_BY_CLASS,
    TRANSFORM_CONTRACT_CASES,
    TransformContractCase,
)
from tests.utils import get_transforms

POSITIONAL_PARAMETERS_SNAPSHOT = Path(__file__).parents[1] / "files" / "public_transform_positional_parameters.json"


def _differs_from_default(value: object, default: object) -> bool:
    if default is inspect.Parameter.empty:
        return True
    if isinstance(value, np.ndarray) or isinstance(default, np.ndarray):
        return not np.array_equal(value, default)
    return bool(value != default)


def _has_non_default_value(transform_cls: type[A.BasicTransform], name: str, parameter: inspect.Parameter) -> bool:
    annotation = get_type_hints(transform_cls.__init__).get(name, parameter.annotation)
    if get_origin(annotation) is Literal:
        return any(_differs_from_default(value, parameter.default) for value in get_args(annotation))
    return True


def test_every_serializable_transform_has_a_contract_case() -> None:
    public_transforms = {
        transform_cls
        for transform_cls, _ in get_transforms(except_augmentations={A.Lambda})
        if transform_cls.is_serializable()
    }
    registered_transforms = {case.transform_cls for case in TRANSFORM_CONTRACT_CASES}

    assert registered_transforms == public_transforms


def test_contract_case_ids_are_unique() -> None:
    case_ids = [case.case_id for case in TRANSFORM_CONTRACT_CASES]

    assert len(case_ids) == len(set(case_ids))


@pytest.mark.parametrize(
    ("case_kwargs", "message"),
    [
        ({"case_id": "Invalid_ID", "transform_cls": A.HorizontalFlip}, "case_id"),
        (
            {"case_id": "forbidden-harness-argument", "transform_cls": A.HorizontalFlip, "init_kwargs": {"p": 1}},
            "harness-owned",
        ),
        (
            {"case_id": "unknown-argument", "transform_cls": A.HorizontalFlip, "init_kwargs": {"mode": "x"}},
            "unknown public constructor",
        ),
        ({"case_id": "missing-seed", "transform_cls": A.HorizontalFlip, "seeds": ()}, "at least one"),
    ],
)
def test_contract_case_rejects_invalid_declarations(case_kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TransformContractCase(**case_kwargs)


def test_contract_case_copies_and_freezes_input_mappings() -> None:
    source = {"group_element": "r90"}
    case = TransformContractCase(
        case_id="immutable-mapping",
        transform_cls=A.RandomRotate90,
        init_kwargs=source,
    )

    source["group_element"] = "r270"

    assert case.init_kwargs == {"group_element": "r90"}
    with pytest.raises(TypeError):
        case.init_kwargs["group_element"] = "r180"


def test_every_public_constructor_parameter_has_a_non_default_case() -> None:
    covered: set[tuple[type[A.BasicTransform], str]] = set()
    parameters: set[tuple[type[A.BasicTransform], str]] = set()

    for case in TRANSFORM_CONTRACT_CASES:
        for name, parameter in inspect.signature(case.transform_cls.__init__).parameters.items():
            if name in {"self", "p", "strict"} or parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            key = (case.transform_cls, name)
            if not _has_non_default_value(case.transform_cls, name, parameter):
                continue
            parameters.add(key)
            if name not in case.init_kwargs:
                continue
            if _differs_from_default(case.init_kwargs[name], parameter.default):
                covered.add(key)

    missing = sorted(f"{transform_cls.__name__}.{name}" for transform_cls, name in parameters - covered)

    assert not missing, "Public parameters without a non-default contract case:\n" + "\n".join(missing)


def test_exact_replay_profile_applies_to_every_registered_mode() -> None:
    exact_transform_classes = {
        case.transform_cls for case in PRIMARY_TRANSFORM_CONTRACT_CASES if case.replay_profile is ReplayProfile.EXACT
    }
    non_exact_cases = [
        case.case_id
        for transform_cls in exact_transform_classes
        for case in TRANSFORM_CASES_BY_CLASS[transform_cls]
        if case.replay_profile is not ReplayProfile.EXACT
    ]

    assert not non_exact_cases, "Modes missing the transform's exact replay profile:\n" + "\n".join(non_exact_cases)


def test_crop_and_pad_registered_modes_require_exact_replay() -> None:
    non_exact_cases = [
        case.case_id
        for case in TRANSFORM_CASES_BY_CLASS[A.CropAndPad]
        if case.replay_profile is not ReplayProfile.EXACT
    ]

    assert not non_exact_cases, "CropAndPad modes without exact replay:\n" + "\n".join(non_exact_cases)


def test_public_transform_positional_parameters_are_stable() -> None:
    expected = json.loads(POSITIONAL_PARAMETERS_SNAPSHOT.read_text())
    registered_transform_classes = {case.transform_cls for case in TRANSFORM_CONTRACT_CASES}
    actual = {
        transform_cls.__name__: [
            name
            for name, parameter in inspect.signature(transform_cls.__init__).parameters.items()
            if name != "self" and parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        for transform_cls in sorted(registered_transform_classes, key=lambda cls: cls.__name__)
    }

    assert actual == expected, (
        "Public positional constructor parameters changed. Existing positional parameters are frozen; "
        "add new parameters as keyword-only."
    )
