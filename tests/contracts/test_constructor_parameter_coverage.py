"""Completeness contracts for the transform-case registry."""

import inspect
from typing import Literal, get_args, get_origin, get_type_hints

import numpy as np
import pytest

import albumentations as A
from tests.helpers.transform_cases import TRANSFORM_CONTRACT_CASES, TransformContractCase
from tests.utils import get_transforms


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
