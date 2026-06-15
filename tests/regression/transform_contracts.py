"""Regression contract registry for selected transform behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import albumentations as A
from tests.helpers.transforms import TransformTestHelper
from tests.utils import get_all_valid_transforms

StabilityMode = Literal["exact", "tolerance", "digest", "structural"]


@dataclass(frozen=True)
class TransformContract:
    name: str
    params: dict[str, Any]
    targets: tuple[str, ...]
    stability: StabilityMode
    seed: int = 137
    behavior_epoch: str = "2.4"


REGRESSION_CONTRACTS: tuple[TransformContract, ...] = (
    TransformContract("HorizontalFlip", {}, ("image", "mask"), "exact"),
    TransformContract("VerticalFlip", {}, ("image", "mask"), "exact"),
    TransformContract("Transpose", {}, ("image", "mask"), "exact"),
    TransformContract("RandomRotate90", {}, ("image", "mask"), "exact"),
)


def contract_by_name(transform_name: str) -> TransformContract:
    for contract in REGRESSION_CONTRACTS:
        if contract.name == transform_name:
            return contract
    msg = f"No regression contract registered for {transform_name}"
    raise KeyError(msg)


def public_transform_names() -> set[str]:
    return {transform.__name__ for transform in get_all_valid_transforms(use_cache=True)}


def registered_transform_names() -> set[str]:
    return {contract.name for contract in REGRESSION_CONTRACTS}


def planned_transform_reasons() -> dict[str, str]:
    reasons: dict[str, str] = {}
    for transform in get_all_valid_transforms(use_cache=True):
        transform_name = transform.__name__
        if transform_name in registered_transform_names():
            continue
        if TransformTestHelper.requires_metadata(transform):
            reasons[transform_name] = "planned structural coverage for metadata/reference-data transform"
        elif TransformTestHelper.requires_special_setup(transform):
            reasons[transform_name] = "planned special-input regression coverage"
        elif TransformTestHelper.is_rgb_only(transform):
            reasons[transform_name] = "planned RGB-only regression coverage"
        elif issubclass(transform, A.Transform3D):
            reasons[transform_name] = "planned volumetric regression coverage"
        elif issubclass(transform, A.BaseCompose):
            reasons[transform_name] = "covered through composition and serialization tests"
        else:
            reasons[transform_name] = "planned phase-2 or phase-3 regression coverage"
    return reasons


def unaccounted_public_transforms() -> set[str]:
    accounted = registered_transform_names() | set(planned_transform_reasons())
    return public_transform_names() - accounted
