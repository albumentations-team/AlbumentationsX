"""Regression and coverage registry for public transform behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import albumentations as A
from tests.utils import get_all_valid_transforms, get_primary_public_transform_params

StabilityMode = Literal["exact", "tolerance", "digest", "structural"]
InputRecipe = Literal["image_mask", "hbb_keypoints", "volume_mask3d"]


@dataclass(frozen=True)
class TransformContract:
    name: str
    params: dict[str, Any]
    targets: tuple[str, ...]
    stability: StabilityMode
    seed: int = 137
    behavior_epoch: str = "2.4"
    input_recipe: InputRecipe = "image_mask"
    tolerance: float = 1e-5


REGRESSION_CONTRACTS: tuple[TransformContract, ...] = (
    TransformContract("HorizontalFlip", {}, ("image", "mask"), "exact"),
    TransformContract("VerticalFlip", {}, ("image", "mask"), "exact"),
    TransformContract("Transpose", {}, ("image", "mask"), "exact"),
    TransformContract("RandomRotate90", {}, ("image", "mask"), "exact"),
    TransformContract(
        "Resize",
        {"height": 16, "width": 20},
        ("bboxes", "bbox_labels", "keypoints", "keypoint_labels"),
        "tolerance",
        input_recipe="hbb_keypoints",
    ),
    TransformContract(
        "CenterCrop3D",
        {"size": (4, 6, 6)},
        ("volume", "mask3d"),
        "exact",
        input_recipe="volume_mask3d",
    ),
)

ABSTRACT_PUBLIC_APIS = frozenset(
    {
        "BasicTransform",
        "DualTransform",
        "ImageOnlyTransform",
        "Transform3D",
        "VolumeOnlyTransform",
    },
)
COMPOSITION_PUBLIC_APIS = frozenset(
    {
        "BaseCompose",
        "Compose",
        "OneOf",
        "OneOrOther",
        "RandomOrder",
        "ReplayCompose",
        "SelectiveChannelTransform",
        "Sequential",
        "SomeOf",
    },
)
PYTORCH_PUBLIC_APIS = frozenset({"ToTensor3D", "ToTensorV2"})
CUSTOM_TRANSFORM_PUBLIC_APIS = frozenset({"Lambda"})


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


def transform_sweep_names() -> set[str]:
    """Names exercised by the established parameterized transform sweeps."""
    return {
        transform.__name__
        for transform, _ in get_primary_public_transform_params(
            except_augmentations={
                A.Lambda,
            },
        )
    }


def public_transform_coverage_routes() -> dict[str, str]:
    """Return the primary test route for every public transform-like API.

    This is intentionally not a golden-vector coverage map. AlbumentationsX has
    mature parameterized tests that sweep transform construction, target routing,
    dtype/shape behavior, serialization, and mask/annotation semantics. Golden
    vectors are a small compatibility sentinel layer on top of those tests.
    """
    reasons: dict[str, str] = {}

    golden_names = registered_transform_names()
    swept_names = transform_sweep_names()
    for transform_name in public_transform_names():
        if transform_name in golden_names and transform_name in swept_names:
            reasons[transform_name] = "golden vector plus parameterized transform sweeps"
        elif transform_name in golden_names:
            reasons[transform_name] = "golden vector compatibility sentinel"
        elif transform_name in swept_names:
            reasons[transform_name] = "parameterized transform sweeps"
        elif transform_name in COMPOSITION_PUBLIC_APIS:
            reasons[transform_name] = "composition, operator, serialization, and replay tests"
        elif transform_name in PYTORCH_PUBLIC_APIS:
            reasons[transform_name] = "PyTorch target conversion tests"
        elif transform_name in CUSTOM_TRANSFORM_PUBLIC_APIS:
            reasons[transform_name] = "custom transform and serialization edge-case tests"
        elif transform_name in ABSTRACT_PUBLIC_APIS:
            reasons[transform_name] = "abstract public base API covered through subclasses and core tests"
    return reasons


def unaccounted_public_transforms() -> set[str]:
    accounted = set(public_transform_coverage_routes())
    return public_transform_names() - accounted


def coverage_summary() -> dict[str, int]:
    routes = public_transform_coverage_routes()
    return {
        "public_transform_apis": len(public_transform_names()),
        "covered_public_transform_apis": len(routes),
        "parameterized_transform_sweep": len(transform_sweep_names()),
        "golden_contracts": len(registered_transform_names()),
        "composition_public_apis": len(COMPOSITION_PUBLIC_APIS),
        "pytorch_public_apis": len(PYTORCH_PUBLIC_APIS),
        "custom_transform_public_apis": len(CUSTOM_TRANSFORM_PUBLIC_APIS),
        "abstract_public_apis": len(ABSTRACT_PUBLIC_APIS),
        "unaccounted_public_transform_apis": len(unaccounted_public_transforms()),
    }
