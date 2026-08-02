"""Reusable target workloads for shared transform contracts."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import albumentations as A
from tests.helpers.contract_data import (
    ContractDataFactory,
    make_target_empty_hbb_data,
    make_target_empty_keypoint_data,
    make_target_float_image_mask_data,
    make_target_grayscale_image_mask_data,
    make_target_hbb_data,
    make_target_image_batch_data,
    make_target_image_mask_data,
    make_target_keypoint_data,
    make_target_mask_batch_data,
    make_target_multispectral_image_mask_data,
    make_target_noncontiguous_image_mask_data,
    make_target_obb_data,
    make_target_readonly_image_mask_data,
    make_target_volume_data,
)

if TYPE_CHECKING:
    from tests.helpers.transform_cases import TransformContractCase

TargetAssertion = Callable[["TransformContractCase", dict[str, Any], dict[str, Any]], None]


class ProfileCost(Enum):
    """Execution tier for a reusable target workload."""

    CORE = auto()
    EXTENDED = auto()


def _assert_image_mask(
    case: TransformContractCase,
    source: dict[str, Any],
    result: dict[str, Any],
) -> None:
    image = result["image"]
    mask = result["mask"]
    assert image.shape[:2] == mask.shape[:2]
    assert image.dtype == source["image"].dtype
    assert mask.dtype == source["mask"].dtype


def _assert_bbox_fields(source: dict[str, Any], result: dict[str, Any], geometry_columns: int) -> None:
    bboxes = np.asarray(result["bboxes"])
    assert len(bboxes) == len(result["bbox_labels"])
    assert len(bboxes) == len(result["bbox_scores"])
    assert set(result["bbox_labels"]) <= _collect_field_values(source, "bbox_labels")
    assert set(result["bbox_scores"]) <= _collect_field_values(source, "bbox_scores")
    if len(bboxes) == 0:
        return
    assert bboxes.ndim == 2
    assert bboxes.shape[1] == geometry_columns
    assert np.isfinite(bboxes).all()


def _collect_field_values(value: Any, field_name: str) -> set[Any]:
    values: set[Any] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == field_name:
                if isinstance(item, dict):
                    pass
                elif isinstance(item, (list, tuple, np.ndarray)):
                    values.update(item)
                else:
                    values.add(item)
            values.update(_collect_field_values(item, field_name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            values.update(_collect_field_values(item, field_name))
    return values


def _assert_hbb(case: TransformContractCase, source: dict[str, Any], result: dict[str, Any]) -> None:
    _assert_image_mask(case, source, result)
    _assert_bbox_fields(source, result, geometry_columns=4)
    bboxes = np.asarray(result["bboxes"])
    if len(bboxes) > 0:
        assert ((bboxes >= 0) & (bboxes <= 1)).all()


def _assert_obb(case: TransformContractCase, source: dict[str, Any], result: dict[str, Any]) -> None:
    _assert_image_mask(case, source, result)
    _assert_bbox_fields(source, result, geometry_columns=5)
    bboxes = np.asarray(result["bboxes"])
    if len(bboxes) > 0:
        assert ((bboxes[:, :4] >= 0) & (bboxes[:, :4] <= 1)).all()


def _assert_keypoints(case: TransformContractCase, source: dict[str, Any], result: dict[str, Any]) -> None:
    _assert_image_mask(case, source, result)
    keypoints = np.asarray(result["keypoints"])
    assert keypoints.ndim == 2
    assert keypoints.shape[1] == 2
    assert len(keypoints) == len(result["keypoint_labels"])
    assert set(result["keypoint_labels"]) <= set(source["keypoint_labels"])
    assert np.isfinite(keypoints).all()


def _assert_volume_mask3d(
    case: TransformContractCase,
    source: dict[str, Any],
    result: dict[str, Any],
) -> None:
    volume = result["volume"]
    mask3d = result["mask3d"]
    if not issubclass(case.transform_cls, A.Transform3D):
        assert volume.shape[0] == source["volume"].shape[0]
        assert mask3d.shape[0] == source["mask3d"].shape[0]
    assert volume.shape[:3] == mask3d.shape[:3]
    assert volume.dtype == source["volume"].dtype
    assert mask3d.dtype == source["mask3d"].dtype


def _assert_images(case: TransformContractCase, source: dict[str, Any], result: dict[str, Any]) -> None:
    images = result["images"]
    assert images.ndim == 4
    assert images.shape[0] == source["images"].shape[0]
    assert images.dtype == source["images"].dtype


def _assert_masks(case: TransformContractCase, source: dict[str, Any], result: dict[str, Any]) -> None:
    masks = result["masks"]
    assert masks.ndim == 3
    assert masks.shape[0] > 0
    assert masks.dtype == source["masks"].dtype


@dataclass(frozen=True)
class TargetProfile:
    """One reusable input workload and its shared output assertions."""

    profile_id: str
    required_targets: frozenset[str]
    data_factory: ContractDataFactory
    assert_result: TargetAssertion
    compose_kwargs: Mapping[str, Any] = field(default_factory=dict)
    cost: ProfileCost = ProfileCost.CORE
    bbox_type: Literal["hbb", "obb"] | None = None
    empty_targets: frozenset[str] = frozenset()
    channel_count: int | None = 3

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", self.profile_id):
            raise ValueError(f"Invalid target profile_id: {self.profile_id!r}")
        object.__setattr__(self, "compose_kwargs", MappingProxyType(copy.deepcopy(dict(self.compose_kwargs))))


TARGET_CONTRACT_PROFILES = (
    TargetProfile(
        profile_id="image-mask",
        required_targets=frozenset({"image", "mask"}),
        data_factory=make_target_image_mask_data,
        assert_result=_assert_image_mask,
    ),
    TargetProfile(
        profile_id="hbb-labels",
        required_targets=frozenset({"image", "mask", "bboxes"}),
        data_factory=make_target_hbb_data,
        assert_result=_assert_hbb,
        compose_kwargs={
            "bbox_params": A.BboxParams(
                coord_format="albumentations",
                label_fields=["bbox_labels", "bbox_scores"],
            ),
        },
        bbox_type="hbb",
    ),
    TargetProfile(
        profile_id="obb-labels",
        required_targets=frozenset({"image", "mask", "bboxes"}),
        data_factory=make_target_obb_data,
        assert_result=_assert_obb,
        compose_kwargs={
            "bbox_params": A.BboxParams(
                coord_format="albumentations",
                bbox_type="obb",
                label_fields=["bbox_labels", "bbox_scores"],
            ),
        },
        bbox_type="obb",
    ),
    TargetProfile(
        profile_id="keypoints-labels",
        required_targets=frozenset({"image", "mask", "keypoints"}),
        data_factory=make_target_keypoint_data,
        assert_result=_assert_keypoints,
        compose_kwargs={
            "keypoint_params": A.KeypointParams(
                coord_format="xy",
                label_fields=["keypoint_labels"],
                label_mapping={},
            ),
        },
    ),
    TargetProfile(
        profile_id="volume-mask3d",
        required_targets=frozenset({"volume", "mask3d"}),
        data_factory=make_target_volume_data,
        assert_result=_assert_volume_mask3d,
        channel_count=3,
    ),
    TargetProfile(
        profile_id="float-image-mask",
        required_targets=frozenset({"image", "mask"}),
        data_factory=make_target_float_image_mask_data,
        assert_result=_assert_image_mask,
        cost=ProfileCost.EXTENDED,
    ),
    TargetProfile(
        profile_id="grayscale-image-mask",
        required_targets=frozenset({"image", "mask"}),
        data_factory=make_target_grayscale_image_mask_data,
        assert_result=_assert_image_mask,
        cost=ProfileCost.EXTENDED,
        channel_count=1,
    ),
    TargetProfile(
        profile_id="multispectral-image-mask",
        required_targets=frozenset({"image", "mask"}),
        data_factory=make_target_multispectral_image_mask_data,
        assert_result=_assert_image_mask,
        cost=ProfileCost.EXTENDED,
        channel_count=5,
    ),
    TargetProfile(
        profile_id="images-batch",
        required_targets=frozenset({"images"}),
        data_factory=make_target_image_batch_data,
        assert_result=_assert_images,
        cost=ProfileCost.EXTENDED,
    ),
    TargetProfile(
        profile_id="masks-batch",
        required_targets=frozenset({"masks"}),
        data_factory=make_target_mask_batch_data,
        assert_result=_assert_masks,
        cost=ProfileCost.EXTENDED,
    ),
    TargetProfile(
        profile_id="empty-hbb",
        required_targets=frozenset({"image", "mask", "bboxes"}),
        data_factory=make_target_empty_hbb_data,
        assert_result=_assert_hbb,
        compose_kwargs={
            "bbox_params": A.BboxParams(
                coord_format="albumentations",
                label_fields=["bbox_labels", "bbox_scores"],
            ),
        },
        cost=ProfileCost.EXTENDED,
        bbox_type="hbb",
        empty_targets=frozenset({"bboxes"}),
    ),
    TargetProfile(
        profile_id="empty-keypoints",
        required_targets=frozenset({"image", "mask", "keypoints"}),
        data_factory=make_target_empty_keypoint_data,
        assert_result=_assert_keypoints,
        compose_kwargs={
            "keypoint_params": A.KeypointParams(
                coord_format="xy",
                label_fields=["keypoint_labels"],
                label_mapping={},
            ),
        },
        cost=ProfileCost.EXTENDED,
        empty_targets=frozenset({"keypoints"}),
    ),
    TargetProfile(
        profile_id="noncontiguous-image-mask",
        required_targets=frozenset({"image", "mask"}),
        data_factory=make_target_noncontiguous_image_mask_data,
        assert_result=_assert_image_mask,
        cost=ProfileCost.EXTENDED,
    ),
    TargetProfile(
        profile_id="readonly-image-mask",
        required_targets=frozenset({"image", "mask"}),
        data_factory=make_target_readonly_image_mask_data,
        assert_result=_assert_image_mask,
        cost=ProfileCost.EXTENDED,
    ),
)

TARGET_PROFILES_BY_ID = {profile.profile_id: profile for profile in TARGET_CONTRACT_PROFILES}
