"""Public-path runner for reusable transform target contracts."""

from __future__ import annotations

import copy
import json
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

import albumentations as A
from albumentations.core.tensor import (
    TENSOR_ANNOTATION_TARGETS,
    TENSOR_CHANNELLESS_RANKS,
    TENSOR_METADATA_FIELD_TARGETS,
    TENSOR_TARGETS,
    tensor_to_numpy_spatial,
)
from tests.helpers.applied_config import ReplayProfile
from tests.helpers.contract_assertions import assert_contract_values_equal
from tests.helpers.target_profiles import TARGET_CONTRACT_PROFILES, ProfileCost, TargetProfile
from tests.helpers.transform_cases import (
    PRIMARY_DUAL_TRANSFORM_CONTRACT_CASES,
    TRANSFORM_CONTRACT_CASES,
    TransformContractCase,
)


@dataclass(frozen=True)
class TargetContractPair:
    """One canonical transform mode paired with one applicable target profile."""

    case: TransformContractCase
    profile: TargetProfile

    @property
    def pair_id(self) -> str:
        return f"{self.case.case_id}--{self.profile.profile_id}"


def _supports_profile(case: TransformContractCase, profile: TargetProfile) -> bool:
    if not issubclass(case.transform_cls, A.DualTransform):
        return False
    if not case.required_targets <= profile.required_targets:
        return False
    if case.required_targets & profile.empty_targets:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transform = case.transform_cls(**copy.deepcopy(dict(case.init_kwargs)), p=1)
    raw_targets = transform._targets
    target_values = raw_targets if isinstance(raw_targets, tuple) else (raw_targets,)
    declared_targets = {target.name.lower() for target in target_values}
    if "image" in declared_targets:
        declared_targets.add("images")
    if "mask" in declared_targets:
        declared_targets.add("masks")
    if not profile.required_targets <= declared_targets:
        return False
    supported_channel_counts = getattr(transform, "_supported_channel_counts", None)
    if supported_channel_counts is not None and profile.channel_count not in supported_channel_counts:
        return False
    return profile.bbox_type is None or profile.bbox_type in transform._supported_bbox_types


CORE_TARGET_CONTRACT_PAIRS = tuple(
    TargetContractPair(case=case, profile=profile)
    for case in TRANSFORM_CONTRACT_CASES
    for profile in TARGET_CONTRACT_PROFILES
    if profile.cost is ProfileCost.CORE and _supports_profile(case, profile)
)

EXTENDED_TARGET_CONTRACT_PAIRS = tuple(
    TargetContractPair(case=case, profile=profile)
    for case in PRIMARY_DUAL_TRANSFORM_CONTRACT_CASES
    for profile in TARGET_CONTRACT_PROFILES
    if profile.cost is ProfileCost.EXTENDED and _supports_profile(case, profile)
)

_TENSOR_EXTENDED_PROFILE_IDS = frozenset(
    {
        "float-image-mask",
        "grayscale-image-mask",
        "multispectral-image-mask",
        "images-batch",
        "masks-batch",
        "noncontiguous-image-mask",
    },
)
TENSOR_TARGET_CONTRACT_PAIRS = (
    *CORE_TARGET_CONTRACT_PAIRS,
    *(pair for pair in EXTENDED_TARGET_CONTRACT_PAIRS if pair.profile.profile_id in _TENSOR_EXTENDED_PROFILE_IDS),
)


def _numpy_target_to_tensor(value: Any, target: str | None) -> Any:
    if not isinstance(value, np.ndarray):
        return value
    if target is None or target in TENSOR_ANNOTATION_TARGETS:
        return torch.from_numpy(value)
    if target in {"image", "mask"}:
        return torch.from_numpy(value if value.ndim == 2 else np.moveaxis(value, -1, 0))
    if target in {"images", "masks"}:
        return torch.from_numpy(value if value.ndim == 3 else np.moveaxis(value, -1, 1))
    if target in {"volume", "mask3d"}:
        return torch.from_numpy(value if value.ndim == 3 else np.moveaxis(value, -1, 0))
    return value


def _tensor_target_to_numpy(value: Any, target: str | None) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    if target is None or target in TENSOR_ANNOTATION_TARGETS:
        return value.numpy()
    if value.ndim == TENSOR_CHANNELLESS_RANKS.get(target):
        return value.numpy()
    return tensor_to_numpy_spatial(value, target)


def _convert_metadata(
    value: Any,
    target: str | None,
    convert: Callable[[Any, str | None], Any],
) -> Any:
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return convert(value, target)
    if isinstance(value, Mapping):
        converted = dict(value)
        for key, nested in value.items():
            nested_target = TENSOR_METADATA_FIELD_TARGETS.get(key)
            if key in TENSOR_METADATA_FIELD_TARGETS or isinstance(nested, (np.ndarray, torch.Tensor)):
                converted[key] = _convert_metadata(nested, nested_target, convert)
        return converted
    if isinstance(value, list):
        return [_convert_metadata_item(item, target, convert) for item in value]
    if isinstance(value, tuple):
        return tuple(_convert_metadata_item(item, target, convert) for item in value)
    return value


def _convert_metadata_item(
    value: Any,
    target: str | None,
    convert: Callable[[Any, str | None], Any],
) -> Any:
    if target is not None or isinstance(value, Mapping):
        return _convert_metadata(value, target, convert)
    if isinstance(value, (np.ndarray, torch.Tensor)):
        sequence_target = "image" if value.ndim in {2, 3} else None
        return _convert_metadata(value, sequence_target, convert)
    return value


def _convert_contract_data(
    data: dict[str, Any],
    transform: A.BasicTransform,
    convert: Callable[[Any, str | None], Any],
) -> dict[str, Any]:
    converted = copy.deepcopy(data)
    for data_name, value in data.items():
        if data_name in transform.get_tensor_metadata_keys():
            converted[data_name] = _convert_metadata(value, None, convert)
            continue
        canonical_target = transform._additional_targets.get(data_name, data_name)
        if canonical_target in TENSOR_TARGETS:
            converted[data_name] = convert(value, canonical_target)
    return converted


def _assert_tensor_result_containers(
    source: dict[str, Any],
    result: dict[str, Any],
    transform: A.BasicTransform,
) -> None:
    metadata_keys = transform.get_tensor_metadata_keys()
    for data_name, value in source.items():
        canonical_target = transform._additional_targets.get(data_name, data_name)
        if canonical_target in TENSOR_TARGETS and isinstance(value, torch.Tensor):
            assert isinstance(result[data_name], torch.Tensor)
        elif data_name in metadata_keys:
            assert result[data_name] is value


def run_tensor_target_cluster_contract(case: TransformContractCase, profile: TargetProfile, seed: int) -> None:
    """Replay one generated NumPy case with equivalent public Tensor targets."""
    source = case.make_data(np.random.default_rng(seed), profile.data_factory)
    numpy_data = copy.deepcopy(source)
    transform = case.transform_cls(**copy.deepcopy(dict(case.init_kwargs)), p=1)
    tensor_data = _convert_contract_data(
        source,
        transform,
        _numpy_target_to_tensor,
    )
    tensor_snapshot = copy.deepcopy(tensor_data)
    compose_kwargs = copy.deepcopy(dict(profile.compose_kwargs))
    numpy_pipeline = A.ReplayCompose(
        [case.transform_cls(**copy.deepcopy(dict(case.init_kwargs)), p=1)],
        **compose_kwargs,
    )
    numpy_pipeline.set_random_seed(seed)

    numpy_result = numpy_pipeline(**numpy_data)
    tensor_result = A.ReplayCompose.replay(numpy_result["replay"], **tensor_data)

    assert_contract_values_equal(tensor_data, tensor_snapshot, "tensor_input")
    _assert_tensor_result_containers(tensor_data, tensor_result, transform)
    profile.assert_result(case, tensor_snapshot, tensor_result)

    comparable_tensor_result = _convert_contract_data(
        tensor_result,
        transform,
        _tensor_target_to_numpy,
    )
    for key in source:
        expected = numpy_result[key]
        canonical_target = transform._additional_targets.get(key, key)
        if canonical_target in TENSOR_ANNOTATION_TARGETS and isinstance(expected, np.ndarray):
            expected = expected.astype(np.float32, copy=False)
        assert_contract_values_equal(comparable_tensor_result[key], expected, key)


def run_target_cluster_contract(case: TransformContractCase, profile: TargetProfile, seed: int) -> None:
    """Execute one registered mode against one reusable target workload."""
    source_data = case.make_data(np.random.default_rng(seed), profile.data_factory)
    replay_data = case.make_data(np.random.default_rng(seed), profile.data_factory)
    source_snapshot = copy.deepcopy(source_data)
    replay_snapshot = copy.deepcopy(replay_data)
    compose_kwargs = copy.deepcopy(dict(profile.compose_kwargs))
    transform = case.transform_cls(**copy.deepcopy(dict(case.init_kwargs)), p=1)
    pipeline = A.Compose(
        [transform],
        save_applied_params=True,
        seed=seed,
        strict=True,
        **compose_kwargs,
    )

    result = pipeline(**source_data)
    assert_contract_values_equal(source_data, source_snapshot, "input")
    profile.assert_result(case, source_snapshot, result)

    transported_record = json.loads(json.dumps(result["applied_transforms"], allow_nan=False))
    replay = A.Compose.from_applied_transforms(transported_record, seed=seed, **compose_kwargs)
    replay_result = replay(**replay_data)
    assert_contract_values_equal(replay_data, replay_snapshot, "replay_input")
    profile.assert_result(case, replay_snapshot, replay_result)

    if case.replay_profile is ReplayProfile.EXACT:
        for key in source_snapshot:
            assert_contract_values_equal(replay_result[key], result[key], key)

    replay_compose_data = case.make_data(np.random.default_rng(seed), profile.data_factory)
    replay_compose_snapshot = copy.deepcopy(replay_compose_data)
    replay_transform = case.transform_cls(**copy.deepcopy(dict(case.init_kwargs)), p=1)
    replay_pipeline = A.ReplayCompose([replay_transform], **copy.deepcopy(dict(profile.compose_kwargs)))
    replay_pipeline.set_random_seed(seed)
    replay_compose_result = replay_pipeline(**replay_compose_data)
    assert_contract_values_equal(replay_compose_data, replay_compose_snapshot, "replay_compose_input")
    replayed_result = A.ReplayCompose.replay(
        replay_compose_result["replay"],
        **case.make_data(np.random.default_rng(seed), profile.data_factory),
    )
    for key in replay_compose_snapshot:
        assert_contract_values_equal(replayed_result[key], replay_compose_result[key], key)
