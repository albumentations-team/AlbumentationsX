"""Public-path runner for reusable transform target contracts."""

from __future__ import annotations

import copy
import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import torch

import albumentations as A
from albumentations.core.tensor import (
    TENSOR_ANNOTATION_TARGETS,
    TENSOR_CANONICAL_RANKS,
    TENSOR_CHANNELLESS_RANKS,
    TENSOR_SPATIAL_TARGETS,
    TENSOR_TARGETS,
    numpy_to_tensor_spatial,
    tensor_to_numpy_spatial,
)
from tests.helpers.applied_config import ReplayProfile
from tests.helpers.contract_assertions import assert_contract_values_equal
from tests.helpers.contract_data import ContractDataFactory
from tests.helpers.target_profiles import (
    IMAGE_ONLY_TENSOR_TARGET_PROFILES,
    TARGET_CONTRACT_PROFILES,
    VOLUME_ONLY_TARGET_PROFILES,
    ProfileCost,
    TargetProfile,
)
from tests.helpers.transform_cases import (
    PRIMARY_DUAL_TRANSFORM_CONTRACT_CASES,
    PRIMARY_IMAGE_ONLY_TRANSFORM_CONTRACT_CASES,
    PRIMARY_TRANSFORM_CONTRACT_CASES,
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


def _supports_profile(
    case: TransformContractCase,
    profile: TargetProfile,
    transform_base: type[A.BasicTransform] = A.DualTransform,
) -> bool:
    if not issubclass(case.transform_cls, transform_base):
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

TENSOR_IMAGE_ONLY_TARGET_CONTRACT_PAIRS = tuple(
    TargetContractPair(case=case, profile=profile)
    for case in PRIMARY_IMAGE_ONLY_TRANSFORM_CONTRACT_CASES
    for profile in IMAGE_ONLY_TENSOR_TARGET_PROFILES
    if _supports_profile(case, profile, A.ImageOnlyTransform)
)

VOLUME_ONLY_TARGET_CONTRACT_PAIRS = tuple(
    TargetContractPair(case=case, profile=profile)
    for case in PRIMARY_TRANSFORM_CONTRACT_CASES
    for profile in VOLUME_ONLY_TARGET_PROFILES
    if _supports_profile(case, profile, A.VolumeOnlyTransform)
)


_TENSOR_EXTENDED_PROFILE_IDS = frozenset(
    {
        "float-image-mask",
        "grayscale-image-mask",
        "multispectral-image-mask",
        "images-batch",
        "masks-batch",
        "volumes-masks3d",
        "noncontiguous-image-mask",
    },
)
TENSOR_TARGET_CONTRACT_PAIRS = (
    *CORE_TARGET_CONTRACT_PAIRS,
    *(pair for pair in EXTENDED_TARGET_CONTRACT_PAIRS if pair.profile.profile_id in _TENSOR_EXTENDED_PROFILE_IDS),
    *TENSOR_IMAGE_ONLY_TARGET_CONTRACT_PAIRS,
    *VOLUME_ONLY_TARGET_CONTRACT_PAIRS,
)


def _numpy_target_to_tensor(value: Any, target: str | None) -> Any:
    if not isinstance(value, np.ndarray):
        return value
    if target is None or target in TENSOR_ANNOTATION_TARGETS:
        return torch.from_numpy(value)
    if target not in TENSOR_SPATIAL_TARGETS:
        return value
    if value.ndim < TENSOR_CANONICAL_RANKS[target]:
        if target != "volumes":
            return torch.from_numpy(value)
        value = value[..., None]
    return numpy_to_tensor_spatial(value, target)


def _tensor_target_to_numpy(value: Any, target: str | None) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    if target is None or target in TENSOR_ANNOTATION_TARGETS:
        return value.numpy()
    if value.ndim == TENSOR_CHANNELLESS_RANKS.get(target):
        return value.numpy()
    return tensor_to_numpy_spatial(value, target)


def _convert_contract_data(
    data: dict[str, Any],
    transform: A.BasicTransform,
    convert: Callable[[Any, str | None], Any],
) -> dict[str, Any]:
    converted = copy.deepcopy(data)
    for data_name, value in data.items():
        canonical_target = transform._additional_targets.get(data_name, data_name)
        if canonical_target in TENSOR_TARGETS:
            converted[data_name] = convert(value, canonical_target)
    return converted


def _assert_tensor_result_containers(
    source: dict[str, Any],
    result: dict[str, Any],
    transform: A.BasicTransform,
) -> None:
    for data_name, value in source.items():
        canonical_target = transform._additional_targets.get(data_name, data_name)
        if canonical_target in TENSOR_TARGETS and isinstance(value, torch.Tensor):
            assert isinstance(result[data_name], torch.Tensor)


IMAGE_TARGET_NAMES = frozenset({"image", "images", "volume", "volumes"})


def _floating_image_data(
    factory: ContractDataFactory, rng: np.random.Generator, dtype: np.dtype[np.float16]
) -> dict[str, Any]:
    data = factory(rng)
    for target in IMAGE_TARGET_NAMES & data.keys():
        image = data[target]
        floating = image.astype(np.float32) / np.float32(255) if image.dtype == np.uint8 else image
        if not image.flags.c_contiguous and not image.flags.f_contiguous:
            backing = np.empty((*image.shape, 2), dtype=dtype)
            converted = backing[..., 0]
            np.copyto(converted, floating, casting="unsafe")
        else:
            converted = floating.astype(dtype, order="K", copy=True)
        if not image.flags.writeable:
            converted.flags.writeable = False
        data[target] = converted
    return data


def _assert_image_dtype(
    case: TransformContractCase,
    source: dict[str, Any],
    result: dict[str, Any],
    dtype: np.dtype[np.float16] | None,
) -> None:
    if dtype is None:
        return
    expected = np.dtype(case.init_kwargs.get("dtype", dtype))
    for target in IMAGE_TARGET_NAMES & source.keys():
        input_image = source[target]
        source_dtype = input_image.numpy().dtype if isinstance(input_image, torch.Tensor) else input_image.dtype
        assert source_dtype == dtype, f"{case.case_id}: {target} input has {source_dtype}, expected {dtype}"
        image = result[target]
        observed = image.numpy().dtype if isinstance(image, torch.Tensor) else image.dtype
        assert observed == expected, f"{case.case_id}: {target} returned {observed}, expected {expected}"


def make_target_contract_data(
    case: TransformContractCase,
    profile: TargetProfile,
    rng: np.random.Generator,
    *,
    image_dtype: np.dtype[np.float16] | None = None,
) -> dict[str, Any]:
    """Build profile targets and transform-specific context for one contract execution."""
    if profile.primary_data_adapter is not None:
        data_factory = partial(profile.primary_data_adapter, case.primary_data_factory)
    elif profile.data_factory is not None:
        data_factory = profile.data_factory
    else:
        raise ValueError(f"{profile.profile_id}: missing data factory")
    if image_dtype is not None:
        data_factory = partial(_floating_image_data, data_factory, dtype=image_dtype)
    return case.make_data(rng, data_factory)


def run_tensor_target_cluster_contract(
    case: TransformContractCase,
    profile: TargetProfile,
    seed: int,
    *,
    image_dtype: np.dtype[np.float16] | None = None,
) -> None:
    """Replay one generated NumPy case with equivalent public Tensor targets."""
    source = make_target_contract_data(case, profile, np.random.default_rng(seed), image_dtype=image_dtype)
    numpy_data = copy.deepcopy(source)
    transform = case.transform_cls(**copy.deepcopy(dict(case.init_kwargs)), p=1)
    tensor_data = _convert_contract_data(source, transform, _numpy_target_to_tensor)
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
    _assert_image_dtype(case, numpy_data, numpy_result, image_dtype)
    _assert_image_dtype(case, tensor_snapshot, tensor_result, image_dtype)

    comparable_tensor_result = _convert_contract_data(tensor_result, transform, _tensor_target_to_numpy)
    for key in source:
        expected = numpy_result[key]
        canonical_target = transform._additional_targets.get(key, key)
        if canonical_target in TENSOR_ANNOTATION_TARGETS and isinstance(expected, np.ndarray):
            expected = expected.astype(np.float32, copy=False)
        assert_contract_values_equal(
            comparable_tensor_result[key],
            expected,
            key,
            uint8_atol=case.tensor_uint8_atol if canonical_target in IMAGE_TARGET_NAMES else 0,
        )


def run_target_cluster_contract(
    case: TransformContractCase,
    profile: TargetProfile,
    seed: int,
    *,
    image_dtype: np.dtype[np.float16] | None = None,
) -> None:
    """Execute one registered mode against one reusable target workload."""
    source_data = make_target_contract_data(case, profile, np.random.default_rng(seed), image_dtype=image_dtype)
    replay_data = make_target_contract_data(case, profile, np.random.default_rng(seed), image_dtype=image_dtype)
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
    _assert_image_dtype(case, source_snapshot, result, image_dtype)

    transported_record = json.loads(json.dumps(result["applied_transforms"], allow_nan=False))
    replay = A.Compose.from_applied_transforms(transported_record, seed=seed, **compose_kwargs)
    replay_result = replay(**replay_data)
    assert_contract_values_equal(replay_data, replay_snapshot, "replay_input")
    profile.assert_result(case, replay_snapshot, replay_result)
    _assert_image_dtype(case, replay_snapshot, replay_result, image_dtype)

    if case.replay_profile is ReplayProfile.EXACT:
        for key in source_snapshot:
            assert_contract_values_equal(replay_result[key], result[key], key)

    replay_compose_data = make_target_contract_data(case, profile, np.random.default_rng(seed), image_dtype=image_dtype)
    replay_compose_snapshot = copy.deepcopy(replay_compose_data)
    replay_transform = case.transform_cls(**copy.deepcopy(dict(case.init_kwargs)), p=1)
    replay_pipeline = A.ReplayCompose([replay_transform], **copy.deepcopy(dict(profile.compose_kwargs)))
    replay_pipeline.set_random_seed(seed)
    replay_compose_result = replay_pipeline(**replay_compose_data)
    assert_contract_values_equal(replay_compose_data, replay_compose_snapshot, "replay_compose_input")
    replayed_result = A.ReplayCompose.replay(
        replay_compose_result["replay"],
        **make_target_contract_data(case, profile, np.random.default_rng(seed), image_dtype=image_dtype),
    )
    for key in replay_compose_snapshot:
        assert_contract_values_equal(replayed_result[key], replay_compose_result[key], key)
