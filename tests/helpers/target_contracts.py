"""Public-path runner for reusable transform target contracts."""

from __future__ import annotations

import copy
import json
import warnings
from dataclasses import dataclass

import numpy as np

import albumentations as A
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
    if "mask3d" in declared_targets:
        declared_targets.add("masks3d")
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
