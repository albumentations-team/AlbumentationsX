import numpy as np
import pytest

import albumentations as A
from tests.helpers.target_contracts import (
    CORE_TARGET_CONTRACT_PAIRS,
    EXTENDED_TARGET_CONTRACT_PAIRS,
    TENSOR_TARGET_CONTRACT_PAIRS,
    TargetContractPair,
    run_target_cluster_contract,
    run_tensor_target_cluster_contract,
)
from tests.helpers.target_profiles import TARGET_PROFILES_BY_ID
from tests.helpers.transform_cases import ALL_DUAL_TRANSFORM_CONTRACT_CASES, PRIMARY_IMAGE_ONLY_TRANSFORM_CONTRACT_CASES


@pytest.mark.parametrize("pair", CORE_TARGET_CONTRACT_PAIRS, ids=lambda pair: pair.pair_id)
def test_core_target_cluster_contract(pair: TargetContractPair) -> None:
    for seed in pair.case.seeds:
        run_target_cluster_contract(pair.case, pair.profile, seed)


@pytest.mark.parametrize("pair", EXTENDED_TARGET_CONTRACT_PAIRS, ids=lambda pair: pair.pair_id)
def test_extended_target_cluster_contract(pair: TargetContractPair) -> None:
    run_target_cluster_contract(pair.case, pair.profile, seed=137)


@pytest.mark.parametrize("pair", TENSOR_TARGET_CONTRACT_PAIRS, ids=lambda pair: f"tensor-{pair.pair_id}")
def test_tensor_target_cluster_contract(pair: TargetContractPair) -> None:
    run_tensor_target_cluster_contract(pair.case, pair.profile, seed=137)


def test_target_contract_ids_are_unique() -> None:
    profile_ids = list(TARGET_PROFILES_BY_ID)
    pair_ids = [pair.pair_id for pair in (*CORE_TARGET_CONTRACT_PAIRS, *EXTENDED_TARGET_CONTRACT_PAIRS)]

    assert len(profile_ids) == len(set(profile_ids))
    assert len(pair_ids) == len(set(pair_ids))


def test_every_dual_transform_case_has_core_target_coverage() -> None:
    covered_case_ids = {pair.case.case_id for pair in CORE_TARGET_CONTRACT_PAIRS}
    missing = {case.case_id for case in ALL_DUAL_TRANSFORM_CONTRACT_CASES} - covered_case_ids

    assert not missing, f"DualTransform cases without core target coverage: {sorted(missing)}"


def test_every_primary_image_only_transform_case_has_tensor_target_coverage() -> None:
    image_only_case_ids = {case.case_id for case in PRIMARY_IMAGE_ONLY_TRANSFORM_CONTRACT_CASES}
    expected_pairs = {
        (case_id, profile_id) for case_id in image_only_case_ids for profile_id in ("image", "images-batch", "volume")
    }
    covered_pairs = {(pair.case.case_id, pair.profile.profile_id) for pair in TENSOR_TARGET_CONTRACT_PAIRS}
    missing = expected_pairs - covered_pairs

    assert not missing, f"ImageOnlyTransform Tensor target coverage is missing: {sorted(missing)}"


def test_every_target_profile_is_collected() -> None:
    collected_profile_ids = {
        pair.profile.profile_id for pair in (*CORE_TARGET_CONTRACT_PAIRS, *EXTENDED_TARGET_CONTRACT_PAIRS)
    }

    assert collected_profile_ids == set(TARGET_PROFILES_BY_ID)


@pytest.mark.parametrize("transform_cls", [A.Mosaic, A.CopyAndPaste])
def test_image_required_transforms_exclude_profiles_without_image(transform_cls: type[A.DualTransform]) -> None:
    cases = [case for case in ALL_DUAL_TRANSFORM_CONTRACT_CASES if case.transform_cls is transform_cls]
    assert cases
    assert all("image" in case.required_targets for case in cases)

    invalid_pairs = [
        pair.pair_id
        for pair in (*CORE_TARGET_CONTRACT_PAIRS, *EXTENDED_TARGET_CONTRACT_PAIRS)
        if pair.case.transform_cls is transform_cls and "image" not in pair.profile.required_targets
    ]
    assert not invalid_pairs


def test_volume_profile_rejects_mismatched_transformed_depth() -> None:
    case = next(case for case in ALL_DUAL_TRANSFORM_CONTRACT_CASES if case.transform_cls is A.Pad3D)
    profile = TARGET_PROFILES_BY_ID["volume-mask3d"]
    source = {
        "volume": np.zeros((2, 8, 12, 1), dtype=np.uint8),
        "mask3d": np.zeros((2, 8, 12), dtype=np.uint8),
    }
    result = {
        "volume": np.zeros((3, 8, 12, 1), dtype=np.uint8),
        "mask3d": np.zeros((4, 8, 12), dtype=np.uint8),
    }

    with pytest.raises(AssertionError):
        profile.assert_result(case, source, result)
