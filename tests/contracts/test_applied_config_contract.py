import pytest

import albumentations as A
from albumentations.core.composition import _normalize_config_for_replay
from tests.helpers.applied_config import run_applied_config_contract
from tests.helpers.transform_cases import TRANSFORM_CONTRACT_CASES


@pytest.mark.parametrize("case", TRANSFORM_CONTRACT_CASES, ids=lambda case: case.case_id)
def test_applied_config_contract(case) -> None:
    for seed in case.seeds:
        run_applied_config_contract(case, seed)


def test_replay_normalization_keeps_valid_scalars_and_wraps_tuple_only_fields() -> None:
    config = {"fill": 0, "angle_range": 15}

    normalized = _normalize_config_for_replay(A.Rotate, config)

    assert normalized == {"fill": 0, "angle_range": (15, 15)}

    assert _normalize_config_for_replay(A.HistogramMatching, {"blend_ratio": 0.75}) == {
        "blend_ratio": (0.75, 0.75),
    }
