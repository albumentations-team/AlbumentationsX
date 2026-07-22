import json

import numpy as np
import pytest

import albumentations as A
from albumentations.core.composition import _normalize_config_for_replay
from tests.helpers import TestDataFactory
from tests.helpers.applied_config import run_applied_config_contract
from tests.helpers.transform_cases import TRANSFORM_CONTRACT_CASES


@pytest.mark.parametrize("case", TRANSFORM_CONTRACT_CASES, ids=lambda case: case.case_id)
def test_applied_config_contract(case) -> None:
    for seed in case.seeds:
        run_applied_config_contract(case, seed)


def test_random_rotate90_subset_applied_record_reconstructs_exactly() -> None:
    image = TestDataFactory.create_image((64, 64, 3), seed=137)
    pipeline = A.Compose(
        [A.RandomRotate90(group_elements=("r90", "r270"), p=1.0, strict=True)],
        save_applied_params=True,
        seed=137,
        strict=True,
    )

    original_result = pipeline(image=image.copy())
    transported_record = json.loads(json.dumps(original_result["applied_transforms"]))

    replay = A.Compose.from_applied_transforms(transported_record)
    replay_result = replay(image=image.copy())

    np.testing.assert_array_equal(replay_result["image"], original_result["image"])


def test_replay_normalization_keeps_valid_scalars_and_wraps_tuple_only_fields() -> None:
    config = {"fill": 0, "angle_range": 15}

    normalized = _normalize_config_for_replay(A.Rotate, config)

    assert normalized == {"fill": 0, "angle_range": (15, 15)}

    assert _normalize_config_for_replay(A.HistogramMatching, {"blend_ratio": 0.75}) == {
        "blend_ratio": (0.75, 0.75),
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"group_element": "r90", "group_elements": ("r90", "r270")}, "mutually exclusive"),
        ({"group_elements": ()}, "non-empty"),
        ({"group_elements": ("r90", "r90")}, "unique"),
        ({"group_elements": ("r90", "invalid")}, "Input should be"),
    ],
)
def test_random_rotate90_rejects_invalid_group_subsets(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        A.RandomRotate90(**kwargs)


def test_random_rotate90_subset_serialization_roundtrip() -> None:
    transform = A.RandomRotate90(group_elements=("r90", "r270"), p=0.75)

    restored = A.from_dict(A.to_dict(transform))

    assert isinstance(restored, A.RandomRotate90)
    assert restored.group_elements == ("r90", "r270")
    assert restored.group_element is None
    assert restored.p == 0.75


def test_random_rotate90_subset_cannot_be_inverted_before_sampling() -> None:
    with pytest.raises(ValueError, match="group_element"):
        A.RandomRotate90(group_elements=("r90", "r270")).inverse()
