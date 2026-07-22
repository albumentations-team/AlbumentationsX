import json

import numpy as np
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


@pytest.mark.parametrize(
    ("transform_cls", "kwargs"),
    [
        (A.TimeMasking, {"time_mask_param": 8}),
        (A.FrequencyMasking, {"freq_mask_param": 8}),
    ],
)
def test_spectrogram_alias_applied_replay_preserves_mask_fill(
    transform_cls: type[A.BasicTransform],
    kwargs: dict[str, int],
) -> None:
    image = np.ones((16, 16, 1), dtype=np.uint8)
    mask = np.ones_like(image)
    with pytest.warns(UserWarning):
        original = A.Compose(
            [transform_cls(p=1, **kwargs)],
            save_applied_params=True,
            seed=137,
        )

    original_result = original(image=image.copy(), mask=mask.copy())
    record = json.loads(json.dumps(original_result["applied_transforms"], allow_nan=False))
    replay = A.Compose.from_applied_transforms(record, seed=137)
    replay_result = replay(image=image.copy(), mask=mask.copy())

    assert record[0][0] == "XYMasking"
    assert record[0][1]["fill_mask"] == 0
    np.testing.assert_array_equal(replay_result["image"], original_result["image"])
    np.testing.assert_array_equal(replay_result["mask"], original_result["mask"])
