"""Bounded generative checks for high-risk replay normalization and mode interactions."""

import json

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import albumentations as A
from albumentations.core.composition import _normalize_config_for_replay


@pytest.mark.property
@given(sample=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False))
def test_replay_normalization_is_idempotent(sample: float) -> None:
    config = {"angle_range": sample, "fill": sample}

    once = _normalize_config_for_replay(A.Rotate, config)
    twice = _normalize_config_for_replay(A.Rotate, once)

    assert twice == once
    assert once["fill"] == sample


@pytest.mark.property
@given(
    low=st.integers(min_value=3, max_value=11).filter(lambda value: value % 2 == 1),
    width=st.integers(min_value=0, max_value=3),
)
def test_json_lists_and_sampled_scalars_normalize_to_constructor_valid_ranges(low: int, width: int) -> None:
    high = low + 2 * width
    transform = A.Blur(blur_range=(low, high), p=1.0)
    result = A.Compose([transform], save_applied_params=True, seed=137)(
        image=np.zeros((16, 16, 3), dtype=np.uint8),
    )
    transported = json.loads(json.dumps(result["applied_transforms"]))

    replay = A.Compose.from_applied_transforms(transported)

    replay(image=np.zeros((16, 16, 3), dtype=np.uint8))
