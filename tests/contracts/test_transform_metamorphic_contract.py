"""Cross-mode relations that reuse the canonical transform and target registries."""

import copy
from typing import Any

import numpy as np
import pytest

import albumentations as A
from tests.helpers.contract_assertions import assert_contract_values_equal
from tests.helpers.target_contracts import CORE_TARGET_CONTRACT_PAIRS, TargetContractPair

_CROP_AND_PAD_CHOICE_PAIRS = tuple(
    pair
    for pair in CORE_TARGET_CONTRACT_PAIRS
    if pair.case.transform_cls is A.CropAndPad
    and ("px_choices" in pair.case.init_kwargs or "percent_choices" in pair.case.init_kwargs)
)


def _fixed_and_singleton_choice_kwargs(pair: TargetContractPair) -> tuple[dict[str, Any], dict[str, Any]]:
    choice_parameter = "px_choices" if "px_choices" in pair.case.init_kwargs else "percent_choices"
    fixed_parameter = "px" if choice_parameter == "px_choices" else "percent"
    choices = pair.case.init_kwargs[choice_parameter]
    assert isinstance(choices, tuple) and choices
    value = choices[0]

    fixed_kwargs = copy.deepcopy(dict(pair.case.init_kwargs))
    fixed_kwargs.pop(choice_parameter)
    fixed_kwargs[fixed_parameter] = value

    singleton_kwargs = copy.deepcopy(dict(pair.case.init_kwargs))
    singleton_kwargs[choice_parameter] = (value,)
    return fixed_kwargs, singleton_kwargs


@pytest.mark.parametrize("pair", _CROP_AND_PAD_CHOICE_PAIRS, ids=lambda pair: pair.pair_id)
def test_crop_and_pad_singleton_choice_matches_the_fixed_amount_across_targets(pair: TargetContractPair) -> None:
    seed = 137
    fixed_kwargs, singleton_kwargs = _fixed_and_singleton_choice_kwargs(pair)
    source = pair.case.make_data(np.random.default_rng(seed), pair.profile.data_factory)
    fixed_data = copy.deepcopy(source)
    choice_data = copy.deepcopy(source)
    compose_kwargs = copy.deepcopy(dict(pair.profile.compose_kwargs))
    fixed_pipeline = A.Compose([A.CropAndPad(**fixed_kwargs, p=1)], seed=seed, strict=True, **compose_kwargs)
    choice_pipeline = A.Compose([A.CropAndPad(**singleton_kwargs, p=1)], seed=seed, strict=True, **compose_kwargs)

    fixed_result = fixed_pipeline(**fixed_data)
    choice_result = choice_pipeline(**choice_data)

    pair.profile.assert_result(pair.case, source, fixed_result)
    pair.profile.assert_result(pair.case, source, choice_result)
    assert_contract_values_equal(choice_result, fixed_result, pair.pair_id)
