"""Every registered primary mode releases per-call arrays while its pipeline stays alive."""

from __future__ import annotations

import copy
import weakref
from typing import Any

import numpy as np
import pytest

import albumentations as A
from tests.helpers.lifetime import watch_array_references
from tests.helpers.transform_cases import PRIMARY_TRANSFORM_CONTRACT_CASES, TransformContractCase


@pytest.mark.usefixtures("refcount_only")
@pytest.mark.parametrize("case", PRIMARY_TRANSFORM_CONTRACT_CASES, ids=lambda case: case.case_id)
@pytest.mark.parametrize("invocation_seed", [None, 137])
def test_primary_transform_releases_per_call_arrays(case: TransformContractCase, invocation_seed: int | None) -> None:
    pipeline = A.Compose(
        [case.transform_cls(**copy.deepcopy(dict(case.init_kwargs)), p=1)],
        **copy.deepcopy(dict(case.primary_compose_kwargs)),
        seed=137,
        strict=True,
    )

    def call() -> list[weakref.ReferenceType[Any]]:
        data = case.make_data(np.random.default_rng(137))
        references: list[weakref.ReferenceType[Any]] = []
        watch_array_references(data, references)
        result = pipeline(**data, invocation_seed=invocation_seed)
        watch_array_references(result, references)
        return references

    for _ in range(3):
        references = call()
        assert references, f"{case.case_id}: workload must own an array"
        assert all(reference() is None for reference in references), f"{case.case_id}: retained per-call array"
