"""Golden-vector regression checks."""

from __future__ import annotations

import pytest

from tests.regression.transform_contracts import unaccounted_public_transforms
from tools.verify_regression_vectors import verify_cases

pytestmark = pytest.mark.regression


def test_all_public_transforms_are_accounted_for() -> None:
    assert unaccounted_public_transforms() == set()


def test_golden_vectors_match_current_behavior() -> None:
    assert verify_cases() == []
