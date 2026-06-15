"""Golden-vector regression checks."""

from __future__ import annotations

import json

import pytest

from tests.regression.transform_contracts import registered_transform_names, unaccounted_public_transforms
from tools.generate_regression_vectors import MANIFEST_PATH
from tools.verify_regression_vectors import verify_cases

pytestmark = pytest.mark.regression


def test_public_transform_surface_has_test_coverage_route() -> None:
    assert unaccounted_public_transforms() == set()


def test_registered_golden_contracts_have_manifest_cases() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest_transform_names = {case["transform"] for case in manifest["cases"]}

    assert manifest_transform_names == registered_transform_names()


def test_golden_vectors_match_current_behavior() -> None:
    assert verify_cases() == []
