"""Shared structural assertions for generated transform contracts."""

from __future__ import annotations

from typing import Any

import numpy as np


def find_contract_difference(actual: Any, expected: Any, path: str = "value") -> str | None:
    """Return the first structural or value difference between transport-safe values."""
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            return f"{path} changed type from ndarray to {type(actual).__name__}"
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            return (
                f"{path} changed array metadata from {expected.dtype}/{expected.shape} to {actual.dtype}/{actual.shape}"
            )
        if not np.array_equal(actual, expected):
            return f"{path} array contents changed"
        return None

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return f"{path} changed type from dict to {type(actual).__name__}"
        if actual.keys() != expected.keys():
            return f"{path} keys changed from {sorted(expected)} to {sorted(actual)}"
        for key in expected:
            difference = find_contract_difference(actual[key], expected[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None

    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)):
            return f"{path} changed type from {type(expected).__name__} to {type(actual).__name__}"
        if len(actual) != len(expected):
            return f"{path} length changed from {len(expected)} to {len(actual)}"
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            difference = find_contract_difference(actual_item, expected_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None

    if actual != expected:
        return f"{path} changed from {expected!r} to {actual!r}"
    return None


def assert_contract_values_equal(actual: Any, expected: Any, path: str = "value") -> None:
    """Assert exact structural equality and identify the first mismatching path."""
    difference = find_contract_difference(actual, expected, path)
    assert difference is None, difference
