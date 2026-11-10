"""Shared structural assertions for generated transform contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _array_difference(actual: Any, expected: np.ndarray, path: str) -> str | None:
    if not isinstance(actual, np.ndarray):
        return f"{path} changed type from ndarray to {type(actual).__name__}"
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        return f"{path} changed array metadata from {expected.dtype}/{expected.shape} to {actual.dtype}/{actual.shape}"
    arrays_equal = (
        np.array_equal(actual, expected, equal_nan=True)
        if actual.dtype.kind == "f"
        else np.array_equal(actual, expected)
    )
    if not arrays_equal:
        return f"{path} array contents changed"
    return None


def _tensor_difference(actual: Any, expected: torch.Tensor, path: str) -> str | None:
    if type(actual) is not torch.Tensor:
        return f"{path} changed type from Tensor to {type(actual).__name__}"
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        return f"{path} changed Tensor metadata from {expected.dtype}/{expected.shape} to {actual.dtype}/{actual.shape}"
    tensors_equal = (
        torch.allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
        if actual.dtype.is_floating_point
        else torch.equal(actual, expected)
    )
    if not tensors_equal:
        return f"{path} Tensor contents changed"
    return None


def _mapping_difference(actual: Any, expected: dict[Any, Any], path: str) -> str | None:
    if not isinstance(actual, dict):
        return f"{path} changed type from dict to {type(actual).__name__}"
    if actual.keys() != expected.keys():
        return f"{path} keys changed from {sorted(expected)} to {sorted(actual)}"
    for key, expected_value in expected.items():
        difference = find_contract_difference(actual[key], expected_value, f"{path}.{key}")
        if difference is not None:
            return difference
    return None


def _sequence_difference(actual: Any, expected: list[Any] | tuple[Any, ...], path: str) -> str | None:
    if not isinstance(actual, type(expected)):
        return f"{path} changed type from {type(expected).__name__} to {type(actual).__name__}"
    if len(actual) != len(expected):
        return f"{path} length changed from {len(expected)} to {len(actual)}"
    for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
        difference = find_contract_difference(actual_item, expected_item, f"{path}[{index}]")
        if difference is not None:
            return difference
    return None


def find_contract_difference(actual: Any, expected: Any, path: str = "value") -> str | None:
    """Return the first structural or value difference between transport-safe values."""
    if isinstance(expected, np.ndarray):
        return _array_difference(actual, expected, path)
    if isinstance(expected, torch.Tensor):
        return _tensor_difference(actual, expected, path)
    if isinstance(expected, dict):
        return _mapping_difference(actual, expected, path)
    if isinstance(expected, (list, tuple)):
        return _sequence_difference(actual, expected, path)
    return f"{path} changed from {expected!r} to {actual!r}" if actual != expected else None


def assert_contract_values_equal(actual: Any, expected: Any, path: str = "value", *, uint8_atol: int = 0) -> None:
    """Assert exact structure and values, with an explicitly requested uint8 rounding allowance."""
    if uint8_atol and isinstance(expected, np.ndarray) and expected.dtype == np.uint8:
        assert isinstance(actual, np.ndarray), f"{path} changed container"
        assert actual.shape == expected.shape, f"{path} changed shape"
        assert actual.dtype == expected.dtype, f"{path} changed dtype"
        delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        assert delta.max(initial=0) <= uint8_atol, f"{path} exceeds uint8 tolerance {uint8_atol}"
        return
    difference = find_contract_difference(actual, expected, path)
    assert difference is None, difference
