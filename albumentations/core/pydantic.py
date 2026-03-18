"""Module containing Pydantic validation utilities for Albumentations.
This module provides a collection of validators and utility functions used for validating
parameters in the Pydantic models throughout the Albumentations library. It includes
functions for ensuring numeric ranges are valid, handling type conversions, and creating
standardized validation patterns that are reused across the codebase.
"""

from collections.abc import Callable
from typing import TypeVar, overload

from albumentations.core.type_definitions import Number
from albumentations.core.utils import to_tuple


def nondecreasing(value: tuple[Number, Number]) -> tuple[Number, Number]:
    """Ensure a tuple of two numbers is in non-decreasing order (value[0] <= value[1]). Raises
    ValueError otherwise. Use as a Pydantic validator for range params.

    Args:
        value (tuple[Number, Number]): Tuple of two numeric values to validate.

    Returns:
        tuple[Number, Number]: The original tuple if valid.

    Raises:
        ValueError: If the first value is greater than the second value.

    """
    if not value[0] <= value[1]:
        raise ValueError(f"First value should be less than the second value, got {value} instead")
    return value


def process_non_negative_range(value: tuple[float, float] | float | None) -> tuple[float, float]:
    """Process and validate a non-negative range from a tuple, single float, or None (treated as 0).
    Raises if negative. For Pydantic.

    Args:
        value (tuple[float, float] | float | None): Value to process. Can be:
            - A tuple of two floats
            - A single float (converted to symmetric range)
            - None (defaults to 0)

    Returns:
        tuple[float, float]: Validated non-negative range.

    Raises:
        ValueError: If any values in the range are negative.

    """
    result = to_tuple(value if value is not None else 0, 0)
    if not all(x >= 0 for x in result):
        msg = "All values in the non negative range should be non negative"
        raise ValueError(msg)
    return result


def convert_to_1plus_int_range(
    value: tuple[int, int] | tuple[float, float] | float,
) -> tuple[int, int]:
    """Normalize int range with lower bound 1: scalar or tuple -> tuple[int, int].
    Accepts int or float input (e.g. from JSON); output is always (int, int).
    """
    result = to_tuple(value, low=1)
    return (int(result[0]), int(result[1]))


def process_non_negative_int_range(
    value: tuple[int, int] | tuple[float, float] | float | None,
) -> tuple[int, int]:
    """Normalize non-negative int range: scalar or tuple -> tuple[int, int].
    None treated as 0. Accepts int or float input; output is always (int, int).
    """
    result = to_tuple(value if value is not None else 0, 0)
    if not all(x >= 0 for x in result):
        raise ValueError("All values in the non negative range should be non negative")
    return (int(result[0]), int(result[1]))


@overload
def create_symmetric_range(value: tuple[int, int] | int) -> tuple[int, int]: ...


@overload
def create_symmetric_range(value: tuple[float, float] | float) -> tuple[float, float]: ...


def create_symmetric_range(value: tuple[float, float] | float) -> tuple[float, float]:
    """Create a symmetric range around zero: scalar x becomes (-x, x); tuple used as-is. Used for
    symmetric params in Pydantic.

    Args:
        value (tuple[float, float] | float): Input value, either:
            - A tuple of two floats (used directly)
            - A single float (converted to (-value, value))

    Returns:
        tuple[float, float]: Symmetric range.

    """
    return to_tuple(value)


def convert_to_1plus_range(value: tuple[float, float] | float) -> tuple[float, float]:
    """Convert value to a range with lower bound of 1 (via to_tuple with low=1). Scalar or tuple
    input. For Pydantic range fields.

    Args:
        value (tuple[float, float] | float): Input value.

    Returns:
        tuple[float, float]: Range with minimum value of at least 1.

    """
    return to_tuple(value, low=1)


def convert_to_0plus_range(value: tuple[float, float] | float) -> tuple[float, float]:
    """Convert value to a range with lower bound of 0 (via to_tuple with low=0). Scalar or tuple
    input. For Pydantic range fields.

    Args:
        value (tuple[float, float] | float): Input value.

    Returns:
        tuple[float, float]: Range with minimum value of at least 0.

    """
    return to_tuple(value, low=0)


def convert_to_1centered_range(value: tuple[float, float] | float) -> tuple[float, float]:
    """Convert value to a range centered at 1: scalar x gives (max(0, 1-x), 1+x). Used for
    brightness/contrast factors. For Pydantic.

    Args:
        value (tuple[float, float] | float): Input value. Scalar x becomes (max(0, 1-x), 1+x).
            Tuple is used as-is. Scalar must be non-negative.

    Returns:
        tuple[float, float]: Range suitable for brightness/contrast/saturation factors.

    """
    if isinstance(value, (int, float)) and value < 0:
        raise ValueError("If value is a single number, it must be non negative.")
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    result = to_tuple(value, bias=1)
    return (max(0.0, result[0]), result[1])


def repeat_if_scalar(value: tuple[float, float] | float) -> tuple[float, float]:
    """Convert a scalar to (value, value) or return the tuple unchanged. Used to normalize range
    params to (min, max). For Pydantic.

    Args:
        value (tuple[float, float] | float): Input value, either a scalar or tuple.

    Returns:
        tuple[float, float]: If input is scalar, returns (value, value), otherwise returns input unchanged.

    """
    return (value, value) if isinstance(value, (int, float)) else value


T = TypeVar("T", int, float)


def check_range_bounds(
    min_val: Number,
    max_val: Number | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> Callable[[tuple[T, ...] | None], tuple[T, ...] | None]:
    """Return a validator that ensures all values in a tuple are within min/max bounds
    (inclusive or exclusive). Use in Pydantic model field validators.

    Args:
        min_val (Number):
            Minimum allowed value.
        max_val (Number | None):
            Maximum allowed value. If None, only lower bound is checked.
        min_inclusive (bool):
            If True, min_val is inclusive (>=). If False, exclusive (>).
        max_inclusive (bool):
            If True, max_val is inclusive (<=). If False, exclusive (<).

    Returns:
        Callable[[tuple[T, ...] | None], tuple[T, ...] | None]: Validator function that
            checks if all values in tuple are within bounds. Returns None if input is None.

    Raises:
        ValueError: If any value in tuple is outside the allowed range

    Examples:
        >>> validator = check_range_bounds(0, 1)  # For [0, 1] range
        >>> validator((0.1, 0.5))  # Valid 2D
        (0.1, 0.5)
        >>> validator((0.1, 0.5, 0.7))  # Valid 3D
        (0.1, 0.5, 0.7)
        >>> validator((1.1, 0.5))  # Raises ValueError - outside range
        >>> validator = check_range_bounds(0, 1, max_inclusive=False)  # For [0, 1) range
        >>> validator((0, 1))  # Raises ValueError - 1 not included

    """

    def validator(value: tuple[T, ...] | None) -> tuple[T, ...] | None:
        if value is None:
            return None

        min_op = (lambda x, y: x >= y) if min_inclusive else (lambda x, y: x > y)
        max_op = (lambda x, y: x <= y) if max_inclusive else (lambda x, y: x < y)

        if max_val is None:
            if not all(min_op(x, min_val) for x in value):
                op_symbol = ">=" if min_inclusive else ">"
                raise ValueError(f"All values in {value} must be {op_symbol} {min_val}")
        else:
            min_symbol = ">=" if min_inclusive else ">"
            max_symbol = "<=" if max_inclusive else "<"
            if not all(min_op(x, min_val) and max_op(x, max_val) for x in value):
                raise ValueError(f"All values in {value} must be {min_symbol} {min_val} and {max_symbol} {max_val}")
        return value

    return validator


def convert_to_int_pair(value: tuple[int, int] | int) -> tuple[int, int]:
    """Convert an int to (value, value) or return a tuple unchanged.
    For Pydantic range fields that accept either a single int or (min, max) tuple.
    """
    return to_tuple(value, value)
