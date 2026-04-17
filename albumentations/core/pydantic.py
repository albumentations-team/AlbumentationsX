"""Pydantic validation utilities for Albumentations.

All validators here are tuple-only and reject scalar inputs. Scalar shorthand for
sampling-range parameters has been removed across the public API.
"""

from collections.abc import Callable
from typing import TypeVar

from albumentations.core.type_definitions import Number


def nondecreasing(value: tuple[Number, Number]) -> tuple[Number, Number]:
    """Ensure the tuple is non-decreasing (`value[0] <= value[1]`); raise `ValueError` otherwise.
    Used as a Pydantic `AfterValidator` on ordered ranges.

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
