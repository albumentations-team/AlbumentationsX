from inspect import Parameter, signature
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError
from pydantic.functional_validators import AfterValidator

import albumentations as A
from albumentations.core.pydantic import (
    check_range_bounds,
    nondecreasing,
)
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.core.validation import ValidatedTransformMeta
from tests.utils import get_all_valid_transforms


class ValidationModel(BaseModel):
    non_negative_range_float: (
        Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0)),
            AfterValidator(nondecreasing),
        ]
        | None
    ) = None
    non_negative_range_int: (
        Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(0)),
            AfterValidator(nondecreasing),
        ]
        | None
    ) = None
    one_plus_range_float: (
        Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(1)),
        ]
        | None
    ) = None
    one_plus_range_int: (
        Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1)),
        ]
        | None
    ) = None
    zero_one_range: (
        Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        | None
    ) = None


@pytest.mark.parametrize("non_negative_range", [(0, 5), (10, 10), None])
def test_non_negative_range_valid(non_negative_range: tuple[int, int] | None) -> None:
    assert ValidationModel(non_negative_range_float=non_negative_range)
    assert ValidationModel(non_negative_range_int=non_negative_range)


@pytest.mark.parametrize(
    "non_negative_range",
    [(-1, 5), (-10, -1)],
)
def test_non_negative_range_invalid(non_negative_range: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        ValidationModel(non_negative_range_float=non_negative_range)

    with pytest.raises(ValidationError):
        ValidationModel(non_negative_range_int=non_negative_range)


@pytest.mark.parametrize("scalar_value", [5, 10, -1])
def test_non_negative_range_rejects_scalar(scalar_value: int) -> None:
    """Sampling-range validators must reject scalar inputs (tuple-only API)."""
    with pytest.raises(ValidationError):
        ValidationModel(non_negative_range_int=scalar_value)
    with pytest.raises(ValidationError):
        ValidationModel(non_negative_range_float=scalar_value)


@pytest.mark.parametrize(
    "value, expected",
    [
        ((0, 10), (0, 10)),
        ((1, 5), (1, 5)),
    ],
)
def test_non_negative_int_range_normalization(value, expected):
    model = ValidationModel(non_negative_range_int=value)
    assert model.non_negative_range_int == expected


@pytest.mark.parametrize(
    "value",
    [
        (-1, 5),
        (0, -5),
    ],
)
def test_non_negative_int_range_rejects_negative_values(value):
    with pytest.raises(ValidationError) as excinfo:
        ValidationModel(non_negative_range_int=value)
    msg = str(excinfo.value).lower()
    assert "must be >= 0" in msg


@pytest.mark.parametrize(
    "value, expected",
    [
        ((1, 3), (1, 3)),
        ((2, 5), (2, 5)),
        ((1, 1), (1, 1)),
    ],
)
def test_one_plus_int_range_normalization(value, expected):
    model = ValidationModel(one_plus_range_int=value)
    assert model.one_plus_range_int == expected


@pytest.mark.parametrize(
    "value",
    [
        (0, 5),
        (1, 0),
    ],
)
def test_one_plus_int_range_rejects_values_below_1(value):
    with pytest.raises(ValidationError):
        ValidationModel(one_plus_range_int=value)


@pytest.mark.parametrize("scalar", [3, 1, 0])
def test_one_plus_int_range_rejects_scalar(scalar: int) -> None:
    with pytest.raises(ValidationError):
        ValidationModel(one_plus_range_int=scalar)


@pytest.mark.parametrize("one_plus_range", [(0, 5), (0.5, 0.9)])
def test_one_plus_range_invalid(one_plus_range: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        ValidationModel(one_plus_range_float=one_plus_range)


@pytest.mark.parametrize("zero_one_range", [(-0.1, 0.5), (0.0, 1.1)])
def test_zero_one_range_invalid(zero_one_range: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        ValidationModel(zero_one_range=zero_one_range)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interpolation": 999, "size": (1, 1)},
        {"interpolation": -1, "size": (1, 1)},
        {"scale": (-4, 1), "size": (1, 1)},
        {"ratio": (-1, 2), "size": (1, 1)},
        {"size": (-1, 1)},
        {"size": (0, 1)},
    ],
)
def test_RandomResizedCrop(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        A.RandomResizedCrop(**kwargs)


class MyTransformInitSchema(BaseModel):
    param_a: int
    param_b: float = 1.0


class MyTransform(metaclass=ValidatedTransformMeta):
    class InitSchema(MyTransformInitSchema):
        pass

    def __init__(self, param_a: int, param_b: float = 1.0):
        self.param_a = param_a
        self.param_b = param_b


def test_my_transform_valid_initialization() -> None:
    transform = MyTransform(param_a=10, param_b=2.0)
    assert transform.param_a == 10
    assert transform.param_b == 2.0

    transform = MyTransform(param_a=5)
    assert transform.param_a == 5
    assert transform.param_b == 1.0


def test_my_transform_missing_required_param() -> None:
    with pytest.raises(ValueError):
        MyTransform()


@pytest.mark.parametrize(
    "invalid_a, invalid_b",
    [
        ("not an int", 2.0),
        (10, "not a float"),
    ],
)
def test_my_transform_invalid_types(invalid_a: int, invalid_b: float) -> None:
    with pytest.raises(ValueError):
        MyTransform(param_a=invalid_a, param_b=invalid_b)


class SimpleTransform(metaclass=ValidatedTransformMeta):
    def __init__(self, param_a: int, param_b: str = "default"):
        self.param_a = param_a
        self.param_b = param_b


def test_transform_without_schema() -> None:
    transform = SimpleTransform(param_a=10)
    assert transform.param_a == 10
    assert transform.param_b == "default", "Default parameter should be unchanged"

    transform = SimpleTransform(param_a=20, param_b="custom")
    assert transform.param_a == 20
    assert transform.param_b == "custom", "Custom parameter should be correctly assigned"

    transform = SimpleTransform(
        param_a="should not fail due to type annotations not enforcing type checks at runtime",
    )
    assert transform.param_a == "should not fail due to type annotations not enforcing type checks at runtime"


class CustomImageTransform(ImageOnlyTransform):
    def __init__(self, custom_param: int, p: float = 0.5):
        super().__init__(p=p)
        self.custom_param = custom_param


def test_custom_image_transform_signature() -> None:
    expected_signature = signature(CustomImageTransform)
    expected_params = expected_signature.parameters

    assert "custom_param" in expected_params
    assert expected_params["custom_param"] == Parameter(
        "custom_param",
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=int,
    )

    assert "p" in expected_params
    assert expected_params["p"] == Parameter(
        "p",
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=0.5,
        annotation=float,
    )

    assert expected_params["p"].default == 0.5
    assert expected_params["custom_param"].annotation is int


def test_check_range_bounds_doctest():
    validator = check_range_bounds(0, 1)
    assert validator((0.1, 0.5)) == (0.1, 0.5)
    assert validator((0.1, 0.5, 0.7)) == (0.1, 0.5, 0.7)

    with pytest.raises(ValueError):
        validator((1.1, 0.5))

    validator_exclusive = check_range_bounds(0, 1, max_inclusive=False)
    with pytest.raises(ValueError):
        validator_exclusive((0, 1))


@pytest.mark.parametrize(
    ["min_val", "max_val", "min_inclusive", "max_inclusive", "test_value", "expected"],
    [
        (0, 1, True, True, (0, 0.5, 1), (0, 0.5, 1)),
        (0, 1, False, False, (0.1, 0.5, 0.9), (0.1, 0.5, 0.9)),
        (0, None, True, True, None, None),
        (0, None, True, True, (1, 2, 3), (1, 2, 3)),
        (0, 1, True, True, (0.5,), (0.5,)),
        (0, 1, True, True, (0.2, 0.4), (0.2, 0.4)),
        (0, 1, True, True, (0.2, 0.4, 0.6, 0.8), (0.2, 0.4, 0.6, 0.8)),
        (0, 1, True, True, (0, 1), (0, 1)),
        (-1, 1, True, True, (-1, 0, 1), (-1, 0, 1)),
        (0.5, 1.5, True, True, (0.5, 1.0, 1.5), (0.5, 1.0, 1.5)),
    ],
)
def test_check_range_bounds_valid(min_val, max_val, min_inclusive, max_inclusive, test_value, expected):
    validator = check_range_bounds(min_val, max_val, min_inclusive, max_inclusive)
    assert validator(test_value) == expected


@pytest.mark.parametrize(
    ["min_val", "max_val", "min_inclusive", "max_inclusive", "test_value", "error_pattern"],
    [
        (0, 1, True, True, (-0.1, 0.5), "must be >= 0"),
        (0, 1, False, True, (0, 0.5), "must be > 0"),
        (0, 1, True, True, (0.5, 1.1), "must be >= 0 and <= 1"),
        (0, 1, True, False, (0.5, 1), "must be >= 0 and < 1"),
        (0, 1, True, True, (-0.1, 1.1), "must be >= 0 and <= 1"),
        (0, 1, False, False, (0, 1), "must be > 0 and < 1"),
        (0, None, True, True, (-1, -0.5), "must be >= 0"),
        (0, None, False, True, (0, 1), "must be > 0"),
    ],
)
def test_check_range_bounds_invalid(min_val, max_val, min_inclusive, max_inclusive, test_value, error_pattern):
    validator = check_range_bounds(min_val, max_val, min_inclusive, max_inclusive)
    with pytest.raises(ValueError, match=error_pattern):
        validator(test_value)


@pytest.mark.parametrize(
    ["min_val", "max_val", "values"],
    [
        (0, 1, [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]),
        (0, None, [(1, 2), (3, 4), (5, 6)]),
    ],
)
def test_check_range_bounds_multiple_calls(min_val, max_val, values):
    validator = check_range_bounds(min_val, max_val)
    for value in values:
        assert validator(value) == value


def test_check_range_bounds_type_preservation():
    validator = check_range_bounds(0, 1)

    int_tuple = (0, 1)
    assert isinstance(validator(int_tuple)[0], int)

    float_tuple = (0.5, 0.7)
    assert isinstance(validator(float_tuple)[0], float)


# ---------------------------------------------------------------------------
# API convention: any InitSchema field whose name ends with `_range` must be
# annotated as a two-number tuple (`int, int` or `float, float`), optionally
# wrapped in `| None` for runtime-defaulted fields. Scalar shorthand was
# dropped library-wide; this test prevents regressions.
# ---------------------------------------------------------------------------


def _unwrap(annotation: Any) -> Any:
    """Strip a single Annotated[...] layer if present."""
    if get_origin(annotation) is Annotated:
        return get_args(annotation)[0]
    # `Annotated` from `typing` shows up with `__metadata__` attr too
    if hasattr(annotation, "__origin__") and hasattr(annotation, "__metadata__"):
        return annotation.__origin__
    return annotation


def _arms(annotation: Any) -> list[Any]:
    """Return the union arms (excluding `None`) of an annotation; non-unions return `[ann]`."""
    annotation = _unwrap(annotation)
    if get_origin(annotation) in (Union, UnionType):
        return [_unwrap(a) for a in get_args(annotation) if a is not type(None)]
    return [annotation]


def _is_two_number_tuple(annotation: Any) -> bool:
    """Return True iff `annotation` is exactly `tuple[int, int]` or `tuple[float, float]`."""
    annotation = _unwrap(annotation)
    if get_origin(annotation) is not tuple:
        return False
    args = get_args(annotation)
    return len(args) == 2 and args[0] in (int, float) and args[0] is args[1]


def _is_valid_range_annotation(annotation: Any) -> bool:
    """Each union arm (after stripping `None`/`Annotated`) must be a two-number tuple."""
    arms = _arms(annotation)
    return bool(arms) and all(_is_two_number_tuple(a) for a in arms)


def _collect_range_fields() -> list[tuple[type, str, Any]]:
    """Yield `(transform_cls, field_name, annotation)` for every InitSchema field
    whose name ends with `_range`.
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[type, str, Any]] = []
    for cls in get_all_valid_transforms():
        schema = getattr(cls, "InitSchema", None)
        if schema is None or not hasattr(schema, "model_fields"):
            continue
        for name, field in schema.model_fields.items():
            if not name.endswith("_range"):
                continue
            key = (cls.__name__, name)
            if key in seen:
                continue
            seen.add(key)
            out.append((cls, name, field.annotation))
    return out


_RANGE_FIELDS = _collect_range_fields()


def test_range_fields_discovered() -> None:
    """Sanity check: we actually find some _range fields to validate."""
    assert len(_RANGE_FIELDS) > 20, f"expected to discover many _range fields, got {len(_RANGE_FIELDS)}"


@pytest.mark.parametrize(
    ("cls", "field_name", "annotation"),
    [pytest.param(c, n, a, id=f"{c.__name__}.{n}") for c, n, a in _RANGE_FIELDS],
)
def test_range_field_is_two_number_tuple(cls: type, field_name: str, annotation: Any) -> None:
    """Every InitSchema field ending in `_range` must be `tuple[int, int]` or
    `tuple[float, float]` (optionally wrapped in `| None` or unioned across int/float
    arms for parameters that accept either pixel or fraction inputs). Scalar shorthand
    is forbidden.
    """
    assert _is_valid_range_annotation(annotation), (
        f"{cls.__name__}.{field_name} must be typed as tuple[int, int] or tuple[float, float] "
        f"(optionally `| None`, optionally union of int/float tuple arms); got {annotation!r}"
    )
