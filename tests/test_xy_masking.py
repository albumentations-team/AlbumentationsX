import itertools
import json
import warnings
from collections import deque
from collections.abc import Callable, Iterator
from decimal import Decimal
from enum import IntEnum
from fractions import Fraction
from numbers import Real

import numpy as np
import pytest

import albumentations as A


class _Length(IntEnum):
    ZERO = 0
    THREE = 3


class _CustomReal:
    def __float__(self) -> float:
        return 0.5


Real.register(_CustomReal)


def _sample_holes(transform: A.XYMasking, shape: tuple[int, int, int], seed: int = 137) -> np.ndarray:
    pipeline = A.Compose([transform], seed=seed, strict=True)
    pipeline(image=np.ones(shape, dtype=np.uint8))
    return transform.get_applied_params()["holes"]


def _assert_canonical_empty_holes(holes: np.ndarray) -> None:
    assert holes.shape == (0, 4)
    assert holes.dtype == np.int32


def test_relative_ranges_scale_x_by_width_and_y_by_height() -> None:
    holes = _sample_holes(
        A.XYMasking(
            num_masks_x_range=(1, 1),
            num_masks_y_range=(1, 1),
            mask_x_length_range=(0.5, 0.5),
            mask_y_length_range=(0.5, 0.5),
            p=1,
        ),
        (7, 11, 1),
    )

    np.testing.assert_array_equal(holes[:, 2] - holes[:, 0], [5, 11])
    np.testing.assert_array_equal(holes[:, 3] - holes[:, 1], [7, 3])


@pytest.mark.parametrize(
    ("fraction", "width", "expected_length"),
    [
        (0.5, 9, 4),
        (0.1, 5, 0),
        (0.58, 100, 57),
        (1.0, 9, 9),
    ],
)
def test_relative_x_range_floor_zero_and_full_axis(
    fraction: float,
    width: int,
    expected_length: int,
) -> None:
    holes = _sample_holes(
        A.XYMasking(
            num_masks_x_range=(1, 1),
            mask_x_length_range=(fraction, fraction),
            p=1,
        ),
        (7, width, 1),
    )

    if expected_length == 0:
        _assert_canonical_empty_holes(holes)
        return

    assert holes.shape == (1, 4)
    x_min, y_min, x_max, y_max = holes[0]
    assert x_max - x_min == expected_length
    assert 0 <= x_min <= x_max <= width
    assert (y_min, y_max) == (0, 7)


def test_fractional_zero_length_is_noop_for_every_supported_target() -> None:
    image = np.arange(7 * 5 * 3, dtype=np.uint8).reshape(7, 5, 3)
    mask = np.arange(7 * 5, dtype=np.uint8).reshape(7, 5)
    bboxes = np.array([[0.1, 0.2, 0.8, 0.9]], dtype=np.float32)
    bbox_labels = ["object"]
    keypoints = np.array([[2.0, 3.0]], dtype=np.float32)
    keypoint_labels = ["point"]
    volume = np.arange(2 * 7 * 5 * 3, dtype=np.uint8).reshape(2, 7, 5, 3)
    mask3d = np.arange(2 * 7 * 5, dtype=np.uint8).reshape(2, 7, 5)
    transform = A.XYMasking(
        num_masks_x_range=(1, 1),
        mask_x_length_range=(0.1, 0.1),
        fill=0,
        fill_mask=99,
        p=1,
    )
    pipeline = A.Compose(
        [transform],
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="hbb",
            label_fields=["bbox_labels"],
        ),
        keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["keypoint_labels"]),
        seed=137,
        strict=True,
    )

    result = pipeline(
        image=image,
        mask=mask,
        bboxes=bboxes,
        bbox_labels=bbox_labels,
        keypoints=keypoints,
        keypoint_labels=keypoint_labels,
        volume=volume,
        mask3d=mask3d,
    )

    _assert_canonical_empty_holes(transform.get_applied_params()["holes"])
    np.testing.assert_array_equal(result["image"], image)
    np.testing.assert_array_equal(result["mask"], mask)
    np.testing.assert_array_equal(result["bboxes"], bboxes)
    assert result["bbox_labels"] == bbox_labels
    np.testing.assert_array_equal(result["keypoints"], keypoints)
    assert result["keypoint_labels"] == keypoint_labels
    np.testing.assert_array_equal(result["volume"], volume)
    np.testing.assert_array_equal(result["mask3d"], mask3d)


@pytest.mark.parametrize("mode", ["float-float", "float-int", "int-float"])
@pytest.mark.parametrize("zero_noop", [False, True])
def test_relative_and_mixed_modes_cover_batch_and_volume_targets(mode: str, zero_noop: bool) -> None:
    height, width = 5, 7
    relative_length = 0.1 if zero_noop else 0.5
    if mode == "float-float":
        mask_x_length_range: tuple[int, int] | tuple[float, float] = (relative_length, relative_length)
        mask_y_length_range: tuple[int, int] | tuple[float, float] = (relative_length, relative_length)
        num_masks_x_range = (1, 1)
        num_masks_y_range = (1, 1)
    elif mode == "float-int":
        mask_x_length_range = (relative_length, relative_length)
        mask_y_length_range = (2, 2)
        num_masks_x_range = (1, 1)
        num_masks_y_range = (0, 0) if zero_noop else (1, 1)
    else:
        mask_x_length_range = (2, 2)
        mask_y_length_range = (relative_length, relative_length)
        num_masks_x_range = (0, 0) if zero_noop else (1, 1)
        num_masks_y_range = (1, 1)

    transform = A.Compose(
        [
            A.XYMasking(
                num_masks_x_range=num_masks_x_range,
                num_masks_y_range=num_masks_y_range,
                mask_x_length_range=mask_x_length_range,
                mask_y_length_range=mask_y_length_range,
                fill=0,
                fill_mask=0,
                p=1,
            ),
        ],
        seed=137,
        strict=True,
    )
    inputs = {
        "image": np.full((height, width, 1), 9, dtype=np.uint8),
        "images": np.full((2, height, width, 1), 9, dtype=np.uint8),
        "masks": np.full((2, height, width), 9, dtype=np.uint8),
        "volume": np.full((2, height, width, 1), 9, dtype=np.uint8),
        "mask3d": np.full((2, height, width), 9, dtype=np.uint8),
    }

    result = transform(**inputs)

    for target_name, target in inputs.items():
        assert result[target_name].shape == target.shape
        assert result[target_name].dtype == target.dtype
        if zero_noop:
            np.testing.assert_array_equal(result[target_name], target)
        else:
            assert np.any(result[target_name] == 0)


def test_fractional_zero_length_is_noop_for_float32_inpainting() -> None:
    image = np.linspace(0, 1, 7 * 5 * 3, dtype=np.float32).reshape(7, 5, 3)
    transform = A.XYMasking(
        num_masks_x_range=(1, 1),
        mask_x_length_range=(0.1, 0.1),
        fill="inpaint_telea",
        p=1,
    )

    result = A.Compose([transform], seed=137, strict=True)(image=image)

    _assert_canonical_empty_holes(transform.get_applied_params()["holes"])
    np.testing.assert_array_equal(result["image"], image)


def test_fractional_zero_length_keeps_rng_order_for_later_masks() -> None:
    transform = A.XYMasking(
        num_masks_x_range=(3, 3),
        mask_x_length_range=(0.1, 0.4),
        p=1,
    )
    transform.set_random_seed(137)

    transform(image=np.ones((7, 5, 1), dtype=np.uint8))
    holes = transform.get_applied_params()["holes"]

    np.testing.assert_array_equal(holes, [[1, 0, 2, 7], [0, 0, 1, 7]])


@pytest.mark.parametrize("seed", range(137, 157))
def test_relative_mask_coordinates_stay_within_image(seed: int) -> None:
    height, width = 13, 17
    holes = _sample_holes(
        A.XYMasking(
            num_masks_x_range=(2, 4),
            num_masks_y_range=(2, 4),
            mask_x_length_range=(0.0, 1.0),
            mask_y_length_range=(0.0, 1.0),
            p=1,
        ),
        (height, width, 1),
        seed,
    )

    assert np.all(holes[:, 0] >= 0)
    assert np.all(holes[:, 1] >= 0)
    assert np.all(holes[:, 2] <= width)
    assert np.all(holes[:, 3] <= height)
    assert np.all(holes[:, 0] < holes[:, 2])
    assert np.all(holes[:, 1] < holes[:, 3])


@pytest.mark.parametrize("shape", [(12, 20, 1), (24, 40, 1)])
def test_fractional_x_scales_while_integer_y_stays_fixed(shape: tuple[int, int, int]) -> None:
    height, width = shape[:2]
    holes = _sample_holes(
        A.XYMasking(
            num_masks_x_range=(1, 1),
            num_masks_y_range=(1, 1),
            mask_x_length_range=(0.25, 0.25),
            mask_y_length_range=(4, 4),
            p=1,
        ),
        shape,
    )

    assert holes.shape == (2, 4)
    np.testing.assert_array_equal(holes[:, 2] - holes[:, 0], [int(0.25 * width), width])
    np.testing.assert_array_equal(holes[:, 3] - holes[:, 1], [height, 4])


def test_relative_geometry_respects_counts_orientation_and_length_bounds() -> None:
    height, width = 23, 37
    holes = _sample_holes(
        A.XYMasking(
            num_masks_x_range=(2, 3),
            num_masks_y_range=(1, 2),
            mask_x_length_range=(0.2, 0.4),
            mask_y_length_range=(3, 7),
            p=1,
        ),
        (height, width, 1),
    )
    x_holes = holes[(holes[:, 1] == 0) & (holes[:, 3] == height)]
    y_holes = holes[(holes[:, 0] == 0) & (holes[:, 2] == width)]

    assert 2 <= len(x_holes) <= 3
    assert 1 <= len(y_holes) <= 2
    assert np.all(x_holes[:, 2] - x_holes[:, 0] >= int(0.2 * width))
    assert np.all(x_holes[:, 2] - x_holes[:, 0] <= int(0.4 * width))
    assert np.all(y_holes[:, 3] - y_holes[:, 1] >= 3)
    assert np.all(y_holes[:, 3] - y_holes[:, 1] <= 7)
    assert np.all((holes[:, 2] - holes[:, 0]) * (holes[:, 3] - holes[:, 1]) > 0)


def test_axes_accept_different_range_representations() -> None:
    transform = A.XYMasking(
        num_masks_x_range=(1, 1),
        num_masks_y_range=(1, 1),
        mask_x_length_range=(3, 3),
        mask_y_length_range=(0.5, 0.5),
        p=1,
    )

    holes = _sample_holes(transform, (9, 13, 1))

    assert transform.mask_x_length_range == (3, 3)
    assert transform.mask_y_length_range == (0.5, 0.5)
    np.testing.assert_array_equal(holes[:, 2] - holes[:, 0], [3, 13])
    np.testing.assert_array_equal(holes[:, 3] - holes[:, 1], [9, 4])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ((np.int32(0), np.int64(3)), (0, 3)),
        ((_Length.ZERO, _Length.THREE), (0, 3)),
        ((np.float32(0.0), np.float64(1.0)), (0.0, 1.0)),
        ((0.0, np.float32(0.5)), (0.0, 0.5)),
        ((np.longdouble("0.25"), np.longdouble("0.5")), (0.25, 0.5)),
        ([0.0, np.float32(0.5)], (0.0, 0.5)),
    ],
)
@pytest.mark.parametrize("field_name", ["mask_x_length_range", "mask_y_length_range"])
@pytest.mark.parametrize("strict", [False, True])
def test_mask_length_range_accepts_numeric_scalars_and_normalizes_to_builtin(
    value: object,
    expected: tuple[int, int] | tuple[float, float],
    field_name: str,
    strict: bool,
) -> None:
    transform = A.XYMasking(**{field_name: value}, strict=strict)

    normalized_range = getattr(transform, field_name)
    assert normalized_range == expected
    assert all(type(element) is type(expected[0]) for element in normalized_range)


@pytest.mark.parametrize(
    "value_factory",
    [
        lambda: (0, 3),
        lambda: [0, 3],
        lambda: deque([0, 3]),
        lambda: np.array([0, 3]),
        lambda: range(2),
        lambda: (element for element in (0, 3)),
    ],
)
@pytest.mark.parametrize("field_name", ["mask_x_length_range", "mask_y_length_range"])
@pytest.mark.parametrize("strict", [False, True])
def test_mask_length_range_accepts_ordered_two_item_iterables(
    value_factory: Callable[[], object],
    field_name: str,
    strict: bool,
) -> None:
    transform = A.XYMasking(**{field_name: value_factory()}, strict=strict)

    normalized_range = getattr(transform, field_name)
    assert normalized_range in {(0, 3), (0, 1)}
    assert all(type(element) is int for element in normalized_range)


def test_wrong_length_generator_consumes_at_most_three_values() -> None:
    consumed = []

    def values() -> Iterator[int]:
        for value in itertools.count():
            consumed.append(value)
            yield value

    with pytest.raises(ValueError, match="exactly two values"):
        A.XYMasking(mask_x_length_range=values())

    assert consumed == [0, 1, 2]


@pytest.mark.parametrize("field_name", ["mask_x_length_range", "mask_y_length_range"])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "value",
    [
        (0, 0.5),
        (0.0, 1),
        [0, 0.5],
        [0.0, 1],
        (np.int64(0), np.float32(0.5)),
    ],
)
def test_mask_length_range_rejects_mixed_element_types(field_name: str, value: object, strict: bool) -> None:
    kwargs = {field_name: value, "strict": strict}

    with pytest.raises(ValueError, match="all integers or all floats"):
        A.XYMasking(**kwargs)


@pytest.mark.parametrize(
    "value_factory",
    [
        lambda: deque([0.0, 1.0]),
        lambda: np.array([0.0, 1.0]),
        lambda: (value for value in (0.0, 1.0)),
    ],
)
@pytest.mark.parametrize("strict", [False, True])
def test_float_ranges_reject_non_tuple_list_containers(
    value_factory: Callable[[], object],
    strict: bool,
) -> None:
    with pytest.raises(ValueError, match="must use a tuple or list"):
        A.XYMasking(mask_x_length_range=value_factory(), strict=strict)


@pytest.mark.parametrize(
    "value",
    [
        "01",
        b"01",
        {0: "lower", 1: "upper"},
        {0, 1},
        frozenset({0, 1}),
        (value for value in (0,)),
        (value for value in (0, 1, 2)),
        np.array([[0, 1]]),
        3,
        np.float32(0.5),
        Decimal("0.5"),
        (Decimal("0.25"), Decimal("0.5")),
        Fraction(1, 2),
        (_CustomReal(), _CustomReal()),
    ],
)
@pytest.mark.parametrize("strict", [False, True])
def test_mask_length_range_rejects_unordered_non_numeric_and_wrong_arity_inputs(value: object, strict: bool) -> None:
    with pytest.raises(ValueError):
        A.XYMasking(mask_x_length_range=value, strict=strict)


@pytest.mark.parametrize("value", [(1.0, 1.0), [1.0, 1.0]])
def test_float_range_transport_preserves_integral_valued_float_identity(value: object) -> None:
    transform = A.XYMasking(
        num_masks_x_range=(1, 1),
        num_masks_y_range=(1, 1),
        mask_x_length_range=value,
        mask_y_length_range=value,
        p=1,
    )
    pipeline = A.Compose([transform], seed=137)
    transported = json.loads(json.dumps(A.to_dict(pipeline), allow_nan=False))
    restored = A.from_dict(transported)
    restored_transform = restored.transforms[0]
    restored(image=np.ones((7, 11, 1), dtype=np.uint8))
    holes = restored_transform.get_applied_params()["holes"]

    assert transform.mask_x_length_range == (1.0, 1.0)
    assert all(type(element) is float for element in transform.mask_x_length_range)
    assert restored_transform.mask_x_length_range == (1.0, 1.0)
    assert restored_transform.mask_y_length_range == (1.0, 1.0)
    assert all(type(element) is float for element in restored_transform.mask_x_length_range)
    assert all(type(element) is float for element in restored_transform.mask_y_length_range)
    np.testing.assert_array_equal(holes, [[0, 0, 11, 7], [0, 0, 11, 7]])


@pytest.mark.parametrize("field_name", ["mask_x_length_range", "mask_y_length_range"])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "value",
    [
        (-1, 1),
        (2, 1),
        (-0.1, 0.5),
        (0.5, 1.1),
        (0.75, 0.25),
        (float("nan"), 0.5),
        (0.5, float("nan")),
        (-float("inf"), 0.5),
        (0.5, float("inf")),
        (False, True),
        (np.bool_(False), np.bool_(True)),
        [False, 1],
        ("0", "1"),
    ],
)
def test_mask_length_range_rejects_invalid_values(field_name: str, value: object, strict: bool) -> None:
    kwargs = {field_name: value, "strict": strict}

    with pytest.raises(ValueError):
        A.XYMasking(**kwargs)


@pytest.mark.parametrize(
    "value",
    [
        (np.nextafter(np.longdouble(1), np.longdouble(2)), np.longdouble(1)),
        (np.nextafter(np.longdouble(0), np.longdouble(-1)), np.longdouble(0.5)),
        (np.longdouble(0.5), np.longdouble("inf")),
        (np.longdouble("nan"), np.longdouble(0.5)),
        (np.nextafter(np.longdouble(0.5), np.longdouble(1)), np.longdouble(0.5)),
    ],
)
@pytest.mark.parametrize("strict", [False, True])
def test_float_range_validates_raw_values_before_builtin_float_normalization(value: object, strict: bool) -> None:
    with pytest.raises(ValueError):
        A.XYMasking(mask_x_length_range=value, strict=strict)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "value",
    [
        (np.longdouble("0.25"), Fraction(1, 2)),
        (Fraction(10**400), Fraction(10**400)),
    ],
)
def test_invalid_real_ranges_fail_closed_without_warning(value: object, strict: bool) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as raised:
            A.XYMasking(mask_x_length_range=value, strict=strict)

    assert caught == []
    assert raised.value.__cause__ is not None
    assert type(raised.value.__cause__).__name__ == "ValidationError"


@pytest.mark.parametrize(
    ("field_name", "range_value", "shape"),
    [
        ("mask_x_length_range", (0, 18), (11, 17, 1)),
        ("mask_y_length_range", (0, 12), (11, 17, 1)),
    ],
)
def test_integer_ranges_retain_runtime_dimension_validation(
    field_name: str,
    range_value: tuple[int, int],
    shape: tuple[int, int, int],
) -> None:
    transform = A.XYMasking(**{field_name: range_value}, p=1)
    pipeline = A.Compose([transform], seed=137, strict=True)

    with pytest.raises(ValueError, match="out of valid range"):
        pipeline(image=np.ones(shape, dtype=np.uint8))


@pytest.mark.parametrize("invalid_axis", ["x", "y"])
def test_integer_dimension_error_does_not_advance_rng(invalid_axis: str) -> None:
    invalid_kwargs = {
        "num_masks_x_range": (1, 2),
        "num_masks_y_range": (1, 2),
        "mask_x_length_range": (2, 18) if invalid_axis == "x" else (2, 5),
        "mask_y_length_range": (2, 12) if invalid_axis == "y" else (2, 5),
        "fill": 0,
        "p": 1,
    }
    retried = A.XYMasking(**invalid_kwargs)
    fresh = A.XYMasking(**invalid_kwargs)
    retried.set_random_seed(137)
    fresh.set_random_seed(137)
    invalid_image = np.ones((11, 17, 1), dtype=np.uint8)
    valid_image = np.ones((13, 19, 1), dtype=np.uint8)

    with pytest.raises(ValueError, match=f"mask_{invalid_axis}_length_range"):
        retried(image=invalid_image)

    retried_result = retried(image=valid_image)
    fresh_result = fresh(image=valid_image)
    np.testing.assert_array_equal(retried.get_applied_params()["holes"], fresh.get_applied_params()["holes"])
    np.testing.assert_array_equal(retried_result["image"], fresh_result["image"])

    retried(image=valid_image)
    fresh(image=valid_image)
    np.testing.assert_array_equal(retried.get_applied_params()["holes"], fresh.get_applied_params()["holes"])


def test_integer_dimension_preflight_checks_x_before_y() -> None:
    transform = A.XYMasking(
        num_masks_x_range=(1, 1),
        num_masks_y_range=(1, 1),
        mask_x_length_range=(2, 18),
        mask_y_length_range=(2, 12),
        p=1,
    )

    with pytest.raises(ValueError, match="mask_x_length_range"):
        transform(image=np.ones((11, 17, 1), dtype=np.uint8))


def test_integer_length_equal_to_dimension_is_valid() -> None:
    holes = _sample_holes(
        A.XYMasking(
            num_masks_x_range=(1, 1),
            mask_x_length_range=(5, 5),
            p=1,
        ),
        (7, 5, 1),
    )

    np.testing.assert_array_equal(holes, [[0, 0, 5, 7]])


def test_strict_json_applied_configuration_reconstructs_runnable_policy_at_new_resolution() -> None:
    image = np.ones((12, 20, 1), dtype=np.uint8)
    pipeline = A.Compose(
        [
            A.XYMasking(
                num_masks_x_range=(1, 1),
                num_masks_y_range=(1, 1),
                mask_x_length_range=(0.25, 0.25),
                mask_y_length_range=(3, 3),
                fill=0,
                p=1,
            ),
        ],
        save_applied_params=True,
        seed=137,
    )

    result = pipeline(image=image)
    transported = json.loads(json.dumps(result["applied_transforms"], allow_nan=False))
    reconstructed = A.Compose.from_applied_transforms(transported, seed=151)
    reconstructed_transform = reconstructed.transforms[0]
    reconstructed(image=np.ones((24, 40, 1), dtype=np.uint8))
    holes = reconstructed_transform.get_applied_params()["holes"]

    assert reconstructed_transform.mask_x_length_range == (0.25, 0.25)
    assert reconstructed_transform.mask_y_length_range == (3, 3)
    assert all(type(element) is float for element in reconstructed_transform.mask_x_length_range)
    assert all(type(element) is int for element in reconstructed_transform.mask_y_length_range)
    np.testing.assert_array_equal(holes[:, 2] - holes[:, 0], [10, 40])
    np.testing.assert_array_equal(holes[:, 3] - holes[:, 1], [24, 3])


def test_one_serialized_relative_pipeline_scales_across_resolutions() -> None:
    pipeline = A.from_dict(
        json.loads(
            json.dumps(
                A.to_dict(
                    A.Compose(
                        [
                            A.XYMasking(
                                num_masks_x_range=(1, 1),
                                mask_x_length_range=(0.5, 0.5),
                                p=1,
                            ),
                        ],
                        seed=137,
                    ),
                ),
            ),
        ),
    )
    transform = pipeline.transforms[0]

    pipeline(image=np.ones((7, 10, 1), dtype=np.uint8))
    first_holes = transform.get_applied_params()["holes"]
    pipeline(image=np.ones((7, 20, 1), dtype=np.uint8))
    second_holes = transform.get_applied_params()["holes"]

    np.testing.assert_array_equal(first_holes[:, 2] - first_holes[:, 0], [5])
    np.testing.assert_array_equal(second_holes[:, 2] - second_holes[:, 0], [10])


def test_inactive_axes_return_canonical_empty_holes() -> None:
    holes = _sample_holes(A.XYMasking(mask_x_length_range=(3, 3), p=1), (11, 17, 1))

    _assert_canonical_empty_holes(holes)


@pytest.mark.parametrize(
    ("kwargs", "shape", "seed", "expected"),
    [
        (
            {"num_masks_x_range": (2, 2), "mask_x_length_range": (2, 5)},
            (11, 17, 1),
            137,
            [[6, 0, 9, 11], [3, 0, 7, 11]],
        ),
        (
            {"num_masks_y_range": (2, 2), "mask_y_length_range": (3, 6)},
            (11, 17, 1),
            137,
            [[0, 6, 17, 10], [0, 1, 17, 6]],
        ),
        (
            {
                "num_masks_x_range": (1, 3),
                "num_masks_y_range": (2, 4),
                "mask_x_length_range": (1, 7),
                "mask_y_length_range": (2, 8),
            },
            (13, 19, 1),
            137,
            [[13, 0, 15, 13], [0, 3, 19, 6], [0, 5, 19, 10], [0, 6, 19, 8]],
        ),
        (
            {
                "num_masks_x_range": (0, 0),
                "num_masks_y_range": (1, 2),
                "mask_x_length_range": (4, 8),
                "mask_y_length_range": (2, 5),
            },
            (13, 19, 1),
            151,
            [[0, 5, 19, 8], [0, 5, 19, 9]],
        ),
        (
            {
                "num_masks_x_range": (1, 2),
                "num_masks_y_range": (0, 0),
                "mask_x_length_range": (2, 6),
                "mask_y_length_range": (3, 7),
            },
            (13, 19, 1),
            151,
            [[11, 0, 14, 13], [11, 0, 15, 13]],
        ),
        (
            {"num_masks_x_range": (1, 1), "mask_x_length_range": (19, 19)},
            (13, 19, 1),
            211,
            [[0, 0, 19, 13]],
        ),
        (
            {"num_masks_x_range": (1, 1), "mask_x_length_range": (0, 1)},
            (13, 19, 1),
            137,
            [[13, 0, 13, 13]],
        ),
        (
            {
                "num_masks_x_range": (1, 2),
                "num_masks_y_range": (1, 2),
                "mask_x_length_range": (0, 4),
                "mask_y_length_range": (0, 5),
            },
            (13, 19, 1),
            211,
            [[0, 0, 1, 13], [17, 0, 19, 13], [0, 9, 19, 13], [0, 2, 19, 7]],
        ),
    ],
)
def test_integer_sampling_matches_permanent_baseline_golden_vectors(
    kwargs: dict[str, tuple[int, int]],
    shape: tuple[int, int, int],
    seed: int,
    expected: list[list[int]],
) -> None:
    transform = A.XYMasking(**kwargs, p=1)
    transform.set_random_seed(seed)
    transform(image=np.ones(shape, dtype=np.uint8))
    holes = transform.get_applied_params()["holes"]

    np.testing.assert_array_equal(holes, expected)


def test_integer_sampling_preserves_sequential_rng_progression() -> None:
    transform = A.XYMasking(
        num_masks_x_range=(1, 2),
        num_masks_y_range=(1, 2),
        mask_x_length_range=(0, 4),
        mask_y_length_range=(0, 5),
        p=1,
    )
    transform.set_random_seed(137)
    image = np.ones((13, 19, 1), dtype=np.uint8)
    expected_calls = [
        [[13, 0, 14, 13], [0, 3, 19, 4], [0, 5, 19, 8]],
        [[16, 0, 19, 13], [0, 8, 19, 11], [0, 2, 19, 5]],
        [[10, 0, 14, 13], [19, 0, 19, 13], [0, 2, 19, 7]],
    ]

    for expected in expected_calls:
        transform(image=image)
        np.testing.assert_array_equal(transform.get_applied_params()["holes"], expected)
