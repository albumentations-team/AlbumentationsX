"""Finite-group contracts for geometric symmetries through public Compose routes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.transforms3d import functional as f3d
from albumentations.core.bbox_utils import obb_to_polygons
from albumentations.core.transforms_interface import BasicTransform
from tests.helpers import obb_corners_equivalent

BboxType = Literal["hbb", "obb"]
D4_REFLECTIONS = ("v", "hvt", "h", "t")
CUBIC_SYMMETRY_ORDERS = (
    1,
    4,
    2,
    4,
    2,
    2,
    2,
    2,
    4,
    3,
    2,
    3,
    4,
    3,
    2,
    3,
    4,
    3,
    2,
    3,
    4,
    3,
    2,
    3,
    2,
    2,
    2,
    2,
    2,
    4,
    2,
    4,
    2,
    6,
    4,
    6,
    2,
    6,
    4,
    6,
    4,
    6,
    2,
    6,
    4,
    6,
    2,
    6,
)


def _make_2d_data(bbox_type: BboxType) -> dict[str, object]:
    image = np.arange(4 * 6, dtype=np.uint8).reshape(4, 6, 1)
    volume = np.arange(2 * 4 * 6, dtype=np.uint8).reshape(2, 4, 6, 1)
    bboxes: list[list[float]] = [[0.2, 0.3, 0.4, 0.6]]
    if bbox_type == "obb":
        bboxes[0].append(30.0)
    return {
        "image": image,
        "mask": image[..., 0] % 3,
        "volume": volume,
        "mask3d": volume[..., 0] % 5,
        "keypoints": [
            [1.0, 1.0, 0.1],
            [4.0, 2.0, 1.5 * np.pi],
            [2.0, 1.0, -1e-6],
            [3.0, 1.0, 0.5 * np.pi - 1e-6],
            [3.0, 2.0, 0.5 * np.pi + 1e-6],
            [2.0, 2.0, np.pi - 1e-6],
            [2.0, 3.0, np.pi + 1e-6],
            [1.0, 3.0, 2.0 * np.pi - 1e-6],
        ],
        "bboxes": bboxes,
    }


def _make_2d_compose(transforms: Sequence[BasicTransform], bbox_type: BboxType) -> A.Compose:
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(coord_format="albumentations", bbox_type=bbox_type),
        keypoint_params=A.KeypointParams(coord_format="xya", angle_in_degrees=False),
        strict=True,
        telemetry=False,
    )


def _apply_repeatedly(transform: A.Compose, data: dict[str, object], count: int) -> dict[str, object]:
    result = data
    for _ in range(count):
        result = transform(**result)
    return result


def _assert_keypoints_equal_modulo_2pi(actual: object, expected: object) -> None:
    actual_keypoints = np.asarray(actual)
    expected_keypoints = np.asarray(expected)
    np.testing.assert_allclose(actual_keypoints[:, :2], expected_keypoints[:, :2], atol=1e-6)
    angle_difference = np.mod(actual_keypoints[:, 2] - expected_keypoints[:, 2] + np.pi, 2 * np.pi) - np.pi
    np.testing.assert_allclose(angle_difference, 0.0, atol=1e-6)


def _assert_bboxes_equal(actual: object, expected: object, bbox_type: BboxType) -> None:
    actual_bboxes = np.asarray(actual)
    expected_bboxes = np.asarray(expected)
    if bbox_type == "hbb":
        np.testing.assert_allclose(actual_bboxes, expected_bboxes, atol=1e-6)
        return

    assert np.all((actual_bboxes[:, 4] >= -90.0) & (actual_bboxes[:, 4] < 90.0))
    for actual_bbox, expected_bbox in zip(actual_bboxes, expected_bboxes, strict=True):
        assert obb_corners_equivalent(
            obb_to_polygons(actual_bbox[None, :])[0],
            obb_to_polygons(expected_bbox[None, :])[0],
        )


def _assert_2d_targets_equal(actual: dict[str, object], expected: dict[str, object], bbox_type: BboxType) -> None:
    for target in ("image", "mask", "volume", "mask3d"):
        np.testing.assert_array_equal(actual[target], expected[target])
    _assert_keypoints_equal_modulo_2pi(actual["keypoints"], expected["keypoints"])
    _assert_bboxes_equal(actual["bboxes"], expected["bboxes"], bbox_type)


@pytest.mark.parametrize("bbox_type", ["hbb", "obb"])
@pytest.mark.parametrize("transform_cls", [A.HorizontalFlip, A.VerticalFlip, A.Transpose])
def test_c2_symmetries_are_involutions_for_all_2d_targets(
    transform_cls: type[BasicTransform],
    bbox_type: BboxType,
) -> None:
    data = _make_2d_data(bbox_type)
    compose = _make_2d_compose([transform_cls(p=1.0)], bbox_type)

    _assert_2d_targets_equal(_apply_repeatedly(compose, data, 2), data, bbox_type)


@pytest.mark.parametrize("flip_axes", [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)])
def test_fixed_flip3d_is_an_involution(flip_axes: tuple[int, ...]) -> None:
    volume = np.arange(2 * 3 * 5, dtype=np.uint8).reshape(2, 3, 5, 1)
    data: dict[str, object] = {
        "volume": volume,
        "mask3d": volume[..., 0] % 3,
        "keypoints": [[0.0, 0.0, 0.0], [4.0, 2.0, 1.0]],
    }
    compose = A.Compose(
        [A.Flip3D(flip_axes=flip_axes, p=1.0)],
        keypoint_params=A.KeypointParams(coord_format="xyz"),
        strict=True,
        telemetry=False,
    )

    result = _apply_repeatedly(compose, data, 2)

    for target in ("volume", "mask3d", "keypoints"):
        np.testing.assert_array_equal(result[target], data[target])


@pytest.mark.parametrize("bbox_type", ["hbb", "obb"])
def test_d4_r90_has_order_four_for_all_2d_targets(bbox_type: BboxType) -> None:
    data = _make_2d_data(bbox_type)
    compose = _make_2d_compose([A.D4(group_element="r90", p=1.0)], bbox_type)

    _assert_2d_targets_equal(_apply_repeatedly(compose, data, 4), data, bbox_type)


@pytest.mark.parametrize("axis_pair", [(0, 1), (0, 2), (1, 2)])
def test_fixed_3d_r90_has_order_four_for_every_axis_pair(axis_pair: tuple[int, int]) -> None:
    volume = np.arange(2 * 3 * 5, dtype=np.uint8).reshape(2, 3, 5, 1)
    data: dict[str, object] = {
        "volume": volume,
        "mask3d": volume[..., 0] % 3,
        "keypoints": [[0.0, 0.0, 0.0], [4.0, 2.0, 1.0]],
    }
    compose = A.Compose(
        [A.RandomRotate90_3D(axis_pair=axis_pair, group_element="r90", p=1.0)],
        keypoint_params=A.KeypointParams(coord_format="xyz"),
        strict=True,
        telemetry=False,
    )

    result = _apply_repeatedly(compose, data, 4)

    for target in ("volume", "mask3d", "keypoints"):
        np.testing.assert_array_equal(result[target], data[target])


@pytest.mark.parametrize("bbox_type", ["hbb", "obb"])
@pytest.mark.parametrize("reflection", D4_REFLECTIONS)
def test_d4_reflections_are_involutions_for_all_2d_targets(reflection: str, bbox_type: BboxType) -> None:
    data = _make_2d_data(bbox_type)
    compose = _make_2d_compose([A.D4(group_element=reflection, p=1.0)], bbox_type)

    _assert_2d_targets_equal(_apply_repeatedly(compose, data, 2), data, bbox_type)


@pytest.mark.parametrize("bbox_type", ["hbb", "obb"])
def test_d4_transpose_conjugates_r90_to_r270_for_all_2d_targets(bbox_type: BboxType) -> None:
    data = _make_2d_data(bbox_type)
    conjugated = _make_2d_compose(
        [A.D4(group_element="t", p=1.0), A.D4(group_element="r90", p=1.0), A.D4(group_element="t", p=1.0)],
        bbox_type,
    )
    r270 = _make_2d_compose([A.D4(group_element="r270", p=1.0)], bbox_type)

    _assert_2d_targets_equal(conjugated(**data), r270(**data), bbox_type)


def _transform_d4_corners(corners: np.ndarray, group_element: str) -> np.ndarray:
    transformed = corners.copy()
    x_coordinates = corners[:, 0]
    y_coordinates = corners[:, 1]
    if group_element == "e":
        return transformed
    if group_element == "r90":
        transformed[:, 0], transformed[:, 1] = y_coordinates, 1.0 - x_coordinates
    elif group_element == "r180":
        transformed[:, 0], transformed[:, 1] = 1.0 - x_coordinates, 1.0 - y_coordinates
    elif group_element == "r270":
        transformed[:, 0], transformed[:, 1] = 1.0 - y_coordinates, x_coordinates
    elif group_element == "v":
        transformed[:, 1] = 1.0 - y_coordinates
    elif group_element == "hvt":
        transformed[:, 0], transformed[:, 1] = 1.0 - y_coordinates, 1.0 - x_coordinates
    elif group_element == "h":
        transformed[:, 0] = 1.0 - x_coordinates
    elif group_element == "t":
        transformed[:, 0], transformed[:, 1] = y_coordinates, x_coordinates
    else:
        msg = f"Unknown D4 group element: {group_element}"
        raise ValueError(msg)
    return transformed


@pytest.mark.parametrize("group_element", ["e", "r90", "r180", "r270", *D4_REFLECTIONS])
def test_d4_obb_corners_follow_the_independent_group_mapping(group_element: str) -> None:
    data = _make_2d_data("obb")
    compose = _make_2d_compose([A.D4(group_element=group_element, p=1.0)], "obb")

    result = compose(**data)
    source_corners = obb_to_polygons(np.asarray(data["bboxes"], dtype=np.float32))[0]
    actual_corners = obb_to_polygons(np.asarray(result["bboxes"], dtype=np.float32))[0]

    assert obb_corners_equivalent(actual_corners, _transform_d4_corners(source_corners, group_element))
    assert -90.0 <= result["bboxes"][0][4] < 90.0


@pytest.mark.parametrize("source_angle", [-450.0, 450.0])
@pytest.mark.parametrize("group_element", ["r90", "r180", "r270", *D4_REFLECTIONS])
def test_d4_obb_symmetries_canonicalize_noncanonical_input_angles(group_element: str, source_angle: float) -> None:
    data = _make_2d_data("obb")
    data["bboxes"][0][4] = source_angle
    compose = _make_2d_compose([A.D4(group_element=group_element, p=1.0)], "obb")

    result = compose(**data)
    source_corners = obb_to_polygons(np.asarray(data["bboxes"], dtype=np.float32))[0]
    actual_corners = obb_to_polygons(np.asarray(result["bboxes"], dtype=np.float32))[0]

    assert obb_corners_equivalent(actual_corners, _transform_d4_corners(source_corners, group_element))
    assert -90.0 <= result["bboxes"][0][4] < 90.0


@pytest.mark.parametrize(("index", "order"), list(enumerate(CUBIC_SYMMETRY_ORDERS)))
def test_cubic_symmetry_table_preserves_target_alignment_and_element_order(index: int, order: int) -> None:
    volume = np.arange(3 * 4 * 5, dtype=np.int16).reshape(3, 4, 5)
    mask3d = volume % 7
    z_coordinates, y_coordinates, x_coordinates = np.indices(volume.shape)
    keypoints = np.column_stack((x_coordinates.ravel(), y_coordinates.ravel(), z_coordinates.ravel())).astype(
        np.float32
    )

    transformed_volume = f3d.transform_cube(volume, index)
    transformed_mask3d = f3d.transform_cube(mask3d, index)
    transformed_keypoints = f3d.transform_cube_keypoints(keypoints, index, volume.shape)
    transformed_x = transformed_keypoints[:, 0].astype(int)
    transformed_y = transformed_keypoints[:, 1].astype(int)
    transformed_z = transformed_keypoints[:, 2].astype(int)

    np.testing.assert_array_equal(transformed_mask3d, transformed_volume % 7)
    np.testing.assert_array_equal(
        transformed_volume[transformed_z, transformed_y, transformed_x],
        volume[z_coordinates.ravel(), y_coordinates.ravel(), x_coordinates.ravel()],
    )

    current_volume = volume
    current_keypoints = keypoints
    current_shape = volume.shape
    for _ in range(order):
        current_volume = f3d.transform_cube(current_volume, index)
        current_keypoints = f3d.transform_cube_keypoints(current_keypoints, index, current_shape)
        current_shape = current_volume.shape

    np.testing.assert_array_equal(current_volume, volume)
    np.testing.assert_array_equal(current_keypoints, keypoints)
