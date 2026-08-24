"""Focused contracts for the greenfield bounded ElasticTransform."""

import json
from typing import Any

import cv2
import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.invocation import SamplingContext
from tests.utils import make_sampling_args


class _LabelMappingElasticTransform(A.ElasticTransform):
    def _get_label_transform_name(self, **params: Any) -> str:
        return "HorizontalFlip"


def _image(height: int = 32, width: int = 40, channels: int = 3) -> np.ndarray:
    return np.random.default_rng(137).random((height, width, channels), dtype=np.float32)


def _cubic_basis(t: float) -> np.ndarray:
    return np.array(
        [
            (1 - t) ** 3 / 6,
            (3 * t**3 - 6 * t**2 + 4) / 6,
            (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6,
            t**3 / 6,
        ],
    )


def _forward(control: np.ndarray, point: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    rows, columns, _ = control.shape
    spans_y, spans_x = rows - 3, columns - 3
    normalized_x = point[0] * spans_x / (width - 1)
    normalized_y = point[1] * spans_y / (height - 1)
    base_x = min(int(np.floor(normalized_x)), spans_x - 1)
    base_y = min(int(np.floor(normalized_y)), spans_y - 1)
    weights_x = _cubic_basis(normalized_x - base_x)
    weights_y = _cubic_basis(normalized_y - base_y)
    displacement = np.zeros(2, dtype=np.float64)
    for row_tap in range(4):
        for column_tap in range(4):
            displacement += weights_y[row_tap] * weights_x[column_tap] * control[base_y + row_tap, base_x + column_tap]
    return point + displacement


@pytest.mark.parametrize(
    "kwargs",
    [
        {"displacement_range": (-0.1, 0.1)},
        {"displacement_range": (0.1, 0.01)},
        {"control_grid_shape": (1, 5)},
        {"displacement_range": (0.2, 0.2)},
    ],
)
def test_elastic_constructor_rejects_invalid_contract(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        A.ElasticTransform(**kwargs)


def test_cubic_control_grid_expansion_matches_scalar_reference() -> None:
    control = np.random.default_rng(137).uniform(-1, 1, (6, 7, 2)).astype(np.float32)
    image_shape = (9, 13)

    dense = fgeometric.expand_control_grid(control, image_shape)
    expected = np.array(
        [
            [_forward(control, np.array([x, y], dtype=np.float64), image_shape) - [x, y] for x in range(image_shape[1])]
            for y in range(image_shape[0])
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(np.moveaxis(dense, 0, -1), expected, atol=1e-6)
    assert dense.dtype == np.float32


def test_cubic_control_grid_constant_field_is_constant() -> None:
    control = np.broadcast_to(np.array([1.5, -2.0], dtype=np.float32), (7, 6, 2)).copy()
    dense = fgeometric.expand_control_grid(control, (17, 13))
    expected = np.broadcast_to(control[0, 0, :, None, None], dense.shape)
    np.testing.assert_allclose(dense, expected, atol=1e-6)


def test_cubic_control_grid_preserves_coefficient_radius() -> None:
    rng = np.random.default_rng(137)
    radius = 4.25
    control = rng.uniform(-1.0, 1.0, (8, 9, 2)).astype(np.float32)
    norms = np.linalg.norm(control, axis=-1, keepdims=True)
    control *= np.minimum(1.0, radius / np.maximum(norms, np.finfo(np.float32).tiny))

    dense = fgeometric.expand_control_grid(control, (97, 113))
    dense_norm = np.linalg.norm(np.moveaxis(dense, 0, -1), axis=-1)
    assert np.max(dense_norm) <= radius + 1e-5


def test_cubic_field_has_continuous_first_and_second_derivatives() -> None:
    control = np.random.default_rng(137).uniform(-1, 1, (7, 8, 2)).astype(np.float64)
    image_shape = (81, 101)
    knot_x = 40.0
    step = 1e-3
    y = 37.0

    values = np.stack(
        [
            _forward(control, np.array([knot_x + offset, y]), image_shape) - [knot_x + offset, y]
            for offset in (-2 * step, -step, 0.0, step, 2 * step)
        ],
    )
    left_first = (values[2] - values[1]) / step
    right_first = (values[3] - values[2]) / step
    left_second = (values[2] - 2 * values[1] + values[0]) / step**2
    right_second = (values[4] - 2 * values[3] + values[2]) / step**2
    np.testing.assert_allclose(left_first, right_first, atol=1e-5)
    np.testing.assert_allclose(left_second, right_second, atol=2e-3)


def test_sampled_control_coefficients_are_bounded_and_replayable() -> None:
    image = _image(64, 48)
    transform = A.ReplayCompose(
        [A.ElasticTransform(displacement_range=(0.05, 0.05), control_grid_shape=(7, 7), p=1.0)],
        seed=137,
    )
    result = transform(image=image)
    params = result["replay"]["transforms"][0]["params"]["params"]
    coefficients = np.asarray(params["control_coefficients"], dtype=np.float32)
    radius = 0.05 * min(image.shape[0] - 1, image.shape[1] - 1)

    assert params["displacement_magnitude"] == 0.05
    assert np.max(np.linalg.norm(coefficients, axis=-1)) <= radius + 1e-5
    dense = fgeometric.expand_control_grid(coefficients, image.shape[:2])
    assert np.max(np.linalg.norm(np.moveaxis(dense, 0, -1), axis=-1)) <= radius + 1e-4
    transported = json.loads(json.dumps(result["replay"], allow_nan=False))
    replayed = A.ReplayCompose.replay(transported, image=image)
    np.testing.assert_array_equal(result["image"], replayed["image"])


def test_replay_rejects_a_different_spatial_shape() -> None:
    image = _image()
    result = A.ReplayCompose([A.ElasticTransform(p=1.0)], seed=137)(image=image)

    with pytest.raises(ValueError, match="same spatial shape"):
        A.ReplayCompose.replay(result["replay"], image=image[:-1])


def test_replay_roundtrip_keeps_all_spatial_targets() -> None:
    image = _image(24, 28)
    mask = np.arange(24 * 28, dtype=np.uint8).reshape(24, 28)
    images = np.stack([image, image], axis=0)
    volume = np.stack([image, image], axis=0)
    bboxes = np.array([[4, 5, 19, 18]], dtype=np.float32)
    keypoints = np.array([[8, 9], [17, 14]], dtype=np.float32)
    pipeline = A.ReplayCompose(
        [A.ElasticTransform(displacement_range=(0.01, 0.03), control_grid_shape=(4, 5), p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["bbox_labels"]),
        keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["keypoint_labels"]),
        seed=137,
    )
    result = pipeline(
        image=image,
        mask=mask,
        images=images,
        volume=volume,
        bboxes=bboxes,
        bbox_labels=[3],
        keypoints=keypoints,
        keypoint_labels=[4, 5],
    )
    replayed = A.ReplayCompose.replay(
        json.loads(json.dumps(result["replay"], allow_nan=False)),
        image=image,
        mask=mask,
        images=images,
        volume=volume,
        bboxes=bboxes,
        bbox_labels=[3],
        keypoints=keypoints,
        keypoint_labels=[4, 5],
    )

    for key in ("image", "mask", "images", "volume", "bboxes", "keypoints"):
        np.testing.assert_array_equal(result[key], replayed[key])
    assert result["bbox_labels"] == replayed["bbox_labels"]
    assert result["keypoint_labels"] == replayed["keypoint_labels"]


def test_applied_config_fixes_magnitude_but_remains_runnable() -> None:
    image = _image()
    pipeline = A.Compose(
        [A.ElasticTransform(displacement_range=(0.02, 0.05), p=1.0)],
        save_applied_params=True,
        seed=137,
    )
    result = pipeline(image=image)
    transported = json.loads(json.dumps(result["applied_transforms"], allow_nan=False))
    _, config = transported[0]

    assert config["displacement_range"][0] == config["displacement_range"][1]
    assert "control_coefficients" not in config
    reconstructed = A.Compose.from_applied_transforms(transported, seed=137)
    assert reconstructed(image=image)["image"].shape == image.shape


def test_keypoint_inverse_recovers_a_scalar_reference() -> None:
    rng = np.random.default_rng(137)
    control = rng.uniform(-0.5, 0.5, (4, 5, 2)).astype(np.float32)
    image_shape = (48, 64)
    output_points = rng.uniform([0.5, 0.5], [63.5, 47.5], (32, 2))
    source_points = np.stack([_forward(control, point, image_shape) for point in output_points])
    keypoints = np.column_stack([source_points, np.full(len(source_points), 7.0, dtype=np.float32)]).astype(np.float32)

    transformed = fgeometric.remap_elastic_keypoints(keypoints, control, image_shape)

    np.testing.assert_allclose(transformed[:, :2], output_points, atol=1e-3)
    np.testing.assert_array_equal(transformed[:, 2], 7.0)


def test_keypoint_inverse_preserves_continuous_boundary_points() -> None:
    control = np.zeros((4, 4, 2), dtype=np.float32)
    keypoints = np.array([[39.5, 31.5], [39.999, 31.999]], dtype=np.float32)

    transformed = fgeometric.remap_elastic_keypoints(keypoints, control, (32, 40))

    np.testing.assert_allclose(transformed, keypoints, atol=1e-6)


def test_keypoint_inverse_rejects_nonfinite_coordinates_without_index_errors() -> None:
    control = np.zeros((4, 4, 2), dtype=np.float32)
    keypoints = np.array([[np.nan, 1.0], [np.inf, 2.0], [-np.inf, 3.0], [1.0, 2.0]])

    with np.errstate(all="raise"):
        transformed = fgeometric.remap_elastic_keypoints(keypoints, control, (32, 40))

    np.testing.assert_array_equal(transformed[:3], -1.0)
    np.testing.assert_array_equal(transformed[3], keypoints[3])


def test_keypoint_inverse_marks_divergent_iterations_invalid() -> None:
    control = np.full((4, 4, 2), np.finfo(np.float64).max)
    keypoints = np.array([[1.0, 2.0]])

    with np.errstate(all="raise"):
        transformed = fgeometric.remap_elastic_keypoints(keypoints, control, (32, 40))

    np.testing.assert_array_equal(transformed, -1.0)


def test_zero_spatial_extent_does_not_sample_coefficients() -> None:
    transform = A.ElasticTransform(displacement_range=(0.01, 0.01), p=1.0)

    data = {"image": np.empty((0, 10, 3), dtype=np.float32)}
    params = transform.sample_parameters(
        *make_sampling_args(transform, data),
        SamplingContext.from_owner(transform, {}),
    ).params

    assert params["control_coefficients"] == []


def test_volume_uses_one_xy_map_for_every_slice() -> None:
    first = _image(32, 40)
    volume = np.stack([first, first], axis=0)

    transformed = A.ElasticTransform(p=1.0)(volume=volume)["volume"]

    np.testing.assert_array_equal(transformed[0], transformed[1])


def test_zero_range_is_exact_identity() -> None:
    image = _image()
    mask = np.arange(image.shape[0] * image.shape[1], dtype=np.uint8).reshape(image.shape[:2])
    result = A.Compose(
        [A.ElasticTransform(displacement_range=(0.0, 0.0), interpolation=cv2.INTER_CUBIC, p=1.0)],
        seed=137,
    )(image=image, mask=mask)

    np.testing.assert_array_equal(result["image"], image)
    np.testing.assert_array_equal(result["mask"], mask)


def test_zero_range_applies_active_label_mappings() -> None:
    image = _image()
    mask = np.resize(np.array([[2, 0, 3], [3, 2, 0]], dtype=np.uint8), image.shape[:2])
    expected_mask = mask.copy()
    expected_mask[mask == 2] = 3
    expected_mask[mask == 3] = 2
    keypoints = np.array([[5.0, 15.0], [10.0, 20.0]], dtype=np.float32)
    pipeline = A.Compose(
        [_LabelMappingElasticTransform(displacement_range=(0.0, 0.0), p=1.0)],
        keypoint_params=A.KeypointParams(
            coord_format="xy",
            label_fields=["keypoint_labels"],
            label_mapping={
                "HorizontalFlip": {
                    "keypoint_labels": {
                        "left_eye": "right_eye",
                        "right_eye": "left_eye",
                    },
                },
            },
        ),
        semantic_mask_label_mappings={"HorizontalFlip": {2: 3, 3: 2}},
        seed=137,
    )

    result = pipeline(
        image=image,
        mask=mask,
        keypoints=keypoints,
        keypoint_labels=["left_eye", "right_eye"],
    )

    np.testing.assert_array_equal(result["image"], image)
    np.testing.assert_array_equal(result["mask"], expected_mask)
    np.testing.assert_array_equal(result["keypoints"], keypoints[::-1])
    assert result["keypoint_labels"] == ["right_eye", "left_eye"]
