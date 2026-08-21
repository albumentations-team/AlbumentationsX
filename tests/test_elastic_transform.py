"""Focused contracts for the greenfield bounded ElasticTransform."""

import json

import cv2
import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.geometric import functional as fgeometric


def _image(height: int = 32, width: int = 40, channels: int = 3) -> np.ndarray:
    return np.random.default_rng(137).random((height, width, channels), dtype=np.float32)


def _forward(control: np.ndarray, point: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    rows, columns, _ = control.shape
    x_grid = np.linspace(0, width - 1, columns)
    y_grid = np.linspace(0, height - 1, rows)
    column = min(np.searchsorted(x_grid, point[0], side="right") - 1, columns - 2)
    row = min(np.searchsorted(y_grid, point[1], side="right") - 1, rows - 2)
    u = (point[0] - x_grid[column]) / (x_grid[column + 1] - x_grid[column])
    v = (point[1] - y_grid[row]) / (y_grid[row + 1] - y_grid[row])
    weights = np.array([(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v])
    displacement = weights @ control[row : row + 2, column : column + 2].reshape(4, 2)
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


@pytest.mark.parametrize(
    "legacy_name",
    [
        "alpha",
        "sigma",
        "approximate",
        "same_dxdy",
        "noise_distribution",
        "keypoint_remapping_method",
        "map_resolution_range",
    ],
)
def test_elastic_constructor_has_no_legacy_parameters(legacy_name: str) -> None:
    with pytest.warns(UserWarning, match="not valid for transform ElasticTransform"):
        transform = A.ElasticTransform(**{legacy_name: 1})
    assert not hasattr(transform, legacy_name)


def test_endpoint_aligned_control_grid_expansion() -> None:
    control = np.zeros((3, 4, 2), dtype=np.float32)
    control[..., 0] = np.arange(12, dtype=np.float32).reshape(3, 4)
    control[..., 1] = -control[..., 0]

    dense = fgeometric.expand_control_grid(control, (9, 13))

    np.testing.assert_array_equal(dense[:, 0, 0], control[0, 0])
    np.testing.assert_array_equal(dense[:, 0, -1], control[0, -1])
    np.testing.assert_array_equal(dense[:, -1, 0], control[-1, 0])
    np.testing.assert_array_equal(dense[:, -1, -1], control[-1, -1])
    assert dense.dtype == np.float32

    expected_center = np.array([5.5, -5.5], dtype=np.float32)
    np.testing.assert_allclose(dense[:, 4, 6], expected_center, atol=1e-6)


def test_sampled_control_vectors_are_bounded_and_replayable() -> None:
    image = _image(64, 48)
    transform = A.ReplayCompose(
        [A.ElasticTransform(displacement_range=(0.05, 0.05), control_grid_shape=(5, 5), p=1.0)],
        seed=137,
    )
    result = transform(image=image)
    params = result["replay"]["transforms"][0]["params"]
    vectors = np.asarray(params["control_vectors"], dtype=np.float32)
    radius = 0.05 * min(image.shape[0] - 1, image.shape[1] - 1)

    assert params["displacement_magnitude"] == 0.05
    assert "elastic_identity" not in params
    assert np.max(np.linalg.norm(vectors, axis=-1)) <= radius + 1e-5
    dense = fgeometric.expand_control_grid(vectors, image.shape[:2])
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
    assert "control_vectors" not in config
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
