"""Shape, dtype, bounds, and categorical-mask invariants."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import albumentations as A
from tests.property.strategies import image_and_mask_arrays, image_arrays

pytestmark = pytest.mark.property


@given(image=image_arrays(dtypes=(np.uint8, np.float32)))
def test_horizontal_flip_preserves_shape_and_dtype(image: np.ndarray) -> None:
    transform = A.Compose([A.HorizontalFlip(p=1.0)], strict=True)

    result = transform(image=image)["image"]

    assert result.shape == image.shape
    assert result.dtype == image.dtype


@given(data=image_and_mask_arrays())
def test_mask_values_remain_categorical_under_mask_path(data: tuple[np.ndarray, np.ndarray]) -> None:
    image, mask = data
    transform = A.Compose([A.HorizontalFlip(p=1.0)], strict=True)

    result = transform(image=image, mask=mask)

    assert set(np.unique(result["mask"])).issubset(set(np.unique(mask)))


@given(image=image_arrays())
def test_batch_images_keep_channel_last_shape(image: np.ndarray) -> None:
    images = np.stack([image, image], axis=0)
    transform = A.Compose([A.HorizontalFlip(p=1.0)], strict=True)

    result = transform(images=images)["images"]

    assert result.shape == images.shape
    assert result.dtype == images.dtype


def test_bboxes_remain_finite_and_bounded() -> None:
    image = np.zeros((32, 40, 3), dtype=np.uint8)
    bboxes = np.array([[4, 5, 20, 23], [12, 8, 30, 29]], dtype=np.float32)
    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", bbox_type="hbb"),
        strict=True,
    )

    result = np.asarray(transform(image=image, bboxes=bboxes)["bboxes"], dtype=np.float32)

    assert np.isfinite(result).all()
    assert (result[:, [0, 2]] >= 0).all()
    assert (result[:, [0, 2]] <= image.shape[1]).all()
    assert (result[:, [1, 3]] >= 0).all()
    assert (result[:, [1, 3]] <= image.shape[0]).all()


@given(height=st.integers(5, 20), width=st.integers(5, 20))
def test_crop_and_flip_preserve_subpixel_edges_and_labels(height: int, width: int) -> None:
    image = np.zeros((height, width), dtype=np.uint8)
    right, bottom = width - 1.5, height - 1.5
    corners = np.array([(0.5, 0.5), (right, 0.5), (right, bottom), (0.5, bottom)], dtype=np.float32)
    edges = np.array([(0.5, height / 2), (right, height / 2), (width / 2, 0.5), (width / 2, bottom)], dtype=np.float32)
    inward = np.array([(0.125, 0), (-0.125, 0), (0, 0.125), (0, -0.125)], dtype=np.float32)
    outside = edges - inward
    keypoints = np.concatenate((corners, outside, edges, edges + inward))
    point_labels = list(range(137, 137 + len(keypoints)))
    pipeline = A.Compose(
        [A.Crop(x_min=1, y_min=1, x_max=width - 1, y_max=height - 1, p=1), A.HorizontalFlip(p=1)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["box_labels"]),
        keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["point_labels"], label_mapping={}),
    )

    result = pipeline(
        image=image,
        keypoints=keypoints,
        point_labels=point_labels,
        bboxes=[(0.75, 0.75, 1.25, 1.25), (0.125, 1.25, 0.875, 2.25)],
        box_labels=["partial", "outside"],
    )

    output_width = width - 2
    visible = np.concatenate((corners, edges, edges + inward))
    np.testing.assert_array_equal(
        result["keypoints"], np.column_stack((output_width - visible[:, 0], visible[:, 1] - 1))
    )
    assert result["point_labels"] == point_labels[: len(corners)] + point_labels[len(corners) + len(outside) :]
    np.testing.assert_allclose(result["bboxes"], [[output_width - 0.25, 0, output_width, 0.25]], atol=1e-5)
    assert result["box_labels"] == ["partial"]
