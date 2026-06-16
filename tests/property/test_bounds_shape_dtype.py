"""Shape, dtype, bounds, and categorical-mask invariants."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given

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
