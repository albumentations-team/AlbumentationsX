"""Composition invariants for additional targets and NoOp identity behavior."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given

import albumentations as A
from tests.property.strategies import image_and_mask_arrays, image_arrays

pytestmark = pytest.mark.property


@given(data=image_and_mask_arrays())
def test_additional_targets_receive_matching_semantics(data: tuple[np.ndarray, np.ndarray]) -> None:
    image, mask = data
    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        additional_targets={"paired_image": "image", "paired_mask": "mask"},
        strict=True,
    )

    result = transform(image=image, paired_image=image.copy(), mask=mask, paired_mask=mask.copy())

    np.testing.assert_array_equal(result["image"], result["paired_image"])
    np.testing.assert_array_equal(result["mask"], result["paired_mask"])


@given(image=image_arrays())
def test_noop_before_transform_preserves_result(image: np.ndarray) -> None:
    with_noop = A.Compose([A.NoOp(p=1.0), A.HorizontalFlip(p=1.0)], strict=True)
    without_noop = A.Compose([A.HorizontalFlip(p=1.0)], strict=True)

    np.testing.assert_array_equal(with_noop(image=image)["image"], without_noop(image=image)["image"])


@given(image=image_arrays())
def test_noop_after_transform_preserves_result(image: np.ndarray) -> None:
    with_noop = A.Compose([A.HorizontalFlip(p=1.0), A.NoOp(p=1.0)], strict=True)
    without_noop = A.Compose([A.HorizontalFlip(p=1.0)], strict=True)

    np.testing.assert_array_equal(with_noop(image=image)["image"], without_noop(image=image)["image"])
