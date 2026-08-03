"""Tests for Compose additional target source requirements and routing."""

from __future__ import annotations

import numpy as np
import pytest

import albumentations as A


@pytest.mark.parametrize(
    ("additional_targets", "data", "alias", "target"),
    [
        (
            {"image2": "image"},
            {"image2": np.zeros((4, 5, 3), dtype=np.uint8)},
            "image2",
            "image",
        ),
        (
            {"mask2": "mask"},
            {
                "image": np.zeros((4, 5, 3), dtype=np.uint8),
                "mask2": np.zeros((4, 5), dtype=np.uint8),
            },
            "mask2",
            "mask",
        ),
        (
            {"scan": "volume"},
            {"scan": np.zeros((3, 4, 5, 1), dtype=np.uint8)},
            "scan",
            "volume",
        ),
        (
            {"segmentation": "mask3d"},
            {"segmentation": np.zeros((3, 4, 5), dtype=np.uint8)},
            "segmentation",
            "mask3d",
        ),
    ],
)
def test_additional_target_requires_canonical_source(
    additional_targets: dict[str, str],
    data: dict[str, np.ndarray],
    alias: str,
    target: str,
) -> None:
    transform = A.Compose([A.NoOp(p=1.0)], additional_targets=additional_targets, strict=True)

    with pytest.raises(
        ValueError,
        match=rf"Additional target '{alias}' requires canonical target '{target}' to be present",
    ):
        transform(**data)


def test_unused_additional_target_does_not_require_its_source() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    transform = A.Compose([A.NoOp(p=1.0)], additional_targets={"image2": "image"}, strict=True)

    result = transform(image=image)

    assert result["image"] is image


def test_additional_image_is_transformed_with_its_canonical_source() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(3, 4, 1)
    image2 = np.arange(12, 24, dtype=np.uint8).reshape(3, 4, 1)
    transform = A.Compose([A.HorizontalFlip(p=1.0)], additional_targets={"image2": "image"})

    result = transform(image=image, image2=image2)

    np.testing.assert_array_equal(result["image"], image[:, ::-1])
    np.testing.assert_array_equal(result["image2"], image2[:, ::-1])


def test_grayscale_additional_image_restores_its_original_shape() -> None:
    image = np.zeros((3, 4, 1), dtype=np.uint8)
    image2 = np.arange(12, dtype=np.uint8).reshape(3, 4)
    transform = A.Compose([A.HorizontalFlip(p=1.0)], additional_targets={"image2": "image"})

    result = transform(image=image, image2=image2)

    assert result["image2"].shape == image2.shape
    np.testing.assert_array_equal(result["image2"], image2[:, ::-1])
