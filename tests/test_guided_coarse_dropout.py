"""Focused contract tests for :class:`GuidedCoarseDropout`."""

import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.dropout import functional as fdropout


def test_dropout_uses_top_level_binary_region_and_preserves_metadata() -> None:
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    region = np.zeros((16, 16), dtype=np.uint8)
    region[4:12, 4:12] = 1
    transform = A.Compose(
        [
            A.GuidedCoarseDropout(
                region_key="sal",
                num_holes_range=(8, 8),
                hole_height_range=(0.25, 0.25),
                hole_width_range=(0.25, 0.25),
                fill=0,
                p=1.0,
            ),
        ],
        seed=7,
        strict=True,
    )

    result = transform(image=image, sal=region)
    changed = np.any(result["image"] != image, axis=-1)

    assert changed.any()
    assert not changed[~region.astype(bool)].any()
    np.testing.assert_array_equal(result["sal"], region)


@pytest.mark.parametrize(
    "region",
    [
        np.ones((8, 8, 1), dtype=np.uint8),
        np.full((8, 8), 2, dtype=np.uint8),
        np.full((8, 8), np.nan, dtype=np.float32),
    ],
    ids=["three-dimensional", "non-binary", "non-finite"],
)
def test_dropout_region_must_be_binary_2d_array(region: np.ndarray) -> None:
    transform = A.Compose([A.GuidedCoarseDropout(p=1.0)], strict=True)

    with pytest.raises(ValueError, match="must be"):
        transform(image=np.zeros((8, 8, 3), dtype=np.uint8), dropout_region=region)


def test_dropout_region_is_required() -> None:
    transform = A.Compose([A.GuidedCoarseDropout(p=1.0)], strict=True)

    with pytest.raises(ValueError, match="requires"):
        transform(image=np.zeros((8, 8, 3), dtype=np.uint8))


@pytest.mark.parametrize("margin", [-0.1, (0.1, 0.1)])
def test_protection_margin_must_be_a_nonnegative_float(margin: object) -> None:
    with pytest.raises(ValueError, match="protection_margin"):
        A.GuidedCoarseDropout(protection_margin=margin)  # type: ignore[arg-type]


def test_protected_box_margin_is_clipped_at_image_border() -> None:
    image = np.full((12, 12, 3), 255, dtype=np.uint8)
    transform = A.Compose(
        [
            A.GuidedCoarseDropout(
                protected_bbox_labels=["cat"],
                protection_margin=0.5,
                num_holes_range=(40, 40),
                hole_height_range=(0.5, 0.5),
                hole_width_range=(0.5, 0.5),
                fill=0,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
        seed=5,
        strict=True,
    )

    result = transform(
        image=image,
        dropout_region=np.ones((12, 12), dtype=np.uint8),
        bboxes=[[0, 0, 4, 4]],
        labels=["cat"],
    )

    np.testing.assert_array_equal(result["image"][:6, :6], image[:6, :6])
    assert np.any(result["image"][6:] != image[6:])


def test_fully_protected_region_is_a_no_op() -> None:
    image = np.full((12, 12, 3), 255, dtype=np.uint8)
    transform = A.Compose(
        [A.GuidedCoarseDropout(protected_bbox_labels=[1], fill=0, p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
        strict=True,
    )

    result = transform(
        image=image,
        dropout_region=np.ones((12, 12), dtype=np.uint8),
        bboxes=[[0, 0, 12, 12]],
        labels=[1],
    )

    np.testing.assert_array_equal(result["image"], image)


def test_odd_hole_dimensions_are_preserved() -> None:
    holes = A.GuidedCoarseDropout._centers_to_holes(
        np.array([[5, 5]], dtype=np.int32),
        np.array([3], dtype=np.int32),
        np.array([5], dtype=np.int32),
        (12, 12),
    )

    np.testing.assert_array_equal(holes, np.array([[3, 4, 8, 7]], dtype=np.int32))


def test_mask_and_bboxes_follow_the_actual_dropout_mask() -> None:
    image = np.full((12, 12, 3), 255, dtype=np.uint8)
    mask = np.ones((12, 12), dtype=np.uint8)
    transform = A.Compose(
        [
            A.GuidedCoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(0.5, 0.5),
                hole_width_range=(0.5, 0.5),
                fill=0,
                fill_mask=7,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"], min_visibility=1.0),
        seed=3,
        strict=True,
    )

    result = transform(
        image=image,
        mask=mask,
        dropout_region=np.ones((12, 12), dtype=np.uint8),
        bboxes=[[0, 0, 12, 12]],
        labels=[1],
    )
    changed = np.any(result["image"] != image, axis=-1)

    np.testing.assert_array_equal(result["mask"] == 7, changed)
    assert len(result["bboxes"]) == 0


def test_random_uniform_uses_one_value_per_original_hole() -> None:
    image = np.zeros((2, 6, 3), dtype=np.uint8)
    holes = np.array([[0, 0, 2, 2], [4, 0, 6, 2]], dtype=np.int32)
    dropout_mask = np.zeros((2, 6), dtype=bool)
    dropout_mask[:, :2] = True
    dropout_mask[:, 4:] = True

    result = fdropout.fill_masked_holes(
        image,
        holes,
        dropout_mask,
        "random_uniform",
        np.random.default_rng(0),
    )

    np.testing.assert_array_equal(result[:, :2], np.broadcast_to(result[0, 0], result[:, :2].shape))
    np.testing.assert_array_equal(result[:, 4:], np.broadcast_to(result[0, 4], result[:, 4:].shape))
    assert not np.array_equal(result[0, 0], result[0, 4])
