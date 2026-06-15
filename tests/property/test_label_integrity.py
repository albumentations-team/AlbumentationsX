"""Annotation label integrity invariants."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import albumentations as A

pytestmark = pytest.mark.property


@given(width=st.integers(32, 96), height=st.integers(32, 96))
def test_bbox_labels_stay_paired_with_surviving_boxes(width: int, height: int) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    bboxes = np.array(
        [
            [2, 3, width // 3, height // 2],
            [width // 2, height // 3, width - 2, height - 2],
        ],
        dtype=np.float32,
    )
    labels = ["first", "second"]
    scores = [0.5, 0.9]
    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(
            coord_format="pascal_voc",
            label_fields=["bbox_labels", "bbox_scores"],
            bbox_type="hbb",
        ),
        strict=True,
    )

    result = transform(image=image, bboxes=bboxes, bbox_labels=labels, bbox_scores=scores)

    assert len(result["bboxes"]) == len(result["bbox_labels"]) == len(result["bbox_scores"])
    assert result["bbox_labels"] == labels
    np.testing.assert_allclose(result["bbox_scores"], scores, atol=1e-6)


def test_hbb_extra_columns_do_not_define_bbox_type() -> None:
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    bboxes = np.array([[4, 5, 20, 23, 101, 202], [12, 8, 30, 29, 303, 404]], dtype=np.float32)
    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", bbox_type="hbb"),
        strict=True,
    )

    result = np.asarray(transform(image=image, bboxes=bboxes)["bboxes"], dtype=np.float32)

    assert result.shape[1] == bboxes.shape[1]
    np.testing.assert_array_equal(result[:, 4:], bboxes[:, 4:])


def test_keypoint_labels_stay_paired() -> None:
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    keypoints = np.array([[6, 8], [22, 18]], dtype=np.float32)
    labels = ["left", "right"]
    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        keypoint_params=A.KeypointParams(
            coord_format="xy",
            label_fields=["keypoint_labels"],
            remove_invisible=False,
            label_mapping={},
        ),
        strict=True,
    )

    result = transform(image=image, keypoints=keypoints, keypoint_labels=labels)

    assert len(result["keypoints"]) == len(result["keypoint_labels"])
    assert result["keypoint_labels"] == labels
