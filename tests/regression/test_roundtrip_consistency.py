"""Round-trip and consistency regression checks."""

from __future__ import annotations

import numpy as np
import pytest

import albumentations as A
from tests.helpers import TestDataFactory

pytestmark = pytest.mark.regression


@pytest.mark.parametrize("transform_cls", [A.HorizontalFlip, A.VerticalFlip, A.Transpose])
def test_dual_transform_applied_twice_returns_original(transform_cls: type[A.DualTransform]) -> None:
    image = TestDataFactory.create_image((32, 40, 3), dtype=np.uint8)
    mask = TestDataFactory.create_mask((32, 40))
    bboxes = np.array([[4, 5, 20, 23], [12, 8, 30, 29]], dtype=np.float32)
    bbox_labels = ["left", "right"]
    keypoints = np.array([[8, 9], [22, 18]], dtype=np.float32)
    keypoint_labels = ["a", "b"]

    transform = A.Compose(
        [transform_cls(p=1.0), transform_cls(p=1.0)],
        bbox_params=A.BboxParams(
            coord_format="pascal_voc",
            label_fields=["bbox_labels"],
            bbox_type="hbb",
        ),
        keypoint_params=A.KeypointParams(
            coord_format="xy",
            label_fields=["keypoint_labels"],
            remove_invisible=False,
            label_mapping={},
        ),
        strict=True,
    )

    result = transform(
        image=image,
        mask=mask,
        bboxes=bboxes,
        bbox_labels=bbox_labels,
        keypoints=keypoints,
        keypoint_labels=keypoint_labels,
    )

    np.testing.assert_array_equal(result["image"], image)
    np.testing.assert_array_equal(result["mask"], mask)
    np.testing.assert_allclose(np.asarray(result["bboxes"], dtype=np.float32), bboxes, atol=1e-5)
    np.testing.assert_allclose(np.asarray(result["keypoints"], dtype=np.float32), keypoints, atol=1e-5)
    assert result["bbox_labels"] == bbox_labels
    assert result["keypoint_labels"] == keypoint_labels


def test_replay_compose_replays_same_output() -> None:
    image = TestDataFactory.create_image((24, 26, 3), dtype=np.uint8)
    replay_transform = A.ReplayCompose([A.HorizontalFlip(p=1.0), A.RandomRotate90(p=1.0)], seed=137)

    first = replay_transform(image=image)
    replayed = A.ReplayCompose.replay(first["replay"], image=image)

    np.testing.assert_array_equal(first["image"], replayed["image"])


def test_to_dict_from_dict_preserves_seeded_behavior() -> None:
    image = TestDataFactory.create_image((24, 26, 3), dtype=np.uint8)
    transform = A.Compose([A.HorizontalFlip(p=1.0), A.RandomRotate90(p=1.0)], seed=137, strict=True)
    restored = A.from_dict(A.to_dict(transform))

    np.testing.assert_array_equal(transform(image=image)["image"], restored(image=image)["image"])


def test_additional_targets_follow_image_and_mask_semantics() -> None:
    image = TestDataFactory.create_image((18, 22, 3), dtype=np.uint8)
    paired_image = image.copy()
    mask = TestDataFactory.create_mask((18, 22))
    paired_mask = mask.copy()
    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        additional_targets={"paired_image": "image", "paired_mask": "mask"},
        strict=True,
    )

    result = transform(image=image, paired_image=paired_image, mask=mask, paired_mask=paired_mask)

    np.testing.assert_array_equal(result["image"], result["paired_image"])
    np.testing.assert_array_equal(result["mask"], result["paired_mask"])
