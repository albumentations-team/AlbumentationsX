import random
from typing import Any

import numpy as np
import pytest
from deepdiff import DeepDiff

import albumentations as A
from albumentations.augmentations.mixing import functional as fmixing


def image_generator():
    yield {"image": np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)}


def complex_image_generator():
    height = 100
    width = 100
    yield {"image": (height, width)}


def complex_read_fn_image(x):
    return {"image": np.random.randint(0, 256, (x["image"][0], x["image"][1], 3), dtype=np.uint8)}


# Mock random.randint to produce consistent results
@pytest.fixture(autouse=True)
def mock_random(monkeypatch):
    def mock_randint(start, end):
        return start  # always return the start value for consistency in tests

    monkeypatch.setattr(random, "randint", mock_randint)


@pytest.mark.parametrize(
    "metadata, img_shape, expected_output",
    [
        (
            # Image + bbox without label + mask + mask_id + label_id + no offset
            {
                "image": np.ones((20, 20, 3), dtype=np.uint8) * 255,
                "bbox": [0.3, 0.3, 0.5, 0.5],
                "mask": np.ones((20, 20), dtype=np.uint8) * 127,
                "mask_id": 1,
                "bbox_id": 99,
            },
            (100, 100),
            {
                "overlay_image": np.ones((20, 20, 3), dtype=np.uint8) * 255,
                "overlay_mask": np.ones((20, 20), dtype=np.uint8) * 127,
                "offset": (30, 30),
                "mask_id": 1,
                "bbox": [30, 30, 50, 50, 99],
            },
        ),
        # Image + bbox with label + mask_id + no mask
        (
            {"image": np.ones((20, 20, 3), dtype=np.uint8) * 255, "bbox": [0.3, 0.3, 0.5, 0.5, 99], "mask_id": 1},
            (100, 100),
            {
                "overlay_image": np.ones((20, 20, 3), dtype=np.uint8) * 255,
                "overlay_mask": np.ones((20, 20), dtype=np.uint8),
                "offset": (30, 30),
                "mask_id": 1,
                "bbox": [30, 30, 50, 50, 99],
            },
        ),
        # Test case with triangular mask
        (
            {
                "image": np.ones((20, 20, 3), dtype=np.uint8) * 255,
                "bbox": [0, 0, 0.2, 0.2],
                "mask": np.tri(20, 20, dtype=np.uint8) * 127,
                "mask_id": 2,
                "bbox_id": 100,
            },
            (100, 100),
            {
                "overlay_image": np.ones((20, 20, 3), dtype=np.uint8) * 255,
                "overlay_mask": np.tri(20, 20, dtype=np.uint8) * 127,
                "offset": (0, 0),
                "mask_id": 2,
                "bbox": [0, 0, 20, 20, 100],
            },
        ),
        # Test case with overlay_image having the same size as img_shape
        (
            {
                "image": np.ones((100, 100, 3), dtype=np.uint8) * 255,
                "bbox": [0, 0, 1, 1],
                "mask": np.ones((100, 100), dtype=np.uint8) * 127,
                "mask_id": 3,
                "bbox_id": 101,
            },
            (100, 100),
            {
                "overlay_image": np.ones((100, 100, 3), dtype=np.uint8) * 255,
                "overlay_mask": np.ones((100, 100), dtype=np.uint8) * 127,
                "offset": (0, 0),
                "mask_id": 3,
                "bbox": [0, 0, 100, 100, 101],
            },
        ),
    ],
)
def test_preprocess_metadata(metadata: dict[str, Any], img_shape: tuple[int, int], expected_output: dict[str, Any]):
    result = A.OverlayElements.preprocess_metadata(metadata, img_shape, random.Random(0))

    assert DeepDiff(result, expected_output, ignore_type_in_groups=[(tuple, list)]) == {}


@pytest.mark.parametrize(
    "metadata, expected_output",
    [
        (
            {
                "image": np.ones((10, 10, 3), dtype=np.uint8) * 255,
                "bbox": [0.1, 0.2, 0.2, 0.3],
            },
            {
                "expected_overlay": np.ones((10, 10, 3), dtype=np.uint8) * 255,
                "expected_bbox": [10, 20, 20, 30],
            },
        ),
        (
            {
                "image": np.ones((10, 10, 3), dtype=np.uint8) * 255,
                "bbox": [0.3, 0.4, 0.4, 0.5],
                "label_id": 99,
            },
            {
                "expected_overlay": np.ones((10, 10, 3), dtype=np.uint8) * 255,
                "expected_bbox": [30, 40, 40, 50, 99],
            },
        ),
        (
            {
                "image": np.ones((10, 10, 3), dtype=np.uint8) * 255,
            },
            {
                "expected_overlay": np.ones((10, 10, 3), dtype=np.uint8) * 255,
                "expected_bbox": [0, 0, 10, 10],
            },
        ),
    ],
)
def test_end_to_end(metadata, expected_output):
    transform = A.Compose([A.OverlayElements(p=1)], strict=True)

    img = np.zeros((100, 100, 3), dtype=np.uint8)

    transformed = transform(image=img, overlay_metadata=metadata)

    expected_img = np.zeros((100, 100, 3), dtype=np.uint8)

    x_min, y_min, x_max, y_max = expected_output["expected_bbox"][:4]

    expected_img[y_min:y_max, x_min:x_max] = expected_output["expected_overlay"]

    if "bbox" in metadata:
        np.testing.assert_array_equal(transformed["image"], expected_img)
    else:
        assert expected_img.sum() == transformed["image"].sum()


# ============================================================
# CopyAndPaste tests
# ============================================================


def _make_object(
    fill: int,
    region: tuple[int, int, int, int],
    height: int = 100,
    width: int = 100,
) -> dict[str, Any]:
    """Helper: one object dict with image + mask for the given region."""
    img = np.full((height, width, 3), fill, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    y0, y1, x0, x1 = region
    mask[y0:y1, x0:x1] = 1
    return {"image": img, "mask": mask}


class TestCopyAndPasteBasic:
    """Basic copy-and-paste pasting behaviour."""

    def test_image_pixels_pasted(self):
        """Pasted region should contain donor pixels."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        obj = _make_object(200, (10, 30, 10, 30))
        transform = A.Compose([A.CopyAndPaste(p=1.0)], seed=137)
        result = transform(image=image, copy_paste_metadata=[obj])
        assert np.any(result["image"] > 0), "Donor pixels should appear in the result"

    def test_no_op_when_empty_list(self):
        """No crash and no change when metadata list is empty."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        original = image.copy()
        transform = A.Compose([A.CopyAndPaste(p=1.0)], seed=137)
        result = transform(image=image, copy_paste_metadata=[])
        np.testing.assert_array_equal(result["image"], original)

    def test_all_objects_pasted(self):
        """Every object in the list should be pasted."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        obj1 = _make_object(100, (10, 20, 10, 20))
        obj2 = _make_object(200, (60, 80, 60, 80))

        transform = A.Compose([A.CopyAndPaste(p=1.0)], seed=137)
        result = transform(image=image, copy_paste_metadata=[obj1, obj2])
        assert result["image"][15, 15, 0] == 100
        assert result["image"][70, 70, 0] == 200

    def test_donor_resized_to_target(self):
        """Object from a different-sized source image should be resized."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        obj = _make_object(137, (10, 30, 10, 30), height=50, width=50)

        transform = A.Compose([A.CopyAndPaste(p=1.0)], seed=137)
        result = transform(image=image, copy_paste_metadata=[obj])
        assert result["image"].shape == (100, 100, 3)

    def test_metadata_all_zero_masks_is_no_op(self):
        """Entries whose paste mask has no foreground after resize are skipped; all skipped => no paste."""
        image = np.full((100, 100, 3), 50, dtype=np.uint8)
        empty = {
            "image": np.zeros((100, 100, 3), dtype=np.uint8),
            "mask": np.zeros((100, 100), dtype=np.uint8),
        }
        transform = A.Compose([A.CopyAndPaste(p=1.0)], seed=137)
        result = transform(image=image.copy(), copy_paste_metadata=[empty])
        np.testing.assert_array_equal(result["image"], image)

    def test_empty_paste_mask_skipped_other_objects_still_paste(self):
        """Zero-only paste masks are dropped; remaining valid metadata still pastes."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        empty = {
            "image": np.zeros((100, 100, 3), dtype=np.uint8),
            "mask": np.zeros((100, 100), dtype=np.uint8),
        }
        good = _make_object(137, (40, 60, 40, 60))
        transform = A.Compose([A.CopyAndPaste(p=1.0)], seed=137)
        result = transform(image=image, copy_paste_metadata=[empty, good])
        assert np.any(result["image"] > 0)


class TestCopyAndPasteMasks:
    """Instance mask handling -- occlusion and appending."""

    def test_pasted_masks_appended(self):
        """Pasted instance masks should be appended to the output."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        primary_masks = np.zeros((1, 100, 100), dtype=np.uint8)
        primary_masks[0, 5:15, 5:15] = 1

        objs = [
            _make_object(137, (40, 60, 40, 60)),
            _make_object(137, (70, 90, 70, 90)),
        ]

        transform = A.Compose(
            [
                A.CopyAndPaste(min_visibility_after_paste=0.0, p=1.0),
            ],
            seed=137,
        )
        result = transform(image=image, masks=primary_masks, copy_paste_metadata=objs)
        assert result["masks"].shape[0] == 3

    def test_occluded_instance_removed(self):
        """Existing instance fully covered by pasted object should be removed."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        primary_masks = np.zeros((1, 100, 100), dtype=np.uint8)
        primary_masks[0, 20:40, 20:40] = 1

        obj = _make_object(137, (10, 50, 10, 50))

        transform = A.Compose(
            [
                A.CopyAndPaste(min_visibility_after_paste=0.5, p=1.0),
            ],
            seed=137,
        )
        result = transform(image=image, masks=primary_masks, copy_paste_metadata=[obj])
        assert result["masks"].shape[0] == 1

    def test_surviving_masks_erased_in_pasted_region(self):
        """Surviving masks should have zeros where pasted objects are placed."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        primary_masks = np.zeros((1, 100, 100), dtype=np.uint8)
        primary_masks[0, :, :] = 1

        obj = _make_object(137, (40, 60, 40, 60))

        transform = A.Compose(
            [
                A.CopyAndPaste(min_visibility_after_paste=0.0, p=1.0),
            ],
            seed=137,
        )
        result = transform(image=image, masks=primary_masks, copy_paste_metadata=[obj])
        surviving_mask = result["masks"][0]
        assert np.sum(surviving_mask[40:60, 40:60]) == 0

    def test_primary_masks_as_list_of_arrays(self):
        """CopyAndPaste should accept masks as a list of (H, W) arrays like other targets."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        m = np.zeros((100, 100), dtype=np.uint8)
        m[20:40, 20:40] = 1
        primary_masks_list = [m]

        obj = _make_object(137, (10, 50, 10, 50))

        transform = A.Compose(
            [
                A.CopyAndPaste(min_visibility_after_paste=0.5, p=1.0),
            ],
            seed=137,
        )
        result = transform(image=image, masks=primary_masks_list, copy_paste_metadata=[obj])
        assert result["masks"].shape[0] == 1


class TestCopyAndPasteBboxes:
    """Bounding box handling with label sync."""

    def test_pasted_bboxes_added(self):
        """Pasted bboxes should be appended to the output."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        primary_masks = np.zeros((1, 100, 100), dtype=np.uint8)
        primary_masks[0, 5:15, 5:15] = 1
        bboxes = np.array([[5, 5, 15, 15]], dtype=np.float32)
        class_labels = [1]

        obj = _make_object(137, (50, 70, 50, 70))
        obj["bbox"] = [50, 50, 70, 70]
        obj["bbox_labels"] = {"class_labels": 2}

        transform = A.Compose(
            [A.CopyAndPaste(min_visibility_after_paste=0.0, p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_labels"]),
            seed=137,
        )
        result = transform(
            image=image,
            masks=primary_masks,
            bboxes=bboxes,
            class_labels=class_labels,
            copy_paste_metadata=[obj],
        )
        assert len(result["bboxes"]) == 2
        assert len(result["class_labels"]) == len(result["bboxes"])

    def test_bboxes_derived_from_masks(self):
        """When object has no bbox, it should be derived from mask."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bboxes = np.array([[5, 5, 15, 15]], dtype=np.float32)
        class_labels = [1]

        obj = _make_object(137, (50, 70, 50, 70))
        obj["bbox_labels"] = {"class_labels": 2}

        transform = A.Compose(
            [A.CopyAndPaste(min_visibility_after_paste=0.0, p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_labels"]),
            seed=137,
        )
        result = transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels,
            copy_paste_metadata=[obj],
        )
        assert len(result["bboxes"]) == 2

    def test_derived_bbox_matches_yolo_coord_format(self):
        """Mask-derived bbox must match the pipeline coord_format (not raw Pascal pixels)."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bboxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        class_labels = [1]

        obj = _make_object(137, (50, 70, 50, 70))
        obj["bbox_labels"] = {"class_labels": 2}

        transform = A.Compose(
            [A.CopyAndPaste(min_visibility_after_paste=0.0, p=1.0)],
            bbox_params=A.BboxParams(coord_format="yolo", label_fields=["class_labels"]),
            seed=137,
        )
        result = transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels,
            copy_paste_metadata=[obj],
        )
        assert len(result["bboxes"]) == 2
        pasted = result["bboxes"][1]
        assert np.all((pasted[:4] > 0) & (pasted[:4] <= 1.0))

    def test_occluded_bbox_removed_with_mask(self):
        """Bbox of fully occluded instance should be removed when masks are present."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        primary_masks = np.zeros((1, 100, 100), dtype=np.uint8)
        primary_masks[0, 20:40, 20:40] = 1
        bboxes = np.array([[20, 20, 40, 40]], dtype=np.float32)
        class_labels = [1]

        obj = _make_object(137, (10, 50, 10, 50))
        obj["bbox"] = [10, 10, 50, 50]
        obj["bbox_labels"] = {"class_labels": 2}

        transform = A.Compose(
            [A.CopyAndPaste(min_visibility_after_paste=0.5, p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_labels"]),
            seed=137,
        )
        result = transform(
            image=image,
            masks=primary_masks,
            bboxes=bboxes,
            class_labels=class_labels,
            copy_paste_metadata=[obj],
        )
        assert len(result["bboxes"]) == 1
        assert result["class_labels"] == [2]


class TestCopyAndPasteKeypoints:
    """Keypoint rows aligned with instance masks must follow paste survivor filtering."""

    def test_surviving_keypoints_match_instance_mask_indices(self):
        """When one of two instance masks is removed, keep only keypoints for surviving instances."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        primary_masks = np.zeros((2, 100, 100), dtype=np.uint8)
        primary_masks[0, 20:40, 20:40] = 1
        primary_masks[1, 60:80, 60:80] = 1
        keypoints = np.array([[30.0, 30.0], [70.0, 70.0]], dtype=np.float32)
        keypoint_labels = [0, 1]

        obj = _make_object(137, (10, 50, 10, 50))

        transform = A.Compose(
            [A.CopyAndPaste(min_visibility_after_paste=0.5, p=1.0)],
            keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["keypoint_labels"]),
            seed=137,
        )
        result = transform(
            image=image,
            masks=primary_masks,
            keypoints=keypoints,
            keypoint_labels=keypoint_labels,
            copy_paste_metadata=[obj],
        )
        np.testing.assert_array_equal(result["keypoints"], np.array([[70.0, 70.0]], dtype=np.float32))
        assert result["keypoint_labels"] == [1]

    def test_keypoints_preserved_when_row_count_not_instance_count(self):
        """If keypoint rows are not one-per-instance, do not drop rows using mask survivor indices."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        primary_masks = np.zeros((2, 100, 100), dtype=np.uint8)
        primary_masks[0, 20:40, 20:40] = 1
        primary_masks[1, 60:80, 60:80] = 1
        keypoints = np.array([[10.0, 10.0], [30.0, 30.0], [70.0, 70.0], [75.0, 75.0]], dtype=np.float32)
        keypoint_labels = [0, 1, 2, 3]

        obj = _make_object(137, (10, 50, 10, 50))

        transform = A.Compose(
            [A.CopyAndPaste(min_visibility_after_paste=0.5, p=1.0)],
            keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["keypoint_labels"]),
            seed=137,
        )
        result = transform(
            image=image,
            masks=primary_masks,
            keypoints=keypoints,
            keypoint_labels=keypoint_labels,
            copy_paste_metadata=[obj],
        )
        assert len(result["keypoints"]) == 4
        assert result["keypoint_labels"] == [0, 1, 2, 3]


class TestCopyAndPasteBlending:
    """Blend mode tests."""

    def test_hard_blend_exact_copy(self):
        """Hard blend should exactly copy donor pixels in pasted region."""
        base = np.zeros((50, 50, 3), dtype=np.uint8)
        donor = np.full((50, 50, 3), 137, dtype=np.uint8)
        alpha = np.zeros((50, 50), dtype=np.float32)
        alpha[10:20, 10:20] = 1.0

        result = fmixing.blend_images_using_alpha(base, donor, alpha)
        np.testing.assert_array_equal(result[10:20, 10:20], 137)
        np.testing.assert_array_equal(result[0:10, 0:10], 0)

    def test_soft_blend_float_high_range_clip(self):
        """Float images in ~[0, 255] should clip to 255, not 1.0."""
        base = np.zeros((20, 20, 3), dtype=np.float32)
        donor = np.full((20, 20, 3), 200.0, dtype=np.float32)
        alpha = np.zeros((20, 20), dtype=np.float32)
        alpha[5:15, 5:15] = 0.5

        result = fmixing.blend_images_using_alpha(base, donor, alpha)
        assert result.dtype == np.float32
        assert float(np.max(result)) > 1.0

    def test_gaussian_blend_smooth_edges(self):
        """Gaussian blend should produce smooth transitions at edges."""
        masks = np.zeros((1, 50, 50), dtype=np.uint8)
        masks[0, 15:35, 15:35] = 1
        alpha = fmixing.create_copy_paste_alpha(masks, "gaussian", 3.0)

        assert alpha.max() <= 1.0
        assert alpha.min() >= 0.0
        has_intermediate = np.any((alpha > 0.01) & (alpha < 0.99))
        assert has_intermediate, "Gaussian alpha should have intermediate values"

    def test_hard_alpha_binary(self):
        """Hard blend alpha should be strictly 0 or 1."""
        masks = np.zeros((1, 50, 50), dtype=np.uint8)
        masks[0, 15:35, 15:35] = 1
        alpha = fmixing.create_copy_paste_alpha(masks, "hard", 0.0)

        unique_values = np.unique(alpha)
        for val in unique_values:
            assert val in {0.0, 1.0}

    def test_gaussian_blend_compose(self):
        """Full pipeline with gaussian blending should not crash."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        obj = _make_object(137, (30, 70, 30, 70))

        transform = A.Compose(
            [
                A.CopyAndPaste(blend_mode="gaussian", blend_sigma_range=(2.0, 2.0), p=1.0),
            ],
            seed=137,
        )
        result = transform(image=image, copy_paste_metadata=[obj])
        assert result["image"].shape == (100, 100, 3)


class TestCopyAndPasteFunctional:
    """Tests for functional layer helpers."""

    def test_compute_visibility_no_occlusion(self):
        """No occlusion means visibility = 1.0."""
        masks = np.zeros((2, 50, 50), dtype=np.uint8)
        masks[0, 5:15, 5:15] = 1
        masks[1, 30:40, 30:40] = 1
        alpha = np.zeros((50, 50), dtype=np.float32)

        visibility = fmixing.compute_instance_visibility(masks, alpha)
        np.testing.assert_allclose(visibility, [1.0, 1.0])

    def test_compute_visibility_full_occlusion(self):
        """Full occlusion means visibility = 0.0."""
        masks = np.zeros((1, 50, 50), dtype=np.uint8)
        masks[0, 10:20, 10:20] = 1
        alpha = np.zeros((50, 50), dtype=np.float32)
        alpha[5:25, 5:25] = 1.0

        visibility = fmixing.compute_instance_visibility(masks, alpha)
        np.testing.assert_allclose(visibility, [0.0])

    def test_compute_visibility_partial_occlusion(self):
        """Partial occlusion gives intermediate visibility."""
        masks = np.zeros((1, 50, 50), dtype=np.uint8)
        masks[0, 10:20, 10:20] = 1
        alpha = np.zeros((50, 50), dtype=np.float32)
        alpha[15:20, 10:20] = 1.0

        visibility = fmixing.compute_instance_visibility(masks, alpha)
        np.testing.assert_allclose(visibility, [0.5])

    def test_compute_visibility_empty_mask(self):
        """Empty mask should have visibility 1.0 (nothing to occlude)."""
        masks = np.zeros((1, 50, 50), dtype=np.uint8)
        alpha = np.ones((50, 50), dtype=np.float32)

        visibility = fmixing.compute_instance_visibility(masks, alpha)
        np.testing.assert_allclose(visibility, [1.0])

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    def test_blend_preserves_dtype(self, dtype: np.dtype):
        """Blend should preserve the input image dtype."""
        if dtype == np.uint8:
            base = np.zeros((50, 50, 3), dtype=dtype)
            donor = np.full((50, 50, 3), 137, dtype=dtype)
        else:
            base = np.zeros((50, 50, 3), dtype=dtype)
            donor = np.full((50, 50, 3), 0.5, dtype=dtype)
        alpha = np.zeros((50, 50), dtype=np.float32)
        alpha[10:20, 10:20] = 1.0

        result = fmixing.blend_images_using_alpha(base, donor, alpha)
        assert result.dtype == dtype


class TestCopyAndPasteEdgeCases:
    """Edge cases and robustness."""

    def test_single_channel_image(self):
        """Should work with single-channel images."""
        image = np.zeros((100, 100, 1), dtype=np.uint8)
        img_src = np.full((100, 100, 1), 137, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:60, 30:60] = 1

        transform = A.Compose([A.CopyAndPaste(p=1.0)], seed=137)
        result = transform(image=image, copy_paste_metadata=[{"image": img_src, "mask": mask}])
        assert result["image"].shape == (100, 100, 1)

    def test_float32_image(self):
        """Should work with float32 images."""
        image = np.zeros((100, 100, 3), dtype=np.float32)
        img_src = np.full((100, 100, 3), 0.5, dtype=np.float32)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:60, 30:60] = 1

        transform = A.Compose([A.CopyAndPaste(p=1.0)], seed=137)
        result = transform(image=image, copy_paste_metadata=[{"image": img_src, "mask": mask}])
        assert result["image"].dtype == np.float32

    def test_no_masks_in_primary(self):
        """When primary has no masks, should still paste pixels and add bboxes."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bboxes = np.array([[5, 5, 15, 15]], dtype=np.float32)
        class_labels = [1]

        obj = _make_object(137, (50, 70, 50, 70))
        obj["bbox"] = [50, 50, 70, 70]
        obj["bbox_labels"] = {"class_labels": 2}

        transform = A.Compose(
            [A.CopyAndPaste(p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_labels"]),
            seed=137,
        )
        result = transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels,
            copy_paste_metadata=[obj],
        )
        assert len(result["bboxes"]) == 2
        assert np.any(result["image"] > 0)


class TestUnpackLabelWrappers:
    """Tests for `unpack_label_wrappers` reserved-key handling."""

    def test_no_wrapper_keys_returns_input_unchanged(self):
        item = {"image": np.zeros((4, 4, 3), dtype=np.uint8), "class_labels": [1, 2]}
        result = fmixing.unpack_label_wrappers(item)
        assert result is item

    def test_wrapper_dict_is_flattened(self):
        item = {
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
            "bbox_labels": {"class_labels": [1, 2]},
            "keypoint_labels": {"kp_classes": ["eye"]},
        }
        result = fmixing.unpack_label_wrappers(item)
        assert "bbox_labels" not in result
        assert "keypoint_labels" not in result
        assert result["class_labels"] == [1, 2]
        assert result["kp_classes"] == ["eye"]

    def test_none_wrapper_value_is_skipped(self):
        item = {"image": np.zeros((4, 4, 3), dtype=np.uint8), "bbox_labels": None}
        result = fmixing.unpack_label_wrappers(item)
        assert "bbox_labels" not in result

    @pytest.mark.parametrize(
        ("wrapper_key", "bad_value"),
        [
            ("bbox_labels", [1, 2, 3]),
            ("bbox_labels", (1, 2)),
            ("keypoint_labels", ["eye", "nose"]),
            ("keypoint_labels", np.array([1, 2])),
        ],
    )
    def test_non_dict_wrapper_raises_typeerror(self, wrapper_key, bad_value):
        item = {"image": np.zeros((4, 4, 3), dtype=np.uint8), wrapper_key: bad_value}
        with pytest.raises(TypeError, match=f"`{wrapper_key}` must be a dict"):
            fmixing.unpack_label_wrappers(item)
