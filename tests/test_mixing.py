import random
import warnings
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
    """Helper: one donor dict with image + mask. The mask region defines the tight crop the
    transform pastes; the donor is then randomly placed inside the target. For deterministic
    full-target pastes (e.g. occlusion tests) use `_make_full_cover_object` instead.
    """
    img = np.full((height, width, 3), fill, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    y0, y1, x0, x1 = region
    mask[y0:y1, x0:x1] = 1
    return {"image": img, "mask": mask}


def _make_full_cover_object(fill: int, target_shape: tuple[int, int] = (100, 100)) -> dict[str, Any]:
    """Helper: donor whose tight crop equals target dims, so the new transform places it at (0, 0)
    and the paste covers the whole target deterministically. Use for occlusion tests.
    """
    h, w = target_shape
    img = np.full((h, w, 3), fill, dtype=np.uint8)
    mask = np.ones((h, w), dtype=np.uint8)
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
        """Every object in the list should be pasted somewhere on the target (random placement)."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        obj1 = _make_object(100, (10, 20, 10, 20))
        obj2 = _make_object(200, (60, 80, 60, 80))

        transform = A.Compose([A.CopyAndPaste(p=1.0)], seed=137)
        result = transform(image=image, copy_paste_metadata=[obj1, obj2])
        unique = set(np.unique(result["image"]).tolist())
        assert 100 in unique
        assert 200 in unique

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
        """Existing instance fully covered by a full-cover pasted donor should be removed
        (full-cover donor pastes deterministically over the whole target).
        """
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        primary_masks = np.zeros((1, 100, 100), dtype=np.uint8)
        primary_masks[0, 20:40, 20:40] = 1

        obj = _make_full_cover_object(137)

        transform = A.Compose(
            [
                A.CopyAndPaste(min_visibility_after_paste=0.5, p=1.0),
            ],
            seed=137,
        )
        result = transform(image=image, masks=primary_masks, copy_paste_metadata=[obj])
        assert result["masks"].shape[0] == 1

    def test_surviving_masks_erased_in_pasted_region(self):
        """Surviving masks should have zeros where pasted objects are placed.

        Locate the actual paste union (random placement) and assert it was zeroed in the survivor.
        """
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
        paste_region = np.any(result["image"] > 0, axis=-1)
        assert paste_region.sum() > 0
        assert np.sum(surviving_mask[paste_region]) == 0

    def test_primary_masks_as_list_of_arrays(self):
        """CopyAndPaste should accept masks as a list of (H, W) arrays like other targets."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        m = np.zeros((100, 100), dtype=np.uint8)
        m[20:40, 20:40] = 1
        primary_masks_list = [m]

        obj = _make_full_cover_object(137)

        transform = A.Compose(
            [
                A.CopyAndPaste(min_visibility_after_paste=0.5, p=1.0),
            ],
            seed=137,
        )
        result = transform(image=image, masks=primary_masks_list, copy_paste_metadata=[obj])
        assert result["masks"].shape[0] == 1

    def test_warns_when_mask_target_present_but_no_semantic_mask(self):
        """If a `mask` target is in the pipeline but no donor item provides `semantic_mask`,
        a UserWarning should fire so users notice the silent no-op on the semantic mask.
        """
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        primary_mask = np.zeros((100, 100), dtype=np.uint8)
        primary_mask[10:30, 10:30] = 5

        obj = _make_object(137, (40, 60, 40, 60))

        transform = A.Compose([A.CopyAndPaste(p=1.0)], seed=137)
        with pytest.warns(UserWarning, match="no donor item provided `semantic_mask`"):
            result = transform(image=image, mask=primary_mask, copy_paste_metadata=[obj])
        np.testing.assert_array_equal(result["mask"], primary_mask)

    def test_no_warning_when_semantic_mask_provided(self):
        """Donors that provide `semantic_mask` should suppress the warning and stamp class ids
        at the (random) paste position. We locate that position from the result, not assume it.
        """
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        primary_mask = np.zeros((100, 100), dtype=np.uint8)

        obj = _make_full_cover_object(137)
        obj["semantic_mask"] = np.full((100, 100), 7, dtype=np.uint8)

        transform = A.Compose([A.CopyAndPaste(p=1.0)], seed=137)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = transform(image=image, mask=primary_mask, copy_paste_metadata=[obj])
        assert (result["mask"] == 7).all()

    def test_no_warning_when_no_mask_target(self):
        """Without a `mask` target the warning must stay silent even if no donor has `semantic_mask`."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        obj = _make_object(137, (40, 60, 40, 60))

        transform = A.Compose([A.CopyAndPaste(p=1.0)], seed=137)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            transform(image=image, copy_paste_metadata=[obj])


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
        """Bbox of fully occluded instance should be removed when masks are present.

        Uses a full-cover donor that pastes deterministically over the entire target so the primary
        instance is guaranteed to be fully occluded regardless of the random placement RNG.
        """
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        primary_masks = np.zeros((1, 100, 100), dtype=np.uint8)
        primary_masks[0, 20:40, 20:40] = 1
        bboxes = np.array([[20, 20, 40, 40]], dtype=np.float32)
        class_labels = [1]

        obj = _make_full_cover_object(137)
        obj["bbox"] = [0, 0, 100, 100]
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
        """When all instance masks are occluded, all corresponding keypoints are removed.

        Uses a full-cover donor that deterministically pastes over the whole target so both
        primary instance masks are fully occluded regardless of placement RNG.
        """
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        primary_masks = np.zeros((2, 100, 100), dtype=np.uint8)
        primary_masks[0, 20:40, 20:40] = 1
        primary_masks[1, 60:80, 60:80] = 1
        keypoints = np.array([[30.0, 30.0], [70.0, 70.0]], dtype=np.float32)
        keypoint_labels = [0, 1]

        obj = _make_full_cover_object(137)

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
        assert len(result["keypoints"]) == 0
        assert result["keypoint_labels"] == []

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


class TestCopyAndPasteRandomPlacement:
    """Random placement, shrink-to-fit, aspect preservation."""

    def test_donor_bigger_than_target_is_shrunk(self):
        """A donor whose tight crop is bigger than the target gets shrunk to fit; the paste
        footprint area cannot exceed the target area.
        """
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        donor_image = np.full((200, 200, 3), 137, dtype=np.uint8)
        donor_mask = np.ones((200, 200), dtype=np.uint8)

        transform = A.Compose([A.CopyAndPaste(p=1.0)], seed=137)
        result = transform(
            image=image,
            copy_paste_metadata=[{"image": donor_image, "mask": donor_mask}],
        )
        paste_region = np.any(result["image"] > 0, axis=-1)
        assert paste_region.sum() <= 100 * 100
        assert paste_region.sum() > 0

    def test_donor_aspect_preserved_when_shrunk(self):
        """Aspect ratio of the donor's tight crop is preserved by the shrink-to-fit step."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # 200x100 donor (aspect 2:1), full mask -> shrink-to-fit gives 100x50 (aspect preserved).
        donor_image = np.full((200, 100, 3), 137, dtype=np.uint8)
        donor_mask = np.ones((200, 100), dtype=np.uint8)

        transform = A.Compose([A.CopyAndPaste(p=1.0)], seed=137)
        result = transform(
            image=image,
            copy_paste_metadata=[{"image": donor_image, "mask": donor_mask}],
        )
        paste_region = np.any(result["image"] > 0, axis=-1)
        rows = np.any(paste_region, axis=1)
        cols = np.any(paste_region, axis=0)
        h = int(rows.sum())
        w = int(cols.sum())
        assert h == 100
        assert w == 50

    def test_no_upscaling_for_small_donor(self):
        """A donor smaller than the target keeps its tight-crop size (no upscaling)."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        donor_image = np.full((30, 30, 3), 137, dtype=np.uint8)
        donor_mask = np.ones((30, 30), dtype=np.uint8)

        transform = A.Compose([A.CopyAndPaste(p=1.0)], seed=137)
        result = transform(
            image=image,
            copy_paste_metadata=[{"image": donor_image, "mask": donor_mask}],
        )
        paste_region = np.any(result["image"] > 0, axis=-1)
        assert int(paste_region.sum()) == 30 * 30

    def test_random_placement_is_random(self):
        """Two different RNG seeds should produce different placements for the same donor."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        donor_image = np.full((20, 20, 3), 137, dtype=np.uint8)
        donor_mask = np.ones((20, 20), dtype=np.uint8)

        def _placement(seed: int) -> tuple[int, int]:
            transform = A.Compose([A.CopyAndPaste(p=1.0)], seed=seed)
            result = transform(
                image=image.copy(),
                copy_paste_metadata=[{"image": donor_image, "mask": donor_mask}],
            )
            paste_region = np.any(result["image"] > 0, axis=-1)
            ys, xs = np.where(paste_region)
            return int(ys.min()), int(xs.min())

        # Try a small bag of seeds and assert at least one differs from seed 0.
        positions = {_placement(s) for s in range(8)}
        assert len(positions) > 1, "random placement should produce >1 distinct positions across seeds"

    def test_paste_stays_inside_target(self):
        """Random placement must keep the paste footprint fully inside the target image."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        donor_image = np.full((37, 41, 3), 137, dtype=np.uint8)
        donor_mask = np.ones((37, 41), dtype=np.uint8)

        for seed in range(20):
            transform = A.Compose([A.CopyAndPaste(p=1.0)], seed=seed)
            result = transform(
                image=image.copy(),
                copy_paste_metadata=[{"image": donor_image, "mask": donor_mask}],
            )
            paste_region = np.any(result["image"] > 0, axis=-1)
            ys, xs = np.where(paste_region)
            assert ys.min() >= 0 and ys.max() < 100
            assert xs.min() >= 0 and xs.max() < 100


class TestCopyAndPasteBboxOnlyDonor:
    """Donors that supply only a `bbox` (no instance mask) — rectangle paste footprint."""

    @pytest.mark.parametrize("coord_format", ["pascal_voc", "yolo"])
    def test_bbox_only_donor_pastes_rectangle(self, coord_format: str):
        """Bbox-only donor should paste a rectangular footprint matching the bbox dims and the
        returned bbox should round-trip through the pipeline `coord_format`.
        """
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Donor is 60x80 (Hd, Wd); bbox covers a 30x40 (h, w) rectangle.
        donor_image = np.full((60, 80, 3), 137, dtype=np.uint8)
        if coord_format == "pascal_voc":
            donor_bbox: list[float] = [10.0, 5.0, 50.0, 35.0]
        else:  # yolo: cx, cy, w, h normalized to donor (Hd=60, Wd=80)
            donor_bbox = [(10 + 50) / 2 / 80, (5 + 35) / 2 / 60, (50 - 10) / 80, (35 - 5) / 60]

        transform = A.Compose(
            [A.CopyAndPaste(p=1.0)],
            bbox_params=A.BboxParams(coord_format=coord_format, label_fields=["class_labels"]),
            seed=137,
        )
        result = transform(
            image=image,
            bboxes=np.zeros((0, 4), dtype=np.float32),
            class_labels=[],
            copy_paste_metadata=[
                {
                    "image": donor_image,
                    "bbox": donor_bbox,
                    "bbox_labels": {"class_labels": 2},
                },
            ],
        )
        paste_region = np.any(result["image"] > 0, axis=-1)
        rows = np.any(paste_region, axis=1)
        cols = np.any(paste_region, axis=0)
        # Paste footprint should match the donor bbox dims (40w x 30h pixels) since both fit in target.
        # Allow ±1 px slack for non-pixel-aligned coord formats (yolo normalization).
        assert abs(int(rows.sum()) - 30) <= 1
        assert abs(int(cols.sum()) - 40) <= 1
        assert len(result["bboxes"]) == 1
        out_bbox = np.asarray(result["bboxes"][0], dtype=np.float32)
        if coord_format == "pascal_voc":
            x_min, y_min, x_max, y_max = out_bbox[:4]
            assert abs((x_max - x_min) - 40) <= 1.0
            assert abs((y_max - y_min) - 30) <= 1.0
        else:  # yolo: width/height are normalized to target (100x100)
            assert np.isclose(out_bbox[2], 40 / 100, atol=1e-2)
            assert np.isclose(out_bbox[3], 30 / 100, atol=1e-2)
        assert result["class_labels"] == [2]

    def test_bbox_only_donor_warns_when_mask_target_present(self):
        """Bbox-only donor still triggers the no-semantic-mask warning when a `mask` target is present."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        donor_image = np.full((40, 40, 3), 137, dtype=np.uint8)
        primary_mask = np.zeros((100, 100), dtype=np.uint8)

        transform = A.Compose(
            [A.CopyAndPaste(p=1.0)],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["class_labels"]),
            seed=137,
        )
        with pytest.warns(UserWarning, match="no donor item provided `semantic_mask`"):
            transform(
                image=image,
                mask=primary_mask,
                bboxes=np.zeros((0, 4), dtype=np.float32),
                class_labels=[],
                copy_paste_metadata=[
                    {
                        "image": donor_image,
                        "bbox": [0, 0, 40, 40],
                        "bbox_labels": {"class_labels": 2},
                    },
                ],
            )


class TestCopyAndPasteScaleJitter:
    """`scale_range` shrink jitter and the cap at fit-to-target."""

    def test_scale_range_halves_donor(self):
        """`scale_range=(0.5, 0.5)` halves the donor relative to the shrink-to-fit scale."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        donor_image = np.full((40, 40, 3), 137, dtype=np.uint8)
        donor_mask = np.ones((40, 40), dtype=np.uint8)

        transform = A.Compose([A.CopyAndPaste(scale_range=(0.5, 0.5), p=1.0)], seed=137)
        result = transform(
            image=image,
            copy_paste_metadata=[{"image": donor_image, "mask": donor_mask}],
        )
        paste_region = np.any(result["image"] > 0, axis=-1)
        rows = np.any(paste_region, axis=1)
        cols = np.any(paste_region, axis=0)
        # Donor is 40x40, fits with scale 1.0 -> jitter 0.5 -> 20x20.
        assert int(rows.sum()) == 20
        assert int(cols.sum()) == 20

    def test_scale_range_capped_by_fit(self):
        """`scale_range=(2.0, 2.0)` is capped at fit-to-target — output never exceeds fit dims."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # 200x200 donor: fit is 100x100; jitter 2.0 would normally double, but is capped.
        donor_image = np.full((200, 200, 3), 137, dtype=np.uint8)
        donor_mask = np.ones((200, 200), dtype=np.uint8)

        transform = A.Compose([A.CopyAndPaste(scale_range=(2.0, 2.0), p=1.0)], seed=137)
        result = transform(
            image=image,
            copy_paste_metadata=[{"image": donor_image, "mask": donor_mask}],
        )
        paste_region = np.any(result["image"] > 0, axis=-1)
        assert int(paste_region.sum()) == 100 * 100


class TestCopyAndPasteMinPasteArea:
    """`min_paste_area` silently drops donors whose scaled footprint is too small."""

    def test_min_paste_area_skips_tiny_donors(self):
        """A 1000x1000 donor onto a 10x10 target with `min_paste_area=200` should drop the donor
        (scale-to-fit gives a 10x10 footprint with area 100 < 200).
        """
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        donor_image = np.full((1000, 1000, 3), 137, dtype=np.uint8)
        donor_mask = np.ones((1000, 1000), dtype=np.uint8)

        transform = A.Compose([A.CopyAndPaste(min_paste_area=200, p=1.0)], seed=137)
        result = transform(
            image=image.copy(),
            copy_paste_metadata=[{"image": donor_image, "mask": donor_mask}],
        )
        np.testing.assert_array_equal(result["image"], image)


class TestCopyAndPasteCropExpansion:
    """Keypoints outside the mask/bbox tight bbox extend the crop bounds."""

    def test_keypoints_outside_mask_extend_crop(self):
        """A keypoint outside the mask tight bbox should be preserved in the output (the crop is
        expanded to include it).
        """
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        donor_image = np.full((40, 40, 3), 137, dtype=np.uint8)
        donor_mask = np.zeros((40, 40), dtype=np.uint8)
        donor_mask[5:10, 5:10] = 1  # tight bbox: rows 5-10, cols 5-10
        # Keypoint at donor (35, 35) -- well outside the mask tight bbox.
        donor_keypoints = np.array([[35.0, 35.0]], dtype=np.float32)

        transform = A.Compose(
            [A.CopyAndPaste(p=1.0)],
            keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["keypoint_labels"]),
            seed=137,
        )
        result = transform(
            image=image,
            keypoints=np.zeros((0, 2), dtype=np.float32),
            keypoint_labels=[],
            copy_paste_metadata=[
                {
                    "image": donor_image,
                    "mask": donor_mask,
                    "keypoints": donor_keypoints,
                    "keypoint_labels": {"keypoint_labels": 7},
                },
            ],
        )
        assert len(result["keypoints"]) == 1
        assert result["keypoint_labels"] == [7]

    @pytest.mark.parametrize("coord_format", ["pascal_voc", "yolo"])
    def test_bbox_coords_after_paste_match_target_position(self, coord_format: str):
        """Round-trip: returned bbox in pipeline coord_format should map back to the actual paste
        footprint on the target image.
        """
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        donor_image = np.full((40, 40, 3), 137, dtype=np.uint8)
        donor_mask = np.ones((40, 40), dtype=np.uint8)
        if coord_format == "pascal_voc":
            donor_bbox: list[float] = [0.0, 0.0, 40.0, 40.0]
        else:  # yolo
            donor_bbox = [0.5, 0.5, 1.0, 1.0]

        transform = A.Compose(
            [A.CopyAndPaste(p=1.0)],
            bbox_params=A.BboxParams(coord_format=coord_format, label_fields=["class_labels"]),
            seed=137,
        )
        result = transform(
            image=image,
            bboxes=np.zeros((0, 4), dtype=np.float32),
            class_labels=[],
            copy_paste_metadata=[
                {
                    "image": donor_image,
                    "mask": donor_mask,
                    "bbox": donor_bbox,
                    "bbox_labels": {"class_labels": 2},
                },
            ],
        )
        paste_region = np.any(result["image"] > 0, axis=-1)
        ys, xs = np.where(paste_region)
        actual_x_min, actual_y_min = float(xs.min()), float(ys.min())
        actual_x_max, actual_y_max = float(xs.max()) + 1, float(ys.max()) + 1

        out_bbox = np.asarray(result["bboxes"][0], dtype=np.float32)
        if coord_format == "pascal_voc":
            assert np.allclose(out_bbox[:4], [actual_x_min, actual_y_min, actual_x_max, actual_y_max], atol=1.0)
        else:  # yolo: convert returned cxcywh-normalized to pascal pixels for comparison
            cx, cy, w, h = out_bbox[:4]
            x_min = (cx - w / 2) * 100
            y_min = (cy - h / 2) * 100
            x_max = (cx + w / 2) * 100
            y_max = (cy + h / 2) * 100
            assert np.allclose(
                [x_min, y_min, x_max, y_max],
                [actual_x_min, actual_y_min, actual_x_max, actual_y_max],
                atol=1.0,
            )


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
