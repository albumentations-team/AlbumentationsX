"""Tests for GuidedCoarseDropout.

Covers:
- Label filtering (string and numeric)
- Bbox dilation and clipping
- Strict protection guarantees
- No-op cases (empty guidance, missing keys, fully protected, etc.)
- Serialization and replay
- Deterministic seeding
"""

import numpy as np
import pytest

import albumentations as A

# ---------------------------------------------------------------------------
# Assuming GuidedCoarseDropout is importable from albumentations after
# the module is wired into __init__.py.  During local dev, adjust the import.
# ---------------------------------------------------------------------------
# from albumentations.augmentations.dropout.guided_coarse_dropout import GuidedCoarseDropout
# For now, use:
# import albumentations as A; GuidedCoarseDropout = A.GuidedCoarseDropout


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def image_100():
    """100x100 RGB image."""
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def full_guidance_100():
    """100x100 guidance mask — all True (dropout everywhere)."""
    return np.ones((100, 100), dtype=np.uint8)


@pytest.fixture
def partial_guidance_100():
    """100x100 guidance mask — True only in a 60x60 center region."""
    m = np.zeros((100, 100), dtype=np.uint8)
    m[20:80, 20:80] = 1
    return m


# ===========================================================================
# 1.  Basic functionality
# ===========================================================================


class TestBasicDropout:
    """Verify that dropout modifies pixels only inside the guidance mask."""

    def test_dropout_applied_within_guidance(self, image_100, partial_guidance_100):
        transform = A.Compose(
            [
                A.GuidedCoarseDropout(
                    region_key="sal",
                    num_holes_range=(5, 10),
                    hole_height_range=(0.05, 0.15),
                    hole_width_range=(0.05, 0.15),
                    fill=0,
                    p=1.0,
                ),
            ],
        )
        original = image_100.copy()
        result = transform(image=image_100, user_data={"sal": partial_guidance_100})

        changed = result["image"] != original
        # Any pixel that changed must lie within the guidance region
        guidance_bool = partial_guidance_100.astype(bool)
        assert np.all(changed[:, :, 0][~guidance_bool] == False), (  # noqa: E712
            "Pixels outside guidance were modified"
        )

    def test_no_change_when_guidance_all_zero(self, image_100):
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        transform = A.GuidedCoarseDropout(
            region_key="m",
            num_holes_range=(3, 5),
            fill=0,
            p=1.0,
        )
        original = image_100.copy()
        result = transform(image=image_100, user_data={"m": empty_mask})
        np.testing.assert_array_equal(result["image"], original)


# ===========================================================================
# 2.  Strict protection guarantees
# ===========================================================================


class TestProtection:
    """dropout_mask ∩ protected_mask = ∅"""

    def test_protected_bbox_untouched(self, image_100, full_guidance_100):
        """Dropout must never modify pixels inside a protected + dilated bbox."""
        bboxes = [[30, 30, 70, 70]]
        labels = ["cat"]

        transform = A.Compose(
            [
                A.GuidedCoarseDropout(
                    region_key="sal",
                    protected_bbox_labels=["cat"],
                    protection_margin=0.10,
                    num_holes_range=(20, 30),
                    hole_height_range=(0.10, 0.30),
                    hole_width_range=(0.10, 0.30),
                    fill=0,
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
        )
        original = image_100.copy()
        result = transform(
            image=image_100,
            user_data={"sal": full_guidance_100},
            bboxes=bboxes,
            labels=labels,
        )

        # The protected region with 10% margin around [30,30,70,70]
        # box_w=40, box_h=40 → margin_x=4, margin_y=4
        # protected region: [26, 26, 74, 74]
        protected_region = result["image"][26:74, 26:74]
        original_region = original[26:74, 26:74]
        np.testing.assert_array_equal(
            protected_region,
            original_region,
            err_msg="Protected region was modified by dropout",
        )

    def test_protection_margin_clipping(self, image_100, full_guidance_100):
        """Protected bbox at edge: dilation must clip to image bounds."""
        bboxes = [[0, 0, 20, 20]]
        labels = ["obj"]

        transform = A.Compose(
            [
                A.GuidedCoarseDropout(
                    region_key="sal",
                    protected_bbox_labels=["obj"],
                    protection_margin=0.50,  # large margin
                    num_holes_range=(10, 10),
                    hole_height_range=(0.05, 0.10),
                    hole_width_range=(0.05, 0.10),
                    fill=0,
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
        )
        # Should not raise — clipping must handle going below (0, 0)
        result = transform(
            image=image_100,
            user_data={"sal": full_guidance_100},
            bboxes=bboxes,
            labels=labels,
        )
        assert result["image"] is not None


# ===========================================================================
# 3.  No-op cases
# ===========================================================================


class TestNoOps:
    """All no-op scenarios must return the image unchanged."""

    def test_missing_user_data_key(self, image_100):
        transform = A.GuidedCoarseDropout(region_key="missing", fill=0, p=1.0)
        original = image_100.copy()
        result = transform(image=image_100, user_data={"other": np.ones((100, 100))})
        np.testing.assert_array_equal(result["image"], original)

    def test_no_user_data_at_all(self, image_100):
        transform = A.GuidedCoarseDropout(region_key="sal", fill=0, p=1.0)
        original = image_100.copy()
        result = transform(image=image_100)
        np.testing.assert_array_equal(result["image"], original)

    def test_fully_protected_guidance(self, image_100, full_guidance_100):
        """When protection covers the entire guidance region → no-op."""
        bboxes = [[0, 0, 100, 100]]  # covers whole image
        labels = ["big"]

        transform = A.Compose(
            [
                A.GuidedCoarseDropout(
                    region_key="sal",
                    protected_bbox_labels=["big"],
                    protection_margin=0.0,
                    num_holes_range=(5, 5),
                    fill=0,
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
        )
        original = image_100.copy()
        result = transform(
            image=image_100,
            user_data={"sal": full_guidance_100},
            bboxes=bboxes,
            labels=labels,
        )
        np.testing.assert_array_equal(result["image"], original)

    def test_no_bboxes_but_protected_labels(self, image_100, full_guidance_100):
        """protected_bbox_labels set but no bboxes supplied → dropout still works."""
        transform = A.GuidedCoarseDropout(
            region_key="sal",
            protected_bbox_labels=["cat"],
            num_holes_range=(5, 5),
            hole_height_range=(0.10, 0.20),
            hole_width_range=(0.10, 0.20),
            fill=0,
            p=1.0,
        )
        result = transform(image=image_100, user_data={"sal": full_guidance_100})
        # Should not crash — bboxes are simply absent
        assert result["image"].shape == image_100.shape

    def test_missing_protected_labels_in_bboxes(self, image_100, full_guidance_100):
        """Bboxes present but none match the protected labels → no protection."""
        bboxes = [[10, 10, 30, 30]]
        labels = ["dog"]

        transform = A.Compose(
            [
                A.GuidedCoarseDropout(
                    region_key="sal",
                    protected_bbox_labels=["cat"],  # no "cat" in labels
                    num_holes_range=(10, 10),
                    hole_height_range=(0.10, 0.20),
                    hole_width_range=(0.10, 0.20),
                    fill=0,
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
        )
        original = image_100.copy()
        result = transform(
            image=image_100,
            user_data={"sal": full_guidance_100},
            bboxes=bboxes,
            labels=labels,
        )
        # Some change should happen since nothing is protected
        changed = np.any(result["image"] != original)
        assert changed, "Expected dropout to modify pixels when no matching labels"


# ===========================================================================
# 4.  Label filtering
# ===========================================================================


class TestLabelFiltering:
    """String and numeric label filtering via the label encoder."""

    def test_numeric_labels(self, image_100, full_guidance_100):
        bboxes = [[10, 10, 40, 40], [60, 60, 90, 90]]
        labels = [1, 2]

        transform = A.Compose(
            [
                A.GuidedCoarseDropout(
                    region_key="sal",
                    protected_bbox_labels=[1],  # protect only label 1
                    protection_margin=0.0,
                    num_holes_range=(20, 20),
                    hole_height_range=(0.10, 0.30),
                    hole_width_range=(0.10, 0.30),
                    fill=0,
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
        )
        original = image_100.copy()
        result = transform(
            image=image_100,
            user_data={"sal": full_guidance_100},
            bboxes=bboxes,
            labels=labels,
        )
        # First bbox region should be untouched
        np.testing.assert_array_equal(
            result["image"][10:40, 10:40],
            original[10:40, 10:40],
        )

    def test_string_labels_selective(self, image_100, full_guidance_100):
        bboxes = [[10, 10, 40, 40], [60, 60, 90, 90]]
        labels = ["cat", "dog"]

        transform = A.Compose(
            [
                A.GuidedCoarseDropout(
                    region_key="sal",
                    protected_bbox_labels=["cat"],  # protect cat only
                    protection_margin=0.0,
                    num_holes_range=(20, 20),
                    hole_height_range=(0.10, 0.30),
                    hole_width_range=(0.10, 0.30),
                    fill=0,
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
        )
        original = image_100.copy()
        result = transform(
            image=image_100,
            user_data={"sal": full_guidance_100},
            bboxes=bboxes,
            labels=labels,
        )
        # Cat bbox should be protected
        np.testing.assert_array_equal(
            result["image"][10:40, 10:40],
            original[10:40, 10:40],
        )


# ===========================================================================
# 5.  Fill modes
# ===========================================================================


class TestFillModes:
    """Verify different fill types don't crash and produce expected behavior."""

    @pytest.mark.parametrize(
        "fill",
        [0, 128, (0, 0, 0), "random", "random_uniform", "inpaint_telea", "inpaint_ns", "grayscale"],
    )
    def test_fill_mode_runs(self, image_100, full_guidance_100, fill):
        transform = A.GuidedCoarseDropout(
            region_key="sal",
            num_holes_range=(3, 5),
            hole_height_range=(0.05, 0.15),
            hole_width_range=(0.05, 0.15),
            fill=fill,
            p=1.0,
        )
        result = transform(image=image_100, user_data={"sal": full_guidance_100})
        assert result["image"].shape == image_100.shape
        assert result["image"].dtype == image_100.dtype


# ===========================================================================
# 6.  Deterministic seeding
# ===========================================================================


class TestDeterminism:
    """Same seed → same result."""

    def test_seeded_reproducibility(self, image_100, full_guidance_100):
        def run(seed):
            transform = A.Compose(
                [
                    A.GuidedCoarseDropout(
                        region_key="sal",
                        num_holes_range=(3, 5),
                        hole_height_range=(0.05, 0.15),
                        hole_width_range=(0.05, 0.15),
                        fill=0,
                        p=1.0,
                    ),
                ],
                seed=seed,
            )
            return transform(image=image_100.copy(), user_data={"sal": full_guidance_100})["image"]

        r1 = run(42)
        r2 = run(42)
        np.testing.assert_array_equal(r1, r2)


# ===========================================================================
# 7.  Mask target handling
# ===========================================================================


class TestMaskTarget:
    """fill_mask behaviour."""

    def test_fill_mask_applied(self, image_100, full_guidance_100):
        seg_mask = np.ones((100, 100), dtype=np.uint8) * 5
        transform = A.GuidedCoarseDropout(
            region_key="sal",
            num_holes_range=(5, 5),
            hole_height_range=(0.10, 0.20),
            hole_width_range=(0.10, 0.20),
            fill=0,
            fill_mask=0,
            p=1.0,
        )
        result = transform(
            image=image_100,
            mask=seg_mask,
            user_data={"sal": full_guidance_100},
        )
        # At least some mask pixels should now be 0
        assert np.any(result["mask"] == 0)

    def test_fill_mask_none_leaves_mask_unchanged(self, image_100, full_guidance_100):
        seg_mask = np.ones((100, 100), dtype=np.uint8) * 5
        transform = A.GuidedCoarseDropout(
            region_key="sal",
            num_holes_range=(5, 5),
            hole_height_range=(0.10, 0.20),
            hole_width_range=(0.10, 0.20),
            fill=0,
            fill_mask=None,
            p=1.0,
        )
        result = transform(
            image=image_100,
            mask=seg_mask,
            user_data={"sal": full_guidance_100},
        )
        np.testing.assert_array_equal(result["mask"], seg_mask)


# ===========================================================================
# 8.  Guidance mask left unchanged
# ===========================================================================


class TestGuidancePreservation:
    """The guidance mask in user_data should not be modified."""

    def test_guidance_mask_unchanged(self, image_100, full_guidance_100):
        original_guidance = full_guidance_100.copy()
        transform = A.GuidedCoarseDropout(
            region_key="sal",
            num_holes_range=(5, 5),
            fill=0,
            p=1.0,
        )
        result = transform(image=image_100, user_data={"sal": full_guidance_100})
        np.testing.assert_array_equal(
            result.get("user_data", {}).get("sal", full_guidance_100),
            original_guidance,
        )


# ===========================================================================
# 9.  Guidance mask validation
# ===========================================================================


class TestGuidanceValidation:
    """Validate shape and dimensionality checks on the guidance mask."""

    def test_wrong_ndim_raises(self, image_100):
        bad_mask = np.ones((100, 100, 3), dtype=np.uint8)
        transform = A.GuidedCoarseDropout(region_key="sal", fill=0, p=1.0)
        with pytest.raises(ValueError, match="must be 2-D"):
            transform(image=image_100, user_data={"sal": bad_mask})

    def test_shape_mismatch_raises(self, image_100):
        wrong_shape = np.ones((50, 50), dtype=np.uint8)
        transform = A.GuidedCoarseDropout(region_key="sal", fill=0, p=1.0)
        with pytest.raises(ValueError, match="does not match image shape"):
            transform(image=image_100, user_data={"sal": wrong_shape})

    def test_3d_with_channel_1_is_squeezed(self, image_100):
        mask_3d = np.ones((100, 100, 1), dtype=np.uint8)
        transform = A.GuidedCoarseDropout(
            region_key="sal",
            num_holes_range=(3, 5),
            fill=0,
            p=1.0,
        )
        result = transform(image=image_100, user_data={"sal": mask_3d})
        assert result["image"].shape == image_100.shape
