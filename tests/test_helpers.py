"""Tests for test helper utilities.

This module tests the helper utilities themselves to ensure they work correctly.
"""

import numpy as np
import pytest

import albumentations as A
from tests.helpers import (
    ComposeBuilder,
    TestDataFactory,
    TransformTestHelper,
    build_exclude_set,
    create_compose,
    get_transforms_with_categories,
)


class TestTestDataFactory:
    """Tests for TestDataFactory."""

    def test_create_rng_default_seed(self):
        """Test RNG creation with default seed."""
        rng1 = TestDataFactory.create_rng()
        rng2 = TestDataFactory.create_rng()

        # Should produce same sequence
        assert rng1.integers(0, 100) == rng2.integers(0, 100)

    def test_create_rng_custom_seed(self):
        """Test RNG creation with custom seed."""
        rng1 = TestDataFactory.create_rng(seed=42)
        rng2 = TestDataFactory.create_rng(seed=42)

        # Should produce same sequence
        assert rng1.integers(0, 100) == rng2.integers(0, 100)

    def test_create_image_uint8(self):
        """Test uint8 image creation."""
        img = TestDataFactory.create_image((100, 100, 3), dtype=np.uint8)

        assert img.shape == (100, 100, 3)
        assert img.dtype == np.uint8
        assert np.min(img) >= 0
        assert np.max(img) <= 255

    def test_create_image_float32(self):
        """Test float32 image creation."""
        img = TestDataFactory.create_image((100, 100, 3), dtype=np.float32)

        assert img.shape == (100, 100, 3)
        assert img.dtype == np.float32
        assert np.min(img) >= 0.0
        assert np.max(img) <= 1.0

    def test_create_image_reproducible(self):
        """Test that same seed produces same image."""
        img1 = TestDataFactory.create_image((50, 50, 3), seed=137)
        img2 = TestDataFactory.create_image((50, 50, 3), seed=137)

        np.testing.assert_array_equal(img1, img2)

    def test_create_mask(self):
        """Test mask creation."""
        mask = TestDataFactory.create_mask((100, 100))

        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})

    def test_create_volume(self):
        """Test volume creation."""
        volume = TestDataFactory.create_volume((4, 100, 100, 3))

        assert volume.shape == (4, 100, 100, 3)
        assert volume.dtype == np.uint8

    def test_create_bboxes_pascal_voc(self):
        """Test bbox creation in pascal_voc format."""
        bboxes = TestDataFactory.create_bboxes(
            num_boxes=5,
            format="pascal_voc",
            seed=137,
        )

        assert bboxes.shape == (5, 4)
        assert bboxes.dtype == np.float32
        # Check valid boxes (x_max > x_min, y_max > y_min)
        assert np.all(bboxes[:, 2] > bboxes[:, 0])
        assert np.all(bboxes[:, 3] > bboxes[:, 1])

    def test_create_bboxes_albumentations(self):
        """Test bbox creation in albumentations format (normalized)."""
        bboxes = TestDataFactory.create_bboxes(
            num_boxes=3,
            format="albumentations",
            seed=137,
        )

        assert bboxes.shape == (3, 4)
        # All coordinates should be normalized [0, 1]
        assert np.all(bboxes >= 0.0)
        assert np.all(bboxes <= 1.0)

    def test_create_keypoints(self):
        """Test keypoint creation."""
        kpts = TestDataFactory.create_keypoints(
            num_points=10,
            format="xy",
            seed=137,
        )

        assert kpts.shape == (10, 2)
        assert kpts.dtype == np.float32


class TestTransformTestHelper:
    """Tests for TransformTestHelper."""

    def test_safe_copy_params(self):
        """Test params copying."""
        original = {"a": 1, "b": {"c": 2}}
        copied = TransformTestHelper.safe_copy_params(original)

        # Should be equal
        assert copied == original

        # But not same object
        assert copied is not original
        assert copied["b"] is not original["b"]

        # Modifying copy shouldn't affect original
        copied["b"]["c"] = 999
        assert original["b"]["c"] == 2

    def test_is_rgb_only(self):
        """Test RGB-only transform detection."""
        assert TransformTestHelper.is_rgb_only(A.ChannelDropout)
        assert TransformTestHelper.is_rgb_only(A.HEStain)
        assert not TransformTestHelper.is_rgb_only(A.HorizontalFlip)

    def test_requires_metadata(self):
        """Test metadata requirement detection."""
        assert TransformTestHelper.requires_metadata(A.FDA)
        assert TransformTestHelper.requires_metadata(A.Mosaic)
        assert not TransformTestHelper.requires_metadata(A.HorizontalFlip)

    def test_requires_mask(self):
        """Test mask requirement detection."""
        assert TransformTestHelper.requires_mask(A.MaskDropout)
        assert TransformTestHelper.requires_mask(A.ConstrainedCoarseDropout)
        assert not TransformTestHelper.requires_mask(A.HorizontalFlip)

    def test_requires_special_setup(self):
        """Test special setup requirement detection."""
        assert TransformTestHelper.requires_special_setup(A.OverlayElements)
        assert TransformTestHelper.requires_special_setup(A.TextImage)
        assert not TransformTestHelper.requires_special_setup(A.HorizontalFlip)

    def test_changes_dimensions(self):
        """Test dimension-changing transform detection."""
        assert TransformTestHelper.changes_dimensions(A.RandomCrop)
        assert TransformTestHelper.changes_dimensions(A.Resize)
        assert not TransformTestHelper.changes_dimensions(A.HorizontalFlip)

    def test_prepare_test_data_simple(self):
        """Test data preparation for simple transform."""
        image = np.ones((100, 100, 3), dtype=np.uint8)
        data = TransformTestHelper.prepare_test_data(A.HorizontalFlip, image)

        assert "image" in data
        assert data["image"] is image

    def test_prepare_test_data_with_mask(self):
        """Test data preparation for mask-required transform."""
        image = np.ones((100, 100, 3), dtype=np.uint8)
        data = TransformTestHelper.prepare_test_data(A.MaskDropout, image)

        assert "image" in data
        assert "mask" in data
        assert data["mask"].shape == (100, 100)

    def test_prepare_test_data_overlay_elements(self):
        """Test data preparation for OverlayElements."""
        image = np.ones((100, 100, 3), dtype=np.uint8)
        data = TransformTestHelper.prepare_test_data(A.OverlayElements, image)

        assert "overlay_metadata" in data
        assert isinstance(data["overlay_metadata"], list)

    def test_prepare_test_data_mosaic(self):
        """Test data preparation for Mosaic."""
        image = np.ones((100, 100, 3), dtype=np.uint8)
        data = TransformTestHelper.prepare_test_data(A.Mosaic, image)

        assert "mosaic_metadata" in data
        assert isinstance(data["mosaic_metadata"], list)

    def test_adjust_params_for_grayscale(self):
        """Test params adjustment for grayscale."""
        params = {"fill": (10, 20, 30), "other": "value"}
        adjusted = TransformTestHelper.adjust_params_for_grayscale(params)

        assert adjusted["fill"] == 10
        assert adjusted["other"] == "value"
        # Should be a copy
        assert adjusted is not params


class TestComposeBuilder:
    """Tests for ComposeBuilder."""

    def test_basic_build(self):
        """Test basic Compose building."""
        builder = ComposeBuilder([A.HorizontalFlip(p=1.0)])
        compose = builder.build()

        assert isinstance(compose, A.Compose)
        assert len(compose.transforms) == 1

    def test_with_seed(self):
        """Test seed setting."""
        # We can't directly check the seed, but we can verify deterministic behavior
        image = np.ones((50, 50, 3), dtype=np.uint8)

        compose1 = ComposeBuilder([A.RandomBrightnessContrast(p=1.0)]).with_seed(42).build()
        compose2 = ComposeBuilder([A.RandomBrightnessContrast(p=1.0)]).with_seed(42).build()

        result1 = compose1(image=image.copy())["image"]
        result2 = compose2(image=image.copy())["image"]

        # Same seed should produce same results
        np.testing.assert_array_equal(result1, result2)

    def test_with_bboxes(self):
        """Test bbox params."""
        compose = ComposeBuilder([A.HorizontalFlip(p=1.0)]).with_bboxes("pascal_voc").build()

        assert compose.processors["bboxes"] is not None

    def test_with_keypoints(self):
        """Test keypoint params."""
        compose = ComposeBuilder([A.HorizontalFlip(p=1.0)]).with_keypoints("xy").build()

        assert compose.processors["keypoints"] is not None

    def test_chaining(self):
        """Test method chaining."""
        compose = (
            ComposeBuilder([A.HorizontalFlip(p=1.0)])
            .with_seed(42)
            .with_strict(True)
            .with_bboxes("pascal_voc")
            .with_keypoints("xy")
            .build()
        )

        assert isinstance(compose, A.Compose)
        assert compose.processors["bboxes"] is not None
        assert compose.processors["keypoints"] is not None

    def test_create_compose_factory(self):
        """Test quick factory function."""
        # Test that factory creates a valid compose
        compose = create_compose([A.HorizontalFlip(p=1.0)], seed=42)

        assert isinstance(compose, A.Compose)
        assert len(compose.transforms) == 1


class TestParametrizeHelpers:
    """Tests for parametrize helpers."""

    def test_get_transforms_with_categories_all(self):
        """Test getting all 2D transforms."""
        transforms = get_transforms_with_categories(transform_type="2d")

        assert len(transforms) > 0
        assert all(isinstance(t, tuple) and len(t) == 2 for t in transforms)

    def test_get_transforms_exclude_metadata(self):
        """Test excluding metadata transforms."""
        all_transforms = get_transforms_with_categories(transform_type="2d")
        filtered = get_transforms_with_categories(
            transform_type="2d",
            exclude_categories=["metadata"],
        )

        # Should have fewer transforms
        assert len(filtered) < len(all_transforms)

        # Should not contain metadata transforms
        transform_classes = {t[0] for t in filtered}
        assert A.FDA not in transform_classes
        assert A.Mosaic not in transform_classes

    def test_get_transforms_exclude_multiple(self):
        """Test excluding multiple categories."""
        filtered = get_transforms_with_categories(
            transform_type="2d",
            exclude_categories=["metadata", "rgb_only"],
        )

        transform_classes = {t[0] for t in filtered}
        # Should not contain metadata or RGB-only transforms
        assert A.FDA not in transform_classes
        assert A.ChannelDropout not in transform_classes

    def test_build_exclude_set(self):
        """Test building exclude set."""
        exclude_set = build_exclude_set("metadata", "rgb_only")

        assert A.FDA in exclude_set
        assert A.Mosaic in exclude_set
        assert A.ChannelDropout in exclude_set
        assert A.HEStain in exclude_set

    def test_build_exclude_set_invalid_category(self):
        """Test error on invalid category."""
        with pytest.raises(ValueError, match="Unknown category"):
            build_exclude_set("invalid_category")
