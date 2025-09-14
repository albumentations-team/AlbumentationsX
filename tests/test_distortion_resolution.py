"""Tests for distortion map resolution functionality."""

import numpy as np
import pytest
import cv2

import albumentations as A
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.geometric.distortion import (
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    PiecewiseAffine,
    ThinPlateSpline,
)


class TestUpscaleDistortionMaps:
    """Test the upscale_distortion_maps function."""

    def test_upscale_identity(self):
        """Test that upscaling when source and target are same size returns identity."""
        h, w = 100, 100
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        result_x, result_y = fgeometric.upscale_distortion_maps(
            map_x, map_y, (h, w), cv2.INTER_LINEAR
        )

        np.testing.assert_array_equal(result_x, map_x)
        np.testing.assert_array_equal(result_y, map_y)

    def test_upscale_2x(self):
        """Test upscaling from half resolution to full resolution."""
        # Create small distortion maps
        small_h, small_w = 50, 50
        map_x, map_y = np.meshgrid(np.arange(small_w), np.arange(small_h))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        # Upscale to double size
        target_h, target_w = 100, 100
        result_x, result_y = fgeometric.upscale_distortion_maps(
            map_x, map_y, (target_h, target_w), cv2.INTER_LINEAR
        )

        assert result_x.shape == (target_h, target_w)
        assert result_y.shape == (target_h, target_w)

        # Check that coordinates are properly scaled
        # Top-left should be (0, 0)
        assert result_x[0, 0] == pytest.approx(0, abs=0.1)
        assert result_y[0, 0] == pytest.approx(0, abs=0.1)

        # Bottom-right should be close to (99, 99)
        assert result_x[-1, -1] == pytest.approx(target_w - 1, abs=1.0)
        assert result_y[-1, -1] == pytest.approx(target_h - 1, abs=1.0)

    def test_upscale_non_uniform(self):
        """Test upscaling with non-uniform scaling factors."""
        small_h, small_w = 30, 40
        map_x, map_y = np.meshgrid(np.arange(small_w), np.arange(small_h))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        target_h, target_w = 90, 160
        result_x, result_y = fgeometric.upscale_distortion_maps(
            map_x, map_y, (target_h, target_w), cv2.INTER_LINEAR
        )

        assert result_x.shape == (target_h, target_w)
        assert result_y.shape == (target_h, target_w)

    @pytest.mark.parametrize("interpolation", [
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_NEAREST,
    ])
    def test_upscale_interpolation_methods(self, interpolation):
        """Test different interpolation methods."""
        small_h, small_w = 25, 25
        map_x = np.random.rand(small_h, small_w).astype(np.float32) * small_w
        map_y = np.random.rand(small_h, small_w).astype(np.float32) * small_h

        target_h, target_w = 100, 100
        result_x, result_y = fgeometric.upscale_distortion_maps(
            map_x, map_y, (target_h, target_w), interpolation
        )

        assert result_x.shape == (target_h, target_w)
        assert result_y.shape == (target_h, target_w)
        assert result_x.dtype == np.float32
        assert result_y.dtype == np.float32


class TestMapResolutionRange:
    """Test map_resolution_range parameter for all distortion transforms."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_mask(self):
        """Create a sample mask for testing."""
        return np.random.randint(0, 2, (100, 100), dtype=np.uint8)

    @pytest.fixture
    def sample_bboxes(self):
        """Create sample bounding boxes."""
        return [[10, 10, 50, 50], [60, 60, 90, 90]]

    @pytest.fixture
    def sample_keypoints(self):
        """Create sample keypoints."""
        return [[25, 25], [75, 75]]

    @pytest.mark.parametrize("transform_cls,params", [
        (ElasticTransform, {"alpha": 1, "sigma": 50}),
        (GridDistortion, {"num_steps": 5}),
        (OpticalDistortion, {"distort_limit": 0.1}),
        (PiecewiseAffine, {"scale": (0.03, 0.05)}),
        (ThinPlateSpline, {"scale_range": (0.2, 0.4)}),
    ])
    def test_default_resolution(self, transform_cls, params, sample_image):
        """Test that default map_resolution_range=(1.0, 1.0) works."""
        transform = transform_cls(**params, map_resolution_range=(1.0, 1.0), p=1.0)
        result = transform(image=sample_image)

        assert result["image"].shape == sample_image.shape
        assert result["image"].dtype == sample_image.dtype

    @pytest.mark.parametrize("transform_cls,params", [
        (ElasticTransform, {"alpha": 1, "sigma": 50}),
        (GridDistortion, {"num_steps": 5}),
        (OpticalDistortion, {"distort_limit": 0.1}),
        (PiecewiseAffine, {"scale": (0.03, 0.05)}),
        (ThinPlateSpline, {"scale_range": (0.2, 0.4)}),
    ])
    def test_half_resolution(self, transform_cls, params, sample_image):
        """Test that map_resolution_range=(0.5, 0.5) works."""
        transform = transform_cls(**params, map_resolution_range=(0.5, 0.5), p=1.0)
        result = transform(image=sample_image)

        assert result["image"].shape == sample_image.shape
        assert result["image"].dtype == sample_image.dtype

    @pytest.mark.parametrize("transform_cls,params", [
        (ElasticTransform, {"alpha": 1, "sigma": 50}),
        (GridDistortion, {"num_steps": 5}),
        (OpticalDistortion, {"distort_limit": 0.1}),
        (PiecewiseAffine, {"scale": (0.03, 0.05)}),
        (ThinPlateSpline, {"scale_range": (0.2, 0.4)}),
    ])
    def test_resolution_range(self, transform_cls, params, sample_image):
        """Test that variable map_resolution_range works."""
        transform = transform_cls(**params, map_resolution_range=(0.3, 1.0), p=1.0)

        # Run multiple times to test random sampling
        for _ in range(5):
            result = transform(image=sample_image)
            assert result["image"].shape == sample_image.shape
            assert result["image"].dtype == sample_image.dtype

    @pytest.mark.parametrize("transform_cls,params", [
        (ElasticTransform, {"alpha": 1, "sigma": 50}),
        (GridDistortion, {"num_steps": 5}),
        (OpticalDistortion, {"distort_limit": 0.1}),
        (PiecewiseAffine, {"scale": (0.03, 0.05)}),
        (ThinPlateSpline, {"scale_range": (0.2, 0.4)}),
    ])
    def test_very_low_resolution(self, transform_cls, params, sample_image):
        """Test that very low resolution (0.1) still works."""
        transform = transform_cls(**params, map_resolution_range=(0.1, 0.1), p=1.0)
        result = transform(image=sample_image)

        assert result["image"].shape == sample_image.shape
        assert result["image"].dtype == sample_image.dtype

    def test_compose_with_resolution(self, sample_image, sample_mask, sample_bboxes, sample_keypoints):
        """Test that transforms work in Compose with map_resolution_range."""
        transform = A.Compose([
            A.ElasticTransform(alpha=1, sigma=50, map_resolution_range=(0.5, 0.8), p=1.0),
            A.GridDistortion(num_steps=5, map_resolution_range=(0.5, 0.8), p=1.0),
        ], bbox_params=A.BboxParams(format='pascal_voc'),
           keypoint_params=A.KeypointParams(format='xy'))

        result = transform(
            image=sample_image,
            mask=sample_mask,
            bboxes=sample_bboxes,
            keypoints=sample_keypoints
        )

        assert result["image"].shape == sample_image.shape
        assert result["mask"].shape == sample_mask.shape
        assert len(result["bboxes"]) <= len(sample_bboxes)  # Some might be outside
        assert len(result["keypoints"]) <= len(sample_keypoints)  # Some might be outside

    @pytest.mark.parametrize("invalid_range", [
        (0.0, 1.0),  # 0 is not allowed (exclusive lower bound)
        (-0.1, 0.5),  # Negative not allowed
        (0.5, 0.3),  # Not non-decreasing
        (1.1, 1.2),  # Above 1.0 not allowed
    ])
    def test_invalid_resolution_range(self, invalid_range):
        """Test that invalid map_resolution_range values raise errors."""
        with pytest.raises((ValueError, TypeError)):
            ElasticTransform(alpha=1, sigma=50, map_resolution_range=invalid_range)

    def test_performance_benefit(self, sample_image):
        """Test that lower resolution actually provides performance benefit."""
        import time

        # Large image for noticeable performance difference
        large_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        # Full resolution
        transform_full = ElasticTransform(
            alpha=1, sigma=50, map_resolution_range=(1.0, 1.0), p=1.0
        )

        # Quarter resolution
        transform_quarter = ElasticTransform(
            alpha=1, sigma=50, map_resolution_range=(0.25, 0.25), p=1.0
        )

        # Warm up
        for _ in range(3):
            transform_full(image=large_image)
            transform_quarter(image=large_image)

        # Time full resolution
        start = time.time()
        for _ in range(10):
            transform_full(image=large_image)
        time_full = time.time() - start

        # Time quarter resolution
        start = time.time()
        for _ in range(10):
            transform_quarter(image=large_image)
        time_quarter = time.time() - start

        # Quarter resolution should be noticeably faster
        # We expect at least 2x speedup for map generation
        # (not 16x because remapping still takes time)
        assert time_quarter < time_full * 0.7, (
            f"Quarter resolution ({time_quarter:.3f}s) should be faster than "
            f"full resolution ({time_full:.3f}s)"
        )


class TestElasticTransformOptimization:
    """Test the ElasticTransform coordinate generation optimization."""

    def test_elastic_transform_correctness(self):
        """Test that optimized ElasticTransform produces valid results."""
        transform = ElasticTransform(alpha=1, sigma=50, p=1.0)
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        result = transform(image=image)

        # Check output shape and type
        assert result["image"].shape == image.shape
        assert result["image"].dtype == image.dtype

        # Check that transformation actually changes the image
        # (with p=1.0 it should always apply)
        assert not np.array_equal(result["image"], image)

    def test_coordinate_grid_dimensions(self):
        """Test that coordinate grids have correct dimensions."""
        # This tests the internal logic by checking a simple case
        transform = ElasticTransform(alpha=0, sigma=50, p=1.0)  # alpha=0 means no displacement
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128

        result = transform(image=image)

        # With alpha=0, the image should remain unchanged
        np.testing.assert_array_almost_equal(result["image"], image, decimal=0)
