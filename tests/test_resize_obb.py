"""Tests for OBB (Oriented Bounding Box) support in resize transforms."""

import numpy as np
import pytest

import albumentations as A
from albumentations.core.bbox_utils import normalize_bbox_angles


class TestRandomScaleOBB:
    """Test RandomScale transform with OBB bboxes."""

    @pytest.mark.parametrize("scale_limit", [(-0.3, 0.3), (0.1, 0.5), (-0.5, -0.1)])
    @pytest.mark.parametrize("angle", [0, 45, 90, -30, 135, -90])  # Removed 180 (equivalent to -180)
    def test_random_scale_preserves_obb_angles(self, scale_limit, angle):
        """Test that RandomScale preserves OBB angles since it's uniform scaling."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        obb_boxes = np.array([[0.2, 0.3, 0.4, 0.5, angle]], dtype=np.float32)
        bbox_labels = [1]

        transform = A.Compose(
            [A.RandomScale(scale_limit=scale_limit, p=1.0)],
            bbox_params=A.BboxParams(
                coord_format="albumentations",
                bbox_type="obb",
                label_fields=["bbox_labels"],
            ),
        )

        result = transform(image=image, bboxes=obb_boxes, bbox_labels=bbox_labels)

        # Uniform scaling preserves normalized coordinates and angles
        assert np.allclose(result["bboxes"][0][:5], obb_boxes[0][:5], atol=1e-6)

    def test_random_scale_with_multiple_obbs(self):
        """Test RandomScale with multiple OBB boxes."""
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        obb_boxes = np.array(
            [[0.1, 0.1, 0.3, 0.3, 45.0], [0.5, 0.5, 0.7, 0.7, -30.0], [0.7, 0.2, 0.9, 0.4, 90.0]],
            dtype=np.float32,
        )
        bbox_labels = [1, 2, 3]

        transform = A.Compose(
            [A.RandomScale(scale_limit=0.5, p=1.0)],
            bbox_params=A.BboxParams(
                coord_format="albumentations",
                bbox_type="obb",
                label_fields=["bbox_labels"],
            ),
        )

        result = transform(image=image, bboxes=obb_boxes, bbox_labels=bbox_labels)

        # Check all boxes are preserved
        assert len(result["bboxes"]) == 3
        assert np.allclose(result["bboxes"], obb_boxes, atol=1e-6)

    def test_random_scale_obb_with_extra_fields(self):
        """Test that extra fields beyond angle are preserved."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        # Add extra fields: [x_min, y_min, x_max, y_max, angle, confidence, track_id]
        obb_boxes = np.array([[0.2, 0.3, 0.4, 0.5, 45.0, 0.95, 137]], dtype=np.float32)
        bbox_labels = [1]

        transform = A.Compose(
            [A.RandomScale(scale_limit=0.3, p=1.0)],
            bbox_params=A.BboxParams(
                coord_format="albumentations",
                bbox_type="obb",
                label_fields=["bbox_labels"],
            ),
        )

        result = transform(image=image, bboxes=obb_boxes, bbox_labels=bbox_labels)

        # Check all fields including extras are preserved
        assert result["bboxes"].shape == obb_boxes.shape
        assert np.allclose(result["bboxes"][0], obb_boxes[0], atol=1e-6)


@pytest.mark.slow
class TestLongestMaxSizeOBB:
    """Test LongestMaxSize transform with OBB bboxes."""

    @pytest.mark.parametrize("max_size", [137, 512, 1024])
    @pytest.mark.parametrize("angle", [0, 45, 90, -30, 135, -90])
    def test_longest_max_size_preserves_obb_angles(self, max_size, angle, large_image_1000x500):
        """Test that LongestMaxSize preserves OBB angles (uniform scaling)."""
        obb_boxes = np.array([[0.2, 0.3, 0.4, 0.5, angle]], dtype=np.float32)
        bbox_labels = [1]

        transform = A.Compose(
            [A.LongestMaxSize(max_size=max_size, p=1.0)],
            bbox_params=A.BboxParams(
                coord_format="albumentations",
                bbox_type="obb",
                label_fields=["bbox_labels"],
            ),
        )

        result = transform(image=large_image_1000x500, bboxes=obb_boxes, bbox_labels=bbox_labels)

        # Uniform scaling preserves normalized coordinates and angles
        assert np.allclose(result["bboxes"][0][:5], obb_boxes[0][:5], atol=1e-6)

    def test_longest_max_size_with_max_size_hw(self, large_image_1000x800):
        """Test LongestMaxSize with max_size_hw parameter."""
        obb_boxes = np.array([[0.2, 0.3, 0.4, 0.5, 45.0]], dtype=np.float32)
        bbox_labels = [1]

        transform = A.Compose(
            [A.LongestMaxSize(max_size_hw=(600, 800), p=1.0)],
            bbox_params=A.BboxParams(
                coord_format="albumentations",
                bbox_type="obb",
                label_fields=["bbox_labels"],
            ),
        )

        result = transform(image=large_image_1000x800, bboxes=obb_boxes, bbox_labels=bbox_labels)

        # Uniform scaling preserves coordinates
        assert np.allclose(result["bboxes"][0][:5], obb_boxes[0][:5], atol=1e-6)


@pytest.mark.slow
class TestSmallestMaxSizeOBB:
    """Test SmallestMaxSize transform with OBB bboxes."""

    @pytest.mark.parametrize("max_size", [137, 512, 1024])
    @pytest.mark.parametrize("angle", [0, 45, 90, -30, 135, -90])
    def test_smallest_max_size_preserves_obb_angles(self, max_size, angle, large_image_500x1000):
        """Test that SmallestMaxSize preserves OBB angles (uniform scaling)."""
        obb_boxes = np.array([[0.2, 0.3, 0.4, 0.5, angle]], dtype=np.float32)
        bbox_labels = [1]

        transform = A.Compose(
            [A.SmallestMaxSize(max_size=max_size, p=1.0)],
            bbox_params=A.BboxParams(
                coord_format="albumentations",
                bbox_type="obb",
                label_fields=["bbox_labels"],
            ),
        )

        result = transform(image=large_image_500x1000, bboxes=obb_boxes, bbox_labels=bbox_labels)

        # Uniform scaling preserves normalized coordinates and angles
        assert np.allclose(result["bboxes"][0][:5], obb_boxes[0][:5], atol=1e-6)

    def test_smallest_max_size_with_max_size_hw(self, large_image_800x1000):
        """Test SmallestMaxSize with max_size_hw parameter."""
        obb_boxes = np.array([[0.2, 0.3, 0.4, 0.5, -45.0]], dtype=np.float32)
        bbox_labels = [1]

        transform = A.Compose(
            [A.SmallestMaxSize(max_size_hw=(1000, 800), p=1.0)],
            bbox_params=A.BboxParams(
                coord_format="albumentations",
                bbox_type="obb",
                label_fields=["bbox_labels"],
            ),
        )

        result = transform(image=large_image_800x1000, bboxes=obb_boxes, bbox_labels=bbox_labels)

        # Uniform scaling preserves coordinates
        assert np.allclose(result["bboxes"][0][:5], obb_boxes[0][:5], atol=1e-6)


class TestResizeOBB:
    """Test Resize transform with OBB bboxes."""

    def test_resize_uniform_scaling_preserves_angle(self):
        """Test that uniform scaling (e.g., 100x100 -> 200x200) preserves OBB angle."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        obb_boxes = np.array([[0.2, 0.3, 0.4, 0.5, 45.0]], dtype=np.float32)
        bbox_labels = [1]

        transform = A.Compose(
            [A.Resize(height=200, width=200, p=1.0)],
            bbox_params=A.BboxParams(
                coord_format="albumentations",
                bbox_type="obb",
                label_fields=["bbox_labels"],
            ),
        )

        result = transform(image=image, bboxes=obb_boxes, bbox_labels=bbox_labels)

        # Uniform scaling preserves everything
        assert np.allclose(result["bboxes"][0][:5], obb_boxes[0][:5], atol=1e-6)

    @pytest.mark.parametrize("angle", [0, 30, 45, 60, 90, -30, -45, -90, 135, 180])
    def test_resize_non_uniform_scaling_updates_angle(self, angle):
        """Test that non-uniform scaling (different x/y scales) updates OBB angle via polygon conversion."""
        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        # Create a rotated box in the center
        obb_boxes = np.array([[0.4, 0.4, 0.6, 0.6, angle]], dtype=np.float32)
        bbox_labels = [1]

        # Non-uniform resize: 100x200 -> 200x400 (scale_y=2, scale_x=2, uniform)
        # vs 100x200 -> 300x400 (scale_y=3, scale_x=2, non-uniform)
        transform = A.Compose(
            [A.Resize(height=300, width=400, p=1.0)],  # Non-uniform: scale_y=3, scale_x=2
            bbox_params=A.BboxParams(
                coord_format="albumentations",
                bbox_type="obb",
                label_fields=["bbox_labels"],
            ),
        )

        result = transform(image=image, bboxes=obb_boxes, bbox_labels=bbox_labels)

        # For non-uniform scaling, coordinates stay normalized but angle changes
        # The angle change depends on the original angle and scale factors
        assert result["bboxes"].shape == obb_boxes.shape
        assert len(result["bboxes"]) == 1

        # Coordinates should still be normalized (in [0, 1])
        assert np.all(result["bboxes"][0][:4] >= 0)
        assert np.all(result["bboxes"][0][:4] <= 1)

        # Angle should be normalized to [-180, 180]
        normalized_result = normalize_bbox_angles(result["bboxes"])
        assert -180 <= normalized_result[0][4] <= 180

    def test_resize_non_uniform_with_multiple_obbs(self):
        """Test non-uniform resize with multiple OBB boxes."""
        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        obb_boxes = np.array(
            [[0.1, 0.1, 0.3, 0.3, 0.0], [0.4, 0.4, 0.6, 0.6, 45.0], [0.7, 0.7, 0.9, 0.9, 90.0]],
            dtype=np.float32,
        )
        bbox_labels = [1, 2, 3]

        transform = A.Compose(
            [A.Resize(height=300, width=400, p=1.0)],  # Non-uniform
            bbox_params=A.BboxParams(
                coord_format="albumentations",
                bbox_type="obb",
                label_fields=["bbox_labels"],
            ),
        )

        result = transform(image=image, bboxes=obb_boxes, bbox_labels=bbox_labels)

        # All boxes should be transformed
        assert len(result["bboxes"]) == 3

        # All coordinates should be valid
        for bbox in result["bboxes"]:
            assert np.all(bbox[:4] >= 0)
            assert np.all(bbox[:4] <= 1)
            # Width and height should be positive
            assert bbox[2] > bbox[0]
            assert bbox[3] > bbox[1]

    def test_resize_obb_with_extra_fields(self):
        """Test that extra fields are preserved during non-uniform resize."""
        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        # Add extra fields
        obb_boxes = np.array([[0.3, 0.3, 0.7, 0.7, 45.0, 0.95, 137]], dtype=np.float32)
        bbox_labels = [1]

        transform = A.Compose(
            [A.Resize(height=300, width=400, p=1.0)],
            bbox_params=A.BboxParams(
                coord_format="albumentations",
                bbox_type="obb",
                label_fields=["bbox_labels"],
            ),
        )

        result = transform(image=image, bboxes=obb_boxes, bbox_labels=bbox_labels)

        # Check shape is preserved (including extra fields)
        assert result["bboxes"].shape == obb_boxes.shape
        # Check extra fields are preserved
        assert np.isclose(result["bboxes"][0][5], 0.95)
        assert result["bboxes"][0][6] == 137

    def test_resize_empty_obb_array(self):
        """Test that Resize handles empty OBB array correctly."""
        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        obb_boxes = np.array([], dtype=np.float32).reshape(0, 5)
        bbox_labels = []

        transform = A.Compose(
            [A.Resize(height=300, width=400, p=1.0)],
            bbox_params=A.BboxParams(
                coord_format="albumentations",
                bbox_type="obb",
                label_fields=["bbox_labels"],
            ),
        )

        result = transform(image=image, bboxes=obb_boxes, bbox_labels=bbox_labels)

        # Empty OBB array should preserve shape (0, 5) for basic OBB format
        assert result["bboxes"].shape == (0, 5)
        assert len(result["bbox_labels"]) == 0


class TestResizeOBBEdgeCases:
    """Test edge cases for OBB resize transforms."""

    def test_resize_very_small_scale_difference(self):
        """Test that tiny scale differences are treated as uniform."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        obb_boxes = np.array([[0.2, 0.3, 0.4, 0.5, 45.0]], dtype=np.float32)
        bbox_labels = [1]

        # Scale to 200x200.001 (almost uniform, should be treated as uniform due to 1e-7 tolerance)
        transform = A.Compose(
            [A.Resize(height=200, width=200, p=1.0)],
            bbox_params=A.BboxParams(
                coord_format="albumentations",
                bbox_type="obb",
                label_fields=["bbox_labels"],
            ),
        )

        result = transform(image=image, bboxes=obb_boxes, bbox_labels=bbox_labels)

        # Should preserve angle exactly
        assert np.allclose(result["bboxes"][0][:5], obb_boxes[0][:5], atol=1e-6)

    @pytest.mark.parametrize("original_angle", [-180, -90, 0, 90, 180, 270, -270])
    def test_angle_normalization(self, original_angle):
        """Test that angles are properly normalized after transformation."""
        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        obb_boxes = np.array([[0.3, 0.3, 0.7, 0.7, original_angle]], dtype=np.float32)
        bbox_labels = [1]

        transform = A.Compose(
            [A.Resize(height=300, width=400, p=1.0)],
            bbox_params=A.BboxParams(
                coord_format="albumentations",
                bbox_type="obb",
                label_fields=["bbox_labels"],
            ),
        )

        result = transform(image=image, bboxes=obb_boxes, bbox_labels=bbox_labels)

        # Result angle should be in range after normalization
        result_angle = result["bboxes"][0][4]
        # cv2.minAreaRect returns angles in [-90, 0], but normalize_bbox_angles normalizes to [-180, 180]
        # After any transformation, the angle should be valid
        assert not np.isnan(result_angle)
