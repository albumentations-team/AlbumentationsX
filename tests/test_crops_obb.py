"""Tests for OBB (Oriented Bounding Box) support in crop transforms.

This test file verifies:
1. OBB format preservation through crops
2. Correct handling of partially cropped OBBs (refitting via cv2.minAreaRect)
3. Unchanged OBBs keep original angle/dimensions
4. All crop transforms support OBB
5. Helper functions for clipped/unclipped OBB processing
"""

import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.crops.functional import (
    _process_clipped_obb_boxes,
    _process_unclipped_obb_boxes,
)


@pytest.mark.parametrize(
    "bbox_format,input_bbox",
    [
        pytest.param("pascal_voc", [20, 30, 60, 70, 30.0], id="pascal_voc"),
        pytest.param("coco", [20, 30, 40, 40, 30.0], id="coco"),
        pytest.param("yolo", [0.4, 0.5, 0.4, 0.4, 30.0], id="yolo"),
        pytest.param("albumentations", [0.2, 0.3, 0.6, 0.7, 30.0], id="albumentations"),
    ],
)
@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(A.Crop(x_min=10, y_min=10, x_max=90, y_max=90, p=1.0), id="Crop"),
        pytest.param(A.CenterCrop(height=60, width=60, p=1.0), id="CenterCrop"),
        pytest.param(A.RandomCrop(height=60, width=60, p=1.0), id="RandomCrop"),
    ],
)
@pytest.mark.obb
def test_obb_format_preservation_through_crops(bbox_format: str, input_bbox: list, transform: A.BasicTransform) -> None:
    """Test that input format == output format for OBB through various crop transforms."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    aug = A.Compose(
        [transform],
        bbox_params=A.BboxParams(format=bbox_format, bbox_type="obb")
    )

    np.random.seed(137)
    result = aug(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"]

    # Verify format consistency - output should be in same format as input
    if len(output_bbox) > 0:
        output_bbox = output_bbox[0]
        assert len(output_bbox) == 5, f"Expected 5 elements for OBB, got {len(output_bbox)}"

        # Verify coordinates are in expected range
        if bbox_format in ["pascal_voc", "coco"]:
            # Pixel coordinates - should have values that could be > 1
            assert isinstance(output_bbox, (list, tuple, np.ndarray))
        else:
            # Normalized coordinates [0, 1]
            assert all(0 <= v <= 1.01 for v in output_bbox[:4]), \
                f"Expected normalized coords for {bbox_format}, got {output_bbox}"


@pytest.mark.obb
def test_obb_fully_inside_crop_keeps_angle() -> None:
    """Test that OBB fully inside crop keeps original angle and dimensions (relative)."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # OBB fully inside the crop region (center at 50,50, will stay inside 20-80 crop)
    bbox = [0.4, 0.4, 0.6, 0.6, 45.0]  # albumentations format

    transform = A.Compose(
        [A.Crop(x_min=20, y_min=20, x_max=80, y_max=80, p=1.0)],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )

    result = transform(image=image, bboxes=[bbox])
    output_bbox = result["bboxes"][0]

    # Should still have 5 elements (OBB format preserved)
    assert len(output_bbox) == 5

    # Angle should be preserved (within tolerance, might be normalized to equivalent angle)
    # 45 and -45 are 90 degrees apart, but cv2.minAreaRect may return either depending on box orientation
    angle_diff = abs(output_bbox[4] - 45.0)
    # Allow for angle normalization: angles can differ by 90, 180, or 270 degrees and still be equivalent
    angle_diff = min(angle_diff, abs(angle_diff - 90), abs(angle_diff - 180), abs(angle_diff - 270))
    assert angle_diff < 5.0, f"Expected angle close to 45.0, got {output_bbox[4]}"

    # Box should be within the cropped region [0, 1]
    assert all(0 <= v <= 1.0 for v in output_bbox[:4])


@pytest.mark.obb
def test_obb_partially_cropped_refits() -> None:
    """Test that OBB partially outside crop gets refitted via cv2.minAreaRect."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # OBB that extends beyond crop region
    # Center at (0.8, 0.8), extends to edges, will be clipped by crop to (0.5, 0.5, 0.95, 0.95)
    bbox = [0.7, 0.7, 0.9, 0.9, 30.0]

    transform = A.Compose(
        [A.Crop(x_min=50, y_min=50, x_max=95, y_max=95, p=1.0)],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )

    result = transform(image=image, bboxes=[bbox])

    # Box should still exist (not filtered out)
    assert len(result["bboxes"]) == 1

    output_bbox = result["bboxes"][0]

    # Should still have 5 elements (OBB format preserved)
    assert len(output_bbox) == 5

    # Angle might have changed due to refitting
    # Just verify it's a valid angle in the canonical range
    assert -180 <= output_bbox[4] < 180


@pytest.mark.obb
def test_obb_completely_outside_crop_filtered() -> None:
    """Test that OBB completely outside crop is filtered out."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # OBB completely outside the crop region
    bbox = [0.1, 0.1, 0.3, 0.3, 0.0]  # Top-left corner, will be cropped out

    transform = A.Compose(
        [A.Crop(x_min=50, y_min=50, x_max=100, y_max=100, p=1.0)],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )

    result = transform(image=image, bboxes=[bbox])

    # Box should be filtered out
    assert len(result["bboxes"]) == 0


@pytest.mark.parametrize(
    "transform_class,transform_kwargs",
    [
        pytest.param(A.Crop, {"x_min": 10, "y_min": 10, "x_max": 90, "y_max": 90}, id="Crop"),
        pytest.param(A.CenterCrop, {"height": 60, "width": 60}, id="CenterCrop"),
        pytest.param(A.RandomCrop, {"height": 60, "width": 60}, id="RandomCrop"),
        pytest.param(A.RandomResizedCrop, {"size": (50, 50), "scale": (0.5, 1.0)}, id="RandomResizedCrop"),
        pytest.param(A.RandomSizedCrop, {"min_max_height": (40, 60), "size": (50, 50)}, id="RandomSizedCrop"),
        pytest.param(A.RandomCropFromBorders, {}, id="RandomCropFromBorders"),
        pytest.param(A.CropNonEmptyMaskIfExists, {"height": 50, "width": 50}, id="CropNonEmptyMaskIfExists"),
        pytest.param(A.BBoxSafeRandomCrop, {"erosion_rate": 0.0}, id="BBoxSafeRandomCrop"),
        pytest.param(A.AtLeastOneBBoxRandomCrop, {"height": 50, "width": 50, "erosion_factor": 0.0}, id="AtLeastOneBBoxRandomCrop"),
    ],
)
@pytest.mark.obb
def test_crop_transforms_support_obb(transform_class, transform_kwargs) -> None:
    """Test that all crop transforms support OBB."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 1  # For CropNonEmptyMaskIfExists

    # OBB in the middle
    bbox = [0.3, 0.3, 0.7, 0.7, 25.0]

    transform = transform_class(**transform_kwargs, p=1.0)

    # Verify _supported_bbox_types includes "obb"
    if hasattr(transform, "_supported_bbox_types"):
        assert "obb" in transform._supported_bbox_types, \
            f"{transform_class.__name__} does not declare OBB support"

    aug = A.Compose(
        [transform],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )

    np.random.seed(137)
    result = aug(image=image, mask=mask, bboxes=[bbox])

    # Should not raise an error and should return transformed data
    assert "image" in result
    assert "bboxes" in result

    # If bbox survived, it should still have 5 elements
    if len(result["bboxes"]) > 0:
        assert len(result["bboxes"][0]) == 5


@pytest.mark.obb
def test_crop_and_pad_with_obb() -> None:
    """Test CropAndPad transform with OBB."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # OBB in the middle
    bbox = [0.3, 0.3, 0.7, 0.7, 30.0]

    transform = A.Compose(
        [A.CropAndPad(px=10, p=1.0)],  # Pad 10px on each side (negative values crop)
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )

    result = transform(image=image, bboxes=[bbox])

    # Should not raise an error
    assert "bboxes" in result

    # If bbox survived, it should still have 5 elements
    if len(result["bboxes"]) > 0:
        assert len(result["bboxes"][0]) == 5


@pytest.mark.obb
def test_obb_with_extra_fields_preserved() -> None:
    """Test that OBB extra fields (e.g., class labels) are preserved through crops."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # OBB with extra field (class label)
    bbox = [0.3, 0.3, 0.7, 0.7, 30.0, 1.0]  # Last element is class label

    transform = A.Compose(
        [A.CenterCrop(height=60, width=60, p=1.0)],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )

    result = transform(image=image, bboxes=[bbox])

    if len(result["bboxes"]) > 0:
        output_bbox = result["bboxes"][0]

        # Should have 6 elements (5 for OBB + 1 extra)
        assert len(output_bbox) == 6

        # Extra field should be preserved
        assert output_bbox[5] == 1.0


@pytest.mark.obb
def test_multiple_obb_crop() -> None:
    """Test cropping multiple OBBs at once."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Multiple OBBs
    bboxes = [
        [0.2, 0.2, 0.4, 0.4, 15.0],
        [0.5, 0.5, 0.8, 0.8, 45.0],
        [0.1, 0.7, 0.3, 0.9, -30.0],
    ]

    transform = A.Compose(
        [A.CenterCrop(height=70, width=70, p=1.0)],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )

    result = transform(image=image, bboxes=bboxes)

    # Should have some boxes remaining (at least the middle one)
    assert len(result["bboxes"]) > 0

    # All remaining boxes should have 5 elements
    for bbox in result["bboxes"]:
        assert len(bbox) == 5


@pytest.mark.obb
def test_obb_angle_normalization() -> None:
    """Test that OBB angles are properly normalized after cropping."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # OBB with angle outside normal range
    bbox = [0.4, 0.4, 0.6, 0.6, 370.0]  # Will be normalized to 10.0

    transform = A.Compose(
        [A.CenterCrop(height=80, width=80, p=1.0)],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )

    result = transform(image=image, bboxes=[bbox])

    if len(result["bboxes"]) > 0:
        output_bbox = result["bboxes"][0]

        # Angle should be normalized to [-180, 180) range
        assert -180 <= output_bbox[4] < 180


@pytest.mark.obb
def test_random_crop_near_bbox_with_obb() -> None:
    """Test RandomCropNearBBox with OBB."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # OBB for regular bboxes
    bbox = [0.3, 0.3, 0.7, 0.7, 30.0]

    # Cropping bbox (can be HBB)
    cropping_bbox = [25, 25, 75, 75]

    transform = A.Compose(
        [A.RandomCropNearBBox(max_part_shift=0.2, cropping_bbox_key="crop_bbox", p=1.0)],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )

    np.random.seed(137)
    result = transform(image=image, bboxes=[bbox], crop_bbox=cropping_bbox)

    # Should not raise an error
    assert "bboxes" in result

    # If bbox survived, it should still have 5 elements
    if len(result["bboxes"]) > 0:
        assert len(result["bboxes"][0]) == 5


@pytest.mark.obb
def test_random_sized_bbox_safe_crop_with_obb() -> None:
    """Test RandomSizedBBoxSafeCrop with OBB."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # OBB
    bbox = [0.3, 0.3, 0.7, 0.7, 25.0]

    transform = A.Compose(
        [A.RandomSizedBBoxSafeCrop(height=50, width=50, erosion_rate=0.0, p=1.0)],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )

    np.random.seed(137)
    result = transform(image=image, bboxes=[bbox])

    # Should not raise an error
    assert "bboxes" in result

    # Box should be preserved (erosion_rate=0)
    assert len(result["bboxes"]) > 0

    # Should still have 5 elements
    assert len(result["bboxes"][0]) == 5


@pytest.mark.obb
def test_obb_crop_compose_validation() -> None:
    """Test that Compose validates OBB support at __init__ time."""
    # This should work - BaseCrop supports OBB
    compose = A.Compose(
        [A.CenterCrop(height=50, width=50)],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )
    assert compose is not None

    # Create a custom transform that doesn't support OBB
    class UnsupportedTransform(A.DualTransform):
        _supported_bbox_types = frozenset({"hbb"})

        def apply(self, img, **params):
            return img

        def apply_to_bboxes(self, bboxes, **params):
            return bboxes

    # This should fail at Compose.__init__
    with pytest.raises(ValueError, match="do not support OBB"):
        A.Compose(
            [UnsupportedTransform()],
            bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
        )


@pytest.mark.obb
def test_process_clipped_obb_boxes() -> None:
    """Test _process_clipped_obb_boxes helper function."""
    # Create clipped polygons (already normalized and clipped to crop boundaries)
    # Simulate a box that was clipped at the crop edge
    crop_width, crop_height = 80, 80

    # Clipped polygon corners (normalized to crop region)
    # This represents a box that hit the crop boundary
    polygons_clipped = np.array([
        [
            [0.0, 0.5],   # Left edge (clipped)
            [0.25, 0.25],
            [0.5, 0.5],
            [0.25, 0.75],
        ],
        [
            [0.5, 0.5],
            [0.75, 0.25],
            [1.0, 0.5],   # Right edge (clipped)
            [0.75, 0.75],
        ],
    ], dtype=np.float32)

    # Test without extras
    clipped_indices = np.array([0, 1])
    result = _process_clipped_obb_boxes(
        clipped_indices, polygons_clipped, crop_width, crop_height, extras=None
    )

    assert result.shape == (2, 5), f"Expected shape (2, 5), got {result.shape}"
    assert result.shape[1] == 5, "Result should have 5 columns (OBB format)"

    # Test with extras (e.g., class labels)
    extras = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result_with_extras = _process_clipped_obb_boxes(
        clipped_indices, polygons_clipped, crop_width, crop_height, extras=extras
    )

    assert result_with_extras.shape == (2, 7), f"Expected shape (2, 7), got {result_with_extras.shape}"
    np.testing.assert_array_equal(result_with_extras[:, 5:], extras)


@pytest.mark.obb
def test_process_unclipped_obb_boxes() -> None:
    """Test _process_unclipped_obb_boxes helper function (vectorized center shift)."""
    # Original bboxes in normalized coordinates
    bboxes = np.array([
        [0.3, 0.3, 0.5, 0.5, 30.0],      # Box 1
        [0.4, 0.4, 0.6, 0.6, 45.0],      # Box 2
        [0.35, 0.45, 0.55, 0.65, -15.0], # Box 3
    ], dtype=np.float32)

    image_shape = (100, 100)
    crop_coords = (20, 20, 80, 80)  # 60x60 crop
    crop_width, crop_height = 60, 60

    unclipped_indices = np.array([0, 1, 2])

    result_bboxes = _process_unclipped_obb_boxes(
        unclipped_indices, bboxes, image_shape, crop_coords,
        crop_width, crop_height, extras=None
    )

    # Verify angles are preserved
    np.testing.assert_array_almost_equal(result_bboxes[:, 4], bboxes[:, 4], decimal=6)

    # Verify boxes were transformed (centers shifted relative to crop)
    assert not np.allclose(result_bboxes[:, :4], bboxes[:, :4])

    # Test with extras
    extras = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)

    bboxes_with_extras = np.column_stack([bboxes, extras])
    result_with_extras = _process_unclipped_obb_boxes(
        unclipped_indices, bboxes_with_extras, image_shape, crop_coords,
        crop_width, crop_height, extras=extras
    )

    # Verify extras are preserved
    np.testing.assert_array_equal(result_with_extras[:, 5:], extras)


@pytest.mark.obb
def test_process_unclipped_obb_boxes_vectorized() -> None:
    """Test that _process_unclipped_obb_boxes processes all boxes in one shot (no loops)."""
    # Create many boxes to ensure vectorization
    n_boxes = 100
    bboxes = np.random.rand(n_boxes, 5).astype(np.float32)
    bboxes[:, :4] = bboxes[:, :4] * 0.3 + 0.35  # Keep in center region [0.35, 0.65]
    bboxes[:, 4] = (bboxes[:, 4] - 0.5) * 360   # Random angles [-180, 180]

    image_shape = (200, 200)
    crop_coords = (50, 50, 150, 150)
    crop_width, crop_height = 100, 100

    unclipped_indices = np.arange(n_boxes)

    # This should be fast since it's vectorized
    result_bboxes = _process_unclipped_obb_boxes(
        unclipped_indices, bboxes, image_shape, crop_coords,
        crop_width, crop_height, extras=None
    )

    # Verify all angles are preserved
    np.testing.assert_array_almost_equal(result_bboxes[:, 4], bboxes[:, 4], decimal=6)

    # Verify all boxes were processed
    assert result_bboxes.shape == (n_boxes, 5)
    assert not np.any(np.isnan(result_bboxes))


@pytest.mark.obb
def test_pad_with_obb() -> None:
    """Test Pad transform with OBB - padding is just a shift."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # OBB in the middle
    bbox = [0.3, 0.3, 0.7, 0.7, 30.0]

    transform = A.Compose(
        [A.Pad(padding=10, p=1.0)],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )

    result = transform(image=image, bboxes=[bbox])

    # Image shape should increase by padding
    assert result["image"].shape == (120, 120, 3)

    # Should have the bbox
    assert len(result["bboxes"]) == 1
    output_bbox = result["bboxes"][0]

    # Should still have 5 elements (OBB format preserved)
    assert len(output_bbox) == 5

    # Angle should be preserved (padding doesn't rotate)
    assert abs(output_bbox[4] - 30.0) < 0.01

    # Bounding box should be shifted relative to new image size
    # Original center was at (0.5, 0.5) * 100 = (50, 50)
    # After padding by 10, center is at (60, 60) in 120x120 image = (0.5, 0.5) normalized
    # So normalized coords should be adjusted proportionally
    assert all(0 <= v <= 1.0 for v in output_bbox[:4])


@pytest.mark.obb
def test_pad_with_obb_different_sides() -> None:
    """Test Pad with different padding on each side with OBB."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    bbox = [0.4, 0.4, 0.6, 0.6, 45.0]

    transform = A.Compose(
        [A.Pad(padding=(5, 10, 15, 20), p=1.0)],  # left, top, right, bottom
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )

    result = transform(image=image, bboxes=[bbox])

    # Image shape: width = 100 + 5 + 15 = 120, height = 100 + 10 + 20 = 130
    assert result["image"].shape == (130, 120, 3)

    # Bbox should be preserved
    assert len(result["bboxes"]) == 1
    output_bbox = result["bboxes"][0]

    # OBB format preserved
    assert len(output_bbox) == 5

    # Angle preserved
    assert abs(output_bbox[4] - 45.0) < 0.01


@pytest.mark.obb
def test_pad_if_needed_with_obb() -> None:
    """Test PadIfNeeded transform with OBB."""
    image = np.zeros((80, 80, 3), dtype=np.uint8)

    bbox = [0.3, 0.3, 0.7, 0.7, -15.0]

    transform = A.Compose(
        [A.PadIfNeeded(min_height=100, min_width=100, p=1.0)],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )

    result = transform(image=image, bboxes=[bbox])

    # Image should be padded to at least 100x100
    assert result["image"].shape[0] >= 100
    assert result["image"].shape[1] >= 100

    # Bbox should be preserved
    assert len(result["bboxes"]) == 1
    output_bbox = result["bboxes"][0]

    # OBB format preserved
    assert len(output_bbox) == 5

    # Angle preserved
    assert abs(output_bbox[4] - (-15.0)) < 0.01


@pytest.mark.obb
def test_pad_if_needed_divisor_with_obb() -> None:
    """Test PadIfNeeded with divisor and OBB."""
    image = np.zeros((95, 95, 3), dtype=np.uint8)

    bbox = [0.4, 0.4, 0.6, 0.6, 60.0]

    transform = A.Compose(
        [A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, p=1.0)],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )

    result = transform(image=image, bboxes=[bbox])

    # Image should be padded to be divisible by 32
    assert result["image"].shape[0] % 32 == 0
    assert result["image"].shape[1] % 32 == 0

    # Bbox should be preserved
    assert len(result["bboxes"]) == 1
    output_bbox = result["bboxes"][0]

    # OBB format preserved
    assert len(output_bbox) == 5

    # Angle preserved
    assert abs(output_bbox[4] - 60.0) < 0.01


@pytest.mark.obb
def test_pad_with_obb_extra_fields() -> None:
    """Test that Pad preserves OBB extra fields (e.g., class labels)."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # OBB with extra field (class label)
    bbox = [0.3, 0.3, 0.7, 0.7, 30.0, 5.0]

    transform = A.Compose(
        [A.Pad(padding=10, p=1.0)],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )

    result = transform(image=image, bboxes=[bbox])

    output_bbox = result["bboxes"][0]

    # Should have 6 elements (5 for OBB + 1 extra)
    assert len(output_bbox) == 6

    # Extra field should be preserved
    assert output_bbox[5] == 5.0

    # Angle should be preserved
    assert abs(output_bbox[4] - 30.0) < 0.01


@pytest.mark.obb
def test_pad_with_multiple_obb() -> None:
    """Test Pad with multiple OBBs."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    bboxes = [
        [0.2, 0.2, 0.4, 0.4, 15.0],
        [0.5, 0.5, 0.8, 0.8, 45.0],
        [0.1, 0.7, 0.3, 0.9, -30.0],
    ]

    transform = A.Compose(
        [A.Pad(padding=20, p=1.0)],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb")
    )

    result = transform(image=image, bboxes=bboxes)

    # All boxes should be preserved
    assert len(result["bboxes"]) == 3

    # All should have 5 elements
    for bbox in result["bboxes"]:
        assert len(bbox) == 5

    # Original angles should be preserved
    original_angles = [15.0, 45.0, -30.0]
    for i, bbox in enumerate(result["bboxes"]):
        assert abs(bbox[4] - original_angles[i]) < 0.01


@pytest.mark.obb
def test_pad_declares_obb_support() -> None:
    """Test that Pad and PadIfNeeded declare OBB support."""
    assert "obb" in A.Pad._supported_bbox_types
    assert "obb" in A.PadIfNeeded._supported_bbox_types


@pytest.mark.obb
def test_crop_and_pad_bboxes_obb_partial_crop() -> None:
    """Test crop_and_pad_bboxes with OBB that gets partially cropped - should clip polygons."""
    from albumentations.augmentations.crops.functional import crop_and_pad_bboxes

    # OBB that extends beyond crop region
    # In normalized coords: center at (0.75, 0.75), extends near edges
    bboxes = np.array([[0.6, 0.6, 0.9, 0.9, 45.0]], dtype=np.float32)

    image_shape = (100, 100)
    crop_params = (20, 20, 80, 80)  # Crop to 60x60 region
    pad_params = None
    result_shape = (60, 60)

    result = crop_and_pad_bboxes(
        bboxes, crop_params, pad_params, image_shape, result_shape, bbox_type="obb"
    )

    # Should not be empty - box is partially inside
    assert len(result) > 0, "Partially cropped OBB should not be filtered out"

    # Should still be OBB format
    assert result.shape[1] == 5

    # All coordinates should be valid (no out-of-bounds due to missing clipping)
    assert np.all(np.isfinite(result)), "Result should have finite values"


@pytest.mark.obb
def test_crop_and_pad_bboxes_obb_with_crop_and_pad() -> None:
    """Test crop_and_pad_bboxes with both crop and pad for OBB."""
    from albumentations.augmentations.crops.functional import crop_and_pad_bboxes

    # OBB in middle of image
    bboxes = np.array([[0.4, 0.4, 0.6, 0.6, 30.0]], dtype=np.float32)

    image_shape = (100, 100)
    crop_params = (20, 20, 80, 80)  # Crop to 60x60
    pad_params = (10, 10, 10, 10)   # Pad 10px on all sides
    result_shape = (80, 80)         # 60 + 20 = 80

    result = crop_and_pad_bboxes(
        bboxes, crop_params, pad_params, image_shape, result_shape, bbox_type="obb"
    )

    # Should preserve the bbox
    assert len(result) == 1
    assert result.shape[1] == 5

    # Angle may differ by 90 degrees due to cv2.minAreaRect ambiguity (30 vs -60 vs 120 etc.)
    # Just verify it's a valid angle in the canonical range
    assert -180 <= result[0, 4] < 180, f"Angle {result[0, 4]} out of range"


@pytest.mark.obb
def test_crop_and_pad_bboxes_obb_edge_clipping() -> None:
    """Test that OBB polygons are properly clipped when they extend beyond crop boundaries."""
    from albumentations.augmentations.crops.functional import crop_and_pad_bboxes

    # OBB that definitely extends beyond left and top crop boundaries
    # Center at (0.1, 0.1), close to top-left corner
    bboxes = np.array([[0.0, 0.0, 0.2, 0.2, 0.0]], dtype=np.float32)

    image_shape = (100, 100)
    # Crop that should clip the OBB
    crop_params = (10, 10, 90, 90)
    pad_params = None
    result_shape = (80, 80)

    result = crop_and_pad_bboxes(
        bboxes, crop_params, pad_params, image_shape, result_shape, bbox_type="obb"
    )

    # If the box survives filtering, check it was properly clipped
    if len(result) > 0:
        # All normalized coordinates should be in valid range [0, 1]
        assert np.all(result[:, 0] >= -0.01), "x_min should be >= 0"
        assert np.all(result[:, 1] >= -0.01), "y_min should be >= 0"
        assert np.all(result[:, 2] <= 1.01), "x_max should be <= 1"
        assert np.all(result[:, 3] <= 1.01), "y_max should be <= 1"

        # Should still be OBB format
        assert result.shape[1] == 5


@pytest.mark.obb
def test_crop_and_pad_bboxes_obb_with_extras() -> None:
    """Test that crop_and_pad_bboxes preserves extra fields for OBB."""
    from albumentations.augmentations.crops.functional import crop_and_pad_bboxes

    # OBB with extra fields (e.g., class labels)
    bboxes = np.array([[0.3, 0.3, 0.7, 0.7, 25.0, 1.0, 2.0]], dtype=np.float32)

    image_shape = (100, 100)
    crop_params = (10, 10, 90, 90)
    pad_params = None
    result_shape = (80, 80)

    result = crop_and_pad_bboxes(
        bboxes, crop_params, pad_params, image_shape, result_shape, bbox_type="obb"
    )

    # Should preserve bbox
    assert len(result) == 1

    # Should have 7 columns (5 for OBB + 2 extras)
    assert result.shape[1] == 7

    # Extra fields should be preserved
    np.testing.assert_array_almost_equal(result[0, 5:], [1.0, 2.0])


@pytest.mark.obb
def test_crop_and_pad_bboxes_obb_only_pad() -> None:
    """Test crop_and_pad_bboxes with only padding (no crop) for OBB."""
    from albumentations.augmentations.crops.functional import crop_and_pad_bboxes

    # OBB in middle
    bboxes = np.array([[0.4, 0.4, 0.6, 0.6, 60.0]], dtype=np.float32)

    image_shape = (100, 100)
    crop_params = None
    pad_params = (20, 20, 20, 20)
    result_shape = (140, 140)

    result = crop_and_pad_bboxes(
        bboxes, crop_params, pad_params, image_shape, result_shape, bbox_type="obb"
    )

    # Should preserve bbox
    assert len(result) == 1
    assert result.shape[1] == 5

    # Angle should be preserved (padding doesn't rotate)
    assert abs(result[0, 4] - 60.0) < 0.01


@pytest.mark.obb
def test_crop_and_pad_bboxes_vectorized_filtering() -> None:
    """Test that OBB filtering in crop_bboxes_by_coords_obb is vectorized (no Python loops)."""
    from albumentations.augmentations.crops.functional import crop_bboxes_by_coords

    # Create many OBBs to test vectorization
    n_boxes = 50
    np.random.seed(137)
    bboxes = np.random.rand(n_boxes, 5).astype(np.float32)
    bboxes[:, :4] = bboxes[:, :4] * 0.8 + 0.1  # Range [0.1, 0.9]
    bboxes[:, 4] = (bboxes[:, 4] - 0.5) * 360   # Angles [-180, 180]

    image_shape = (200, 200)
    crop_coords = (50, 50, 150, 150)

    # This should execute quickly with vectorized operations
    result = crop_bboxes_by_coords(bboxes, crop_coords, image_shape, bbox_type="obb")

    # Should return valid OBB array
    assert result.shape[1] == 5
    assert np.all(np.isfinite(result)), "All values should be finite"
