"""Analytical tests for OBB (Oriented Bounding Box) transformations using mask comparison.

This test file verifies OBB transformations by converting OBBs to pixel masks and comparing
them pixel-wise. This ensures we're testing the actual geometric transformation, not just
properties like center position.

For transformations where we can compute exact results mathematically, we:
1. Create an input OBB
2. Apply the transformation
3. Convert both input and output OBBs to masks
4. Apply the same transformation to the input mask
5. Compare the transformed mask with the mask from the output OBB

This validates that the OBB transformation matches the actual geometric operation.
"""

import cv2
import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.geometric import functional as fgeometric


def obb_to_mask(
    obb: list[float] | np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Convert OBB to a binary mask by drawing the rotated rectangle.

    Args:
        obb: [x_min, y_min, x_max, y_max, angle] in normalized coordinates
        image_shape: (height, width) of the output mask

    Returns:
        Binary mask with the OBB filled

    """
    from albumentations.core.bbox_utils import obb_to_polygons

    height, width = image_shape

    # Convert OBB to polygon using the same function the library uses
    obb_array = np.array([obb], dtype=np.float32)
    polygon = obb_to_polygons(obb_array)[0]  # Shape: (4, 2)

    # Convert normalized coordinates to pixels
    polygon_px = polygon.copy()
    polygon_px[:, 0] *= width
    polygon_px[:, 1] *= height
    polygon_px = np.int32(polygon_px)

    # Create mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Fill the polygon
    cv2.fillPoly(mask, [polygon_px], 1)

    return mask


@pytest.mark.obb
@pytest.mark.parametrize(
    "rotation_deg",
    [30, 45, 60, 90, 120, 180, 270],
)
def test_obb_rotation_centered_square_analytical(rotation_deg: int) -> None:
    """Test rotation of centered square box by comparing masks.

    We rotate both the OBB and a mask representation, then verify they match.
    """
    image_shape = (200, 200)
    image = np.zeros((*image_shape, 3), dtype=np.uint8)

    # Centered square box
    cx, cy = 0.5, 0.5
    size = 0.4
    initial_angle = 0.0

    input_obb = [
        cx - size / 2,
        cy - size / 2,
        cx + size / 2,
        cy + size / 2,
        initial_angle,
    ]

    # Create input mask from OBB
    input_mask = obb_to_mask(input_obb, image_shape)

    # Apply rotation to OBB
    transform = A.Compose(
        [
            A.Rotate(limit=(rotation_deg, rotation_deg), p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, mask=input_mask, bboxes=[input_obb])
    output_obb = result["bboxes"][0]
    rotated_mask = result["mask"]

    # Create mask from output OBB
    output_mask = obb_to_mask(output_obb, image_shape)

    # Compare masks - they should be very similar (allowing for discretization)
    # Use IoU (Intersection over Union) as metric
    intersection = np.logical_and(rotated_mask, output_mask).sum()
    union = np.logical_or(rotated_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.70, f"IoU {iou:.3f} too low for {rotation_deg}° rotation (expected > 0.70)"


@pytest.mark.obb
@pytest.mark.parametrize(
    "rotation_deg",
    [30, 45, 60, 90, 135, 180],
)
def test_obb_rotation_centered_rectangular_analytical(rotation_deg: int) -> None:
    """Test rotation of centered rectangular box by comparing masks."""
    image_shape = (200, 200)
    image = np.zeros((*image_shape, 3), dtype=np.uint8)

    # Centered rectangular box
    cx, cy = 0.5, 0.5
    width, height = 0.6, 0.4
    initial_angle = 0.0

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        initial_angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)

    transform = A.Compose(
        [
            A.Rotate(limit=(rotation_deg, rotation_deg), p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, mask=input_mask, bboxes=[input_obb])
    output_obb = result["bboxes"][0]
    rotated_mask = result["mask"]

    output_mask = obb_to_mask(output_obb, image_shape)

    intersection = np.logical_and(rotated_mask, output_mask).sum()
    union = np.logical_or(rotated_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.45, f"IoU {iou:.3f} too low for {rotation_deg}° rotation"


@pytest.mark.obb
@pytest.mark.parametrize(
    "rotation_deg,initial_angle",
    [
        (30, 15),
        (45, 30),
        (90, 45),
        (180, 60),
    ],
)
def test_obb_rotation_with_initial_angle_analytical(rotation_deg: int, initial_angle: float) -> None:
    """Test rotation of OBB that already has an angle by comparing masks."""
    image_shape = (200, 200)
    image = np.zeros((*image_shape, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.3

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        initial_angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)

    transform = A.Compose(
        [
            A.Rotate(limit=(rotation_deg, rotation_deg), p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, mask=input_mask, bboxes=[input_obb])
    output_obb = result["bboxes"][0]
    rotated_mask = result["mask"]

    output_mask = obb_to_mask(output_obb, image_shape)

    intersection = np.logical_and(rotated_mask, output_mask).sum()
    union = np.logical_or(rotated_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.45, f"IoU {iou:.3f} too low for rotation {rotation_deg}° with initial angle {initial_angle}°"


@pytest.mark.obb
@pytest.mark.parametrize(
    "offset_x,offset_y,rotation_deg",
    [
        (0.2, 0.0, 90),
        (0.0, 0.2, 90),
        (0.15, 0.15, 45),
        (-0.1, 0.1, 60),
    ],
)
def test_obb_rotation_offset_box_analytical(offset_x: float, offset_y: float, rotation_deg: int) -> None:
    """Test rotation of offset box by comparing masks."""
    image_shape = (200, 200)
    image = np.zeros((*image_shape, 3), dtype=np.uint8)

    cx, cy = 0.5 + offset_x, 0.5 + offset_y
    width, height = 0.2, 0.2
    initial_angle = 0.0

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        initial_angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)

    transform = A.Compose(
        [
            A.Rotate(limit=(rotation_deg, rotation_deg), p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, mask=input_mask, bboxes=[input_obb])
    output_obb = result["bboxes"][0]
    rotated_mask = result["mask"]

    output_mask = obb_to_mask(output_obb, image_shape)

    intersection = np.logical_and(rotated_mask, output_mask).sum()
    union = np.logical_or(rotated_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.70, f"IoU {iou:.3f} too low for offset rotation"


@pytest.mark.obb
def test_obb_horizontal_flip_centered_box_analytical() -> None:
    """Test horizontal flip of centered box by comparing masks."""
    image_shape = (200, 200)
    image = np.zeros((*image_shape, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.3
    angle = 30.0

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)

    transform = A.Compose(
        [
            A.HorizontalFlip(p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, mask=input_mask, bboxes=[input_obb])
    output_obb = result["bboxes"][0]
    flipped_mask = result["mask"]

    output_mask = obb_to_mask(output_obb, image_shape)

    intersection = np.logical_and(flipped_mask, output_mask).sum()
    union = np.logical_or(flipped_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.98, f"IoU {iou:.3f} too low for horizontal flip (expected > 0.98)"


@pytest.mark.obb
def test_obb_vertical_flip_centered_box_analytical() -> None:
    """Test vertical flip of centered box by comparing masks."""
    image_shape = (200, 200)
    image = np.zeros((*image_shape, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.3
    angle = 30.0

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)

    transform = A.Compose(
        [
            A.VerticalFlip(p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, mask=input_mask, bboxes=[input_obb])
    output_obb = result["bboxes"][0]
    flipped_mask = result["mask"]

    output_mask = obb_to_mask(output_obb, image_shape)

    intersection = np.logical_and(flipped_mask, output_mask).sum()
    union = np.logical_or(flipped_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.98, f"IoU {iou:.3f} too low for vertical flip"


@pytest.mark.obb
@pytest.mark.parametrize(
    "offset_x,offset_y",
    [
        (0.2, 0.0),
        (-0.15, 0.1),
        (0.1, -0.1),
    ],
)
def test_obb_horizontal_flip_offset_box_analytical(offset_x: float, offset_y: float) -> None:
    """Test horizontal flip of offset box by comparing masks."""
    image_shape = (200, 200)
    image = np.zeros((*image_shape, 3), dtype=np.uint8)

    cx, cy = 0.5 + offset_x, 0.5 + offset_y
    width, height = 0.2, 0.15
    angle = 25.0

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)

    transform = A.Compose(
        [
            A.HorizontalFlip(p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, mask=input_mask, bboxes=[input_obb])
    output_obb = result["bboxes"][0]
    flipped_mask = result["mask"]

    output_mask = obb_to_mask(output_obb, image_shape)

    intersection = np.logical_and(flipped_mask, output_mask).sum()
    union = np.logical_or(flipped_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.98, f"IoU {iou:.3f} too low for horizontal flip with offset ({offset_x}, {offset_y})"


@pytest.mark.obb
@pytest.mark.parametrize(
    "offset_x,offset_y",
    [
        (0.0, 0.2),
        (0.1, -0.15),
        (-0.1, 0.1),
    ],
)
def test_obb_vertical_flip_offset_box_analytical(offset_x: float, offset_y: float) -> None:
    """Test vertical flip of offset box by comparing masks."""
    image_shape = (200, 200)
    image = np.zeros((*image_shape, 3), dtype=np.uint8)

    cx, cy = 0.5 + offset_x, 0.5 + offset_y
    width, height = 0.2, 0.15
    angle = 25.0

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)

    transform = A.Compose(
        [
            A.VerticalFlip(p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, mask=input_mask, bboxes=[input_obb])
    output_obb = result["bboxes"][0]
    flipped_mask = result["mask"]

    output_mask = obb_to_mask(output_obb, image_shape)

    intersection = np.logical_and(flipped_mask, output_mask).sum()
    union = np.logical_or(flipped_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.98, f"IoU {iou:.3f} too low for vertical flip with offset ({offset_x}, {offset_y})"


@pytest.mark.obb
@pytest.mark.parametrize(
    "k",
    [1, 2, 3],
)
def test_obb_rot90_centered_box_analytical(k: int) -> None:
    """Test 90-degree rotations by comparing masks."""
    image_shape = (200, 200)

    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.3
    angle = 15.0

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)

    # Apply rot90 to mask directly
    rotated_mask = np.rot90(input_mask, k)

    # Apply rot90 to OBB using functional API
    bboxes = np.array([input_obb], dtype=np.float32)
    result_bboxes = fgeometric.bboxes_rot90(bboxes, k, bbox_type="obb")
    output_obb = result_bboxes[0]

    # Create mask from output OBB
    output_mask = obb_to_mask(output_obb, image_shape)

    intersection = np.logical_and(rotated_mask, output_mask).sum()
    union = np.logical_or(rotated_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.55, f"IoU {iou:.3f} too low for rot90 with k={k}"


@pytest.mark.obb
@pytest.mark.parametrize(
    "offset_x,offset_y,k",
    [
        (0.2, 0.0, 1),
        (0.0, 0.2, 1),
        (0.15, 0.1, 2),
        (-0.1, 0.15, 3),
    ],
)
def test_obb_rot90_offset_box_analytical(offset_x: float, offset_y: float, k: int) -> None:
    """Test 90-degree rotations of offset box by comparing masks."""
    image_shape = (200, 200)

    cx, cy = 0.5 + offset_x, 0.5 + offset_y
    width, height = 0.2, 0.15
    angle = 20.0

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)
    rotated_mask = np.rot90(input_mask, k)

    bboxes = np.array([input_obb], dtype=np.float32)
    result_bboxes = fgeometric.bboxes_rot90(bboxes, k, bbox_type="obb")
    output_obb = result_bboxes[0]

    output_mask = obb_to_mask(output_obb, image_shape)

    intersection = np.logical_and(rotated_mask, output_mask).sum()
    union = np.logical_or(rotated_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.55, f"IoU {iou:.3f} too low for rot90 with k={k}, offset=({offset_x}, {offset_y})"


@pytest.mark.obb
def test_obb_transpose_centered_box_analytical() -> None:
    """Test transpose of centered box by comparing masks."""
    image_shape = (200, 200)
    image = np.zeros((*image_shape, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.3
    angle = 25.0

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)

    transform = A.Compose(
        [
            A.Transpose(p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, mask=input_mask, bboxes=[input_obb])
    output_obb = result["bboxes"][0]
    transposed_mask = result["mask"]

    output_mask = obb_to_mask(output_obb, image_shape)

    intersection = np.logical_and(transposed_mask, output_mask).sum()
    union = np.logical_or(transposed_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.55, f"IoU {iou:.3f} too low for transpose"


@pytest.mark.obb
@pytest.mark.parametrize(
    "offset_x,offset_y",
    [
        (0.2, 0.1),
        (-0.1, 0.15),
        (0.15, -0.1),
    ],
)
def test_obb_transpose_offset_box_analytical(offset_x: float, offset_y: float) -> None:
    """Test transpose of offset box by comparing masks."""
    image_shape = (200, 200)
    image = np.zeros((*image_shape, 3), dtype=np.uint8)

    cx, cy = 0.5 + offset_x, 0.5 + offset_y
    width, height = 0.2, 0.15
    angle = 30.0

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)

    transform = A.Compose(
        [
            A.Transpose(p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, mask=input_mask, bboxes=[input_obb])
    output_obb = result["bboxes"][0]
    transposed_mask = result["mask"]

    output_mask = obb_to_mask(output_obb, image_shape)

    intersection = np.logical_and(transposed_mask, output_mask).sum()
    union = np.logical_or(transposed_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.55, f"IoU {iou:.3f} too low for transpose with offset ({offset_x}, {offset_y})"


@pytest.mark.obb
def test_obb_identity_transform_analytical() -> None:
    """Test that identity transform preserves OBB exactly by comparing masks."""
    image_shape = (200, 200)
    image = np.zeros((*image_shape, 3), dtype=np.uint8)

    cx, cy = 0.4, 0.6
    width, height = 0.3, 0.25
    angle = 42.0

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)

    transform = A.Compose(
        [
            A.Affine(scale=1.0, rotate=0, translate_px={"x": 0, "y": 0}, p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, mask=input_mask, bboxes=[input_obb])
    output_obb = result["bboxes"][0]
    output_image_mask = result["mask"]

    output_mask = obb_to_mask(output_obb, image_shape)

    # For identity transform, masks should be identical
    assert np.array_equal(input_mask, output_image_mask), "Identity transform should not change mask"
    assert np.array_equal(input_mask, output_mask), "Output OBB mask should match input mask"


@pytest.mark.obb
def test_obb_360_rotation_analytical() -> None:
    """Test that 360° rotation returns close to original by comparing masks."""
    image_shape = (200, 200)
    image = np.zeros((*image_shape, 3), dtype=np.uint8)

    cx, cy = 0.4, 0.6
    width, height = 0.3, 0.2
    angle = 25.0

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)

    transform = A.Compose(
        [
            A.Rotate(limit=(360, 360), p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, mask=input_mask, bboxes=[input_obb])
    output_obb = result["bboxes"][0]
    result["mask"]

    output_mask = obb_to_mask(output_obb, image_shape)

    # After 360° rotation, mask should be very similar to original
    intersection = np.logical_and(input_mask, output_mask).sum()
    union = np.logical_or(input_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.90, f"IoU {iou:.3f} too low for 360° rotation (expected > 0.90)"


@pytest.mark.obb
def test_obb_combined_transforms_analytical() -> None:
    """Test combined transformations by comparing masks."""
    image_shape = (200, 200)
    image = np.zeros((*image_shape, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.3
    angle = 0.0

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)

    # Apply HFlip then Rotate 90°
    transform = A.Compose(
        [
            A.HorizontalFlip(p=1.0),
            A.Rotate(limit=(90, 90), p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, mask=input_mask, bboxes=[input_obb])
    output_obb = result["bboxes"][0]
    transformed_mask = result["mask"]

    output_mask = obb_to_mask(output_obb, image_shape)

    intersection = np.logical_and(transformed_mask, output_mask).sum()
    union = np.logical_or(transformed_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.70, f"IoU {iou:.3f} too low for combined HFlip + Rotate"


@pytest.mark.obb
@pytest.mark.parametrize(
    "width,height,angle",
    [
        (0.3, 0.2, 0),
        (0.4, 0.3, 30),
        (0.5, 0.2, 45),
        (0.3, 0.3, 60),
    ],
)
def test_obb_mask_conversion_roundtrip(width: float, height: float, angle: float) -> None:
    """Test that OBB -> mask -> visual representation is consistent.

    This validates our obb_to_mask function works correctly.
    """
    image_shape = (200, 200)

    cx, cy = 0.5, 0.5
    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    # Create mask from OBB
    mask = obb_to_mask(input_obb, image_shape)

    # Basic sanity checks
    assert mask.shape == image_shape, "Mask shape should match image shape"
    assert mask.dtype == np.uint8, "Mask should be uint8"
    assert np.all((mask == 0) | (mask == 1)), "Mask should be binary"

    # Check that mask has reasonable area
    expected_area_pixels = width * height * image_shape[0] * image_shape[1]
    actual_area = mask.sum()

    # Allow 20% tolerance for discretization and rotation effects
    assert 0.8 * expected_area_pixels <= actual_area <= 1.2 * expected_area_pixels, (
        f"Mask area {actual_area} should be close to expected {expected_area_pixels:.0f}"
    )


@pytest.mark.obb
@pytest.mark.parametrize(
    "scale,rotation_deg",
    [
        (1.0, 0),
        (1.0, 45),
        (1.2, 30),
        (0.8, 90),
    ],
)
def test_obb_affine_identity_and_simple_transforms_mask(scale: float, rotation_deg: int) -> None:
    """Test Affine transforms with mask comparison.

    Validates that OBB transformation matches actual image transformation.
    """
    image_shape = (200, 200)
    image = np.zeros((*image_shape, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.3
    angle = 20.0

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)

    transform = A.Compose(
        [
            A.Affine(
                scale=scale,
                rotate=rotation_deg,
                translate_px=0,
                shear=0,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, mask=input_mask, bboxes=[input_obb])
    output_obb = result["bboxes"][0]
    transformed_mask = result["mask"]

    output_mask = obb_to_mask(output_obb, image_shape)

    intersection = np.logical_and(transformed_mask, output_mask).sum()
    union = np.logical_or(transformed_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.60, f"IoU {iou:.3f} too low for Affine(scale={scale}, rotate={rotation_deg})"


@pytest.mark.obb
@pytest.mark.parametrize(
    "translate_x,translate_y",
    [
        (0.1, 0.0),
        (0.0, 0.1),
        (0.15, 0.1),
    ],
)
def test_obb_affine_translation_mask(translate_x: float, translate_y: float) -> None:
    """Test Affine translation by comparing masks."""
    image_shape = (200, 200)
    image = np.zeros((*image_shape, 3), dtype=np.uint8)

    cx, cy = 0.4, 0.4
    width, height = 0.3, 0.2
    angle = 30.0

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)

    transform = A.Compose(
        [
            A.Affine(
                scale=1.0,
                rotate=0,
                translate_percent={"x": translate_x, "y": translate_y},
                shear=0,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, mask=input_mask, bboxes=[input_obb])
    output_obb = result["bboxes"][0]
    transformed_mask = result["mask"]

    output_mask = obb_to_mask(output_obb, image_shape)

    intersection = np.logical_and(transformed_mask, output_mask).sum()
    union = np.logical_or(transformed_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.60, f"IoU {iou:.3f} too low for translation ({translate_x}, {translate_y})"


@pytest.mark.obb
@pytest.mark.parametrize(
    "scale",
    [0.5, 0.8, 1.2, 1.5],
)
def test_obb_affine_scaling_mask(scale: float) -> None:
    """Test Affine scaling by comparing masks."""
    image_shape = (200, 200)
    image = np.zeros((*image_shape, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.3
    angle = 25.0

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)

    transform = A.Compose(
        [
            A.Affine(
                scale=scale,
                rotate=0,
                translate_px=0,
                shear=0,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, mask=input_mask, bboxes=[input_obb])
    output_obb = result["bboxes"][0]
    transformed_mask = result["mask"]

    output_mask = obb_to_mask(output_obb, image_shape)

    intersection = np.logical_and(transformed_mask, output_mask).sum()
    union = np.logical_or(transformed_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.60, f"IoU {iou:.3f} too low for scale={scale}"


@pytest.mark.obb
@pytest.mark.parametrize(
    "shear_x,shear_y",
    [
        (10, 0),
        (0, 10),
        (15, 10),
    ],
)
def test_obb_affine_shear_mask(shear_x: float, shear_y: float) -> None:
    """Test Affine shear by comparing masks."""
    image_shape = (200, 200)
    image = np.zeros((*image_shape, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.3, 0.2
    angle = 0.0  # Start unrotated for clarity

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)

    transform = A.Compose(
        [
            A.Affine(
                scale=1.0,
                rotate=0,
                translate_px=0,
                shear={"x": shear_x, "y": shear_y},
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, mask=input_mask, bboxes=[input_obb])
    output_obb = result["bboxes"][0]
    transformed_mask = result["mask"]

    output_mask = obb_to_mask(output_obb, image_shape)

    intersection = np.logical_and(transformed_mask, output_mask).sum()
    union = np.logical_or(transformed_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.55, f"IoU {iou:.3f} too low for shear ({shear_x}, {shear_y})"


@pytest.mark.obb
@pytest.mark.parametrize(
    "scale,rotation_deg,translate",
    [
        (1.2, 30, 0.1),
        (0.8, 45, 0.05),
        (1.1, 60, 0.15),
    ],
)
def test_obb_affine_combined_transforms_mask(scale: float, rotation_deg: int, translate: float) -> None:
    """Test combined Affine transformations by comparing masks."""
    image_shape = (200, 200)
    image = np.zeros((*image_shape, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.3, 0.25
    angle = 15.0

    input_obb = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    input_mask = obb_to_mask(input_obb, image_shape)

    transform = A.Compose(
        [
            A.Affine(
                scale=scale,
                rotate=rotation_deg,
                translate_percent=translate,
                shear=0,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, mask=input_mask, bboxes=[input_obb])
    output_obb = result["bboxes"][0]
    transformed_mask = result["mask"]

    output_mask = obb_to_mask(output_obb, image_shape)

    intersection = np.logical_and(transformed_mask, output_mask).sum()
    union = np.logical_or(transformed_mask, output_mask).sum()
    iou = intersection / union if union > 0 else 0

    assert iou > 0.60, (
        f"IoU {iou:.3f} too low for combined transform (scale={scale}, rotate={rotation_deg}, translate={translate})"
    )
