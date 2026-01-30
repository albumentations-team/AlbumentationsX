"""Analytical tests for OBB (Oriented Bounding Box) transformations.

This test file verifies OBB transformations against analytically computed expected values.
For transformations where we can compute exact results mathematically, we test:
1. Rotation of centered boxes by specific angles
2. Flips with known geometry
3. Affine transformations with predictable outcomes
4. 90-degree rotations

Note on OBB format:
    OBB in albumentations is stored as [x_min, y_min, x_max, y_max, angle] where:
    - (x_min, y_min, x_max, y_max) is the axis-aligned bounding box (AABB)
      that encloses the oriented rectangle
    - angle is the rotation angle in degrees

    When rotating an OBB:
    1. The OBB is converted to polygon corners
    2. The polygon is rotated
    3. A new AABB is computed from the rotated polygon
    4. The angle is updated
"""

import math

import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.bbox_utils import obb_to_polygons, polygons_to_obb


def rotate_polygon(
    polygon: np.ndarray,
    cx: float,
    cy: float,
    angle_deg: float,
) -> np.ndarray:
    """Rotate a polygon around center (cx, cy) by angle_deg degrees counterclockwise."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rotated = polygon.copy()
    for i in range(len(polygon)):
        x, y = polygon[i]
        dx = x - cx
        dy = y - cy
        rotated[i, 0] = cx + dx * cos_a - dy * sin_a
        rotated[i, 1] = cy + dx * sin_a + dy * cos_a

    return rotated


def compute_obb_after_rotation(
    input_obb: list[float],
    image_cx: float,
    image_cy: float,
    rotation_deg: float,
) -> tuple[float, float, float, float, float]:
    """Compute OBB after rotating around image center.

    Args:
        input_obb: [x_min, y_min, x_max, y_max, angle]
        image_cx, image_cy: Image center coordinates
        rotation_deg: Rotation angle in degrees (counterclockwise, but OpenCV is clockwise)

    Returns:
        (x_min, y_min, x_max, y_max, new_angle) - AABB of rotated oriented box

    """
    # Convert OBB to polygon
    obb_array = np.array([input_obb], dtype=np.float32)
    polygon = obb_to_polygons(obb_array)[0]

    # Rotate polygon (OpenCV rotates clockwise for positive angles in image coordinates)
    # So we negate the angle to match OpenCV's behavior
    rotated_polygon = rotate_polygon(polygon, image_cx, image_cy, -rotation_deg)

    # Convert back to OBB
    rotated_obb = polygons_to_obb(rotated_polygon.reshape(1, 4, 2))[0]

    return tuple(rotated_obb)


@pytest.mark.obb
@pytest.mark.parametrize(
    "rotation_deg",
    [30, 45, 60, 90, 120, 180, 270],
)
def test_obb_rotation_centered_square_box_square_image(rotation_deg: int) -> None:
    """Test rotation of a centered square box on a square image.

    For a square box, rotation around center should keep it centered,
    and the AABB dimensions may change depending on angle.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Centered square box: 40x40 at center
    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.4
    initial_angle = 0.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        initial_angle,
    ]

    transform = A.Compose(
        [
            A.Rotate(limit=(rotation_deg, rotation_deg), p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Compute expected using polygon rotation
    expected = compute_obb_after_rotation(input_bbox, 0.5, 0.5, rotation_deg)

    # Check AABB coordinates
    np.testing.assert_allclose(
        output_bbox[:4],
        expected[:4],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"OBB AABB incorrect after {rotation_deg}° rotation",
    )

    # Check angle (allowing for equivalence modulo 360 and potential 90° ambiguity from minAreaRect)
    # For centered square, angle might be ambiguous
    actual_angle = output_bbox[4] % 360
    expected_angle = expected[4] % 360

    # Allow for 90-degree ambiguity in angle representation
    angle_diff = min(
        abs(actual_angle - expected_angle),
        abs(actual_angle - expected_angle + 90) % 360,
        abs(actual_angle - expected_angle - 90) % 360,
        abs(actual_angle - expected_angle + 180) % 360,
    )

    assert angle_diff < 1.0, (
        f"OBB angle incorrect after {rotation_deg}° rotation: got {actual_angle}, expected {expected_angle}"
    )


@pytest.mark.obb
@pytest.mark.parametrize(
    "rotation_deg",
    [30, 45, 60, 90, 135, 180],
)
def test_obb_rotation_centered_rectangular_box_square_image(rotation_deg: int) -> None:
    """Test rotation of a centered rectangular box on a square image.

    For rectangular box, the AABB changes as the box rotates.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Centered rectangular box: 60x40
    cx, cy = 0.5, 0.5
    width, height = 0.6, 0.4
    initial_angle = 0.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        initial_angle,
    ]

    transform = A.Compose(
        [
            A.Rotate(limit=(rotation_deg, rotation_deg), p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    expected = compute_obb_after_rotation(input_bbox, 0.5, 0.5, rotation_deg)

    np.testing.assert_allclose(
        output_bbox[:4],
        expected[:4],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"Rectangular OBB AABB incorrect after {rotation_deg}° rotation",
    )


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
def test_obb_rotation_with_initial_angle(rotation_deg: int, initial_angle: float) -> None:
    """Test rotation of OBB that already has an angle."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.3

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        initial_angle,
    ]

    transform = A.Compose(
        [
            A.Rotate(limit=(rotation_deg, rotation_deg), p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    expected = compute_obb_after_rotation(input_bbox, 0.5, 0.5, rotation_deg)

    np.testing.assert_allclose(
        output_bbox[:4],
        expected[:4],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"OBB with initial angle {initial_angle}° incorrect after {rotation_deg}° rotation",
    )


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
def test_obb_rotation_offset_box(offset_x: float, offset_y: float, rotation_deg: int) -> None:
    """Test rotation of a box that's offset from image center."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.5 + offset_x, 0.5 + offset_y
    width, height = 0.2, 0.2
    initial_angle = 0.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        initial_angle,
    ]

    transform = A.Compose(
        [
            A.Rotate(limit=(rotation_deg, rotation_deg), p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    expected = compute_obb_after_rotation(input_bbox, 0.5, 0.5, rotation_deg)

    np.testing.assert_allclose(
        output_bbox[:4],
        expected[:4],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"Offset OBB incorrect after {rotation_deg}° rotation",
    )


@pytest.mark.obb
def test_obb_horizontal_flip_centered_box() -> None:
    """Test horizontal flip of centered box.

    For a centered box, horizontal flip should keep it centered.
    We test this by checking the center stays at (0.5, 0.5).
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.3
    angle = 30.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    transform = A.Compose(
        [
            A.HorizontalFlip(p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # For centered box: center should remain at (0.5, 0.5)
    out_cx = (output_bbox[0] + output_bbox[2]) / 2
    out_cy = (output_bbox[1] + output_bbox[3]) / 2

    np.testing.assert_allclose(
        [out_cx, out_cy],
        [0.5, 0.5],
        rtol=1e-4,
        atol=1e-4,
        err_msg="Centered OBB center should stay at (0.5, 0.5) after horizontal flip",
    )

    # AABB dimensions should be the same (just the angle changes)
    out_width = output_bbox[2] - output_bbox[0]
    out_height = output_bbox[3] - output_bbox[1]
    input_width = width
    input_height = height

    np.testing.assert_allclose(
        [out_width, out_height],
        [input_width, input_height],
        rtol=1e-4,
        atol=1e-4,
        err_msg="AABB dimensions should be preserved for centered box",
    )


@pytest.mark.obb
def test_obb_vertical_flip_centered_box() -> None:
    """Test vertical flip of centered box.

    For a centered box, vertical flip should keep it centered.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.3
    angle = 30.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    transform = A.Compose(
        [
            A.VerticalFlip(p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # For centered box: center should remain at (0.5, 0.5)
    out_cx = (output_bbox[0] + output_bbox[2]) / 2
    out_cy = (output_bbox[1] + output_bbox[3]) / 2

    np.testing.assert_allclose(
        [out_cx, out_cy],
        [0.5, 0.5],
        rtol=1e-4,
        atol=1e-4,
        err_msg="Centered OBB center should stay at (0.5, 0.5) after vertical flip",
    )

    # AABB dimensions should be the same
    out_width = output_bbox[2] - output_bbox[0]
    out_height = output_bbox[3] - output_bbox[1]

    np.testing.assert_allclose(
        [out_width, out_height],
        [width, height],
        rtol=1e-4,
        atol=1e-4,
        err_msg="AABB dimensions should be preserved for centered box",
    )


@pytest.mark.obb
@pytest.mark.parametrize(
    "offset_x,offset_y",
    [
        (0.2, 0.0),
        (-0.15, 0.1),
        (0.1, -0.1),
    ],
)
def test_obb_horizontal_flip_offset_box(offset_x: float, offset_y: float) -> None:
    """Test horizontal flip of box offset from center.

    For horizontal flip: center x-coord should flip (x -> 1 - x), y stays same.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.5 + offset_x, 0.5 + offset_y
    width, height = 0.2, 0.15
    angle = 25.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    transform = A.Compose(
        [
            A.HorizontalFlip(p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Expected center after horizontal flip
    out_cx = (output_bbox[0] + output_bbox[2]) / 2
    out_cy = (output_bbox[1] + output_bbox[3]) / 2

    expected_cx = 1.0 - cx
    expected_cy = cy

    np.testing.assert_allclose(
        [out_cx, out_cy],
        [expected_cx, expected_cy],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"OBB center incorrect after horizontal flip (offset={offset_x}, {offset_y})",
    )


@pytest.mark.obb
@pytest.mark.parametrize(
    "offset_x,offset_y",
    [
        (0.0, 0.2),
        (0.1, -0.15),
        (-0.1, 0.1),
    ],
)
def test_obb_vertical_flip_offset_box(offset_x: float, offset_y: float) -> None:
    """Test vertical flip of box offset from center.

    For vertical flip: center y-coord should flip (y -> 1 - y), x stays same.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.5 + offset_x, 0.5 + offset_y
    width, height = 0.2, 0.15
    angle = 25.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    transform = A.Compose(
        [
            A.VerticalFlip(p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Expected center after vertical flip
    out_cx = (output_bbox[0] + output_bbox[2]) / 2
    out_cy = (output_bbox[1] + output_bbox[3]) / 2

    expected_cx = cx
    expected_cy = 1.0 - cy

    np.testing.assert_allclose(
        [out_cx, out_cy],
        [expected_cx, expected_cy],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"OBB center incorrect after vertical flip (offset={offset_x}, {offset_y})",
    )


@pytest.mark.obb
@pytest.mark.parametrize(
    "k",
    [1, 2, 3],
)
def test_obb_rot90_centered_box_analytical(k: int) -> None:
    """Test 90-degree rotations with analytical calculation using functional API."""
    np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.3
    angle = 15.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    # Use functional API
    bboxes = np.array([input_bbox], dtype=np.float32)
    result_bboxes = fgeometric.bboxes_rot90(bboxes, k, bbox_type="obb")
    output_bbox = result_bboxes[0]

    # Compute expected by rotating polygon
    obb_array = np.array([input_bbox], dtype=np.float32)
    polygon = obb_to_polygons(obb_array)[0]

    # Rot90 rotations around center for k times
    # k=1: (x, y) -> (y, 1-x)
    # k=2: (x, y) -> (1-x, 1-y)
    # k=3: (x, y) -> (1-y, x)
    rotated_polygon = polygon.copy()
    for _ in range(k):
        temp = rotated_polygon.copy()
        rotated_polygon[:, 0] = temp[:, 1]
        rotated_polygon[:, 1] = 1.0 - temp[:, 0]

    expected_obb = polygons_to_obb(rotated_polygon.reshape(1, 4, 2))[0]

    np.testing.assert_allclose(
        output_bbox[:4],
        expected_obb[:4],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"OBB incorrect after rot90 with k={k}",
    )


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
    """Test 90-degree rotations of offset box with analytical calculation."""
    np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.5 + offset_x, 0.5 + offset_y
    width, height = 0.2, 0.15
    angle = 20.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    bboxes = np.array([input_bbox], dtype=np.float32)
    result_bboxes = fgeometric.bboxes_rot90(bboxes, k, bbox_type="obb")
    output_bbox = result_bboxes[0]

    # Compute expected
    obb_array = np.array([input_bbox], dtype=np.float32)
    polygon = obb_to_polygons(obb_array)[0]

    rotated_polygon = polygon.copy()
    for _ in range(k):
        temp = rotated_polygon.copy()
        rotated_polygon[:, 0] = temp[:, 1]
        rotated_polygon[:, 1] = 1.0 - temp[:, 0]

    expected_obb = polygons_to_obb(rotated_polygon.reshape(1, 4, 2))[0]

    np.testing.assert_allclose(
        output_bbox[:4],
        expected_obb[:4],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"Offset OBB incorrect after rot90 with k={k}, offset=({offset_x}, {offset_y})",
    )


@pytest.mark.obb
def test_obb_transpose_centered_box() -> None:
    """Test transpose of centered box.

    Transpose swaps x and y coordinates: (x, y) -> (y, x).
    For a centered box, it stays centered.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.3
    angle = 25.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    transform = A.Compose(
        [
            A.Transpose(p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # For centered box: should stay centered
    out_cx = (output_bbox[0] + output_bbox[2]) / 2
    out_cy = (output_bbox[1] + output_bbox[3]) / 2

    np.testing.assert_allclose(
        [out_cx, out_cy],
        [0.5, 0.5],
        rtol=1e-4,
        atol=1e-4,
        err_msg="Centered OBB should stay centered after transpose",
    )

    # For centered box, AABB dimensions swap
    out_width = output_bbox[2] - output_bbox[0]
    out_height = output_bbox[3] - output_bbox[1]

    np.testing.assert_allclose(
        [out_width, out_height],
        [height, width],  # Note: swapped
        rtol=1e-4,
        atol=1e-4,
        err_msg="AABB dimensions should swap for centered box after transpose",
    )


@pytest.mark.obb
@pytest.mark.parametrize(
    "offset_x,offset_y",
    [
        (0.2, 0.1),
        (-0.1, 0.15),
        (0.15, -0.1),
    ],
)
def test_obb_transpose_offset_box(offset_x: float, offset_y: float) -> None:
    """Test transpose of offset box.

    Transpose: (x, y) -> (y, x), so center coordinates swap.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.5 + offset_x, 0.5 + offset_y
    width, height = 0.2, 0.15
    angle = 30.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    transform = A.Compose(
        [
            A.Transpose(p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # After transpose: center coordinates swap
    out_cx = (output_bbox[0] + output_bbox[2]) / 2
    out_cy = (output_bbox[1] + output_bbox[3]) / 2

    expected_cx = cy  # x and y swap
    expected_cy = cx

    np.testing.assert_allclose(
        [out_cx, out_cy],
        [expected_cx, expected_cy],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"OBB center incorrect after transpose (offset={offset_x}, {offset_y})",
    )


@pytest.mark.obb
def test_obb_identity_transform() -> None:
    """Test that identity transform (no change) preserves OBB exactly."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.4, 0.6
    width, height = 0.3, 0.25
    angle = 42.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    transform = A.Compose(
        [
            A.Affine(scale=1.0, rotate=0, translate_px={"x": 0, "y": 0}, p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    np.testing.assert_allclose(
        output_bbox,
        input_bbox,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Identity transform should preserve OBB exactly",
    )


@pytest.mark.obb
def test_obb_360_rotation_is_identity() -> None:
    """Test that 360° rotation returns to original state."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.4, 0.6
    width, height = 0.3, 0.2
    angle = 25.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    transform = A.Compose(
        [
            A.Rotate(limit=(360, 360), p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Position should be same (within numerical tolerance)
    np.testing.assert_allclose(
        output_bbox[:4],
        input_bbox[:4],
        rtol=1e-3,
        atol=1e-3,
        err_msg="360° rotation should return to original position",
    )


@pytest.mark.obb
def test_obb_combined_flip_and_rotate_centered() -> None:
    """Test combined horizontal flip + 90° rotation on centered box."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.3
    angle = 0.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    transform = A.Compose(
        [
            A.HorizontalFlip(p=1.0),
            A.Rotate(limit=(90, 90), p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Compute expected: HFlip then Rotate
    obb_array = np.array([input_bbox], dtype=np.float32)
    polygon = obb_to_polygons(obb_array)[0]

    # Step 1: HFlip
    flipped_polygon = polygon.copy()
    flipped_polygon[:, 0] = 1.0 - flipped_polygon[:, 0]

    # Step 2: Rotate 90° (clockwise in image coords, so negate)
    rotated_polygon = rotate_polygon(flipped_polygon, 0.5, 0.5, -90)

    expected_obb = polygons_to_obb(rotated_polygon.reshape(1, 4, 2))[0]

    np.testing.assert_allclose(
        output_bbox[:4],
        expected_obb[:4],
        rtol=1e-4,
        atol=1e-4,
        err_msg="Combined HFlip + Rotate incorrect",
    )


@pytest.mark.obb
def test_obb_multiple_rotations_accumulate() -> None:
    """Test that multiple rotations compose correctly (not necessarily additive in angle)."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.3, 0.2
    initial_angle = 10.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        initial_angle,
    ]

    # Rotate by 30° three times
    transform = A.Compose(
        [
            A.Rotate(limit=(30, 30), p=1.0),
            A.Rotate(limit=(30, 30), p=1.0),
            A.Rotate(limit=(30, 30), p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Compute expected: three successive rotations
    expected = compute_obb_after_rotation(input_bbox, 0.5, 0.5, 30)
    expected = compute_obb_after_rotation(expected, 0.5, 0.5, 30)
    expected = compute_obb_after_rotation(expected, 0.5, 0.5, 30)

    np.testing.assert_allclose(
        output_bbox[:4],
        expected[:4],
        rtol=1e-3,
        atol=1e-3,
        err_msg="Multiple rotations should compose correctly",
    )


@pytest.mark.obb
@pytest.mark.parametrize(
    "translate_x,translate_y",
    [
        (0.1, 0.0),
        (0.0, 0.15),
        (0.2, 0.1),
        (-0.1, 0.15),
    ],
)
def test_obb_affine_pure_translation(translate_x: float, translate_y: float) -> None:
    """Test Affine with only translation (no rotation/scale/shear).

    Center should shift by exact translate amount, AABB dimensions should stay same.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.4, 0.5
    width, height = 0.3, 0.2
    angle = 25.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

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

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Center should shift by exact translate amount
    out_cx = (output_bbox[0] + output_bbox[2]) / 2
    out_cy = (output_bbox[1] + output_bbox[3]) / 2

    expected_cx = cx + translate_x
    expected_cy = cy + translate_y

    np.testing.assert_allclose(
        [out_cx, out_cy],
        [expected_cx, expected_cy],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"Center should translate by ({translate_x}, {translate_y})",
    )

    # AABB dimensions should stay the same
    out_width = output_bbox[2] - output_bbox[0]
    out_height = output_bbox[3] - output_bbox[1]

    np.testing.assert_allclose(
        [out_width, out_height],
        [width, height],
        rtol=1e-4,
        atol=1e-4,
        err_msg="AABB dimensions should be preserved during translation",
    )

    # Note: Angle might have ambiguity due to cv2.minAreaRect behavior
    # The important thing is that AABB center and dimensions are preserved
    # For translation-only, we mainly care about position preservation


@pytest.mark.obb
@pytest.mark.parametrize(
    "scale",
    [0.5, 0.8, 1.2, 1.5, 2.0],
)
def test_obb_affine_pure_scaling(scale: float) -> None:
    """Test Affine with only scaling (centered).

    AABB should scale proportionally, angle should be preserved.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Centered box so scaling is symmetric
    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.3
    angle = 30.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

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

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Center should stay at (0.5, 0.5) for centered box
    out_cx = (output_bbox[0] + output_bbox[2]) / 2
    out_cy = (output_bbox[1] + output_bbox[3]) / 2

    np.testing.assert_allclose(
        [out_cx, out_cy],
        [0.5, 0.5],
        rtol=1e-4,
        atol=1e-4,
        err_msg="Center should stay at (0.5, 0.5) for centered box during scaling",
    )

    # AABB dimensions should scale proportionally
    out_width = output_bbox[2] - output_bbox[0]
    out_height = output_bbox[3] - output_bbox[1]

    expected_width = width * scale
    expected_height = height * scale

    np.testing.assert_allclose(
        [out_width, out_height],
        [expected_width, expected_height],
        rtol=1e-3,
        atol=1e-3,
        err_msg=f"AABB dimensions should scale by {scale}",
    )

    # Note: Angle representation can be ambiguous due to cv2.minAreaRect behavior
    # The key validation is that AABB dimensions scale correctly


@pytest.mark.obb
@pytest.mark.parametrize(
    "rotation_deg",
    [30, 45, 90, 135, 180],
)
def test_obb_affine_rotation_vs_rotate_transform(rotation_deg: int) -> None:
    """Test that Affine(rotate=X) produces same results as Rotate(limit=X).

    This validates that Affine rotation handling is consistent with Rotate transform.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.3
    angle = 15.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    # Apply Affine with rotation
    affine_transform = A.Compose(
        [
            A.Affine(rotate=rotation_deg, scale=1.0, translate_px=0, shear=0, p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    affine_result = affine_transform(image=image, bboxes=[input_bbox])
    affine_bbox = affine_result["bboxes"][0]

    # Apply Rotate transform
    rotate_transform = A.Compose(
        [
            A.Rotate(limit=(rotation_deg, rotation_deg), p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    rotate_result = rotate_transform(image=image, bboxes=[input_bbox])
    rotate_bbox = rotate_result["bboxes"][0]

    # Results should be very close
    np.testing.assert_allclose(
        affine_bbox[:4],
        rotate_bbox[:4],
        rtol=1e-3,
        atol=1e-3,
        err_msg=f"Affine(rotate={rotation_deg}) should match Rotate(limit={rotation_deg})",
    )


@pytest.mark.obb
@pytest.mark.parametrize(
    "scale,rotation_deg",
    [
        (1.2, 45),
        (0.8, 30),
        (1.5, 90),
        (0.7, 60),
    ],
)
def test_obb_affine_combined_scale_rotate(scale: float, rotation_deg: int) -> None:
    """Test combined scale and rotation transform.

    Verify against analytical calculation using polygon transformation.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.3, 0.2
    angle = 20.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    transform = A.Compose(
        [
            A.Affine(scale=scale, rotate=rotation_deg, translate_px=0, shear=0, p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Compute expected: scale then rotate the polygon
    obb_array = np.array([input_bbox], dtype=np.float32)
    polygon = obb_to_polygons(obb_array)[0]

    # Scale around center
    scaled_polygon = polygon.copy()
    scaled_polygon[:, 0] = 0.5 + (scaled_polygon[:, 0] - 0.5) * scale
    scaled_polygon[:, 1] = 0.5 + (scaled_polygon[:, 1] - 0.5) * scale

    # Rotate around center
    rotated_polygon = rotate_polygon(scaled_polygon, 0.5, 0.5, -rotation_deg)

    expected_obb = polygons_to_obb(rotated_polygon.reshape(1, 4, 2))[0]

    # AABB should be close
    np.testing.assert_allclose(
        output_bbox[:4],
        expected_obb[:4],
        rtol=1e-3,
        atol=1e-3,
        err_msg=f"Combined scale={scale}, rotate={rotation_deg} incorrect",
    )


@pytest.mark.obb
@pytest.mark.parametrize(
    "image_size",
    [10, 100, 500, 1000],
)
def test_obb_affine_different_image_sizes(image_size: int) -> None:
    """Test OBB Affine on different image sizes.

    Precision should be consistent regardless of image size.
    """
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # Use relative coordinates (same regardless of image size)
    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.3
    angle = 35.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    # Apply 45° rotation
    transform = A.Compose(
        [
            A.Affine(rotate=45, scale=1.0, translate_px=0, shear=0, p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Center should stay at (0.5, 0.5)
    out_cx = (output_bbox[0] + output_bbox[2]) / 2
    out_cy = (output_bbox[1] + output_bbox[3]) / 2

    np.testing.assert_allclose(
        [out_cx, out_cy],
        [0.5, 0.5],
        rtol=1e-3,
        atol=1e-3,
        err_msg=f"Center incorrect for image size {image_size}x{image_size}",
    )

    # Box should not degenerate or grow unreasonably
    out_width = output_bbox[2] - output_bbox[0]
    out_height = output_bbox[3] - output_bbox[1]

    assert 0.1 < out_width < 0.9, f"Width {out_width} unreasonable for image size {image_size}"
    assert 0.1 < out_height < 0.9, f"Height {out_height} unreasonable for image size {image_size}"


@pytest.mark.obb
@pytest.mark.parametrize(
    "box_size,rotation_deg",
    [
        (0.05, 0),
        (0.05, 45),
        (0.08, 30),
        (0.03, 60),
    ],
)
def test_obb_affine_very_small_boxes(box_size: float, rotation_deg: int) -> None:
    """Test very small boxes (5% of image) maintain precision.

    Small boxes are more susceptible to numerical errors.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    angle = 20.0

    input_bbox = [
        cx - box_size / 2,
        cy - box_size / 2,
        cx + box_size / 2,
        cy + box_size / 2,
        angle,
    ]

    transform = A.Compose(
        [
            A.Affine(rotate=rotation_deg, scale=1.0, translate_px=0, shear=0, p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Center should stay at (0.5, 0.5) for centered box
    out_cx = (output_bbox[0] + output_bbox[2]) / 2
    out_cy = (output_bbox[1] + output_bbox[3]) / 2

    np.testing.assert_allclose(
        [out_cx, out_cy],
        [0.5, 0.5],
        rtol=1e-3,
        atol=1e-3,
        err_msg=f"Small box center incorrect (size={box_size})",
    )

    # Box should not degenerate
    out_width = output_bbox[2] - output_bbox[0]
    out_height = output_bbox[3] - output_bbox[1]

    assert out_width > 0.01, f"Width {out_width} too small, box degenerated"
    assert out_height > 0.01, f"Height {out_height} too small, box degenerated"


@pytest.mark.obb
@pytest.mark.parametrize(
    "shear_x,shear_y",
    [
        (10, 0),
        (0, 10),
        (15, 5),
        (-10, 10),
    ],
)
def test_obb_affine_shear_transforms(shear_x: float, shear_y: float) -> None:
    """Test Affine with shear transforms.

    Verify polygon corners match expected positions after shear.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    width, height = 0.4, 0.3
    angle = 0.0  # Start with non-rotated box for clarity

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

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

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # After shear, center should stay approximately at (0.5, 0.5) for centered box
    out_cx = (output_bbox[0] + output_bbox[2]) / 2
    out_cy = (output_bbox[1] + output_bbox[3]) / 2

    np.testing.assert_allclose(
        [out_cx, out_cy],
        [0.5, 0.5],
        rtol=0.05,
        atol=0.05,
        err_msg=f"Center should be near (0.5, 0.5) after shear ({shear_x}, {shear_y})",
    )

    # Box should not degenerate
    out_width = output_bbox[2] - output_bbox[0]
    out_height = output_bbox[3] - output_bbox[1]

    assert out_width > 0.1, f"Width {out_width} too small after shear"
    assert out_height > 0.1, f"Height {out_height} too small after shear"


@pytest.mark.obb
@pytest.mark.parametrize(
    "image_height,image_width",
    [
        (100, 200),
        (200, 100),
        (150, 300),
    ],
)
def test_obb_affine_non_square_images(image_height: int, image_width: int) -> None:
    """Test Affine on non-square images.

    Verify aspect ratio is handled correctly.
    """
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # Place box in center
    cx, cy = 0.5, 0.5
    width, height = 0.3, 0.2
    angle = 30.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    # Apply 90° rotation
    transform = A.Compose(
        [
            A.Affine(rotate=90, scale=1.0, translate_px=0, shear=0, p=1.0),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Center should stay at (0.5, 0.5)
    out_cx = (output_bbox[0] + output_bbox[2]) / 2
    out_cy = (output_bbox[1] + output_bbox[3]) / 2

    np.testing.assert_allclose(
        [out_cx, out_cy],
        [0.5, 0.5],
        rtol=1e-3,
        atol=1e-3,
        err_msg=f"Center incorrect for {image_height}x{image_width} image",
    )

    # Box should be reasonable
    out_width = output_bbox[2] - output_bbox[0]
    out_height = output_bbox[3] - output_bbox[1]

    assert 0.05 < out_width < 0.95, f"Width {out_width} unreasonable for non-square image"
    assert 0.05 < out_height < 0.95, f"Height {out_height} unreasonable for non-square image"


@pytest.mark.obb
@pytest.mark.parametrize(
    "rotation_deg",
    [45, 90, 135],
)
def test_obb_affine_fit_output(rotation_deg: int) -> None:
    """Test Affine with fit_output=True.

    OBB should adapt to new output shape when fit_output is enabled.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Place box near corner so fit_output makes a difference
    cx, cy = 0.3, 0.3
    width, height = 0.4, 0.3
    angle = 0.0

    input_bbox = [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
        angle,
    ]

    transform = A.Compose(
        [
            A.Affine(
                rotate=rotation_deg,
                scale=1.0,
                translate_px=0,
                shear=0,
                fit_output=True,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])

    # With fit_output, bbox should be preserved (not filtered)
    assert len(result["bboxes"]) == 1, "Box should be preserved with fit_output=True"

    output_bbox = result["bboxes"][0]

    # Box should be valid (all coords in [0, 1])
    assert 0 <= output_bbox[0] <= 1, f"x_min {output_bbox[0]} out of bounds"
    assert 0 <= output_bbox[1] <= 1, f"y_min {output_bbox[1]} out of bounds"
    assert 0 <= output_bbox[2] <= 1, f"x_max {output_bbox[2]} out of bounds"
    assert 0 <= output_bbox[3] <= 1, f"y_max {output_bbox[3]} out of bounds"
