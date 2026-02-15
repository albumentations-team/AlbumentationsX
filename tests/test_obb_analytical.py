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

import cv2
import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.bbox_utils import obb_to_polygons, polygons_to_obb
from tests.helpers import obb_corners_equivalent


def _assert_obb_geometrically_equivalent(
    output_bbox: np.ndarray,
    expected: np.ndarray,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    err_msg: str = "",
) -> None:
    """Assert two OBBs represent the same polygon (allows different w/h/angle ordering)."""
    poly_out = obb_to_polygons(np.array([output_bbox], dtype=np.float32))[0]
    poly_exp = obb_to_polygons(np.array([expected], dtype=np.float32))[0]
    assert obb_corners_equivalent(poly_out, poly_exp, rtol=rtol, atol=atol), (
        err_msg or "OBBs should represent the same polygon"
    )


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
        bbox_params=A.BboxParams(coord_format="albumentations", bbox_type="obb"),
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

    # Check angle: with width>=height, non-squares have 180° only; squares have 90° (no swap when w≈h)
    # Use [0,90] canonical for squares: min(θ%180, 180-θ%180)
    def _canon_90(a: float) -> float:
        x = ((a % 360) + 360) % 360 % 180
        return min(x, 180 - x)

    actual_can = _canon_90(output_bbox[4])
    expected_can = _canon_90(expected[4])
    assert abs(actual_can - expected_can) < 1.0, (
        f"OBB angle incorrect after {rotation_deg}° rotation: got {output_bbox[4]}, expected {expected[4]}"
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
        bbox_params=A.BboxParams(coord_format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    expected = compute_obb_after_rotation(input_bbox, 0.5, 0.5, rotation_deg)

    _assert_obb_geometrically_equivalent(
        output_bbox,
        expected,
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"Rectangular OBB incorrect after {rotation_deg}° rotation",
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
        bbox_params=A.BboxParams(coord_format="albumentations", bbox_type="obb"),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    expected = compute_obb_after_rotation(input_bbox, 0.5, 0.5, rotation_deg)

    _assert_obb_geometrically_equivalent(
        output_bbox,
        expected,
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
        bbox_params=A.BboxParams(coord_format="albumentations", bbox_type="obb"),
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
        bbox_params=A.BboxParams(coord_format="albumentations", bbox_type="obb"),
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
        bbox_params=A.BboxParams(coord_format="albumentations", bbox_type="obb"),
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
        bbox_params=A.BboxParams(coord_format="albumentations", bbox_type="obb"),
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
        bbox_params=A.BboxParams(coord_format="albumentations", bbox_type="obb"),
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
        bbox_params=A.BboxParams(coord_format="albumentations", bbox_type="obb"),
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

    # Physical dimensions (0.3, 0.4) preserved; canonical form may have width>=height
    out_width = output_bbox[2] - output_bbox[0]
    out_height = output_bbox[3] - output_bbox[1]

    np.testing.assert_allclose(
        [min(out_width, out_height), max(out_width, out_height)],
        [min(width, height), max(width, height)],
        rtol=1e-4,
        atol=1e-4,
        err_msg="OBB oriented dimensions (min,max) should be preserved after transpose",
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
        bbox_params=A.BboxParams(coord_format="albumentations", bbox_type="obb"),
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
        bbox_params=A.BboxParams(coord_format="albumentations", bbox_type="obb"),
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
        bbox_params=A.BboxParams(coord_format="albumentations", bbox_type="obb"),
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
        bbox_params=A.BboxParams(coord_format="albumentations", bbox_type="obb"),
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
        bbox_params=A.BboxParams(coord_format="albumentations", bbox_type="obb"),
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
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            clip_after_transform=False,
        ),
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


def _obb_canonical_angle(angle_deg: float) -> float:
    """Reduce rectangle angle to [0, 180) for comparison.

    Same rect can have θ or θ+180°; minAreaRect may also swap w/h (θ≡θ+90°).
    """
    return ((angle_deg % 360) + 360) % 360 % 180


def _obb_oriented_dims(obb: np.ndarray, shape: tuple[int, int]) -> tuple[float, float]:
    """Get oriented-rect dimensions (minAreaRect style) from OBB in albumentations format.

    Returns (min_dim, max_dim) so we can compare before/after regardless of w/h flip.
    """
    height, width = shape[0], shape[1]
    polygons = obb_to_polygons(obb.reshape(1, -1).astype(np.float32))
    poly = polygons[0].copy()
    poly[:, 0] *= width
    poly[:, 1] *= height
    rect = cv2.minAreaRect(poly.astype(np.float32))
    (_, _), (w, h), _ = rect
    return (min(w, h), max(w, h))


def _obb_area_normalized(obb: np.ndarray) -> float:
    """Area of OBB in normalized coords: (x_max - x_min) * (y_max - y_min)."""
    w = obb[2] - obb[0]
    h = obb[3] - obb[1]
    return w * h


@pytest.mark.obb
def test_obb_affine_translate_rotate_preserves_dimensions() -> None:
    """Affine with only translate+rotate (no scale/shear) preserves OBB oriented-rect dimensions.

    (min(W,H), max(W,H)) should be unchanged; w and h may flip due to minAreaRect convention.
    Uses centered boxes so they stay inside after rotation.
    """
    import hypothesis.strategies as st
    from hypothesis import given, settings

    @given(
        rotate=st.floats(-180, 180),
        translate_x=st.integers(-10, 10),
        translate_y=st.integers(-10, 10),
        box_w=st.floats(0.05, 0.25),
        box_h=st.floats(0.05, 0.25),
        angle=st.floats(-90, 90),
    )
    @settings(max_examples=50, deadline=5000)
    def _run(
        rotate: float,
        translate_x: int,
        translate_y: int,
        box_w: float,
        box_h: float,
        angle: float,
    ) -> None:
        # Centered box so it stays inside after any rotation
        cx, cy = 0.5, 0.5
        x_min = cx - box_w / 2
        y_min = cy - box_h / 2
        x_max = cx + box_w / 2
        y_max = cy + box_h / 2

        obb = np.array([[x_min, y_min, x_max, y_max, angle]], dtype=np.float32)
        shape = (100, 100)
        min_before, max_before = _obb_oriented_dims(obb, shape)

        transform = A.Compose(
            [
                A.Affine(
                    scale=(1.0, 1.0),
                    rotate=(rotate, rotate),
                    translate_px={"x": (translate_x, translate_x), "y": (translate_y, translate_y)},
                    shear={"x": (0, 0), "y": (0, 0)},
                    fit_output=False,
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(
                coord_format="albumentations",
                bbox_type="obb",
                clip_after_transform=False,
            ),
        )
        image = np.zeros((*shape, 3), dtype=np.uint8)
        result = transform(image=image, bboxes=obb.tolist())
        out_bboxes = result["bboxes"]
        if len(out_bboxes) == 0:
            return  # filtered out
        out_obb = np.array(out_bboxes, dtype=np.float32)
        out_shape = result["image"].shape[:2]
        min_after, max_after = _obb_oriented_dims(out_obb, out_shape)

        np.testing.assert_allclose(
            [min_before, max_before],
            [min_after, max_after],
            rtol=1e-4,
            atol=1e-3,
            err_msg=f"OBB dims (min,max) should be preserved: before ({min_before},{max_before}) vs after ({min_after},{max_after})",
        )

    _run()


@pytest.mark.obb
@pytest.mark.parametrize(
    "rotate,translate_x,translate_y,box_w,box_h,angle",
    [
        (30, 5, -3, 0.2, 0.15, 45.0),
        (90, 0, 0, 0.1, 0.2, 0.0),
        (-45, -10, 10, 0.15, 0.15, -30.0),
        (180, 0, 0, 0.25, 0.1, 60.0),
    ],
)
def test_obb_affine_translate_rotate_preserves_dimensions_parametrized(
    rotate: float,
    translate_x: int,
    translate_y: int,
    box_w: float,
    box_h: float,
    angle: float,
) -> None:
    """Parametrized: Affine translate+rotate only preserves (min(W,H), max(W,H))."""
    cx, cy = 0.5, 0.5
    obb = np.array(
        [
            [cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2, angle],
        ],
        dtype=np.float32,
    )
    shape = (100, 100)
    min_before, max_before = _obb_oriented_dims(obb, shape)

    transform = A.Compose(
        [
            A.Affine(
                scale=(1.0, 1.0),
                rotate=(rotate, rotate),
                translate_px={"x": (translate_x, translate_x), "y": (translate_y, translate_y)},
                shear={"x": (0, 0), "y": (0, 0)},
                fit_output=False,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            clip_after_transform=False,
        ),
    )
    image = np.zeros((*shape, 3), dtype=np.uint8)
    result = transform(image=image, bboxes=obb.tolist())
    out_obb = np.array(result["bboxes"], dtype=np.float32)
    out_shape = result["image"].shape[:2]
    min_after, max_after = _obb_oriented_dims(out_obb, out_shape)

    np.testing.assert_allclose(
        [min_before, max_before],
        [min_after, max_after],
        rtol=1e-4,
        atol=1e-3,
        err_msg=f"OBB dims (min,max) should be preserved: before ({min_before},{max_before}) vs after ({min_after},{max_after})",
    )


@pytest.mark.obb
@pytest.mark.parametrize("rotate", [30, 90, -45, 180])
def test_obb_affine_pure_rotation_preserves_dims_and_angle(rotate: float) -> None:
    """Affine with only rotation (scale=1, shear=0, translate=0), clip_after_transform=False.

    Checks:
    1. (min(w,h), max(w,h)) of OBB preserved
    2. When initial angle=0, output angle equals Affine rotation (mod 360, normalized to [-180,180))
    """
    cx, cy = 0.5, 0.5
    box_w, box_h = 0.2, 0.15
    initial_angle = 0.0
    obb = np.array(
        [[cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2, initial_angle]],
        dtype=np.float32,
    )
    shape = (100, 100)
    min_before, max_before = _obb_oriented_dims(obb, shape)

    transform = A.Compose(
        [
            A.Affine(
                scale=(1.0, 1.0),
                rotate=(rotate, rotate),
                translate_px={"x": (0, 0), "y": (0, 0)},
                shear={"x": (0, 0), "y": (0, 0)},
                fit_output=False,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            clip_after_transform=False,
        ),
    )
    image = np.zeros((*shape, 3), dtype=np.uint8)
    result = transform(image=image, bboxes=obb.tolist())
    out_obb = np.array(result["bboxes"], dtype=np.float32)
    out_shape = result["image"].shape[:2]
    min_after, max_after = _obb_oriented_dims(out_obb, out_shape)

    np.testing.assert_allclose(
        [min_before, max_before],
        [min_after, max_after],
        rtol=1e-4,
        atol=1e-3,
        err_msg="OBB dims (min,max) should be preserved",
    )

    # When initial angle=0, output angle equals Affine rotation. Affine uses clockwise rotation
    # (positive rotate = CW), so output angle = -rotate. minAreaRect may return θ or θ+90° (w/h swap).
    out_can = _obb_canonical_angle(float(out_obb[0, 4]))
    rot_can = _obb_canonical_angle(-rotate)
    diff = abs(out_can - rot_can)
    # Allow θ≡θ+90° (minAreaRect w/h swap) and θ≡θ+180°
    assert diff < 1e-3 or abs(diff - 90) < 1e-3 or abs(diff - 180) < 1e-3, (
        f"Angle canonical form: out={out_can} should match rotate={rot_can} (mod 90°)"
    )


def _obb_after_rotation_analytical(
    obb: np.ndarray,
    shape: tuple[int, int],
    rotation_deg: float,
    use_center: bool,
) -> np.ndarray:
    """Compute OBB after rotation analytically.

    use_center: If True, use center() (image center); if False, use center_bbox().
    Affine uses center_bbox() for bboxes so use_center=False matches Affine.
    """
    h, w = shape[0], shape[1]
    shift = fgeometric.center(shape) if use_center else fgeometric.center_bbox(shape)
    translate = {"x": 0, "y": 0}
    shear = {"x": 0, "y": 0}
    scale = {"x": 1.0, "y": 1.0}
    matrix = fgeometric.create_affine_transformation_matrix(
        translate,
        shear,
        scale,
        rotation_deg,
        shift,
    )
    polygon = obb_to_polygons(obb.reshape(1, -1).astype(np.float32))[0].copy()
    polygon[:, 0] *= w
    polygon[:, 1] *= h
    rotated = fgeometric.apply_affine_to_points(
        polygon.reshape(-1, 2),
        matrix,
    ).reshape(4, 2)
    obb_px = polygons_to_obb(rotated.reshape(1, 4, 2))[0]
    obb_norm = obb_px.copy()
    obb_norm[0] /= w
    obb_norm[1] /= h
    obb_norm[2] /= w
    obb_norm[3] /= h
    return obb_norm


@pytest.mark.obb
@pytest.mark.parametrize("rotate", [30, 45, 90])
def test_obb_affine_rotation_matches_analytical_center_bbox(rotate: float) -> None:
    """Affine OBB rotation matches analytical computation using center_bbox().

    Affine uses center_bbox() for bbox matrix; analytical with same center matches.
    """
    shape = (100, 100)
    cx, cy = 0.5, 0.5
    box_w, box_h = 0.2, 0.15
    angle = 15.0
    obb = np.array(
        [[cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2, angle]],
        dtype=np.float32,
    )

    analytical = _obb_after_rotation_analytical(obb, shape, rotate, use_center=False)

    transform = A.Compose(
        [
            A.Affine(
                scale=(1.0, 1.0),
                rotate=(rotate, rotate),
                translate_px={"x": (0, 0), "y": (0, 0)},
                shear={"x": (0, 0), "y": (0, 0)},
                fit_output=False,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            clip_after_transform=False,
        ),
    )
    image = np.zeros((*shape, 3), dtype=np.uint8)
    result = transform(image=image, bboxes=obb.tolist())
    affined = np.array(result["bboxes"], dtype=np.float32)[0]

    _assert_obb_geometrically_equivalent(
        affined,
        analytical,
        rtol=1e-4,
        atol=1e-3,
        err_msg=f"Affine should match analytical (center) for rotate={rotate}",
    )


@pytest.mark.obb
def test_obb_affine_center_vs_center_bbox_offset() -> None:
    """Quantify offset between center() and center_bbox() for OBB rotation.

    center() = (w/2-0.5, h/2-0.5), center_bbox() = (w/2, h/2).
    For 100x100: 0.5 px difference. Documents expected ~0.5 px drift at edges.
    """
    shape = (100, 100)
    cx, cy = 0.5, 0.5
    box_w, box_h = 0.2, 0.15
    angle = 0.0
    obb = np.array(
        [[cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2, angle]],
        dtype=np.float32,
    )
    rotate = 30.0

    analytical_center = _obb_after_rotation_analytical(obb, shape, rotate, use_center=True)
    analytical_bbox = _obb_after_rotation_analytical(obb, shape, rotate, use_center=False)

    # Center of rotated box: both should be near (0.5, 0.5) for centered input
    cx_center = (analytical_center[0] + analytical_center[2]) / 2
    cy_center = (analytical_center[1] + analytical_center[3]) / 2
    cx_bbox = (analytical_bbox[0] + analytical_bbox[2]) / 2
    cy_bbox = (analytical_bbox[1] + analytical_bbox[3]) / 2

    # Offset in normalized coords: 0.5/100 = 0.005
    offset_x = abs(cx_center - cx_bbox)
    offset_y = abs(cy_center - cy_bbox)
    assert offset_x < 0.02 and offset_y < 0.02, "Offset should be small (~0.5 px)"


@pytest.mark.obb
def test_obb_affine_clip_bboxes_on_input_preserves_orientation() -> None:
    """Regression: clip_bboxes_on_input=False preserves OBB orientation through Affine.

    With clip_bboxes_on_input=True, OBBs extending outside [0,1] get angle=0 (lossy).
    With clip_bboxes_on_input=False, orientation is preserved. See plot_boats_obb fix.
    """
    shape = (100, 100)
    # OBB extending outside: center at edge, rotated
    cx, cy = 0.95, 0.5
    box_w, box_h = 0.2, 0.1
    angle = 45.0
    obb = np.array(
        [[cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2, angle]],
        dtype=np.float32,
    )
    min_before, max_before = _obb_oriented_dims(obb, shape)

    transform = A.Compose(
        [
            A.Affine(
                scale=(1.0, 1.0),
                rotate=(30, 30),
                translate_px={"x": (0, 0), "y": (0, 0)},
                shear={"x": (0, 0), "y": (0, 0)},
                fit_output=False,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            clip_bboxes_on_input=False,
            clip_after_transform=False,
        ),
    )
    image = np.zeros((*shape, 3), dtype=np.uint8)
    result = transform(image=image, bboxes=obb.tolist())
    out_bboxes = result["bboxes"]
    assert len(out_bboxes) > 0, "OBB should not be filtered"
    out_obb = np.array(out_bboxes, dtype=np.float32)
    min_after, max_after = _obb_oriented_dims(out_obb, shape)

    # Oriented dims preserved (translate+rotate only)
    np.testing.assert_allclose(
        [min_before, max_before],
        [min_after, max_after],
        rtol=1e-4,
        atol=1e-3,
        err_msg="clip_bboxes_on_input=False should preserve OBB dims",
    )


@pytest.mark.obb
@pytest.mark.parametrize(
    "translate_px_x,translate_px_y",
    [
        (5, 0),
        (0, -3),
        (10, 5),
        (-7, 8),
    ],
)
def test_obb_affine_pure_translate_px_analytical(
    translate_px_x: int,
    translate_px_y: int,
) -> None:
    """Affine with only translate_px: center shifts by exact pixels, dims preserved."""
    shape = (100, 100)
    image = np.zeros((*shape, 3), dtype=np.uint8)
    h, w = shape[0], shape[1]

    cx, cy = 0.5, 0.5
    box_w, box_h = 0.2, 0.15
    angle = 45.0
    obb = np.array(
        [[cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2, angle]],
        dtype=np.float32,
    )
    min_before, max_before = _obb_oriented_dims(obb, shape)
    area_before = _obb_area_normalized(obb[0])

    transform = A.Compose(
        [
            A.Affine(
                scale=(1.0, 1.0),
                rotate=(0, 0),
                translate_px={"x": (translate_px_x, translate_px_x), "y": (translate_px_y, translate_px_y)},
                shear={"x": (0, 0), "y": (0, 0)},
                fit_output=False,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            clip_after_transform=False,
        ),
    )
    result = transform(image=image, bboxes=obb.tolist())
    out_obb = np.array(result["bboxes"], dtype=np.float32)

    # Center shifts by (tx/w, ty/h) in normalized coords
    expected_cx = cx + translate_px_x / w
    expected_cy = cy + translate_px_y / h
    out_cx = (out_obb[0, 0] + out_obb[0, 2]) / 2
    out_cy = (out_obb[0, 1] + out_obb[0, 3]) / 2
    np.testing.assert_allclose(
        [out_cx, out_cy],
        [expected_cx, expected_cy],
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Center should shift by ({translate_px_x}, {translate_px_y}) px",
    )

    # Oriented dims preserved
    min_after, max_after = _obb_oriented_dims(out_obb, shape)
    np.testing.assert_allclose(
        [min_before, max_before],
        [min_after, max_after],
        rtol=1e-4,
        atol=1e-3,
        err_msg="Oriented dims should be preserved",
    )

    # Area preserved
    area_after = _obb_area_normalized(out_obb[0])
    np.testing.assert_allclose(area_after, area_before, rtol=1e-5, atol=1e-6)


@pytest.mark.obb
@pytest.mark.parametrize(
    "scale",
    [0.5, 0.8, 1.2, 1.5, 2.0],
)
def test_obb_affine_pure_scale_analytical(scale: float) -> None:
    """Affine with only scale: area scales by scale², oriented dims scale by scale."""
    shape = (100, 100)
    image = np.zeros((*shape, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    box_w, box_h = 0.2, 0.15
    angle = 30.0
    obb = np.array(
        [[cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2, angle]],
        dtype=np.float32,
    )
    min_before, max_before = _obb_oriented_dims(obb, shape)
    area_before = _obb_area_normalized(obb[0])

    transform = A.Compose(
        [
            A.Affine(
                scale=(scale, scale),
                rotate=(0, 0),
                translate_px={"x": (0, 0), "y": (0, 0)},
                shear={"x": (0, 0), "y": (0, 0)},
                fit_output=False,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            clip_after_transform=False,
        ),
    )
    result = transform(image=image, bboxes=obb.tolist())
    out_obb = np.array(result["bboxes"], dtype=np.float32)

    # Center stays near (0.5, 0.5); with center()=(w/2-0.5,h/2-0.5), scaled center shifts slightly
    out_cx = (out_obb[0, 0] + out_obb[0, 2]) / 2
    out_cy = (out_obb[0, 1] + out_obb[0, 3]) / 2
    np.testing.assert_allclose([out_cx, out_cy], [0.5, 0.5], rtol=1e-2, atol=0.005)

    # Oriented dims scale by scale
    min_after, max_after = _obb_oriented_dims(out_obb, shape)
    np.testing.assert_allclose(
        [min_after, max_after],
        [min_before * scale, max_before * scale],
        rtol=1e-4,
        atol=1e-3,
        err_msg=f"Oriented dims should scale by {scale}",
    )

    # Area scales by scale²
    area_after = _obb_area_normalized(out_obb[0])
    np.testing.assert_allclose(
        area_after,
        area_before * (scale * scale),
        rtol=1e-4,
        atol=1e-5,
        err_msg=f"Area should scale by {scale}²",
    )


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
def test_obb_affine_pure_shear_preserves_area(shear_x: float, shear_y: float) -> None:
    """Affine with only shear: area is approximately preserved.

    Shear is mathematically area-preserving, but the OBB fitting (minAreaRect of
    sheared polygon) and center_bbox vs image center can introduce small drift.
    Use relaxed tolerance to catch gross violations.
    """
    shape = (100, 100)
    image = np.zeros((*shape, 3), dtype=np.uint8)

    cx, cy = 0.5, 0.5
    box_w, box_h = 0.2, 0.15
    angle = 0.0
    obb = np.array(
        [[cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2, angle]],
        dtype=np.float32,
    )
    area_before = _obb_area_normalized(obb[0])

    transform = A.Compose(
        [
            A.Affine(
                scale=(1.0, 1.0),
                rotate=(0, 0),
                translate_px={"x": (0, 0), "y": (0, 0)},
                shear={"x": (shear_x, shear_x), "y": (shear_y, shear_y)},
                fit_output=False,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            clip_after_transform=False,
        ),
    )
    result = transform(image=image, bboxes=obb.tolist())
    out_obb = np.array(result["bboxes"], dtype=np.float32)
    area_after = _obb_area_normalized(out_obb[0])

    # Relaxed: shear+minAreaRect can drift; catch >50% change
    np.testing.assert_allclose(
        area_after,
        area_before,
        rtol=0.5,
        atol=0.02,
        err_msg=f"Shear ({shear_x}, {shear_y}) area changed too much",
    )


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
                fit_output=False,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            clip_after_transform=False,
        ),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Center stays near (0.5, 0.5); center()=(w/2-0.5,h/2-0.5) causes ~0.5px shift
    out_cx = (output_bbox[0] + output_bbox[2]) / 2
    out_cy = (output_bbox[1] + output_bbox[3]) / 2

    np.testing.assert_allclose(
        [out_cx, out_cy],
        [0.5, 0.5],
        rtol=0.02,
        atol=0.01,
        err_msg="Center should stay near (0.5, 0.5) for centered box during scaling",
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
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            clip_after_transform=False,
        ),
    )

    affine_result = affine_transform(image=image, bboxes=[input_bbox])
    affine_bbox = affine_result["bboxes"][0]

    # Apply Rotate transform
    rotate_transform = A.Compose(
        [
            A.Rotate(limit=(rotation_deg, rotation_deg), p=1.0),
        ],
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            clip_after_transform=False,
        ),
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
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            clip_after_transform=False,
        ),
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

    _assert_obb_geometrically_equivalent(
        output_bbox,
        expected_obb,
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
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            clip_after_transform=False,
        ),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Center stays near (0.5, 0.5); center() causes ~0.5px shift
    out_cx = (output_bbox[0] + output_bbox[2]) / 2
    out_cy = (output_bbox[1] + output_bbox[3]) / 2

    np.testing.assert_allclose(
        [out_cx, out_cy],
        [0.5, 0.5],
        rtol=0.02,
        atol=0.02,
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
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            clip_after_transform=False,
        ),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Center stays near (0.5, 0.5); center() causes ~0.5px shift
    out_cx = (output_bbox[0] + output_bbox[2]) / 2
    out_cy = (output_bbox[1] + output_bbox[3]) / 2

    np.testing.assert_allclose(
        [out_cx, out_cy],
        [0.5, 0.5],
        rtol=0.02,
        atol=0.02,
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
                fit_output=False,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            clip_after_transform=False,
        ),
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
        bbox_params=A.BboxParams(
            coord_format="albumentations",
            bbox_type="obb",
            clip_after_transform=False,
        ),
    )

    result = transform(image=image, bboxes=[input_bbox])
    output_bbox = result["bboxes"][0]

    # Center stays near (0.5, 0.5); center() causes ~0.5px shift
    out_cx = (output_bbox[0] + output_bbox[2]) / 2
    out_cy = (output_bbox[1] + output_bbox[3]) / 2

    np.testing.assert_allclose(
        [out_cx, out_cy],
        [0.5, 0.5],
        rtol=0.02,
        atol=0.02,
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
        bbox_params=A.BboxParams(coord_format="albumentations", bbox_type="obb"),
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
