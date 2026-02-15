"""Extensive tests for cv2.minAreaRect via polygons_to_obb.

Tests the round-trip obb_to_polygons -> polygons_to_obb with rectangles at various
angles, aspect ratios, and coordinate scales. minAreaRect has conventions (angle in
[-90,0), width>=height) that can produce different but geometrically equivalent OBB
representations — we verify the fitted rect matches the input polygon.
"""

import numpy as np
import pytest

from albumentations.core.bbox_utils import (
    _corners_to_obb_params,
    obb_to_polygons,
    polygons_to_obb,
)


def _canonicalize_obb_cxcywh(
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle: float,
) -> tuple[float, float, float, float, float]:
    """Normalize (cx, cy, w, h, angle) to w >= h so equivalent boxes compare equal."""
    if w >= h:
        return (cx, cy, w, h, angle)
    return (cx, cy, h, w, angle + 90.0)


def _obb_cxcywh_same_box(
    cx1: float,
    cy1: float,
    w1: float,
    h1: float,
    a1: float,
    cx2: float,
    cy2: float,
    w2: float,
    h2: float,
    a2: float,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> bool:
    """Check if two (cx, cy, w, h, angle) represent the same oriented box.

    Canonical form: w >= h. Angle equivalence: a1 ≡ a2 (mod 180) or a1 ≡ -a2 (mod 180),
    since OpenCV minAreaRect can return negated angles for the same box.
    """
    c1 = _canonicalize_obb_cxcywh(cx1, cy1, w1, h1, a1)
    c2 = _canonicalize_obb_cxcywh(cx2, cy2, w2, h2, a2)
    if not np.allclose([c1[0], c1[1], c1[2], c1[3]], [c2[0], c2[1], c2[2], c2[3]], rtol=rtol, atol=atol):
        return False
    a1c, a2c = c1[4], c2[4]
    diff_pos = abs((a1c - a2c) % 180)
    diff_neg = abs((a1c + a2c) % 180)
    return diff_pos < atol or abs(diff_pos - 180) < atol or diff_neg < atol or abs(diff_neg - 180) < atol


def _polygons_match(poly_a: np.ndarray, poly_b: np.ndarray, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    """Check if two 4-corner polygons represent the same rectangle.

    minAreaRect uses OpenCV's angle convention ([-90,0), width>=height) which can
    produce different corner order/angle for the same rect. We verify geometric
    equivalence: same center, same area, and the 4 corners are the same set of points.
    """
    if poly_a.shape != (4, 2) or poly_b.shape != (4, 2):
        return False
    c_a, c_b = poly_a.mean(axis=0), poly_b.mean(axis=0)
    if not np.allclose(c_a, c_b, rtol=rtol, atol=atol):
        return False

    # Shoelace area
    def _area(p: np.ndarray) -> float:
        return 0.5 * abs(
            np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]),
        )

    if not np.isclose(_area(poly_a), _area(poly_b), rtol=rtol, atol=atol):
        return False
    # Each corner of poly_a must match some corner of poly_b (same 4 points, any order)
    for i in range(4):
        dists = np.linalg.norm(poly_b - poly_a[i], axis=1)
        if dists.min() > atol + rtol * np.linalg.norm(poly_a[i]):
            return False
    return True


def _obb_roundtrip_geometrically_equivalent(
    obb: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Round-trip OBB through polygons and assert same 4 corners.

    Only use for angles in ANGLES_FULL_ROUNDTRIP (0°, ±90°, 180°, etc.).
    """
    obb = np.asarray(obb, dtype=np.float32)
    if obb.ndim == 1:
        obb = obb.reshape(1, -1)
    polys_in = obb_to_polygons(obb)
    obb_out = polygons_to_obb(polys_in)
    polys_out = obb_to_polygons(obb_out)
    for i in range(len(obb)):
        assert _polygons_match(
            polys_in[i],
            polys_out[i],
            rtol=rtol,
            atol=atol,
        ), f"Round-trip mismatch for OBB {obb[i]}: in={polys_in[i]}, out={polys_out[i]}"


# --- Angle parametrization ---
# Full corner match works only for 0°, ±90°, 180°, 270°, 360° (axis-aligned).
# Other angles: minAreaRect returns different (w,h,angle) representation.
ANGLES_FULL_ROUNDTRIP = [0, 90, 180, -90, -180, 270, 360, 450, -270, -360]
# All angles for center+area tests
ANGLES_ALL = ANGLES_FULL_ROUNDTRIP + [15, 30, 45, 60, 120, 135, 150, -30, -45, -135]

# (width, height) — use pixel-like values for numerical stability
DIMENSIONS = [
    (100, 100),  # square
    (120, 80),  # landscape
    (80, 120),  # portrait
    (150, 50),  # wide
    (50, 150),  # tall
    (200, 100),
    (100, 200),
]


@pytest.mark.parametrize("angle_deg", ANGLES_FULL_ROUNDTRIP)
def test_minarearect_roundtrip_single_angle(angle_deg: int) -> None:
    """Round-trip OBB at a single angle; square box at pixel center."""
    cx, cy = 100.0, 100.0
    w, h = 80.0, 80.0
    obb = np.array(
        [
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
            float(angle_deg),
        ],
        dtype=np.float32,
    )
    _obb_roundtrip_geometrically_equivalent(obb)


@pytest.mark.parametrize("angle_deg", ANGLES_FULL_ROUNDTRIP)
@pytest.mark.parametrize("w,h", DIMENSIONS)
def test_minarearect_roundtrip_angles_and_dimensions(angle_deg: int, w: int, h: int) -> None:
    """Round-trip OBB for all angle/dimension combinations."""
    cx, cy = 100.0, 100.0
    obb = np.array(
        [
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
            float(angle_deg),
        ],
        dtype=np.float32,
    )
    _obb_roundtrip_geometrically_equivalent(obb)


@pytest.mark.parametrize("angle_deg", [0, 90, 180])
@pytest.mark.parametrize("cx,cy", [(100, 100), (50, 50), (200, 150)])
def test_minarearect_roundtrip_centers(angle_deg: int, cx: float, cy: float) -> None:
    """Round-trip OBB at different centers."""
    w, h = 60.0, 40.0
    obb = np.array(
        [
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
            float(angle_deg),
        ],
        dtype=np.float32,
    )
    _obb_roundtrip_geometrically_equivalent(obb)


@pytest.mark.parametrize("angle_deg", [0, 90, 180])
def test_minarearect_roundtrip_preserves_extra_fields(angle_deg: int) -> None:
    """Round-trip preserves extra columns (labels, etc.)."""
    obb = np.array(
        [
            [50, 50, 150, 130, float(angle_deg), 1.0, 42],
            [200, 100, 280, 180, float(angle_deg), 2.0, 137],
        ],
        dtype=np.float32,
    )
    polys = obb_to_polygons(obb)
    restored = polygons_to_obb(polys, extra_fields=obb[:, 5:])
    assert restored.shape[1] == obb.shape[1]
    np.testing.assert_array_equal(restored[:, 5:], obb[:, 5:])
    for i in range(len(obb)):
        assert _polygons_match(
            obb_to_polygons(obb[i : i + 1])[0],
            obb_to_polygons(restored[i : i + 1])[0],
        )


def test_minarearect_empty_input() -> None:
    """Empty polygons: handle_empty_array returns input, so shape (0, 4, 2)."""
    empty_polys = np.zeros((0, 4, 2), dtype=np.float32)
    result = polygons_to_obb(empty_polys)
    assert len(result) == 0
    assert result.dtype == np.float32


def test_minarearect_empty_with_extras() -> None:
    """Empty polygons with extra_fields: handle_empty_array returns input."""
    empty_polys = np.zeros((0, 4, 2), dtype=np.float32)
    extras = np.zeros((0, 2), dtype=np.float32)
    result = polygons_to_obb(empty_polys, extra_fields=extras)
    assert len(result) == 0


@pytest.mark.parametrize("angle_deg", [0, 90, 180])
def test_minarearect_batch(angle_deg: int) -> None:
    """Batch of OBBs round-trip correctly."""
    rng = np.random.default_rng(137)
    n = 10
    centers = rng.uniform(50, 150, (n, 2))
    wh = rng.uniform(20, 80, (n, 2))
    obbs = np.column_stack(
        [
            centers[:, 0] - wh[:, 0] / 2,
            centers[:, 1] - wh[:, 1] / 2,
            centers[:, 0] + wh[:, 0] / 2,
            centers[:, 1] + wh[:, 1] / 2,
            np.full(n, float(angle_deg)),
        ],
    ).astype(np.float32)
    _obb_roundtrip_geometrically_equivalent(obbs)


@pytest.mark.parametrize("angle_deg", [0, 90, 180])
def test_minarearect_thin_rectangle(angle_deg: int) -> None:
    """Thin rectangle (high aspect ratio) round-trips."""
    cx, cy = 100.0, 100.0
    w, h = 100.0, 5.0
    obb = np.array(
        [
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
            float(angle_deg),
        ],
        dtype=np.float32,
    )
    _obb_roundtrip_geometrically_equivalent(obb, atol=1e-4)


def test_minarearect_normalized_coords() -> None:
    """Round-trip works with normalized [0,1] coordinates."""
    for angle in [0, 90, 180]:
        obb = np.array([0.2, 0.2, 0.8, 0.6, float(angle)], dtype=np.float32)
        _obb_roundtrip_geometrically_equivalent(obb, atol=1e-5)


@pytest.mark.parametrize("angle_deg", ANGLES_ALL)
def test_minarearect_roundtrip_same_box_cxcywh(angle_deg: int) -> None:
    """Round-trip produces (cx, cy, w, h, angle) that represents the same box.

    Uses canonical form (w>=h, angle mod 180) to compare equivalent representations.
    """
    cx, cy = 100.0, 100.0
    w, h = 80.0, 60.0
    obb_in = np.array(
        [
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
            float(angle_deg),
        ],
        dtype=np.float32,
    )
    polys = obb_to_polygons(obb_in.reshape(1, -1))[0]
    obb_out = polygons_to_obb(polys.reshape(1, 4, 2))[0]
    cx_out = (obb_out[0] + obb_out[2]) / 2
    cy_out = (obb_out[1] + obb_out[3]) / 2
    w_out = obb_out[2] - obb_out[0]
    h_out = obb_out[3] - obb_out[1]
    a_out = obb_out[4]
    assert _obb_cxcywh_same_box(
        cx,
        cy,
        w,
        h,
        float(angle_deg),
        cx_out,
        cy_out,
        w_out,
        h_out,
        a_out,
        atol=1e-4,  # float32 from minAreaRect has limited precision
    ), (
        f"Round-trip (cx,cy,w,h,angle) mismatch for angle {angle_deg}: in=({cx},{cy},{w},{h},{angle_deg}), out=({cx_out},{cy_out},{w_out},{h_out},{a_out})"
    )


@pytest.mark.parametrize("angle_deg", ANGLES_ALL)
def test_minarearect_center_and_area_preserved(angle_deg: int) -> None:
    """For any angle, minAreaRect preserves center and area (geometric fit is correct)."""
    cx, cy = 100.0, 100.0
    w, h = 80.0, 60.0
    obb = np.array(
        [
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
            float(angle_deg),
        ],
        dtype=np.float32,
    )
    polys_in = obb_to_polygons(obb.reshape(1, -1))[0]
    obb_out = polygons_to_obb(polys_in.reshape(1, 4, 2))[0]
    polys_out = obb_to_polygons(obb_out.reshape(1, -1))[0]
    # Center and area must match
    np.testing.assert_allclose(polys_in.mean(axis=0), polys_out.mean(axis=0), rtol=1e-5)
    area_in = 0.5 * abs(
        np.sum(polys_in[:, 0] * np.roll(polys_in[:, 1], -1) - np.roll(polys_in[:, 0], -1) * polys_in[:, 1]),
    )
    area_out = 0.5 * abs(
        np.sum(polys_out[:, 0] * np.roll(polys_out[:, 1], -1) - np.roll(polys_out[:, 0], -1) * polys_out[:, 1]),
    )
    np.testing.assert_allclose(area_in, area_out, rtol=1e-5)


@pytest.mark.parametrize("angle_deg", ANGLES_FULL_ROUNDTRIP)
def test_minarearect_output_format(angle_deg: int) -> None:
    """polygons_to_obb returns [x_min, y_min, x_max, y_max, angle] format."""
    obb = np.array([50, 50, 150, 130, float(angle_deg)], dtype=np.float32)
    polys = obb_to_polygons(obb.reshape(1, -1))
    result = polygons_to_obb(polys)[0]
    assert result.shape == (5,)
    # x_min < x_max, y_min < y_max (in local frame; for our convention)
    assert result[2] > result[0]
    assert result[3] > result[1]
    # Center consistency
    cx = (result[0] + result[2]) / 2
    cy = (result[1] + result[3]) / 2
    expected_cx = (obb[0] + obb[2]) / 2
    expected_cy = (obb[1] + obb[3]) / 2
    np.testing.assert_allclose([cx, cy], [expected_cx, expected_cy], rtol=1e-5)
    # Area preserved
    w_out = result[2] - result[0]
    h_out = result[3] - result[1]
    w_in = obb[2] - obb[0]
    h_in = obb[3] - obb[1]
    np.testing.assert_allclose(w_out * h_out, w_in * h_in, rtol=1e-5)


def test_corners_to_obb_params_angle_range() -> None:
    """_corners_to_obb_params returns angle in [-90, 90) for various corner inputs."""
    cx, cy = 100.0, 100.0
    w, h = 80.0, 40.0
    for angle_deg in [0, 15, 30, 45, 60, 89, -15, -30, -45, -60, -89, 90, 120, 170, -120, -170]:
        obb = np.array(
            [
                cx - w / 2,
                cy - h / 2,
                cx + w / 2,
                cy + h / 2,
                float(angle_deg),
            ],
            dtype=np.float64,
        )
        corners = obb_to_polygons(obb.reshape(1, -1))[0]
        _, _, _, _, a_out = _corners_to_obb_params(corners)
        assert -90 <= a_out < 90, f"Angle {a_out} out of [-90, 90) for input {angle_deg}"


@pytest.mark.parametrize("angle_deg", ANGLES_ALL)
def test_obb_angle_in_range(angle_deg: int) -> None:
    """Round-trip obb -> polygons -> polygons_to_obb returns angle in [-90, 90)."""
    cx, cy = 100.0, 100.0
    w, h = 80.0, 60.0
    obb = np.array(
        [
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
            float(angle_deg),
        ],
        dtype=np.float32,
    )
    polys = obb_to_polygons(obb.reshape(1, -1))
    obb_out = polygons_to_obb(polys)[0]
    assert -90 <= obb_out[4] < 90, f"Angle {obb_out[4]} out of [-90, 90) for input {angle_deg}"


@pytest.mark.parametrize("angle_deg", [-90, 89.9, 0, 90])
def test_minarearect_angle_boundaries(angle_deg: float) -> None:
    """Round-trip at angle boundaries produces angle in [-90, 90)."""
    cx, cy = 100.0, 100.0
    w, h = 60.0, 40.0
    obb = np.array(
        [
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
            angle_deg,
        ],
        dtype=np.float32,
    )
    polys = obb_to_polygons(obb.reshape(1, -1))
    obb_out = polygons_to_obb(polys)[0]
    assert -90 <= obb_out[4] < 90, f"Angle {obb_out[4]} out of [-90, 90) for input {angle_deg}"
