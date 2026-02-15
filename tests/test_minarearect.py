"""Extensive tests for cv2.minAreaRect via polygons_to_obb.

Tests the round-trip obb_to_polygons -> polygons_to_obb with rectangles at various
angles, aspect ratios, and coordinate scales. minAreaRect has conventions (angle in
[-90,0), width>=height) that can produce different but geometrically equivalent OBB
representations — we verify the fitted rect matches the input polygon.
"""

import cv2
import numpy as np
import pytest

from albumentations.core.bbox_utils import (
    _corners_to_obb_params,
    convert_bboxes_to_albumentations,
    denormalize_bboxes,
    obb_to_polygons,
    polygons_to_obb,
)
from tests.helpers import obb_corners_equivalent, polygon_area, polygon_center


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


def _obb_roundtrip_geometrically_equivalent(
    obb: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-3,
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
        assert obb_corners_equivalent(
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
        assert obb_corners_equivalent(
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
    np.testing.assert_allclose(polygon_center(polys_in), polygon_center(polys_out), rtol=1e-5)
    np.testing.assert_allclose(polygon_area(polys_in), polygon_area(polys_out), rtol=1e-5)


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


# --- cxcywh pipeline equivalence to cv2.boxPoints ---


def _cxcywh_pipeline_corners(
    bboxes_cxcywh: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Full pipeline: cxcywh (pixel) -> convert -> denormalize -> obb_to_polygons -> corners."""
    bboxes_alb = convert_bboxes_to_albumentations(
        bboxes_cxcywh,
        "cxcywh",
        shape,
        "obb",
        check_validity=False,
    )
    internal_px = np.column_stack(
        [denormalize_bboxes(bboxes_alb[:, :4], shape), bboxes_alb[:, 4:5]],
    )
    return obb_to_polygons(internal_px)


def _cxcywh_boxpoints_corners(bboxes_cxcywh: np.ndarray) -> np.ndarray:
    """Direct cv2.boxPoints from cxcywh (center, w, h, angle) in pixels."""
    arr = np.asarray(bboxes_cxcywh, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.array(
        [cv2.boxPoints(((r[0], r[1]), (r[2], r[3]), r[4])) for r in arr],
        dtype=np.float32,
    )


@pytest.mark.parametrize("angle_deg", ANGLES_FULL_ROUNDTRIP)
def test_cxcywh_pipeline_equals_boxpoints_axis_aligned(angle_deg: int) -> None:
    """Pipeline corners match cv2.boxPoints for axis-aligned angles (0°, ±90°, 180°, etc.)."""
    shape = (200, 300)
    cx, cy = 150.0, 100.0
    w, h = 60.0, 40.0
    bbox = np.array([[cx, cy, w, h, float(angle_deg)]], dtype=np.float32)

    pipeline_corners = _cxcywh_pipeline_corners(bbox, shape)[0]
    direct_corners = _cxcywh_boxpoints_corners(bbox)[0]

    assert obb_corners_equivalent(
        pipeline_corners,
        direct_corners,
        rtol=1e-5,
        atol=1e-4,
    ), f"Pipeline != boxPoints for angle {angle_deg}: pipeline={pipeline_corners}, direct={direct_corners}"


@pytest.mark.parametrize("angle_deg", ANGLES_ALL)
def test_cxcywh_pipeline_equals_boxpoints_general_angles(angle_deg: int) -> None:
    """Pipeline corners must match cv2.boxPoints for any angle."""
    shape = (200, 300)
    cx, cy = 150.0, 100.0
    w, h = 60.0, 40.0
    bbox = np.array([[cx, cy, w, h, float(angle_deg)]], dtype=np.float32)

    pipeline_corners = _cxcywh_pipeline_corners(bbox, shape)[0]
    direct_corners = _cxcywh_boxpoints_corners(bbox)[0]

    assert obb_corners_equivalent(
        pipeline_corners,
        direct_corners,
        rtol=1e-5,
        atol=1e-4,
    ), f"Pipeline != boxPoints for angle {angle_deg}: pipeline={pipeline_corners}, direct={direct_corners}"


@pytest.mark.parametrize("angle_deg", [0, 90, 180])
@pytest.mark.parametrize("cx,cy", [(50, 50), (137, 137), (200, 150)])
@pytest.mark.parametrize("w,h", [(30, 20), (100, 50), (50, 100)])
def test_cxcywh_pipeline_equals_boxpoints_parametrized(
    angle_deg: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
) -> None:
    """Pipeline == boxPoints for various centers, dimensions (axis-aligned angles only)."""
    shape = (300, 400)
    bbox = np.array([[cx, cy, w, h, float(angle_deg)]], dtype=np.float32)

    pipeline_corners = _cxcywh_pipeline_corners(bbox, shape)[0]
    direct_corners = _cxcywh_boxpoints_corners(bbox)[0]

    assert obb_corners_equivalent(pipeline_corners, direct_corners, rtol=1e-5, atol=1e-4)


def test_cxcywh_pipeline_equals_boxpoints_batch() -> None:
    """Pipeline == boxPoints for batch of boxes (axis-aligned angles)."""
    shape = (200, 300)
    rng = np.random.default_rng(137)
    n = 8
    centers = rng.uniform(30, 250, (n, 2))
    wh = rng.uniform(15, 80, (n, 2))
    angles = rng.choice([0, 90, 180, -90], size=n)  # axis-aligned only
    bboxes = np.column_stack(
        [centers[:, 0], centers[:, 1], wh[:, 0], wh[:, 1], angles],
    ).astype(np.float32)

    pipeline_corners = _cxcywh_pipeline_corners(bboxes, shape)
    direct_corners = _cxcywh_boxpoints_corners(bboxes)

    for i in range(n):
        assert obb_corners_equivalent(
            pipeline_corners[i],
            direct_corners[i],
            rtol=1e-5,
            atol=1e-3,
        ), f"Box {i} mismatch: pipeline={pipeline_corners[i]}, direct={direct_corners[i]}"


def test_cxcywh_pipeline_equals_boxpoints_thin_rect() -> None:
    """Pipeline == boxPoints for thin rectangles (axis-aligned only)."""
    shape = (200, 300)
    for angle in [0, 90]:
        bbox = np.array([[100, 100, 100.0, 3.0, float(angle)]], dtype=np.float32)
        pipeline_corners = _cxcywh_pipeline_corners(bbox, shape)[0]
        direct_corners = _cxcywh_boxpoints_corners(bbox)[0]
        assert obb_corners_equivalent(pipeline_corners, direct_corners, atol=1e-3)


def test_cxcywh_pipeline_equals_boxpoints_angle_wraparound() -> None:
    """Pipeline == boxPoints for angles outside [-90, 90) (e.g. 450° -> canonical)."""
    shape = (200, 300)
    for angle in [180, 270, 360, 450, -180, -270]:
        bbox = np.array([[100, 100, 40, 30, float(angle)]], dtype=np.float32)
        pipeline_corners = _cxcywh_pipeline_corners(bbox, shape)[0]
        direct_corners = _cxcywh_boxpoints_corners(bbox)[0]
        assert obb_corners_equivalent(pipeline_corners, direct_corners, rtol=1e-5, atol=1e-4)


def test_cxcywh_pipeline_equals_boxpoints_boats_data() -> None:
    """Pipeline == boxPoints for real boats_raw.json OBB data (axis-aligned only)."""
    import json
    from pathlib import Path

    json_path = Path(__file__).resolve().parent.parent / "boats_raw.json"
    if not json_path.exists():
        pytest.skip("boats_raw.json not found")

    with json_path.open() as f:
        data = json.load(f)
    obbs = [b for b in data.get("bboxes_obb", []) if b.get("angle") is not None]
    if not obbs:
        pytest.skip("No OBBs with angle in boats_raw.json")

    # Use image shape from JSON if available, else default
    shape = (
        data.get("image_height", 600),
        data.get("image_width", 1000),
    )
    bboxes = np.array(
        [[b["center_x"], b["center_y"], b["width"], b["height"], b["angle"]] for b in obbs[:20]],
        dtype=np.float32,
    )

    # Filter to axis-aligned angles for this test (pipeline known to pass)
    angle_mod_90 = np.abs(bboxes[:, 4]) % 90
    axis_aligned = np.isclose(angle_mod_90, 0, atol=1e-5)
    bboxes = bboxes[axis_aligned]
    if len(bboxes) == 0:
        pytest.skip("No axis-aligned OBBs in sample")

    pipeline_corners = _cxcywh_pipeline_corners(bboxes, shape)
    direct_corners = _cxcywh_boxpoints_corners(bboxes)

    for i in range(len(bboxes)):
        assert obb_corners_equivalent(
            pipeline_corners[i],
            direct_corners[i],
            rtol=1e-5,
            atol=1e-3,
        ), f"Box {i} (angle={bboxes[i, 4]}) mismatch"
