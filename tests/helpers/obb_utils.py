"""OBB test helpers: convention-agnostic comparison using invariants."""

import numpy as np


def _polygon_area(corners: np.ndarray) -> float:
    """Shoelace formula for polygon area."""
    return 0.5 * abs(
        np.sum(corners[:, 0] * np.roll(corners[:, 1], -1)) - np.sum(np.roll(corners[:, 0], -1) * corners[:, 1]),
    )


def obb_corners_equivalent(
    corners_a: np.ndarray,
    corners_b: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> bool:
    """Check if two 4-corner polygons represent the same rectangle.

    Uses invariants only (convention-agnostic):
    - Center (mean of corners)
    - Area (shoelace)
    - Sorted corner coordinates (lexicographic by x then y)

    Independent of angle, width/height ordering, or corner order.
    """
    if corners_a.shape != (4, 2) or corners_b.shape != (4, 2):
        return False

    # Center
    c_a = corners_a.mean(axis=0)
    c_b = corners_b.mean(axis=0)
    if not np.allclose(c_a, c_b, rtol=rtol, atol=atol):
        return False

    # Area
    area_a = _polygon_area(corners_a)
    area_b = _polygon_area(corners_b)
    if not np.isclose(area_a, area_b, rtol=rtol, atol=atol):
        return False

    # Match corners: for each corner in A, find closest in B. All 4 pairs must be within tolerance.
    # More robust than lexicographic sort when float precision causes different order.
    used = np.zeros(4, dtype=bool)
    for i in range(4):
        dists = np.linalg.norm(corners_b - corners_a[i], axis=1)
        dists[used] = np.inf
        j = np.argmin(dists)
        if dists[j] > atol + rtol * np.linalg.norm(corners_a[i]):
            return False
        used[j] = True
    return True
