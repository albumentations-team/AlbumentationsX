"""OBB test helpers: convention-agnostic comparison using invariants."""

import numpy as np


def polygon_area(corners: np.ndarray) -> float | np.ndarray:
    """Shoelace formula for polygon area.

    Args:
        corners: (4, 2) or (N, 4, 2) array of corner coordinates.

    Returns:
        Scalar area for single polygon, or (N,) array for batched.

    """
    if corners.ndim == 2:
        term = corners[:, 0] * np.roll(corners[:, 1], -1) - np.roll(corners[:, 0], -1) * corners[:, 1]
        return 0.5 * abs(float(np.sum(term)))
    term = (
        corners[:, :, 0] * np.roll(corners[:, :, 1], -1, axis=1)
        - np.roll(corners[:, :, 0], -1, axis=1) * corners[:, :, 1]
    )
    return 0.5 * np.abs(np.sum(term, axis=1))


def polygon_center(corners: np.ndarray) -> np.ndarray:
    """Mean of polygon corners (center).

    Args:
        corners: (4, 2) or (N, 4, 2) array of corner coordinates.

    Returns:
        (2,) or (N, 2) array of center coordinates.

    """
    axis = 0 if corners.ndim == 2 else 1
    return corners.mean(axis=axis)


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
    c_a = polygon_center(corners_a)
    c_b = polygon_center(corners_b)
    if not np.allclose(c_a, c_b, rtol=rtol, atol=atol):
        return False

    # Area
    area_a = polygon_area(corners_a)
    area_b = polygon_area(corners_b)
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
