"""Distortion map, piecewise affine, optical, and TPS functional helpers."""

from __future__ import annotations

from typing import NamedTuple, cast

from albucore import pairwise_distances_squared

from ._functional_shared import (
    cv2,
    np,
)


class _ElasticCellCoefficients(NamedTuple):
    origin: np.ndarray
    horizontal_basis: np.ndarray
    vertical_basis: np.ndarray
    bilinear_twist: np.ndarray
    x_origins: np.ndarray
    y_origins: np.ndarray
    widths: np.ndarray
    heights: np.ndarray


def generate_inverse_distortion_map(
    map_x: np.ndarray,
    map_y: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate inverse mapping for strong distortions. From forward map_x, map_y;
    returns inverse map for sampling. For PiecewiseAffine and similar.
    """
    h, w = shape

    src_y, src_x = np.mgrid[:h, :w]
    src_x_flat = src_x.ravel().astype(np.float32)
    src_y_flat = src_y.ravel().astype(np.float32)

    valid = (map_x >= 0) & (map_x < w) & (map_y >= 0) & (map_y < h)

    dst_x_floor = np.floor(map_x).astype(np.int32)
    dst_y_floor = np.floor(map_y).astype(np.int32)

    inv_map_x = np.zeros((h, w), dtype=np.float32)
    inv_map_y = np.zeros((h, w), dtype=np.float32)
    best_dist = np.full((h, w), np.inf, dtype=np.float32)

    map_x_flat = map_x.ravel()
    map_y_flat = map_y.ravel()

    for dy in range(2):
        for dx in range(2):
            ny = dst_y_floor + dy
            nx = dst_x_floor + dx

            mask = valid & (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            flat_mask = np.flatnonzero(mask.ravel())

            ny_m = ny.ravel()[flat_mask]
            nx_m = nx.ravel()[flat_mask]
            dist = np.abs(nx_m.astype(np.float32) - map_x_flat[flat_mask]) + np.abs(
                ny_m.astype(np.float32) - map_y_flat[flat_mask],
            )

            improve = dist < best_dist[ny_m, nx_m]

            ny_upd = ny_m[improve]
            nx_upd = nx_m[improve]
            flat_upd = flat_mask[improve]
            dist_upd = dist[improve]

            # Sort descending by dist so the minimum-dist source is written last and wins
            # when multiple source pixels compete for the same destination cell.
            order = np.argsort(dist_upd)[::-1]
            ny_upd = ny_upd[order]
            nx_upd = nx_upd[order]
            flat_upd = flat_upd[order]
            dist_upd = dist_upd[order]

            inv_map_x[ny_upd, nx_upd] = src_x_flat[flat_upd]
            inv_map_y[ny_upd, nx_upd] = src_y_flat[flat_upd]
            best_dist[ny_upd, nx_upd] = dist_upd

    return inv_map_x, inv_map_y


def upscale_distortion_maps(
    map_x: np.ndarray,
    map_y: np.ndarray,
    target_shape: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, np.ndarray]:
    """Upscale coarse distortion coordinate maps to full image size and rescale coordinates,
    enabling faster map generation while preserving full-resolution remapping.

    Distortion transforms can generate coordinate maps at reduced resolution, then upscale
    them before remapping full-resolution targets to trade geometric precision for speed.

    Args:
        map_x (np.ndarray): X-coordinate map generated at the lower resolution.
        map_y (np.ndarray): Y-coordinate map generated at the lower resolution.
        target_shape (tuple[int, int]): Target image shape as `(height, width)`.
        interpolation (int): OpenCV interpolation flag used for resizing the maps.

    Returns:
        tuple[np.ndarray, np.ndarray]: Upscaled `map_x` and `map_y` with coordinates
            adjusted for the target shape.

    """
    height, width = target_shape
    map_height, map_width = map_x.shape[:2]

    if (map_height, map_width) == (height, width):
        return map_x, map_y

    dx = map_x - np.arange(map_width, dtype=np.float32)
    dy = map_y - np.arange(map_height, dtype=np.float32)[:, None]

    scale_y = 1 if height == 1 or map_height == 1 else (map_height - 1) / (height - 1)
    scale_x = 1 if width == 1 or map_width == 1 else (map_width - 1) / (width - 1)
    dx = cv2.resize(dx, (width, height), interpolation=interpolation) / scale_x
    dy = cv2.resize(dy, (width, height), interpolation=interpolation) / scale_y

    dx += np.arange(width, dtype=np.float32)
    dy += np.arange(height, dtype=np.float32)[:, None]
    return dx, dy


def expand_control_grid(
    control_vectors: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Expand endpoint-aligned control vectors into a dense component-first bilinear field with exact anchors across
    the output shape.

    Args:
        control_vectors (np.ndarray): Control vectors with shape `(rows, columns, 2)`.
        output_shape (tuple[int, int]): Dense output shape as `(height, width)`.

    Returns:
        np.ndarray: Component-first float32 displacement field with shape `(2, height, width)`.

    """
    control = np.asarray(control_vectors, dtype=np.float32)
    if control.ndim != 3 or control.shape[-1] != 2 or min(control.shape[:2]) < 2:
        raise ValueError("control_vectors must have shape (rows >= 2, columns >= 2, 2)")
    rows, columns, _ = control.shape

    height, width = output_shape
    x = np.linspace(0, columns - 1, width, dtype=np.float32)
    x0 = np.floor(x).astype(np.intp)
    x1 = np.minimum(x0 + 1, columns - 1)
    x_weight = x - x0.astype(np.float32)
    horizontal = control[:, x0, :] * (1.0 - x_weight)[None, :, None]
    horizontal += control[:, x1, :] * x_weight[None, :, None]

    y = np.linspace(0, rows - 1, height, dtype=np.float32)
    y0 = np.floor(y).astype(np.intp)
    y1 = np.minimum(y0 + 1, rows - 1)
    y_weight = y - y0.astype(np.float32)
    dense = horizontal[y0, :, :] * (1.0 - y_weight)[:, None, None]
    dense += horizontal[y1, :, :] * y_weight[:, None, None]
    return np.moveaxis(dense, -1, 0)


def create_elastic_maps(
    control_vectors: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Create float32 target-to-source maps from an endpoint-aligned control grid for synchronized raster
    remapping in one run.
    """
    height, width = image_shape
    displacement = expand_control_grid(control_vectors, image_shape)
    map_x = displacement[0]
    map_y = displacement[1]
    map_x += np.arange(width, dtype=np.float32)
    map_y += np.arange(height, dtype=np.float32)[:, None]
    return map_x, map_y


def _cross2d(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _elastic_cell_coefficients(
    control_vectors: np.ndarray,
    image_shape: tuple[int, int],
) -> _ElasticCellCoefficients:
    control = np.asarray(control_vectors, dtype=np.float64)
    rows, columns, _ = control.shape
    height, width = image_shape
    x_anchors = np.linspace(0.0, width - 1.0, columns, dtype=np.float64)
    y_anchors = np.linspace(0.0, height - 1.0, rows, dtype=np.float64)
    cell_widths = np.diff(x_anchors)
    cell_heights = np.diff(y_anchors)

    top_left = control[:-1, :-1]
    top_right = control[:-1, 1:]
    bottom_left = control[1:, :-1]
    bottom_right = control[1:, 1:]
    grid_shape = (rows - 1, columns - 1)
    x_origins = np.broadcast_to(x_anchors[:-1], grid_shape)
    y_origins = np.broadcast_to(y_anchors[:-1, None], grid_shape)
    origin = np.stack([x_origins, y_origins], axis=-1) + top_left
    horizontal_basis = top_right - top_left
    horizontal_basis[..., 0] += cell_widths[None, :]
    vertical_basis = bottom_left - top_left
    vertical_basis[..., 1] += cell_heights[:, None]
    bilinear_twist = bottom_right - top_right - bottom_left + top_left

    return _ElasticCellCoefficients(
        origin=origin.reshape(-1, 2),
        horizontal_basis=horizontal_basis.reshape(-1, 2),
        vertical_basis=vertical_basis.reshape(-1, 2),
        bilinear_twist=bilinear_twist.reshape(-1, 2),
        x_origins=x_origins.reshape(-1),
        y_origins=y_origins.reshape(-1),
        widths=np.broadcast_to(cell_widths, grid_shape).reshape(-1),
        heights=np.broadcast_to(cell_heights[:, None], grid_shape).reshape(-1),
    )


def _solve_elastic_cells(
    points: np.ndarray,
    coefficients: _ElasticCellCoefficients,
    tolerance: float,
) -> np.ndarray:
    origin = coefficients.origin
    horizontal_basis = coefficients.horizontal_basis
    vertical_basis = coefficients.vertical_basis
    bilinear_twist = coefficients.bilinear_twist
    points64 = np.asarray(points, dtype=np.float64)
    residual_points = points64[:, None, :] - origin[None, :, :]
    quadratic_a = -_cross2d(horizontal_basis, bilinear_twist)[None, :]
    quadratic_b = (
        _cross2d(residual_points, bilinear_twist[None, :, :])
        - _cross2d(
            horizontal_basis,
            vertical_basis,
        )[None, :]
    )
    quadratic_c = _cross2d(residual_points, vertical_basis[None, :, :])

    with np.errstate(invalid="ignore", divide="ignore"):
        discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c
        sqrt_discriminant = np.sqrt(np.maximum(discriminant, 0.0))
        horizontal_fractions = np.stack(
            [
                (-quadratic_b + sqrt_discriminant) / (2.0 * quadratic_a),
                (-quadratic_b - sqrt_discriminant) / (2.0 * quadratic_a),
            ],
            axis=-1,
        )
        linear_fraction = -quadratic_c / quadratic_b
    linear = np.abs(quadratic_a) <= 1e-12
    horizontal_fractions = np.where(linear[..., None], linear_fraction[..., None], horizontal_fractions)
    horizontal_fractions[..., 1] = np.where(linear, np.nan, horizontal_fractions[..., 1])
    horizontal_fractions = np.where(discriminant[..., None] >= -1e-10, horizontal_fractions, np.nan)

    vertical_numerator = (
        points64[:, None, None, :]
        - origin[None, :, None, :]
        - horizontal_basis[None, :, None, :] * horizontal_fractions[..., None]
    )
    vertical_denominator = vertical_basis[None, :, None, :] + (
        bilinear_twist[None, :, None, :] * horizontal_fractions[..., None]
    )
    denominator_x = np.where(np.abs(vertical_denominator[..., 0]) > 1e-12, vertical_denominator[..., 0], 1.0)
    denominator_y = np.where(np.abs(vertical_denominator[..., 1]) > 1e-12, vertical_denominator[..., 1], 1.0)
    use_x = np.abs(vertical_denominator[..., 0]) >= np.abs(vertical_denominator[..., 1])
    vertical_fractions = np.where(
        use_x,
        vertical_numerator[..., 0] / denominator_x,
        vertical_numerator[..., 1] / denominator_y,
    )

    valid = np.isfinite(horizontal_fractions) & np.isfinite(vertical_fractions)
    valid &= (horizontal_fractions >= -1e-7) & (horizontal_fractions <= 1.0 + 1e-7)
    valid &= (vertical_fractions >= -1e-7) & (vertical_fractions <= 1.0 + 1e-7)
    forward = (
        origin[None, :, None, :]
        + horizontal_basis[None, :, None, :] * horizontal_fractions[..., None]
        + vertical_basis[None, :, None, :] * vertical_fractions[..., None]
        + bilinear_twist[None, :, None, :] * horizontal_fractions[..., None] * vertical_fractions[..., None]
    )
    residual = np.max(np.abs(forward - points64[:, None, None, :]), axis=-1)
    valid &= residual <= tolerance

    output_x = coefficients.x_origins[None, :, None] + horizontal_fractions * coefficients.widths[None, :, None]
    output_y = coefficients.y_origins[None, :, None] + vertical_fractions * coefficients.heights[None, :, None]
    candidates = np.stack([output_x, output_y], axis=-1)
    candidate_residuals = np.where(valid, residual, np.inf).reshape(len(points64), -1)
    candidate_indices = np.argmin(candidate_residuals, axis=1)
    best_residuals = candidate_residuals[np.arange(len(points64)), candidate_indices]
    best_candidates = candidates.reshape(len(points64), -1, 2)[np.arange(len(points64)), candidate_indices]
    best_candidates[~np.isfinite(best_residuals)] = -1.0
    return best_candidates


def remap_elastic_keypoints(
    keypoints: np.ndarray,
    control_vectors: np.ndarray,
    image_shape: tuple[int, int],
    tolerance: float = 1e-3,
) -> np.ndarray:
    """Invert an injective bilinear control-grid map analytically for keypoints with strict residual
    validation for each candidate.
    """
    if keypoints.size == 0:
        return keypoints.copy()
    coefficients = _elastic_cell_coefficients(control_vectors, image_shape)
    transformed_xy = _solve_elastic_cells(keypoints[:, :2], coefficients, tolerance)
    result = keypoints.copy()
    result[:, :2] = transformed_xy.astype(result.dtype, copy=False)
    return result


def generate_distorted_grid_polygons(
    dimensions: np.ndarray,
    magnitude: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate distorted grid polygons from dimensions and magnitude. Internal
    vertices randomized; boundary fixed. For PiecewiseAffine mesh generation.

    This function creates a grid of polygons and applies random distortions to the internal vertices,
    while keeping the boundary vertices fixed. The distortion is applied consistently across shared
    vertices to avoid gaps or overlaps in the resulting grid.

    Args:
        dimensions (np.ndarray): A 3D array of shape (grid_height, grid_width, 4) where each element
                                 is [x_min, y_min, x_max, y_max] representing the dimensions of a grid cell.
        magnitude (int): Maximum pixel-wise displacement for distortion. The actual displacement
                         will be randomly chosen in the range [-magnitude, magnitude].
        random_generator (np.random.Generator): A random number generator.

    Returns:
        np.ndarray: A 2D array of shape (total_cells, 8) where each row represents a distorted polygon
                    as [x1, y1, x2, y1, x2, y2, x1, y2]. The total_cells is equal to grid_height * grid_width.

    Note:
        - Only internal grid points are distorted; boundary points remain fixed.
        - The function ensures consistent distortion across shared vertices of adjacent cells.
        - The distortion is applied to the following points of each internal cell:
            * Bottom-right of the cell above and to the left
            * Bottom-left of the cell above
            * Top-right of the cell to the left
            * Top-left of the current cell
        - Each square represents a cell, and the X marks indicate the coordinates where displacement occurs.
            +--+--+--+--+
            |  |  |  |  |
            +--X--X--X--+
            |  |  |  |  |
            +--X--X--X--+
            |  |  |  |  |
            +--X--X--X--+
            |  |  |  |  |
            +--+--+--+--+
        - For each X, the coordinates of the left, right, top, and bottom edges
          in the four adjacent cells are displaced.

    Examples:
        >>> dimensions = np.array([[[0, 0, 50, 50], [50, 0, 100, 50]],
        ...                        [[0, 50, 50, 100], [50, 50, 100, 100]]])
        >>> distorted = generate_distorted_grid_polygons(dimensions, magnitude=10)
        >>> distorted.shape
        (4, 8)

    """
    grid_height, grid_width = dimensions.shape[:2]
    total_cells = grid_height * grid_width

    # Initialize polygons
    polygons = np.zeros((total_cells, 8), dtype=np.float32)
    polygons[:, 0:2] = dimensions.reshape(-1, 4)[:, [0, 1]]  # x1, y1
    polygons[:, 2:4] = dimensions.reshape(-1, 4)[:, [2, 1]]  # x2, y1
    polygons[:, 4:6] = dimensions.reshape(-1, 4)[:, [2, 3]]  # x2, y2
    polygons[:, 6:8] = dimensions.reshape(-1, 4)[:, [0, 3]]  # x1, y2

    # Generate displacements for internal grid points only
    internal_points_height, internal_points_width = grid_height - 1, grid_width - 1
    displacements = random_generator.integers(
        -magnitude,
        magnitude + 1,
        size=(internal_points_height, internal_points_width, 2),
    ).astype(np.float32)

    # Apply displacements to internal polygon vertices
    for i in range(1, grid_height):
        for j in range(1, grid_width):
            dx, dy = displacements[i - 1, j - 1]

            # Bottom-right of cell (i-1, j-1)
            polygons[(i - 1) * grid_width + (j - 1), 4:6] += [dx, dy]

            # Bottom-left of cell (i-1, j)
            polygons[(i - 1) * grid_width + j, 6:8] += [dx, dy]

            # Top-right of cell (i, j-1)
            polygons[i * grid_width + (j - 1), 2:4] += [dx, dy]

            # Top-left of cell (i, j)
            polygons[i * grid_width + j, 0:2] += [dx, dy]

    return polygons


def create_piecewise_affine_maps(
    image_shape: tuple[int, int],
    grid: tuple[int, int],
    scale: float,
    absolute_scale: bool,
    random_generator: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Create map_x and map_y for PiecewiseAffine: jittered grid and IDW yield full-resolution
    remap maps. Used by the transform; result is passed to OpenCV remap.

    It generates the control points for the transformation, then uses the remap function to create
    the transformation maps.

    Args:
        image_shape (tuple[int, int]): The shape of the image as (height, width).
        grid (tuple[int, int]): The grid size as (rows, columns).
        scale (float): The scale of the transformation.
        absolute_scale (bool): Whether to use absolute scale.
        random_generator (np.random.Generator): The random generator to use for generating the points.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: The transformation maps.

    """
    height, width = image_shape[:2]
    nb_rows, nb_cols = grid

    # Input validation
    if height <= 0 or width <= 0 or nb_rows <= 0 or nb_cols <= 0:
        raise ValueError("Dimensions must be positive")
    if scale <= 0:
        return None, None

    # Create source points grid
    y = np.linspace(0, height - 1, nb_rows, dtype=np.float32)
    x = np.linspace(0, width - 1, nb_cols, dtype=np.float32)
    xx_src, yy_src = np.meshgrid(x, y)

    # Generate jitter for control points
    jitter_scale = scale / 3 if absolute_scale else scale * min(width, height) / 3

    jitter = random_generator.normal(0, jitter_scale, (nb_rows, nb_cols, 2)).astype(
        np.float32,
    )

    # Control points: source (x,y) and jittered destination (x,y)
    control_points = np.zeros((nb_rows * nb_cols, 4), dtype=np.float32)
    control_points[:, 0] = xx_src.ravel()
    control_points[:, 1] = yy_src.ravel()
    np.clip(
        xx_src.ravel() + jitter[:, :, 1].ravel(),
        0,
        width - 1,
        out=control_points[:, 2],
    )
    np.clip(
        yy_src.ravel() + jitter[:, :, 0].ravel(),
        0,
        height - 1,
        out=control_points[:, 3],
    )

    # IDW: loop over control points, accumulate weights and weighted dest on full grid.
    # O(H*W*K) memory would be large; we keep O(H*W) by accumulating per control point.
    yy, xx = np.mgrid[:height, :width]
    xx_f = xx.astype(np.float32)
    yy_f = yy.astype(np.float32)

    numerator_x = np.zeros((height, width), dtype=np.float32)
    numerator_y = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    for cp in control_points:
        dx = xx_f - cp[0]
        dy = yy_f - cp[1]
        w = np.float32(1.0) / (dx * dx + dy * dy + np.float32(1e-8))
        weight_sum += w
        numerator_x += w * cp[2]
        numerator_y += w * cp[3]

    map_x = numerator_x / weight_sum
    map_y = numerator_y / weight_sum

    map_x = np.clip(map_x, 0, width - 1, out=map_x)
    map_y = np.clip(map_y, 0, height - 1, out=map_y)

    return map_x, map_y


def _compute_tps_kernel(distances: np.ndarray) -> np.ndarray:
    log_distances = cv2.log(distances + np.float32(1e-6))
    return np.multiply(distances, log_distances, out=log_distances)


def compute_tps_weights(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Thin Plate Spline weights from src_points and dst_points. Returns
    (nonlinear_weights, affine_weights) for TPS warp. For ThinPlateSpline.

    Args:
        src_points (np.ndarray): Source control points with shape (num_points, 2)
        dst_points (np.ndarray): Destination control points with shape (num_points, 2)

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (nonlinear_weights, affine_weights)
        - nonlinear_weights: TPS kernel weights for nonlinear deformation (num_points, 2)
        - affine_weights: Weights for affine transformation (3, 2)
            [constant term, x scale/shear, y scale/shear]

    Note:
        The TPS interpolation is decomposed into:
        1. Nonlinear part (controlled by kernel weights)
        2. Affine part (global scaling, rotation, translation)

    """
    num_points = src_points.shape[0]

    # Compute pairwise distances
    distances = pairwise_distances_squared(src_points, src_points)
    kernel_matrix = _compute_tps_kernel(distances)

    # Build system matrix efficiently
    affine_terms = np.empty((num_points, 3), dtype=np.float32)
    affine_terms[:, 0] = 1
    affine_terms[:, 1:] = src_points

    # Construct system matrix
    system_matrix = np.zeros((num_points + 3, num_points + 3), dtype=np.float32)
    system_matrix[:num_points, :num_points] = kernel_matrix
    system_matrix[:num_points, num_points:] = affine_terms
    system_matrix[num_points:, :num_points] = affine_terms.T

    # Prepare target coordinates
    target = np.zeros((num_points + 3, 2), dtype=np.float32)
    target[:num_points] = dst_points

    weights = cv2.solve(system_matrix, target, flags=cv2.DECOMP_LU)[1]

    return weights[:num_points], weights[num_points:]


def tps_transform(
    target_points: np.ndarray,
    control_points: np.ndarray,
    nonlinear_weights: np.ndarray,
    affine_weights: np.ndarray,
) -> np.ndarray:
    """Apply TPS transformation to target_points given control_points and
    nonlinear_weights, affine_weights. All float32. For ThinPlateSpline remap.
    """
    # Ensure float32 type for all inputs
    target_points = np.ascontiguousarray(target_points, dtype=np.float32)
    control_points = np.ascontiguousarray(control_points, dtype=np.float32)
    nonlinear_weights = np.ascontiguousarray(nonlinear_weights, dtype=np.float32)
    affine_weights = np.ascontiguousarray(affine_weights, dtype=np.float32)

    distances = pairwise_distances_squared(target_points, control_points)
    kernel_matrix = _compute_tps_kernel(distances)

    # Prepare affine terms
    num_points = len(target_points)
    affine_terms = np.empty((num_points, 3), dtype=np.float32)
    affine_terms[:, 0] = 1
    affine_terms[:, 1:] = target_points

    # Matrix multiplications with consistent float32 type
    empty_matrix = np.empty((0,), dtype=np.float32)
    nonlinear_part = cast("np.ndarray", cv2.gemm(kernel_matrix, nonlinear_weights, 1.0, empty_matrix, 0.0))
    affine_part = cast("np.ndarray", cv2.gemm(affine_terms, affine_weights, 1.0, empty_matrix, 0.0))

    return nonlinear_part + affine_part


def get_camera_matrix_distortion_maps(
    image_shape: tuple[int, int],
    k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (map_x, map_y) from camera matrix model. image_shape, k.
    For OpticalDistortion. cv2.initUndistortRectifyMap style.

    Args:
        image_shape (tuple[int, int]): Image shape (height, width)
        k (float): Distortion coefficient

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (map_x, map_y) distortion maps

    """
    height, width = image_shape[:2]

    center_x, center_y = width / 2, height / 2

    camera_matrix = np.array(
        [[width, 0, center_x], [0, height, center_y], [0, 0, 1]],
        dtype=np.float32,
    )
    distortion = np.array([k, k, 0, 0, 0], dtype=np.float32)
    empty_matrix = np.empty((0,), dtype=np.float32)
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        empty_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    return cast("np.ndarray", map_x), cast("np.ndarray", map_y)


def get_fisheye_distortion_maps(
    image_shape: tuple[int, int],
    k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (map_x, map_y) distortion maps from fisheye model. image_shape, k.
    Radial distortion r*(1+k*r_norm^2). For OpticalDistortion fisheye.

    Args:
        image_shape (tuple[int, int]): Image shape (height, width)
        k (float): Distortion coefficient

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (map_x, map_y) distortion maps

    """
    height, width = image_shape[:2]

    center_x, center_y = width / 2, height / 2
    x = np.arange(width, dtype=np.float32)[np.newaxis, :] - center_x
    y = np.arange(height, dtype=np.float32)[:, np.newaxis] - center_y

    max_radius_squared = max(center_x, width - center_x) ** 2 + max(center_y, height - center_y) ** 2
    distortion_scale = x * x + y * y
    distortion_scale *= np.float32(k / max_radius_squared)
    distortion_scale += 1

    map_x = x * distortion_scale + center_x
    map_y = y * distortion_scale + center_y

    return map_x, map_y


def generate_control_points(num_control_points: int) -> np.ndarray:
    """Generate control points for TPS in unit square. num_control_points per side;
    special case 2 -> 4 corners + center. Returns (N, 2). For ThinPlateSpline.

    Args:
        num_control_points (int): Number of control points per side

    Returns:
        np.ndarray: Control points with shape (N, 2)

    """
    if num_control_points == 2:
        # Generate 4 corners + center point similar to Kornia
        return np.array(
            [
                [0, 0],  # top-left
                [0, 1],  # bottom-left
                [1, 0],  # top-right
                [1, 1],  # bottom-right
                [0.5, 0.5],  # center
            ],
            dtype=np.float32,
        )

        # Generate regular grid
    x = np.linspace(0, 1, num_control_points, dtype=np.float32)
    y = np.linspace(0, 1, num_control_points, dtype=np.float32)
    return np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)


__all__ = [
    "compute_tps_weights",
    "create_elastic_maps",
    "create_piecewise_affine_maps",
    "expand_control_grid",
    "generate_control_points",
    "generate_distorted_grid_polygons",
    "generate_inverse_distortion_map",
    "get_camera_matrix_distortion_maps",
    "get_fisheye_distortion_maps",
    "remap_elastic_keypoints",
    "tps_transform",
    "upscale_distortion_maps",
]
