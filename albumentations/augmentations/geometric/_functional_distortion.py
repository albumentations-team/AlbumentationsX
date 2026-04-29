"""Distortion map, piecewise affine, optical, and TPS functional helpers."""

from __future__ import annotations

from typing import Literal

from ._functional_shared import (
    cv2,
    math,
    np,
    reduce_sum,
)


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


def generate_displacement_fields(
    image_shape: tuple[int, int],
    alpha: float,
    sigma: float,
    same_dxdy: bool,
    kernel_size: tuple[int, int],
    random_generator: np.random.Generator,
    noise_distribution: Literal["gaussian", "uniform"],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate displacement fields for elastic transform. Params: alpha, sigma,
    shape; random_generator for reproducibility. Returns map_x, map_y.

    This function generates displacement fields for elastic transform based on the provided parameters.
    It generates noise either from a Gaussian or uniform distribution and normalizes it to the range [-1, 1].

    Args:
        image_shape (tuple[int, int]): The shape of the image as (height, width).
        alpha (float): The alpha parameter for the elastic transform.
        sigma (float): The sigma parameter for the elastic transform.
        same_dxdy (bool): Whether to use the same displacement field for both x and y directions.
        kernel_size (tuple[int, int]): The size of the kernel for the elastic transform.
        random_generator (np.random.Generator): The random number generator to use.
        noise_distribution (Literal['gaussian', 'uniform']): The distribution of the noise.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - fields: The displacement fields for the elastic transform.
            - output_shape: The output shape of the elastic warp.

    """
    # Pre-allocate memory and generate noise in one step
    if noise_distribution == "gaussian":
        # Generate and normalize in one step, directly as float32
        fields = random_generator.standard_normal(
            (1 if same_dxdy else 2, *image_shape[:2]),
            dtype=np.float32,
        )
        # Normalize inplace
        max_abs = np.abs(fields, out=np.empty_like(fields)).max()
        if max_abs > 1e-6:
            fields /= max_abs
    else:  # uniform is already normalized to [-1, 1]
        fields = random_generator.uniform(
            -1,
            1,
            size=(1 if same_dxdy else 2, *image_shape[:2]),
        ).astype(np.float32)

    # # Apply Gaussian blur if needed using fast OpenCV operations
    # When kernel_size is (0,0) cv2.GaussianBlur uses automatic kernel size. Kernel == (0,0) is NOT a noop.
    # Reshape to 2D array (combining first dimension with height)
    shape = fields.shape
    fields = fields.reshape(-1, shape[-1])

    # Apply blur to all fields at once
    cv2.GaussianBlur(
        fields,
        kernel_size,
        sigma,
        dst=fields,
        borderType=cv2.BORDER_REPLICATE,
    )

    # Restore original shape
    fields = fields.reshape(shape)

    # Scale by alpha inplace
    fields *= alpha

    # Return views of the array to avoid copies
    return (fields[0], fields[0]) if same_dxdy else (fields[0], fields[1])


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


def compute_pairwise_distances(
    points1: np.ndarray,
    points2: np.ndarray,
) -> np.ndarray:
    """Compute pairwise Euclidean squared distances between points1 (N, 2) and points2 (M, 2).
    Returns (N, M) matrix. For TPS and nearest-neighbor. Uses cv2.gemm.

    Args:
        points1 (np.ndarray): First set of points with shape (N, 2)
        points2 (np.ndarray): Second set of points with shape (M, 2)

    Returns:
        np.ndarray: Matrix of pairwise distances with shape (N, M)

    """
    points1 = np.ascontiguousarray(points1, dtype=np.float32)
    points2 = np.ascontiguousarray(points2, dtype=np.float32)

    # Compute squared terms
    p1_squared = reduce_sum(cv2.multiply(points1, points1), axis=1, keepdims=True)
    p2_squared = reduce_sum(cv2.multiply(points2, points2), axis=1)[None, :]

    # Compute dot product
    dot_product = cv2.gemm(points1, points2.T, 1, None, 0)

    return p1_squared + p2_squared - 2 * dot_product


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
    distances = compute_pairwise_distances(src_points, src_points)

    kernel_matrix = np.where(
        distances > 0,
        distances * distances * cv2.log(distances + 1e-6),
        0,
    ).astype(np.float32)

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

    distances = compute_pairwise_distances(target_points, control_points)

    # Ensure kernel matrix is float32
    kernel_matrix = np.where(
        distances > 0,
        distances * cv2.log(distances + 1e-6),
        0,
    ).astype(np.float32)

    # Prepare affine terms
    num_points = len(target_points)
    affine_terms = np.empty((num_points, 3), dtype=np.float32)
    affine_terms[:, 0] = 1
    affine_terms[:, 1:] = target_points

    # Matrix multiplications with consistent float32 type
    nonlinear_part = cv2.gemm(kernel_matrix, nonlinear_weights, 1, None, 0)
    affine_part = cv2.gemm(affine_terms, affine_weights, 1, None, 0)

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
    return cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        None,
        (width, height),
        cv2.CV_32FC1,
    )


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
    # Create coordinate grid
    y, x = np.mgrid[:height, :width].astype(np.float32)

    x = x - center_x
    y = y - center_y

    # Calculate polar coordinates
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    # Normalize radius by the maximum possible radius to keep distortion in check
    max_radius = math.sqrt(max(center_x, width - center_x) ** 2 + max(center_y, height - center_y) ** 2)
    r_norm = r / max_radius

    # Apply fisheye distortion to normalized radius
    r_dist = r * (1 + k * r_norm * r_norm)

    # Convert back to cartesian coordinates
    map_x = r_dist * np.cos(theta) + center_x
    map_y = r_dist * np.sin(theta) + center_y

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
    "compute_pairwise_distances",
    "compute_tps_weights",
    "create_piecewise_affine_maps",
    "generate_control_points",
    "generate_displacement_fields",
    "generate_distorted_grid_polygons",
    "generate_inverse_distortion_map",
    "get_camera_matrix_distortion_maps",
    "get_fisheye_distortion_maps",
    "tps_transform",
]
