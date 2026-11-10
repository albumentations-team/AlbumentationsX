"""Numerical operations shared by geometric vector-field targets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from albucore import remap, remap3d
from numpy.typing import NDArray

from albumentations.core.type_definitions import CV2_BORDER_CONSTANT, CV2_INTER_LINEAR


def resample_field_2d(
    field: NDArray[np.float16] | NDArray[np.float32],
    map_x: NDArray[np.float32],
    map_y: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Bilinearly resample a channel-last field, excluding invalid vectors and border samples."""
    output_shape = map_x.shape
    if field.size == 0 or output_shape[0] == 0 or output_shape[1] == 0:
        return np.full((*output_shape, field.shape[-1]), np.nan, dtype=np.float32)

    finite_map = np.isfinite(map_x) & np.isfinite(map_y)
    source_height, source_width = field.shape[:2]
    in_domain = (
        finite_map & (map_x >= -0.5) & (map_x < source_width - 0.5) & (map_y >= -0.5) & (map_y < source_height - 0.5)
    )
    safe_map_x = np.where(finite_map, map_x, -1.0).astype(np.float32, copy=False)
    safe_map_y = np.where(finite_map, map_y, -1.0).astype(np.float32, copy=False)
    valid_vectors = np.asarray(np.isfinite(field).all(axis=-1), dtype=np.bool_)
    source_float = field.astype(np.float32, copy=False)
    numerator_input = np.where(valid_vectors[..., None], source_float, np.float32(0))
    support_input = valid_vectors[..., None].astype(np.float32)

    numerator = remap(
        numerator_input,
        safe_map_x,
        safe_map_y,
        interpolation=CV2_INTER_LINEAR,
        border_mode=CV2_BORDER_CONSTANT,
        border_value=0,
    )
    support = remap(
        support_input,
        safe_map_x,
        safe_map_y,
        interpolation=CV2_INTER_LINEAR,
        border_mode=CV2_BORDER_CONSTANT,
        border_value=0,
    )[..., 0]

    result = np.full((*output_shape, field.shape[-1]), np.nan, dtype=np.float32)
    has_support = in_domain & (support > 0)
    np.divide(numerator, support[..., None], out=result, where=has_support[..., None])
    result_valid = np.isfinite(result).all(axis=-1)
    result[~result_valid] = np.nan
    return result


def pull_map_from_matrix_2d(
    matrix: NDArray[np.float32],
    output_shape: tuple[int, int],
    source_shape: tuple[int, int],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
    """Build a pixel-center pull map and source-domain mask from a forward affine or perspective matrix."""
    forward_matrix = np.asarray(matrix, dtype=np.float32)
    if forward_matrix.shape == (2, 3):
        homogeneous_matrix = np.vstack((forward_matrix, np.asarray((0.0, 0.0, 1.0), dtype=np.float32)))
    elif forward_matrix.shape == (3, 3):
        homogeneous_matrix = forward_matrix
    else:
        output_height, output_width = output_shape
        invalid_map = np.full((output_height, output_width), np.nan, dtype=np.float32)
        invalid_domain = np.zeros((output_height, output_width), dtype=bool)
        return invalid_map, invalid_map.copy(), invalid_domain

    try:
        inverse = np.linalg.inv(homogeneous_matrix).astype(np.float32, copy=False)
    except np.linalg.LinAlgError:
        output_height, output_width = output_shape
        invalid_map = np.full((output_height, output_width), np.nan, dtype=np.float32)
        return invalid_map, invalid_map.copy(), np.zeros((output_height, output_width), dtype=bool)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        output_height, output_width = output_shape
        output_y = np.arange(output_height, dtype=np.float32)[:, None]
        output_x = np.arange(output_width, dtype=np.float32)[None, :]
        denominator = inverse[2, 0] * output_x + inverse[2, 1] * output_y + inverse[2, 2]
        map_x = (inverse[0, 0] * output_x + inverse[0, 1] * output_y + inverse[0, 2]) / denominator
        map_y = (inverse[1, 0] * output_x + inverse[1, 1] * output_y + inverse[1, 2]) / denominator

    source_height, source_width = source_shape
    domain = np.isfinite(map_x) & np.isfinite(map_y) & (denominator != 0)
    domain &= (map_x >= -0.5) & (map_x < source_width - 0.5) & (map_y >= -0.5) & (map_y < source_height - 0.5)
    return np.asarray(map_x, dtype=np.float32), np.asarray(map_y, dtype=np.float32), domain


def _coordinates_to_grid_3d(
    x_coordinates: NDArray[np.float32],
    y_coordinates: NDArray[np.float32],
    z_coordinates: NDArray[np.float32],
    source_shape: tuple[int, int, int],
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Convert voxel-center coordinates to an align-corners-false grid and its source-cell domain."""
    source_depth, source_height, source_width = source_shape
    finite = np.isfinite(x_coordinates) & np.isfinite(y_coordinates) & np.isfinite(z_coordinates)
    in_domain = finite & (x_coordinates >= -0.5) & (x_coordinates < source_width - 0.5)
    in_domain &= (y_coordinates >= -0.5) & (y_coordinates < source_height - 0.5)
    in_domain &= (z_coordinates >= -0.5) & (z_coordinates < source_depth - 0.5)
    grid = np.stack(
        (
            2 * (x_coordinates + 0.5) / source_width - 1,
            2 * (y_coordinates + 0.5) / source_height - 1,
            2 * (z_coordinates + 0.5) / source_depth - 1,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    grid[~finite] = np.float32(2)
    return grid, in_domain


def pull_map_from_matrix_3d(
    matrix: NDArray[np.float32],
    output_shape: tuple[int, int, int],
    source_shape: tuple[int, int, int],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.bool_],
]:
    """Build a pixel-center pull grid for a forward 3D affine matrix."""
    forward_matrix = np.asarray(matrix, dtype=np.float32)
    if forward_matrix.shape == (3, 4):
        forward_matrix = np.vstack((forward_matrix, np.asarray((0, 0, 0, 1), dtype=np.float32)))
    if forward_matrix.shape != (4, 4):
        output_depth, output_height, output_width = output_shape
        invalid = np.full((output_depth, output_height, output_width), np.nan, dtype=np.float32)
        return invalid, invalid.copy(), invalid.copy(), invalid.copy(), np.zeros(invalid.shape, dtype=bool)
    try:
        inverse = np.linalg.inv(forward_matrix).astype(np.float32, copy=False)
    except np.linalg.LinAlgError:
        output_depth, output_height, output_width = output_shape
        invalid = np.full((output_depth, output_height, output_width), np.nan, dtype=np.float32)
        return invalid, invalid.copy(), invalid.copy(), invalid.copy(), np.zeros(invalid.shape, dtype=bool)

    output_depth, output_height, output_width = output_shape
    z_coordinates = np.arange(output_depth, dtype=np.float32)[:, None, None]
    y_coordinates = np.arange(output_height, dtype=np.float32)[None, :, None]
    x_coordinates = np.arange(output_width, dtype=np.float32)[None, None, :]
    denominator = (
        inverse[3, 0] * x_coordinates + inverse[3, 1] * y_coordinates + inverse[3, 2] * z_coordinates + inverse[3, 3]
    )
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        source_x = (
            inverse[0, 0] * x_coordinates
            + inverse[0, 1] * y_coordinates
            + inverse[0, 2] * z_coordinates
            + inverse[0, 3]
        ) / denominator
        source_y = (
            inverse[1, 0] * x_coordinates
            + inverse[1, 1] * y_coordinates
            + inverse[1, 2] * z_coordinates
            + inverse[1, 3]
        ) / denominator
        source_z = (
            inverse[2, 0] * x_coordinates
            + inverse[2, 1] * y_coordinates
            + inverse[2, 2] * z_coordinates
            + inverse[2, 3]
        ) / denominator
    grid, domain = _coordinates_to_grid_3d(source_x, source_y, source_z, source_shape)
    domain &= denominator != 0
    return (
        grid,
        source_x.astype(np.float32, copy=False),
        source_y.astype(np.float32, copy=False),
        source_z.astype(np.float32, copy=False),
        domain,
    )


def resample_field_3d(
    field: NDArray[np.float16] | NDArray[np.float32],
    sampling_grid: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Trilinearly resample a channel-last field with shared finite-vector support."""
    output_shape = sampling_grid.shape[:3]
    source_depth, source_height, source_width = field.shape[:3]
    if field.size == 0 or any(size == 0 for size in output_shape):
        empty = np.full((*output_shape, field.shape[-1]), np.nan, dtype=np.float32)
        return empty, np.zeros(output_shape, dtype=bool)

    normalized_grid = np.asarray(sampling_grid, dtype=np.float32)
    source_x = (normalized_grid[..., 0] + 1) * (source_width / 2) - 0.5
    source_y = (normalized_grid[..., 1] + 1) * (source_height / 2) - 0.5
    source_z = (normalized_grid[..., 2] + 1) * (source_depth / 2) - 0.5
    finite_map = np.isfinite(normalized_grid).all(axis=-1)
    in_domain = finite_map & (source_x >= -0.5) & (source_x < source_width - 0.5)
    in_domain &= (source_y >= -0.5) & (source_y < source_height - 0.5)
    in_domain &= (source_z >= -0.5) & (source_z < source_depth - 0.5)
    valid_vectors = np.asarray(np.isfinite(field).all(axis=-1), dtype=np.bool_)
    source_float = field.astype(np.float32, copy=False)
    numerator_source = np.where(valid_vectors[..., None], source_float, np.float32(0))
    support_source = valid_vectors[..., None].astype(np.float32)
    numerator = remap3d(numerator_source, normalized_grid, interpolation=CV2_INTER_LINEAR)
    support = remap3d(support_source, normalized_grid, interpolation=CV2_INTER_LINEAR)[..., 0]
    result = np.full((*output_shape, field.shape[-1]), np.nan, dtype=np.float32)
    supported = in_domain & (support > 0)
    np.divide(numerator, support[..., None], out=result, where=supported[..., None])
    supported &= np.isfinite(result).all(axis=-1)
    result[~supported] = np.nan
    return result, supported


def warp_vector_field_3d(
    field: NDArray[np.float16] | NDArray[np.float32],
    matrix: NDArray[np.float32],
    output_shape: tuple[int, int, int],
    *,
    semantics: str,
) -> NDArray[np.float32]:
    """Resample a 3D field and apply an affine differential or endpoint mapping."""
    forward_matrix = np.asarray(matrix, dtype=np.float32)
    if forward_matrix.shape == (3, 4):
        forward_matrix = np.vstack((forward_matrix, np.asarray((0, 0, 0, 1), dtype=np.float32)))
    if forward_matrix.shape != (4, 4):
        return np.full((*output_shape, field.shape[-1]), np.nan, dtype=np.float32)
    grid, source_x, source_y, source_z, source_domain = pull_map_from_matrix_3d(
        forward_matrix,
        output_shape,
        field.shape[:3],
    )
    sampled, supported = resample_field_3d(field, grid)
    supported &= source_domain
    output_depth, output_height, output_width = output_shape
    output_z = np.arange(output_depth, dtype=np.float32)[:, None, None]
    output_y = np.arange(output_height, dtype=np.float32)[None, :, None]
    output_x = np.arange(output_width, dtype=np.float32)[None, None, :]

    if semantics == "displacement3d":
        endpoint_x = source_x + sampled[..., 0]
        endpoint_y = source_y + sampled[..., 1]
        endpoint_z = source_z + sampled[..., 2]
        homogeneous = (
            forward_matrix[3, 0] * endpoint_x
            + forward_matrix[3, 1] * endpoint_y
            + forward_matrix[3, 2] * endpoint_z
            + forward_matrix[3, 3]
        )
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            mapped_x = (
                forward_matrix[0, 0] * endpoint_x
                + forward_matrix[0, 1] * endpoint_y
                + forward_matrix[0, 2] * endpoint_z
                + forward_matrix[0, 3]
            ) / homogeneous
            mapped_y = (
                forward_matrix[1, 0] * endpoint_x
                + forward_matrix[1, 1] * endpoint_y
                + forward_matrix[1, 2] * endpoint_z
                + forward_matrix[1, 3]
            ) / homogeneous
            mapped_z = (
                forward_matrix[2, 0] * endpoint_x
                + forward_matrix[2, 1] * endpoint_y
                + forward_matrix[2, 2] * endpoint_z
                + forward_matrix[2, 3]
            ) / homogeneous
        result = np.stack((mapped_x - output_x, mapped_y - output_y, mapped_z - output_z), axis=-1)
        supported &= np.isfinite(result).all(axis=-1) & (homogeneous != 0)
        result[~supported] = np.nan
        return result.astype(np.float32, copy=False)

    result = transform_vector_components_3d(
        sampled,
        forward_matrix[:3, :3],
        semantics=semantics,
    )
    result[~supported] = np.nan
    return result


def crop_and_resize_vector_field_3d(
    field: NDArray[np.float16] | NDArray[np.float32],
    crop_coords: tuple[int, int, int, int, int, int],
    output_shape: tuple[int, int, int],
    *,
    semantics: str,
) -> NDArray[np.float32]:
    """Crop a 3D field before resizing so discarded voxels cannot enter interpolation support."""
    z_min, z_max, y_min, y_max, x_min, x_max = crop_coords
    cropped = field[z_min:z_max, y_min:y_max, x_min:x_max]
    crop_depth, crop_height, crop_width = cropped.shape[:3]
    output_depth, output_height, output_width = output_shape
    scale_x = output_width / crop_width
    scale_y = output_height / crop_height
    scale_z = output_depth / crop_depth
    matrix = np.asarray(
        (
            (scale_x, 0, 0, (scale_x - 1) / 2),
            (0, scale_y, 0, (scale_y - 1) / 2),
            (0, 0, scale_z, (scale_z - 1) / 2),
        ),
        dtype=np.float32,
    )
    return warp_vector_field_3d(cropped, matrix, output_shape, semantics=semantics)


def _evaluate_control_grid_float32(
    points: NDArray[np.float32],
    coefficients: NDArray[np.float32],
    image_shape: tuple[int, int],
) -> NDArray[np.float32]:
    """Evaluate one cubic control lattice with the float32 basis used by raster map generation."""
    rows, columns = coefficients.shape[:2]
    height, width = image_shape

    def axis_parameters(coordinates: NDArray[np.float32], image_length: int, count: int) -> tuple[Any, Any]:
        spans = count - 3
        normalized = np.zeros_like(coordinates, dtype=np.float32)
        if image_length > 1:
            np.multiply(coordinates, np.float32(spans / (image_length - 1)), out=normalized)
        finite = np.isfinite(coordinates) & np.isfinite(normalized)
        normalized = np.nan_to_num(
            normalized,
            copy=False,
            nan=0.0,
            posinf=float(count + 4),
            neginf=-4.0,
        )
        np.clip(normalized, np.float32(-4), np.float32(count + 4), out=normalized)
        base = np.floor(normalized).astype(np.intp)
        t = normalized - base.astype(np.float32)
        t_squared = t * t
        t_cubed = t_squared * t
        one_minus_t = np.float32(1) - t
        weights = np.stack(
            (
                one_minus_t * one_minus_t * one_minus_t / np.float32(6),
                (np.float32(3) * t_cubed - np.float32(6) * t_squared + np.float32(4)) / np.float32(6),
                (-np.float32(3) * t_cubed + np.float32(3) * t_squared + np.float32(3) * t + np.float32(1))
                / np.float32(6),
                t_cubed / np.float32(6),
            ),
            axis=-1,
        )
        indices = base[:, None] + np.arange(4, dtype=np.intp)[None, :]
        valid = (indices >= 0) & (indices < count) & finite[:, None]
        return np.clip(indices, 0, count - 1), weights * valid

    row_indices, row_weights = axis_parameters(points[:, 1], height, rows)
    column_indices, column_weights = axis_parameters(points[:, 0], width, columns)
    control = np.asarray(coefficients, dtype=np.float32)
    output = np.zeros((len(points), control.shape[-1]), dtype=np.float32)
    for row_tap in range(4):
        for column_tap in range(4):
            weight = row_weights[:, row_tap] * column_weights[:, column_tap]
            output += control[row_indices[:, row_tap], column_indices[:, column_tap]] * weight[:, None]
    return output


def _elastic_pull_displacement_points_3d(
    points: NDArray[np.float32],
    coefficients: Mapping[str, Any],
    volume_shape: tuple[int, int, int],
) -> NDArray[np.float32]:
    depth, height, width = volume_shape
    points = np.asarray(points, dtype=np.float32)
    displacement = np.zeros((len(points), 3), dtype=np.float32)
    displacement[:, :2] += _evaluate_control_grid_float32(
        points[:, :2],
        np.asarray(coefficients["xy"], dtype=np.float32),
        (height, width),
    )
    displacement[:, (0, 2)] += _evaluate_control_grid_float32(
        points[:, (0, 2)],
        np.asarray(coefficients["xz"], dtype=np.float32),
        (depth, width),
    )
    displacement[:, 1:3] += _evaluate_control_grid_float32(
        points[:, 1:3],
        np.asarray(coefficients["yz"], dtype=np.float32),
        (depth, height),
    )
    return displacement / np.float32(3)


def _inverse_elastic_pull_points_3d(
    source_points: NDArray[np.float32],
    coefficients: Mapping[str, Any],
    volume_shape: tuple[int, int, int],
    *,
    max_iterations: int,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    transformed_points = np.asarray(source_points, dtype=np.float32).copy()
    finite_points = np.isfinite(transformed_points).all(axis=-1)
    converged = np.zeros(len(transformed_points), dtype=bool)
    for _ in range(max_iterations):
        pull_displacement = _elastic_pull_displacement_points_3d(
            transformed_points,
            coefficients,
            volume_shape,
        )
        updated = source_points - pull_displacement
        residual = np.max(np.abs(updated - transformed_points), axis=-1)
        converged |= finite_points & np.isfinite(residual) & (residual <= np.float32(1e-3))
        active = finite_points & ~converged & np.isfinite(updated).all(axis=-1)
        transformed_points[active] = updated[active]
        if not np.any(active):
            break

    final_pull = _elastic_pull_displacement_points_3d(transformed_points, coefficients, volume_shape)
    residual = np.max(np.abs(transformed_points + final_pull - source_points), axis=-1)
    converged &= np.isfinite(residual) & (residual <= np.float32(1e-3))
    return transformed_points, converged


_PullJacobian3D = tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]


def _coordinate_derivative_3d(
    values: NDArray[np.float32],
    axis: int,
    identity_axis: int,
) -> NDArray[np.float32]:
    if values.shape[axis] == 1:
        return np.full(values.shape, 1 if axis == identity_axis else 0, dtype=np.float32)
    return np.gradient(values, axis=axis).astype(np.float32, copy=False)


def _pull_coordinates_and_jacobian_3d(
    sampling_grid: NDArray[np.float32],
    block_start: int,
    block_end: int,
    source_shape: tuple[int, int, int],
) -> tuple[NDArray[np.float32], _PullJacobian3D, NDArray[np.float32]]:
    output_depth = sampling_grid.shape[0]
    source_depth, source_height, source_width = source_shape
    halo_start = max(0, block_start - 1)
    halo_end = min(output_depth, block_end + 1)
    halo_grid = sampling_grid[halo_start:halo_end]
    halo_x = np.asarray((halo_grid[..., 0] + 1) * (source_width / 2) - 0.5, dtype=np.float32)
    halo_y = np.asarray((halo_grid[..., 1] + 1) * (source_height / 2) - 0.5, dtype=np.float32)
    halo_z = np.asarray((halo_grid[..., 2] + 1) * (source_depth / 2) - 0.5, dtype=np.float32)
    halo_slice = slice(block_start - halo_start, block_end - halo_start)
    source_x, source_y, source_z = halo_x[halo_slice], halo_y[halo_slice], halo_z[halo_slice]
    jacobian: _PullJacobian3D = (
        _coordinate_derivative_3d(source_x, 2, 2),
        _coordinate_derivative_3d(source_x, 1, 2),
        _coordinate_derivative_3d(halo_x, 0, 2)[halo_slice],
        _coordinate_derivative_3d(source_y, 2, 1),
        _coordinate_derivative_3d(source_y, 1, 1),
        _coordinate_derivative_3d(halo_y, 0, 1)[halo_slice],
        _coordinate_derivative_3d(source_z, 2, 0),
        _coordinate_derivative_3d(source_z, 1, 0),
        _coordinate_derivative_3d(halo_z, 0, 0)[halo_slice],
    )
    a, b, c, d, e, f, g, h, i = jacobian
    determinant = np.asarray(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g), dtype=np.float32)
    source_coordinates = np.asarray(np.stack((source_x, source_y, source_z), axis=-1), dtype=np.float32)
    return source_coordinates, jacobian, determinant


def _transform_pull_field_block_3d(
    sampled: NDArray[np.float32],
    jacobian: _PullJacobian3D,
    determinant: NDArray[np.float32],
    supported: NDArray[np.bool_],
    semantics: str,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    a, b, c, d, e, f, g, h, i = jacobian
    valid = np.asarray(supported & np.isfinite(determinant) & (determinant > np.float32(1e-8)), dtype=np.bool_)
    gx, gy, gz = (sampled[..., index] for index in range(3))
    if semantics in {"gradient3d", "normal3d"}:
        result = np.stack((a * gx + d * gy + g * gz, b * gx + e * gy + h * gz, c * gx + f * gy + i * gz), axis=-1)
    else:
        cofactors = (
            e * i - f * h,
            c * h - b * i,
            b * f - c * e,
            f * g - d * i,
            a * i - c * g,
            c * d - a * f,
            d * h - e * g,
            b * g - a * h,
            a * e - b * d,
        )
        c00, c01, c02, c10, c11, c12, c20, c21, c22 = cofactors
        result = np.stack(
            (c00 * gx + c01 * gy + c02 * gz, c10 * gx + c11 * gy + c12 * gz, c20 * gx + c21 * gy + c22 * gz),
            axis=-1,
        )
        np.divide(result, determinant[..., None], out=result, where=valid[..., None])
    if semantics == "normal3d":
        lengths = np.linalg.norm(result, axis=-1)
        normal_valid = np.isfinite(lengths) & (lengths > 0)
        np.divide(result, lengths[..., None], out=result, where=normal_valid[..., None])
        valid &= normal_valid
    if semantics != "displacement3d":
        result[~valid] = np.nan
    return np.asarray(result, dtype=np.float32), valid


def _invert_displacement_block_3d(
    result: NDArray[np.float32],
    valid: NDArray[np.bool_],
    source_coordinates: NDArray[np.float32],
    sampled: NDArray[np.float32],
    normalized_grid: NDArray[np.float32],
    block_start: int,
    inverse_iterations: int,
    inverse_model: Mapping[str, Any] | None,
    source_shape: tuple[int, int, int],
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    output_depth, output_height, output_width = normalized_grid.shape[:3]
    output_z, output_y, output_x = np.mgrid[
        block_start : block_start + result.shape[0],
        :output_height,
        :output_width,
    ].astype(np.float32)
    target_x = source_coordinates[..., 0] + sampled[..., 0]
    target_y = source_coordinates[..., 1] + sampled[..., 1]
    target_z = source_coordinates[..., 2] + sampled[..., 2]
    query_x = output_x + result[..., 0]
    query_y = output_y + result[..., 1]
    query_z = output_z + result[..., 2]
    converged = np.zeros(valid.shape, dtype=bool)
    model_resolved = np.zeros(valid.shape, dtype=bool)

    for _ in range(inverse_iterations):
        endpoint_grid, endpoint_domain = _coordinates_to_grid_3d(
            query_x,
            query_y,
            query_z,
            (output_depth, output_height, output_width),
        )
        sampled_map, pull_supported = resample_field_3d(normalized_grid, endpoint_grid)
        endpoint_x = (sampled_map[..., 0] + 1) * (source_shape[2] / 2) - 0.5
        endpoint_y = (sampled_map[..., 1] + 1) * (source_shape[1] / 2) - 0.5
        endpoint_z = (sampled_map[..., 2] + 1) * (source_shape[0] / 2) - 0.5
        residual_x, residual_y, residual_z = target_x - endpoint_x, target_y - endpoint_y, target_z - endpoint_z
        residual = np.sqrt(residual_x * residual_x + residual_y * residual_y + residual_z * residual_z)
        converged |= np.isfinite(residual) & (residual <= np.float32(1e-3))
        active = valid & endpoint_domain & pull_supported & ~converged
        query_x = np.where(active, residual_x + query_x, query_x)
        query_y = np.where(active, residual_y + query_y, query_y)
        query_z = np.where(active, residual_z + query_z, query_z)
        if not active.any():
            break

    if inverse_model is not None:
        fallback = valid & ~converged & np.isfinite(target_x) & np.isfinite(target_y) & np.isfinite(target_z)
        if fallback.any():
            endpoints = np.stack((target_x[fallback], target_y[fallback], target_z[fallback]), axis=-1)
            mapped, mapped_valid = _inverse_elastic_pull_points_3d(
                endpoints,
                inverse_model,
                source_shape,
                max_iterations=inverse_iterations,
            )
            query_x[fallback], query_y[fallback], query_z[fallback] = mapped.T
            converged[fallback] = mapped_valid
            model_resolved[fallback] = mapped_valid

    inside_output = (
        (query_x >= -0.5)
        & (query_x < output_width - 0.5)
        & (query_y >= -0.5)
        & (query_y < output_height - 0.5)
        & (query_z >= -0.5)
        & (query_z < output_depth - 0.5)
    )
    valid &= converged & (inside_output | model_resolved)
    result[..., 0] = query_x - output_x
    result[..., 1] = query_y - output_y
    result[..., 2] = query_z - output_z
    result[~valid] = np.nan
    return result, valid


def warp_vector_field_from_pull_grid_3d(
    field: NDArray[np.float16] | NDArray[np.float32],
    sampling_grid: NDArray[np.float32],
    *,
    semantics: str,
    inverse_iterations: int = 32,
    inverse_model: Mapping[str, Any] | None = None,
) -> NDArray[np.float32]:
    """Transform a 3D field through a dense normalized pull map in bounded depth slabs."""
    normalized_grid = np.asarray(sampling_grid, dtype=np.float32)
    source_shape = field.shape[:3]
    output_depth = normalized_grid.shape[0]
    result = np.full((*normalized_grid.shape[:3], 3), np.nan, dtype=np.float32)

    for block_start in range(0, output_depth, 8):
        block_end = min(output_depth, block_start + 8)
        source_coordinates, jacobian, determinant = _pull_coordinates_and_jacobian_3d(
            normalized_grid,
            block_start,
            block_end,
            source_shape,
        )
        sampled, supported = resample_field_3d(field, normalized_grid[block_start:block_end])
        block_result, valid = _transform_pull_field_block_3d(sampled, jacobian, determinant, supported, semantics)
        if semantics == "displacement3d":
            block_result, valid = _invert_displacement_block_3d(
                block_result,
                valid,
                source_coordinates,
                sampled,
                normalized_grid,
                block_start,
                inverse_iterations,
                inverse_model,
                source_shape,
            )
        result[block_start:block_end] = block_result
    return result


def _project_points_2d(
    matrix: NDArray[np.float32],
    x_coordinates: NDArray[np.float32],
    y_coordinates: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
    """Project pixel-coordinate arrays through an affine or perspective forward matrix."""
    forward_matrix = np.asarray(matrix, dtype=np.float32)
    if forward_matrix.shape == (2, 3):
        output_x = forward_matrix[0, 0] * x_coordinates + forward_matrix[0, 1] * y_coordinates + forward_matrix[0, 2]
        output_y = forward_matrix[1, 0] * x_coordinates + forward_matrix[1, 1] * y_coordinates + forward_matrix[1, 2]
        finite = np.isfinite(output_x) & np.isfinite(output_y)
    else:
        denominator = forward_matrix[2, 0] * x_coordinates + forward_matrix[2, 1] * y_coordinates + forward_matrix[2, 2]
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            output_x = (
                forward_matrix[0, 0] * x_coordinates + forward_matrix[0, 1] * y_coordinates + forward_matrix[0, 2]
            ) / denominator
            output_y = (
                forward_matrix[1, 0] * x_coordinates + forward_matrix[1, 1] * y_coordinates + forward_matrix[1, 2]
            ) / denominator
        finite = np.isfinite(output_x) & np.isfinite(output_y) & (denominator != 0)
    return (
        np.asarray(output_x, dtype=np.float32),
        np.asarray(output_y, dtype=np.float32),
        finite,
    )


def warp_vector_field_2d(
    field: NDArray[np.float16] | NDArray[np.float32],
    matrix: NDArray[np.float32],
    output_shape: tuple[int, int],
    *,
    semantics: str,
) -> NDArray[np.float32]:
    """Resample a vector field and apply the matrix's differential or endpoint mapping."""
    if field.ndim == 4:
        return np.stack(
            [warp_vector_field_2d(field[z], matrix, output_shape, semantics=semantics) for z in range(field.shape[0])],
            axis=0,
        )
    source_shape = field.shape[:2]
    map_x, map_y, source_domain = pull_map_from_matrix_2d(matrix, output_shape, source_shape)
    sampled = resample_field_2d(field, map_x, map_y)
    output_y = np.arange(output_shape[0], dtype=np.float32)[:, None]
    output_x = np.arange(output_shape[1], dtype=np.float32)[None, :]

    if semantics in {"displacement", "displacement3d"}:
        endpoint_x, endpoint_y, endpoint_finite = _project_points_2d(
            matrix,
            map_x + sampled[..., 0],
            map_y + sampled[..., 1],
        )
        result = sampled.copy()
        result[..., 0] = endpoint_x - output_x
        result[..., 1] = endpoint_y - output_y
        if semantics == "displacement3d":
            result[..., 2] = sampled[..., 2]
        result[~(source_domain & endpoint_finite)] = np.nan
        return result

    if matrix.shape == (2, 3):
        lifted_to_3d = semantics in {"gradient3d", "vector_field3d", "normal", "normal3d"}
        forward_jacobian = np.eye(3 if lifted_to_3d else 2, dtype=np.float32)
        forward_jacobian[:2, :2] = matrix[:2, :2]
        transformed = (
            transform_vector_components_3d(
                sampled,
                forward_jacobian,
                semantics=semantics,
            )
            if semantics in {"gradient3d", "vector_field3d", "normal3d"}
            else transform_vector_components_2d(
                sampled,
                matrix[:2, :2],
                semantics=semantics,
            )
        )
        transformed[~source_domain] = np.nan
        return transformed

    m = np.asarray(matrix, dtype=np.float32)
    denominator = m[2, 0] * map_x + m[2, 1] * map_y + m[2, 2]
    projected_x = m[0, 0] * map_x + m[0, 1] * map_y + m[0, 2]
    projected_y = m[1, 0] * map_x + m[1, 1] * map_y + m[1, 2]
    denominator_squared = denominator * denominator
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        jacobian_xx = (m[0, 0] * denominator - projected_x * m[2, 0]) / denominator_squared
        jacobian_xy = (m[0, 1] * denominator - projected_x * m[2, 1]) / denominator_squared
        jacobian_yx = (m[1, 0] * denominator - projected_y * m[2, 0]) / denominator_squared
        jacobian_yy = (m[1, 1] * denominator - projected_y * m[2, 1]) / denominator_squared
    jacobian_determinant = jacobian_xx * jacobian_yy - jacobian_xy * jacobian_yx
    local_valid = source_domain & np.isfinite(jacobian_determinant) & (jacobian_determinant != 0)
    if semantics in {"gradient", "gradient3d", "normal", "normal3d"}:
        transformed_x = (jacobian_yy * sampled[..., 0] - jacobian_yx * sampled[..., 1]) / jacobian_determinant
        transformed_y = (-jacobian_xy * sampled[..., 0] + jacobian_xx * sampled[..., 1]) / jacobian_determinant
    else:
        transformed_x = jacobian_xx * sampled[..., 0] + jacobian_xy * sampled[..., 1]
        transformed_y = jacobian_yx * sampled[..., 0] + jacobian_yy * sampled[..., 1]
    result = sampled.copy()
    result[..., 0] = transformed_x
    result[..., 1] = transformed_y
    if semantics in {"normal", "normal3d"}:
        lengths = np.linalg.norm(result, axis=-1)
        normal_valid = np.isfinite(lengths) & (lengths > 0)
        np.divide(result, lengths[..., None], out=result, where=normal_valid[..., None])
        local_valid &= normal_valid
    result[~local_valid] = np.nan
    return result


def crop_vector_field_2d(
    field: NDArray[np.float16] | NDArray[np.float32],
    crop_coords: tuple[int, int, int, int],
) -> NDArray[np.float16] | NDArray[np.float32]:
    """Return the 2D crop view for a single field or a depth stack of fields."""
    x_min, y_min, x_max, y_max = crop_coords
    if field.ndim == 4:
        return field[:, y_min:y_max, x_min:x_max]
    return field[y_min:y_max, x_min:x_max]


def crop_and_resize_vector_field_2d(
    field: NDArray[np.float16] | NDArray[np.float32],
    crop_coords: tuple[int, int, int, int],
    output_shape: tuple[int, int],
    *,
    semantics: str,
) -> NDArray[np.float32]:
    """Crop a 2D field before resizing so discarded pixels cannot enter interpolation support."""
    x_min, y_min, x_max, y_max = crop_coords
    cropped = crop_vector_field_2d(field, crop_coords)
    output_height, output_width = output_shape
    scale_x = output_width / (x_max - x_min)
    scale_y = output_height / (y_max - y_min)
    matrix = np.asarray(
        (
            (scale_x, 0, (scale_x - 1) / 2),
            (0, scale_y, (scale_y - 1) / 2),
        ),
        dtype=np.float32,
    )
    return warp_vector_field_2d(cropped, matrix, output_shape, semantics=semantics)


def _pull_jacobian_2d(
    map_x: NDArray[np.float32],
    map_y: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Differentiate a pixel-coordinate pull map along output X and Y."""
    if min(map_x.shape) < 2:
        invalid = np.full(map_x.shape, np.nan, dtype=np.float32)
        return invalid, invalid.copy(), invalid.copy(), invalid.copy()
    map_x_dy, map_x_dx = np.gradient(map_x.astype(np.float32, copy=False))
    map_y_dy, map_y_dx = np.gradient(map_y.astype(np.float32, copy=False))
    return (
        np.asarray(map_x_dx, dtype=np.float32),
        np.asarray(map_x_dy, dtype=np.float32),
        np.asarray(map_y_dx, dtype=np.float32),
        np.asarray(map_y_dy, dtype=np.float32),
    )


def _map_sample_2d(
    pull_map: NDArray[np.float32],
    points_x: NDArray[np.float32],
    points_y: NDArray[np.float32],
) -> NDArray[np.float32]:
    return resample_field_2d(pull_map, points_x, points_y)


def _inverse_elastic_pull_points_2d(
    source_points: NDArray[np.float32],
    coefficients: NDArray[np.float32],
    image_shape: tuple[int, int],
    *,
    max_iterations: int = 32,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    transformed = np.asarray(source_points, dtype=np.float32).copy()
    finite = np.isfinite(transformed).all(axis=-1)
    converged = np.zeros(len(transformed), dtype=bool)
    for _ in range(max_iterations):
        pull_displacement = _evaluate_control_grid_float32(transformed, coefficients, image_shape)
        updated = source_points - pull_displacement
        residual = np.max(np.abs(updated - transformed), axis=-1)
        converged |= finite & np.isfinite(residual) & (residual <= np.float32(1e-3))
        active = finite & ~converged & np.isfinite(updated).all(axis=-1)
        transformed[active] = updated[active]
        if not active.any():
            break

    final_displacement = _evaluate_control_grid_float32(transformed, coefficients, image_shape)
    residual = np.max(np.abs(transformed + final_displacement - source_points), axis=-1)
    converged &= np.isfinite(residual) & (residual <= np.float32(1e-3))
    return transformed, converged


def _newton_inverse_pull_map_2d(
    target_x: NDArray[np.float32],
    target_y: NDArray[np.float32],
    query_x: NDArray[np.float32],
    query_y: NDArray[np.float32],
    valid: NDArray[np.bool_],
    pull_map: NDArray[np.float32],
    jacobian_map: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_]]:
    converged = np.zeros(query_x.shape, dtype=bool)
    for _ in range(12):
        estimated_source = _map_sample_2d(pull_map, query_x, query_y)
        residual_x = target_x - estimated_source[..., 0]
        residual_y = target_y - estimated_source[..., 1]
        residual_norm = np.hypot(residual_x, residual_y)
        converged |= np.isfinite(residual_norm) & (residual_norm <= np.float32(1e-3))
        active = valid & ~converged & np.isfinite(residual_norm)
        if not np.any(active):
            break
        jacobian = _map_sample_2d(jacobian_map, query_x, query_y)
        a, b, c, d = (jacobian[..., index] for index in range(4))
        determinant = a * d - b * c
        stable = np.isfinite(determinant) & (np.abs(determinant) > np.float32(1e-8))
        update_x = np.zeros_like(query_x)
        update_y = np.zeros_like(query_y)
        np.divide(d * residual_x - b * residual_y, determinant, out=update_x, where=stable)
        np.divide(-c * residual_x + a * residual_y, determinant, out=update_y, where=stable)
        valid &= stable | converged
        query_x += np.where(active, update_x, np.float32(0))
        query_y += np.where(active, update_y, np.float32(0))
    return query_x, query_y, valid, converged


def _resolve_elastic_pull_endpoints_2d(
    target_x: NDArray[np.float32],
    target_y: NDArray[np.float32],
    query_x: NDArray[np.float32],
    query_y: NDArray[np.float32],
    valid: NDArray[np.bool_],
    converged: NDArray[np.bool_],
    map_shape: tuple[int, int],
    inverse_model: NDArray[np.float32] | None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
    inside_map = (query_x >= -0.5) & (query_x < map_shape[1] - 0.5) & (query_y >= -0.5) & (query_y < map_shape[0] - 0.5)
    model_resolved = np.zeros(valid.shape, dtype=bool)
    if inverse_model is not None:
        fallback = valid & (~converged | ~inside_map)
        if fallback.any():
            endpoints = np.stack((target_x[fallback], target_y[fallback]), axis=-1)
            mapped_points, mapped_valid = _inverse_elastic_pull_points_2d(endpoints, inverse_model, map_shape)
            query_x[fallback] = mapped_points[:, 0]
            query_y[fallback] = mapped_points[:, 1]
            converged[fallback] = mapped_valid
            model_resolved[fallback] = mapped_valid
    return query_x, query_y, converged & (inside_map | model_resolved)


def _push_displacements_through_pull_map_2d(
    sampled: NDArray[np.float32],
    map_x: NDArray[np.float32],
    map_y: NDArray[np.float32],
    source_x: NDArray[np.float32],
    source_y: NDArray[np.float32],
    semantics: str,
    inverse_model: NDArray[np.float32] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Invert a smooth pull map at finite-displacement endpoints with bounded Newton steps."""
    dsource_x = sampled[..., 0]
    dsource_y = sampled[..., 1]
    map_dx_dx, map_dx_dy, map_dy_dx, map_dy_dy = _pull_jacobian_2d(map_x, map_y)
    determinant = map_dx_dx * map_dy_dy - map_dx_dy * map_dy_dx
    valid = np.isfinite(determinant) & (np.abs(determinant) > np.float32(1e-8))
    valid &= np.isfinite(sampled).all(axis=-1)
    target_x = source_x + dsource_x
    target_y = source_y + dsource_y
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        output_dx = (map_dy_dy * dsource_x - map_dx_dy * dsource_y) / determinant
        output_dy = (-map_dy_dx * dsource_x + map_dx_dx * dsource_y) / determinant
    query_x = np.asarray(
        np.broadcast_to(np.arange(map_x.shape[1], dtype=np.float32)[None, :], map_x.shape) + output_dx,
        dtype=np.float32,
    )
    query_y = np.asarray(
        np.broadcast_to(np.arange(map_x.shape[0], dtype=np.float32)[:, None], map_x.shape) + output_dy,
        dtype=np.float32,
    )
    pull_map = np.stack((map_x, map_y), axis=-1)
    jacobian_map = np.stack((map_dx_dx, map_dx_dy, map_dy_dx, map_dy_dy), axis=-1)
    query_x, query_y, valid, converged = _newton_inverse_pull_map_2d(
        target_x,
        target_y,
        query_x,
        query_y,
        valid,
        pull_map,
        jacobian_map,
    )
    output_y = np.arange(map_x.shape[0], dtype=np.float32)[:, None]
    output_x = np.arange(map_x.shape[1], dtype=np.float32)[None, :]
    output = np.full((*map_x.shape, sampled.shape[-1]), np.nan, dtype=np.float32)
    query_x, query_y, valid = _resolve_elastic_pull_endpoints_2d(
        target_x,
        target_y,
        query_x,
        query_y,
        valid,
        converged,
        map_x.shape,
        inverse_model,
    )
    valid &= np.isfinite(source_x) & np.isfinite(source_y)
    output[..., 0] = query_x - output_x
    output[..., 1] = query_y - output_y
    if semantics == "displacement3d":
        output[..., 2] = sampled[..., 2]
    output[~valid] = np.nan
    return output, valid


def warp_vector_field_from_pull_map_2d(
    field: NDArray[np.float16] | NDArray[np.float32],
    map_x: NDArray[np.float32],
    map_y: NDArray[np.float32],
    *,
    semantics: str,
    inverse_model: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Resample fields through a pixel-coordinate pull map and apply its local differential."""
    if field.ndim == 4:
        return np.stack(
            [
                warp_vector_field_from_pull_map_2d(
                    field[z],
                    map_x,
                    map_y,
                    semantics=semantics,
                    inverse_model=inverse_model,
                )
                for z in range(field.shape[0])
            ],
            axis=0,
        )

    map_x = np.asarray(map_x, dtype=np.float32)
    map_y = np.asarray(map_y, dtype=np.float32)
    sampled = resample_field_2d(field, map_x, map_y)
    map_dx_dx, map_dx_dy, map_dy_dx, map_dy_dy = _pull_jacobian_2d(map_x, map_y)
    determinant = map_dx_dx * map_dy_dy - map_dx_dy * map_dy_dx
    valid = np.isfinite(determinant) & (np.abs(determinant) > np.float32(1e-8))
    valid &= np.isfinite(sampled).all(axis=-1)

    if semantics in {"displacement", "displacement3d"}:
        result, endpoint_valid = _push_displacements_through_pull_map_2d(
            sampled,
            map_x,
            map_y,
            map_x,
            map_y,
            semantics,
            inverse_model,
        )
        result[~valid | ~endpoint_valid] = np.nan
        return result

    result = sampled.copy()
    source_float = sampled.astype(np.float32, copy=False)
    if semantics in {"gradient", "gradient3d", "normal", "normal3d"}:
        result[..., 0] = map_dx_dx * source_float[..., 0] + map_dy_dx * source_float[..., 1]
        result[..., 1] = map_dx_dy * source_float[..., 0] + map_dy_dy * source_float[..., 1]
    else:
        result[..., 0] = (map_dy_dy * source_float[..., 0] - map_dx_dy * source_float[..., 1]) / determinant
        result[..., 1] = (-map_dy_dx * source_float[..., 0] + map_dx_dx * source_float[..., 1]) / determinant

    if semantics in {"normal", "normal3d"}:
        lengths = np.linalg.norm(result, axis=-1)
        normal_valid = np.isfinite(lengths) & (lengths > 0)
        np.divide(result, lengths[..., None], out=result, where=normal_valid[..., None])
        valid &= normal_valid
    valid &= np.isfinite(result).all(axis=-1)
    result[~valid] = np.nan
    return result


def restore_vector_field_dtype(
    field: NDArray[np.float32],
    dtype: np.dtype[np.float16] | np.dtype[np.float32],
) -> NDArray[np.float16] | NDArray[np.float32]:
    """Round a computed field back to its input dtype, marking overflowed vectors invalid."""
    with np.errstate(over="ignore", invalid="ignore"):
        restored = field.astype(dtype, copy=False)
    restored_valid = np.isfinite(restored).all(axis=-1)
    restored[~restored_valid] = np.nan
    return cast("NDArray[np.float16] | NDArray[np.float32]", restored)


def transform_vector_components_2d(
    field: NDArray[np.float16] | NDArray[np.float32],
    matrix: NDArray[np.float32],
    *,
    semantics: str,
) -> NDArray[np.float32]:
    """Apply a spatial linear map to 2D gradients, vectors, displacements, or lifted normals."""
    component_count = field.shape[-1]
    expected_components = 3 if semantics == "normal" else 2
    if component_count != expected_components or matrix.shape != (2, 2):
        return np.full(field.shape, np.nan, dtype=np.float32)
    if field.size == 0:
        return field.astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        return np.full(field.shape, np.nan, dtype=np.float32)

    matrix_float = np.asarray(matrix, dtype=np.float32)
    component_matrix = _component_matrix_2d(matrix_float, semantics)
    if component_matrix is None:
        return np.full(field.shape, np.nan, dtype=np.float32)

    valid_vectors = np.asarray(np.isfinite(field).all(axis=-1), dtype=np.bool_)
    source_float = field.astype(np.float32, copy=False)
    safe_field = np.where(valid_vectors[..., None], source_float, np.float32(0))
    result = np.einsum("ij,...j->...i", component_matrix, safe_field, optimize=True)
    if semantics == "normal":
        lengths = np.linalg.norm(result, axis=-1)
        normal_valid = np.isfinite(lengths) & (lengths > 0)
        np.divide(result, lengths[..., None], out=result, where=normal_valid[..., None])
        valid_vectors &= normal_valid
    else:
        valid_vectors &= np.isfinite(result).all(axis=-1)
    result[~valid_vectors] = np.nan
    return result


def _component_matrix_2d(matrix: NDArray[np.float32], semantics: str) -> NDArray[np.float32] | None:
    if semantics == "gradient":
        determinant = np.linalg.det(matrix)
        return np.linalg.inv(matrix).T if np.isfinite(determinant) and determinant != 0 else None
    if semantics in {"vector_field", "displacement"}:
        return matrix
    if semantics == "normal":
        lifted_matrix = np.eye(3, dtype=np.float32)
        lifted_matrix[:2, :2] = matrix
        determinant = np.linalg.det(lifted_matrix)
        return np.linalg.inv(lifted_matrix).T if np.isfinite(determinant) and determinant != 0 else None
    return None


def transform_vector_components_3d(
    field: NDArray[np.float16] | NDArray[np.float32],
    matrix: NDArray[np.float32],
    *,
    semantics: str,
) -> NDArray[np.float32]:
    """Apply a 3D linear map to gradients, local vectors, displacements, or normals."""
    if field.shape[-1] != 3 or matrix.shape != (3, 3):
        return np.full(field.shape, np.nan, dtype=np.float32)
    if field.size == 0:
        return field.astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        return np.full(field.shape, np.nan, dtype=np.float32)

    matrix_float = np.asarray(matrix, dtype=np.float32)
    if semantics in {"gradient3d", "normal3d"}:
        determinant = np.linalg.det(matrix_float)
        if not np.isfinite(determinant) or determinant == 0:
            return np.full(field.shape, np.nan, dtype=np.float32)
        component_matrix = np.linalg.inv(matrix_float).T
    elif semantics in {"vector_field3d", "displacement3d"}:
        component_matrix = matrix_float
    else:
        return np.full(field.shape, np.nan, dtype=np.float32)

    valid_vectors = np.asarray(np.isfinite(field).all(axis=-1), dtype=np.bool_)
    source_float = field.astype(np.float32, copy=False)
    safe_field = np.where(valid_vectors[..., None], source_float, np.float32(0))
    result = np.einsum("ij,...j->...i", component_matrix, safe_field, optimize=True)
    if semantics == "normal3d":
        lengths = np.linalg.norm(result, axis=-1)
        normal_valid = np.isfinite(lengths) & (lengths > 0)
        np.divide(result, lengths[..., None], out=result, where=normal_valid[..., None])
        valid_vectors &= normal_valid
    else:
        valid_vectors &= np.isfinite(result).all(axis=-1)
    result[~valid_vectors] = np.nan
    return result


_D4_COMPONENT_MATRICES = {
    "e": np.eye(2, dtype=np.float32),
    "r90": np.asarray(((0, 1), (-1, 0)), dtype=np.float32),
    "r180": np.asarray(((-1, 0), (0, -1)), dtype=np.float32),
    "r270": np.asarray(((0, -1), (1, 0)), dtype=np.float32),
    "v": np.asarray(((1, 0), (0, -1)), dtype=np.float32),
    "hvt": np.asarray(((0, -1), (-1, 0)), dtype=np.float32),
    "h": np.asarray(((-1, 0), (0, 1)), dtype=np.float32),
    "t": np.asarray(((0, 1), (1, 0)), dtype=np.float32),
}


def transform_vector_field_symmetry(field: NDArray[np.generic], target: str, group_element: str) -> NDArray[np.float32]:
    """Apply one exact D4 spatial permutation and its corresponding vector-basis map."""
    if group_element == "e":
        return field.astype(np.float32, copy=False)
    if field.ndim == 4:
        component_matrix = np.eye(3, dtype=np.float32)
        component_matrix[:2, :2] = _D4_COMPONENT_MATRICES[group_element]
        transformed = _d4_reindex_3d(field, group_element)
        return transform_vector_components_3d(
            cast("NDArray[np.float16] | NDArray[np.float32]", transformed),
            component_matrix,
            semantics=target,
        )

    transformed = _d4_reindex_2d(field, group_element)
    return transform_vector_components_2d(
        cast("NDArray[np.float16] | NDArray[np.float32]", transformed),
        _D4_COMPONENT_MATRICES[group_element],
        semantics=target,
    )


def _d4_reindex_2d(field: NDArray[np.generic], group_element: str) -> NDArray[np.generic]:
    if group_element in {"r90", "r180", "r270"}:
        return np.rot90(field, k={"r90": 1, "r180": 2, "r270": 3}[group_element], axes=(0, 1))
    if group_element == "v":
        return np.flip(field, axis=0)
    if group_element == "h":
        return np.flip(field, axis=1)
    if group_element == "t":
        return np.transpose(field, (1, 0, 2))
    return np.transpose(np.rot90(field, k=2, axes=(0, 1)), (1, 0, 2))


def _d4_reindex_3d(field: NDArray[np.generic], group_element: str) -> NDArray[np.generic]:
    spatial_axes = (1, 2)
    if group_element in {"r90", "r180", "r270"}:
        return np.rot90(field, k={"r90": 1, "r180": 2, "r270": 3}[group_element], axes=spatial_axes)
    if group_element == "v":
        return np.flip(field, axis=1)
    if group_element == "h":
        return np.flip(field, axis=2)
    if group_element == "t":
        return np.transpose(field, (0, 2, 1, 3))
    return np.transpose(np.rot90(field, k=2, axes=spatial_axes), (0, 2, 1, 3))


def permute_vector_field_components_2d(
    field: NDArray[np.float16] | NDArray[np.float32],
    *,
    spatial_axes: tuple[int, int] | None = None,
    quarter_turns: int = 0,
    component_matrix: NDArray[np.float32],
    semantics: str,
) -> NDArray[np.float32]:
    """Exactly reindex a 2D field and reorient its components for an orthogonal symmetry."""
    transformed: NDArray[np.generic] = field
    if spatial_axes is not None:
        transformed = np.flip(field, axis=spatial_axes).copy()
    if quarter_turns % 4:
        transformed = np.rot90(transformed, k=quarter_turns % 4, axes=(0, 1))
    return transform_vector_components_2d(
        cast("NDArray[np.float16] | NDArray[np.float32]", transformed),
        component_matrix,
        semantics=semantics,
    )


def permute_vector_field_components_3d(
    field: NDArray[np.float16] | NDArray[np.float32],
    *,
    spatial_axes: tuple[int, ...] | None = None,
    plane_axes: tuple[int, int] | None = None,
    quarter_turns: int = 0,
    spatial_permutation: tuple[int, int, int] | None = None,
    component_matrix: NDArray[np.float32],
    semantics: str,
) -> NDArray[np.float32]:
    """Exactly reindex a 3D field and reorient its components for an orthogonal symmetry."""
    transformed: NDArray[np.generic] = field
    if spatial_permutation is not None:
        transformed = np.transpose(field, (*spatial_permutation, 3))
    if spatial_axes:
        transformed = np.flip(transformed, axis=spatial_axes).copy()
    if plane_axes is not None and quarter_turns % 4:
        transformed = np.rot90(transformed, k=quarter_turns % 4, axes=plane_axes)
    return transform_vector_components_3d(
        cast("NDArray[np.float16] | NDArray[np.float32]", transformed),
        component_matrix,
        semantics=semantics,
    )
