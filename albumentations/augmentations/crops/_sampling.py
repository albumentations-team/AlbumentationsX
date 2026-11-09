"""Rejection-free geometry samplers for resized crops."""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations, pairwise
from math import exp, log, sqrt, ulp
from typing import Literal, Protocol

import numpy as np

SamplingMethod = Literal["standard", "uniform_scale"]


class _RandomSource(Protocol):
    def random(self) -> float: ...


_LOWER_EDGE = 0.5
_EPSILON = 1e-12


def _line_intersection(first: tuple[float, float], second: tuple[float, float]) -> float | None:
    first_slope, first_intercept = first
    second_slope, second_intercept = second
    denominator = first_slope - second_slope
    if abs(denominator) <= _EPSILON:
        return None
    return (second_intercept - first_intercept) / denominator


def _integral_linear_exp(slope: float, intercept: float, value: float) -> float:
    return exp(value) * (slope * (value - 1.0) + intercept)


def _integral_quadratic_exp(quadratic: float, linear: float, constant: float, value: float) -> float:
    return exp(value) * (quadratic * (value * value - 2.0 * value + 2.0) + linear * (value - 1.0) + constant)


@lru_cache(maxsize=512)
def _standard_2d_segments(
    height: int,
    width: int,
    scale_min: float,
    scale_max: float,
    ratio_min: float,
    ratio_max: float,
) -> tuple[tuple[float, float, float, float, float, float], ...]:
    """Return cached area-CDF segments for the unconstrained standard sampler."""
    return _build_standard_2d_segments(
        height,
        width,
        scale_min,
        scale_max,
        ratio_min,
        ratio_max,
        _LOWER_EDGE,
        _LOWER_EDGE,
    )


def _build_standard_2d_segments(
    height: int,
    width: int,
    scale_min: float,
    scale_max: float,
    ratio_min: float,
    ratio_max: float,
    minimum_height: float,
    minimum_width: float,
) -> tuple[tuple[float, float, float, float, float, float], ...]:
    """Build area-CDF segments for the requested crop and minimum-size bounds."""
    upper_height = height + _LOWER_EDGE
    upper_width = width + _LOWER_EDGE
    area_min, area_max = _crop_area_bounds(
        height,
        width,
        (scale_min, scale_max),
        (ratio_min, ratio_max),
        minimum_height,
        minimum_width,
    )
    if area_min > area_max:
        return ()

    x_min = log(area_min)
    x_max = log(area_max)
    if x_max - x_min <= _EPSILON:
        return ()

    lower_lines = (
        (0.0, log(ratio_min)),
        (1.0, -2.0 * log(upper_height)),
        (-1.0, 2.0 * log(minimum_width)),
    )
    upper_lines = (
        (0.0, log(ratio_max)),
        (-1.0, 2.0 * log(upper_width)),
        (1.0, -2.0 * log(minimum_height)),
    )
    breakpoints = {x_min, x_max}
    for lines in (lower_lines, upper_lines):
        for first, second in combinations(lines, 2):
            intersection = _line_intersection(first, second)
            if intersection is not None and x_min < intersection < x_max:
                breakpoints.add(intersection)

    segments: list[tuple[float, float, float, float, float, float]] = []
    cumulative = 0.0
    ordered = sorted(breakpoints)
    for left, right in pairwise(ordered):
        midpoint = (left + right) / 2.0
        lower_slope, lower_intercept = max(lower_lines, key=lambda line: line[0] * midpoint + line[1])
        upper_slope, upper_intercept = min(upper_lines, key=lambda line: line[0] * midpoint + line[1])
        slope = upper_slope - lower_slope
        intercept = upper_intercept - lower_intercept
        if slope * midpoint + intercept <= _EPSILON:
            continue
        mass = _integral_linear_exp(slope, intercept, right) - _integral_linear_exp(slope, intercept, left)
        if mass <= _EPSILON:
            continue
        segments.append((left, right, slope, intercept, cumulative, mass))
        cumulative += mass
    return tuple(segments)


def _sample_standard_2d_area(
    segments: tuple[tuple[float, float, float, float, float, float], ...],
    random_value: float,
) -> float:
    total_mass = segments[-1][4] + segments[-1][5]
    target_mass = random_value * total_mass
    left, right, slope, intercept, cumulative, mass = segments[-1]
    for candidate in segments:
        if target_mass <= candidate[4] + candidate[5]:
            left, right, slope, intercept, cumulative, mass = candidate
            break
    target = target_mass - cumulative
    if abs(slope) <= _EPSILON:
        return exp(left) + target / intercept
    base = _integral_linear_exp(slope, intercept, left)
    low, high = left, right
    value = left + (right - left) * target / mass
    for _ in range(5):
        exponential = exp(value)
        residual = exponential * (slope * (value - 1.0) + intercept) - base - target
        if residual < 0.0:
            low = value
        else:
            high = value
        density = exponential * (slope * value + intercept)
        next_value = value - residual / density if density > _EPSILON else (low + high) / 2.0
        value = next_value if low < next_value < high else (low + high) / 2.0
    residual = _integral_linear_exp(slope, intercept, value) - base - target
    if abs(residual) > mass * 1e-10:
        for _ in range(32):
            midpoint = (low + high) / 2.0
            if _integral_linear_exp(slope, intercept, midpoint) - base < target:
                low = midpoint
            else:
                high = midpoint
        value = (low + high) / 2.0
    return exp(value)


def _is_roundoff_sized_area_gap(area_min: float, area_max: float) -> bool:
    """Allow four ULPs of error from normalizing and denormalizing area bounds."""
    return area_min - area_max <= 4 * max(ulp(area_min), ulp(area_max))


def _sample_2d_fixed_ratio(
    height: int,
    width: int,
    scale: tuple[float, float],
    ratio: float,
    random_value: float,
    minimum_height: float = _LOWER_EDGE,
    minimum_width: float = _LOWER_EDGE,
    allow_roundoff: bool = False,
) -> float | None:
    """Sample a valid crop area for a fixed aspect ratio."""
    area = height * width
    area_min = max(
        scale[0] * area,
        minimum_height * minimum_width,
        minimum_height**2 * ratio,
        minimum_width**2 / ratio,
    )
    area_max = min(scale[1] * area, (width + _LOWER_EDGE) ** 2 / ratio, (height + _LOWER_EDGE) ** 2 * ratio)
    if area_min > area_max:
        if not allow_roundoff or not _is_roundoff_sized_area_gap(area_min, area_max):
            return None
        area_max = area_min
    return area_min + random_value * (area_max - area_min)


def sample_2d_crop_shape(
    height: int,
    width: int,
    scale: tuple[float, float],
    ratio: tuple[float, float],
    sampling_method: SamplingMethod,
    py_random: _RandomSource,
    min_crop_shape: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """Sample one rounded valid 2D crop shape without stochastic rejection."""
    random = py_random.random
    ratio_min, ratio_max = ratio
    if min_crop_shape is None:
        minimum_height = minimum_width = _LOWER_EDGE
        crop_min_height = crop_min_width = 1
    else:
        crop_min_height, crop_min_width = min_crop_shape
        minimum_height = max(_LOWER_EDGE, crop_min_height - _LOWER_EDGE)
        minimum_width = max(_LOWER_EDGE, crop_min_width - _LOWER_EDGE)
    if ratio_min == ratio_max:
        target_area = _sample_2d_fixed_ratio(
            height,
            width,
            scale,
            ratio_min,
            random(),
            minimum_height,
            minimum_width,
            min_crop_shape is not None,
        )
        if target_area is None:
            return None
        aspect_ratio = ratio_min
    else:
        area = height * width
        area_min = max(
            scale[0] * area,
            minimum_height * minimum_width,
            ratio_min * minimum_height**2,
            minimum_width**2 / ratio_max,
        )
        area_max = min(
            scale[1] * area,
            (height + _LOWER_EDGE) * (width + _LOWER_EDGE),
            (width + _LOWER_EDGE) ** 2 / ratio_min,
            ratio_max * (height + _LOWER_EDGE) ** 2,
        )
        if area_min > area_max:
            if min_crop_shape is None or not _is_roundoff_sized_area_gap(area_min, area_max):
                return None
            area_max = area_min
        if area_max - area_min <= _EPSILON:
            target_area = area_min
        elif sampling_method == "uniform_scale":
            target_area = area_min + random() * (area_max - area_min)
        else:
            segments = (
                _standard_2d_segments(height, width, scale[0], scale[1], ratio_min, ratio_max)
                if min_crop_shape is None
                else _build_standard_2d_segments(
                    height,
                    width,
                    scale[0],
                    scale[1],
                    ratio_min,
                    ratio_max,
                    minimum_height,
                    minimum_width,
                )
            )
            target_area = _sample_standard_2d_area(segments, random()) if segments else area_min

        log_ratio_min = max(
            log(ratio_min),
            log(target_area) - 2.0 * log(height + _LOWER_EDGE),
            2.0 * log(minimum_width) - log(target_area),
        )
        log_ratio_max = min(
            log(ratio_max),
            2.0 * log(width + _LOWER_EDGE) - log(target_area),
            log(target_area) - 2.0 * log(minimum_height),
        )
        if log_ratio_min > log_ratio_max + _EPSILON:
            return None
        aspect_ratio = exp(log_ratio_min + random() * (log_ratio_max - log_ratio_min))

    crop_width = min(width, max(crop_min_width, round(sqrt(target_area * aspect_ratio))))
    crop_height = min(height, max(crop_min_height, round(sqrt(target_area / aspect_ratio))))
    return crop_height, crop_width


def relax_2d_crop_ranges(
    height: int,
    width: int,
    scale: tuple[float, float],
    ratio: tuple[float, float],
    min_crop_shape: tuple[int, int],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Keep the closest feasible ratio, then minimize area deviation from the scale range."""
    minimum_height = max(_LOWER_EDGE, min_crop_shape[0] - _LOWER_EDGE)
    minimum_width = max(_LOWER_EDGE, min_crop_shape[1] - _LOWER_EDGE)
    feasible_ratio_min = minimum_width / (height + _LOWER_EDGE)
    feasible_ratio_max = (width + _LOWER_EDGE) / minimum_height

    ratio_min = max(ratio[0], feasible_ratio_min)
    ratio_max = min(ratio[1], feasible_ratio_max)
    if ratio_min > ratio_max:
        ratio_min = ratio_max = feasible_ratio_min if ratio[1] < feasible_ratio_min else feasible_ratio_max

    image_area = height * width
    feasible_area_min, feasible_area_max = _crop_area_bounds(
        height,
        width,
        (0.0, 1.0),
        (ratio_min, ratio_max),
        minimum_height,
        minimum_width,
    )
    requested_area_min = scale[0] * image_area
    requested_area_max = scale[1] * image_area
    effective_area_min = max(feasible_area_min, requested_area_min)
    effective_area_max = min(feasible_area_max, requested_area_max)

    if effective_area_min > effective_area_max:
        nearest_area = feasible_area_max if requested_area_min > feasible_area_max else feasible_area_min
        effective_area_min = nearest_area
        effective_area_max = nearest_area

    return (
        (effective_area_min / image_area, effective_area_max / image_area),
        (ratio_min, ratio_max),
    )


def _crop_area_bounds(
    height: int,
    width: int,
    scale: tuple[float, float],
    ratio: tuple[float, float],
    minimum_height: float,
    minimum_width: float,
) -> tuple[float, float]:
    """Return the feasible crop-area interval for scale, ratio, and size bounds."""
    image_area = height * width
    ratio_min, ratio_max = ratio
    area_min = max(
        scale[0] * image_area,
        minimum_height * minimum_width,
        ratio_min * minimum_height**2,
        minimum_width**2 / ratio_max,
    )
    area_max = min(
        scale[1] * image_area,
        (height + _LOWER_EDGE) * (width + _LOWER_EDGE),
        (width + _LOWER_EDGE) ** 2 / ratio_min,
        ratio_max * (height + _LOWER_EDGE) ** 2,
    )
    return area_min, area_max


def _fixed_shape_volume_bounds(
    depth: float,
    height: float,
    width: float,
    scale: tuple[float, float],
    depth_to_height: float,
    width_to_height: float,
) -> tuple[float, float]:
    height_min = max(_LOWER_EDGE, _LOWER_EDGE / depth_to_height, _LOWER_EDGE / width_to_height)
    height_max = min(
        height + _LOWER_EDGE,
        (depth + _LOWER_EDGE) / depth_to_height,
        (width + _LOWER_EDGE) / width_to_height,
    )
    volume_factor = depth_to_height * width_to_height
    volume = depth * height * width
    return max(scale[0] * volume, volume_factor * height_min**3), min(scale[1] * volume, volume_factor * height_max**3)


def _round_3d_shape(
    depth: float,
    height: float,
    width: float,
    source_depth: int,
    source_height: int,
    source_width: int,
) -> tuple[int, int, int]:
    return (
        min(source_depth, max(1, round(depth))),
        min(source_height, max(1, round(height))),
        min(source_width, max(1, round(width))),
    )


@lru_cache(maxsize=128)
def _ratio_polytope(
    depth: int,
    height: int,
    width: int,
    ratio: float,
) -> tuple[tuple[tuple[float, float, float, float], ...], tuple[float, ...]]:
    """Return log-volume/shape halfspaces and their feasible-vertex breakpoints."""
    log_ratio = log(ratio)
    lower_log = log(_LOWER_EDGE)
    planes = (
        (0.0, -1.0, 0.0, log_ratio),
        (0.0, 1.0, 0.0, log_ratio),
        (0.0, 0.0, -1.0, log_ratio),
        (0.0, 0.0, 1.0, log_ratio),
        (0.0, 1.0, -1.0, log_ratio),
        (0.0, -1.0, 1.0, log_ratio),
        (1.0, -1.0, -1.0, 3.0 * log(height + _LOWER_EDGE)),
        (1.0, 2.0, -1.0, 3.0 * log(width + _LOWER_EDGE)),
        (-1.0, 1.0, -2.0, -3.0 * lower_log),
        (-1.0, 1.0, 1.0, -3.0 * lower_log),
        (-1.0, -2.0, 1.0, -3.0 * lower_log),
        (1.0, -1.0, 2.0, 3.0 * log(depth + _LOWER_EDGE)),
    )
    plane_array = np.asarray(planes, dtype=np.float64)
    breakpoints: set[float] = set()
    for indices in combinations(range(len(planes)), 3):
        matrix = plane_array[list(indices), :3]
        if abs(np.linalg.det(matrix)) <= _EPSILON:
            continue
        point = np.linalg.solve(matrix, plane_array[list(indices), 3])
        if np.all(plane_array[:, :3] @ point <= plane_array[:, 3] + _EPSILON):
            breakpoints.add(float(point[0]))
    return planes, tuple(sorted(breakpoints))


def _clip_polygon(
    polygon: list[tuple[float, float]],
    coefficient_a: float,
    coefficient_b: float,
    limit: float,
) -> list[tuple[float, float]]:
    if not polygon:
        return []
    result: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_value = coefficient_a * previous[0] + coefficient_b * previous[1] - limit
    previous_inside = previous_value <= _EPSILON
    for current in polygon:
        current_value = coefficient_a * current[0] + coefficient_b * current[1] - limit
        current_inside = current_value <= _EPSILON
        if current_inside != previous_inside:
            fraction = previous_value / (previous_value - current_value)
            result.append(
                (
                    previous[0] + fraction * (current[0] - previous[0]),
                    previous[1] + fraction * (current[1] - previous[1]),
                ),
            )
        if current_inside:
            result.append(current)
        previous, previous_value, previous_inside = current, current_value, current_inside
    return result


def _shape_polygon(
    planes: tuple[tuple[float, float, float, float], ...],
    log_ratio: float,
    log_volume: float,
) -> list[tuple[float, float]]:
    polygon = [(-log_ratio, -log_ratio), (log_ratio, -log_ratio), (log_ratio, log_ratio), (-log_ratio, log_ratio)]
    for coefficient_u, coefficient_a, coefficient_b, limit in planes:
        polygon = _clip_polygon(polygon, coefficient_a, coefficient_b, limit - coefficient_u * log_volume)
        if not polygon:
            return []
    return polygon


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return (
        abs(
            sum(
                first[0] * second[1] - first[1] * second[0]
                for first, second in zip(polygon, [*polygon[1:], polygon[0]], strict=True)
            ),
        )
        / 2.0
    )


def _standard_3d_segments(
    planes: tuple[tuple[float, float, float, float], ...],
    critical_points: tuple[float, ...],
    log_ratio: float,
    lower: float,
    upper: float,
) -> tuple[tuple[float, float, float, float, float, float], ...]:
    breakpoints = sorted({lower, upper, *(point for point in critical_points if lower < point < upper)})
    segments: list[tuple[float, float, float, float, float, float]] = []
    cumulative = 0.0
    for left, right in pairwise(breakpoints):
        delta = (right - left) / 4.0
        midpoint = (left + right) / 2.0
        area_left = _polygon_area(_shape_polygon(planes, log_ratio, midpoint - delta))
        area_mid = _polygon_area(_shape_polygon(planes, log_ratio, midpoint))
        area_right = _polygon_area(_shape_polygon(planes, log_ratio, midpoint + delta))
        quadratic = (area_left - 2.0 * area_mid + area_right) / (2.0 * delta * delta)
        linear = (area_right - area_left) / (2.0 * delta) - 2.0 * quadratic * midpoint
        constant = area_mid - quadratic * midpoint * midpoint - linear * midpoint
        mass = _integral_quadratic_exp(quadratic, linear, constant, right) - _integral_quadratic_exp(
            quadratic,
            linear,
            constant,
            left,
        )
        if mass > _EPSILON:
            segments.append((left, right, quadratic, linear, constant, cumulative))
            cumulative += mass
    return tuple(segments)


def _sample_standard_3d_volume(
    segments: tuple[tuple[float, float, float, float, float, float], ...],
    random_value: float,
) -> float:
    final = segments[-1]
    total_mass = (
        _integral_quadratic_exp(final[2], final[3], final[4], final[1])
        - _integral_quadratic_exp(
            final[2],
            final[3],
            final[4],
            final[0],
        )
        + final[5]
    )
    target_mass = random_value * total_mass
    selected = final
    for candidate in segments:
        mass = _integral_quadratic_exp(candidate[2], candidate[3], candidate[4], candidate[1]) - (
            _integral_quadratic_exp(candidate[2], candidate[3], candidate[4], candidate[0])
        )
        if target_mass <= candidate[5] + mass:
            selected = candidate
            break
    left, right, quadratic, linear, constant, cumulative = selected
    target = target_mass - cumulative
    base = _integral_quadratic_exp(quadratic, linear, constant, left)
    low, high = left, right
    for _ in range(48):
        midpoint = (low + high) / 2.0
        if _integral_quadratic_exp(quadratic, linear, constant, midpoint) - base < target:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def _sample_polygon_point(polygon: list[tuple[float, float]], py_random: _RandomSource) -> tuple[float, float]:
    if len(polygon) == 1:
        return polygon[0]
    if len(polygon) == 2:
        proportion = py_random.random()
        return (
            polygon[0][0] + proportion * (polygon[1][0] - polygon[0][0]),
            polygon[0][1] + proportion * (polygon[1][1] - polygon[0][1]),
        )

    origin = polygon[0]
    triangles: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    total_area = 0.0
    for first, second in pairwise(polygon[1:]):
        area = (
            abs(
                (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0]),
            )
            / 2.0
        )
        if area > _EPSILON:
            triangles.append((first, second, area))
            total_area += area
    if total_area <= _EPSILON:
        return origin
    target = py_random.random() * total_area
    first, second = triangles[-1][:2]
    cumulative = 0.0
    for candidate_first, candidate_second, area in triangles:
        cumulative += area
        if target <= cumulative:
            first, second = candidate_first, candidate_second
            break
    root = sqrt(py_random.random())
    second_weight = py_random.random()
    return (
        (1.0 - root) * origin[0] + root * (1.0 - second_weight) * first[0] + root * second_weight * second[0],
        (1.0 - root) * origin[1] + root * (1.0 - second_weight) * first[1] + root * second_weight * second[1],
    )


def sample_3d_crop_shape(
    depth: int,
    height: int,
    width: int,
    output_size: tuple[int, int, int],
    scale: tuple[float, float],
    ratio: float | None,
    sampling_method: SamplingMethod,
    py_random: _RandomSource,
) -> tuple[int, int, int] | None:
    """Sample one rounded valid 3D crop shape without stochastic rejection."""
    if ratio is None or ratio == 1.0:
        if ratio is None:
            output_depth, output_height, output_width = output_size
            depth_to_height = output_depth / output_height
            width_to_height = output_width / output_height
        else:
            depth_to_height = 1.0
            width_to_height = 1.0
        volume_min, volume_max = _fixed_shape_volume_bounds(
            depth,
            height,
            width,
            scale,
            depth_to_height,
            width_to_height,
        )
        if volume_min > volume_max:
            return None
        volume = volume_min + py_random.random() * (volume_max - volume_min)
        crop_height = (volume / (depth_to_height * width_to_height)) ** (1.0 / 3.0)
        return _round_3d_shape(
            depth_to_height * crop_height,
            crop_height,
            width_to_height * crop_height,
            depth,
            height,
            width,
        )

    planes, critical_points = _ratio_polytope(depth, height, width, ratio)
    if not critical_points:
        return None
    volume = depth * height * width
    log_volume_min = log(max(scale[0] * volume, _LOWER_EDGE**3))
    log_volume_max = log(scale[1] * volume) if scale[1] > 0.0 else float("-inf")
    lower = max(log_volume_min, critical_points[0])
    upper = min(log_volume_max, critical_points[-1])
    if lower > upper:
        return None
    if upper - lower <= _EPSILON:
        log_volume = lower
    elif sampling_method == "uniform_scale":
        volume_min = exp(lower)
        volume_max = exp(upper)
        log_volume = log(volume_min + py_random.random() * (volume_max - volume_min))
    else:
        segments = _standard_3d_segments(planes, critical_points, log(ratio), lower, upper)
        log_volume = _sample_standard_3d_volume(segments, py_random.random()) if segments else (lower + upper) / 2.0

    polygon = _shape_polygon(planes, log(ratio), log_volume)
    if not polygon:
        return None
    log_width_to_height, log_depth_to_height = _sample_polygon_point(polygon, py_random)
    crop_height = exp((log_volume - log_width_to_height - log_depth_to_height) / 3.0)
    return _round_3d_shape(
        exp(log_depth_to_height) * crop_height,
        crop_height,
        exp(log_width_to_height) * crop_height,
        depth,
        height,
        width,
    )
