"""Independent mathematical references for vector-field geometry kernels."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pytest_mock import MockerFixture

import albumentations as A
from albumentations.augmentations.geometric._functional_vector_fields import (
    permute_vector_field_components_2d,
    permute_vector_field_components_3d,
    pull_map_from_matrix_2d,
    pull_map_from_matrix_3d,
    resample_field_2d,
    resample_field_3d,
    transform_vector_components_2d,
    transform_vector_components_3d,
    warp_vector_field_2d,
    warp_vector_field_3d,
    warp_vector_field_from_pull_grid_3d,
    warp_vector_field_from_pull_map_2d,
)
from albumentations.core.transform_params import SampledParams


@pytest.mark.parametrize("dtype", (np.float16, np.float32))
def test_bilinear_resampling_renormalizes_finite_vector_neighbours(dtype: type[np.generic]) -> None:
    field = np.full((2, 2, 2), (10.0, -4.0), dtype=dtype)
    field[1, 1] = np.nan
    map_x = np.array([[0.5, -0.25, -0.75]], dtype=np.float32)
    map_y = np.full_like(map_x, 0.5)

    result = resample_field_2d(field, map_x, map_y)

    np.testing.assert_array_equal(np.isnan(result), [[[False, False], [False, False], [True, True]]])
    np.testing.assert_array_equal(result[0, :2], np.array([[10.0, -4.0], [10.0, -4.0]], dtype=np.float32))


def test_bilinear_resampling_marks_empty_support_and_partial_vectors_as_invalid() -> None:
    field = np.full((2, 2, 2), np.nan, dtype=np.float32)
    field[0, 0] = (3.0, np.nan)
    result = resample_field_2d(
        field,
        np.array([[0.0, 4.0]], dtype=np.float32),
        np.array([[0.0, 4.0]], dtype=np.float32),
    )

    assert np.isnan(result).all()


@pytest.mark.parametrize(
    ("semantics", "source", "expected"),
    (
        ("gradient", (2.0, 8.0), (1.0, 2.0)),
        ("vector_field", (2.0, 8.0), (4.0, 32.0)),
        ("displacement", (2.0, 8.0), (4.0, 32.0)),
    ),
)
def test_anisotropic_scale_uses_target_component_semantics(
    semantics: str,
    source: tuple[float, ...],
    expected: tuple[float, ...],
) -> None:
    field = np.asarray(source, dtype=np.float16).reshape(1, 1, -1)

    result = transform_vector_components_2d(
        field,
        np.diag((2.0, 4.0)).astype(np.float32),
        semantics=semantics,
    )

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[0, 0], expected)


def test_normal_lift_preserves_depth_axis_and_renormalizes() -> None:
    source = np.asarray((-2.0, -8.0, 1.0), dtype=np.float32) / np.sqrt(np.float32(69))

    result = transform_vector_components_2d(
        source.reshape(1, 1, 3),
        np.diag((2.0, 4.0)).astype(np.float32),
        semantics="normal",
    )

    expected = np.asarray((-1.0, -2.0, 1.0), dtype=np.float32) / np.sqrt(np.float32(6))
    np.testing.assert_allclose(result[0, 0], expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("semantics", "source", "expected"),
    (
        ("gradient3d", (4.0, 8.0, 3.0), (2.0, 2.0, 6.0)),
        ("vector_field3d", (4.0, 8.0, 3.0), (8.0, 32.0, 1.5)),
        ("displacement3d", (4.0, 8.0, 3.0), (8.0, 32.0, 1.5)),
    ),
)
def test_anisotropic_3d_scale_uses_target_component_semantics(
    semantics: str,
    source: tuple[float, ...],
    expected: tuple[float, ...],
) -> None:
    result = transform_vector_components_3d(
        np.asarray(source, dtype=np.float16).reshape(1, 1, 1, 3),
        np.diag((2.0, 4.0, 0.5)).astype(np.float32),
        semantics=semantics,
    )

    np.testing.assert_array_equal(result[0, 0, 0], expected)


def test_3d_normal_uses_inverse_transpose_and_becomes_unit_length() -> None:
    source = np.asarray((-2.0, -8.0, -0.5), dtype=np.float32)
    source /= np.linalg.norm(source)

    result = transform_vector_components_3d(
        source.reshape(1, 1, 1, 3),
        np.diag((2.0, 4.0, 0.5)).astype(np.float32),
        semantics="normal3d",
    )

    expected = np.asarray((-1.0, -2.0, -1.0), dtype=np.float32)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(result[0, 0, 0], expected, rtol=1e-6, atol=1e-7)


def test_horizontal_flip_reindexes_and_reorients_each_field_type() -> None:
    field = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    result = permute_vector_field_components_2d(
        field,
        spatial_axes=(1,),
        component_matrix=np.diag((-1.0, 1.0)).astype(np.float32),
        semantics="gradient",
    )

    np.testing.assert_array_equal(result, [[[(-3.0), 4.0], [(-1.0), 2.0]]])


def test_3d_axis_permutation_reorders_spatial_axes_and_components() -> None:
    field = np.empty((2, 3, 4, 3), dtype=np.float32)
    field[...] = (1.0, 2.0, 3.0)
    result = permute_vector_field_components_3d(
        field,
        spatial_permutation=(2, 1, 0),
        component_matrix=np.asarray(((0, 0, 1), (0, 1, 0), (1, 0, 0)), dtype=np.float32),
        semantics="gradient3d",
    )

    assert result.shape == (4, 3, 2, 3)
    np.testing.assert_array_equal(result, np.broadcast_to((3.0, 2.0, 1.0), result.shape))


def test_affine_pull_map_uses_half_pixel_resize_coordinates_and_source_domain() -> None:
    matrix = np.asarray(((2.0, 0.0, 0.5), (0.0, 2.0, 0.5)), dtype=np.float32)

    map_x, map_y, valid = pull_map_from_matrix_2d(matrix, (9, 9), (4, 4))

    np.testing.assert_allclose(map_x[0, :4], (-0.25, 0.25, 0.75, 1.25), rtol=0, atol=1e-6)
    assert map_y[0, 0] == np.float32(-0.25)
    assert valid[0, 0]
    assert not valid[0, -1]


def test_affine_3d_pull_map_uses_half_voxel_resize_coordinates() -> None:
    matrix = np.diag((2.0, 2.0, 2.0, 1.0)).astype(np.float32)
    matrix[:3, 3] = 0.5

    grid, source_x, source_y, source_z, valid = pull_map_from_matrix_3d(matrix, (6, 8, 10), (3, 4, 5))

    np.testing.assert_allclose(source_x[2, 2, :4], (-0.25, 0.25, 0.75, 1.25), atol=1e-6)
    np.testing.assert_allclose(source_y[2, :4, 2], (-0.25, 0.25, 0.75, 1.25), atol=1e-6)
    np.testing.assert_allclose(source_z[:4, 2, 2], (-0.25, 0.25, 0.75, 1.25), atol=1e-6)
    assert grid.shape == (6, 8, 10, 3)
    assert valid.all()


def test_trilinear_resampling_renormalizes_finite_vector_neighbours() -> None:
    field = np.full((2, 2, 2, 3), (10.0, -4.0, 3.0), dtype=np.float32)
    field[1, 1, 1] = np.nan
    normalized_midpoint = np.full((1, 1, 1, 3), 0.0, dtype=np.float32)

    result, supported = resample_field_3d(field, normalized_midpoint)

    assert supported.item()
    np.testing.assert_array_equal(result[0, 0, 0], (10.0, -4.0, 3.0))


@pytest.mark.parametrize(
    ("semantics", "source", "expected"),
    (
        ("gradient3d", (2.0, 4.0, 8.0), (1.0, 2.0, 4.0)),
        ("vector_field3d", (2.0, 4.0, 8.0), (4.0, 8.0, 16.0)),
        ("displacement3d", (0.2, -0.1, 0.3), (0.4, -0.2, 0.6)),
    ),
)
def test_affine_3d_warp_uses_each_target_component_rule(
    semantics: str,
    source: tuple[float, float, float],
    expected: tuple[float, float, float],
) -> None:
    field = np.empty((3, 4, 5, 3), dtype=np.float32)
    field[...] = source
    matrix = np.diag((2.0, 2.0, 2.0, 1.0)).astype(np.float32)
    matrix[:3, 3] = 0.5

    result = warp_vector_field_3d(field, matrix, (6, 8, 10), semantics=semantics)

    np.testing.assert_allclose(result[3, 4, 5], expected, atol=1e-6)


@pytest.mark.parametrize(
    ("semantics", "source", "expected_before_slab_boundary", "expected_after_slab_boundary"),
    (
        ("gradient3d", (1.0, 0.0, 0.0), (1.0, 0.0, 0.028), (1.0, 0.0, 0.032)),
        ("vector_field3d", (0.0, 0.0, 1.0), (-0.028, 0.0, 1.0), (-0.032, 0.0, 1.0)),
    ),
)
def test_dense_pull_grid_uses_neighbor_derivatives_across_depth_slabs(
    semantics: str,
    source: tuple[float, float, float],
    expected_before_slab_boundary: tuple[float, float, float],
    expected_after_slab_boundary: tuple[float, float, float],
) -> None:
    depth, height, width = 12, 3, 4
    alpha = np.float32(0.002)
    z, y, x = np.mgrid[:depth, :height, :width].astype(np.float32)
    pull_x = x + alpha * z**2
    sampling_grid = np.stack(
        (
            2 * (pull_x + 0.5) / width - 1,
            2 * (y + 0.5) / height - 1,
            2 * (z + 0.5) / depth - 1,
        ),
        axis=-1,
    )
    field = np.empty((depth, height, width, 3), dtype=np.float32)
    field[...] = source

    result = warp_vector_field_from_pull_grid_3d(field, sampling_grid, semantics=semantics)

    np.testing.assert_allclose(result[7, 1, 1], expected_before_slab_boundary, rtol=0, atol=1e-6)
    np.testing.assert_allclose(result[8, 1, 1], expected_after_slab_boundary, rtol=0, atol=1e-6)


def test_dense_pull_grid_fold_singularity_marks_the_vector_invalid() -> None:
    depth, height, width = 3, 3, 5
    z, y, x = np.mgrid[:depth, :height, :width].astype(np.float32)
    folded_x = (x - 2) ** 2
    sampling_grid = np.stack(
        (
            2 * (folded_x + 0.5) / width - 1,
            2 * (y + 0.5) / height - 1,
            2 * (z + 0.5) / depth - 1,
        ),
        axis=-1,
    )
    field = np.ones((depth, height, width, 3), dtype=np.float32)

    result = warp_vector_field_from_pull_grid_3d(field, sampling_grid, semantics="vector_field3d")

    assert np.isnan(result[1, 1, 2]).all()


def test_elastic_control_model_keeps_computable_endpoints_outside_output_grid() -> None:
    depth, height, width = 4, 5, 5
    z, y, x = np.mgrid[:depth, :height, :width].astype(np.float32)
    sampling_grid = np.stack(
        (
            2 * (x + 0.5) / width - 1,
            2 * (y + 0.5) / height - 1,
            2 * (z + 0.5) / depth - 1,
        ),
        axis=-1,
    )
    coefficients = {plane: np.zeros((4, 4, 2), dtype=np.float32) for plane in ("xy", "xz", "yz")}
    field = np.zeros((depth, height, width, 3), dtype=np.float32)
    field[..., 2] = 4.0

    result = warp_vector_field_from_pull_grid_3d(
        field,
        sampling_grid,
        semantics="displacement3d",
        inverse_model=coefficients,
    )

    np.testing.assert_array_equal(result[1, 2, 2], (0.0, 0.0, 4.0))


@pytest.mark.parametrize("semantics", ("gradient", "vector_field", "displacement"))
def test_affine_warp_transforms_components_separately_from_sampling(semantics: str) -> None:
    field = np.empty((4, 4, 2), dtype=np.float32)
    field[...] = (2.0, 4.0)
    matrix = np.asarray(((2.0, 0.0, 0.5), (0.0, 2.0, 0.5)), dtype=np.float32)

    result = warp_vector_field_2d(field, matrix, (8, 8), semantics=semantics)

    assert result.shape == (8, 8, 2)
    expected_values = (1.0, 2.0) if semantics == "gradient" else (4.0, 8.0)
    np.testing.assert_allclose(result[3, 3], expected_values)
    assert not np.isnan(result[0, 0]).any()


def test_perspective_warp_uses_local_differential_and_both_displacement_endpoints() -> None:
    field = np.zeros((4, 4, 2), dtype=np.float32)
    field[..., 0] = 1.0
    matrix = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.01, 0.0, 1.0)), dtype=np.float32)

    gradient = warp_vector_field_2d(field, matrix, (4, 4), semantics="gradient")
    vector = warp_vector_field_2d(field, matrix, (4, 4), semantics="vector_field")
    displacement = warp_vector_field_2d(field * np.float32(0.2), matrix, (4, 4), semantics="displacement")
    source_x = 1.0 / (1.0 - np.float32(0.01))
    expected_gradient_x = (1.0 + np.float32(0.01) * source_x) ** 2
    expected_vector_x = 1.0 / expected_gradient_x
    expected_displacement_x = (source_x + 0.2) / (1.0 + np.float32(0.01) * (source_x + 0.2)) - 1.0

    assert gradient[0, 1, 0] == pytest.approx(expected_gradient_x, abs=2e-5)
    assert vector[0, 1, 0] == pytest.approx(expected_vector_x, abs=2e-5)
    assert displacement[0, 1, 0] == pytest.approx(expected_displacement_x, abs=2e-5)


@pytest.mark.parametrize("semantics", ("gradient", "vector_field", "displacement"))
def test_identity_pull_map_preserves_field_components(semantics: str) -> None:
    field = np.empty((3, 5, 2), dtype=np.float32)
    field[...] = (0.25, -0.5)
    map_y, map_x = np.mgrid[:3, :5].astype(np.float32)

    result = warp_vector_field_from_pull_map_2d(field, map_x, map_y, semantics=semantics)

    if semantics == "displacement":
        np.testing.assert_allclose(result[1, 2], field[1, 2], rtol=0, atol=1e-6)
    else:
        np.testing.assert_allclose(result, field, rtol=0, atol=1e-6)


def test_nonlinear_pull_map_transforms_local_vectors_and_finite_displacements() -> None:
    output_y, output_x = np.mgrid[:3, :17].astype(np.float32)
    map_x = np.sqrt(output_x)
    map_y = output_y.copy()
    vector_field = np.zeros((3, 5, 2), dtype=np.float32)
    vector_field[..., 0] = 1.0
    displacement = np.zeros_like(vector_field)
    displacement[..., 0] = 1.0

    vector_result = warp_vector_field_from_pull_map_2d(vector_field, map_x, map_y, semantics="vector_field")
    displacement_result = warp_vector_field_from_pull_map_2d(
        displacement,
        map_x,
        map_y,
        semantics="displacement",
    )

    assert vector_result[1, 4, 0] == pytest.approx(4.0, abs=0.05)
    assert displacement_result[1, 4, 0] == pytest.approx(5.0, abs=2e-3)


def test_elastic_control_model_keeps_computable_2d_endpoints_outside_output_grid() -> None:
    image = np.zeros((9, 11, 1), dtype=np.float32)
    displacement = np.zeros((9, 11, 2), dtype=np.float32)
    displacement[..., 0] = 20.0

    result = A.Compose(
        [A.ElasticTransform(displacement_range=(0.02, 0.02), control_grid_shape=(4, 4), p=1.0)],
    )(image=image, displacement=displacement)

    assert np.isfinite(result["displacement"][4, 5]).all()
    assert result["displacement"][4, 5, 0] > 19.0


def test_grid_shuffle_2d_maps_displacement_endpoints_across_tiles() -> None:
    height, width = 4, 8
    image = np.broadcast_to(np.arange(width, dtype=np.float32)[None, :, None], (height, width, 1)).copy()
    displacement = np.zeros((height, width, 2), dtype=np.float32)
    displacement[..., 0] = 1.0
    vector_field = np.zeros_like(displacement)
    vector_field[..., 0] = 1.0

    result = A.Compose([A.RandomGridShuffle(grid=(1, 2), p=1.0)], seed=2)(
        image=image,
        displacement=displacement,
        vector_field=vector_field,
    )

    np.testing.assert_array_equal(result["image"][0, :, 0], (4, 5, 6, 7, 0, 1, 2, 3))
    np.testing.assert_array_equal(result["displacement"][1, :3, 0], (1, 1, 1))
    assert np.isnan(result["displacement"][1, 3]).all()
    np.testing.assert_array_equal(result["displacement"][1, 4], (1.0, 0.0))
    np.testing.assert_array_equal(result["displacement"][1, 7], (-7.0, 0.0))
    assert np.isnan(result["vector_field"][1, 3]).all()
    assert np.isnan(result["vector_field"][1, 4]).all()
    np.testing.assert_array_equal(result["vector_field"][1, 5], (1.0, 0.0))


def test_grid_shuffle3d_maps_displacement_endpoints_across_tiles() -> None:
    depth, height, width = 2, 2, 8
    volume = np.broadcast_to(np.arange(width, dtype=np.float32)[None, None, :, None], (depth, height, width, 1)).copy()
    displacement = np.zeros((depth, height, width, 3), dtype=np.float32)
    displacement[..., 0] = 1.0

    result = A.Compose([A.GridShuffle3D(grid_zyx=(1, 1, 2), p=1.0)], seed=2)(
        volume=volume,
        displacement3d=displacement,
    )

    np.testing.assert_array_equal(result["volume"][0, 0, :, 0], (4, 5, 6, 7, 0, 1, 2, 3))
    np.testing.assert_array_equal(result["displacement3d"][0, 0, :3, 0], (1, 1, 1))
    assert np.isnan(result["displacement3d"][0, 0, 3]).all()
    np.testing.assert_array_equal(result["displacement3d"][0, 0, 4], (1.0, 0.0, 0.0))
    np.testing.assert_array_equal(result["displacement3d"][0, 0, 7], (-7.0, 0.0, 0.0))


@pytest.mark.parametrize("semantics", ("gradient", "normal"))
def test_invalid_covectors_and_singular_linear_maps_return_nan(semantics: str) -> None:
    component_count = 3 if semantics == "normal" else 2
    field = np.ones((1, 1, component_count), dtype=np.float32)
    field[0, 0, 0] = np.nan

    invalid_input = transform_vector_components_2d(
        field,
        np.eye(2, dtype=np.float32),
        semantics=semantics,
    )
    singular_map = transform_vector_components_2d(
        np.ones_like(field),
        np.zeros((2, 2), dtype=np.float32),
        semantics=semantics,
    )

    assert np.isnan(invalid_input).all()
    assert np.isnan(singular_map).all()


@pytest.mark.parametrize("semantics", ("vector_field", "displacement"))
def test_degenerate_linear_pushforward_keeps_known_zero_vectors(semantics: str) -> None:
    field = np.ones((1, 1, 2), dtype=np.float32)

    result = transform_vector_components_2d(
        field,
        np.zeros((2, 2), dtype=np.float32),
        semantics=semantics,
    )

    np.testing.assert_array_equal(result, np.zeros_like(field))


@pytest.mark.parametrize(
    ("target", "source", "expected"),
    (
        ("gradient", (2.0, 4.0), (1.0, 2.0)),
        ("vector_field", (2.0, 4.0), (4.0, 8.0)),
        ("displacement", (0.25, 0.5), (0.5, 1.0)),
    ),
)
def test_resize_transform_routes_vector_fields_through_the_public_compose_path(
    target: str,
    source: tuple[float, float],
    expected: tuple[float, float],
) -> None:
    field = np.empty((4, 5, 2), dtype=np.float32)
    field[...] = source
    image = np.zeros((4, 5, 1), dtype=np.float32)

    result = A.Compose([A.Resize(height=8, width=10)])(image=image, **{target: field})

    assert result[target].shape == (8, 10, 2)
    np.testing.assert_allclose(result[target][3, 4], expected, atol=1e-6)
    assert np.isfinite(result[target]).all()


def test_resize_normal_uses_the_lifted_2d_deformation() -> None:
    normal = np.empty((4, 5, 3), dtype=np.float32)
    normal[...] = np.asarray((1.0, 0.0, 1.0), dtype=np.float32) / np.sqrt(np.float32(2))
    image = np.zeros((4, 5, 1), dtype=np.float32)

    result = A.Compose([A.Resize(height=8, width=10)])(image=image, normal=normal)

    expected = np.asarray((0.5, 0.0, 1.0), dtype=np.float32) / np.sqrt(np.float32(1.25))
    np.testing.assert_allclose(result["normal"][3, 4], expected, rtol=1e-6, atol=1e-7)


def test_resize_preserves_float16_field_dtype_and_invalidates_overflowed_vectors() -> None:
    image = np.zeros((4, 5, 1), dtype=np.float32)
    field = np.empty((4, 5, 2), dtype=np.float16)
    field[...] = (60000.0, 0.0)

    result = A.Compose([A.Resize(height=8, width=10)])(image=image, vector_field=field)

    assert result["vector_field"].dtype == np.float16
    assert np.isnan(result["vector_field"][3, 4]).all()


def test_horizontal_flip_reorients_components_through_the_public_compose_path() -> None:
    field = np.empty((2, 3, 2), dtype=np.float32)
    field[...] = (2.0, 4.0)
    image = np.zeros((2, 3, 1), dtype=np.float32)

    result = A.Compose([A.HorizontalFlip(p=1.0)])(image=image, vector_field=field)

    np.testing.assert_array_equal(result["vector_field"], np.broadcast_to((-2.0, 4.0), (2, 3, 2)))


def test_pad_marks_added_vector_field_cells_as_nan() -> None:
    field = np.ones((2, 3, 2), dtype=np.float32)
    image = np.zeros((2, 3, 1), dtype=np.float32)

    result = A.Compose([A.Pad(padding=1)])(image=image, gradient=field)

    assert result["gradient"].shape == (4, 5, 2)
    assert np.isnan(result["gradient"][0]).all()
    np.testing.assert_array_equal(result["gradient"][1:3, 1:4], field)
    assert np.isnan(result["gradient"][-1]).all()


def test_pad_if_needed3d_replays_sparse_float16_field() -> None:
    field = np.arange(1, 73, dtype=np.float16).reshape(2, 3, 4, 3)
    field[0, 1, 2] = np.nan
    data = {"volume": np.zeros((2, 3, 4, 1), dtype=np.uint8), "gradient3d": field}
    expected = np.pad(field, ((1, 1), (1, 2), (1, 1), (0, 0)), constant_values=np.nan)
    pipeline = A.Compose(
        [A.PadIfNeeded3D(min_zyx=(3, 5, 5), pad_divisor_zyx=(2, 2, 2))],
        save_applied_params=True,
    )

    result = pipeline(**data)
    record = json.loads(json.dumps(result["applied_transforms"], allow_nan=False))
    replayed = A.Compose.from_applied_transforms(record)(**data)

    for output in (result, replayed):
        assert output["gradient3d"].dtype == np.float16
        np.testing.assert_array_equal(output["gradient3d"], expected)


def test_center_crop_keeps_displacement_values_and_output_endpoint_semantics() -> None:
    field = np.empty((4, 4, 2), dtype=np.float32)
    field[...] = (1.0, -0.5)
    image = np.zeros((4, 4, 1), dtype=np.float32)

    result = A.Compose([A.Crop(x_min=1, y_min=1, x_max=3, y_max=3)])(image=image, displacement=field)

    assert result["displacement"].shape == (2, 2, 2)
    np.testing.assert_array_equal(result["displacement"], field[1:3, 1:3])


@pytest.mark.parametrize(
    ("target", "source", "expected"),
    (
        ("gradient3d", (2.0, 4.0, 8.0), (1.0, 2.0, 4.0)),
        ("vector_field3d", (2.0, 4.0, 8.0), (4.0, 8.0, 16.0)),
        ("displacement3d", (0.25, 0.5, -0.125), (0.5, 1.0, -0.25)),
    ),
)
def test_resize3d_routes_vector_fields_through_the_public_compose_path(
    target: str,
    source: tuple[float, float, float],
    expected: tuple[float, float, float],
) -> None:
    field = np.empty((2, 3, 4, 3), dtype=np.float32)
    field[...] = source
    volume = np.zeros((2, 3, 4, 1), dtype=np.float32)

    result = A.Compose([A.Resize3D(size=(4, 6, 8))])(volume=volume, **{target: field})

    assert result[target].shape == (4, 6, 8, 3)
    assert np.isfinite(result[target]).all()
    np.testing.assert_allclose(result[target][2, 3, 4], expected, atol=1e-6)


def test_flip3d_reorients_components_in_xyz_order() -> None:
    field = np.empty((2, 3, 4, 3), dtype=np.float32)
    field[...] = (1.0, 2.0, 3.0)
    volume = np.zeros((2, 3, 4, 1), dtype=np.float32)

    result = A.Compose([A.Flip3D(flip_axes=(0, 2))])(volume=volume, vector_field3d=field)

    np.testing.assert_array_equal(result["vector_field3d"], np.broadcast_to((-1.0, 2.0, -3.0), field.shape))


def test_random_rotate90_3d_reorients_components_in_the_selected_axis_pair() -> None:
    field = np.empty((2, 3, 5, 3), dtype=np.float32)
    field[...] = (1.0, 2.0, 3.0)
    volume = np.zeros((2, 3, 5, 1), dtype=np.float32)
    transform = A.RandomRotate90_3D(axis_pair=(0, 2), group_element="r90", p=1.0)

    result = A.Compose([transform])(volume=volume, vector_field3d=field)

    assert result["vector_field3d"].shape == (5, 3, 2, 3)
    np.testing.assert_array_equal(
        result["vector_field3d"],
        np.broadcast_to((3.0, 2.0, -1.0), result["vector_field3d"].shape),
    )


def test_elastic_transform3d_reuses_its_pull_grid_for_gradient_and_volume(mocker: MockerFixture) -> None:
    from albumentations.augmentations.transforms3d import functional as f3d

    volume = np.zeros((5, 7, 9, 1), dtype=np.float32)
    gradient = np.ones((5, 7, 9, 3), dtype=np.float32)
    transform = A.ElasticTransform3D(
        displacement_range=(0.02, 0.02),
        control_grid_shape=(4, 4),
        p=1.0,
    )
    grid_creation = mocker.spy(f3d, "create_elastic_grid_3d")

    result = A.Compose([transform], seed=137)(volume=volume, gradient3d=gradient)

    assert grid_creation.call_count == 1
    assert result["volume"].shape == volume.shape
    assert result["gradient3d"].shape == gradient.shape
    assert np.isfinite(result["gradient3d"][2, 3, 4]).all()


@pytest.mark.parametrize(
    ("target", "components", "expected"),
    (
        ("gradient3d", (2.0, 4.0, 6.0), (1.0, 2.0, 6.0)),
        ("displacement3d", (1.0, 2.0, 3.0), (2.0, 4.0, 3.0)),
        ("vector_field3d", (1.0, 2.0, 3.0), (2.0, 4.0, 3.0)),
        ("normal3d", (1.0, 0.0, 1.0), (1 / np.sqrt(5), 0.0, 2 / np.sqrt(5))),
    ),
)
def test_2d_resize_lifts_its_component_map_for_3d_fields(
    target: str,
    components: tuple[float, float, float],
    expected: tuple[float, float, float],
) -> None:
    volume = np.zeros((2, 3, 4, 1), dtype=np.float32)
    field = np.broadcast_to(np.asarray(components, dtype=np.float32), (2, 3, 4, 3)).copy()

    result = A.Compose([A.Resize(height=6, width=8, p=1.0)], strict=True)(volume=volume, **{target: field})

    assert result[target].shape == (2, 6, 8, 3)
    np.testing.assert_allclose(result[target][0, 2, 2], expected, atol=2e-6)


@pytest.mark.parametrize("transform_name", ("RandomSizedCrop", "RandomResizedCrop"))
@pytest.mark.parametrize(("target", "components"), (("vector_field", 2), ("vector_field3d", 3)))
def test_crop_resize_uses_only_values_inside_the_selected_crop(
    monkeypatch: pytest.MonkeyPatch,
    transform_name: str,
    target: str,
    components: int,
) -> None:
    transform = (
        A.RandomSizedCrop(min_max_height=(2, 2), size=(4, 4), p=1.0)
        if transform_name == "RandomSizedCrop"
        else A.RandomResizedCrop(size=(4, 4), p=1.0)
    )
    monkeypatch.setattr(
        transform,
        "sample_parameters",
        lambda *_args, **_kwargs: SampledParams(params={"crop_coords": (1, 1, 3, 3)}),
    )
    if target == "vector_field":
        spatial_input = {"image": np.zeros((4, 4, 1), dtype=np.float32)}
        field = np.ones((4, 4, components), dtype=np.float32)
        field[1:3, 1:3] = 0
        expected_shape = (4, 4, components)
    else:
        spatial_input = {"volume": np.zeros((2, 4, 4, 1), dtype=np.float32)}
        field = np.ones((2, 4, 4, components), dtype=np.float32)
        field[:, 1:3, 1:3] = 0
        expected_shape = (2, 4, 4, components)

    result = A.Compose([transform], strict=True)(**spatial_input, **{target: field})

    assert result[target].shape == expected_shape
    np.testing.assert_array_equal(result[target], np.zeros(expected_shape, dtype=np.float32))


@pytest.mark.parametrize(("target", "components"), (("vector_field", 2), ("vector_field3d", 3)))
def test_crop_and_pad_resizes_only_values_inside_the_retained_crop(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    components: int,
) -> None:
    transform = A.CropAndPad(px=-1, keep_size=True, p=1.0)
    monkeypatch.setattr(
        transform,
        "sample_parameters",
        lambda *_args, **_kwargs: SampledParams(
            params={
                "crop_params": (1, 1, 3, 3),
                "pad_params": None,
                "result_shape": (2, 2),
                "shape": (4, 4),
            },
        ),
    )
    if target == "vector_field":
        spatial_input = {"image": np.zeros((4, 4, 1), dtype=np.float32)}
        field = np.ones((4, 4, components), dtype=np.float32)
        field[1:3, 1:3] = 0
        expected_shape = (4, 4, components)
    else:
        spatial_input = {"volume": np.zeros((2, 4, 4, 1), dtype=np.float32)}
        field = np.ones((2, 4, 4, components), dtype=np.float32)
        field[:, 1:3, 1:3] = 0
        expected_shape = (2, 4, 4, components)

    result = A.Compose([transform], strict=True)(**spatial_input, **{target: field})

    assert result[target].shape == expected_shape
    np.testing.assert_array_equal(result[target], np.zeros(expected_shape, dtype=np.float32))


def test_crop_and_pad_marks_padding_invalid_relative_to_the_cropped_field(monkeypatch: pytest.MonkeyPatch) -> None:
    transform = A.CropAndPad(px=-1, keep_size=True, p=1.0)
    monkeypatch.setattr(
        transform,
        "sample_parameters",
        lambda *_args, **_kwargs: SampledParams(
            params={
                "crop_params": (1, 1, 3, 3),
                "pad_params": (1, 1, 1, 1),
                "result_shape": (4, 4),
                "shape": (4, 4),
            },
        ),
    )
    image = np.zeros((4, 4, 1), dtype=np.float32)
    field = np.ones((4, 4, 2), dtype=np.float32)
    field[1:3, 1:3] = 0

    result = A.Compose([transform], strict=True)(image=image, vector_field=field)

    expected = np.full((4, 4, 2), np.nan, dtype=np.float32)
    expected[1:3, 1:3] = 0
    np.testing.assert_array_equal(result["vector_field"], expected)


@pytest.mark.parametrize(("target", "components"), (("vector_field", 2), ("vector_field3d", 3)))
def test_random_sized_bbox_safe_crop_resizes_vector_fields(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    components: int,
) -> None:
    transform = A.RandomSizedBBoxSafeCrop(height=4, width=4, p=1.0)
    monkeypatch.setattr(
        transform,
        "sample_parameters",
        lambda *_args, **_kwargs: SampledParams(params={"crop_coords": (1, 1, 3, 3)}),
    )
    if target == "vector_field":
        spatial_input = {"image": np.zeros((4, 4, 1), dtype=np.float32)}
        field = np.full((4, 4, components), 50.0, dtype=np.float32)
        field[1:3, 1:3] = (1.0, 2.0)
        expected = (2.0, 4.0)
        expected_shape = (4, 4, components)
    else:
        spatial_input = {"volume": np.zeros((2, 4, 4, 1), dtype=np.float32)}
        field = np.full((2, 4, 4, components), 50.0, dtype=np.float32)
        field[:, 1:3, 1:3] = (1.0, 2.0, 3.0)
        expected = (2.0, 4.0, 3.0)
        expected_shape = (2, 4, 4, components)

    result = A.Compose([transform], strict=True)(**spatial_input, **{target: field})

    assert result[target].shape == expected_shape
    np.testing.assert_allclose(result[target], np.broadcast_to(expected, expected_shape))


def test_random_grid_shuffle_preserves_reindexed_z_displacement(monkeypatch: pytest.MonkeyPatch) -> None:
    height, width = 2, 8
    volume = np.broadcast_to(np.arange(width, dtype=np.float32)[None, None, :, None], (1, height, width, 1)).copy()
    field = np.zeros((1, height, width, 3), dtype=np.float32)
    field[..., 2] = np.arange(width, dtype=np.float32)[None, None, :]
    tiles = np.asarray(((0, 0, height, 4), (0, 4, height, 8)), dtype=np.int32)
    transform = A.RandomGridShuffle(grid=(1, 2), p=1.0)
    monkeypatch.setattr(
        transform,
        "sample_parameters",
        lambda *_args, **_kwargs: SampledParams(params={"tiles": tiles, "mapping": [1, 0]}),
    )

    result = A.Compose([transform], strict=True)(volume=volume, displacement3d=field)

    expected_z = (4, 5, 6, 7, 0, 1, 2, 3)
    np.testing.assert_array_equal(result["volume"][0, 0, :, 0], expected_z)
    np.testing.assert_array_equal(result["displacement3d"][0, 0, :, 2], expected_z)


def test_random_grid_shuffle_reindexes_fields_with_non_involutive_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    height, width = 2, 9
    volume = np.zeros((height, width, 1), dtype=np.float32)
    field = np.zeros((height, width, 2), dtype=np.float32)
    tile_values = (10.0, 20.0, 30.0)
    for tile_index, value in enumerate(tile_values):
        tile = slice(tile_index * 3, (tile_index + 1) * 3)
        volume[:, tile, 0] = value
        field[:, tile, 0] = value
    tiles = np.asarray(((0, 0, height, 3), (0, 3, height, 6), (0, 6, height, 9)), dtype=np.int32)
    transform = A.RandomGridShuffle(grid=(1, 3), p=1.0)
    monkeypatch.setattr(
        transform,
        "sample_parameters",
        lambda *_args, **_kwargs: SampledParams(params={"tiles": tiles, "mapping": [1, 2, 0]}),
    )

    result = A.Compose([transform], strict=True)(image=volume, vector_field=field)

    for x in (1, 4, 7):
        np.testing.assert_array_equal(result["vector_field"][:, x, 0], result["image"][:, x, 0])


def test_grid_shuffle3d_reindexes_fields_with_non_involutive_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    depth, height, width = 2, 2, 9
    volume = np.zeros((depth, height, width, 1), dtype=np.float32)
    field = np.zeros((depth, height, width, 3), dtype=np.float32)
    tile_values = (10.0, 20.0, 30.0)
    for tile_index, value in enumerate(tile_values):
        tile = slice(tile_index * 3, (tile_index + 1) * 3)
        volume[:, :, tile, 0] = value
        field[:, :, tile, 0] = value
    tiles = np.asarray(
        (
            (0, 0, 0, depth, height, 3),
            (0, 0, 3, depth, height, 6),
            (0, 0, 6, depth, height, width),
        ),
        dtype=np.int32,
    )
    transform = A.GridShuffle3D(grid_zyx=(1, 1, 3), p=1.0)
    monkeypatch.setattr(
        transform,
        "sample_parameters",
        lambda *_args, **_kwargs: SampledParams(params={"tiles": tiles, "mapping": [1, 2, 0]}),
    )

    result = A.Compose([transform], strict=True)(volume=volume, vector_field3d=field)

    for x in (1, 4, 7):
        np.testing.assert_array_equal(result["vector_field3d"][:, :, x, 0], result["volume"][:, :, x, 0])
