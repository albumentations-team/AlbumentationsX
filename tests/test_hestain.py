import json

import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.pixel import functional as fpixel
from tests.helpers import TestDataFactory

CUSTOM_STAIN_MATRIX = np.array([[0.71, 0.65, 0.27], [0.18, 0.91, 0.37]], dtype=np.float32)


def _apply_he_residual_reference(
    image: np.ndarray,
    stain_matrix: np.ndarray,
    scale_factors: np.ndarray,
    shift_values: np.ndarray,
) -> np.ndarray:
    max_value = 255.0 if image.dtype == np.uint8 else 1.0
    pixel_matrix = image.reshape(-1, 3).astype(np.float32) / max_value
    pixel_matrix = np.maximum(pixel_matrix, np.float32(1e-6))
    optical_density = -np.log(pixel_matrix)
    residual_vector = np.cross(stain_matrix[0], stain_matrix[1])
    residual_vector /= np.linalg.norm(residual_vector)
    full_stain_matrix = np.vstack([stain_matrix, residual_vector]).astype(np.float32)
    stain_concentrations = np.linalg.solve(full_stain_matrix.T, optical_density.T).T
    augmented_concentrations = stain_concentrations * scale_factors + shift_values
    result = np.clip(np.exp(-(augmented_concentrations @ full_stain_matrix)), 0, 1).reshape(image.shape)
    if image.dtype == np.uint8:
        return np.rint(result * 255).astype(np.uint8)
    return result


def _apply_he_project_legacy_reference(
    image: np.ndarray,
    stain_matrix: np.ndarray,
    scale_factors: np.ndarray,
    shift_values: np.ndarray,
    augment_background: bool,
) -> np.ndarray:
    max_value = 255.0 if image.dtype == np.uint8 else 1.0
    image_float = image.astype(np.float32) / max_value
    pixel_matrix = np.maximum(image_float.reshape(-1, 3), np.float32(1e-6))
    optical_density = -np.log(pixel_matrix)
    regularization = 1e-6
    stain_correlation = stain_matrix @ stain_matrix.T + regularization * np.eye(2)
    density_projection = stain_matrix @ optical_density.T
    stain_concentrations = np.linalg.solve(stain_correlation, density_projection).T
    if augment_background:
        stain_concentrations = stain_concentrations * scale_factors + shift_values
    else:
        luminosity = image_float @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        tissue_mask = luminosity.reshape(-1) < 0.85
        stain_concentrations[tissue_mask] = stain_concentrations[tissue_mask] * scale_factors + shift_values
    result = np.clip(np.exp(-(stain_concentrations @ stain_matrix)), 0, 1).reshape(image.shape)
    if image.dtype == np.uint8:
        return np.rint(result * 255).astype(np.uint8)
    return result.astype(np.float32)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_hestain_augment_residual_matches_full_basis_reference(dtype: type[np.generic]) -> None:
    rng = np.random.default_rng(137)
    if dtype == np.uint8:
        image = rng.integers(13, 256, size=(12, 10, 3), dtype=np.uint8)
    else:
        image = rng.uniform(0.05, 1.0, size=(12, 10, 3)).astype(np.float32)
    transform = A.HEStain(
        method="custom",
        stain_matrix=CUSTOM_STAIN_MATRIX,
        residual_mode="augment",
        intensity_scale_range=(0.8, 1.2),
        intensity_shift_range=(-0.1, 0.1),
        augment_background=True,
        p=1.0,
    )

    result = transform(image=image)["image"]
    params = transform.get_applied_params()

    assert params["scale_factors"].shape == (3,)
    assert params["shift_values"].shape == (3,)
    expected = _apply_he_residual_reference(
        image,
        CUSTOM_STAIN_MATRIX,
        params["scale_factors"],
        params["shift_values"],
    )
    tolerance = 1 if dtype == np.uint8 else 1e-6
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=tolerance)


def test_hestain_preserve_residual_reconstructs_identity() -> None:
    image = np.random.default_rng(137).uniform(0.05, 1.0, size=(12, 10, 3)).astype(np.float32)
    transform = A.HEStain(
        method="custom",
        stain_matrix=CUSTOM_STAIN_MATRIX,
        residual_mode="preserve",
        intensity_scale_range=(1.0, 1.0),
        intensity_shift_range=(0.0, 0.0),
        augment_background=True,
        p=1.0,
    )

    result = transform(image=image)["image"]
    params = transform.get_applied_params()

    np.testing.assert_array_equal(params["scale_factors"], np.array([1.0, 1.0, 1.0]))
    np.testing.assert_array_equal(params["shift_values"], np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(result, image, rtol=1e-5, atol=1e-6)


def test_hestain_default_project_mode_preserves_seeded_parameter_sequence() -> None:
    image = np.random.default_rng(137).integers(0, 256, size=(8, 7, 3), dtype=np.uint8)
    transform = A.HEStain(method="custom", stain_matrix=CUSTOM_STAIN_MATRIX, p=1.0)

    result = A.Compose([transform], seed=137, strict=True)(image=image)["image"]
    params = transform.get_applied_params()
    explicit_project = A.Compose(
        [A.HEStain(method="custom", stain_matrix=CUSTOM_STAIN_MATRIX, residual_mode="project", p=1.0)],
        seed=137,
        strict=True,
    )(image=image)["image"]

    np.testing.assert_array_equal(
        params["scale_factors"],
        np.array([0.9460570722474746, 1.2941528299113085]),
    )
    np.testing.assert_array_equal(
        params["shift_values"],
        np.array([-0.10763109776870272, -0.06048900184007305]),
    )
    np.testing.assert_array_equal(result, explicit_project)


@pytest.mark.parametrize("residual_mode", ["preserve", "augment"])
def test_hestain_degenerate_extracted_residual_basis_falls_back_to_project(residual_mode: str) -> None:
    image = np.full((32, 32, 3), 32, dtype=np.uint8)
    project = A.Compose(
        [A.HEStain(method="macenko", residual_mode="project", p=1.0)],
        seed=137,
        strict=True,
    )(image=image)["image"]
    residual = A.Compose(
        [A.HEStain(method="macenko", residual_mode=residual_mode, p=1.0)],
        seed=137,
        strict=True,
    )(image=image)["image"]

    np.testing.assert_array_equal(residual, project)


def test_apply_he_stain_augmentation_nearly_collinear_residual_basis_falls_back_to_project() -> None:
    image = np.random.default_rng(137).uniform(0.05, 1.0, size=(12, 10, 3)).astype(np.float32)
    nearly_collinear_matrix = np.array([[1.0, 0.0, 0.0], [1.0, 5e-7, 0.0]], dtype=np.float32)
    scale_factors = np.array([1.2, 0.8, 1.5], dtype=np.float32)
    shift_values = np.array([0.1, -0.05, 0.2], dtype=np.float32)

    residual = fpixel.apply_he_stain_augmentation(
        image,
        nearly_collinear_matrix,
        scale_factors,
        shift_values,
        augment_background=True,
        residual_mode="augment",
    )
    project = fpixel.apply_he_stain_augmentation(
        image,
        nearly_collinear_matrix,
        scale_factors[:2],
        shift_values[:2],
        augment_background=True,
        residual_mode="project",
    )

    np.testing.assert_array_equal(residual, project)


@pytest.mark.parametrize(
    ("scale_factors", "shift_values", "invalid_parameter"),
    [
        (np.ones(2, dtype=np.float32), np.zeros(3, dtype=np.float32), "scale_factors"),
        (np.ones(3, dtype=np.float32), np.zeros(2, dtype=np.float32), "shift_values"),
    ],
)
def test_apply_he_stain_augmentation_rejects_mismatched_residual_parameters(
    scale_factors: np.ndarray,
    shift_values: np.ndarray,
    invalid_parameter: str,
) -> None:
    image = np.full((8, 8, 3), 0.5, dtype=np.float32)

    with pytest.raises(ValueError, match=rf"{invalid_parameter} must have shape \(3,\)"):
        fpixel.apply_he_stain_augmentation(
            image,
            CUSTOM_STAIN_MATRIX,
            scale_factors,
            shift_values,
            augment_background=True,
            residual_mode="augment",
        )


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("augment_background", [False, True])
def test_hestain_project_mode_stays_within_accepted_legacy_tolerance(
    dtype: type[np.generic],
    augment_background: bool,
) -> None:
    data = TestDataFactory.create_image((24, 20, 3), dtype=dtype, seed=137)
    transform = A.HEStain(
        method="custom",
        stain_matrix=CUSTOM_STAIN_MATRIX,
        residual_mode="project",
        intensity_scale_range=(0.8, 1.2),
        intensity_shift_range=(-0.1, 0.1),
        augment_background=augment_background,
        p=1.0,
    )

    result = transform(image=data)["image"]
    params = transform.get_applied_params()
    expected = _apply_he_project_legacy_reference(
        data,
        CUSTOM_STAIN_MATRIX,
        params["scale_factors"],
        params["shift_values"],
        augment_background,
    )

    tolerance = 1 if dtype == np.uint8 else 1.5e-6
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=tolerance)


@pytest.mark.parametrize(
    ("target", "shape"),
    [
        ("image", (12, 10, 3)),
        ("images", (2, 12, 10, 3)),
        ("volume", (2, 12, 10, 3)),
        ("volumes", (2, 2, 12, 10, 3)),
    ],
)
def test_hestain_augment_residual_replay_reuses_all_sampled_parameters(
    target: str,
    shape: tuple[int, ...],
) -> None:
    data = TestDataFactory.create_image(shape, dtype=np.uint8, seed=137)
    pipeline = A.ReplayCompose(
        [
            A.HEStain(
                method="custom",
                stain_matrix=CUSTOM_STAIN_MATRIX,
                residual_mode="augment",
                augment_background=True,
                p=1.0,
            ),
        ],
        seed=137,
    )

    result = pipeline(**{target: data})
    replay_params = result["replay"]["transforms"][0]["params"]
    replayed = A.ReplayCompose.replay(result["replay"], **{target: data})

    assert len(replay_params["scale_factors"]) == 3
    assert len(replay_params["shift_values"]) == 3
    assert replay_params["scale_factors"][2] != 1.0
    assert replay_params["shift_values"][2] != 0.0
    np.testing.assert_array_equal(replayed[target], result[target])


@pytest.mark.parametrize("residual_mode", ["preserve", "augment"])
@pytest.mark.parametrize("dense_tissue", [False, True])
def test_hestain_residual_modes_respect_tissue_mask(residual_mode: str, dense_tissue: bool) -> None:
    if dense_tissue:
        image = np.full((12, 10, 3), np.array([0.2, 0.3, 0.4]), dtype=np.float32)
        image[:2] = 1.0
    else:
        image = np.ones((12, 10, 3), dtype=np.float32)
        image[3:9, 2:8] = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    transform = A.HEStain(
        method="custom",
        stain_matrix=CUSTOM_STAIN_MATRIX,
        residual_mode=residual_mode,
        intensity_scale_range=(1.2, 1.2),
        intensity_shift_range=(0.1, 0.1),
        augment_background=False,
        p=1.0,
    )

    result = transform(image=image)["image"]
    tissue_mask = fpixel.get_tissue_mask(image).reshape(image.shape[:2])

    np.testing.assert_array_equal(result[~tissue_mask], image[~tissue_mask])
    assert not np.allclose(result[tissue_mask], image[tissue_mask])


def test_hestain_uses_custom_stain_matrix() -> None:
    image = TestDataFactory.create_image((32, 24, 3), dtype=np.uint8, seed=137)
    transform = A.HEStain(
        method="custom",
        stain_matrix=CUSTOM_STAIN_MATRIX,
        intensity_scale_range=(1.2, 1.2),
        intensity_shift_range=(0.05, 0.05),
        augment_background=True,
        p=1.0,
    )

    result = transform(image=image)

    np.testing.assert_array_equal(transform.get_applied_params()["stain_matrix"], CUSTOM_STAIN_MATRIX)
    expected = fpixel.apply_he_stain_augmentation(
        img=image,
        stain_matrix=CUSTOM_STAIN_MATRIX,
        scale_factors=np.array([1.2, 1.2]),
        shift_values=np.array([0.05, 0.05]),
        augment_background=True,
    )
    np.testing.assert_array_equal(result["image"], expected)


def test_hestain_custom_method_requires_stain_matrix() -> None:
    with pytest.raises(ValueError, match="stain_matrix is required"):
        A.HEStain(method="custom")


def test_hestain_rejects_stain_matrix_for_non_custom_method() -> None:
    with pytest.raises(ValueError, match="stain_matrix is only valid"):
        A.HEStain(method="preset", stain_matrix=CUSTOM_STAIN_MATRIX)


def test_hestain_custom_method_rejects_preset() -> None:
    with pytest.raises(ValueError, match="preset should not be specified"):
        A.HEStain(method="custom", preset="standard", stain_matrix=CUSTOM_STAIN_MATRIX)


def test_hestain_converts_custom_stain_matrix_to_owned_float32_array() -> None:
    stain_matrix = [[0.71, 0.65, 0.27], [0.18, 0.91, 0.37]]

    transform = A.HEStain(method="custom", stain_matrix=stain_matrix)
    stain_matrix[0][0] = 137

    assert transform.stain_matrix is not None
    assert transform.stain_matrix.dtype == np.float32
    np.testing.assert_allclose(
        transform.stain_matrix,
        np.array([[0.71, 0.65, 0.27], [0.18, 0.91, 0.37]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("stain_matrix", "message"),
    [
        (np.ones((3, 3)), "shape"),
        (np.array([[0.71, np.nan, 0.27], [0.18, 0.91, 0.37]]), "finite"),
        (np.array([[0.0, 0.0, 0.0], [0.18, 0.91, 0.37]]), "non-zero"),
        (np.array([[0.71, 0.65, 0.27], [1.42, 1.30, 0.54]]), "linearly independent"),
    ],
)
def test_hestain_rejects_invalid_custom_stain_matrix(stain_matrix: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        A.HEStain(method="custom", stain_matrix=stain_matrix)


def test_hestain_custom_stain_matrix_survives_json_serialization() -> None:
    pipeline = A.Compose(
        [A.HEStain(method="custom", stain_matrix=CUSTOM_STAIN_MATRIX, p=1.0)],
        seed=137,
        strict=True,
    )

    serialized = json.loads(json.dumps(A.to_dict(pipeline), allow_nan=False))
    restored = A.from_dict(serialized)

    restored_transform = restored.transforms[0]
    assert isinstance(restored_transform, A.HEStain)
    assert restored_transform.stain_matrix is not None
    assert restored_transform.stain_matrix.dtype == np.float32
    np.testing.assert_array_equal(restored_transform.stain_matrix, CUSTOM_STAIN_MATRIX)


@pytest.mark.parametrize("residual_mode", ["preserve", "augment"])
def test_hestain_residual_mode_survives_json_serialization(residual_mode: str) -> None:
    pipeline = A.Compose(
        [
            A.HEStain(
                method="custom",
                stain_matrix=CUSTOM_STAIN_MATRIX,
                residual_mode=residual_mode,
                p=1.0,
            ),
        ],
        seed=137,
        strict=True,
    )

    serialized = json.loads(json.dumps(A.to_dict(pipeline), allow_nan=False))
    restored = A.from_dict(serialized)

    restored_transform = restored.transforms[0]
    assert isinstance(restored_transform, A.HEStain)
    assert restored_transform.residual_mode == residual_mode


def test_hestain_rejects_invalid_residual_mode() -> None:
    with pytest.raises(ValueError, match="residual_mode"):
        A.HEStain(residual_mode="invalid")


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("residual_mode", ["project", "preserve", "augment"])
@pytest.mark.parametrize(
    ("target", "shape"),
    [
        ("image", (12, 10, 3)),
        ("images", (2, 12, 10, 3)),
        ("volume", (2, 12, 10, 3)),
        ("volumes", (2, 2, 12, 10, 3)),
    ],
)
def test_hestain_custom_matrix_supports_all_image_targets(
    target: str,
    shape: tuple[int, ...],
    dtype: type[np.generic],
    residual_mode: str,
) -> None:
    data = TestDataFactory.create_image(shape, dtype=dtype, seed=137)
    transform = A.Compose(
        [
            A.HEStain(
                method="custom",
                stain_matrix=CUSTOM_STAIN_MATRIX,
                residual_mode=residual_mode,
                intensity_scale_range=(1.1, 1.1),
                intensity_shift_range=(0.05, 0.05),
                augment_background=True,
                p=1.0,
            ),
        ],
        strict=True,
        seed=137,
    )

    result = transform(**{target: data})[target]

    assert result.shape == shape
    assert result.dtype == dtype
    assert np.isfinite(result).all()
