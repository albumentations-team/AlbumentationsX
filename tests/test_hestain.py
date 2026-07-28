import json

import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.pixel import functional as fpixel
from tests.helpers import TestDataFactory

CUSTOM_STAIN_MATRIX = np.array([[0.71, 0.65, 0.27], [0.18, 0.91, 0.37]], dtype=np.float32)


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


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
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
) -> None:
    data = TestDataFactory.create_image(shape, dtype=dtype, seed=137)
    transform = A.Compose(
        [
            A.HEStain(
                method="custom",
                stain_matrix=CUSTOM_STAIN_MATRIX,
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
