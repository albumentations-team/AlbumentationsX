import numpy as np
import pytest

import albumentations as A


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [None, 1, 3, 5])
def test_random_gamma_volumes_match_individual_volumes_and_leave_masks3d_unchanged(
    dtype: np.dtype,
    channels: int | None,
) -> None:
    rng = np.random.default_rng(137)
    shape = (3, 2, 4, 5) + (() if channels is None else (channels,))
    volumes = rng.integers(0, 256, shape, dtype=np.uint8) if dtype == np.uint8 else rng.random(shape, dtype=np.float32)
    masks3d = rng.integers(0, 8, (2, 2, 4, 5), dtype=np.uint8)
    transform = A.Compose([A.RandomGamma(gamma_range=(90, 90), p=1)], strict=True, telemetry=False)

    result = transform(volumes=volumes, masks3d=masks3d)
    expected = np.stack([transform(volume=volume)["volume"] for volume in volumes])

    np.testing.assert_array_equal(result["volumes"], expected)
    np.testing.assert_array_equal(result["masks3d"], masks3d)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("brightness_by_max", [False, True])
@pytest.mark.parametrize("ensure_safe_output", [False, True])
@pytest.mark.parametrize(
    ("channels", "brightness", "contrast"),
    [
        (channels, brightness, contrast)
        for channels in (None, 1, 3, 5)
        for brightness, contrast in ((0.2, -0.3), (-0.2, 0.3), (0.2, 0.0), (-0.2, 0.0))
    ]
    + [(None, 0.0, -1.0), (1, 0.0, -1.0), (3, 0.0, -1.0)],
)
def test_brightness_contrast_volumes_match_individual_volumes(
    dtype: np.dtype,
    channels: int | None,
    brightness_by_max: bool,
    ensure_safe_output: bool,
    brightness: float,
    contrast: float,
) -> None:
    rng = np.random.default_rng(137)
    shape = (3, 2, 4, 5) + (() if channels is None else (channels,))
    volumes = rng.integers(0, 256, shape, dtype=np.uint8) if dtype == np.uint8 else rng.random(shape, dtype=np.float32)
    masks3d = rng.integers(0, 8, (2, 2, 4, 5), dtype=np.uint8)
    transform = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_range=(brightness, brightness),
                contrast_range=(contrast, contrast),
                brightness_by_max=brightness_by_max,
                ensure_safe_output=ensure_safe_output,
                p=1,
            ),
        ],
        strict=True,
        telemetry=False,
    )

    result = transform(volumes=volumes, masks3d=masks3d)
    expected = np.stack([transform(volume=volume)["volume"] for volume in volumes])

    if dtype == np.float32:
        np.testing.assert_allclose(result["volumes"], expected, rtol=1e-6, atol=1e-6)
    else:
        np.testing.assert_array_equal(result["volumes"], expected)
    np.testing.assert_array_equal(result["masks3d"], masks3d)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("transform_cls", [A.RandomGamma, A.RandomBrightnessContrast])
def test_pixel_transforms_keep_empty_volume_collections(dtype: np.dtype, transform_cls: type[A.BasicTransform]) -> None:
    volumes = np.empty((0, 2, 4, 5, 1), dtype=dtype)
    masks3d = np.empty((0, 2, 4, 5), dtype=np.uint8)
    result = A.Compose([transform_cls(p=1)], strict=True, telemetry=False)(volumes=volumes, masks3d=masks3d)

    assert result["volumes"].shape == volumes.shape
    assert result["volumes"].dtype == volumes.dtype
    np.testing.assert_array_equal(result["masks3d"], masks3d)
