import warnings
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image, ImageFilter

import albumentations as A
from albumentations.augmentations.blur import functional as fblur
from albumentations.core.transforms_interface import BasicTransform
from tests.conftest import UINT8_IMAGES


@pytest.mark.parametrize("aug", [A.Blur, A.MedianBlur, A.MotionBlur])
@pytest.mark.parametrize(
    "blur_limit_input, blur_limit_used",
    [[(3, 3), (3, 3)], [(13, 13), (13, 13)]],
)
@pytest.mark.parametrize("image", UINT8_IMAGES)
def test_blur_kernel_generation(
    image: np.ndarray,
    aug: BasicTransform,
    blur_limit_input: tuple[int, int],
    blur_limit_used: tuple[int, int],
) -> None:
    aug = aug(blur_limit=blur_limit_input, p=1)

    assert aug.blur_limit == blur_limit_used
    aug(image=image)["image"]


@pytest.mark.parametrize("val_uint8", [0, 1, 128, 255])
def test_glass_blur_float_uint8_diff_less_than_two(val_uint8: list[int]) -> None:
    x_uint8 = np.zeros((5, 5, 1)).astype(np.uint8)
    x_uint8[2, 2] = val_uint8

    x_float32 = np.zeros((5, 5, 1)).astype(np.float32)
    x_float32[2, 2] = val_uint8 / 255.0

    glassblur = A.GlassBlur(p=1, max_delta=1)
    glassblur.random_generator = np.random.default_rng(0)

    blur_uint8 = glassblur(image=x_uint8)["image"]
    glassblur.random_generator = np.random.default_rng(0)

    blur_float32 = glassblur(image=x_float32)["image"]

    # Before comparison, rescale the blur_float32 to [0, 255]
    diff = np.abs(blur_uint8 - blur_float32 * 255)

    # The difference between the results of float32 and uint8 will be at most 2.
    assert np.all(diff <= 2.0)


@pytest.mark.parametrize("val_uint8", [0, 1, 128, 255])
def test_advanced_blur_float_uint8_diff_less_than_two(val_uint8: list[int]) -> None:
    x_uint8 = np.zeros((5, 5, 1)).astype(np.uint8)
    x_uint8[2, 2] = val_uint8

    x_float32 = np.zeros((5, 5, 1)).astype(np.float32)
    x_float32[2, 2] = val_uint8 / 255.0

    adv_blur = A.AdvancedBlur(blur_limit=(3, 5), p=1)
    adv_blur.set_random_seed(0)

    adv_blur_uint8 = adv_blur(image=x_uint8)["image"]

    adv_blur.set_random_seed(0)
    adv_blur_float32 = adv_blur(image=x_float32)["image"]

    # Before comparison, rescale the adv_blur_float32 to [0, 255]
    diff = np.abs(adv_blur_uint8 - adv_blur_float32 * 255)

    # The difference between the results of float32 and uint8 will be at most 2.
    assert np.all(diff <= 2.0)


@pytest.mark.parametrize(
    "params",
    [
        {"sigma_x_limit": (0.0, 1.0), "sigma_y_limit": (0.0, 1.0)},
        {"beta_limit": (0.1, 0.9)},
        {"beta_limit": (1.1, 8.0)},
    ],
)
def test_advanced_blur_raises_on_incorrect_params(
    params: dict[str, list[float]],
) -> None:
    with pytest.raises(ValueError):
        A.AdvancedBlur(**params)


class MockValidationInfo:
    def __init__(self, field_name: str):
        self.field_name = field_name


@pytest.mark.parametrize(
    ["value", "min_value", "expected", "warning_messages"],
    [
        # Basic valid cases - no warnings
        ((3, 5), 3, (3, 5), []),
        ((0, 3), 0, (0, 3), []),
        (5, 3, (3, 5), []),
        # Adjust values below min_value
        (
            (1, 2),
            3,
            (3, 3),
            [
                "test_field: Invalid kernel size range (1, 2). Values less than 3 are not allowed. Range automatically adjusted to (3, 3).",
            ],
        ),
        # Adjust values below min_value (with automatic odd adjustment)
        (
            (-1, 4),
            0,
            (0, 5),
            [
                "test_field: Non-zero kernel sizes must be odd. Range (0, 4) automatically adjusted to (0, 5)",
                "test_field: Invalid kernel size range (-1, 4). "
                "Values less than 0 are not allowed. Range automatically adjusted to (0, 4).",
            ],
        ),
        # Adjust non-odd values
        (
            (3, 4),
            3,
            (3, 5),
            ["test_field: Non-zero kernel sizes must be odd. Range (3, 4) automatically adjusted to (3, 5)."],
        ),
        (
            (4, 8),
            0,
            (5, 9),
            ["test_field: Non-zero kernel sizes must be odd. Range (4, 8) automatically adjusted to (5, 9)."],
        ),
        # Special case: keep zero values
        (
            (0, 4),
            0,
            (0, 5),
            ["test_field: Non-zero kernel sizes must be odd. Range (0, 4) automatically adjusted to (0, 5)."],
        ),
        # Fix min > max
        (
            (7, 5),
            3,
            (5, 5),
            ["test_field: Invalid range (7, 5) (min > max). Range automatically adjusted to (5, 5)."],
        ),
        # Multiple adjustments
        (
            (2, 4),
            3,
            (3, 5),
            [
                "test_field: Invalid kernel size range (2, 4). "
                "Values less than 3 are not allowed. Range automatically adjusted to (3, 4).",
                "test_field: Non-zero kernel sizes must be odd. Range (3, 4) automatically adjusted to (3, 5).",
            ],
        ),
    ],
)
def test_process_blur_limit(
    value: Any,
    min_value: int,
    expected: tuple[int, int],
    warning_messages: list[str],
) -> None:
    info = MockValidationInfo("test_field")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = fblur.process_blur_limit(value, info, min_value)

        assert result == expected
        assert len(w) == len(warning_messages)


def test_process_blur_limit_sequence_check() -> None:
    """Test that non-sequence values are properly converted to tuples."""
    info = MockValidationInfo("test_field")

    # Test with integer input
    result = fblur.process_blur_limit(5, info, min_value=0)
    assert isinstance(result, tuple)
    assert result == (0, 5)

    # Test with float input
    result = fblur.process_blur_limit(5.0, info, min_value=0)
    assert isinstance(result, tuple)
    assert result == (0, 5)


def compute_sharpness(image: np.ndarray) -> float:
    kernel = np.array(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1],
        ],
    )
    edges = cv2.filter2D(image.astype(np.float32), -1, kernel)
    return np.std(edges)


def test_gaussian_blur_matches_pil():
    # Create a test image with high-frequency details
    image = np.zeros((100, 100, 1), dtype=np.uint8)
    image[::10, :] = 255  # horizontal lines
    image[:, ::10] = 255  # vertical lines

    # Test points
    sigmas = np.linspace(0.2, 10, 50)

    # Get blur progression for PIL
    pil_sharpness = []
    alb_sharpness = []

    pil_image = Image.fromarray(image[:, :, 0])

    for sigma in sigmas:
        # PIL blur
        pil_blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=sigma))
        pil_sharpness.append(compute_sharpness(np.array(pil_blurred)))

        # Albumentations blur
        alb_blurred = A.GaussianBlur(blur_limit=0, sigma_limit=(sigma, sigma), p=1.0)(image=image)["image"]
        alb_sharpness.append(compute_sharpness(alb_blurred))

    # Convert to numpy arrays for easier comparison
    pil_sharpness = np.array(pil_sharpness)
    alb_sharpness = np.array(alb_sharpness)

    # Compare curves directly using absolute differences
    abs_diff = np.abs(pil_sharpness - alb_sharpness)
    mean_diff = np.mean(abs_diff)
    max_diff = np.max(abs_diff)

    # Assert reasonable absolute differences
    assert mean_diff < 10, f"Average absolute difference too high: {mean_diff:.2f}"
    assert max_diff < 83, f"Maximum absolute difference too high: {max_diff:.2f}"


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("num_channels", [1, 3, 5])
@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_mode_filter_output_shape_and_dtype(dtype: np.dtype, num_channels: int, kernel_size: int) -> None:
    rng = np.random.default_rng(137)
    shape = (64, 64, num_channels)
    if dtype == np.uint8:
        image = rng.integers(0, 256, shape, dtype=np.uint8)
    else:
        image = rng.random(shape, dtype=np.float32)

    transform = A.Compose([A.ModeFilter(kernel_range=(kernel_size, kernel_size), p=1.0)])
    result = transform(image=image)["image"]

    assert result.shape == image.shape
    assert result.dtype == dtype


def test_mode_filter_constant_image_passes_through() -> None:
    """A constant-valued image should be unchanged: mode of identical values is that value."""
    image = np.full((32, 32, 3), fill_value=137, dtype=np.uint8)
    transform = A.Compose([A.ModeFilter(kernel_range=(5, 5), p=1.0)])
    result = transform(image=image)["image"]
    np.testing.assert_array_equal(result, image)


def test_mode_filter_constant_image_float32_passes_through() -> None:
    image = np.full((32, 32, 3), fill_value=0.5, dtype=np.float32)
    transform = A.Compose([A.ModeFilter(kernel_range=(5, 5), p=1.0)])
    result = transform(image=image)["image"]
    # After uint8_io round-trip (0.5 → 127 → ~0.498), check within quantisation tolerance
    np.testing.assert_allclose(result, image, atol=1.0 / 255)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("num_channels", [1, 3, 5])
@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_mode_filter_apply_to_images_parity(dtype: np.dtype, num_channels: int, kernel_size: int) -> None:
    """Batch processing via images= must match per-image processing."""
    rng = np.random.default_rng(137)
    shape = (32, 32, num_channels)
    if dtype == np.uint8:
        images = rng.integers(0, 256, (4, *shape), dtype=np.uint8)
    else:
        images = rng.random((4, *shape), dtype=np.float32)

    transform = A.Compose([A.ModeFilter(kernel_range=(kernel_size, kernel_size), p=1.0)])

    batch_result = transform(images=images)["images"]
    per_image_results = np.stack([transform(image=img)["image"] for img in images])

    np.testing.assert_array_equal(batch_result, per_image_results)


@pytest.mark.parametrize(
    "kernel_range_input, kernel_range_stored",
    [
        ((3, 3), (3, 3)),
        ((5, 7), (5, 7)),
        ((4, 6), (5, 7)),  # even values bumped to next odd with a UserWarning
    ],
)
def test_mode_filter_kernel_range_stored(
    kernel_range_input: tuple[int, int],
    kernel_range_stored: tuple[int, int],
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        aug = A.ModeFilter(kernel_range=kernel_range_input, p=1.0)
    assert aug.kernel_range == kernel_range_stored
