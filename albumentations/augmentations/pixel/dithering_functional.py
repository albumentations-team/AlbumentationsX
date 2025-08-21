"""Functional implementations of dithering algorithms for color depth reduction."""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np
from albucore import float32_io

from albumentations.augmentations.pixel.functional import to_gray_average, to_gray_weighted_average

# Error diffusion kernels
FLOYD_STEINBERG_KERNEL = {
    "offsets": [(1, 0), (-1, 1), (0, 1), (1, 1)],
    "weights": [7 / 16, 3 / 16, 5 / 16, 1 / 16],
}

JARVIS_KERNEL = {
    "offsets": [
        (1, 0),
        (2, 0),
        (-2, 1),
        (-1, 1),
        (0, 1),
        (1, 1),
        (2, 1),
        (-2, 2),
        (-1, 2),
        (0, 2),
        (1, 2),
        (2, 2),
    ],
    "weights": [
        7 / 48,
        5 / 48,
        3 / 48,
        5 / 48,
        7 / 48,
        5 / 48,
        3 / 48,
        1 / 48,
        3 / 48,
        5 / 48,
        3 / 48,
        1 / 48,
    ],
}

STUCKI_KERNEL = {
    "offsets": [
        (1, 0),
        (2, 0),
        (-2, 1),
        (-1, 1),
        (0, 1),
        (1, 1),
        (2, 1),
        (-2, 2),
        (-1, 2),
        (0, 2),
        (1, 2),
        (2, 2),
    ],
    "weights": [
        8 / 42,
        4 / 42,
        2 / 42,
        4 / 42,
        8 / 42,
        4 / 42,
        2 / 42,
        1 / 42,
        2 / 42,
        4 / 42,
        2 / 42,
        1 / 42,
    ],
}

ATKINSON_KERNEL = {
    "offsets": [
        (1, 0),
        (2, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (0, 2),
    ],
    "weights": [
        1 / 8,
        1 / 8,
        1 / 8,
        1 / 8,
        1 / 8,
        1 / 8,
    ],
}

BURKES_KERNEL = {
    "offsets": [
        (1, 0),
        (2, 0),
        (-2, 1),
        (-1, 1),
        (0, 1),
        (1, 1),
        (2, 1),
    ],
    "weights": [
        8 / 32,
        4 / 32,
        2 / 32,
        4 / 32,
        8 / 32,
        4 / 32,
        2 / 32,
    ],
}

SIERRA_KERNEL = {
    "offsets": [
        (1, 0),
        (2, 0),
        (-2, 1),
        (-1, 1),
        (0, 1),
        (1, 1),
        (2, 1),
        (-1, 2),
        (0, 2),
        (1, 2),
    ],
    "weights": [
        5 / 32,
        3 / 32,
        2 / 32,
        4 / 32,
        5 / 32,
        4 / 32,
        2 / 32,
        2 / 32,
        3 / 32,
        2 / 32,
    ],
}

SIERRA_2ROW_KERNEL = {
    "offsets": [
        (1, 0),
        (2, 0),
        (-2, 1),
        (-1, 1),
        (0, 1),
        (1, 1),
        (2, 1),
    ],
    "weights": [
        4 / 16,
        3 / 16,
        1 / 16,
        2 / 16,
        3 / 16,
        2 / 16,
        1 / 16,
    ],
}

SIERRA_LITE_KERNEL = {
    "offsets": [
        (1, 0),
        (-1, 1),
        (0, 1),
    ],
    "weights": [
        2 / 4,
        1 / 4,
        1 / 4,
    ],
}

ERROR_DIFFUSION_KERNELS = {
    "floyd_steinberg": FLOYD_STEINBERG_KERNEL,
    "jarvis": JARVIS_KERNEL,
    "stucki": STUCKI_KERNEL,
    "atkinson": ATKINSON_KERNEL,
    "burkes": BURKES_KERNEL,
    "sierra": SIERRA_KERNEL,
    "sierra_2row": SIERRA_2ROW_KERNEL,
    "sierra_lite": SIERRA_LITE_KERNEL,
}


def generate_bayer_matrix(size: int) -> np.ndarray:
    """Generate Bayer threshold matrix of given size (cached).

    Args:
        size: Size of the matrix (2, 4, 8, or 16).

    Returns:
        Bayer matrix normalized to [0, 1] range.

    """
    if size == 2:
        matrix = np.array([[0, 2], [3, 1]], dtype=np.float32)
    elif size == 4:
        matrix = np.array(
            [
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5],
            ],
            dtype=np.float32,
        )
    elif size == 8:
        # Generate 8x8 from 4x4
        base = generate_bayer_matrix(4) * 4
        matrix = np.zeros((8, 8), dtype=np.float32)
        matrix[:4, :4] = base
        matrix[:4, 4:] = base + 2
        matrix[4:, :4] = base + 3
        matrix[4:, 4:] = base + 1
    elif size == 16:
        # Generate 16x16 from 8x8
        base = generate_bayer_matrix(8) * 4
        matrix = np.zeros((16, 16), dtype=np.float32)
        matrix[:8, :8] = base
        matrix[:8, 8:] = base + 2
        matrix[8:, :8] = base + 3
        matrix[8:, 8:] = base + 1
    else:
        msg = f"Unsupported Bayer matrix size: {size}"
        raise ValueError(msg)

    # Normalize to [0, 1]
    return matrix / (size * size)


def quantize_value(value: float, n_levels: int) -> float:
    """Quantize a single value to n discrete levels.

    Args:
        value: Input value in [0, 1] range.
        n_levels: Number of discrete levels.

    Returns:
        Quantized value in [0, 1] range.

    """
    if n_levels == 2:
        return 1.0 if value >= 0.5 else 0.0

    # Scale to [0, n_levels-1], round, then scale back
    scaled = value * (n_levels - 1)
    quantized = round(scaled)
    return quantized / (n_levels - 1)


def quantize_array(arr: np.ndarray, n_levels: int) -> np.ndarray:
    """Quantize an array to n discrete levels efficiently using vectorized operations.

    Args:
        arr: Input array in [0, 1] range.
        n_levels: Number of discrete levels.

    Returns:
        Quantized array in [0, 1] range.

    """
    if n_levels == 2:
        return (arr >= 0.5).astype(np.float32)

    # Vectorized quantization
    # Scale to [0, n_levels-1], round, then scale back
    scaled = arr * (n_levels - 1)
    quantized = np.round(scaled) / (n_levels - 1)
    return quantized.astype(np.float32)


def threshold_dither_uint8(
    img: np.ndarray,
    n_colors: int,
    threshold: int = 127,
) -> np.ndarray:
    """Apply threshold dithering optimized for uint8 images using cv2.LUT.

    Args:
        img: Input uint8 image with shape (H, W, C).
        n_colors: Number of colors per channel.
        threshold: Threshold value in [0, 255].

    Returns:
        Dithered uint8 image.

    """
    if n_colors == 2:
        # Simple binary threshold
        lut = np.zeros(256, dtype=np.uint8)
        lut[threshold:] = 255
    else:
        # Multi-level quantization
        lut = np.zeros(256, dtype=np.uint8)
        step = 256 // n_colors
        for i in range(256):
            level = min(i // step, n_colors - 1)
            lut[i] = level * (255 // (n_colors - 1))

    # Apply LUT to each channel
    return cv2.LUT(img, lut)


@float32_io
def threshold_dither(
    img: np.ndarray,
    n_colors: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """Apply simple threshold dithering.

    Args:
        img: Input image in [0, 1] range with shape (H, W, C).
        n_colors: Number of colors per channel.
        threshold: Threshold value in [0, 1].

    Returns:
        Dithered image in [0, 1] range.

    """
    if n_colors == 2:
        return (img >= threshold).astype(np.float32)

    # For multi-level, use efficient array quantization
    result = np.empty_like(img)
    for channel_idx in range(img.shape[2]):
        channel = img[:, :, channel_idx]
        result[:, :, channel_idx] = quantize_array(channel, n_colors)

    return result


@float32_io
def random_dither(
    img: np.ndarray,
    n_colors: int,
    noise_range: tuple[float, float],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Apply random noise dithering.

    Args:
        img: Input image in [0, 1] range with shape (H, W, C).
        n_colors: Number of colors per channel.
        noise_range: Range of noise to add.
        random_generator: Random number generator.

    Returns:
        Dithered image in [0, 1] range.

    """
    # Add random noise
    noise = random_generator.uniform(noise_range[0], noise_range[1], size=img.shape)
    noisy = np.clip(img + noise, 0, 1)

    # Quantize
    if n_colors == 2:
        return (noisy >= 0.5).astype(np.float32)

    result = np.empty_like(img)
    for channel_idx in range(img.shape[2]):
        channel = noisy[:, :, channel_idx]
        result[:, :, channel_idx] = quantize_array(channel, n_colors)

    return result


def ordered_dither_uint8(
    img: np.ndarray,
    n_colors: int,
    matrix_size: int = 4,
) -> np.ndarray:
    """Apply ordered dithering optimized for uint8 images.

    Args:
        img: Input uint8 image with shape (H, W, C).
        n_colors: Number of colors per channel.
        matrix_size: Size of Bayer matrix (2, 4, 8, or 16).

    Returns:
        Dithered uint8 image.

    """
    # Generate Bayer matrix scaled to [0, 255]
    bayer = (generate_bayer_matrix(matrix_size) * 255).astype(np.uint8)

    # Tile the matrix to cover the image
    height, width = img.shape[:2]
    tiles_height = (height + matrix_size - 1) // matrix_size
    tiles_width = (width + matrix_size - 1) // matrix_size
    tiled = np.tile(bayer, (tiles_height, tiles_width))[:height, :width]

    if n_colors == 2:
        # Binary dithering - compare with tiled threshold
        result = np.zeros_like(img)
        for channel_idx in range(img.shape[2]):
            result[:, :, channel_idx] = (img[:, :, channel_idx] > tiled) * 255
        return result.astype(np.uint8)
    # Multi-level: Create LUT for each threshold level
    result = np.zeros_like(img)
    levels = np.linspace(0, 255, n_colors).astype(np.uint8)

    for channel_idx in range(img.shape[2]):
        channel = img[:, :, channel_idx]
        # Add dither pattern and quantize
        dithered = channel.astype(np.int16) + (tiled.astype(np.int16) - 128) // n_colors
        dithered = np.clip(dithered, 0, 255)

        # Quantize to n_colors levels
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            level_idx = min(i * n_colors // 256, n_colors - 1)
            lut[i] = levels[level_idx]

        result[:, :, channel_idx] = cv2.LUT(dithered.astype(np.uint8), lut)

    return result


@float32_io
def ordered_dither(
    img: np.ndarray,
    n_colors: int,
    matrix_size: int = 4,
) -> np.ndarray:
    """Apply ordered dithering using Bayer matrix.

    Args:
        img: Input image in [0, 1] range with shape (H, W, C).
        n_colors: Number of colors per channel.
        matrix_size: Size of Bayer matrix (2, 4, 8, or 16).

    Returns:
        Dithered image in [0, 1] range.

    """
    # Generate Bayer matrix
    bayer = generate_bayer_matrix(matrix_size)

    # Tile the matrix to cover the image
    height, width = img.shape[:2]
    tiles_height = (height + matrix_size - 1) // matrix_size
    tiles_width = (width + matrix_size - 1) // matrix_size
    tiled = np.tile(bayer, (tiles_height, tiles_width))[:height, :width]

    # Expand for color channels
    tiled = np.expand_dims(tiled, axis=2)

    # Apply threshold with Bayer matrix
    if n_colors == 2:
        return (img > tiled).astype(np.float32)

    # For multiple levels, use the Bayer matrix to add noise then quantize
    # The Bayer matrix values are in [0, 1], center them around 0
    dither_noise = (tiled - 0.5) / n_colors

    # Add dither noise to the image
    dithered = np.clip(img + dither_noise, 0, 1)

    # Quantize to n_colors levels
    result = np.empty_like(img)
    for channel_idx in range(img.shape[2]):
        channel = dithered[:, :, channel_idx]
        quantized = np.floor(channel * n_colors) / n_colors
        # Ensure the result is in [0, 1] and properly quantized
        quantized = np.clip(quantized, 0, (n_colors - 1) / n_colors)
        result[:, :, channel_idx] = quantized

    return result


@float32_io
def error_diffusion_dither(
    img: np.ndarray,
    n_colors: int,
    algorithm: str = "floyd_steinberg",
    serpentine: bool = False,
) -> np.ndarray:
    """Apply error diffusion dithering.

    Args:
        img: Input image in [0, 1] range with shape (H, W, C).
        n_colors: Number of colors per channel.
        algorithm: Error diffusion algorithm name.
        serpentine: Use serpentine (back-and-forth) scanning.

    Returns:
        Dithered image in [0, 1] range.

    """
    if algorithm not in ERROR_DIFFUSION_KERNELS:
        msg = f"Unknown error diffusion algorithm: {algorithm}"
        raise ValueError(msg)

    kernel = ERROR_DIFFUSION_KERNELS[algorithm]
    offsets: list[tuple[int, int]] = cast("list[tuple[int, int]]", kernel["offsets"])
    weights: list[float] = cast("list[float]", kernel["weights"])

    # Work on a copy
    result = img.copy()
    height, width = img.shape[:2]
    num_channels = img.shape[2]

    # Process each channel independently
    for channel_idx in range(num_channels):
        channel = result[:, :, channel_idx].copy()

        # Process pixels
        for row_idx in range(height):
            # Determine scan direction
            if serpentine and row_idx % 2 == 1:
                col_range = range(width - 1, -1, -1)
                x_offsets = [(-ox, oy) for ox, oy in offsets]
            else:
                col_range = range(width)
                x_offsets = offsets

            for col_idx in col_range:
                # Get current pixel value
                old_val = channel[row_idx, col_idx]

                # Quantize
                new_val = quantize_value(old_val, n_colors)
                channel[row_idx, col_idx] = new_val

                # Calculate error
                error = old_val - new_val

                # Distribute error to neighbors
                for (offset_x, offset_y), weight in zip(x_offsets, weights):
                    neighbor_x, neighbor_y = col_idx + offset_x, row_idx + offset_y
                    if 0 <= neighbor_x < width and 0 <= neighbor_y < height:
                        channel[neighbor_y, neighbor_x] += error * weight

        # Clip to valid range
        channel = np.clip(channel, 0, 1)
        result[:, :, channel_idx] = channel

    return result


def _apply_dithering_to_grayscale(
    img: np.ndarray,
    method: str,
    n_colors: int,
    **kwargs: Any,
) -> np.ndarray:
    """Apply dithering to grayscale image."""
    # Convert to grayscale
    if img.shape[2] == 3:
        gray = to_gray_weighted_average(img)
    elif img.shape[2] == 1:
        gray = img
    else:
        gray = to_gray_average(img)

    # Ensure gray has shape (H, W, 1)
    if gray.ndim == 2:
        gray = gray[:, :, np.newaxis]

    # Apply dithering method
    dithered = _apply_single_dithering_method(gray, method, n_colors, **kwargs)

    # Expand back to RGB (handle both 2D and 3D cases)
    if dithered.ndim == 2:
        dithered = dithered[:, :, np.newaxis]
    return np.repeat(dithered, 3, axis=2)


def _apply_single_dithering_method(
    img: np.ndarray,
    method: str,
    n_colors: int,
    **kwargs: Any,
) -> np.ndarray:
    """Apply a single dithering method to an image."""
    # Choose optimized uint8 versions when possible
    if img.dtype == np.uint8 and method in ["threshold", "ordered"]:
        if method == "threshold":
            threshold_uint8 = int(kwargs.get("threshold", 0.5) * 255)
            return threshold_dither_uint8(img, n_colors, threshold_uint8)
        if method == "ordered":
            return ordered_dither_uint8(img, n_colors, kwargs.get("matrix_size", 4))

    # Use float32 versions
    if method == "threshold":
        return threshold_dither(img, n_colors, kwargs.get("threshold", 0.5))
    if method == "random":
        return random_dither(
            img,
            n_colors,
            kwargs.get("noise_range", (-0.5, 0.5)),
            kwargs.get("random_generator"),
        )
    if method == "ordered":
        return ordered_dither(img, n_colors, kwargs.get("matrix_size", 4))
    if method == "error_diffusion":
        return error_diffusion_dither(
            img,
            n_colors,
            kwargs.get("error_diffusion_algorithm", "floyd_steinberg"),
            kwargs.get("serpentine", False),
        )

    msg = f"Unknown dithering method: {method}"
    raise ValueError(msg)


def apply_dithering(
    img: np.ndarray,
    method: str,
    n_colors: int,
    color_mode: str = "per_channel",
    **kwargs: Any,
) -> np.ndarray:
    """Apply dithering to an image.

    Args:
        img: Input image in [0, 1] range with shape (H, W, C).
        method: Dithering method to use.
        n_colors: Number of colors per channel.
        color_mode: How to handle colors ("grayscale", "per_channel", "rgb").
        **kwargs: Additional parameters for specific methods.

    Returns:
        Dithered image in [0, 1] range with shape (H, W, C).

    """
    if color_mode == "grayscale":
        return _apply_dithering_to_grayscale(img, method, n_colors, **kwargs)

    # Apply dithering directly (per_channel mode)
    return _apply_single_dithering_method(img, method, n_colors, **kwargs)
