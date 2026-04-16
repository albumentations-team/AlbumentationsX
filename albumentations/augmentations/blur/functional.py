"""Functional implementations of various blur operations for image processing.

This module provides a collection of low-level functions for applying different blur effects
to images, including standard blur, glass blur, defocus, and zoom effects.
These functions form the foundation for the corresponding transform classes.
"""

import random
from collections.abc import Sequence
from itertools import product
from math import ceil
from typing import Literal
from warnings import warn

import cv2
import numpy as np
from albucore import (
    clipped,
    float32_io,
    preserve_channel_dim,
    reduce_sum,
    uint8_io,
)
from pydantic import ValidationInfo
from scipy.stats import mode as scipy_mode

from albumentations.augmentations.geometric.functional import scale
from albumentations.augmentations.pixel.functional import convolve
from albumentations.core.type_definitions import EIGHT, ImageType

__all__ = ["box_blur", "central_zoom", "defocus", "glass_blur", "mode_filter", "zoom_blur"]


@preserve_channel_dim
def box_blur(img: ImageType, ksize: int) -> ImageType:
    """Smooth image with uniform rectangular kernel (moving average). ksize sets size. Use for
    mild noise reduction or downscale prep.

    This function applies a blur to an image.

    Args:
        img (ImageType): Input image.
        ksize (int): Kernel size.

    Returns:
        ImageType: Blurred image.

    """
    img = np.array(img, copy=True, order="C")
    cv2.blur(img, (ksize, ksize), dst=img)
    return img


@preserve_channel_dim
def glass_blur(
    img: ImageType,
    sigma: float,
    max_delta: int,
    iterations: int,
    dxy: np.ndarray,
    mode: Literal["fast", "exact"],
) -> ImageType:
    """Glass-like effect: Gaussian blur then random pixel swaps. Sigma, max_delta, iterations, dxy.
    Use for frosted-glass look.

    This function applies a glass blur to an image.

    Args:
        img (ImageType): Input image.
        sigma (float): Sigma.
        max_delta (int): Maximum delta.
        iterations (int): Number of iterations.
        dxy (np.ndarray): Dxy.
        mode (Literal['fast', 'exact']): Mode.

    Returns:
        ImageType: Glass blurred image.

    """
    x = cv2.GaussianBlur(np.array(img), sigmaX=sigma, ksize=(0, 0))

    if mode == "fast":
        hs = np.arange(img.shape[0] - max_delta, max_delta, -1)
        ws = np.arange(img.shape[1] - max_delta, max_delta, -1)
        h: int | np.ndarray = np.tile(hs, ws.shape[0])
        w: int | np.ndarray = np.repeat(ws, hs.shape[0])

        for i in range(iterations):
            dy = dxy[:, i, 0]
            dx = dxy[:, i, 1]
            x[h, w], x[h + dy, w + dx] = x[h + dy, w + dx], x[h, w]

    elif mode == "exact":
        for ind, (i, h, w) in enumerate(
            product(
                range(iterations),
                range(img.shape[0] - max_delta, max_delta, -1),
                range(img.shape[1] - max_delta, max_delta, -1),
            ),
        ):
            idx = ind if ind < len(dxy) else ind % len(dxy)
            dy = dxy[idx, i, 0]
            dx = dxy[idx, i, 1]
            x[h, w], x[h + dy, w + dx] = x[h + dy, w + dx], x[h, w]
    else:
        raise ValueError(f"Unsupported mode `{mode}`. Supports only `fast` and `exact`.")

    return cv2.GaussianBlur(x, sigmaX=sigma, ksize=(0, 0))


def create_defocus_kernel(radius: int, alias_blur: float) -> np.ndarray:
    """Create defocus (aliased disk) convolution kernel. radius, alias_blur control disk
    shape and smoothing. Returns kernel for convolve.
    """
    length = np.arange(-max(8, radius), max(8, radius) + 1)
    ksize = 3 if radius <= EIGHT else 5

    x, y = np.meshgrid(length, length)
    aliased_disk = np.array((x**2 + y**2) <= radius**2, dtype=np.float32)
    aliased_disk /= reduce_sum(aliased_disk)

    return cv2.GaussianBlur(aliased_disk, (ksize, ksize), sigmaX=alias_blur)


def defocus(img: ImageType, radius: int, alias_blur: float) -> ImageType:
    """Blur with aliased disk kernel to simulate out-of-focus. radius, alias_blur set size and
    softness. Use for depth-of-field or bokeh-style effects.
    """
    return convolve(img, kernel=create_defocus_kernel(radius, alias_blur))


def central_zoom(img: ImageType, zoom_factor: int) -> ImageType:
    """Zoom from center by integer factor: crop center, upsample, trim to original size.
    Used in zoom-blur pipeline; zoom_factor must be positive.

    This function zooms an image.

    Args:
        img (ImageType): Input image.
        zoom_factor (int): Zoom factor.

    Returns:
        ImageType: Zoomed image.

    """
    height, width = img.shape[:2]
    h_ch, w_ch = ceil(height / zoom_factor), ceil(width / zoom_factor)
    h_top, w_top = (height - h_ch) // 2, (width - w_ch) // 2

    img = scale(img[h_top : h_top + h_ch, w_top : w_top + w_ch], zoom_factor, cv2.INTER_LINEAR)
    h_trim_top, w_trim_top = (img.shape[0] - height) // 2, (img.shape[1] - width) // 2
    return img[h_trim_top : h_trim_top + height, w_trim_top : w_trim_top + width]


@float32_io
@clipped
def zoom_blur(img: ImageType, zoom_factors: np.ndarray | Sequence[int]) -> ImageType:
    """Radial zoom blur: blend image with center-zoomed copies. zoom_factors; normalized result.
    Use for motion or out-of-focus style. Float32 I/O, clipped.

    This function zooms and blurs an image.

    Args:
        img (ImageType): Input image.
        zoom_factors (np.ndarray | Sequence[int]): Zoom factors.

    Returns:
        ImageType: Zoomed and blurred image.

    """
    out = np.zeros_like(img, dtype=np.float32)

    for zoom_factor in zoom_factors:
        out += central_zoom(img, zoom_factor)

    return (img + out) * np.float32(1.0 / (len(zoom_factors) + 1))


@preserve_channel_dim
@uint8_io
def mode_filter(img: ImageType, kernel_size: int) -> ImageType:
    """Replace each pixel with the most frequent value (mode) in its local square neighborhood,
    computed per channel; ties broken by smallest value (scipy default).

    Args:
        img (ImageType): Input image (uint8 after @uint8_io conversion).
        kernel_size (int): Side length of the square neighborhood (must be odd, ≥ 3).

    Returns:
        ImageType: Filtered image with the same shape and dtype as the input.

    """
    pad = kernel_size // 2
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    # Slide a (kernel_size, kernel_size, 1) window over the padded HWC image.
    # Output shape: (H, W, C, kernel_size, kernel_size, 1)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size, 1))
    # Flatten neighborhood into trailing axis: (H, W, C, kernel_size * kernel_size)
    flat = windows.reshape(windows.shape[0], windows.shape[1], windows.shape[2], -1)
    return scipy_mode(flat, axis=-1, keepdims=False).mode.astype(img.dtype, copy=False)


def _ensure_min_value(result: tuple[int, int], min_value: int, field_name: str | None) -> tuple[int, int]:
    if result[0] < min_value or result[1] < min_value:
        new_result = (max(min_value, result[0]), max(min_value, result[1]))
        warn(
            f"{field_name}: Invalid kernel size range {result}. "
            f"Values less than {min_value} are not allowed. "
            f"Range automatically adjusted to {new_result}.",
            UserWarning,
            stacklevel=2,
        )
        return new_result
    return result


def _ensure_odd_values(result: tuple[int, int], field_name: str | None = None) -> tuple[int, int]:
    new_result = (
        result[0] if result[0] == 0 or result[0] % 2 == 1 else result[0] + 1,
        result[1] if result[1] == 0 or result[1] % 2 == 1 else result[1] + 1,
    )
    if new_result != result:
        warn(
            f"{field_name}: Non-zero kernel sizes must be odd. Range {result} automatically adjusted to {new_result}.",
            UserWarning,
            stacklevel=2,
        )
    return new_result


def process_blur_limit(value: int | tuple[int, int], info: ValidationInfo, min_value: int = 0) -> tuple[int, int]:
    """Process blur limit to valid kernel sizes (min, odd). Converts int or tuple to
    (min, max); enforces constraints. For blur InitSchema validators.
    """
    # Convert value to tuple[int, int]
    if isinstance(value, Sequence):
        if len(value) != 2:
            raise ValueError("Sequence must contain exactly 2 elements")
        result = (int(value[0]), int(value[1]))
    else:
        result = (min_value, int(value))

    result = _ensure_min_value(result, min_value, info.field_name)
    result = _ensure_odd_values(result, info.field_name)

    if result[0] > result[1]:
        final_result = (result[1], result[1])
        warn(
            f"{info.field_name}: Invalid range {result} (min > max). Range automatically adjusted to {final_result}.",
            UserWarning,
            stacklevel=2,
        )
        return final_result

    return result


def create_motion_kernel(
    kernel_size: int,
    angle: float,
    direction: float,
    allow_shifted: bool,
    random_state: random.Random,
) -> np.ndarray:
    """Create motion blur kernel (2D float32). kernel_size (odd), angle, direction (-1 to 1),
    allow_shifted, random_state. Returns normalized kernel.

    Args:
        kernel_size (int): Size of the kernel (must be odd)
        angle (float): Angle in degrees (counter-clockwise)
        direction (float): Blur direction (-1.0 to 1.0)
        allow_shifted (bool): Allow kernel to be randomly shifted from center
        random_state (random.Random): Python's random.Random instance

    Returns:
        np.ndarray: Motion blur kernel

    """
    # Validate direction range to prevent unexpected interpolation results
    direction = np.clip(direction, -1.0, 1.0)

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2

    # Convert angle to radians
    angle_rad = np.deg2rad(angle)

    # Calculate direction vector
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    # Create line points with direction bias
    line_length = kernel_size // 2

    # Apply direction bias to control the distribution of blur
    if direction < 0:
        # Backward bias: interpolate between symmetric and backward-only
        # direction = -1: only backward, direction = 0: symmetric
        bias_factor = abs(direction)
        t_start = float(-line_length)
        t_end = line_length * (1 - bias_factor)
    elif direction > 0:
        # Forward bias: interpolate between symmetric and forward-only
        # direction = 1: only forward, direction = 0: symmetric
        bias_factor = direction
        t_start = -line_length * (1 - bias_factor)
        t_end = float(line_length)
    else:
        # Symmetric case (direction = 0)
        t_start = float(-line_length)
        t_end = float(line_length)

    # Generate points along the biased line
    t = np.linspace(t_start, t_end, kernel_size)

    # Generate line coordinates
    x = center + dx * t
    y = center + dy * t

    # Apply random shift if allowed
    if allow_shifted:
        shift_x = random_state.uniform(-1, 1) * line_length / 2
        shift_y = random_state.uniform(-1, 1) * line_length / 2
        x += shift_x
        y += shift_y

    # Round coordinates and clip to kernel bounds
    x = np.clip(np.round(x), 0, kernel_size - 1).astype(int)
    y = np.clip(np.round(y), 0, kernel_size - 1).astype(int)

    # Keep only unique points to avoid multiple assignments
    points = np.unique(np.column_stack([y, x]), axis=0)
    kernel[points[:, 0], points[:, 1]] = 1

    # Ensure at least one point is set
    if not kernel.any():
        kernel[center, center] = 1

    return kernel


def sample_odd_from_range(random_state: random.Random, low: int, high: int) -> int:
    """Sample odd number from [low, high] (inclusive). Low/high normalized to odd (min 3).
    For blur transforms when sampling kernel size from a range.

    Args:
        random_state (random.Random): instance of random.Random
        low (int): lower bound (will be converted to nearest valid odd number)
        high (int): upper bound (will be converted to nearest valid odd number)

    Returns:
        int: Randomly sampled odd number from the range

    Note:
        - Input values will be converted to nearest valid odd numbers:
          * Values less than 3 will become 3
          * Even values will be rounded up to next odd number
        - After normalization, high must be >= low

    """
    # Normalize low value
    low = max(3, low + (low % 2 == 0))
    # Normalize high value
    high = max(3, high + (high % 2 == 0))

    # Ensure high >= low after normalization
    high = max(high, low)

    if low == high:
        return low

    # Calculate number of possible odd values
    num_odd_values = (high - low) // 2 + 1
    # Generate random index and convert to corresponding odd number
    rand_idx = random_state.randint(0, num_odd_values - 1)
    return low + (2 * rand_idx)


def create_gaussian_kernel(sigma: float, ksize: int = 0) -> np.ndarray:
    """Create 2D Gaussian kernel (PIL-style). Sigma and ksize (0 = auto). Returns normalized
    float32 kernel for separable or 2D convolution.

    Args:
        sigma (float): Standard deviation for Gaussian kernel.
        ksize (int): Kernel size. If 0, size is computed as int(sigma * 3.5) * 2 + 1
               to match PIL's implementation. Otherwise, must be positive and odd.

    Returns:
        np.ndarray: 2D normalized Gaussian kernel.

    """
    # PIL's kernel creation approach
    size = int(sigma * 3.5) * 2 + 1 if ksize == 0 else ksize

    # Ensure odd size
    size = size + 1 if size % 2 == 0 else size

    # Create x coordinates
    x = np.linspace(-(size // 2), size // 2, size)

    # Compute 1D kernel using vectorized operations
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / reduce_sum(kernel_1d)

    # Create 2D kernel
    return kernel_1d[:, np.newaxis] @ kernel_1d[np.newaxis, :]


def create_gaussian_kernel_1d(sigma: float, ksize: int = 0) -> np.ndarray:
    """Create 1D Gaussian kernel (PIL-style). Sigma and ksize (0 = auto). For separable
    Gaussian blur; returns normalized float32 1D array.

    Args:
        sigma (float): Standard deviation for Gaussian kernel.
        ksize (int): Kernel size. If 0, size is computed as int(sigma * 3.5) * 2 + 1
               to match PIL's implementation. Otherwise, must be positive and odd.

    Returns:
        np.ndarray: 1D normalized Gaussian kernel.

    """
    # PIL's kernel creation approach
    size = int(sigma * 3.5) * 2 + 1 if ksize == 0 else ksize

    # Ensure odd size
    size = size + 1 if size % 2 == 0 else size

    # Create x coordinates
    x = create_gaussian_kernel_input_array(size=size)

    # Guard against sigma=0 (would cause division by zero)
    if sigma == 0:
        kernel_1d = np.zeros(size, dtype=np.float32)
        kernel_1d[size // 2] = 1.0
        return kernel_1d

    x_f32 = x.astype(np.float32)
    kernel_1d = np.exp(np.float32(-0.5) * (x_f32 / np.float32(sigma)) ** 2)
    return (kernel_1d / reduce_sum(kernel_1d)).astype(np.float32)


def create_gaussian_kernel_input_array(size: int) -> np.ndarray:
    """1-D x-coordinates -size/2 to size/2 for Gaussian kernel. Piecewise for size < 100
    (faster than np.linspace). Returns float array.

    Piecewise function is needed as equivalent python list comprehension is faster than np.linspace
    for values of size < 100

    Args:
        size (int): kernel size

    Returns:
        np.ndarray: x-coordinate array which will be input for gaussian function that will be used for
        separable gaussian blur

    """
    if size < 100:
        return np.array(list(range(-(size // 2), (size // 2) + 1, 1)))

    return np.linspace(-(size // 2), size // 2, size)
