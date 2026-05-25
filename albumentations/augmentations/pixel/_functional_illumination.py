"""Plasma, illumination, dropout, vignette, flare, and halftone functional helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

from ._functional_noise import (
    DIAMOND_KERNEL,
    SQUARE_KERNEL,
)
from ._functional_shared import (
    MAX_VALUES_BY_DTYPE,
    MONO_CHANNEL_DIMENSIONS,
    MULTICHANNEL_LUT_MEDIUM_IMAGE_PIXELS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    ImageType,
    ImageUInt8,
    add,
    add_array,
    add_weighted,
    apply_multichannel_lut,
    clip,
    clipped,
    cv2,
    float32_io,
    get_num_channels,
    math,
    mean,
    multiply,
    multiply_by_array,
    np,
    reduce_sum,
    sz_lut,
    uint8_io,
)


def _normalize_minmax_float32(src: np.ndarray) -> np.ndarray:
    dst = np.empty(src.shape, dtype=np.float32)
    return cast("np.ndarray", cv2.normalize(src, dst, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F))


def _multiply_scalar_inplace(dst: np.ndarray, value: float) -> None:
    multiply_op = cast("Any", cv2.multiply)
    multiply_op(dst, value, dst=dst)


def _add_scalar_inplace(dst: np.ndarray, value: float) -> None:
    add_op = cast("Any", cv2.add)
    add_op(dst, value, dst=dst)


def generate_plasma_pattern(
    target_shape: tuple[int, int],
    roughness: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate a plasma pattern using diamond-square algorithm. Returns float32
    (H, W) or (H, W, 1) for plasma-based brightness/contrast and shadow.

    This function generates a plasma pattern using the diamond-square algorithm.

    Args:
        target_shape (tuple[int, int]): The shape of the plasma pattern to generate.
        roughness (float): The roughness of the plasma pattern.
        random_generator (np.random.Generator): The random number generator to use.

    Returns:
        np.ndarray: The plasma pattern generated.

    """

    def one_diamond_square_step(current_grid: np.ndarray, noise_scale: float) -> np.ndarray:
        next_height = (current_grid.shape[0] - 1) * 2 + 1
        next_width = (current_grid.shape[1] - 1) * 2 + 1

        # Pre-allocate expanded grid
        expanded_grid = np.zeros((next_height, next_width), dtype=np.float32)

        # Generate all noise at once for both steps (already scaled by noise_scale)
        all_noise = random_generator.uniform(-noise_scale, noise_scale, (next_height, next_width)).astype(np.float32)

        # Copy existing points with noise
        expanded_grid[::2, ::2] = current_grid + all_noise[::2, ::2]

        # Diamond step - keep separate for natural look
        diamond_interpolation = cv2.filter2D(expanded_grid, -1, DIAMOND_KERNEL, borderType=cv2.BORDER_CONSTANT)
        diamond_mask = diamond_interpolation > 0
        expanded_grid += (diamond_interpolation + all_noise) * diamond_mask

        # Square step - keep separate for natural look
        square_interpolation = cv2.filter2D(expanded_grid, -1, SQUARE_KERNEL, borderType=cv2.BORDER_CONSTANT)
        square_mask = square_interpolation > 0
        expanded_grid += (square_interpolation + all_noise) * square_mask

        # Normalize after each step to prevent value drift
        return _normalize_minmax_float32(expanded_grid)

    # Pre-compute noise scales
    max_dimension = max(target_shape)
    power_of_two_size = 2 ** np.ceil(np.log2(max_dimension - 1)) + 1
    total_steps = int(np.log2(power_of_two_size - 1) - 1)
    noise_scales = np.asarray([roughness**i for i in range(total_steps)], dtype=np.float32)

    # Initialize with small random grid
    plasma_grid = random_generator.uniform(-1, 1, (3, 3)).astype(np.float32)

    # Recursively apply diamond-square steps
    for noise_scale in noise_scales:
        plasma_grid = one_diamond_square_step(plasma_grid, noise_scale)

    return np.clip(
        _normalize_minmax_float32(plasma_grid[: target_shape[0], : target_shape[1]]),
        0,
        1,
    )


@clipped
@float32_io
def apply_plasma_brightness_contrast(
    img: ImageType,
    brightness_factor: float,
    contrast_factor: float,
    plasma_pattern: np.ndarray,
) -> ImageType:
    """Modulate brightness and contrast with a plasma pattern. brightness_factor,
    contrast_factor scale effect. Use for spatially varying lighting.

    This function applies plasma-based brightness and contrast adjustments to an image.

    Args:
        img (ImageType): The image to apply the brightness and contrast adjustments to.
        brightness_factor (float): The brightness factor to apply.
        contrast_factor (float): The contrast factor to apply.
        plasma_pattern (np.ndarray): The plasma pattern to use for the brightness and contrast adjustments.

    Returns:
        ImageType: The image with the brightness and contrast adjustments applied.

    """
    # Early return if no adjustments needed
    if brightness_factor == 0 and contrast_factor == 0:
        return img

    img = img.copy()

    # Expand plasma pattern once if needed
    if img.ndim > MONO_CHANNEL_DIMENSIONS:
        plasma_pattern = np.tile(plasma_pattern[..., np.newaxis], (1, 1, img.shape[-1]))

    # Apply brightness adjustment
    if brightness_factor != 0:
        brightness_adjustment = multiply(plasma_pattern, brightness_factor, inplace=False)
        img = add(img, brightness_adjustment, inplace=True)

    # Apply contrast adjustment
    if contrast_factor != 0:
        img_mean = mean(img)
        contrast_weights = multiply(plasma_pattern, contrast_factor, inplace=False) + 1

        img = multiply(img, contrast_weights, inplace=True)

        mean_factor = img_mean * (1.0 - contrast_weights)
        return add(img, mean_factor, inplace=True)

    return img


@clipped
def apply_plasma_shadow(
    img: ImageType,
    intensity: float,
    plasma_pattern: np.ndarray,
) -> ImageType:
    """Darken image in regions defined by a plasma pattern. intensity controls strength. Use for
    soft shadows or vignette-like effects. Same dtype and channels.

    Args:
        img (ImageType): Input image
        intensity (float): Shadow intensity
        plasma_pattern (np.ndarray): Plasma pattern to use

    Returns:
        ImageType: Image with plasma shadow

    """
    # Scale plasma pattern by intensity first (scalar operation)
    scaled_pattern = plasma_pattern * intensity

    # Expand dimensions only once if needed
    if img.ndim > MONO_CHANNEL_DIMENSIONS:
        scaled_pattern = scaled_pattern[..., np.newaxis]

    # Single multiply operation
    return cast("ImageType", img * (1 - scaled_pattern))


def create_directional_gradient(height: int, width: int, angle: float) -> np.ndarray:
    """Create a directional gradient in [0, 1] range. Angle and optional smoothing;
    used for illumination and vignette helpers. Returns (H, W) float32.

    This function creates a directional gradient in the [0, 1] range.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.
        angle (float): The angle of the gradient.

    Returns:
        np.ndarray: The directional gradient.

    """
    # Fast path for horizontal gradients
    if angle == 0:
        gradient = np.empty((height, width), dtype=np.float32)
        gradient[:] = np.linspace(0, 1, width, dtype=np.float32)
        return gradient
    if angle == 180:
        gradient = np.empty((height, width), dtype=np.float32)
        gradient[:] = np.linspace(1, 0, width, dtype=np.float32)
        return gradient

    # Fast path for vertical gradients
    if angle == 90:
        gradient = np.empty((height, width), dtype=np.float32)
        gradient[:] = np.linspace(0, 1, height, dtype=np.float32)[:, np.newaxis]
        return gradient
    if angle == 270:
        gradient = np.empty((height, width), dtype=np.float32)
        gradient[:] = np.linspace(1, 0, height, dtype=np.float32)[:, np.newaxis]
        return gradient

    # Fast path for diagonal gradients using broadcasting
    if angle in (45, 135, 225, 315):
        x = np.linspace(0, 1, width, dtype=np.float32)[None, :]  # Horizontal
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None]  # Vertical

        if angle == 45:  # Bottom-left to top-right
            return _normalize_minmax_float32(x + y)
        if angle == 135:  # Bottom-right to top-left
            return _normalize_minmax_float32((1 - x) + y)
        if angle == 225:  # Top-right to bottom-left
            return _normalize_minmax_float32((1 - x) + (1 - y))
        # angle == 315:  # Top-left to bottom-right
        return _normalize_minmax_float32(x + (1 - y))

    # General case for arbitrary angles using broadcasting
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]  # Column vector
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]  # Row vector

    angle_rad = np.deg2rad(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    _multiply_scalar_inplace(x, cos_a)
    _multiply_scalar_inplace(y, sin_a)

    return x + y


def create_corner_illumination_gradient(
    height: int,
    width: int,
    intensity: float,
    corner: Literal[0, 1, 2, 3],
) -> np.ndarray:
    """Create float32 (H, W) corner illumination map for multiply_by_array
    using Euclidean distance from a selected corner with diagonal scaling.

    The map follows `1 + scale * distance`, where `distance` is the Euclidean distance from each pixel
    to the selected image corner and `scale` combines `intensity` with a diagonal-based normalization
    (numerically aligned with the previous `cv2.distanceTransform` implementation).

    Args:
        height (int): Output height `H`.
        width (int): Output width `W`.
        intensity (float): Signed strength. `0` returns an all-ones map (no effect). Positive values
            increase multipliers with distance from the selected corner (typical vignette-style corner
            darkening when applied via `multiply_by_array`). Negative values invert the radial scaling
            (relative brightening toward the corner vs. the image edges), matching the prior corner
            illumination behavior.
        corner (Literal[0, 1, 2, 3]): Corner that anchors the distance field: `0` top-left, `1`
            top-right, `2` bottom-right, `3` bottom-left.

    Returns:
        np.ndarray: Float32 array with shape `(height, width)`.

    """
    if intensity == 0:
        return np.ones((height, width), dtype=np.float32)

    corners = [(0, 0), (0, width - 1), (height - 1, width - 1), (height - 1, 0)]
    corner_y, corner_x = corners[corner]

    y = np.arange(height, dtype=np.float32)[:, np.newaxis] - corner_y
    x = np.arange(width, dtype=np.float32)[np.newaxis, :] - corner_x

    pattern = x * x + y * y
    cv2.sqrt(pattern, dst=pattern)
    _multiply_scalar_inplace(pattern, -intensity / math.sqrt(height * height + width * width))
    _add_scalar_inplace(pattern, 1.0)
    return pattern


def create_illumination_gradient(
    height: int,
    width: int,
    mode: str,
    params: dict[str, Any],
) -> np.ndarray:
    """Create illumination gradient map (H, W) or (H, W, 1). mode: linear, corner, gaussian. Float32;
    apply via multiply_by_array.

    Returns a float32 gradient that can be applied via multiply_by_array.
    The returned gradient does NOT have a channel dimension.
    """
    intensity = params["intensity"]

    if mode == "linear":
        gradient = create_directional_gradient(height, width, params["angle"])
        _multiply_scalar_inplace(gradient, intensity)
        return gradient

    if mode == "corner":
        return create_corner_illumination_gradient(height, width, intensity, params["corner"])

    # gaussian
    if intensity == 0:
        return np.ones((height, width), dtype=np.float32)
    center = params["center"]
    sigma = params["sigma"]
    center_x = width * center[0]
    center_y = height * center[1]
    sigma2 = 2 * (max(height, width) * sigma) ** 2
    y, x = np.ogrid[:height, :width]
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x -= center_x
    y -= center_y
    cv2.multiply(x, x, dst=x)
    cv2.multiply(y, y, dst=y)
    x = x + y
    _multiply_scalar_inplace(x, -1 / sigma2)
    cv2.exp(x, dst=x)
    _multiply_scalar_inplace(x, intensity)
    _add_scalar_inplace(x, 1.0)
    return x


@float32_io
def apply_linear_illumination(img: ImageType, intensity: float, angle: float) -> ImageType:
    """Apply linear illumination gradient to the image. Adds a signed gradient; intensity and angle
    control direction and strength. float32 I/O.

    Args:
        img (ImageType): Input image
        intensity (float): Illumination intensity
        angle (float): Illumination angle in radians

    Returns:
        ImageType: Image with linear illumination

    """
    height, width = img.shape[:2]
    gradient = create_directional_gradient(height, width, angle)
    _multiply_scalar_inplace(gradient, intensity)

    # Add channel dimension if needed
    if img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
        num_channels = img.shape[2]
        if num_channels > 1:
            gradient = cv2.merge([gradient] * num_channels)
            result = cast("ImageType", cv2.add(img, gradient))
            np.clip(result, 0, 1, out=result)
            return result
        gradient = gradient[..., np.newaxis]

    result = img + gradient
    np.clip(result, 0, 1, out=result)
    return result


@clipped
def apply_corner_illumination(
    img: ImageType,
    intensity: float,
    corner: Literal[0, 1, 2, 3],
) -> ImageType:
    """Apply corner illumination (darkened corners) to the image. Gradient from center; intensity and
    corner (0-3) control strength and which corner. Clipped.

    Args:
        img (ImageType): Input image
        intensity (float): Illumination intensity
        corner (Literal[0, 1, 2, 3]): The corner to apply the illumination to.

    Returns:
        ImageType: Image with corner illumination applied.

    """
    if intensity == 0:
        return img.copy()

    height, width = img.shape[:2]

    pattern = create_corner_illumination_gradient(height, width, intensity, corner)

    if img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
        num_channels = img.shape[2]
        pattern = pattern[..., np.newaxis] if num_channels == 1 else cv2.merge([pattern] * num_channels)

    return multiply_by_array(img, pattern)


@clipped
def apply_gaussian_illumination(
    img: ImageType,
    intensity: float,
    center: tuple[float, float],
    sigma: float,
) -> ImageType:
    """Add a Gaussian-shaped bright or dark spot; intensity, center, and sigma define the falloff.
    Use for spotlight or vignette effects. Clipped. Same channel count.

    Args:
        img (ImageType): Input image
        intensity (float): Illumination intensity
        center (tuple[float, float]): The center of the illumination.
        sigma (float): The sigma of the illumination.

    """
    if intensity == 0:
        return img.copy()

    height, width = img.shape[:2]

    # Pre-compute constants
    center_x = width * center[0]
    center_y = height * center[1]
    sigma2 = 2 * (max(height, width) * sigma) ** 2  # Pre-compute denominator

    # Create coordinate grid and calculate distances in-place
    y, x = np.ogrid[:height, :width]
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x -= center_x
    y -= center_y

    # Calculate squared distances in-place
    cv2.multiply(x, x, dst=x)
    cv2.multiply(y, y, dst=y)

    x = x + y

    # Calculate gaussian directly into x array
    _multiply_scalar_inplace(x, -1 / sigma2)
    cv2.exp(x, dst=x)

    # Scale by intensity
    _multiply_scalar_inplace(x, intensity)
    _add_scalar_inplace(x, 1.0)

    if img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
        num_channels = img.shape[2]
        x = x[..., np.newaxis] if num_channels == 1 else cv2.merge([x] * num_channels)

    return multiply_by_array(img, x)


def _auto_contrast_single_channel(
    img: ImageUInt8,
    cutoff: float,
    ignore: int | None,
    method: Literal["cdf", "pil"],
    max_value: int,
) -> ImageUInt8:
    mask = None if ignore is None else (img != ignore)
    hist = cv2.calcHist([img], [0], mask, [256], [0, max_value]).ravel()
    lut = _create_auto_contrast_lut(hist, cutoff, ignore, method, max_value)
    if lut is None:
        return img.copy()

    return sz_lut(img, lut, inplace=False)


def _auto_contrast_pil_zero_cutoff(img: ImageUInt8, max_value: int) -> ImageUInt8:
    """Apply Pillow-style autocontrast for single-channel images without building a
    full histogram when no cutoff or ignored value is requested.
    """
    channel = img if img.ndim == MONO_CHANNEL_DIMENSIONS else img[..., 0]
    min_intensity = int(np.min(channel))
    max_intensity = int(np.max(channel))
    if min_intensity >= max_intensity:
        return img.copy()
    lut = create_contrast_lut(np.empty(0), min_intensity, max_intensity, max_value, "pil")
    result = sz_lut(channel, lut, inplace=False)
    return result if img.ndim == MONO_CHANNEL_DIMENSIONS else result[..., np.newaxis]


def _auto_contrast_multichannel_lut(
    img: ImageUInt8,
    cutoff: float,
    method: Literal["cdf", "pil"],
    max_value: int,
) -> ImageUInt8:
    """Apply per-channel autocontrast LUTs with one OpenCV pass for large RGB
    images and multispectral inputs where split channel assignment is slower.
    """
    luts = []
    for channel_idx in range(get_num_channels(img)):
        channel = img[..., channel_idx]
        hist = cv2.calcHist([channel], [0], None, [256], [0, max_value]).ravel()
        lut = _create_auto_contrast_lut(hist, cutoff, None, method, max_value)
        luts.append(np.arange(256, dtype=np.uint8) if lut is None else lut)

    return cast("ImageUInt8", apply_multichannel_lut(img, np.stack(luts), get_num_channels(img)))


def _auto_contrast_multichannel_hist(
    img: ImageUInt8,
    cutoff: float,
    ignore: int | None,
    method: Literal["cdf", "pil"],
    max_value: int,
) -> ImageUInt8:
    result = img.copy()
    channels = cv2.split(img)
    hists: list[np.ndarray | None] = []
    for channel_idx, channel in enumerate(channels):
        if ignore is not None and channel_idx == ignore:
            hists.append(None)
            continue
        mask = None if ignore is None else (channel != ignore)
        hist = cv2.calcHist([channel], [0], mask, [256], [0, max_value])
        hists.append(hist.ravel())

    for channel_idx, channel in enumerate(channels):
        if ignore is not None and channel_idx == ignore:
            continue

        hist = hists[channel_idx]
        if hist is None:
            continue

        lut = _create_auto_contrast_lut(hist, cutoff, ignore, method, max_value)
        if lut is None:
            continue
        result[..., channel_idx] = sz_lut(channel, lut)

    return result


def _create_auto_contrast_lut(
    hist: np.ndarray,
    cutoff: float,
    ignore: int | None,
    method: Literal["cdf", "pil"],
    max_value: int,
) -> np.ndarray | None:
    lo, hi = get_histogram_bounds(hist, cutoff)
    if hi <= lo:
        return None

    lut = create_contrast_lut(hist, lo, hi, max_value, method)
    if ignore is not None:
        lut[ignore] = ignore
    return lut


def _should_use_auto_contrast_multichannel_lut(
    img: ImageUInt8,
    ignore: int | None,
    method: Literal["cdf", "pil"],
    num_channels: int,
) -> bool:
    return (
        method == "cdf"
        and ignore is None
        and (num_channels > 3 or img.shape[0] * img.shape[1] >= MULTICHANNEL_LUT_MEDIUM_IMAGE_PIXELS)
    )


@uint8_io
def auto_contrast(
    img: ImageType,
    cutoff: float,
    ignore: int | None,
    method: Literal["cdf", "pil"],
) -> ImageType:
    """Apply automatic contrast enhancement. Stretches histogram to full range. cutoff, ignore, method
    (cdf or pil) limit clip. uint8 I/O.

    Args:
        img (ImageType): Input image
        cutoff (float): Cutoff percentage for histogram
        ignore (int | None): Value to ignore in histogram
        method (Literal['cdf', 'pil']): Method to use for contrast enhancement

    Returns:
        ImageType: Image with enhanced contrast

    """
    num_channels = get_num_channels(img)
    img = cast("ImageUInt8", img)
    max_value = int(MAX_VALUES_BY_DTYPE[img.dtype])

    if method == "pil" and cutoff == 0 and ignore is None and num_channels == 1:
        return _auto_contrast_pil_zero_cutoff(img, max_value)

    if img.ndim == MONO_CHANNEL_DIMENSIONS:
        return _auto_contrast_single_channel(img, cutoff, ignore, method, max_value)

    if _should_use_auto_contrast_multichannel_lut(img, ignore, method, num_channels):
        return _auto_contrast_multichannel_lut(img, cutoff, method, max_value)

    return _auto_contrast_multichannel_hist(img, cutoff, ignore, method, max_value)


def create_contrast_lut(
    hist: np.ndarray,
    min_intensity: int,
    max_intensity: int,
    max_value: int,
    method: Literal["cdf", "pil"],
) -> np.ndarray:
    """Create lookup table for contrast adjustment. LUT maps [0, max_val] using
    clip low/high for contrast transforms and auto_contrast.

    This function creates a lookup table for contrast adjustment.

    Args:
        hist (np.ndarray): Histogram of the image.
        min_intensity (int): Minimum intensity of the histogram.
        max_intensity (int): Maximum intensity of the histogram.
        max_value (int): Maximum value of the lookup table.
        method (Literal['cdf', 'pil']): Method to use for contrast enhancement.

    Returns:
        np.ndarray: Lookup table for contrast adjustment.

    """
    if min_intensity >= max_intensity:
        return np.zeros(256, dtype=np.uint8)

    if method == "cdf":
        hist_range = hist[min_intensity : max_intensity + 1]
        cdf = hist_range.cumsum()

        if cdf[-1] == 0:  # No valid pixels
            return np.arange(256, dtype=np.uint8)

        # Normalize CDF to full range
        cdf = (cdf - cdf[0]) * max_value / (cdf[-1] - cdf[0])

        # Create lookup table
        lut = np.zeros(256, dtype=np.uint8)
        lut[min_intensity : max_intensity + 1] = np.clip(np.round(cdf), 0, max_value).astype(np.uint8)
        lut[max_intensity + 1 :] = max_value
        return lut

    # "pil" method
    scale = max_value / (max_intensity - min_intensity)
    indices = np.arange(256, dtype=np.float32)
    # Changed: Use np.round to get 128 for middle value
    # Test expects [0, 128, 255] for range [0, 2]
    lut = np.clip(np.round((indices - min_intensity) * scale), 0, max_value).astype(np.uint8)
    lut[:min_intensity] = 0
    lut[max_intensity + 1 :] = max_value
    return lut


def get_histogram_bounds(hist: np.ndarray, cutoff: float) -> tuple[int, int]:
    """Get the low and high bounds of the histogram. Percentile-based clipping;
    returns (low, high) for auto_contrast and create_contrast_lut.

    This function gets the low and high bounds of the histogram.

    Args:
        hist (np.ndarray): Histogram of the image.
        cutoff (float): Cutoff percentage for histogram.

    Returns:
        tuple[int, int]: Low and high bounds of the histogram.

    """
    if not cutoff:
        non_zero_intensities = np.nonzero(hist)[0]
        if len(non_zero_intensities) == 0:
            return 0, 0
        return int(non_zero_intensities[0]), int(non_zero_intensities[-1])

    total_pixels = float(reduce_sum(hist))
    if total_pixels == 0:
        return 0, 0

    pixels_to_cut = total_pixels * cutoff / 100.0

    # Special case for uniform 256-bin histogram
    if len(hist) == 256 and np.all(hist == hist[0]):
        min_intensity = int(len(hist) * cutoff / 100)  # floor division
        max_intensity = len(hist) - min_intensity - 1
        return min_intensity, max_intensity

    # Find minimum intensity
    cumsum = 0.0
    min_intensity = 0
    for i in range(len(hist)):
        cumsum += hist[i]
        if cumsum >= pixels_to_cut:  # Use >= for left bound
            min_intensity = i + 1
            break
    min_intensity = min(min_intensity, len(hist) - 1)

    # Find maximum intensity
    cumsum = 0.0
    max_intensity = len(hist) - 1
    for i in range(len(hist) - 1, -1, -1):
        cumsum += hist[i]
        if cumsum >= pixels_to_cut:  # Use >= for right bound
            max_intensity = i
            break

    # Handle edge cases
    if min_intensity > max_intensity:
        mid_point = (len(hist) - 1) // 2
        return mid_point, mid_point

    return min_intensity, max_intensity


def get_drop_mask(
    shape: tuple[int, ...],
    per_channel: bool,
    dropout_prob: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate dropout mask (boolean or per-pixel drop prob). shape, per_channel, dropout_prob,
    random_generator. Returns bool or float mask.

    This function generates a dropout mask.

    Args:
        shape (tuple[int, ...]): Shape of the output mask
        per_channel (bool): Whether to apply dropout per channel
        dropout_prob (float): Dropout probability
        random_generator (np.random.Generator): Random number generator

    Returns:
        np.ndarray: Dropout mask

    """
    if per_channel or len(shape) == 2:
        return random_generator.random(shape) < dropout_prob

    mask_2d = random_generator.random(shape[:2]) < dropout_prob
    return np.repeat(mask_2d[..., None], shape[2], axis=2)


def generate_random_values(
    channels: int,
    dtype: np.dtype,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate random values for dropout fill. channels, dtype, random_generator. Returns 1D array
    per channel when value is None in prepare_drop_values.

    Args:
        channels (int): Number of channels
        dtype (np.dtype): Data type of the output array
        random_generator (np.random.Generator): Random number generator

    Returns:
        np.ndarray: Random values

    """
    if dtype == np.uint8:
        return random_generator.integers(
            0,
            int(MAX_VALUES_BY_DTYPE[dtype]),
            size=channels,
            dtype=dtype,
        )
    if dtype == np.float32:
        return random_generator.uniform(0, 1, size=channels).astype(dtype)

    raise ValueError(f"Unsupported dtype: {dtype}")


def prepare_drop_values(
    array: np.ndarray,
    value: float | Sequence[float] | np.ndarray | None,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Prepare values to fill dropped pixels. array shape/dtype, value (or None for random),
    random_generator. Returns fill array matching shape.

    Args:
        array (np.ndarray): Input array to determine shape and dtype
        value (float | Sequence[float] | np.ndarray | None): User-specified drop values or None for random
        random_generator (np.random.Generator): Random number generator

    Returns:
        np.ndarray: Array of values matching input shape

    """
    if value is None:
        channels = get_num_channels(array)
        values = generate_random_values(channels, array.dtype, random_generator)
    elif isinstance(value, (int, float)):
        return np.full(array.shape, value, dtype=array.dtype)
    else:
        values = np.array(value, dtype=array.dtype).reshape(-1)

    # For monochannel input, return single value
    if array.ndim == 2:
        return np.full(array.shape, values[0], dtype=array.dtype)

    # For multichannel input, broadcast values to match image shape
    channels = array.shape[2]
    if len(values) != channels:
        # If number of values doesn't match channels:
        # - Single value: repeat for all channels
        # - Multiple values: cycle through them to match channel count
        if len(values) == 1:
            # Single value for all channels
            broadcast_values = np.full(channels, values[0], dtype=array.dtype)
        else:
            # Use tile for better performance, especially with many channels
            # (e.g., 4x faster for 100 channels, 18x for 512 channels)
            broadcast_values = np.tile(values, (channels + len(values) - 1) // len(values))[:channels].astype(
                array.dtype,
            )
    else:
        broadcast_values = values

    return np.full(array.shape, broadcast_values, dtype=array.dtype)


def get_mask_array(data: dict[str, Any]) -> np.ndarray | None:
    """Get mask array from input data if it exists. Returns data['mask'] or None;
    helper for transforms that accept optional mask.
    """
    if "mask" in data:
        return data["mask"]
    return data["masks"][0] if "masks" in data else None


STAIN_MATRICES = {
    "ruifrok": np.array(
        [  # Ruifrok & Johnston standard reference
            [0.644211, 0.716556, 0.266844],  # Hematoxylin
            [0.092789, 0.954111, 0.283111],  # Eosin
        ],
    ),
    "macenko": np.array(
        [  # Macenko's reference
            [0.5626, 0.7201, 0.4062],
            [0.2159, 0.8012, 0.5581],
        ],
    ),
    "standard": np.array(
        [  # Standard bright-field microscopy
            [0.65, 0.70, 0.29],
            [0.07, 0.99, 0.11],
        ],
    ),
    "high_contrast": np.array(
        [  # Enhanced contrast
            [0.55, 0.88, 0.11],
            [0.12, 0.86, 0.49],
        ],
    ),
    "h_heavy": np.array(
        [  # Hematoxylin dominant
            [0.75, 0.61, 0.32],
            [0.04, 0.93, 0.36],
        ],
    ),
    "e_heavy": np.array(
        [  # Eosin dominant
            [0.60, 0.75, 0.28],
            [0.17, 0.95, 0.25],
        ],
    ),
    "dark": np.array(
        [  # Darker staining
            [0.78, 0.55, 0.28],
            [0.09, 0.97, 0.21],
        ],
    ),
    "light": np.array(
        [  # Lighter staining
            [0.57, 0.71, 0.38],
            [0.15, 0.89, 0.42],
        ],
    ),
}


def _create_vignette_mask(
    height: int,
    width: int,
    intensity: float,
    center_x: float,
    center_y: float,
) -> np.ndarray:
    """Create a 2D vignette falloff mask. Radial gradient from center. Returns
    float32 (H, W) in [0, 1] for apply_vignette to darken corners.

    Returns:
        np.ndarray: (H, W) float32 array with values in [1-intensity, 1].

    """
    pixel_cols = np.arange(width, dtype=np.float32)
    pixel_rows = np.arange(height, dtype=np.float32)

    center_col = center_x * width
    center_row = center_y * height

    norm_x = (pixel_cols - center_col) / (width * 0.5)
    norm_y = (pixel_rows - center_row) / (height * 0.5)

    dist_sq = norm_y[:, np.newaxis] ** 2 + norm_x[np.newaxis, :] ** 2

    max_dist_sq = float(np.max(dist_sq))
    if max_dist_sq > 0:
        dist_sq /= max_dist_sq

    return (1.0 - intensity * dist_sq).astype(np.float32)


def apply_vignette(
    img: ImageType,
    intensity: float,
    center_x: float,
    center_y: float,
) -> ImageType:
    """Apply vignetting by darkening corners with radial gradient. intensity, center_x, center_y
    control strength and center. Supports (H,W) or (H,W,C).

    Args:
        img (ImageType): Input image of shape (H, W) or (H, W, C).
        intensity (float): Strength of darkening at corners, in [0, 1].
        center_x (float): Horizontal center of the vignette as fraction of width, in [0, 1].
        center_y (float): Vertical center of the vignette as fraction of height, in [0, 1].

    Returns:
        ImageType: Image with vignetting applied.

    """
    height, width = img.shape[:2]

    vignette_mask = _create_vignette_mask(height, width, intensity, center_x, center_y)

    return multiply(img, vignette_mask[:, :, np.newaxis])


def apply_film_grain(
    img: ImageType,
    grain: np.ndarray,
    intensity: float,
) -> ImageType:
    """Apply film grain noise to an image (2D or 3D). Luminance-dependent and
    spatially correlated; params control grain size and intensity.

    Film grain is luminance-dependent and spatially correlated, unlike simple Gaussian noise.

    Args:
        img (ImageType): Input image, shape (H, W, C) or (H, W, 1).
        grain (np.ndarray): Pre-generated grain pattern, shape (H, W), float32.
        intensity (float): Grain strength multiplier.

    Returns:
        ImageType: Image with film grain applied.

    """
    num_channels = img.shape[-1]

    luminance = mean(img, axis=-1) if num_channels > 1 else img[..., 0]

    max_val = MAX_VALUES_BY_DTYPE[img.dtype]

    inv_lum = 1.0 - np.asarray(luminance).astype(np.float32) / max_val if img.dtype == np.uint8 else 1.0 - luminance

    modulated = (grain * inv_lum * intensity * max_val).astype(np.float32)

    return add_array(img, modulated[..., np.newaxis])


@uint8_io
def apply_halftone(
    img: ImageType,
    dot_size: int,
    blend: float,
) -> ImageType:
    """Convert image to halftone dot pattern. Simulates printed halftone; dot size
    varies by intensity. Params: scale, dot_shape.

    Args:
        img (ImageType): Input image (H, W, C), uint8 or float32.
        dot_size (int): Size of each halftone grid cell in pixels.
        blend (float): Blend factor between halftone and original. 0 = pure halftone, 1 = original.

    Returns:
        ImageType: Image with halftone effect applied.

    """
    img = cast("ImageType", np.ascontiguousarray(img))
    height, width = img.shape[:2]
    num_channels = img.shape[-1]

    luminance = (
        np.asarray(mean(img, axis=-1)).astype(np.float32) / MAX_VALUES_BY_DTYPE[np.uint8]
        if num_channels > 1
        else img[..., 0].astype(np.float32) / MAX_VALUES_BY_DTYPE[np.uint8]
    )

    use_cell_mask = num_channels > 4

    if use_cell_mask:
        canvas = np.zeros_like(img)
        for y_start in range(0, height, dot_size):
            for x_start in range(0, width, dot_size):
                y_end = min(y_start + dot_size, height)
                x_end = min(x_start + dot_size, width)

                cell_lum = float(mean(luminance[y_start:y_end, x_start:x_end]))
                cell_h = y_end - y_start
                cell_w = x_end - x_start
                radius = max(1, int(dot_size * 0.5 * cell_lum))

                cell = img[y_start:y_end, x_start:x_end]
                avg_color = np.asarray(mean(cell.reshape(-1, num_channels), axis=0)).astype(img.dtype)

                # Cell-local mask: avoids O(N_cells * H * W) allocation
                local_cx = cell_w // 2
                local_cy = cell_h // 2
                dot_mask = np.zeros((cell_h, cell_w), dtype=np.uint8)
                cv2.circle(dot_mask, (local_cx, local_cy), radius, 255, -1, lineType=cv2.LINE_AA)
                mask_bool = dot_mask > 0
                canvas[y_start:y_end, x_start:x_end][mask_bool] = avg_color
    else:
        canvas = np.zeros_like(img)
        for y_start in range(0, height, dot_size):
            for x_start in range(0, width, dot_size):
                y_end = min(y_start + dot_size, height)
                x_end = min(x_start + dot_size, width)

                cell_lum = float(mean(luminance[y_start:y_end, x_start:x_end]))
                cx = (x_start + x_end) // 2
                cy = (y_start + y_end) // 2
                radius = max(1, int(dot_size * 0.5 * cell_lum))

                cell = img[y_start:y_end, x_start:x_end]
                if num_channels > 1:
                    color: tuple[int, ...] | int = tuple(
                        int(v) for v in np.asarray(mean(cell.reshape(-1, num_channels), axis=0))
                    )
                else:
                    color = int(mean(cell))

                cv2.circle(canvas, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)

    if blend > 0:
        return add_weighted(img, blend, canvas, 1.0 - blend)

    return canvas


def apply_lens_flare(
    img: ImageType,
    flare_center: tuple[int, int],
    ghosts: list[tuple[int, int, int, float]],
    starburst_angles: np.ndarray,
    starburst_intensity: float,
    bloom_radius: int,
) -> ImageType:
    """Apply realistic lens flare with ghosts and starburst. Params control position,
    intensity, and number of ghosts. RGB input.

    Args:
        img (ImageType): Input image (H, W, C), must be 3-channel.
        flare_center (tuple[int, int]): (x, y) position of the flare source.
        ghosts (list[tuple[int, int, int, float]]): List of (x, y, radius, alpha) for each ghost circle.
        starburst_angles (np.ndarray): Array of angles in radians for starburst rays.
        starburst_intensity (float): Brightness of starburst rays, 0-1.
        bloom_radius (int): Gaussian blur radius for bloom effect.

    Returns:
        ImageType: Image with lens flare applied.

    """
    height, width = img.shape[:2]
    max_val = MAX_VALUES_BY_DTYPE[img.dtype]
    result = img.copy()

    flare_layer = np.zeros((height, width), dtype=np.float32)

    fx, fy = flare_center
    for angle in starburst_angles:
        dx = np.cos(angle)
        dy = np.sin(angle)
        length = max(height, width)
        x2 = int(fx + dx * length)
        y2 = int(fy + dy * length)
        cv2.line(flare_layer, (fx, fy), (x2, y2), starburst_intensity, 1, lineType=cv2.LINE_AA)

    if bloom_radius > 0:
        ksize = bloom_radius * 2 + 1
        flare_layer = cv2.GaussianBlur(flare_layer, (ksize, ksize), 0)

    for gx, gy, gradius, galpha in ghosts:
        ghost = np.zeros((height, width), dtype=np.float32)
        cv2.circle(ghost, (gx, gy), gradius, 1.0, -1, lineType=cv2.LINE_AA)
        if gradius > 2:
            gk = max(3, gradius | 1)
            ghost = cv2.GaussianBlur(ghost, (gk, gk), 0)
        flare_layer += ghost * galpha

    np.clip(flare_layer, 0, 1, out=flare_layer)

    flare_3d = flare_layer[:, :, np.newaxis] * max_val

    return clip(result.astype(np.float32) + flare_3d, result.dtype)


def generate_water_displacement_maps(
    image_shape: tuple[int, int],
    amplitude: float,
    wavelength: float,
    num_waves: int,
    random_generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate displacement maps simulating water refraction. image_shape, amplitude, wavelength,
    num_waves, random_generator. Returns (map_x, map_y) for cv2.remap.

    Args:
        image_shape (tuple[int, int]): (height, width).
        amplitude (float): Maximum displacement in pixels.
        wavelength (float): Base wavelength of waves in pixels.
        num_waves (int): Number of overlaid sine waves.
        random_generator (np.random.Generator): NumPy random generator.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (map_x, map_y) float32 arrays for cv2.remap.

    """
    height, width = image_shape
    y, x = np.mgrid[:height, :width].astype(np.float32)

    dx = np.zeros((height, width), dtype=np.float32)
    dy = np.zeros((height, width), dtype=np.float32)

    for _ in range(num_waves):
        angle = random_generator.uniform(0, 2 * np.pi)
        phase = random_generator.uniform(0, 2 * np.pi)
        freq = random_generator.uniform(0.7, 1.3) / wavelength
        amp = amplitude * random_generator.uniform(0.5, 1.0)

        wave_x = np.cos(angle)
        wave_y = np.sin(angle)

        projection = x * wave_x + y * wave_y
        displacement = amp * np.sin(2 * np.pi * freq * projection + phase)

        dx += displacement * (-wave_y)
        dy += displacement * wave_x

    map_x = x + dx
    map_y = y + dy

    return map_x.astype(np.float32), map_y.astype(np.float32)


__all__ = [
    "STAIN_MATRICES",
    "_auto_contrast_single_channel",
    "_create_vignette_mask",
    "apply_corner_illumination",
    "apply_film_grain",
    "apply_gaussian_illumination",
    "apply_halftone",
    "apply_lens_flare",
    "apply_linear_illumination",
    "apply_plasma_brightness_contrast",
    "apply_plasma_shadow",
    "apply_vignette",
    "auto_contrast",
    "create_contrast_lut",
    "create_corner_illumination_gradient",
    "create_directional_gradient",
    "create_illumination_gradient",
    "generate_plasma_pattern",
    "generate_random_values",
    "generate_water_displacement_maps",
    "get_drop_mask",
    "get_histogram_bounds",
    "get_mask_array",
    "prepare_drop_values",
]
