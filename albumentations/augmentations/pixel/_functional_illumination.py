"""Plasma, illumination, dropout, vignette, flare, and halftone functional helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import numkong as nk
from albucore import exp as albucore_exp
from albucore import sqrt as albucore_sqrt
from numpy.typing import NDArray

from ._functional_noise import (
    DIAMOND_KERNEL,
    SQUARE_KERNEL,
)
from ._functional_shared import (
    MAX_VALUES_BY_DTYPE,
    MONO_CHANNEL_DIMENSIONS,
    MULTICHANNEL_LUT_MEDIUM_IMAGE_PIXELS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    ImageFloat32,
    ImageUInt8,
    add,
    add_weighted,
    apply_uint8_lut,
    clip,
    convert_dtype,
    cv2,
    get_num_channels,
    math,
    mean,
    multiply,
    np,
    reduce_sum,
    restore_uint8_working_image,
    to_uint8_working_image,
)

_PLASMA_FLOAT32_DEDICATED_MAX_BATCH_SIZE = 4
_PLASMA_UINT8_BRIGHTNESS_ONLY_PER_IMAGE_MAX_BATCH_SIZE = 4
_PLASMA_UINT8_BRIGHTNESS_ONLY_PER_IMAGE_MIN_CHANNELS = 5
_PLASMA_UINT8_BRIGHTNESS_ONLY_PER_IMAGE_MIN_PIXELS = 1024 * 1024


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

        grid_shape = (next_height, next_width)
        expanded_grid = np.zeros(grid_shape, dtype=np.float32)
        all_noise = random_generator.uniform(-noise_scale, noise_scale, grid_shape).astype(np.float32)
        np.add(current_grid, all_noise[::2, ::2], out=expanded_grid[::2, ::2])

        interpolation = np.empty_like(expanded_grid)
        mask = np.empty_like(expanded_grid)
        for kernel in (DIAMOND_KERNEL, SQUARE_KERNEL):
            cv2.filter2D(expanded_grid, -1, kernel, dst=interpolation, borderType=cv2.BORDER_CONSTANT)
            cv2.threshold(interpolation, 0, 1, cv2.THRESH_BINARY, dst=mask)
            cv2.add(interpolation, all_noise, dst=interpolation)
            cv2.multiply(interpolation, mask, dst=interpolation)
            cv2.add(expanded_grid, interpolation, dst=expanded_grid)

        # Normalize after each step to prevent value drift.
        cv2.normalize(expanded_grid, expanded_grid, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return expanded_grid

    # Pre-compute noise scales
    max_dimension = max(target_shape)
    power_of_two_size = 2 ** np.ceil(np.log2(max_dimension - 1)) + 1
    total_steps = int(np.log2(power_of_two_size - 1) - 1)
    noise_scales = np.asarray([roughness**i for i in range(total_steps)], dtype=np.float32)

    plasma_grid = random_generator.uniform(-1, 1, (3, 3)).astype(np.float32)

    # Recursively apply diamond-square steps
    for noise_scale in noise_scales:
        plasma_grid = one_diamond_square_step(plasma_grid, cast("float", noise_scale))

    pattern = _normalize_minmax_float32(plasma_grid[: target_shape[0], : target_shape[1]])
    return np.clip(pattern, 0, 1, out=pattern)


def apply_plasma_brightness_contrast(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    brightness_factor: float,
    contrast_factor: float,
    plasma_pattern: np.ndarray,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Modulate brightness and contrast with a plasma pattern. brightness_factor,
    contrast_factor scale effect. Use for spatially varying lighting.

    This function applies plasma-based brightness and contrast adjustments to an image.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): The image to apply the brightness and
            contrast adjustments to.
        brightness_factor (float): The brightness factor to apply.
        contrast_factor (float): The contrast factor to apply.
        plasma_pattern (np.ndarray): The plasma pattern to use for the brightness and contrast adjustments.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: The image with the brightness and contrast
            adjustments applied.

    """
    input_dtype = img.dtype
    img = convert_dtype(img, np.float32)
    if brightness_factor == 0 and contrast_factor == 0:
        result = img if input_dtype == np.float32 else convert_dtype(img, dtype=input_dtype)
        return result if input_dtype == np.uint8 else clip(result, input_dtype)

    if img.ndim > MONO_CHANNEL_DIMENSIONS:
        plasma_pattern = np.tile(plasma_pattern[..., np.newaxis], (1, 1, img.shape[-1]))

    if brightness_factor != 0:
        brightness_adjustment = multiply(plasma_pattern, brightness_factor)
        img = add(img, brightness_adjustment)

    if contrast_factor != 0:
        img_mean = mean(img)
        contrast_weights = multiply(plasma_pattern, contrast_factor) + 1

        img = multiply(img, contrast_weights, out=img if brightness_factor != 0 else None)

        mean_factor = img_mean * (1.0 - contrast_weights)
        img = add(img, mean_factor, out=img)

    result = img if input_dtype == np.float32 else convert_dtype(img, dtype=input_dtype)
    return result if input_dtype == np.uint8 else clip(result, input_dtype)


def _plasma_pattern_for_batch(
    images: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32], plasma_pattern: np.ndarray
) -> np.ndarray:
    """Expand a shared plasma pattern once for a batch with an explicit channel axis."""
    if images.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
        return plasma_pattern

    channel_pattern = plasma_pattern[..., np.newaxis]
    if images.shape[-1] == 1:
        return channel_pattern
    return np.broadcast_to(channel_pattern, (*plasma_pattern.shape, images.shape[-1])).copy()


def _finish_plasma_brightness_contrast_batch(
    work: NDArray[np.float16] | ImageFloat32,
    output_dtype: np.dtype[Any],
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Clip and restore an owned floating batch to the public dtype and contiguous layout."""
    if output_dtype.type in (np.float16, np.float32):
        np.clip(work, 0.0, 1.0, out=work, dtype=np.float32, casting="unsafe")
    return cast(
        "NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]",
        np.ascontiguousarray(convert_dtype(work, output_dtype)),
    )


def _apply_plasma_brightness_contrast_dedicated(
    images: ImageFloat32,
    brightness_factor: float,
    contrast_factor: float,
    plasma_pattern: np.ndarray,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Apply brightness and contrast to a small float32 batch using one mean per image."""
    work = np.ascontiguousarray(images)
    image_pattern = _plasma_pattern_for_batch(images, plasma_pattern)

    if brightness_factor != 0:
        work = add(work, np.multiply(image_pattern, np.float32(brightness_factor)))

    if contrast_factor != 0:
        image_means = tuple(mean(image) for image in work)
        image_contrast_weights = np.add(
            np.multiply(image_pattern, np.float32(contrast_factor)),
            np.float32(1.0),
        )
        contrast_weights = image_contrast_weights[np.newaxis]
        image_contrast_offsets = np.subtract(np.float32(1.0), image_contrast_weights)
        work = multiply(work, contrast_weights, out=work if brightness_factor != 0 else None)
        for index, image_mean in enumerate(image_means):
            work[index] = cast(
                "ImageFloat32",
                add(
                    work[index],
                    np.multiply(image_mean, image_contrast_offsets),
                    out=work[index],
                ),
            )
    return _finish_plasma_brightness_contrast_batch(work, images.dtype)


def _apply_plasma_brightness_float16_batch(
    images: NDArray[np.float16],
    brightness_factor: float,
    plasma_pattern: np.ndarray,
) -> NDArray[np.float16]:
    brightness_map = np.multiply(plasma_pattern, np.float32(brightness_factor))
    if images.ndim != NUM_MULTI_CHANNEL_DIMENSIONS:
        brightness_map = brightness_map.reshape(*brightness_map.shape, 1)
        if images.shape[-1] > 1:
            brightness_map = np.repeat(brightness_map, images.shape[-1], axis=-1)
    result = np.empty_like(images, order="C")
    add(images, brightness_map[np.newaxis], out=result)
    return clip(result, np.float16, out=result)


def _apply_plasma_brightness_contrast_loop(
    images: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    brightness_factor: float,
    contrast_factor: float,
    plasma_pattern: np.ndarray,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Apply shared plasma maps in a preallocated loop for larger batches and uint8 input."""
    output_dtype = images.dtype
    result = np.empty(images.shape, dtype=output_dtype)
    image_pattern = _plasma_pattern_for_batch(images, plasma_pattern)
    brightness_map = np.multiply(image_pattern, np.float32(brightness_factor)) if brightness_factor != 0 else None
    contrast_weights = (
        np.add(np.multiply(image_pattern, np.float32(contrast_factor)), np.float32(1.0))
        if contrast_factor != 0
        else None
    )
    contrast_offsets = np.subtract(np.float32(1.0), contrast_weights) if contrast_weights is not None else None
    # Early narrowing must retain half overflow and NumPy underflow behavior.
    mean_offsets = (
        np.empty_like(contrast_offsets, dtype=np.float32)
        if contrast_offsets is not None and abs(contrast_factor) <= 1 and np.geterr()["under"] == "ignore"
        else None
    )

    for index, image in enumerate(images):
        work = (
            convert_dtype(image, np.float32)
            if output_dtype == np.uint8
            else cast("NDArray[np.float16] | ImageFloat32", image)
        )

        if brightness_map is not None:
            work = add(work, brightness_map)

        if contrast_weights is not None and contrast_offsets is not None:
            image_mean = mean(work)
            work = multiply(work, contrast_weights)
            mean_factor = (
                np.multiply(image_mean, contrast_offsets, out=mean_offsets)
                if mean_offsets is not None and np.all(np.abs(image_mean) <= 1)
                else np.multiply(image_mean, contrast_offsets)
            )
            work = add(work, mean_factor)
        result[index] = _finish_plasma_brightness_contrast_batch(work, output_dtype)

    return cast("NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]", result)


def _apply_plasma_brightness_contrast_per_image(
    images: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    brightness_factor: float,
    contrast_factor: float,
    plasma_pattern: np.ndarray,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Use the single-image implementation for small, high-resolution uint8 brightness-only batches."""
    return cast(
        "NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]",
        np.stack(
            [
                apply_plasma_brightness_contrast(
                    image,
                    brightness_factor,
                    contrast_factor,
                    plasma_pattern,
                )
                for image in images
            ],
        ),
    )


def apply_plasma_brightness_contrast_batch(
    images: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    brightness_factor: float,
    contrast_factor: float,
    plasma_pattern: np.ndarray,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Apply one plasma lighting pattern to an image batch while retaining independent image means and
    reusing shared spatial setup for efficient lighting augmentation.

    The result matches stacking :func:`apply_plasma_brightness_contrast` for nonzero adjustments. Empty batches
    preserve identity; zero-effect batches are independent copies, with floating-point values clipped to `[0, 1]`.

    Args:
        images (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Image batch in `(N, H, W, C)` format, or
            `(N, H, W)` for direct grayscale calls.
        brightness_factor (float): Strength of the spatial brightness adjustment.
        contrast_factor (float): Strength of the spatial contrast adjustment.
        plasma_pattern (np.ndarray): Shared `(H, W)` plasma pattern with values in `[0, 1]`.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Transformed batch with the input shape and dtype.

    Examples:
        >>> images = np.full((2, 8, 8, 3), 0.5, dtype=np.float32)
        >>> pattern = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
        >>> result = apply_plasma_brightness_contrast_batch(images, 0.2, -0.1, pattern)
        >>> result.shape
        (2, 8, 8, 3)

    """
    if images.shape[0] == 0:
        return images

    if brightness_factor == 0 and contrast_factor == 0:
        if images.dtype.type in (np.float16, np.float32):
            result: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] = np.empty_like(images, order="C")
            np.clip(images, 0.0, 1.0, out=result, dtype=np.float32, casting="unsafe")
        else:
            result = cast("ImageUInt8", np.array(images, copy=True, order="C"))
        return result

    if images.shape[0] == 1:
        result = apply_plasma_brightness_contrast(
            images[0],
            brightness_factor,
            contrast_factor,
            plasma_pattern,
        )
        return result[np.newaxis]

    if (
        images.dtype == np.uint8
        and contrast_factor == 0
        and images.ndim > NUM_MULTI_CHANNEL_DIMENSIONS
        and images.shape[0] <= _PLASMA_UINT8_BRIGHTNESS_ONLY_PER_IMAGE_MAX_BATCH_SIZE
        and images.shape[1] * images.shape[2] >= _PLASMA_UINT8_BRIGHTNESS_ONLY_PER_IMAGE_MIN_PIXELS
        and images.shape[-1] >= _PLASMA_UINT8_BRIGHTNESS_ONLY_PER_IMAGE_MIN_CHANNELS
    ):
        return _apply_plasma_brightness_contrast_per_image(
            images,
            brightness_factor,
            contrast_factor,
            plasma_pattern,
        )

    if images.dtype == np.float32 and images.shape[0] <= _PLASMA_FLOAT32_DEDICATED_MAX_BATCH_SIZE:
        return _apply_plasma_brightness_contrast_dedicated(
            cast("ImageFloat32", images),
            brightness_factor,
            contrast_factor,
            plasma_pattern,
        )

    if images.dtype == np.float16 and contrast_factor == 0:
        result = cast(
            "NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]",
            _apply_plasma_brightness_float16_batch(
                cast("NDArray[np.float16]", images), brightness_factor, plasma_pattern
            ),
        )
    else:
        result = _apply_plasma_brightness_contrast_loop(
            images,
            brightness_factor,
            contrast_factor,
            plasma_pattern,
        )
    return result


def apply_plasma_shadow(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    intensity: float,
    plasma_pattern: np.ndarray,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Darken image in regions defined by a plasma pattern. intensity controls strength. Use for
    soft shadows or vignette-like effects. Same dtype and channels.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image
        intensity (float): Shadow intensity
        plasma_pattern (np.ndarray): Plasma pattern to use

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Image with plasma shadow

    """
    # Scale plasma pattern by intensity first (scalar operation)
    scaled_pattern = plasma_pattern * intensity

    # Expand dimensions only once if needed
    if img.ndim > MONO_CHANNEL_DIMENSIONS:
        scaled_pattern = scaled_pattern[..., np.newaxis]

    result = cast("NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]", img * (1 - scaled_pattern))
    output_dtype = img.dtype if isinstance(img, np.ndarray) else np.dtype(str(img.dtype).removeprefix("torch."))
    return clip(result, output_dtype)


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
    if angle in (0, 90, 180, 270):
        return _create_cardinal_gradient(height, width, angle)
    if angle in (45, 135, 225, 315):
        return _create_diagonal_gradient(height, width, angle)
    return _create_arbitrary_gradient(height, width, angle)


def _create_cardinal_gradient(height: int, width: int, angle: float) -> np.ndarray:
    horizontal = angle in (0, 180)
    reversed_direction = angle in (180, 270)
    size = width if horizontal else height
    start, stop = (1, 0) if reversed_direction else (0, 1)
    values = np.linspace(start, stop, size, dtype=np.float32)
    gradient = np.empty((height, width), dtype=np.float32)
    gradient[:] = values if horizontal else values[:, np.newaxis]
    return gradient


def _create_diagonal_gradient(height: int, width: int, angle: float) -> np.ndarray:
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]  # Horizontal
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]  # Vertical
    if angle == 45:  # Bottom-left to top-right
        return _normalize_minmax_float32(x + y)
    if angle == 135:  # Bottom-right to top-left
        return _normalize_minmax_float32((1 - x) + y)
    if angle == 225:  # Top-right to bottom-left
        return _normalize_minmax_float32((1 - x) + (1 - y))
    # angle == 315: Top-left to bottom-right
    return _normalize_minmax_float32(x + (1 - y))


def _create_arbitrary_gradient(height: int, width: int, angle: float) -> np.ndarray:
    """Create a directional gradient for arbitrary angles without a specialized cardinal or diagonal fast path,
    preserving the normalized float32 output range.
    """
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
    """Create float32 (H, W) corner illumination map for multiply
    using Euclidean distance from a selected corner with diagonal scaling.

    The map follows `1 + scale * distance`, where `distance` is the Euclidean distance from each pixel
    to the selected image corner and `scale` combines `intensity` with a diagonal-based normalization
    (numerically aligned with the previous `cv2.distanceTransform` implementation).

    Args:
        height (int): Output height `H`.
        width (int): Output width `W`.
        intensity (float): Signed strength. `0` returns an all-ones map (no effect). Positive values
            increase multipliers with distance from the selected corner (typical vignette-style corner
            darkening when applied via `multiply`). Negative values invert the radial scaling
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

    pattern = np.add(x * x, y * y, dtype=np.float32)
    pattern = albucore_sqrt(pattern, out=pattern)
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
    apply via multiply.

    Returns a float32 gradient that can be applied via multiply.
    The returned gradient does NOT have a channel dimension.
    """
    if mode == "linear":
        intensity = params["intensity"]
        gradient = create_directional_gradient(height, width, params["angle"])
        _multiply_scalar_inplace(gradient, intensity)
        return gradient

    if mode == "corner":
        return create_corner_illumination_gradient(height, width, params["intensity"], params["corner"])

    # gaussian
    spots = params.get("spots")
    if spots is not None:
        return create_gaussian_illumination_gradient_from_spots(height, width, spots)
    return create_gaussian_illumination_gradient(
        height,
        width,
        params["intensity"],
        params["center"],
        params["sigma"],
    )


def create_gaussian_illumination_gradient(
    height: int,
    width: int,
    intensity: float,
    center: tuple[float, float],
    sigma: float,
) -> np.ndarray:
    """Create a float32 Gaussian illumination multiplier for one independently parameterized spot over an image of the
    requested height and width.
    """
    if intensity == 0:
        return np.ones((height, width), dtype=np.float32)
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
    x = albucore_exp(x, out=x)
    _multiply_scalar_inplace(x, intensity)
    _add_scalar_inplace(x, 1.0)
    return x


def create_gaussian_illumination_gradient_from_spots(
    height: int,
    width: int,
    spots: Sequence[tuple[float, float, float, float]],
) -> np.ndarray:
    """Multiply the float32 multipliers from independently parameterized Gaussian spots into one shared illumination
    field for an image.
    """
    if not spots:
        return np.ones((height, width), dtype=np.float32)
    center_x, center_y, sigma, intensity = spots[0]
    gradient = create_gaussian_illumination_gradient(height, width, intensity, (center_x, center_y), sigma)
    for center_x, center_y, sigma, intensity in spots[1:]:
        spot_gradient = create_gaussian_illumination_gradient(height, width, intensity, (center_x, center_y), sigma)
        np.multiply(gradient, spot_gradient, out=gradient)
    return gradient


def apply_linear_illumination(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32], intensity: float, angle: float
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Apply linear illumination gradient to the image. Adds a signed gradient; intensity and angle
    control direction and strength. Uses float32 working arithmetic and restores the input dtype.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image
        intensity (float): Illumination intensity
        angle (float): Illumination angle in radians

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Image with linear illumination

    """
    input_dtype = img.dtype
    if input_dtype.type is np.uint8 or input_dtype.type is np.float16:
        img = convert_dtype(img, np.float32)
    else:
        img = cast("ImageFloat32", img)

    height, width = img.shape[:2]
    gradient = create_directional_gradient(height, width, angle)
    _multiply_scalar_inplace(gradient, intensity)

    # Add channel dimension if needed
    if img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
        num_channels = img.shape[2]
        if num_channels > 1:
            gradient = cv2.merge([gradient] * num_channels)
            result = cast("ImageFloat32", cv2.add(img, gradient))
            clip(result, out=result)
            return _restore_linear_illumination_dtype(result, input_dtype)
        gradient = gradient[..., np.newaxis]

    result = img + gradient
    clip(result, out=result)
    return _restore_linear_illumination_dtype(result, input_dtype)


def apply_linear_illumination_batch(
    images: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32], intensity: float, angle: float, **params: Any
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Applies one sampled linear gradient to every `(N, H, W, C)` image, matching the single-image kernel's
    normalization, clipping, and dtype-conversion semantics.
    """
    input_dtype = images.dtype
    if input_dtype.type is np.uint8 or input_dtype.type is np.float16:
        images = convert_dtype(images, np.float32)
    else:
        images = cast("ImageFloat32", images)

    height, width = images.shape[1:3]
    gradient = create_directional_gradient(height, width, angle)
    _multiply_scalar_inplace(gradient, intensity)
    batch_gradient = gradient if images.ndim == NUM_MULTI_CHANNEL_DIMENSIONS else gradient[..., np.newaxis]
    result = images + batch_gradient
    clip(result, out=result)
    return _restore_linear_illumination_dtype(result, input_dtype)


def _restore_linear_illumination_dtype(
    img: ImageFloat32, input_dtype: np.dtype
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    if input_dtype.type is np.uint8:
        return convert_dtype(img, np.uint8)
    if input_dtype.type is np.float16:
        return convert_dtype(img, np.float16)
    return img


def _apply_clipped_illumination_batch(
    images: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    gradient: np.ndarray,
    implicit_grayscale: bool,
    num_channels: int,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Multiply float32 illumination batches into preallocated output and upper-clip factors of at least one without
    a second full-batch pass.

    Args:
        images (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input float32 image batch.
        gradient (np.ndarray): Shared multiplicative illumination gradient.
        implicit_grayscale (bool): Whether the batch omits an explicit channel dimension.
        num_channels (int): Number of image channels.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Multiplied and upper-clipped image batch.

    """
    result = np.empty_like(images)
    # Whole-batch iteration wins for tiny single-channel batches; the per-image path has better cache locality beyond 4.
    if num_channels == 1 and images.shape[0] <= 4:
        image_view = images if implicit_grayscale else images[..., 0]
        result_view = result if implicit_grayscale else result[..., 0]
        gradient_view = gradient if implicit_grayscale else gradient[..., 0]
        np.multiply(image_view, gradient_view, out=result_view)
        np.minimum(result_view, 1.0, out=result_view)
        return result

    # Valid illumination multipliers remain nonnegative, so only the upper image bound can be crossed.
    for index, image in enumerate(images):
        np.multiply(image, gradient, out=result[index])
        np.minimum(result[index], 1.0, out=result[index])
    return result


def apply_illumination_batch(
    images: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    mode: Literal["linear", "corner", "gaussian"],
    **params: Any,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Apply one sampled illumination pattern to an image batch, with empty-input handling, gradient reuse, kernel
    routing, and clipping in one functional operation.

    Args:
        images (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input batch in `(N, H, W, C)` format, or
            `(N, H, W)` for direct implicit-grayscale calls.
        mode (Literal['linear', 'corner', 'gaussian']): Illumination pattern.
        **params (Any): Sampled parameters for the selected mode.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Illuminated batch with the input shape and dtype.

    """
    if images.shape[0] == 0:
        return images.copy()

    if mode == "linear":
        return apply_linear_illumination_batch(images, **params)

    height, width = images.shape[1:3]
    implicit_grayscale = images.ndim == NUM_MULTI_CHANNEL_DIMENSIONS
    gradient = create_illumination_gradient(height, width, mode, params)
    if not implicit_grayscale:
        gradient = gradient[..., np.newaxis]
    spots = params.get("spots")
    gaussian_brightens = mode == "gaussian" and (
        any(spot[3] > 0 for spot in spots) if spots is not None else params["intensity"] > 0
    )
    clip_required = (mode == "corner" and params["intensity"] < 0) or gaussian_brightens
    num_channels = 1 if implicit_grayscale else images.shape[-1]

    # The uint8 path regresses against the per-image kernels.
    if images.dtype == np.uint8:
        result = np.empty_like(images)
        for index, image in enumerate(images):
            result[index] = multiply(image, gradient)
    elif num_channels <= 4 and clip_required:
        result = _apply_clipped_illumination_batch(images, gradient, implicit_grayscale, num_channels)
    else:
        result = multiply(images, gradient)
        if images.dtype in (np.float16, np.float32) and clip_required:
            result = clip(result, out=result)
    return result


def apply_corner_illumination(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    intensity: float,
    corner: Literal[0, 1, 2, 3],
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Apply corner illumination (darkened corners) to the image. Gradient from center; intensity and
    corner (0-3) control strength and which corner. Floating results are clipped to the input dtype range.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image
        intensity (float): Illumination intensity
        corner (Literal[0, 1, 2, 3]): The corner to apply the illumination to.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Image with corner illumination applied.

    """
    if intensity == 0:
        result = img.copy()
    else:
        height, width = img.shape[:2]
        pattern = create_corner_illumination_gradient(height, width, intensity, corner)

        if img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
            num_channels = img.shape[2]
            pattern = pattern[..., np.newaxis] if num_channels == 1 else cv2.merge([pattern] * num_channels)

        result = multiply(img, pattern)
    return result if img.dtype == np.uint8 else clip(result, img.dtype)


def apply_gaussian_illumination(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    intensity: float,
    center: tuple[float, float],
    sigma: float,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Add a Gaussian-shaped bright or dark spot; intensity, center, and sigma define the falloff.
    Use for spotlight or vignette effects. Same channel count; floating results are clipped to the input range.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image
        intensity (float): Illumination intensity
        center (tuple[float, float]): The center of the illumination.
        sigma (float): The sigma of the illumination.

    """
    if intensity == 0:
        result = img.copy()
    else:
        height, width = img.shape[:2]
        x = create_gaussian_illumination_gradient(height, width, intensity, center, sigma)

        if img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
            num_channels = img.shape[2]
            x = x[..., np.newaxis] if num_channels == 1 else cv2.merge([x] * num_channels)

        result = multiply(img, x)
    return result if img.dtype == np.uint8 else clip(result, img.dtype)


def apply_gaussian_illumination_spots(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    spots: Sequence[tuple[float, float, float, float]],
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Apply a multiplicatively combined field of independently parameterized Gaussian spots to an image and preserve
    its input dtype.
    """
    if all(spot[3] == 0 for spot in spots):
        result = img.copy()
    else:
        height, width = img.shape[:2]
        x = create_gaussian_illumination_gradient_from_spots(height, width, spots)

        if img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
            num_channels = img.shape[2]
            x = x[..., np.newaxis] if num_channels == 1 else cv2.merge([x] * num_channels)

        result = multiply(img, x)
    return result if img.dtype == np.uint8 else clip(result, img.dtype)


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

    return apply_uint8_lut(img, lut)


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
    result = apply_uint8_lut(channel, lut)
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
    num_channels = get_num_channels(img)
    lut = np.empty((256, 1, num_channels), dtype=np.uint8)
    identity_lut = np.arange(256, dtype=np.uint8)
    for channel_idx in range(num_channels):
        channel = img[..., channel_idx]
        hist = cv2.calcHist([channel], [0], None, [256], [0, max_value]).ravel()
        channel_lut = _create_auto_contrast_lut(hist, cutoff, None, method, max_value)
        lut[:, 0, channel_idx] = identity_lut if channel_lut is None else channel_lut

    return apply_uint8_lut(img, lut)


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

        channel_hist = hists[channel_idx]
        if channel_hist is None:
            continue

        lut = _create_auto_contrast_lut(channel_hist, cutoff, ignore, method, max_value)
        if lut is None:
            continue
        result[..., channel_idx] = apply_uint8_lut(cast("ImageUInt8", channel), lut, out=cast("ImageUInt8", channel))

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


def auto_contrast(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    cutoff: float,
    ignore: int | None,
    method: Literal["cdf", "pil"],
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Apply automatic contrast enhancement. Stretches histogram to full range. cutoff, ignore, method
    (cdf or pil) limit clip. uint8 I/O.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image
        cutoff (float): Cutoff percentage for histogram
        ignore (int | None): Value to ignore in histogram
        method (Literal['cdf', 'pil']): Method to use for contrast enhancement

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Image with enhanced contrast

    """
    img, input_dtype = to_uint8_working_image(img)
    num_channels = get_num_channels(img)
    max_value = int(MAX_VALUES_BY_DTYPE[img.dtype])

    if method == "pil" and cutoff == 0 and ignore is None and num_channels == 1:
        result = _auto_contrast_pil_zero_cutoff(img, max_value)
        return restore_uint8_working_image(result, input_dtype)

    if img.ndim == MONO_CHANNEL_DIMENSIONS:
        result = _auto_contrast_single_channel(img, cutoff, ignore, method, max_value)
        return restore_uint8_working_image(result, input_dtype)

    if _should_use_auto_contrast_multichannel_lut(img, ignore, method, num_channels):
        result = _auto_contrast_multichannel_lut(img, cutoff, method, max_value)
        return restore_uint8_working_image(result, input_dtype)

    result = _auto_contrast_multichannel_hist(img, cutoff, ignore, method, max_value)
    return restore_uint8_working_image(result, input_dtype)


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
    if dtype.type in (np.float16, np.float32):
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
    """Return the first usable 2D mask plane from singular or batched 2D and 3D targets when no image data is available
    during parameter sampling.
    """
    if "mask" in data:
        return data["mask"]
    if "masks" in data and data["masks"].shape[0] > 0:
        return data["masks"][0]
    if "mask3d" in data and data["mask3d"].shape[0] > 0:
        return data["mask3d"][0]
    return None


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
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    intensity: float,
    center_x: float,
    center_y: float,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Apply vignetting by darkening corners with radial gradient. intensity, center_x, center_y
    control strength and center. Supports (H,W) or (H,W,C).

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image of shape (H, W) or (H, W, C).
        intensity (float): Strength of darkening at corners, in [0, 1].
        center_x (float): Horizontal center of the vignette as fraction of width, in [0, 1].
        center_y (float): Vertical center of the vignette as fraction of height, in [0, 1].

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Image with vignetting applied.

    """
    height, width = img.shape[:2]

    vignette_mask = _create_vignette_mask(height, width, intensity, center_x, center_y)

    return multiply(img, vignette_mask[:, :, np.newaxis])


def apply_film_grain(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    grain: np.ndarray,
    intensity: float,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Apply film grain noise to an image (2D or 3D). Luminance-dependent and
    spatially correlated; params control grain size and intensity.

    Film grain is luminance-dependent and spatially correlated, unlike simple Gaussian noise.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image, shape (H, W, C) or (H, W, 1).
        grain (np.ndarray): Pre-generated grain pattern, shape (H, W), float32.
        intensity (float): Grain strength multiplier.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Image with film grain applied.

    """
    num_channels = img.shape[-1]

    luminance = mean(img, axis=-1) if num_channels > 1 else img[..., 0]

    max_val = MAX_VALUES_BY_DTYPE[img.dtype]

    inv_lum = 1.0 - np.asarray(luminance).astype(np.float32) / max_val if img.dtype == np.uint8 else 1.0 - luminance

    modulated = (grain * inv_lum * intensity * max_val).astype(np.float32)

    result = add(img, modulated[..., np.newaxis])
    return clip(result, out=result) if img.dtype in (np.float16, np.float32) else result


def apply_halftone(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    dot_size: int,
    blend: float,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Convert image to halftone dot pattern. Simulates printed halftone; dot size
    varies by intensity. Params: scale, dot_shape.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image (H, W, C), uint8 or float32.
        dot_size (int): Size of each halftone grid cell in pixels.
        blend (float): Blend factor between halftone and original. 0 = pure halftone, 1 = original.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Image with halftone effect applied.

    """
    img, input_dtype = to_uint8_working_image(img)
    img = np.ascontiguousarray(img)
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
        result = add_weighted(img, blend, canvas, 1.0 - blend)
        return restore_uint8_working_image(result, input_dtype)

    return restore_uint8_working_image(canvas, input_dtype)


def apply_lens_flare(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    flare_center: tuple[int, int],
    ghosts: list[tuple[int, int, int, float]],
    starburst_angles: np.ndarray,
    starburst_intensity: float,
    bloom_radius: int,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Apply realistic lens flare with ghosts and starburst. Params control position,
    intensity, and number of ghosts. RGB input.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image (H, W, C), must be 3-channel.
        flare_center (tuple[int, int]): (x, y) position of the flare source.
        ghosts (list[tuple[int, int, int, float]]): List of (x, y, radius, alpha) for each ghost circle.
        starburst_angles (np.ndarray): Array of angles in radians for starburst rays.
        starburst_intensity (float): Brightness of starburst rays, 0-1.
        bloom_radius (int): Gaussian blur radius for bloom effect.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Image with lens flare applied.

    """
    height, width = img.shape[:2]
    max_val = MAX_VALUES_BY_DTYPE[img.dtype]
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
        flare_layer = cast("NDArray[np.float32]", cv2.GaussianBlur(flare_layer, (ksize, ksize), 0))

    for gx, gy, gradius, galpha in ghosts:
        ghost = np.zeros((height, width), dtype=np.float32)
        cv2.circle(ghost, (gx, gy), gradius, 1.0, -1, lineType=cv2.LINE_AA)
        if gradius > 2:
            gk = max(3, gradius | 1)
            ghost = cast("NDArray[np.float32]", cv2.GaussianBlur(ghost, (gk, gk), 0))
        flare_layer += ghost * galpha

    np.clip(flare_layer, 0, 1, out=flare_layer)

    flare_3d = flare_layer[:, :, np.newaxis] * max_val

    return clip(np.add(img, flare_3d, dtype=np.float32), img.dtype)


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
        phase_field = 2 * np.pi * freq * projection + phase
        sine = nk.sin(phase_field.reshape(-1))
        if sine is None:
            raise RuntimeError("nk.sin returned None")
        displacement_buffer = nk.scale(sine, alpha=amp, beta=0.0)
        if displacement_buffer is None:
            raise RuntimeError("nk.scale returned None")
        displacement = np.asarray(displacement_buffer).reshape(height, width)

        dx += displacement * (-wave_y)
        dy += displacement * wave_x

    map_x = x + dx
    map_y = y + dy

    return map_x.astype(np.float32), map_y.astype(np.float32)


__all__ = [
    "_auto_contrast_single_channel",
    "_create_vignette_mask",
    "apply_corner_illumination",
    "apply_film_grain",
    "apply_gaussian_illumination",
    "apply_gaussian_illumination_spots",
    "apply_halftone",
    "apply_illumination_batch",
    "apply_lens_flare",
    "apply_linear_illumination",
    "apply_linear_illumination_batch",
    "apply_plasma_brightness_contrast",
    "apply_plasma_brightness_contrast_batch",
    "apply_plasma_shadow",
    "apply_vignette",
    "auto_contrast",
    "create_contrast_lut",
    "create_corner_illumination_gradient",
    "create_directional_gradient",
    "create_gaussian_illumination_gradient",
    "create_gaussian_illumination_gradient_from_spots",
    "create_illumination_gradient",
    "generate_plasma_pattern",
    "generate_random_values",
    "generate_water_displacement_maps",
    "get_drop_mask",
    "get_histogram_bounds",
    "get_mask_array",
    "prepare_drop_values",
]
