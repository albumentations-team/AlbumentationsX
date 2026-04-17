"""Functional implementations of image augmentation operations.

This module contains low-level functions for various image augmentation techniques including
color transformations, blur effects, tone curve adjustments, noise additions, and other visual
modifications. These functions form the foundation for the transform classes and provide
the core functionality for manipulating image data during the augmentation process.
"""

import math
import random
from collections.abc import Sequence
from typing import Any, Literal
from warnings import warn

import cv2
import numpy as np
from albucore import (
    MAX_VALUES_BY_DTYPE,
    add,
    add_array,
    add_constant,
    add_vector,
    add_weighted,
    clip,
    clipped,
    float32_io,
    from_float,
    get_num_channels,
    is_grayscale_image,
    is_rgb_image,
    maybe_process_in_chunks,
    mean,
    multiply,
    multiply_add,
    multiply_by_array,
    multiply_by_constant,
    normalize_per_image,
    power,
    preserve_channel_dim,
    reduce_sum,
    reshape_ndhwc_channel,
    reshape_xhwc_channel,
    restore_ndhwc_channel,
    restore_xhwc_channel,
    std,
    sz_lut,
    to_float,
    uint8_io,
)

import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.augmentations.utils import (
    PCA,
    non_rgb_error,
)
from albumentations.core.type_definitions import (
    MONO_CHANNEL_DIMENSIONS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    NUM_RGB_CHANNELS,
    ImageType,
    ImageUInt8,
)


@uint8_io
@preserve_channel_dim
def shift_hsv(
    img: ImageType,
    hue_shift: float,
    sat_shift: float,
    val_shift: float,
) -> ImageType:
    """Shift hue, saturation, and value in HSV space. hue_shift, sat_shift, val_shift control
    amount; grayscale gets value only. uint8 I/O.

    Args:
        img (ImageType): The image to shift.
        hue_shift (float): The amount to shift the hue.
        sat_shift (float): The amount to shift the saturation.
        val_shift (float): The amount to shift the value.

    Returns:
        ImageType: The shifted image.

    """
    if hue_shift == 0 and sat_shift == 0 and val_shift == 0:
        return img

    is_gray = is_grayscale_image(img)

    if is_gray:
        if hue_shift != 0 or sat_shift != 0:
            hue_shift = 0
            sat_shift = 0
            warn(
                "HueSaturationValue: hue_shift and sat_shift are not applicable to grayscale image. "
                "Set them to 0 or use RGB image",
                stacklevel=2,
            )
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(img)

    if hue_shift != 0:
        lut_hue = np.arange(0, 256, dtype=np.int16)
        lut_hue = np.mod(lut_hue + hue_shift, 180).astype(np.uint8)
        hue = sz_lut(hue, lut_hue, inplace=False)

    if sat_shift != 0:
        # Create a mask for all grayscale pixels (S=0)
        # These should remain grayscale regardless of saturation change
        grayscale_mask = sat == 0

        # Apply saturation shift only to non-white pixels
        sat = add_constant(sat, sat_shift, inplace=True)

        # Reset saturation for white pixels
        sat[grayscale_mask] = 0

    if val_shift != 0:
        val = add_constant(val, val_shift, inplace=True)

    img = cv2.merge((hue, sat, val))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if is_gray else img


def shift_hsv_images(
    images: np.ndarray,
    hue_shift: float,
    sat_shift: float,
    val_shift: float,
) -> np.ndarray:
    """Apply HSV shift to a batch of images (N, H, W, C) or (N, H, W). Uses pre-allocated
    output and per-frame shift_hsv; faster than stack loop for video batches.

    Uses a pre-allocated output array with per-frame shift_hsv calls.
    This is faster than the base-class np.stack loop because it avoids
    N intermediate allocations and the final stack copy.

    A reshape-to-tall-image approach (N,H,W,3) -> (N*H,W,3) was benchmarked
    but was slower for images >= 512x512 due to cv2.cvtColor and cv2.LUT
    cache thrashing on the large intermediate arrays.
    """
    res = np.empty_like(images)
    for i in range(images.shape[0]):
        res[i] = shift_hsv(images[i], hue_shift, sat_shift, val_shift)
    return res


@clipped
def solarize(img: ImageType, threshold: float) -> ImageType:
    """Invert pixel values above a normalized threshold (solarization). threshold in [0, 1];
    works for uint8 and float32; pixels below threshold unchanged.

    Args:
        img (ImageType): The image to solarize. Can be uint8 or float32.
        threshold (float): Normalized threshold value in range [0, 1].
            For uint8 images: pixels above threshold * 255 are inverted
            For float32 images: pixels above threshold are inverted

    Returns:
        ImageType: Solarized image.

    Note:
        The threshold is normalized to [0, 1] range for both uint8 and float32 images.
        For uint8 images, the threshold is internally scaled by 255.

    """
    dtype = img.dtype
    max_val = MAX_VALUES_BY_DTYPE[dtype]

    if dtype == np.uint8:
        indices = np.arange(int(max_val) + 1, dtype=dtype)
        thresh_val = threshold * max_val
        lut = np.where(indices >= thresh_val, max_val - indices, indices).astype(dtype)
        prev_shape = img.shape
        img = sz_lut(img, lut, inplace=False)
        return img if len(prev_shape) == img.ndim else np.expand_dims(img, -1)
    return np.where(img >= threshold, max_val - img, img)


@uint8_io
@clipped
def posterize(img: ImageType, bits: Literal[1, 2, 3, 4, 5, 6, 7] | list[Literal[1, 2, 3, 4, 5, 6, 7]]) -> ImageType:
    """Reduce bit depth by keeping only the highest N bits per channel. bits: 1-7 or list per
    channel; LUT-based. uint8 I/O, clipped.

    Args:
        img (ImageType): Input image. Can be single or multi-channel.
        bits (Literal[1, 2, 3, 4, 5, 6, 7] | list[Literal[1, 2, 3, 4, 5, 6, 7]]): Number of high bits to keep..
            Can be either:
            - A single value to apply the same bit reduction to all channels
            - A list of values to apply different bit reduction per channel.
              Length of list must match number of channels in image.

    Returns:
        ImageType: Image with reduced bit depth. Has same shape and dtype as input.

    Note:
        - The transform keeps the N highest bits and sets all other bits to 0
        - For example, if bits=3:
            - Original value: 11010110 (214)
            - Keep 3 bits:   11000000 (192)
        - The number of unique colors per channel will be 2^bits
        - Higher bits values = more colors = more subtle effect
        - Lower bits values = fewer colors = more dramatic posterization

    Examples:
        >>> import numpy as np
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> # Same posterization for all channels
        >>> result = posterize(image, bits=3)
        >>> # Different posterization per channel
        >>> result = posterize(image, bits=[3, 4, 5])  # RGB channels

    """
    bits_array = np.uint8(bits)

    if not bits_array.shape or len(bits_array) == 1:
        lut = np.arange(0, 256, dtype=np.uint8)
        mask = ~np.uint8(2 ** (8 - bits_array) - 1)
        lut &= mask

        return sz_lut(img, lut, inplace=False)

    result_img = np.empty_like(img)
    for i, channel_bits in enumerate(bits_array):
        lut = np.arange(0, 256, dtype=np.uint8)
        mask = ~np.uint8(2 ** (8 - channel_bits) - 1)
        lut &= mask

        result_img[..., i] = sz_lut(img[..., i], lut, inplace=True)

    return result_img


def _equalize_pil(img: ImageType, mask: np.ndarray | None = None) -> ImageType:
    histogram = cv2.calcHist([img], [0], mask, [256], (0, 256)).ravel()
    h = histogram[histogram > 0]

    if len(h) <= 1:
        return img.copy()

    step = reduce_sum(h[:-1]) // 255
    if not step:
        return img.copy()

    lut = np.minimum((np.cumsum(histogram) + step // 2) // step, 255).astype(np.uint8)

    return sz_lut(img, lut, inplace=True)


def _equalize_cv(img: ImageType, mask: np.ndarray | None = None) -> ImageType:
    if mask is None:
        return cv2.equalizeHist(img)

    histogram = cv2.calcHist([img], [0], mask, [256], (0, 256)).ravel()

    # Find the first non-zero index with a numpy operation
    i = np.flatnonzero(histogram)[0] if np.any(histogram) else 255

    total = reduce_sum(histogram)

    # Safe division for equalize: handle edge case of uniform histograms
    # If histogram is uniform (denominator == 0), return image unchanged
    denominator = total - histogram[i]
    if denominator == 0:
        # Uniform histogram - no equalization needed
        return img

    scale = 255.0 / denominator
    # Optimize cumulative sum and scale to generate LUT
    cumsum_histogram = np.cumsum(histogram)
    lut = np.clip(((cumsum_histogram - cumsum_histogram[i]) * scale).round(), 0, 255).astype(np.uint8)

    return sz_lut(img, lut, inplace=True)


def _check_preconditions(
    img: ImageType,
    mask: np.ndarray | None,
    by_channels: bool,
) -> None:
    if mask is not None:
        if is_rgb_image(mask) and is_grayscale_image(img):
            raise ValueError(
                f"Wrong mask shape. Image shape: {img.shape}. Mask shape: {mask.shape}",
            )
        if not by_channels and not is_grayscale_image(mask):
            msg = f"When by_channels=False only 1-channel mask supports. Mask shape: {mask.shape}"
            raise ValueError(msg)


def _handle_mask(
    mask: np.ndarray | None,
    i: int | None = None,
) -> np.ndarray | None:
    if mask is None:
        return None
    mask = mask.astype(
        np.uint8,
        copy=False,
    )  # Use copy=False to avoid unnecessary copying
    # Check for grayscale image and avoid slicing if i is None
    if i is not None and not is_grayscale_image(mask):
        mask = mask[..., i]

    return mask


@uint8_io
@preserve_channel_dim
def equalize(
    img: ImageType,
    mask: np.ndarray | None = None,
    mode: Literal["cv", "pil"] = "cv",
    by_channels: bool = True,
) -> ImageType:
    """Apply histogram equalization to the input image. Enhances contrast; supports
    grayscale and color, mode cv/pil, optional mask and by_channels.

    This function enhances the contrast of the input image by equalizing its histogram.
    It supports both grayscale and color images, and can operate on individual channels
    or on the luminance channel of the image.

    Args:
        img (ImageType): Input image. Can be grayscale (2D array) or RGB (3D array).
        mask (np.ndarray | None): Optional mask to apply the equalization selectively.
            If provided, must have the same shape as the input image. Default: None.
        mode (Literal['cv', 'pil']): The backend to use for equalization. Can be either "cv" for
            OpenCV or "pil" for Pillow-style equalization. Default: "cv".
        by_channels (bool): If True, applies equalization to each channel independently.
            If False, converts the image to YCrCb color space and equalizes only the
            luminance channel. Only applicable to color images. Default: True.

    Returns:
        ImageType: Equalized image. The output has the same dtype as the input.

    Raises:
        ValueError: If the input image or mask have invalid shapes or types.

    Note:
        - If the input image is not uint8, it will be temporarily converted to uint8
          for processing and then converted back to its original dtype.
        - For color images, when by_channels=False, the image is converted to YCrCb
          color space, equalized on the Y channel, and then converted back to RGB.
        - The function preserves the original number of channels in the image.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> equalized = A.equalize(image, mode="cv", by_channels=True)
        >>> assert equalized.shape == image.shape
        >>> assert equalized.dtype == image.dtype

    """
    _check_preconditions(img, mask, by_channels)
    function = _equalize_pil if mode == "pil" else _equalize_cv

    if is_grayscale_image(img):
        return function(img, _handle_mask(mask))

    if not by_channels:
        result_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        result_img[..., 0] = function(result_img[..., 0], _handle_mask(mask))
        return cv2.cvtColor(result_img, cv2.COLOR_YCrCb2RGB)

    result_img = np.empty_like(img)
    for i in range(NUM_RGB_CHANNELS):
        _mask = _handle_mask(mask, i)
        # Extract channel, process, and ensure we maintain 2D shape
        channel_result = function(img[..., i], _mask)
        # Remove any extra dimensions that might have been added
        if channel_result.ndim > 2:
            channel_result = channel_result.squeeze()
        result_img[..., i] = channel_result

    return result_img


def evaluate_bez(
    low_y: float | np.ndarray,
    high_y: float | np.ndarray,
) -> np.ndarray:
    """Evaluate the Bezier curve at the given t values. Used for tone-curve control points;
    returns y coordinates for input t in [0, 1].

    Args:
        t (np.ndarray): The t values to evaluate the Bezier curve at.
        low_y (float | np.ndarray): The low y values to evaluate the Bezier curve at.
        high_y (float | np.ndarray): The high y values to evaluate the Bezier curve at.

    Returns:
        np.ndarray: The Bezier curve values.

    """
    t = np.linspace(0.0, 1.0, 256)[..., None]

    one_minus_t = 1 - t
    return (3 * one_minus_t**2 * t * low_y + 3 * one_minus_t * t**2 * high_y + t**3) * 255


@uint8_io
def move_tone_curve(
    img: ImageType,
    low_y: float | np.ndarray,
    high_y: float | np.ndarray,
    num_channels: int,
) -> ImageType:
    """Rescale bright/dark via Bezier tone curve. low_y, high_y (per-channel or scalar), num_channels
    for per-channel curves. uint8 I/O.

    Args:
        img (ImageType): Any number of channels
        low_y (float | np.ndarray): per-channel or single y-position of a Bezier control point used
            to adjust the tone curve, must be in range [0, 1]
        high_y (float | np.ndarray): per-channel or single y-position of a Bezier control point used
            to adjust image tone curve, must be in range [0, 1]
        num_channels (int): The number of channels in the input image.

    Returns:
        ImageType: Image with adjusted tone curve

    """
    if np.isscalar(low_y) and np.isscalar(high_y):
        lut = clip(np.rint(evaluate_bez(low_y, high_y)), np.uint8, inplace=False)
        return sz_lut(img, lut, inplace=False)

    if isinstance(low_y, np.ndarray) and isinstance(high_y, np.ndarray):
        luts = clip(
            np.rint(evaluate_bez(low_y, high_y).T),
            np.uint8,
            inplace=False,
        )
        result = np.empty_like(img)
        for i in range(num_channels):
            result[..., i] = sz_lut(img[..., i], np.ascontiguousarray(luts[i]), inplace=False)
        return result

    raise TypeError(
        f"low_y and high_y must both be of type float or np.ndarray. Got {type(low_y)} and {type(high_y)}",
    )


@clipped
def linear_transformation_rgb(
    img: ImageType,
    transformation_matrix: np.ndarray,
) -> ImageType:
    """3x3 linear transformation to RGB. transformation_matrix (or batch) multiplies channel
    vector. Supports (H,W,3), (B,H,W,3), (B,D,H,W,3).

    This function applies a 3x3 linear transformation matrix (or batch of matrices)
    to the RGB channels of either a single image or a batch of images.

    Args:
        img (ImageType): A single RGB image of shape (H, W, 3), or a batch of images (B, H, W, 3),
            or a batch of volumes (B, D, H, W, 3).
        transformation_matrix (np.ndarray): A 3x3 matrix

    Returns:
        ImageType: Transformed image or batch of images, matching the input shape and dtype.

    Raises:
        ValueError: If input shapes do not conform to the supported configurations.

    """
    if img.ndim == 3:
        return cv2.transform(img, transformation_matrix)
    if img.ndim == 4:
        transformed, original_shape = reshape_xhwc_channel(img)
        transformed = cv2.transform(transformed, transformation_matrix)
        return restore_xhwc_channel(transformed, original_shape)
    if img.ndim == 5:
        transformed, original_shape = reshape_ndhwc_channel(img)
        transformed = cv2.transform(transformed, transformation_matrix)
        return restore_ndhwc_channel(transformed, original_shape)
    raise ValueError(f"Expected input shape (H, W, 3), (B, H, W, 3), (B, D, H, W, 3), got {img.shape}")


@uint8_io
@preserve_channel_dim
def clahe(
    img: ImageType,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> ImageType:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) per tile. clip_limit, tile_grid_size.
    For color, applied in LAB to L channel. uint8 I/O.

    This function enhances the contrast of the input image using CLAHE. For color images,
    it converts the image to the LAB color space, applies CLAHE to the L channel, and then
    converts the image back to RGB.

    Args:
        img (ImageType): Input image. Can be grayscale (2D array) or RGB (3D array).
        clip_limit (float): Threshold for contrast limiting. Higher values give more contrast.
        tile_grid_size (tuple[int, int]): Size of grid for histogram equalization.
            Width and height of the grid.

    Returns:
        ImageType: Image with CLAHE applied. The output has the same dtype as the input.

    Note:
        - If the input image is float32, it's temporarily converted to uint8 for processing
          and then converted back to float32.
        - For color images, CLAHE is applied only to the luminance channel in the LAB color space.

    Raises:
        ValueError: If the input image is not 2D or 3D.

    Examples:
        >>> import numpy as np
        >>> img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> result = clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
        >>> assert result.shape == img.shape
        >>> assert result.dtype == img.dtype

    """
    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if is_grayscale_image(img):
        return clahe_mat.apply(img)

    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_lab[:, :, 0] = clahe_mat.apply(img_lab[:, :, 0])

    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)


@uint8_io
@preserve_channel_dim
def image_compression(
    img: ImageType,
    quality: int,
    image_type: Literal[".jpg", ".webp"],
) -> ImageType:
    """Compress image with JPEG or WebP to simulate artifacts. quality, image_type (.jpg/.webp).
    Lower quality increases blockiness. uint8 I/O.

    Args:
        img (ImageType): Input image
        quality (int): Quality of compression in range [1, 100]
        image_type (Literal['.jpg', '.webp']): Type of compression to use

    Returns:
        ImageType: Compressed image

    """
    # Determine the quality flag for compression
    quality_flag = cv2.IMWRITE_JPEG_QUALITY if image_type == ".jpg" else cv2.IMWRITE_WEBP_QUALITY
    num_channels = get_num_channels(img)

    # Prepare to encode and decode
    def encode_decode(src_img: ImageType, read_mode: int) -> np.ndarray:
        _, encoded_img = cv2.imencode(image_type, src_img, (int(quality_flag), quality))
        return cv2.imdecode(encoded_img, read_mode)

    if num_channels == 1:
        # Grayscale image
        decoded = encode_decode(img, cv2.IMREAD_GRAYSCALE)
        return decoded[..., np.newaxis]  # Add channel dimension back

    if num_channels in (2, NUM_RGB_CHANNELS):
        # 2 channels: pad to 3, or 3 (RGB) channels
        padded_img = np.pad(img, ((0, 0), (0, 0), (0, 1)), mode="constant") if num_channels == 2 else img
        decoded_bgr = encode_decode(padded_img, cv2.IMREAD_UNCHANGED)
        return decoded_bgr[..., :num_channels]  # Return only the required number of channels

    # More than 3 channels
    bgr = img[..., :NUM_RGB_CHANNELS]
    decoded_bgr = encode_decode(bgr, cv2.IMREAD_UNCHANGED)

    # Process additional channels
    extra_channels = [
        encode_decode(img[..., i], cv2.IMREAD_GRAYSCALE)[..., np.newaxis] for i in range(NUM_RGB_CHANNELS, num_channels)
    ]
    return np.dstack([decoded_bgr, *extra_channels])


@uint8_io
def add_snow_bleach(
    img: ImageType,
    snow_point: float,
    brightness_coeff: float,
) -> ImageType:
    """Add a simple snow effect by bleaching out pixels. Brightness increase and
    optional mask; used as a building block for more complex snow augmentations.

    This function simulates a basic snow effect by increasing the brightness of pixels
    that are above a certain threshold (snow_point). It operates in the HLS color space
    to modify the lightness channel.

    Args:
        img (ImageType): Input image. Can be either RGB uint8 or float32.
        snow_point (float): A float in the range [0, 1], scaled and adjusted to determine
            the threshold for pixel modification. Higher values result in less snow effect.
        brightness_coeff (float): Coefficient applied to increase the brightness of pixels
            below the snow_point threshold. Larger values lead to more pronounced snow effects.
            Should be greater than 1.0 for a visible effect.

    Returns:
        ImageType: Image with simulated snow effect. The output has the same dtype as the input.

    Note:
        - This function converts the image to the HLS color space to modify the lightness channel.
        - The snow effect is created by selectively increasing the brightness of pixels.
        - This method tends to create a 'bleached' look, which may not be as realistic as more
          advanced snow simulation techniques.
        - The function automatically handles both uint8 and float32 input images.

    The snow effect is created through the following steps:
    1. Convert the image from RGB to HLS color space.
    2. Adjust the snow_point threshold.
    3. Increase the lightness of pixels below the threshold.
    4. Convert the image back to RGB.

    Mathematical Formulation:
        Let L be the lightness channel in HLS space.
        For each pixel (i, j):
        If L[i, j] < snow_point:
            L[i, j] = L[i, j] * brightness_coeff

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> snowy_image = A.functional.add_snow_v1(image, snow_point=0.5, brightness_coeff=1.5)

    References:
        - HLS Color Space: https://en.wikipedia.org/wiki/HSL_and_HSV
        - Original implementation: https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    """
    max_value = MAX_VALUES_BY_DTYPE[np.uint8]

    # Precompute snow_point threshold
    snow_point = (snow_point * max_value / 2) + (max_value / 3)

    # Convert image to HLS color space once and avoid repeated dtype casting
    image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    lightness_channel = image_hls[:, :, 1].astype(np.float32)

    # Utilize boolean indexing for efficient lightness adjustment
    mask = lightness_channel < snow_point
    lightness_channel[mask] *= brightness_coeff

    # Clip the lightness values in place
    lightness_channel = clip(lightness_channel, np.uint8, inplace=True)

    # Update the lightness channel in the original image
    image_hls[:, :, 1] = lightness_channel

    # Convert back to RGB
    return cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)


def generate_snow_textures(
    img_shape: tuple[int, int],
    random_generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate snow texture and sparkle mask for add_snow_texture. Returns texture
    and mask arrays; uses random generator for reproducibility.

    Args:
        img_shape (tuple[int, int]): Image shape.
        random_generator (np.random.Generator): Random generator to use.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (snow_texture, sparkle_mask) arrays.

    """
    # Generate base snow texture
    snow_texture = random_generator.normal(size=img_shape[:2], loc=0.5, scale=0.3)
    snow_texture = cv2.GaussianBlur(snow_texture, (0, 0), sigmaX=1, sigmaY=1)

    # Generate sparkle mask
    sparkle_mask = random_generator.random(img_shape[:2]) > 0.99

    return snow_texture, sparkle_mask


@uint8_io
def add_snow_texture(
    img: ImageType,
    snow_point: float,
    brightness_coeff: float,
    snow_texture: np.ndarray,
    sparkle_mask: np.ndarray,
) -> ImageType:
    """Add snow effect: texture overlay, sparkle, depth gradient, blue tint. snow_point,
    brightness_coeff; takes precomputed snow_texture and sparkle_mask. uint8 I/O.

    This function simulates snowfall by applying multiple visual effects to the image,
    including brightness adjustment, snow texture overlay, depth simulation, and color tinting.
    The result is a more natural-looking snow effect compared to simple pixel bleaching methods.

    Args:
        img (ImageType): Input image in RGB format.
        snow_point (float): Coefficient that controls the amount and intensity of snow.
            Should be in the range [0, 1], where 0 means no snow and 1 means maximum snow effect.
        brightness_coeff (float): Coefficient for brightness adjustment to simulate the
            reflective nature of snow. Should be in the range [0, 1], where higher values
            result in a brighter image.
        snow_texture (np.ndarray): Snow texture.
        sparkle_mask (np.ndarray): Sparkle mask.

    Returns:
        ImageType: Image with added snow effect. The output has the same dtype as the input.

    Note:
        - The function first converts the image to HSV color space for better control over
          brightness and color adjustments.
        - A snow texture is generated using Gaussian noise and then filtered for a more
          natural appearance.
        - A depth effect is simulated, with more snow at the top of the image and less at the bottom.
        - A slight blue tint is added to simulate the cool color of snow.
        - Random sparkle effects are added to simulate light reflecting off snow crystals.

    The snow effect is created through the following steps:
    1. Brightness adjustment in HSV space
    2. Generation of a snow texture using Gaussian noise
    3. Application of a depth effect to the snow texture
    4. Blending of the snow texture with the original image
    5. Addition of a cool blue tint
    6. Addition of sparkle effects

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> snowy_image = A.functional.add_snow_v2(image, snow_coeff=0.5, brightness_coeff=0.2)

    Note:
        This function works with both uint8 and float32 image types, automatically
        handling the conversion between them.

    References:
        - Perlin Noise: https://en.wikipedia.org/wiki/Perlin_noise
        - HSV Color Space: https://en.wikipedia.org/wiki/HSL_and_HSV

    """
    max_value = MAX_VALUES_BY_DTYPE[np.uint8]

    # Convert to HSV for better color control
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Increase brightness
    np.multiply(img_hsv[:, :, 2], 1 + brightness_coeff * snow_point, out=img_hsv[:, :, 2])
    np.clip(img_hsv[:, :, 2], 0, max_value, out=img_hsv[:, :, 2])

    # Generate snow texture
    snow_texture = cv2.GaussianBlur(snow_texture, (0, 0), sigmaX=1, sigmaY=1)

    # Create depth effect for snow simulation
    # More snow accumulates at the top of the image, gradually decreasing towards the bottom
    # This simulates natural snow distribution on surfaces
    # The effect is achieved using a linear gradient from 1 (full snow) to 0.2 (less snow)
    rows = img.shape[0]
    depth_effect = np.linspace(1, 0.2, rows)[:, np.newaxis]
    snow_texture *= depth_effect

    # Apply snow texture
    snow_layer = (snow_texture[:, :, np.newaxis] * (max_value * snow_point)).astype(
        np.float32,
    )

    # Blend snow with original image
    img_with_snow = cv2.add(img_hsv, snow_layer)

    # Add a slight blue tint to simulate cool snow color
    blue_tint = np.full_like(img_with_snow, (0.6, 0.75, 1))  # Slight blue in HSV

    img_with_snow = cv2.addWeighted(
        img_with_snow,
        0.85,
        blue_tint,
        0.15 * snow_point,
        0,
    )

    # Convert back to RGB
    img_with_snow = cv2.cvtColor(img_with_snow.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Add some sparkle effects for snow glitter
    img_with_snow[sparkle_mask] = [max_value, max_value, max_value]

    return img_with_snow


@uint8_io
@preserve_channel_dim
def add_rain(
    img: ImageType,
    slant: float,
    drop_length: int,
    drop_width: int,
    drop_color: tuple[int, int, int],
    blur_value: int,
    brightness_coefficient: float,
    rain_drops: np.ndarray,
) -> ImageType:
    """Add rain streaks. slant, drop_length, drop_width, drop_color, blur_value,
    brightness_coefficient, rain_drops. Polylines; optional blur. uint8 I/O.

    This function adds rain to an image by drawing rain drops on the image.
    The rain drops are drawn using the OpenCV function cv2.polylines.

    Args:
        img (ImageType): The image to add rain to.
        slant (float): The slant of the rain drops.
        drop_length (int): The length of the rain drops.
        drop_width (int): The width of the rain drops.
        drop_color (tuple[int, int, int]): The color of the rain drops.
        blur_value (int): The blur value of the rain drops.
        brightness_coefficient (float): The brightness coefficient of the rain drops.
        rain_drops (np.ndarray): The rain drops to draw on the image.

    Returns:
        ImageType: The image with rain added.

    """
    if not rain_drops.size:
        return img.copy()

    img = img.copy()

    # Pre-allocate rain layer
    rain_layer = np.zeros_like(img, dtype=np.uint8)

    # Calculate end points correctly
    end_points = rain_drops + np.array([[slant, drop_length]])  # This creates correct shape

    # Stack arrays properly - both must be same shape arrays
    lines = np.stack((rain_drops, end_points), axis=1)  # Use tuple and proper axis

    cv2.polylines(
        rain_layer,
        lines.astype(np.int32),
        False,
        drop_color,
        drop_width,
        lineType=cv2.LINE_4,
    )

    if blur_value > 1:
        cv2.blur(rain_layer, (blur_value, blur_value), dst=rain_layer)

    cv2.add(img, rain_layer, dst=img)

    if brightness_coefficient != 1.0:
        cv2.multiply(img, brightness_coefficient, dst=img, dtype=cv2.CV_8U)

    return img


def get_fog_particle_radiuses(
    img_shape: tuple[int, int],
    num_particles: int,
    fog_intensity: float,
    random_generator: np.random.Generator,
) -> list[int]:
    """Generate per-particle radius list for add_fog. num_particles, fog_intensity, image size;
    random_generator samples. Returns list[int].

    Args:
        img_shape (tuple[int, int]): Image shape.
        num_particles (int): Number of fog particles.
        fog_intensity (float): Intensity of the fog effect, between 0 and 1.
        random_generator (np.random.Generator): Random generator to use.

    Returns:
        list[int]: List of radiuses for each fog particle.

    """
    height, width = img_shape[:2]
    max_fog_radius = max(2, int(min(height, width) * 0.1 * fog_intensity))
    min_radius = max(1, max_fog_radius // 2)

    return [random_generator.integers(min_radius, max_fog_radius) for _ in range(num_particles)]


@uint8_io
@clipped
@preserve_channel_dim
def add_fog(
    img: ImageType,
    fog_intensity: float,
    alpha_coef: float,
    fog_particle_positions: list[tuple[int, int]],
    fog_particle_radiuses: list[int],
) -> ImageType:
    """Add fog with circular particles and alpha blending. fog_intensity, alpha_coef, positions,
    radiuses (lists from get_fog_particle_radiuses). uint8 I/O, clipped.

    This function adds fog to an image by drawing fog particles on the image.
    The fog particles are drawn using the OpenCV function cv2.circle.

    Args:
        img (ImageType): The image to add fog to.
        fog_intensity (float): The intensity of the fog effect, between 0 and 1.
        alpha_coef (float): The coefficient for the alpha blending.
        fog_particle_positions (list[tuple[int, int]]): The positions of the fog particles.
        fog_particle_radiuses (list[int]): The radiuses of the fog particles.

    Returns:
        ImageType: The image with fog added.

    """
    result = img.copy()

    # Apply fog particles progressively like in old version
    for (x, y), radius in zip(fog_particle_positions, fog_particle_radiuses, strict=True):
        overlay = result.copy()
        cv2.circle(
            overlay,
            center=(x, y),
            radius=radius,
            color=(255, 255, 255),
            thickness=-1,
        )

        # Progressive blending
        alpha = alpha_coef * fog_intensity
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, dst=result)

    # Final subtle blur
    blur_size = max(3, int(min(img.shape[:2]) // 30))
    if blur_size % 2 == 0:
        blur_size += 1

    result = cv2.GaussianBlur(result, (blur_size, blur_size), 0)

    return clip(result, np.uint8, inplace=True)


@uint8_io
@preserve_channel_dim
@maybe_process_in_chunks
def add_sun_flare_overlay(
    img: ImageType,
    flare_center: tuple[float, float],
    src_radius: int,
    src_color: tuple[int, ...],
    circles: list[Any],
) -> ImageType:
    """Add a sun flare effect using a simple overlay. Params: src_radius, num_flare_circles;
    used as helper for physics-based sun flare.

    This function creates a basic sun flare effect by overlaying multiple semi-transparent
    circles of varying sizes and intensities on the input image. The effect simulates
    a simple lens flare caused by bright light sources.

    Args:
        img (ImageType): The input image.
        flare_center (tuple[float, float]): (x, y) coordinates of the flare center
            in pixel coordinates.
        src_radius (int): The radius of the main sun circle in pixels.
        src_color (tuple[int, ...]): The color of the sun, represented as a tuple of RGB values.
        circles (list[Any]): A list of tuples, each representing a circle that contributes
            to the flare effect. Each tuple contains:
            - alpha (float): The transparency of the circle (0.0 to 1.0).
            - center (tuple[int, int]): (x, y) coordinates of the circle center.
            - radius (int): The radius of the circle.
            - color (tuple[int, int, int]): RGB color of the circle.

    Returns:
        ImageType: The output image with the sun flare effect added.

    Note:
        - This function uses a simple alpha blending technique to overlay flare elements.
        - The main sun is created as a gradient circle, fading from the center outwards.
        - Additional flare circles are added along an imaginary line from the sun's position.
        - This method is computationally efficient but may produce less realistic results
          compared to more advanced techniques.

    The flare effect is created through the following steps:
    1. Create an overlay image and output image as copies of the input.
    2. Add smaller flare circles to the overlay.
    3. Blend the overlay with the output image using alpha compositing.
    4. Add the main sun circle with a radial gradient.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> flare_center = (50, 50)
        >>> src_radius = 20
        >>> src_color = (255, 255, 200)
        >>> circles = [
        ...     (0.1, (60, 60), 5, (255, 200, 200)),
        ...     (0.2, (70, 70), 3, (200, 255, 200))
        ... ]
        >>> flared_image = A.functional.add_sun_flare_overlay(
        ...     image, flare_center, src_radius, src_color, circles
        ... )

    References:
        - Alpha compositing: https://en.wikipedia.org/wiki/Alpha_compositing
        - Lens flare: https://en.wikipedia.org/wiki/Lens_flare

    """
    overlay = img.copy()
    output = img.copy()

    weighted_brightness = 0.0
    total_radius_length = 0.0

    for alpha, (x, y), rad3, circle_color in circles:
        weighted_brightness += alpha * rad3
        total_radius_length += rad3
        cv2.circle(overlay, (x, y), rad3, circle_color, -1)
        output = add_weighted(overlay, alpha, output, 1 - alpha)

    point = [int(x) for x in flare_center]

    overlay = output.copy()
    num_times = src_radius // 10

    # max_alpha is calculated using weighted_brightness and total_radii_length times 5
    # meaning the higher the alpha with larger area, the brighter the bright spot will be
    # for list of alphas in range [0.05, 0.2], the max_alpha should below 1
    max_alpha = weighted_brightness / total_radius_length * 5
    alpha = np.linspace(0.0, min(max_alpha, 1.0), num=num_times)

    rad = np.linspace(1, src_radius, num=num_times)

    for i in range(num_times):
        cv2.circle(overlay, point, int(rad[i]), src_color, -1)
        alp = alpha[num_times - i - 1] * alpha[num_times - i - 1] * alpha[num_times - i - 1]
        output = add_weighted(overlay, alp, output, 1 - alp)

    return output


@uint8_io
@clipped
def add_sun_flare_physics_based(
    img: ImageType,
    flare_center: tuple[int, int],
    src_radius: int,
    src_color: tuple[int, int, int],
    circles: list[Any],
) -> ImageType:
    """Physics-based sun flare: circle, spikes, ghosts, chromatic aberration, screen blend.
    flare_center, src_radius, src_color, circles.

    This function creates a complex sun flare effect by simulating various optical phenomena
    that occur in real camera lenses when capturing bright light sources. The result is a
    more realistic and physically plausible lens flare effect.

    Args:
        img (ImageType): Input image.
        flare_center (tuple[int, int]): (x, y) coordinates of the sun's center in pixels.
        src_radius (int): Radius of the main sun circle in pixels.
        src_color (tuple[int, int, int]): Color of the sun in RGB format.
        circles (list[Any]): List of tuples, each representing a flare circle with parameters:
            (alpha, center, size, color)
            - alpha (float): Transparency of the circle (0.0 to 1.0).
            - center (tuple[int, int]): (x, y) coordinates of the circle center.
            - size (float): Size factor for the circle radius.
            - color (tuple[int, int, int]): RGB color of the circle.

    Returns:
        ImageType: Image with added sun flare effect.

    Note:
        This function implements several techniques to create a more realistic flare:
        1. Separate flare layer: Allows for complex manipulations of the flare effect.
        2. Lens diffraction spikes: Simulates light diffraction in camera aperture.
        3. Radial gradient mask: Creates natural fading of the flare from the center.
        4. Gaussian blur: Softens the flare for a more natural glow effect.
        5. Chromatic aberration: Simulates color fringing often seen in real lens flares.
        6. Screen blending: Provides a more realistic blending of the flare with the image.

    The flare effect is created through the following steps:
    1. Create a separate flare layer.
    2. Add the main sun circle and diffraction spikes to the flare layer.
    3. Add additional flare circles based on the input parameters.
    4. Apply Gaussian blur to soften the flare.
    5. Create and apply a radial gradient mask for natural fading.
    6. Simulate chromatic aberration by applying different blurs to color channels.
    7. Blend the flare with the original image using screen blending mode.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [1000, 1000, 3], dtype=np.uint8)
        >>> flare_center = (500, 500)
        >>> src_radius = 50
        >>> src_color = (255, 255, 200)
        >>> circles = [
        ...     (0.1, (550, 550), 10, (255, 200, 200)),
        ...     (0.2, (600, 600), 5, (200, 255, 200))
        ... ]
        >>> flared_image = A.functional.add_sun_flare_physics_based(
        ...     image, flare_center, src_radius, src_color, circles
        ... )

    References:
        - Lens flare: https://en.wikipedia.org/wiki/Lens_flare
        - Diffraction: https://en.wikipedia.org/wiki/Diffraction
        - Chromatic aberration: https://en.wikipedia.org/wiki/Chromatic_aberration
        - Screen blending: https://en.wikipedia.org/wiki/Blend_modes#Screen

    """
    output = img.copy()
    height, width = img.shape[:2]

    # Create a separate flare layer
    flare_layer = np.zeros_like(img, dtype=np.float32)

    # Add the main sun
    cv2.circle(flare_layer, flare_center, src_radius, src_color, -1)

    # Add lens diffraction spikes
    for angle in [0, 45, 90, 135]:
        end_point = (
            int(flare_center[0] + np.cos(np.radians(angle)) * max(width, height)),
            int(flare_center[1] + np.sin(np.radians(angle)) * max(width, height)),
        )
        cv2.line(flare_layer, flare_center, end_point, src_color, 2)

    # Add flare circles
    for _, center, size, color in circles:
        cv2.circle(flare_layer, center, int(size**0.33), color, -1)

    # Apply gaussian blur to soften the flare
    flare_layer = cv2.GaussianBlur(flare_layer, (0, 0), sigmaX=15, sigmaY=15)

    # Create a radial gradient mask
    y, x = np.ogrid[:height, :width]
    mask = np.sqrt((x - flare_center[0]) ** 2 + (y - flare_center[1]) ** 2)
    mask = 1 - np.clip(mask / (max(width, height) * 0.7), 0, 1)
    mask = np.dstack([mask] * 3)

    # Apply the mask to the flare layer
    flare_layer *= mask

    # Add chromatic aberration
    channels = list(cv2.split(flare_layer))
    channels[0] = cv2.GaussianBlur(
        channels[0],
        (0, 0),
        sigmaX=3,
        sigmaY=3,
    )  # Blue channel
    channels[2] = cv2.GaussianBlur(
        channels[2],
        (0, 0),
        sigmaX=5,
        sigmaY=5,
    )  # Red channel
    flare_layer = cv2.merge(channels)

    # Blend the flare with the original image using screen blending
    return 255 - ((255 - output) * (255 - flare_layer) / 255)


@uint8_io
@preserve_channel_dim
def add_shadow(
    img: ImageType,
    vertices_list: list[np.ndarray],
    intensities: np.ndarray,
) -> ImageType:
    """Darken polygonal regions to simulate shadows. vertices_list and intensities per polygon.
    Use for outdoor or synthetic shadow augmentation. uint8 I/O.

    Args:
        img (ImageType): Input image. Multichannel images are supported.
        vertices_list (list[np.ndarray]): List of vertices for shadow polygons.
        intensities (np.ndarray): Array of shadow intensities. Range is [0, 1].

    Returns:
        ImageType: Image with shadows added.

    References:
        Automold--Road-Augmentation-Library: https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    """
    max_value = MAX_VALUES_BY_DTYPE[np.uint8]

    img_shadowed = img.copy()
    poly_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    for vertices, shadow_intensity in zip(vertices_list, intensities, strict=True):
        poly_mask[:] = 0
        cv2.fillPoly(poly_mask, [vertices], (max_value,))

        shadowed_indices = poly_mask[:, :, 0] == max_value
        darkness = 1 - shadow_intensity
        img_shadowed[shadowed_indices] = clip(
            img_shadowed[shadowed_indices] * darkness,
            np.uint8,
            inplace=True,
        )

    return img_shadowed


@uint8_io
@clipped
@preserve_channel_dim
def add_gravel(img: ImageType, gravels: list[Any]) -> ImageType:
    """Add gravel: write HLS saturation in rectangular regions. gravels: list of
    (min_y, max_y, min_x, max_x, sat). RGB only; uint8 I/O.

    This function adds gravel to an image by drawing gravel particles on the image.
    The gravel particles are drawn using the OpenCV function cv2.circle.

    Args:
        img (ImageType): The image to add gravel to.
        gravels (list[Any]): The gravel particles to draw on the image.

    Returns:
        ImageType: The image with gravel added.

    """
    non_rgb_error(img)
    image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    for gravel in gravels:
        min_y, max_y, min_x, max_x, sat = gravel
        image_hls[min_y:max_y, min_x:max_x, 1] = sat

    return cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)


def invert(img: ImageType) -> ImageType:
    """Produce the negative image: each pixel becomes max_val - pixel. uint8/float32, any
    channels. Use for inversion augmentation or visualization.

    This function inverts the colors of an image by subtracting each pixel value from the maximum possible value.
    The result is a negative of the original image.

    Args:
        img (ImageType): The image to invert.

    Returns:
        ImageType: The inverted image.

    """
    # Supports all the valid dtypes
    # clips the img to avoid unexpected behaviour.
    return MAX_VALUES_BY_DTYPE[img.dtype] - img


def channel_shuffle(img: ImageType, channels_shuffled: list[int]) -> ImageType:
    """Shuffle image channels via cv2.mixChannels. channels_shuffled gives new order; supports
    (H, W, C) or batch (N, H, W, C).

    This function shuffles the channels of an image by using the cv2.mixChannels function.
    The channels are shuffled according to the channels_shuffled array.

    Args:
        img (ImageType): The image to shuffle.
        channels_shuffled (list[int]): The array of channels to shuffle.

    Returns:
        ImageType: The shuffled image.

    """
    img = np.ascontiguousarray(img)
    output = np.empty(img.shape, dtype=img.dtype)
    from_to = []
    for i, j in enumerate(channels_shuffled):
        from_to.extend([j, i])  # Use [src, dst]
    cv2.mixChannels([img], [output], from_to)
    return output


def volume_channel_shuffle(volume: np.ndarray, channels_shuffled: Sequence[int]) -> np.ndarray:
    """Shuffle channels of volume (D, H, W, C) or (D, H, W). Same as channel_shuffle along last
    axis. channels_shuffled is new order. Used for 3D volume augmentation.

    Args:
        volume (np.ndarray): Input volume.
        channels_shuffled (Sequence[int]): New channel order.

    Returns:
        np.ndarray: Volume with channels shuffled.

    """
    return volume.copy()[..., channels_shuffled] if volume.ndim == 4 else volume


def volumes_channel_shuffle(volumes: np.ndarray, channels_shuffled: Sequence[int]) -> np.ndarray:
    """Shuffle channels of a batch of volumes (B, D, H, W, C) or (B, D, H, W).
    Per-volume shuffle; used for 3D batch augmentation.

    Args:
        volumes (np.ndarray): Input batch of volumes.
        channels_shuffled (Sequence[int]): New channel order.

    Returns:
        np.ndarray: Batch of volumes with channels shuffled.

    """
    return volumes.copy()[..., channels_shuffled] if volumes.ndim == 5 else volumes


def gamma_transform(img: ImageType, gamma: float) -> ImageType:
    """Apply gamma transformation: pixel^gamma to brighten or darken. gamma > 1 brightens;
    gamma < 1 darkens. Supports uint8 and float32.

    This function applies gamma transformation to an image by raising each pixel value to the power of gamma.
    The result is a non-linear transformation that can enhance or reduce the contrast of the image.

    Args:
        img (ImageType): The image to apply gamma transformation to.
        gamma (float): The gamma value to apply.

    Returns:
        ImageType: The gamma transformed image.

    """
    if img.dtype == np.uint8:
        table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
        return sz_lut(img, table.astype(np.uint8), inplace=False)

    return np.power(img, gamma)


@float32_io
@clipped
def iso_noise(
    image: np.ndarray,
    color_shift: float,
    intensity: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Apply Poisson noise to simulate camera sensor noise. color_shift, intensity;
    approximates shot noise. float32 I/O, clipped.

    Args:
        image (np.ndarray): Input image. Currently, only RGB images are supported.
        color_shift (float): The amount of color shift to apply.
        intensity (float): Multiplication factor for noise values. Values of ~0.5 produce a noticeable,
                           yet acceptable level of noise.
        random_generator (np.random.Generator): If specified, this will be random generator used
            for noise generation.

    Returns:
        np.ndarray: The noised image.

    Image types:
        uint8, float32

    Number of channels:
        3

    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    _, stddev = cv2.meanStdDev(hls)

    luminance_noise = random_generator.poisson(
        stddev[1] * intensity,
        size=hls.shape[:2],
    )
    color_noise = random_generator.normal(
        0,
        color_shift * intensity,
        size=hls.shape[:2],
    )

    hls[..., 0] += color_noise
    hls[..., 1] = add_array(
        hls[..., 1],
        luminance_noise * intensity * (1.0 - hls[..., 1]),
    )

    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


@float32_io
@clipped
def iso_noise_images(
    images: np.ndarray,
    color_shift: float,
    intensity: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Apply ISO noise to an image batch with vectorized noise. One noise field broadcast to all
    images; matches apply() with same seed. Use for batch augmentation.

    Noise is generated once and broadcast across all images, matching the behavior
    of calling apply() per image with the same random seed.

    Args:
        images (np.ndarray): (N, H, W, 3) RGB images.
        color_shift (float): Amount of color hue shift.
        intensity (float): Noise intensity multiplier.
        random_generator (np.random.Generator): Numpy RNG seeded for reproducibility.

    Returns:
        np.ndarray: (N, H, W, 3) noised images.

    Image types:
        uint8, float32

    Number of channels:
        3

    """
    non_rgb_error(images[0])
    h, w = images.shape[1:3]

    hls_batch = np.empty_like(images)

    for i, image in enumerate(images):
        cv2.cvtColor(image, cv2.COLOR_RGB2HLS, dst=hls_batch[i])

    # Use first image's stddev — matches apply() which creates default_rng(same_seed) per image
    _, stddev = cv2.meanStdDev(hls_batch[0])

    # Generate noise ONCE in same order as iso_noise — mirrors apply() with same seed
    luminance_noise = random_generator.poisson(float(stddev[1, 0]) * intensity, size=(h, w)).astype(np.float32)
    color_noise = random_generator.normal(0, color_shift * intensity, size=(h, w)).astype(np.float32)

    hls_batch[:, :, :, 0] += color_noise
    del color_noise

    # Equivalent to: L += noise * intensity * (1 - L)
    luminance_noise *= intensity
    hls_batch[:, :, :, 1] *= 1.0 - luminance_noise
    hls_batch[:, :, :, 1] += luminance_noise
    del luminance_noise

    for i in range(len(hls_batch)):
        cv2.cvtColor(hls_batch[i], cv2.COLOR_HLS2RGB, dst=hls_batch[i])

    return hls_batch


def to_gray_weighted_average(img: ImageType) -> ImageType:
    """Convert RGB to grayscale with weighted average (0.299*R+0.587*G+0.114*B). Single or batch.
    BT.601. Matches OpenCV perceptual luminance.

    This function uses OpenCV's cvtColor function with COLOR_RGB2GRAY conversion,
    which applies the following formula:
    Y = 0.299*R + 0.587*G + 0.114*B

    The function efficiently handles batches and volumes by reshaping them into
    a tall 2D image for processing, then restoring the original shape structure.

    Args:
        img (ImageType): Input RGB image(s) as a numpy array. Supported shapes:
            - Single image: (H, W, 3)
            - Batch of images: (N, H, W, 3)
            - Volume: (D, H, W, 3)
            - Batch of volumes: (N, D, H, W, 3)

    Returns:
        ImageType: Grayscale image as a 2D numpy array.

    Image types:
        uint8, float32

    Number of channels:
        3

    """
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.ndim == 4:
        im, original_shape = reshape_xhwc_channel(img)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        new_shape = (*original_shape[:-1], 1)

        return restore_xhwc_channel(im, new_shape)

    if img.ndim == 5:
        img, original_shape = reshape_ndhwc_channel(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        new_shape = (*original_shape[:-1], 1)

        return restore_ndhwc_channel(img, new_shape)

    raise ValueError(f"Unsupported number of dimensions: {img.ndim}")


@uint8_io
def to_gray_from_lab(img: ImageType) -> ImageType:
    """Convert RGB to grayscale using the LAB L channel; perceived brightness matches human vision
    better than a simple average. Single image or batch. uint8 I/O.

    This function converts RGB images to grayscale by first converting to LAB color space
    and then extracting the L (lightness) channel. It uses albucore's reshape utilities
    to efficiently handle batches/volumes by processing them as a single tall image.

    Implementation Details:
        The function uses albucore's reshape_for_channel and restore_from_channel functions:
        - reshape_for_channel: Flattens batches/volumes to 2D format for OpenCV processing
        - restore_from_channel: Restores the original shape after processing

        This enables processing all images in a single OpenCV call

    Args:
        img (ImageType): Input RGB image(s) as a numpy array. Must have 3 channels in the last dimension.
            Supported shapes:
            - Single image: (H, W, 3)
            - Batch of images: (N, H, W, 3)
            - Volume: (D, H, W, 3)
            - Batch of volumes: (N, D, H, W, 3)

            Supported dtypes:
            - np.uint8: Values in range [0, 255]
            - np.float32: Values in range [0, 1]

    Returns:
        ImageType:
            - Single image: (H, W)
            - Batch of images: (N, H, W)
            - Volume: (D, H, W)
            - Batch of volumes: (N, D, H, W)

        The output dtype matches the input dtype. For float inputs, the L channel
        is normalized to [0, 1] by dividing by 100.

    Raises:
        ValueError: If the last dimension is not 3 (RGB channels)

    Examples:
        >>> # Single image
        >>> img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> gray = to_gray_from_lab(img)
        >>> assert gray.shape == (100, 100)

        >>> # Batch of images - efficiently processed without loops
        >>> batch = np.random.randint(0, 256, (10, 100, 100, 3), dtype=np.uint8)
        >>> gray_batch = to_gray_from_lab(batch)
        >>> assert gray_batch.shape == (10, 100, 100)

        >>> # Volume (e.g., video frames or 3D medical data)
        >>> volume = np.random.randint(0, 256, (16, 100, 100, 3), dtype=np.uint8)
        >>> gray_volume = to_gray_from_lab(volume)
        >>> assert gray_volume.shape == (16, 100, 100)

        >>> # Float32 input
        >>> img_float = img.astype(np.float32) / 255.0
        >>> gray_float = to_gray_from_lab(img_float)
        >>> assert 0 <= gray_float.min() <= gray_float.max() <= 1.0

    Note:
        The LAB color space provides perceptually uniform grayscale conversion,
        where the L (lightness) channel represents human perception of brightness
        better than simple RGB averaging or other methods.

    """
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0]
    if img.ndim == 4:
        im, original_shape = reshape_xhwc_channel(img)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)[..., 0]

        new_shape = (*original_shape[:-1], 1)

        return restore_xhwc_channel(im, new_shape)

    if img.ndim == 5:
        img, original_shape = reshape_ndhwc_channel(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0]

        new_shape = (*original_shape[:-1], 1)

        return restore_ndhwc_channel(img, new_shape)

    raise ValueError(f"Unsupported number of dimensions: {img.ndim}")


@clipped
def to_gray_desaturation(img: ImageType) -> ImageType:
    """Convert to grayscale with desaturation (max + min) / 2 per pixel. Any channels; single or batch.
    uint8 and float32. uint8 and float32.

    Args:
        img (ImageType): Input image as a numpy array.

    Returns:
        ImageType: Grayscale image as a 2D numpy array.

    Image types:
        uint8, float32

    Number of channels:
        any

    """
    if img.dtype == np.uint8:
        ch_max = np.max(img, axis=-1).astype(np.uint16)
        ch_min = np.min(img, axis=-1).astype(np.uint16)
        return ((ch_max + ch_min) >> 1).astype(np.uint8)
    float_image = img.astype(np.float32)
    return (np.max(float_image, axis=-1) + np.min(float_image, axis=-1)) * 0.5


def to_gray_average(img: ImageType) -> ImageType:
    """Convert to grayscale using per-pixel mean across channels. Simple average; single image or
    batch. Any channel count. uint8 and float32.

    This function computes the arithmetic mean across all channels for each pixel,
    resulting in a grayscale representation of the image.

    Key aspects of this method:
    1. It treats all channels equally, regardless of their perceptual importance.
    2. Works with any number of channels, making it versatile for various image types.
    3. Simple and fast to compute, but may not accurately represent perceived brightness.
    4. For RGB images, the formula is: Gray = (R + G + B) / 3

    Note: This method may produce different results compared to weighted methods
    (like RGB weighted average) which account for human perception of color brightness.
    It may also produce unexpected results for images with alpha channels or
    non-color data in additional channels.

    Args:
        img (ImageType): Input image as a numpy array. Can be any number of channels.

    Returns:
        ImageType: Grayscale image as a 2D numpy array. The output data type
                    matches the input data type.

    Image types:
        uint8, float32

    Number of channels:
        any

    """
    return mean(img, axis=-1).astype(img.dtype)


def to_gray_max(img: ImageType) -> ImageType:
    """Convert to grayscale using max across channels per pixel. Equivalent to HSV V for RGB. Any
    channel count. Single or batch. uint8 and float32.

    This function takes the maximum value across all channels for each pixel,
    resulting in a grayscale image that preserves the brightest parts of the original image.

    Key aspects of this method:
    1. Works with any number of channels, making it versatile for various image types.
    2. For 3-channel (e.g., RGB) images, this method is equivalent to extracting the V (Value)
       channel from the HSV color space.
    3. Preserves the brightest parts of the image but may lose some color contrast information.
    4. Simple and fast to compute.

    Note:
    - This method tends to produce brighter grayscale images compared to other conversion methods,
      as it always selects the highest intensity value from the channels.
    - For RGB images, it may not accurately represent perceived brightness as it doesn't
      account for human color perception.

    Args:
        img (ImageType): Input image as a numpy array. Can be any number of channels.

    Returns:
        ImageType: Grayscale image as a 2D numpy array. The output data type
                    matches the input data type.

    Image types:
        uint8, float32

    Number of channels:
        any

    """
    return np.max(img, axis=-1)


@clipped
def to_gray_pca(img: ImageType) -> ImageType:
    """Reduce to one channel via PCA; captures max variance in color. Single or batch; uint8 or
    float32. Clipped. Use when simple averaging loses info.

    This function applies PCA to reduce a multi-channel image to a single channel,
    effectively creating a grayscale representation that captures the maximum variance
    in the color data.

    Args:
        img (ImageType): Input image as a numpy array. Can be:
            - Single multi-channel image: (H, W, C)
            - Batch of multi-channel images: (N, H, W, C)
            - Single multi-channel volume: (D, H, W, C)
            - Batch of multi-channel volumes: (N, D, H, W, C)

    Returns:
        ImageType: Grayscale image with the same spatial dimensions as input.
                    If input is uint8, output is uint8 in range [0, 255].
                    If input is float32, output is float32 in range [0, 1].

    Note:
        This method can potentially preserve more information from the original image
        compared to standard weighted average methods, as it accounts for the
        correlations between color channels.

    Image types:
        uint8, float32

    Number of channels:
        any

    """
    dtype = img.dtype
    # Reshape the image to a 2D array of pixels
    pixels = img.reshape(-1, img.shape[-1])

    # Perform PCA
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(pixels)

    # Reshape back to image dimensions and scale to 0-255
    grayscale = pca_result.reshape(img.shape[:-1])
    grayscale = normalize_per_image(grayscale, "min_max")

    return from_float(grayscale, target_dtype=dtype) if dtype == np.uint8 else grayscale


def to_gray(
    img: ImageType,
    num_output_channels: int,
    method: Literal[
        "weighted_average",
        "from_lab",
        "desaturation",
        "average",
        "max",
        "pca",
    ],
) -> ImageType:
    """Convert image to grayscale using a specified method. Choices: weighted_average,
    from_lab, desaturation, average, max, pca.

    This function converts an image to grayscale using a specified method.
    The method can be one of the following:
    - "weighted_average": Use the weighted average method.
    - "from_lab": Use the L channel from the LAB color space.
    - "desaturation": Use the desaturation method.
    - "average": Use the average method.
    - "max": Use the maximum channel value method.
    - "pca": Use the Principal Component Analysis method.

    Args:
        img (ImageType): Input image as a numpy array.
        num_output_channels (int): The number of channels in the output image.
        method (Literal['weighted_average', 'from_lab', 'desaturation', 'average', 'max', 'pca']):
            The method to use for grayscale conversion.

    Returns:
        ImageType: Grayscale image as a 2D numpy array.

    """
    if method == "weighted_average":
        result = to_gray_weighted_average(img)
    elif method == "from_lab":
        result = to_gray_from_lab(img)
    elif method == "desaturation":
        result = to_gray_desaturation(img)
    elif method == "average":
        result = to_gray_average(img)
    elif method == "max":
        result = to_gray_max(img)
    elif method == "pca":
        result = to_gray_pca(img)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return grayscale_to_multichannel(result, num_output_channels)


def grayscale_to_multichannel(
    grayscale_image: np.ndarray,
    num_output_channels: int = 3,
) -> np.ndarray:
    """Convert grayscale to multi-channel by repeating. num_output_channels (default 3).
    For blending gray with color in saturation/hue adjustments.

    This function takes a 2D grayscale image or a 3D image with a single channel
    and converts it to a multi-channel image by repeating the grayscale data
    across the specified number of channels.

    Args:
        grayscale_image (np.ndarray): Input grayscale image. Can be 2D (height, width)
                                      or 3D (height, width, 1).
        num_output_channels (int, optional): Number of channels in the output image. Defaults to 3.

    Returns:
        np.ndarray: Multi-channel image with shape (height, width, num_channels)

    """
    # If output should be single channel, add channel dimension if needed
    if num_output_channels == 1:
        return grayscale_image

    if num_output_channels == 3 and grayscale_image.ndim == 2:
        return cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)

    squeezed = np.squeeze(grayscale_image)
    # For multi-channel output, use tile for better performance
    return np.tile(squeezed[..., np.newaxis], (1,) * squeezed.ndim + (num_output_channels,))


def _build_colorize_lut(
    black: tuple[int, int, int],
    white: tuple[int, int, int],
    mid: tuple[int, int, int] | None,
    mid_value: int,
) -> np.ndarray:
    """Build a (256, 3) float32 LUT mapping uint8 intensity to an RGB ramp via linear
    interpolation between (black, [mid], white) anchors. Used by `colorize`.

    When `mid` is None the ramp has two anchors (0 -> black, 255 -> white). When `mid` is
    given the ramp is piecewise-linear with three anchors (0 -> black, mid_value -> mid,
    255 -> white).
    """
    if mid is None:
        anchor_intensity = np.array([0.0, 255.0], dtype=np.float32)
        anchor_color = np.array([black, white], dtype=np.float32)
    else:
        anchor_intensity = np.array([0.0, float(mid_value), 255.0], dtype=np.float32)
        anchor_color = np.array([black, mid, white], dtype=np.float32)

    intensities = np.arange(256, dtype=np.float32)
    lut = np.empty((256, 3), dtype=np.float32)
    for color_channel in range(3):
        lut[:, color_channel] = np.interp(intensities, anchor_intensity, anchor_color[:, color_channel])
    return lut


def colorize(
    img: ImageType,
    black: tuple[int, int, int],
    white: tuple[int, int, int],
    mid: tuple[int, int, int] | None,
    mid_value: int,
) -> ImageType:
    """Map grayscale intensity to a 2- or 3-color RGB gradient (Pillow `ImageOps.colorize`
    style) using a (256, 3) LUT for uint8 and `np.interp` for float32.

    Anchor colors are specified as RGB tuples in 0-255 regardless of input dtype; for float32
    inputs the anchors are scaled into [0, 1] before interpolation. Output always has 3 channels.

    Args:
        img (ImageType): Single-channel image, shape `(H, W, 1)` or batch `(N, H, W, 1)`.
        black (tuple[int, int, int]): RGB color for intensity 0.
        white (tuple[int, int, int]): RGB color for intensity 255 (or 1.0 for float32).
        mid (tuple[int, int, int] | None): Optional RGB color for the midpoint. `None` gives a
            2-color ramp.
        mid_value (int): Intensity (0-255) that maps to `mid`. Ignored when `mid is None`.

    Returns:
        ImageType: Image with the trailing channel dimension expanded to 3.

    """
    gray = img[..., 0]

    if img.dtype == np.uint8:
        lut_float = _build_colorize_lut(black, white, mid, mid_value)
        # Floor-quantize to match `PIL.ImageOps.colorize` (integer floor division) bit-for-bit.
        lut_uint8 = np.clip(np.floor(lut_float), 0, 255).astype(np.uint8)
        # cv2.LUT on a 3-channel uint8 image with a (1, 256, 3) LUT is the fastest path
        # (~30x vs numpy fancy indexing). Replicate the gray channel into 3 first; cv2 then
        # applies a different per-channel LUT in C.
        lut_cv = lut_uint8.reshape(1, 256, 3)
        if gray.ndim == 2:
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return cv2.LUT(gray_3ch, lut_cv)
        # Batched / volumetric: gray has a leading axis (N or D). Loop in Python over the
        # leading axis but keep the per-frame work in cv2 — far cheaper than 4D fancy indexing.
        out = np.empty((*gray.shape, 3), dtype=np.uint8)
        flat_in = gray.reshape(-1, *gray.shape[-2:])
        flat_out = out.reshape(-1, *gray.shape[-2:], 3)
        for frame_index in range(flat_in.shape[0]):
            flat_out[frame_index] = cv2.LUT(cv2.cvtColor(flat_in[frame_index], cv2.COLOR_GRAY2BGR), lut_cv)
        return out

    if img.dtype != np.float32:
        raise ValueError(f"colorize: unsupported dtype {img.dtype}; expected uint8 or float32")

    anchor_intensity_float = (
        np.array([0.0, 1.0], dtype=np.float32)
        if mid is None
        else np.array([0.0, mid_value / 255.0, 1.0], dtype=np.float32)
    )
    anchor_color_float = (
        np.array([black, white], dtype=np.float32) / 255.0
        if mid is None
        else np.array([black, mid, white], dtype=np.float32) / 255.0
    )
    out = np.empty((*gray.shape, 3), dtype=np.float32)
    gray_clipped = np.clip(gray, 0.0, 1.0)
    for color_channel in range(3):
        out[..., color_channel] = np.interp(gray_clipped, anchor_intensity_float, anchor_color_float[:, color_channel])
    return out


@preserve_channel_dim
@uint8_io
def downscale(
    img: ImageType,
    scale: float,
    down_interpolation: int,
    up_interpolation: int,
) -> ImageType:
    """Simulate resolution loss: downscale then upscale. down/up_interpolation control quality.
    Use for low-res or compression-style augmentation. uint8 I/O.

    This function downscales and upscales an image using the specified interpolation methods.
    The downscaling and upscaling are performed using albucore.resize.

    Args:
        img (ImageType): Input image as a numpy array.
        scale (float): The scale factor for the downscaling and upscaling.
        down_interpolation (int): The interpolation method for the downscaling.
        up_interpolation (int): The interpolation method for the upscaling.

    Returns:
        ImageType: The downscaled and upscaled image.

    """
    height, width = img.shape[:2]

    downscaled = fgeometric.resize(img, (int(height * scale), int(width * scale)), down_interpolation)
    return fgeometric.resize(downscaled, (height, width), up_interpolation)


def noop(input_obj: Any, **params: Any) -> Any:
    """No-op: return input unchanged. Used to satisfy type checker and pipeline
    placeholders; accepts any input and optional kwargs.

    This function is a no-op and returns the input object unchanged.
    It is used to satisfy the type checker requirements for the `noop` function.

    Args:
        input_obj (Any): The input object to return unchanged.
        **params (Any): Additional keyword arguments.

    Returns:
        Any: The input object unchanged.

    """
    return input_obj


@float32_io
@clipped
@preserve_channel_dim
def fancy_pca(img: ImageType, alpha_vector: np.ndarray) -> ImageType:
    """Fancy PCA color augmentation: add noise along principal components. alpha_vector; any channel
    count. float32 I/O, clipped.

    Args:
        img (ImageType): Input image
        alpha_vector (np.ndarray): Vector of scale factors for each principal component.
                                   Should have the same length as the number of channels in the image.

    Returns:
        ImageType: Augmented image of the same shape, type, and range as the input.

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - This function generalizes the Fancy PCA augmentation to work with any number of channels.
        - It preserves the original range of the image ([0, 255] for uint8, [0, 1] for float32).
        - For single-channel images, the augmentation is applied as a simple scaling of pixel intensity variation.
        - For multi-channel images, PCA is performed on the entire image, treating each pixel
          as a point in N-dimensional space (where N is the number of channels).
        - The augmentation preserves the correlation between channels while adding controlled noise.
        - Computation time may increase significantly for images with a large number of channels.

    References:
        ImageNet classification with deep convolutional neural networks: Krizhevsky, A., Sutskever, I.,
            & Hinton, G. E. (2012): In Advances in neural information processing systems (pp. 1097-1105).

    """
    orig_shape = img.shape
    num_channels = get_num_channels(img)

    # Reshape image to 2D array of pixels
    img_reshaped = img.reshape(-1, num_channels)

    # Center the pixel values
    img_mean = mean(img_reshaped, axis=0, dtype=np.float32)
    img_centered = img_reshaped - img_mean

    if num_channels == 1:
        # For grayscale images, apply a simple scaling
        std_dev = std(img_centered, eps=0)
        noise = alpha_vector[0] * std_dev * img_centered
    else:
        # Compute covariance matrix
        img_cov = np.cov(img_centered, rowvar=False)

        # Compute eigenvectors & eigenvalues of the covariance matrix
        eig_vals, eig_vecs = np.linalg.eigh(img_cov)

        # Sort eigenvectors by eigenvalues in descending order
        sort_perm = eig_vals[::-1].argsort()
        eig_vals = eig_vals[sort_perm]
        eig_vecs = eig_vecs[:, sort_perm]

        # Create noise vector
        noise = np.dot(
            np.dot(eig_vecs, np.diag(alpha_vector * eig_vals)),
            img_centered.T,
        ).T

    # Add noise to the image
    img_pca = img_reshaped + noise

    # Reshape back to original shape
    img_pca = img_pca.reshape(orig_shape)

    # Clip values to [0, 1] range
    return np.clip(img_pca, 0, 1, out=img_pca)


@preserve_channel_dim
def adjust_brightness_torchvision(img: ImageType, factor: np.ndarray) -> ImageType:
    """Adjust brightness by multiplying pixels by factor. Torchvision-compatible; uint8 and float32.
    factor scalar or per-channel array.

    This function adjusts the brightness of an image by multiplying each pixel value by a factor.
    The brightness is adjusted by multiplying the image by the factor.

    Args:
        img (ImageType): Input image as a numpy array.
        factor (np.ndarray): The factor to adjust the brightness by.

    Returns:
        ImageType: The adjusted image.

    """
    if factor == 0:
        return np.zeros_like(img)
    if factor == 1:
        return img

    return multiply(img, factor, inplace=False)


@preserve_channel_dim
def adjust_contrast_torchvision(img: ImageType, factor: float) -> ImageType:
    """Adjust contrast by multiplying by factor (relative to grayscale mean). Torchvision-compatible;
    uint8 and float32. factor 0 yields flat gray.

    This function adjusts the contrast of an image by multiplying each pixel value by a factor.
    The contrast is adjusted by multiplying the image by the factor.

    Args:
        img (ImageType): Input image as a numpy array.
        factor (float): The factor to adjust the contrast by.

    Returns:
        ImageType: The adjusted image.

    """
    if factor == 1:
        return img

    img_mean = mean(img) if is_grayscale_image(img) else mean(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

    if factor == 0:
        if img.dtype != np.float32:
            img_mean = int(img_mean + 0.5)
        return np.full_like(img, img_mean, dtype=img.dtype)

    return multiply_add(img, factor, img_mean * (1 - factor), inplace=False)


@clipped
@preserve_channel_dim
def adjust_saturation_torchvision(
    img: ImageType,
    factor: float,
    gamma: float = 0,
) -> ImageType:
    """Adjust saturation by blending with grayscale. Factor in [0, inf]; uses
    to_gray (weighted_average for RGB). Torchvision-compatible.

    Uses to_gray for conversion: weighted_average for RGB (matches OpenCV), average for
    arbitrary channels. Works on batches (4D) and volumes (5D).

    Args:
        img (ImageType): Input image as a numpy array.
        factor (float): The factor to adjust the saturation by.
        gamma (float): Unused, kept for API compatibility.

    Returns:
        ImageType: The adjusted image.

    """
    if factor == 1 or is_grayscale_image(img):
        return img

    gray = to_gray_weighted_average(img) if is_rgb_image(img) else to_gray_average(img)
    gray_expanded = grayscale_to_multichannel(gray, img.shape[-1])

    return gray_expanded if factor == 0 else add_weighted(img, factor, gray_expanded, 1 - factor)


def _adjust_hue_torchvision_uint8(img: ImageUInt8, factor: float) -> ImageUInt8:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lut = np.arange(0, 256, dtype=np.int16)
    lut = np.mod(lut + 180 * factor, 180).astype(np.uint8)
    img[..., 0] = sz_lut(img[..., 0], lut, inplace=False)

    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def adjust_hue_torchvision(img: ImageType, factor: float) -> ImageType:
    """Adjust hue by shifting in HSV. factor in [-0.5, 0.5]. Torchvision-compatible; RGB only.
    LUT for uint8; in-place mod for float32.

    This function adjusts the hue of an image by adding a factor to the hue value.

    Args:
        img (ImageType): Input image.
        factor (float): The factor to adjust the hue by.

    Returns:
        ImageType: The adjusted image.

    """
    if is_grayscale_image(img) or factor == 0:
        return img

    if img.dtype == np.uint8:
        return _adjust_hue_torchvision_uint8(img, factor)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[..., 0] = np.mod(img[..., 0] + factor * 360, 360)
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def apply_brightness_contrast_torchvision(
    img: ImageType,
    brightness_factor: float,
    contrast_factor: float,
    brightness_first: bool,
) -> ImageType:
    """Fused brightness and contrast: single LUT (uint8) or two passes (float32). clip(a*x+b);
    torchvision-compatible; order via brightness_first.

    Both operations are `clip(a*x + b)`. The image grayscale mean is computed once and propagated
    analytically through the pipeline: if brightness comes first, `mean_at_contrast = mean * brightness_factor`
    (clipped to valid range). This avoids re-reading the image after brightness is applied.

    For uint8 images the composition is pre-computed over all 256 input values into a single
    256-entry LUT applied in one `cv2.LUT` call. For float32, two sequential clipped passes are used.

    Args:
        img (ImageType): Input image (uint8 or float32).
        brightness_factor (float): Brightness multiplicative factor.
        contrast_factor (float): Contrast multiplicative factor.
        brightness_first (bool): Whether brightness is applied before contrast.

    Returns:
        ImageType: Adjusted image with the same dtype as input.

    """
    # Compute original grayscale mean once, normalised to [0, 1].
    gray_for_mean = (
        img
        if is_grayscale_image(img)
        else (to_gray_weighted_average(img) if is_rgb_image(img) else to_gray_average(img))
    )
    img_mean = float(mean(gray_for_mean))
    if img.dtype == np.uint8:
        img_mean /= 255.0

    # Propagate mean analytically: brightness scales the mean, clipped to [0, 1].
    mean_at_contrast = float(np.clip(img_mean * brightness_factor, 0.0, 1.0)) if brightness_first else img_mean

    if img.dtype == np.uint8:
        lut = np.arange(256, dtype=np.float32)
        if brightness_first:
            lut = np.clip(lut * brightness_factor, 0.0, 255.0)
            lut = np.clip(lut * contrast_factor + mean_at_contrast * 255.0 * (1.0 - contrast_factor), 0.0, 255.0)
        else:
            lut = np.clip(lut * contrast_factor + mean_at_contrast * 255.0 * (1.0 - contrast_factor), 0.0, 255.0)
            lut = np.clip(lut * brightness_factor, 0.0, 255.0)
        return sz_lut(img, lut.astype(np.uint8), inplace=False)

    # float32: two clipped passes, single buffer, in-place ops
    offset = mean_at_contrast * (1.0 - contrast_factor)
    out = np.empty_like(img)
    if brightness_first:
        np.multiply(img, brightness_factor, out=out)
        np.clip(out, 0.0, 1.0, out=out)
        np.multiply(out, contrast_factor, out=out)
        np.add(out, offset, out=out)
        np.clip(out, 0.0, 1.0, out=out)
    else:
        np.multiply(img, contrast_factor, out=out)
        np.add(out, offset, out=out)
        np.clip(out, 0.0, 1.0, out=out)
        np.multiply(out, brightness_factor, out=out)
        np.clip(out, 0.0, 1.0, out=out)
    return out


@uint8_io
@preserve_channel_dim
def superpixels(
    image: np.ndarray,
    n_segments: int,
    replace_samples: Sequence[bool],
    max_size: int | None,
    interpolation: int,
) -> np.ndarray:
    """Apply superpixels using SLIC: replace pixels with segment mean. n_segments, replace_samples,
    max_size, interpolation. uint8 I/O.

    This function applies superpixels to an image using the SLIC algorithm.
    The superpixels are applied by replacing the pixels in the image with the mean intensity of the superpixel.

    Args:
        image (np.ndarray): Input image as a numpy array.
        n_segments (int): The number of segments to use for the superpixels.
        replace_samples (Sequence[bool]): The samples to replace.
        max_size (int | None): The maximum size of the superpixels.
        interpolation (int): The interpolation method to use.

    Returns:
        np.ndarray: The superpixels applied to the image.

    """
    if not np.any(replace_samples):
        return image

    orig_shape = image.shape
    if max_size is not None:
        size = max(image.shape[:2])
        if size > max_size:
            scale = max_size / size
            height, width = image.shape[:2]
            new_height, new_width = int(height * scale), int(width * scale)
            image = fgeometric.resize(image, (new_height, new_width), interpolation)

    segments = slic(
        image,
        n_segments=n_segments,
        compactness=10,
    )

    min_value = 0
    max_value = MAX_VALUES_BY_DTYPE[image.dtype]
    image = np.copy(image)

    num_channels = get_num_channels(image)

    for c in range(num_channels):
        image_sp_c = image[..., c]
        # Get unique segment labels (skip 0 if it exists as it's typically background)
        unique_labels = np.unique(segments)
        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]

        # Calculate mean intensity for each segment
        for idx, label in enumerate(unique_labels):
            # with mod here, because slic can sometimes create more superpixel than requested.
            # replace_samples then does not have enough values, so we just start over with the first one again.
            if replace_samples[idx % len(replace_samples)]:
                mask = segments == label
                mean_intensity = mean(image_sp_c[mask])

                if image_sp_c.dtype.kind in ["i", "u", "b"]:
                    # After rounding the value can end up slightly outside of the value_range. Hence, we need to clip.
                    # We do clip via min(max(...)) instead of np.clip because
                    # the latter one does not seem to keep dtypes for dtypes with large itemsizes (e.g. uint64).
                    value: int | float
                    value = int(np.round(mean_intensity))
                    value = min(max(value, min_value), max_value)
                else:
                    value = mean_intensity

                image_sp_c[mask] = value

    return fgeometric.resize(image, orig_shape[:2], interpolation) if orig_shape != image.shape else image


def unsharp_mask_images(
    images: np.ndarray,
    ksize: int,
    sigma: float,
    alpha: float,
    threshold: int,
) -> np.ndarray:
    """Apply unsharp mask to batch (N, H, W, C). Sharpen via blur-subtract-add. ksize, sigma, alpha,
    threshold. Pre-allocated output for batch path.

    Processes a batch of images (N, H, W, C) by applying the unsharp mask to each image
    and writing directly into a pre-allocated output array.

    Args:
        images (np.ndarray): Batch of images with shape (N, H, W, C) or (N, H, W).
        ksize (int): The kernel size for Gaussian blur.
        sigma (float): The sigma value for Gaussian blur.
        alpha (float): The alpha value for the unsharp mask.
        threshold (int): The threshold value for the unsharp mask.

    Returns:
        np.ndarray: Batch of unsharp masked images with same shape and dtype as input.

    Note: we intentionally avoid @float32_io/@clipped decorators here.
        Those decorators convert the entire batch at once, which is ~2x slower for uint8
        than per-image conversion due to poor cache locality on large 4D arrays.

    """
    input_dtype = images.dtype
    need_conversion = input_dtype != np.float32
    num_channels = images.shape[-1] if images.ndim > 3 else 0
    ksize_tuple = (ksize, ksize)
    threshold_f = threshold / 255.0

    result = np.empty_like(images)

    # Single-image float32 working buffer for uint8 path
    buf: np.ndarray = np.empty(images.shape[1:], dtype=np.float32) if need_conversion else result

    for i in range(images.shape[0]):
        if need_conversion:
            image = to_float(images[i])
            dst = buf if num_channels != 1 else buf[:, :, 0]
        else:
            image = images[i]
            dst = result[i] if num_channels != 1 else result[i, :, :, 0]

        cv2.subtract(image, cv2.GaussianBlur(image, ksize_tuple, sigmaX=sigma), dst=dst)

        mask = np.abs(dst)
        cv2.threshold(mask, threshold_f, 1.0, cv2.THRESH_BINARY, dst=mask)

        cv2.scaleAdd(dst, alpha, image, dst=dst)
        np.clip(dst, 0, 1, out=dst)

        cv2.GaussianBlur(mask, ksize_tuple, sigmaX=sigma, dst=mask)

        # Blend: image + mask * (sharp - image), all in-place
        cv2.subtract(dst, image, dst=dst)
        cv2.multiply(dst, mask, dst=dst)
        cv2.add(dst, image, dst=dst)

        if need_conversion:
            np.clip(buf, 0, 1, out=buf)
            result[i] = from_float(buf, target_dtype=input_dtype)

    if not need_conversion:
        np.clip(result, 0, 1, out=result)

    return result


def unsharp_mask(
    image: np.ndarray,
    ksize: int,
    sigma: float,
    alpha: float,
    threshold: int,
) -> np.ndarray:
    """Apply unsharp mask to a single image. Sharpen via blur-subtract-add;
    backward-compatible wrapper around unsharp_mask_images.

    Backward-compatible wrapper around unsharp_mask_images for a single image.

    Args:
        image (np.ndarray): Single image, shape (H, W, C) or (H, W).
        ksize (int): The kernel size for Gaussian blur.
        sigma (float): The sigma value for Gaussian blur.
        alpha (float): The alpha value for the unsharp mask.
        threshold (int): The threshold value for the unsharp mask.

    Returns:
        np.ndarray: Unsharp masked image with same shape and dtype as input.

    """
    return unsharp_mask_images(np.expand_dims(image, axis=0), ksize, sigma, alpha, threshold)[0]


@preserve_channel_dim
def pixel_dropout(
    image: np.ndarray,
    drop_mask: np.ndarray,
    drop_values: np.ndarray,
) -> np.ndarray:
    """Replace pixels where drop_mask is True with drop_values. Use get_drop_mask,
    prepare_drop_values. For coarse dropout or inpainting-style augmentation.

    Args:
        image (np.ndarray): Input image
        drop_mask (np.ndarray): Boolean mask indicating which pixels to drop
        drop_values (np.ndarray): Values to replace dropped pixels with

    Returns:
        np.ndarray: Image with dropped pixels

    """
    return np.where(drop_mask, drop_values, image)


@float32_io
@clipped
@preserve_channel_dim
def spatter_rain(img: ImageType, rain: np.ndarray) -> ImageType:
    """Add rain layer using precomputed pattern from get_rain_params. Simulates wet surfaces.
    Used by Spatter. float32 I/O, clipped.

    This function applies spatter rain to an image by adding the rain to the image.

    Args:
        img (ImageType): Input image as a numpy array.
        rain (np.ndarray): Rain image as a numpy array.

    Returns:
        ImageType: The spatter rain applied to the image.

    """
    return add(img, rain, inplace=False)


@float32_io
@clipped
@preserve_channel_dim
def spatter_mud(img: ImageType, non_mud: np.ndarray, mud: np.ndarray) -> ImageType:
    """Spatter mud: blend non_mud and mud layers. non_mud, mud from get_mud_params. Simulates dirt on
    lens/surface. float32 I/O, clipped.

    This function applies spatter mud to an image by adding the mud to the image.

    Args:
        img (ImageType): Input image as a numpy array.
        non_mud (np.ndarray): Non-mud image as a numpy array.
        mud (np.ndarray): Mud image as a numpy array.

    Returns:
        ImageType: The spatter mud applied to the image.

    """
    return add(img * non_mud, mud, inplace=False)


@uint8_io
@clipped
def chromatic_aberration(
    img: ImageType,
    primary_distortion_red: float,
    secondary_distortion_red: float,
    primary_distortion_blue: float,
    secondary_distortion_blue: float,
    interpolation: int,
) -> ImageType:
    """Chromatic aberration: shift R/B channels. primary/secondary_distortion_red/blue, interpolation.
    Simulates lens R/B shift. uint8 I/O, clipped.

    This function applies chromatic aberration to an image by distorting the red and blue channels.

    Args:
        img (ImageType): Input image as a numpy array.
        primary_distortion_red (float): The primary distortion of the red channel.
        secondary_distortion_red (float): The secondary distortion of the red channel.
        primary_distortion_blue (float): The primary distortion of the blue channel.
        secondary_distortion_blue (float): The secondary distortion of the blue channel.
        interpolation (int): The interpolation method to use.

    Returns:
        ImageType: The chromatic aberration applied to the image.

    """
    height, width = img.shape[:2]

    # Build camera matrix
    camera_mat = np.eye(3, dtype=np.float32)
    camera_mat[0, 0] = width
    camera_mat[1, 1] = height
    camera_mat[0, 2] = width / 2.0
    camera_mat[1, 2] = height / 2.0

    # Build distortion coefficients
    distortion_coeffs_red = np.array(
        [primary_distortion_red, secondary_distortion_red, 0, 0],
        dtype=np.float32,
    )
    distortion_coeffs_blue = np.array(
        [primary_distortion_blue, secondary_distortion_blue, 0, 0],
        dtype=np.float32,
    )

    # Distort the red and blue channels
    red_distorted = _distort_channel(
        img[..., 0],
        camera_mat,
        distortion_coeffs_red,
        height,
        width,
        interpolation,
    )
    blue_distorted = _distort_channel(
        img[..., 2],
        camera_mat,
        distortion_coeffs_blue,
        height,
        width,
        interpolation,
    )

    return np.dstack([red_distorted, img[..., 1], blue_distorted])


def _distort_channel(
    channel: np.ndarray,
    camera_mat: np.ndarray,
    distortion_coeffs: np.ndarray,
    height: int,
    width: int,
    interpolation: int,
) -> np.ndarray:
    map_x, map_y = cv2.initUndistortRectifyMap(
        cameraMatrix=camera_mat,
        distCoeffs=distortion_coeffs,
        R=None,
        newCameraMatrix=camera_mat,
        size=(width, height),
        m1type=cv2.CV_32FC1,
    )
    return cv2.remap(
        channel,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_REPLICATE,
    )


PLANCKIAN_COEFFS: dict[str, dict[int, list[float]]] = {
    "blackbody": {
        3_000: [0.6743, 0.4029, 0.0013],
        3_500: [0.6281, 0.4241, 0.1665],
        4_000: [0.5919, 0.4372, 0.2513],
        4_500: [0.5623, 0.4457, 0.3154],
        5_000: [0.5376, 0.4515, 0.3672],
        5_500: [0.5163, 0.4555, 0.4103],
        6_000: [0.4979, 0.4584, 0.4468],
        6_500: [0.4816, 0.4604, 0.4782],
        7_000: [0.4672, 0.4619, 0.5053],
        7_500: [0.4542, 0.4630, 0.5289],
        8_000: [0.4426, 0.4638, 0.5497],
        8_500: [0.4320, 0.4644, 0.5681],
        9_000: [0.4223, 0.4648, 0.5844],
        9_500: [0.4135, 0.4651, 0.5990],
        10_000: [0.4054, 0.4653, 0.6121],
        10_500: [0.3980, 0.4654, 0.6239],
        11_000: [0.3911, 0.4655, 0.6346],
        11_500: [0.3847, 0.4656, 0.6444],
        12_000: [0.3787, 0.4656, 0.6532],
        12_500: [0.3732, 0.4656, 0.6613],
        13_000: [0.3680, 0.4655, 0.6688],
        13_500: [0.3632, 0.4655, 0.6756],
        14_000: [0.3586, 0.4655, 0.6820],
        14_500: [0.3544, 0.4654, 0.6878],
        15_000: [0.3503, 0.4653, 0.6933],
    },
    "cied": {
        4_000: [0.5829, 0.4421, 0.2288],
        4_500: [0.5510, 0.4514, 0.2948],
        5_000: [0.5246, 0.4576, 0.3488],
        5_500: [0.5021, 0.4618, 0.3941],
        6_000: [0.4826, 0.4646, 0.4325],
        6_500: [0.4654, 0.4667, 0.4654],
        7_000: [0.4502, 0.4681, 0.4938],
        7_500: [0.4364, 0.4692, 0.5186],
        8_000: [0.4240, 0.4700, 0.5403],
        8_500: [0.4127, 0.4705, 0.5594],
        9_000: [0.4023, 0.4709, 0.5763],
        9_500: [0.3928, 0.4713, 0.5914],
        10_000: [0.3839, 0.4715, 0.6049],
        10_500: [0.3757, 0.4716, 0.6171],
        11_000: [0.3681, 0.4717, 0.6281],
        11_500: [0.3609, 0.4718, 0.6380],
        12_000: [0.3543, 0.4719, 0.6472],
        12_500: [0.3480, 0.4719, 0.6555],
        13_000: [0.3421, 0.4719, 0.6631],
        13_500: [0.3365, 0.4719, 0.6702],
        14_000: [0.3313, 0.4719, 0.6766],
        14_500: [0.3263, 0.4719, 0.6826],
        15_000: [0.3217, 0.4719, 0.6882],
    },
}


@clipped
def planckian_jitter(
    img: ImageType,
    temperature: int,
    mode: Literal["blackbody", "cied"],
) -> ImageType:
    """Apply Planckian jitter (color temp shift) to an image. Params: temperature,
    mode (blackbody/cied). Used for color augmentation; RGB only.

    This function applies Planckian jitter to an image by linearly interpolating
    between the two closest temperatures in the PLANCKIAN_COEFFS dictionary.

    Args:
        img (ImageType): Input image as a numpy array.
        temperature (int): The temperature to apply.
        mode (Literal['blackbody', 'cied']): The mode to use.

    Returns:
        ImageType: The Planckian jitter applied to the image.

    """
    img = img.copy()
    # Get the min and max temperatures for the given mode
    min_temp = min(PLANCKIAN_COEFFS[mode].keys())
    max_temp = max(PLANCKIAN_COEFFS[mode].keys())

    # Clamp the temperature to the available range
    temperature = np.clip(temperature, min_temp, max_temp)

    # Linearly interpolate between 2 closest temperatures
    step = 500
    t_left = max(
        (temperature // step) * step,
        min_temp,
    )  # Ensure t_left doesn't go below min_temp
    t_right = min(
        (temperature // step + 1) * step,
        max_temp,
    )  # Ensure t_right doesn't exceed max_temp

    # Handle the case where temperature is at or near min_temp or max_temp
    if t_left == t_right:
        coeffs = np.array(PLANCKIAN_COEFFS[mode][t_left])
    else:
        w_right = (temperature - t_left) / (t_right - t_left)
        w_left = 1 - w_right
        coeffs = w_left * np.array(PLANCKIAN_COEFFS[mode][t_left]) + w_right * np.array(
            PLANCKIAN_COEFFS[mode][t_right],
        )

    img[..., 0] = multiply_by_constant(
        img[..., 0],
        coeffs[0] / coeffs[1],
        inplace=True,
    )
    img[..., 2] = multiply_by_constant(
        img[..., 2],
        coeffs[2] / coeffs[1],
        inplace=True,
    )

    return img


@clipped
def add_noise(img: ImageType, noise: np.ndarray) -> ImageType:
    """Add pre-generated noise to an image. Noise shape can match image or be
    broadcast; supports per-channel (vector) or full array.

    This function adds noise to an image by adding the noise to the image.

    Args:
        img (ImageType): Input image as a numpy array.
        noise (np.ndarray): Noise as a numpy array.

    Returns:
        ImageType: The noise added to the image.

    """
    if img.ndim == 3 and noise.ndim == 1:
        return add_vector(img, noise, inplace=False)

    n_tiles = np.prod(img.shape) // np.prod(noise.shape)
    noise = np.tile(noise, (n_tiles,) + (1,) * noise.ndim).reshape(img.shape)

    return add_array(img, noise)


def slic(
    image: np.ndarray,
    n_segments: int,
    compactness: float = 10.0,
    max_iterations: int = 10,
) -> np.ndarray:
    """SLIC superpixel segmentation. n_segments, compactness, max_iterations. Returns label mask for
    oversegmentation; used by superpixels() or standalone.

    Args:
        image (np.ndarray): Input image (3D numpy array with shape (H, W, C)).
        n_segments (int): Approximate number of superpixels to generate.
        compactness (float): Balance between color proximity and space proximity.
        max_iterations (int): Maximum number of iterations for k-means.

    Returns:
        np.ndarray: Segmentation mask where each superpixel has a unique label.

    """
    height, width = image.shape[:2]
    num_pixels = height * width

    # Normalize image to [0, 1] range
    max_val = np.float32(image.max())
    image_normalized = image.astype(np.float32) / (max_val + np.float32(1e-6))

    # Initialize cluster centers via meshgrid
    grid_step = int((num_pixels / n_segments) ** 0.5)
    x_range = np.arange(grid_step // 2, width, grid_step)
    y_range = np.arange(grid_step // 2, height, grid_step)
    xx_grid, yy_grid = np.meshgrid(x_range, y_range)
    centers = np.column_stack([xx_grid.ravel(), yy_grid.ravel()]).astype(np.float32)

    # Initialize labels and distances
    labels = np.full((height, width), -1, dtype=np.int32)
    distances = np.full((height, width), np.inf, dtype=np.float32)

    inv_grid_step_sq = np.float32(1.0 / (grid_step * grid_step))

    for _ in range(max_iterations):
        for i, center in enumerate(centers):
            y, x = int(center[1]), int(center[0])

            y_low, y_high = max(0, y - grid_step), min(height, y + grid_step + 1)
            x_low, x_high = max(0, x - grid_step), min(width, x + grid_step + 1)

            crop = image_normalized[y_low:y_high, x_low:x_high]
            color_diff = crop - image_normalized[y, x]
            color_distance = reduce_sum(color_diff**2, axis=-1)

            yy, xx = np.ogrid[y_low:y_high, x_low:x_high]
            spatial_distance = ((yy - y) ** 2 + (xx - x) ** 2) * inv_grid_step_sq

            distance = color_distance + compactness * spatial_distance

            dist_slice = distances[y_low:y_high, x_low:x_high]
            mask = distance < dist_slice
            dist_slice[mask] = distance[mask]
            labels[y_low:y_high, x_low:x_high][mask] = i

        for i in range(len(centers)):
            mask = labels == i
            if np.any(mask):
                ys, xs = np.where(mask)
                centers[i] = [mean(xs.astype(np.float32)), mean(ys.astype(np.float32))]

    return labels


@preserve_channel_dim
@float32_io
def shot_noise(
    img: ImageType,
    scale: float,
    random_generator: np.random.Generator,
) -> ImageType:
    """Add shot (Poisson) noise in linear light space; scale controls strength. Simulates
    sensor noise. Use for camera noise augmentation. float32 I/O, clipped.

    Args:
        img (ImageType): Input image
        scale (float): Scale factor for the noise
        random_generator (np.random.Generator): Random number generator

    Returns:
        ImageType: Image with shot noise

    """
    # Apply inverse gamma correction to work in linear space
    img_linear = cv2.pow(img, 2.2)

    # Scale image values and add small constant to avoid zero values
    scaled_img = (img_linear + scale * 1e-6) / scale

    # Generate Poisson noise
    noisy_img = multiply_by_constant(
        random_generator.poisson(scaled_img).astype(np.float32),
        scale,
        inplace=True,
    )

    # Scale back and apply gamma correction
    return power(np.clip(noisy_img, 0, 1, out=noisy_img), 1 / 2.2)


def get_safe_brightness_contrast_params(
    alpha: float,
    beta: float,
    max_value: float,
) -> tuple[float, float]:
    """Get (alpha, beta) brightness/contrast params clipped to valid LUT range. alpha, beta, max_value.
    Returns (alpha, beta). For LUT-based brightness/contrast.

    Args:
        alpha (float): Contrast factor
        beta (float): Brightness factor
        max_value (float): Maximum pixel value

    Returns:
        tuple[float, float]: Safe alpha and beta values

    """
    if alpha > 0:
        # For x = max_value: alpha * max_value + beta <= max_value
        # For x = 0: beta >= 0
        safe_beta = np.clip(beta, 0, max_value)
        # From alpha * max_value + safe_beta <= max_value
        safe_alpha = min(alpha, (max_value - safe_beta) / max_value)
    else:
        # For x = 0: beta <= max_value
        # For x = max_value: alpha * max_value + beta >= 0
        safe_beta = min(beta, max_value)
        # From alpha * max_value + safe_beta >= 0
        safe_alpha = max(alpha, -safe_beta / max_value)

    return safe_alpha, safe_beta


def generate_constant_noise_with_py_random(
    noise_type: Literal["uniform", "gaussian", "laplace", "beta"],
    shape: tuple[int, ...],
    params: dict[str, Any] | None,
    max_value: float,
    py_random: random.Random,
) -> np.ndarray:
    """Generate constant (per-channel) noise using Python's random. Faster for small
    arrays; used when one value per channel suffices.

    This function generates constant noise using Python's random generator, which is
    more efficient for generating a small number of values (one per channel).

    Args:
        noise_type (Literal['uniform', 'gaussian', 'laplace', 'beta']): The type of noise to generate.
        shape (tuple[int, ...]): The shape of the noise to generate.
        params (dict[str, Any] | None): The parameters of the noise to generate.
        max_value (float): The maximum value of the noise to generate.
        py_random (random.Random): Python's random generator to use.

    Returns:
        np.ndarray: The noise generated.

    """
    if params is None:
        return np.zeros(shape[-1], dtype=np.float32)

    num_channels = shape[-1]

    if noise_type == "uniform":
        ranges = params["ranges"]
        if len(ranges) == 1:
            ranges = ranges * num_channels
        elif len(ranges) < num_channels:
            raise ValueError(
                f"Not enough ranges provided. Expected {num_channels}, got {len(ranges)}",
            )
        # Use py_random for constant mode (faster for small number of values)
        return (
            np.array(
                [py_random.uniform(low, high) for low, high in ranges[:num_channels]],
                dtype=np.float32,
            )
            * max_value
        )

    if noise_type == "gaussian":
        # Sample mean and std once for all channels
        mean = py_random.uniform(*params["mean_range"])
        std = py_random.uniform(*params["std_range"])
        # Python's random has gauss() method
        return (
            np.array(
                [py_random.gauss(mean, std) for _ in range(num_channels)],
                dtype=np.float32,
            )
            * max_value
        )

    if noise_type == "laplace":
        # Sample location and scale once for all channels
        loc = py_random.uniform(*params["mean_range"])
        scale = py_random.uniform(*params["scale_range"])

        # Implement laplace using inverse transform method
        # Laplace CDF inverse: F^(-1)(p) = loc - scale * sign(p - 0.5) * ln(1 - 2*|p - 0.5|)
        def laplace_sample() -> float:
            u = py_random.random()
            if u < 0.5:
                return loc + scale * math.log(2 * u)
            return loc - scale * math.log(2 * (1 - u))

        return (
            np.array(
                [laplace_sample() for _ in range(num_channels)],
                dtype=np.float32,
            )
            * max_value
        )

    if noise_type == "beta":
        # Sample alpha, beta, and scale once for all channels
        alpha = py_random.uniform(*params["alpha_range"])
        beta = py_random.uniform(*params["beta_range"])
        scale = py_random.uniform(*params["scale_range"])
        # Python's random has betavariate() method
        # Transform from [0,1] to [-scale, scale]
        return (
            np.array(
                [(2 * py_random.betavariate(alpha, beta) - 1) * scale for _ in range(num_channels)],
                dtype=np.float32,
            )
            * max_value
        )

    raise ValueError(f"Unknown noise type: {noise_type}")


def generate_spatial_noise(
    noise_type: Literal["uniform", "gaussian", "laplace", "beta"],
    spatial_mode: Literal["per_pixel", "shared"],
    shape: tuple[int, ...],
    params: dict[str, Any] | None,
    max_value: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate per-pixel or spatially-shared noise of the requested distribution and shape,
    scaling samples by `max_value` into the image dtype range.

    Args:
        noise_type (Literal['uniform', 'gaussian', 'laplace', 'beta']): The type of noise to generate.
        spatial_mode (Literal['per_pixel', 'shared']): The spatial mode to use.
        shape (tuple[int, ...]): The shape of the noise to generate.
        params (dict[str, Any] | None): The parameters of the noise to generate.
        max_value (float): The maximum value of the noise to generate.
        random_generator (np.random.Generator): The random number generator to use.

    Returns:
        np.ndarray: The noise generated.

    """
    if params is None:
        return np.zeros(shape, dtype=np.float32)

    cv2_seed = random_generator.integers(0, 2**16)
    cv2.setRNGSeed(cv2_seed)

    if spatial_mode == "shared":
        return generate_shared_noise(
            noise_type,
            shape,
            params,
            max_value,
            random_generator,
        )
    return generate_per_pixel_noise(
        noise_type,
        shape,
        params,
        max_value,
        random_generator,
    )


def generate_per_pixel_noise(
    noise_type: Literal["uniform", "gaussian", "laplace", "beta"],
    shape: tuple[int, ...],
    params: dict[str, Any],
    max_value: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate per-pixel noise from chosen distribution (uniform, gaussian, laplace, beta). Shape and
    spatial_mode from params.
    matches image.

    This function generates per-pixel noise by sampling from the noise distribution.

    Args:
        noise_type (Literal['uniform', 'gaussian', 'laplace', 'beta']): The type of noise to generate.
        shape (tuple[int, ...]): The shape of the noise to generate.
        params (dict[str, Any]): The parameters of the noise to generate.
        max_value (float): The maximum value of the noise to generate.
        random_generator (np.random.Generator): The random number generator to use.

    Returns:
        np.ndarray: The per-pixel noise generated.

    """
    return sample_noise(noise_type, shape, params, max_value, random_generator)


def sample_noise(
    noise_type: Literal["uniform", "gaussian", "laplace", "beta"],
    size: tuple[int, ...],
    params: dict[str, Any],
    max_value: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Sample from noise distribution (uniform, gaussian, laplace, beta). noise_type and params.
    Dispatches to sample_*; returns array of given size.

    This function samples from a specific noise distribution.

    Args:
        noise_type (Literal['uniform', 'gaussian', 'laplace', 'beta']): The type of noise to generate.
        size (tuple[int, ...]): The size of the noise to generate.
        params (dict[str, Any]): The parameters of the noise to generate.
        max_value (float): The maximum value of the noise to generate.
        random_generator (np.random.Generator): The random number generator to use.

    Returns:
        np.ndarray: The noise sampled.

    """
    if noise_type == "uniform":
        return sample_uniform(size, params, random_generator) * max_value
    if noise_type == "gaussian":
        return sample_gaussian(size, params, random_generator) * max_value
    if noise_type == "laplace":
        return sample_laplace(size, params, random_generator) * max_value
    if noise_type == "beta":
        return sample_beta(size, params, random_generator) * max_value

    raise ValueError(f"Unknown noise type: {noise_type}")


def sample_uniform(
    size: tuple[int, ...],
    params: dict[str, Any],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Sample from uniform distribution for spatial noise. params['ranges'][0] per channel.
    Returns array of given size. Used by noise augmentation.

    Args:
        size (tuple[int, ...]): Size of the output array
        params (dict[str, Any]): Distribution parameters
        random_generator (np.random.Generator): Random number generator

    Returns:
        np.ndarray: Sampled values

    """
    # use first range for spatial noise
    low, high = params["ranges"][0]
    return random_generator.uniform(low, high, size=size)


def sample_gaussian(
    size: tuple[int, ...],
    params: dict[str, Any],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Sample from Gaussian distribution. Mean and std from params or uniform range.
    Uses cv2.randn; returns float32 array of given size.

    This function samples from a Gaussian distribution.

    Args:
        size (tuple[int, ...]): The size of the noise to generate.
        params (dict[str, Any]): The parameters of the noise to generate.
        random_generator (np.random.Generator): The random number generator to use.

    Returns:
        np.ndarray: The Gaussian noise sampled.

    """
    mean = (
        params["mean_range"][0]
        if params["mean_range"][0] == params["mean_range"][1]
        else random_generator.uniform(*params["mean_range"])
    )
    std = (
        params["std_range"][0]
        if params["std_range"][0] == params["std_range"][1]
        else random_generator.uniform(*params["std_range"])
    )
    num_channels = size[2] if len(size) > MONO_CHANNEL_DIMENSIONS else 1
    mean_vector = mean * np.ones(shape=(num_channels,), dtype=np.float32)
    std_dev_vector = std * np.ones(shape=(num_channels,), dtype=np.float32)
    gaussian_sampled_arr = np.zeros(shape=size)

    cv2.randn(dst=gaussian_sampled_arr, mean=mean_vector, stddev=std_dev_vector)
    return gaussian_sampled_arr.astype(np.float32)


def sample_laplace(
    size: tuple[int, ...],
    params: dict[str, Any],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Sample from Laplace distribution. Location and scale from params.
    Returns array of given size. Heavier tails than Gaussian.

    This function samples from a Laplace distribution.

    Args:
        size (tuple[int, ...]): The size of the noise to generate.
        params (dict[str, Any]): The parameters of the noise to generate.
        random_generator (np.random.Generator): The random number generator to use.

    Returns:
        np.ndarray: The Laplace noise sampled.

    """
    loc = random_generator.uniform(*params["mean_range"])
    scale = random_generator.uniform(*params["scale_range"])
    return random_generator.laplace(loc=loc, scale=scale, size=size)


def sample_beta(
    size: tuple[int, ...],
    params: dict[str, Any],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Sample from Beta distribution. Alpha, beta, scale from params. Returns values in
    [-scale, scale]. For structured noise augmentation.

    This function samples from a Beta distribution.

    Args:
        size (tuple[int, ...]): The size of the noise to generate.
        params (dict[str, Any]): The parameters of the noise to generate.
        random_generator (np.random.Generator): The random number generator to use.

    Returns:
        np.ndarray: The Beta noise sampled.

    """
    alpha = random_generator.uniform(*params["alpha_range"])
    beta = random_generator.uniform(*params["beta_range"])
    scale = random_generator.uniform(*params["scale_range"])

    # Sample from Beta[0,1] and transform to [-scale,scale]
    samples = random_generator.beta(alpha, beta, size=size)
    return (2 * samples - 1) * scale


def generate_shared_noise(
    noise_type: Literal["uniform", "gaussian", "laplace", "beta"],
    shape: tuple[int, ...],
    params: dict[str, Any],
    max_value: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate shared noise (one value per pixel, broadcast across channels). Used
    by MultiplicativeNoise and generate_spatial_noise when spatial_mode is shared.

    Args:
        noise_type (Literal['uniform', 'gaussian', 'laplace', 'beta']): Type of noise to generate
        shape (tuple[int, ...]): Shape of the output array
        params (dict[str, Any]): Distribution parameters
        max_value (float): Maximum value for the noise
        random_generator (np.random.Generator): Random number generator

    Returns:
        np.ndarray: Generated noise

    """
    # Generate noise for (H, W)
    height, width = shape[:2]
    noise_map = sample_noise(
        noise_type,
        (height, width),
        params,
        max_value,
        random_generator,
    )

    # If input is multichannel, broadcast noise to all channels
    if len(shape) > MONO_CHANNEL_DIMENSIONS:
        return np.broadcast_to(noise_map[..., None], shape)
    return noise_map


@clipped
@preserve_channel_dim
def sharpen_gaussian(
    img: ImageType,
    alpha: float,
    kernel_size: int,
    sigma: float,
) -> ImageType:
    """Sharpen via unsharp mask: subtract Gaussian blur, add back with alpha. kernel_size, sigma
    control blur. Use for crisp edges. Clipped.

    This function sharpens an image using a Gaussian blur.

    Args:
        img (ImageType): The image to sharpen.
        alpha (float): The alpha value to use for the sharpening.
        kernel_size (int): The kernel size to use for the Gaussian blur.
        sigma (float): The sigma value to use for the Gaussian blur.

    Returns:
        ImageType: The sharpened image.

    """
    blurred = cv2.GaussianBlur(
        img,
        ksize=(kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    return add_weighted(img, 1.0 + alpha, blurred, -alpha)


# Native Albumentations equivalents of Pillow's enhancement filters. Each kernel is
# normalized so its weights sum to 1 (DC-preserving). With the blend formula
#     K(alpha) = (1 - alpha) * I + alpha * E
# alpha=0 leaves the image untouched, alpha=1 reproduces the Pillow preset, and
# larger alpha overshoots into stronger variants. For "edge" specifically, alpha=2
# reproduces Pillow's EDGE_ENHANCE_MORE.
_ENHANCE_KERNELS: dict[str, np.ndarray] = {
    "edge": np.array(
        [
            [-0.5, -0.5, -0.5],
            [-0.5, 5.0, -0.5],
            [-0.5, -0.5, -0.5],
        ],
        dtype=np.float32,
    ),
    "detail": np.array(
        [
            [0.0, -1.0 / 6.0, 0.0],
            [-1.0 / 6.0, 10.0 / 6.0, -1.0 / 6.0],
            [0.0, -1.0 / 6.0, 0.0],
        ],
        dtype=np.float32,
    ),
}

_ENHANCE_IDENTITY = np.array(
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    dtype=np.float32,
)


def generate_enhance_matrix(mode: Literal["edge", "detail"], alpha: float) -> np.ndarray:
    """Build a Pillow-inspired 3x3 enhancement kernel via `(1 - alpha) * I + alpha * E`,
    where E is selected by `mode` (edge or detail). Apply via `convolve`.

    Args:
        mode (Literal['edge', 'detail']): Which native enhancement operator to use.
            - "edge": crispens contours; alpha=1 matches Pillow's EDGE_ENHANCE,
              alpha=2 matches Pillow's EDGE_ENHANCE_MORE.
            - "detail": mild local detail boost; alpha=1 matches Pillow's DETAIL.
        alpha (float): Blend strength. 0 returns the identity kernel; 1 returns the
            full preset; values >1 push past the preset for a stronger effect.

    Returns:
        np.ndarray: A `(3, 3)` float32 convolution kernel.

    Raises:
        ValueError: If `mode` is not one of the supported enhancement modes.

    """
    if mode not in _ENHANCE_KERNELS:
        msg = f"Unsupported enhance mode: {mode!r}. Supported modes are: {tuple(_ENHANCE_KERNELS)}"
        raise ValueError(msg)
    kernel = (1.0 - alpha) * _ENHANCE_IDENTITY + alpha * _ENHANCE_KERNELS[mode]
    return kernel.astype(np.float32, copy=False)


def apply_salt_and_pepper(
    img: ImageType,
    salt_mask: np.ndarray,
    pepper_mask: np.ndarray,
) -> ImageType:
    """Apply salt and pepper noise using pre-computed masks. Replaces pixels with
    min or max value; amount controlled by masks.

    This function applies salt and pepper noise to an image using pre-computed masks.
    Salt pixels are set to maximum value, pepper pixels are set to 0.

    Args:
        img (ImageType): Input image of any dtype and dimensions:
            - 2D: (H, W) - grayscale
            - 3D: (H, W, C) - RGB/multi-channel
            - 4D: (D, H, W, C) - volume with depth
        salt_mask (np.ndarray): Boolean mask indicating salt pixels (H, W)
        pepper_mask (np.ndarray): Boolean mask indicating pepper pixels (H, W)

    Returns:
        ImageType: The image with salt and pepper noise applied.

    """
    max_value = MAX_VALUES_BY_DTYPE[img.dtype]
    return np.where(salt_mask[..., None], max_value, np.where(pepper_mask[..., None], 0, img))


# Pre-compute constant kernels
DIAMOND_KERNEL = np.array(
    [
        [0.25, 0.0, 0.25],
        [0.0, 0.0, 0.0],
        [0.25, 0.0, 0.25],
    ],
    dtype=np.float32,
)

SQUARE_KERNEL = np.array(
    [
        [0.0, 0.25, 0.0],
        [0.25, 0.0, 0.25],
        [0.0, 0.25, 0.0],
    ],
    dtype=np.float32,
)

# Pre-compute initial grid
INITIAL_GRID_SIZE = (3, 3)


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
        return cv2.normalize(expanded_grid, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Pre-compute noise scales
    max_dimension = max(target_shape)
    power_of_two_size = 2 ** np.ceil(np.log2(max_dimension - 1)) + 1
    total_steps = int(np.log2(power_of_two_size - 1) - 1)
    noise_scales = np.float32([roughness**i for i in range(total_steps)])

    # Initialize with small random grid
    plasma_grid = random_generator.uniform(-1, 1, (3, 3)).astype(np.float32)

    # Recursively apply diamond-square steps
    for noise_scale in noise_scales:
        plasma_grid = one_diamond_square_step(plasma_grid, noise_scale)

    return np.clip(
        cv2.normalize(plasma_grid[: target_shape[0], : target_shape[1]], None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F),
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
    return img * (1 - scaled_pattern)


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
        return np.linspace(0, 1, width, dtype=np.float32)[None, :] * np.ones((height, 1), dtype=np.float32)
    if angle == 180:
        return np.linspace(1, 0, width, dtype=np.float32)[None, :] * np.ones((height, 1), dtype=np.float32)

    # Fast path for vertical gradients
    if angle == 90:
        return np.linspace(0, 1, height, dtype=np.float32)[:, None] * np.ones((1, width), dtype=np.float32)
    if angle == 270:
        return np.linspace(1, 0, height, dtype=np.float32)[:, None] * np.ones((1, width), dtype=np.float32)

    # Fast path for diagonal gradients using broadcasting
    if angle in (45, 135, 225, 315):
        x = np.linspace(0, 1, width, dtype=np.float32)[None, :]  # Horizontal
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None]  # Vertical

        if angle == 45:  # Bottom-left to top-right
            return cv2.normalize(x + y, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if angle == 135:  # Bottom-right to top-left
            return cv2.normalize((1 - x) + y, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if angle == 225:  # Top-right to bottom-left
            return cv2.normalize((1 - x) + (1 - y), None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # angle == 315:  # Top-left to bottom-right
        return cv2.normalize(x + (1 - y), None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # General case for arbitrary angles using broadcasting
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]  # Column vector
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]  # Row vector

    angle_rad = np.deg2rad(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    cv2.multiply(x, cos_a, dst=x)
    cv2.multiply(y, sin_a, dst=y)

    return x + y


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
    abs_intensity = abs(intensity)

    if mode == "linear":
        gradient = create_directional_gradient(height, width, params["angle"])
        if intensity < 0:
            cv2.subtract(1, gradient, dst=gradient)
        cv2.multiply(gradient, 2 * abs_intensity, dst=gradient)
        cv2.add(gradient, 1 - abs_intensity, dst=gradient)
        return gradient

    if mode == "corner":
        if intensity == 0:
            return np.ones((height, width), dtype=np.float32)
        corner = params["corner"]
        diagonal_length = math.sqrt(height * height + width * width)
        mask = np.full((height, width), 255, dtype=np.uint8)
        corners = [(0, 0), (0, width - 1), (height - 1, width - 1), (height - 1, 0)]
        mask[corners[corner]] = 0
        pattern = cv2.distanceTransform(
            mask,
            distanceType=cv2.DIST_L2,
            maskSize=cv2.DIST_MASK_PRECISE,
            dstType=cv2.CV_32F,
        )
        cv2.multiply(pattern, -intensity / diagonal_length, dst=pattern)
        cv2.add(pattern, 1, dst=pattern)
        return pattern

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
    cv2.multiply(x, -1 / sigma2, dst=x)
    cv2.exp(x, dst=x)
    cv2.multiply(x, intensity, dst=x)
    cv2.add(x, 1, dst=x)
    return x


@float32_io
def apply_linear_illumination(img: ImageType, intensity: float, angle: float) -> ImageType:
    """Apply linear illumination gradient to the image. Multiplies by gradient; intensity and angle
    control direction and strength. float32 I/O.

    Args:
        img (ImageType): Input image
        intensity (float): Illumination intensity
        angle (float): Illumination angle in radians

    Returns:
        ImageType: Image with linear illumination

    """
    height, width = img.shape[:2]
    abs_intensity = abs(intensity)

    # Create gradient and handle negative intensity in one step
    gradient = create_directional_gradient(height, width, angle)

    if intensity < 0:
        cv2.subtract(1, gradient, dst=gradient)

    cv2.multiply(gradient, 2 * abs_intensity, dst=gradient)
    cv2.add(gradient, 1 - abs_intensity, dst=gradient)

    # Add channel dimension if needed
    if img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
        gradient = gradient[..., np.newaxis]

    return multiply_by_array(img, gradient)


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

    # Pre-compute diagonal length once
    diagonal_length = math.sqrt(height * height + width * width)

    # Create inverted distance map mask directly
    # Use uint8 for distanceTransform regardless of input dtype
    mask = np.full((height, width), 255, dtype=np.uint8)

    # Use array indexing instead of conditionals
    corners = [(0, 0), (0, width - 1), (height - 1, width - 1), (height - 1, 0)]
    mask[corners[corner]] = 0

    # Calculate distance transform
    pattern = cv2.distanceTransform(
        mask,
        distanceType=cv2.DIST_L2,
        maskSize=cv2.DIST_MASK_PRECISE,
        dstType=cv2.CV_32F,  # Specify float output directly
    )

    # Combine operations to reduce array copies
    cv2.multiply(pattern, -intensity / diagonal_length, dst=pattern)
    cv2.add(pattern, 1, dst=pattern)

    if img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
        pattern = cv2.merge([pattern] * img.shape[2])

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
    cv2.multiply(x, -1 / sigma2, dst=x)
    cv2.exp(x, dst=x)

    # Scale by intensity
    cv2.multiply(x, intensity, dst=x)
    cv2.add(x, 1, dst=x)

    if img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
        x = cv2.merge([x] * img.shape[2])

    return multiply_by_array(img, x)


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
    result = img.copy()
    num_channels = img.shape[-1]
    max_value = MAX_VALUES_BY_DTYPE[img.dtype]

    # Pre-compute histograms using cv2.calcHist - much faster than np.histogram
    channels = cv2.split(img)
    hists: list[np.ndarray] = []
    for i, channel in enumerate(channels):
        if ignore is not None and i == ignore:
            hists.append(None)
            continue
        mask = None if ignore is None else (channel != ignore)
        hist = cv2.calcHist([channel], [0], mask, [256], [0, max_value])
        hists.append(hist.ravel())

    for i in range(num_channels):
        if ignore is not None and i == ignore:
            continue

        hist = hists[i]
        channel = channels[i]

        lo, hi = get_histogram_bounds(hist, cutoff)
        if hi <= lo:
            continue

        lut = create_contrast_lut(hist, lo, hi, max_value, method)
        if ignore is not None:
            lut[ignore] = ignore

        result[..., i] = sz_lut(channel, lut)

    return result


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
        return random_generator.choice(
            [True, False],
            shape,
            p=[dropout_prob, 1 - dropout_prob],
        )

    # Generate 2D mask and expand to match channels
    mask_2d = random_generator.choice(
        [True, False],
        shape[:2],
        p=[dropout_prob, 1 - dropout_prob],
    )

    # If input is 2D, return 2D mask
    if len(shape) == 2:
        return mask_2d

    # For 3D input, expand and repeat across channels
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


def get_rain_params(
    liquid_layer: np.ndarray,
    color: np.ndarray,
    intensity: float,
) -> dict[str, Any]:
    """Generate parameters for rain effect. liquid_layer, color, intensity. Returns dict with 'drops'
    for add_rain/spatter_rain.

    This function generates parameters for a rain effect.

    Args:
        liquid_layer (np.ndarray): Liquid layer of the image.
        color (np.ndarray): Color of the rain.
        intensity (float): Intensity of the rain.

    Returns:
        dict[str, Any]: Parameters for the rain effect.

    """
    liquid_layer = clip(liquid_layer * 255, np.uint8, inplace=False)

    # Generate distance transform with more defined edges
    dist = 255 - cv2.Canny(liquid_layer, 50, 150)
    dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
    _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)

    # Use separate blur operations for better drop formation
    dist = cv2.GaussianBlur(
        dist,
        ksize=(3, 3),
        sigmaX=1,  # Add slight sigma for smoother drops
        sigmaY=1,
        borderType=cv2.BORDER_REPLICATE,
    )
    dist = clip(dist, np.uint8, inplace=True)

    dist = dist[..., np.newaxis]

    # Enhance contrast in the distance map
    dist = equalize(dist)
    # Modified kernel for more natural drop shapes
    ker = np.array(
        [
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2],
        ],
        dtype=np.float32,
    )

    # Apply convolution with better precision
    dist = convolve(dist, ker)

    # Final blur with larger kernel for smoother drops
    dist = cv2.GaussianBlur(
        dist,
        ksize=(5, 5),  # Increased kernel size
        sigmaX=1.5,  # Adjusted sigma
        sigmaY=1.5,
        borderType=cv2.BORDER_REPLICATE,
    ).astype(np.float32)

    # Calculate final rain mask with better blending
    m = liquid_layer.astype(np.float32) * dist

    # Normalize with better handling of edge cases
    m_max = np.max(m, axis=(0, 1))
    if m_max > 0:
        m *= 1 / m_max
    else:
        m = np.zeros_like(m)

    # Apply color with adjusted intensity for more natural look
    drops = m[:, :, None] * color * (intensity * 0.9)  # Slightly reduced intensity

    return {
        "drops": drops,
    }


def get_mud_params(
    liquid_layer: np.ndarray,
    color: np.ndarray,
    cutout_threshold: float,
    sigma: float,
    intensity: float,
    random_generator: np.random.Generator,
) -> dict[str, Any]:
    """Generate parameters for mud effect. liquid_layer, color, cutout_threshold, sigma, intensity,
    random_generator. Returns dict for spatter_mud.

    This function generates parameters for a mud effect.

    Args:
        liquid_layer (np.ndarray): Liquid layer of the image.
        color (np.ndarray): Color of the mud.
        cutout_threshold (float): Cutout threshold for the mud.
        sigma (float): Sigma for the Gaussian blur.
        intensity (float): Intensity of the mud.
        random_generator (np.random.Generator): Random number generator.

    Returns:
        dict[str, Any]: Parameters for the mud effect.

    """
    height, width = liquid_layer.shape

    # Create initial mask (ensure we have some non-zero values)
    mask = (liquid_layer > cutout_threshold).astype(np.float32)
    if reduce_sum(mask) == 0:  # If mask is all zeros
        # Force minimum coverage of 10%
        num_pixels = height * width
        num_needed = max(1, int(0.1 * num_pixels))  # At least 1 pixel
        flat_indices = random_generator.choice(num_pixels, num_needed, replace=False)
        mask = np.zeros_like(liquid_layer, dtype=np.float32)
        mask.flat[flat_indices] = 1.0

    # Apply Gaussian blur if sigma > 0
    if sigma > 0:
        mask = cv2.GaussianBlur(
            mask,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE,
        )

    # Safe normalization (avoid division by zero)
    mask_max = np.max(mask)
    if mask_max > 0:
        mask = mask / mask_max
    else:
        # If mask is somehow all zeros after blur, force some effect
        mask[0, 0] = 1.0

    # Scale by intensity directly (no minimum)
    mask = mask * intensity

    # Create mud effect array
    mud = np.zeros((height, width, 3), dtype=np.float32)

    # Apply color directly - the intensity scaling is already handled
    for i in range(3):
        mud[..., i] = mask * color[i]

    # Create complementary non-mud array
    non_mud = np.ones_like(mud)
    for i in range(3):
        if color[i] > 0:
            non_mud[..., i] = np.clip((color[i] - mud[..., i]) / color[i], 0, 1)
        else:
            non_mud[..., i] = 1.0 - mask

    return {
        "mud": mud.astype(np.float32),
        "non_mud": non_mud.astype(np.float32),
    }


# Standard reference H&E stain matrices
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


def rgb_to_optical_density(img: ImageType, eps: float = 1e-6) -> np.ndarray:
    """Convert RGB image to optical density (-log10). eps avoids log(0). Expects uint8 or float32 in
    [0,1]. Returns (N*H*W, 3) float64. For stain normalization.

    This function converts an RGB image to optical density.

    Args:
        img (ImageType): Input image.
        eps (float): Epsilon value.

    Returns:
        np.ndarray: Optical density image.

    """
    max_value = MAX_VALUES_BY_DTYPE[img.dtype]
    pixel_matrix = img.reshape(-1, 3).astype(np.float32)
    pixel_matrix = np.maximum(pixel_matrix / max_value, eps)
    return -np.log(pixel_matrix)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length (L2). Axis and dtype preserved; 1D or 2D. For stain
    normalization (e.g. Macenko) stain vector normalization.

    This function normalizes vectors.

    Args:
        vectors (np.ndarray): Vectors to normalize.

    Returns:
        np.ndarray: Normalized vectors.

    """
    norms = np.sqrt(reduce_sum(vectors**2, axis=1, keepdims=True))
    return vectors / norms


def get_normalizer(method: Literal["vahadane", "macenko"]) -> "StainNormalizer":
    """Get stain normalizer based on method ('vahadane' or 'macenko'). Returns
    VahadaneNormalizer or MacenkoNormalizer instance for histology stain norm.

    This function gets a stain normalizer based on a method.

    Args:
        method (Literal['vahadane', 'macenko']): Method to use for stain normalization.

    Returns:
        StainNormalizer: Stain normalizer.

    """
    return VahadaneNormalizer() if method == "vahadane" else MacenkoNormalizer()


class StainNormalizer:
    """Base class for stain normalizers. Subclass and implement fit/transform for
    histology stain normalization (e.g. Vahadane, Macenko).
    """

    def __init__(self) -> None:
        self.stain_matrix_target = None

    def fit(self, img: ImageType) -> None:
        """Fit the stain normalizer to a reference image. Learns stain matrix from img; call transform
        on target images after. Subclass implements the actual extraction.

        This function fits the stain normalizer to an image.

        Args:
            img (ImageType): Input image.

        """
        raise NotImplementedError


class SimpleNMF:
    """Simple NMF for histology stain separation. Factorizes OD matrix into stain basis and
    concentrations. Iterative multiplicative updates, non-negativity.

    This class implements a simplified version of the Non-negative Matrix Factorization algorithm
    specifically designed for separating Hematoxylin and Eosin (H&E) stains in histopathology images.
    It is used as part of the Vahadane stain normalization method.

    The algorithm decomposes optical density values of H&E stained images into stain color appearances
    (the stain color vectors) and stain concentrations (the density of each stain at each pixel).

    The implementation uses an iterative multiplicative update approach that preserves non-negativity
    constraints, which are physically meaningful for stain separation as concentrations and
    absorption coefficients cannot be negative.

    This implementation is optimized for stability by:
    1. Initializing with standard H&E reference colors from Ruifrok
    2. Using normalized projection for initial concentrations
    3. Applying careful normalization to avoid numerical issues

    Args:
        n_iter (int): Number of iterations for the NMF algorithm. Default: 100

    References:
        - Vahadane, A., et al. (2016): Structure-preserving color normalization and
          sparse stain separation for histological images. IEEE Transactions on
          Medical Imaging, 35(8), 1962-1971.
        - Ruifrok, A. C., & Johnston, D. A. (2001): Quantification of histochemical
          staining by color deconvolution. Analytical and Quantitative Cytology and
          Histology, 23(4), 291-299.

    """

    def __init__(self, n_iter: int = 100):
        self.n_iter = n_iter
        # Initialize with standard H&E colors from Ruifrok
        self.initial_colors = np.array(
            [
                [0.644211, 0.716556, 0.266844],  # Hematoxylin
                [0.092789, 0.954111, 0.283111],  # Eosin
            ],
            dtype=np.float32,
        )

    def fit_transform(self, optical_density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit the NMF model to optical density matrix. Learns stain basis and
        concentrations; used internally by VahadaneNormalizer for stain separation.

        This function fits the NMF model to optical density.

        Args:
            optical_density (np.ndarray): Optical density image.

        Returns:
            tuple[np.ndarray, np.ndarray]: Stain concentrations and stain colors.

        """
        # Start with known H&E colors
        stain_colors = self.initial_colors.copy()

        # Initialize concentrations based on projection onto initial colors
        # This gives us a physically meaningful starting point
        stain_colors_normalized = normalize_vectors(stain_colors)

        # Suppress numerical warnings for edge cases (handled by eps)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            stain_concentrations = np.maximum(optical_density @ stain_colors_normalized.T, 0)

            # Iterative updates with careful normalization
            eps = 1e-6
            for _ in range(self.n_iter):
                # Update concentrations
                numerator = optical_density @ stain_colors.T
                denominator = stain_concentrations @ (stain_colors @ stain_colors.T)
                stain_concentrations *= numerator / (denominator + eps)

                # Ensure non-negativity
                stain_concentrations = np.maximum(stain_concentrations, 0)

                # Update colors
                numerator = stain_concentrations.T @ optical_density
                denominator = (stain_concentrations.T @ stain_concentrations) @ stain_colors
                stain_colors *= numerator / (denominator + eps)

                # Ensure non-negativity and normalize
                stain_colors = np.maximum(stain_colors, 0)
                stain_colors = normalize_vectors(stain_colors)

        return stain_concentrations, stain_colors


def order_stains_combined(stain_colors: np.ndarray) -> tuple[int, int]:
    """Order stains using a combination of methods (angular and spectral).
    Returns ordered stain matrix for consistent H/E ordering.

    This combines both angular information and spectral characteristics
    for more robust identification.

    Args:
        stain_colors (np.ndarray): Stain colors.

    Returns:
        tuple[int, int]: Hematoxylin and eosin indices.

    """
    # Normalize stain vectors
    stain_colors = normalize_vectors(stain_colors)

    # Calculate angles (Macenko)
    angles = np.mod(np.arctan2(stain_colors[:, 1], stain_colors[:, 0]), np.pi)

    # Calculate spectral ratios (Ruifrok)
    blue_ratio = stain_colors[:, 2] / (reduce_sum(stain_colors, axis=1) + 1e-6)
    red_ratio = stain_colors[:, 0] / (reduce_sum(stain_colors, axis=1) + 1e-6)

    # Combine scores
    # High angle and high blue ratio indicates Hematoxylin
    # Low angle and high red ratio indicates Eosin
    scores = angles * blue_ratio - red_ratio

    hematoxylin_idx = np.argmax(scores)
    eosin_idx = 1 - hematoxylin_idx

    return hematoxylin_idx, eosin_idx


class VahadaneNormalizer(StainNormalizer):
    """Vahadane stain normalizer for histopathology. NMF-based stain separation;
    fit on reference image, then transform. Used for H&E normalization.

    This class implements the "Structure-Preserving Color Normalization and Sparse Stain Separation
    for Histological Images" method proposed by Vahadane et al. The technique uses Non-negative
    Matrix Factorization (NMF) to separate Hematoxylin and Eosin (H&E) stains in histopathology
    images and then normalizes them to a target standard.

    The Vahadane method is particularly effective for histology image normalization because:
    1. It maintains tissue structure during color normalization
    2. It performs sparse stain separation, reducing color bleeding
    3. It adaptively estimates stain vectors from each image
    4. It preserves biologically relevant information

    This implementation uses SimpleNMF as its core matrix factorization algorithm to extract
    stain color vectors (appearance matrix) and concentration matrices from optical
    density-transformed images. It identifies the Hematoxylin and Eosin stains by their
    characteristic color profiles and spatial distribution.

    References:
        Vahadane, et al., 2016: Structure-preserving color normalization
        and sparse stain separation for histological images. IEEE transactions on medical imaging,
        35(8), pp.1962-1971.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> from albumentations.augmentations.pixel import functional as F
        >>> import cv2
        >>>
        >>> # Load source and target images (H&E stained histopathology)
        >>> source_img = cv2.imread('source_image.png')
        >>> source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        >>> target_img = cv2.imread('target_image.png')
        >>> target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        >>>
        >>> # Create and fit the normalizer to the target image
        >>> normalizer = F.VahadaneNormalizer()
        >>> normalizer.fit(target_img)
        >>>
        >>> # Normalize the source image to match the target's stain characteristics
        >>> normalized_img = normalizer.transform(source_img)

    """

    def fit(self, img: ImageType) -> None:
        """Fit the Vahadane stain normalizer to a reference image. Runs NMF on OD
        matrix; call transform on target images for normalization.

        This function fits the Vahadane stain normalizer to an image.

        Args:
            img (ImageType): Input image.

        """
        optical_density = rgb_to_optical_density(img)

        nmf = SimpleNMF(n_iter=100)
        _, stain_colors = nmf.fit_transform(optical_density)

        # Use combined method for robust stain ordering
        hematoxylin_idx, eosin_idx = order_stains_combined(stain_colors)

        self.stain_matrix_target = np.array(
            [
                stain_colors[hematoxylin_idx],
                stain_colors[eosin_idx],
            ],
        )


class MacenkoNormalizer(StainNormalizer):
    """Macenko stain normalizer with optimized computations. SVD-based stain
    separation; fit on reference, then transform. Used for H&E normalization.
    """

    def __init__(self, angular_percentile: float = 99):
        super().__init__()
        self.angular_percentile = angular_percentile

    def fit(self, img: ImageType, angular_percentile: float = 99) -> None:
        """Fit the Macenko stain normalizer to a reference image. SVD-based;
        call transform on target images for H&E normalization.

        This function fits the Macenko stain normalizer to an image.

        Args:
            img (ImageType): Input image.
            angular_percentile (float): Angular percentile.

        """
        # Step 1: Convert RGB to optical density (OD) space
        optical_density = rgb_to_optical_density(img)

        # Step 2: Remove background pixels
        od_threshold = 0.05
        threshold_mask = (optical_density > od_threshold).any(axis=1)
        tissue_density = optical_density[threshold_mask]

        if len(tissue_density) < 1:
            raise ValueError(f"No tissue pixels found (threshold={od_threshold})")

        # Step 3: Compute covariance matrix
        tissue_density = np.ascontiguousarray(tissue_density, dtype=np.float32)
        od_covariance = cv2.calcCovarMatrix(
            tissue_density,
            None,
            cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE,
        )[0]

        # Step 4: Get principal components
        eigenvalues, eigenvectors = cv2.eigen(od_covariance)[1:]
        idx = np.argsort(eigenvalues.ravel())[-2:]
        principal_eigenvectors = np.ascontiguousarray(eigenvectors[:, idx], dtype=np.float32)

        # Step 5: Project onto eigenvector plane
        # Add small epsilon to avoid numerical instability
        epsilon = 1e-8
        if np.any(np.abs(principal_eigenvectors) < epsilon):
            # Regularize near-zero entries by assigning ±ε based on original sign
            principal_eigenvectors = np.where(
                np.abs(principal_eigenvectors) < epsilon,
                np.where(principal_eigenvectors < 0, -epsilon, epsilon),
                principal_eigenvectors,
            )

        # Add small epsilon to tissue_density to avoid numerical issues
        safe_tissue_density = tissue_density + epsilon

        # Suppress numerical warnings for edge cases with extreme optical densities
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            plane_coordinates = safe_tissue_density @ principal_eigenvectors

        # Step 6: Find angles of extreme points
        polar_angles = np.arctan2(
            plane_coordinates[:, 1],
            plane_coordinates[:, 0],
        )

        # Get robust angle estimates
        hematoxylin_angle = np.percentile(polar_angles, 100 - angular_percentile)
        eosin_angle = np.percentile(polar_angles, angular_percentile)

        # Step 7: Convert angles back to RGB space
        hem_cos, hem_sin = np.cos(hematoxylin_angle), np.sin(hematoxylin_angle)
        eos_cos, eos_sin = np.cos(eosin_angle), np.sin(eosin_angle)

        angle_to_vector = np.array(
            [[hem_cos, hem_sin], [eos_cos, eos_sin]],
            dtype=np.float32,
        )

        # Ensure both matrices have the same data type for cv2.gemm
        principal_eigenvectors_t = np.ascontiguousarray(principal_eigenvectors.T, dtype=np.float32)
        stain_vectors = cv2.gemm(
            angle_to_vector,
            principal_eigenvectors_t,
            1,
            None,
            0,
        )

        # Step 8: Ensure non-negativity by taking absolute values
        stain_vectors = np.abs(stain_vectors)

        # Step 9: Normalize vectors to unit length
        stain_vectors = stain_vectors / np.sqrt(reduce_sum(stain_vectors**2, axis=1, keepdims=True) + epsilon)

        # Step 10: Order vectors as [hematoxylin, eosin]
        self.stain_matrix_target = stain_vectors if stain_vectors[0, 0] > stain_vectors[1, 0] else stain_vectors[::-1]


def get_tissue_mask(img: ImageType, threshold: float = 0.85) -> np.ndarray:
    """Get tissue mask from image (exclude background). threshold for intensity-based masking of
    non-tissue. Returns 1D bool mask.

    Args:
        img (ImageType): Input image
        threshold (float): Threshold for tissue detection. Default: 0.85

    Returns:
        np.ndarray: Binary mask where True indicates tissue regions

    """
    # Convert to grayscale using RGB weights: R*0.299 + G*0.587 + B*0.114
    luminosity = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114

    # Tissue is darker, so we want pixels below threshold
    mask = luminosity < threshold

    return mask.reshape(-1)


@clipped
@float32_io
def apply_he_stain_augmentation(
    img: ImageType,
    stain_matrix: np.ndarray,
    scale_factors: np.ndarray,
    shift_values: np.ndarray,
    augment_background: bool,
) -> ImageType:
    """Apply HE (hematoxylin-eosin) stain augmentation. Shifts stain concentrations;
    params control strength. Used for histology augmentation. Returns RGB image.

    This function applies HE stain augmentation to an image.

    Args:
        img (ImageType): Input image.
        stain_matrix (np.ndarray): Stain matrix.
        scale_factors (np.ndarray): Scale factors.
        shift_values (np.ndarray): Shift values.
        augment_background (bool): Whether to augment the background.

    Returns:
        ImageType: Augmented image.

    """
    # Step 1: Convert RGB to optical density space
    optical_density = rgb_to_optical_density(img)

    # Step 2: Calculate stain concentrations using regularized pseudo-inverse
    stain_matrix = np.ascontiguousarray(stain_matrix, dtype=np.float32)

    # Add small regularization term for numerical stability
    regularization = 1e-6

    # Suppress numerical warnings for edge cases with extreme optical densities
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        stain_correlation = stain_matrix @ stain_matrix.T + regularization * np.eye(2)
        density_projection = stain_matrix @ optical_density.T

        try:
            # Solve for stain concentrations
            stain_concentrations = np.linalg.solve(stain_correlation, density_projection).T
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if direct solve fails
            stain_concentrations = np.linalg.lstsq(
                stain_matrix.T,
                optical_density,
                rcond=regularization,
            )[0].T

        # Step 3: Apply concentration adjustments
        if not augment_background:
            # Only modify tissue regions
            tissue_mask = get_tissue_mask(img).reshape(-1)
            stain_concentrations[tissue_mask] = stain_concentrations[tissue_mask] * scale_factors + shift_values
        else:
            # Modify all pixels
            stain_concentrations = stain_concentrations * scale_factors + shift_values

        # Step 4: Reconstruct RGB image
        optical_density_result = stain_concentrations @ stain_matrix
        rgb_result = np.exp(-optical_density_result)

    return rgb_result.reshape(img.shape)


@clipped
@preserve_channel_dim
def convolve(img: ImageType, kernel: np.ndarray) -> ImageType:
    """Convolve image with 2D kernel via cv2.filter2D. Any channel count. Use for custom blur,
    sharpen, or edge kernels. Clipped.

    This function convolves an image with a kernel.

    Args:
        img (ImageType): Input image.
        kernel (np.ndarray): Kernel.

    Returns:
        ImageType: Convolved image.

    """
    img = np.array(img, copy=True, order="C")
    cv2.filter2D(img, ddepth=-1, kernel=kernel, dst=img)
    return img


@clipped
@preserve_channel_dim
def separable_convolve(img: ImageType, kernel: np.ndarray) -> ImageType:
    """Convolve with separable 1D kernel in two passes. Faster than full 2D for large kernels.
    Use for Gaussian-like blur or custom separable filters. Clipped.

    This function convolves an image with a separable kernel.

    Args:
        img (ImageType): Input image.
        kernel (np.ndarray): Kernel.

    Returns:
        ImageType: Convolved image.

    """
    img = np.array(img, copy=True, order="C")
    cv2.sepFilter2D(img, ddepth=-1, kernelX=kernel, kernelY=kernel, dst=img)
    return img


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

    inv_lum = 1.0 - luminance.astype(np.float32) / max_val if img.dtype == np.uint8 else 1.0 - luminance

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
    img = np.ascontiguousarray(img)
    height, width = img.shape[:2]
    num_channels = img.shape[-1]

    luminance = (
        mean(img, axis=-1).astype(np.float32) / MAX_VALUES_BY_DTYPE[np.uint8]
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
                avg_color = mean(cell.reshape(-1, num_channels), axis=0).astype(img.dtype)

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
                    color: tuple[int, ...] | int = tuple(int(v) for v in mean(cell.reshape(-1, num_channels), axis=0))
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
        cv2.line(flare_layer, (fx, fy), (x2, y2), float(starburst_intensity), 1, lineType=cv2.LINE_AA)

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


def apply_atmospheric_fog(
    img: ImageType,
    density: float,
    fog_color: tuple[float, ...],
    depth_map: np.ndarray,
) -> ImageType:
    """Apply depth-aware atmospheric fog using standard scattering. Formula: img *
    exp(-density*depth) + fog_color*(1 - exp(-density*depth)).

    Formula: result = img * exp(-density * depth) + fog_color * (1 - exp(-density * depth))

    Args:
        img (ImageType): Input image (H, W, C).
        density (float): Fog density factor.
        fog_color (tuple[float, ...]): Color of the fog, values in [0, max_val].
        depth_map (np.ndarray): (H, W) float32 array with values in [0, 1], where 1 is farthest.

    Returns:
        ImageType: Image with fog applied.

    """
    num_channels = img.shape[-1]
    transmission = np.exp(-density * depth_map).astype(np.float32)[:, :, np.newaxis]

    fog_array = np.array(fog_color, dtype=np.float32)
    if len(fog_array) < num_channels:
        fog_array = np.pad(fog_array, (0, num_channels - len(fog_array)), mode="edge")
    fog_array = fog_array[:num_channels].reshape(1, 1, -1)

    result = img.astype(np.float32) * transmission + fog_array * (1.0 - transmission)

    return clip(result, img.dtype)
