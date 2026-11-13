"""Color, tone, histogram, grayscale, and color conversion functional helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import albucore
import numkong as nk
import torch

if TYPE_CHECKING:
    from albucore import ImageInputT
    from numpy.typing import NDArray

from albumentations.core.utils import get_volume_shape

from ._functional_shared import (
    MAX_VALUES_BY_DTYPE,
    MULTICHANNEL_LUT_LARGE_IMAGE_PIXELS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    NUM_RGB_CHANNELS,
    PCA,
    ImageFloat32,
    ImageUInt8,
    add,
    apply_uint8_lut,
    clip,
    convert_dtype,
    cv2,
    fgeometric,
    get_num_channels,
    is_grayscale_image,
    is_rgb_image,
    mean,
    non_rgb_error,
    normalize_per_image,
    np,
    power,
    reduce_sum,
    restore_uint8_working_image,
    to_uint8_working_image,
    warn,
)


def _reshape_xhwc_for_opencv(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Flatten the leading batch or depth axis before a channel-last OpenCV operation."""
    return data.reshape(-1, data.shape[2], data.shape[3]), data.shape


def _restore_xhwc_from_opencv(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore a flattened OpenCV result to its batch or depth layout."""
    return data.reshape(original_shape[0], original_shape[1], data.shape[1], original_shape[3])


def shift_hsv(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    hue_shift: float,
    sat_shift: float,
    val_shift: float,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Shift hue, saturation, and value in HSV space. hue_shift, sat_shift, val_shift control
    amount; grayscale gets value only. Floating inputs are quantized to uint8 and restored.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): The image to shift.
        hue_shift (float): The amount to shift the hue.
        sat_shift (float): The amount to shift the saturation.
        val_shift (float): The amount to shift the value.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: The shifted image.

    """
    img, input_dtype = to_uint8_working_image(img)
    if hue_shift == 0 and sat_shift == 0 and val_shift == 0:
        return restore_uint8_working_image(img, input_dtype)

    is_gray = is_grayscale_image(img)
    has_explicit_gray_channel = img.ndim == 3 and img.shape[-1] == 1

    if is_gray:
        if hue_shift != 0 or sat_shift != 0:
            hue_shift = 0
            sat_shift = 0
            warn(
                "HueSaturationValue: hue_shift and sat_shift are not applicable to grayscale image. "
                "Set them to 0 or use RGB image",
                stacklevel=2,
            )
        img = cast(
            "NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]", cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        )

    img = cast("NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]", cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    hue, sat, val = (cast("ImageUInt8", channel) for channel in cv2.split(img))

    if hue_shift != 0:
        lut_hue = np.arange(0, 256, dtype=np.int16)
        lut_hue = np.mod(lut_hue + hue_shift, 180).astype(np.uint8)
        hue = apply_uint8_lut(hue, lut_hue)

    if sat_shift != 0:
        # These should remain grayscale regardless of saturation change
        grayscale_mask = sat == 0

        saturation = sat[..., np.newaxis]
        add(saturation, sat_shift, out=saturation)

        # Reset saturation for white pixels
        sat[grayscale_mask] = 0

    if val_shift != 0:
        value_channel = val[..., np.newaxis]
        add(value_channel, val_shift, out=value_channel)

    img = cast("NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]", cv2.merge((hue, sat, val)))
    img = cast("NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]", cv2.cvtColor(img, cv2.COLOR_HSV2RGB))

    if is_gray:
        result = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if has_explicit_gray_channel:
            result = result[..., np.newaxis]
        return restore_uint8_working_image(cast("ImageUInt8", result), input_dtype)
    return restore_uint8_working_image(cast("ImageUInt8", img), input_dtype)


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


def solarize(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32], threshold: float
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Invert pixel values above a normalized threshold (solarization). threshold in [0, 1];
    supports uint8, float16, and float32; pixels below threshold unchanged.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): The image to solarize.
        threshold (float): Normalized threshold value in range [0, 1].
            For uint8 images: pixels above threshold * 255 are inverted
            For float32 images: pixels above threshold are inverted

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Solarized image.

    Note:
        The threshold is normalized to [0, 1] range for both uint8 and float32 images.
        For uint8 images, the threshold is internally scaled by 255.

    """
    dtype = img.dtype
    max_val = MAX_VALUES_BY_DTYPE[dtype]

    if dtype == np.uint8:
        uint8_img = cast("ImageUInt8", img)
        indices = np.arange(int(max_val) + 1, dtype=np.uint8)
        thresh_val = threshold * max_val
        lut = cast("ImageUInt8", np.where(indices >= thresh_val, max_val - indices, indices).astype(dtype))
        prev_shape = img.shape
        result = apply_uint8_lut(uint8_img, lut)
        return (
            result
            if len(prev_shape) == result.ndim
            else cast("NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]", np.expand_dims(result, -1))
        )
    result = np.where(img >= threshold, max_val - img, img)
    return clip(result, dtype)


def posterize(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    bits: Literal[1, 2, 3, 4, 5, 6, 7] | list[Literal[1, 2, 3, 4, 5, 6, 7]],
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Reduce bit depth by keeping only the highest N bits per channel. bits: 1-7 or list per
    channel; quantizes to uint8 for bit masking, then restores the input dtype.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image. Can be single or
            multi-channel.
        bits (Literal[1, 2, 3, 4, 5, 6, 7] | list[Literal[1, 2, 3, 4, 5, 6, 7]]): Number of high bits to keep..
            Can be either:
            - A single value to apply the same bit reduction to all channels
            - A list of values to apply different bit reduction per channel.
              Length of list must match number of channels in image.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Image with reduced bit depth. Has same shape and
            dtype as input.

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
    uint8_img, input_dtype = to_uint8_working_image(img)
    bits_array = np.asarray(bits, dtype=np.uint8)

    if bits_array.ndim == 0 or bits_array.size == 1:
        mask = ~np.uint8(2 ** (8 - bits_array.item()) - 1)
        result = cast("NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]", uint8_img & mask)
        return restore_uint8_working_image(cast("ImageUInt8", result), input_dtype)

    result_img = np.empty_like(uint8_img)
    for i, channel_bits in enumerate(bits_array):
        mask = ~np.uint8(2 ** (8 - int(channel_bits)) - 1)
        np.bitwise_and(uint8_img[..., i], mask, out=result_img[..., i])

    result = cast("NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]", result_img)
    return restore_uint8_working_image(cast("ImageUInt8", result), input_dtype)


def _equalize_pil(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32], mask: np.ndarray | None = None
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    histogram = cv2.calcHist([img], [0], mask, [256], (0, 256)).ravel()
    h = histogram[histogram > 0]

    if len(h) <= 1:
        return img.copy()

    step = int(reduce_sum(h[:-1])) // 255
    if not step:
        return img.copy()

    # Pillow emits each LUT entry before adding the current histogram bin.
    cumulative = np.empty(256, dtype=np.int64)
    cumulative[0] = 0
    np.cumsum(histogram[:-1], dtype=np.int64, out=cumulative[1:])
    lut = np.minimum((cumulative + step // 2) // step, 255).astype(np.uint8)

    result = cast("ImageUInt8", img.copy())
    return apply_uint8_lut(result, lut, out=result)


def _equalize_cv(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32], mask: np.ndarray | None = None
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    if mask is None:
        if img.ndim == 3 and img.shape[-1] == 1:
            result = np.empty_like(img)
            cv2.equalizeHist(img, dst=result)
            return result
        return cast("NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]", cv2.equalizeHist(img))

    histogram = cv2.calcHist([img], [0], mask, [256], (0, 256)).ravel()
    lut = _create_equalize_cv_lut(histogram)
    if lut is None:
        return img

    result = cast("ImageUInt8", img.copy())
    return apply_uint8_lut(result, lut, out=result)


def _create_equalize_cv_lut(histogram: np.ndarray) -> np.ndarray | None:
    nonzero = np.flatnonzero(histogram)
    if len(nonzero) == 0:
        return np.arange(256, dtype=np.uint8)

    first_nonzero = nonzero[0]
    total = reduce_sum(histogram)
    denominator = total - histogram[first_nonzero]
    if denominator == 0:
        return None

    scale = 255.0 / denominator
    cumsum_histogram = np.cumsum(histogram)
    return np.clip(((cumsum_histogram - cumsum_histogram[first_nonzero]) * scale).round(), 0, 255).astype(np.uint8)


def _equalize_cv_multichannel_lut(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Apply OpenCV-style equalization with one multichannel LUT pass for large
    RGB images and multispectral inputs where per-channel assignment is slower.
    """
    num_channels = get_num_channels(img)
    lut = np.empty((256, 1, num_channels), dtype=np.uint8)
    identity_lut = np.arange(256, dtype=np.uint8)
    for channel_idx in range(num_channels):
        channel = img[..., channel_idx]
        histogram = cv2.calcHist([channel], [0], None, [256], (0, 256)).ravel()
        channel_lut = _create_equalize_cv_lut(histogram)
        lut[:, 0, channel_idx] = identity_lut if channel_lut is None else channel_lut

    return apply_uint8_lut(cast("ImageUInt8", img), lut)


def _check_preconditions(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
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
    if i is not None and not is_grayscale_image(mask):
        mask = mask[..., i]

    return mask


def equalize(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    mask: np.ndarray | None = None,
    mode: Literal["cv", "pil"] = "cv",
    by_channels: bool = True,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Apply histogram equalization to the input image. Enhances contrast; supports
    grayscale and color, mode cv/pil, optional mask and by_channels.

    This function enhances the contrast of the input image by equalizing its histogram.
    It supports both grayscale and color images, and can operate on individual channels
    or on the luminance channel of the image.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image. Can be grayscale (2D array) or
            RGB (3D array).
        mask (np.ndarray | None): Optional histogram-selection mask. Nonzero pixels select
            the histogram sample; the resulting mapping is applied to the whole image.
            Must match the image's spatial shape, with one shared channel or a mask channel
            per image channel when by_channels=True. Default: None.
        mode (Literal['cv', 'pil']): The backend to use for equalization. Can be either "cv" for
            OpenCV or "pil" for Pillow-style equalization. Default: "cv".
        by_channels (bool): If True, applies equalization to each channel independently.
            If False, converts the image to YCrCb color space and equalizes only the
            luminance channel. Only applicable to color images. Default: True.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Equalized image. The output has the same dtype as
            the input.

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
    uint8_img, input_dtype = to_uint8_working_image(img)
    img = uint8_img
    _check_preconditions(img, mask, by_channels)
    function = _equalize_pil if mode == "pil" else _equalize_cv

    if is_grayscale_image(img):
        result = cast("ImageUInt8", function(img, _handle_mask(mask)))
        return restore_uint8_working_image(result, input_dtype)

    if mask is None and by_channels and mode == "cv" and is_rgb_image(img):
        if img.shape[0] * img.shape[1] >= MULTICHANNEL_LUT_LARGE_IMAGE_PIXELS:
            result = cast("ImageUInt8", _equalize_cv_multichannel_lut(img))
            return restore_uint8_working_image(result, input_dtype)
        channels = cv2.split(img)
        result = cast("ImageUInt8", cv2.merge([cv2.equalizeHist(channel) for channel in channels]))
        return restore_uint8_working_image(result, input_dtype)

    if mask is None and by_channels and mode == "cv" and get_num_channels(img) > NUM_RGB_CHANNELS:
        result = cast("ImageUInt8", _equalize_cv_multichannel_lut(img))
        return restore_uint8_working_image(result, input_dtype)

    if not by_channels:
        result_img = cast(
            "NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]", cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        )
        result_img[..., 0] = function(result_img[..., 0], _handle_mask(mask))
        result = cast("ImageUInt8", cv2.cvtColor(result_img, cv2.COLOR_YCrCb2RGB))
        return restore_uint8_working_image(result, input_dtype)

    result_img = np.empty_like(img)
    for i in range(get_num_channels(img)):
        _mask = _handle_mask(mask, i)
        channel_result = function(img[..., i], _mask)
        # Remove any extra dimensions that might have been added
        if channel_result.ndim > 2:
            channel_result = channel_result.squeeze()
        result_img[..., i] = channel_result

    return restore_uint8_working_image(result_img, input_dtype)


def evaluate_bez(
    low_y: float | np.ndarray,
    high_y: float | np.ndarray,
) -> np.ndarray:
    """Build a 256-entry cubic Bezier tone curve from shared or per-channel control points for fast lookup-table
    evaluation of uint8 images.

    Args:
        low_y (float | np.ndarray): The first inner control ordinate of the Bezier curve.
        high_y (float | np.ndarray): The second inner control ordinate of the Bezier curve.

    Returns:
        np.ndarray: The Bezier curve values.

    """
    t = np.linspace(0.0, 1.0, 256)[..., None]

    one_minus_t = 1 - t
    return (3 * one_minus_t**2 * t * low_y + 3 * one_minus_t * t**2 * high_y + t**3) * 255


def _move_tone_curve_float32(
    img: ImageFloat32,
    low_y: float | np.ndarray,
    high_y: float | np.ndarray,
) -> ImageFloat32:
    """Evaluate a cubic Bezier tone curve continuously in float32 while reusing one image-sized output buffer for
    memory-efficient processing.
    """
    low_y_float32 = np.asarray(low_y, dtype=np.float32)
    high_y_float32 = np.asarray(high_y, dtype=np.float32)

    coefficient_1 = np.float32(3) * low_y_float32
    coefficient_2 = np.float32(3) * high_y_float32 - np.float32(6) * low_y_float32
    coefficient_3 = np.float32(3) * low_y_float32 - np.float32(3) * high_y_float32 + np.float32(1)

    result = np.empty_like(img)
    if coefficient_1.shape == (3,) and img.shape[-1] == 3:
        # The loop wins at the measured 256/512/1024 single-image C=3 sizes but loses at
        # the measured 1x1-32x32 sizes; the single-machine crossover does not establish a stable cutoff.
        for channel_idx in range(3):
            image_channel = img[..., channel_idx]
            result_channel = result[..., channel_idx]
            np.multiply(image_channel, coefficient_3[channel_idx], out=result_channel)
            np.add(result_channel, coefficient_2[channel_idx], out=result_channel)
            np.multiply(result_channel, image_channel, out=result_channel)
            np.add(result_channel, coefficient_1[channel_idx], out=result_channel)
            np.multiply(result_channel, image_channel, out=result_channel)
    else:
        np.multiply(img, coefficient_3, out=result)
        np.add(result, coefficient_2, out=result)
        np.multiply(result, img, out=result)
        np.add(result, coefficient_1, out=result)
        np.multiply(result, img, out=result)
    np.clip(result, np.float32(0), np.float32(1), out=result)
    # Float32 coefficient rounding can undershoot the cubic curve's exact t=1 endpoint.
    result[img == np.float32(1)] = np.float32(1)
    return result


def move_tone_curve(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    low_y: float | np.ndarray,
    high_y: float | np.ndarray,
    num_channels: int,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Apply a cubic Bezier tone curve with scalar or per-channel controls, dispatching to continuous float32
    arithmetic or a uint8 lookup table.

    Floating inputs use float32 arithmetic and retain dtype. Uint8 uses a rounded 256-entry LUT.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Any number of channels
        low_y (float | np.ndarray): per-channel or single y-position of a Bezier control point used
            to adjust the tone curve, must be in range [0, 1]
        high_y (float | np.ndarray): per-channel or single y-position of a Bezier control point used
            to adjust image tone curve, must be in range [0, 1]
        num_channels (int): The number of channels in the input image.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Image with adjusted tone curve

    """
    input_dtype = img.dtype
    if input_dtype.type is np.float16:
        img = convert_dtype(img, dtype=np.float32)

    if np.isscalar(low_y) and np.isscalar(high_y):
        low_y_scalar = cast("Any", low_y)
        high_y_scalar = cast("Any", high_y)
        if img.dtype == np.float32:
            result = _move_tone_curve_float32(
                cast("ImageFloat32", img),
                float(low_y_scalar),
                float(high_y_scalar),
            )
            return convert_dtype(result, dtype=input_dtype)
        lut = clip(np.rint(evaluate_bez(float(low_y_scalar), float(high_y_scalar))), np.dtype(np.uint8)).reshape(-1)
        return apply_uint8_lut(cast("ImageUInt8", img), lut)

    if isinstance(low_y, np.ndarray) and isinstance(high_y, np.ndarray):
        if img.dtype == np.float32:
            result = _move_tone_curve_float32(cast("ImageFloat32", img), low_y, high_y)
            return convert_dtype(result, dtype=input_dtype)
        lut = clip(np.rint(evaluate_bez(low_y, high_y)), np.dtype(np.uint8))
        return apply_uint8_lut(cast("ImageUInt8", img), lut[:, np.newaxis, :])

    raise TypeError(
        f"low_y and high_y must both be of type float or np.ndarray. Got {type(low_y)} and {type(high_y)}",
    )


def linear_transformation_rgb(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    transformation_matrix: np.ndarray,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """3x3 linear transformation to RGB. transformation_matrix (or batch) multiplies channel
    vector. Supports (H,W,3) and (B,H,W,3).

    This function applies a 3x3 linear transformation matrix (or batch of matrices)
    to the RGB channels of either a single image or a batch of images.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): A single RGB image of shape (H, W, 3) or a
            batch of images (B, H, W, 3).
        transformation_matrix (np.ndarray): A 3x3 matrix

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Transformed image or batch of images, matching
            the input shape and dtype.

    Raises:
        ValueError: If input shapes do not conform to the supported configurations.

    """
    if img.ndim not in (3, 4):
        raise ValueError(f"Expected input shape (H, W, 3) or (B, H, W, 3), got {img.shape}")
    return albucore.linear_transformation_rgb(img, transformation_matrix)


def clahe(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) per tile. clip_limit, tile_grid_size.
    For color, applied in LAB to L channel. Floating inputs are quantized to uint8 and restored.

    This function enhances the contrast of the input image using CLAHE. For color images,
    it converts the image to the LAB color space, applies CLAHE to the L channel, and then
    converts the image back to RGB.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image. Can be grayscale (2D array) or
            RGB (3D array).
        clip_limit (float): Threshold for contrast limiting. Higher values give more contrast.
        tile_grid_size (tuple[int, int]): Size of grid for histogram equalization.
            Width and height of the grid.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Image with CLAHE applied. The output has the same
            dtype as the input.

    Note:
        - Floating inputs are temporarily quantized to uint8 for processing and restored to their original dtype.
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
    img, input_dtype = to_uint8_working_image(img)
    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if is_grayscale_image(img):
        if img.ndim == 3 and img.shape[-1] == 1:
            result = np.empty_like(img)
            clahe_mat.apply(img, dst=result)
            return restore_uint8_working_image(result, input_dtype)
        return restore_uint8_working_image(cast("ImageUInt8", clahe_mat.apply(img)), input_dtype)

    img_lab = cast(
        "NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]", cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    )
    img_lab[:, :, 0] = clahe_mat.apply(img_lab[:, :, 0])

    return restore_uint8_working_image(cast("ImageUInt8", cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)), input_dtype)


def image_compression(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    quality: int,
    image_type: Literal[".jpg", ".webp"],
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Compress image with JPEG or WebP to simulate artifacts. quality, image_type (.jpg/.webp).
    Lower quality increases blockiness. Floating inputs are quantized to uint8 and restored.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image
        quality (int): Quality of compression in range [1, 100]
        image_type (Literal['.jpg', '.webp']): Type of compression to use

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Compressed image

    """
    img, input_dtype = to_uint8_working_image(img)
    # Determine the quality flag for compression
    quality_flag = cv2.IMWRITE_JPEG_QUALITY if image_type == ".jpg" else cv2.IMWRITE_WEBP_QUALITY
    num_channels = get_num_channels(img)

    # Prepare to encode and decode
    def encode_decode(
        src_img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32], read_mode: int
    ) -> np.ndarray:
        _, encoded_img = cv2.imencode(image_type, src_img, (quality_flag, quality))
        return cast("np.ndarray", cv2.imdecode(encoded_img, read_mode))

    if num_channels == 1:
        # Grayscale image
        decoded = encode_decode(img, cv2.IMREAD_GRAYSCALE)
        result = cast("ImageUInt8", decoded[..., np.newaxis])
        return restore_uint8_working_image(result, input_dtype)

    if num_channels in (2, NUM_RGB_CHANNELS):
        # 2 channels: pad to 3, or 3 (RGB) channels
        padded_img = (
            cast(
                "NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]",
                np.pad(img, ((0, 0), (0, 0), (0, 1)), mode="constant"),
            )
            if num_channels == 2
            else img
        )
        decoded_bgr = encode_decode(padded_img, cv2.IMREAD_UNCHANGED)
        result = cast("ImageUInt8", decoded_bgr[..., :num_channels])
        return restore_uint8_working_image(result, input_dtype)

    # More than 3 channels
    bgr = img[..., :NUM_RGB_CHANNELS]
    decoded_bgr = encode_decode(bgr, cv2.IMREAD_UNCHANGED)

    # Process additional channels
    extra_channels = [
        encode_decode(img[..., i], cv2.IMREAD_GRAYSCALE)[..., np.newaxis] for i in range(NUM_RGB_CHANNELS, num_channels)
    ]
    result = cast("ImageUInt8", np.dstack([decoded_bgr, *extra_channels]))
    return restore_uint8_working_image(result, input_dtype)


def invert(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Produce the negative image: each pixel becomes max_val - pixel. uint8/float32, any
    channels. Use for inversion augmentation or visualization.

    This function inverts the colors of an image by subtracting each pixel value from the maximum possible value.
    The result is a negative of the original image.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): The image to invert.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: The inverted image.

    """
    # Supports all the valid dtypes
    # clips the img to avoid unexpected behaviour.
    return MAX_VALUES_BY_DTYPE[img.dtype] - img


def channel_shuffle(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32], channels_shuffled: list[int]
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Shuffle image channels. channels_shuffled gives new order; supports
    (H, W, C) or batch (N, H, W, C).

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): The image to shuffle.
        channels_shuffled (list[int]): The array of channels to shuffle.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: The shuffled image.

    """
    if (
        img.dtype in (np.uint8, np.float16, np.float32)
        and channels_shuffled == [2, 1, 0]
        and img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS
        and img.shape[-1] == NUM_RGB_CHANNELS
        and img.size
    ):
        if img.dtype == np.float16:
            # OpenCV accepts uint16 for this channel permutation. A view preserves every FP16 bit.
            return cast("NDArray[np.float16]", cv2.cvtColor(img.view(np.uint16), cv2.COLOR_RGB2BGR).view(np.float16))
        return cast("NDArray[np.uint8] | NDArray[np.float32]", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    img = cast("NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]", np.ascontiguousarray(img))
    output = np.empty(img.shape, dtype=img.dtype)
    from_to = []
    for i, j in enumerate(channels_shuffled):
        from_to.extend([j, i])  # Use [src, dst]
    cv2.mixChannels([img], [output], from_to)
    return cast("NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]", output)


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


def get_exposure_gains(
    images: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    target_mean: float,
    gain_range: tuple[float, float] | None,
) -> float | np.ndarray:
    """Compute exposure gains from normalized global means for one image or a tensor of images,
    enabling batch-aware brightness scaling without per-image dispatch.

    Args:
        images (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Image or tensor of images. The final
            three dimensions must be `(H, W, C)`.
        target_mean (float): Desired normalized mean in [0, 1].
        gain_range (tuple[float, float] | None): Optional inclusive lower and upper gain bounds.

    Returns:
        float | np.ndarray: One scalar gain for an image or an array matching the leading batch dimensions.

    """
    image_axes = tuple(range(images.ndim - 3, images.ndim))
    values_per_image = images.shape[-3] * images.shape[-2] * images.shape[-1]

    image_sums: int | float | nk.Tensor | NDArray[np.float64] | None
    if images.dtype == np.uint8:
        image_sums = nk.sum(images, axis=image_axes)
        if image_sums is None:
            raise RuntimeError("NumKong did not return exposure sums")
    else:
        image_sums = np.sum(images, axis=image_axes, dtype=np.float64)

    normalized_means = np.asarray(image_sums, dtype=np.float64) / (values_per_image * MAX_VALUES_BY_DTYPE[images.dtype])
    gains = np.asarray(target_mean / np.maximum(normalized_means, 1e-6), dtype=np.float64)

    if gain_range is not None:
        np.clip(gains, gain_range[0], gain_range[1], out=gains)

    return float(gains) if gains.ndim == 0 else gains


def exposure_match_batch(
    images: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32], gains: np.ndarray
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Scale an image tensor by per-image exposure gains in one vectorized operation for image batches
    and the `volume` target without per-image dispatch.

    Args:
        images (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Image batch or `volume` data ending in
            `(H, W, C)`.
        gains (np.ndarray): Gain array matching every leading dimension before `(H, W, C)`.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Scaled tensor with the input shape and dtype.

    Raises:
        ValueError: If the gain shape does not match the image tensor's leading dimensions.

    """
    expected_shape = images.shape[:-3]
    if gains.shape != expected_shape:
        raise ValueError(f"Expected exposure gains with shape {expected_shape}, got {gains.shape}")

    if gains.size == 0:
        return images.copy()

    if images.dtype == np.uint8:
        flattened_images = images.reshape(gains.size, -1)
        gain_matrix = np.broadcast_to(gains.reshape(-1, 1), flattened_images.shape)
        return cast(
            "NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]",
            cv2.multiply(flattened_images, gain_matrix, dtype=cv2.CV_8U).reshape(images.shape),
        )

    gain_shape = (*gains.shape, 1, 1, 1)
    result = np.multiply(images, gains.reshape(gain_shape), dtype=np.float32)
    np.clip(result, 0, MAX_VALUES_BY_DTYPE[images.dtype], out=result)
    return convert_dtype(result, dtype=images.dtype)


def gamma_transform(img: ImageInputT, gamma: float) -> ImageInputT:
    """Apply a power-law correction (gamma correction) to image intensities, brightening or darkening
    midtones to adjust exposure and tonal contrast.

    For normalized intensities in [0, 1], output = input ** gamma: gamma > 1 darkens,
    0 < gamma < 1 brightens, and gamma = 1 leaves intensities unchanged.
    uint8 values are normalized by 255 before the power operation, then scaled back by 255
    and converted to uint8, truncating fractional values. Floating arrays use the existing power kernel.
    NumPy inputs are channel-last; Tensor inputs are CPU CHW images.

    Args:
        img (ImageInputT): Input uint8 image in [0, 255], or float16/float32 image in [0, 1].
        gamma (float): The exponent applied to normalized intensities.

    Returns:
        ImageInputT: The gamma transformed image, preserving container and dtype.

    """
    if isinstance(img, torch.Tensor):
        if img.numel() == 0:
            return img
        if img.dtype != torch.uint8:
            return cast("ImageInputT", power(img, gamma, layout="CHW"))
    else:
        if img.size == 0:
            return img
        if img.dtype != np.uint8:
            return cast(
                "ImageInputT", power(cast("NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]", img), gamma)
            )
    table = (np.arange(256, dtype=np.float64) / 255) ** gamma * 255
    return cast("ImageInputT", apply_uint8_lut(cast("ImageUInt8 | torch.Tensor", img), table.astype(np.uint8)))


def _iso_noise_restore(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    clip(image, out=image)
    return convert_dtype(image, dtype=dtype)


def iso_noise(
    image: np.ndarray,
    color_shift: float,
    intensity: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Apply Poisson noise to simulate camera sensor noise. color_shift, intensity;
    approximates shot noise. Float32 working arithmetic with explicit output clipping.

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
        uint8, float16, float32

    Number of channels:
        3

    """
    original_dtype = image.dtype
    image = convert_dtype(image, np.float32)
    hls = cast("NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]", cv2.cvtColor(image, cv2.COLOR_RGB2HLS))
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
    hls[..., 1] = add(
        hls[..., 1],
        cast(
            "NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]",
            (luminance_noise * intensity * (1.0 - hls[..., 1])).astype(np.float32, copy=False),
        ),
    )

    result = cast("np.ndarray", cv2.cvtColor(hls, cv2.COLOR_HLS2RGB))
    return _iso_noise_restore(result, original_dtype)


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
        uint8, float16, float32

    Number of channels:
        3

    """
    original_dtype = images.dtype
    images = convert_dtype(images, np.float32)
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

    return _iso_noise_restore(hls_batch, original_dtype)


def iso_noise_volume(
    volume: np.ndarray,
    color_shift: float,
    intensity: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Apply camera-like ISO noise to an RGB volume with a single seeded voxel-wise field, so hue and
    luminance vary through depth. Use for 3D RGB data.
    """
    original_dtype = volume.dtype
    volume = convert_dtype(volume, np.float32)
    non_rgb_error(volume[0])
    hls_volume = np.empty_like(volume)
    for depth_index, image in enumerate(volume):
        cv2.cvtColor(image, cv2.COLOR_RGB2HLS, dst=hls_volume[depth_index])

    luminance_std = float(np.std(hls_volume[..., 1]))
    volume_shape = get_volume_shape(volume)
    luminance_noise = random_generator.poisson(luminance_std * intensity, size=volume_shape).astype(np.float32)
    color_noise = random_generator.normal(0, color_shift * intensity, size=volume_shape).astype(np.float32)

    hls_volume[..., 0] += color_noise
    luminance_noise *= intensity
    hls_volume[..., 1] *= 1.0 - luminance_noise
    hls_volume[..., 1] += luminance_noise

    for image in hls_volume:
        cv2.cvtColor(image, cv2.COLOR_HLS2RGB, dst=image)

    return _iso_noise_restore(hls_volume, original_dtype)


def to_gray_weighted_average(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Convert RGB to grayscale with weighted average (0.299*R+0.587*G+0.114*B). Single or batch.
    BT.601. Matches OpenCV perceptual luminance.

    This function uses OpenCV's cvtColor function with COLOR_RGB2GRAY conversion,
    which applies the following formula:
    Y = 0.299*R + 0.587*G + 0.114*B

    The function efficiently handles batches and the `volume` target by reshaping them into
    a tall 2D image for processing, then restoring the original shape structure.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input RGB image(s) as a numpy array.
            Supported shapes:
            - Single image: (H, W, 3)
            - Batch of images: (N, H, W, 3)
            - Volume: (D, H, W, 3)

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Grayscale image with unchanged spatial and
            collection axes and C=1.

    Image types:
        uint8, float16, float32

    Number of channels:
        3

    """
    input_dtype = img.dtype
    if input_dtype.type is np.float16:
        img = convert_dtype(img, dtype=np.float32)
    if img.ndim == 3:
        gray = np.empty((*img.shape[:-1], 1), dtype=img.dtype)
        result = cast(
            "NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]",
            cv2.cvtColor(img, cv2.COLOR_RGB2GRAY, dst=gray),
        )
    elif img.ndim == 4:
        im, original_shape = _reshape_xhwc_for_opencv(img)
        gray = np.empty((*im.shape[:-1], 1), dtype=im.dtype)
        im = cast(
            "NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]",
            cv2.cvtColor(im, cv2.COLOR_RGB2GRAY, dst=gray),
        )
        result = cast(
            "NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]",
            _restore_xhwc_from_opencv(im, (*original_shape[:-1], 1)),
        )
    else:
        raise ValueError(f"Unsupported number of dimensions: {img.ndim}")
    if input_dtype.type is np.float16:
        return convert_dtype(result, cast("np.dtype[np.float16]", input_dtype))
    return result


def to_gray_from_lab(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Convert RGB to grayscale using the LAB L channel; perceived brightness matches human vision
    better than a simple average. Single image or batch. Floating inputs are quantized to uint8 and restored.

    This function converts RGB images to grayscale by first converting to LAB color space
    and then extracting the L (lightness) channel. It flattens batches and volumes into one tall image
    for the OpenCV call.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input RGB image(s) as a numpy array. Must
            have 3 channels in the last dimension.
            Supported shapes:
            - Single image: (H, W, 3)
            - Batch of images: (N, H, W, 3)
            - Volume: (D, H, W, 3)

            Supported dtypes:
            - np.uint8: Values in range [0, 255]
            - np.float16 / np.float32: Values in range [0, 1]

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
            - Single image: (H, W, 1)
            - Batch of images: (N, H, W, 1)
            - Volume: (D, H, W, 1)

        The output dtype matches the input dtype. Floating outputs are normalized to [0, 1].

    Raises:
        ValueError: If the last dimension is not 3 (RGB channels)

    Examples:
        >>> # Single image
        >>> img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> gray = to_gray_from_lab(img)
        >>> assert gray.shape == (100, 100, 1)

        >>> # Batch of images - efficiently processed without loops
        >>> batch = np.random.randint(0, 256, (10, 100, 100, 3), dtype=np.uint8)
        >>> gray_batch = to_gray_from_lab(batch)
        >>> assert gray_batch.shape == (10, 100, 100, 1)

        >>> # Volume (e.g., video frames or 3D medical data)
        >>> volume = np.random.randint(0, 256, (16, 100, 100, 3), dtype=np.uint8)
        >>> gray_volume = to_gray_from_lab(volume)
        >>> assert gray_volume.shape == (16, 100, 100, 1)

        >>> # Float32 input
        >>> img_float = img.astype(np.float32) / 255.0
        >>> gray_float = to_gray_from_lab(img_float)
        >>> assert 0 <= gray_float.min() <= gray_float.max() <= 1.0

    Note:
        The LAB color space provides perceptually uniform grayscale conversion,
        where the L (lightness) channel represents human perception of brightness
        better than simple RGB averaging or other methods.

    """
    img, input_dtype = to_uint8_working_image(img)
    if img.ndim == 3:
        result = cast("ImageUInt8", cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., :1])
        return restore_uint8_working_image(result, input_dtype)
    if img.ndim == 4:
        im, original_shape = _reshape_xhwc_for_opencv(img)
        im = cast("ImageUInt8", cv2.cvtColor(im, cv2.COLOR_RGB2LAB)[..., :1])

        new_shape = (*original_shape[:-1], 1)
        result = cast("ImageUInt8", _restore_xhwc_from_opencv(im, new_shape))
        return restore_uint8_working_image(result, input_dtype)

    raise ValueError(f"Unsupported number of dimensions: {img.ndim}")


def to_gray_desaturation(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Convert to grayscale with desaturation (max + min) / 2 per pixel. Any channels; single or batch.
    uint8, float16, and float32.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image as a numpy array.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Grayscale image with unchanged spatial and
            collection axes and C=1.

    Image types:
        uint8, float16, float32

    Number of channels:
        any

    """
    if img.dtype == np.uint8:
        if img.shape[-1] == 1:
            return img
        if img.ndim > NUM_MULTI_CHANNEL_DIMENSIONS:
            wide_max = np.max(img, axis=-1, keepdims=True).astype(np.uint16)
            wide_min = np.min(img, axis=-1, keepdims=True).astype(np.uint16)
            return ((wide_max + wide_min) >> 1).astype(np.uint8)
        channels = cv2.split(img)
        ch_max = channels[0]
        ch_min = channels[0]
        for channel in channels[1:]:
            ch_max = cv2.max(ch_max, channel)
            ch_min = cv2.min(ch_min, channel)
        channel_sum = np.empty((*img.shape[:-1], 1), dtype=np.uint16)
        cv2.add(ch_max, ch_min, dst=channel_sum, dtype=cv2.CV_16U)
        return (channel_sum >> 1).astype(np.uint8)
    float_image = convert_dtype(img, np.float32)
    result = (np.max(float_image, axis=-1, keepdims=True) + np.min(float_image, axis=-1, keepdims=True)) * 0.5
    return clip(result, img.dtype)


def to_gray_average(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
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
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image as a numpy array. Can be any
            number of channels.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Grayscale image with unchanged leading axes, C=1,
            and the input dtype.

    Image types:
        uint8, float16, float32

    Number of channels:
        any

    """
    return cast(
        "NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]",
        np.asarray(mean(img, axis=-1, keepdims=True)).astype(img.dtype),
    )


def to_gray_max(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
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
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image as a numpy array. Can be any
            number of channels.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Grayscale image with unchanged leading axes, C=1,
            and the input dtype.

    Image types:
        uint8, float16, float32

    Number of channels:
        any

    """
    return np.max(img, axis=-1, keepdims=True)


def to_gray_pca(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Reduce to one channel via PCA; captures max variance in color. Single or batch; uint8 or
    float16/float32. Use when simple averaging loses info.

    This function applies PCA to reduce a multi-channel image to a single channel,
    effectively creating a grayscale representation that captures the maximum variance
    in the color data.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image as a numpy array. Can be:
            - Single multi-channel image: (H, W, C)
            - Batch of multi-channel images: (N, H, W, C)
            - Single multi-channel volume: (D, H, W, C)
            - Five-dimensional multi-channel tensor: (N, D, H, W, C)

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Grayscale image with the same leading dimensions
            as input and C=1.
                    If input is uint8, output is uint8 in range [0, 255].
                    If input is float16 or float32, output retains that dtype and lies in [0, 1].

    Note:
        This method can potentially preserve more information from the original image
        compared to standard weighted average methods, as it accounts for the
        correlations between color channels.

    Image types:
        uint8, float16, float32

    Number of channels:
        any

    """
    dtype = img.dtype
    pixels = img.reshape(-1, img.shape[-1])

    pca = PCA(n_components=1, dtype=np.float32)
    pca_result: ImageFloat32 = pca.fit_transform(pixels)

    grayscale = pca_result.reshape(*img.shape[:-1], 1)
    grayscale = normalize_per_image(grayscale, "min_max")

    if dtype == np.uint8:
        return convert_dtype(grayscale, dtype=np.uint8)
    return clip(grayscale, dtype)


def to_gray(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    num_output_channels: int,
    method: Literal[
        "weighted_average",
        "from_lab",
        "desaturation",
        "average",
        "max",
        "pca",
    ],
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
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
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image as a numpy array.
        num_output_channels (int): The number of channels in the output image.
        method (Literal['weighted_average', 'from_lab', 'desaturation', 'average', 'max', 'pca']):
            The method to use for grayscale conversion.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Grayscale image with unchanged leading axes and
            num_output_channels channels.

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
    """Repeat an explicit grayscale channel, preserving spatial and collection axes.

    Args:
        grayscale_image (np.ndarray): Channel-last grayscale data with C=1, such as
            (H, W, 1), (N, H, W, 1), or (N, D, H, W, 1).
        num_output_channels (int): Number of output channels. Defaults to 3.

    Returns:
        np.ndarray: Grayscale values repeated along the last axis. Requesting one
            output channel returns the input unchanged.

    """
    if num_output_channels == 1:
        return grayscale_image

    if num_output_channels == 3 and grayscale_image.ndim == 3 and grayscale_image.dtype in (np.uint8, np.float32):
        return cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)

    return np.tile(grayscale_image, (1,) * (grayscale_image.ndim - 1) + (num_output_channels,))


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
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    black: tuple[int, int, int],
    white: tuple[int, int, int],
    mid: tuple[int, int, int] | None,
    mid_value: int,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Map grayscale intensity to a 2- or 3-color RGB gradient (Pillow `ImageOps.colorize`
    style) using a (256, 3) LUT for uint8 and float32-working `np.interp` for floating inputs.

    Anchor colors are specified as RGB tuples in 0-255 regardless of input dtype; for float32
    inputs the anchors are scaled into [0, 1] before interpolation. Output always has 3 channels.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Single-channel image, shape `(H, W, 1)` or
            batch `(N, H, W, 1)`.
        black (tuple[int, int, int]): RGB color for intensity 0.
        white (tuple[int, int, int]): RGB color for intensity 255 (or 1.0 for float32).
        mid (tuple[int, int, int] | None): Optional RGB color for the midpoint. `None` gives a
            2-color ramp.
        mid_value (int): Intensity (0-255) that maps to `mid`. Ignored when `mid is None`.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: Image with the trailing channel dimension
            expanded to 3.

    """
    if img.dtype == np.uint8:
        lut_float = _build_colorize_lut(black, white, mid, mid_value)
        # Floor-quantize to match `PIL.ImageOps.colorize` (integer floor division) bit-for-bit.
        lut_uint8 = np.clip(np.floor(lut_float), 0, 255).astype(np.uint8)
        lut_cv = lut_uint8.reshape(1, 256, 3)
        if img.ndim == 3:
            gray_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return cast("NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]", cv2.LUT(gray_3ch, lut_cv))
        out = np.empty((*img.shape[:-1], 3), dtype=np.uint8)
        flat_in = img.reshape(-1, *img.shape[-3:])
        flat_out = out.reshape(-1, *img.shape[-3:-1], 3)
        for frame_index in range(flat_in.shape[0]):
            flat_out[frame_index] = cv2.LUT(cv2.cvtColor(flat_in[frame_index], cv2.COLOR_GRAY2BGR), lut_cv)
        return out

    if img.dtype.type not in (np.float16, np.float32):
        raise ValueError(f"colorize: unsupported dtype {img.dtype}; expected uint8, float16 or float32")

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
    out = np.empty((*img.shape[:-1], 3), dtype=np.float32)
    gray_clipped = cast("NDArray[np.float16 | np.float32]", np.clip(img, 0.0, 1.0))
    for color_channel in range(3):
        out[..., color_channel : color_channel + 1] = np.interp(
            gray_clipped, anchor_intensity_float, anchor_color_float[:, color_channel]
        )
    return convert_dtype(out, dtype=img.dtype)


def downscale(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    scale: float,
    down_interpolation: int,
    up_interpolation: int,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Simulate resolution loss: downscale then upscale. down/up_interpolation control quality.
    Use for low-res or compression-style augmentation. Floating inputs are quantized to uint8 and restored.

    This function downscales and upscales an image using the specified interpolation methods.
    The downscaling and upscaling are performed using albucore.resize.

    Args:
        img (NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]): Input image as a numpy array.
        scale (float): The scale factor for the downscaling and upscaling.
        down_interpolation (int): The interpolation method for the downscaling.
        up_interpolation (int): The interpolation method for the upscaling.

    Returns:
        NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: The downscaled and upscaled image.

    """
    img, input_dtype = to_uint8_working_image(img)
    height, width = img.shape[:2]

    downscaled = fgeometric.resize(img, (int(height * scale), int(width * scale)), down_interpolation)
    result = fgeometric.resize(downscaled, (height, width), up_interpolation)
    return restore_uint8_working_image(cast("ImageUInt8", result), input_dtype)


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


__all__ = [
    "_build_colorize_lut",
    "_check_preconditions",
    "_equalize_cv",
    "_equalize_pil",
    "_handle_mask",
    "channel_shuffle",
    "clahe",
    "colorize",
    "downscale",
    "equalize",
    "evaluate_bez",
    "exposure_match_batch",
    "gamma_transform",
    "get_exposure_gains",
    "grayscale_to_multichannel",
    "image_compression",
    "invert",
    "iso_noise",
    "iso_noise_images",
    "iso_noise_volume",
    "linear_transformation_rgb",
    "move_tone_curve",
    "noop",
    "posterize",
    "shift_hsv",
    "shift_hsv_images",
    "solarize",
    "to_gray",
    "to_gray_average",
    "to_gray_desaturation",
    "to_gray_from_lab",
    "to_gray_max",
    "to_gray_pca",
    "to_gray_weighted_average",
    "volume_channel_shuffle",
]
