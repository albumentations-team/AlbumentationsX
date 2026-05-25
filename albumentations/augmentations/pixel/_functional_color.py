"""Color, tone, histogram, grayscale, and color conversion functional helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

from ._functional_shared import (
    MAX_VALUES_BY_DTYPE,
    MULTICHANNEL_LUT_LARGE_IMAGE_PIXELS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    NUM_RGB_CHANNELS,
    PCA,
    ImageType,
    ImageUInt8,
    add_array,
    add_constant,
    apply_multichannel_lut,
    clip,
    clipped,
    cv2,
    fgeometric,
    float32_io,
    from_float,
    get_num_channels,
    is_grayscale_image,
    is_rgb_image,
    mean,
    non_rgb_error,
    normalize_per_image,
    np,
    preserve_channel_dim,
    reduce_sum,
    reshape_ndhwc_channel,
    reshape_xhwc_channel,
    restore_ndhwc_channel,
    restore_xhwc_channel,
    sz_lut,
    uint8_io,
    warn,
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
        img = cast("ImageType", cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

    img = cast("ImageType", cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    hue, sat, val = (cast("ImageType", channel) for channel in cv2.split(img))

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

    img = cast("ImageType", cv2.merge((hue, sat, val)))
    img = cast("ImageType", cv2.cvtColor(img, cv2.COLOR_HSV2RGB))

    return cast("ImageType", cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)) if is_gray else img


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
        uint8_img = cast("ImageUInt8", img)
        indices = np.arange(int(max_val) + 1, dtype=dtype)
        thresh_val = threshold * max_val
        lut = cast("ImageUInt8", np.where(indices >= thresh_val, max_val - indices, indices).astype(dtype))
        prev_shape = img.shape
        result = sz_lut(uint8_img, lut, inplace=False)
        return result if len(prev_shape) == result.ndim else cast("ImageType", np.expand_dims(result, -1))
    return cast("ImageType", np.where(img >= threshold, max_val - img, img))


@uint8_io
@clipped
def posterize(img: ImageType, bits: Literal[1, 2, 3, 4, 5, 6, 7] | list[Literal[1, 2, 3, 4, 5, 6, 7]]) -> ImageType:
    """Reduce bit depth by keeping only the highest N bits per channel. bits: 1-7 or list per
    channel; implemented via bitwise masking on uint8. uint8 I/O, clipped.

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
    bits_array = np.asarray(bits, dtype=np.uint8)
    uint8_img = cast("ImageUInt8", img)

    if bits_array.ndim == 0 or bits_array.size == 1:
        mask = ~np.uint8(2 ** (8 - bits_array.item()) - 1)
        return cast("ImageType", uint8_img & mask)

    result_img = np.empty_like(uint8_img)
    for i, channel_bits in enumerate(bits_array):
        mask = ~np.uint8(2 ** (8 - int(channel_bits)) - 1)
        np.bitwise_and(uint8_img[..., i], mask, out=result_img[..., i])

    return cast("ImageType", result_img)


def _equalize_pil(img: ImageType, mask: np.ndarray | None = None) -> ImageType:
    histogram = cv2.calcHist([img], [0], mask, [256], (0, 256)).ravel()
    h = histogram[histogram > 0]

    if len(h) <= 1:
        return img.copy()

    step = int(reduce_sum(h[:-1])) // 255
    if not step:
        return img.copy()

    lut = np.minimum((np.cumsum(histogram) + step // 2) // step, 255).astype(np.uint8)

    return sz_lut(cast("ImageUInt8", img), lut, inplace=True)


def _equalize_cv(img: ImageType, mask: np.ndarray | None = None) -> ImageType:
    if mask is None:
        return cast("ImageType", cv2.equalizeHist(img))

    histogram = cv2.calcHist([img], [0], mask, [256], (0, 256)).ravel()
    lut = _create_equalize_cv_lut(histogram)
    if lut is None:
        return img

    return sz_lut(img, lut, inplace=True)


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


def _equalize_cv_multichannel_lut(img: ImageType) -> ImageType:
    """Apply OpenCV-style equalization with one multichannel LUT pass for large
    RGB images and multispectral inputs where per-channel assignment is slower.
    """
    luts = []
    for channel_idx in range(get_num_channels(img)):
        channel = img[..., channel_idx]
        histogram = cv2.calcHist([channel], [0], None, [256], (0, 256)).ravel()
        lut = _create_equalize_cv_lut(histogram)
        luts.append(np.arange(256, dtype=np.uint8) if lut is None else lut)

    return apply_multichannel_lut(img, np.stack(luts), get_num_channels(img))


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

    if mask is None and by_channels and mode == "cv" and is_rgb_image(img):
        if img.shape[0] * img.shape[1] >= MULTICHANNEL_LUT_LARGE_IMAGE_PIXELS:
            return _equalize_cv_multichannel_lut(img)
        channels = cv2.split(img)
        return cast("ImageType", cv2.merge([cv2.equalizeHist(channel) for channel in channels]))

    if mask is None and by_channels and mode == "cv" and get_num_channels(img) > NUM_RGB_CHANNELS:
        return _equalize_cv_multichannel_lut(img)

    if not by_channels:
        result_img = cast("ImageType", cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb))
        result_img[..., 0] = function(result_img[..., 0], _handle_mask(mask))
        return cast("ImageType", cv2.cvtColor(result_img, cv2.COLOR_YCrCb2RGB))

    result_img = np.empty_like(img)
    for i in range(get_num_channels(img)):
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
        low_y_scalar = cast("Any", low_y)
        high_y_scalar = cast("Any", high_y)
        lut = cast(
            "ImageUInt8",
            clip(np.rint(evaluate_bez(float(low_y_scalar), float(high_y_scalar))), np.dtype(np.uint8), inplace=False),
        )
        return sz_lut(cast("ImageUInt8", img), lut, inplace=False)

    if isinstance(low_y, np.ndarray) and isinstance(high_y, np.ndarray):
        luts = cast(
            "ImageUInt8",
            clip(
                np.rint(evaluate_bez(low_y, high_y).T),
                np.dtype(np.uint8),
                inplace=False,
            ),
        )
        return apply_multichannel_lut(img, luts, num_channels)

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
        return cast("ImageType", cv2.transform(img, transformation_matrix))
    if img.ndim == 4:
        transformed, original_shape = reshape_xhwc_channel(img)
        transformed = cast("ImageType", cv2.transform(transformed, transformation_matrix))
        return cast("ImageType", restore_xhwc_channel(transformed, original_shape))
    if img.ndim == 5:
        transformed, original_shape = reshape_ndhwc_channel(img)
        transformed = cast("ImageType", cv2.transform(transformed, transformation_matrix))
        return cast("ImageType", restore_ndhwc_channel(transformed, original_shape))
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
        return cast("ImageType", clahe_mat.apply(img))

    img_lab = cast("ImageType", cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
    img_lab[:, :, 0] = clahe_mat.apply(img_lab[:, :, 0])

    return cast("ImageType", cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB))


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
        _, encoded_img = cv2.imencode(image_type, src_img, (quality_flag, quality))
        return cast("np.ndarray", cv2.imdecode(encoded_img, read_mode))

    if num_channels == 1:
        # Grayscale image
        decoded = encode_decode(img, cv2.IMREAD_GRAYSCALE)
        return cast("ImageType", decoded[..., np.newaxis])  # Add channel dimension back

    if num_channels in (2, NUM_RGB_CHANNELS):
        # 2 channels: pad to 3, or 3 (RGB) channels
        padded_img = (
            cast("ImageType", np.pad(img, ((0, 0), (0, 0), (0, 1)), mode="constant")) if num_channels == 2 else img
        )
        decoded_bgr = encode_decode(padded_img, cv2.IMREAD_UNCHANGED)
        return cast("ImageType", decoded_bgr[..., :num_channels])  # Return only the required number of channels

    # More than 3 channels
    bgr = img[..., :NUM_RGB_CHANNELS]
    decoded_bgr = encode_decode(bgr, cv2.IMREAD_UNCHANGED)

    # Process additional channels
    extra_channels = [
        encode_decode(img[..., i], cv2.IMREAD_GRAYSCALE)[..., np.newaxis] for i in range(NUM_RGB_CHANNELS, num_channels)
    ]
    return cast("ImageType", np.dstack([decoded_bgr, *extra_channels]))


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
    img = cast("ImageType", np.ascontiguousarray(img))
    output = np.empty(img.shape, dtype=img.dtype)
    from_to = []
    for i, j in enumerate(channels_shuffled):
        from_to.extend([j, i])  # Use [src, dst]
    cv2.mixChannels([img], [output], from_to)
    return cast("ImageType", output)


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
    hls = cast("ImageType", cv2.cvtColor(image, cv2.COLOR_RGB2HLS))
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
        cast("ImageType", (luminance_noise * intensity * (1.0 - hls[..., 1])).astype(np.float32, copy=False)),
    )

    return cast("np.ndarray", cv2.cvtColor(hls, cv2.COLOR_HLS2RGB))


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
        return cast("ImageType", cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    if img.ndim == 4:
        im, original_shape = reshape_xhwc_channel(img)
        im = cast("ImageType", cv2.cvtColor(im, cv2.COLOR_RGB2GRAY))

        new_shape = (*original_shape[:-1], 1)

        return cast("ImageType", restore_xhwc_channel(im, new_shape))

    if img.ndim == 5:
        img, original_shape = reshape_ndhwc_channel(img)
        img = cast("ImageType", cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

        new_shape = (*original_shape[:-1], 1)

        return cast("ImageType", restore_ndhwc_channel(img, new_shape))

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
        return cast("ImageType", cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0])
    if img.ndim == 4:
        im, original_shape = reshape_xhwc_channel(img)
        im = cast("ImageType", cv2.cvtColor(im, cv2.COLOR_RGB2LAB)[..., 0])

        new_shape = (*original_shape[:-1], 1)

        return cast("ImageType", restore_xhwc_channel(im, new_shape))

    if img.ndim == 5:
        img, original_shape = reshape_ndhwc_channel(img)
        img = cast("ImageType", cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0])

        new_shape = (*original_shape[:-1], 1)

        return cast("ImageType", restore_ndhwc_channel(img, new_shape))

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
        if img.shape[-1] == 1:
            return img[..., 0]
        if img.ndim > NUM_MULTI_CHANNEL_DIMENSIONS:
            ch_max = np.max(img, axis=-1).astype(np.uint16)
            ch_min = np.min(img, axis=-1).astype(np.uint16)
            return ((ch_max + ch_min) >> 1).astype(np.uint8)
        channels = cv2.split(img)
        ch_max = channels[0]
        ch_min = channels[0]
        for channel in channels[1:]:
            ch_max = cv2.max(ch_max, channel)
            ch_min = cv2.min(ch_min, channel)
        channel_sum = cast("np.ndarray", cv2.add(ch_max, ch_min, dtype=cv2.CV_16U))
        return (channel_sum >> 1).astype(np.uint8)
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
    return cast("ImageType", np.asarray(mean(img, axis=-1)).astype(img.dtype))


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
            return cast("ImageType", cv2.LUT(gray_3ch, lut_cv))
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
    "gamma_transform",
    "grayscale_to_multichannel",
    "image_compression",
    "invert",
    "iso_noise",
    "iso_noise_images",
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
    "volumes_channel_shuffle",
]
