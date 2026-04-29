"""Sharpness, convolution, and enhancement functional helpers."""

from __future__ import annotations

from typing import Literal

from ._functional_shared import (
    NUM_MULTI_CHANNEL_DIMENSIONS,
    ImageType,
    clipped,
    cv2,
    lru_cache,
    multiply_by_constant,
    np,
    preserve_channel_dim,
    remap,
    uint8_io,
)


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

    """
    result = np.empty_like(images)
    for i in range(images.shape[0]):
        result[i] = unsharp_mask(images[i], ksize, sigma, alpha, threshold)
    return result


def unsharp_mask(
    image: np.ndarray,
    ksize: int,
    sigma: float,
    alpha: float,
    threshold: int,
) -> np.ndarray:
    """Apply unsharp mask to a single image using one blur, weighted residual sharpening,
    and threshold-gated replacement for faster Pillow-style sharpening.

    Args:
        image (np.ndarray): Single image, shape (H, W, C) or (H, W).
        ksize (int): The kernel size for Gaussian blur.
        sigma (float): The sigma value for Gaussian blur.
        alpha (float): The alpha value for the unsharp mask.
        threshold (int): The threshold value for the unsharp mask.

    Returns:
        np.ndarray: Unsharp masked image with same shape and dtype as input.

    """
    if image.ndim == NUM_MULTI_CHANNEL_DIMENSIONS and image.shape[2] == 1:
        return unsharp_mask(image[..., 0], ksize, sigma, alpha, threshold)[..., np.newaxis]

    ksize_tuple = (ksize, ksize)
    blurred = cv2.GaussianBlur(image, ksize_tuple, sigmaX=sigma)

    sharpened = cv2.addWeighted(image, 1.0 + alpha, blurred, -alpha, 0)
    if image.dtype == np.float32:
        np.clip(sharpened, 0, 1, out=sharpened)

    if threshold <= 0:
        return sharpened

    diff = cv2.absdiff(image, blurred)
    threshold_value = threshold / 255.0 if image.dtype == np.float32 else threshold
    mask = diff > threshold_value

    return np.where(mask, sharpened, image).astype(image.dtype, copy=False)


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
    if image.ndim == 4:
        if drop_mask.ndim == 2:
            drop_mask = drop_mask[None, :, :, None]
        elif drop_mask.ndim == 3:
            drop_mask = drop_mask[None, ...]
    elif drop_mask.ndim == image.ndim - 1:
        drop_mask = drop_mask[..., None]
    return np.where(drop_mask, drop_values, image)


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
    return remap(
        channel,
        map_x,
        map_y,
        interpolation=interpolation,
        border_mode=cv2.BORDER_REPLICATE,
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


@lru_cache(maxsize=128)
def _get_planckian_coeffs(
    mode: Literal["blackbody", "cied"],
    temperature: int,
) -> tuple[float, float]:
    min_temp = min(PLANCKIAN_COEFFS[mode].keys())
    max_temp = max(PLANCKIAN_COEFFS[mode].keys())
    temperature = int(np.clip(temperature, min_temp, max_temp))

    step = 500
    t_left = max((temperature // step) * step, min_temp)
    t_right = min((temperature // step + 1) * step, max_temp)

    if t_left == t_right:
        coeffs = PLANCKIAN_COEFFS[mode][t_left]
    else:
        w_right = (temperature - t_left) / (t_right - t_left)
        w_left = 1 - w_right
        left_coeffs = PLANCKIAN_COEFFS[mode][t_left]
        right_coeffs = PLANCKIAN_COEFFS[mode][t_right]
        coeffs = [
            w_left * left_coeff + w_right * right_coeff
            for left_coeff, right_coeff in zip(left_coeffs, right_coeffs, strict=True)
        ]

    return coeffs[0] / coeffs[1], coeffs[2] / coeffs[1]


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
    red_multiplier, blue_multiplier = _get_planckian_coeffs(mode, temperature)

    img[..., 0] = multiply_by_constant(
        img[..., 0],
        red_multiplier,
        inplace=True,
    )
    img[..., 2] = multiply_by_constant(
        img[..., 2],
        blue_multiplier,
        inplace=True,
    )

    return img


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


__all__ = [
    "PLANCKIAN_COEFFS",
    "_distort_channel",
    "_get_planckian_coeffs",
    "chromatic_aberration",
    "convolve",
    "pixel_dropout",
    "planckian_jitter",
    "separable_convolve",
    "unsharp_mask",
    "unsharp_mask_images",
]
