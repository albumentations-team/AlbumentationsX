import cv2
_OPENCV_MAX_ROW_ELEMENTS = 2**31 - 1
_cv2_compare_scalar=cv2.compare
def random_dither_uint8(
    img: ImageUInt8,
    n_colors: int,
    noise_range: tuple[float, float],
    random_generator: np.random.Generator | None = None,
    random_noise: np.ndarray | None = None,
) -> ImageUInt8:
    """Random dithering for uint8: add noise then quantize with LUT. n_colors, noise_range,
    random_generator. Fast path for Dithering transform when input is uint8.

    Args:
        img (ImageUInt8): Input uint8 image with shape (H, W, C) in [0, 255] range.
        n_colors (int): Number of colors per channel after quantization.
        noise_range (tuple[float, float]): Range of noise to add (min_noise, max_noise) in [0, 1] range.
        random_generator (np.random.Generator | None): Random source when a pre-sampled noise map is unavailable.
        random_noise (np.ndarray | None): Pre-sampled int16 noise values for replayable transform execution.

    Returns:
        ImageUInt8: Dithered uint8 image in [0, 255] range.

    """
    # Add random noise (scale noise_range to uint8 range)
    if random_noise is None:
        if random_generator is None:
            msg = "random dithering requires random_noise or random_generator"
            raise ValueError(msg)
        noise_uint8 = random_generator.uniform(
            noise_range[0] * 255,
            noise_range[1] * 255,
            size=img.shape,
        ).astype(np.int16)
    else:
        noise_uint8 = random_noise

    if (
        n_colors == 2
        and noise_uint8.dtype == np.int16
        and noise_uint8.shape == img.shape
        and 0 < img.size <= _OPENCV_MAX_ROW_ELEMENTS
    ):
        noisy_int16 = np.add(img, noise_uint8, dtype=np.int16).reshape(1, -1)
        return _cv2_compare_scalar(noisy_int16, 127, cv2.CMP_GT).reshape(img.shape)

    # Add noise and clip to valid range
    noisy = np.clip(img.astype(np.int16) + noise_uint8, 0, 255).astype(np.uint8)

    # Quantize using LUT for maximum performance
    if n_colors == 2:
        lut_binary = np.zeros(256, dtype=np.uint8)
        lut_binary[128:] = 255
        return apply_uint8_lut(noisy, lut_binary)

    lut = (np.round(np.arange(256) * (n_colors - 1) / 255) / (n_colors - 1) * 255).astype(np.uint8)

    return apply_uint8_lut(noisy, lut)

def random_dither(
    img: ImageFloat32,
    n_colors: int,
    noise_range: tuple[float, float],
    random_generator: np.random.Generator | None = None,
    random_noise: np.ndarray | None = None,
) -> ImageFloat32:
    """Random dithering for float32: add noise then quantize to n_colors. noise_range and
    random_generator control the dither. Float32 input and output in [0, 1].

    Args:
        img (ImageFloat32): Input float32 image with shape (H, W, C) in [0, 1] range.
        n_colors (int): Number of colors per channel after quantization.
        noise_range (tuple[float, float]): Range of noise to add (min_noise, max_noise).
        random_generator (np.random.Generator | None): Random source when a pre-sampled noise map is unavailable.
        random_noise (np.ndarray | None): Pre-sampled noise values for replayable transform execution.

    Returns:
        ImageFloat32: Dithered float32 image in [0, 1] range.

    """
    # Add random noise
    if random_noise is None:
        if random_generator is None:
            msg = "random dithering requires random_noise or random_generator"
            raise ValueError(msg)
        noisy = random_generator.uniform(noise_range[0], noise_range[1], size=img.shape)
    else:
        noisy = random_noise.copy()
    np.add(noisy, img, out=noisy)
    if n_colors == 2:
        # Clipping does not affect the binary decision. OpenCV uses >, so use
        # the predecessor of 0.5 to preserve >=, including exact ties.
        if 4_096 <= noisy.size <= _OPENCV_MAX_ROW_ELEMENTS and noisy.dtype in (np.float32, np.float64):
            threshold = np.nextafter(noisy.dtype.type(0.5), noisy.dtype.type(-np.inf))
            flat = noisy.reshape(1, -1)
            cv2.threshold(flat, float(threshold), 1, cv2.THRESH_BINARY, dst=flat)
            return flat.reshape(noisy.shape).astype(np.float32, copy=False)
        return (noisy >= 0.5).astype(np.float32)

    np.clip(noisy, 0, 1, out=noisy)

    # Vectorized quantization for float32
    np.multiply(noisy, n_colors - 1, out=noisy)
    np.round(noisy, out=noisy)
    np.divide(noisy, n_colors - 1, out=noisy)
    return noisy.astype(np.float32)