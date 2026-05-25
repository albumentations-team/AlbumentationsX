"""Noise generation and application functional helpers."""

from __future__ import annotations

from typing import Any, Literal, cast

from ._functional_shared import (
    MAX_VALUES_BY_DTYPE,
    MONO_CHANNEL_DIMENSIONS,
    ImageType,
    add_array,
    add_vector,
    add_weighted,
    clipped,
    cv2,
    float32_io,
    math,
    multiply_by_constant,
    np,
    power,
    preserve_channel_dim,
    random,
)


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

    cv2_seed = int(random_generator.integers(0, 2**16))
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
    blurred = cast(
        "ImageType",
        cv2.GaussianBlur(
            img,
            ksize=(kernel_size, kernel_size),
            sigmaX=sigma,
            sigmaY=sigma,
        ),
    )
    return add_weighted(img, 1.0 + alpha, blurred, -alpha)


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


SPARSE_SALT_AND_PEPPER_THRESHOLD = 0.16


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
            - 3D: (H, W, C) - RGB/multi-channel
            - 4D: (D, H, W, C) - volume with depth
        salt_mask (np.ndarray): Boolean mask indicating salt pixels (H, W)
        pepper_mask (np.ndarray): Boolean mask indicating pepper pixels (H, W)

    Returns:
        ImageType: The image with salt and pepper noise applied.

    """
    max_value = MAX_VALUES_BY_DTYPE[img.dtype]
    if img.shape[-1] > 1:
        noisy_fraction = (np.count_nonzero(salt_mask) + np.count_nonzero(pepper_mask)) / salt_mask.size
        if noisy_fraction <= SPARSE_SALT_AND_PEPPER_THRESHOLD:
            result = img.copy()
            result[..., salt_mask, :] = max_value
            result[..., pepper_mask, :] = 0
            return result

    salt_mask = salt_mask[..., None]
    pepper_mask = pepper_mask[..., None]
    return np.where(salt_mask, max_value, np.where(pepper_mask, 0, img))


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

INITIAL_GRID_SIZE = (3, 3)

__all__ = [
    "DIAMOND_KERNEL",
    "INITIAL_GRID_SIZE",
    "SPARSE_SALT_AND_PEPPER_THRESHOLD",
    "SQUARE_KERNEL",
    "_ENHANCE_IDENTITY",
    "_ENHANCE_KERNELS",
    "add_noise",
    "apply_salt_and_pepper",
    "generate_constant_noise_with_py_random",
    "generate_enhance_matrix",
    "generate_per_pixel_noise",
    "generate_shared_noise",
    "generate_spatial_noise",
    "get_safe_brightness_contrast_params",
    "sample_beta",
    "sample_gaussian",
    "sample_laplace",
    "sample_noise",
    "sample_uniform",
    "sharpen_gaussian",
    "shot_noise",
]
