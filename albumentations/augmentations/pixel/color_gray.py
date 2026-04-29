"""Grayscale, RGB conversion, colorization, sepia, and PCA color transforms."""

from typing import Annotated, Any, Literal

from ._color_shared import (
    NUM_RGB_CHANNELS,
    AfterValidator,
    BaseTransformInitSchema,
    Field,
    ImageOnlyTransform,
    ImageType,
    VolumeType,
    check_range_bounds,
    fpixel,
    get_num_channels,
    is_grayscale_image,
    is_rgb_image,
    nondecreasing,
    np,
    warnings,
)
from .color_advanced import (
    ColorRange,
)


class ToGray(ImageOnlyTransform):
    """Convert to grayscale (weighted by channel weights). Optionally replicate to keep
    shape. Useful for grayscale training or channel reduction.

    This transform first converts a color image to a single-channel grayscale image using various methods,
    then replicates the grayscale channel if num_output_channels is greater than 1.

    Args:
        num_output_channels (int): The number of channels in the output image. If greater than 1,
            the grayscale channel will be replicated. Default: 3.
        method (Literal['weighted_average', 'from_lab', 'desaturation', 'average', 'max', 'pca']):
            The method used for grayscale conversion:
            - "weighted_average": Uses a weighted sum of RGB channels (0.299R + 0.587G + 0.114B).
              Works only with 3-channel images. Provides realistic results based on human perception.
            - "from_lab": Extracts the L channel from the LAB color space.
              Works only with 3-channel images. Gives perceptually uniform results.
            - "desaturation": Averages the maximum and minimum values across channels.
              Works with any number of channels. Fast but may not preserve perceived brightness well.
            - "average": Simple average of all channels.
              Works with any number of channels. Fast but may not give realistic results.
            - "max": Takes the maximum value across all channels.
              Works with any number of channels. Tends to produce brighter results.
            - "pca": Applies Principal Component Analysis to reduce channels.
              Works with any number of channels. Can preserve more information but is computationally intensive.
        p (float): Probability of applying the transform. Default: 0.5.

    Raises:
        TypeError: If the input image doesn't have 3 channels for methods that require it.

    Note:
        - The transform first converts the input image to single-channel grayscale, then replicates
          this channel if num_output_channels > 1.
        - "weighted_average" and "from_lab" are typically used in image processing and computer vision
          applications where accurate representation of human perception is important.
        - "desaturation" and "average" are often used in simple image manipulation tools or when
          computational speed is a priority.
        - "max" method can be useful in scenarios where preserving bright features is important,
          such as in some medical imaging applications.
        - "pca" might be used in advanced image analysis tasks or when dealing with hyperspectral images.

    Image types:
        uint8, float32

    Returns:
        np.ndarray: Grayscale image with the specified number of channels.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create a sample color image with distinct RGB values
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> # Red square in top-left
        >>> image[10:40, 10:40, 0] = 200
        >>> # Green square in top-right
        >>> image[10:40, 60:90, 1] = 200
        >>> # Blue square in bottom-left
        >>> image[60:90, 10:40, 2] = 200
        >>> # Yellow square in bottom-right (Red + Green)
        >>> image[60:90, 60:90, 0] = 200
        >>> image[60:90, 60:90, 1] = 200
        >>>
        >>> # Example 1: Default conversion (weighted average, 3 channels)
        >>> transform = A.ToGray(p=1.0)
        >>> result = transform(image=image)
        >>> gray_image = result['image']
        >>> # Output has 3 duplicate channels with values based on RGB perception weights
        >>> # R=0.299, G=0.587, B=0.114
        >>> assert gray_image.shape == (100, 100, 3)
        >>> assert np.allclose(gray_image[:, :, 0], gray_image[:, :, 1])
        >>> assert np.allclose(gray_image[:, :, 1], gray_image[:, :, 2])
        >>>
        >>> # Example 2: Single-channel output
        >>> transform = A.ToGray(num_output_channels=1, p=1.0)
        >>> result = transform(image=image)
        >>> gray_image = result['image']
        >>> assert gray_image.shape == (100, 100, 1)
        >>>
        >>> # Example 3: Using different conversion methods
        >>> # "desaturation" method (min+max)/2
        >>> transform_desaturate = A.ToGray(
        ...     method="desaturation",
        ...     p=1.0
        ... )
        >>> result = transform_desaturate(image=image)
        >>> gray_desaturate = result['image']
        >>>
        >>> # "from_lab" method (using L channel from LAB colorspace)
        >>> transform_lab = A.ToGray(
        ...     method="from_lab",
        ...     p=1.0
        >>> )
        >>> result = transform_lab(image=image)
        >>> gray_lab = result['image']
        >>>
        >>> # "average" method (simple average of channels)
        >>> transform_avg = A.ToGray(
        ...     method="average",
        ...     p=1.0
        >>> )
        >>> result = transform_avg(image=image)
        >>> gray_avg = result['image']
        >>>
        >>> # "max" method (takes max value across channels)
        >>> transform_max = A.ToGray(
        ...     method="max",
        ...     p=1.0
        >>> )
        >>> result = transform_max(image=image)
        >>> gray_max = result['image']
        >>>
        >>> # Example 4: Using grayscale in an augmentation pipeline
        >>> pipeline = A.Compose([
        ...     A.ToGray(p=0.5),           # 50% chance of grayscale conversion
        ...     A.RandomBrightnessContrast(p=1.0)  # Always apply brightness/contrast
        ... ])
        >>> result = pipeline(image=image)
        >>> augmented_image = result['image']  # May be grayscale or color
        >>>
        >>> # Example 5: Converting float32 image
        >>> float_image = image.astype(np.float32) / 255.0  # Range [0, 1]
        >>> transform = A.ToGray(p=1.0)
        >>> result = transform(image=float_image)
        >>> gray_float_image = result['image']
        >>> assert gray_float_image.dtype == np.float32
        >>> assert gray_float_image.max() <= 1.0

    """

    class InitSchema(BaseTransformInitSchema):
        num_output_channels: int = Field(
            description="The number of output channels.",
            ge=1,
        )
        method: Literal[
            "weighted_average",
            "from_lab",
            "desaturation",
            "average",
            "max",
            "pca",
        ]

    def __init__(
        self,
        num_output_channels: int = 3,
        method: Literal[
            "weighted_average",
            "from_lab",
            "desaturation",
            "average",
            "max",
            "pca",
        ] = "weighted_average",
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.num_output_channels = num_output_channels
        self.method = method

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        if is_grayscale_image(img):
            warnings.warn("The image is already gray.", stacklevel=2)
            return img

        num_channels = get_num_channels(img)

        if num_channels != NUM_RGB_CHANNELS and self.method not in {
            "desaturation",
            "average",
            "max",
            "pca",
        }:
            msg = "ToGray transformation expects 3-channel images."
            raise TypeError(msg)

        return fpixel.to_gray(img, self.num_output_channels, self.method)

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        # Check if images are already grayscale by checking number of channels
        if images.shape[-1] == 1:
            warnings.warn("The image is already gray.", stacklevel=2)
            return images

        return fpixel.to_gray(images, self.num_output_channels, self.method)

    def apply_to_volumes(self, volumes: VolumeType, **params: Any) -> VolumeType:
        # Check if volumes are already grayscale by checking number of channels
        if volumes.shape[-1] == 1:
            warnings.warn("The volumes are already gray.", stacklevel=2)
            return volumes

        return fpixel.to_gray(volumes, self.num_output_channels, self.method)


class ToRGB(ImageOnlyTransform):
    """Convert grayscale image to RGB by replicating the single channel to three. No color
    information added; use when a model expects 3-channel input.

    Args:
        num_output_channels (int): The number of channels in the output image. Default: 3.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        1

    Note:
        - For single-channel (grayscale) images, the channel is replicated to create an RGB image.
        - If the input is already a 3-channel RGB image, it is returned unchanged.
        - This transform does not change the data type of the image (e.g., uint8 remains uint8).

    Raises:
        TypeError: If the input image has more than 1 channel.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Convert a grayscale image to RGB
        >>> transform = A.Compose([A.ToRGB(p=1.0)])
        >>> grayscale_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> rgb_image = transform(image=grayscale_image)['image']
        >>> assert rgb_image.shape == (100, 100, 3)

    """

    class InitSchema(BaseTransformInitSchema):
        num_output_channels: int = Field(ge=1)

    def __init__(
        self,
        num_output_channels: int = 3,
        p: float = 1.0,
    ):
        super().__init__(p=p)

        self.num_output_channels = num_output_channels

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        if is_rgb_image(img):
            warnings.warn("The image is already an RGB.", stacklevel=2)
            return np.ascontiguousarray(img)
        if not is_grayscale_image(img):
            msg = "ToRGB transformation expects images with the number of channels equal to 1."
            raise TypeError(msg)

        return fpixel.grayscale_to_multichannel(
            img,
            num_output_channels=self.num_output_channels,
        )

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self.apply(images, **params)

    def apply_to_volumes(self, volumes: VolumeType, **params: Any) -> VolumeType:
        return self.apply(volumes, **params)


def _validate_color_range(rng: ColorRange) -> ColorRange:
    """Validate a Colorize anchor range: each entry is an RGB triple in [0, 255], and the lower
    bound must not exceed the upper bound on any channel.
    """
    lo, hi = rng
    if len(lo) != NUM_RGB_CHANNELS or len(hi) != NUM_RGB_CHANNELS:
        raise ValueError(f"Color range must be (rgb_low, rgb_high) with 3 channels each, got {rng}")
    for channel in range(NUM_RGB_CHANNELS):
        if not (0 <= lo[channel] <= 255 and 0 <= hi[channel] <= 255):
            raise ValueError(f"Color range entries must be in [0, 255], got {rng}")
        if lo[channel] > hi[channel]:
            raise ValueError(f"Color range lower bound must be <= upper bound per channel, got {rng}")
    return rng


class Colorize(ImageOnlyTransform):
    """Map a single-channel grayscale image to a 2- or 3-color RGB gradient with per-call sampled
    anchor colors (Pillow `ImageOps.colorize` style).

    Intensity acts as a coordinate along a sampled color ramp:

    - `0` maps to a sample from `black_range`
    - `255` (or `1.0` for float32) maps to a sample from `white_range`
    - if `mid_range` is set, intensity sampled from `mid_value_range` maps to a sample from
      `mid_range` and the ramp becomes piecewise linear

    Each anchor range is given as `(low_rgb, high_rgb)` and sampled per-channel uniformly on
    every call. Pass identical low/high tuples to fix a color
    (e.g. `black_range=((0, 0, 255), (0, 0, 255))`). Anchors are always specified in 0-255 RGB;
    for float32 inputs they are rescaled to [0, 1] internally.

    Args:
        black_range (tuple[tuple[int, int, int], tuple[int, int, int]]): Inclusive per-channel
            range from which the dark anchor is sampled. Default: ((0, 0, 0), (0, 0, 0)).
        white_range (tuple[tuple[int, int, int], tuple[int, int, int]]): Inclusive per-channel
            range from which the bright anchor is sampled.
            Default: ((255, 255, 255), (255, 255, 255)).
        mid_range (tuple[tuple[int, int, int], tuple[int, int, int]] | None): Optional inclusive
            range from which the midpoint anchor is sampled. `None` disables 3-color mode.
            Default: None.
        mid_value_range (tuple[int, int]): Inclusive intensity range (each in 1-254) from which
            the midpoint position is sampled. Ignored when `mid is None`. Default: (127, 127).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        1

    Note:
        - Input must be single-channel; multi-channel input is a no-op with a warning.
        - Interpolation is linear in RGB space.
        - For uint8 inputs the per-call mapping is a (256, 3) LUT applied via `cv2.LUT`;
          for float32 inputs `np.interp` is used per channel.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8)
        >>>
        >>> # Fixed blue -> yellow ramp (low == high)
        >>> fixed = A.Compose([A.Colorize(
        ...     black_range=((0, 0, 255), (0, 0, 255)),
        ...     white_range=((255, 255, 0), (255, 255, 0)),
        ...     p=1.0,
        ... )])
        >>> assert fixed(image=image)["image"].shape == (100, 100, 3)
        >>>
        >>> # Random thermal-ish ramp with random midpoint position
        >>> random_thermal = A.Compose([A.Colorize(
        ...     black_range=((0, 0, 64), (32, 0, 192)),
        ...     mid_range=((96, 0, 96), (160, 64, 160)),
        ...     white_range=((220, 160, 0), (255, 220, 32)),
        ...     mid_value_range=(96, 160),
        ...     p=1.0,
        ... )])
        >>> assert random_thermal(image=image)["image"].shape == (100, 100, 3)

    """

    class InitSchema(BaseTransformInitSchema):
        black_range: Annotated[ColorRange, AfterValidator(_validate_color_range)]
        white_range: Annotated[ColorRange, AfterValidator(_validate_color_range)]
        mid_range: Annotated[ColorRange, AfterValidator(_validate_color_range)] | None
        mid_value_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, 254)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        black_range: tuple[tuple[int, int, int], tuple[int, int, int]] = ((0, 0, 0), (0, 0, 0)),
        white_range: tuple[tuple[int, int, int], tuple[int, int, int]] = (
            (255, 255, 255),
            (255, 255, 255),
        ),
        mid_range: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
        mid_value_range: tuple[int, int] = (127, 127),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.black_range = black_range
        self.white_range = white_range
        self.mid_range = mid_range
        self.mid_value_range = mid_value_range

    def _sample_color(self, color_range: ColorRange) -> tuple[int, int, int]:
        lo, hi = color_range
        return (
            self.py_random.randint(lo[0], hi[0]),
            self.py_random.randint(lo[1], hi[1]),
            self.py_random.randint(lo[2], hi[2]),
        )

    def get_params(self) -> dict[str, Any]:
        black_color = self._sample_color(self.black_range)
        white_color = self._sample_color(self.white_range)
        mid_color = self._sample_color(self.mid_range) if self.mid_range is not None else None
        # `fpixel.colorize(..., mid=None, mid_value=...)` ignores `mid_value`, so don't waste
        # an RNG draw or report a phantom sampled value when no midpoint anchor is configured.
        if mid_color is not None:
            mid_value = self.py_random.randint(*self.mid_value_range)
            applied_mid_value: int | tuple[int, int] = mid_value
        else:
            mid_value = self.mid_value_range[0]
            applied_mid_value = self.mid_value_range

        # Resolve ranges to sampled scalars in applied_config (per BasicTransform contract).
        self.applied_config["black_range"] = black_color
        self.applied_config["white_range"] = white_color
        self.applied_config["mid_range"] = mid_color
        self.applied_config["mid_value_range"] = applied_mid_value

        return {
            "black_color": black_color,
            "white_color": white_color,
            "mid_color": mid_color,
            "mid_value": mid_value,
        }

    def apply(
        self,
        img: ImageType,
        black_color: tuple[int, int, int],
        white_color: tuple[int, int, int],
        mid_color: tuple[int, int, int] | None,
        mid_value: int,
        **params: Any,
    ) -> ImageType:
        if get_num_channels(img) != 1:
            warnings.warn(
                "Colorize expects a single-channel image; got a multi-channel image, returning it unchanged.",
                stacklevel=2,
            )
            return img
        return fpixel.colorize(img, black_color, white_color, mid_color, mid_value)

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self.apply(images, **params)

    def apply_to_volumes(self, volumes: VolumeType, **params: Any) -> VolumeType:
        return self.apply(volumes, **params)


class ToSepia(ImageOnlyTransform):
    """Apply sepia (brownish vintage) filter via fixed color matrix. Optional alpha for
    blending with original. Good for style or temporal variation in datasets.

    This transform converts a color image to a sepia tone, giving it a warm, brownish tint
    that is reminiscent of old photographs. The sepia effect is achieved by applying a
    specific color transformation matrix to the RGB channels of the input image.
    For grayscale images, the transform is a no-op and returns the original image.

    Args:
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        1,3

    Note:
        - The sepia effect only works with RGB images (3 channels). For grayscale images,
          the original image is returned unchanged since the sepia transformation would
          have no visible effect when R=G=B.
        - The sepia effect is created using a fixed color transformation matrix:
          [[0.393, 0.769, 0.189],
           [0.349, 0.686, 0.168],
           [0.272, 0.534, 0.131]]
        - The output image will have the same data type as the input image.
        - For float32 images, ensure the input values are in the range [0, 1].

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        # Apply sepia effect to a uint8 RGB image
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.ToSepia(p=1.0)
        >>> sepia_image = transform(image=image)['image']
        >>> assert sepia_image.shape == image.shape
        >>> assert sepia_image.dtype == np.uint8
        >>>
        # Apply sepia effect to a float32 RGB image
        >>> image = np.random.rand(100, 100, 3).astype(np.float32)
        >>> transform = A.ToSepia(p=1.0)
        >>> sepia_image = transform(image=image)['image']
        >>> assert sepia_image.shape == image.shape
        >>> assert sepia_image.dtype == np.float32
        >>> assert 0 <= sepia_image.min() <= sepia_image.max() <= 1.0
        >>>
        # No effect on grayscale images
        >>> gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> transform = A.ToSepia(p=1.0)
        >>> result = transform(image=gray_image)['image']
        >>> assert np.array_equal(result, gray_image)

    Mathematical Formulation:
        Given an input pixel [R, G, B], the sepia tone is calculated as:
        R_sepia = 0.393*R + 0.769*G + 0.189*B
        G_sepia = 0.349*R + 0.686*G + 0.168*B
        B_sepia = 0.272*R + 0.534*G + 0.131*B

        For grayscale images where R=G=B, this transformation would result in a simple
        scaling of the original value, so we skip it.

        The output values are clipped to the valid range for the image's data type.

    See Also:
        ToGray: For converting images to grayscale instead of sepia.

    """

    def __init__(self, p: float = 0.5):
        super().__init__(p=p)
        self.sepia_transformation_matrix = np.array(
            [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]],
        )

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        if is_grayscale_image(img):
            return img

        if not is_rgb_image(img):
            msg = "ToSepia transformation expects 1 or 3-channel images."
            raise TypeError(msg)
        return fpixel.linear_transformation_rgb(img, self.sepia_transformation_matrix)

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self.apply(images, **params)

    def apply_to_volumes(self, volumes: VolumeType, **params: Any) -> VolumeType:
        return self.apply(volumes, **params)


class FancyPCA(ImageOnlyTransform):
    """Add color variation via PCA on RGB: perturb components by alpha_std. Simulates natural
    lighting variation (ImageNet-style). Good for object recognition.

    This augmentation technique applies PCA (Principal Component Analysis) to the image's color channels,
    then adds multiples of the principal components to the image, with magnitudes proportional to the
    corresponding eigenvalues times a random variable drawn from a Gaussian with mean 0 and standard
    deviation 'alpha'.

    Args:
        alpha (float): Standard deviation of the Gaussian distribution used to generate
            random noise for each principal component. Default: 0.1.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        any

    Note:
        - This augmentation is particularly effective for RGB images but can work with any number of channels.
        - For grayscale images, it applies a simplified version of the augmentation.
        - The transform preserves the mean of the image while adjusting the color/intensity variation.
        - This implementation is based on the paper by Krizhevsky et al. and is similar to the one used
          in the original AlexNet paper.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.FancyPCA(alpha=0.1, p=1.0)
        >>> result = transform(image=image)
        >>> augmented_image = result["image"]

    References:
        ImageNet Classification with Deep Convolutional Neural Networks: In Advances in Neural Information
        Processing Systems (Vol. 25). Curran Associates, Inc.

    """

    class InitSchema(BaseTransformInitSchema):
        alpha: float = Field(ge=0)

    def __init__(
        self,
        alpha: float = 0.1,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.alpha = alpha

    def apply(
        self,
        img: ImageType,
        alpha_vector: np.ndarray,
        **params: Any,
    ) -> ImageType:
        return fpixel.fancy_pca(img, alpha_vector)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        shape = params["shape"]
        # All images now have channel dimension
        num_channels = shape[-1]
        alpha_vector = self.random_generator.normal(0, self.alpha, num_channels).astype(
            np.float32,
        )
        return {"alpha_vector": alpha_vector}


__all__ = [
    "Colorize",
    "FancyPCA",
    "ToGray",
    "ToRGB",
    "ToSepia",
    "_validate_color_range",
]
