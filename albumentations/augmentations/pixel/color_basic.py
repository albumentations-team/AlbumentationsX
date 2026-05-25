"""Basic color, tone, histogram, brightness, and contrast transforms."""

from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal

from ._color_shared import (
    MAX_VALUES_BY_DTYPE,
    PAIR,
    SEVEN,
    AfterValidator,
    BaseTransformInitSchema,
    Field,
    ImageOnlyTransform,
    ImageType,
    VolumeType,
    albucore,
    batch_transform,
    check_range_bounds,
    field_validator,
    fpixel,
    is_grayscale_image,
    is_rgb_image,
    mean,
    nondecreasing,
    np,
)


class RandomToneCurve(ImageOnlyTransform):
    """Randomly warp the tone curve to change contrast and tonal distribution. scale and
    scale_upper control strength. Good for exposure variation.

    This transform applies a random S-curve to the image's tone curve, adjusting the brightness and contrast
    in a non-linear manner. It can be applied to the entire image or to each channel separately.

    Args:
        scale (float): Standard deviation of the normal distribution used to sample random distances
            to move two control points that modify the image's curve. Values should be in range [0, 1].
            Higher values will result in more dramatic changes to the image. Default: 0.1
        per_channel (bool): If True, the tone curve will be applied to each channel of the input image separately,
            which can lead to color distortion. If False, the same curve is applied to all channels,
            preserving the original color relationships. Default: False
        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - This transform modifies the image's histogram by applying a smooth, S-shaped curve to it.
        - The S-curve is defined by moving two control points of a quadratic Bézier curve.
        - When per_channel is False, the same curve is applied to all channels, maintaining color balance.
        - When per_channel is True, different curves are applied to each channel, which can create color shifts.
        - This transform can be used to adjust image contrast and brightness in a more natural way than linear
            transforms.
        - The effect can range from subtle contrast adjustments to more dramatic "vintage" or "faded" looks.

    Mathematical Formulation:
        1. Two control points are randomly moved from their default positions (0.25, 0.25) and (0.75, 0.75).
        2. The new positions are sampled from a normal distribution: N(μ, σ²), where μ is the original position
        and alpha is the scale parameter.
        3. These points, along with fixed points at (0, 0) and (1, 1), define a quadratic Bézier curve.
        4. The curve is applied as a lookup table to the image intensities:
           new_intensity = curve(original_intensity)

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Apply a random tone curve to all channels together
        >>> transform = A.RandomToneCurve(scale=0.1, per_channel=False, p=1.0)
        >>> augmented_image = transform(image=image)['image']

        # Apply random tone curves to each channel separately
        >>> transform = A.RandomToneCurve(scale=0.2, per_channel=True, p=1.0)
        >>> augmented_image = transform(image=image)['image']

    References:
        - "What Else Can Fool Deep Learning? Addressing Color Constancy Errors on Deep Neural Network Performance":
          https://arxiv.org/abs/1912.06960
        - Bézier curve: https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Quadratic_B%C3%A9zier_curves
        - Tone mapping: https://en.wikipedia.org/wiki/Tone_mapping

    """

    class InitSchema(BaseTransformInitSchema):
        scale: float = Field(
            ge=0,
            le=1,
        )
        per_channel: bool

    def __init__(
        self,
        scale: float = 0.1,
        per_channel: bool = False,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.scale = scale
        self.per_channel = per_channel

    def apply(
        self,
        img: ImageType,
        low_y: float | np.ndarray,
        high_y: float | np.ndarray,
        num_channels: int,
        **params: Any,
    ) -> ImageType:
        return fpixel.move_tone_curve(img, low_y, high_y, num_channels)

    def apply_to_images(
        self,
        images: ImageType,
        low_y: float | np.ndarray,
        high_y: float | np.ndarray,
        num_channels: int,
        **params: Any,
    ) -> ImageType:
        return fpixel.move_tone_curve(images, low_y, high_y, num_channels)

    def apply_to_volumes(
        self,
        volumes: VolumeType,
        low_y: float | np.ndarray,
        high_y: float | np.ndarray,
        num_channels: int,
        **params: Any,
    ) -> VolumeType:
        return fpixel.move_tone_curve(volumes, low_y, high_y, num_channels)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        num_channels = self.get_image_data(data)["num_channels"]
        result = {
            "num_channels": num_channels,
        }

        self.applied_config = {"scale": self.scale}

        if self.per_channel and result["num_channels"] != 1:
            result["low_y"] = np.clip(
                self.random_generator.normal(
                    loc=0.25,
                    scale=self.scale,
                    size=(num_channels,),
                ),
                0,
                1,
            )
            result["high_y"] = np.clip(
                self.random_generator.normal(
                    loc=0.75,
                    scale=self.scale,
                    size=(num_channels,),
                ),
                0,
                1,
            )
            return result

        low_y = np.clip(self.random_generator.normal(loc=0.25, scale=self.scale), 0, 1)
        high_y = np.clip(self.random_generator.normal(loc=0.75, scale=self.scale), 0, 1)

        return {"low_y": low_y, "high_y": high_y, "num_channels": num_channels}


class HueSaturationValue(ImageOnlyTransform):
    """Randomly shift hue, saturation, and value (HSV). Separate ranges per channel. Common
    for color augmentation in classification.

    This transform adjusts the HSV (Hue, Saturation, Value) channels of an input RGB image.
    It allows for independent control over each channel, providing a wide range of color
    and brightness modifications.

    Args:
        hue_shift_range (tuple[float, float]): Range for changing hue, sampled per image.
            Values should be in the range [-180, 180]. Default: (-20, 20).

        sat_shift_range (tuple[float, float]): Range for changing saturation, sampled per image.
            Values should be in the range [-255, 255]. Default: (-30, 30).

        val_shift_range (tuple[float, float]): Range for changing value (brightness),
            sampled per image. Values should be in the range [-255, 255]. Default: (-20, 20).

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        3

    Note:
        - The transform first converts the input RGB image to the HSV color space.
        - Each channel (Hue, Saturation, Value) is adjusted independently.
        - Hue is circular, so it wraps around at 180 degrees.
        - For float32 images, the shift values are applied as percentages of the full range.
        - This transform is particularly useful for color augmentation and simulating
          different lighting conditions.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.HueSaturationValue(
        ...     hue_shift_range=(-20, 20),
        ...     sat_shift_range=(-30, 30),
        ...     val_shift_range=(-20, 20),
        ...     p=0.7,
        ... )
        >>> result = transform(image=image)
        >>> augmented_image = result["image"]

    References:
        HSV color space: https://en.wikipedia.org/wiki/HSL_and_HSV

    """

    class InitSchema(BaseTransformInitSchema):
        hue_shift_range: tuple[float, float]
        sat_shift_range: tuple[float, float]
        val_shift_range: tuple[float, float]

    def __init__(
        self,
        hue_shift_range: tuple[float, float] = (-20, 20),
        sat_shift_range: tuple[float, float] = (-30, 30),
        val_shift_range: tuple[float, float] = (-20, 20),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.hue_shift_range = hue_shift_range
        self.sat_shift_range = sat_shift_range
        self.val_shift_range = val_shift_range

    def apply(
        self,
        img: ImageType,
        hue_shift: int,
        sat_shift: int,
        val_shift: int,
        **params: Any,
    ) -> ImageType:
        if not is_rgb_image(img) and not is_grayscale_image(img):
            msg = "HueSaturationValue transformation expects 1-channel or 3-channel images."
            raise TypeError(msg)
        return fpixel.shift_hsv(img, hue_shift, sat_shift, val_shift)

    def apply_to_images(
        self,
        images: ImageType,
        hue_shift: float,
        sat_shift: float,
        val_shift: float,
        **params: Any,
    ) -> ImageType:
        return fpixel.shift_hsv_images(images, hue_shift, sat_shift, val_shift)

    def get_params(self) -> dict[str, float]:
        hue_shift = self.py_random.uniform(*self.hue_shift_range)
        sat_shift = self.py_random.uniform(*self.sat_shift_range)
        val_shift = self.py_random.uniform(*self.val_shift_range)

        self.applied_config = {
            "hue_shift_range": hue_shift,
            "sat_shift_range": sat_shift,
            "val_shift_range": val_shift,
        }

        return {
            "hue_shift": hue_shift,
            "sat_shift": sat_shift,
            "val_shift": val_shift,
        }


class Solarize(ImageOnlyTransform):
    """Invert pixel values above a threshold. threshold_range controls cutoff. Strong
    highlight inversion; useful for data augmentation.

    This transform applies a solarization effect to the input image. Solarization is a phenomenon in
    photography in which the image recorded on a negative or on a photographic print is wholly or
    partially reversed in tone. Dark areas appear light or light areas appear dark.

    In this implementation, all pixel values above a threshold are inverted.

    Args:
        threshold_range (tuple[float, float]): Range for solarizing threshold as a fraction
            of maximum value. The threshold_range should be in the range [0, 1] and will be multiplied by the
            maximum value of the image type (255 for uint8 images or 1.0 for float images).
            Default: (0.5, 0.5) (corresponds to 127.5 for uint8 and 0.5 for float32).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - For uint8 images, pixel values above the threshold are inverted as: 255 - pixel_value
        - For float32 images, pixel values above the threshold are inverted as: 1.0 - pixel_value
        - The threshold is applied to each channel independently
        - The threshold is calculated in two steps:
          1. Sample a value from threshold_range
          2. Multiply by the image's maximum value:
             * For uint8: threshold = sampled_value * 255
             * For float32: threshold = sampled_value * 1.0
        - This transform can create interesting artistic effects or be used for data augmentation

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        # Solarize uint8 image with fixed threshold at 50% of max value (127.5)
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.Solarize(threshold_range=(0.5, 0.5), p=1.0)
        >>> solarized_image = transform(image=image)['image']
        >>>
        # Solarize uint8 image with random threshold between 40-60% of max value (102-153)
        >>> transform = A.Solarize(threshold_range=(0.4, 0.6), p=1.0)
        >>> solarized_image = transform(image=image)['image']
        >>>
        # Solarize float32 image at 50% of max value (0.5)
        >>> image = np.random.rand(100, 100, 3).astype(np.float32)
        >>> transform = A.Solarize(threshold_range=(0.5, 0.5), p=1.0)
        >>> solarized_image = transform(image=image)['image']

    Mathematical Formulation:
        Let f be a value sampled from threshold_range (min, max).
        For each pixel value p:
        threshold = f * max_value
        if p > threshold:
            p_new = max_value - p
        else:
            p_new = p

        Where max_value is 255 for uint8 images and 1.0 for float32 images.

    See Also:
        Invert: For inverting all pixel values regardless of a threshold.

    """

    class InitSchema(BaseTransformInitSchema):
        threshold_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        threshold_range: tuple[float, float] = (0.5, 0.5),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.threshold_range = threshold_range

    def apply(self, img: ImageType, threshold: float, **params: Any) -> ImageType:
        return fpixel.solarize(img, threshold)

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self.apply(images, **params)

    def apply_to_volumes(self, volumes: VolumeType, **params: Any) -> VolumeType:
        return self.apply(volumes, **params)

    def get_params(self) -> dict[str, float]:
        threshold = self.py_random.uniform(*self.threshold_range)

        self.applied_config = {"threshold_range": threshold}

        return {"threshold": threshold}


class Posterize(ImageOnlyTransform):
    """Reduce bits per color channel (e.g. 8→4). num_bits_range controls strength; lower
    gives stronger posterization. Simulates low-bit-depth or compression.

    This transform applies color posterization, a technique that reduces the number of distinct
    colors used in an image. It works by lowering the number of bits used to represent each
    color channel, effectively creating a "poster-like" effect with fewer color gradations.

    Args:
        num_bits (tuple[int, int] | list[tuple[int, int]]):
            Defines the number of bits to keep for each color channel. Can be specified as:
            - tuple of two ints: (min_bits, max_bits) to randomly choose from. Range for each: [1, 7].
            - list of per-channel tuples: Ranges per channel [(r_min, r_max), (g_min, g_max), ...].
            Default: (4, 4)

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - The effect becomes more pronounced as the number of bits is reduced.
        - This transform can create interesting artistic effects or be used for image compression simulation.
        - Posterization is particularly useful for:
          * Creating stylized or retro-looking images
          * Reducing the color palette for specific artistic effects
          * Simulating the look of older or lower-quality digital images
          * Data augmentation in scenarios where color depth might vary

    Mathematical Background:
        For an 8-bit color channel, posterization to n bits can be expressed as:
        new_value = (old_value >> (8 - n)) << (8 - n)
        This operation keeps the n most significant bits and sets the rest to zero.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Posterize all channels to 3 bits
        >>> transform = A.Posterize(num_bits=(3, 3), p=1.0)
        >>> posterized_image = transform(image=image)["image"]

        # Randomly posterize between 2 and 5 bits
        >>> transform = A.Posterize(num_bits=(2, 5), p=1.0)
        >>> posterized_image = transform(image=image)["image"]

        # Range of bits for each channel
        >>> transform = A.Posterize(num_bits=[(1, 3), (3, 5), (2, 4)], p=1.0)
        >>> posterized_image = transform(image=image)["image"]

    References:
        - Color Quantization: https://en.wikipedia.org/wiki/Color_quantization
        - Posterization: https://en.wikipedia.org/wiki/Posterization

    """

    class InitSchema(BaseTransformInitSchema):
        num_bits: tuple[int, int] | list[tuple[int, int]]

        @field_validator("num_bits")
        @classmethod
        def _validate_num_bits(
            cls,
            num_bits: Any,
        ) -> tuple[int, int] | list[tuple[int, int]]:
            def _check_pair(pair: Any) -> tuple[int, int]:
                if not isinstance(pair, Sequence) or isinstance(pair, str) or len(pair) != PAIR:
                    raise ValueError("num_bits must be a tuple of two ints or a list of such tuples")
                lo, hi = int(pair[0]), int(pair[1])
                if not (1 <= lo <= SEVEN and 1 <= hi <= SEVEN):
                    raise ValueError("num_bits values must be in [1, 7]")
                if lo > hi:
                    raise ValueError("num_bits min must be <= max")
                return (lo, hi)

            if isinstance(num_bits, list):
                return [_check_pair(item) for item in num_bits]
            return _check_pair(num_bits)

    def __init__(
        self,
        num_bits: tuple[int, int] | list[tuple[int, int]] = (4, 4),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.num_bits = num_bits

    def apply(
        self,
        img: ImageType,
        num_bits: Literal[1, 2, 3, 4, 5, 6, 7] | list[Literal[1, 2, 3, 4, 5, 6, 7]],
        **params: Any,
    ) -> ImageType:
        return fpixel.posterize(img, num_bits)

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self.apply(images, **params)

    def apply_to_volumes(self, volumes: VolumeType, **params: Any) -> VolumeType:
        return self.apply(volumes, **params)

    def get_params(self) -> dict[str, Any]:
        if isinstance(self.num_bits, list):
            num_bits_list = [self.py_random.randint(*i) for i in self.num_bits]
            self.applied_config = {"num_bits": num_bits_list}
            return {"num_bits": num_bits_list}
        num_bits = self.py_random.randint(*self.num_bits)
        self.applied_config = {"num_bits": num_bits}
        return {"num_bits": num_bits}


class Equalize(ImageOnlyTransform):
    """Equalize histogram to spread intensities. mode: global or adaptive; mask optional.
    Improves contrast normalization across datasets.

    This transform applies histogram equalization to the input image. Histogram equalization
    is a method in image processing of contrast adjustment using the image's histogram.

    Args:
        mode (Literal['cv', 'pil']): Use OpenCV or Pillow equalization method.
            Default: 'cv'
        by_channels (bool): If True, use equalization by channels separately,
            else convert image to YCbCr representation and use equalization by `Y` channel.
            Default: True
        mask (np.ndarray, callable): If given, only the pixels selected by
            the mask are included in the analysis. Can be:
            - A 1-channel or 3-channel numpy array of the same size as the input image.
            - A callable (function) that generates a mask. The function should accept 'image'
              as its first argument, and can accept additional arguments specified in mask_params.
            Default: None
        mask_params (list[str]): Additional parameters to pass to the mask function.
            These parameters will be taken from the data dict passed to __call__.
            Default: ()
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        1,3

    Note:
        - When mode='cv', OpenCV's equalizeHist() function is used.
        - When mode='pil', Pillow's equalize() function is used.
        - The 'by_channels' parameter determines whether equalization is applied to each color channel
          independently (True) or to the luminance channel only (False).
        - If a mask is provided as a numpy array, it should have the same height and width as the input image.
        - If a mask is provided as a function, it allows for dynamic mask generation based on the input image
          and additional parameters. This is useful for scenarios where the mask depends on the image content
          or external data (e.g., bounding boxes, segmentation masks).

    Mask Function:
        When mask is a callable, it should have the following signature:
        mask_func(image, *args) -> np.ndarray

        - image: The input image (numpy array)
        - *args: Additional arguments as specified in mask_params

        The function should return a numpy array of the same height and width as the input image,
        where non-zero pixels indicate areas to be equalized.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>>
        >>> # Using a static mask
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> transform = A.Equalize(mask=mask, p=1.0)
        >>> result = transform(image=image)
        >>>
        >>> # Using a dynamic mask function
        >>> def mask_func(image, bboxes):
        ...     mask = np.ones_like(image[:, :, 0], dtype=np.uint8)
        ...     for bbox in bboxes:
        ...         x1, y1, x2, y2 = map(int, bbox)
        ...         mask[y1:y2, x1:x2] = 0  # Exclude areas inside bounding boxes
        ...     return mask
        >>>
        >>> transform = A.Equalize(mask=mask_func, mask_params=['bboxes'], p=1.0)
        >>> bboxes = [(10, 10, 50, 50), (60, 60, 90, 90)]  # Example bounding boxes
        >>> result = transform(image=image, bboxes=bboxes)

    References:
        - OpenCV equalizeHist: https://docs.opencv.org/3.4/d6/dc7/group__imgproc__hist.html#ga7e54091f0c937d49bf84152a16f76d6e
        - Pillow ImageOps.equalize: https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.equalize
        - Histogram Equalization: https://en.wikipedia.org/wiki/Histogram_equalization

    """

    class InitSchema(BaseTransformInitSchema):
        mode: Literal["cv", "pil"]
        by_channels: bool
        mask: np.ndarray | Callable[..., Any] | None
        mask_params: Sequence[str]

    def __init__(
        self,
        mode: Literal["cv", "pil"] = "cv",
        by_channels: bool = True,
        mask: np.ndarray | Callable[..., Any] | None = None,
        mask_params: Sequence[str] = (),
        p: float = 0.5,
    ):
        super().__init__(p=p)

        self.mode = mode
        self.by_channels = by_channels
        self.mask = mask
        self.mask_params = mask_params

    def apply(self, img: ImageType, mask: np.ndarray, **params: Any) -> ImageType:
        if not is_rgb_image(img) and not is_grayscale_image(img):
            raise ValueError("Equalize transform is only supported for RGB and grayscale images.")
        return fpixel.equalize(
            img,
            mode=self.mode,
            by_channels=self.by_channels,
            mask=mask,
        )

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        if not callable(self.mask):
            return {"mask": self.mask}

        mask_params = {"image": data["image"]}
        for key in self.mask_params:
            if key not in data:
                raise KeyError(
                    f"Required parameter '{key}' for mask function is missing in data.",
                )
            mask_params[key] = data[key]

        return {"mask": self.mask(**mask_params)}

    @property
    def targets_as_params(self) -> list[str]:
        return [*list(self.mask_params)]


class RandomBrightnessContrast(ImageOnlyTransform):
    """Randomly adjust brightness and contrast with separate ranges. Simple and fast;
    good baseline color augmentation for classification and detection.

    This transform adjusts the brightness and contrast of an image simultaneously, allowing for
    a wide range of lighting and contrast variations. It's particularly useful for data augmentation
    in computer vision tasks, helping models become more robust to different lighting conditions.

    Args:
        brightness_range (tuple[float, float]): Factor range for changing brightness, sampled
            per image. Values should typically be in the range [-1.0, 1.0], where 0 means no
            change, 1.0 means maximum brightness, and -1.0 means minimum brightness.
            Default: (-0.2, 0.2).

        contrast_range (tuple[float, float]): Factor range for changing contrast, sampled per
            image. Values should typically be in the range [-1.0, 1.0], where 0 means no change,
            1.0 means maximum increase in contrast, and -1.0 means maximum decrease in contrast.
            Default: (-0.2, 0.2).

        brightness_by_max (bool): If True, adjusts brightness by scaling pixel values up to the
            maximum value of the image's dtype. If False, uses the mean pixel value for adjustment.
            Default: True.

        ensure_safe_output (bool): If True, adjusts alpha and beta to prevent overflow/underflow.
            This keeps output values inside the valid range for the image dtype without clipping.
            Default: False.

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - The order of operation is: contrast adjustment, then brightness adjustment.
        - For uint8 images, the output is clipped to [0, 255] range.
        - For float32 images, the output is clipped to [0, 1] range.
        - The `brightness_by_max` parameter affects how brightness is adjusted:
          * If True, brightness adjustment is more pronounced and can lead to more saturated results.
          * If False, brightness adjustment is more subtle and preserves the overall lighting better.
        - This transform is useful for:
          * Simulating different lighting conditions
          * Enhancing low-light or overexposed images
          * Data augmentation to improve model robustness

    Mathematical Formulation:
        Let a be the contrast adjustment factor and β be the brightness adjustment factor.
        For each pixel value x:
        1. Contrast adjustment: x' = clip((x - mean) * (1 + a) + mean)
        2. Brightness adjustment:
           If brightness_by_max is True:  x'' = clip(x' * (1 + β))
           If brightness_by_max is False: x'' = clip(x' + β * max_value)
        Where clip() ensures values stay within the valid range for the image dtype.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Default usage
        >>> transform = A.RandomBrightnessContrast(p=1.0)
        >>> augmented_image = transform(image=image)["image"]

        # Custom brightness and contrast limits
        >>> transform = A.RandomBrightnessContrast(
        ...     brightness_range=(-0.3, 0.3),
        ...     contrast_range=(-0.3, 0.3),
        ...     p=1.0,
        ... )
        >>> augmented_image = transform(image=image)["image"]

        # Adjust brightness based on mean value
        >>> transform = A.RandomBrightnessContrast(
        ...     brightness_range=(-0.2, 0.2),
        ...     contrast_range=(-0.2, 0.2),
        ...     brightness_by_max=False,
        ...     p=1.0,
        ... )
        >>> augmented_image = transform(image=image)["image"]

    References:
        - Brightness: https://en.wikipedia.org/wiki/Brightness
        - Contrast: https://en.wikipedia.org/wiki/Contrast_(vision)

    """

    class InitSchema(BaseTransformInitSchema):
        brightness_range: tuple[float, float]
        contrast_range: tuple[float, float]
        brightness_by_max: bool
        ensure_safe_output: bool

    def __init__(
        self,
        brightness_range: tuple[float, float] = (-0.2, 0.2),
        contrast_range: tuple[float, float] = (-0.2, 0.2),
        brightness_by_max: bool = True,
        ensure_safe_output: bool = False,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.brightness_by_max = brightness_by_max
        self.ensure_safe_output = ensure_safe_output

    def apply(
        self,
        img: ImageType,
        alpha: float,
        beta: float,
        **params: Any,
    ) -> ImageType:
        max_value = MAX_VALUES_BY_DTYPE[img.dtype]
        # Scale beta according to brightness_by_max setting
        beta = beta * max_value if self.brightness_by_max else beta * float(mean(img))

        if self.ensure_safe_output:
            alpha, beta = fpixel.get_safe_brightness_contrast_params(
                alpha,
                beta,
                max_value,
            )

        return albucore.multiply_add(img, alpha, beta, inplace=False)

    def apply_to_images(self, images: ImageType, *args: Any, **params: Any) -> ImageType:
        return self.apply(images, *args, **params)

    def apply_to_volumes(self, volumes: VolumeType, *args: Any, **params: Any) -> VolumeType:
        return self.apply(volumes, *args, **params)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, float]:
        # Sample initial values
        alpha = 1.0 + self.py_random.uniform(*self.contrast_range)
        beta = self.py_random.uniform(*self.brightness_range)

        self.applied_config = {
            "brightness_range": beta,
            "contrast_range": alpha - 1.0,
        }

        return {
            "alpha": alpha,
            "beta": beta,
        }


class CLAHE(ImageOnlyTransform):
    """Contrast Limited Adaptive Histogram Equalization: local contrast with clip_range and
    tile_grid_size. Good for non-uniform lighting; preserves detail.

    CLAHE is an advanced method of improving the contrast in an image. Unlike regular histogram
    equalization, which operates on the entire image, CLAHE operates on small regions (tiles)
    in the image. This results in a more balanced equalization, preventing over-amplification
    of contrast in areas with initially low contrast.

    Args:
        clip_range (tuple[float, float]): Range for the contrast enhancement clip limit.
            Higher values allow for more contrast enhancement, but may also increase noise.
            Both bounds must be >= 1. Default: (1, 4)

        tile_grid_size (tuple[int, int]): Defines the number of tiles in the row and column directions.
            Format is (rows, columns). Smaller tile sizes can lead to more localized enhancements,
            while larger sizes give results closer to global histogram equalization.
            Default: (8, 8)

        p (float): Probability of applying the transform. Default: 0.5

    Notes:
        - Supports only RGB or grayscale images.
        - For color images, CLAHE is applied to the L channel in the LAB color space.
        - The clip limit determines the maximum slope of the cumulative histogram. A lower
          clip limit will result in more contrast limiting.
        - Tile grid size affects the adaptiveness of the method. More tiles increase local
          adaptiveness but can lead to an unnatural look if set too high.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        1, 3

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.CLAHE(clip_range=(1, 4), tile_grid_size=(8, 8), p=1.0)
        >>> result = transform(image=image)
        >>> clahe_image = result["image"]

    References:
        - Tutorial: https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
        - "Contrast Limited Adaptive Histogram Equalization.": https://ieeexplore.ieee.org/document/109340

    """

    class InitSchema(BaseTransformInitSchema):
        clip_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(1, None)),
            AfterValidator(nondecreasing),
        ]
        tile_grid_size: Annotated[tuple[int, int], AfterValidator(check_range_bounds(1, None))]

    def __init__(
        self,
        clip_range: tuple[float, float] = (1.0, 4.0),
        tile_grid_size: tuple[int, int] = (8, 8),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.clip_range = clip_range
        self.tile_grid_size = tile_grid_size

    def apply(self, img: ImageType, clip_limit: float, **params: Any) -> ImageType:
        if not is_rgb_image(img) and not is_grayscale_image(img):
            msg = "CLAHE transformation expects 1-channel or 3-channel images."
            raise TypeError(msg)

        return fpixel.clahe(img, clip_limit, self.tile_grid_size)

    def get_params(self) -> dict[str, float]:
        clip_limit = self.py_random.uniform(*self.clip_range)

        self.applied_config = {"clip_range": clip_limit}

        return {"clip_limit": clip_limit}


class RandomGamma(ImageOnlyTransform):
    """Apply random gamma correction (power-law on intensity). gamma_range controls range.
    Common for exposure and display variation.

    Gamma correction, or simply gamma, is a nonlinear operation used to encode and decode luminance
    or tristimulus values in imaging systems. This transform can adjust the brightness of an image
    while preserving the relative differences between darker and lighter areas, making it useful
    for simulating different lighting conditions or correcting for display characteristics.

    Args:
        gamma_range (tuple[float, float]): Lower and upper bounds for gamma adjustment, sampled
            per image. Values are in terms of percentage change, e.g. (80, 120) means the gamma
            will be between 80% and 120% of the original. Default: (80, 120).
        eps (float): A small value added to the gamma to avoid division by zero or log of zero errors.
            Default: 1e-7.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - The gamma correction is applied using the formula: output = input^gamma
        - Gamma values > 1 will make the image darker, while values < 1 will make it brighter
        - This transform is particularly useful for:
          * Simulating different lighting conditions
          * Correcting for non-linear display characteristics
          * Enhancing contrast in certain regions of the image
          * Data augmentation in computer vision tasks

    Mathematical Formulation:
        Let I be the input image and G (gamma) be the correction factor.
        The gamma correction is applied as follows:
        1. Normalize the image to [0, 1] range: I_norm = I / 255 (for uint8 images)
        2. Apply gamma correction: I_corrected = I_norm ^ (1 / G)
        3. Scale back to original range: output = I_corrected * 255 (for uint8 images)

        The actual gamma value used is calculated as:
        G = 1 + (random_value / 100), where random_value is sampled from gamma_range range.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Default usage
        >>> transform = A.RandomGamma(p=1.0)
        >>> augmented_image = transform(image=image)["image"]

        # Custom gamma range
        >>> transform = A.RandomGamma(gamma_range=(50, 150), p=1.0)
        >>> augmented_image = transform(image=image)["image"]

        # Applying with other transforms
        >>> transform = A.Compose([
        ...     A.RandomGamma(gamma_range=(80, 120), p=0.5),
        ...     A.RandomBrightnessContrast(p=0.5),
        ... ])
        >>> augmented_image = transform(image=image)["image"]

    References:
        - Gamma correction: https://en.wikipedia.org/wiki/Gamma_correction
        - Power law (Gamma) encoding: https://www.cambridgeincolour.com/tutorials/gamma-correction.htm

    """

    class InitSchema(BaseTransformInitSchema):
        gamma_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(1)),
        ]

    def __init__(
        self,
        gamma_range: tuple[float, float] = (80, 120),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.gamma_range = gamma_range

    def apply(self, img: ImageType, gamma: float, **params: Any) -> ImageType:
        return fpixel.gamma_transform(img, gamma=gamma)

    def apply_to_volumes(self, volumes: VolumeType, gamma: float, **params: Any) -> VolumeType:
        return self.apply(volumes, gamma=gamma)

    def apply_to_images(self, images: ImageType, gamma: float, **params: Any) -> ImageType:
        return self.apply(images, gamma=gamma)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        gamma = self.py_random.uniform(*self.gamma_range)

        self.applied_config = {"gamma_range": gamma}

        return {
            "gamma": gamma / 100.0,
        }


class AutoContrast(ImageOnlyTransform):
    """Stretch intensity to full range (autocontrast). method: CDF or PIL-style. cutoff, ignore trim
    extremes. Use for normalizing brightness/contrast across images.

    This transform provides two methods for contrast enhancement:
    1. CDF method (default): Uses cumulative distribution function for more gradual adjustment
    2. PIL method: Uses linear scaling like PIL.ImageOps.autocontrast

    The transform can optionally exclude extreme values from both ends of the
    intensity range and preserve specific intensity values (e.g., alpha channel).

    Args:
        cutoff (float): Percentage of pixels to exclude from both ends of the histogram.
            Range: [0, 100]. Default: 0 (use full intensity range)
            - 0 means use the minimum and maximum intensity values found
            - 20 means exclude darkest and brightest 20% of pixels
        ignore (int, optional): Intensity value to preserve (e.g., alpha channel).
            Range: [0, 255]. Default: None
            - If specified, this intensity value will not be modified
            - Useful for images with alpha channel or special marker values
        method (Literal['cdf', 'pil']): Algorithm to use for contrast enhancement.
            Default: "cdf"
            - "cdf": Uses cumulative distribution for smoother adjustment
            - "pil": Uses linear scaling like PIL.ImageOps.autocontrast
        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image, volume

    Image types:
        uint8, float32

    Note:
        - The transform processes each color channel independently
        - For grayscale images, only one channel is processed
        - The output maintains the same dtype as input
        - Empty or single-color channels remain unchanged

    Examples:
        >>> import albumentations as A
        >>> # Basic usage
        >>> transform = A.AutoContrast(p=1.0)
        >>>
        >>> # Exclude extreme values
        >>> transform = A.AutoContrast(cutoff=20, p=1.0)
        >>>
        >>> # Preserve alpha channel
        >>> transform = A.AutoContrast(ignore=255, p=1.0)
        >>>
        >>> # Use PIL-like contrast enhancement
        >>> transform = A.AutoContrast(method="pil", p=1.0)

    """

    class InitSchema(BaseTransformInitSchema):
        cutoff: float = Field(ge=0, le=100)
        ignore: int | None = Field(ge=0, le=255)
        method: Literal["cdf", "pil"]

    def __init__(
        self,
        cutoff: float = 0,
        ignore: int | None = None,
        method: Literal["cdf", "pil"] = "cdf",
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.cutoff = cutoff
        self.ignore = ignore
        self.method = method

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        return fpixel.auto_contrast(img, self.cutoff, self.ignore, self.method)

    @batch_transform("channel")
    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self.apply(images, **params)

    @batch_transform("channel")
    def apply_to_volumes(self, volumes: VolumeType, **params: Any) -> VolumeType:
        return self.apply(volumes, **params)


__all__ = [
    "CLAHE",
    "AutoContrast",
    "Equalize",
    "HueSaturationValue",
    "Posterize",
    "RandomBrightnessContrast",
    "RandomGamma",
    "RandomToneCurve",
    "Solarize",
]
