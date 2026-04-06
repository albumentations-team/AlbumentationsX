"""Color, tone, and brightness transforms.

Transforms that modify color properties, tone curves, brightness, contrast,
saturation, hue, and other color-space operations.
"""

import warnings
from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal, cast

import albucore
import cv2
import numpy as np
from albucore import (
    MAX_VALUES_BY_DTYPE,
    batch_transform,
    get_image_data,
    get_num_channels,
    is_grayscale_image,
    is_rgb_image,
    mean,
)
from pydantic import Field, field_validator, model_validator
from pydantic.functional_validators import AfterValidator
from typing_extensions import Self

from albumentations.augmentations.pixel import functional as fpixel
from albumentations.augmentations.pixel.noise import AdditiveNoise
from albumentations.augmentations.utils import non_rgb_error
from albumentations.core.pydantic import (
    check_range_bounds,
    convert_to_1centered_range,
    convert_to_1plus_range,
    create_symmetric_range,
    nondecreasing,
)
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    ImageOnlyTransform,
)
from albumentations.core.type_definitions import (
    NUM_RGB_CHANNELS,
    PAIR,
    SEVEN,
    ImageType,
    VolumeType,
)
from albumentations.core.utils import to_tuple

__all__ = [
    "CLAHE",
    "AutoContrast",
    "ChromaticAberration",
    "ColorJitter",
    "Equalize",
    "FancyPCA",
    "HEStain",
    "HueSaturationValue",
    "Illumination",
    "PhotoMetricDistort",
    "PlanckianJitter",
    "PlasmaBrightnessContrast",
    "PlasmaShadow",
    "Posterize",
    "RGBShift",
    "RandomBrightnessContrast",
    "RandomGamma",
    "RandomToneCurve",
    "Solarize",
    "ToGray",
    "ToRGB",
    "ToSepia",
    "Vignetting",
]


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
        num_channels = get_image_data(data)["num_channels"]
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
        hue_shift_limit (float | tuple[float, float]): Range for changing hue.
            If a single float value is provided, the range will be (-hue_shift_limit, hue_shift_limit).
            Values should be in the range [-180, 180]. Default: (-20, 20).

        sat_shift_limit (float | tuple[float, float]): Range for changing saturation.
            If a single float value is provided, the range will be (-sat_shift_limit, sat_shift_limit).
            Values should be in the range [-255, 255]. Default: (-30, 30).

        val_shift_limit (float | tuple[float, float]): Range for changing value (brightness).
            If a single float value is provided, the range will be (-val_shift_limit, val_shift_limit).
            Values should be in the range [-255, 255]. Default: (-20, 20).

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
        ...     hue_shift_limit=20,
        ...     sat_shift_limit=30,
        ...     val_shift_limit=20,
        ...     p=0.7
        ... )
        >>> result = transform(image=image)
        >>> augmented_image = result["image"]

    References:
        HSV color space: https://en.wikipedia.org/wiki/HSL_and_HSV

    """

    class InitSchema(BaseTransformInitSchema):
        hue_shift_limit: Annotated[
            tuple[float, float] | float,
            AfterValidator(create_symmetric_range),
        ]
        sat_shift_limit: Annotated[
            tuple[float, float] | float,
            AfterValidator(create_symmetric_range),
        ]
        val_shift_limit: Annotated[
            tuple[float, float] | float,
            AfterValidator(create_symmetric_range),
        ]

    def __init__(
        self,
        hue_shift_limit: tuple[float, float] | float = (-20, 20),
        sat_shift_limit: tuple[float, float] | float = (-30, 30),
        val_shift_limit: tuple[float, float] | float = (-20, 20),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.hue_shift_limit = cast("tuple[float, float]", hue_shift_limit)
        self.sat_shift_limit = cast("tuple[float, float]", sat_shift_limit)
        self.val_shift_limit = cast("tuple[float, float]", val_shift_limit)

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
        hue_shift = self.py_random.uniform(*self.hue_shift_limit)
        sat_shift = self.py_random.uniform(*self.sat_shift_limit)
        val_shift = self.py_random.uniform(*self.val_shift_limit)

        self.applied_config = {
            "hue_shift_limit": hue_shift,
            "sat_shift_limit": sat_shift,
            "val_shift_limit": val_shift,
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
        num_bits (int | tuple[int, int] | list[int] | list[tuple[int, int]]):
            Defines the number of bits to keep for each color channel. Can be specified in several ways:
            - Single int: Same number of bits for all channels. Range: [1, 7].
            - tuple of two ints: (min_bits, max_bits) to randomly choose from. Range for each: [1, 7].
            - list of three ints: Specific number of bits for each channel [r_bits, g_bits, b_bits].
            - list of three tuples: Ranges for each channel [(r_min, r_max), (g_min, g_max), (b_min, b_max)].
            Default: 4

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
        >>> transform = A.Posterize(num_bits=3, p=1.0)
        >>> posterized_image = transform(image=image)["image"]

        # Randomly posterize between 2 and 5 bits
        >>> transform = A.Posterize(num_bits=(2, 5), p=1.0)
        >>> posterized_image = transform(image=image)["image"]

        # Different bits for each channel
        >>> transform = A.Posterize(num_bits=[3, 5, 2], p=1.0)
        >>> posterized_image = transform(image=image)["image"]

        # Range of bits for each channel
        >>> transform = A.Posterize(num_bits=[(1, 3), (3, 5), (2, 4)], p=1.0)
        >>> posterized_image = transform(image=image)["image"]

    References:
        - Color Quantization: https://en.wikipedia.org/wiki/Color_quantization
        - Posterization: https://en.wikipedia.org/wiki/Posterization

    """

    class InitSchema(BaseTransformInitSchema):
        num_bits: int | tuple[int, int] | list[tuple[int, int]]

        @field_validator("num_bits")
        @classmethod
        def _validate_num_bits(
            cls,
            num_bits: Any,
        ) -> tuple[int, int] | list[tuple[int, int]]:
            if isinstance(num_bits, int):
                if num_bits < 1 or num_bits > SEVEN:
                    raise ValueError("num_bits must be in the range [1, 7]")
                return (num_bits, num_bits)
            if isinstance(num_bits, Sequence) and len(num_bits) > PAIR:
                return [to_tuple(i, i) for i in num_bits]
            return to_tuple(num_bits, num_bits)

    def __init__(
        self,
        num_bits: int | tuple[int, int] | list[tuple[int, int]] = 4,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.num_bits = cast("tuple[int, int] | list[tuple[int, int]]", num_bits)

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
        brightness_limit (float | tuple[float, float]): Factor range for changing brightness.
            If a single float value is provided, the range will be (-brightness_limit, brightness_limit).
            Values should typically be in the range [-1.0, 1.0], where 0 means no change,
            1.0 means maximum brightness, and -1.0 means minimum brightness.
            Default: (-0.2, 0.2).

        contrast_limit (float | tuple[float, float]): Factor range for changing contrast.
            If a single float value is provided, the range will be (-contrast_limit, contrast_limit).
            Values should typically be in the range [-1.0, 1.0], where 0 means no change,
            1.0 means maximum increase in contrast, and -1.0 means maximum decrease in contrast.
            Default: (-0.2, 0.2).

        brightness_by_max (bool): If True, adjusts brightness by scaling pixel values up to the
            maximum value of the image's dtype. If False, uses the mean pixel value for adjustment.
            Default: True.

        ensure_safe_range (bool): If True, adjusts alpha and beta to prevent overflow/underflow.
            This ensures output values stay within the valid range for the image dtype without clipping.
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
        ...     brightness_limit=0.3,
        ...     contrast_limit=0.3,
        ...     p=1.0
        ... )
        >>> augmented_image = transform(image=image)["image"]

        # Adjust brightness based on mean value
        >>> transform = A.RandomBrightnessContrast(
        ...     brightness_limit=0.2,
        ...     contrast_limit=0.2,
        ...     brightness_by_max=False,
        ...     p=1.0
        ... )
        >>> augmented_image = transform(image=image)["image"]

    References:
        - Brightness: https://en.wikipedia.org/wiki/Brightness
        - Contrast: https://en.wikipedia.org/wiki/Contrast_(vision)

    """

    class InitSchema(BaseTransformInitSchema):
        brightness_limit: Annotated[
            tuple[float, float] | float,
            AfterValidator(create_symmetric_range),
        ]
        contrast_limit: Annotated[
            tuple[float, float] | float,
            AfterValidator(create_symmetric_range),
        ]
        brightness_by_max: bool
        ensure_safe_range: bool

    def __init__(
        self,
        brightness_limit: tuple[float, float] | float = (-0.2, 0.2),
        contrast_limit: tuple[float, float] | float = (-0.2, 0.2),
        brightness_by_max: bool = True,
        ensure_safe_range: bool = False,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.brightness_limit = cast("tuple[float, float]", brightness_limit)
        self.contrast_limit = cast("tuple[float, float]", contrast_limit)
        self.brightness_by_max = brightness_by_max
        self.ensure_safe_range = ensure_safe_range

    def apply(
        self,
        img: ImageType,
        alpha: float,
        beta: float,
        **params: Any,
    ) -> ImageType:
        max_value = MAX_VALUES_BY_DTYPE[img.dtype]
        # Scale beta according to brightness_by_max setting
        beta = beta * max_value if self.brightness_by_max else beta * mean(img)

        # Clip values to safe ranges if needed
        if self.ensure_safe_range:
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
        alpha = 1.0 + self.py_random.uniform(*self.contrast_limit)
        beta = self.py_random.uniform(*self.brightness_limit)

        self.applied_config = {
            "brightness_limit": beta,
            "contrast_limit": alpha - 1.0,
        }

        return {
            "alpha": alpha,
            "beta": beta,
        }


class CLAHE(ImageOnlyTransform):
    """Contrast Limited Adaptive Histogram Equalization: local contrast with clip_limit and
    tile_grid_size. Good for non-uniform lighting; preserves detail.

    CLAHE is an advanced method of improving the contrast in an image. Unlike regular histogram
    equalization, which operates on the entire image, CLAHE operates on small regions (tiles)
    in the image. This results in a more balanced equalization, preventing over-amplification
    of contrast in areas with initially low contrast.

    Args:
        clip_limit (tuple[float, float] | float): Controls the contrast enhancement limit.
            - If a single float is provided, the range will be (1, clip_limit).
            - If a tuple of two floats is provided, it defines the range for random selection.
            Higher values allow for more contrast enhancement, but may also increase noise.
            Default: (1, 4)

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
        >>> transform = A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0)
        >>> result = transform(image=image)
        >>> clahe_image = result["image"]

    References:
        - Tutorial: https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
        - "Contrast Limited Adaptive Histogram Equalization.": https://ieeexplore.ieee.org/document/109340

    """

    class InitSchema(BaseTransformInitSchema):
        clip_limit: Annotated[
            tuple[float, float] | float,
            AfterValidator(convert_to_1plus_range),
            AfterValidator(check_range_bounds(1, None)),
        ]
        tile_grid_size: Annotated[tuple[int, int], AfterValidator(check_range_bounds(1, None))]

    def __init__(
        self,
        clip_limit: tuple[float, float] | float = 4.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.clip_limit = cast("tuple[float, float]", clip_limit)
        self.tile_grid_size = tile_grid_size

    def apply(self, img: ImageType, clip_limit: float, **params: Any) -> ImageType:
        if not is_rgb_image(img) and not is_grayscale_image(img):
            msg = "CLAHE transformation expects 1-channel or 3-channel images."
            raise TypeError(msg)

        return fpixel.clahe(img, clip_limit, self.tile_grid_size)

    def get_params(self) -> dict[str, float]:
        clip_limit = self.py_random.uniform(*self.clip_limit)

        self.applied_config = {"clip_limit": clip_limit}

        return {"clip_limit": clip_limit}


class RandomGamma(ImageOnlyTransform):
    """Apply random gamma correction (power-law on intensity). gamma_limit controls range.
    Common for exposure and display variation.

    Gamma correction, or simply gamma, is a nonlinear operation used to encode and decode luminance
    or tristimulus values in imaging systems. This transform can adjust the brightness of an image
    while preserving the relative differences between darker and lighter areas, making it useful
    for simulating different lighting conditions or correcting for display characteristics.

    Args:
        gamma_limit (float | tuple[float, float]): If gamma_limit is a single float value, the range
            will be (1, gamma_limit). If it's a tuple of two floats, they will serve as
            the lower and upper bounds for gamma adjustment. Values are in terms of percentage change,
            e.g., (80, 120) means the gamma will be between 80% and 120% of the original.
            Default: (80, 120).
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
        G = 1 + (random_value / 100), where random_value is sampled from gamma_limit range.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Default usage
        >>> transform = A.RandomGamma(p=1.0)
        >>> augmented_image = transform(image=image)["image"]

        # Custom gamma range
        >>> transform = A.RandomGamma(gamma_limit=(50, 150), p=1.0)
        >>> augmented_image = transform(image=image)["image"]

        # Applying with other transforms
        >>> transform = A.Compose([
        ...     A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ...     A.RandomBrightnessContrast(p=0.5),
        ... ])
        >>> augmented_image = transform(image=image)["image"]

    References:
        - Gamma correction: https://en.wikipedia.org/wiki/Gamma_correction
        - Power law (Gamma) encoding: https://www.cambridgeincolour.com/tutorials/gamma-correction.htm

    """

    class InitSchema(BaseTransformInitSchema):
        gamma_limit: Annotated[
            tuple[float, float] | float,
            AfterValidator(convert_to_1plus_range),
            AfterValidator(check_range_bounds(1, None)),
        ]

    def __init__(
        self,
        gamma_limit: tuple[float, float] | float = (80, 120),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.gamma_limit = cast("tuple[float, float]", gamma_limit)

    def apply(self, img: ImageType, gamma: float, **params: Any) -> ImageType:
        return fpixel.gamma_transform(img, gamma=gamma)

    def apply_to_volumes(self, volumes: VolumeType, gamma: float, **params: Any) -> VolumeType:
        return self.apply(volumes, gamma=gamma)

    def apply_to_images(self, images: ImageType, gamma: float, **params: Any) -> ImageType:
        return self.apply(images, gamma=gamma)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        gamma = self.py_random.uniform(*self.gamma_limit)

        self.applied_config = {"gamma_limit": gamma}

        return {
            "gamma": gamma / 100.0,
        }


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


class ColorJitter(ImageOnlyTransform):
    """Randomly apply brightness, contrast, saturation, hue in random order. Separate ranges per
    effect. Strong color augmentation for classification and detection.

    This transform is similar to torchvision's ColorJitter but with some differences due to the use of OpenCV
    instead of Pillow. The main differences are:
    1. OpenCV and Pillow use different formulas to convert images to HSV format.
    2. This implementation uses value saturation instead of uint8 overflow as in Pillow.

    These differences may result in slightly different output compared to torchvision's ColorJitter.

    Args:
        brightness (tuple[float, float] | float): How much to jitter brightness.
            If float:
                The brightness factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
            If tuple:
                The brightness factor is sampled from the range specified.
            Should be non-negative numbers.
            Default: (0.8, 1.2)

        contrast (tuple[float, float] | float): How much to jitter contrast.
            If float:
                The contrast factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
            If tuple:
                The contrast factor is sampled from the range specified.
            Should be non-negative numbers.
            Default: (0.8, 1.2)

        saturation (tuple[float, float] | float): How much to jitter saturation.
            If float:
                The saturation factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
            If tuple:
                The saturation factor is sampled from the range specified.
            Should be non-negative numbers.
            Default: (0.8, 1.2)

        hue (float or tuple of float (min, max)): How much to jitter hue.
            If float:
                The hue factor is chosen uniformly from [-hue, hue]. Should have 0 <= hue <= 0.5.
            If tuple:
                The hue factor is sampled from the range specified. Values should be in range [-0.5, 0.5].
            Default: (-0.5, 0.5)

         p (float): Probability of applying the transform. Should be in the range [0, 1].
            Default: 0.5


    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        1, 3

    Note:
        - The order of application for these color transformations is random for each image.
        - The ranges for brightness, contrast, and saturation are applied as multiplicative factors.
        - The range for hue is applied as an additive factor.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0)
        >>> result = transform(image=image)
        >>> jittered_image = result['image']

    References:
        - ColorJitter: https://pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html
        - Color Conversions: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html

    """

    class InitSchema(BaseTransformInitSchema):
        brightness: Annotated[
            tuple[float, float] | float,
            AfterValidator(convert_to_1centered_range),
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        contrast: Annotated[
            tuple[float, float] | float,
            AfterValidator(convert_to_1centered_range),
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        saturation: Annotated[
            tuple[float, float] | float,
            AfterValidator(convert_to_1centered_range),
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        hue: Annotated[
            tuple[float, float] | float,
            AfterValidator(create_symmetric_range),
            AfterValidator(check_range_bounds(-0.5, 0.5)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        brightness: tuple[float, float] | float = (0.8, 1.2),
        contrast: tuple[float, float] | float = (0.8, 1.2),
        saturation: tuple[float, float] | float = (0.8, 1.2),
        hue: tuple[float, float] | float = (-0.5, 0.5),
        p: float = 0.5,
    ):
        super().__init__(p=p)

        self.brightness = cast("tuple[float, float]", brightness)
        self.contrast = cast("tuple[float, float]", contrast)
        self.saturation = cast("tuple[float, float]", saturation)
        self.hue = cast("tuple[float, float]", hue)

    def get_params(self) -> dict[str, Any]:
        brightness = self.py_random.uniform(*self.brightness)
        contrast = self.py_random.uniform(*self.contrast)
        saturation = self.py_random.uniform(*self.saturation)
        hue = self.py_random.uniform(*self.hue)

        self.applied_config = {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue,
        }

        order = ["brightness", "contrast", "saturation", "hue"]
        self.random_generator.shuffle(order)

        # Merge adjacent brightness+contrast into one slot for fused LUT.
        idx_b, idx_c = order.index("brightness"), order.index("contrast")
        if abs(idx_b - idx_c) == 1:
            merged = "brightness_contrast" if idx_b < idx_c else "contrast_brightness"
            order = [o for o in order if o not in ("brightness", "contrast")]
            order.insert(min(idx_b, idx_c), merged)

        return {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue,
            "order": order,
        }

    def apply(
        self,
        img: ImageType,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        order: list[str],
        **params: Any,
    ) -> ImageType:
        if not is_rgb_image(img) and not is_grayscale_image(img):
            msg = "ColorJitter transformation expects 1-channel or 3-channel images."
            raise TypeError(msg)

        for op in order:
            if op == "brightness_contrast":
                img = fpixel.apply_brightness_contrast_torchvision(
                    img,
                    brightness,
                    contrast,
                    brightness_first=True,
                )
            elif op == "contrast_brightness":
                img = fpixel.apply_brightness_contrast_torchvision(
                    img,
                    brightness,
                    contrast,
                    brightness_first=False,
                )
            elif op == "brightness":
                img = fpixel.adjust_brightness_torchvision(img, brightness)
            elif op == "contrast":
                img = fpixel.adjust_contrast_torchvision(img, contrast)
            elif op == "saturation":
                img = fpixel.adjust_saturation_torchvision(img, saturation)
            elif op == "hue":
                img = fpixel.adjust_hue_torchvision(img, hue)
        return img

    def apply_to_images(self, images: ImageType, *args: Any, **params: Any) -> ImageType:
        return self._apply_to_batch_same_shape(images, lambda image: self.apply(image, **params))


class ChromaticAberration(ImageOnlyTransform):
    """Add lateral chromatic aberration: shift red and blue relative to green. distortion_limit
    and shift_limit control strength. Simulates lens color fringing.

    Chromatic aberration is an optical effect that occurs when a lens fails to focus all colors to the same point.
    This transform simulates this effect by applying different radial distortions to the red and blue channels
    of the image, while leaving the green channel unchanged.

    Args:
        primary_distortion_limit (tuple[float, float] | float): Range of the primary radial distortion coefficient.
            If a single float value is provided, the range
            will be (-primary_distortion_limit, primary_distortion_limit).
            This parameter controls the distortion in the center of the image:
            - Positive values result in pincushion distortion (edges bend inward)
            - Negative values result in barrel distortion (edges bend outward)
            Default: (-0.02, 0.02).

        secondary_distortion_limit (tuple[float, float] | float): Range of the secondary radial distortion coefficient.
            If a single float value is provided, the range
            will be (-secondary_distortion_limit, secondary_distortion_limit).
            This parameter controls the distortion in the corners of the image:
            - Positive values enhance pincushion distortion
            - Negative values enhance barrel distortion
            Default: (-0.05, 0.05).

        mode (Literal['green_purple', 'red_blue', 'random']): Type of color fringing to apply. Options are:
            - 'green_purple': Distorts red and blue channels in opposite directions, creating green-purple fringing.
            - 'red_blue': Distorts red and blue channels in the same direction, creating red-blue fringing.
            - 'random': Randomly chooses between 'green_purple' and 'red_blue' modes for each application.
            Default: 'green_purple'.

        interpolation (InterpolationType): Flag specifying the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.

        p (float): Probability of applying the transform. Should be in the range [0, 1].
            Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        3

    Note:
        - This transform only affects RGB images. Grayscale images will raise an error.
        - The strength of the effect depends on both primary and secondary distortion limits.
        - Higher absolute values for distortion limits will result in more pronounced chromatic aberration.
        - The 'green_purple' mode tends to produce more noticeable effects than 'red_blue'.

    Examples:
        >>> import albumentations as A
        >>> import cv2
        >>> transform = A.ChromaticAberration(
        ...     primary_distortion_limit=0.05,
        ...     secondary_distortion_limit=0.1,
        ...     mode='green_purple',
        ...     interpolation=cv2.INTER_LINEAR,
        ...     p=1.0
        ... )
        >>> transformed = transform(image=image)
        >>> aberrated_image = transformed['image']

    References:
        Chromatic Aberration: https://en.wikipedia.org/wiki/Chromatic_aberration

    """

    class InitSchema(BaseTransformInitSchema):
        primary_distortion_limit: Annotated[
            tuple[float, float] | float,
            AfterValidator(create_symmetric_range),
        ]
        secondary_distortion_limit: Annotated[
            tuple[float, float] | float,
            AfterValidator(create_symmetric_range),
        ]
        mode: Literal["green_purple", "red_blue", "random"]
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]

    def __init__(
        self,
        primary_distortion_limit: tuple[float, float] | float = (-0.02, 0.02),
        secondary_distortion_limit: tuple[float, float] | float = (-0.05, 0.05),
        mode: Literal["green_purple", "red_blue", "random"] = "green_purple",
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_LINEAR,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.primary_distortion_limit = cast(
            "tuple[float, float]",
            primary_distortion_limit,
        )
        self.secondary_distortion_limit = cast(
            "tuple[float, float]",
            secondary_distortion_limit,
        )
        self.mode = mode
        self.interpolation = interpolation

    def apply(
        self,
        img: ImageType,
        primary_distortion_red: float,
        secondary_distortion_red: float,
        primary_distortion_blue: float,
        secondary_distortion_blue: float,
        **params: Any,
    ) -> ImageType:
        non_rgb_error(img)
        return fpixel.chromatic_aberration(
            img,
            primary_distortion_red,
            secondary_distortion_red,
            primary_distortion_blue,
            secondary_distortion_blue,
            self.interpolation,
        )

    def get_params(self) -> dict[str, float]:
        primary_distortion_red = self.py_random.uniform(*self.primary_distortion_limit)
        secondary_distortion_red = self.py_random.uniform(
            *self.secondary_distortion_limit,
        )
        primary_distortion_blue = self.py_random.uniform(*self.primary_distortion_limit)
        secondary_distortion_blue = self.py_random.uniform(
            *self.secondary_distortion_limit,
        )

        secondary_distortion_red = self._match_sign(
            primary_distortion_red,
            secondary_distortion_red,
        )
        secondary_distortion_blue = self._match_sign(
            primary_distortion_blue,
            secondary_distortion_blue,
        )

        if self.mode == "green_purple":
            # distortion coefficients of the red and blue channels have the same sign
            primary_distortion_blue = self._match_sign(
                primary_distortion_red,
                primary_distortion_blue,
            )
            secondary_distortion_blue = self._match_sign(
                secondary_distortion_red,
                secondary_distortion_blue,
            )
        if self.mode == "red_blue":
            # distortion coefficients of the red and blue channels have the opposite sign
            primary_distortion_blue = self._unmatch_sign(
                primary_distortion_red,
                primary_distortion_blue,
            )
            secondary_distortion_blue = self._unmatch_sign(
                secondary_distortion_red,
                secondary_distortion_blue,
            )

        self.applied_config = {
            "primary_distortion_limit": (primary_distortion_red, primary_distortion_blue),
            "secondary_distortion_limit": (secondary_distortion_red, secondary_distortion_blue),
        }
        return {
            "primary_distortion_red": primary_distortion_red,
            "secondary_distortion_red": secondary_distortion_red,
            "primary_distortion_blue": primary_distortion_blue,
            "secondary_distortion_blue": secondary_distortion_blue,
        }

    @staticmethod
    def _match_sign(a: float, b: float) -> float:
        # Match the sign of b to a
        if (a < 0 < b) or (a > 0 > b):
            return -b
        return b

    @staticmethod
    def _unmatch_sign(a: float, b: float) -> float:
        # Unmatch the sign of b to a
        if (a < 0 and b < 0) or (a > 0 and b > 0):
            return -b
        return b


PLANKIAN_JITTER_CONST = {
    "MAX_TEMP": max(
        *fpixel.PLANCKIAN_COEFFS["blackbody"].keys(),
        *fpixel.PLANCKIAN_COEFFS["cied"].keys(),
    ),
    "MIN_BLACKBODY_TEMP": min(fpixel.PLANCKIAN_COEFFS["blackbody"].keys()),
    "MIN_CIED_TEMP": min(fpixel.PLANCKIAN_COEFFS["cied"].keys()),
    "WHITE_TEMP": 6_000,
    "SAMPLING_TEMP_PROB": 0.4,
}


class PlanckianJitter(ImageOnlyTransform):
    """Simulate color temperature variation via Planckian locus jitter. mode and magnitude
    control the shift. Good for robustness to different light sources.

    This transform adjusts the color of an image to mimic the effect of different color temperatures
    of light sources, based on Planck's law of black body radiation. It can simulate the appearance
    of an image under various lighting conditions, from warm (reddish) to cool (bluish) color casts.

    PlanckianJitter vs. ColorJitter:
    PlanckianJitter is fundamentally different from ColorJitter in its approach and use cases:
    1. Physics-based: PlanckianJitter is grounded in the physics of light, simulating real-world
       color temperature changes. ColorJitter applies arbitrary color adjustments.
    2. Natural effects: This transform produces color shifts that correspond to natural lighting
       variations, making it ideal for outdoor scene simulation or color constancy problems.
    3. Single parameter: Color changes are controlled by a single, physically meaningful parameter
       (color temperature), unlike ColorJitter's multiple abstract parameters.
    4. Correlated changes: Color shifts are correlated across channels in a way that mimics natural
       light, whereas ColorJitter can make independent channel adjustments.

    When to use PlanckianJitter:
    - Simulating different times of day or lighting conditions in outdoor scenes
    - Augmenting data for computer vision tasks that need to be robust to natural lighting changes
    - Preparing synthetic data to better match real-world lighting variations
    - Color constancy research or applications
    - When you need physically plausible color variations rather than arbitrary color changes

    The logic behind PlanckianJitter:
    As the color temperature increases:
    1. Lower temperatures (around 3000K) produce warm, reddish tones, simulating sunset or incandescent lighting.
    2. Mid-range temperatures (around 5500K) correspond to daylight.
    3. Higher temperatures (above 7000K) result in cool, bluish tones, similar to overcast sky or shade.
    This progression mimics the natural variation of sunlight throughout the day and in different weather conditions.

    Args:
        mode (Literal['blackbody', 'cied']): The mode of the transformation.
            - "blackbody": Simulates blackbody radiation color changes.
            - "cied": Uses the CIE D illuminant series for color temperature simulation.
            Default: "blackbody"

        temperature_limit (tuple[int, int] | None): The range of color temperatures (in Kelvin) to sample from.
            - For "blackbody" mode: Should be within [3000K, 15000K]. Default: (3000, 15000)
            - For "cied" mode: Should be within [4000K, 15000K]. Default: (4000, 15000)
            If None, the default ranges will be used based on the selected mode.
            Higher temperatures produce cooler (bluish) images, lower temperatures produce warmer (reddish) images.

        sampling_method (Literal['uniform', 'gaussian']): Method to sample the temperature.
            - "uniform": Samples uniformly across the specified range.
            - "gaussian": Samples from a Gaussian distribution centered at 6500K (approximate daylight).
            Default: "uniform"

        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        3

    Note:
        - The transform preserves the overall brightness of the image while shifting its color.
        - The "blackbody" mode provides a wider range of color shifts, especially in the lower (warmer) temperatures.
        - The "cied" mode is based on standard illuminants and may provide more realistic daylight variations.
        - The Gaussian sampling method tends to produce more subtle variations, as it's centered around daylight.
        - Unlike ColorJitter, this transform ensures that color changes are physically plausible and correlated
          across channels, maintaining the natural appearance of the scene under different lighting conditions.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> transform = A.PlanckianJitter(mode="blackbody",
        ...                               temperature_range=(3000, 9000),
        ...                               sampling_method="uniform",
        ...                               p=1.0)
        >>> result = transform(image=image)
        >>> jittered_image = result["image"]

    References:
        - Planck's law: https://en.wikipedia.org/wiki/Planck%27s_law
        - CIE Standard Illuminants: https://en.wikipedia.org/wiki/Standard_illuminant
        - Color temperature: https://en.wikipedia.org/wiki/Color_temperature
        - Implementation inspired by: https://github.com/TheZino/PlanckianJitter

    """

    class InitSchema(BaseTransformInitSchema):
        mode: Literal["blackbody", "cied"]
        temperature_limit: Annotated[tuple[int, int], AfterValidator(nondecreasing)] | None
        sampling_method: Literal["uniform", "gaussian"]

        @model_validator(mode="after")
        def _validate_temperature(self) -> Self:
            max_temp = int(PLANKIAN_JITTER_CONST["MAX_TEMP"])

            if self.temperature_limit is None:
                if self.mode == "blackbody":
                    self.temperature_limit = (
                        int(PLANKIAN_JITTER_CONST["MIN_BLACKBODY_TEMP"]),
                        max_temp,
                    )
                elif self.mode == "cied":
                    self.temperature_limit = (
                        int(PLANKIAN_JITTER_CONST["MIN_CIED_TEMP"]),
                        max_temp,
                    )
            else:
                if self.mode == "blackbody" and (
                    min(self.temperature_limit) < PLANKIAN_JITTER_CONST["MIN_BLACKBODY_TEMP"]
                    or max(self.temperature_limit) > max_temp
                ):
                    raise ValueError(
                        "Temperature limits for blackbody should be in [3000, 15000] range",
                    )
                if self.mode == "cied" and (
                    min(self.temperature_limit) < PLANKIAN_JITTER_CONST["MIN_CIED_TEMP"]
                    or max(self.temperature_limit) > max_temp
                ):
                    raise ValueError(
                        "Temperature limits for CIED should be in [4000, 15000] range",
                    )

                if not self.temperature_limit[0] <= PLANKIAN_JITTER_CONST["WHITE_TEMP"] <= self.temperature_limit[1]:
                    raise ValueError(
                        "White temperature should be within the temperature limits",
                    )

            return self

    def __init__(
        self,
        mode: Literal["blackbody", "cied"] = "blackbody",
        temperature_limit: tuple[int, int] | None = None,
        sampling_method: Literal["uniform", "gaussian"] = "uniform",
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)

        self.mode = mode
        self.temperature_limit = cast("tuple[int, int]", temperature_limit)
        self.sampling_method = sampling_method

    def apply(self, img: ImageType, temperature: int, **params: Any) -> ImageType:
        non_rgb_error(img)
        return fpixel.planckian_jitter(img, temperature, mode=self.mode)

    def apply_to_images(self, images: ImageType, temperature: int, **params: Any) -> ImageType:
        non_rgb_error(images)
        return self.apply(images, temperature, **params)

    def apply_to_volumes(self, volumes: VolumeType, temperature: int, **params: Any) -> VolumeType:
        non_rgb_error(volumes)
        return self.apply(volumes, temperature, **params)

    def get_params(self) -> dict[str, Any]:
        sampling_prob_boundary = PLANKIAN_JITTER_CONST["SAMPLING_TEMP_PROB"]
        sampling_temp_boundary = PLANKIAN_JITTER_CONST["WHITE_TEMP"]

        if self.sampling_method == "uniform":
            # Split into 2 cases to avoid selecting cold temperatures (>6000) too often
            if self.py_random.random() < sampling_prob_boundary:
                temperature = self.py_random.uniform(
                    self.temperature_limit[0],
                    sampling_temp_boundary,
                )
            else:
                temperature = self.py_random.uniform(
                    sampling_temp_boundary,
                    self.temperature_limit[1],
                )
        elif self.sampling_method == "gaussian":
            # Sample values from asymmetric gaussian distribution
            if self.py_random.random() < sampling_prob_boundary:
                # Left side
                shift = np.abs(
                    self.py_random.gauss(
                        0,
                        np.abs(sampling_temp_boundary - self.temperature_limit[0]) / 3,
                    ),
                )
                temperature = sampling_temp_boundary - shift
            else:
                # Right side
                shift = np.abs(
                    self.py_random.gauss(
                        0,
                        np.abs(self.temperature_limit[1] - sampling_temp_boundary) / 3,
                    ),
                )
                temperature = sampling_temp_boundary + shift
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

        # Ensure temperature is within the valid range
        temperature = np.clip(
            temperature,
            self.temperature_limit[0],
            self.temperature_limit[1],
        )

        self.applied_config = {"temperature_limit": int(temperature)}
        return {"temperature": int(temperature)}


class RGBShift(AdditiveNoise):
    """Shift R, G, B with separate ranges. Specialized AdditiveNoise with constant uniform shifts.
    Params: r_shift_limit, g_shift_limit, b_shift_limit.

    A specialized version of AdditiveNoise that applies constant uniform shifts to RGB channels.
    Each channel (R,G,B) can have its own shift range specified.

    Args:
        r_shift_limit ((int, int) or int): Range for shifting the red channel. Options:
            - If tuple (min, max): Sample shift value from this range
            - If int: Sample shift value from (-r_shift_limit, r_shift_limit)
            - For uint8 images: Values represent absolute shifts in [0, 255]
            - For float images: Values represent relative shifts in [0, 1]
            Default: (-20, 20)

        g_shift_limit ((int, int) or int): Range for shifting the green channel. Options:
            - If tuple (min, max): Sample shift value from this range
            - If int: Sample shift value from (-g_shift_limit, g_shift_limit)
            - For uint8 images: Values represent absolute shifts in [0, 255]
            - For float images: Values represent relative shifts in [0, 1]
            Default: (-20, 20)

        b_shift_limit ((int, int) or int): Range for shifting the blue channel. Options:
            - If tuple (min, max): Sample shift value from this range
            - If int: Sample shift value from (-b_shift_limit, b_shift_limit)
            - For uint8 images: Values represent absolute shifts in [0, 255]
            - For float images: Values represent relative shifts in [0, 1]
            Default: (-20, 20)

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Note:
        - Values are shifted independently for each channel
        - For uint8 images:
            * Input ranges like (-20, 20) represent pixel value shifts
            * A shift of 20 means adding 20 to that channel
            * Final values are clipped to [0, 255]
        - For float32 images:
            * Input ranges like (-0.1, 0.1) represent relative shifts
            * A shift of 0.1 means adding 0.1 to that channel
            * Final values are clipped to [0, 1]

    Examples:
        >>> import numpy as np
        >>> import albumentations as A

        # Shift RGB channels of uint8 image
        >>> transform = A.RGBShift(
        ...     r_shift_limit=30,  # Will sample red shift from [-30, 30]
        ...     g_shift_limit=(-20, 20),  # Will sample green shift from [-20, 20]
        ...     b_shift_limit=(-10, 10),  # Will sample blue shift from [-10, 10]
        ...     p=1.0
        ... )
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> shifted = transform(image=image)["image"]

        # Same effect using AdditiveNoise
        >>> transform = A.AdditiveNoise(
        ...     noise_type="uniform",
        ...     spatial_mode="constant",  # One value per channel
        ...     noise_params={
        ...         "ranges": [(-30/255, 30/255), (-20/255, 20/255), (-10/255, 10/255)]
        ...     },
        ...     p=1.0
        ... )

    See Also:
        - AdditiveNoise: More general noise transform with various options:
            * Different noise distributions (uniform, gaussian, laplace, beta)
            * Spatial modes (constant, per-pixel, shared)
            * Approximation for faster computation
        - RandomToneCurve: For non-linear color transformations
        - RandomBrightnessContrast: For combined brightness and contrast adjustments
        - PlankianJitter: For color temperature adjustments
        - HueSaturationValue: For HSV color space adjustments
        - ColorJitter: For combined brightness, contrast, saturation adjustments

    """

    class InitSchema(BaseTransformInitSchema):
        r_shift_limit: Annotated[
            tuple[float, float] | float,
            AfterValidator(create_symmetric_range),
        ]
        g_shift_limit: Annotated[
            tuple[float, float] | float,
            AfterValidator(create_symmetric_range),
        ]
        b_shift_limit: Annotated[
            tuple[float, float] | float,
            AfterValidator(create_symmetric_range),
        ]

    def __init__(
        self,
        r_shift_limit: tuple[float, float] | float = (-20, 20),
        g_shift_limit: tuple[float, float] | float = (-20, 20),
        b_shift_limit: tuple[float, float] | float = (-20, 20),
        p: float = 0.5,
    ):
        # Convert RGB shift limits to normalized ranges if needed
        def normalize_range(limit: tuple[float, float]) -> tuple[float, float]:
            # If any value is > 1, assume uint8 range and normalize
            if abs(limit[0]) > 1 or abs(limit[1]) > 1:
                return (limit[0] / 255.0, limit[1] / 255.0)
            return limit

        ranges = [
            normalize_range(cast("tuple[float, float]", r_shift_limit)),
            normalize_range(cast("tuple[float, float]", g_shift_limit)),
            normalize_range(cast("tuple[float, float]", b_shift_limit)),
        ]

        # Initialize with fixed noise type and spatial mode
        super().__init__(
            noise_type="uniform",
            spatial_mode="constant",
            noise_params={"ranges": ranges},
            approximation=1.0,
            p=p,
        )

        # Store original limits for get_transform_init_args
        self.r_shift_limit = cast("tuple[float, float]", r_shift_limit)
        self.g_shift_limit = cast("tuple[float, float]", g_shift_limit)
        self.b_shift_limit = cast("tuple[float, float]", b_shift_limit)


class PlasmaBrightnessContrast(ImageOnlyTransform):
    """Plasma fractal (Diamond-Square) pattern varies brightness and contrast spatially.
    brightness_range, contrast_range. Organic, non-uniform look.

    Uses Diamond-Square algorithm to generate organic-looking fractal patterns
    that create spatially-varying brightness and contrast adjustments.

    Args:
        brightness_range ((float, float)): Range for brightness adjustment strength.
            Values between -1 and 1:
            - Positive values increase brightness
            - Negative values decrease brightness
            - 0 means no brightness change
            Default: (-0.3, 0.3)

        contrast_range ((float, float)): Range for contrast adjustment strength.
            Values between -1 and 1:
            - Positive values increase contrast
            - Negative values decrease contrast
            - 0 means no contrast change
            Default: (-0.3, 0.3)

        plasma_size (int): Size of the initial plasma pattern grid.
            Larger values create more detailed patterns but are slower to compute.
            The pattern will be resized to match the input image dimensions.
            Default: 256

        roughness (float): Controls how quickly the noise amplitude increases at each iteration.
            Must be greater than 0:
            - Low values (< 1.0): Smoother, more gradual pattern
            - Medium values (~2.0): Natural-looking pattern
            - High values (> 3.0): Very rough, noisy pattern
            Default: 3.0

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Note:
        - Works with any number of channels (grayscale, RGB, multispectral)
        - The same plasma pattern is applied to all channels
        - Operations are performed in float32 precision
        - Final values are clipped to valid range [0, max_value]

    Mathematical Formulation:
        1. Plasma Pattern Generation (Diamond-Square Algorithm):
           Starting with a 3x3 grid of random values in [-1, 1], iteratively:
           a) Diamond Step: For each 2x2 cell, compute center using diamond kernel:
              [[0.25, 0.0, 0.25],
               [0.0,  0.0, 0.0 ],
               [0.25, 0.0, 0.25]]

           b) Square Step: Fill remaining points using square kernel:
              [[0.0,  0.25, 0.0 ],
               [0.25, 0.0,  0.25],
               [0.0,  0.25, 0.0 ]]

           c) Add random noise scaled by roughness^iteration

           d) Normalize final pattern P to [0,1] range using min-max normalization

        2. Brightness Adjustment:
           For each pixel (x,y):
           O(x,y) = I(x,y) + b·P(x,y)
           where:
           - I is the input image
           - b is the brightness factor
           - P is the normalized plasma pattern

        3. Contrast Adjustment:
           For each pixel (x,y):
           O(x,y) = I(x,y)·(1 + c·P(x,y)) + μ·(1 - (1 + c·P(x,y)))
           where:
           - I is the input image
           - c is the contrast factor
           - P is the normalized plasma pattern
           - μ is the mean pixel value

    Examples:
        >>> import albumentations as A
        >>> import numpy as np

        # Default parameters
        >>> transform = A.PlasmaBrightnessContrast(p=1.0)

        # Custom adjustments
        >>> transform = A.PlasmaBrightnessContrast(
        ...     brightness_range=(-0.5, 0.5),
        ...     contrast_range=(-0.3, 0.3),
        ...     plasma_size=512,    # More detailed pattern
        ...     roughness=0.7,      # Smoother transitions
        ...     p=1.0
        ... )

    References:
        - Fournier, Fussell, and Carpenter, "Computer rendering of stochastic models,": Communications of
            the ACM, 1982. Paper introducing the Diamond-Square algorithm.
        - Diamond-Square algorithm: https://en.wikipedia.org/wiki/Diamond-square_algorithm

    See Also:
        - RandomBrightnessContrast: For uniform brightness/contrast adjustments
        - CLAHE: For contrast limited adaptive histogram equalization
        - FancyPCA: For color-based contrast enhancement
        - HistogramMatching: For reference-based contrast adjustment

    """

    class InitSchema(BaseTransformInitSchema):
        brightness_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(-1, 1)),
        ]
        contrast_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(-1, 1)),
        ]
        plasma_size: int = Field(ge=1)
        roughness: float = Field(gt=0)

    def __init__(
        self,
        brightness_range: tuple[float, float] = (-0.3, 0.3),
        contrast_range: tuple[float, float] = (-0.3, 0.3),
        plasma_size: int = 256,
        roughness: float = 3.0,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.plasma_size = plasma_size
        self.roughness = roughness

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        shape = params["shape"]

        # Sample adjustment strengths
        brightness = self.py_random.uniform(*self.brightness_range)
        contrast = self.py_random.uniform(*self.contrast_range)

        self.applied_config = {"brightness_range": brightness, "contrast_range": contrast}

        # Generate plasma pattern
        plasma = fpixel.generate_plasma_pattern(
            target_shape=shape[:2],
            roughness=self.roughness,
            random_generator=self.random_generator,
        )

        return {
            "brightness_factor": brightness,
            "contrast_factor": contrast,
            "plasma_pattern": plasma,
        }

    def apply(
        self,
        img: ImageType,
        brightness_factor: float,
        contrast_factor: float,
        plasma_pattern: np.ndarray,
        **params: Any,
    ) -> ImageType:
        return fpixel.apply_plasma_brightness_contrast(
            img,
            brightness_factor,
            contrast_factor,
            plasma_pattern,
        )

    @batch_transform("spatial")
    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self.apply(images, **params)

    @batch_transform("spatial", keep_depth_dim=True)
    def apply_to_volumes(self, volumes: VolumeType, **params: Any) -> VolumeType:
        return self.apply(volumes, **params)


class PlasmaShadow(ImageOnlyTransform):
    """Plasma fractal (Diamond-Square) shadow: organic darkening. shadow_intensity_range, roughness.
    Good for natural shading and lighting variation.

    Creates organic-looking shadows using plasma fractal noise pattern.
    The shadow intensity varies smoothly across the image, creating natural-looking
    darkening effects that can simulate shadows, shading, or lighting variations.

    Args:
        shadow_intensity_range (tuple[float, float]): Range for shadow intensity.
            Values between 0 and 1:
            - 0 means no shadow (original image)
            - 1 means maximum darkening (black)
            - Values between create partial shadows
            Default: (0.3, 0.7)

        roughness (float): Controls how quickly the noise amplitude increases at each iteration.
            Must be greater than 0:
            - Low values (< 1.0): Smoother, more gradual shadows
            - Medium values (~2.0): Natural-looking shadows
            - High values (> 3.0): Very rough, noisy shadows
            Default: 3.0

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Note:
        - The transform darkens the image using a plasma pattern
        - Works with any number of channels (grayscale, RGB, multispectral)
        - Shadow pattern is generated using Diamond-Square algorithm with specific kernels
        - The same shadow pattern is applied to all channels
        - Final values are clipped to valid range [0, max_value]

    Mathematical Formulation:
        1. Plasma Pattern Generation (Diamond-Square Algorithm):
           Starting with a 3x3 grid of random values in [-1, 1], iteratively:
           a) Diamond Step: For each 2x2 cell, compute center using diamond kernel:
              [[0.25, 0.0, 0.25],
               [0.0,  0.0, 0.0 ],
               [0.25, 0.0, 0.25]]

           b) Square Step: Fill remaining points using square kernel:
              [[0.0,  0.25, 0.0 ],
               [0.25, 0.0,  0.25],
               [0.0,  0.25, 0.0 ]]

           c) Add random noise scaled by roughness^iteration

           d) Normalize final pattern P to [0,1] range using min-max normalization

        2. Shadow Application:
           For each pixel (x,y):
           O(x,y) = I(x,y) * (1 - i*P(x,y))
           where:
           - I is the input image
           - P is the normalized plasma pattern
           - i is the sampled shadow intensity
           - O is the output image

    Examples:
        >>> import albumentations as A
        >>> import numpy as np

        # Default parameters for natural shadows
        >>> transform = A.PlasmaShadow(p=1.0)

        # Subtle, smooth shadows
        >>> transform = A.PlasmaShadow(
        ...     shadow_intensity_range=(0.1, 0.3),
        ...     roughness=0.7,
        ...     p=1.0
        ... )

        # Dramatic, detailed shadows
        >>> transform = A.PlasmaShadow(
        ...     shadow_intensity_range=(0.5, 0.9),
        ...     roughness=0.3,
        ...     p=1.0
        ... )

    References:
        - Fournier, Fussell, and Carpenter, "Computer rendering of stochastic models,": Communications of
            the ACM, 1982. Paper introducing the Diamond-Square algorithm.
        - Diamond-Square algorithm: https://en.wikipedia.org/wiki/Diamond-square_algorithm

    See Also:
        - PlasmaBrightnessContrast: For brightness/contrast adjustments using plasma patterns
        - RandomShadow: For geometric shadow effects
        - RandomToneCurve: For global lighting adjustments
        - PlasmaBrightnessContrast: For brightness/contrast adjustments using plasma patterns

    """

    class InitSchema(BaseTransformInitSchema):
        shadow_intensity_range: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, 1))]
        roughness: float = Field(gt=0)

    def __init__(
        self,
        shadow_intensity_range: tuple[float, float] = (0.3, 0.7),
        plasma_size: int = 256,
        roughness: float = 3.0,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.shadow_intensity_range = shadow_intensity_range
        self.plasma_size = plasma_size
        self.roughness = roughness

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        shape = params["shape"]

        # Sample shadow intensity
        intensity = self.py_random.uniform(*self.shadow_intensity_range)

        self.applied_config = {"shadow_intensity_range": intensity}

        # Generate plasma pattern
        plasma = fpixel.generate_plasma_pattern(
            target_shape=shape[:2],
            roughness=self.roughness,
            random_generator=self.random_generator,
        )

        return {
            "intensity": intensity,
            "plasma_pattern": plasma,
        }

    def apply(
        self,
        img: ImageType,
        intensity: float,
        plasma_pattern: np.ndarray,
        **params: Any,
    ) -> ImageType:
        return fpixel.apply_plasma_shadow(img, intensity, plasma_pattern)

    @batch_transform("spatial")
    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self.apply(images, **params)

    @batch_transform("spatial", keep_depth_dim=True)
    def apply_to_volumes(self, volumes: VolumeType, **params: Any) -> VolumeType:
        return self.apply(volumes, **params)


class Illumination(ImageOnlyTransform):
    """Illumination patterns: directional (linear), corner shadows/highlights, or gaussian.
    mode and params control shape and strength. Simulates lighting variation.

    This transform simulates different lighting conditions by applying controlled
    illumination patterns. It can create effects like:
    - Directional lighting (linear mode)
    - Corner shadows/highlights (corner mode)
    - Spotlights or local lighting (gaussian mode)

    These effects can be used to:
    - Simulate natural lighting variations
    - Add dramatic lighting effects
    - Create synthetic shadows or highlights
    - Augment training data with different lighting conditions

    Args:
        mode (Literal['linear', 'corner', 'gaussian']): Type of illumination pattern:
            - 'linear': Creates a smooth gradient across the image,
                       simulating directional lighting like sunlight
                       through a window
            - 'corner': Applies gradient from any corner,
                       simulating light source from a corner
            - 'gaussian': Creates a circular spotlight effect,
                         simulating local light sources
            Default: 'linear'

        intensity_range (tuple[float, float]): Range for effect strength.
            Values between 0.01 and 0.2:
            - 0.01-0.05: Subtle lighting changes
            - 0.05-0.1: Moderate lighting effects
            - 0.1-0.2: Strong lighting effects
            Default: (0.01, 0.2)

        effect_type (str): Type of lighting change:
            - 'brighten': Only adds light (like a spotlight)
            - 'darken': Only removes light (like a shadow)
            - 'both': Randomly chooses between brightening and darkening
            Default: 'both'

        angle_range (tuple[float, float]): Range for gradient angle in degrees.
            Controls direction of linear gradient:
            - 0°: Left to right
            - 90°: Top to bottom
            - 180°: Right to left
            - 270°: Bottom to top
            Only used for 'linear' mode.
            Default: (0, 360)

        center_range (tuple[float, float]): Range for spotlight position.
            Values between 0 and 1 representing relative position:
            - (0, 0): Top-left corner
            - (1, 1): Bottom-right corner
            - (0.5, 0.5): Center of image
            Only used for 'gaussian' mode.
            Default: (0.1, 0.9)

        sigma_range (tuple[float, float]): Range for spotlight size.
            Values between 0.2 and 1.0:
            - 0.2: Small, focused spotlight
            - 0.5: Medium-sized light area
            - 1.0: Broad, soft lighting
            Only used for 'gaussian' mode.
            Default: (0.2, 1.0)

        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image, volume

    Image types:
        uint8, float32

    Examples:
        >>> import albumentations as A
        >>> # Simulate sunlight through window
        >>> transform = A.Illumination(
        ...     mode='linear',
        ...     intensity_range=(0.05, 0.1),
        ...     effect_type='brighten',
        ...     angle_range=(30, 60)
        ... )
        >>>
        >>> # Create dramatic corner shadow
        >>> transform = A.Illumination(
        ...     mode='corner',
        ...     intensity_range=(0.1, 0.2),
        ...     effect_type='darken'
        ... )
        >>>
        >>> # Add multiple spotlights
        >>> transform1 = A.Illumination(
        ...     mode='gaussian',
        ...     intensity_range=(0.05, 0.15),
        ...     effect_type='brighten',
        ...     center_range=(0.2, 0.4),
        ...     sigma_range=(0.2, 0.3)
        ... )
        >>> transform2 = A.Illumination(
        ...     mode='gaussian',
        ...     intensity_range=(0.05, 0.15),
        ...     effect_type='darken',
        ...     center_range=(0.6, 0.8),
        ...     sigma_range=(0.3, 0.5)
        ... )
        >>> transforms = A.Compose([transform1, transform2])

    References:
        - Lighting in Computer Vision:
          https://en.wikipedia.org/wiki/Lighting_in_computer_vision

        - Image-based lighting:
          https://en.wikipedia.org/wiki/Image-based_lighting

        - Similar implementation in Kornia:
          https://kornia.readthedocs.io/en/latest/augmentation.html#randomlinearillumination

        - Research on lighting augmentation:
          "Learning Deep Representations of Fine-grained Visual Descriptions"
          https://arxiv.org/abs/1605.05395

        - Photography lighting patterns:
          https://en.wikipedia.org/wiki/Lighting_pattern

    Note:
        - The transform preserves image range and dtype
        - Effects are applied multiplicatively to preserve texture
        - Can be combined with other transforms for complex lighting scenarios
        - Useful for training models to be robust to lighting variations

    """

    class InitSchema(BaseTransformInitSchema):
        mode: Literal["linear", "corner", "gaussian"]
        intensity_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0.01, 0.2)),
        ]
        effect_type: Literal["brighten", "darken", "both"]
        angle_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 360)),
        ]
        center_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
        ]
        sigma_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0.2, 1.0)),
        ]

    def __init__(
        self,
        mode: Literal["linear", "corner", "gaussian"] = "linear",
        intensity_range: tuple[float, float] = (0.01, 0.2),
        effect_type: Literal["brighten", "darken", "both"] = "both",
        angle_range: tuple[float, float] = (0, 360),
        center_range: tuple[float, float] = (0.1, 0.9),
        sigma_range: tuple[float, float] = (0.2, 1.0),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.mode = mode
        self.intensity_range = intensity_range
        self.effect_type = effect_type
        self.angle_range = angle_range
        self.center_range = center_range
        self.sigma_range = sigma_range

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        intensity = self.py_random.uniform(*self.intensity_range)

        # Determine if brightening or darkening
        sign = 1  # brighten
        if self.effect_type == "both":
            sign = 1 if self.py_random.random() > 0.5 else -1
        elif self.effect_type == "darken":
            sign = -1

        intensity *= sign

        if self.mode == "linear":
            angle = self.py_random.uniform(*self.angle_range)
            self.applied_config = {"intensity_range": abs(intensity), "angle_range": angle}
            return {
                "intensity": intensity,
                "angle": angle,
            }
        if self.mode == "corner":
            corner = self.py_random.randint(0, 3)  # Choose random corner
            self.applied_config = {"intensity_range": abs(intensity)}
            return {
                "intensity": intensity,
                "corner": corner,
            }

        x = self.py_random.uniform(*self.center_range)
        y = self.py_random.uniform(*self.center_range)
        sigma = self.py_random.uniform(*self.sigma_range)
        self.applied_config = {"intensity_range": abs(intensity), "center_range": (x, y), "sigma_range": sigma}
        return {
            "intensity": intensity,
            "center": (x, y),
            "sigma": sigma,
        }

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        if self.mode == "linear":
            return fpixel.apply_linear_illumination(
                img,
                intensity=params["intensity"],
                angle=params["angle"],
            )
        if self.mode == "corner":
            return fpixel.apply_corner_illumination(
                img,
                intensity=params["intensity"],
                corner=params["corner"],
            )

        return fpixel.apply_gaussian_illumination(
            img,
            intensity=params["intensity"],
            center=params["center"],
            sigma=params["sigma"],
        )

    def apply_to_images(self, images: ImageType, *args: Any, **params: Any) -> ImageType:
        height, width = images.shape[1], images.shape[2]
        gradient = fpixel.create_illumination_gradient(
            height,
            width,
            self.mode,
            params,
        )
        gradient = gradient[..., np.newaxis]

        return self._apply_to_batch_same_shape(images, lambda image: albucore.multiply_by_array(image, gradient))


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


class HEStain(ImageOnlyTransform):
    """H&E stain augmentation for histopathology. method: preset, random_preset, vahadane, macenko.
    Simulates staining variation for robust pathology models.

    This transform simulates different H&E staining conditions using either:
    1. Predefined stain matrices (8 standard references)
    2. Vahadane method for stain extraction
    3. Macenko method for stain extraction
    4. Custom stain matrices

    Args:
        method(Literal['preset', 'random_preset', 'vahadane', 'macenko']): Method to use for stain augmentation:
            - "preset": Use predefined stain matrices
            - "random_preset": Randomly select a preset matrix each time
            - "vahadane": Extract using Vahadane method
            - "macenko": Extract using Macenko method
            Default: "preset"

        preset(str | None): Preset stain matrix to use when method="preset":
            - "ruifrok": Standard reference from Ruifrok & Johnston
            - "macenko": Reference from Macenko's method
            - "standard": Typical bright-field microscopy
            - "high_contrast": Enhanced contrast
            - "h_heavy": Hematoxylin dominant
            - "e_heavy": Eosin dominant
            - "dark": Darker staining
            - "light": Lighter staining
            Default: "standard"

        intensity_scale_range(tuple[float, float]): Range for multiplicative stain intensity variation.
            Values are multipliers between 0.5 and 1.5. For example:
            - (0.7, 1.3) means stain intensities will vary from 70% to 130%
            - (0.9, 1.1) gives subtle variations
            - (0.5, 1.5) gives dramatic variations
            Default: (0.7, 1.3)

        intensity_shift_range(tuple[float, float]): Range for additive stain intensity variation.
            Values between -0.3 and 0.3. For example:
            - (-0.2, 0.2) means intensities will be shifted by -20% to +20%
            - (-0.1, 0.1) gives subtle shifts
            - (-0.3, 0.3) gives dramatic shifts
            Default: (-0.2, 0.2)

        augment_background(bool): Whether to apply augmentation to background regions.
            Default: False

    Targets:
        image, volume

    Number of channels:
        3

    Image types:
        uint8, float32

    References:
        - A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical": Analytical and quantitative
            cytology and histology, 2001.
        - M. Macenko et al., "A method for normalizing histology slides for: 2009 IEEE International Symposium on
            quantitative analysis," 2009 IEEE International Symposium on Biomedical Imaging, 2009.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create a sample H&E stained histopathology image
        >>> # For real use cases, load an actual H&E stained image
        >>> image = np.zeros((300, 300, 3), dtype=np.uint8)
        >>> # Simulate tissue regions with different staining patterns
        >>> image[50:150, 50:150] = np.array([120, 140, 180], dtype=np.uint8)  # Hematoxylin-rich region
        >>> image[150:250, 150:250] = np.array([140, 160, 120], dtype=np.uint8)  # Eosin-rich region
        >>>
        >>> # Example 1: Using a specific preset stain matrix
        >>> transform = A.HEStain(
        ...     method="preset",
        ...     preset="standard",
        ...     intensity_scale_range=(0.8, 1.2),
        ...     intensity_shift_range=(-0.1, 0.1),
        ...     augment_background=False,
        ...     p=1.0
        ... )
        >>> result = transform(image=image)
        >>> transformed_image = result['image']
        >>>
        >>> # Example 2: Using random preset selection
        >>> transform = A.HEStain(
        ...     method="random_preset",
        ...     intensity_scale_range=(0.7, 1.3),
        ...     intensity_shift_range=(-0.15, 0.15),
        ...     p=1.0
        ... )
        >>> result = transform(image=image)
        >>> transformed_image = result['image']
        >>>
        >>> # Example 3: Using Vahadane method (requires H&E stained input)
        >>> transform = A.HEStain(
        ...     method="vahadane",
        ...     intensity_scale_range=(0.7, 1.3),
        ...     p=1.0
        ... )
        >>> result = transform(image=image)
        >>> transformed_image = result['image']
        >>>
        >>> # Example 4: Using Macenko method (requires H&E stained input)
        >>> transform = A.HEStain(
        ...     method="macenko",
        ...     intensity_scale_range=(0.7, 1.3),
        ...     intensity_shift_range=(-0.2, 0.2),
        ...     p=1.0
        ... )
        >>> result = transform(image=image)
        >>> transformed_image = result['image']
        >>>
        >>> # Example 5: Combining with other transforms in a pipeline
        >>> transform = A.Compose([
        ...     A.HEStain(method="preset", preset="high_contrast", p=1.0),
        ...     A.RandomBrightnessContrast(p=0.5),
        ... ])
        >>> result = transform(image=image)
        >>> transformed_image = result['image']

    """

    class InitSchema(BaseTransformInitSchema):
        method: Literal["preset", "random_preset", "vahadane", "macenko"]
        preset: (
            Literal[
                "ruifrok",
                "macenko",
                "standard",
                "high_contrast",
                "h_heavy",
                "e_heavy",
                "dark",
                "light",
            ]
            | None
        )
        intensity_scale_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, None)),
        ]
        intensity_shift_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-1, 1)),
        ]
        augment_background: bool

        @model_validator(mode="after")
        def _validate_matrix_selection(self) -> Self:
            if self.method == "preset" and self.preset is None:
                self.preset = "standard"
            elif self.method == "random_preset" and self.preset is not None:
                raise ValueError("preset should not be specified when method='random_preset'")
            return self

    def __init__(
        self,
        method: Literal["preset", "random_preset", "vahadane", "macenko"] = "random_preset",
        preset: Literal[
            "ruifrok",
            "macenko",
            "standard",
            "high_contrast",
            "h_heavy",
            "e_heavy",
            "dark",
            "light",
        ]
        | None = None,
        intensity_scale_range: tuple[float, float] = (0.7, 1.3),
        intensity_shift_range: tuple[float, float] = (-0.2, 0.2),
        augment_background: bool = False,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.method = method
        self.preset = preset
        self.intensity_scale_range = intensity_scale_range
        self.intensity_shift_range = intensity_shift_range
        self.augment_background = augment_background
        self.stain_normalizer = None

        # Initialize stain extractor here if needed
        if method in ["vahadane", "macenko"]:
            self.stain_extractor = fpixel.get_normalizer(
                cast("Literal['vahadane', 'macenko']", method),
            )

        self.preset_names = [
            "ruifrok",
            "macenko",
            "standard",
            "high_contrast",
            "h_heavy",
            "e_heavy",
            "dark",
            "light",
        ]

    def _get_stain_matrix(self, img: ImageType) -> np.ndarray:
        """Return stain matrix for HEStain: from preset, random_preset, or vahadane/macenko
        extraction from img. Determines per-call stain appearance.
        """
        if self.method == "preset" and self.preset is not None:
            return fpixel.STAIN_MATRICES[self.preset]
        if self.method == "random_preset":
            random_preset = self.py_random.choice(self.preset_names)
            return fpixel.STAIN_MATRICES[random_preset]
        # vahadane or macenko
        self.stain_extractor.fit(img)
        return self.stain_extractor.stain_matrix_target

    def apply(
        self,
        img: ImageType,
        stain_matrix: np.ndarray,
        scale_factors: np.ndarray,
        shift_values: np.ndarray,
        **params: Any,
    ) -> ImageType:
        non_rgb_error(img)
        return fpixel.apply_he_stain_augmentation(
            img=img,
            stain_matrix=stain_matrix,
            scale_factors=scale_factors,
            shift_values=shift_values,
            augment_background=self.augment_background,
        )

    @batch_transform("channel")
    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self.apply(images, **params)

    @batch_transform("channel")
    def apply_to_volumes(self, volumes: VolumeType, **params: Any) -> VolumeType:
        return self.apply(volumes, **params)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        # Get stain matrix
        if "image" in data:
            image = data["image"]
        elif "images" in data:
            image = data["images"][0]
        elif "volume" in data:
            image = data["volume"][0]
        elif "volumes" in data:
            image = data["volumes"][0][0]

        stain_matrix = self._get_stain_matrix(image)

        # Generate random scaling and shift parameters for both H&E channels
        scale_h = self.py_random.uniform(*self.intensity_scale_range)
        scale_e = self.py_random.uniform(*self.intensity_scale_range)
        shift_h = self.py_random.uniform(*self.intensity_shift_range)
        shift_e = self.py_random.uniform(*self.intensity_shift_range)

        scale_factors = np.array([scale_h, scale_e])
        shift_values = np.array([shift_h, shift_e])

        self.applied_config = {
            "intensity_scale_range": (scale_h, scale_e),
            "intensity_shift_range": (shift_h, shift_e),
        }

        return {
            "stain_matrix": stain_matrix,
            "scale_factors": scale_factors,
            "shift_values": shift_values,
        }


class PhotoMetricDistort(ImageOnlyTransform):
    """SSD-style photometric distortion: brightness, contrast, saturation, hue, channel shuffle; each
    with probability distort_p. For detection training.

    Applies brightness, contrast, saturation, and hue adjustments independently with probability
    `distort_p` each. Contrast is applied either before or after the HSV-space adjustments
    (randomly chosen). Optionally permutes channels with probability `distort_p`.

    This mirrors the `RandomPhotometricDistort` transform from torchvision but uses our
    existing `adjust_*_torchvision` functional primitives.

    Args:
        brightness_range (tuple[float, float]): Multiplicative factor range for brightness.
            Factor is drawn uniformly from this range. Must be non-negative.
            Default: `(0.875, 1.125)`.
        contrast_range (tuple[float, float]): Multiplicative factor range for contrast.
            Factor is drawn uniformly from this range. Must be non-negative.
            Default: `(0.5, 1.5)`.
        saturation_range (tuple[float, float]): Multiplicative factor range for saturation.
            Factor is drawn uniformly from this range. Must be non-negative.
            Default: `(0.5, 1.5)`.
        hue_range (tuple[float, float]): Additive factor range for hue.
            Factor is drawn uniformly from this range. Must be in `[-0.5, 0.5]`.
            Default: `(-0.05, 0.05)`.
        distort_p (float): Probability of applying each individual distortion (brightness,
            contrast, saturation, hue, channel permutation). Default: `0.5`.
        p (float): Probability of applying the overall transform. Default: `0.5`.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        1, 3

    Note:
        - Each of the five distortions (brightness, contrast, saturation, hue, channel shuffle)
          is applied independently with probability `distort_p`.
        - Contrast is randomly applied either before or after saturation/hue adjustment.
        - For single-channel images, saturation and hue adjustments have no effect.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        >>> bbox_labels = [1]
        >>> keypoints = np.array([[20, 30]], dtype=np.float32)
        >>> keypoint_labels = [0]
        >>>
        >>> transform = A.Compose([
        ...     A.PhotoMetricDistort(
        ...         brightness_range=(0.875, 1.125),
        ...         contrast_range=(0.5, 1.5),
        ...         saturation_range=(0.5, 1.5),
        ...         hue_range=(-0.05, 0.05),
        ...         distort_p=0.5,
        ...         p=1.0,
        ...     )
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> result = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels,
        ... )
        >>> transformed_image = result['image']

    References:
        - SSD: https://arxiv.org/abs/1512.02325
        - torchvision RandomPhotometricDistort:
          https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomPhotometricDistort.html

    """

    class InitSchema(BaseTransformInitSchema):
        brightness_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        contrast_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        saturation_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        hue_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(-0.5, 0.5)),
            AfterValidator(nondecreasing),
        ]
        distort_p: float = Field(ge=0.0, le=1.0)

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.875, 1.125),
        contrast_range: tuple[float, float] = (0.5, 1.5),
        saturation_range: tuple[float, float] = (0.5, 1.5),
        hue_range: tuple[float, float] = (-0.05, 0.05),
        distort_p: float = 0.5,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.distort_p = distort_p

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        shape = params["shape"]
        num_channels = 1 if len(shape) == 2 else shape[-1]

        brightness_factor = (
            self.py_random.uniform(*self.brightness_range) if self.py_random.random() < self.distort_p else None
        )
        contrast_factor = (
            self.py_random.uniform(*self.contrast_range) if self.py_random.random() < self.distort_p else None
        )
        saturation_factor = (
            self.py_random.uniform(*self.saturation_range) if self.py_random.random() < self.distort_p else None
        )
        hue_factor = self.py_random.uniform(*self.hue_range) if self.py_random.random() < self.distort_p else None
        # contrast_before controls where contrast sits relative to sat/hue; brightness always precedes contrast
        contrast_before = self.py_random.random() < 0.5

        if self.py_random.random() < self.distort_p and num_channels > 1:
            ch_arr = list(range(num_channels))
            self.py_random.shuffle(ch_arr)
            channel_permutation: list[int] | None = ch_arr
        else:
            channel_permutation = None

        applied: dict[str, Any] = {}
        if brightness_factor is not None:
            applied["brightness_range"] = brightness_factor
        if contrast_factor is not None:
            applied["contrast_range"] = contrast_factor
        if saturation_factor is not None:
            applied["saturation_range"] = saturation_factor
        if hue_factor is not None:
            applied["hue_range"] = hue_factor
        self.applied_config = applied

        return {
            "brightness_factor": brightness_factor,
            "contrast_factor": contrast_factor,
            "saturation_factor": saturation_factor,
            "hue_factor": hue_factor,
            "contrast_before": contrast_before,
            "channel_permutation": channel_permutation,
        }

    def _apply_brightness_contrast_before(
        self,
        img: ImageType,
        brightness_factor: float | None,
        contrast_factor: float | None,
    ) -> ImageType:
        if brightness_factor is not None and contrast_factor is not None:
            return fpixel.apply_brightness_contrast_torchvision(
                img,
                brightness_factor,
                contrast_factor,
                brightness_first=True,
            )
        if brightness_factor is not None:
            return fpixel.adjust_brightness_torchvision(img, brightness_factor)
        if contrast_factor is not None:
            return fpixel.adjust_contrast_torchvision(img, contrast_factor)
        return img

    def apply(
        self,
        img: ImageType,
        brightness_factor: float | None,
        contrast_factor: float | None,
        saturation_factor: float | None,
        hue_factor: float | None,
        contrast_before: bool,
        channel_permutation: list[int] | None,
        **params: Any,
    ) -> ImageType:
        if not is_rgb_image(img) and not is_grayscale_image(img):
            msg = "PhotoMetricDistort expects 1-channel or 3-channel images."
            raise TypeError(msg)

        if contrast_before:
            img = self._apply_brightness_contrast_before(
                img,
                brightness_factor,
                contrast_factor,
            )
        elif brightness_factor is not None:
            img = fpixel.adjust_brightness_torchvision(img, brightness_factor)

        if saturation_factor is not None:
            img = fpixel.adjust_saturation_torchvision(img, saturation_factor)
        if hue_factor is not None:
            img = fpixel.adjust_hue_torchvision(img, hue_factor)

        if not contrast_before and contrast_factor is not None:
            img = fpixel.adjust_contrast_torchvision(img, contrast_factor)

        if channel_permutation is not None:
            img = fpixel.channel_shuffle(img, channel_permutation)
        return img

    def apply_to_images(self, images: ImageType, *args: Any, **params: Any) -> ImageType:
        return self._apply_to_batch_same_shape(images, lambda image: self.apply(image, **params))


class Vignetting(ImageOnlyTransform):
    """Darken corners with a radial (elliptical) gradient. Simulates lens vignetting or
    natural light falloff. Use for lens realism or stylistic darkening.

    Center of the image stays bright; corners and edges are darkened. Center position
    can be jittered for variety.

    Args:
        intensity_range (tuple[float, float]): Darkening at corners: 0 = no effect, 1 = black.
            Default: (0.2, 0.5).
        center_range (tuple[float, float]): Range for vignette center as fraction of width/height.
            (0.5, 0.5) = image center. Default: (0.3, 0.7).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - Elliptical gradient centered at a random point (within center_range).
        - Quadratic falloff from center to edges.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>>
        >>> transform = A.Vignetting(intensity_range=(0.2, 0.5), p=1.0)
        >>> result = transform(image=image)["image"]

    See Also:
        - Halftone: Dot pattern (printing-style) for vintage or print aesthetic.
        - FilmGrain: Luminance-dependent film grain for vintage texture.

    """

    class InitSchema(BaseTransformInitSchema):
        intensity_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        center_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        intensity_range: tuple[float, float] = (0.2, 0.5),
        center_range: tuple[float, float] = (0.3, 0.7),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.intensity_range = intensity_range
        self.center_range = center_range

    def apply(
        self,
        img: ImageType,
        intensity: float,
        center_x: float,
        center_y: float,
        **params: Any,
    ) -> ImageType:
        return fpixel.apply_vignette(img, intensity, center_x, center_y)

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self._apply_to_batch_same_shape(images, lambda image: self.apply(image, **params))

    def get_params(self) -> dict[str, float]:
        intensity = self.py_random.uniform(*self.intensity_range)
        center_x = self.py_random.uniform(*self.center_range)
        center_y = self.py_random.uniform(*self.center_range)
        self.applied_config = {"intensity_range": intensity, "center_range": (center_x, center_y)}
        return {
            "intensity": intensity,
            "center_x": center_x,
            "center_y": center_y,
        }
