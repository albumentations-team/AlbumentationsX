"""Advanced color jitter, channel shift, aberration, stain, and photometric transforms."""

from typing import Annotated, Any, Literal, cast

from typing_extensions import Self

from ._color_shared import (
    AdditiveNoise,
    AfterValidator,
    BaseTransformInitSchema,
    Field,
    ImageOnlyTransform,
    ImageType,
    VolumeType,
    batch_transform,
    check_range_bounds,
    cv2,
    fpixel,
    is_grayscale_image,
    is_rgb_image,
    model_validator,
    non_rgb_error,
    nondecreasing,
    np,
)

ColorRange = tuple[tuple[int, int, int], tuple[int, int, int]]


class ColorJitter(ImageOnlyTransform):
    """Randomly jitter brightness/contrast/saturation/hue in random order. Separate _range per
    effect. Strong color augmentation for classification and detection.

    This transform is similar to torchvision's ColorJitter but with some differences due to the use of OpenCV
    instead of Pillow. The main differences are:
    1. OpenCV and Pillow use different formulas to convert images to HSV format.
    2. This implementation uses value saturation instead of uint8 overflow as in Pillow.

    These differences may result in slightly different output compared to torchvision's ColorJitter.

    Args:
        brightness_range (tuple[float, float]): Range for the brightness factor, sampled per
            image. Both ends should be non-negative. Default: (0.8, 1.2)

        contrast_range (tuple[float, float]): Range for the contrast factor, sampled per image.
            Both ends should be non-negative. Default: (0.8, 1.2)

        saturation_range (tuple[float, float]): Range for the saturation factor, sampled per
            image. Both ends should be non-negative. Default: (0.8, 1.2)

        hue_range (tuple[float, float]): Range for the hue factor, sampled per image. Values
            should be in [-0.5, 0.5]. Default: (-0.5, 0.5)

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
        - The ranges for brightness_range, contrast_range, and saturation_range are applied as multiplicative factors.
        - The range for hue_range is applied as an additive factor.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.ColorJitter(
        ...     brightness_range=(0.8, 1.2),
        ...     contrast_range=(0.8, 1.2),
        ...     saturation_range=(0.8, 1.2),
        ...     hue_range=(-0.1, 0.1),
        ...     p=1.0,
        ... )
        >>> result = transform(image=image)
        >>> jittered_image = result['image']

    References:
        - ColorJitter: https://pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html
        - Color Conversions: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html

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

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        contrast_range: tuple[float, float] = (0.8, 1.2),
        saturation_range: tuple[float, float] = (0.8, 1.2),
        hue_range: tuple[float, float] = (-0.5, 0.5),
        p: float = 0.5,
    ):
        super().__init__(p=p)

        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range

    def get_params(self) -> dict[str, Any]:
        brightness = self.py_random.uniform(*self.brightness_range)
        contrast = self.py_random.uniform(*self.contrast_range)
        saturation = self.py_random.uniform(*self.saturation_range)
        hue = self.py_random.uniform(*self.hue_range)

        self.applied_config = {
            "brightness_range": brightness,
            "contrast_range": contrast,
            "saturation_range": saturation,
            "hue_range": hue,
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
    """Add lateral chromatic aberration: shift red/blue channels relative to green.
    Simulates lens color fringing via primary/secondary distortion ranges.

    Chromatic aberration is an optical effect that occurs when a lens fails to focus all colors to the same point.
    This transform simulates this effect by applying different radial distortions to the red and blue channels
    of the image, while leaving the green channel unchanged.

    Args:
        primary_distortion_range (tuple[float, float]): Range of the primary radial distortion
            coefficient, sampled per image. Controls distortion in the center of the image:
            - Positive values result in pincushion distortion (edges bend inward)
            - Negative values result in barrel distortion (edges bend outward)
            Default: (-0.02, 0.02).

        secondary_distortion_range (tuple[float, float]): Range of the secondary radial
            distortion coefficient, sampled per image. Controls distortion in the corners:
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
        ...     primary_distortion_range=(-0.05, 0.05),
        ...     secondary_distortion_range=(-0.1, 0.1),
        ...     mode='green_purple',
        ...     interpolation=cv2.INTER_LINEAR,
        ...     p=1.0,
        ... )
        >>> transformed = transform(image=image)
        >>> aberrated_image = transformed['image']

    References:
        Chromatic Aberration: https://en.wikipedia.org/wiki/Chromatic_aberration

    """

    class InitSchema(BaseTransformInitSchema):
        primary_distortion_range: tuple[float, float]
        secondary_distortion_range: tuple[float, float]
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
        primary_distortion_range: tuple[float, float] = (-0.02, 0.02),
        secondary_distortion_range: tuple[float, float] = (-0.05, 0.05),
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
        self.primary_distortion_range = primary_distortion_range
        self.secondary_distortion_range = secondary_distortion_range
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
        primary_distortion_red = self.py_random.uniform(*self.primary_distortion_range)
        secondary_distortion_red = self.py_random.uniform(
            *self.secondary_distortion_range,
        )
        primary_distortion_blue = self.py_random.uniform(*self.primary_distortion_range)
        secondary_distortion_blue = self.py_random.uniform(
            *self.secondary_distortion_range,
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
            "primary_distortion_range": (primary_distortion_red, primary_distortion_blue),
            "secondary_distortion_range": (secondary_distortion_red, secondary_distortion_blue),
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

        temperature_range (tuple[int, int] | None): The range of color temperatures (in Kelvin) to sample from.
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
        temperature_range: Annotated[tuple[int, int], AfterValidator(nondecreasing)] | None
        sampling_method: Literal["uniform", "gaussian"]

        @model_validator(mode="after")
        def _validate_temperature(self) -> Self:
            max_temp = int(PLANKIAN_JITTER_CONST["MAX_TEMP"])

            if self.temperature_range is None:
                if self.mode == "blackbody":
                    self.temperature_range = (
                        int(PLANKIAN_JITTER_CONST["MIN_BLACKBODY_TEMP"]),
                        max_temp,
                    )
                elif self.mode == "cied":
                    self.temperature_range = (
                        int(PLANKIAN_JITTER_CONST["MIN_CIED_TEMP"]),
                        max_temp,
                    )
            else:
                if self.mode == "blackbody" and (
                    min(self.temperature_range) < PLANKIAN_JITTER_CONST["MIN_BLACKBODY_TEMP"]
                    or max(self.temperature_range) > max_temp
                ):
                    raise ValueError(
                        "Temperature limits for blackbody should be in [3000, 15000] range",
                    )
                if self.mode == "cied" and (
                    min(self.temperature_range) < PLANKIAN_JITTER_CONST["MIN_CIED_TEMP"]
                    or max(self.temperature_range) > max_temp
                ):
                    raise ValueError(
                        "Temperature limits for CIED should be in [4000, 15000] range",
                    )

                if not self.temperature_range[0] <= PLANKIAN_JITTER_CONST["WHITE_TEMP"] <= self.temperature_range[1]:
                    raise ValueError(
                        "White temperature should be within the temperature limits",
                    )

            return self

    def __init__(
        self,
        mode: Literal["blackbody", "cied"] = "blackbody",
        temperature_range: tuple[int, int] | None = None,
        sampling_method: Literal["uniform", "gaussian"] = "uniform",
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)

        self.mode = mode
        self.temperature_range = cast("tuple[int, int]", temperature_range)
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
                    self.temperature_range[0],
                    sampling_temp_boundary,
                )
            else:
                temperature = self.py_random.uniform(
                    sampling_temp_boundary,
                    self.temperature_range[1],
                )
        elif self.sampling_method == "gaussian":
            # Sample values from asymmetric gaussian distribution
            if self.py_random.random() < sampling_prob_boundary:
                # Left side
                shift = np.abs(
                    self.py_random.gauss(
                        0,
                        np.abs(sampling_temp_boundary - self.temperature_range[0]) / 3,
                    ),
                )
                temperature = sampling_temp_boundary - shift
            else:
                # Right side
                shift = np.abs(
                    self.py_random.gauss(
                        0,
                        np.abs(self.temperature_range[1] - sampling_temp_boundary) / 3,
                    ),
                )
                temperature = sampling_temp_boundary + shift
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

        # Ensure temperature is within the valid range
        temperature = np.clip(
            temperature,
            self.temperature_range[0],
            self.temperature_range[1],
        )

        self.applied_config = {"temperature_range": int(temperature)}
        return {"temperature": int(temperature)}


class RGBShift(AdditiveNoise):
    """Shift R, G, B with separate ranges. Specialized AdditiveNoise with constant uniform shifts.
    Params: r_shift_range, g_shift_range, b_shift_range.

    A specialized version of AdditiveNoise that applies constant uniform shifts to RGB channels.
    Each channel (R,G,B) can have its own shift range specified.

    Args:
        r_shift_range (tuple[int, int]): Range (min, max) for shifting the red channel,
            sampled per image. For uint8 images values are absolute shifts in [0, 255];
            for float images they are relative shifts in [0, 1]. Default: (-20, 20)

        g_shift_range (tuple[int, int]): Range (min, max) for shifting the green channel,
            sampled per image. Same units as r_shift_range. Default: (-20, 20)

        b_shift_range (tuple[int, int]): Range (min, max) for shifting the blue channel,
            sampled per image. Same units as r_shift_range. Default: (-20, 20)

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
        ...     r_shift_range=(-30, 30),  # Will sample red shift from [-30, 30]
        ...     g_shift_range=(-20, 20),  # Will sample green shift from [-20, 20]
        ...     b_shift_range=(-10, 10),  # Will sample blue shift from [-10, 10]
        ...     p=1.0,
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
        - RandomToneCurve: For non-linear color transformations
        - RandomBrightnessContrast: For combined brightness and contrast adjustments
        - PlankianJitter: For color temperature adjustments
        - HueSaturationValue: For HSV color space adjustments
        - ColorJitter: For combined brightness, contrast, saturation adjustments

    """

    class InitSchema(BaseTransformInitSchema):
        r_shift_range: tuple[float, float]
        g_shift_range: tuple[float, float]
        b_shift_range: tuple[float, float]

    def __init__(
        self,
        r_shift_range: tuple[float, float] = (-20, 20),
        g_shift_range: tuple[float, float] = (-20, 20),
        b_shift_range: tuple[float, float] = (-20, 20),
        p: float = 0.5,
    ):
        def normalize_range(limit: tuple[float, float]) -> tuple[float, float]:
            if abs(limit[0]) > 1 or abs(limit[1]) > 1:
                return (limit[0] / 255.0, limit[1] / 255.0)
            return limit

        ranges = [
            normalize_range(r_shift_range),
            normalize_range(g_shift_range),
            normalize_range(b_shift_range),
        ]

        # Initialize with fixed noise type and spatial mode
        super().__init__(
            noise_type="uniform",
            spatial_mode="constant",
            noise_params={"ranges": ranges},
            p=p,
        )

        self.r_shift_range = r_shift_range
        self.g_shift_range = g_shift_range
        self.b_shift_range = b_shift_range

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        result = super().get_params_dependent_on_data(params=params, data=data)

        # spatial_mode="constant" produces a (C,) noise_map of per-channel shifts already
        # scaled to image dtype (uint8: [-255, 255], float: [-1, 1]). Record them as sampled scalars.
        noise_map = result.get("noise_map")
        if noise_map is not None and noise_map.size >= 3:
            shifts = noise_map.reshape(-1)[:3].tolist()
            self.applied_config["r_shift_range"] = float(shifts[0])
            self.applied_config["g_shift_range"] = float(shifts[1])
            self.applied_config["b_shift_range"] = float(shifts[2])

        return result


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


__all__ = [
    "PLANKIAN_JITTER_CONST",
    "ChromaticAberration",
    "ColorJitter",
    "ColorRange",
    "HEStain",
    "PhotoMetricDistort",
    "PlanckianJitter",
    "RGBShift",
]
