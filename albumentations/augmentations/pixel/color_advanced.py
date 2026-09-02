"""Advanced color jitter, channel shift, aberration, stain, and photometric transforms."""

from typing import Annotated, Any, Literal, TypedDict, cast

from typing_extensions import Self

from albumentations.core.invocation import SamplingContext
from albumentations.core.transform_params import (
    SampledParams,
    TargetParams,
    TargetSet,
    requirements_for_views,
)

from ._color_shared import (
    CV2_INTER_LINEAR,
    AdditiveNoise,
    AfterValidator,
    BaseTransformInitSchema,
    Field,
    FullInterpolationType,
    ImageOnlyTransform,
    ImageType,
    check_range_bounds,
    field_validator,
    fpixel,
    is_grayscale_image,
    is_rgb_image,
    model_validator,
    non_rgb_error,
    nondecreasing,
    np,
)

ColorRange = tuple[tuple[int, int, int], tuple[int, int, int]]


class PlanckianJitterConst(TypedDict):
    MAX_TEMP: int
    MIN_BLACKBODY_TEMP: int
    MIN_CIED_TEMP: int
    WHITE_TEMP: int
    SAMPLING_TEMP_PROB: float


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

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        brightness = sampling.py_random.uniform(*self.brightness_range)
        contrast = sampling.py_random.uniform(*self.contrast_range)
        saturation = sampling.py_random.uniform(*self.saturation_range)
        hue = sampling.py_random.uniform(*self.hue_range)

        sampling.applied_overrides.update(
            {
                "brightness_range": brightness,
                "contrast_range": contrast,
                "saturation_range": saturation,
                "hue_range": hue,
            },
        )

        order = ["brightness", "contrast", "saturation", "hue"]
        sampling.random_generator.shuffle(order)

        # Merge adjacent brightness+contrast into one slot for fused LUT.
        idx_b, idx_c = order.index("brightness"), order.index("contrast")
        if abs(idx_b - idx_c) == 1:
            merged = "brightness_contrast" if idx_b < idx_c else "contrast_brightness"
            order = [o for o in order if o not in ("brightness", "contrast")]
            order.insert(min(idx_b, idx_c), merged)

        return SampledParams(
            params={
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
                "hue": hue,
                "order": order,
            }
        )

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
        return fpixel.apply_color_jitter(img, brightness, contrast, saturation, hue, order)

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
        interpolation: FullInterpolationType

    def __init__(
        self,
        primary_distortion_range: tuple[float, float] = (-0.02, 0.02),
        secondary_distortion_range: tuple[float, float] = (-0.05, 0.05),
        mode: Literal["green_purple", "red_blue", "random"] = "green_purple",
        interpolation: FullInterpolationType = CV2_INTER_LINEAR,
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

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        primary_distortion_red = sampling.py_random.uniform(*self.primary_distortion_range)
        secondary_distortion_red = sampling.py_random.uniform(
            *self.secondary_distortion_range,
        )
        primary_distortion_blue = sampling.py_random.uniform(*self.primary_distortion_range)
        secondary_distortion_blue = sampling.py_random.uniform(
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

        sampling.applied_overrides.update(
            {
                "primary_distortion_range": (primary_distortion_red, primary_distortion_blue),
                "secondary_distortion_range": (secondary_distortion_red, secondary_distortion_blue),
            },
        )
        return SampledParams(
            params={
                "primary_distortion_red": primary_distortion_red,
                "secondary_distortion_red": secondary_distortion_red,
                "primary_distortion_blue": primary_distortion_blue,
                "secondary_distortion_blue": secondary_distortion_blue,
            }
        )

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


PLANKIAN_JITTER_CONST: PlanckianJitterConst = {
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
            max_temp = PLANKIAN_JITTER_CONST["MAX_TEMP"]

            if self.temperature_range is None:
                if self.mode == "blackbody":
                    self.temperature_range = (
                        PLANKIAN_JITTER_CONST["MIN_BLACKBODY_TEMP"],
                        max_temp,
                    )
                elif self.mode == "cied":
                    self.temperature_range = (
                        PLANKIAN_JITTER_CONST["MIN_CIED_TEMP"],
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

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        sampling_prob_boundary = PLANKIAN_JITTER_CONST["SAMPLING_TEMP_PROB"]
        sampling_temp_boundary = PLANKIAN_JITTER_CONST["WHITE_TEMP"]

        if self.sampling_method == "uniform":
            # Split into 2 cases to avoid selecting cold temperatures (>6000) too often
            if sampling.py_random.random() < sampling_prob_boundary:
                temperature = sampling.py_random.uniform(
                    self.temperature_range[0],
                    sampling_temp_boundary,
                )
            else:
                temperature = sampling.py_random.uniform(
                    sampling_temp_boundary,
                    self.temperature_range[1],
                )
        elif self.sampling_method == "gaussian":
            # Sample values from asymmetric gaussian distribution
            if sampling.py_random.random() < sampling_prob_boundary:
                # Left side
                shift = np.abs(
                    sampling.py_random.gauss(
                        0,
                        np.abs(sampling_temp_boundary - self.temperature_range[0]) / 3,
                    ),
                )
                temperature = sampling_temp_boundary - shift
            else:
                # Right side
                shift = np.abs(
                    sampling.py_random.gauss(
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

        white_temperature = PLANKIAN_JITTER_CONST["WHITE_TEMP"]
        sampling.applied_overrides.update(
            {
                "temperature_range": (
                    min(int(temperature), white_temperature),
                    max(int(temperature), white_temperature),
                ),
            },
        )
        return SampledParams(params={"temperature": int(temperature)})


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

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        groups: list[TargetParams] = []
        for views in targets.group_image_like_by(
            lambda view: (
                view.descriptor.channels,
                view.descriptor.value_scale,
            ),
        ):
            view = views[0]
            value_scale = view.descriptor.value_scale
            if value_scale is None:
                raise ValueError(f"RGBShift target {view.name!r} has an unsupported dtype")
            sampled = self._sample_noise_map(
                (view.descriptor.channels or 1,),
                value_scale,
                sampling,
            )

            noise_map = sampled.get("noise_map")
            if not groups and noise_map is not None and noise_map.size >= 3:
                shifts = noise_map.reshape(-1)[:3].tolist()
                sampling.applied_overrides["r_shift_range"] = float(shifts[0])
                sampling.applied_overrides["g_shift_range"] = float(shifts[1])
                sampling.applied_overrides["b_shift_range"] = float(shifts[2])

            groups.append(
                TargetParams(
                    targets=tuple(item.name for item in views),
                    params=sampled,
                    requirements=requirements_for_views(
                        views,
                        channels=True,
                        dtype=True,
                    ),
                ),
            )

        return SampledParams(params={}, target_params=tuple(groups))


class HEStain(ImageOnlyTransform):
    """Perturb stain concentrations in histology images to simulate color variation across laboratories,
    scanners, protocols, and staining panels.

    Use this transform to train pathology models against expected staining variation. It converts RGB values to
    optical density, separates the stain concentrations with the selected basis, perturbs the configured components,
    and reconstructs the RGB image.

    Args:
        method (Literal["preset", "random_preset", "vahadane", "macenko", "custom"]): Selects the stain basis:
            - "preset": Use the matrix named by `preset`.
            - "random_preset": Select one of the eight preset matrices for each call.
            - "vahadane": Extract the matrix from the input with the Vahadane method.
            - "macenko": Extract the matrix from the input with the Macenko method.
            - "custom": Use the fixed matrix supplied through `stain_matrix`.
            Default: "random_preset".

        preset (str | None): Preset stain matrix used when `method="preset"`:
            - "ruifrok": Standard reference from Ruifrok and Johnston.
            - "macenko": Reference from the Macenko method.
            - "standard": Typical bright-field microscopy.
            - "high_contrast": Enhanced contrast.
            - "h_heavy": Hematoxylin-dominant staining.
            - "e_heavy": Eosin-dominant staining.
            - "dark": Darker staining.
            - "light": Lighter staining.
            When None with `method="preset"`, "standard" is used. Default: None.

        intensity_scale_range (tuple[float, float]): Non-negative range for the multiplicative concentration factor
            sampled independently for hematoxylin, eosin, and any augmented third component. For example,
            `(0.7, 1.3)` varies each concentration from 70% to 130%. Default: (0.7, 1.3).

        intensity_shift_range (tuple[float, float]): Range within `[-1.0, 1.0]` for the additive concentration shift
            sampled independently for hematoxylin, eosin, and any augmented third component. Default: (-0.2, 0.2).

        augment_background (bool): Whether to perturb background pixels along with tissue pixels. Default: False.
        residual_mode (Literal["project", "preserve", "augment"]): Controls the third optical-density component:
            - `"project"`: Reconstruct from H&E only, retaining the two-stain model from earlier releases.
            - `"preserve"`: Keep the derived residual or explicit third-stain concentration unchanged.
            - `"augment"`: Independently perturb the derived residual or explicit third stain along with H&E.
            Default: `"project"`.
        p (float): Probability of applying the transform. Default: 0.5.
        stain_matrix (np.ndarray | None): Fixed stain basis used when `method="custom"`. A `(2, 3)` matrix contains
            hematoxylin and eosin RGB optical-density vectors; `"preserve"` and `"augment"` derive the third vector
            as `normalize(cross(H, E))`. A `(3, 3)` matrix supplies the third stain directly and requires
            `residual_mode="preserve"` or `"augment"`. Every row must contain finite values and be non-zero, and the
            matrix must have full row rank. The transform copies the matrix as `float32` without row normalization.
            Default: None.

    Targets:
        image, volume

    Number of channels:
        3

    Image types:
        uint8, float32

    Note:
        - Let `M` be the stain matrix and `C` the per-pixel concentrations. `"project"` solves
          `OD ~= C @ M`, perturbs H&E, and reconstructs `RGB = exp(-(C * scale + shift) @ M)`.
        - For a `(2, 3)` matrix, `"preserve"` and `"augment"` derive `R = normalize(cross(H, E))` and solve the full
          H&E+R basis. A `(3, 3)` matrix uses its third row directly.
        - A custom matrix is fixed for the lifetime of the transform. Per-image callable extraction is not supported.

    References:
        - A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical": Analytical and quantitative
            cytology and histology, 2001.
        - M. Macenko et al., "A method for normalizing histology slides for: 2009 IEEE International Symposium on
            quantitative analysis," 2009 IEEE International Symposium on Biomedical Imaging, 2009.
        - D. Tellez et al., "H&E stain augmentation improves generalization of convolutional networks for
            histopathological mitosis detection": Medical Imaging, 2018.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Create a sample H&E stained histopathology image
        >>> # For real use cases, load an actual H&E stained image
        >>> image = np.zeros((300, 300, 3), dtype=np.uint8)
        >>> # Simulate tissue regions with different staining patterns
        >>> image[50:150, 50:150] = np.array([120, 140, 180], dtype=np.uint8)  # Hematoxylin-rich region
        >>> image[150:250, 150:250] = np.array([140, 160, 120], dtype=np.uint8)  # Eosin-rich region
        >>>
        >>> # Example 1: Map HEDJitter(theta) to a full H&E+DAB basis
        >>> theta = 0.05
        >>> hed_basis = np.array(
        ...     [
        ...         [0.65, 0.70, 0.29],  # Hematoxylin
        ...         [0.07, 0.99, 0.11],  # Eosin
        ...         [0.27, 0.57, 0.78],  # DAB
        ...     ],
        ...     dtype=np.float32,
        ... )
        >>> transform = A.HEStain(
        ...     method="custom",
        ...     stain_matrix=hed_basis,
        ...     residual_mode="augment",
        ...     intensity_scale_range=(1 - theta, 1 + theta),
        ...     intensity_shift_range=(-theta, theta),
        ...     augment_background=True,
        ...     p=1.0,
        ... )
        >>> transformed_image = transform(image=image)["image"]
        >>>
        >>> # Example 2: Using a specific preset stain matrix
        >>> transform = A.HEStain(
        ...     method="preset",
        ...     preset="standard",
        ...     intensity_scale_range=(0.8, 1.2),
        ...     intensity_shift_range=(-0.1, 0.1),
        ...     augment_background=False,
        ...     p=1.0,
        ... )
        >>> transformed_image = transform(image=image)["image"]
        >>>
        >>> # Example 3: Using random preset selection
        >>> transform = A.HEStain(
        ...     method="random_preset",
        ...     intensity_scale_range=(0.7, 1.3),
        ...     intensity_shift_range=(-0.15, 0.15),
        ...     p=1.0,
        ... )
        >>> transformed_image = transform(image=image)["image"]
        >>>
        >>> # Example 4: Using Vahadane extraction (requires an H&E stained input)
        >>> transform = A.HEStain(
        ...     method="vahadane",
        ...     intensity_scale_range=(0.7, 1.3),
        ...     p=1.0,
        ... )
        >>> transformed_image = transform(image=image)["image"]
        >>>
        >>> # Example 5: Using Macenko extraction (requires an H&E stained input)
        >>> transform = A.HEStain(
        ...     method="macenko",
        ...     intensity_scale_range=(0.7, 1.3),
        ...     intensity_shift_range=(-0.2, 0.2),
        ...     p=1.0,
        ... )
        >>> transformed_image = transform(image=image)["image"]
        >>>
        >>> # Example 6: Combining stain and brightness variation in one pipeline
        >>> transform = A.Compose([
        ...     A.HEStain(method="preset", preset="high_contrast", p=1.0),
        ...     A.RandomBrightnessContrast(p=0.5),
        ... ])
        >>> transformed_image = transform(image=image)["image"]

    """

    class InitSchema(BaseTransformInitSchema):
        method: Literal["preset", "random_preset", "vahadane", "macenko", "custom"]
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
        stain_matrix: np.ndarray | None
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
        residual_mode: Literal["project", "preserve", "augment"]

        @field_validator("stain_matrix", mode="before")
        @classmethod
        def _convert_stain_matrix(cls, value: Any) -> np.ndarray | None:
            if value is None:
                return None
            try:
                stain_matrix = np.array(value, dtype=np.float32, copy=True)
            except (TypeError, ValueError) as exc:
                raise ValueError("stain_matrix must contain numeric values") from exc
            if stain_matrix.shape not in {(2, 3), (3, 3)}:
                raise ValueError(f"stain_matrix must have shape (2, 3) or (3, 3), got {stain_matrix.shape}")
            if not np.isfinite(stain_matrix).all():
                raise ValueError("stain_matrix must contain only finite values")
            if np.any(np.linalg.norm(stain_matrix, axis=1) == 0):
                raise ValueError("stain_matrix rows must be non-zero stain vectors")
            if np.linalg.matrix_rank(stain_matrix) < stain_matrix.shape[0]:
                raise ValueError("stain_matrix rows must be linearly independent")
            return stain_matrix

        @model_validator(mode="after")
        def _validate_matrix_selection(self) -> Self:
            if self.method == "custom" and self.stain_matrix is None:
                raise ValueError("stain_matrix is required when method='custom'")
            if self.method != "custom" and self.stain_matrix is not None:
                raise ValueError("stain_matrix is only valid when method='custom'")
            if (
                self.method == "custom"
                and self.stain_matrix is not None
                and self.stain_matrix.shape == (3, 3)
                and self.residual_mode == "project"
            ):
                raise ValueError("A full stain basis is incompatible with residual_mode='project'")
            if self.method == "preset" and self.preset is None:
                self.preset = "standard"
            elif self.method in {"random_preset", "custom"} and self.preset is not None:
                raise ValueError(f"preset should not be specified when method='{self.method}'")
            return self

    def __init__(
        self,
        method: Literal["preset", "random_preset", "vahadane", "macenko", "custom"] = "random_preset",
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
        *,
        residual_mode: Literal["project", "preserve", "augment"] = "project",
        stain_matrix: np.ndarray | None = None,
    ):
        super().__init__(p=p)
        self.method = method
        self.preset = preset
        self.intensity_scale_range = intensity_scale_range
        self.intensity_shift_range = intensity_shift_range
        self.augment_background = augment_background
        self.residual_mode = residual_mode
        self.stain_matrix = stain_matrix

        # Initialize stain extractor here if needed
        if method in ["vahadane", "macenko"]:
            self.stain_extractor = fpixel.get_normalizer(method)

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

    def get_transform_init_args(self) -> dict[str, Any]:
        """Return constructor arguments with a custom stain matrix converted to nested lists so JSON and YAML
        serialization can reconstruct HEStain.
        """
        args = super().get_transform_init_args()
        if isinstance(args.get("stain_matrix"), np.ndarray):
            args["stain_matrix"] = args["stain_matrix"].tolist()
        return args

    def _get_stain_matrix(self, img: ImageType | None, sampling: SamplingContext) -> np.ndarray:
        if self.method == "preset" and self.preset is not None:
            return fpixel.STAIN_MATRICES[self.preset]
        if self.method == "random_preset":
            random_preset = sampling.py_random.choice(self.preset_names)
            return fpixel.STAIN_MATRICES[random_preset]
        if self.method == "custom":
            return cast("np.ndarray", self.stain_matrix)
        # vahadane or macenko
        if img is None:
            raise RuntimeError("Stain extraction requires an image-like target")
        self.stain_extractor.fit(img)
        stain_matrix = self.stain_extractor.stain_matrix_target
        if stain_matrix is None:
            raise RuntimeError("Stain extractor did not produce a stain matrix.")
        return stain_matrix

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
            residual_mode=self.residual_mode,
        )

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        shared_stain_matrix = (
            self._get_stain_matrix(None, sampling) if self.method not in {"vahadane", "macenko"} else None
        )
        scale_h = sampling.py_random.uniform(*self.intensity_scale_range)
        scale_e = sampling.py_random.uniform(*self.intensity_scale_range)
        shift_h = sampling.py_random.uniform(*self.intensity_shift_range)
        shift_e = sampling.py_random.uniform(*self.intensity_shift_range)

        sampled_scales: tuple[float, ...]
        sampled_shifts: tuple[float, ...]
        if self.residual_mode == "preserve":
            scale_factors = np.array([scale_h, scale_e, 1.0], dtype=np.float32)
            shift_values = np.array([shift_h, shift_e, 0.0], dtype=np.float32)
            sampled_scales = (scale_h, scale_e)
            sampled_shifts = (shift_h, shift_e)
        elif self.residual_mode == "augment":
            scale_residual = sampling.py_random.uniform(*self.intensity_scale_range)
            shift_residual = sampling.py_random.uniform(*self.intensity_shift_range)
            scale_factors = np.array([scale_h, scale_e, scale_residual], dtype=np.float32)
            shift_values = np.array([shift_h, shift_e, shift_residual], dtype=np.float32)
            sampled_scales = (scale_h, scale_e, scale_residual)
            sampled_shifts = (shift_h, shift_e, shift_residual)
        else:
            scale_factors = np.array([scale_h, scale_e])
            shift_values = np.array([shift_h, shift_e])
            sampled_scales = (scale_h, scale_e)
            sampled_shifts = (shift_h, shift_e)

        sampling.applied_overrides.update(
            {
                "intensity_scale_range": (min(sampled_scales), max(sampled_scales)),
                "intensity_shift_range": (min(sampled_shifts), max(sampled_shifts)),
            },
        )

        shared_params = {
            "scale_factors": scale_factors,
            "shift_values": shift_values,
        }
        if self.method not in {"vahadane", "macenko"}:
            return SampledParams(params={**shared_params, "stain_matrix": shared_stain_matrix})

        groups = []
        for view in targets.image_like():
            image = view.value if view.canonical_type == "image" else view.value[0]
            groups.append(
                TargetParams(
                    targets=(view.name,),
                    params={"stain_matrix": self._get_stain_matrix(image, sampling)},
                    requirements=requirements_for_views((view,), channels=True),
                ),
            )
        if not groups:
            raise RuntimeError("Expected an image-like target for stain augmentation")
        return SampledParams(params=shared_params, target_params=tuple(groups))


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

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        brightness_factor = (
            sampling.py_random.uniform(*self.brightness_range) if sampling.py_random.random() < self.distort_p else None
        )
        contrast_factor = (
            sampling.py_random.uniform(*self.contrast_range) if sampling.py_random.random() < self.distort_p else None
        )
        saturation_factor = (
            sampling.py_random.uniform(*self.saturation_range) if sampling.py_random.random() < self.distort_p else None
        )
        hue_factor = (
            sampling.py_random.uniform(*self.hue_range) if sampling.py_random.random() < self.distort_p else None
        )
        # contrast_before controls where contrast sits relative to sat/hue; brightness always precedes contrast
        contrast_before = sampling.py_random.random() < 0.5

        applied: dict[str, Any] = {}
        if brightness_factor is not None:
            applied["brightness_range"] = brightness_factor
        if contrast_factor is not None:
            applied["contrast_range"] = contrast_factor
        if saturation_factor is not None:
            applied["saturation_range"] = saturation_factor
        if hue_factor is not None:
            applied["hue_range"] = hue_factor
        sampling.applied_overrides.update(applied)

        groups: list[TargetParams] = []
        for views in targets.group_image_like_by(lambda view: view.descriptor.channels):
            num_channels = views[0].descriptor.channels or 1
            if sampling.py_random.random() < self.distort_p and num_channels > 1:
                channel_permutation = list(range(num_channels))
                sampling.py_random.shuffle(channel_permutation)
            else:
                channel_permutation = None
            groups.append(
                TargetParams(
                    targets=tuple(view.name for view in views),
                    params={"channel_permutation": channel_permutation},
                    requirements=requirements_for_views(views, channels=True),
                ),
            )
        return SampledParams(
            params={
                "brightness_factor": brightness_factor,
                "contrast_factor": contrast_factor,
                "saturation_factor": saturation_factor,
                "hue_factor": hue_factor,
                "contrast_before": contrast_before,
            },
            target_params=tuple(groups),
        )

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
        return fpixel.apply_photometric_distort(
            img,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
            contrast_before,
            channel_permutation,
        )

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
