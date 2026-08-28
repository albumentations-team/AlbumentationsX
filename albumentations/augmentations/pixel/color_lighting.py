"""Lighting, plasma, illumination, and vignetting transforms."""

from typing import Annotated, Any, Literal

from albumentations.core.invocation import SamplingContext
from albumentations.core.transform_params import (
    SampledParams,
    TargetParams,
    TargetSet,
    requirements_for_views,
)

from ._color_shared import (
    AfterValidator,
    BaseTransformInitSchema,
    Field,
    ImageOnlyTransform,
    ImageType,
    Self,
    albucore,
    check_range_bounds,
    cv2,
    fpixel,
    model_validator,
    nondecreasing,
    np,
)

GaussianIlluminationSpot = tuple[float, float, float, float]


def _validate_gaussian_illumination_spot(spot: GaussianIlluminationSpot) -> None:
    center_x, center_y, sigma, intensity = spot
    if not 0 <= center_x <= 1 or not 0 <= center_y <= 1:
        raise ValueError("gaussian_spots centers must be in the [0, 1] range")
    if not 0.2 <= sigma <= 1.0:
        raise ValueError("gaussian_spots sigma must be in the [0.2, 1.0] range")
    if not 0.01 <= abs(intensity) <= 0.2:
        raise ValueError("gaussian_spots signed intensity magnitude must be in the [0.01, 0.2] range")


def _generate_resized_plasma(
    target_shape: tuple[int, int],
    plasma_size: int,
    roughness: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    plasma_shape = (min(target_shape[0], plasma_size), min(target_shape[1], plasma_size))
    plasma = fpixel.generate_plasma_pattern(
        target_shape=plasma_shape,
        roughness=roughness,
        random_generator=random_generator,
    )
    if plasma_shape == target_shape:
        return plasma
    return albucore.resize(plasma, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


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

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        # Sample adjustment strengths
        brightness = sampling.py_random.uniform(*self.brightness_range)
        contrast = sampling.py_random.uniform(*self.contrast_range)

        sampling.applied_overrides.update({"brightness_range": brightness, "contrast_range": contrast})

        groups = []
        for views in targets.group_image_like_by(
            lambda view: tuple((view.descriptor.spatial_shape or ())[-2:]),
        ):
            spatial_shape = views[0].descriptor.spatial_shape
            if spatial_shape is None:
                raise ValueError("PlasmaBrightnessContrast requires image-like targets with known shapes")
            target_shape = (spatial_shape[-2], spatial_shape[-1])
            plasma = _generate_resized_plasma(
                target_shape=target_shape,
                plasma_size=self.plasma_size,
                roughness=self.roughness,
                random_generator=sampling.random_generator,
            )
            groups.append(
                TargetParams(
                    targets=tuple(view.name for view in views),
                    params={"plasma_pattern": plasma},
                    requirements=requirements_for_views(views, spatial_shape_suffix=True),
                ),
            )
        return SampledParams(
            params={"brightness_factor": brightness, "contrast_factor": contrast},
            target_params=tuple(groups),
        )

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
        plasma_size: int = Field(ge=1)
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

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        # Sample shadow intensity
        intensity = sampling.py_random.uniform(*self.shadow_intensity_range)

        sampling.applied_overrides["shadow_intensity_range"] = intensity

        groups = []
        for views in targets.group_image_like_by(
            lambda view: tuple((view.descriptor.spatial_shape or ())[-2:]),
        ):
            spatial_shape = views[0].descriptor.spatial_shape
            if spatial_shape is None:
                raise ValueError("PlasmaShadow requires image-like targets with known shapes")
            target_shape = (spatial_shape[-2], spatial_shape[-1])
            plasma = _generate_resized_plasma(
                target_shape=target_shape,
                plasma_size=self.plasma_size,
                roughness=self.roughness,
                random_generator=sampling.random_generator,
            )
            groups.append(
                TargetParams(
                    targets=tuple(view.name for view in views),
                    params={"plasma_pattern": plasma},
                    requirements=requirements_for_views(views, spatial_shape_suffix=True),
                ),
            )
        return SampledParams(params={"intensity": intensity}, target_params=tuple(groups))

    def apply(
        self,
        img: ImageType,
        intensity: float,
        plasma_pattern: np.ndarray,
        **params: Any,
    ) -> ImageType:
        return fpixel.apply_plasma_shadow(img, intensity, plasma_pattern)


class Illumination(ImageOnlyTransform):
    """Simulate directional, corner, or local Gaussian lighting patterns to make training images more robust
    to varied illumination conditions.

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

        num_spots_range (tuple[int, int]): Inclusive range for the number of independently sampled Gaussian spots.
            Each spot has its own center, sigma, intensity, and effect sign. Overlapping spot fields are multiplied,
            so several spots can produce a stronger combined effect. Only used for 'gaussian' mode.
            Default: (1, 1)

        gaussian_spots (tuple[tuple[float, float, float, float], ...] | None): Optional fixed Gaussian spots.
            Each spot is represented as `(center_x, center_y, sigma, signed_intensity)`. Providing this value bypasses
            random spot sampling, and `num_spots_range` must equal the exact number of spots. Default: None

        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image, volume

    Image types:
        uint8, float32

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.full((100, 100, 3), 128, dtype=np.uint8)
        >>> # Simulate sunlight through window
        >>> transform = A.Illumination(
        ...     mode='linear',
        ...     intensity_range=(0.05, 0.1),
        ...     effect_type='brighten',
        ...     angle_range=(30, 60)
        ... )
        >>> transformed_image = transform(image=image)["image"]
        >>>
        >>> # Create dramatic corner shadow
        >>> transform = A.Illumination(
        ...     mode='corner',
        ...     intensity_range=(0.1, 0.2),
        ...     effect_type='darken'
        ... )
        >>> transformed_image = transform(image=image)["image"]
        >>>
        >>> # Add a random number of independently sampled bright and dark spots
        >>> transform = A.Illumination(
        ...     mode='gaussian',
        ...     num_spots_range=(2, 4),
        ...     intensity_range=(0.05, 0.15),
        ...     effect_type='both',
        ...     center_range=(0.1, 0.9),
        ...     sigma_range=(0.2, 0.5),
        ...     p=1.0,
        ... )
        >>> transformed_image = transform(image=image)["image"]

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
        - Linear mode adds a signed gradient, matching Kornia's RandomLinearIllumination behavior
        - Corner and gaussian modes apply multiplicative masks to preserve texture
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
        num_spots_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
            AfterValidator(nondecreasing),
        ]
        gaussian_spots: tuple[GaussianIlluminationSpot, ...] | None

        @model_validator(mode="after")
        def validate_gaussian_spots(self) -> Self:
            """Validate Gaussian spot configuration and mode-specific options before sampling.
            Keep constructor state consistent for random and deterministic replay.
            """
            if self.mode != "gaussian":
                if self.num_spots_range != (1, 1):
                    raise ValueError("num_spots_range can only be changed for gaussian mode")
                if self.gaussian_spots is not None:
                    raise ValueError("gaussian_spots can only be provided for gaussian mode")
                return self

            if self.gaussian_spots is None:
                return self
            if not self.gaussian_spots:
                raise ValueError("gaussian_spots must contain at least one spot")
            expected_range = (len(self.gaussian_spots),) * 2
            if self.num_spots_range != expected_range:
                raise ValueError(
                    "num_spots_range must equal the number of deterministic gaussian_spots",
                )
            for spot in self.gaussian_spots:
                _validate_gaussian_illumination_spot(spot)
            return self

    def __init__(
        self,
        mode: Literal["linear", "corner", "gaussian"] = "linear",
        intensity_range: tuple[float, float] = (0.01, 0.2),
        effect_type: Literal["brighten", "darken", "both"] = "both",
        angle_range: tuple[float, float] = (0, 360),
        center_range: tuple[float, float] = (0.1, 0.9),
        sigma_range: tuple[float, float] = (0.2, 1.0),
        p: float = 0.5,
        *,
        num_spots_range: tuple[int, int] = (1, 1),
        gaussian_spots: tuple[GaussianIlluminationSpot, ...] | None = None,
    ):
        super().__init__(p=p)
        self.mode = mode
        self.intensity_range = intensity_range
        self.effect_type = effect_type
        self.angle_range = angle_range
        self.center_range = center_range
        self.sigma_range = sigma_range
        self.num_spots_range = num_spots_range
        self.gaussian_spots = gaussian_spots

    def _sample_signed_intensity(self, sampling: SamplingContext) -> float:
        intensity = sampling.py_random.uniform(*self.intensity_range)
        if self.effect_type == "both":
            return intensity if sampling.py_random.random() > 0.5 else -intensity
        if self.effect_type == "darken":
            return -intensity
        return intensity

    def _sample_gaussian_spot(self, sampling: SamplingContext) -> GaussianIlluminationSpot:
        intensity = self._sample_signed_intensity(sampling)
        center_x = sampling.py_random.uniform(*self.center_range)
        center_y = sampling.py_random.uniform(*self.center_range)
        sigma = sampling.py_random.uniform(*self.sigma_range)
        return center_x, center_y, sigma, intensity

    def _sample_gaussian_parameters(self, sampling: SamplingContext) -> SampledParams:
        if self.gaussian_spots is None:
            num_spots = 1 if self.num_spots_range == (1, 1) else sampling.py_random.randint(*self.num_spots_range)
            spots = tuple(self._sample_gaussian_spot(sampling) for _ in range(num_spots))
            intensities = [abs(spot[3]) for spot in spots]
            sampling.applied_overrides.update(
                {
                    "intensity_range": intensities[0] if num_spots == 1 else (min(intensities), max(intensities)),
                    "angle_range": self.angle_range,
                    "center_range": (spots[0][0], spots[0][1]) if num_spots == 1 else self.center_range,
                    "sigma_range": spots[0][2]
                    if num_spots == 1
                    else (min(spot[2] for spot in spots), max(spot[2] for spot in spots)),
                },
            )
        else:
            spots = self.gaussian_spots
            num_spots = len(spots)

        sampling.applied_overrides.update(
            {
                "num_spots_range": num_spots,
                "gaussian_spots": spots,
            },
        )
        if num_spots == 1:
            center_x, center_y, sigma, intensity = spots[0]
            return SampledParams(
                params={
                    "intensity": intensity,
                    "angle": None,
                    "corner": None,
                    "center": (center_x, center_y),
                    "sigma": sigma,
                    "spots": None,
                },
            )
        return SampledParams(
            params={
                "intensity": None,
                "angle": None,
                "corner": None,
                "center": None,
                "sigma": None,
                "spots": spots,
            },
        )

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        if self.mode == "gaussian":
            return self._sample_gaussian_parameters(sampling)

        intensity = self._sample_signed_intensity(sampling)

        # Always record all _range overrides so the applied record consistently reflects what was used,
        # echoing the constructor range for params not active in this mode.
        sampling.applied_overrides.update(
            {
                "intensity_range": abs(intensity),
                "angle_range": self.angle_range,
                "center_range": self.center_range,
                "sigma_range": self.sigma_range,
            },
        )

        if self.mode == "linear":
            angle = sampling.py_random.uniform(*self.angle_range)
            sampling.applied_overrides["angle_range"] = angle
            return SampledParams(
                params={
                    "intensity": intensity,
                    "angle": angle,
                    "corner": None,
                    "center": None,
                    "sigma": None,
                    "spots": None,
                }
            )
        if self.mode == "corner":
            corner = sampling.py_random.randint(0, 3)  # Choose random corner
            return SampledParams(
                params={
                    "intensity": intensity,
                    "angle": None,
                    "corner": corner,
                    "center": None,
                    "sigma": None,
                    "spots": None,
                }
            )
        raise RuntimeError(f"Unsupported illumination mode: {self.mode}")

    def apply(
        self,
        img: ImageType,
        intensity: float | None,
        angle: float | None,
        corner: Literal[0, 1, 2, 3] | None,
        center: tuple[float, float] | None,
        sigma: float | None,
        spots: tuple[GaussianIlluminationSpot, ...] | None,
        **params: Any,
    ) -> ImageType:
        if self.mode == "linear" and intensity is not None and angle is not None:
            return fpixel.apply_linear_illumination(img, intensity=intensity, angle=angle)
        if self.mode == "corner" and intensity is not None and corner is not None:
            return fpixel.apply_corner_illumination(img, intensity=intensity, corner=corner)
        if self.mode == "gaussian" and spots is not None:
            return fpixel.apply_gaussian_illumination_spots(img, spots)
        if self.mode == "gaussian" and intensity is not None and center is not None and sigma is not None:
            return fpixel.apply_gaussian_illumination(img, intensity=intensity, center=center, sigma=sigma)
        raise RuntimeError(f"Illumination sampled parameters are incompatible with mode {self.mode!r}")

    def apply_to_images(self, images: ImageType, *args: Any, **params: Any) -> ImageType:
        return fpixel.apply_illumination_batch(images, self.mode, **params)


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

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        intensity = sampling.py_random.uniform(*self.intensity_range)
        center_x = sampling.py_random.uniform(*self.center_range)
        center_y = sampling.py_random.uniform(*self.center_range)
        sampling.applied_overrides.update(
            {
                "intensity_range": intensity,
                "center_range": (min(center_x, center_y), max(center_x, center_y)),
            },
        )
        return SampledParams(
            params={
                "intensity": intensity,
                "center_x": center_x,
                "center_y": center_y,
            }
        )


__all__ = [
    "Illumination",
    "PlasmaBrightnessContrast",
    "PlasmaShadow",
    "Vignetting",
    "_generate_resized_plasma",
]
