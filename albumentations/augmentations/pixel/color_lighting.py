"""Lighting, plasma, illumination, and vignetting transforms."""

from typing import Annotated, Any, Literal

from ._color_shared import (
    AfterValidator,
    BaseTransformInitSchema,
    Field,
    ImageOnlyTransform,
    ImageType,
    VolumeType,
    albucore,
    batch_transform,
    check_range_bounds,
    cv2,
    fpixel,
    nondecreasing,
    np,
)


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

        plasma = _generate_resized_plasma(
            target_shape=shape[:2],
            plasma_size=self.plasma_size,
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

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        shape = params["shape"]

        # Sample shadow intensity
        intensity = self.py_random.uniform(*self.shadow_intensity_range)

        self.applied_config = {"shadow_intensity_range": intensity}

        plasma = _generate_resized_plasma(
            target_shape=shape[:2],
            plasma_size=self.plasma_size,
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

        # Always record all _range overrides so applied_config consistently reflects what was used,
        # echoing the constructor range for params not active in this mode.
        self.applied_config = {
            "intensity_range": abs(intensity),
            "angle_range": self.angle_range,
            "center_range": self.center_range,
            "sigma_range": self.sigma_range,
        }

        if self.mode == "linear":
            angle = self.py_random.uniform(*self.angle_range)
            self.applied_config["angle_range"] = angle
            return {
                "intensity": intensity,
                "angle": angle,
            }
        if self.mode == "corner":
            corner = self.py_random.randint(0, 3)  # Choose random corner
            return {
                "intensity": intensity,
                "corner": corner,
            }

        x = self.py_random.uniform(*self.center_range)
        y = self.py_random.uniform(*self.center_range)
        sigma = self.py_random.uniform(*self.sigma_range)
        self.applied_config["center_range"] = (x, y)
        self.applied_config["sigma_range"] = sigma
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

        if self.mode == "linear":
            return self._apply_to_batch_same_shape(
                images,
                lambda image: albucore.add_array(image, gradient),
            )

        return self._apply_to_batch_same_shape(images, lambda image: albucore.multiply_by_array(image, gradient))


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


__all__ = [
    "Illumination",
    "PlasmaBrightnessContrast",
    "PlasmaShadow",
    "Vignetting",
    "_generate_resized_plasma",
]
