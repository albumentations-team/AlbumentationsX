"""Weather and atmospheric effect transforms.

Transforms that simulate weather conditions like snow, rain, fog, sun flare,
shadows, and atmospheric effects.
"""

import math
from collections.abc import Sequence
from typing import Annotated, Any, Literal, cast

import albucore
import cv2
import numpy as np
from pydantic import Field, model_validator
from pydantic.functional_validators import AfterValidator
from typing_extensions import Self

import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.augmentations.pixel import functional as fpixel
from albumentations.augmentations.utils import non_rgb_error
from albumentations.core.pydantic import (
    check_range_bounds,
    nondecreasing,
)
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    ImageOnlyTransform,
)
from albumentations.core.type_definitions import (
    MAX_RAIN_ANGLE,
    NUM_RGB_CHANNELS,
    ImageType,
)

__all__ = [
    "AtmosphericFog",
    "RandomFog",
    "RandomGravel",
    "RandomRain",
    "RandomShadow",
    "RandomSnow",
    "RandomSunFlare",
    "Spatter",
]


class RandomSnow(ImageOnlyTransform):
    """Add snow overlay via bleach (brightness threshold) or texture (noise-based overlay).
    Good for winter or snowy-scene robustness in outdoor imagery.

    Two methods: "bleach" brightens pixels above a threshold (faster, simpler); "texture"
    adds a depth-weighted snow layer with sparkle (more realistic, heavier).

    Args:
        snow_point_range (tuple[float, float]): Range for snow intensity threshold in (0, 1).
            Default: (0.1, 0.3).
        brightness_coeff (float): Brightness multiplier for snow; must be > 0. Default: 2.5.
        method (Literal['bleach', 'texture']): "bleach" = threshold + brighten; "texture" =
            noise-based overlay with depth and sparkle. Default: "bleach".
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Note:
        - "bleach": brightness threshold in HLS; pixels above snow_point are scaled by
          brightness_coeff. Fast, less realistic.
        - "texture": HSV brightness boost, Gaussian noise texture, depth gradient (stronger
          at top), alpha blend, blue tint, sparkle. More realistic, heavier.

    Mathematical Formulation:
        For the "bleach" method:
        Let L be the lightness channel in HLS color space.
        For each pixel (i, j):
        If L[i, j] > snow_point:
            L[i, j] = L[i, j] * brightness_coeff

        For the "texture" method:
        1. Brightness adjustment: V_new = V * (1 + brightness_coeff * snow_point)
        2. Snow texture generation: T = GaussianFilter(GaussianNoise(μ=0.5, sigma=0.3))
        3. Depth effect: D = LinearGradient(1.0 to 0.2)
        4. Final pixel value: P = (1 - alpha) * original_pixel + alpha * (T * D * 255)
           where alpha is the snow intensity factor derived from snow_point.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Default usage (bleach method)
        >>> transform = A.RandomSnow(p=1.0)
        >>> snowy_image = transform(image=image)["image"]

        # Using texture method with custom parameters
        >>> transform = A.RandomSnow(
        ...     snow_point_range=(0.2, 0.4),
        ...     brightness_coeff=2.0,
        ...     method="texture",
        ...     p=1.0
        ... )
        >>> snowy_image = transform(image=image)["image"]

    References:
        - Bleach method: https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
        - Texture method: Inspired by computer graphics techniques for snow rendering
          and atmospheric scattering simulations.

    See Also:
        - RandomRain: Rain streaks and blur for rainy conditions.
        - RandomFog: Patch-based fog without depth.
        - AtmosphericFog: Depth-dependent fog via scattering.

    """

    class InitSchema(BaseTransformInitSchema):
        snow_point_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]

        brightness_coeff: float = Field(gt=0)
        method: Literal["bleach", "texture"]

    def __init__(
        self,
        brightness_coeff: float = 2.5,
        snow_point_range: tuple[float, float] = (0.1, 0.3),
        method: Literal["bleach", "texture"] = "bleach",
        p: float = 0.5,
    ):
        super().__init__(p=p)

        self.snow_point_range = snow_point_range
        self.brightness_coeff = brightness_coeff
        self.method = method

    def apply(
        self,
        img: ImageType,
        snow_point: float,
        snow_texture: np.ndarray,
        sparkle_mask: np.ndarray,
        **params: Any,
    ) -> ImageType:
        non_rgb_error(img)

        if self.method == "bleach":
            return fpixel.add_snow_bleach(img, snow_point, self.brightness_coeff)
        if self.method == "texture":
            return fpixel.add_snow_texture(
                img,
                snow_point,
                self.brightness_coeff,
                snow_texture,
                sparkle_mask,
            )

        raise ValueError(f"Unknown snow method: {self.method}")

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, np.ndarray | None]:
        image_shape = params["shape"][:2]
        snow_point = self.py_random.uniform(*self.snow_point_range)

        result = {
            "snow_point": snow_point,
            "snow_texture": None,
            "sparkle_mask": None,
        }

        if self.method == "texture":
            snow_texture, sparkle_mask = fpixel.generate_snow_textures(
                img_shape=image_shape,
                random_generator=self.random_generator,
            )
            result["snow_texture"] = snow_texture
            result["sparkle_mask"] = sparkle_mask

        self.applied_config = {"snow_point_range": snow_point}

        return result


class RandomGravel(ImageOnlyTransform):
    """Add gravel-like particle artifacts on the image. Number and size of particles and
    ROI are configurable. Simulates dirt or debris on a lens or surface.

    This transform simulates the appearance of gravel or small stones scattered across
    specific regions of an image. It's particularly useful for augmenting datasets of
    road or terrain images, adding realistic texture variations.

    Args:
        gravel_roi (tuple[float, float, float, float]): Region of interest where gravel
            will be added, specified as (x_min, y_min, x_max, y_max) in relative coordinates
            [0, 1]. Default: (0.1, 0.4, 0.9, 0.9).
        number_of_patches (int): Number of gravel patch regions to generate within the ROI.
            Each patch will contain multiple gravel particles. Default: 2.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        3

    Note:
        - The gravel effect is created by modifying the saturation channel in the HLS color space.
        - Gravel particles are distributed within randomly generated patches inside the specified ROI.
        - This transform is particularly useful for:
          * Augmenting datasets for road condition analysis
          * Simulating variations in terrain for computer vision tasks
          * Adding realistic texture to synthetic images of outdoor scenes

    Mathematical Formulation:
        For each gravel patch:
        1. A rectangular region is randomly generated within the specified ROI.
        2. Within this region, multiple gravel particles are placed.
        3. For each particle:
           - Random (x, y) coordinates are generated within the patch.
           - A random radius (r) between 1 and 3 pixels is assigned.
           - A random saturation value (sat) between 0 and 255 is assigned.
        4. The saturation channel of the image is modified for each particle:
           image_hls[y-r:y+r, x-r:x+r, 1] = sat

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Default usage
        >>> transform = A.RandomGravel(p=1.0)
        >>> augmented_image = transform(image=image)["image"]

        # Custom ROI and number of patches
        >>> transform = A.RandomGravel(
        ...     gravel_roi=(0.2, 0.2, 0.8, 0.8),
        ...     number_of_patches=5,
        ...     p=1.0
        ... )
        >>> augmented_image = transform(image=image)["image"]

        # Combining with other transforms
        >>> transform = A.Compose([
        ...     A.RandomGravel(p=0.7),
        ...     A.RandomBrightnessContrast(p=0.5),
        ... ])
        >>> augmented_image = transform(image=image)["image"]

    References:
        - Road surface textures: https://en.wikipedia.org/wiki/Road_surface
        - HLS color space: https://en.wikipedia.org/wiki/HSL_and_HSV

    """

    class InitSchema(BaseTransformInitSchema):
        gravel_roi: tuple[float, float, float, float]
        number_of_patches: int = Field(ge=1)

        @model_validator(mode="after")
        def _validate_gravel_roi(self) -> Self:
            gravel_lower_x, gravel_lower_y, gravel_upper_x, gravel_upper_y = self.gravel_roi
            if not 0 <= gravel_lower_x < gravel_upper_x <= 1 or not 0 <= gravel_lower_y < gravel_upper_y <= 1:
                raise ValueError(f"Invalid gravel_roi. Got: {self.gravel_roi}.")
            return self

    def __init__(
        self,
        gravel_roi: tuple[float, float, float, float] = (0.1, 0.4, 0.9, 0.9),
        number_of_patches: int = 2,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.gravel_roi = gravel_roi
        self.number_of_patches = number_of_patches

    def generate_gravel_patch(
        self,
        rectangular_roi: tuple[int, int, int, int],
    ) -> ImageType:
        """Generate gravel (x,y) coordinates inside rectangular_roi (x_min,y_min,x_max,y_max).
        Returns (N, 2) array for RandomGravel overlay.

        Args:
            rectangular_roi (tuple[int, int, int, int]): The rectangular region where gravel
                particles will be generated, specified as (x_min, y_min, x_max, y_max) in pixel coordinates.

        Returns:
            ImageType: An array of gravel particles with shape (count, 2), where count is the number of particles.
            Each row contains the (x, y) coordinates of a gravel particle.

        """
        x_min, y_min, x_max, y_max = rectangular_roi
        area = abs((x_max - x_min) * (y_max - y_min))
        count = area // 10
        gravels = np.empty([count, 2], dtype=np.int64)
        gravels[:, 0] = self.random_generator.integers(x_min, x_max, count)
        gravels[:, 1] = self.random_generator.integers(y_min, y_max, count)
        return gravels

    def apply(
        self,
        img: ImageType,
        gravels_infos: list[Any],
        **params: Any,
    ) -> ImageType:
        return fpixel.add_gravel(img, gravels_infos)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        metadata = self.get_image_data(data)
        height, width = (metadata["height"], metadata["width"])

        # Calculate ROI in pixels
        x_min, y_min, x_max, y_max = (
            int(coord * dim) for coord, dim in zip(self.gravel_roi, [width, height, width, height], strict=True)
        )

        roi_width = x_max - x_min
        roi_height = y_max - y_min

        gravels_info = []

        for _ in range(self.number_of_patches):
            # Generate a random rectangular region within the ROI
            patch_width = self.py_random.randint(roi_width // 10, roi_width // 5)
            patch_height = self.py_random.randint(roi_height // 10, roi_height // 5)

            patch_x = self.py_random.randint(x_min, x_max - patch_width)
            patch_y = self.py_random.randint(y_min, y_max - patch_height)

            # Generate gravel particles within this patch
            num_particles = (patch_width * patch_height) // 100  # Adjust this divisor to control density

            for _ in range(num_particles):
                x = self.py_random.randint(patch_x, patch_x + patch_width)
                y = self.py_random.randint(patch_y, patch_y + patch_height)
                r = self.py_random.randint(1, 3)
                sat = self.py_random.randint(0, 255)

                gravels_info.append(
                    [
                        max(y - r, 0),  # min_y
                        min(y + r, height - 1),  # max_y
                        max(x - r, 0),  # min_x
                        min(x + r, width - 1),  # max_x
                        sat,  # saturation
                    ],
                )

        self.applied_config = {"number_of_patches": self.number_of_patches, "gravel_roi": self.gravel_roi}

        return {"gravels_infos": np.array(gravels_info, dtype=np.int64)}


class RandomRain(ImageOnlyTransform):
    """Add rain streaks (semi-transparent lines), optional blur and brightness reduction.
    Good for outdoor or driving robustness to rainy conditions.

    Streaks are drawn with configurable slant, length, and width; blur and darkening
    simulate wet, low-contrast views. Density and style are configurable (e.g. drizzle,
    heavy, torrential).

    Args:
        slant_range (tuple[float, float]): Range for the rain slant angle in degrees.
            Negative values slant to the left, positive to the right. Default: (-10, 10).
        drop_length (int | None): Length of the rain drops in pixels.
            If None, drop length will be automatically calculated as height // 8.
            This allows the rain effect to scale with the image size.
            Default: None
        drop_width (int): Width of the rain drops in pixels. Default: 1.
        drop_color (tuple[int, int, int]): Color of the rain drops in RGB format. Default: (200, 200, 200).
        blur_value (int): Blur value for simulating rain effect. Rainy views are typically blurry. Default: 7.
        brightness_coefficient (float): Coefficient to adjust the brightness of the image.
            Rainy scenes are usually darker. Should be in the range (0, 1]. Default: 0.7.
        rain_type (Literal['drizzle', 'heavy', 'torrential', 'default']): Type of rain to simulate.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        3

    Note:
        - Rain is drawn as semi-transparent lines; slant simulates wind.
        - rain_type (drizzle, heavy, torrential, default) controls drop count and style.
        - Blur and brightness reduction mimic wet, darker scenes.

    Mathematical Formulation:
        For each raindrop:
        1. Start position (x1, y1) is randomly generated within the image.
        2. End position (x2, y2) is calculated based on drop_length and slant:
           x2 = x1 + drop_length * sin(slant)
           y2 = y1 + drop_length * cos(slant)
        3. A line is drawn from (x1, y1) to (x2, y2) with the specified drop_color and drop_width.
        4. The image is then blurred and its brightness is adjusted.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>>
        >>> # Default usage
        >>> transform = A.RandomRain(p=1.0)
        >>> rainy_image = transform(image=image)["image"]
        >>>
        >>> # Custom rain parameters
        >>> transform = A.RandomRain(
        ...     slant_range=(-15, 15),
        ...     drop_length=30,
        ...     drop_width=2,
        ...     drop_color=(180, 180, 180),
        ...     blur_value=5,
        ...     brightness_coefficient=0.8,
        ...     p=1.0
        ... )
        >>> rainy_image = transform(image=image)["image"]
        >>>
        >>> # Heavy rain
        >>> transform = A.RandomRain(rain_type="heavy", p=1.0)
        >>> heavy_rain_image = transform(image=image)["image"]

    References:
        - Rain visualization techniques: https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-27-real-time-rain-rendering
        - Weather effects in computer vision: https://www.sciencedirect.com/science/article/pii/S1077314220300692

    See Also:
        - RandomSnow: Snow overlay for winter conditions.
        - RandomFog: Patch-based fog without depth.
        - AtmosphericFog: Depth-dependent fog via scattering.

    """

    class InitSchema(BaseTransformInitSchema):
        slant_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-MAX_RAIN_ANGLE, MAX_RAIN_ANGLE)),
        ]
        drop_length: int | None
        drop_width: int = Field(ge=1)
        drop_color: tuple[int, int, int]
        blur_value: int = Field(ge=1)
        brightness_coefficient: float = Field(gt=0, le=1)
        rain_type: Literal["drizzle", "heavy", "torrential", "default"]

    def __init__(
        self,
        slant_range: tuple[float, float] = (-10, 10),
        drop_length: int | None = None,
        drop_width: int = 1,
        drop_color: tuple[int, int, int] = (200, 200, 200),
        blur_value: int = 7,
        brightness_coefficient: float = 0.7,
        rain_type: Literal["drizzle", "heavy", "torrential", "default"] = "default",
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.slant_range = slant_range
        self.drop_length = drop_length
        self.drop_width = drop_width
        self.drop_color = drop_color
        self.blur_value = blur_value
        self.brightness_coefficient = brightness_coefficient
        self.rain_type = rain_type

    def apply(
        self,
        img: ImageType,
        slant: float,
        drop_length: int,
        rain_drops: np.ndarray,
        **params: Any,
    ) -> ImageType:
        non_rgb_error(img)

        return fpixel.add_rain(
            img,
            slant,
            drop_length,
            self.drop_width,
            self.drop_color,
            self.blur_value,
            self.brightness_coefficient,
            rain_drops,
        )

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = self.get_image_data(data)
        height, width = (metadata["height"], metadata["width"])

        # Simpler calculations, directly following Kornia
        if self.rain_type == "drizzle":
            num_drops = height // 4
        elif self.rain_type == "heavy":
            num_drops = height
        elif self.rain_type == "torrential":
            num_drops = height * 2
        else:
            num_drops = height // 3

        drop_length = max(1, height // 8) if self.drop_length is None else self.drop_length

        # Simplified slant calculation
        slant = self.py_random.uniform(*self.slant_range)

        # Single random call for all coordinates
        if num_drops > 0:
            # Generate all coordinates in one call
            coords = self.random_generator.integers(
                low=[0, 0],
                high=[width, height - drop_length],
                size=(num_drops, 2),
                dtype=np.int32,
            )
            rain_drops = coords
        else:
            rain_drops = np.empty((0, 2), dtype=np.int32)

        self.applied_config = {
            "slant_range": slant,
            "drop_length": drop_length,
            "drop_width": self.drop_width,
            "blur_value": self.blur_value,
            "brightness_coefficient": self.brightness_coefficient,
            "drop_color": self.drop_color,
        }

        return {"drop_length": drop_length, "slant": slant, "rain_drops": rain_drops}


class RandomFog(ImageOnlyTransform):
    """Simulate fog by overlaying semi-transparent circles and blending with a fog color.
    Good for driving or outdoor robustness to weather.

    Fog is built from random circles with controllable intensity; an image-size-dependent
    Gaussian blur is applied to the result. Patch-based (no depth); for distance-dependent
    fog use AtmosphericFog.

    Args:
        fog_coef_range (tuple[float, float]): Range for fog intensity coefficient in [0, 1].
            Default: (0.3, 1).
        alpha_coef (float): Transparency of the fog circles in [0, 1]. Default: 0.08.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        3

    Note:
        - Fog is created by overlaying semi-transparent circles at random positions
          and with random radius; alpha is controlled by alpha_coef.
        - Higher fog_coef values give denser fog; effect is typically stronger toward center
          and gradually decreases toward the edges.
        - A Gaussian blur (dependent on the shorter image dimension) is applied after blending
          to reduce sharpness.

    Mathematical Formulation:
        For each fog particle:
        1. A position (x, y) is randomly generated within the image.
        2. A circle with random radius is drawn at this position.
        3. The circle's alpha (transparency) is determined by the alpha_coef.
        4. These circles are overlaid on the original image to create the fog effect.
        5. A Gaussian blur dependent on the shorter dimension is applied

        The final pixel value is calculated as:
        output = blur((1 - alpha) * original_pixel + alpha * fog_color)

        where alpha is influenced by the fog_coef and alpha_coef parameters.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Default usage
        >>> transform = A.RandomFog(p=1.0)
        >>> foggy_image = transform(image=image)["image"]

        # Custom fog intensity range
        >>> transform = A.RandomFog(fog_coef_range=(0.3, 0.8), p=1.0)
        >>> foggy_image = transform(image=image)["image"]

        # Adjust fog transparency
        >>> transform = A.RandomFog(fog_coef_range=(0.2, 0.5), alpha_coef=0.1, p=1.0)
        >>> foggy_image = transform(image=image)["image"]

    References:
        - Fog: https://en.wikipedia.org/wiki/Fog
        - Atmospheric perspective: https://en.wikipedia.org/wiki/Aerial_perspective

    See Also:
        - AtmosphericFog: Depth-dependent fog via scattering; use when you need
          distance-based haze.
        - RandomRain: Rain streaks and blur for rainy conditions.
        - RandomSnow: Snow overlay for winter conditions.

    """

    class InitSchema(BaseTransformInitSchema):
        fog_coef_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]

        alpha_coef: float = Field(ge=0, le=1)

    def __init__(
        self,
        alpha_coef: float = 0.08,
        fog_coef_range: tuple[float, float] = (0.3, 1),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.fog_coef_range = fog_coef_range
        self.alpha_coef = alpha_coef

    def apply(
        self,
        img: ImageType,
        particle_positions: list[tuple[int, int]],
        radiuses: list[int],
        intensity: float,
        **params: Any,
    ) -> ImageType:
        non_rgb_error(img)
        return fpixel.add_fog(
            img,
            intensity,
            self.alpha_coef,
            particle_positions,
            radiuses,
        )

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        # Select a random fog intensity within the specified range
        intensity = self.py_random.uniform(*self.fog_coef_range)

        image_shape = params["shape"][:2]

        image_height, image_width = image_shape

        # Calculate the size of the fog effect region based on image width and fog intensity
        fog_region_size = max(1, int(image_width // 3 * intensity))

        particle_positions = []

        # Initialize the central region where fog will be most dense
        center_x, center_y = (int(x) for x in fgeometric.center(image_shape))

        # Define the initial size of the foggy area
        current_width = image_width
        current_height = image_height

        # Define shrink factor for reducing the foggy area each iteration
        shrink_factor = 0.1

        max_iterations = 10  # Prevent infinite loop
        iteration = 0

        while current_width > fog_region_size and current_height > fog_region_size and iteration < max_iterations:
            # Calculate the number of particles for this region
            area = current_width * current_height
            particles_in_region = int(
                area / (fog_region_size * fog_region_size) * intensity * 10,
            )

            for _ in range(particles_in_region):
                # Generate random positions within the current region
                x = self.py_random.randint(
                    center_x - current_width // 2,
                    center_x + current_width // 2,
                )
                y = self.py_random.randint(
                    center_y - current_height // 2,
                    center_y + current_height // 2,
                )
                particle_positions.append((x, y))

            # Shrink the region for the next iteration
            current_width = int(current_width * (1 - shrink_factor))
            current_height = int(current_height * (1 - shrink_factor))

            iteration += 1

        radiuses = fpixel.get_fog_particle_radiuses(
            image_shape,
            len(particle_positions),
            intensity,
            self.random_generator,
        )

        self.applied_config = {"fog_coef_range": intensity, "alpha_coef": self.alpha_coef}

        return {
            "particle_positions": particle_positions,
            "intensity": intensity,
            "radiuses": radiuses,
        }


class RandomSunFlare(ImageOnlyTransform):
    """Simulate lens flare: circles of light and rays. src_radius, num_flare_circles, angle
    control the effect. Good for outdoor robustness.

    This transform creates a sun flare effect by overlaying multiple semi-transparent
    circles of varying sizes and intensities along a line originating from a "sun" point.
    It offers two methods: a simple overlay technique and a more complex physics-based approach.

    Args:
        flare_roi (tuple[float, float, float, float]): Region of interest where the sun flare
            can appear. Values are in the range [0, 1] and represent (x_min, y_min, x_max, y_max)
            in relative coordinates. Default: (0, 0, 1, 0.5).
        angle_range (tuple[float, float]): Range of angles (in radians) for the flare direction.
            Values should be in the range [0, 1], where 0 represents 0 radians and 1 represents 2π radians.
            Default: (0, 1).
        num_flare_circles_range (tuple[int, int]): Range for the number of flare circles to generate.
            Default: (6, 10).
        src_radius (int): Radius of the sun circle in pixels. Default: 400.
        src_color (tuple[int, int, int]): Color of the sun in RGB format. Default: (255, 255, 255).
        method (Literal['overlay', 'physics_based']): Method to use for generating the sun flare.
            "overlay" uses a simple alpha blending technique, while "physics_based" simulates
            more realistic optical phenomena. Default: "overlay".

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        3

    Note:
        The transform offers two methods for generating sun flares:

        1. Overlay Method ("overlay']:
           - Creates a simple sun flare effect using basic alpha blending.
           - Steps:
             a. Generate the main sun circle with a radial gradient.
             b. Create smaller flare circles along the flare line.
             c. Blend these elements with the original image using alpha compositing.
           - Characteristics:
             * Faster computation
             * Less realistic appearance
             * Suitable for basic augmentation or when performance is a priority

        2. Physics-based Method ("physics_based']:
           - Simulates more realistic optical phenomena observed in actual lens flares.
           - Steps:
             a. Create a separate flare layer for complex manipulations.
             b. Add the main sun circle and diffraction spikes to simulate light diffraction.
             c. Generate and add multiple flare circles with varying properties.
             d. Apply Gaussian blur to create a soft, glowing effect.
             e. Create and apply a radial gradient mask for natural fading from the center.
             f. Simulate chromatic aberration by applying different blurs to color channels.
             g. Blend the flare with the original image using screen blending mode.
           - Characteristics:
             * More computationally intensive
             * Produces more realistic and visually appealing results
             * Includes effects like diffraction spikes and chromatic aberration
             * Suitable for high-quality augmentation or realistic image synthesis

    Mathematical Formulation:
        For both methods:
        1. Sun position (x_s, y_s) is randomly chosen within the specified ROI.
        2. Flare angle θ is randomly chosen from the angle_range.
        3. For each flare circle i:
           - Position (x_i, y_i) = (x_s + t_i * cos(θ), y_s + t_i * sin(θ))
             where t_i is a random distance along the flare line.
           - Radius r_i is randomly chosen, with larger circles closer to the sun.
           - Alpha (transparency) alpha_i is randomly chosen in the range [0.05, 0.2].
           - Color (R_i, G_i, B_i) is randomly chosen close to src_color.

        Overlay method blending:
        new_pixel = (1 - alpha_i) * original_pixel + alpha_i * flare_color_i

        Physics-based method blending:
        new_pixel = 255 - ((255 - original_pixel) * (255 - flare_pixel) / 255)

        4. Each flare circle is blended with the image using alpha compositing:
           new_pixel = (1 - alpha_i) * original_pixel + alpha_i * flare_color_i

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [1000, 1000, 3], dtype=np.uint8)

        # Default sun flare (overlay method)
        >>> transform = A.RandomSunFlare(p=1.0)
        >>> flared_image = transform(image=image)["image"]

        # Physics-based sun flare with custom parameters

        # Default sun flare
        >>> transform = A.RandomSunFlare(p=1.0)
        >>> flared_image = transform(image=image)["image"]

        # Custom sun flare parameters

        >>> transform = A.RandomSunFlare(
        ...     flare_roi=(0.1, 0, 0.9, 0.3),
        ...     angle_range=(0.25, 0.75),
        ...     num_flare_circles_range=(5, 15),
        ...     src_radius=200,
        ...     src_color=(255, 200, 100),
        ...     method="physics_based",
        ...     p=1.0
        ... )
        >>> flared_image = transform(image=image)["image"]

    References:
        - Lens flare: https://en.wikipedia.org/wiki/Lens_flare
        - Alpha compositing: https://en.wikipedia.org/wiki/Alpha_compositing
        - Diffraction: https://en.wikipedia.org/wiki/Diffraction
        - Chromatic aberration: https://en.wikipedia.org/wiki/Chromatic_aberration
        - Screen blending: https://en.wikipedia.org/wiki/Blend_modes#Screen

    """

    class InitSchema(BaseTransformInitSchema):
        flare_roi: tuple[float, float, float, float]
        src_radius: int = Field(gt=1)
        src_color: tuple[int, ...]

        angle_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]

        num_flare_circles_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
            AfterValidator(nondecreasing),
        ]
        method: Literal["overlay", "physics_based"]

        @model_validator(mode="after")
        def _validate_parameters(self) -> Self:
            (
                flare_center_lower_x,
                flare_center_lower_y,
                flare_center_upper_x,
                flare_center_upper_y,
            ) = self.flare_roi
            if (
                not 0 <= flare_center_lower_x < flare_center_upper_x <= 1
                or not 0 <= flare_center_lower_y < flare_center_upper_y <= 1
            ):
                raise ValueError(f"Invalid flare_roi. Got: {self.flare_roi}")

            return self

    def __init__(
        self,
        flare_roi: tuple[float, float, float, float] = (0, 0, 1, 0.5),
        src_radius: int = 400,
        src_color: tuple[int, ...] = (255, 255, 255),
        angle_range: tuple[float, float] = (0, 1),
        num_flare_circles_range: tuple[int, int] = (6, 10),
        method: Literal["overlay", "physics_based"] = "overlay",
        p: float = 0.5,
    ):
        super().__init__(p=p)

        self.angle_range = angle_range
        self.num_flare_circles_range = num_flare_circles_range

        self.src_radius = src_radius
        self.src_color = src_color
        self.flare_roi = flare_roi
        self.method = method

    def apply(
        self,
        img: ImageType,
        flare_center: tuple[float, float],
        circles: list[Any],
        **params: Any,
    ) -> ImageType:
        non_rgb_error(img)
        if self.method == "overlay":
            return fpixel.add_sun_flare_overlay(
                img,
                flare_center,
                self.src_radius,
                self.src_color,
                circles,
            )
        if self.method == "physics_based":
            return fpixel.add_sun_flare_physics_based(
                img,
                flare_center,
                self.src_radius,
                self.src_color,
                circles,
            )

        raise ValueError(f"Invalid method: {self.method}")

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = self.get_image_data(data)
        height, width = (metadata["height"], metadata["width"])
        diagonal = math.sqrt(height**2 + width**2)

        angle = 2 * math.pi * self.py_random.uniform(*self.angle_range)

        # Calculate flare center in pixel coordinates
        x_min, y_min, x_max, y_max = self.flare_roi
        flare_center_x = int(width * self.py_random.uniform(x_min, x_max))
        flare_center_y = int(height * self.py_random.uniform(y_min, y_max))

        num_circles = self.py_random.randint(*self.num_flare_circles_range)

        # Calculate parameters relative to image size
        step_size = max(1, int(diagonal * 0.01))  # 1% of diagonal, minimum 1 pixel
        max_radius = max(2, int(height * 0.01))  # 1% of height, minimum 2 pixels
        color_range = int(max(self.src_color) * 0.2)  # 20% of max color value

        def line(t: float) -> tuple[float, float]:
            return (
                flare_center_x + t * math.cos(angle),
                flare_center_y + t * math.sin(angle),
            )

        # Generate points along the flare line
        t_range = range(-flare_center_x, width - flare_center_x, step_size)
        points = [line(t) for t in t_range]

        circles = []
        for _ in range(num_circles):
            alpha = self.py_random.uniform(0.05, 0.2)
            point = self.py_random.choice(points)
            rad = self.py_random.randint(1, max_radius)

            # Generate colors relative to src_color
            colors = [self.py_random.randint(max(c - color_range, 0), c) for c in self.src_color]

            circles.append(
                (
                    alpha,
                    (int(point[0]), int(point[1])),
                    pow(rad, 3),
                    tuple(colors),
                ),
            )

        self.applied_config = {
            "angle_range": angle,
            "num_flare_circles_range": num_circles,
            "src_radius": self.src_radius,
            "src_color": self.src_color,
        }

        return {
            "circles": circles,
            "flare_center": (flare_center_x, flare_center_y),
        }


class RandomShadow(ImageOnlyTransform):
    """Simulate cast shadows by darkening random regions. shadow_roi, num_shadows, shadow_dimension
    control placement and softness. Improves lighting robustness.

    This transform adds realistic shadow effects to images, which can be useful for augmenting
    datasets for outdoor scene analysis, autonomous driving, or any computer vision task where
    shadows may be present.

    Args:
        shadow_roi (tuple[float, float, float, float]): Region of the image where shadows
            will appear (x_min, y_min, x_max, y_max). All values should be in range [0, 1].
            Default: (0, 0.5, 1, 1).
        num_shadows_range (tuple[int, int]): Lower and upper limits for the possible number of shadows.
            Default: (1, 2).
        shadow_dimension (int): Number of edges in the shadow polygons. Default: 5.
        shadow_intensity_range (tuple[float, float]): Range for the shadow intensity. Larger value
            means darker shadow. Should be two float values between 0 and 1. Default: (0.5, 0.5).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - Shadows are created by generating random polygons within the specified ROI and
          reducing the brightness of the image in these areas.
        - The number of shadows, their shapes, and intensities can be randomized for variety.
        - This transform is particularly useful for:
          * Augmenting datasets for outdoor scene understanding
          * Improving robustness of object detection models to shadowed conditions
          * Simulating different lighting conditions in synthetic datasets

    Mathematical Formulation:
        For each shadow:
        1. A polygon with `shadow_dimension` vertices is generated within the shadow ROI.
        2. The shadow intensity a is randomly chosen from `shadow_intensity_range`.
        3. For each pixel (x, y) within the polygon:
           new_pixel_value = original_pixel_value * (1 - a)

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Default usage
        >>> transform = A.RandomShadow(p=1.0)
        >>> shadowed_image = transform(image=image)["image"]

        # Custom shadow parameters
        >>> transform = A.RandomShadow(
        ...     shadow_roi=(0.2, 0.2, 0.8, 0.8),
        ...     num_shadows_range=(2, 4),
        ...     shadow_dimension=8,
        ...     shadow_intensity_range=(0.3, 0.7),
        ...     p=1.0
        ... )
        >>> shadowed_image = transform(image=image)["image"]

        # Combining with other transforms
        >>> transform = A.Compose([
        ...     A.RandomShadow(p=0.5),
        ...     A.RandomBrightnessContrast(p=0.5),
        ... ])
        >>> augmented_image = transform(image=image)["image"]

    References:
        - Shadow detection and removal: https://www.sciencedirect.com/science/article/pii/S1047320315002035
        - Shadows in computer vision: https://en.wikipedia.org/wiki/Shadow_detection

    """

    class InitSchema(BaseTransformInitSchema):
        shadow_roi: tuple[float, float, float, float]
        num_shadows_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
            AfterValidator(nondecreasing),
        ]
        shadow_dimension: int = Field(ge=3)

        shadow_intensity_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]

        @model_validator(mode="after")
        def _validate_shadows(self) -> Self:
            shadow_lower_x, shadow_lower_y, shadow_upper_x, shadow_upper_y = self.shadow_roi

            if not 0 <= shadow_lower_x <= shadow_upper_x <= 1 or not 0 <= shadow_lower_y <= shadow_upper_y <= 1:
                raise ValueError(f"Invalid shadow_roi. Got: {self.shadow_roi}")

            return self

    def __init__(
        self,
        shadow_roi: tuple[float, float, float, float] = (0, 0.5, 1, 1),
        num_shadows_range: tuple[int, int] = (1, 2),
        shadow_dimension: int = 5,
        shadow_intensity_range: tuple[float, float] = (0.5, 0.5),
        p: float = 0.5,
    ):
        super().__init__(p=p)

        self.shadow_roi = shadow_roi
        self.shadow_dimension = shadow_dimension
        self.num_shadows_range = num_shadows_range
        self.shadow_intensity_range = shadow_intensity_range

    def apply(
        self,
        img: ImageType,
        vertices_list: list[np.ndarray],
        intensities: np.ndarray,
        **params: Any,
    ) -> ImageType:
        return fpixel.add_shadow(img, vertices_list, intensities)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, list[np.ndarray]]:
        metadata = self.get_image_data(data)
        height, width = (metadata["height"], metadata["width"])

        num_shadows = self.py_random.randint(*self.num_shadows_range)

        x_min, y_min, x_max, y_max = self.shadow_roi

        x_min = int(x_min * width)
        x_max = int(x_max * width)
        y_min = int(y_min * height)
        y_max = int(y_max * height)

        vertices_list = [
            np.stack(
                [
                    self.random_generator.integers(
                        x_min,
                        x_max,
                        size=self.shadow_dimension,
                    ),
                    self.random_generator.integers(
                        y_min,
                        y_max,
                        size=self.shadow_dimension,
                    ),
                ],
                axis=1,
            )
            for _ in range(num_shadows)
        ]

        # Sample shadow intensity for each shadow
        intensities = self.random_generator.uniform(
            *self.shadow_intensity_range,
            size=num_shadows,
        )

        self.applied_config = {
            "num_shadows_range": num_shadows,
            "shadow_dimension": self.shadow_dimension,
            "shadow_roi": self.shadow_roi,
            "shadow_intensity_range": intensities.tolist(),
        }

        return {"vertices_list": vertices_list, "intensities": intensities}


class Spatter(ImageOnlyTransform):
    """Simulate lens occlusion from rain or mud: splatter patterns and optional blur. fill
    and spread control appearance. Good for dirty or wet lens robustness.

    Args:
        mean_range (tuple[float, float]): Mean of the normal distribution for generating the
            liquid layer; sampled per image from `(mean_range[0], mean_range[1])`. For a
            constant value use `(mean, mean)`. Default: (0.65, 0.65).
        std_range (tuple[float, float]): Standard deviation of the normal distribution for
            generating the liquid layer; sampled per image from
            `(std_range[0], std_range[1])`. For a constant value use `(std, std)`.
            Default: (0.3, 0.3).
        gauss_sigma_range (tuple[float, float]): Sigma for Gaussian filtering of the liquid
            layer; sampled per image from `(gauss_sigma_range[0], gauss_sigma_range[1])`. For a
            constant value use `(sigma, sigma)`. Default: (2, 3).
        cutout_threshold_range (tuple[float, float]): Threshold for filtering the liquid layer
            (controls number of drops); sampled per image from
            `(cutout_threshold_range[0], cutout_threshold_range[1])`. For a constant value use
            `(t, t)`. Default: (0.68, 0.68).
        intensity_range (tuple[float, float]): Intensity of corruption; sampled per image from
            `(intensity_range[0], intensity_range[1])`. For a constant value use `(i, i)`.
            Default: (0.6, 0.6).
        mode (Literal['rain', 'mud']): Type of corruption. Default: "rain".
        color (tuple[int, ...] | None): Corruption elements color.
            If list uses provided list as color for the effect.
            If None uses default colors based on mode (rain: (238, 238, 175), mud: (20, 42, 63)).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    References:
        Benchmarking Neural Network Robustness to Common Corruptions and Perturbations: https://arxiv.org/abs/1903.12261

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create a sample image
        >>> image = np.ones((300, 300, 3), dtype=np.uint8) * 200  # Light gray background
        >>> # Add some gradient to make effects more visible
        >>> for i in range(300):
        ...     image[i, :, :] = np.clip(image[i, :, :] - i // 3, 0, 255)
        >>>
        >>> # Example 1: Rain effect with default parameters
        >>> rain_transform = A.Spatter(
        ...     mode="rain",
        ...     p=1.0
        ... )
        >>> rain_result = rain_transform(image=image)
        >>> rain_image = rain_result['image']  # Image with rain drops
        >>>
        >>> # Example 2: Heavy rain with custom parameters
        >>> heavy_rain = A.Spatter(
        ...     mode="rain",
        ...     mean_range=(0.7, 0.7),                # Higher mean = more coverage
        ...     std_range=(0.2, 0.2),                 # Lower std = more uniform effect
        ...     cutout_threshold_range=(0.65, 0.65),  # Lower threshold = more drops
        ...     intensity_range=(0.8, 0.8),           # Higher intensity = more visible effect
        ...     color=(200, 200, 255),                # Blueish rain drops
        ...     p=1.0
        ... )
        >>> heavy_rain_result = heavy_rain(image=image)
        >>> heavy_rain_image = heavy_rain_result['image']
        >>>
        >>> # Example 3: Mud effect
        >>> mud_transform = A.Spatter(
        ...     mode="mud",
        ...     mean_range=(0.6, 0.6),
        ...     std_range=(0.3, 0.3),
        ...     cutout_threshold_range=(0.62, 0.62),
        ...     intensity_range=(0.7, 0.7),
        ...     p=1.0
        ... )
        >>> mud_result = mud_transform(image=image)
        >>> mud_image = mud_result['image']  # Image with mud splatters
        >>>
        >>> # Example 4: Custom colored mud
        >>> red_mud = A.Spatter(
        ...     mode="mud",
        ...     mean_range=(0.55, 0.55),
        ...     std_range=(0.25, 0.25),
        ...     cutout_threshold_range=(0.7, 0.7),
        ...     intensity_range=(0.6, 0.6),
        ...     color=(120, 40, 40),  # Reddish-brown mud
        ...     p=1.0
        ... )
        >>> red_mud_result = red_mud(image=image)
        >>> red_mud_image = red_mud_result['image']
        >>>
        >>> # Example 5: Random effect (50% chance of applying)
        >>> random_spatter = A.Compose([
        ...     A.Spatter(
        ...         mode="rain" if np.random.random() < 0.5 else "mud",
        ...         p=0.5
        ...     )
        ... ])
        >>> random_result = random_spatter(image=image)
        >>> result_image = random_result['image']  # May or may not have spatter effect

    """

    class InitSchema(BaseTransformInitSchema):
        mean_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        std_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        gauss_sigma_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0)),
            AfterValidator(nondecreasing),
        ]
        cutout_threshold_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        intensity_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        mode: Literal["rain", "mud"]
        color: Sequence[int] | None

        @model_validator(mode="after")
        def _check_color(self) -> Self:
            # Default colors for each mode
            default_colors = {"rain": [238, 238, 175], "mud": [20, 42, 63]}

            if self.color is None:
                # Use default color for the selected mode
                self.color = default_colors[self.mode]
            # Validate the provided color
            elif len(self.color) != NUM_RGB_CHANNELS:
                msg = "Color must be a list of three integers for RGB format."
                raise ValueError(msg)
            return self

    def __init__(
        self,
        mean_range: tuple[float, float] = (0.65, 0.65),
        std_range: tuple[float, float] = (0.3, 0.3),
        gauss_sigma_range: tuple[float, float] = (2, 2),
        cutout_threshold_range: tuple[float, float] = (0.68, 0.68),
        intensity_range: tuple[float, float] = (0.6, 0.6),
        mode: Literal["rain", "mud"] = "rain",
        color: Sequence[int] | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.mean_range = mean_range
        self.std_range = std_range
        self.gauss_sigma_range = gauss_sigma_range
        self.cutout_threshold_range = cutout_threshold_range
        self.intensity_range = intensity_range
        self.mode = mode
        self.color = cast("tuple[int, ...]", color)

    def apply(
        self,
        img: ImageType,
        **params: dict[str, Any],
    ) -> ImageType:
        non_rgb_error(img)

        if params["mode"] == "rain":
            return fpixel.spatter_rain(img, params["drops"])

        return fpixel.spatter_mud(img, params["non_mud"], params["mud"])

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = self.get_image_data(data)
        height, width = (metadata["height"], metadata["width"])

        mean = self.py_random.uniform(*self.mean_range)
        std = self.py_random.uniform(*self.std_range)
        cutout_threshold = self.py_random.uniform(*self.cutout_threshold_range)
        sigma = self.py_random.uniform(*self.gauss_sigma_range)
        mode = self.mode
        intensity = self.py_random.uniform(*self.intensity_range)
        color = np.array(self.color) / 255.0

        liquid_layer = self.random_generator.normal(
            size=(height, width),
            loc=mean,
            scale=std,
        )
        # Convert sigma to kernel size (must be odd)
        ksize = int(2 * round(3 * sigma) + 1)  # 3 sigma rule, rounded to nearest odd
        cv2.GaussianBlur(
            src=liquid_layer,
            dst=liquid_layer,  # in-place operation
            ksize=(ksize, ksize),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE,
        )

        # Important line, without it the rain effect looses drops
        liquid_layer[liquid_layer < cutout_threshold] = 0

        self.applied_config = {
            "mean_range": mean,
            "std_range": std,
            "cutout_threshold_range": cutout_threshold,
            "gauss_sigma_range": sigma,
            "intensity_range": intensity,
        }

        if mode == "rain":
            return {
                "mode": "rain",
                **fpixel.get_rain_params(liquid_layer=liquid_layer, color=color, intensity=intensity),
            }

        return {
            "mode": "mud",
            **fpixel.get_mud_params(
                liquid_layer=liquid_layer,
                color=color,
                cutout_threshold=cutout_threshold,
                sigma=sigma,
                intensity=intensity,
                random_generator=self.random_generator,
            ),
        }


class AtmosphericFog(ImageOnlyTransform):
    """Add depth-dependent fog via the atmospheric scattering equation and a synthetic depth map.
    Use for outdoor and driving robustness to haze.

    Unlike RandomFog (which overlays circular fog patches), this transform uses a
    physically-based scattering model: farther pixels (by synthetic depth) get more
    fog, producing realistic distance-dependent haze. Depth is derived from image
    position (linear, diagonal, or radial), not from a real depth map.

    Formula: `result = image * exp(-density * depth) + fog_color * (1 - exp(-density * depth))`

    Args:
        density_range (tuple[float, float]): Range for fog density. Higher values
            give thicker fog. Default: (1.0, 3.0).
        fog_color (tuple[int, ...]): Fog color per channel, e.g. (R, G, B) for 3
            channels. Length must match image channels. Default: (200, 200, 200).
        depth_mode (Literal['linear', 'diagonal', 'radial']): How synthetic depth
            is generated:
            - "linear": top of image = far, bottom = near (sky vs ground).
            - "diagonal": top-left = far.
            - "radial": center = near, edges = far.
            Default: "linear".
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - Depth is synthetic (from pixel position), not from scene geometry.
        - For typical outdoor frames, "linear" matches sky far / ground near.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.AtmosphericFog(density_range=(1.0, 2.5), depth_mode="linear", p=1.0)
        >>> result = transform(image=image)["image"]
        >>> # Radial fog (center clear, edges foggy)
        >>> transform_radial = A.AtmosphericFog(density_range=(1.5, 3.0), depth_mode="radial", p=1.0)
        >>> result_radial = transform_radial(image=image)["image"]

    See Also:
        - RandomFog: Patch-based fog without depth; simpler and faster when
          distance-dependent haze is not needed.
        - RandomRain: Rain streaks and blur for rainy-scene robustness.
        - RandomSnow: Snow overlay (bleach or texture) for winter conditions.
        - LensFlare: Starburst and ghost reflections for optical artifacts.

    """

    class InitSchema(BaseTransformInitSchema):
        density_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        fog_color: tuple[int, ...]
        depth_mode: Literal["linear", "diagonal", "radial"]

    def __init__(
        self,
        density_range: tuple[float, float] = (1.0, 3.0),
        fog_color: tuple[int, ...] = (200, 200, 200),
        depth_mode: Literal["linear", "diagonal", "radial"] = "linear",
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.density_range = density_range
        self.fog_color = fog_color
        self.depth_mode = depth_mode

    def apply(
        self,
        img: ImageType,
        density: float,
        depth_map: np.ndarray,
        fog_color_scaled: tuple[float, ...],
        **params: Any,
    ) -> ImageType:
        return fpixel.apply_atmospheric_fog(img, density, fog_color_scaled, depth_map)

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self._apply_to_batch_same_shape(images, lambda image: self.apply(image, **params))

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        height, width = params["shape"][:2]
        density = self.py_random.uniform(*self.density_range)

        if self.depth_mode == "linear":
            depth_map = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, np.newaxis]
            depth_map = np.broadcast_to(depth_map, (height, width)).copy()
        elif self.depth_mode == "diagonal":
            y = np.linspace(1.0, 0.0, height, dtype=np.float32)
            x = np.linspace(1.0, 0.0, width, dtype=np.float32)
            depth_map = (y[:, np.newaxis] + x[np.newaxis, :]) / 2.0
        else:
            cy, cx = height / 2.0, width / 2.0
            y = np.arange(height, dtype=np.float32)
            x = np.arange(width, dtype=np.float32)
            dist = np.sqrt((y[:, np.newaxis] - cy) ** 2 + (x[np.newaxis, :] - cx) ** 2)
            max_dist = np.sqrt(cy**2 + cx**2)
            depth_map = (dist / max_dist).astype(np.float32)

        max_val = float(albucore.MAX_VALUES_BY_DTYPE[np.uint8])
        image_data = self.get_image_data(data)
        img_dtype = image_data["dtype"]
        actual_max = float(albucore.MAX_VALUES_BY_DTYPE[img_dtype])
        fog_color_scaled = tuple(c / max_val * actual_max for c in self.fog_color)

        self.applied_config = {"density_range": density}
        return {
            "density": density,
            "depth_map": depth_map,
            "fog_color_scaled": fog_color_scaled,
        }
