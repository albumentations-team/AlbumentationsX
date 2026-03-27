"""Noise injection transforms.

Transforms that add various types of noise to images, including Gaussian,
ISO, multiplicative, shot, salt-and-pepper, additive, and film grain noise.
"""

from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypeAlias

import cv2
import numpy as np
from albucore import (
    MAX_VALUES_BY_DTYPE,
    get_image_data,
    multiply,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
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
from albumentations.core.type_definitions import PAIR, ImageType, VolumeType

__all__ = [
    "AdditiveNoise",
    "FilmGrain",
    "GaussNoise",
    "ISONoise",
    "MultiplicativeNoise",
    "SaltAndPepper",
    "ShotNoise",
]


class GaussNoise(ImageOnlyTransform):
    """Add Gaussian (normal) noise to the image. i.i.d. per pixel (or per block if scaled).
    Use for robustness to sensor or transmission noise.

    Noise standard deviation and mean are sampled from configurable ranges and scaled
    to image dtype (255 for uint8, 1.0 for float32). Optional per-channel sampling
    and lower-resolution noise for speed.

    Args:
        std_range (tuple[float, float]): Range for noise standard deviation as a fraction
            of the max value (255 for uint8, 1.0 for float32). In [0, 1]. Default: (0.2, 0.44).
        mean_range (tuple[float, float]): Range for noise mean as a fraction of max.
            In [-1, 1]. Default: (0.0, 0.0).
        per_channel (bool): If True, sample noise per channel; else same noise for all.
            Default: False.
        noise_scale_factor (float): If < 1, noise is generated at lower resolution and
            resized (faster, coarser). 1 = per-pixel. In (0, 1]. Default: 1.0.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - std_range and mean_range are in [0, 1] / [-1, 1]; scaled by 255 (uint8) or
          used directly (float32).
        - per_channel=False: faster, same noise on all channels (grayscale-like on RGB).
        - per_channel=True: different noise per channel (colored noise).
        - noise_scale_factor < 1 trades speed for noise granularity.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>>
        >>> transform = A.GaussNoise(std_range=(0.1, 0.2), p=1.0)
        >>> noisy_image = transform(image=image)["image"]

    See Also:
        - FilmGrain: Luminance-dependent, spatially correlated (film-like) noise.
        - ShotNoise: Poisson noise in linear space; sensor-realistic for low light.

    """

    class InitSchema(BaseTransformInitSchema):
        std_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        mean_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(-1, 1)),
            AfterValidator(nondecreasing),
        ]
        per_channel: bool
        noise_scale_factor: float = Field(gt=0, le=1)

    def __init__(
        self,
        std_range: tuple[float, float] = (0.2, 0.44),  # sqrt(10 / 255), sqrt(50 / 255)
        mean_range: tuple[float, float] = (0.0, 0.0),
        per_channel: bool = False,
        noise_scale_factor: float = 1,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.std_range = std_range
        self.mean_range = mean_range
        self.per_channel = per_channel
        self.noise_scale_factor = noise_scale_factor

    def apply(
        self,
        img: ImageType,
        noise_map: np.ndarray,
        **params: Any,
    ) -> ImageType:
        return fpixel.add_noise(img, noise_map)

    def apply_to_images(self, images: ImageType, noise_map: np.ndarray, **params: Any) -> ImageType:
        return fpixel.add_noise(images, noise_map)

    def apply_to_volumes(self, volumes: VolumeType, noise_map: np.ndarray, **params: Any) -> VolumeType:
        return fpixel.add_noise(volumes, noise_map)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, float]:
        metadata = get_image_data(data)
        max_value = MAX_VALUES_BY_DTYPE[metadata["dtype"]]
        shape = (metadata["height"], metadata["width"], metadata["num_channels"])

        sigma = self.py_random.uniform(*self.std_range)
        mean = self.py_random.uniform(*self.mean_range)

        self.applied_config = {"std_range": sigma, "mean_range": mean}

        noise_map = fpixel.generate_spatial_noise(
            noise_type="gaussian",
            spatial_mode="per_pixel" if self.per_channel else "shared",
            shape=shape,
            params={"mean_range": (mean, mean), "std_range": (sigma, sigma)},
            max_value=max_value,
            approximation=self.noise_scale_factor,
            random_generator=self.random_generator,
        )
        return {"noise_map": noise_map}


class ISONoise(ImageOnlyTransform):
    """Add camera-sensor-like noise scaling with intensity (high ISO). color_shift and
    intensity range control strength. Good for low-light or camera noise simulation.

    This transform adds random noise to an image, mimicking the effect of using high ISO settings
    in digital photography. It simulates two main components of ISO noise:
    1. Color noise: random shifts in color hue
    2. Luminance noise: random variations in pixel intensity

    Args:
        color_shift (tuple[float, float]): Range for changing color hue.
            Values should be in the range [0, 1], where 1 represents a full 360° hue rotation.
            Default: (0.01, 0.05)

        intensity (tuple[float, float]): Range for the noise intensity.
            Higher values increase the strength of both color and luminance noise.
            Default: (0.1, 0.5)

        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        3

    Note:
        - This transform only works with RGB images. It will raise a TypeError if applied to
          non-RGB images.
        - The color shift is applied in the HSV color space, affecting the hue channel.
        - Luminance noise is added to all channels independently.
        - This transform can be useful for data augmentation in low-light scenarios or when
          training models to be robust against noisy inputs.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5)
        >>> result = transform(image=image)
        >>> noisy_image = result["image"]

    References:
        ISO noise in digital photography: https://en.wikipedia.org/wiki/Image_noise#In_digital_cameras

    """

    class InitSchema(BaseTransformInitSchema):
        color_shift: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        intensity: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        color_shift: tuple[float, float] = (0.01, 0.05),
        intensity: tuple[float, float] = (0.1, 0.5),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.intensity = intensity
        self.color_shift = color_shift

    def apply(
        self,
        img: ImageType,
        color_shift: float,
        intensity: float,
        random_seed: int,
        **params: Any,
    ) -> ImageType:
        non_rgb_error(img)
        return fpixel.iso_noise(
            img,
            color_shift,
            intensity,
            np.random.default_rng(random_seed),
        )

    def apply_to_images(
        self,
        images: ImageType,
        color_shift: float,
        intensity: float,
        random_seed: int,
        **params: Any,
    ) -> ImageType:
        return fpixel.iso_noise_images(
            images,
            color_shift,
            intensity,
            np.random.default_rng(random_seed),
        )

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        random_seed = self.random_generator.integers(0, 2**32 - 1)
        color_shift = self.py_random.uniform(*self.color_shift)
        intensity = self.py_random.uniform(*self.intensity)

        self.applied_config = {"color_shift": color_shift, "intensity": intensity}

        return {
            "color_shift": color_shift,
            "intensity": intensity,
            "random_seed": random_seed,
        }


class MultiplicativeNoise(ImageOnlyTransform):
    """Multiply image by random per-pixel or per-channel factor. multiplier_range controls
    strength. Simulates illumination or gain variation; preserves zeros.

    This transform multiplies each pixel in the image by a random value or array of values,
    effectively creating a noise pattern that scales with the image intensity.

    Args:
        multiplier (tuple[float, float]): The range for the random multiplier.
            Defines the range from which the multiplier is sampled.
            Default: (0.9, 1.1)

        per_channel (bool): If True, use a different random multiplier for each channel.
            If False, use the same multiplier for all channels.
            Setting this to False is slightly faster.
            Default: False

        elementwise (bool): If True, generates a unique multiplier for each pixel.
            If False, generates a single multiplier (or one per channel if per_channel=True).
            Default: False

        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - When elementwise=False and per_channel=False, a single multiplier is applied to the entire image.
        - When elementwise=False and per_channel=True, each channel gets a different multiplier.
        - When elementwise=True and per_channel=False, each pixel gets the same multiplier across all channels.
        - When elementwise=True and per_channel=True, each pixel in each channel gets a unique multiplier.
        - Setting per_channel=False is slightly faster, especially for larger images.
        - This transform can be used to simulate various lighting conditions or to create noise that
          scales with image intensity.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=1.0)
        >>> result = transform(image=image)
        >>> noisy_image = result["image"]

    References:
        Multiplicative noise: https://en.wikipedia.org/wiki/Multiplicative_noise

    """

    class InitSchema(BaseTransformInitSchema):
        multiplier: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        per_channel: bool
        elementwise: bool

    def __init__(
        self,
        multiplier: tuple[float, float] = (0.9, 1.1),
        per_channel: bool = False,
        elementwise: bool = False,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.multiplier = multiplier
        self.elementwise = elementwise
        self.per_channel = per_channel

    def apply(
        self,
        img: ImageType,
        multiplier: float | np.ndarray,
        **kwargs: Any,
    ) -> ImageType:
        return multiply(img, multiplier)

    def apply_to_images(self, images: ImageType, multiplier: float | np.ndarray, **kwargs: Any) -> ImageType:
        return self.apply(images, multiplier, **kwargs)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = get_image_data(data)
        image_shape = (metadata["height"], metadata["width"], metadata["num_channels"])
        num_channels = image_shape[-1]

        if self.elementwise:
            multiplier_shape: tuple[int, ...] = image_shape if self.per_channel else (*image_shape[:2], 1)
        else:
            multiplier_shape = (num_channels,) if self.per_channel else (1,)

        multiplier = self.random_generator.uniform(
            self.multiplier[0],
            self.multiplier[1],
            multiplier_shape,
        ).astype(np.float32)

        if not self.per_channel and num_channels > 1:
            # Replicate the multiplier for all channels if not per_channel
            multiplier = np.repeat(multiplier, num_channels, axis=-1)

        if not self.elementwise and self.per_channel:
            # Reshape to broadcast correctly when not elementwise but per_channel
            multiplier = multiplier.reshape(1, 1, -1)

        if multiplier.shape != image_shape:
            multiplier = multiplier.squeeze()

        return {"multiplier": multiplier}


class ShotNoise(ImageOnlyTransform):
    """Shot noise (Poisson) in linear light space. Sensor-realistic; use for low-light
    or photon-limited imaging and camera simulation.

    Simulates photon-counting: convert to linear space (gamma removed), treat pixel
    values as expected photon counts, sample from Poisson, convert back. Variance
    equals mean in linear space; brighter regions have more absolute noise, less relative.

    Args:
        scale_range (tuple[float, float]): Reciprocal of photons per unit intensity.
            Higher = more noise. e.g. 0.1 ≈ low, 1.0 ≈ moderate, 10.0 ≈ high. Default: (0.1, 0.3).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - Pipeline: linear space (gamma = 2.2), Poisson sample, back to display space.
        - Preserves mean intensity. Per-pixel, per-channel independent.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>>
        >>> transform = A.ShotNoise(scale_range=(0.1, 1.0), p=1.0)
        >>> noisy_image = transform(image=image)["image"]

    References:
        - Shot noise: https://en.wikipedia.org/wiki/Shot_noise
        - Original paper: https://doi.org/10.1002/andp.19183622304 (Schottky, 1918)
        - Poisson process: https://en.wikipedia.org/wiki/Poisson_point_process
        - Gamma correction: https://en.wikipedia.org/wiki/Gamma_correction

    See Also:
        - GaussNoise: i.i.d. Gaussian noise; use for sensor or transmission noise.
        - FilmGrain: Luminance-dependent, spatially correlated (film-like) noise.

    """

    class InitSchema(BaseTransformInitSchema):
        scale_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, None)),
        ]

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.1, 0.3),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.scale_range = scale_range

    def apply(
        self,
        img: ImageType,
        scale: float,
        random_seed: int,
        **params: Any,
    ) -> ImageType:
        return fpixel.shot_noise(img, scale, np.random.default_rng(random_seed))

    def get_params(self) -> dict[str, Any]:
        scale = self.py_random.uniform(*self.scale_range)
        self.applied_config = {"scale_range": scale}
        return {
            "scale": scale,
            "random_seed": self.random_generator.integers(0, 2**32 - 1),
        }


class NoiseParamsBase(BaseModel):
    """Base Pydantic model for AdditiveNoise noise params (uniform, gaussian, laplace, beta).
    Subclasses define noise_type and distribution-specific fields.
    """

    model_config = ConfigDict(extra="forbid")
    noise_type: str


class UniformParams(NoiseParamsBase):
    noise_type: Literal["uniform"] = "uniform"
    ranges: list[Sequence[float]] = Field(min_length=1)

    @field_validator("ranges", mode="after")
    @classmethod
    def _validate_ranges(cls, v: list[Sequence[float]]) -> list[tuple[float, float]]:
        result = []
        for range_values in v:
            if len(range_values) != PAIR:
                raise ValueError("Each range must have exactly 2 values")
            min_val, max_val = range_values
            if not (-1 <= min_val <= max_val <= 1):
                raise ValueError("Range values must be in [-1, 1] and min <= max")
            result.append((float(min_val), float(max_val)))
        return result


class GaussianParams(NoiseParamsBase):
    noise_type: Literal["gaussian"] = "gaussian"
    mean_range: Annotated[
        Sequence[float],
        AfterValidator(check_range_bounds(min_val=-1, max_val=1)),
    ]
    std_range: Annotated[
        Sequence[float],
        AfterValidator(check_range_bounds(min_val=0, max_val=1)),
    ]


class LaplaceParams(NoiseParamsBase):
    noise_type: Literal["laplace"] = "laplace"
    mean_range: Annotated[
        Sequence[float],
        AfterValidator(check_range_bounds(min_val=-1, max_val=1)),
    ]
    scale_range: Annotated[
        Sequence[float],
        AfterValidator(check_range_bounds(min_val=0, max_val=1)),
    ]


class BetaParams(NoiseParamsBase):
    noise_type: Literal["beta"] = "beta"
    alpha_range: Annotated[
        Sequence[float],
        AfterValidator(check_range_bounds(min_val=0)),
    ]
    beta_range: Annotated[
        Sequence[float],
        AfterValidator(check_range_bounds(min_val=0)),
    ]
    scale_range: Annotated[
        Sequence[float],
        AfterValidator(check_range_bounds(min_val=0, max_val=1)),
    ]


NoiseParams: TypeAlias = Annotated[
    UniformParams | GaussianParams | LaplaceParams | BetaParams,
    Field(discriminator="noise_type"),
]


class AdditiveNoise(ImageOnlyTransform):
    """Random noise to channels: uniform, gaussian, laplace, or beta. spatial_mode: constant,
    per_pixel, or shared. Params depend on noise_type.

    This transform generates noise using different probability distributions and applies it
    to image channels. The noise can be generated in three spatial modes and supports
    multiple noise distributions, each with configurable parameters.

    Args:
        noise_type(Literal['uniform', 'gaussian', 'laplace', 'beta']): Type of noise distribution to use. Options:
            - "uniform": Uniform distribution, good for simple random perturbations
            - "gaussian": Normal distribution, models natural random processes
            - "laplace": Similar to Gaussian but with heavier tails, good for outliers
            - "beta": Flexible bounded distribution, can be symmetric or skewed

        spatial_mode(Literal['constant', 'per_pixel', 'shared']): How to generate and apply the noise. Options:
            - "constant": One noise value per channel, fastest
            - "per_pixel": Independent noise value for each pixel and channel, slowest
            - "shared": One noise map shared across all channels, medium speed

        approximation(float): float in [0, 1], default=1.0
            Controls noise generation speed vs quality tradeoff.
            - 1.0: Generate full resolution noise (slowest, highest quality)
            - 0.5: Generate noise at half resolution and upsample
            - 0.25: Generate noise at quarter resolution and upsample
            Only affects 'per_pixel' and 'shared' spatial modes.

        noise_params(dict[str, Any] | None): Parameters for the chosen noise distribution.
            Must match the noise_type:

            uniform:
                ranges: list[tuple[float, float]]
                    List of (min, max) ranges for each channel.
                    Each range must be in [-1, 1].
                    If only one range is provided, it will be used for all channels.

                    [(-0.2, 0.2)]  # Same range for all channels
                    [(-0.2, 0.2), (-0.1, 0.1), (-0.1, 0.1)]  # Different ranges for RGB

            gaussian:
                mean_range: tuple[float, float], default (0.0, 0.0)
                    Range for sampling mean value, in [-1, 1]
                std_range: tuple[float, float], default (0.1, 0.1)
                    Range for sampling standard deviation, in [0, 1]

            laplace:
                mean_range: tuple[float, float], default (0.0, 0.0)
                    Range for sampling location parameter, in [-1, 1]
                scale_range: tuple[float, float], default (0.1, 0.1)
                    Range for sampling scale parameter, in [0, 1]

            beta:
                alpha_range: tuple[float, float], default (0.5, 1.5)
                    Value < 1 = U-shaped, Value > 1 = Bell-shaped
                    Range for sampling first shape parameter, in (0, inf)
                beta_range: tuple[float, float], default (0.5, 1.5)
                    Value < 1 = U-shaped, Value > 1 = Bell-shaped
                    Range for sampling second shape parameter, in (0, inf)
                scale_range: tuple[float, float], default (0.1, 0.3)
                    Smaller scale for subtler noise
                    Range for sampling output scale, in [0, 1]

    Examples:
        >>> # Constant RGB shift with different ranges per channel:
        >>> transform = AdditiveNoise(
        ...     noise_type="uniform",
        ...     spatial_mode="constant",
        ...     noise_params={"ranges": [(-0.2, 0.2), (-0.1, 0.1), (-0.1, 0.1)]}
        ... )

        Gaussian noise shared across channels:
        >>> transform = AdditiveNoise(
        ...     noise_type="gaussian",
        ...     spatial_mode="shared",
        ...     noise_params={"mean_range": (0.0, 0.0), "std_range": (0.05, 0.15)}
        ... )

    Note:
        Performance considerations:
            - "constant" mode is fastest as it generates only C values (C = number of channels)
            - "shared" mode generates HxW values and reuses them for all channels
            - "per_pixel" mode generates HxWxC values, slowest but most flexible

        Distribution characteristics:
            - uniform: Equal probability within range, good for simple perturbations
            - gaussian: Bell-shaped, symmetric, good for natural noise
            - laplace: Like gaussian but with heavier tails, good for outliers
            - beta: Very flexible shape, can be uniform, bell-shaped, or U-shaped

        Implementation details:
            - All noise is generated in normalized range and scaled by image max value
            - For uint8 images, final noise range is [-255, 255]
            - For float images, final noise range is [-1, 1]

    """

    class InitSchema(BaseTransformInitSchema):
        noise_type: Literal["uniform", "gaussian", "laplace", "beta"]
        spatial_mode: Literal["constant", "per_pixel", "shared"]
        noise_params: dict[str, Any] | None
        approximation: float = Field(ge=0.0, le=1.0)

        @model_validator(mode="after")
        def _validate_noise_params(self) -> Self:
            # Default parameters for each noise type
            default_params = {
                "uniform": {
                    "ranges": [(-0.1, 0.1)],  # Single channel by default
                },
                "gaussian": {"mean_range": (0.0, 0.0), "std_range": (0.05, 0.15)},
                "laplace": {"mean_range": (0.0, 0.0), "scale_range": (0.05, 0.15)},
                "beta": {
                    "alpha_range": (0.5, 1.5),
                    "beta_range": (0.5, 1.5),
                    "scale_range": (0.1, 0.3),
                },
            }

            # Use default params if none provided
            params_dict = self.noise_params if self.noise_params is not None else default_params[self.noise_type]

            # Add noise_type to params if not present
            params_dict = {**params_dict, "noise_type": self.noise_type}  # type: ignore[dict-item]

            # Convert dict to appropriate NoiseParams object and validate
            params_class = {
                "uniform": UniformParams,
                "gaussian": GaussianParams,
                "laplace": LaplaceParams,
                "beta": BetaParams,
            }[self.noise_type]

            # Validate using the appropriate NoiseParams class
            validated_params = params_class(**params_dict)

            # Store the validated parameters as a dict
            self.noise_params = validated_params.model_dump()

            return self

    def __init__(
        self,
        noise_type: Literal["uniform", "gaussian", "laplace", "beta"] = "uniform",
        spatial_mode: Literal["constant", "per_pixel", "shared"] = "constant",
        noise_params: dict[str, Any] | None = None,
        approximation: float = 1.0,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.noise_type = noise_type
        self.spatial_mode = spatial_mode
        self.noise_params = noise_params
        self.approximation = approximation

    def apply(
        self,
        img: ImageType,
        noise_map: np.ndarray,
        **params: Any,
    ) -> ImageType:
        return fpixel.add_noise(img, noise_map)

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self.apply(images, **params)

    def apply_to_volumes(self, volumes: VolumeType, **params: Any) -> VolumeType:
        return self.apply(volumes, **params)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = get_image_data(data)
        max_value = MAX_VALUES_BY_DTYPE[metadata["dtype"]]
        shape = (metadata["height"], metadata["width"], metadata["num_channels"])

        if self.spatial_mode == "constant":
            noise_map = fpixel.generate_constant_noise_with_py_random(
                noise_type=self.noise_type,
                shape=shape,
                params=self.noise_params,
                max_value=max_value,
                py_random=self.py_random,
            )
        else:
            noise_map = fpixel.generate_spatial_noise(
                noise_type=self.noise_type,
                spatial_mode=self.spatial_mode,
                shape=shape,
                params=self.noise_params,
                max_value=max_value,
                approximation=self.approximation,
                random_generator=self.random_generator,
            )
        return {"noise_map": noise_map}


class SaltAndPepper(ImageOnlyTransform):
    """Apply salt-and-pepper (impulse) noise: randomly set pixels to min or max. amount and
    salt_vs_pepper control density and ratio. Same mask for all channels.

    Salt and pepper noise is a form of impulse noise that randomly sets pixels to either maximum value (salt)
    or minimum value (pepper). The amount and proportion of salt vs pepper can be controlled.
    The same noise mask is applied to all channels of the image to preserve color consistency.

    Args:
        amount ((float, float)): Range for total amount of noise (both salt and pepper).
            Values between 0 and 1. For example:
            - 0.05 means 5% of all pixels will be replaced with noise
            - (0.01, 0.06) will sample amount uniformly from 1% to 6%
            Default: (0.01, 0.06)

        salt_vs_pepper ((float, float)): Range for ratio of salt (white) vs pepper (black) noise.
            Values between 0 and 1. For example:
            - 0.5 means equal amounts of salt and pepper
            - 0.7 means 70% of noisy pixels will be salt, 30% pepper
            - (0.4, 0.6) will sample ratio uniformly from 40% to 60%
            Default: (0.4, 0.6)

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Note:
        - Salt noise sets pixels to maximum value (255 for uint8, 1.0 for float32)
        - Pepper noise sets pixels to 0
        - The noise mask is generated once and applied to all channels to maintain
          color consistency (i.e., if a pixel is set to salt, all its color channels
          will be set to maximum value)
        - The exact number of affected pixels matches the specified amount as masks
          are generated without overlap

    Mathematical Formulation:
        For an input image I, the output O is:
        O[c,x,y] = max_value,  if salt_mask[x,y] = True
        O[c,x,y] = 0,         if pepper_mask[x,y] = True
        O[c,x,y] = I[c,x,y],  otherwise

        where:
        - c is the channel index
        - salt_mask and pepper_mask are 2D boolean arrays applied to all channels
        - Number of True values in salt_mask = floor(H*W * amount * salt_ratio)
        - Number of True values in pepper_mask = floor(H*W * amount * (1 - salt_ratio))
        - amount ∈ [amount_min, amount_max]
        - salt_ratio ∈ [salt_vs_pepper_min, salt_vs_pepper_max]

    Examples:
        >>> import albumentations as A
        >>> import numpy as np

        # Apply salt and pepper noise with default parameters
        >>> transform = A.SaltAndPepper(p=1.0)
        >>> noisy_image = transform(image=image)["image"]

        # Heavy noise with more salt than pepper
        >>> transform = A.SaltAndPepper(
        ...     amount=(0.1, 0.2),       # 10-20% of pixels will be noisy
        ...     salt_vs_pepper=(0.7, 0.9),  # 70-90% of noise will be salt
        ...     p=1.0
        ... )
        >>> noisy_image = transform(image=image)["image"]

    References:
        - Digital Image Processing: Rafael C. Gonzalez and Richard E. Woods, 4th Edition,
            Chapter 5: Image Restoration and Reconstruction.
        - Fundamentals of Digital Image Processing: A. K. Jain, Chapter 7: Image Degradation and Restoration.
        - Salt and pepper noise: https://en.wikipedia.org/wiki/Salt-and-pepper_noise

    See Also:
        - GaussNoise: For additive Gaussian noise
        - MultiplicativeNoise: For multiplicative noise
        - ISONoise: For camera sensor noise simulation

    """

    class InitSchema(BaseTransformInitSchema):
        amount: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, 1))]
        salt_vs_pepper: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, 1))]

    def __init__(
        self,
        amount: tuple[float, float] = (0.01, 0.06),
        salt_vs_pepper: tuple[float, float] = (0.4, 0.6),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = get_image_data(data)
        height, width = (metadata["height"], metadata["width"])

        total_amount = self.py_random.uniform(*self.amount)
        salt_ratio = self.py_random.uniform(*self.salt_vs_pepper)

        self.applied_config = {"amount": total_amount, "salt_vs_pepper": salt_ratio}

        area = height * width

        num_pixels = int(area * total_amount)
        num_salt = int(num_pixels * salt_ratio)

        # Generate all positions at once
        noise_positions = self.random_generator.choice(area, size=num_pixels, replace=False)

        # Create masks
        salt_mask = np.zeros(area, dtype=bool)
        pepper_mask = np.zeros(area, dtype=bool)

        # Set salt and pepper positions
        salt_mask[noise_positions[:num_salt]] = True
        pepper_mask[noise_positions[num_salt:]] = True

        # Reshape to 2D
        salt_mask = salt_mask.reshape(height, width)
        pepper_mask = pepper_mask.reshape(height, width)

        return {
            "salt_mask": salt_mask,
            "pepper_mask": pepper_mask,
        }

    def apply(
        self,
        img: ImageType,
        salt_mask: np.ndarray,
        pepper_mask: np.ndarray,
        **params: Any,
    ) -> ImageType:
        return fpixel.apply_salt_and_pepper(img, salt_mask, pepper_mask)

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self.apply(images, **params)

    def apply_to_volumes(self, volumes: VolumeType, **params: Any) -> VolumeType:
        return self.apply(volumes, **params)


class FilmGrain(ImageOnlyTransform):
    """Analog film grain: luminance-dependent, spatially correlated noise. Distinct from
    i.i.d. GaussNoise or ShotNoise. Use for vintage or film-like augmentation.

    Unlike GaussNoise or ShotNoise, film grain is:
    - Luminance-dependent: darker areas show more visible grain
    - Spatially correlated: grain is clumped, not i.i.d. per-pixel
    - Optionally chromatic: separate grain patterns per channel

    Args:
        intensity_range (tuple[float, float]): Range for grain intensity. Higher values
            give more prominent grain. Default: (0.1, 0.3).
        grain_size_range (tuple[int, int]): Grain resolution as divisor of image size.
            1 = full resolution (fine); larger = coarser, more clumped. Default: (1, 3).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - Grain is generated at lower resolution and upscaled → spatial correlation
          (clumping) like real film.
        - Visibility modulated by inverse luminance; darker regions show more grain
          (silver halide-like behavior).

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>>
        >>> transform = A.FilmGrain(intensity_range=(0.1, 0.3), grain_size_range=(1, 3), p=1.0)
        >>> result = transform(image=image)["image"]

    See Also:
        - GaussNoise: i.i.d. Gaussian noise; use for sensor or transmission noise.
        - ShotNoise: Poisson (photon) noise in linear space; use for low-light sensor noise.

    """

    class InitSchema(BaseTransformInitSchema):
        intensity_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        grain_size_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        intensity_range: tuple[float, float] = (0.1, 0.3),
        grain_size_range: tuple[int, int] = (1, 3),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.intensity_range = intensity_range
        self.grain_size_range = grain_size_range

    def apply(
        self,
        img: ImageType,
        grain: np.ndarray,
        intensity: float,
        **params: Any,
    ) -> ImageType:
        return fpixel.apply_film_grain(img, grain, intensity)

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        result = np.empty_like(images)
        for i, image in enumerate(images):
            result[i] = self.apply(image, **params)
        return result

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        image_shape = params["shape"][:2]
        height, width = image_shape

        intensity = self.py_random.uniform(*self.intensity_range)
        grain_size = (
            self.py_random.randint(*self.grain_size_range)
            if self.grain_size_range[0] != self.grain_size_range[1]
            else self.grain_size_range[0]
        )

        grain_h = max(1, height // grain_size)
        grain_w = max(1, width // grain_size)

        grain = self.random_generator.standard_normal((grain_h, grain_w, 1), dtype=np.float32)

        if grain_h != height or grain_w != width:
            grain = fgeometric.resize(grain, (height, width), interpolation=cv2.INTER_LINEAR)

        grain = grain[:, :, 0]

        self.applied_config = {"intensity_range": intensity, "grain_size_range": grain_size}
        return {
            "grain": grain,
            "intensity": intensity,
        }
