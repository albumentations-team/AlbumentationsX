"""Noise injection transforms.

Transforms that add various types of noise to images, including Gaussian,
ISO, multiplicative, Rician, shot, salt-and-pepper, additive, and film grain noise.
"""

from collections.abc import Sequence
from typing import Annotated, Any, ClassVar, Literal, TypeAlias, cast

import cv2
import numpy as np
from albucore import clip, multiply, resize3d
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.functional_validators import AfterValidator
from typing_extensions import Self

import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.augmentations.pixel import functional as fpixel
from albumentations.augmentations.utils import non_rgb_error
from albumentations.core.invocation import SamplingContext
from albumentations.core.pydantic import (
    check_range_bounds,
    nondecreasing,
)
from albumentations.core.transform_params import (
    SampledParams,
    TargetParams,
    TargetSet,
    TargetView,
    requirements_for_views,
)
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    ImageOnlyTransform,
)
from albumentations.core.type_definitions import (
    CV2_BORDER_REFLECT_101,
    PAIR,
    ImageType,
)

__all__ = [
    "AdditiveNoise",
    "FilmGrain",
    "GaussNoise",
    "ISONoise",
    "MultiplicativeNoise",
    "RicianNoise",
    "SaltAndPepper",
    "ShotNoise",
    "StochasticConvolution",
]


def _target_noise_map_shape(view: Any) -> tuple[int, ...]:
    descriptor = view.descriptor
    if descriptor.canonical_type == "image":
        return (*descriptor.spatial_shape, descriptor.channels or 1)
    if descriptor.canonical_type == "images":
        return (*descriptor.spatial_shape, descriptor.channels or 1)
    if descriptor.canonical_type == "volume":
        if descriptor.shape is None:
            raise ValueError(f"Volume target {view.name!r} has no shape")
        return descriptor.shape
    raise ValueError(f"Noise transforms do not support target {view.name!r}")


def _target_additive_shape(transform: Any, view: Any) -> tuple[int, ...]:
    shape = _target_noise_map_shape(view)
    if view.canonical_type == "volume" and transform._volume_sampling_is_slice_wise:  # noqa: SLF001
        return (shape[1], shape[2], shape[3])
    return shape


def _sampling_family(view: Any, *, volume_is_3d: bool = True) -> str:
    if view.canonical_type == "volume" and volume_is_3d:
        return "volume_3d"
    return "image_2d"


def _target_value_scale(view: TargetView) -> float:
    value_scale = view.descriptor.value_scale
    if value_scale is None:
        raise ValueError(f"Noise target {view.name!r} has an unsupported dtype")
    return value_scale


def _target_parameter_group(
    views: tuple[Any, ...],
    parameter_name: str,
    value: Any,
    *,
    shape: bool = False,
    spatial_shape: bool = False,
    channels: bool = False,
    dtype: bool = False,
    topology: bool = False,
) -> TargetParams:
    return TargetParams(
        targets=tuple(view.name for view in views),
        params={parameter_name: value},
        requirements=requirements_for_views(
            views,
            shape=shape,
            spatial_shape=spatial_shape,
            channels=channels,
            dtype=dtype,
            sampling_topology=topology,
        ),
    )


class _FullVolumeNoiseTransform(ImageOnlyTransform):
    _volume_sampling_is_slice_wise: ClassVar[bool] = False


class StochasticConvolution(_FullVolumeNoiseTransform):
    """Apply a stochastic identity-centered convolution kernel with configurable spectral strength and channel sharing
    for images and volumes.

    The kernel is a discrete impulse plus a zero-mean Gaussian field. `kernel_range` controls the
    odd side length `K` (the spectral resolution in PRIME), while `strength_range` controls the
    perturbation energy. The random field is scaled by `strength / K` so the expected perturbation
    energy remains comparable across kernel sizes.

    Args:
        kernel_range (tuple[int, int]): Inclusive odd range for the square kernel side length. Values must be
            greater than or equal to 3. Default: (3, 7).
        strength_range (tuple[float, float]): Non-negative range for the random field strength. Zero is an exact
            identity. Default: (0.0, 1.0).
        per_channel (bool): If True, sample an independent kernel for each channel. If False, share one kernel
            across all channels. Default: False.
        border_mode (BorderModeType): OpenCV border policy. Supported values are constant, replicate, reflect,
            and reflect-101; wrap is rejected because it is not supported by the convolution backend. Default:
            `cv2.BORDER_REFLECT_101`.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Notes:
        - The random weights are not normalized or mean-subtracted. They may be signed, and the realized DC gain
          is the sampled kernel sum (with expected gain 1).
        - `cv2.BORDER_CONSTANT` uses zero padding, matching the PRIME reference implementation. The default
          reflect-101 border is the project-wide image-friendly choice.
        - One kernel realization is sampled per transform invocation and reused for every image in a batch and every
          depth slice in a volume.
        - Applied configuration records the sampled scalar values for the range fields and remains runnable after JSON
          transport. The in-memory replay path retains the realized kernel through transform parameters.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>> image = np.random.default_rng(137).random((128, 128, 3), dtype=np.float32)
        >>> transform = A.Compose(
        ...     [
        ...         A.StochasticConvolution(
        ...             kernel_range=(3, 7),
        ...             strength_range=(0.05, 0.25),
        ...             border_mode=cv2.BORDER_REFLECT_101,
        ...             p=1.0,
        ...         ),
        ...     ],
        ...     seed=137,
        ... )
        >>> transformed = transform(image=image)["image"]

        Use `per_channel=True` for independent spectral perturbations, or `strength_range=(0.0, 0.0)` for an
        exact identity while keeping the transform in a pipeline.

    References:
        - PRIME issue: https://github.com/albumentations-team/AlbumentationsX/issues/330
        - PRIME randomized-filter construction: https://github.com/amodas/PRIME-augmentations/blob/main/utils/rand_filter.py

    """

    class InitSchema(BaseTransformInitSchema):
        kernel_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(3)),
            AfterValidator(nondecreasing),
        ]
        strength_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0)),
            AfterValidator(nondecreasing),
        ]
        per_channel: bool
        border_mode: Literal[0, 1, 2, 4]

        @field_validator("kernel_range")
        @classmethod
        def _check_odd_kernel_range(cls, value: tuple[int, int]) -> tuple[int, int]:
            if any(size % 2 == 0 for size in value):
                raise ValueError(f"kernel_range values must be odd, got {value}")
            return value

    def __init__(
        self,
        *,
        kernel_range: tuple[int, int] = (3, 7),
        strength_range: tuple[float, float] = (0.0, 1.0),
        per_channel: bool = False,
        border_mode: Literal[0, 1, 2, 4] = CV2_BORDER_REFLECT_101,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.kernel_range = kernel_range
        self.strength_range = strength_range
        self.per_channel = per_channel
        self.border_mode = border_mode

    def apply(self, img: ImageType, kernel: np.ndarray, **params: Any) -> ImageType:
        return fpixel.convolve(img, kernel=kernel, border_mode=self.border_mode)

    @staticmethod
    def _sample_kernel(
        kernel_shape: tuple[int, ...],
        strength: float,
        sampling: SamplingContext,
    ) -> np.ndarray:
        random_field = (
            np.zeros(kernel_shape, dtype=np.float32)
            if strength == 0
            else sampling.random_generator.standard_normal(kernel_shape, dtype=np.float32)
        )
        return fpixel.create_stochastic_convolution_kernel(random_field, strength)

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        del params, data
        kernel_size = sampling.py_random.randrange(self.kernel_range[0], self.kernel_range[1] + 1, 2)
        strength = sampling.py_random.uniform(*self.strength_range)
        sampling.applied_overrides.update({"kernel_range": kernel_size, "strength_range": strength})

        if not self.per_channel:
            kernel = self._sample_kernel((kernel_size, kernel_size), strength, sampling)
            return SampledParams(params={"kernel": kernel})

        groups: list[TargetParams] = []
        for views in targets.group_image_like_by(lambda view: view.descriptor.channels):
            channel_count = views[0].descriptor.channels
            if channel_count is None:
                raise ValueError("StochasticConvolution requires image-like targets with a known channel count")
            kernel = self._sample_kernel((channel_count, kernel_size, kernel_size), strength, sampling)
            groups.append(_target_parameter_group(views, "kernel", kernel, channels=True))
        return SampledParams(params={}, target_params=tuple(groups))


class GaussNoise(_FullVolumeNoiseTransform):
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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>>
        >>> transform = A.GaussNoise(std_range=(0.1, 0.2), p=1.0)
        >>> noisy_image = transform(image=image)["image"]

    See Also:
        - FilmGrain: Luminance-dependent, spatially correlated (film-like) noise.
        - RicianNoise: MRI magnitude-reconstruction noise with a low-signal floor.
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

    def __init__(
        self,
        std_range: tuple[float, float] = (0.2, 0.44),  # sqrt(10 / 255), sqrt(50 / 255)
        mean_range: tuple[float, float] = (0.0, 0.0),
        per_channel: bool = False,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.std_range = std_range
        self.mean_range = mean_range
        self.per_channel = per_channel

    def apply(
        self,
        img: ImageType,
        noise_map: np.ndarray,
        **params: Any,
    ) -> ImageType:
        return fpixel.add_noise(img, noise_map)

    def apply_to_images(self, images: ImageType, noise_map: np.ndarray, **params: Any) -> ImageType:
        return fpixel.add_noise(images, noise_map)

    def apply_to_volume(
        self,
        volume: ImageType,
        noise_map: np.ndarray,
        **params: Any,
    ) -> ImageType:
        return fpixel.add_noise(volume, noise_map)

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        sigma = sampling.py_random.uniform(*self.std_range)
        mean = sampling.py_random.uniform(*self.mean_range)
        sampling.applied_overrides.update({"std_range": sigma, "mean_range": mean})
        spatial_mode: Literal["per_pixel", "shared"] = "per_pixel" if self.per_channel else "shared"
        noise_params = {"mean_range": (mean, mean), "std_range": (sigma, sigma)}
        groups: list[TargetParams] = []
        for views in targets.group_image_like_by(
            lambda view: (
                _target_noise_map_shape(view),
                view.descriptor.value_scale,
                _sampling_family(view),
            ),
        ):
            view = views[0]
            noise_map = fpixel.generate_spatial_noise(
                noise_type="gaussian",
                spatial_mode=spatial_mode,
                shape=_target_noise_map_shape(view),
                params=noise_params,
                max_value=_target_value_scale(view),
                random_generator=sampling.random_generator,
            )
            groups.append(
                _target_parameter_group(
                    views,
                    "noise_map",
                    noise_map,
                    shape=True,
                    dtype=True,
                    topology=True,
                ),
            )
        return SampledParams(params={}, target_params=tuple(groups))


class ISONoise(_FullVolumeNoiseTransform):
    """Add camera-sensor-like noise scaling with intensity (high ISO), useful for low-light or
    camera noise simulation. See `color_shift_range` and `intensity_range`.

    This transform adds random noise to an image, mimicking the effect of using high ISO settings
    in digital photography. It simulates two main components of ISO noise:
    1. Color noise: random shifts in color hue
    2. Luminance noise: random variations in pixel intensity

    Args:
        color_shift_range (tuple[float, float]): Range for changing color hue.
            Values should be in the range [0, 1], where 1 represents a full 360° hue rotation.
            Default: (0.01, 0.05)

        intensity_range (tuple[float, float]): Range for the noise intensity.
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
        >>> transform = A.ISONoise(color_shift_range=(0.01, 0.05), intensity_range=(0.1, 0.5), p=0.5)
        >>> result = transform(image=image)
        >>> noisy_image = result["image"]

    References:
        ISO noise in digital photography: https://en.wikipedia.org/wiki/Image_noise#In_digital_cameras

    """

    class InitSchema(BaseTransformInitSchema):
        color_shift_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        intensity_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        color_shift_range: tuple[float, float] = (0.01, 0.05),
        intensity_range: tuple[float, float] = (0.1, 0.5),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.intensity_range = intensity_range
        self.color_shift_range = color_shift_range

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

    def apply_to_volume(
        self,
        volume: ImageType,
        color_shift: float,
        intensity: float,
        random_seed: int,
        **params: Any,
    ) -> ImageType:
        return fpixel.iso_noise_volume(
            volume,
            color_shift,
            intensity,
            np.random.default_rng(random_seed),
        )

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        random_seed = sampling.random_generator.integers(0, 2**32 - 1)
        color_shift = sampling.py_random.uniform(*self.color_shift_range)
        intensity = sampling.py_random.uniform(*self.intensity_range)

        sampling.applied_overrides.update({"color_shift_range": color_shift, "intensity_range": intensity})

        return SampledParams(
            params={
                "color_shift": color_shift,
                "intensity": intensity,
                "random_seed": random_seed,
            }
        )


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

    @property
    def _volume_sampling_is_slice_wise(self) -> bool:
        return not self.elementwise

    def apply(
        self,
        img: ImageType,
        multiplier: float | np.ndarray,
        **kwargs: Any,
    ) -> ImageType:
        result = multiply(img, multiplier)
        return clip(result, img.dtype, inplace=True) if img.dtype == np.float32 else result

    def apply_to_images(self, images: ImageType, multiplier: float | np.ndarray, **kwargs: Any) -> ImageType:
        return self.apply(images, multiplier, **kwargs)

    def apply_to_volume(
        self,
        volume: ImageType,
        multiplier: float | np.ndarray,
        **kwargs: Any,
    ) -> ImageType:
        return self.apply(volume, multiplier, **kwargs)

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        if not self.elementwise and not self.per_channel:
            multiplier = sampling.random_generator.uniform(self.multiplier[0], self.multiplier[1])
            return SampledParams(params={"multiplier": multiplier})

        groups: list[TargetParams] = []
        if not self.elementwise:
            for compatible_views in targets.group_image_like_by(lambda view: view.descriptor.channels):
                channels = compatible_views[0].descriptor.channels or 1
                groups.append(
                    _target_parameter_group(
                        compatible_views,
                        "multiplier",
                        sampling.random_generator.uniform(
                            self.multiplier[0],
                            self.multiplier[1],
                            size=(channels,),
                        ).astype(np.float32),
                        channels=True,
                    ),
                )
            return SampledParams(params={}, target_params=tuple(groups))

        for compatible_views in targets.group_image_like_by(
            lambda view: (self._multiplier_shape(_target_noise_map_shape(view)), _sampling_family(view)),
        ):
            view = compatible_views[0]
            groups.append(
                _target_parameter_group(
                    compatible_views,
                    "multiplier",
                    self._sample_multiplier(_target_noise_map_shape(view), sampling),
                    spatial_shape=True,
                    channels=self.per_channel,
                    topology=True,
                ),
            )
        return SampledParams(params={}, target_params=tuple(groups))

    def _sample_multiplier(
        self,
        target_shape: tuple[int, ...],
        sampling: SamplingContext,
    ) -> np.ndarray:
        """Generates the multiplier for the current layout without replay-policy overrides, keeping pixel execution
        independent from constructor serialization concerns.
        """
        return sampling.random_generator.uniform(
            self.multiplier[0],
            self.multiplier[1],
            self._multiplier_shape(target_shape),
        ).astype(np.float32)

    def _multiplier_shape(self, target_shape: tuple[int, ...]) -> tuple[int, ...]:
        return target_shape if self.per_channel else (*target_shape[:-1], 1)


class ShotNoise(_FullVolumeNoiseTransform):
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
        - RicianNoise: MRI magnitude-reconstruction noise with a low-signal floor.

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

    def apply_to_volume(
        self,
        volume: ImageType,
        scale: float,
        random_seed: int,
        **params: Any,
    ) -> ImageType:
        return fpixel.shot_noise(volume, scale, np.random.default_rng(random_seed))

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        scale = sampling.py_random.uniform(*self.scale_range)
        sampling.applied_overrides["scale_range"] = scale
        return SampledParams(
            params={
                "scale": scale,
                "random_seed": sampling.random_generator.integers(0, 2**32 - 1),
            }
        )


class NoiseParamsBase(BaseModel):
    """Base Pydantic model for AdditiveNoise noise params (uniform, gaussian, laplace, beta).
    Subclasses define noise_type and distribution-specific fields.
    """

    model_config = ConfigDict(extra="forbid")


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
            result.append((min_val, max_val))
        return result  # pyrefly: ignore[bad-return]


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


class _AdditiveNoiseInitSchema(BaseTransformInitSchema):
    noise_type: Literal["uniform", "gaussian", "laplace", "beta"]
    spatial_mode: Literal["constant", "per_pixel", "shared", "patch"]
    noise_params: dict[str, Any] | None
    patch_count_range: Annotated[
        tuple[int, int],
        AfterValidator(check_range_bounds(1, None)),
        AfterValidator(nondecreasing),
    ]
    patch_height_range: Annotated[
        tuple[float, float],
        AfterValidator(check_range_bounds(0, 1, min_inclusive=False)),
        AfterValidator(nondecreasing),
    ]
    patch_width_range: Annotated[
        tuple[float, float],
        AfterValidator(check_range_bounds(0, 1, min_inclusive=False)),
        AfterValidator(nondecreasing),
    ]
    per_channel: bool

    @model_validator(mode="after")
    def _validate_noise_params(self) -> Self:
        # Default parameters for each noise type
        default_params: dict[str, dict[str, Any]] = {
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
        params_dict: dict[str, Any] = (
            self.noise_params if self.noise_params is not None else default_params[self.noise_type]
        )

        # Add noise_type to params if not present
        params_dict = {**params_dict, "noise_type": self.noise_type}

        # Convert dict to appropriate NoiseParams object and validate
        params_class: Any = {
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

    @model_validator(mode="after")
    def _validate_patch_options(self) -> Self:
        if self.spatial_mode != "patch" and (
            self.patch_count_range != (1, 1)
            or self.patch_height_range != (0.1, 1.0)
            or self.patch_width_range != (0.1, 1.0)
            or self.per_channel
        ):
            raise ValueError("Patch options can only be used when spatial_mode='patch'")
        return self


class AdditiveNoise(ImageOnlyTransform):
    """Add uniform, Gaussian, Laplace, or beta-distributed noise in constant, per-pixel, channel-shared, or randomly
    localized rectangular patch modes.

    Noise can be constant per channel, independent per pixel and channel, shared across channels, or localized inside
    one or more randomly sampled rectangular patches. Patch-localized noise is useful when spatially restricted
    corruption should improve robustness without perturbing the complete image.

    Args:
        noise_type (Literal['uniform', 'gaussian', 'laplace', 'beta']): Noise distribution. Default: "uniform".
        spatial_mode (Literal['constant', 'per_pixel', 'shared', 'patch']): Spatial sampling mode. Default: "constant".
            - `"constant"` samples one value per channel.
            - `"per_pixel"` samples each pixel and channel independently.
            - `"shared"` samples one spatial map and shares it across channels.
            - `"patch"` samples noise only inside random rectangular patches.
        noise_params (dict[str, Any] | None): Parameters for the chosen noise distribution.
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
        p (float): Probability of applying the transform. Default: 0.5.
        patch_count_range (tuple[int, int]): Inclusive range for the number of patches when
            `spatial_mode="patch"`. Default: (1, 1).
        patch_height_range (tuple[float, float]): Patch height as a fraction of image height. Values must be in
            `(0, 1]`. Default: (0.1, 1.0).
        patch_width_range (tuple[float, float]): Patch width as a fraction of image width. Values must be in
            `(0, 1]`. Default: (0.1, 1.0).
        per_channel (bool): When `spatial_mode="patch"`, whether to sample independent noise for every channel.
            If False, the same noise is shared across channels. Default: False.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Targets:
        image, volume

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.Compose(
        ...     [
        ...         A.AdditiveNoise(
        ...             noise_type="gaussian",
        ...             spatial_mode="patch",
        ...             noise_params={"mean_range": (0.0, 0.0), "std_range": (0.05, 0.15)},
        ...             patch_count_range=(1, 3),
        ...             patch_height_range=(0.1, 0.4),
        ...             patch_width_range=(0.1, 0.4),
        ...             p=1.0,
        ...         ),
        ...     ],
        ...     seed=137,
        ... )
        >>> noisy_image = transform(image=image)["image"]

    Note:
        - Patch positions and sizes are shared across channels. `per_channel` controls only the sampled noise values.
        - Overlapping patches are processed in order, and later patch noise replaces earlier noise in the overlap.
        - Image batches and volume slices receive the same sampled patch program, matching the existing batch behavior.
        - All noise is generated in normalized units and scaled by the image dtype maximum.

    References:
        Patch Gaussian: Improving Generalization of Convolutional Neural Networks without Encouraging Invariance:
          https://openreview.net/forum?id=HkxWXkStDB

    """

    InitSchema: ClassVar[type[BaseTransformInitSchema]] = _AdditiveNoiseInitSchema

    def __init__(
        self,
        noise_type: Literal["uniform", "gaussian", "laplace", "beta"] = "uniform",
        spatial_mode: Literal["constant", "per_pixel", "shared", "patch"] = "constant",
        noise_params: dict[str, Any] | None = None,
        p: float = 0.5,
        *,
        patch_count_range: tuple[int, int] = (1, 1),
        patch_height_range: tuple[float, float] = (0.1, 1.0),
        patch_width_range: tuple[float, float] = (0.1, 1.0),
        per_channel: bool = False,
    ):
        super().__init__(p=p)
        self.noise_type = noise_type
        self.spatial_mode = spatial_mode
        self.noise_params = noise_params
        self.patch_count_range = patch_count_range
        self.patch_height_range = patch_height_range
        self.patch_width_range = patch_width_range
        self.per_channel = per_channel

    @property
    def _volume_sampling_is_slice_wise(self) -> bool:
        return self.spatial_mode in {"constant", "patch"}

    def apply(
        self,
        img: ImageType,
        noise_map: np.ndarray,
        patches: np.ndarray | None,
        **params: Any,
    ) -> ImageType:
        if patches is not None:
            return fpixel.add_noise_by_patches(img, noise_map, patches)
        return fpixel.add_noise(img, noise_map)

    def apply_to_images(self, images: ImageType, **params: Any) -> ImageType:
        return self.apply(images, **params)

    def apply_to_volume(
        self,
        volume: ImageType,
        noise_map: np.ndarray,
        **params: Any,
    ) -> ImageType:
        return self.apply(volume, noise_map, **params)

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
                _target_additive_shape(self, view),
                view.descriptor.value_scale,
                _sampling_family(view, volume_is_3d=not self._volume_sampling_is_slice_wise),
            ),
        ):
            view = views[0]
            sampled = self._sample_noise_map(
                _target_additive_shape(self, view),
                _target_value_scale(view),
                sampling,
            )
            groups.append(
                TargetParams(
                    targets=tuple(item.name for item in views),
                    params=sampled,
                    requirements=requirements_for_views(
                        views,
                        shape=True,
                        dtype=True,
                        sampling_topology=True,
                    ),
                ),
            )
        return SampledParams(params={}, target_params=tuple(groups))

    def _sample_noise_map(
        self,
        shape: tuple[int, ...],
        max_value: float,
        sampling: SamplingContext,
    ) -> dict[str, Any]:
        """Generates the noise map for the current layout without replay-policy overrides, keeping pixel execution
        independent from constructor serialization concerns.
        """
        num_channels = shape[-1]
        noise_params = cast("dict[str, Any]", self.noise_params)

        if self.noise_type == "uniform":
            ranges = noise_params["ranges"]
            range_count = len(ranges)
            if range_count > 1:
                uses_channel_ranges = self.spatial_mode in {"constant", "per_pixel"} or (
                    self.spatial_mode == "patch" and self.per_channel
                )
                if uses_channel_ranges and range_count < num_channels:
                    raise ValueError(
                        f"Not enough ranges provided. Expected 1 or at least {num_channels}, got {range_count}",
                    )
                if not uses_channel_ranges or range_count != num_channels:
                    resolved_count = num_channels if uses_channel_ranges else 1
                    noise_params = {**noise_params, "ranges": ranges[:resolved_count]}

        if self.spatial_mode == "constant":
            noise_map = fpixel.generate_constant_noise_with_py_random(
                noise_type=self.noise_type,
                shape=shape,
                params=noise_params,
                max_value=max_value,
                py_random=sampling.py_random,
            )
            return {"noise_map": noise_map, "patches": None}

        if self.spatial_mode == "patch":
            patch_shape = cast("tuple[int, int, int]", shape)
            patch_count = sampling.py_random.randint(*self.patch_count_range)
            patch_heights = np.ceil(
                shape[0] * sampling.random_generator.uniform(*self.patch_height_range, size=patch_count),
            ).astype(np.int32)
            patch_widths = np.ceil(
                shape[1] * sampling.random_generator.uniform(*self.patch_width_range, size=patch_count),
            ).astype(np.int32)
            y_min = sampling.random_generator.integers(0, shape[0] - patch_heights + 1)
            x_min = sampling.random_generator.integers(0, shape[1] - patch_widths + 1)
            patches = np.stack([x_min, y_min, x_min + patch_widths, y_min + patch_heights], axis=-1)
            noise_map = fpixel.generate_patch_noise(
                noise_type=self.noise_type,
                shape=patch_shape,
                params=noise_params,
                max_value=max_value,
                random_generator=sampling.random_generator,
                patches=patches,
                per_channel=self.per_channel,
            )
            return {"noise_map": noise_map, "patches": patches}

        noise_map = fpixel.generate_spatial_noise(
            noise_type=self.noise_type,
            spatial_mode=self.spatial_mode,
            shape=shape,
            params=noise_params,
            max_value=max_value,
            random_generator=sampling.random_generator,
        )
        return {"noise_map": noise_map, "patches": None}


class SaltAndPepper(_FullVolumeNoiseTransform):
    """Apply salt-and-pepper (impulse) noise: randomly set pixels to min or max with
    density and ratio controlled by `amount_range` and `salt_vs_pepper_range`.

    Salt and pepper noise is a form of impulse noise that randomly sets pixels to either maximum value (salt)
    or minimum value (pepper). The amount and proportion of salt vs pepper can be controlled.
    The same noise mask is applied to all channels of the image to preserve color consistency.

    Args:
        amount_range ((float, float)): Range for total amount of noise (both salt and pepper).
            Values between 0 and 1. For example:
            - 0.05 means 5% of all pixels will be replaced with noise
            - (0.01, 0.06) will sample amount uniformly from 1% to 6%
            Default: (0.01, 0.06)

        salt_vs_pepper_range ((float, float)): Range for ratio of salt (white) vs pepper (black) noise.
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
        - amount ∈ amount_range
        - salt_ratio ∈ salt_vs_pepper_range

    Examples:
        >>> import albumentations as A
        >>> import numpy as np

        # Apply salt and pepper noise with default parameters
        >>> transform = A.SaltAndPepper(p=1.0)
        >>> noisy_image = transform(image=image)["image"]

        # Heavy noise with more salt than pepper
        >>> transform = A.SaltAndPepper(
        ...     amount_range=(0.1, 0.2),         # 10-20% of pixels will be noisy
        ...     salt_vs_pepper_range=(0.7, 0.9), # 70-90% of noise will be salt
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
        amount_range: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, 1))]
        salt_vs_pepper_range: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, 1))]

    def __init__(
        self,
        amount_range: tuple[float, float] = (0.01, 0.06),
        salt_vs_pepper_range: tuple[float, float] = (0.4, 0.6),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.amount_range = amount_range
        self.salt_vs_pepper_range = salt_vs_pepper_range

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        total_amount = sampling.py_random.uniform(*self.amount_range)
        salt_ratio = sampling.py_random.uniform(*self.salt_vs_pepper_range)
        sampling.applied_overrides.update({"amount_range": total_amount, "salt_vs_pepper_range": salt_ratio})
        groups: list[TargetParams] = []
        for views in targets.group_image_like_by(
            lambda view: (
                tuple((view.descriptor.shape or ())[:3])
                if view.canonical_type == "volume"
                else tuple(view.descriptor.spatial_shape or ()),
                _sampling_family(view),
            ),
        ):
            view = views[0]
            spatial_shape = (
                tuple(view.descriptor.shape[:-1])
                if view.canonical_type == "volume" and view.descriptor.shape is not None
                else tuple(view.descriptor.spatial_shape or ())
            )
            salt_mask, pepper_mask = self._sample_masks(spatial_shape, total_amount, salt_ratio, sampling)
            groups.append(
                TargetParams(
                    targets=tuple(item.name for item in views),
                    params={"salt_mask": salt_mask, "pepper_mask": pepper_mask},
                    requirements=requirements_for_views(views, shape=True, sampling_topology=True),
                ),
            )
        return SampledParams(params={}, target_params=tuple(groups))

    @staticmethod
    def _sample_masks(
        spatial_shape: tuple[int, ...],
        total_amount: float,
        salt_ratio: float,
        sampling: SamplingContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        area = int(np.prod(spatial_shape))
        num_pixels = int(area * total_amount)
        num_salt = int(num_pixels * salt_ratio)
        noise_positions = sampling.random_generator.choice(area, size=num_pixels, replace=False)
        salt_mask: np.ndarray = np.zeros(area, dtype=bool)
        pepper_mask: np.ndarray = np.zeros(area, dtype=bool)
        salt_mask[noise_positions[:num_salt]] = True
        pepper_mask[noise_positions[num_salt:]] = True
        return salt_mask.reshape(spatial_shape), pepper_mask.reshape(spatial_shape)

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

    def apply_to_volume(
        self,
        volume: ImageType,
        salt_mask: np.ndarray,
        pepper_mask: np.ndarray,
        **params: Any,
    ) -> ImageType:
        return self.apply(volume, salt_mask, pepper_mask, **params)


class FilmGrain(_FullVolumeNoiseTransform):
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
        return self._apply_to_batch_same_shape(images, lambda image: self.apply(image, **params))

    def apply_to_volume(
        self,
        volume: ImageType,
        grain: np.ndarray,
        intensity: float,
        **params: Any,
    ) -> ImageType:
        return self.apply(volume, grain, intensity, **params)

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        intensity = sampling.py_random.uniform(*self.intensity_range)
        grain_size = (
            sampling.py_random.randint(*self.grain_size_range)
            if self.grain_size_range[0] != self.grain_size_range[1]
            else self.grain_size_range[0]
        )
        sampling.applied_overrides.update({"intensity_range": intensity, "grain_size_range": grain_size})
        groups: list[TargetParams] = []
        for views in targets.group_image_like_by(
            lambda view: (
                tuple(view.descriptor.spatial_shape or ()),
                _sampling_family(view),
            ),
        ):
            view = views[0]
            spatial_shape = tuple(view.descriptor.spatial_shape or ())
            groups.append(
                TargetParams(
                    targets=tuple(item.name for item in views),
                    params={"grain": self._sample_grain(spatial_shape, grain_size, sampling)},
                    requirements=requirements_for_views(views, shape=True, sampling_topology=True),
                ),
            )
        return SampledParams(params={"intensity": intensity}, target_params=tuple(groups))

    @staticmethod
    def _sample_grain(
        spatial_shape: tuple[int, ...],
        grain_size: int,
        sampling: SamplingContext,
    ) -> np.ndarray:
        grain_shape = tuple(max(1, size // grain_size) for size in spatial_shape)
        grain = sampling.random_generator.standard_normal((*grain_shape, 1), dtype=np.float32)
        if grain_shape == spatial_shape:
            return grain[..., 0]
        if len(spatial_shape) == 3:
            return resize3d(grain, spatial_shape, cv2.INTER_LINEAR)[..., 0]
        spatial_shape_2d = (spatial_shape[0], spatial_shape[1])
        return fgeometric.resize(grain, spatial_shape_2d, interpolation=cv2.INTER_LINEAR)[:, :, 0]


class RicianNoise(_FullVolumeNoiseTransform):
    """Simulate MRI magnitude reconstruction with Gaussian real and imaginary components, yielding Rician noise
    and a positive low-signal noise floor.

    The transform computes sqrt((signal + n_real)^2 + n_imag^2). Unlike additive Gaussian noise, this model
    remains biased upward at low signal-to-noise ratios, matching magnitude MRI reconstruction.

    Args:
        std_range (tuple[float, float]): Nondecreasing range in [0, 1] for the Gaussian component standard deviation
            as a fraction of the dtype range. Default: (0.05, 0.15).
        per_channel (bool): If True, sample independent real and imaginary fields for each channel. If False,
            share one pair of fields across channels. Default: False.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - Volumes receive one independently sampled full-depth field rather than a slice-wise image batch.
        - A sampled standard deviation of zero is an exact identity.

    Examples:
        >>> import albumentations as A
        >>> import numpy as np
        >>> image = np.random.default_rng(137).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.RicianNoise(std_range=(0.05, 0.15), p=1.0)
        >>> noisy_image = transform(image=image)["image"]

    References:
        Gudbjartsson & Patz (1995): https://doi.org/10.1002/mrm.1910340618

    See Also:
        - GaussNoise: Additive Gaussian noise for sensor or transmission robustness.
        - ShotNoise: Poisson noise in linear space for photon-limited acquisition.

    """

    class InitSchema(BaseTransformInitSchema):
        std_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        per_channel: bool

    def __init__(
        self,
        std_range: tuple[float, float] = (0.05, 0.15),
        per_channel: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.std_range = std_range
        self.per_channel = per_channel

    def apply(
        self,
        img: ImageType,
        std: float,
        real_noise: np.ndarray | None,
        imaginary_noise: np.ndarray | None,
        **params: Any,
    ) -> ImageType:
        if std == 0:
            return img
        if real_noise is None or imaginary_noise is None:
            msg = "RicianNoise requires sampled real and imaginary noise fields."
            raise RuntimeError(msg)
        return fpixel.rician_noise(img, real_noise, imaginary_noise)

    def apply_to_images(
        self,
        images: ImageType,
        std: float,
        real_noise: np.ndarray | None,
        imaginary_noise: np.ndarray | None,
        **params: Any,
    ) -> ImageType:
        return self.apply(images, std, real_noise, imaginary_noise, **params)

    def apply_to_volume(
        self,
        volume: ImageType,
        std: float,
        real_noise: np.ndarray | None,
        imaginary_noise: np.ndarray | None,
        **params: Any,
    ) -> ImageType:
        return self.apply(volume, std, real_noise, imaginary_noise)

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        std = sampling.py_random.uniform(*self.std_range)
        sampling.applied_overrides["std_range"] = std
        if std == 0:
            return SampledParams(
                params={"std": std, "real_noise": None, "imaginary_noise": None},
            )

        groups: list[TargetParams] = []
        for views in targets.group_image_like_by(
            lambda view: (
                _target_noise_map_shape(view),
                self.per_channel,
                _sampling_family(view),
            ),
        ):
            view = views[0]
            noise_shape = _target_noise_map_shape(view)
            if not self.per_channel:
                noise_shape = (*noise_shape[:-1], 1)
            real_noise, imaginary_noise = self._sample_noise(noise_shape, std, sampling)
            groups.append(
                TargetParams(
                    targets=tuple(item.name for item in views),
                    params={"real_noise": real_noise, "imaginary_noise": imaginary_noise},
                    requirements=requirements_for_views(views, shape=True, sampling_topology=True),
                ),
            )
        return SampledParams(params={"std": std}, target_params=tuple(groups))

    @staticmethod
    def _sample_noise(
        shape: tuple[int, ...],
        std: float,
        sampling: SamplingContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        real_noise = sampling.random_generator.standard_normal(shape, dtype=np.float32)
        imaginary_noise = sampling.random_generator.standard_normal(shape, dtype=np.float32)
        np.multiply(real_noise, std, out=real_noise)
        np.multiply(imaginary_noise, std, out=imaginary_noise)
        return real_noise, imaginary_noise
