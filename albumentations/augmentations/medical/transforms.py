"""Medical acquisition artifacts and histology stain augmentation."""

from typing import Annotated, Any, Literal, Self, cast

import numpy as np
import torch
from pydantic import AfterValidator, field_validator, model_validator

from albumentations.augmentations.medical import functional as fmedical
from albumentations.augmentations.pixel.noise import (
    _FullVolumeNoiseTransform,
    _sampling_family,
    _target_noise_map_shape,
    _target_parameter_group,
)
from albumentations.augmentations.transforms3d import functional as f3d
from albumentations.augmentations.transforms3d.transforms import AXIS_NAMES_3D, AxisIndex3D, _sampling_volume_shape
from albumentations.augmentations.utils import non_rgb_error
from albumentations.core.invocation import SamplingContext
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transform_params import (
    SampledParams,
    TargetParams,
    TargetSet,
    TargetView,
    requirements_for_views,
)
from albumentations.core.transforms_interface import BaseTransformInitSchema, ImageOnlyTransform, VolumeOnlyTransform
from albumentations.core.type_definitions import CV2_INTER_LINEAR, ImageType, VolumeType

__all__ = [
    "Anisotropy3D",
    "BiasField",
    "GhostingArtifact",
    "GibbsRinging",
    "HEStain",
    "KSpaceSpikeNoise",
    "MotionArtifact",
    "RicianNoise",
]


class Anisotropy3D(VolumeOnlyTransform):
    """Simulate thicker or lower-resolution volume acquisition by degrading selected spatial axes and restoring the
    original grid for 3D model robustness training.

    The transform samples a subset from `axes` and one downsampling factor for every selected axis. Both NumPy and CPU
    Tensor routes delegate spatial resizing to Albucore `resize3d`. PyTorch does not currently provide antialiasing for
    5D trilinear interpolation, so Tensor input remains non-antialiased when `antialias=True`. `mask3d` remains
    unchanged because this is an image-acquisition artifact, not a geometry transform.

    Args:
        axes (tuple[int, ...]): Eligible spatial axes in `(depth, height, width)` order. Default: `(0, 1, 2)`.
        num_axes_range (tuple[int, int]): Inclusive range for the number of eligible axes to degrade.
            Default: `(1, 1)`.
        downscale_factor_range (tuple[float, float]): Inclusive range of downsampling factors, each greater than one.
            Default: `(1.5, 4.0)`.
        antialias (bool): Apply a low-pass filter while reducing spatial resolution. Default: `True`.
        p (float): Probability of applying the transform. Default: `0.5`.

    Targets:
        volume, volumes

    Image types:
        uint8, float32

    Examples:
        >>> import albumentations as A
        >>> import numpy as np
        >>> volume = np.random.default_rng(137).integers(0, 256, (32, 128, 128, 1), dtype=np.uint8)
        >>> transform = A.Compose([
        ...     A.Anisotropy3D(
        ...         axes=(0,),
        ...         num_axes_range=(1, 1),
        ...         downscale_factor_range=(2.0, 2.0),
        ...         p=1.0,
        ...     ),
        ... ])
        >>> result = transform(volume=volume)
        >>> result["volume"].shape
        (32, 128, 128, 1)

    """

    class InitSchema(BaseTransformInitSchema):
        axes: tuple[AxisIndex3D, ...]
        num_axes_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
            AfterValidator(nondecreasing),
        ]
        downscale_factor_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(1, None, min_inclusive=False)),
            AfterValidator(nondecreasing),
        ]
        antialias: bool

        @model_validator(mode="after")
        def _validate_axes(self) -> Self:
            if not self.axes:
                raise ValueError("axes must contain at least one spatial axis")
            if len(self.axes) != len(set(self.axes)):
                raise ValueError("axes must not contain duplicates")
            if self.num_axes_range[1] > len(self.axes):
                raise ValueError("num_axes_range cannot select more axes than are available in axes")
            return self

    def __init__(
        self,
        axes: tuple[AxisIndex3D, ...] = (0, 1, 2),
        num_axes_range: tuple[int, int] = (1, 1),
        downscale_factor_range: tuple[float, float] = (1.5, 4.0),
        antialias: bool = True,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.axes = axes
        self.num_axes_range = num_axes_range
        self.downscale_factor_range = downscale_factor_range
        self.antialias = antialias

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        selected_axis_count = sampling.py_random.randint(*self.num_axes_range)
        selected_axes = tuple(sorted(sampling.py_random.sample(self.axes, selected_axis_count)))
        downscale_factor = sampling.py_random.uniform(*self.downscale_factor_range)
        downsample_shape = fmedical.get_anisotropy_downsample_shape(
            _sampling_volume_shape(targets),
            selected_axes,
            downscale_factor,
        )

        sampling.applied_overrides.update(
            {
                "axes": selected_axes,
                "num_axes_range": (selected_axis_count, selected_axis_count),
                "downscale_factor_range": (downscale_factor, downscale_factor),
            },
        )
        return SampledParams(params={"downsample_shape": downsample_shape})

    def apply_to_volume(
        self,
        volume: VolumeType | torch.Tensor,
        downsample_shape: tuple[int, int, int],
        **params: Any,
    ) -> VolumeType:
        return cast("VolumeType", fmedical.anisotropy_3d(volume, downsample_shape, self.antialias))


class BiasField(_FullVolumeNoiseTransform):
    """Simulate smooth MRI intensity inhomogeneity by multiplying images and volumes by a positive field
    sampled on a coarse spatial grid.

    The transform samples zero-mean Gaussian log-gain coefficients, interpolates them across all spatial axes,
    and exponentiates the result. The field changes tissue intensity across space while preserving anatomy labels.

    Args:
        std_range (tuple[float, float]): Nondecreasing range in `[0, 1]` for the coarse log-gain standard deviation.
            Larger values produce stronger intensity variation. Zero gives exact identity. Default: `(0.0, 0.5)`.
        scale_range (tuple[float, float]): Nondecreasing range in `(0, 1]` for the ratio of coarse axis length
            to input axis length.
            Smaller values produce smoother fields; `1.0` uses an unsmoothed voxel-wise log-gain grid.
            Each non-singleton coarse axis has at least two points. Default: `(0.025, 0.025)`.
        per_channel (bool): If True, sample independent log-gain coefficients for each channel. If False,
            share one field across channels. Default: `False`.
        p (float): Probability of applying the transform. Default: `0.5`.

    Targets:
        image, images, volume, volumes

    Image types:
        uint8, float32

    Number of channels:
        Any

    Notes:
        - Images use bilinear interpolation over `(H, W)`; volumes use trilinear interpolation over `(D, H, W)`.
          Singleton axes remain singleton. Batch elements and aligned additional targets share sampled coefficients.
        - The coarse axis length is `min(size, max(2, int(size * scale)))`. The standard deviation describes
          the coarse coefficients; interpolation generally reduces the final log-field variance.
        - The positive gain is `exp(interpolated_log_gain)`. Results are clipped to `[0, 1]` for float32
          and rounded back into `[0, 255]` for uint8. Clipping can change contrast near the intensity maximum.
        - TorchIO uses `std=0.5` and `scale=0.025` as reference MRI settings. The default range samples up to
          that strength. Strong fields can obscure tissue contrast or alter clinical semantics; inspect augmented
          examples for the intended anatomy and task.
        - Parameters use voxel dimensions. Physical spacing and scanner-specific coil sensitivity are not modelled.

    Examples:
        >>> import albumentations as A
        >>> import numpy as np
        >>> volume = np.full((16, 64, 96, 1), 0.25, dtype=np.float32)
        >>> mask3d = np.zeros((16, 64, 96), dtype=np.uint8)
        >>> pipeline = A.Compose([
        ...     A.BiasField(std_range=(0.1, 0.3), scale_range=(0.02, 0.04), p=1),
        ... ], seed=137, strict=True)
        >>> result = pipeline(volume=volume, mask3d=mask3d)
        >>> result["volume"].shape
        (16, 64, 96, 1)

    References:
        TorchIO BiasField: https://docs.torchio.org/2.0/reference/transforms/bias_field/

    """

    class InitSchema(BaseTransformInitSchema):
        std_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        scale_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1, min_inclusive=False)),
            AfterValidator(nondecreasing),
        ]
        per_channel: bool

    def __init__(
        self,
        *,
        std_range: tuple[float, float] = (0.0, 0.5),
        scale_range: tuple[float, float] = (0.025, 0.025),
        per_channel: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.std_range = std_range
        self.scale_range = scale_range
        self.per_channel = per_channel

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        """Share coefficients by spatial shape and sampling family, including channel count only for per_channel.

        Replay validates each target's sampling topology against the captured requirements.
        """
        std = sampling.py_random.uniform(*self.std_range)
        scale = sampling.py_random.uniform(*self.scale_range)
        sampling.applied_overrides.update({"std_range": (std, std), "scale_range": (scale, scale)})
        if std == 0:
            return SampledParams(params={"coarse_field": None})

        groups: list[TargetParams] = []
        for views in targets.group_image_like_by(
            lambda view: (
                tuple(view.descriptor.spatial_shape or ()),
                view.descriptor.channels if self.per_channel else None,
                _sampling_family(view),
            ),
        ):
            view = views[0]
            spatial_shape = tuple(view.descriptor.spatial_shape or ())
            coarse_shape = tuple(min(size, max(2, int(size * scale))) for size in spatial_shape)
            channels = (view.descriptor.channels or 1) if self.per_channel else 1
            coarse_field = sampling.random_generator.standard_normal((*coarse_shape, channels), dtype=np.float32)
            np.multiply(coarse_field, std, out=coarse_field)
            groups.append(
                _target_parameter_group(
                    views,
                    "coarse_field",
                    coarse_field,
                    spatial_shape=True,
                    channels=self.per_channel,
                    topology=True,
                ),
            )
        return SampledParams(params={}, target_params=tuple(groups))

    @staticmethod
    def _apply_field(
        img: ImageType,
        coarse_field: np.ndarray | None,
        *,
        is_batch: bool,
    ) -> ImageType:
        if img.dtype not in (np.uint8, np.float32):
            raise ValueError("BiasField supports uint8 or float32 image/volume targets")
        if coarse_field is None:
            return img
        return fmedical.bias_field(img, coarse_field, is_batch=is_batch)

    def apply(self, img: ImageType, coarse_field: np.ndarray | None, **params: Any) -> ImageType:
        return self._apply_field(img, coarse_field, is_batch=False)

    def apply_to_images(self, images: ImageType, coarse_field: np.ndarray | None, **params: Any) -> ImageType:
        return self._apply_field(images, coarse_field, is_batch=True)

    def apply_to_volume(self, volume: ImageType, coarse_field: np.ndarray | None, **params: Any) -> ImageType:
        return self.apply(volume, coarse_field, **params)

    def apply_to_volumes(self, volumes: ImageType, coarse_field: np.ndarray | None, **params: Any) -> ImageType:
        return self.apply_to_images(volumes, coarse_field, **params)


class GhostingArtifact(VolumeOnlyTransform):
    """Simulate repeated MRI ghost replicas along one spatial axis by attenuating periodic k-space planes
    while protecting central frequencies.

    Periodic frequency loss creates displaced copies of anatomy in the reconstructed magnitude volume.
    The same acquisition parameters apply to every channel and collection item; anatomy annotations stay unchanged.

    Args:
        num_ghosts_range (tuple[int, int]): Inclusive frequency-comb period, at least two. When the axis length
            is divisible by the period, replicas are spaced by `axis_length / num_ghosts` voxels.
            Default: `(2, 4)`.
        intensity_range (tuple[float, float]): Nondecreasing range in `[0, 1]` for the fraction removed from
            affected Fourier coefficients. Zero gives exact identity; one removes those coefficients.
            Default: `(0.1, 0.5)`.
        axis (Literal[0, 1, 2]): Phase-encode axis in array order `(D, H, W)`. Default: `2`.
        restore_range (tuple[float, float]): Nondecreasing range in `[0, 1]` for the protected central fraction
            of the frequency axis. Frequencies with `abs(k) <= floor(restore * axis_length / 2)` are unchanged.
            Zero still protects DC; one gives exact identity. Default: `(0.02, 0.08)`.
        p (float): Probability of applying the transform. Default: `0.5`.

    Targets:
        volume, volumes

    Image types:
        uint8, float32

    Number of channels:
        Any

    Notes:
        - Inputs are real magnitude volumes. `Compose` supplies canonical NumPy `(D, H, W, C)` layouts
          and restores optional singleton channel dimensions. Complex input is unsupported.
        - The comb is anchored at DC: positive and negative frequencies that are multiples of `num_ghosts`
          are multiplied by `1 - intensity` outside the protected band. This preserves conjugate symmetry.
        - FFT/IFFT use backward normalization. A real FFT along the selected spatial axis is mathematically
          equivalent to a full 3D FFT with this plane mask; batch and channel axes are never transformed.
        - Reconstruction takes the absolute value and clips to `[0, 1]` for float32, or rounds back into
          `[0, 255]` for uint8. Magnitude and clipping can change the mean despite preserving central coefficients.
        - Non-divisible axis lengths give sampled, broadened replicas. If no affected frequency fits outside
          the protected band, the transform is an exact identity. This includes a singleton selected axis.
        - Physical spacing and scanner trajectories are not modelled. Select the axis for the input layout
          and inspect strong artifacts for the intended anatomy and task.

    Examples:
        >>> import albumentations as A
        >>> import numpy as np
        >>> volume = np.random.default_rng(137).random((16, 64, 96, 1), dtype=np.float32)
        >>> mask3d = np.zeros((16, 64, 96), dtype=np.uint8)
        >>> pipeline = A.Compose([
        ...     A.GhostingArtifact(num_ghosts_range=(2, 4), intensity_range=(0.2, 0.5), axis=1, p=1),
        ... ], seed=137, strict=True)
        >>> result = pipeline(volume=volume, mask3d=mask3d)
        >>> result["volume"].shape
        (16, 64, 96, 1)

    References:
        TorchIO Ghosting: https://docs.torchio.org/2.0/reference/transforms/ghosting/

    """

    class InitSchema(BaseTransformInitSchema):
        num_ghosts_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(2, None)),
            AfterValidator(nondecreasing),
        ]
        intensity_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        axis: AxisIndex3D
        restore_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        *,
        num_ghosts_range: tuple[int, int] = (2, 4),
        intensity_range: tuple[float, float] = (0.1, 0.5),
        axis: AxisIndex3D = 2,
        restore_range: tuple[float, float] = (0.02, 0.08),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.num_ghosts_range = num_ghosts_range
        self.intensity_range = intensity_range
        self.axis = axis
        self.restore_range = restore_range

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        """Share the sampled comb period, strength and central fraction across all volume targets."""
        num_ghosts = sampling.py_random.randint(*self.num_ghosts_range)
        intensity = sampling.py_random.uniform(*self.intensity_range)
        restore = sampling.py_random.uniform(*self.restore_range)
        sampling.applied_overrides.update(
            {
                "num_ghosts_range": (num_ghosts, num_ghosts),
                "intensity_range": (intensity, intensity),
                "restore_range": (restore, restore),
            },
        )
        return SampledParams(params={"num_ghosts": num_ghosts, "intensity": intensity, "restore": restore})

    def apply_to_volume(
        self,
        volume: ImageType,
        num_ghosts: int,
        intensity: float,
        restore: float,
        **params: Any,
    ) -> ImageType:
        if volume.dtype not in (np.uint8, np.float32):
            raise ValueError("GhostingArtifact supports uint8 or float32 real magnitude volumes")
        if intensity == 0 or restore == 1:
            return volume
        return fmedical.ghosting_artifact(volume, num_ghosts, intensity, self.axis, restore)


class GibbsRinging(_FullVolumeNoiseTransform):
    """Simulate Gibbs ringing near sharp intensity boundaries by truncating high-frequency k-space
    in an image or a whole 3D volume.

    A hard rectangular frequency cutoff creates oscillations and reduces spatial resolution. Use it to model
    finite Cartesian acquisition bandwidth; it is a generic artifact simulation, not a correction algorithm.

    Args:
        retained_fraction_range (tuple[float, float]): Nondecreasing range in `[0, 1]` for the fraction of
            the frequency bandwidth retained along every spatial axis. Smaller values remove more frequencies.
            One is an exact identity; zero keeps only DC, giving each channel its spatial mean.
            Default: `(0.5, 0.9)`.
        p (float): Probability of applying the transform. Default: `0.5`.

    Targets:
        image, images, volume, volumes

    Image types:
        uint8, float32

    Number of channels:
        Any

    Notes:
        - `Compose` normalizes optional channel axes. Direct calls use explicit channel-last `(H, W, C)`
          images or `(D, H, W, C)` volumes.
        - FFT/IFFT operate over `(H, W)` for images and `(D, H, W)` for volumes, including singleton depth.
          Channels and collection items are independent signals with one shared sampled cutoff fraction.
        - On an axis of length `N`, retain signed frequency indices with
          `abs(k) <= floor(retained_fraction * N / 2)`. DC is always included. Even-length Nyquist bins
          are retained only when the cutoff reaches `N / 2`; odd lengths use their actual signed bins.
        - The retained region is a rectangle in 2D and a rectangular box in 3D. The fraction describes
          bandwidth along each axis, not the fraction of all Fourier coefficients or physical voxel spacing.
        - FFT normalization is backward. The symmetric mask gives a real inverse reconstruction;
          no magnitude operation is applied. Fourier boundaries are periodic, so image-border jumps can ring.
        - Inputs must be finite real uint8 or float32 arrays; float32 inputs are expected in `[0, 1]`.
          Float32 reconstruction is clipped to `[0, 1]`; uint8 is processed in float32 and rounded back
          into `[0, 255]`. Clipping can change the mean.
        - Cutoffs are discrete. Nearby fractions can select the same bins, and sufficiently small arrays can
          retain their entire spectrum. Removing more frequencies does not guarantee a larger local overshoot.

    Examples:
        >>> import albumentations as A
        >>> import numpy as np
        >>> image = np.zeros((64, 96, 1), dtype=np.float32)
        >>> image[16:48, 24:72] = 0.75
        >>> transform = A.GibbsRinging(retained_fraction_range=(0.4, 0.8), p=1)
        >>> augmented = transform(image=image)["image"]
        >>> volume = np.zeros((16, 64, 96, 1), dtype=np.float32)
        >>> volume[4:12, 16:48, 24:72] = 0.75
        >>> pipeline = A.Compose([transform], seed=137, strict=True)
        >>> result = pipeline(volume=volume)["volume"]
        >>> result.shape
        (16, 64, 96, 1)

    References:
        Gibbs Ringing in Diffusion MRI: https://pmc.ncbi.nlm.nih.gov/articles/PMC4915073/

    """

    class InitSchema(BaseTransformInitSchema):
        retained_fraction_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        *,
        retained_fraction_range: tuple[float, float] = (0.5, 0.9),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.retained_fraction_range = retained_fraction_range

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        retained_fraction = sampling.py_random.uniform(*self.retained_fraction_range)
        sampling.applied_overrides["retained_fraction_range"] = (retained_fraction, retained_fraction)
        return SampledParams(params={"retained_fraction": retained_fraction})

    def apply(self, img: ImageType, retained_fraction: float, **params: Any) -> ImageType:
        if img.dtype not in (np.uint8, np.float32):
            raise ValueError("GibbsRinging supports uint8 or float32 real inputs")
        if img.dtype == np.float32 and not np.isfinite(img).all():
            raise ValueError("GibbsRinging requires finite inputs")
        if retained_fraction == 1:
            return img
        return fmedical.gibbs_ringing(img, retained_fraction)

    def apply_to_volume(self, volume: ImageType, retained_fraction: float, **params: Any) -> ImageType:
        return self.apply(volume, retained_fraction, **params)


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
        image, images, volume, volumes

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
        >>> image = np.zeros((300, 300, 3), dtype=np.uint8)
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

        if method in ["vahadane", "macenko"]:
            self.stain_extractor = fmedical.get_normalizer(method)

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
            return fmedical.STAIN_MATRICES[self.preset]
        if self.method == "random_preset":
            random_preset = sampling.py_random.choice(self.preset_names)
            return fmedical.STAIN_MATRICES[random_preset]
        if self.method == "custom":
            return cast("np.ndarray", self.stain_matrix)
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
        return fmedical.apply_he_stain_augmentation(
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
            if view.canonical_type == "image":
                image = view.value
            elif view.canonical_type == "volumes":
                image = view.value[0, 0]
            else:
                image = view.value[0]
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


class KSpaceSpikeNoise(_FullVolumeNoiseTransform):
    """Inject point spikes into the MRI k-space spectrum and reconstruct the image or volume,
    producing structured stripes typical of acquisition failures.

    K-space spike artifacts arise from isolated high-energy points in the Fourier
    representation of MRI data (e.g. scanner spikes, radio-frequency interference).
    Each sampled spike adds a real amplitude at its frequency and at the conjugate
    mirror, keeping the spectrum Hermitian so the reconstruction is real.

    Args:
        num_spikes_range (tuple[int, int]): Inclusive range for the number of spikes sampled
            per invocation. Zero spikes is an exact identity. Default: (1, 5).
        intensity_range (tuple[float, float]): Range for the spike amplitude as a fraction of
            the spectrum maximum magnitude. Zero is an exact identity. Default: (0.1, 0.5).
        per_channel (bool): If True, sample independent spike locations and amplitudes for each
            channel. If False, share one set of spikes across all channels. Default: False.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, images, volume, volumes

    Image types:
        uint8, float32

    Number of channels:
        Any

    Notes:
        - Fourier frequencies refer to spatial axes only; batch and channel dimensions are excluded.
          Shared float32 volume spikes can be reconstructed analytically. Float32 results may differ
          from FFT reconstruction by roundoff; uint8 retains FFT reconstruction and its rounding.
        - Each spike injects a real amplitude `intensity * max|F|` at the sampled bin and at its
          conjugate mirror, so the half-spectrum stays Hermitian and the reconstruction is real
          without discarding imaginary parts. Self-conjugate bins (DC and the Nyquist bin of even
          axes) are injected once.
        - Spikes are uniform over the full frequency grid, including DC. A spike at DC shifts the
          global mean rather than creating stripes; this is intentional and documented.
        - A spike of relative amplitude `i` turns a flat field of value `c` into a cosine
          pattern of amplitude `2 * i * c` along the spike's frequency direction.
        - One spike realization is sampled per transform invocation and reused across all channels
          (shared mode), all images in a batch, and the whole volume as a single 3D transform.
        - uint8 inputs are processed as float32 in [0, 1] and converted back with rounding, so
          outputs stay within [0, 255]; float32 outputs are clipped to [0, 1].
        - This differs from image-space impulse noise (SaltAndPepper), which replaces individual
          pixels, and from RingingOvershoot, which convolves in the image domain.

    Mathematical Formulation:
        F = rfftn(I, axes=spatial)
        a = intensity * max(|F|)

        for each spike k:
            F[k] += a
            F[(-k) mod N] += a

        I = clip(irfftn(F), 0, 1)

        For a flat field the pair injection produces a cosine artifact of amplitude 2a / (H * W)
        at the spike frequency, oriented along the spike's direction.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        >>> transform = A.Compose(
        ...     [A.KSpaceSpikeNoise(num_spikes_range=(2, 4), intensity_range=(0.1, 0.3), p=1.0)],
        ...     seed=137,
        ... )
        >>> spiked = transform(image=image)["image"]

        Volumes are supported with one 3D spike realization per invocation:
        >>> volume = np.random.rand(4, 32, 32, 1).astype(np.float32)
        >>> transform = A.Compose([A.KSpaceSpikeNoise(p=1.0)], seed=137)
        >>> spiked_volume = transform(volume=volume)["volume"]

    References:
        - TorchIO RandomSpike: https://docs.torchio.org/2.0/reference/transforms/spike/
        - TorchIO paper: https://www.sciencedirect.com/science/article/pii/S0169260721003102

    See Also:
        - SaltAndPepper: Image-space impulse noise; use when corruption lives in pixel space.
        - RingingOvershoot: Image-domain convolution ringing; use for sharpening artifacts.
        - RicianNoise: MRI magnitude-reconstruction noise with a low-signal floor.
        - GaussNoise: Additive Gaussian noise for general sensor or transmission noise.

    """

    class InitSchema(BaseTransformInitSchema):
        num_spikes_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(0)),
            AfterValidator(nondecreasing),
        ]
        intensity_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0)),
            AfterValidator(nondecreasing),
        ]
        per_channel: bool

    def __init__(
        self,
        *,
        num_spikes_range: tuple[int, int] = (1, 5),
        intensity_range: tuple[float, float] = (0.1, 0.5),
        per_channel: bool = False,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.num_spikes_range = num_spikes_range
        self.intensity_range = intensity_range
        self.per_channel = per_channel

    def apply(self, img: ImageType, spikes: np.ndarray, intensity: float, **params: Any) -> ImageType:
        if intensity == 0 or spikes.size == 0:
            return img
        return fmedical.k_space_spike(img, spikes, intensity)

    def apply_to_images(self, images: ImageType, spikes: np.ndarray, intensity: float, **params: Any) -> ImageType:
        if intensity == 0 or spikes.size == 0:
            return images
        return fmedical.k_space_spike(images, spikes, intensity)

    def apply_to_volume(self, volume: ImageType, spikes: np.ndarray, intensity: float, **params: Any) -> ImageType:
        if intensity == 0 or spikes.size == 0:
            return volume
        return fmedical.k_space_spike(volume, spikes, intensity)

    @staticmethod
    def _spatial_rank(view: TargetView) -> int:
        spatial_shape = view.descriptor.spatial_shape
        if spatial_shape is not None:
            return len(spatial_shape)
        return 3 if view.canonical_type in {"volume", "volumes", "mask3d", "masks3d"} else 2

    @staticmethod
    def _sample_spikes(
        spatial_shape: tuple[int, ...],
        num_spikes: int,
        channel_count: int | None,
        sampling: SamplingContext,
    ) -> np.ndarray:
        rank = len(spatial_shape)
        bounds = np.asarray(spatial_shape, dtype=np.int64)
        if channel_count is None:
            return sampling.random_generator.integers(0, bounds, size=(num_spikes, rank))
        return sampling.random_generator.integers(0, bounds, size=(channel_count, num_spikes, rank))

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        del params, data
        num_spikes = sampling.py_random.randrange(self.num_spikes_range[0], self.num_spikes_range[1] + 1)
        intensity = sampling.py_random.uniform(*self.intensity_range)
        sampling.applied_overrides.update({"num_spikes_range": num_spikes, "intensity_range": intensity})

        groups: list[TargetParams] = []
        key = (
            (lambda view: (view.descriptor.channels, self._spatial_rank(view)))
            if self.per_channel
            else self._spatial_rank
        )
        for views in targets.group_image_like_by(key):
            rank = self._spatial_rank(views[0])
            spatial_shape = targets.require_aligned_spatial_shape(rank)
            channel_count = views[0].descriptor.channels if self.per_channel else None
            if self.per_channel and channel_count is None:
                raise ValueError(
                    "KSpaceSpikeNoise requires image-like targets with a known channel count for per_channel=True",
                )
            spikes = self._sample_spikes(spatial_shape, num_spikes, channel_count, sampling)
            groups.append(
                _target_parameter_group(
                    views,
                    "spikes",
                    spikes,
                    spatial_shape=True,
                    channels=self.per_channel,
                ),
            )
        return SampledParams(params={"intensity": intensity}, target_params=tuple(groups))


class MotionArtifact(VolumeOnlyTransform):
    """Simulate motion artifacts in 3D MRI volumes by combining k-space from successive rigid movement states
    during image acquisition.

    The transform acquires consecutive planes of centred k-space along one selected axis. The initial segment uses
    the original volume; each subsequent segment uses an independently sampled absolute rotation and translation
    of that volume. A magnitude inverse FFT reconstructs the corrupted image. Anatomy annotations stay unchanged.

    Args:
        num_events_range (tuple[int, int]): Inclusive number of motion events. Zero events give exact identity.
            The upper bound must be smaller than the acquisition-axis length. Default: `(2, 2)`.
        rotate_range (tuple[float, float]): Degree range sampled independently around each voxel-coordinate
            axis `(x, y, z)`, rotating about the volume centre. Default: `(-5.0, 5.0)`.
        translate_percent_range (tuple[float, float]): Translation range sampled independently along `(x, y, z)`.
            A value of `0.01` shifts by one percent of that axis length. Default: `(-0.02, 0.02)`.
        axis (Literal[0, 1, 2]): Acquisition axis in array order `(D, H, W)`. Default: `2`.
        interpolation (Literal[0, 1]): Resampling interpolation: `cv2.INTER_NEAREST` or `cv2.INTER_LINEAR`.
            Default: `cv2.INTER_LINEAR`.
        p (float): Probability of applying the transform. Default: `0.5`.

    Targets:
        volume, volumes

    Image types:
        uint8, float32

    Notes:
        - Inputs are real magnitude volumes `(D, H, W)` or `(D, H, W, C)`. Every spatial axis must have at least
          two voxels. Complex-valued input is unsupported.
        - Event boundaries are sampled uniformly without replacement from the interior acquisition-plane boundaries.
          The event time is the boundary index divided by the selected axis length.
        - Channels, collection items, and additional volumes share the same event times and motion states.
          FFTs operate over `(D, H, W)` only, independently for each channel and collection item.
        - Resampling uses zero padding. FFT/IFFT use backward normalization; the magnitude of the complex
          reconstruction is clipped to `[0, 1]` for float32 or converted back to `[0, 255]` for uint8.
        - Rotations and translations use voxel coordinates. Physical spacing and scanner acquisition trajectories
          are not modelled. Parameter strength must be chosen for the anatomy and task.
        - Runtime grows with the number of events. Moved volumes and spectra are processed one at a time.

    Examples:
        >>> import albumentations as A
        >>> import numpy as np
        >>> volume = np.random.default_rng(137).random((16, 64, 96, 1), dtype=np.float32)
        >>> transform = A.Compose([A.MotionArtifact(num_events_range=(1, 2), p=1)], seed=137, strict=True)
        >>> result = transform(volume=volume)
        >>> result["volume"].shape
        (16, 64, 96, 1)

    References:
        - TorchIO Motion: https://docs.torchio.org/2.0/reference/transforms/motion/
        - Shaw et al., 2019: https://proceedings.mlr.press/v102/shaw19a.html

    """

    class InitSchema(BaseTransformInitSchema):
        num_events_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        rotate_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(-180, 180)),
            AfterValidator(nondecreasing),
        ]
        translate_percent_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(-1, 1)),
            AfterValidator(nondecreasing),
        ]
        axis: AxisIndex3D
        interpolation: Literal[0, 1]

    def __init__(
        self,
        *,
        num_events_range: tuple[int, int] = (2, 2),
        rotate_range: tuple[float, float] = (-5.0, 5.0),
        translate_percent_range: tuple[float, float] = (-0.02, 0.02),
        axis: AxisIndex3D = 2,
        interpolation: Literal[0, 1] = CV2_INTER_LINEAR,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.num_events_range = num_events_range
        self.rotate_range = rotate_range
        self.translate_percent_range = translate_percent_range
        self.axis = axis
        self.interpolation = interpolation

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        source_shape = _sampling_volume_shape(targets)
        if min(source_shape) < 2:
            raise ValueError("MotionArtifact requires at least two voxels along every spatial axis, including D")
        if self.num_events_range[1] >= source_shape[self.axis]:
            raise ValueError("num_events_range must be smaller than the acquisition-axis length")

        num_events = sampling.py_random.randint(*self.num_events_range)
        boundaries = tuple(sorted(sampling.py_random.sample(range(1, source_shape[self.axis]), num_events)))
        matrices = []
        for _ in range(num_events):
            rotate = {axis: sampling.py_random.uniform(*self.rotate_range) for axis in AXIS_NAMES_3D}
            translate = {
                axis: sampling.py_random.uniform(*self.translate_percent_range) * length
                for axis, length in zip(AXIS_NAMES_3D, reversed(source_shape), strict=True)
            }
            matrices.append(
                f3d.create_affine_transformation_matrix_3d(
                    translate,
                    dict.fromkeys(AXIS_NAMES_3D, 1.0),
                    rotate,
                    source_shape,
                ),
            )
        sampling.applied_overrides["num_events_range"] = (num_events, num_events)
        return SampledParams(params={"matrices": np.asarray(matrices, dtype=np.float32), "boundaries": boundaries})

    def apply_to_volume(
        self,
        volume: VolumeType,
        matrices: np.ndarray,
        boundaries: tuple[int, ...],
        **params: Any,
    ) -> VolumeType:
        if volume.dtype not in (np.uint8, np.float32):
            raise ValueError("MotionArtifact supports uint8 or float32 real magnitude volumes")
        return fmedical.motion_artifact(volume, matrices, boundaries, self.axis, self.interpolation)


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
        image, images, volume, volumes

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
        return fmedical.rician_noise(img, real_noise, imaginary_noise)

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
