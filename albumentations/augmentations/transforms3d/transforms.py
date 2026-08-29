"""Module containing 3D transformation classes for volumetric data augmentation.

This module provides a collection of transformation classes designed specifically for
3D volumetric data (such as medical CT/MRI scans). These transforms can manipulate properties
such as spatial dimensions, apply dropout effects, and perform symmetry operations on
3D volume data, masks, and keypoints. Each transformation inherits from a base transform
interface and implements specific 3D augmentation logic.
"""

from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Final, Literal, cast

import numpy as np
import torch
from albucore import resize3d
from pydantic import AfterValidator, field_validator, model_validator
from typing_extensions import Self

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.transforms3d import functional as f3d
from albumentations.core.invocation import SamplingContext
from albumentations.core.keypoints_utils import KeypointsProcessor
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transform_params import SampledParams, TargetSet
from albumentations.core.transforms_interface import BaseTransformInitSchema, Transform3D, VolumeOnlyTransform
from albumentations.core.type_definitions import (
    C4_INVERSE,
    CV2_BORDER_CONSTANT,
    CV2_INTER_LINEAR,
    CV2_INTER_NEAREST,
    C4GroupElement,
    Targets,
    VolumeType,
    c4_group_elements,
)

__all__ = [
    "Affine3D",
    "Anisotropy3D",
    "CenterCrop3D",
    "CoarseDropout3D",
    "CubicSymmetry",
    "ElasticTransform3D",
    "Flip3D",
    "GridShuffle3D",
    "Pad3D",
    "PadIfNeeded3D",
    "RandomCrop3D",
    "RandomRotate90_3D",
    "Resize3D",
]

NUM_DIMENSIONS: Final = 3
AxisIndex3D = Literal[0, 1, 2]
AxisPair3D = tuple[AxisIndex3D, AxisIndex3D]
RotationCount3D = Literal[0, 1, 2, 3]
AxisName3D = Literal["x", "y", "z"]
AxisRange3D = Annotated[tuple[float, float], AfterValidator(nondecreasing)]
PositiveAxisRange3D = Annotated[
    tuple[float, float],
    AfterValidator(check_range_bounds(0, None, min_inclusive=False)),
    AfterValidator(nondecreasing),
]
AxisRanges3D = dict[AxisName3D, AxisRange3D]
PositiveAxisRanges3D = dict[AxisName3D, PositiveAxisRange3D]
DEFAULT_ROTATION_AXIS_PAIRS: tuple[AxisPair3D, ...] = ((0, 1), (0, 2), (1, 2))
DEFAULT_FLIP_AXES: tuple[AxisIndex3D, ...] = (0, 1, 2)
AXIS_NAMES_3D: tuple[AxisName3D, ...] = ("x", "y", "z")


def _sampling_volume_shape(targets: TargetSet) -> tuple[int, int, int]:
    return targets.require_aligned_spatial_shape(NUM_DIMENSIONS)


class Affine3D(Transform3D):
    """Apply a sampled 3D affine mapping to volume and mask3d by rotating, scaling, and shifting
    voxel coordinates for robust medical-imaging augmentation.

    `Affine3D` resamples depth, height, and width jointly through Albucore `warp_affine3d`; it never treats depth as a
    batch axis. The output grid keeps the input `(D, H, W)` shape. It samples positive per-axis scales, rotations, and
    relative translations independently, then applies the same forward matrix to `volume`, `mask3d`, and `xyz`
    keypoints.

    Args:
        rotate_range (dict[str, tuple[float, float]]): Inclusive degree ranges around the `x`, `y`, and `z` axes.
            Axis names use the `(x, y, z)` voxel coordinate order. Default: all `(0.0, 0.0)`.
        scale_range (dict[str, tuple[float, float]]): Positive multiplicative scale ranges for `x`, `y`, and `z`.
            `1.0` leaves an axis unchanged. Default: all `(1.0, 1.0)`.
        translate_percent_range (dict[str, tuple[float, float]]): Relative translation ranges for `x`, `y`, and `z`.
            A value of `1.0` moves by the corresponding input-axis length. Default: all `(0.0, 0.0)`.
        interpolation (Literal[0, 1]): Volume interpolation: `cv2.INTER_NEAREST` or `cv2.INTER_LINEAR`.
            Default: `cv2.INTER_LINEAR`.
        mask_interpolation (Literal[0, 1]): `mask3d` interpolation: `cv2.INTER_NEAREST` or `cv2.INTER_LINEAR`.
            Default: `cv2.INTER_NEAREST`.
        border_mode (Literal[0, 1]): Border policy: `cv2.BORDER_CONSTANT` or `cv2.BORDER_REPLICATE`.
            Default: `cv2.BORDER_CONSTANT`.
        fill (tuple[float, ...] | float): Constant fill for volume channels when `border_mode` is constant.
            Default: `0`.
        fill_mask (tuple[float, ...] | float): Constant fill for `mask3d` when `border_mode` is constant.
            Default: `0`.
        p (float): Probability of applying the transform. Default: `0.5`.

    Targets:
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Notes:
        - Volume arrays are `(D, H, W, C)` and keypoints are `(x, y, z)`.
          The centred forward matrix applies scale, then x-, y-, and z-axis rotations, then translation. One sampled
          matrix is shared across all elements of a sampled transform.
        - Scale factors must be positive, so this transform does not sample reflections. Use `Flip3D` for reflections.
        - Transform parameters use voxel coordinates only; physical spacing, orientation, and affine metadata remain
          unchanged.

    Examples:
        >>> import albumentations as A
        >>> import cv2
        >>> import numpy as np
        >>> volume = np.random.default_rng(137).random((16, 64, 96, 1), dtype=np.float32)
        >>> mask3d = np.zeros((16, 64, 96), dtype=np.uint8)
        >>> keypoints = np.array([[48.0, 32.0, 8.0]], dtype=np.float32)
        >>> transform = A.Compose([
        ...     A.Affine3D(
        ...         rotate_range={"x": (-10.0, 10.0), "y": (-5.0, 5.0), "z": (-15.0, 15.0)},
        ...         scale_range={"x": (0.9, 1.1), "y": (0.9, 1.1), "z": (0.95, 1.05)},
        ...         translate_percent_range={"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05)},
        ...         interpolation=cv2.INTER_LINEAR,
        ...         mask_interpolation=cv2.INTER_NEAREST,
        ...         p=1.0,
        ...     ),
        ... ], keypoint_params=A.KeypointParams(coord_format="xyz"), strict=True)
        >>> result = transform(volume=volume, mask3d=mask3d, keypoints=keypoints)
        >>> result["volume"].shape, result["mask3d"].shape
        ((16, 64, 96, 1), (16, 64, 96))

    See Also:
        - `Resize3D`: Resize every spatial axis without sampling an affine matrix.
        - `RandomRotate90_3D`: Use exact right-angle rotations without interpolation.
        - `Flip3D`: Apply discrete axis reflections without resampling.

    Returns:
        dict[str, Any]: Augmented targets when the transform is executed through `Compose`.

    """

    _targets = (Targets.VOLUME, Targets.MASK3D, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        rotate_range: AxisRanges3D
        scale_range: PositiveAxisRanges3D
        translate_percent_range: AxisRanges3D
        interpolation: Literal[0, 1]
        mask_interpolation: Literal[0, 1]
        border_mode: Literal[0, 1]
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

        @model_validator(mode="after")
        def _validate_axis_ranges(self) -> Self:
            expected_axes = set(AXIS_NAMES_3D)
            for field_name, axis_ranges in (
                ("rotate_range", self.rotate_range),
                ("scale_range", self.scale_range),
                ("translate_percent_range", self.translate_percent_range),
            ):
                if set(axis_ranges) != expected_axes:
                    raise ValueError(f"{field_name} must define exactly the x, y, and z axes")
            return self

    def __init__(
        self,
        rotate_range: AxisRanges3D = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        scale_range: PositiveAxisRanges3D = {"x": (1.0, 1.0), "y": (1.0, 1.0), "z": (1.0, 1.0)},
        translate_percent_range: AxisRanges3D = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        interpolation: Literal[0, 1] = CV2_INTER_LINEAR,
        mask_interpolation: Literal[0, 1] = CV2_INTER_NEAREST,
        border_mode: Literal[0, 1] = CV2_BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.rotate_range = rotate_range
        self.scale_range = scale_range
        self.translate_percent_range = translate_percent_range
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.border_mode = border_mode
        self.fill = fill
        self.fill_mask = fill_mask

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        source_shape = _sampling_volume_shape(targets)
        rotate = {axis: sampling.py_random.uniform(*self.rotate_range[axis]) for axis in AXIS_NAMES_3D}
        scale = {axis: sampling.py_random.uniform(*self.scale_range[axis]) for axis in AXIS_NAMES_3D}
        translate_percent = {
            axis: sampling.py_random.uniform(*self.translate_percent_range[axis]) for axis in AXIS_NAMES_3D
        }
        depth, height, width = source_shape
        translate = {
            "x": translate_percent["x"] * width,
            "y": translate_percent["y"] * height,
            "z": translate_percent["z"] * depth,
        }
        matrix = f3d.create_affine_transformation_matrix_3d(translate, scale, rotate, source_shape)

        sampling.applied_overrides.update(
            {
                "rotate_range": {axis: (rotate[axis], rotate[axis]) for axis in AXIS_NAMES_3D},
                "scale_range": {axis: (scale[axis], scale[axis]) for axis in AXIS_NAMES_3D},
                "translate_percent_range": {
                    axis: (translate_percent[axis], translate_percent[axis]) for axis in AXIS_NAMES_3D
                },
            },
        )
        return SampledParams(params={"matrix": matrix, "output_shape": source_shape})

    def apply_to_volume(
        self,
        volume: VolumeType | torch.Tensor,
        matrix: np.ndarray,
        output_shape: tuple[int, int, int],
        **params: Any,
    ) -> VolumeType:
        return cast(
            "VolumeType",
            f3d.affine_3d(volume, matrix, output_shape, self.interpolation, self.border_mode, self.fill),
        )

    def apply_to_mask3d(
        self,
        mask3d: VolumeType | torch.Tensor,
        matrix: np.ndarray,
        output_shape: tuple[int, int, int],
        **params: Any,
    ) -> VolumeType:
        return cast(
            "VolumeType",
            f3d.affine_3d(
                mask3d,
                matrix,
                output_shape,
                self.mask_interpolation,
                self.border_mode,
                self.fill_mask,
                is_mask=True,
            ),
        )

    def apply_to_keypoints(self, keypoints: np.ndarray, matrix: np.ndarray, **params: Any) -> np.ndarray:
        return f3d.keypoints_affine_3d(keypoints, matrix)


class ElasticTransform3D(Transform3D):
    """Apply a smooth bounded 3D elastic deformation from compact cubic control planes, useful for volumetric
    segmentation and anatomy-shape robustness.

    The transform samples XY, XZ, and YZ cubic B-spline coefficient planes, averages their embedded displacement
    fields, and resamples each raster target once through Albucore `remap3d`. The same compact field drives volume,
    `mask3d`, and XYZ keypoint geometry. `displacement_range` scales coefficients by the shortest active voxel-center
    span, so the policy transfers across volume sizes without a dense noise or smoothing pass.

    Args:
        displacement_range (tuple[float, float]): Inclusive relative coefficient-radius range. Default: `(0.02, 0.05)`.
        control_grid_shape (tuple[int, int]): Cubic coefficient rows and columns for each plane. Default:
            `(7, 7)`.
        interpolation (Literal[0, 1]): Volume interpolation: `cv2.INTER_NEAREST` or `cv2.INTER_LINEAR`. Default:
            `cv2.INTER_LINEAR`.
        mask_interpolation (Literal[0, 1]): `mask3d` interpolation: `cv2.INTER_NEAREST` or `cv2.INTER_LINEAR`.
            Default: `cv2.INTER_NEAREST`.
        border_mode (Literal[0, 1]): Border policy: `cv2.BORDER_CONSTANT` or `cv2.BORDER_REPLICATE`. Default:
            `cv2.BORDER_CONSTANT`.
        fill (tuple[float, ...] | float): Constant fill for volume channels. Default: `0`.
        fill_mask (tuple[float, ...] | float): Constant fill for `mask3d`. Default: `0`.
        p (float): Probability of applying the transform. Default: `0.5`.

    Targets:
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Notes:
        - The constructor enforces `2 * high * sqrt((rows - 3)^2 + (columns - 3)^2) < 0.75`, which bounds the
          deformation's Lipschitz constant below one.
        - Replay stores compact coefficients and requires the same `(depth, height, width)` shape. Applied
          configuration fixes only the realized magnitude and samples fresh coefficient planes.
        - CPU Tensor volumes run natively for every channel count and retain their `(C, D, H, W)` layout.
        - This transform changes voxel-index coordinates. Physical spacing, orientation metadata, and 3D bounding
          boxes are outside its contract.

    See Also:
        - ElasticTransform: Applies the same bounded cubic field policy to 2D images and annotations.
        - Affine3D: Applies global rotation, scale, and translation when local elastic deformation is unnecessary.

    Examples:
        >>> import albumentations as A
        >>> import cv2
        >>> import numpy as np
        >>> volume = np.random.default_rng(137).random((16, 64, 96, 1), dtype=np.float32)
        >>> mask3d = np.zeros((16, 64, 96), dtype=np.uint8)
        >>> keypoints = np.array([[48.0, 32.0, 8.0]], dtype=np.float32)
        >>> keypoint_labels = [3]
        >>> transform = A.Compose([
        ...     A.ElasticTransform3D(
        ...         displacement_range=(0.02, 0.05),
        ...         control_grid_shape=(7, 7),
        ...         interpolation=cv2.INTER_LINEAR,
        ...         mask_interpolation=cv2.INTER_NEAREST,
        ...         p=1.0,
        ...     ),
        ... ], keypoint_params=A.KeypointParams(coord_format="xyz", label_fields=["keypoint_labels"]))
        >>> result = transform(
        ...     volume=volume,
        ...     mask3d=mask3d,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels,
        ... )
        >>> result["volume"].shape, result["mask3d"].shape, result["keypoint_labels"]
        ((16, 64, 96, 1), (16, 64, 96), [3])

    """

    _targets = (Targets.VOLUME, Targets.MASK3D, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        displacement_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0)),
            AfterValidator(nondecreasing),
        ]
        control_grid_shape: tuple[int, int]
        interpolation: Literal[0, 1]
        mask_interpolation: Literal[0, 1]
        border_mode: Literal[0, 1]
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

        @field_validator("control_grid_shape")
        @classmethod
        def _validate_control_grid_shape(cls, value: tuple[int, int]) -> tuple[int, int]:
            if len(value) != 2 or min(value) < 4:
                raise ValueError("control_grid_shape must contain two dimensions, each at least 4")
            return value

        @model_validator(mode="after")
        def _validate_topology_bound(self) -> Self:
            rows, columns = self.control_grid_shape
            high = self.displacement_range[1]
            bound = 2 * high * float(np.sqrt((rows - 3) ** 2 + (columns - 3) ** 2))
            if bound >= 0.75:
                raise ValueError(
                    "displacement_range and control_grid_shape violate the strict topology bound: "
                    "2 * high * sqrt((rows - 3)^2 + (columns - 3)^2) must be less than 0.75",
                )
            return self

    def __init__(
        self,
        displacement_range: tuple[float, float] = (0.02, 0.05),
        control_grid_shape: tuple[int, int] = (7, 7),
        interpolation: Literal[0, 1] = CV2_INTER_LINEAR,
        mask_interpolation: Literal[0, 1] = CV2_INTER_NEAREST,
        border_mode: Literal[0, 1] = CV2_BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.displacement_range = displacement_range
        self.control_grid_shape = control_grid_shape
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.border_mode = border_mode
        self.fill = fill
        self.fill_mask = fill_mask

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        del params, data
        volume_shape = _sampling_volume_shape(targets)
        raster_target_count = len(targets.by_canonical_type("volume")) + len(targets.by_canonical_type("mask3d"))
        low, high = self.displacement_range
        magnitude = low if low == high else sampling.py_random.uniform(low, high)
        sampling.applied_overrides["displacement_range"] = (magnitude, magnitude)
        if magnitude == 0:
            return SampledParams(
                params={
                    "sampler": f3d.ElasticTransform3DSampler({}, volume_shape, raster_target_count),
                }
            )

        shortest_active_span = min(axis_length - 1 for axis_length in volume_shape if axis_length > 1)
        coefficient_radius = magnitude * shortest_active_span
        control_coefficients = {
            plane: fgeometric.sample_elastic_control_coefficients(
                self.control_grid_shape,
                coefficient_radius,
                sampling.random_generator,
            ).tolist()
            for plane in ("xy", "xz", "yz")
        }
        return SampledParams(
            params={
                "sampler": f3d.ElasticTransform3DSampler(
                    control_coefficients,
                    volume_shape,
                    raster_target_count,
                ),
            }
        )

    def apply_to_volume(
        self,
        volume: VolumeType | torch.Tensor,
        sampler: Mapping[str, Any],
        **params: Any,
    ) -> VolumeType:
        return cast(
            "VolumeType",
            f3d.elastic_transform_3d(
                volume,
                sampler,
                self.interpolation,
                self.border_mode,
                self.fill,
            ),
        )

    def apply_to_mask3d(
        self,
        mask3d: VolumeType | torch.Tensor,
        sampler: Mapping[str, Any],
        **params: Any,
    ) -> VolumeType:
        return cast(
            "VolumeType",
            f3d.elastic_transform_3d(
                mask3d,
                sampler,
                self.mask_interpolation,
                self.border_mode,
                self.fill_mask,
                is_mask=True,
            ),
        )

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        sampler: Mapping[str, Any],
        **params: Any,
    ) -> np.ndarray:
        control_coefficients = sampler["control_coefficients"]
        if not control_coefficients:
            return keypoints
        return f3d.remap_elastic_keypoints_3d(
            keypoints,
            control_coefficients,
            tuple(sampler["volume_shape"]),
        )


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
        volume

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
        downsample_shape = f3d.get_anisotropy_downsample_shape(
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
        return cast("VolumeType", f3d.anisotropy_3d(volume, downsample_shape, self.antialias))


class Resize3D(Transform3D):
    """Resize a volume to a fixed `(depth, height, width)` shape, preserving all\
    channels, dtype, and its public layout intact.

    `Resize3D` resamples all three spatial axes together; depth is never treated as a
    batch axis. Single-volume intensity data and categorical masks use independently configurable
    interpolation. The routed Albucore backend supports only linear and nearest-neighbor
    interpolation, ensuring the same public contract for NumPy and CPU Tensor inputs.

    Args:
        size (tuple[int, int, int]): Target spatial shape in `(depth, height, width)` order.
        interpolation (Literal[0, 1]): Interpolation for a volume: `cv2.INTER_LINEAR` or
            `cv2.INTER_NEAREST`. Default: `cv2.INTER_LINEAR`.
        mask_interpolation (Literal[0, 1]): Interpolation for `mask3d`:
            `cv2.INTER_LINEAR` or `cv2.INTER_NEAREST`. Default: `cv2.INTER_NEAREST`.
        p (float): Probability of applying the transform. Default: `1.0`.

    Targets:
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Notes:
        - NumPy volume data use channel-last `(D, H, W, C)` layout. CPU Tensor volume data
          use channel-first `(C, D, H, W)` layout.
        - `uint8` output preserves dtype; linear resampling rounds and saturates to
          `[0, 255]`. Float32 output remains float32.
        - Keypoints use `(x, y, z)` order and scale from the voxel-grid origin by
          `(W2 / W1, H2 / H1, D2 / D1)`, matching the 2D resize convention.
        - This transform changes voxel-grid coordinates only. It does not update physical
          voxel spacing or orientation metadata.

    Examples:
        >>> import albumentations as A
        >>> import cv2
        >>> import numpy as np
        >>> volume = np.random.default_rng(137).random((16, 64, 96, 1), dtype=np.float32)
        >>> mask3d = np.zeros((16, 64, 96), dtype=np.uint8)
        >>> transform = A.Compose([
        ...     A.Resize3D(
        ...         size=(32, 128, 128),
        ...         interpolation=cv2.INTER_LINEAR,
        ...         mask_interpolation=cv2.INTER_NEAREST,
        ...     ),
        ... ])
        >>> result = transform(volume=volume, mask3d=mask3d)
        >>> result["volume"].shape, result["mask3d"].shape
        ((32, 128, 128, 1), (32, 128, 128))

    See Also:
        - `Affine3D`: Sample continuous rotations, scales, and translations while keeping the input output grid.

    """

    _targets = (Targets.VOLUME, Targets.MASK3D, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        size: Annotated[tuple[int, int, int], AfterValidator(check_range_bounds(1, None))]
        interpolation: Literal[0, 1]
        mask_interpolation: Literal[0, 1]

    def __init__(
        self,
        size: tuple[int, int, int],
        interpolation: Literal[0, 1] = CV2_INTER_LINEAR,
        mask_interpolation: Literal[0, 1] = CV2_INTER_NEAREST,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.size = size
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        return SampledParams(params={"source_shape": _sampling_volume_shape(targets)})

    def apply_to_volume(self, volume: VolumeType | torch.Tensor, **params: Any) -> VolumeType:
        return cast("VolumeType", resize3d(volume, self.size, self.interpolation))

    def apply_to_mask3d(self, mask3d: VolumeType, **params: Any) -> VolumeType:
        return cast("VolumeType", resize3d(mask3d, self.size, self.mask_interpolation))

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        source_shape: tuple[int, int, int],
        **params: Any,
    ) -> np.ndarray:
        return f3d.keypoints_scale_3d(keypoints, source_shape, self.size)


class _BaseCropAndPad3DInitSchema(BaseTransformInitSchema):
    pad_if_needed: bool
    fill: tuple[float, ...] | float
    fill_mask: tuple[float, ...] | float
    pad_position: Literal["center", "random"]


class BasePad3D(Transform3D):
    """Base class for 3D padding transforms. Common logic for volume data, masks, keypoints; fill, fill_mask.
    Subclasses implement sample_parameters.

    This class serves as a foundation for all 3D transforms that perform padding operations
    on volumetric data. It provides common functionality for padding 3D volume data, masks,
    and processing 3D keypoints during padding operations.

    The class handles different types of padding values (scalar or per-channel) and
    provides separate fill values for volume data and masks.

    Args:
        fill (tuple[float, ...] | float): Value to fill the padded voxels for volume data.
            Can be a single value for all channels or a tuple of values per channel.
        fill_mask (tuple[float, ...] | float): Value to fill the padded voxels for 3D masks.
            Can be a single value for all channels or a tuple of values per channel.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        volume, mask3d, keypoints

    Note:
        This is a base class and not intended to be used directly. Use its derivatives
        like Pad3D or PadIfNeeded3D instead, or create a custom padding transform
        by inheriting from this class.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Example of a custom padding transform inheriting from BasePad3D
        >>> class CustomPad3D(A.BasePad3D):
        ...     def __init__(self, padding_size: tuple[int, int, int] = (5, 5, 5), *args, **kwargs):
        ...         super().__init__(*args, **kwargs)
        ...         self.padding_size = padding_size
        ...
        ...     def sample_parameters(self, params, data, targets, sampling):
        ...         # Create symmetric padding: same amount on all sides of each dimension
        ...         pad_d, pad_h, pad_w = self.padding_size
        ...         padding = (pad_d, pad_d, pad_h, pad_h, pad_w, pad_w)
        ...         return SampledParams(params={"padding": padding})
        >>>
        >>> # Prepare sample data
        >>> volume = np.random.randint(0, 256, (10, 100, 100), dtype=np.uint8)  # (D, H, W)
        >>> mask3d = np.random.randint(0, 2, (10, 100, 100), dtype=np.uint8)    # (D, H, W)
        >>> keypoints = np.array([[20, 30, 5], [60, 70, 8]], dtype=np.float32)  # (x, y, z)
        >>> keypoint_labels = [1, 2]  # Labels for each keypoint
        >>>
        >>> # Use the custom transform in a pipeline
        >>> transform = A.Compose([
        ...     CustomPad3D(
        ...         padding_size=(2, 10, 10),
        ...         fill=0,
        ...         fill_mask=1,
        ...         p=1.0
        ...     )
        ... ], keypoint_params=A.KeypointParams(coord_format='xyz', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     volume=volume,
        ...     mask3d=mask3d,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> transformed_volume = transformed["volume"]           # Shape: (14, 120, 120)
        >>> transformed_mask3d = transformed["mask3d"]           # Shape: (14, 120, 120)
        >>> transformed_keypoints = transformed["keypoints"]     # Keypoints shifted by padding offsets
        >>> transformed_keypoint_labels = transformed["keypoint_labels"]  # Labels remain unchanged

    """

    _targets = (Targets.VOLUME, Targets.MASK3D, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

    def __init__(
        self,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.fill = fill
        self.fill_mask = fill_mask

    def apply_to_volume(
        self,
        volume: VolumeType,
        padding: tuple[int, int, int, int, int, int],
        **params: Any,
    ) -> VolumeType:
        if padding == (0, 0, 0, 0, 0, 0):
            return volume
        return f3d.pad_3d_with_params(
            volume=volume,
            padding=padding,
            value=self.fill,
        )

    def apply_to_mask3d(
        self,
        mask3d: VolumeType,
        padding: tuple[int, int, int, int, int, int],
        **params: Any,
    ) -> VolumeType:
        if padding == (0, 0, 0, 0, 0, 0):
            return mask3d
        return f3d.pad_3d_with_params(
            volume=mask3d,
            padding=padding,
            value=cast("tuple[float, ...] | float", self.fill_mask),
        )

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        padding: tuple[int, int, int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        shift_vector = np.array([padding[4], padding[2], padding[0]])
        return fgeometric.shift_keypoints(keypoints, shift_vector)


class Pad3D(BasePad3D):
    """Add voxels around a 3D volume. Padding: int or per-side (depth, height, width); fill,
    fill_mask. For fixed-size batches or avoiding crop boundaries.

    Targets: volume, mask3d, keypoints

    Args:
        padding (int, tuple[int, int, int] or tuple[int, int, int, int, int, int]): Padding values. Can be:
            * int - pad all sides by this value
            * tuple[int, int, int] - symmetric padding (depth, height, width) where each value
              is applied to both sides of the corresponding dimension
            * tuple[int, int, int, int, int, int] - explicit padding per side in order:
              (depth_front, depth_back, height_top, height_bottom, width_left, width_right)

        fill (tuple[float, ...] | float): Padding value for image
        fill_mask (tuple[float, ...] | float): Padding value for mask
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Note:
        Input volume should be a numpy array with dimensions ordered as (z, y, x) or (depth, height, width),
        with optional channel dimension as the last axis.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Prepare sample data
        >>> volume = np.random.randint(0, 256, (10, 100, 100), dtype=np.uint8)  # (D, H, W)
        >>> mask3d = np.random.randint(0, 2, (10, 100, 100), dtype=np.uint8)    # (D, H, W)
        >>> keypoints = np.array([[20, 30, 5], [60, 70, 8]], dtype=np.float32)  # (x, y, z)
        >>> keypoint_labels = [1, 2]  # Labels for each keypoint
        >>>
        >>> # Create the transform with symmetric padding
        >>> transform = A.Compose([
        ...     A.Pad3D(
        ...         padding=(2, 5, 10),  # (depth, height, width) applied symmetrically
        ...         fill=0,
        ...         fill_mask=1,
        ...         p=1.0
        ...     )
        ... ], keypoint_params=A.KeypointParams(coord_format='xyz', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     volume=volume,
        ...     mask3d=mask3d,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> padded_volume = transformed["volume"]  # Shape: (14, 110, 120)
        >>> padded_mask3d = transformed["mask3d"]  # Shape: (14, 110, 120)
        >>> padded_keypoints = transformed["keypoints"]  # Keypoints shifted by padding
        >>> padded_keypoint_labels = transformed["keypoint_labels"]  # Labels remain unchanged

    """

    class InitSchema(BasePad3D.InitSchema):
        padding: int | tuple[int, int, int] | tuple[int, int, int, int, int, int]

        @field_validator("padding")
        @classmethod
        def validate_padding(
            cls,
            v: int | tuple[int, int, int] | tuple[int, int, int, int, int, int],
        ) -> int | tuple[int, int, int] | tuple[int, int, int, int, int, int]:
            """Validate padding: int or tuple of 3 or 6 non-negative ints. Raises if invalid or wrong length.
            For Pad3D InitSchema field validator.

            Args:
                cls (type): The class object
                v (int | tuple[int, int, int] | tuple[int, int, int, int, int, int]): The padding value to validate,
                    can be an integer or tuple of integers

            Returns:
                int | tuple[int, int, int] | tuple[int, int, int, int, int, int]: The validated padding value

            Raises:
                ValueError: If padding is negative or contains negative values

            """
            if isinstance(v, int) and v < 0:
                raise ValueError("Padding value must be non-negative")
            if isinstance(v, tuple) and not all(isinstance(i, int) and i >= 0 for i in v):
                raise ValueError("Padding tuple must contain non-negative integers")

            return v

    def __init__(
        self,
        padding: int | tuple[int, int, int] | tuple[int, int, int, int, int, int],
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        p: float = 1.0,
    ):
        super().__init__(fill=fill, fill_mask=fill_mask, p=p)
        self.padding = padding
        self.fill = fill
        self.fill_mask = fill_mask

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        """Get padding parameters from input data (volume shape) as a structured plan with a padding tuple
        (d_front, d_back, h_top, h_bottom, w_left, w_right). For Pad3D.

        Args:
            params (dict[str, Any]): Common execution parameters.
            data (dict[str, Any]): Preprocessed invocation data.
            targets (TargetSet): Active target metadata.
            sampling (SamplingContext): Call-local random streams and applied-configuration capture.

        Returns:
            SampledParams: Plan containing the padding parameter tuple in format:
                (depth_front, depth_back, height_top, height_bottom, width_left, width_right)

        """
        if isinstance(self.padding, int):
            pad_d = pad_h = pad_w = self.padding
            padding = (pad_d, pad_d, pad_h, pad_h, pad_w, pad_w)
        elif len(self.padding) == NUM_DIMENSIONS:
            pad_d, pad_h, pad_w = self.padding
            padding = (pad_d, pad_d, pad_h, pad_h, pad_w, pad_w)
        else:
            padding = self.padding

        return SampledParams(params={"padding": padding})


class PadIfNeeded3D(BasePad3D):
    """Pad 3D volume to min dimensions (min_zyx) and/or divisibility (pad_divisor_zyx). position,
    fill, fill_mask. At least one of min_zyx or pad_divisor_zyx required.

    Args:
        min_zyx (tuple[int, int, int] | None): Minimum desired size as (depth, height, width).
            Ensures volume dimensions are at least these values.
            If not specified, pad_divisor_zyx must be provided.
        pad_divisor_zyx (tuple[int, int, int] | None): If set, pads each dimension to make it
            divisible by corresponding value in format (depth_div, height_div, width_div).
            If not specified, min_zyx must be provided.
        position (Literal['center', 'random']): Position where the volume is to be placed after padding.
            Default is 'center'.
        fill (tuple[float, ...] | float): Value to fill the border voxels for volume. Default: 0
        fill_mask (tuple[float, ...] | float): Value to fill the border voxels for masks. Default: 0
        p (float): Probability of applying the transform. Default: 1.0

    Targets:
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Note:
        Input volume should be a numpy array with dimensions ordered as (z, y, x) or (depth, height, width),
        with optional channel dimension as the last axis.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Prepare sample data
        >>> volume = np.random.randint(0, 256, (10, 100, 100), dtype=np.uint8)  # (D, H, W)
        >>> mask3d = np.random.randint(0, 2, (10, 100, 100), dtype=np.uint8)    # (D, H, W)
        >>> keypoints = np.array([[20, 30, 5], [60, 70, 8]], dtype=np.float32)  # (x, y, z)
        >>> keypoint_labels = [1, 2]  # Labels for each keypoint
        >>>
        >>> # Create a transform with both min_zyx and pad_divisor_zyx
        >>> transform = A.Compose([
        ...     A.PadIfNeeded3D(
        ...         min_zyx=(16, 128, 128),        # Minimum size (depth, height, width)
        ...         pad_divisor_zyx=(8, 16, 16),   # Make dimensions divisible by these values
        ...         position="center",              # Center the volume in the padded space
        ...         fill=0,                         # Fill value for volume
        ...         fill_mask=1,                    # Fill value for mask
        ...         p=1.0
        ...     )
        ... ], keypoint_params=A.KeypointParams(coord_format='xyz', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     volume=volume,
        ...     mask3d=mask3d,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> padded_volume = transformed["volume"]           # Shape: (16, 128, 128)
        >>> padded_mask3d = transformed["mask3d"]           # Shape: (16, 128, 128)
        >>> padded_keypoints = transformed["keypoints"]     # Keypoints shifted by padding
        >>> padded_keypoint_labels = transformed["keypoint_labels"]  # Labels remain unchanged

    """

    class InitSchema(BasePad3D.InitSchema):
        min_zyx: Annotated[tuple[int, int, int] | None, AfterValidator(check_range_bounds(0, None))]
        pad_divisor_zyx: Annotated[tuple[int, int, int] | None, AfterValidator(check_range_bounds(1, None))]
        position: Literal["center", "random"]

        @model_validator(mode="after")
        def validate_params(self) -> Self:
            """Validate that exactly one of min_zyx or pad_divisor_zyx is provided. Raises ValueError
            if both None or both set. For PadIfNeeded3D InitSchema.

            Returns:
                Self: Self reference for method chaining

            Raises:
                ValueError: If both min_zyx and pad_divisor_zyx are None

            """
            if self.min_zyx is None and self.pad_divisor_zyx is None:
                msg = "At least one of min_zyx or pad_divisor_zyx must be set"
                raise ValueError(msg)
            return self

    def __init__(
        self,
        min_zyx: tuple[int, int, int] | None = None,
        pad_divisor_zyx: tuple[int, int, int] | None = None,
        position: Literal["center", "random"] = "center",
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        p: float = 1.0,
    ):
        super().__init__(fill=fill, fill_mask=fill_mask, p=p)
        self.min_zyx = min_zyx
        self.pad_divisor_zyx = pad_divisor_zyx
        self.position = position

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        """Calculate padding parameters from volume shape to meet min_zyx or pad_divisor_zyx.
        Returns a structured plan with a padding tuple. For PadIfNeeded3D.

        Args:
            params (dict[str, Any]): Common execution parameters.
            data (dict[str, Any]): Preprocessed invocation data.
            targets (TargetSet): Active target metadata.
            sampling (SamplingContext): Call-local random streams and applied-configuration capture.

        Returns:
            SampledParams: Plan containing calculated padding parameters

        """
        depth, height, width = _sampling_volume_shape(targets)
        sizes = (depth, height, width)

        paddings = [
            fgeometric.get_dimension_padding(
                current_size=size,
                min_size=self.min_zyx[i] if self.min_zyx else None,
                divisor=self.pad_divisor_zyx[i] if self.pad_divisor_zyx else None,
            )
            for i, size in enumerate(sizes)
        ]

        padding = f3d.adjust_padding_by_position3d(
            paddings=paddings,
            position=self.position,
            py_random=sampling.py_random,
        )

        return SampledParams(params={"padding": padding})


class BaseCropAndPad3D(Transform3D):
    """Base class for 3D transforms that crop and optionally pad. pad_if_needed, fill, fill_mask,
    pad_position. Subclasses implement sample_parameters.

    This class serves as a foundation for transforms that combine cropping and padding operations
    on 3D volumetric data. It provides functionality for calculating padding parameters,
    applying crop and pad operations to volume data, masks, and handling keypoint coordinate shifts.

    Args:
        pad_if_needed (bool): Whether to pad if the volume is smaller than target dimensions
        fill (tuple[float, ...] | float): Value to fill the padded voxels for volume
        fill_mask (tuple[float, ...] | float): Value to fill the padded voxels for mask
        pad_position (Literal['center', 'random']): How to distribute padding when needed
            "center" - equal amount on both sides, "random" - random distribution
        p (float): Probability of applying the transform. Default: 1.0

    Targets:
        volume, mask3d, keypoints

    Note:
        This is a base class and not intended to be used directly. Use its derivatives
        like CenterCrop3D or RandomCrop3D instead, or create a custom transform
        by inheriting from this class.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Example of a custom crop transform inheriting from BaseCropAndPad3D
        >>> class CustomFixedCrop3D(A.BaseCropAndPad3D):
        ...     def __init__(self, crop_size: tuple[int, int, int] = (8, 64, 64), *args, **kwargs):
        ...         super().__init__(
        ...             pad_if_needed=True,
        ...             fill=0,
        ...             fill_mask=0,
        ...             pad_position="center",
        ...             *args,
        ...             **kwargs
        ...         )
        ...         self.crop_size = crop_size
        ...
        ...     def sample_parameters(self, params, data, targets, sampling):
        ...         # Get the validated spatial frame
        ...         z, h, w = targets.require_aligned_spatial_shape(3)
        ...         target_z, target_h, target_w = self.crop_size
        ...
        ...         # Check if padding is needed and calculate parameters
        ...         pad_params = self._get_pad_params(
        ...             image_shape=(z, h, w),
        ...             target_shape=self.crop_size,
        ...             sampling=sampling,
        ...         )
        ...
        ...         # Update dimensions if padding is applied
        ...         if pad_params is not None:
        ...             z = z + pad_params["pad_front"] + pad_params["pad_back"]
        ...             h = h + pad_params["pad_top"] + pad_params["pad_bottom"]
        ...             w = w + pad_params["pad_left"] + pad_params["pad_right"]
        ...
        ...         # Calculate fixed crop coordinates - always start at position (0,0,0)
        ...         crop_coords = (0, target_z, 0, target_h, 0, target_w)
        ...
        ...         from albumentations.core.transform_params import SampledParams
        ...
        ...         return SampledParams(params={
        ...             "crop_coords": crop_coords,
        ...             "pad_params": pad_params,
        ...         })
        >>>
        >>> # Prepare sample data
        >>> volume = np.random.randint(0, 256, (10, 100, 100), dtype=np.uint8)  # (D, H, W)
        >>> mask3d = np.random.randint(0, 2, (10, 100, 100), dtype=np.uint8)    # (D, H, W)
        >>> keypoints = np.array([[20, 30, 5], [60, 70, 8]], dtype=np.float32)  # (x, y, z)
        >>> keypoint_labels = [1, 2]  # Labels for each keypoint
        >>>
        >>> # Use the custom transform in a pipeline
        >>> transform = A.Compose([
        ...     CustomFixedCrop3D(
        ...         crop_size=(8, 64, 64),  # Crop first 8x64x64 voxels (with padding if needed)
        ...         p=1.0
        ...     )
        ... ], keypoint_params=A.KeypointParams(coord_format='xyz', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     volume=volume,
        ...     mask3d=mask3d,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> cropped_volume = transformed["volume"]           # Shape: (8, 64, 64)
        >>> cropped_mask3d = transformed["mask3d"]           # Shape: (8, 64, 64)
        >>> cropped_keypoints = transformed["keypoints"]     # Keypoints shifted relative to crop
        >>> cropped_keypoint_labels = transformed["keypoint_labels"]  # Labels remain unchanged

    """

    _targets = (Targets.VOLUME, Targets.MASK3D, Targets.KEYPOINTS)

    InitSchema: ClassVar[type[BaseTransformInitSchema]] = _BaseCropAndPad3DInitSchema

    def __init__(
        self,
        pad_if_needed: bool,
        fill: tuple[float, ...] | float,
        fill_mask: tuple[float, ...] | float,
        pad_position: Literal["center", "random"],
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.fill_mask = fill_mask
        self.pad_position = pad_position

    def _random_pad(self, pad: int, sampling: SamplingContext) -> tuple[int, int]:
        """Generate random (front, back) padding that sum to pad. Used when pad_position is random.
        Returns tuple (front, back) for one dimension.

        Args:
            pad (int): Total padding value to distribute
            sampling (SamplingContext): Call-local random streams for selecting the split.

        Returns:
            tuple[int, int]: Random padding values (front, back)

        """
        if pad > 0:
            pad_start = sampling.py_random.randint(0, pad)
            pad_end = pad - pad_start
        else:
            pad_start = pad_end = 0
        return pad_start, pad_end

    def _center_pad(self, pad: int) -> tuple[int, int]:
        """Generate centered (front, back) padding that sum to pad. Used when pad_position is center.
        Returns tuple (front, back) for one dimension. For BaseCropAndPad3D.

        Args:
            pad (int): Total padding value to distribute

        Returns:
            tuple[int, int]: Centered padding values (front, back)

        """
        pad_start = pad // 2
        pad_end = pad - pad_start
        return pad_start, pad_end

    def _get_pad_params(
        self,
        image_shape: tuple[int, int, int],
        target_shape: tuple[int, int, int],
        sampling: SamplingContext,
    ) -> dict[str, int] | None:
        """Calculate padding to reach target shape from image_shape. Returns dict or None. For
        BaseCropAndPad3D when pad_if_needed True.

        Args:
            image_shape (tuple[int, int, int]): Current shape (depth, height, width)
            target_shape (tuple[int, int, int]): Target shape (depth, height, width)
            sampling (SamplingContext): Call-local random streams for random padding.

        Returns:
            dict[str, int] | None: Padding parameters or None if no padding needed

        """
        if not self.pad_if_needed:
            return None

        z, h, w = image_shape
        target_z, target_h, target_w = target_shape

        # Calculate total padding needed for each dimension
        z_pad = max(0, target_z - z)
        h_pad = max(0, target_h - h)
        w_pad = max(0, target_w - w)

        if z_pad == 0 and h_pad == 0 and w_pad == 0:
            return None

        # For center padding, split equally
        if self.pad_position == "center":
            z_front, z_back = self._center_pad(z_pad)
            h_top, h_bottom = self._center_pad(h_pad)
            w_left, w_right = self._center_pad(w_pad)
        # For random padding, randomly distribute the padding
        else:  # random
            z_front, z_back = self._random_pad(z_pad, sampling)
            h_top, h_bottom = self._random_pad(h_pad, sampling)
            w_left, w_right = self._random_pad(w_pad, sampling)

        return {
            "pad_front": z_front,
            "pad_back": z_back,
            "pad_top": h_top,
            "pad_bottom": h_bottom,
            "pad_left": w_left,
            "pad_right": w_right,
        }

    def apply_to_volume(
        self,
        volume: VolumeType,
        crop_coords: tuple[int, int, int, int, int, int],
        pad_params: dict[str, int] | None,
        **params: Any,
    ) -> VolumeType:
        # First crop
        cropped = f3d.crop3d(volume, crop_coords)

        # Then pad if needed
        if pad_params is not None:
            padding = (
                pad_params["pad_front"],
                pad_params["pad_back"],
                pad_params["pad_top"],
                pad_params["pad_bottom"],
                pad_params["pad_left"],
                pad_params["pad_right"],
            )
            return f3d.pad_3d_with_params(
                cropped,
                padding=padding,
                value=self.fill,
            )

        return cropped

    def apply_to_mask3d(
        self,
        mask3d: VolumeType,
        crop_coords: tuple[int, int, int, int, int, int],
        pad_params: dict[str, int] | None,
        **params: Any,
    ) -> VolumeType:
        # First crop
        cropped = f3d.crop3d(mask3d, crop_coords)

        # Then pad if needed
        if pad_params is not None:
            padding = (
                pad_params["pad_front"],
                pad_params["pad_back"],
                pad_params["pad_top"],
                pad_params["pad_bottom"],
                pad_params["pad_left"],
                pad_params["pad_right"],
            )
            return f3d.pad_3d_with_params(
                cropped,
                padding=padding,
                value=cast("tuple[float, ...] | float", self.fill_mask),
            )

        return cropped

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        crop_coords: tuple[int, int, int, int, int, int],
        pad_params: dict[str, int] | None,
        **params: Any,
    ) -> np.ndarray:
        # Extract crop start coordinates (z1,y1,x1)
        crop_z1, _, crop_y1, _, crop_x1, _ = crop_coords

        # Initialize shift vector with negative crop coordinates
        shift = np.array(
            [
                -crop_x1,  # X shift
                -crop_y1,  # Y shift
                -crop_z1,  # Z shift
            ],
        )

        # Add padding shift if needed
        if pad_params is not None:
            shift += np.array(
                [
                    pad_params["pad_left"],  # X shift
                    pad_params["pad_top"],  # Y shift
                    pad_params["pad_front"],  # Z shift
                ],
            )

        # Apply combined shift
        return fgeometric.shift_keypoints(keypoints, shift)


class CenterCrop3D(BaseCropAndPad3D):
    """Take the center sub-volume to fixed (depth, height, width). pad_if_needed fills when smaller;
    fill, fill_mask. For fixed-size 3D inputs (e.g. CT, MRI).

    Targets: volume, mask3d, keypoints

    Args:
        size (tuple[int, int, int]): Desired output size of the crop in format (depth, height, width)
        pad_if_needed (bool): Whether to pad if the volume is smaller than desired crop size. Default: False
        fill (tuple[float, float] | float): Padding value for image if pad_if_needed is True. Default: 0
        fill_mask (tuple[float, float] | float): Padding value for mask if pad_if_needed is True. Default: 0
        p (float): probability of applying the transform. Default: 1.0

    Targets:
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Note:
        If you want to perform cropping only in the XY plane while preserving all slices along
        the Z axis, consider using CenterCrop instead. CenterCrop will apply the same XY crop
        to each slice independently, maintaining the full depth of the volume.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Prepare sample data
        >>> volume = np.random.randint(0, 256, (20, 200, 200), dtype=np.uint8)  # (D, H, W)
        >>> mask3d = np.random.randint(0, 2, (20, 200, 200), dtype=np.uint8)    # (D, H, W)
        >>> keypoints = np.array([[100, 100, 10], [150, 150, 15]], dtype=np.float32)  # (x, y, z)
        >>> keypoint_labels = [1, 2]  # Labels for each keypoint
        >>>
        >>> # Create the transform - crop to 16x128x128 from center
        >>> transform = A.Compose([
        ...     A.CenterCrop3D(
        ...         size=(16, 128, 128),        # Output size (depth, height, width)
        ...         pad_if_needed=True,         # Pad if input is smaller than crop size
        ...         fill=0,                     # Fill value for volume padding
        ...         fill_mask=1,                # Fill value for mask padding
        ...         p=1.0
        ...     )
        ... ], keypoint_params=A.KeypointParams(coord_format='xyz', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     volume=volume,
        ...     mask3d=mask3d,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> cropped_volume = transformed["volume"]           # Shape: (16, 128, 128)
        >>> cropped_mask3d = transformed["mask3d"]           # Shape: (16, 128, 128)
        >>> cropped_keypoints = transformed["keypoints"]     # Keypoints shifted relative to center crop
        >>> cropped_keypoint_labels = transformed["keypoint_labels"]  # Labels remain unchanged
        >>>
        >>> # Example with a small volume that requires padding
        >>> small_volume = np.random.randint(0, 256, (10, 100, 100), dtype=np.uint8)
        >>> small_transform = A.Compose([
        ...     A.CenterCrop3D(
        ...         size=(16, 128, 128),
        ...         pad_if_needed=True,   # Will pad since the input is smaller
        ...         fill=0,
        ...         p=1.0
        ...     )
        ... ])
        >>> small_result = small_transform(volume=small_volume)
        >>> padded_and_cropped = small_result["volume"]  # Shape: (16, 128, 128), padded to size

    """

    class InitSchema(BaseTransformInitSchema):
        size: Annotated[tuple[int, int, int], AfterValidator(check_range_bounds(1, None))]
        pad_if_needed: bool
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

    def __init__(
        self,
        size: tuple[int, int, int],
        pad_if_needed: bool = False,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        p: float = 1.0,
    ):
        super().__init__(
            pad_if_needed=pad_if_needed,
            fill=fill,
            fill_mask=fill_mask,
            pad_position="center",  # Center crop always uses center padding
            p=p,
        )
        self.size = size

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        """Calculate center crop coordinates from volume shape and self.size as a structured plan with
        crop_coords (z_min, z_max, y_min, y_max, x_min, x_max). For CenterCrop3D.

        Args:
            params (dict[str, Any]): Common execution parameters.
            data (dict[str, Any]): Preprocessed invocation data.
            targets (TargetSet): Active target metadata.
            sampling (SamplingContext): Call-local random streams and applied-configuration capture.

        Returns:
            SampledParams: Plan containing crop coordinates and optional padding parameters

        """
        z, h, w = _sampling_volume_shape(targets)
        target_z, target_h, target_w = self.size

        # Get padding params if needed
        pad_params = self._get_pad_params(
            image_shape=(z, h, w),
            target_shape=self.size,
            sampling=sampling,
        )

        # Update dimensions if padding is applied
        if pad_params is not None:
            z = z + pad_params["pad_front"] + pad_params["pad_back"]
            h = h + pad_params["pad_top"] + pad_params["pad_bottom"]
            w = w + pad_params["pad_left"] + pad_params["pad_right"]

        # Validate dimensions after padding
        if z < target_z or h < target_h or w < target_w:
            msg = (
                f"Crop size {self.size} is larger than padded image size ({z}, {h}, {w}). "
                f"This should not happen - please report this as a bug."
            )
            raise ValueError(msg)

        # For CenterCrop3D:
        z_start = (z - target_z) // 2
        h_start = (h - target_h) // 2
        w_start = (w - target_w) // 2

        crop_coords = (
            z_start,
            z_start + target_z,
            h_start,
            h_start + target_h,
            w_start,
            w_start + target_w,
        )

        return SampledParams(
            params={
                "crop_coords": crop_coords,
                "pad_params": pad_params,
            }
        )


class RandomCrop3D(BaseCropAndPad3D):
    """Extract a random 3D sub-volume of given (depth, height, width). pad_if_needed when smaller;
    fill, fill_mask. For spatial augmentation of volumetric data.

    Targets: volume, mask3d, keypoints

    Args:
        size (tuple[int, int, int]): Desired output size of the crop in format (depth, height, width)
        pad_if_needed (bool): Whether to pad if the volume is smaller than desired crop size. Default: False
        fill (tuple[float, float] | float): Padding value for image if pad_if_needed is True. Default: 0
        fill_mask (tuple[float, float] | float): Padding value for mask if pad_if_needed is True. Default: 0
        p (float): probability of applying the transform. Default: 1.0

    Targets:
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Note:
        If you want to perform random cropping only in the XY plane while preserving all slices along
        the Z axis, consider using RandomCrop instead. RandomCrop will apply the same XY crop
        to each slice independently, maintaining the full depth of the volume.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Prepare sample data
        >>> volume = np.random.randint(0, 256, (20, 200, 200), dtype=np.uint8)  # (D, H, W)
        >>> mask3d = np.random.randint(0, 2, (20, 200, 200), dtype=np.uint8)    # (D, H, W)
        >>> keypoints = np.array([[100, 100, 10], [150, 150, 15]], dtype=np.float32)  # (x, y, z)
        >>> keypoint_labels = [1, 2]  # Labels for each keypoint
        >>>
        >>> # Create the transform with random crop and padding if needed
        >>> transform = A.Compose([
        ...     A.RandomCrop3D(
        ...         size=(16, 128, 128),        # Output size (depth, height, width)
        ...         pad_if_needed=True,         # Pad if input is smaller than crop size
        ...         fill=0,                     # Fill value for volume padding
        ...         fill_mask=1,                # Fill value for mask padding
        ...         p=1.0
        ...     )
        ... ], keypoint_params=A.KeypointParams(coord_format='xyz', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     volume=volume,
        ...     mask3d=mask3d,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> cropped_volume = transformed["volume"]           # Shape: (16, 128, 128)
        >>> cropped_mask3d = transformed["mask3d"]           # Shape: (16, 128, 128)
        >>> cropped_keypoints = transformed["keypoints"]     # Keypoints shifted relative to random crop
        >>> cropped_keypoint_labels = transformed["keypoint_labels"]  # Labels remain unchanged

    """

    class InitSchema(BaseTransformInitSchema):
        size: Annotated[tuple[int, int, int], AfterValidator(check_range_bounds(1, None))]
        pad_if_needed: bool
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

    def __init__(
        self,
        size: tuple[int, int, int],
        pad_if_needed: bool = False,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        p: float = 1.0,
    ):
        super().__init__(
            pad_if_needed=pad_if_needed,
            fill=fill,
            fill_mask=fill_mask,
            pad_position="random",  # Random crop uses random padding position
            p=p,
        )
        self.size = size

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        """Calculate random crop coordinates from volume shape and self.size as a structured plan with
        crop_coords (z_min, z_max, y_min, y_max, x_min, x_max). For RandomCrop3D.

        Args:
            params (dict[str, Any]): Common execution parameters.
            data (dict[str, Any]): Preprocessed invocation data.
            targets (TargetSet): Active target metadata.
            sampling (SamplingContext): Call-local random streams and applied-configuration capture.

        Returns:
            SampledParams: Plan containing randomly generated crop coordinates and optional padding parameters

        """
        z, h, w = _sampling_volume_shape(targets)
        target_z, target_h, target_w = self.size

        # Get padding params if needed
        pad_params = self._get_pad_params(
            image_shape=(z, h, w),
            target_shape=self.size,
            sampling=sampling,
        )

        # Update dimensions if padding is applied
        if pad_params is not None:
            z = z + pad_params["pad_front"] + pad_params["pad_back"]
            h = h + pad_params["pad_top"] + pad_params["pad_bottom"]
            w = w + pad_params["pad_left"] + pad_params["pad_right"]

        # Calculate random crop coordinates
        z_start = sampling.py_random.randint(0, max(0, z - target_z))
        h_start = sampling.py_random.randint(0, max(0, h - target_h))
        w_start = sampling.py_random.randint(0, max(0, w - target_w))

        crop_coords = (
            z_start,
            z_start + target_z,
            h_start,
            h_start + target_h,
            w_start,
            w_start + target_w,
        )

        return SampledParams(
            params={
                "crop_coords": crop_coords,
                "pad_params": pad_params,
            }
        )


class CoarseDropout3D(Transform3D):
    """Randomly drop cuboid regions from a 3D volume (and optionally mask) to simulate occlusion.
    Hole size/count configurable.

    Args:
        num_holes_range (tuple[int, int]): Range (min, max) for the number of cuboid
            regions to drop out. Default: (1, 1)
        hole_depth_range (tuple[float, float]): Range (min, max) for the depth
            of dropout regions as a fraction of the volume depth (between 0 and 1). Default: (0.1, 0.2)
        hole_height_range (tuple[float, float]): Range (min, max) for the height
            of dropout regions as a fraction of the volume height (between 0 and 1). Default: (0.1, 0.2)
        hole_width_range (tuple[float, float]): Range (min, max) for the width
            of dropout regions as a fraction of the volume width (between 0 and 1). Default: (0.1, 0.2)
        fill (tuple[float, float] | float): Value for the dropped voxels. Can be:
            - int or float: all channels are filled with this value
            - tuple: tuple of values for each channel
            Default: 0
        fill_mask (tuple[float, float] | float | None): Fill value for dropout regions in the 3D mask.
            If None, mask regions corresponding to volume dropouts are unchanged. Default: None
        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Note:
        - The actual number and size of dropout regions are randomly chosen within the specified ranges.
        - All values in hole_depth_range, hole_height_range and hole_width_range must be between 0 and 1.
        - If you want to apply dropout only in the XY plane while preserving the full depth dimension,
          consider using CoarseDropout instead. CoarseDropout will apply the same rectangular dropout
          to each slice independently, effectively creating cylindrical dropout regions that extend
          through the entire depth of the volume.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> volume = np.random.randint(0, 256, (10, 100, 100), dtype=np.uint8)  # (D, H, W)
        >>> mask3d = np.random.randint(0, 2, (10, 100, 100), dtype=np.uint8)    # (D, H, W)
        >>> aug = A.CoarseDropout3D(
        ...     num_holes_range=(3, 6),
        ...     hole_depth_range=(0.1, 0.2),
        ...     hole_height_range=(0.1, 0.2),
        ...     hole_width_range=(0.1, 0.2),
        ...     fill=0,
        ...     p=1.0
        ... )
        >>> transformed = aug(volume=volume, mask3d=mask3d)
        >>> transformed_volume, transformed_mask3d = transformed["volume"], transformed["mask3d"]

    """

    _targets = (Targets.VOLUME, Targets.MASK3D, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        num_holes_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        hole_depth_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        hole_height_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        hole_width_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float | None

        @staticmethod
        def validate_range(range_value: tuple[float, float], range_name: str) -> None:
            """Validate that range values are in [0, 1] and non-decreasing. Raises ValueError if not.
            For CoarseDropout3D InitSchema depth/height/width range fields.

            Args:
                range_value (tuple[float, float]): Tuple of (min, max) values to check
                range_name (str): Name of the range for error reporting

            Raises:
                ValueError: If range values are invalid

            """
            if not 0 <= range_value[0] <= range_value[1] <= 1:
                raise ValueError(
                    f"All values in {range_name} should be in [0, 1] range and first value "
                    f"should be less or equal than the second value. Got: {range_value}",
                )

        @model_validator(mode="after")
        def _check_ranges(self) -> Self:
            self.validate_range(self.hole_depth_range, "hole_depth_range")
            self.validate_range(self.hole_height_range, "hole_height_range")
            self.validate_range(self.hole_width_range, "hole_width_range")
            return self

    def __init__(
        self,
        num_holes_range: tuple[int, int] = (1, 1),
        hole_depth_range: tuple[float, float] = (0.1, 0.2),
        hole_height_range: tuple[float, float] = (0.1, 0.2),
        hole_width_range: tuple[float, float] = (0.1, 0.2),
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.num_holes_range = num_holes_range
        self.hole_depth_range = hole_depth_range
        self.hole_height_range = hole_height_range
        self.hole_width_range = hole_width_range
        self.fill = fill
        self.fill_mask = fill_mask

    def calculate_hole_dimensions(
        self,
        volume_shape: tuple[int, int, int],
        depth_range: tuple[float, float],
        height_range: tuple[float, float],
        width_range: tuple[float, float],
        size: int,
        sampling: SamplingContext,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate 3D dropout hole dimensions from volume shape and fraction ranges. Returns
        (depths, heights, widths) arrays. For CoarseDropout3D.

        Args:
            volume_shape (tuple[int, int, int]): Shape of the volume (depth, height, width)
            depth_range (tuple[float, float]): Range for hole depth as fraction of volume depth
            height_range (tuple[float, float]): Range for hole height as fraction of volume height
            width_range (tuple[float, float]): Range for hole width as fraction of volume width
            size (int): Number of holes to generate
            sampling (SamplingContext): Call-local random streams for hole dimensions.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of hole dimensions (depths, heights, widths)

        """
        depth, height, width = volume_shape

        hole_depths = np.maximum(1, np.ceil(depth * sampling.random_generator.uniform(*depth_range, size=size))).astype(
            int,
        )
        hole_heights = np.maximum(
            1, np.ceil(height * sampling.random_generator.uniform(*height_range, size=size))
        ).astype(
            int,
        )
        hole_widths = np.maximum(1, np.ceil(width * sampling.random_generator.uniform(*width_range, size=size))).astype(
            int,
        )

        return hole_depths, hole_heights, hole_widths

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        """Generate hole parameters for CoarseDropout3D from volume shape as a structured plan with holes
        (n, 6) and params. Uses depth/height/width ranges, num_holes.

        Args:
            params (dict[str, Any]): Common execution parameters.
            data (dict[str, Any]): Preprocessed invocation data.
            targets (TargetSet): Active target metadata.
            sampling (SamplingContext): Call-local random streams and applied-configuration capture.

        Returns:
            SampledParams: Plan containing generated hole parameters for dropout

        """
        volume_shape = _sampling_volume_shape(targets)

        num_holes = sampling.py_random.randint(*self.num_holes_range)

        hole_depths, hole_heights, hole_widths = self.calculate_hole_dimensions(
            volume_shape,
            self.hole_depth_range,
            self.hole_height_range,
            self.hole_width_range,
            size=num_holes,
            sampling=sampling,
        )

        depth, height, width = volume_shape

        z_min = sampling.random_generator.integers(0, depth - hole_depths + 1, size=num_holes)
        y_min = sampling.random_generator.integers(0, height - hole_heights + 1, size=num_holes)
        x_min = sampling.random_generator.integers(0, width - hole_widths + 1, size=num_holes)
        z_max = z_min + hole_depths
        y_max = y_min + hole_heights
        x_max = x_min + hole_widths

        holes = np.stack([z_min, y_min, x_min, z_max, y_max, x_max], axis=-1)

        sampling.applied_overrides.update(
            {
                "num_holes_range": num_holes,
                "hole_depth_range": (float(hole_depths.min() / depth), float(hole_depths.max() / depth)),
                "hole_height_range": (float(hole_heights.min() / height), float(hole_heights.max() / height)),
                "hole_width_range": (float(hole_widths.min() / width), float(hole_widths.max() / width)),
            },
        )

        return SampledParams(params={"holes": holes})

    def apply_to_volume(self, volume: VolumeType, holes: np.ndarray, **params: Any) -> VolumeType:
        if holes.size == 0:
            return volume

        return f3d.cutout3d(volume, holes, self.fill)

    def apply_to_mask(self, mask: VolumeType, holes: np.ndarray, **params: Any) -> VolumeType:
        if self.fill_mask is None or holes.size == 0:
            return mask

        return f3d.cutout3d(mask, holes, self.fill_mask)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        holes: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        if holes.size == 0:
            return keypoints
        processor = cast("KeypointsProcessor", self.get_processor("keypoints"))

        if processor is None or not processor.params.remove_invisible:
            return keypoints
        return f3d.filter_keypoints_in_holes3d(keypoints, holes)


class Flip3D(Transform3D):
    """Reflect a volume independently across depth, height, and width voxel-index axes while retaining its shape and
    channel layout.

    In random mode, each allowed axis is independently reflected, including the empty subset (identity). This samples
    the full reflection group uniformly. Set `flip_axes` to a fixed subset, including `()`, for reversible test-time
    augmentation; reflections are self-inverse. This is a voxel-index reflection only: it does not update affine
    metadata or perform a physical-space reorientation.

    Args:
        axes (tuple[int, ...]): Non-empty spatial axes that random mode may flip, in `(depth, height, width)` order.
            Default: `(0, 1, 2)`.
        flip_axes (tuple[int, ...] | None): Fixed subset of `axes` to reflect for deterministic test-time augmentation.
            Use `()` for identity. Default: None.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Note:
        - A realized reflection across an odd number of axes emits the `Flip3D` label-mapping event. Its semantic-mask
          mapping applies only to `mask3d`; keypoint label mappings rename label values without changing coordinate-row
          order.
        - A reflection across an even number of axes, including identity, preserves orientation and does not emit this
          event. Without an explicit mapping, labels stay unchanged.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> volume = np.arange(2 * 3 * 5, dtype=np.uint8).reshape(2, 3, 5, 1)
        >>> transform = A.Flip3D(flip_axes=(0, 2), p=1.0)
        >>> result = transform(volume=volume)
        >>> result["volume"].shape
        (2, 3, 5, 1)

    See Also:
        - `Affine3D`: Use positive scales and continuous rotations or translations without sampling reflections.

    """

    _targets = (Targets.VOLUME, Targets.MASK3D, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        axes: tuple[AxisIndex3D, ...]
        flip_axes: tuple[AxisIndex3D, ...] | None

        @model_validator(mode="after")
        def _validate_flip_config(self) -> Self:
            if not self.axes:
                raise ValueError("axes must contain at least one spatial axis")
            if len(self.axes) != len(set(self.axes)):
                raise ValueError("axes must not contain duplicate spatial axes")
            if self.flip_axes is not None:
                if len(self.flip_axes) != len(set(self.flip_axes)):
                    raise ValueError("flip_axes must not contain duplicate spatial axes")
                if not set(self.flip_axes).issubset(self.axes):
                    raise ValueError("flip_axes must be a subset of axes")
            return self

    def __init__(
        self,
        axes: tuple[AxisIndex3D, ...] = DEFAULT_FLIP_AXES,
        flip_axes: tuple[AxisIndex3D, ...] | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.axes = axes
        self.flip_axes = flip_axes

    def get_transform_init_args(self) -> dict[str, Any]:
        """Return constructor arguments while preserving the empty axis tuple that selects deterministic identity
        instead of random sampling.
        """
        args = super().get_transform_init_args()
        if self.flip_axes == ():
            args["flip_axes"] = self.flip_axes
        return args

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        flip_axes = self.flip_axes
        if flip_axes is None:
            flip_axes = tuple(axis for axis in self.axes if sampling.py_random.random() < 0.5)

        sampling.applied_overrides["flip_axes"] = flip_axes
        return SampledParams(params={"flip_axes": flip_axes, "volume_shape": _sampling_volume_shape(targets)})

    def _get_label_transform_name(self, **params: Any) -> str | None:
        """Return the Flip3D semantic-mapping event for a realized orientation-reversing reflection, allowing mask3d
        and keypoint labels to follow it.

        Each voxel-index reflection changes orientation parity. An odd number of reflected axes has determinant
        negative and therefore applies the configured semantic mapping; an even number, including identity, preserves
        orientation.
        """
        flip_axes = params.get("flip_axes", ())
        return "Flip3D" if len(flip_axes) % 2 else None

    def apply_to_volume(
        self,
        volume: VolumeType,
        flip_axes: tuple[AxisIndex3D, ...],
        **params: Any,
    ) -> VolumeType:
        if not flip_axes:
            return volume
        flipped: np.ndarray = np.flip(volume, axis=flip_axes)
        return cast("VolumeType", flipped)

    def apply_to_mask3d(
        self,
        mask3d: VolumeType,
        flip_axes: tuple[AxisIndex3D, ...],
        **params: Any,
    ) -> VolumeType:
        if not flip_axes:
            return mask3d
        flipped: np.ndarray = np.flip(mask3d, axis=flip_axes)
        return cast("VolumeType", flipped)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        flip_axes: tuple[AxisIndex3D, ...],
        volume_shape: tuple[int, int, int],
        **params: Any,
    ) -> np.ndarray:
        if not flip_axes:
            return keypoints
        return f3d.keypoints_flip_3d(keypoints, flip_axes, volume_shape)

    def inverse(self) -> Self:
        """Return a fixed self-inverse reflection for test-time augmentation when this transform has a deterministic
        axis subset configured.

        Raises:
            ValueError: If this transform is configured in random mode.

        """
        if self.flip_axes is None:
            raise ValueError("Cannot invert Flip3D in random mode. Set flip_axes for TTA.")
        return type(self)(axes=self.axes, flip_axes=self.flip_axes, p=1.0)


class CubicSymmetry(Transform3D):
    """Randomly reorient a 3D volume with one of 48 exact cubic symmetries, using only axis permutations and
    reflections for interpolation-free augmentation.

    This transform is intended for augmentation. Use a transform with explicit deterministic parameters for TTA.

    This transform is a 3D extension of D4. While D4 handles the 8 symmetries
    of a square (4 rotations x 2 reflections), CubicSymmetry handles all 48 symmetries of a cube.
    Like D4, this transform does not create any interpolation artifacts as it only remaps voxels
    from one position to another without any interpolation.

    The 48 transformations consist of:
    - 24 rotations (orientation-preserving):
        * 4 rotations around each face diagonal (6 face diagonals x 4 rotations = 24)
    - 24 rotoreflections (orientation-reversing):
        * Reflection through a plane followed by any of the 24 rotations

    For a cube, these transformations preserve:
    - All face centers (6)
    - All vertex positions (8)
    - All edge centers (12)

    works with 3D volume data and masks of the shape (D, H, W) or (D, H, W, C)

    Args:
        p (float): Probability of applying the transform. Default: 1.0

    Targets:
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Note:
        - This transform is particularly useful for data augmentation in 3D medical imaging,
          crystallography, and voxel-based 3D modeling where the object's orientation
          is arbitrary.
        - All transformations preserve the object's chirality (handedness) when using
          pure rotations (indices 0-23) and invert it when using rotoreflections
          (indices 24-47).
        - A realized rotoreflection emits the `CubicSymmetry` label-mapping event. Configured semantic-mask mappings
          remap `mask3d` labels and aliases; configured keypoint mappings remap label fields without moving their
          transformed coordinate rows. Rotations do not emit this event.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> volume = np.random.randint(0, 256, (10, 100, 100), dtype=np.uint8)  # (D, H, W)
        >>> mask3d = np.random.randint(0, 2, (10, 100, 100), dtype=np.uint8)    # (D, H, W)
        >>> transform = A.CubicSymmetry(p=1.0)
        >>> transformed = transform(volume=volume, mask3d=mask3d)
        >>> transformed_volume = transformed["volume"]
        >>> transformed_mask3d = transformed["mask3d"]

    See Also:
        - D4: The 2D version that handles the 8 symmetries of a square

    """

    _targets = (Targets.VOLUME, Targets.MASK3D, Targets.KEYPOINTS)

    def __init__(
        self,
        p: float = 1.0,
    ):
        super().__init__(p=p)

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        """Generate cubic symmetry parameters: random index 0-47. Returns a structured plan with index for
        transform_cube and transform_cube_keypoints. For CubicSymmetry.

        Args:
            params (dict[str, Any]): Common execution parameters.
            data (dict[str, Any]): Preprocessed invocation data.
            targets (TargetSet): Active target metadata.
            sampling (SamplingContext): Call-local random streams and applied-configuration capture.

        Returns:
            SampledParams: Plan containing the randomly selected transformation index

        """
        # Randomly select one of 48 possible transformations
        volume_shape = _sampling_volume_shape(targets)
        return SampledParams(params={"index": sampling.py_random.randint(0, 47), "volume_shape": volume_shape})

    def _get_label_transform_name(self, **params: Any) -> str | None:
        """Return the CubicSymmetry mapping event for a rotoreflection, while pure cube rotations preserve class
        meanings and must not alter semantic labels.
        """
        index = params.get("index")
        return "CubicSymmetry" if isinstance(index, int) and index >= 24 else None

    def apply_to_volume(self, volume: VolumeType, index: int, **params: Any) -> VolumeType:
        return f3d.transform_cube(volume, index)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        index: int,
        volume_shape: tuple[int, int, int],
        **params: Any,
    ) -> np.ndarray:
        return f3d.transform_cube_keypoints(keypoints, index, volume_shape=volume_shape)


class RandomRotate90_3D(Transform3D):  # noqa: N801 - Public API name specified by issue #388.
    """Rotate a volume by a random 90-degree multiple across one spatial axis pair, rotating mask3d and XYZ keypoints
    without reflections.

    Quarter turns swap the selected depth, height, or width lengths while preserving the channel axis.

    Axis indices always use `(depth, height, width)` order: `(0, 1)` rotates depth with height,
    `(0, 2)` rotates depth with width, and `(1, 2)` rotates height with width. A 90-degree or
    270-degree turn swaps the two selected lengths, so non-cubic input can change shape.
    A 180-degree turn and the identity preserve the original shape. The transform does not reflect data.

    Set both `axis_pair` and `group_element` for deterministic test-time augmentation. `inverse()`
    then returns the rotation that restores the original voxel and keypoint coordinates.

    Args:
        axis_pairs (tuple[tuple[int, int], ...]): Non-empty set of axis pairs sampled in random mode.
            Each pair must list two distinct axes in ascending `(depth, height, width)` order.
            Default: `((0, 1), (0, 2), (1, 2))`.
        axis_pair (tuple[int, int] | None): Fixed spatial axis pair. Set with `group_element` for
            deterministic TTA. Default: None.
        group_element (C4GroupElement | None): If set, always apply this C4 group element:
            `"e"`=identity, `"r90"`=90°, `"r180"`=180°, `"r270"`=270° counterclockwise.
            Use for TTA. Default: None (random choice).
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> volume = np.arange(2 * 3 * 5, dtype=np.uint8).reshape(2, 3, 5, 1)
        >>> transform = A.RandomRotate90_3D(axis_pair=(0, 2), group_element="r90", p=1.0)
        >>> result = transform(volume=volume)
        >>> result["volume"].shape
        (5, 3, 2, 1)

    See Also:
        - `Affine3D`: Sample continuous rotations with interpolation and optional scale or translation.

    """

    _targets = (Targets.VOLUME, Targets.MASK3D, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        axis_pairs: tuple[AxisPair3D, ...]
        axis_pair: AxisPair3D | None
        group_element: C4GroupElement | None

        @model_validator(mode="after")
        def _validate_rotation_config(self) -> Self:
            if not self.axis_pairs:
                raise ValueError("axis_pairs must contain at least one spatial axis pair")
            if len(self.axis_pairs) != len(set(self.axis_pairs)):
                raise ValueError("axis_pairs must not contain duplicate axis pairs")
            pairs_to_validate = (*self.axis_pairs, *((self.axis_pair,) if self.axis_pair is not None else ()))
            for axis_pair in pairs_to_validate:
                if axis_pair[0] >= axis_pair[1]:
                    raise ValueError(
                        "Each axis pair must contain distinct axes in ascending (depth, height, width) order",
                    )
            return self

    def __init__(
        self,
        axis_pairs: tuple[AxisPair3D, ...] = DEFAULT_ROTATION_AXIS_PAIRS,
        axis_pair: AxisPair3D | None = None,
        group_element: C4GroupElement | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.axis_pairs = axis_pairs
        self.axis_pair = axis_pair
        self.group_element = group_element

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        axis_pair = self.axis_pair if self.axis_pair is not None else sampling.py_random.choice(self.axis_pairs)
        if self.group_element is not None:
            group_element = self.group_element
        else:
            group_element = cast("C4GroupElement", sampling.py_random.choice(c4_group_elements))

        rotation_count = cast("RotationCount3D", fgeometric.C4_GROUP_ELEMENT_TO_K[group_element])

        sampling.applied_overrides.update({"axis_pair": axis_pair, "group_element": group_element})
        return SampledParams(
            params={
                "axis_pair": axis_pair,
                "rotation_count": rotation_count,
                "volume_shape": _sampling_volume_shape(targets),
            }
        )

    def apply_to_volume(
        self,
        volume: VolumeType,
        axis_pair: AxisPair3D,
        rotation_count: RotationCount3D,
        **params: Any,
    ) -> VolumeType:
        return f3d.rotate90_3d(volume, rotation_count, axis_pair)

    def apply_to_mask3d(
        self,
        mask3d: VolumeType,
        axis_pair: AxisPair3D,
        rotation_count: RotationCount3D,
        **params: Any,
    ) -> VolumeType:
        return f3d.rotate90_3d(mask3d, rotation_count, axis_pair)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        axis_pair: AxisPair3D,
        rotation_count: RotationCount3D,
        volume_shape: tuple[int, int, int],
        **params: Any,
    ) -> np.ndarray:
        return f3d.keypoints_rotate90_3d(keypoints, rotation_count, axis_pair, volume_shape)

    def inverse(self) -> Self:
        """Return a deterministic transform whose quarter turns undo this rotation. Use after inference in TTA
        to restore predictions to the original orientation.

        Raises:
            ValueError: If this transform is configured in random mode.

        """
        if self.axis_pair is None or self.group_element is None:
            raise ValueError(
                "Cannot invert RandomRotate90_3D in random mode. Set axis_pair and group_element for TTA.",
            )
        return type(self)(
            axis_pair=self.axis_pair,
            group_element=C4_INVERSE[self.group_element],
            p=1.0,
        )


class GridShuffle3D(Transform3D):
    """Randomly shuffles the grid's cells on a 3D volume, mask3d, or keypoints,
    effectively rearranging patches within the volume.

    This transformation divides the volume into a 3D grid and then permutes these grid cells based on a random mapping.
    Unlike the 2D version, this does not support bounding boxes as 3D bounding boxes are not yet implemented.

    Args:
        grid_zyx (tuple[int, int, int]): Size of the grid for splitting the volume into cells along (Z, Y, X) axes,
            corresponding to (depth, height, width) dimensions. Each cell is shuffled randomly.
            For example, (2, 3, 3) will divide the volume into 2 slices along Z, 3 along Y, and 3 along X,
            resulting in 18 cells to be shuffled.
            Default: (2, 2, 2)
        p (float): Probability that the transform will be applied. Should be in the range [0, 1].
            Default: 0.5

    Targets:
        volume, mask3d, keypoints

    Note:
        - This transform maintains consistency across all targets. If applied to a volume and its corresponding
          mask3d or keypoints, the same shuffling will be applied to all.
        - The number of cells in the grid should be at least 2 (i.e., grid_zyx should be at least (1, 1, 2), (1, 2, 1),
          (2, 1, 1) or larger) for the transform to have any effect.
        - Keypoints are moved along with their corresponding grid cell.
        - The grid_zyx parameter corresponds to volume dimensions: Z (depth), Y (height), X (width).

    Mathematical Formulation:
        1. The volume is divided into a grid of size (d, m, n) as specified by the 'grid_zyx' parameter.
        2. A random permutation P of integers from 0 to (d*m*n - 1) is generated.
        3. Each cell in the grid is assigned a number from 0 to (d*m*n - 1) in depth-row-column-major order.
        4. The cells are then rearranged according to the permutation P.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> # Prepare sample data
        >>> volume = np.random.randint(0, 256, (10, 100, 100), dtype=np.uint8)  # (D, H, W)
        >>> mask3d = np.random.randint(0, 2, (10, 100, 100), dtype=np.uint8)    # (D, H, W)
        >>> keypoints = np.array([[20, 30, 5], [60, 70, 8]], dtype=np.float32)  # (x, y, z)
        >>> keypoint_labels = [1, 2]  # Labels for each keypoint
        >>>
        >>> # Define transform with grid_zyx as a tuple (Z, Y, X)
        >>> transform = A.Compose([
        ...     A.GridShuffle3D(grid_zyx=(2, 3, 3), p=1.0),
        ... ], keypoint_params=A.KeypointParams(coord_format='xyz', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     volume=volume,
        ...     mask3d=mask3d,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> transformed_volume = transformed['volume']           # Grid-shuffled volume
        >>> transformed_mask3d = transformed['mask3d']           # Grid-shuffled mask
        >>> transformed_keypoints = transformed['keypoints']     # Grid-shuffled keypoints
        >>> transformed_keypoint_labels = transformed['keypoint_labels']  # Labels remain unchanged

    """

    class InitSchema(BaseTransformInitSchema):
        grid_zyx: Annotated[tuple[int, int, int], AfterValidator(check_range_bounds(1, None))]

    _targets = (Targets.VOLUME, Targets.MASK3D, Targets.KEYPOINTS)

    def __init__(
        self,
        grid_zyx: tuple[int, int, int] = (2, 2, 2),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.grid_zyx = grid_zyx

    def apply_to_volume(
        self,
        volume: VolumeType,
        tiles: np.ndarray,
        mapping: list[int],
        **params: Any,
    ) -> VolumeType:
        return f3d.swap_tiles_on_volume(volume, tiles, mapping)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        tiles: np.ndarray,
        mapping: list[int],
        **params: Any,
    ) -> np.ndarray:
        return f3d.swap_tiles_on_keypoints_3d(keypoints, tiles, mapping)

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        volume_shape = _sampling_volume_shape(targets)

        original_tiles = f3d.split_uniform_grid_3d(
            volume_shape,
            self.grid_zyx,
            sampling.random_generator,
        )
        shape_groups = f3d.create_shape_groups_3d(original_tiles)
        mapping = f3d.shuffle_tiles_within_shape_groups_3d(
            shape_groups,
            sampling.random_generator,
        )

        return SampledParams(params={"tiles": original_tiles, "mapping": mapping})
