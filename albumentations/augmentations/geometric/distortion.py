"""Geometric distortion transforms for image augmentation.
This module provides various geometric distortion transformations that modify the spatial arrangement
of pixels in images while preserving their intensity values. These transforms can create
non-rigid deformations that are useful for data augmentation, especially when training models
that need to be robust to geometric variations.

Available transforms:
- ElasticTransform: Creates random elastic deformations by displacing pixels along random vectors
- GridDistortion: Distorts the image by moving the nodes of a grid placed on the image
- OpticalDistortion: Simulates lens distortion effects (barrel/pincushion) using camera or fisheye models
- PiecewiseAffine: Divides the image into a grid and applies random affine transformations to each cell
- ThinPlateSpline: Applies smooth deformations based on the thin plate spline interpolation technique

Remap transforms share a common interface for applying distortion maps to various target types
(images, masks, bounding boxes, keypoints). `BaseDistortion` retains the map-resolution and
keypoint policy used by the other distortion families; `ElasticTransform` uses
the bounded control-grid contract directly.
These transforms are particularly useful for:

- Data augmentation to increase training set diversity
- Simulating real-world distortion effects like camera lens aberrations
- Creating more challenging test cases for computer vision models
- Medical image analysis where anatomy might appear in different shapes

Each transform supports customization through various parameters controlling the strength,
type, and characteristics of the distortion, as well as interpolation methods for different
target types.
"""

from typing import Annotated, Any, Literal

import numpy as np
from albucore import remap
from pydantic import (
    AfterValidator,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from typing_extensions import Self

import albumentations.augmentations.pixel.functional as fpixel
from albumentations.augmentations.utils import check_range
from albumentations.core.bbox_utils import (
    denormalize_bboxes,
    normalize_bboxes,
)
from albumentations.core.invocation import SamplingContext
from albumentations.core.pydantic import (
    check_range_bounds,
    nondecreasing,
)
from albumentations.core.transform_params import TransformParameterPlan, TransformSamplingInput
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    DualTransform,
)
from albumentations.core.type_definitions import (
    ALL_TARGETS,
    CV2_BORDER_CONSTANT,
    CV2_BORDER_REFLECT_101,
    CV2_INTER_LINEAR,
    CV2_INTER_NEAREST,
    BorderModeType,
    ImageType,
    InterpolationType,
    VolumeType,
)

from . import functional as fgeometric

__all__ = [
    "ElasticTransform",
    "GridDistortion",
    "OpticalDistortion",
    "PiecewiseAffine",
    "PixelSpread",
    "ThinPlateSpline",
    "WaterRefraction",
]


class BaseRemapTransform(DualTransform):
    """Dispatch a sampled coordinate-map pair across raster and annotation targets while subclasses own sampling and
    target-specific keypoint policy.

    Examples:
        A subclass samples its coordinate map and inherits synchronized image, mask, bbox, keypoint, batch, and volume
        dispatch from this base.

    """

    _targets = ALL_TARGETS
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})

    class InitSchema(BaseTransformInitSchema):
        interpolation: InterpolationType
        mask_interpolation: InterpolationType
        border_mode: BorderModeType
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

    def __init__(
        self,
        interpolation: InterpolationType,
        mask_interpolation: InterpolationType,
        p: float,
        border_mode: BorderModeType = CV2_BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
    ):
        super().__init__(p=p)
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.border_mode = border_mode
        self.fill = fill
        self.fill_mask = fill_mask

    def apply(
        self,
        img: ImageType,
        map_x: np.ndarray,
        map_y: np.ndarray,
        **params: Any,
    ) -> ImageType:
        result = remap(
            img,
            map_x,
            map_y,
            interpolation=self.interpolation,
            border_mode=self.border_mode,
            border_value=self.fill,
        )
        return fgeometric.clip_if_interpolation_can_overshoot(result, self.interpolation)

    def apply_to_mask3d(self, mask3d: VolumeType, **params: Any) -> VolumeType:
        return self._apply_to_batch_same_shape(mask3d, lambda mask: self.apply_to_mask(mask, **params))

    def apply_to_mask(
        self,
        mask: ImageType,
        map_x: np.ndarray,
        map_y: np.ndarray,
        **params: Any,
    ) -> ImageType:
        return remap(
            mask,
            map_x,
            map_y,
            interpolation=self.mask_interpolation,
            border_mode=self.border_mode,
            border_value=self.fill_mask,
        )

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        map_x: np.ndarray,
        map_y: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        image_shape = params["shape"][:2]
        bbox_type = params["bbox_type"]
        bboxes_denorm = denormalize_bboxes(bboxes, image_shape)
        bboxes_returned = fgeometric.remap_bboxes(
            bboxes_denorm,
            map_x,
            map_y,
            image_shape,
            bbox_type=bbox_type,
        )
        return normalize_bboxes(bboxes_returned, image_shape)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        *args: Any,
        **params: Any,
    ) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__} must implement apply_to_keypoints")


class BaseDistortion(BaseRemapTransform):
    """Provide map-resolution sampling and direct/mask keypoint policies for remap-based distortion subclasses that
    do not define their own continuous inverse.

    Examples:
        `GridDistortion`, `OpticalDistortion`, and `PiecewiseAffine` inherit this map-resolution and keypoint policy.

    """

    class InitSchema(BaseRemapTransform.InitSchema):
        keypoint_remapping_method: Literal["direct", "mask"]
        map_resolution_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1, min_inclusive=False)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        interpolation: InterpolationType,
        mask_interpolation: InterpolationType,
        keypoint_remapping_method: Literal["direct", "mask"],
        p: float,
        border_mode: BorderModeType = CV2_BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        map_resolution_range: tuple[float, float] = (1.0, 1.0),
    ):
        super().__init__(
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            p=p,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
        )
        self.keypoint_remapping_method = keypoint_remapping_method
        self.map_resolution_range = map_resolution_range

    def _get_map_resolution_and_shape(
        self,
        image_shape: tuple[int, int],
        sampling: SamplingContext,
    ) -> tuple[float, tuple[int, int]]:
        min_resolution, max_resolution = self.map_resolution_range
        map_resolution = (
            min_resolution
            if min_resolution == max_resolution
            else sampling.py_random.uniform(min_resolution, max_resolution)
        )
        sampling.applied_overrides["map_resolution_range"] = map_resolution

        height, width = image_shape
        scaled_shape = (
            max(2, int(height * map_resolution)),
            max(2, int(width * map_resolution)),
        )
        return map_resolution, scaled_shape

    def _maybe_upscale_maps(
        self,
        map_x: np.ndarray,
        map_y: np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        return fgeometric.upscale_distortion_maps(map_x, map_y, image_shape, self.interpolation)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        map_x: np.ndarray,
        map_y: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        if self.keypoint_remapping_method == "direct":
            return fgeometric.remap_keypoints(keypoints, map_x, map_y, params["shape"])
        return fgeometric.remap_keypoints_via_mask(keypoints, map_x, map_y, params["shape"])


class ElasticTransform(BaseRemapTransform):
    """Apply bounded XY deformations from a compact control grid to images and annotations. Use it for
    shape variation in segmentation and medical imaging.

    `displacement_range` is measured relative to the shorter span between the first and last
    pixel centers. The sampled cubic B-spline coefficients use pixel units after scaling. One map
    is shared by every raster and annotation target in an invocation; volumes receive the same XY
    deformation on every depth slice.

    Args:
        displacement_range (tuple[float, float]): Range for the sampled relative displacement magnitude.
        control_grid_shape (tuple[int, int]): Number of cubic B-spline coefficient rows and columns, each at least 4.
        interpolation (int): Interpolation used for images.
        mask_interpolation (int): Interpolation used for masks.
        border_mode (int): OpenCV border mode for raster targets.
        fill (tuple[float, ...] | float): Fill value for images.
        fill_mask (tuple[float, ...] | float): Fill value for masks.
        p (float): Probability of applying the transform.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Supported bboxes:
        hbb, obb

    Note:
        The constructor enforces `2 * high * sqrt((rows - 3)^2 + (columns - 3)^2) < 0.75`.
        `ReplayCompose` stores the compact sampled coefficient lattice and replays it for the same
        spatial shape. Applied configuration fixes the realized magnitude but samples a new lattice.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        >>> bbox_labels = [1]
        >>> keypoints = np.array([[20, 30]], dtype=np.float32)
        >>> keypoint_labels = [0]
        >>> transform = A.Compose(
        ...     [A.ElasticTransform(displacement_range=(0.02, 0.05), control_grid_shape=(7, 7), p=1.0)],
        ...     bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["bbox_labels"]),
        ...     keypoint_params=A.KeypointParams(
        ...         coord_format="xy", label_fields=["keypoint_labels"], label_mapping={}
        ...     ),
        ... )
        >>> transformed = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels,
        ... )
        >>> transformed_image = transformed["image"]
        >>> transformed_mask = transformed["mask"]
        >>> transformed_bboxes = transformed["bboxes"]
        >>> transformed_bbox_labels = transformed["bbox_labels"]
        >>> transformed_keypoints = transformed["keypoints"]
        >>> transformed_keypoint_labels = transformed["keypoint_labels"]

    """

    _runtime_generated_params = frozenset({"map_x", "map_y"})

    class InitSchema(BaseRemapTransform.InitSchema):
        displacement_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0)),
            AfterValidator(nondecreasing),
        ]
        control_grid_shape: tuple[int, int]

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
        interpolation: InterpolationType = CV2_INTER_LINEAR,
        mask_interpolation: InterpolationType = CV2_INTER_NEAREST,
        border_mode: BorderModeType = CV2_BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        p: float = 0.5,
    ):
        super().__init__(
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            p=p,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
        )
        self.displacement_range = displacement_range
        self.control_grid_shape = control_grid_shape

    def sample_parameters(
        self,
        inputs: TransformSamplingInput,
        sampling: SamplingContext,
    ) -> TransformParameterPlan:
        data = inputs.data
        del data
        image_shape = inputs.require_spatial_frame().spatial_shape_2d
        low, high = self.displacement_range
        magnitude = low if low == high else sampling.py_random.uniform(low, high)
        sampling.applied_overrides["displacement_range"] = (magnitude, magnitude)

        height, width = image_shape
        if magnitude == 0 or min(height - 1, width - 1) <= 0:
            return TransformParameterPlan.shared_only(
                {
                    "displacement_magnitude": magnitude,
                    "control_coefficients": [],
                }
            )

        random_values = sampling.random_generator.random((*self.control_grid_shape, 2), dtype=np.float32)
        radius = np.float32(magnitude * min(height - 1, width - 1))
        vector_radius = radius * np.sqrt(random_values[..., 0])
        angle = np.float32(2 * np.pi) * random_values[..., 1]
        control_coefficients = np.empty((*self.control_grid_shape, 2), dtype=np.float32)
        control_coefficients[..., 0] = vector_radius * np.cos(angle)
        control_coefficients[..., 1] = vector_radius * np.sin(angle)
        return TransformParameterPlan.shared_only(
            {
                "displacement_magnitude": magnitude,
                "control_coefficients": control_coefficients.tolist(),
            }
        )

    def _apply_label_mappings_without_geometry(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Preserve every target while applying configured keypoint and semantic-mask
        label mappings for an exact identity deformation.
        """
        result = dict(data)
        if "keypoints" in result and result["keypoints"] is not None:
            result["keypoints"] = self._apply_label_mapping_to_keypoints(result["keypoints"], **params)
        if self._semantic_mask_label_mappings:
            result = self._apply_label_mapping_to_semantic_masks(result, **params)
        return result

    def apply_with_params(self, plan: TransformParameterPlan, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Build one dense map for the invocation and reject replay on another spatial shape before dispatching all
        synchronized targets with exact map sharing.
        """
        params = dict(plan.params_for("image"))
        recorded_shape = tuple(params["shape"][:2])
        if self.replay_mode:
            current_targets = self._build_target_set(kwargs)
            current_frame = self._build_spatial_frame(current_targets)
            if current_frame is not None and current_frame.shape != recorded_shape:
                raise ValueError(
                    f"ElasticTransform replay requires the same spatial shape {recorded_shape}, "
                    f"got {current_frame.shape}",
                )
        if not params["control_coefficients"]:
            return self._apply_label_mappings_without_geometry(params, kwargs)

        runtime_params = dict(params)
        control_coefficients = np.asarray(params["control_coefficients"], dtype=np.float32)
        runtime_params["control_coefficients"] = control_coefficients
        runtime_params["map_x"], runtime_params["map_y"] = fgeometric.create_elastic_maps(
            control_coefficients,
            recorded_shape,
        )
        runtime_plan = TransformParameterPlan(
            shared=runtime_params,
            groups=plan.groups,
            target_schema=plan.target_schema,
        )
        return super().apply_with_params(runtime_plan, *args, **kwargs)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        control_coefficients: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.remap_elastic_keypoints(
            keypoints,
            control_coefficients,
            params["shape"][:2],
        )


class PiecewiseAffine(BaseDistortion):
    """Apply piecewise affine transformations via a regular grid of control points. Params:
    scale_range, nb_rows_range, nb_cols_range, interpolation.

    This augmentation places a regular grid of points on an image and randomly moves the neighborhood of these points
    around via affine transformations. This leads to local distortions in the image.

    Args:
        scale_range (tuple[float, float]): Standard deviation of the normal distributions used
            to sample random corner offsets, sampled per image. Recommended values are in
            (0.01, 0.05) for small distortions and (0.05, 0.1) for larger distortions.
            Default: (0.03, 0.05).
        nb_rows_range (tuple[int, int]): Range for the number of rows in the regular grid;
            a value from the discrete interval [a..b] is uniformly sampled per image. Both ends
            must be >= 2. Default: (4, 4).
        nb_cols_range (tuple[int, int]): Range for the number of columns in the regular grid;
            a value from the discrete interval [a..b] is uniformly sampled per image. Both ends
            must be >= 2. Default: (4, 4).
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        absolute_scale (bool): If set to True, the value of the scale parameter will be treated as an absolute
            pixel value. If set to False, it will be treated as a fraction of the image height and width.
            Default: False.
        keypoint_remapping_method (Literal['direct', 'mask']): Method to use for keypoint remapping.
            - "mask": Uses mask-based remapping. Faster, especially for many keypoints, but may be
              less accurate for large distortions. Recommended for large images or many keypoints.
            - "direct": Uses inverse mapping. More accurate for large distortions but slower.
            Default: "mask"
        map_resolution_range (tuple[float, float]): Range for sampling the displacement map resolution
            relative to the target size. Values below 1.0 generate lower-resolution maps and upscale
            them, trading precision for speed. Default: (1.0, 1.0).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints, bboxes, volume, mask3d

    Image types:
        uint8, float32

    Supported bboxes:
        hbb, obb

    Note:
        - The augmentation may not always produce visible effects, especially with small scale values.
        - For keypoints and bounding boxes, the transformation might move them outside the image boundaries.
          In such cases, the keypoints will be set to (-1, -1) and the bounding boxes will be removed.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.Compose([
        ...     A.PiecewiseAffine(scale_range=(0.03, 0.05), nb_rows_range=(4, 4), nb_cols_range=(4, 4), p=0.5),
        ... ])
        >>> transformed = transform(image=image)
        >>> transformed_image = transformed["image"]

    """

    class InitSchema(BaseDistortion.InitSchema):
        scale_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0)),
            AfterValidator(nondecreasing),
        ]
        nb_rows_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(2, None)),
        ]
        nb_cols_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(2, None)),
        ]
        absolute_scale: bool

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.03, 0.05),
        nb_rows_range: tuple[int, int] = (4, 4),
        nb_cols_range: tuple[int, int] = (4, 4),
        interpolation: InterpolationType = CV2_INTER_LINEAR,
        mask_interpolation: InterpolationType = CV2_INTER_NEAREST,
        absolute_scale: bool = False,
        keypoint_remapping_method: Literal["direct", "mask"] = "mask",
        p: float = 0.5,
        border_mode: BorderModeType = CV2_BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        map_resolution_range: tuple[float, float] = (1.0, 1.0),
    ):
        super().__init__(
            p=p,
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            keypoint_remapping_method=keypoint_remapping_method,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
            map_resolution_range=map_resolution_range,
        )

        self.scale_range = scale_range
        self.nb_rows_range = nb_rows_range
        self.nb_cols_range = nb_cols_range
        self.absolute_scale = absolute_scale

    def sample_parameters(
        self,
        inputs: TransformSamplingInput,
        sampling: SamplingContext,
    ) -> TransformParameterPlan:
        image_shape = inputs.require_spatial_frame().spatial_shape_2d
        _, scaled_shape = self._get_map_resolution_and_shape(image_shape, sampling)

        nb_rows = np.clip(sampling.py_random.randint(*self.nb_rows_range), 2, None)
        nb_cols = np.clip(sampling.py_random.randint(*self.nb_cols_range), 2, None)
        scale = sampling.py_random.uniform(*self.scale_range)

        sampling.applied_overrides["scale_range"] = scale
        sampling.applied_overrides["nb_rows_range"] = int(nb_rows)
        sampling.applied_overrides["nb_cols_range"] = int(nb_cols)

        map_x, map_y = fgeometric.create_piecewise_affine_maps(
            image_shape=scaled_shape,
            grid=(nb_rows, nb_cols),
            scale=scale,
            absolute_scale=self.absolute_scale,
            random_generator=sampling.random_generator,
        )
        if map_x is None or map_y is None:
            map_y, map_x = np.meshgrid(
                np.arange(image_shape[0], dtype=np.float32),
                np.arange(image_shape[1], dtype=np.float32),
                indexing="ij",
            )
        else:
            map_x, map_y = self._maybe_upscale_maps(map_x, map_y, image_shape)

        return TransformParameterPlan.shared_only({"map_x": map_x, "map_y": map_y})


class OpticalDistortion(BaseDistortion):
    """Apply optical distortion (lens/camera or fisheye model) to images, masks, bboxes, keypoints.
    Params: distort_range, mode (camera/fisheye), interpolation.

    Supports two distortion models:
    1. Camera matrix model (original):
       Uses OpenCV's camera calibration model with k1=k2=k distortion coefficients

    2. Fisheye model:
       Direct radial distortion: r_dist = r * (1 + gamma * r²)

    Args:
        distort_range (tuple[float, float]): Range of distortion coefficient, sampled per image.
            For camera model: recommended range (-0.05, 0.05).
            For fisheye model: recommended range (-0.3, 0.3).
            Default: (-0.05, 0.05)

        mode (Literal['camera', 'fisheye']): Distortion model to use:
            - 'camera': Original camera matrix model
            - 'fisheye': Fisheye lens model
            Default: 'camera'

        interpolation (OpenCV flag): Interpolation method used for image transformation.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC,
            cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.

        mask_interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.

        keypoint_remapping_method (Literal['direct', 'mask']): Method to use for keypoint remapping.
            - "mask": Uses mask-based remapping. Faster, especially for many keypoints, but may be
              less accurate for large distortions. Recommended for large images or many keypoints.
            - "direct": Uses inverse mapping. More accurate for large distortions but slower.
            Default: "mask"
        map_resolution_range (tuple[float, float]): Range for sampling the displacement map resolution
            relative to the target size. Values below 1.0 generate lower-resolution maps and upscale
            them, trading precision for speed. Default: (1.0, 1.0).

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32


    Supported bboxes:
        hbb, obb
    Note:
        - The distortion is applied using OpenCV's initUndistortRectifyMap and remap functions.
        - The distortion coefficient (k) is randomly sampled from the distort_range range.
        - Bounding boxes and keypoints are transformed along with the image to maintain consistency.
        - Fisheye model directly applies radial distortion

    Examples:
        >>> import albumentations as A
        >>> transform = A.Compose([
        ...     A.OpticalDistortion(distort_range=(-0.1, 0.1), p=1.0),
        ... ])
        >>> transformed = transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
        >>> transformed_image = transformed['image']
        >>> transformed_mask = transformed['mask']
        >>> transformed_bboxes = transformed['bboxes']
        >>> transformed_keypoints = transformed['keypoints']

    """

    class InitSchema(BaseDistortion.InitSchema):
        distort_range: tuple[float, float]
        mode: Literal["camera", "fisheye"]
        keypoint_remapping_method: Literal["direct", "mask"]

    def __init__(
        self,
        distort_range: tuple[float, float] = (-0.05, 0.05),
        interpolation: InterpolationType = CV2_INTER_LINEAR,
        mask_interpolation: InterpolationType = CV2_INTER_NEAREST,
        mode: Literal["camera", "fisheye"] = "camera",
        keypoint_remapping_method: Literal["direct", "mask"] = "mask",
        p: float = 0.5,
        border_mode: BorderModeType = CV2_BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        map_resolution_range: tuple[float, float] = (1.0, 1.0),
    ):
        super().__init__(
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            keypoint_remapping_method=keypoint_remapping_method,
            p=p,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
            map_resolution_range=map_resolution_range,
        )
        self.distort_range = distort_range
        self.mode = mode

    def sample_parameters(
        self,
        inputs: TransformSamplingInput,
        sampling: SamplingContext,
    ) -> TransformParameterPlan:
        image_shape = inputs.require_spatial_frame().spatial_shape_2d
        _, scaled_shape = self._get_map_resolution_and_shape(image_shape, sampling)

        k = sampling.py_random.uniform(*self.distort_range)

        sampling.applied_overrides["distort_range"] = k

        if k == 0:
            height, width = image_shape
            map_y, map_x = np.meshgrid(
                np.arange(height, dtype=np.float32),
                np.arange(width, dtype=np.float32),
                indexing="ij",
            )
            return TransformParameterPlan.shared_only({"map_x": map_x, "map_y": map_y})

        if self.mode == "camera":
            map_x, map_y = fgeometric.get_camera_matrix_distortion_maps(
                scaled_shape,
                k,
            )
        else:
            map_x, map_y = fgeometric.get_fisheye_distortion_maps(
                scaled_shape,
                k,
            )
        map_x, map_y = self._maybe_upscale_maps(map_x, map_y, image_shape)

        return TransformParameterPlan.shared_only({"map_x": map_x, "map_y": map_y})


class GridDistortion(BaseDistortion):
    """Apply grid distortion by dividing the image into cells and warping each. Params: num_steps,
    distort_range, interpolation, normalized.

    This transformation divides the image into a grid and randomly distorts each cell,
    creating localized warping effects. It's particularly useful for data augmentation
    in tasks like medical image analysis, OCR, and other domains where local geometric
    variations are meaningful.

    Args:
        num_steps (int): Number of grid cells on each side of the image. Higher values
            create more granular distortions. Must be at least 1. Default: 5.
        distort_range (tuple[float, float]): Range of distortion, sampled per image. Higher
            absolute values create stronger distortions. Should be in [-1, 1].
            Default: (-0.3, 0.3).
        interpolation (int): OpenCV interpolation method used for image transformation.
            Options include cv2.INTER_LINEAR, cv2.INTER_CUBIC, etc. Default: cv2.INTER_LINEAR.
        normalized (bool): If True, ensures that the distortion does not move pixels
            outside the image boundaries. This can result in less extreme distortions
            but guarantees that no information is lost. Default: True.
        mask_interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        keypoint_remapping_method (Literal['direct', 'mask']): Method to use for keypoint remapping.
            - "mask": Uses mask-based remapping. Faster, especially for many keypoints, but may be
              less accurate for large distortions. Recommended for large images or many keypoints.
            - "direct": Uses inverse mapping. More accurate for large distortions but slower.
            Default: "mask"
        map_resolution_range (tuple[float, float]): Range for sampling the displacement map resolution
            relative to the target size. Values below 1.0 generate lower-resolution maps and upscale
            them, trading precision for speed. Default: (1.0, 1.0).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32


    Supported bboxes:
        hbb, obb
    Note:
        - The same distortion is applied to all targets (image, mask, bboxes, keypoints)
          to maintain consistency.
        - When normalized=True, the distortion is adjusted to ensure all pixels remain
          within the image boundaries.

    Examples:
        >>> import albumentations as A
        >>> transform = A.Compose([
        ...     A.GridDistortion(num_steps=5, distort_range=(-0.3, 0.3), p=1.0),
        ... ])
        >>> transformed = transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
        >>> transformed_image = transformed['image']
        >>> transformed_mask = transformed['mask']
        >>> transformed_bboxes = transformed['bboxes']
        >>> transformed_keypoints = transformed['keypoints']

    """

    class InitSchema(BaseDistortion.InitSchema):
        num_steps: Annotated[int, Field(ge=1)]
        distort_range: tuple[float, float]
        normalized: bool
        keypoint_remapping_method: Literal["direct", "mask"]

        @field_validator("distort_range")
        @classmethod
        def _check_limits(
            cls,
            v: tuple[float, float],
            info: ValidationInfo,
        ) -> tuple[float, float]:
            check_range(v, -1, 1, info.field_name)
            return v

    def __init__(
        self,
        num_steps: int = 5,
        distort_range: tuple[float, float] = (-0.3, 0.3),
        interpolation: InterpolationType = CV2_INTER_LINEAR,
        normalized: bool = True,
        mask_interpolation: InterpolationType = CV2_INTER_NEAREST,
        keypoint_remapping_method: Literal["direct", "mask"] = "mask",
        p: float = 0.5,
        border_mode: BorderModeType = CV2_BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        map_resolution_range: tuple[float, float] = (1.0, 1.0),
    ):
        super().__init__(
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            keypoint_remapping_method=keypoint_remapping_method,
            p=p,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
            map_resolution_range=map_resolution_range,
        )
        self.num_steps = num_steps
        self.distort_range = distort_range
        self.normalized = normalized

    def sample_parameters(
        self,
        inputs: TransformSamplingInput,
        sampling: SamplingContext,
    ) -> TransformParameterPlan:
        image_shape = inputs.require_spatial_frame().spatial_shape_2d
        _, scaled_shape = self._get_map_resolution_and_shape(image_shape, sampling)
        num_steps = min(self.num_steps, *scaled_shape)

        steps_x = (1 + sampling.random_generator.uniform(*self.distort_range, size=num_steps + 1)).tolist()
        steps_y = (1 + sampling.random_generator.uniform(*self.distort_range, size=num_steps + 1)).tolist()

        # distort_range is per-cell uniform bounds; record realized (min, max) of sampled distortions
        # (steps are stored as 1+sample, so subtract 1 to get the raw distortion values).
        all_steps = np.array(steps_x + steps_y) - 1.0
        sampling.applied_overrides["distort_range"] = (float(all_steps.min()), float(all_steps.max()))

        if np.all(all_steps == 0):
            height, width = image_shape
            map_y, map_x = np.meshgrid(
                np.arange(height, dtype=np.float32),
                np.arange(width, dtype=np.float32),
                indexing="ij",
            )
            return TransformParameterPlan.shared_only({"map_x": map_x, "map_y": map_y})

        if self.normalized:
            normalized_params = fgeometric.normalize_grid_distortion_steps(
                scaled_shape,
                num_steps,
                steps_x,
                steps_y,
            )
            steps_x, steps_y = (
                normalized_params["steps_x"],
                normalized_params["steps_y"],
            )

        map_x, map_y = fgeometric.generate_grid(
            scaled_shape,
            steps_x,
            steps_y,
            num_steps,
        )
        map_x, map_y = self._maybe_upscale_maps(map_x, map_y, image_shape)

        return TransformParameterPlan.shared_only({"map_x": map_x, "map_y": map_y})


class ThinPlateSpline(BaseDistortion):
    r"""Apply Thin Plate Spline (TPS) for smooth, non-rigid deformations. Control points
    warp the image like pins on a thin plate; smooth interpolation between points.

    Imagine the image printed on a thin metal plate that can be bent and warped smoothly:
    - Control points act like pins pushing or pulling the plate
    - The plate resists sharp bending, creating smooth deformations
    - The transformation maintains continuity (no tears or folds)
    - Areas between control points are interpolated naturally

    The transform works by:
    1. Creating a regular grid of control points (like pins in the plate)
    2. Randomly displacing these points (like pushing/pulling the pins)
    3. Computing a smooth interpolation (like the plate bending)
    4. Applying the resulting deformation to the image


    Args:
        scale_range (tuple[float, float]): Range for random displacement of control points.
            Values should be in [0.0, 1.0]:
            - 0.0: No displacement (identity transform)
            - 0.1: Subtle warping
            - 0.2-0.4: Moderate deformation (recommended range)
            - 0.5+: Strong warping
            Default: (0.2, 0.4)

        num_control_points (int): Number of control points per side.
            Creates a grid of num_control_points x num_control_points points.
            - 2: Minimal deformation (affine-like)
            - 3-4: Moderate flexibility (recommended)
            - 5+: More local deformation control
            Must be >= 2. Default: 4

        interpolation (int): OpenCV interpolation flag. Used for image sampling.
            See also: cv2.INTER_*
            Default: cv2.INTER_LINEAR

        mask_interpolation (int): OpenCV interpolation flag. Used for mask sampling.
            See also: cv2.INTER_*
            Default: cv2.INTER_NEAREST

        keypoint_remapping_method (Literal['direct', 'mask']): Method to use for keypoint remapping.
            - "mask": Uses mask-based remapping. Faster, especially for many keypoints, but may be
              less accurate for large distortions. Recommended for large images or many keypoints.
            - "direct": Uses inverse mapping. More accurate for large distortions but slower.
            Default: "mask"
        map_resolution_range (tuple[float, float]): Range for sampling the displacement map resolution
            relative to the target size. Values below 1.0 generate lower-resolution maps and upscale
            them, trading precision for speed. Default: (1.0, 1.0).

        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image, mask, keypoints, bboxes, volume, mask3d

    Image types:
        uint8, float32

    Supported bboxes:
        hbb, obb

    Note:
        - The transformation preserves smoothness and continuity
        - Stronger scale values may create more extreme deformations
        - Higher number of control points allows more local deformations
        - The same deformation is applied consistently to all targets

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create sample data
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[25:75, 25:75] = 1  # Square mask
        >>> bboxes = np.array([[10, 10, 40, 40]])  # Single box
        >>> bbox_labels = [1]
        >>> keypoints = np.array([[50, 50]])  # Single keypoint at center
        >>> keypoint_labels = [0]
        >>>
        >>> # Set up transform with Compose to handle all targets
        >>> transform = A.Compose([
        ...     A.ThinPlateSpline(scale_range=(0.2, 0.4), p=1.0)
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply to all targets
        >>> result = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Access transformed results
        >>> transformed_image = result['image']
        >>> transformed_mask = result['mask']
        >>> transformed_bboxes = result['bboxes']
        >>> transformed_bbox_labels = result['bbox_labels']
        >>> transformed_keypoints = result['keypoints']
        >>> transformed_keypoint_labels = result['keypoint_labels']

    References:
        - "Principal Warps: Thin-Plate Splines and the Decomposition of Deformations"
          by F.L. Bookstein
          https://doi.org/10.1109/34.24792

        - Thin Plate Splines in Computer Vision:
          https://en.wikipedia.org/wiki/Thin_plate_spline

        - Similar implementation in Kornia:
          https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomThinPlateSpline

    See Also:
        - ElasticTransform: For different type of non-rigid deformation
        - GridDistortion: For grid-based warping
        - OpticalDistortion: For lens-like distortions

    """

    class InitSchema(BaseDistortion.InitSchema):
        scale_range: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, 1))]
        num_control_points: int = Field(ge=2)
        keypoint_remapping_method: Literal["direct", "mask"]

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.2, 0.4),
        num_control_points: int = 4,
        interpolation: InterpolationType = CV2_INTER_LINEAR,
        mask_interpolation: InterpolationType = CV2_INTER_NEAREST,
        keypoint_remapping_method: Literal["direct", "mask"] = "mask",
        p: float = 0.5,
        border_mode: BorderModeType = CV2_BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        map_resolution_range: tuple[float, float] = (1.0, 1.0),
    ):
        super().__init__(
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            keypoint_remapping_method=keypoint_remapping_method,
            p=p,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
            map_resolution_range=map_resolution_range,
        )
        self.scale_range = scale_range
        self.num_control_points = num_control_points

    def sample_parameters(
        self,
        inputs: TransformSamplingInput,
        sampling: SamplingContext,
    ) -> TransformParameterPlan:
        height, width = inputs.require_spatial_frame().spatial_shape_2d
        image_shape = (height, width)
        _, scaled_shape = self._get_map_resolution_and_shape(image_shape, sampling)
        scaled_height, scaled_width = scaled_shape

        src_points = fgeometric.generate_control_points(self.num_control_points)

        scale = sampling.py_random.uniform(*self.scale_range) / 10

        sampling.applied_overrides["scale_range"] = scale * 10
        dst_points = src_points + sampling.random_generator.normal(
            0,
            scale,
            src_points.shape,
        )

        weights, affine = fgeometric.compute_tps_weights(src_points, dst_points)

        points = np.empty((scaled_height * scaled_width, 2), dtype=np.float32)
        points[:, 0] = np.tile(np.arange(scaled_width, dtype=np.float32) / scaled_width, scaled_height)
        points[:, 1] = np.repeat(np.arange(scaled_height, dtype=np.float32) / scaled_height, scaled_width)

        transformed = fgeometric.tps_transform(
            points,
            src_points,
            weights,
            affine,
        )
        transformed[:, 0] *= scaled_width
        transformed[:, 1] *= scaled_height

        map_x = transformed[:, 0].reshape(scaled_height, scaled_width).astype(np.float32)
        map_y = transformed[:, 1].reshape(scaled_height, scaled_width).astype(np.float32)
        map_x, map_y = self._maybe_upscale_maps(map_x, map_y, image_shape)

        return TransformParameterPlan.shared_only(
            {
                "map_x": map_x,
                "map_y": map_y,
            }
        )


class WaterRefraction(BaseDistortion):
    """Simulate looking through water or wavy glass via sine-wave displacement maps. Params:
    amplitude_range, wavelength_range, num_waves_range, interpolation.

    Generates displacement maps from overlaid sine waves at random frequencies,
    phases, and angles to create a refraction distortion effect.

    Args:
        amplitude_range (tuple[float, float]): Range for maximum displacement as a
            fraction of image size. Default: (0.01, 0.05).
        wavelength_range (tuple[float, float]): Range for wave period as a fraction
            of image size. Default: (0.05, 0.2).
        num_waves_range (tuple[int, int]): Range for number of overlaid sine waves.
            More waves = more complex distortion. Default: (3, 7).
        interpolation (int): OpenCV interpolation flag. Default: cv2.INTER_LINEAR.
        mask_interpolation (int): OpenCV interpolation for masks. Default: cv2.INTER_NEAREST.
        border_mode (int): OpenCV border mode. Default: cv2.BORDER_REFLECT_101.
        fill (float | tuple): Fill value for constant border. Default: 0.
        fill_mask (float | tuple): Fill value for mask borders. Default: 0.
        map_resolution_range (tuple[float, float]): Range for sampling the displacement map resolution
            relative to the target size. Values below 1.0 generate lower-resolution maps and upscale
            them, trading precision for speed. Default: (1.0, 1.0).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Number of channels:
        Any

    Supported bboxes:
        hbb, obb

    Note:
        This is a geometric (DualTransform) because the displacement warps the
        image geometry - masks, bboxes, and keypoints are transformed accordingly.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.WaterRefraction(amplitude_range=(0.02, 0.04), p=1.0)
        >>> result = transform(image=image)["image"]

    """

    class InitSchema(BaseDistortion.InitSchema):
        amplitude_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        wavelength_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, None, min_inclusive=False)),
            AfterValidator(nondecreasing),
        ]
        num_waves_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        amplitude_range: tuple[float, float] = (0.01, 0.05),
        wavelength_range: tuple[float, float] = (0.05, 0.2),
        num_waves_range: tuple[int, int] = (3, 7),
        interpolation: InterpolationType = CV2_INTER_LINEAR,
        mask_interpolation: InterpolationType = CV2_INTER_NEAREST,
        keypoint_remapping_method: Literal["direct", "mask"] = "mask",
        border_mode: BorderModeType = CV2_BORDER_REFLECT_101,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        map_resolution_range: tuple[float, float] = (1.0, 1.0),
        p: float = 0.5,
    ):
        super().__init__(
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            keypoint_remapping_method=keypoint_remapping_method,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
            map_resolution_range=map_resolution_range,
            p=p,
        )
        self.amplitude_range = amplitude_range
        self.wavelength_range = wavelength_range
        self.num_waves_range = num_waves_range

    def sample_parameters(
        self,
        inputs: TransformSamplingInput,
        sampling: SamplingContext,
    ) -> TransformParameterPlan:
        image_shape = inputs.require_spatial_frame().spatial_shape_2d
        _, scaled_shape = self._get_map_resolution_and_shape(image_shape, sampling)
        scaled_height, scaled_width = scaled_shape

        img_size = min(scaled_height, scaled_width)
        amplitude_frac = sampling.py_random.uniform(*self.amplitude_range)
        wavelength_frac = sampling.py_random.uniform(*self.wavelength_range)
        num_waves = sampling.py_random.randint(*self.num_waves_range)

        sampling.applied_overrides["amplitude_range"] = amplitude_frac
        sampling.applied_overrides["wavelength_range"] = wavelength_frac
        sampling.applied_overrides["num_waves_range"] = num_waves

        amplitude = amplitude_frac * img_size
        wavelength = wavelength_frac * img_size

        map_x, map_y = fpixel.generate_water_displacement_maps(
            scaled_shape,
            amplitude,
            wavelength,
            num_waves,
            sampling.random_generator,
        )
        map_x, map_y = self._maybe_upscale_maps(map_x, map_y, image_shape)

        return TransformParameterPlan.shared_only(
            {
                "map_x": map_x,
                "map_y": map_y,
            }
        )


class PixelSpread(BaseDistortion):
    """Stochastically displaces each pixel by sampling its value from a random source within a
    local square neighborhood, without blurring or coherent warping.

    For every output pixel `(row, col)` an offset `(d_row, d_col)` is drawn independently and
    uniformly from the square neighborhood `[-radius, radius] x [-radius, radius]` and the pixel
    value is read from source position `(row + d_row, col + d_col)`. The same dense remapping
    field is applied to all targets (image, mask, bboxes, keypoints) so spatial annotations remain
    consistent.

    This occupies a useful middle ground between blur (which aggregates a neighborhood) and smooth
    elastic warps (which produce coherent displacement fields): the displacement field is intentionally
    non-smooth and high-frequency, making it suitable for simulating sensor noise, compression
    artifacts, fine-grained texture corruption, and domain shifts where local pixel structure
    becomes unstable but global object geometry is preserved.

    Args:
        radius (int): Maximum pixel displacement in each direction. The sampling neighborhood is
            the square `[-radius, radius] x [-radius, radius]`, giving `(2*radius+1)^2` possible
            source locations per output pixel. Must be >= 1. Default: 2.
        interpolation (int): Interpolation flag used by `cv2.remap`. Default: `cv2.INTER_NEAREST`.
            Nearest-neighbor is the natural choice because the effect is explicitly about discrete
            pixel reassignment, not sub-pixel blending.
        mask_interpolation (int): Interpolation flag for masks. Default: `cv2.INTER_NEAREST`.
        keypoint_remapping_method (Literal["direct", "mask"]): Strategy for remapping keypoints.
            Default: `"mask"`.
        border_mode (int): OpenCV border extrapolation mode for out-of-bounds source lookups.
            Default: `cv2.BORDER_REFLECT_101`.
        fill (float | tuple[float, ...]): Fill value used when `border_mode` is
            `cv2.BORDER_CONSTANT`. Default: 0.
        fill_mask (float | tuple[float, ...]): Fill value for masks under constant border.
            Default: 0.
        map_resolution_range (tuple[float, float]): Range for sampling the displacement map resolution
            relative to the target size. Values below 1.0 generate lower-resolution maps and upscale
            them, trading precision for speed. Default: (1.0, 1.0).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Number of channels:
        Any

    Supported bboxes:
        hbb, obb

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
        ...     A.PixelSpread(radius=3, p=1.0)
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
        >>> transformed_mask = result['mask']

    """

    class InitSchema(BaseDistortion.InitSchema):
        radius: Annotated[int, Field(ge=1)]

    def __init__(
        self,
        radius: int = 2,
        interpolation: InterpolationType = CV2_INTER_NEAREST,
        mask_interpolation: InterpolationType = CV2_INTER_NEAREST,
        keypoint_remapping_method: Literal["direct", "mask"] = "mask",
        border_mode: BorderModeType = CV2_BORDER_REFLECT_101,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        map_resolution_range: tuple[float, float] = (1.0, 1.0),
        p: float = 0.5,
    ):
        super().__init__(
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            keypoint_remapping_method=keypoint_remapping_method,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
            map_resolution_range=map_resolution_range,
            p=p,
        )
        self.radius = radius

    def sample_parameters(
        self,
        inputs: TransformSamplingInput,
        sampling: SamplingContext,
    ) -> TransformParameterPlan:
        image_shape = inputs.require_spatial_frame().spatial_shape_2d
        map_resolution, scaled_shape = self._get_map_resolution_and_shape(image_shape, sampling)
        scaled_height, scaled_width = scaled_shape

        row_coords, col_coords = np.meshgrid(
            np.arange(scaled_height, dtype=np.float32),
            np.arange(scaled_width, dtype=np.float32),
            indexing="ij",
        )
        scaled_radius = max(1, round(self.radius * map_resolution))
        offsets = sampling.random_generator.integers(
            -scaled_radius,
            scaled_radius + 1,
            size=(scaled_height, scaled_width, 2),
            dtype=np.int32,
        )
        map_y = (row_coords + offsets[..., 0]).astype(np.float32)
        map_x = (col_coords + offsets[..., 1]).astype(np.float32)
        map_x, map_y = self._maybe_upscale_maps(map_x, map_y, image_shape)

        return TransformParameterPlan.shared_only({"map_x": map_x, "map_y": map_y})
