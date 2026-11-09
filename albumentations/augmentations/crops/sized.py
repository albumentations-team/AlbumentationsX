"""Random sized crop transforms."""

from math import ceil, cos, floor, radians, sin
from typing import Annotated, Any, Literal

import numpy as np

from albumentations.core.bbox_utils import obb_to_polygons
from albumentations.core.invocation import SamplingContext
from albumentations.core.transform_params import SampledParams, TargetSet

from ._sampling import SamplingMethod, relax_2d_crop_ranges, sample_2d_crop_shape
from ._transforms_shared import (
    ALL_TARGETS,
    CV2_INTER_LINEAR,
    CV2_INTER_NEAREST,
    AfterValidator,
    BaseTransformInitSchema,
    Field,
    FullInterpolationType,
    check_range_bounds,
    fcrops,
    math,
    nondecreasing,
)
from .base import (
    _BaseRandomSizedCrop,
)


def _validate_positive_finite_ratio_range(value: tuple[float, float]) -> tuple[float, float]:
    if not all(math.isfinite(item) and item > 0.0 for item in value):
        raise ValueError("ratio values must be finite and greater than 0")
    return value


def _merge_bbox_envelopes(
    current: tuple[float, float, float, float] | None,
    candidate: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return (
        min(current[0], candidate[0]),
        min(current[1], candidate[1]),
        max(current[2], candidate[2]),
        max(current[3], candidate[3]),
    )


def _clip_polygon_to_image(
    polygon: np.ndarray,
    image_height: int,
    image_width: int,
) -> tuple[float, float, float, float] | None:
    vertices = [(float(x_coord), float(y_coord)) for x_coord, y_coord in polygon]
    boundaries = ((0, 0.0, True), (0, float(image_width), False), (1, 0.0, True), (1, float(image_height), False))
    for axis, boundary, keep_greater in boundaries:
        if not vertices:
            return None
        clipped: list[tuple[float, float]] = []
        previous_x, previous_y = vertices[-1]
        previous = (previous_x, previous_y)
        previous_value = previous[axis] - boundary
        previous_inside = previous_value >= 0.0 if keep_greater else previous_value <= 0.0
        for current in vertices:
            current_value = current[axis] - boundary
            current_inside = current_value >= 0.0 if keep_greater else current_value <= 0.0
            if current_inside != previous_inside:
                fraction = previous_value / (previous_value - current_value)
                clipped.append(
                    (
                        previous[0] + fraction * (current[0] - previous[0]),
                        previous[1] + fraction * (current[1] - previous[1]),
                    ),
                )
            if current_inside:
                clipped.append(current)
            previous, previous_value, previous_inside = current, current_value, current_inside
        vertices = clipped
    if len(vertices) < 3:
        return None
    double_area = abs(
        sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(vertices, [*vertices[1:], vertices[0]], strict=True)
        ),
    )
    if double_area <= np.finfo(np.float64).eps * image_height * image_width * 8:
        return None
    visible = np.asarray(vertices)
    return (
        float(visible[:, 0].min()),
        float(visible[:, 1].min()),
        float(visible[:, 0].max()),
        float(visible[:, 1].max()),
    )


def _hbb_bbox_envelope(bbox: np.ndarray, image_shape: tuple[int, int]) -> tuple[float, float, float, float] | None:
    image_height, image_width = image_shape
    raw_x_min, raw_y_min, raw_x_max, raw_y_max = bbox[:4]
    if not all(math.isfinite(float(value)) for value in (raw_x_min, raw_y_min, raw_x_max, raw_y_max)):
        return None
    if raw_x_max <= raw_x_min or raw_y_max <= raw_y_min:
        return None
    x_min = min(1.0, max(0.0, float(raw_x_min))) * image_width
    y_min = min(1.0, max(0.0, float(raw_y_min))) * image_height
    x_max = min(1.0, max(0.0, float(raw_x_max))) * image_width
    y_max = min(1.0, max(0.0, float(raw_y_max))) * image_height
    return (x_min, y_min, x_max, y_max) if x_min < x_max and y_min < y_max else None


def _bbox_envelope(
    bbox: np.ndarray,
    bbox_type: Literal["hbb", "obb"],
    image_shape: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    if bbox_type == "hbb":
        return _hbb_bbox_envelope(bbox, image_shape)
    image_height, image_width = image_shape

    coordinates = np.asarray(bbox[:5], dtype=np.float64).copy()
    if not np.isfinite(coordinates).all():
        return None

    coordinates[[0, 2]] *= image_width
    coordinates[[1, 3]] *= image_height
    if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
        return None

    coordinates[4] = math.remainder(float(coordinates[4]), 360.0)
    angle = radians(float(coordinates[4]))
    abs_cos = abs(cos(angle))
    abs_sin = abs(sin(angle))
    half_width = (coordinates[2] - coordinates[0]) * 0.5
    half_height = (coordinates[3] - coordinates[1]) * 0.5
    center_x = (coordinates[0] + coordinates[2]) * 0.5
    center_y = (coordinates[1] + coordinates[3]) * 0.5
    # Support radii give the corner envelope without allocating one polygon for a single bbox.
    extent_x = abs_cos * half_width + abs_sin * half_height
    extent_y = abs_sin * half_width + abs_cos * half_height
    x_min, y_min = center_x - extent_x, center_y - extent_y
    x_max, y_max = center_x + extent_x, center_y + extent_y
    if x_min >= 0.0 and y_min >= 0.0 and x_max <= image_width and y_max <= image_height:
        return float(x_min), float(y_min), float(x_max), float(y_max)

    polygon = obb_to_polygons(coordinates.reshape(1, -1))[0]
    if not np.isfinite(polygon).all():
        return None
    return _clip_polygon_to_image(polygon, image_height, image_width)


def _bbox_views_envelope(
    bbox_views: tuple[Any, ...],
    bbox_type: Literal["hbb", "obb"],
    image_shape: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    envelope = None
    image_height, image_width = image_shape
    for view in bbox_views:
        bboxes = np.asarray(view.value)
        if not len(bboxes):
            continue
        if bbox_type == "hbb":
            x_min = np.clip(bboxes[:, 0], 0.0, 1.0) * image_width
            y_min = np.clip(bboxes[:, 1], 0.0, 1.0) * image_height
            x_max = np.clip(bboxes[:, 2], 0.0, 1.0) * image_width
            y_max = np.clip(bboxes[:, 3], 0.0, 1.0) * image_height
            visible = np.isfinite(bboxes[:, :4]).all(axis=1) & (x_min < x_max) & (y_min < y_max)
            if visible.any():
                envelope = _merge_bbox_envelopes(
                    envelope,
                    (
                        float(x_min[visible].min()),
                        float(y_min[visible].min()),
                        float(x_max[visible].max()),
                        float(y_max[visible].max()),
                    ),
                )
            continue

        coordinates = bboxes[:, :5].astype(np.float64, copy=True)
        coordinates[:, [0, 2]] *= image_width
        coordinates[:, [1, 3]] *= image_height
        coordinates[:, 4] = np.remainder(coordinates[:, 4], 360.0)
        valid = np.isfinite(coordinates).all(axis=1)
        valid &= (coordinates[:, 2] > coordinates[:, 0]) & (coordinates[:, 3] > coordinates[:, 1])
        if not valid.all():
            coordinates[~valid] = 0.0
        polygons = obb_to_polygons(coordinates)
        inside = valid & (polygons[:, :, 0] >= 0).all(axis=1)
        inside &= (polygons[:, :, 0] <= image_width).all(axis=1)
        inside &= (polygons[:, :, 1] >= 0).all(axis=1)
        inside &= (polygons[:, :, 1] <= image_height).all(axis=1)
        if inside.any():
            visible_polygons = polygons[inside]
            envelope = _merge_bbox_envelopes(
                envelope,
                (
                    float(visible_polygons[:, :, 0].min()),
                    float(visible_polygons[:, :, 1].min()),
                    float(visible_polygons[:, :, 0].max()),
                    float(visible_polygons[:, :, 1].max()),
                ),
            )

        for index in np.flatnonzero(valid & ~inside):
            polygon = polygons[index]
            candidate = _clip_polygon_to_image(polygon, image_height, image_width)
            envelope = _merge_bbox_envelopes(envelope, candidate)
    return envelope


def _sample_containing_crop_coords(
    envelope: tuple[float, float, float, float],
    crop_shape: tuple[int, int],
    image_shape: tuple[int, int],
    sampling: SamplingContext,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = envelope
    crop_height, crop_width = crop_shape
    image_height, image_width = image_shape
    x_min = max(0, min(image_width - crop_width, ceil(right - crop_width)))
    y_min = max(0, min(image_height - crop_height, ceil(bottom - crop_height)))
    x_max = min(image_width - crop_width, max(x_min, floor(left)))
    y_max = min(image_height - crop_height, max(y_min, floor(top)))
    crop_x = sampling.py_random.randint(x_min, x_max)
    crop_y = sampling.py_random.randint(y_min, y_max)
    return crop_x, crop_y, crop_x + crop_width, crop_y + crop_height


def _bbox_envelope_for_policy(
    bbox_views: tuple[Any, ...],
    bbox_preservation: Literal["one_box", "all_boxes"] | None,
    bbox_type: Literal["hbb", "obb"],
    image_shape: tuple[int, int],
    sampling: SamplingContext,
) -> tuple[float, float, float, float] | None:
    if bbox_preservation is None:
        return None
    bbox_count = sum(len(view.value) for view in bbox_views)
    if not bbox_count:
        return None
    if bbox_preservation == "all_boxes":
        return _bbox_views_envelope(bbox_views, bbox_type, image_shape)

    selected_index = int(sampling.random_generator.integers(bbox_count))
    for view in bbox_views:
        if selected_index < len(view.value):
            return _bbox_envelope(np.asarray(view.value)[selected_index], bbox_type, image_shape)
        selected_index -= len(view.value)
    return None


def _sample_bbox_preserving_shape(
    image_shape: tuple[int, int],
    min_crop_shape: tuple[int, int],
    scale: tuple[float, float],
    ratio: tuple[float, float],
    sampling_method: SamplingMethod,
    sampling: SamplingContext,
) -> tuple[int, int]:
    crop_shape = sample_2d_crop_shape(
        *image_shape,
        scale,
        ratio,
        sampling_method,
        sampling.py_random,
        min_crop_shape,
    )
    if crop_shape is None:
        relaxed_scale, relaxed_ratio = relax_2d_crop_ranges(*image_shape, scale, ratio, min_crop_shape)
        crop_shape = sample_2d_crop_shape(
            *image_shape,
            relaxed_scale,
            relaxed_ratio,
            sampling_method,
            sampling.py_random,
            min_crop_shape,
        )
    return crop_shape if crop_shape is not None else min_crop_shape


def _sample_random_resized_crop_coords(
    image_shape: tuple[int, int],
    scale: tuple[float, float],
    ratio: tuple[float, float],
    sampling_method: SamplingMethod,
    protected_envelope: tuple[float, float, float, float] | None,
    sampling: SamplingContext,
) -> tuple[int, int, int, int]:
    if protected_envelope is not None:
        left, top, right, bottom = protected_envelope
        min_crop_shape = (ceil(bottom) - floor(top), ceil(right) - floor(left))
        protected_crop_shape = _sample_bbox_preserving_shape(
            image_shape,
            min_crop_shape,
            scale,
            ratio,
            sampling_method,
            sampling,
        )
        return _sample_containing_crop_coords(protected_envelope, protected_crop_shape, image_shape, sampling)

    crop_shape = sample_2d_crop_shape(
        *image_shape,
        scale,
        ratio,
        sampling_method,
        sampling.py_random,
    )
    if crop_shape is None:
        image_height, image_width = image_shape
        in_ratio = image_width / image_height
        if in_ratio < ratio[0]:
            width = image_width
            height = min(image_height, max(1, round(image_width / ratio[0])))
        elif in_ratio > ratio[1]:
            height = image_height
            width = min(image_width, max(1, round(image_height * ratio[1])))
        else:
            width = image_width
            height = image_height
        return fcrops.get_center_crop_coords(image_shape, (height, width))

    return fcrops.get_crop_coords(
        image_shape,
        crop_shape,
        sampling.py_random.random(),
        sampling.py_random.random(),
    )


class RandomSizedCrop(_BaseRandomSizedCrop):
    """Random crop with height in min_max_height and aspect ratio (w2h_ratio), then resize to
    size. Scale and aspect variation with fixed output size.

    This transform first crops a random portion of the input and then resizes it to a specified size.
    The size of the random crop is controlled by the 'min_max_height' parameter.

    Args:
        min_max_height (tuple[int, int]): Minimum and maximum height of the crop in pixels.
        size (tuple[int, int]): Target size for the output image, i.e. (height, width) after crop and resize.
        w2h_ratio (float): Aspect ratio (width/height) of crop. Default: 1.0
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        area_for_downscale (Literal[None, "image", "image_mask"]): Controls automatic use of INTER_AREA interpolation
            for downscaling. Options:
            - None: No automatic interpolation selection, always use the specified interpolation method
            - "image": Use INTER_AREA when downscaling images, retain specified interpolation for upscaling and masks
            - "image_mask": Use INTER_AREA when downscaling both images and masks
            Default: None.
        p (float): Probability of applying the transform. Default: 1.0

    Targets:
        image, images, mask, masks, bboxes, keypoints, volume, volumes, mask3d, masks3d

    Image types:
        uint8, float16, float32


    Supported bboxes:
        hbb, obb
    Note:
        - The crop size is randomly selected for each execution within the range specified by 'min_max_height'.
        - The aspect ratio of the crop is determined by the 'w2h_ratio' parameter.
        - After cropping, the result is resized to the specified 'size'.
        - Bounding boxes that end up fully outside the cropped area will be removed.
        - Keypoints that end up outside the cropped area will be removed.
        - This transform differs from RandomResizedCrop in that it allows more control over the crop size
          through the 'min_max_height' parameter, rather than using a scale parameter.
        - When area_for_downscale is set, INTER_AREA interpolation will be used automatically for
          downscaling (when the crop is larger than the target size), which provides better quality for size reduction.

    Mathematical Details:
        1. A random crop height h is sampled from the range [min_max_height[0], min_max_height[1]].
        2. The crop width w is calculated as: w = h * w2h_ratio
        3. A random location for the crop is selected within the input image.
        4. The image is cropped to the size (h, w).
        5. The crop is then resized to the specified 'size'.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Define transform with parameters as tuples
        >>> transform = A.Compose([
        ...     A.RandomSizedCrop(
        ...         min_max_height=(50, 80),
        ...         size=(64, 64),
        ...         w2h_ratio=1.0,
        ...         interpolation=cv2.INTER_LINEAR,
        ...         mask_interpolation=cv2.INTER_NEAREST,
        ...         area_for_downscale="image",  # Use INTER_AREA for image downscaling
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> transformed_image = transformed['image']       # Shape: (64, 64, 3)
        >>> transformed_mask = transformed['mask']         # Shape: (64, 64)
        >>> transformed_bboxes = transformed['bboxes']     # Bounding boxes adjusted to new crop and size
        >>> transformed_bbox_labels = transformed['bbox_labels']  # Labels for the preserved bboxes
        >>> transformed_keypoints = transformed['keypoints']      # Keypoints adjusted to new crop and size
        >>> transformed_keypoint_labels = transformed['keypoint_labels']  # Labels for the preserved keypoints

    """

    _targets = ALL_TARGETS
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})

    class InitSchema(BaseTransformInitSchema):
        interpolation: FullInterpolationType
        mask_interpolation: FullInterpolationType
        min_max_height: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
            AfterValidator(nondecreasing),
        ]
        w2h_ratio: Annotated[float, Field(gt=0)]
        size: Annotated[tuple[int, int], AfterValidator(check_range_bounds(1, None))]
        area_for_downscale: Literal["image", "image_mask"] | None

    def __init__(
        self,
        min_max_height: tuple[int, int],
        size: tuple[int, int],
        w2h_ratio: float = 1.0,
        interpolation: FullInterpolationType = CV2_INTER_LINEAR,
        mask_interpolation: FullInterpolationType = CV2_INTER_NEAREST,
        area_for_downscale: Literal["image", "image_mask"] | None = None,
        p: float = 1.0,
    ):
        super().__init__(
            size=size,
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            area_for_downscale=area_for_downscale,
            p=p,
        )
        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        image_shape = targets.require_aligned_spatial_shape(2)

        crop_height = sampling.py_random.randint(*self.min_max_height)
        crop_width = int(crop_height * self.w2h_ratio)

        crop_shape = (crop_height, crop_width)

        h_start = sampling.py_random.random()
        w_start = sampling.py_random.random()

        crop_coords = fcrops.get_crop_coords(image_shape, crop_shape, h_start, w_start)

        sampling.applied_overrides["min_max_height"] = (crop_height, crop_height)

        return SampledParams(params={"crop_coords": crop_coords})


class RandomResizedCrop(_BaseRandomSizedCrop):
    """Random crop with scale and ratio ranges (torchvision-style), then resize to size.
    Standard for training on varying resolutions; scale and ratio control crop.

    This transform first crops a random portion of the input image (or mask, bounding boxes, keypoints)
    and then resizes the crop to a specified size. It's particularly useful for training neural networks
    on images of varying sizes and aspect ratios.

    Args:
        size (tuple[int, int]): Target size for the output image, i.e. (height, width) after crop and resize.
        scale (tuple[float, float]): Range of the random size of the crop relative to the input size.
            For example, (0.08, 1.0) means the crop size will be between 8% and 100% of the input size.
            Default: (0.08, 1.0)
        ratio (tuple[float, float]): Range of aspect ratios of the random crop.
            For example, (0.75, 1.3333) allows crop aspect ratios from 3:4 to 4:3.
            Default: (0.75, 1.3333333333333333)
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR
        mask_interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST
        area_for_downscale (Literal[None, "image", "image_mask"]): Controls automatic use of INTER_AREA interpolation
            for downscaling. Options:
            - None: No automatic interpolation selection, always use the specified interpolation method
            - "image": Use INTER_AREA when downscaling images, retain specified interpolation for upscaling and masks
            - "image_mask": Use INTER_AREA when downscaling both images and masks
            Default: None.
        sampling_method (Literal["standard", "uniform_scale"]): Distribution for feasible crop area.
            `"standard"` preserves the conditional distribution of uniform-area and log-aspect-ratio
            proposals. `"uniform_scale"` samples uniformly from the feasible crop-area interval.
            Default: `"standard"`.
        bbox_preservation (Literal["one_box", "all_boxes"] | None): Require the crop to contain one uniformly
            selected bbox from all active bbox fields, or all boxes from those fields. Default: None.
        p (float): Probability of applying the transform. Default: 1.0

    Targets:
        image, images, mask, masks, bboxes, keypoints, volume, volumes, mask3d, masks3d

    Image types:
        uint8, float16, float32


    Supported bboxes:
        hbb, obb
    Note:
        - Crop dimensions are sampled directly from the feasible region. A centered crop is used
          only when that region is empty.
        - The crop's aspect ratio is defined as width / height.
        - Bounding boxes that end up fully outside the cropped area will be removed.
        - When `bbox_preservation` is set, bbox-aware sampling uses all active bbox fields, including aliases.
          If no boxes are present, ordinary random-resized-crop sampling is used.
        - Bbox preservation keeps visible geometry inside the input image. `scale` and `ratio` are relaxed only when
          needed, in that priority order, and BboxParams filters continue to apply after the resize.
        - Keypoints that end up outside the cropped area will be removed.
        - After cropping, the result is resized to the specified size.
        - When area_for_downscale is set, INTER_AREA interpolation will be used automatically for
          downscaling (when the crop is larger than the target size), which provides better quality for size reduction.

    Mathematical Details:
        1. A target area A and log-aspect ratio r are sampled from their feasible joint region.
           `sampling_method="standard"` preserves the conditional law of uniform area and
           log-ratio proposals; `"uniform_scale"` makes A uniform on its feasible interval.
        2. The crop width and height are computed as:
           w = sqrt(A * r)
           h = sqrt(A / r)
        3. The crop is then resized to the specified size.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Prepare sample data
        >>> rng = np.random.default_rng(137)
        >>> image = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = rng.integers(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Define transform with parameters as tuples
        >>> transform = A.Compose([
        ...     A.RandomResizedCrop(
        ...         size=(64, 64),
        ...         bbox_preservation="one_box",
        ...         scale=(0.5, 0.9),  # Crop size will be 50-90% of original image
        ...         ratio=(0.75, 1.33),  # Aspect ratio will vary from 3:4 to 4:3
        ...         interpolation=cv2.INTER_LINEAR,
        ...         mask_interpolation=cv2.INTER_NEAREST,
        ...         area_for_downscale="image",  # Use INTER_AREA for image downscaling
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']),
        ...    seed=137)
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> transformed_image = transformed['image']       # Shape: (64, 64, 3)
        >>> transformed_mask = transformed['mask']         # Shape: (64, 64)
        >>> transformed_bboxes = transformed['bboxes']     # Bounding boxes adjusted to new crop and size
        >>> transformed_bbox_labels = transformed['bbox_labels']  # Labels for the preserved bboxes
        >>> transformed_keypoints = transformed['keypoints']      # Keypoints adjusted to new crop and size
        >>> transformed_keypoint_labels = transformed['keypoint_labels']  # Labels for the preserved keypoints

    """

    _targets = ALL_TARGETS
    _supported_bbox_types: frozenset[str] = frozenset({"hbb", "obb"})

    class InitSchema(BaseTransformInitSchema):
        scale: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, 1)), AfterValidator(nondecreasing)]
        ratio: Annotated[
            tuple[float, float],
            AfterValidator(_validate_positive_finite_ratio_range),
            AfterValidator(nondecreasing),
        ]
        size: Annotated[tuple[int, int], AfterValidator(check_range_bounds(1, None))]
        interpolation: FullInterpolationType
        mask_interpolation: FullInterpolationType
        area_for_downscale: Literal["image", "image_mask"] | None
        sampling_method: SamplingMethod
        bbox_preservation: Literal["one_box", "all_boxes"] | None

    def __init__(
        self,
        size: tuple[int, int],
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        interpolation: FullInterpolationType = CV2_INTER_LINEAR,
        mask_interpolation: FullInterpolationType = CV2_INTER_NEAREST,
        area_for_downscale: Literal["image", "image_mask"] | None = None,
        p: float = 1.0,
        *,
        sampling_method: SamplingMethod = "standard",
        bbox_preservation: Literal["one_box", "all_boxes"] | None = None,
    ):
        super().__init__(
            size=size,
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            area_for_downscale=area_for_downscale,
            p=p,
        )
        self.scale = scale
        self.ratio = ratio
        self.sampling_method = sampling_method
        self.bbox_preservation = bbox_preservation

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        image_shape = targets.require_aligned_spatial_shape(2)
        image_height, image_width = image_shape
        bbox_type: Literal["hbb", "obb"] = params.get("bbox_type", "hbb")
        bbox_views = targets.by_canonical_type("bboxes") if self.bbox_preservation is not None else ()
        bbox_envelope = _bbox_envelope_for_policy(
            bbox_views,
            self.bbox_preservation,
            bbox_type,
            image_shape,
            sampling,
        )
        crop_coords = _sample_random_resized_crop_coords(
            image_shape,
            self.scale,
            self.ratio,
            self.sampling_method,
            bbox_envelope,
            sampling,
        )

        crop_height = crop_coords[3] - crop_coords[1]
        crop_width = crop_coords[2] - crop_coords[0]

        sampled_scale = (crop_width * crop_height) / (image_height * image_width)
        sampled_ratio = crop_width / crop_height
        sampling.applied_overrides.update(
            {
                "scale": (sampled_scale, sampled_scale),
                "ratio": (sampled_ratio, sampled_ratio),
            },
        )
        return SampledParams(params={"crop_coords": crop_coords})


__all__ = [
    "RandomResizedCrop",
    "RandomSizedCrop",
]
