"""Implementation of XY masking for time-frequency domain transformations.
This module provides the XYMasking transform, which applies masking strips along the X and Y axes
of an image. This is particularly useful for audio spectrograms, time-series data visualizations,
and other grid-like data representations where masking in specific directions (time or frequency)
can improve model robustness and generalization.
"""

import math
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from numbers import Integral, Real
from typing import Annotated, Any, ClassVar, Literal, cast

import numpy as np
from pydantic import field_validator, model_validator
from pydantic.functional_validators import AfterValidator
from pydantic_core import PydanticCustomError
from typing_extensions import Self

from albumentations.augmentations.dropout.transforms import BaseDropout, DropoutFillValue
from albumentations.core.pydantic import (
    check_range_bounds,
    nondecreasing,
)
from albumentations.core.transforms_interface import BaseTransformInitSchema

__all__ = ["XYMasking"]

MaskLengthRange = tuple[int, int] | tuple[float, float]


def _materialize_range(value: Any) -> tuple[Any, Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping, AbstractSet)):
        raise PydanticCustomError(
            "mask_length_range_type",
            "Mask length range must be an ordered iterable of two values",
        )

    try:
        elements = tuple(value)
    except TypeError as exc:
        raise PydanticCustomError(
            "mask_length_range_type",
            "Mask length range must be an ordered iterable of two values",
        ) from exc

    if len(elements) != 2:
        raise ValueError("Mask length range must contain exactly two values")
    return elements[0], elements[1]


def _normalize_integer_range(elements: tuple[Any, Any]) -> tuple[int, int]:
    if elements[0] < 0:
        raise ValueError("Integer mask length ranges must contain only nonnegative values")
    if elements[0] > elements[1]:
        raise ValueError("Mask length range must be nondecreasing")
    return int(elements[0]), int(elements[1])


def _normalize_float_range(elements: tuple[Any, Any]) -> tuple[float, float]:
    if not all(math.isfinite(element) for element in elements):
        raise ValueError("Float mask length ranges must contain only finite values")
    if any(element < 0 or element > 1 for element in elements):
        raise ValueError("Float mask length ranges must be within [0.0, 1.0]")
    if elements[0] > elements[1]:
        raise ValueError("Mask length range must be nondecreasing")
    return float(elements[0]), float(elements[1])


class _XYMaskingInitSchema(BaseTransformInitSchema):
    num_masks_x_range: Annotated[
        tuple[int, int],
        AfterValidator(check_range_bounds(0)),
        AfterValidator(nondecreasing),
    ]
    num_masks_y_range: Annotated[
        tuple[int, int],
        AfterValidator(check_range_bounds(0)),
        AfterValidator(nondecreasing),
    ]
    mask_x_length_range: MaskLengthRange
    mask_y_length_range: MaskLengthRange

    fill: DropoutFillValue
    fill_mask: tuple[float, ...] | float | None

    @field_validator("mask_x_length_range", "mask_y_length_range", mode="before")
    @classmethod
    def _validate_mask_length_range(cls, value: Any) -> MaskLengthRange:
        elements = _materialize_range(value)

        if all(isinstance(element, Integral) and not isinstance(element, bool) for element in elements):
            return _normalize_integer_range(elements)

        if all(isinstance(element, Real) and not isinstance(element, (Integral, bool)) for element in elements):
            return _normalize_float_range(elements)

        raise ValueError("Mask length range elements must be all integers or all floats")

    @model_validator(mode="after")
    def _check_mask_length(self) -> Self:
        if self.mask_x_length_range[1] <= 0 and self.mask_y_length_range[1] <= 0:
            msg = "At least one of `mask_x_length_range` or `mask_y_length_range` must have a positive max value."
            raise ValueError(msg)

        return self


class XYMasking(BaseDropout):
    """Apply horizontal or vertical masking strips to simulate occlusion.
    Useful for spectrograms (spectral/frequency masking).

    Useful for training with varied visibility conditions; spectral and frequency
    masking can improve model robustness (e.g. SpecAugment-style). At least one of
    `mask_x_length_range` or `mask_y_length_range` must have a positive maximum,
    dictating the mask's maximum size along each axis.

    Args:
        num_masks_x_range (tuple[int, int]): Range of vertical strips to mask. Defaults to (0, 0).
        num_masks_y_range (tuple[int, int]): Range of horizontal strips to mask. Defaults to (0, 0).
        mask_x_length_range (tuple[int, int] | tuple[float, float]): Range (min, max) of mask length along the X
            (horizontal) axis. Integer values specify pixels. Float values specify fractions of the image width and
            must be within [0.0, 1.0]. The length is randomly chosen within this range for each mask. Defaults to
            (0, 0).
        mask_y_length_range (tuple[int, int] | tuple[float, float]): Range (min, max) of mask height along the Y
            (vertical) axis. Integer values specify pixels. Float values specify fractions of the image height and
            must be within [0.0, 1.0]. The height is randomly chosen within this range for each mask. Defaults to
            (0, 0).
        fill (float | tuple[float, ...] | str):
            Value for the dropped pixels. Can be:
            - int or float: all channels are filled with this value
            - tuple: tuple of values for each channel
            - 'random': each pixel is filled with random values
            - 'random_uniform': each hole is filled with a single random color
            - 'inpaint_telea': uses OpenCV Telea inpainting method
            - 'inpaint_ns': uses OpenCV Navier-Stokes inpainting method
            - 'grayscale': converts dropped regions to grayscale while preserving channel count
            Default: 0
        fill_mask (tuple[float, ...] | float | None): Fill value for dropout regions in the mask.
            If None, mask regions corresponding to image dropouts are unchanged. Default: None
        p (float): Probability of applying the transform. Defaults to 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Supported bboxes:
        hbb

    Note:
        - Either `mask_x_length_range` or `mask_y_length_range` or both must have a positive max.
        - Fractional lengths are sampled uniformly, multiplied by the corresponding image dimension, and rounded down
          with `int`. A positive fractional sample can therefore produce a zero-length, no-op mask on a small image;
          such degenerate masks are discarded. A fractional endpoint of 1.0 can span the full axis.
        - `Compose.from_applied_transforms()` reconstructs a runnable length policy and samples new holes.
          Use `ReplayCompose` when the exact holes from one call must be replayed.
        - When using `fill="grayscale"`, `fill_mask` must be None.

    Examples:
        Pixel-based ranges keep their existing inclusive integer sampling behavior:

        >>> import albumentations as A
        >>> import numpy as np
        >>> image = np.ones((80, 120, 3), dtype=np.uint8) * 255
        >>> mask = np.ones((80, 120), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 40, 40]], dtype=np.float32)
        >>> bbox_labels = [1]
        >>> keypoints = np.array([[20, 30]], dtype=np.float32)
        >>> keypoint_labels = [0]
        >>> pixel_transform = A.Compose(
        ...     [
        ...         A.XYMasking(
        ...             num_masks_x_range=(1, 2),
        ...             num_masks_y_range=(1, 1),
        ...             mask_x_length_range=(8, 16),
        ...             mask_y_length_range=(4, 8),
        ...             fill=0,
        ...             p=1.0,
        ...         ),
        ...     ],
        ...     bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["bbox_labels"]),
        ...     keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["keypoint_labels"]),
        ...     seed=137,
        ... )
        >>> pixel_result = pixel_transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels,
        ... )
        >>> pixel_image = pixel_result["image"]
        >>> pixel_mask = pixel_result["mask"]
        >>> pixel_bboxes = pixel_result["bboxes"]
        >>> pixel_bbox_labels = pixel_result["bbox_labels"]
        >>> pixel_keypoints = pixel_result["keypoints"]
        >>> pixel_keypoint_labels = pixel_result["keypoint_labels"]

        Fractional ranges scale X by width and Y by height, so the same pipeline adapts to each input resolution:

        >>> relative_transform = A.Compose(
        ...     [
        ...         A.XYMasking(
        ...             num_masks_x_range=(1, 2),
        ...             num_masks_y_range=(1, 1),
        ...             mask_x_length_range=(0.05, 0.15),
        ...             mask_y_length_range=(0.1, 0.2),
        ...             fill=0,
        ...             p=1.0,
        ...         ),
        ...     ],
        ...     seed=137,
        ... )
        >>> small_relative_image = relative_transform(image=np.ones((40, 60, 3), dtype=np.uint8))["image"]
        >>> large_relative_image = relative_transform(image=np.ones((80, 120, 3), dtype=np.uint8))["image"]

    """

    InitSchema: ClassVar[type[BaseTransformInitSchema]] = _XYMaskingInitSchema

    def __init__(
        self,
        num_masks_x_range: tuple[int, int] = (0, 0),
        num_masks_y_range: tuple[int, int] = (0, 0),
        mask_x_length_range: MaskLengthRange = (0, 0),
        mask_y_length_range: MaskLengthRange = (0, 0),
        fill: DropoutFillValue = 0,
        fill_mask: tuple[float, ...] | float | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, fill=fill, fill_mask=fill_mask)
        self.num_masks_x_range = num_masks_x_range
        self.num_masks_y_range = num_masks_y_range

        self.mask_x_length_range = mask_x_length_range
        self.mask_y_length_range = mask_y_length_range

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, int | np.ndarray]:
        image_shape = params["shape"][:2]

        masks_x = self._generate_axis_masks(
            self.num_masks_x_range,
            image_shape,
            self.mask_x_length_range,
            axis="x",
        )
        masks_y = self._generate_axis_masks(
            self.num_masks_y_range,
            image_shape,
            self.mask_y_length_range,
            axis="y",
        )

        rectangles = masks_x + masks_y
        holes = np.array(rectangles) if rectangles else np.empty((0, 4), dtype=np.int32)

        self.applied_config = {
            "num_masks_x_range": len(masks_x),
            "num_masks_y_range": len(masks_y),
            "mask_x_length_range": self.mask_x_length_range,
            "mask_y_length_range": self.mask_y_length_range,
            "fill": self.fill,
            "fill_mask": self.fill_mask,
        }

        return {"holes": holes, "seed": int(self.random_generator.integers(0, 2**32 - 1))}

    def _generate_axis_masks(
        self,
        num_masks: tuple[int, int],
        image_shape: tuple[int, int],
        mask_length_range: MaskLengthRange,
        axis: Literal["x", "y"],
    ) -> list[tuple[int, int, int, int]]:
        first_length = mask_length_range[0]
        if type(first_length) is int or isinstance(first_length, Integral):
            return self._generate_masks(
                num_masks,
                image_shape,
                cast("tuple[int, int]", mask_length_range),
                axis,
            )

        return self._generate_relative_masks(
            num_masks,
            image_shape,
            cast("tuple[float, float]", mask_length_range),
            axis,
        )

    def _generate_mask_size(self, mask_length: tuple[int, int]) -> int:
        return self.py_random.randint(*mask_length)

    def _generate_masks(
        self,
        num_masks: tuple[int, int],
        image_shape: tuple[int, int],
        max_length: tuple[int, int],
        axis: Literal["x", "y"],
    ) -> list[tuple[int, int, int, int]]:
        height, width = image_shape
        dimension_size = width if axis == "x" else height
        if max_length[0] < 0 or max_length[1] > dimension_size:
            dimension_name = f"mask_{axis}_length_range"
            raise ValueError(
                f"{dimension_name} range {max_length} is out of valid range [0, {dimension_size}]",
            )

        if max_length[1] == 0 or num_masks[1] == 0:
            return []

        masks = []
        num_masks_integer = self.py_random.randint(num_masks[0], num_masks[1])

        for _ in range(num_masks_integer):
            length = self._generate_mask_size(max_length)

            if axis == "x":
                x_min = self.py_random.randint(0, width - length)
                y_min = 0
                x_max, y_max = x_min + length, height
            else:  # axis == 'y'
                y_min = self.py_random.randint(0, height - length)
                x_min = 0
                x_max, y_max = width, y_min + length

            masks.append((x_min, y_min, x_max, y_max))
        return masks

    def _generate_relative_masks(
        self,
        num_masks: tuple[int, int],
        image_shape: tuple[int, int],
        max_length: tuple[float, float],
        axis: Literal["x", "y"],
    ) -> list[tuple[int, int, int, int]]:
        if max_length[1] == 0 or num_masks[1] == 0:
            return []

        masks = []
        num_masks_integer = self.py_random.randint(num_masks[0], num_masks[1])

        height, width = image_shape

        dimension_size = width if axis == "x" else height
        for _ in range(num_masks_integer):
            length = int(self.py_random.uniform(*max_length) * dimension_size)

            if axis == "x":
                x_min = self.py_random.randint(0, width - length)
                y_min = 0
                x_max, y_max = x_min + length, height
            else:  # axis == 'y'
                y_min = self.py_random.randint(0, height - length)
                x_min = 0
                x_max, y_max = width, y_min + length

            if length > 0:
                masks.append((x_min, y_min, x_max, y_max))
        return masks
