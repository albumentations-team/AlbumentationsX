"""Implementation of XY masking for time-frequency domain transformations.
This module provides the XYMasking transform, which applies masking strips along the X and Y axes
of an image. This is particularly useful for audio spectrograms, time-series data visualizations,
and other grid-like data representations where masking in specific directions (time or frequency)
can improve model robustness and generalization.
"""

from typing import Annotated, Any, ClassVar, Literal, cast

import numpy as np
from pydantic import model_validator
from pydantic.functional_validators import AfterValidator, BeforeValidator
from typing_extensions import Self

from albumentations.augmentations.dropout.transforms import BaseDropout, DropoutFillValue
from albumentations.core.invocation import SamplingContext
from albumentations.core.pydantic import (
    check_range_bounds,
    nondecreasing,
)
from albumentations.core.transform_params import SampledParams, TargetSet
from albumentations.core.transforms_interface import BaseTransformInitSchema

__all__ = ["XYMasking"]

MaskLengthRange = tuple[int | float, int | float]


def _normalize_mask_length_range(value: Any) -> MaskLengthRange:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("Mask length range must be a tuple or list of exactly two values")

    endpoint_type = type(value[0])
    if endpoint_type not in (int, float) or any(type(endpoint) is not endpoint_type for endpoint in value):
        raise ValueError("Mask length range elements must be all integers or all floats")
    return value[0], value[1]


def _validate_mask_length_bounds(value: MaskLengthRange) -> MaskLengthRange:
    max_value = None if type(value[0]) is int else 1
    check_range_bounds(0, max_value)(value)
    return value


ValidatedMaskLengthRange = Annotated[
    MaskLengthRange,
    BeforeValidator(_normalize_mask_length_range),
    AfterValidator(_validate_mask_length_bounds),
    AfterValidator(nondecreasing),
]


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
    mask_x_length_range: ValidatedMaskLengthRange
    mask_y_length_range: ValidatedMaskLengthRange

    fill: DropoutFillValue
    fill_mask: tuple[float, ...] | float | None

    @model_validator(mode="after")
    def _check_mask_length(self) -> Self:
        if self.mask_x_length_range[1] <= 0 and self.mask_y_length_range[1] <= 0:
            msg = "At least one of `mask_x_length_range` or `mask_y_length_range` must have a positive max value."
            raise ValueError(msg)

        return self


class XYMasking(BaseDropout):
    """Mask images with random horizontal and vertical strips sized in pixels or axis-relative fractions,
    useful for occlusion and spectrogram augmentation.

    Integer endpoints express strip sizes in pixels. Float endpoints in [0.0, 1.0] scale X lengths by image width
    and Y lengths by image height.

    Args:
        num_masks_x_range (tuple[int, int]): Range for the number of vertical strips. Defaults to (0, 0).
        num_masks_y_range (tuple[int, int]): Range for the number of horizontal strips. Defaults to (0, 0).
        mask_x_length_range (tuple[int | float, int | float]): Range of vertical-strip widths. Use two integers for
            pixels or two floats in [0.0, 1.0] for fractions of image width. Defaults to (0, 0).
        mask_y_length_range (tuple[int | float, int | float]): Range of horizontal-strip heights. Use two integers for
            pixels or two floats in [0.0, 1.0] for fractions of image height. Defaults to (0, 0).
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
            If None, mask regions corresponding to image dropouts are unchanged. Defaults to None.
        p (float): Probability of applying the transform. Defaults to 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Supported bboxes:
        hbb

    Note:
        - Each length range must contain either two integers or two floats; mixed endpoint types are invalid.
        - Integer lengths are sampled inclusively in pixels. Float lengths are sampled between the endpoints, scaled
          by the corresponding axis, and rounded down. Zero-length strips are omitted, and 1.0 can span the full axis.
        - At least one of `mask_x_length_range` and `mask_y_length_range` must have a positive maximum.
        - When using `fill="grayscale"`, `fill_mask` must be None.

    Examples:
        Use integer ranges for pixel-based strip sizes:

        >>> import albumentations as A
        >>> import numpy as np
        >>> image = np.full((80, 120, 3), 255, dtype=np.uint8)
        >>> pixel_transform = A.XYMasking(
        ...     num_masks_x_range=(1, 2),
        ...     num_masks_y_range=(1, 1),
        ...     mask_x_length_range=(8, 16),
        ...     mask_y_length_range=(4, 8),
        ...     fill=0,
        ...     p=1.0,
        ... )
        >>> pixel_image = pixel_transform(image=image)["image"]

        Use float ranges when strip sizes should scale with each input:

        >>> relative_transform = A.XYMasking(
        ...     num_masks_x_range=(1, 2),
        ...     num_masks_y_range=(1, 1),
        ...     mask_x_length_range=(0.05, 0.15),
        ...     mask_y_length_range=(0.1, 0.2),
        ...     fill=0,
        ...     p=1.0,
        ... )
        >>> relative_image = relative_transform(image=image)["image"]

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
        self._mask_x_length_is_integer = type(mask_x_length_range[0]) is int
        self._mask_y_length_is_integer = type(mask_y_length_range[0]) is int

    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        def materialize(image_shape: tuple[int, int], _: tuple[Any, ...]) -> dict[str, Any]:
            self._validate_integer_axis_ranges(image_shape)

            masks_x = self._generate_axis_masks(
                self.num_masks_x_range,
                image_shape,
                self.mask_x_length_range,
                axis="x",
                sampling=sampling,
            )
            masks_y = self._generate_axis_masks(
                self.num_masks_y_range,
                image_shape,
                self.mask_y_length_range,
                axis="y",
                sampling=sampling,
            )

            rectangles = masks_x + masks_y
            holes = np.array(rectangles) if rectangles else np.empty((0, 4), dtype=np.int32)

            sampling.applied_overrides.update(
                {
                    "num_masks_x_range": len(masks_x),
                    "num_masks_y_range": len(masks_y),
                    "mask_x_length_range": self.mask_x_length_range,
                    "mask_y_length_range": self.mask_y_length_range,
                    "fill": self.fill,
                    "fill_mask": self.fill_mask,
                },
            )
            return {"holes": holes, "seed": int(sampling.random_generator.integers(0, 2**32 - 1))}

        return self._build_spatial_parameter_plan(targets, materialize)

    def _validate_integer_axis_ranges(self, image_shape: tuple[int, int]) -> None:
        if self._mask_x_length_is_integer and self.mask_x_length_range[1] > image_shape[1]:
            raise ValueError(
                f"mask_x_length_range range {self.mask_x_length_range} is out of valid range [0, {image_shape[1]}]",
            )
        if self._mask_y_length_is_integer and self.mask_y_length_range[1] > image_shape[0]:
            raise ValueError(
                f"mask_y_length_range range {self.mask_y_length_range} is out of valid range [0, {image_shape[0]}]",
            )

    def _generate_axis_masks(
        self,
        num_masks: tuple[int, int],
        image_shape: tuple[int, int],
        mask_length_range: MaskLengthRange,
        axis: Literal["x", "y"],
        sampling: SamplingContext,
    ) -> list[tuple[int, int, int, int]]:
        first_length = mask_length_range[0]
        if type(first_length) is int:
            return self._generate_masks(
                num_masks,
                image_shape,
                cast("tuple[int, int]", mask_length_range),
                axis,
                sampling,
            )

        return self._generate_relative_masks(
            num_masks,
            image_shape,
            cast("tuple[float, float]", mask_length_range),
            axis,
            sampling,
        )

    def _generate_mask_size(self, mask_length: tuple[int, int], sampling: SamplingContext) -> int:
        return sampling.py_random.randint(*mask_length)

    def _generate_masks(
        self,
        num_masks: tuple[int, int],
        image_shape: tuple[int, int],
        max_length: tuple[int, int],
        axis: Literal["x", "y"],
        sampling: SamplingContext,
    ) -> list[tuple[int, int, int, int]]:
        height, width = image_shape
        if max_length[1] == 0 or num_masks[1] == 0:
            return []

        masks = []
        num_masks_integer = sampling.py_random.randint(num_masks[0], num_masks[1])

        for _ in range(num_masks_integer):
            length = self._generate_mask_size(max_length, sampling)

            if axis == "x":
                x_min = sampling.py_random.randint(0, width - length)
                y_min = 0
                x_max, y_max = x_min + length, height
            else:  # axis == 'y'
                y_min = sampling.py_random.randint(0, height - length)
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
        sampling: SamplingContext,
    ) -> list[tuple[int, int, int, int]]:
        if max_length[1] == 0 or num_masks[1] == 0:
            return []

        masks = []
        num_masks_integer = sampling.py_random.randint(num_masks[0], num_masks[1])

        height, width = image_shape

        dimension_size = width if axis == "x" else height
        for _ in range(num_masks_integer):
            length = int(sampling.py_random.uniform(*max_length) * dimension_size)

            if axis == "x":
                x_min = sampling.py_random.randint(0, width - length)
                y_min = 0
                x_max, y_max = x_min + length, height
            else:  # axis == 'y'
                y_min = sampling.py_random.randint(0, height - length)
                x_min = 0
                x_max, y_max = width, y_min + length

            if length > 0:
                masks.append((x_min, y_min, x_max, y_max))
        return masks
