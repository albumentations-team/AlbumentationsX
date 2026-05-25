"""Implementation of XY masking for time-frequency domain transformations.
This module provides the XYMasking transform, which applies masking strips along the X and Y axes
of an image. This is particularly useful for audio spectrograms, time-series data visualizations,
and other grid-like data representations where masking in specific directions (time or frequency)
can improve model robustness and generalization.
"""

from typing import Annotated, Any, ClassVar, Literal

import numpy as np
from pydantic import model_validator
from pydantic.functional_validators import AfterValidator
from typing_extensions import Self

from albumentations.augmentations.dropout.transforms import BaseDropout
from albumentations.core.pydantic import (
    check_range_bounds,
    nondecreasing,
)
from albumentations.core.transforms_interface import BaseTransformInitSchema

__all__ = ["XYMasking"]


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
    mask_x_length_range: Annotated[
        tuple[int, int],
        AfterValidator(check_range_bounds(0)),
        AfterValidator(nondecreasing),
    ]
    mask_y_length_range: Annotated[
        tuple[int, int],
        AfterValidator(check_range_bounds(0)),
        AfterValidator(nondecreasing),
    ]

    fill: tuple[float, ...] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"]
    fill_mask: tuple[float, ...] | float | None

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
    `max_x_length` or `max_y_length` must be specified, dictating the mask's
    maximum size along each axis.

    Args:
        num_masks_x_range (tuple[int, int]): Range of horizontal regions to mask. Defaults to (0, 0).
        num_masks_y_range (tuple[int, int]): Range of vertical regions to mask. Defaults to (0, 0).
        mask_x_length_range (tuple[int, int]): Range (min, max) of mask length along the X (horizontal)
            axis. The length is randomly chosen within this range for each mask. Defaults to (0, 0).
        mask_y_length_range (tuple[int, int]): Range (min, max) of mask height along the Y (vertical)
            axis. The height is randomly chosen within this range for each mask. Defaults to (0, 0).
        fill (tuple[float, float] | float | Literal['random', 'random_uniform', 'inpaint_telea', 'inpaint_ns']):
            Value for the dropped pixels. Can be:
            - int or float: all channels are filled with this value
            - tuple: tuple of values for each channel
            - 'random': each pixel is filled with random values
            - 'random_uniform': each hole is filled with a single random color
            - 'inpaint_telea': uses OpenCV Telea inpainting method
            - 'inpaint_ns': uses OpenCV Navier-Stokes inpainting method
            Default: 0
        fill_mask (tuple[float, float] | float | None): Fill value for dropout regions in the mask.
            If None, mask regions corresponding to image dropouts are unchanged. Default: None
        p (float): Probability of applying the transform. Defaults to 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Supported bboxes:
        hbb

    Note:
        Either `mask_x_length_range` or `mask_y_length_range` or both must have a positive max.

    """

    InitSchema: ClassVar[type[BaseTransformInitSchema]] = _XYMaskingInitSchema

    def __init__(
        self,
        num_masks_x_range: tuple[int, int] = (0, 0),
        num_masks_y_range: tuple[int, int] = (0, 0),
        mask_x_length_range: tuple[int, int] = (0, 0),
        mask_y_length_range: tuple[int, int] = (0, 0),
        fill: tuple[float, ...] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"] = 0,
        fill_mask: tuple[float, ...] | float | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, fill=fill, fill_mask=fill_mask)
        self.num_masks_x_range = num_masks_x_range
        self.num_masks_y_range = num_masks_y_range

        self.mask_x_length_range = mask_x_length_range
        self.mask_y_length_range = mask_y_length_range

    def _validate_mask_length(
        self,
        mask_length: tuple[int, int],
        dimension_size: int,
        dimension_name: str,
    ) -> None:
        """Validate mask length for XYMasking. Raises if mismatch with image dimension. Used when
        applying horizontal/vertical masks.
        """
        if mask_length[0] < 0 or mask_length[1] > dimension_size:
            raise ValueError(
                f"{dimension_name} range {mask_length} is out of valid range [0, {dimension_size}]",
            )

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, int | np.ndarray]:
        image_shape = params["shape"][:2]

        height, width = image_shape

        self._validate_mask_length(self.mask_x_length_range, width, "mask_x_length_range")
        self._validate_mask_length(self.mask_y_length_range, height, "mask_y_length_range")

        masks_x = self._generate_masks(self.num_masks_x_range, image_shape, self.mask_x_length_range, axis="x")
        masks_y = self._generate_masks(self.num_masks_y_range, image_shape, self.mask_y_length_range, axis="y")

        holes = np.array(masks_x + masks_y)

        self.applied_config = {
            "num_masks_x_range": len(masks_x),
            "num_masks_y_range": len(masks_y),
            "mask_x_length_range": self.mask_x_length_range,
            "mask_y_length_range": self.mask_y_length_range,
        }

        return {"holes": holes, "seed": int(self.random_generator.integers(0, 2**32 - 1))}

    def _generate_mask_size(self, mask_length: tuple[int, int]) -> int:
        return self.py_random.randint(*mask_length)

    def _generate_masks(
        self,
        num_masks: tuple[int, int],
        image_shape: tuple[int, int],
        max_length: tuple[int, int],
        axis: str,
    ) -> list[tuple[int, int, int, int]]:
        if max_length[1] == 0 or num_masks[1] == 0:
            return []

        masks = []
        num_masks_integer = self.py_random.randint(num_masks[0], num_masks[1])

        height, width = image_shape

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
