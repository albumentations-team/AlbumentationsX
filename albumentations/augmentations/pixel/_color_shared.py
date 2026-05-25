"""Shared imports for split modules."""

import warnings
from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal, cast

import albucore
import cv2
import numpy as np
from albucore import (
    MAX_VALUES_BY_DTYPE,
    batch_transform,
    get_num_channels,
    is_grayscale_image,
    is_rgb_image,
    mean,
)
from pydantic import Field, field_validator, model_validator
from pydantic.functional_validators import AfterValidator
from typing_extensions import Self

from albumentations.augmentations.pixel import functional as fpixel
from albumentations.augmentations.pixel.noise import AdditiveNoise
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
    CV2_INTER_LINEAR,
    NUM_RGB_CHANNELS,
    PAIR,
    SEVEN,
    FullInterpolationType,
    ImageType,
    VolumeType,
)

__all__ = [
    "CV2_INTER_LINEAR",
    "MAX_VALUES_BY_DTYPE",
    "NUM_RGB_CHANNELS",
    "PAIR",
    "SEVEN",
    "AdditiveNoise",
    "AfterValidator",
    "Annotated",
    "Any",
    "BaseTransformInitSchema",
    "Callable",
    "Field",
    "FullInterpolationType",
    "ImageOnlyTransform",
    "ImageType",
    "Literal",
    "Self",
    "Sequence",
    "VolumeType",
    "albucore",
    "batch_transform",
    "cast",
    "check_range_bounds",
    "cv2",
    "field_validator",
    "fpixel",
    "get_num_channels",
    "is_grayscale_image",
    "is_rgb_image",
    "mean",
    "model_validator",
    "non_rgb_error",
    "nondecreasing",
    "np",
    "warnings",
]
