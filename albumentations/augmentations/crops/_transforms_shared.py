"""Shared imports for split modules."""

import math
from collections.abc import Sequence
from typing import Annotated, Any, Literal, cast

import cv2
import numpy as np
from albucore import reduce_sum
from pydantic import Field, model_validator
from pydantic.functional_validators import AfterValidator
from typing_extensions import Self

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes, union_of_bboxes
from albumentations.core.pydantic import (
    check_range_bounds,
    nondecreasing,
)
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.type_definitions import (
    ALL_TARGETS,
    CV2_BORDER_CONSTANT,
    CV2_INTER_LINEAR,
    CV2_INTER_NEAREST,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    PAIR,
    BorderModeType,
    FullInterpolationType,
    ImageType,
    PercentType,
    PxType,
    StackedMasks4D,
    VolumeType,
)

from . import functional as fcrops

__all__ = [
    "ALL_TARGETS",
    "CV2_BORDER_CONSTANT",
    "CV2_INTER_LINEAR",
    "CV2_INTER_NEAREST",
    "NUM_MULTI_CHANNEL_DIMENSIONS",
    "PAIR",
    "AfterValidator",
    "Annotated",
    "Any",
    "BaseTransformInitSchema",
    "BorderModeType",
    "DualTransform",
    "Field",
    "FullInterpolationType",
    "ImageType",
    "Literal",
    "PercentType",
    "PxType",
    "Self",
    "Sequence",
    "StackedMasks4D",
    "VolumeType",
    "cast",
    "check_range_bounds",
    "cv2",
    "denormalize_bboxes",
    "fcrops",
    "fgeometric",
    "math",
    "model_validator",
    "nondecreasing",
    "normalize_bboxes",
    "np",
    "reduce_sum",
    "union_of_bboxes",
]
