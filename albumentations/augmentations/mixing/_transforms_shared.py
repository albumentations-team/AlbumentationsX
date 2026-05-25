"""Shared imports for split modules."""

import random
import warnings
from collections.abc import Sequence
from copy import deepcopy
from typing import Annotated, Any, ClassVar, Literal, cast

import cv2
import numpy as np
from pydantic import AfterValidator, Field, model_validator
from typing_extensions import Self

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.mixing import functional as fmixing
from albumentations.core.bbox_utils import (
    BboxProcessor,
    check_bboxes,
    convert_bboxes_from_albumentations,
    convert_bboxes_to_albumentations,
    denormalize_bboxes,
    filter_bboxes_with_mask,
)
from albumentations.core.composition import _BBOX_INSTANCE_ID, _KP_INSTANCE_ID
from albumentations.core.keypoints_utils import (
    KeypointsProcessor,
    convert_keypoints_from_albumentations,
    convert_keypoints_to_albumentations,
)
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.type_definitions import (
    CV2_INTER_LINEAR,
    CV2_INTER_NEAREST,
    LENGTH_RAW_BBOX,
    FullInterpolationType,
    ImageType,
    StackedMasks4D,
    Targets,
)

__all__ = [
    "CV2_INTER_LINEAR",
    "CV2_INTER_NEAREST",
    "LENGTH_RAW_BBOX",
    "_BBOX_INSTANCE_ID",
    "_KP_INSTANCE_ID",
    "AfterValidator",
    "Annotated",
    "Any",
    "BaseTransformInitSchema",
    "BboxProcessor",
    "ClassVar",
    "DualTransform",
    "Field",
    "FullInterpolationType",
    "ImageType",
    "KeypointsProcessor",
    "Literal",
    "Self",
    "Sequence",
    "StackedMasks4D",
    "Targets",
    "cast",
    "check_bboxes",
    "check_range_bounds",
    "convert_bboxes_from_albumentations",
    "convert_bboxes_to_albumentations",
    "convert_keypoints_from_albumentations",
    "convert_keypoints_to_albumentations",
    "cv2",
    "deepcopy",
    "denormalize_bboxes",
    "fgeometric",
    "filter_bboxes_with_mask",
    "fmixing",
    "model_validator",
    "nondecreasing",
    "np",
    "random",
    "warnings",
]
