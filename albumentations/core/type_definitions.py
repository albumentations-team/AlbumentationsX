"""Module containing type definitions and constants used throughout Albumentations.

This module defines common types, constants, and enumerations that are used across the
Albumentations library. It includes type aliases for numeric types, enumerations for
targets supported by transforms, and constants that define standard dimensions or values
used in image and volumetric data processing. These definitions help ensure type safety
and provide a centralized location for commonly used values.
"""

from enum import Enum
from typing import Literal, NewType, TypeAlias, TypeVar

import cv2
import numpy as np
from numpy import float32, uint8
from numpy.typing import NDArray
from typing_extensions import NotRequired, TypedDict

Number = TypeVar("Number", float, int)

IntNumType = np.integer | NDArray[np.integer]
FloatNumType = np.floating | NDArray[np.floating]

# Core image types - restrict to uint8 and float32 only
ImageUInt8: TypeAlias = NDArray[uint8]
ImageFloat32: TypeAlias = NDArray[float32]
ImageType: TypeAlias = ImageUInt8 | ImageFloat32

# Image and Volume types - restrict to uint8 and float32 only
# VolumeType is same as ImageType (volumes are also uint8/float32)
VolumeType: TypeAlias = ImageType

# OpenCV exposes these constants as module-level ints. Literal aliases and typed constants use the
# corresponding values so static checkers can validate transform configuration domains.
CV2_INTER_NEAREST: Literal[0] = 0
CV2_INTER_LINEAR: Literal[1] = 1
CV2_INTER_CUBIC: Literal[2] = 2
CV2_INTER_AREA: Literal[3] = 3
CV2_INTER_LANCZOS4: Literal[4] = 4
CV2_INTER_LINEAR_EXACT: Literal[5] = 5
CV2_INTER_NEAREST_EXACT: Literal[6] = 6
CV2_BORDER_CONSTANT: Literal[0] = 0
CV2_BORDER_REPLICATE: Literal[1] = 1
CV2_BORDER_REFLECT: Literal[2] = 2
CV2_BORDER_WRAP: Literal[3] = 3
CV2_BORDER_REFLECT_101: Literal[4] = 4
InterpolationType: TypeAlias = Literal[0, 1, 2, 3, 4]
FullInterpolationType: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6]
BorderModeType: TypeAlias = Literal[0, 1, 2, 3, 4]

d4_group_elements = ["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]

# C4: cyclic subgroup of D4 (rotations only). RandomRotate90 uses these.
c4_group_elements = ["e", "r90", "r180", "r270"]

# Inverse tables for TTA: applying element then its inverse yields identity.
# Rotations: r90 and r270 are mutual inverses; r180 and e are self-inverse.
# Reflections in D4 are all self-inverse.
C4GroupElement: TypeAlias = Literal["e", "r90", "r180", "r270"]
D4GroupElement: TypeAlias = Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]
C4_INVERSE: dict[C4GroupElement, C4GroupElement] = {
    "e": "e",
    "r90": "r270",
    "r180": "r180",
    "r270": "r90",
}
D4_INVERSE: dict[D4GroupElement, D4GroupElement] = {
    "e": "e",
    "r90": "r270",
    "r180": "r180",
    "r270": "r90",
    "v": "v",
    "h": "h",
    "t": "t",
    "hvt": "hvt",
}


class ReferenceImage(TypedDict):
    """TypedDict for reference image data: image (required), optional mask, bbox, keypoints.
    Use for reference-based transforms (e.g. style transfer, exemplar).

    A typed dictionary defining the structure of reference image data used within
    Albumentations, including optional components like masks, bounding boxes,
    and keypoints.

    Args:
        image (ImageType): The reference image array (uint8 or float32).
        mask (np.ndarray | None): Optional mask array.
        bbox (tuple[float, ...] | np.ndarray | None): Optional bounding box coordinates.
        keypoints (tuple[float, ...] | np.ndarray | None): Optional keypoint coordinates.

    """

    image: ImageType
    mask: NotRequired[np.ndarray]
    bbox: NotRequired[tuple[float, ...] | np.ndarray]
    keypoints: NotRequired[tuple[float, ...] | np.ndarray]


class Targets(Enum):
    """Enum of supported target types: image, mask, bboxes, keypoints, volume, mask3d, user_data.
    Compose and transform targets use this to dispatch apply_* methods.

    This enum defines the different types of data that can be augmented
    by Albumentations transforms, including both 2D and 3D targets.

    Args:
        IMAGE (str): 2D image target.
        MASK (str): 2D mask target.
        BBOXES (str): Bounding box target.
        KEYPOINTS (str): Keypoint coordinates target.
        VOLUME (str): 3D volume target.
        MASK3D (str): 3D mask target.
        USER_DATA (str): Arbitrary user-defined data target.

    """

    IMAGE = "Image"
    MASK = "Mask"
    BBOXES = "BBoxes"
    KEYPOINTS = "Keypoints"
    VOLUME = "Volume"
    MASK3D = "Mask3D"
    USER_DATA = "UserData"


ALL_TARGETS = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS, Targets.VOLUME, Targets.MASK3D)


NUM_VOLUME_DIMENSIONS = 4
NUM_MULTI_CHANNEL_DIMENSIONS = 3
MONO_CHANNEL_DIMENSIONS = 2
NUM_RGB_CHANNELS = 3

PAIR = 2
TWO = 2
THREE = 3
FOUR = 4
SEVEN = 7
EIGHT = 8
THREE_SIXTY = 360

BIG_INTEGER = np.iinfo(np.uint32).max
MAX_RAIN_ANGLE = 45  # Maximum angle for rain augmentation in degrees

LENGTH_RAW_BBOX = 4

PercentType = (
    float
    | tuple[float, float]
    | tuple[float, float, float, float]
    | tuple[
        float | tuple[float, float],
        float | tuple[float, float],
        float | tuple[float, float],
        float | tuple[float, float],
    ]
)


PxType = (
    int
    | tuple[int, int]
    | tuple[int, int, int, int]
    | tuple[
        int | tuple[int, int],
        int | tuple[int, int],
        int | tuple[int, int],
        int | tuple[int, int],
    ]
)


REFLECT_BORDER_MODES = {
    cv2.BORDER_REFLECT_101,
    cv2.BORDER_REFLECT,
}

NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS = 5
NUM_BBOXES_COLUMNS_IN_ALBUMENTATIONS = 4


# Stacked instance masks shape `(N, H, W, C)`. NewType is identity at runtime (zero overhead) and
# distinct for static checkers, so a function annotated `-> StackedMasks4D` rejects a raw
# `np.ndarray` return without an explicit `_make_stacked_masks` round-trip. Constructed only by
# `_make_stacked_masks` (in `albumentations.core.composition`); mixing transforms preserve the
# brand by returning their input shape rank.
#
# The `type: ignore` keeps pre-commit's mypy happy when it runs without numpy installed (np.ndarray
# resolves to Any there, which NewType rejects). Local/CI mypy with numpy installed sees the
# annotation as redundant, so we also disable `warn_unused_ignores` for this specific code below
# via the `unused-ignore` allowance baked into mypy's default behavior for cross-config NewTypes.
StackedMasks4D = NewType("StackedMasks4D", np.ndarray)  # type: ignore[valid-newtype, unused-ignore]
