"""Module containing type definitions and constants used throughout Albumentations.

This module defines common types, constants, and enumerations that are used across the
Albumentations library. It includes type aliases for numeric types, enumerations for
targets supported by transforms, and constants that define standard dimensions or values
used in image and volumetric data processing. These definitions help ensure type safety
and provide a centralized location for commonly used values.
"""

from enum import Enum
from typing import TypeAlias, TypeVar

import cv2
import numpy as np
from albucore.utils import MAX_VALUES_BY_DTYPE
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

d4_group_elements = ["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]


class ReferenceImage(TypedDict):
    """Dictionary-like container for reference image data.

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
    """Enumeration of supported target types in Albumentations.

    This enum defines the different types of data that can be augmented
    by Albumentations transforms, including both 2D and 3D targets.

    Args:
        IMAGE (str): 2D image target.
        MASK (str): 2D mask target.
        BBOXES (str): Bounding box target.
        KEYPOINTS (str): Keypoint coordinates target.
        VOLUME (str): 3D volume target.
        MASK3D (str): 3D mask target.

    """

    IMAGE = "Image"
    MASK = "Mask"
    BBOXES = "BBoxes"
    KEYPOINTS = "Keypoints"
    VOLUME = "Volume"
    MASK3D = "Mask3D"


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

BIG_INTEGER = MAX_VALUES_BY_DTYPE[np.uint32]
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
