"""CPU Tensor target validation and NumPy bridge helpers."""

from typing import Any, Final

import numpy as np
import torch
from albucore import TensorLayout
from albucore.tensor_bridge import (
    numpy_to_tensor_layout,
    numpy_to_tensor_storage,
    tensor_to_numpy_channel_last,
    tensor_to_numpy_storage,
)
from numpy.typing import NDArray

from .type_definitions import (
    VECTOR_FIELD_TARGET_INFO_BY_NAME,
    VECTOR_FIELD_TARGET_NAMES,
)

TENSOR_ANNOTATION_TARGETS: Final[frozenset[str]] = frozenset({"bboxes", "keypoints"})
TENSOR_VECTOR_FIELD_TARGETS = VECTOR_FIELD_TARGET_NAMES
TENSOR_VECTOR_FIELD_2D_TARGETS = frozenset(
    name for name, (spatial_rank, _) in VECTOR_FIELD_TARGET_INFO_BY_NAME.items() if spatial_rank == 2
)
TENSOR_VECTOR_FIELD_3D_TARGETS = frozenset(
    name for name, (spatial_rank, _) in VECTOR_FIELD_TARGET_INFO_BY_NAME.items() if spatial_rank == 3
)
_VECTOR_FIELD_2D_LAYOUT: Final[TensorLayout] = "CHW"
_VECTOR_FIELD_3D_LAYOUT: Final[TensorLayout] = "CDHW"
_TENSOR_SPATIAL_LAYOUTS: Final[dict[str, TensorLayout]] = {
    "image": "CHW",
    "images": "NCHW",
    "volume": "CDHW",
    "volumes": "NCDHW",
    "mask": "CHW",
    "masks": "NCHW",
    "mask3d": "CDHW",
    "masks3d": "NCDHW",
    **dict.fromkeys(TENSOR_VECTOR_FIELD_2D_TARGETS, _VECTOR_FIELD_2D_LAYOUT),
    **dict.fromkeys(TENSOR_VECTOR_FIELD_3D_TARGETS, _VECTOR_FIELD_3D_LAYOUT),
}
TENSOR_SPATIAL_TARGETS: Final[frozenset[str]] = frozenset(_TENSOR_SPATIAL_LAYOUTS)
TENSOR_TARGETS: Final[frozenset[str]] = TENSOR_SPATIAL_TARGETS | TENSOR_ANNOTATION_TARGETS

TENSOR_CANONICAL_RANKS: Final[dict[str, int]] = {
    **{target: len(layout) for target, layout in _TENSOR_SPATIAL_LAYOUTS.items()},
    "bboxes": 2,
    "keypoints": 2,
}
TENSOR_CHANNELLESS_RANKS: Final[dict[str, int]] = {
    "image": 2,
    "mask": 2,
    "masks": 3,
    "mask3d": 3,
    "masks3d": 4,
}
TENSOR_CHANNEL_AXIS: Final[dict[str, int]] = {
    target: layout.index("C") for target, layout in _TENSOR_SPATIAL_LAYOUTS.items()
}
TENSOR_IMAGE_DTYPES: Final[frozenset[torch.dtype]] = frozenset({torch.uint8, torch.float16, torch.float32})
TENSOR_MASK_DTYPES: Final[frozenset[torch.dtype]] = frozenset({torch.uint8, torch.int16, torch.float16, torch.float32})
TENSOR_VECTOR_FIELD_DTYPES: Final[frozenset[torch.dtype]] = frozenset({torch.float16, torch.float32})
TENSOR_ANNOTATION_DTYPES: Final[frozenset[torch.dtype]] = frozenset({torch.float32})
TENSOR_TARGET_DTYPES: Final[dict[str, frozenset[torch.dtype]]] = {
    "image": TENSOR_IMAGE_DTYPES,
    "images": TENSOR_IMAGE_DTYPES,
    "volume": TENSOR_IMAGE_DTYPES,
    "volumes": TENSOR_IMAGE_DTYPES,
    "mask": TENSOR_MASK_DTYPES,
    "masks": TENSOR_MASK_DTYPES,
    "mask3d": TENSOR_MASK_DTYPES,
    "masks3d": TENSOR_MASK_DTYPES,
    **dict.fromkeys(TENSOR_VECTOR_FIELD_TARGETS, TENSOR_VECTOR_FIELD_DTYPES),
    **dict.fromkeys(TENSOR_ANNOTATION_TARGETS, TENSOR_ANNOTATION_DTYPES),
}
TENSOR_SHAPE_DESCRIPTIONS: Final[dict[str, str]] = {
    "image": "(H, W) or (C, H, W)",
    "images": "(N, C, H, W)",
    "volume": "(C, D, H, W)",
    "volumes": "(N, C, D, H, W)",
    "mask": "(H, W) or (C, H, W)",
    "masks": "(N, H, W) or (N, C, H, W)",
    "mask3d": "(D, H, W) or (C, D, H, W)",
    "masks3d": "(N, D, H, W) or (N, C, D, H, W)",
    **{
        name: (f"({channels}, H, W)" if spatial_rank == 2 else f"({channels}, D, H, W)")
        for name, (spatial_rank, channels) in VECTOR_FIELD_TARGET_INFO_BY_NAME.items()
    },
    "bboxes": "(N, K)",
    "keypoints": "(N, K)",
}
TENSOR_CANONICAL_SHAPE_DESCRIPTIONS: Final[dict[str, str]] = {
    "image": "(C, H, W)",
    "images": "(N, C, H, W)",
    "volume": "(C, D, H, W)",
    "volumes": "(N, C, D, H, W)",
    "mask": "(C, H, W)",
    "masks": "(N, C, H, W)",
    "mask3d": "(C, D, H, W)",
    "masks3d": "(N, C, D, H, W)",
    **{
        name: (f"({channels}, H, W)" if spatial_rank == 2 else f"({channels}, D, H, W)")
        for name, (spatial_rank, channels) in VECTOR_FIELD_TARGET_INFO_BY_NAME.items()
    },
    "bboxes": "(N, K)",
    "keypoints": "(N, K)",
}


def _validate_plain_cpu_tensor(value: torch.Tensor, data_name: str) -> None:
    if type(value) is not torch.Tensor:
        raise TypeError(f"{data_name} must be a plain torch.Tensor without a Tensor subclass")
    if value.device.type != "cpu":
        raise ValueError(f"{data_name} must be a CPU torch.Tensor, got device {value.device}")
    if value.requires_grad:
        raise ValueError(f"{data_name} must have requires_grad=False")
    if value.layout is not torch.strided:
        raise TypeError(f"{data_name} must use torch.strided layout, got {value.layout}")


def validate_tensor_input(value: torch.Tensor, data_name: str, canonical_name: str) -> None:
    """Validate a plain CPU Tensor at the public Compose boundary."""
    expected_rank = TENSOR_CANONICAL_RANKS.get(canonical_name)
    if expected_rank is None:
        raise TypeError(
            f"{data_name} is a torch.Tensor, but Tensor input is currently supported only for "
            "image, images, volume, volumes, mask, masks, mask3d, masks3d, bboxes, and keypoints targets",
        )
    _validate_plain_cpu_tensor(value, data_name)

    supported_dtypes = TENSOR_TARGET_DTYPES[canonical_name]
    if value.dtype not in supported_dtypes:
        supported = ", ".join(str(dtype).removeprefix("torch.") for dtype in sorted(supported_dtypes, key=str))
        raise TypeError(f"{data_name} must have dtype one of {supported}, got {value.dtype}")

    channel_less_rank = TENSOR_CHANNELLESS_RANKS.get(canonical_name)
    if value.ndim not in (expected_rank, channel_less_rank):
        raise TypeError(
            f"{data_name} must have shape {TENSOR_SHAPE_DESCRIPTIONS[canonical_name]}, got {tuple(value.shape)}",
        )


def validate_vector_field_numpy_input(value: np.ndarray, data_name: str, canonical_name: str) -> None:
    """Validate the required channel-last NumPy shape and storage dtype for one vector-field target."""
    spatial_rank, channels = VECTOR_FIELD_TARGET_INFO_BY_NAME[canonical_name]
    expected_rank = spatial_rank + 1
    if value.dtype.type not in {np.float16, np.float32}:
        raise TypeError(f"{data_name} must have dtype float16 or float32, got {value.dtype}")
    if value.ndim != expected_rank or value.shape[-1] != channels:
        layout = "H, W" if spatial_rank == 2 else "D, H, W"
        raise TypeError(f"{data_name} must have shape ({layout}, {channels}), got {tuple(value.shape)}")


def validate_vector_field_input(value: Any, data_name: str, canonical_name: str) -> None:
    """Validate one canonical vector-field input in its supported NumPy or CPU Tensor layout."""
    if isinstance(value, np.ndarray):
        validate_vector_field_numpy_input(value, data_name, canonical_name)
    elif isinstance(value, torch.Tensor):
        validate_tensor_input(value, data_name, canonical_name)
        _, components = VECTOR_FIELD_TARGET_INFO_BY_NAME[canonical_name]
        if value.shape[0] != components:
            raise TypeError(
                f"{data_name} must have shape {TENSOR_SHAPE_DESCRIPTIONS[canonical_name]}, got {tuple(value.shape)}",
            )
    else:
        raise TypeError(f"{data_name} must be a NumPy array or torch.Tensor, got {type(value).__name__}")


def tensor_to_numpy_spatial(value: torch.Tensor, target: str) -> NDArray[np.generic]:
    """Return a canonical channel-last NumPy view of a canonical Tensor target."""
    layout = _TENSOR_SPATIAL_LAYOUTS.get(target)
    if layout is None:
        raise TypeError(f"{target} is not a spatial Tensor target")
    return tensor_to_numpy_channel_last(value, layout)


def validate_tensor_metadata_input(value: torch.Tensor, data_name: str, target: str | None) -> None:
    """Validate a Tensor read through `targets_as_params` before sampling."""
    if target is None:
        _validate_plain_cpu_tensor(value, data_name)
    elif target in VECTOR_FIELD_TARGET_INFO_BY_NAME:
        validate_vector_field_input(value, data_name, target)
    else:
        validate_tensor_input(value, data_name, target)


def tensor_metadata_field_target(name: object) -> str | None:
    """Return the canonical target for a conventional Tensor metadata field."""
    if name == "semantic_mask":
        return "mask"
    return name if isinstance(name, str) and name in TENSOR_TARGETS else None


def tensor_metadata_to_numpy(value: torch.Tensor, target: str | None) -> NDArray[np.generic]:
    """Convert one validated `targets_as_params` Tensor to the NumPy layout consumed by a transform."""
    if target is None or target in TENSOR_ANNOTATION_TARGETS or value.ndim == TENSOR_CHANNELLESS_RANKS.get(target):
        return tensor_to_numpy_storage(value)
    return tensor_to_numpy_spatial(value, target)


def numpy_to_tensor_spatial(value: NDArray[np.generic], target: str) -> torch.Tensor:
    """Return a canonical Tensor view of a canonical channel-last NumPy target."""
    expected_rank = TENSOR_CANONICAL_RANKS[target]
    if value.ndim != expected_rank:
        raise TypeError(f"{target} fallback expected a {expected_rank}D NumPy array, got {value.ndim}D")
    layout = _TENSOR_SPATIAL_LAYOUTS.get(target)
    if layout is None:
        raise TypeError(f"{target} is not a spatial Tensor target")
    return numpy_to_tensor_layout(value, layout)


def numpy_to_tensor_annotation(value: NDArray[np.generic], target: str) -> torch.Tensor:
    """Return a float32 Tensor annotation matrix from a NumPy processor result."""
    expected_rank = TENSOR_CANONICAL_RANKS[target]
    if value.ndim != expected_rank:
        raise TypeError(f"{target} bridge expected a {expected_rank}D NumPy array, got {value.ndim}D")
    if value.dtype != np.float32:
        value = value.astype(np.float32, copy=False)
    return numpy_to_tensor_storage(value)
