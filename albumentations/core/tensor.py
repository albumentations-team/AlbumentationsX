"""CPU Tensor input contract and annotation bridge shared by Compose."""

from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray

TENSOR_TARGET_RANKS: Final[dict[str, int]] = {
    "image": 3,
    "images": 4,
    "volume": 4,
    "mask": 2,
    "masks": 3,
    "mask3d": 3,
    "bboxes": 2,
    "keypoints": 2,
}
TENSOR_ANNOTATION_TARGETS: Final[frozenset[str]] = frozenset({"bboxes", "keypoints"})
TENSOR_SPATIAL_TARGETS: Final[frozenset[str]] = frozenset(
    {"image", "images", "volume", "mask", "masks", "mask3d", "bboxes", "keypoints"},
)
TENSOR_IMAGE_DTYPES: Final[frozenset[torch.dtype]] = frozenset({torch.uint8, torch.float32})
TENSOR_MASK_DTYPES: Final[frozenset[torch.dtype]] = frozenset({torch.uint8, torch.uint16, torch.float32})
TENSOR_ANNOTATION_DTYPES: Final[frozenset[torch.dtype]] = frozenset({torch.float32})
TENSOR_TARGET_DTYPES: Final[dict[str, frozenset[torch.dtype]]] = {
    "image": TENSOR_IMAGE_DTYPES,
    "images": TENSOR_IMAGE_DTYPES,
    "volume": TENSOR_IMAGE_DTYPES,
    "mask": TENSOR_MASK_DTYPES,
    "masks": TENSOR_MASK_DTYPES,
    "mask3d": TENSOR_MASK_DTYPES,
    **dict.fromkeys(TENSOR_ANNOTATION_TARGETS, TENSOR_ANNOTATION_DTYPES),
}


def validate_tensor_input(value: torch.Tensor, data_name: str, canonical_name: str) -> None:
    """Validate a CPU Tensor against target-specific Compose shape, dtype, device, layout,
    and autograd boundary rules before it reaches a transform helper.

    The CPU stage accepts explicit-channel image targets, spatial mask targets,
    and float32 annotation matrices. It preserves non-contiguous strides because
    each accepted capability later decides whether a contiguous copy is justified
    by its full-path benchmark.
    """
    expected_rank = TENSOR_TARGET_RANKS.get(canonical_name)
    if expected_rank is None:
        raise TypeError(
            f"{data_name} is a torch.Tensor, but Tensor input is currently supported only for "
            "image, images, volume, mask, masks, mask3d, bboxes, and keypoints targets",
        )
    if value.device.type != "cpu":
        raise ValueError(f"{data_name} must be a CPU torch.Tensor, got device {value.device}")
    if value.requires_grad:
        raise ValueError(f"{data_name} must have requires_grad=False")
    if value.layout is not torch.strided:
        raise TypeError(f"{data_name} must use torch.strided layout, got {value.layout}")
    supported_dtypes = TENSOR_TARGET_DTYPES[canonical_name]
    if value.dtype not in supported_dtypes:
        supported = ", ".join(str(dtype).removeprefix("torch.") for dtype in sorted(supported_dtypes, key=str))
        raise TypeError(f"{data_name} must have dtype one of {supported}, got {value.dtype}")
    if value.ndim != expected_rank:
        expected_shape = {
            "image": "(C, H, W)",
            "images": "(C, L, H, W)",
            "volume": "(C, D, H, W)",
            "mask": "(H, W)",
            "masks": "(N, H, W)",
            "mask3d": "(D, H, W)",
            "bboxes": "(N, K)",
            "keypoints": "(N, K)",
        }[canonical_name]
        raise TypeError(f"{data_name} must have shape {expected_shape}, got {tuple(value.shape)}")


def tensor_to_numpy_annotation(value: torch.Tensor, target: str) -> NDArray[np.generic]:
    """Return a NumPy view of a validated Tensor bbox or keypoint matrix through the shared
    annotation bridge used by the existing geometry processors.
    """
    validate_tensor_input(value, target, target)
    return value.numpy()


def tensor_to_numpy_spatial(value: torch.Tensor, target: str) -> NDArray[np.generic]:
    """Convert a validated Tensor target to the channel-last NumPy layout that existing Compose preprocessing and
    transforms expect.

    The bridge owns every layout conversion. Transform helpers only receive their established NumPy layout and never
    decide whether to convert a caller's Tensor themselves.
    """
    validate_tensor_input(value, target, target)
    if target == "image":
        return value.permute(1, 2, 0).numpy()
    if target in {"images", "volume"}:
        return value.permute(1, 2, 3, 0).numpy()
    return value.numpy()


def numpy_to_tensor_spatial(value: NDArray[np.generic], target: str) -> torch.Tensor:
    """Convert a NumPy result from Compose back to the caller-facing Tensor layout, copying only when negative strides
    require it.

    NumPy transforms may return a negative-stride view, such as after a reflection. PyTorch cannot share that storage,
    so materialize only that incompatible case before returning the Tensor result.
    """
    if target in {"image", "images", "volume"}:
        value = np.moveaxis(value, -1, 0)
    if any(stride < 0 for stride in value.strides):
        value = np.ascontiguousarray(value)
    return torch.from_numpy(value)


def numpy_to_tensor_annotation(value: NDArray[np.generic], target: str) -> torch.Tensor:
    """Return a Tensor bbox or keypoint matrix from a processor result, materializing only
    negative-stride NumPy storage that PyTorch cannot safely share.
    """
    expected_rank = TENSOR_TARGET_RANKS[target]
    if value.ndim != expected_rank:
        raise TypeError(f"{target} bridge expected a {expected_rank}D NumPy array, got {value.ndim}D")
    if value.dtype != np.float32:
        value = value.astype(np.float32, copy=False)
    if any(stride < 0 for stride in value.strides):
        value = np.ascontiguousarray(value)
    return torch.from_numpy(value)
