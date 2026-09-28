"""Frame selection for temporal image and mask sequences."""

from collections.abc import Sequence
from typing import TypeVar, cast

import numpy as np
import torch

__all__ = ["uniform_temporal_subsample"]


ArrayType = TypeVar("ArrayType", bound=np.ndarray | torch.Tensor)


def uniform_temporal_subsample(
    frames: ArrayType,
    frame_indices: Sequence[int] | np.ndarray,
) -> ArrayType:
    """Select frames along the leading axis without changing their spatial layout."""
    if isinstance(frames, torch.Tensor):
        indices = torch.as_tensor(frame_indices, dtype=torch.long, device=frames.device)
        return cast("ArrayType", torch.index_select(frames, 0, indices))
    return cast("ArrayType", frames[np.asarray(frame_indices, dtype=np.intp)])


def select_annotation_rows(
    frame_ids: np.ndarray,
    frame_indices: Sequence[int] | np.ndarray,
    instance_ids: np.ndarray | None = None,
    instance_frame_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rows = []
    frames = []
    instances = []
    offset = 0
    for new_frame, old_frame in enumerate(frame_indices):
        selected = np.flatnonzero(frame_ids == old_frame)
        rows.append(selected)
        frames.append(np.full(len(selected), new_frame, dtype=np.intp))
        if instance_ids is not None:
            parents = np.flatnonzero(instance_frame_ids == old_frame)
            instances.append(np.searchsorted(parents, instance_ids[selected]) + offset)
            offset += len(parents)
    return (
        np.concatenate(rows),
        np.concatenate(frames),
        np.concatenate(instances) if instance_ids is not None else None,
    )
