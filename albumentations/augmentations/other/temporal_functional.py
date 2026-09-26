"""Functional operations for temporal image sequences."""

from collections.abc import Sequence

import numpy as np
import torch

__all__ = ["uniform_temporal_subsample"]


def uniform_temporal_subsample(
    images: np.ndarray | torch.Tensor,
    frame_indices: Sequence[int],
) -> np.ndarray | torch.Tensor:
    """Select temporal frames along the leading axis with one indexed operation."""
    if images.ndim != 4:
        raise ValueError(
            "UniformTemporalSubsample expects a 4D video with shape (T, H, W, C) or (T, C, H, W)",
        )
    if images.shape[0] == 0:
        raise ValueError("UniformTemporalSubsample requires at least one frame")

    if isinstance(images, torch.Tensor):
        indices = torch.as_tensor(frame_indices, dtype=torch.long, device=images.device)
        return torch.index_select(images, 0, indices)

    indices = np.asarray(frame_indices, dtype=np.intp)
    return images[indices]
