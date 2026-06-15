"""Shared benchmark data builders."""

from __future__ import annotations

import numpy as np

SIZES = {
    "small": (256, 256),
    "medium": (512, 512),
    "large": (1024, 1024),
}
CHANNELS = (1, 3, 5)
IMAGE_PARAMS = ([*SIZES], CHANNELS)


def make_image(size_name: str, channels: int, dtype: type[np.generic] = np.uint8) -> np.ndarray:
    """Create a deterministic channel-last image for benchmark inputs."""
    height, width = SIZES[size_name]
    rng = np.random.default_rng(137 + height + width + channels)
    shape = (height, width, channels)
    if dtype == np.uint8:
        return rng.integers(0, 256, shape, dtype=np.uint8)
    if dtype == np.float32:
        return rng.uniform(0, 1, shape).astype(np.float32)
    msg = f"Unsupported benchmark dtype: {dtype}"
    raise ValueError(msg)


def make_batch(size_name: str, channels: int, batch_size: int = 8) -> np.ndarray:
    """Create a deterministic channel-last image batch for benchmark inputs."""
    image = make_image(size_name, channels)
    return np.stack([image.copy() for _ in range(batch_size)], axis=0)


def make_volume() -> np.ndarray:
    """Create a deterministic channel-last volume for benchmark inputs."""
    rng = np.random.default_rng(137)
    return rng.integers(0, 256, (8, 64, 64, 1), dtype=np.uint8)
