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
DTYPES = {
    "uint8": np.uint8,
    "float32": np.float32,
}
IMAGE_DTYPE_PARAMS = ([*SIZES], CHANNELS, tuple(DTYPES))
ANNOTATION_COUNTS = (10, 100, 1000)
VOLUME_SIZES = {
    "small": (8, 64, 64),
    "medium": (16, 128, 128),
}


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


def dtype_from_name(dtype_name: str) -> type[np.generic]:
    """Resolve a benchmark dtype name."""
    return DTYPES[dtype_name]


def make_batch(size_name: str, channels: int, batch_size: int = 8) -> np.ndarray:
    """Create a deterministic channel-last image batch for benchmark inputs."""
    image = make_image(size_name, channels)
    return np.stack([image.copy() for _ in range(batch_size)], axis=0)


def make_mask(size_name: str = "small") -> np.ndarray:
    """Create a deterministic semantic mask."""
    height, width = SIZES[size_name]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[height // 5 : height // 2, width // 4 : width // 2] = 1
    mask[height // 2 : height * 4 // 5, width // 2 : width * 4 // 5] = 2
    return mask


def make_masks(size_name: str = "small", count: int = 8) -> np.ndarray:
    """Create deterministic stacked instance masks."""
    height, width = SIZES[size_name]
    masks = np.zeros((count, height, width), dtype=np.uint8)
    grid = int(np.ceil(np.sqrt(count)))
    cell_h = max(height // (grid + 1), 4)
    cell_w = max(width // (grid + 1), 4)
    for index in range(count):
        row = index // grid
        col = index % grid
        y0 = min((row + 1) * cell_h, height - 3)
        x0 = min((col + 1) * cell_w, width - 3)
        masks[index, y0 : min(y0 + cell_h // 2, height), x0 : min(x0 + cell_w // 2, width)] = 1
    return masks


def make_hbb_bboxes(size_name: str = "small", count: int = 10) -> np.ndarray:
    """Create deterministic pascal_voc horizontal bounding boxes."""
    height, width = SIZES[size_name]
    boxes = np.empty((count, 4), dtype=np.float32)
    grid = int(np.ceil(np.sqrt(count)))
    cell_h = height / (grid + 1)
    cell_w = width / (grid + 1)
    box_h = max(cell_h * 0.35, 4)
    box_w = max(cell_w * 0.35, 4)
    for index in range(count):
        row = index // grid
        col = index % grid
        center_y = (row + 1) * cell_h
        center_x = (col + 1) * cell_w
        x_min = max(center_x - box_w / 2, 0)
        y_min = max(center_y - box_h / 2, 0)
        x_max = min(center_x + box_w / 2, width - 1)
        y_max = min(center_y + box_h / 2, height - 1)
        boxes[index] = (x_min, y_min, x_max, y_max)
    return boxes


def make_obb_bboxes(count: int = 10) -> np.ndarray:
    """Create deterministic normalized OBB boxes in x_min, y_min, x_max, y_max, angle format."""
    boxes = np.empty((count, 5), dtype=np.float32)
    grid = int(np.ceil(np.sqrt(count)))
    cell = 1 / (grid + 1)
    box_w = min(cell * 0.45, 0.04)
    box_h = min(cell * 0.45, 0.05)
    for index in range(count):
        row = index // grid
        col = index % grid
        center_x = (col + 1) / (grid + 1)
        center_y = (row + 1) / (grid + 1)
        boxes[index] = (
            max(center_x - box_w / 2, 0.001),
            max(center_y - box_h / 2, 0.001),
            min(center_x + box_w / 2, 0.999),
            min(center_y + box_h / 2, 0.999),
            float((index % 12) * 15 - 90),
        )
    return boxes


def make_keypoints(size_name: str = "small", count: int = 10) -> np.ndarray:
    """Create deterministic xy keypoints."""
    height, width = SIZES[size_name]
    keypoints = np.empty((count, 2), dtype=np.float32)
    grid = int(np.ceil(np.sqrt(count)))
    for index in range(count):
        row = index // grid
        col = index % grid
        keypoints[index] = ((col + 1) * width / (grid + 1), (row + 1) * height / (grid + 1))
    return keypoints


def make_labels(count: int) -> list[int]:
    """Create deterministic integer labels."""
    return [index % 17 for index in range(count)]


def make_reference_images(
    size_name: str = "small",
    channels: int = 3,
    dtype: type[np.generic] = np.uint8,
    count: int = 4,
) -> list[np.ndarray]:
    """Create deterministic reference images for metadata transforms."""
    image = make_image(size_name, channels, dtype)
    references = [np.flipud(image).copy(), np.fliplr(image).copy(), np.rot90(image).copy(), image.copy()]
    return references[:count]


def make_volume(
    size_name: str = "small",
    channels: int = 1,
    dtype: type[np.generic] = np.uint8,
) -> np.ndarray:
    """Create a deterministic channel-last volume for benchmark inputs."""
    depth, height, width = VOLUME_SIZES[size_name]
    rng = np.random.default_rng(137 + depth + height + width + channels)
    shape = (depth, height, width, channels)
    if dtype == np.uint8:
        return rng.integers(0, 256, shape, dtype=np.uint8)
    if dtype == np.float32:
        return rng.uniform(0, 1, shape).astype(np.float32)
    msg = f"Unsupported benchmark dtype: {dtype}"
    raise ValueError(msg)


def make_mask3d(size_name: str = "small") -> np.ndarray:
    """Create a deterministic 3D mask."""
    volume = make_volume(size_name, 1)
    return (volume[..., 0] > 127).astype(np.uint8)
