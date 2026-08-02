"""Deterministic, fresh input factories for transform contract tests."""

from collections.abc import Callable
from typing import Any

import numpy as np

ContractDataFactory = Callable[[np.random.Generator], dict[str, Any]]
ContractContextFactory = Callable[[np.random.Generator, dict[str, Any]], dict[str, Any]]

IMAGE_SHAPE = (128, 128, 3)
VOLUME_SHAPE = (4, 64, 64, 3)
TARGET_IMAGE_SHAPE = (96, 128, 3)
TARGET_VOLUME_SHAPE = (4, *TARGET_IMAGE_SHAPE)


def make_image_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return a fresh RGB image."""
    return {"image": rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8)}


def make_grayscale_image_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return a fresh single-channel image."""
    return {"image": rng.integers(0, 256, IMAGE_SHAPE[:2], dtype=np.uint8)}


def make_float_image_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return a fresh normalized float32 image."""
    return {"image": rng.random(IMAGE_SHAPE, dtype=np.float32)}


def make_mask_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return an RGB image and a non-empty semantic mask."""
    data = make_image_data(rng)
    mask = np.zeros(IMAGE_SHAPE[:2], dtype=np.uint8)
    mask[16:96, 16:96] = 1
    data["mask"] = mask
    return data


def make_target_image_mask_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return an asymmetric coordinate-coded image and multiclass mask."""
    height, width, _ = TARGET_IMAGE_SHAPE
    offset = int(rng.integers(0, 256))
    rows = np.arange(height, dtype=np.uint16)[:, None]
    columns = np.arange(width, dtype=np.uint16)[None, :]
    image = np.stack(
        [
            (rows + columns + offset) % 256,
            np.broadcast_to((3 * rows + offset) % 256, (height, width)),
            np.broadcast_to((5 * columns + offset) % 256, (height, width)),
        ],
        axis=-1,
    ).astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[9:41, 17:73] = 1
    mask[52:87, 61:117] = 2
    mask[24:75, 92:104] = 3
    return {"image": image, "mask": mask}


def make_target_hbb_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return asymmetric image data with horizontal boxes and aligned fields."""
    data = make_target_image_mask_data(rng)
    data.update(
        bboxes=np.array(
            [
                [0.12, 0.18, 0.53, 0.61],
                [0.58, 0.49, 0.91, 0.88],
            ],
            dtype=np.float32,
        ),
        bbox_labels=[11, 29],
        bbox_scores=[0.25, 0.75],
    )
    return data


def make_target_obb_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return asymmetric image data with oriented boxes and aligned fields."""
    data = make_target_image_mask_data(rng)
    data.update(
        bboxes=np.array(
            [
                [0.15, 0.18, 0.51, 0.57, 27.0],
                [0.59, 0.51, 0.89, 0.86, -33.0],
            ],
            dtype=np.float32,
        ),
        bbox_labels=[11, 29],
        bbox_scores=[0.25, 0.75],
    )
    return data


def make_target_keypoint_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return asymmetric image data with keypoints and aligned fields."""
    data = make_target_image_mask_data(rng)
    data.update(
        keypoints=np.array([[22.0, 18.0], [91.0, 67.0]], dtype=np.float32),
        keypoint_labels=[11, 29],
    )
    return data


def make_target_volume_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return a coordinate-coded volume and a slice-specific multiclass mask3d."""
    depth, height, width, channels = TARGET_VOLUME_SHAPE
    volume = np.empty(TARGET_VOLUME_SHAPE, dtype=np.uint8)
    mask3d = np.empty((depth, height, width), dtype=np.uint8)
    for depth_index in range(depth):
        slice_data = make_target_image_mask_data(rng)
        volume[depth_index] = (slice_data["image"].astype(np.uint16) + 17 * depth_index).astype(np.uint8)
        mask3d[depth_index] = (slice_data["mask"] + depth_index).astype(np.uint8)
    assert volume.shape[-1] == channels
    return {"volume": volume, "mask3d": mask3d}


def make_target_float_image_mask_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return normalized float32 image data with a uint8 semantic mask."""
    data = make_target_image_mask_data(rng)
    data["image"] = data["image"].astype(np.float32) / 255.0
    return data


def make_target_grayscale_image_mask_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return channel-last grayscale image data with a semantic mask."""
    data = make_target_image_mask_data(rng)
    data["image"] = data["image"][..., :1]
    return data


def make_target_multispectral_image_mask_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return five-channel image data with a semantic mask."""
    data = make_target_image_mask_data(rng)
    image = data["image"]
    extra_channels = np.stack(
        [
            (image[..., 0].astype(np.uint16) + image[..., 1]) % 256,
            (image[..., 1].astype(np.uint16) + image[..., 2]) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    data["image"] = np.concatenate([image, extra_channels], axis=-1)
    return data


def make_target_image_batch_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return a batch of distinct coordinate-coded images."""
    images = [make_target_image_mask_data(rng)["image"] for _ in range(2)]
    return {"images": np.stack(images)}


def make_target_mask_batch_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return a batch of asymmetric multiclass masks."""
    masks = [make_target_image_mask_data(rng)["mask"] for _ in range(2)]
    return {"masks": np.stack(masks)}


def make_target_mask3d_batch_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return a batch of slice-dependent 3D masks."""
    masks3d = [make_target_volume_data(rng)["mask3d"] for _ in range(2)]
    return {"masks3d": np.stack(masks3d)}


def make_target_empty_hbb_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return image data with typed empty horizontal-box fields."""
    data = make_target_image_mask_data(rng)
    data.update(
        bboxes=np.empty((0, 4), dtype=np.float32),
        bbox_labels=[],
        bbox_scores=[],
    )
    return data


def make_target_empty_keypoint_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return image data with typed empty keypoint fields."""
    data = make_target_image_mask_data(rng)
    data.update(
        keypoints=np.empty((0, 2), dtype=np.float32),
        keypoint_labels=[],
    )
    return data


def make_target_noncontiguous_image_mask_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return strided image and mask views with asymmetric spatial dimensions."""
    height, width, channels = TARGET_IMAGE_SHAPE
    image = rng.integers(0, 256, (height * 2, width * 2, channels), dtype=np.uint8)[::2, ::2]
    mask = rng.integers(0, 4, (height * 2, width * 2), dtype=np.uint8)[::2, ::2]
    assert not image.flags.c_contiguous
    assert not mask.flags.c_contiguous
    return {"image": image, "mask": mask}


def make_target_readonly_image_mask_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return coordinate-coded image and mask arrays marked read-only."""
    data = make_target_image_mask_data(rng)
    data["image"].setflags(write=False)
    data["mask"].setflags(write=False)
    return data


def make_hbb_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return image, mask, normalized horizontal boxes, and labels."""
    data = make_mask_data(rng)
    data.update(
        bboxes=np.array([[0.15, 0.15, 0.65, 0.65]], dtype=np.float32),
        bbox_labels=[1],
    )
    return data


def make_volume_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return a fresh color volume and a non-empty 3D mask."""
    volume = rng.integers(0, 256, VOLUME_SHAPE, dtype=np.uint8)
    mask3d = np.zeros(VOLUME_SHAPE[:3], dtype=np.uint8)
    mask3d[:, 8:48, 8:48] = 1
    return {"volume": volume, "mask3d": mask3d}


def make_empty_context(rng: np.random.Generator, data: dict[str, Any]) -> dict[str, Any]:
    """Return no transform-specific context."""
    return {}


def _first_image(data: dict[str, Any]) -> np.ndarray:
    if "image" in data:
        return data["image"]
    if "images" in data:
        return data["images"][0]
    if "volume" in data:
        return data["volume"][0]
    if "mask" in data:
        mask = data["mask"]
        return np.repeat(mask[..., None], 3, axis=-1) if mask.ndim == 2 else mask
    if "masks" in data:
        mask = data["masks"][0]
        return np.repeat(mask[..., None], 3, axis=-1) if mask.ndim == 2 else mask
    if "mask3d" in data:
        mask = data["mask3d"][0]
        return np.repeat(mask[..., None], 3, axis=-1) if mask.ndim == 2 else mask
    if "masks3d" in data:
        mask = data["masks3d"][0, 0]
        return np.repeat(mask[..., None], 3, axis=-1) if mask.ndim == 2 else mask
    raise ValueError(f"Cannot derive an image from data keys: {sorted(data)}")


def _first_mask(data: dict[str, Any]) -> np.ndarray | None:
    if "mask" in data:
        return data["mask"]
    if "masks" in data:
        return data["masks"][0]
    if "mask3d" in data:
        return data["mask3d"][0]
    if "masks3d" in data:
        return data["masks3d"][0, 0]
    return None


def make_reference_context(metadata_key: str) -> ContractContextFactory:
    """Build context for transforms that consume reference images."""

    def factory(rng: np.random.Generator, data: dict[str, Any]) -> dict[str, Any]:
        reference = rng.integers(0, 256, _first_image(data).shape, dtype=np.uint8)
        return {metadata_key: [reference]}

    return factory


def make_crop_near_bbox_context(metadata_key: str) -> ContractContextFactory:
    """Build context containing the reference crop box."""

    def factory(rng: np.random.Generator, data: dict[str, Any]) -> dict[str, Any]:
        height, width = _first_image(data).shape[:2]
        return {metadata_key: [width // 6, height // 6, 5 * width // 6, 5 * height // 6]}

    return factory


def make_mosaic_context(metadata_key: str) -> ContractContextFactory:
    """Build context containing a primary image and compatible mosaic sources."""

    def factory(rng: np.random.Generator, data: dict[str, Any]) -> dict[str, Any]:
        image = _first_image(data)
        mask = _first_mask(data)
        sources = []
        for _ in range(3):
            source = {"image": rng.integers(0, 256, image.shape, dtype=np.uint8)}
            if mask is not None:
                source["mask"] = rng.integers(0, 4, mask.shape, dtype=np.uint8)
            sources.append(source)
        return {metadata_key: sources}

    return factory


def make_copy_and_paste_context(metadata_key: str) -> ContractContextFactory:
    """Build context containing a primary image and one paste donor."""

    def factory(rng: np.random.Generator, data: dict[str, Any]) -> dict[str, Any]:
        image = _first_image(data)
        if "bboxes" in data and len(data["bboxes"]) == 0:
            return {metadata_key: []}
        source_height = max(8, image.shape[0] // 3)
        source_width = max(8, image.shape[1] // 3)
        source = rng.integers(0, 256, (source_height, source_width, image.shape[-1]), dtype=np.uint8)
        source_mask = np.zeros((source_height, source_width), dtype=np.uint8)
        source_mask[2:-2, 2:-2] = 1
        donor: dict[str, Any] = {"image": source, "mask": source_mask, "semantic_mask": source_mask}
        if "bboxes" in data:
            donor["bbox_labels"] = {"bbox_labels": 41, "bbox_scores": 0.5}
        return {metadata_key: [donor]}

    return factory


def make_overlay_context(metadata_key: str) -> ContractContextFactory:
    """Build context containing one overlay element."""

    def factory(rng: np.random.Generator, data: dict[str, Any]) -> dict[str, Any]:
        image = _first_image(data)
        overlay_height = min(24, image.shape[0])
        overlay_width = min(24, image.shape[1])
        overlay = rng.integers(
            0,
            256,
            (overlay_height, overlay_width, image.shape[-1]),
            dtype=np.uint8,
        )
        overlay_mask = np.ones((overlay_height, overlay_width), dtype=np.uint8)
        return {metadata_key: [{"image": overlay, "mask": overlay_mask}]}

    return factory


def make_text_context(metadata_key: str) -> ContractContextFactory:
    """Build context containing normalized text placement metadata."""

    def factory(rng: np.random.Generator, data: dict[str, Any]) -> dict[str, Any]:
        return {
            metadata_key: {
                "text": "contract",
                "bbox": (0.1, 0.1, 0.8, 0.25),
            },
        }

    return factory
