"""Deterministic, fresh input factories for transform contract tests."""

from collections.abc import Callable
from typing import Any

import numpy as np

ContractDataFactory = Callable[[np.random.Generator], dict[str, Any]]

IMAGE_SHAPE = (128, 128, 3)
VOLUME_SHAPE = (4, 64, 64, 3)


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


def make_hbb_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return image, mask, normalized horizontal boxes, and labels."""
    data = make_mask_data(rng)
    data.update(
        bboxes=np.array([[0.15, 0.15, 0.65, 0.65]], dtype=np.float32),
        bbox_labels=[1],
    )
    return data


def make_crop_near_bbox_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return horizontal-box data plus the crop reference box."""
    data = make_hbb_data(rng)
    data["cropping_bbox"] = [20, 20, 100, 100]
    return data


def make_obb_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return image, mask, one normalized oriented box, and labels."""
    data = make_mask_data(rng)
    data.update(
        bboxes=np.array([[0.25, 0.25, 0.7, 0.65, 30.0]], dtype=np.float32),
        bbox_labels=[1],
    )
    return data


def make_keypoint_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return image, mask, keypoints, and labels."""
    data = make_mask_data(rng)
    data.update(
        keypoints=np.array([[24.0, 32.0], [72.0, 80.0]], dtype=np.float32),
        keypoint_labels=[1, 2],
    )
    return data


def make_volume_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return a fresh color volume and a non-empty 3D mask."""
    volume = rng.integers(0, 256, VOLUME_SHAPE, dtype=np.uint8)
    mask3d = np.zeros(VOLUME_SHAPE[:3], dtype=np.uint8)
    mask3d[:, 8:48, 8:48] = 1
    return {"volume": volume, "mask3d": mask3d}


def make_image_batch_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return a fresh batch of RGB images."""
    return {"images": rng.integers(0, 256, (2, *IMAGE_SHAPE), dtype=np.uint8)}


def make_volume_batch_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return a fresh batch of color volumes."""
    return {"volumes": rng.integers(0, 256, (2, *VOLUME_SHAPE), dtype=np.uint8)}


def make_reference_data(metadata_key: str) -> ContractDataFactory:
    """Build a factory for transforms that consume a reference-image list."""

    def factory(rng: np.random.Generator) -> dict[str, Any]:
        data = make_image_data(rng)
        reference = rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8)
        data[metadata_key] = [reference]
        return data

    return factory


def remap_data_key(factory: ContractDataFactory, source_key: str, target_key: str) -> ContractDataFactory:
    """Wrap a factory so a configurable target or metadata key remains in sync with init kwargs."""

    def remapped(rng: np.random.Generator) -> dict[str, Any]:
        data = factory(rng)
        data[target_key] = data.pop(source_key)
        return data

    return remapped


def make_mosaic_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return a base sample and enough source samples for a 2x2 mosaic."""
    data = make_mask_data(rng)
    sources = []
    for _ in range(3):
        source = make_mask_data(rng)
        sources.append({"image": source["image"], "mask": source["mask"]})
    data["mosaic_metadata"] = sources
    return data


def make_copy_and_paste_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return a base sample and one paste candidate."""
    data = make_mask_data(rng)
    source = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    source_mask = np.zeros((40, 40), dtype=np.uint8)
    source_mask[5:35, 5:35] = 1
    data["copy_paste_metadata"] = [
        {"image": source, "mask": source_mask, "semantic_mask": source_mask},
    ]
    return data


def make_overlay_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return a base image and one overlay element."""
    data = make_image_data(rng)
    overlay = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    overlay_mask = np.ones((24, 24), dtype=np.uint8)
    data["overlay_metadata"] = [{"image": overlay, "mask": overlay_mask}]
    return data


def make_text_data(rng: np.random.Generator) -> dict[str, Any]:
    """Return an image and normalized text placement metadata."""
    data = make_image_data(rng)
    data["textimage_metadata"] = {
        "text": "contract",
        "bbox": (0.1, 0.1, 0.8, 0.25),
    }
    return data
