"""Catalog-wide benchmark registry.

The registry is shared by ASV benchmarks and local coverage tooling. It keeps
the broad transform smoke layer explicit enough to catch catalog drift while
staying small enough for scheduled CI.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import albumentations
from albumentations.core.transforms_interface import BasicTransform, Transform3D, VolumeOnlyTransform
from benchmarks.common import SIZES, make_image, make_volume

REPO_ROOT = Path(__file__).resolve().parents[2]
FONT_PATH = REPO_ROOT / "tests" / "files" / "LiberationSerif-Bold.ttf"

ABSTRACT_TRANSFORM_NAMES = frozenset(
    {
        "BasicTransform",
        "DualTransform",
        "ImageOnlyTransform",
        "Transform3D",
        "VolumeOnlyTransform",
    },
)

DEDICATED_TENSOR_BENCHMARK_TRANSFORMS = {
    "ToTensor3D": "covered by the dedicated PyTorch Tensor ASV lane",
    "ToTensorV2": "covered by the dedicated PyTorch Tensor ASV lane",
}

EXPECTED_INIT_WARNINGS = (
    "FrequencyMasking is a specialized version of XYMasking.*",
    "ShiftScaleRotate is a special case of Affine transform.*",
    "TimeMasking is a specialized version of XYMasking.*",
    "TimeReverse is an alias for HorizontalFlip transform.*",
)

PARAM_OVERRIDES: Mapping[str, Mapping[str, Any]] = {
    "AtLeastOneBBoxRandomCrop": {"height": 96, "width": 96},
    "Anisotropy3D": {
        "axes": (0, 2),
        "num_axes_range": (2, 2),
        "downscale_factor_range": (2.0, 2.0),
    },
    "Affine3D": {
        "rotate_range": {"x": (3.0, 3.0), "y": (-2.0, -2.0), "z": (5.0, 5.0)},
        "scale_range": {"x": (1.05, 1.05), "y": (0.95, 0.95), "z": (1.0, 1.0)},
        "translate_percent_range": {"x": (0.02, 0.02), "y": (-0.02, -0.02), "z": (0.0, 0.0)},
    },
    "CenterCrop": {"height": 96, "width": 96},
    "CenterCrop3D": {"size": (4, 48, 48)},
    "ConstrainedCoarseDropout": {"mask_indices": [1]},
    "Crop": {"x_max": 128, "y_max": 128},
    "CropAndPad": {"px": 8},
    "CropNonEmptyMaskIfExists": {"height": 96, "width": 96},
    "Flip3D": {"flip_axes": (0, 1, 2)},
    "GridElasticDeform": {"num_grid_xy": (4, 4), "magnitude": 4},
    "LetterBox": {"size": (128, 128)},
    "LongestMaxSize": {"max_size": 160},
    "Mosaic": {"target_size": (128, 128), "cell_shape": (128, 128)},
    "Pad3D": {"padding": (1, 2, 2)},
    "PadIfNeeded3D": {"min_zyx": (10, 72, 72)},
    "PixelDistributionAdaptation": {"transform_type": "standard"},
    "RandomCrop": {"height": 96, "width": 96},
    "RandomCrop3D": {"size": (4, 48, 48)},
    "RandomRotate90_3D": {"axis_pair": (0, 2), "group_element": "r90"},
    "RandomResizedCrop": {"size": (96, 96), "scale": (0.8, 1.0)},
    "RandomSizedBBoxSafeCrop": {"height": 96, "width": 96},
    "RandomSizedCrop": {"min_max_height": (96, 128), "size": (96, 96)},
    "Resize": {"height": 128, "width": 128},
    "Resize3D": {"size": (12, 96, 96)},
    "SmallestMaxSize": {"max_size": 160},
    "TextImage": {
        "augmentations": (None,),
        "font_path": FONT_PATH,
        "font_size_fraction_range": (0.5, 0.5),
    },
    "XYMasking": {
        "mask_x_length_range": (12, 12),
        "mask_y_length_range": (12, 12),
        "num_masks_x_range": (1, 1),
        "num_masks_y_range": (1, 1),
    },
}

CHANNEL_OVERRIDES = {
    "Colorize": 1,
    "ToRGB": 1,
}

MASK_ROUTE_TRANSFORMS = frozenset(
    {
        "ConstrainedCoarseDropout",
        "CropNonEmptyMaskIfExists",
        "MaskDropout",
    },
)

BBOX_ROUTE_TRANSFORMS = frozenset(
    {
        "AtLeastOneBBoxRandomCrop",
        "BBoxSafeRandomCrop",
        "RandomSizedBBoxSafeCrop",
    },
)

METADATA_ROUTE_TRANSFORMS = frozenset(
    {
        "FDA",
        "HistogramMatching",
        "PixelDistributionAdaptation",
    },
)

MIXING_ROUTE_TRANSFORMS = frozenset(
    {
        "CopyAndPaste",
        "Mosaic",
        "OverlayElements",
    },
)


@dataclass(frozen=True)
class TransformBenchmarkSpec:
    """Single transform benchmark route."""

    name: str
    route: str
    params: Mapping[str, Any] = field(default_factory=dict)
    channels: int = 3
    size_name: str = "small"
    benchmark: bool = True
    reason: str = ""


def _public_transform_classes() -> dict[str, type[BasicTransform]]:
    transforms: dict[str, type[BasicTransform]] = {}
    for name in dir(albumentations):
        if name.startswith("_") or name in ABSTRACT_TRANSFORM_NAMES:
            continue
        obj = getattr(albumentations, name)
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, BasicTransform):
            continue
        if not obj.__module__.startswith("albumentations"):
            continue
        transforms[name] = obj
    return dict(sorted(transforms.items()))


def public_transform_names() -> tuple[str, ...]:
    """Return public concrete transform names exposed by the package."""
    return tuple(_public_transform_classes())


def _route_for_transform(name: str, transform_cls: type[BasicTransform]) -> str:
    if name in DEDICATED_TENSOR_BENCHMARK_TRANSFORMS:
        return "dedicated_tensor"
    if issubclass(transform_cls, (Transform3D, VolumeOnlyTransform)):
        return "volume"
    if name in BBOX_ROUTE_TRANSFORMS:
        return "bboxes"
    if name in MASK_ROUTE_TRANSFORMS:
        return "mask"
    if name in METADATA_ROUTE_TRANSFORMS:
        return "metadata"
    if name in MIXING_ROUTE_TRANSFORMS:
        return "mixing"
    if name == "RandomCropNearBBox":
        return "crop_bbox"
    if name == "TextImage":
        return "text"
    return "image"


def benchmark_specs() -> dict[str, TransformBenchmarkSpec]:
    """Return the benchmark route for every public transform."""
    specs: dict[str, TransformBenchmarkSpec] = {}
    for name, transform_cls in _public_transform_classes().items():
        route = _route_for_transform(name, transform_cls)
        benchmark = route != "dedicated_tensor"
        specs[name] = TransformBenchmarkSpec(
            name=name,
            route=route,
            params=PARAM_OVERRIDES.get(name, {}),
            channels=CHANNEL_OVERRIDES.get(name, 3),
            benchmark=benchmark,
            reason=DEDICATED_TENSOR_BENCHMARK_TRANSFORMS.get(name, ""),
        )
    return specs


def asv_case_ids() -> tuple[str, ...]:
    """Return catalog case ids that should be executed by ASV."""
    return tuple(name for name, spec in benchmark_specs().items() if spec.benchmark)


def instantiate_transform(spec: TransformBenchmarkSpec) -> BasicTransform:
    """Build a transform for the catalog smoke path."""
    transform_cls = _public_transform_classes()[spec.name]
    params = dict(spec.params)
    params.setdefault("p", 1.0)
    with warnings.catch_warnings():
        for message in EXPECTED_INIT_WARNINGS:
            warnings.filterwarnings("ignore", message=message, category=UserWarning)
        return transform_cls(**params)


def make_mask(size_name: str = "small") -> np.ndarray:
    """Create a deterministic semantic mask."""
    height, width = SIZES[size_name]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[height // 4 : height // 2, width // 4 : width // 2] = 1
    return mask


def make_instance_masks(size_name: str = "small") -> np.ndarray:
    """Create deterministic stacked instance masks."""
    return np.expand_dims(make_mask(size_name), axis=0)


def make_bboxes(size_name: str = "small") -> list[list[float]]:
    """Create one pascal_voc bbox safely inside the benchmark image."""
    height, width = SIZES[size_name]
    return [[width * 0.2, height * 0.2, width * 0.65, height * 0.7]]


def _reference_images(size_name: str, channels: int) -> list[np.ndarray]:
    image = make_image(size_name, channels)
    return [np.flipud(image).copy(), np.fliplr(image).copy()]


def _overlay_metadata(channels: int) -> list[dict[str, Any]]:
    overlay_image = np.full((32, 48, channels), 180, dtype=np.uint8)
    overlay_mask = np.ones((32, 48), dtype=np.uint8)
    return [
        {
            "bbox": [0.1, 0.1, 0.35, 0.3],
            "image": overlay_image,
            "mask": overlay_mask,
            "mask_id": 2,
        },
    ]


def _copy_paste_metadata(channels: int) -> list[dict[str, Any]]:
    donor_image = np.full((48, 48, channels), 200, dtype=np.uint8)
    donor_mask = np.zeros((48, 48), dtype=np.uint8)
    donor_mask[8:40, 8:40] = 1
    semantic_mask = np.zeros((48, 48), dtype=np.uint8)
    semantic_mask[8:40, 8:40] = 2
    return [
        {
            "bbox_labels": {"class_labels": 2},
            "image": donor_image,
            "mask": donor_mask,
            "semantic_mask": semantic_mask,
        },
    ]


def _mosaic_metadata(spec: TransformBenchmarkSpec) -> list[dict[str, Any]]:
    return [{"image": make_image(spec.size_name, spec.channels)} for _ in range(4)]


def _base_2d_data(spec: TransformBenchmarkSpec) -> dict[str, Any]:
    return {"image": make_image(spec.size_name, spec.channels)}


def _mask_data(spec: TransformBenchmarkSpec) -> dict[str, Any]:
    data = _base_2d_data(spec)
    data["mask"] = make_mask(spec.size_name)
    return data


def _bbox_data(spec: TransformBenchmarkSpec) -> dict[str, Any]:
    data = _mask_data(spec)
    data["bboxes"] = make_bboxes(spec.size_name)
    data["class_labels"] = [1]
    return data


def _crop_bbox_data(spec: TransformBenchmarkSpec) -> dict[str, Any]:
    data = _bbox_data(spec)
    data["cropping_bbox"] = [48, 48, 192, 192]
    return data


def _metadata_data(spec: TransformBenchmarkSpec) -> dict[str, Any]:
    data = _base_2d_data(spec)
    data_key = {
        "FDA": "fda_metadata",
        "HistogramMatching": "hm_metadata",
        "PixelDistributionAdaptation": "pda_metadata",
    }[spec.name]
    data[data_key] = _reference_images(spec.size_name, spec.channels)
    return data


def _mixing_data(spec: TransformBenchmarkSpec) -> dict[str, Any]:
    if spec.name == "OverlayElements":
        data = _mask_data(spec)
        data["overlay_metadata"] = _overlay_metadata(spec.channels)
        return data
    if spec.name == "CopyAndPaste":
        data = _bbox_data(spec)
        data["masks"] = make_instance_masks(spec.size_name)
        data["copy_paste_metadata"] = _copy_paste_metadata(spec.channels)
        return data
    data = _base_2d_data(spec)
    data["mosaic_metadata"] = _mosaic_metadata(spec)
    return data


def _text_data(spec: TransformBenchmarkSpec) -> dict[str, Any]:
    data = _base_2d_data(spec)
    data["textimage_metadata"] = [
        {
            "bbox": [0.1, 0.1, 0.85, 0.25],
            "text": "AlbumentationsX benchmark",
        },
    ]
    return data


def _volume_data(spec: TransformBenchmarkSpec) -> dict[str, Any]:
    volume = make_volume()
    mask3d = (volume[..., 0] > 127).astype(np.uint8)
    return {"mask3d": mask3d, "volume": volume}


DATA_BUILDERS: Mapping[str, Callable[[TransformBenchmarkSpec], dict[str, Any]]] = {
    "bboxes": _bbox_data,
    "crop_bbox": _crop_bbox_data,
    "image": _base_2d_data,
    "mask": _mask_data,
    "metadata": _metadata_data,
    "mixing": _mixing_data,
    "text": _text_data,
    "volume": _volume_data,
}


def make_data(spec: TransformBenchmarkSpec) -> dict[str, Any]:
    """Create a deterministic input payload for a benchmark spec."""
    return DATA_BUILDERS[spec.route](spec)


def make_compose(spec: TransformBenchmarkSpec) -> albumentations.Compose:
    """Create the benchmark Compose wrapper for a spec."""
    transform = instantiate_transform(spec)
    kwargs: dict[str, Any] = {"seed": 137, "strict": True}
    if spec.route in {"bboxes", "crop_bbox"} or spec.name == "CopyAndPaste":
        kwargs["bbox_params"] = albumentations.BboxParams(coord_format="pascal_voc", label_fields=["class_labels"])
    return albumentations.Compose([transform], **kwargs)
