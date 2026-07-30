"""Validate benchmark coverage for the public transform catalog."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
sys.path.insert(0, str(BENCHMARK_ROOT))

from benchmarks.catalog import (  # noqa: E402
    OPTIONAL_BENCHMARK_TRANSFORMS,
    asv_case_ids,
    benchmark_specs,
    instantiate_transform,
    make_compose,
    make_data,
    public_transform_names,
)
from benchmarks.catalog import (  # noqa: E402
    unavailable_optional_transform_names as _unavailable_optional_transform_names,
)
from benchmarks.common import CHANNELS, DTYPES, SIZES, VOLUME_SIZES  # noqa: E402
from benchmarks.test_batch_matrix import (  # noqa: E402
    IMAGE_BATCH_CASES,
    IMAGE_BATCH_TRANSFORMS,
    MASK_BATCH_CASES,
    MASK_BATCH_TRANSFORMS,
    RANDOM_TONE_CURVE_DIRECT_IMAGE_CASES,
    RANDOM_TONE_CURVE_DIRECT_VOLUME_CASES,
    SPATTER_DIRECT_CASES,
    VOLUME_BATCH_CASES,
    VOLUME_BATCH_TRANSFORMS,
)
from benchmarks.test_family_matrix import (  # noqa: E402
    ANNOTATION_CASES,
    ANNOTATION_TRANSFORMS,
    BBOX_SPECIAL_TARGET_TRANSFORMS,
    GEOMETRY_CASES,
    GEOMETRY_TRANSFORMS,
    HBB_KEYPOINT_TRANSFORMS,
    PIXEL_CASES,
    PIXEL_TRANSFORMS,
    REFERENCE_CASES,
    REFERENCE_TRANSFORMS,
    SPECIAL_TARGET_CASES,
    SPECIAL_TARGET_TRANSFORMS,
    VOLUME_CASES,
    VOLUME_TRANSFORMS,
)
from benchmarks.test_functional_kernels import (  # noqa: E402
    FUNCTIONAL_3D_CASES,
    FUNCTIONAL_BLUR_CASES,
    FUNCTIONAL_GEOMETRY_ANNOTATION_CASES,
    FUNCTIONAL_GEOMETRY_IMAGE_CASES,
    FUNCTIONAL_PIXEL_CASES,
)
from benchmarks.test_parameter_sensitivity import (  # noqa: E402
    PARAMETER_SENSITIVITY_CASES,
    PARAMETER_SENSITIVITY_TRANSFORMS,
)
from pytorch_benchmarks import test_tensor as pytorch_tensor_benchmarks  # noqa: E402


def unavailable_optional_transform_names() -> set[str]:
    """Return optional transforms unavailable in the current environment."""
    return _unavailable_optional_transform_names()


BATCH_METHOD_NAMES = (
    "apply_to_images",
    "apply_to_masks",
    "apply_to_volumes",
    "apply_to_masks3d",
)
ANNOTATION_METHOD_NAMES = (
    "apply_to_bboxes",
    "apply_to_keypoints",
)

GEOMETRY_ALIAS_TO_TRANSFORM = {
    "affine": "Affine",
    "center_crop": "CenterCrop",
    "crop": "Crop",
    "crop_and_pad": "CropAndPad",
    "d4": "D4",
    "elastic": "ElasticTransform",
    "grid_distortion": "GridDistortion",
    "grid_elastic": "GridElasticDeform",
    "horizontal_flip": "HorizontalFlip",
    "letterbox": "LetterBox",
    "longest_max_size": "LongestMaxSize",
    "optical_distortion": "OpticalDistortion",
    "pad": "Pad",
    "pad_if_needed": "PadIfNeeded",
    "perspective": "Perspective",
    "piecewise_affine": "PiecewiseAffine",
    "pixel_spread": "PixelSpread",
    "morphological": "Morphological",
    "random_crop": "RandomCrop",
    "random_crop_from_borders": "RandomCropFromBorders",
    "random_grid_shuffle": "RandomGridShuffle",
    "random_resized_crop": "RandomResizedCrop",
    "random_rotate90": "RandomRotate90",
    "random_scale": "RandomScale",
    "random_sized_crop": "RandomSizedCrop",
    "resize": "Resize",
    "rotate": "Rotate",
    "safe_rotate": "SafeRotate",
    "smallest_max_size": "SmallestMaxSize",
    "square_symmetry": "SquareSymmetry",
    "thin_plate_spline": "ThinPlateSpline",
    "transpose": "Transpose",
    "vertical_flip": "VerticalFlip",
    "water_refraction": "WaterRefraction",
}

PIXEL_ALIAS_TO_TRANSFORM = {
    "additive_noise": "AdditiveNoise",
    "advanced_blur": "AdvancedBlur",
    "annotation_artifacts": "AnnotationArtifacts",
    "atmospheric_fog": "AtmosphericFog",
    "auto_contrast": "AutoContrast",
    "blur": "Blur",
    "clahe": "CLAHE",
    "channel_dropout": "ChannelDropout",
    "channel_shuffle": "ChannelShuffle",
    "channel_swap": "ChannelSwap",
    "chromatic_aberration": "ChromaticAberration",
    "color_jitter": "ColorJitter",
    "colorize": "Colorize",
    "coarse_dropout": "CoarseDropout",
    "dithering": "Dithering",
    "downscale": "Downscale",
    "emboss": "Emboss",
    "enhance": "Enhance",
    "erasing": "Erasing",
    "exposure_matching": "ExposureMatching",
    "fancy_pca": "FancyPCA",
    "film_grain": "FilmGrain",
    "defocus": "Defocus",
    "grid_dropout": "GridDropout",
    "grid_mask": "GridMask",
    "equalize": "Equalize",
    "from_float": "FromFloat",
    "gauss_noise": "GaussNoise",
    "gaussian_blur": "GaussianBlur",
    "glass_blur": "GlassBlur",
    "halftone": "Halftone",
    "he_stain": "HEStain",
    "hue_saturation_value": "HueSaturationValue",
    "image_compression": "ImageCompression",
    "illumination": "Illumination",
    "invert_img": "InvertImg",
    "iso_noise": "ISONoise",
    "lambda": "Lambda",
    "lens_flare": "LensFlare",
    "median_blur": "MedianBlur",
    "mode_filter": "ModeFilter",
    "motion_blur": "MotionBlur",
    "multiplicative_noise": "MultiplicativeNoise",
    "noop": "NoOp",
    "normalize": "Normalize",
    "photometric_distort": "PhotoMetricDistort",
    "pixel_dropout": "PixelDropout",
    "planckian_jitter": "PlanckianJitter",
    "plasma_brightness_contrast": "PlasmaBrightnessContrast",
    "plasma_shadow": "PlasmaShadow",
    "random_brightness_contrast": "RandomBrightnessContrast",
    "random_fog": "RandomFog",
    "random_gamma": "RandomGamma",
    "random_gravel": "RandomGravel",
    "random_rain": "RandomRain",
    "random_shadow": "RandomShadow",
    "random_snow": "RandomSnow",
    "random_sun_flare": "RandomSunFlare",
    "random_tone_curve": "RandomToneCurve",
    "random_tone_curve_per_channel": "RandomToneCurve",
    "rgb_shift": "RGBShift",
    "ringing_overshoot": "RingingOvershoot",
    "salt_and_pepper": "SaltAndPepper",
    "sharpen": "Sharpen",
    "shot_noise": "ShotNoise",
    "posterize": "Posterize",
    "solarize": "Solarize",
    "spatter": "Spatter",
    "superpixels": "Superpixels",
    "to_float": "ToFloat",
    "to_gray": "ToGray",
    "to_rgb": "ToRGB",
    "to_sepia": "ToSepia",
    "unsharp_mask": "UnsharpMask",
    "vignetting": "Vignetting",
    "xy_masking": "XYMasking",
    "zoom_blur": "ZoomBlur",
}

ALIAS_COVERAGE_TRANSFORMS = {
    "FrequencyMasking": "XYMasking",
    "ShiftScaleRotate": "Affine",
    "TimeMasking": "XYMasking",
    "TimeReverse": "HorizontalFlip",
}

ANNOTATION_ALIAS_TO_TRANSFORM = {
    "hbb_affine": "Affine",
    "hbb_horizontal_flip": "HorizontalFlip",
    "hbb_perspective": "Perspective",
    "hbb_safe_crop": "RandomSizedBBoxSafeCrop",
    "obb_horizontal_flip": "HorizontalFlip",
    "obb_random_scale": "RandomScale",
    "obb_resize": "Resize",
}

SPECIAL_TARGET_ALIAS_TO_TRANSFORM = {
    "at_least_one_bbox_random_crop": "AtLeastOneBBoxRandomCrop",
    "bbox_safe_random_crop": "BBoxSafeRandomCrop",
    "constrained_coarse_dropout": "ConstrainedCoarseDropout",
    "crop_non_empty_mask_if_exists": "CropNonEmptyMaskIfExists",
    "mask_dropout": "MaskDropout",
    "random_crop_near_bbox": "RandomCropNearBBox",
}

VOLUME_ALIAS_TO_TRANSFORM = {
    "center_crop3d": "CenterCrop3D",
    "coarse_dropout3d": "CoarseDropout3D",
    "cubic_symmetry": "CubicSymmetry",
    "grid_shuffle3d": "GridShuffle3D",
    "pad3d": "Pad3D",
    "pad_if_needed3d": "PadIfNeeded3D",
    "random_crop3d": "RandomCrop3D",
}

BATCH_ALIAS_TO_TRANSFORM = {
    "channel_dropout": "ChannelDropout",
    "coarse_dropout": "CoarseDropout",
    "exposure_matching": "ExposureMatching",
    "d4": "D4",
    "gauss_noise": "GaussNoise",
    "horizontal_flip": "HorizontalFlip",
    "normalize": "Normalize",
    "random_brightness_contrast": "RandomBrightnessContrast",
    "random_rotate90": "RandomRotate90",
    "random_tone_curve": "RandomToneCurve",
    "random_tone_curve_per_channel": "RandomToneCurve",
    "resize": "Resize",
    "spatter_mud": "Spatter",
    "spatter_rain": "Spatter",
    "transpose": "Transpose",
    "vertical_flip": "VerticalFlip",
}

PARAMETER_SENSITIVITY_ALIAS_TO_TRANSFORM = {
    name: spec.public_transform for name, spec in PARAMETER_SENSITIVITY_TRANSFORMS.items()
}

DIRECT_KERNEL_TRANSFORMS = frozenset(
    {
        "Affine",
        "AutoContrast",
        "Blur",
        "CenterCrop3D",
        "ChannelShuffle",
        "CoarseDropout3D",
        "D4",
        "Defocus",
        "Equalize",
        "ExposureMatching",
        "GaussianBlur",
        "GlassBlur",
        "HorizontalFlip",
        "HueSaturationValue",
        "ImageCompression",
        "ModeFilter",
        "Normalize",
        "Pad",
        "Pad3D",
        "Perspective",
        "PixelDropout",
        "Posterize",
        "RandomGamma",
        "RandomToneCurve",
        "Resize",
        "Solarize",
        "ToGray",
        "Transpose",
        "VerticalFlip",
        "ZoomBlur",
    },
)

MEMORY_BENCHMARKS = (
    "peakmem_affine_large_rgb",
    "peakmem_batch_pipeline_medium_rgb",
    "peakmem_copy_paste_small_rgb",
    "peakmem_mosaic_small_rgb",
    "peakmem_normalize_large_rgb",
    "peakmem_resize_large_rgb",
    "peakmem_spatter_batch_large_rgb",
    "peakmem_volume_pad_medium",
)

MEMORY_COVERED_TRANSFORMS = frozenset(
    {
        "Affine",
        "CopyAndPaste",
        "GaussianBlur",
        "HorizontalFlip",
        "Mosaic",
        "Normalize",
        "PadIfNeeded3D",
        "RandomBrightnessContrast",
        "Resize",
        "Spatter",
    },
)

PYTORCH_TENSOR_TRANSFORMS = frozenset({"ToTensor3D", "ToTensorV2"})
PYTORCH_IMAGE_CASES = pytorch_tensor_benchmarks.IMAGE_CASES
PYTORCH_VOLUME_CASES = pytorch_tensor_benchmarks.VOLUME_CASES

DIRECT_KERNEL_CASE_PREFIXES_BY_TRANSFORM = {
    "Affine": ("bboxes_affine", "keypoints_affine"),
    "AutoContrast": ("auto_contrast",),
    "Blur": ("box_blur",),
    "CenterCrop3D": ("crop3d",),
    "ChannelShuffle": ("channel_shuffle",),
    "CoarseDropout3D": ("cutout3d",),
    "D4": ("d4", "bboxes_d4", "keypoints_d4"),
    "Defocus": ("defocus",),
    "Equalize": ("equalize",),
    "ExposureMatching": ("exposure_match_batch",),
    "GaussianBlur": ("convolve",),
    "GlassBlur": ("glass_blur",),
    "HorizontalFlip": ("hflip",),
    "HueSaturationValue": ("shift_hsv",),
    "ImageCompression": ("image_compression",),
    "ModeFilter": ("mode_filter",),
    "Normalize": ("normalize_per_image",),
    "Pad": ("pad_with_params",),
    "Pad3D": ("pad_3d_with_params",),
    "Perspective": ("warp_perspective",),
    "PixelDropout": ("pixel_dropout",),
    "Posterize": ("posterize",),
    "RandomGamma": ("gamma_transform",),
    "RandomToneCurve": ("move_tone_curve_shared", "move_tone_curve_per_channel"),
    "Resize": ("resize", "resize_bboxes"),
    "Solarize": ("solarize",),
    "ToGray": ("to_gray",),
    "Transpose": ("transpose",),
    "VerticalFlip": ("vflip",),
    "ZoomBlur": ("zoom_blur",),
}

MEMORY_CASES_BY_TRANSFORM = {
    "Affine": ("peakmem_affine_large_rgb",),
    "CopyAndPaste": ("peakmem_copy_paste_small_rgb",),
    "GaussianBlur": ("peakmem_batch_pipeline_medium_rgb",),
    "HorizontalFlip": ("peakmem_batch_pipeline_medium_rgb",),
    "Mosaic": ("peakmem_mosaic_small_rgb",),
    "Normalize": ("peakmem_normalize_large_rgb",),
    "PadIfNeeded3D": ("peakmem_volume_pad_medium",),
    "RandomBrightnessContrast": ("peakmem_batch_pipeline_medium_rgb",),
    "Resize": ("peakmem_resize_large_rgb",),
    "Spatter": ("peakmem_spatter_batch_large_rgb",),
}

ASV_BENCHMARKS = {
    "annotation_scaling": "benchmarks.test_family_matrix.TimeAnnotationTargets.time_transform",
    "batch_image": "benchmarks.test_batch_matrix.TimeImageBatchMatrix.time_transform",
    "batch_mask": "benchmarks.test_batch_matrix.TimeMaskBatchMatrix.time_transform",
    "batch_random_tone_curve_direct_image": (
        "benchmarks.test_batch_matrix.TimeRandomToneCurveDirectImageBatchMatrix.time_apply_to_images"
    ),
    "batch_random_tone_curve_direct_volume": (
        "benchmarks.test_batch_matrix.TimeRandomToneCurveDirectVolumeBatchMatrix.time_apply_to_volumes"
    ),
    "batch_spatter_direct": "benchmarks.test_batch_matrix.TimeSpatterDirectBatchMatrix.time_apply_to_images",
    "batch_volume": "benchmarks.test_batch_matrix.TimeVolumeBatchMatrix.time_transform",
    "catalog_smoke": "benchmarks.test_catalog_smoke.TimeCatalogTransformSmoke.time_transform_compose",
    "direct_kernel_3d": "benchmarks.test_functional_kernels.TimeFunctional3DKernels.time_kernel",
    "direct_kernel_blur": "benchmarks.test_functional_kernels.TimeFunctionalBlurKernels.time_kernel",
    "direct_kernel_geometry_annotation": (
        "benchmarks.test_functional_kernels.TimeFunctionalGeometryAnnotationKernels.time_kernel"
    ),
    "direct_kernel_geometry_image": "benchmarks.test_functional_kernels.TimeFunctionalGeometryImageKernels.time_kernel",
    "direct_kernel_pixel": "benchmarks.test_functional_kernels.TimeFunctionalPixelKernels.time_kernel",
    "family_matrix_geometry": "benchmarks.test_family_matrix.TimeGeometryFullMatrix.time_transform",
    "family_matrix_pixel": "benchmarks.test_family_matrix.TimePixelFullMatrix.time_transform",
    "memory": "benchmarks.test_family_matrix.PeakMemoryHotPaths",
    "memory_spatter": "benchmarks.test_batch_matrix.PeakMemorySpatterBatchMatrix",
    "parameter_sensitivity": "benchmarks.test_parameter_sensitivity.TimeParameterSensitivity.time_transform",
    "pytorch_tensor_2d": "pytorch_benchmarks.test_tensor.TimeToTensorV2",
    "pytorch_tensor_3d": "pytorch_benchmarks.test_tensor.TimeToTensor3D",
    "reference_data": "benchmarks.test_family_matrix.TimeReferenceDataFullMatrix.time_transform",
    "target_matrix": "benchmarks.test_family_matrix.TimeSpecialTargetMatrix.time_transform",
    "volumetric_matrix": "benchmarks.test_family_matrix.TimeVolumetricFullMatrix.time_transform",
}

DEEP_COVERAGE_LAYERS = frozenset(
    {
        "alias_coverage",
        "annotation_scaling",
        "batch_matrix",
        "direct_kernel",
        "family_matrix",
        "memory",
        "parameter_sensitivity",
        "pytorch_tensor",
        "reference_data",
        "target_matrix",
        "volumetric_matrix",
    },
)
ASV_CASE_REQUIRED_LAYERS = frozenset(
    {
        "annotation_scaling",
        "batch_matrix",
        "catalog_smoke",
        "direct_kernel",
        "family_matrix",
        "memory",
        "parameter_sensitivity",
        "pytorch_tensor",
        "reference_data",
        "target_matrix",
        "volumetric_matrix",
    },
)


@dataclass(frozen=True)
class CoverageExpectation:
    """Machine-checkable benchmark coverage contract for one transform."""

    required_layers: frozenset[str]
    required_any_layers: tuple[frozenset[str], ...] = ()
    reason: str = ""


SIZE_ORDER = tuple(SIZES)
VOLUME_SIZE_ORDER = tuple(VOLUME_SIZES)
DTYPE_ORDER = tuple(DTYPES)
CHANNEL_ORDER = CHANNELS
ANNOTATION_COUNT_ORDER = (10, 100, 1000)
BATCH_SIZE_ORDER = (2, 4, 8, 16)

AXIS_ORDERS: Mapping[str, tuple[Any, ...]] = {
    "annotation_counts": ANNOTATION_COUNT_ORDER,
    "channels": CHANNEL_ORDER,
    "dtypes": DTYPE_ORDER,
    "sizes": SIZE_ORDER,
    "volume_sizes": VOLUME_SIZE_ORDER,
}

LAYER_AXIS_REFERENCES: Mapping[str, Mapping[str, tuple[Any, ...]]] = {
    "annotation_scaling": {"annotation_counts": ANNOTATION_COUNT_ORDER},
    "batch_matrix": {
        "channels": CHANNEL_ORDER,
        "dtypes": DTYPE_ORDER,
        "sizes": SIZE_ORDER,
        "volume_sizes": VOLUME_SIZE_ORDER,
    },
    "direct_kernel": {
        "annotation_counts": ANNOTATION_COUNT_ORDER,
        "channels": CHANNEL_ORDER,
        "dtypes": DTYPE_ORDER,
        "sizes": SIZE_ORDER,
        "volume_sizes": VOLUME_SIZE_ORDER,
    },
    "family_matrix": {"channels": CHANNEL_ORDER, "dtypes": DTYPE_ORDER, "sizes": SIZE_ORDER},
    "parameter_sensitivity": {"channels": CHANNEL_ORDER, "dtypes": DTYPE_ORDER, "sizes": SIZE_ORDER},
    "pytorch_tensor": {"channels": CHANNEL_ORDER, "dtypes": DTYPE_ORDER, "sizes": SIZE_ORDER},
    "reference_data": {"sizes": SIZE_ORDER},
    "target_matrix": {"channels": CHANNEL_ORDER, "dtypes": DTYPE_ORDER, "sizes": SIZE_ORDER},
    "volumetric_matrix": {"dtypes": DTYPE_ORDER, "volume_sizes": VOLUME_SIZE_ORDER},
}

LAYER_SKIP_REASONS: Mapping[str, str] = {
    "annotation_scaling": "annotation matrix bounds expensive high-count cases to stable release-critical paths",
    "batch_matrix": "batch matrix omits large image batches to keep CI evidence stable and affordable",
    "direct_kernel": "direct-kernel cases cover the axes exercised by each shared functional hot path",
    "family_matrix": "family matrix uses the transform's supported or representative size/channel/dtype axes",
    "parameter_sensitivity": "parameter stress cases are bounded to representative axes for scheduled evidence",
    "pytorch_tensor": "optional PyTorch tensor benchmarks follow supported tensor conversion axes",
    "reference_data": "reference-data benchmarks are bounded to small/medium metadata-heavy cases",
    "target_matrix": "target matrix uses the standard image axes for target-specialized transforms",
    "volumetric_matrix": "volumetric matrix uses the supported volume size and dtype axes",
}


def _mapped_names(mapping: Mapping[str, str], names: Iterable[str]) -> set[str]:
    """Map benchmark aliases to public transform names."""
    return {mapping[name] for name in names}


def _coverage_layer_sets() -> dict[str, set[str]]:
    geometry_matrix = _mapped_names(GEOMETRY_ALIAS_TO_TRANSFORM, GEOMETRY_TRANSFORMS)
    pixel_matrix = _mapped_names(PIXEL_ALIAS_TO_TRANSFORM, PIXEL_TRANSFORMS)
    annotation_scaling = _mapped_names(ANNOTATION_ALIAS_TO_TRANSFORM, ANNOTATION_TRANSFORMS)
    batch_matrix = _mapped_names(BATCH_ALIAS_TO_TRANSFORM, IMAGE_BATCH_TRANSFORMS)
    batch_matrix |= _mapped_names(BATCH_ALIAS_TO_TRANSFORM, MASK_BATCH_TRANSFORMS)
    batch_matrix |= _mapped_names(BATCH_ALIAS_TO_TRANSFORM, VOLUME_BATCH_TRANSFORMS)
    parameter_sensitivity = _mapped_names(
        PARAMETER_SENSITIVITY_ALIAS_TO_TRANSFORM,
        PARAMETER_SENSITIVITY_TRANSFORMS,
    )
    reference_data = set(REFERENCE_TRANSFORMS)
    special_targets = _mapped_names(SPECIAL_TARGET_ALIAS_TO_TRANSFORM, SPECIAL_TARGET_TRANSFORMS)
    volumetric_matrix = _mapped_names(VOLUME_ALIAS_TO_TRANSFORM, VOLUME_TRANSFORMS)

    return {
        "alias_coverage": set(ALIAS_COVERAGE_TRANSFORMS),
        "annotation_scaling": annotation_scaling,
        "batch_matrix": batch_matrix,
        "direct_kernel": set(DIRECT_KERNEL_TRANSFORMS),
        "family_matrix": geometry_matrix | pixel_matrix,
        "memory": set(MEMORY_COVERED_TRANSFORMS),
        "parameter_sensitivity": parameter_sensitivity,
        "pytorch_tensor": set(PYTORCH_TENSOR_TRANSFORMS),
        "reference_data": reference_data,
        "target_matrix": special_targets,
        "volumetric_matrix": volumetric_matrix,
    }


def _coverage_expectation(name: str, route: str) -> CoverageExpectation:
    """Return the expected benchmark coverage layers for a public transform."""
    if name in PYTORCH_TENSOR_TRANSFORMS:
        return CoverageExpectation(
            required_layers=frozenset({"optional", "pytorch_tensor"}),
            reason="optional PyTorch tensor transforms are benchmarked in the dedicated PyTorch ASV lane",
        )
    if name in ALIAS_COVERAGE_TRANSFORMS:
        return CoverageExpectation(
            required_layers=frozenset({"catalog_smoke", "alias_coverage"}),
            reason="warning alias is covered by its canonical transform and still smoke-tested directly",
        )
    if route == "volume":
        return CoverageExpectation(
            required_layers=frozenset({"catalog_smoke", "volumetric_matrix"}),
            reason="public 3D transforms require volumetric matrix coverage",
        )
    if route in {"metadata", "mixing", "text"}:
        return CoverageExpectation(
            required_layers=frozenset({"catalog_smoke", "reference_data"}),
            reason="reference-data transforms require metadata-path coverage beyond smoke",
        )
    if route in {"bboxes", "crop_bbox", "mask"}:
        return CoverageExpectation(
            required_layers=frozenset({"catalog_smoke"}),
            required_any_layers=(frozenset({"annotation_scaling", "target_matrix"}),),
            reason="target-specialized transforms require annotation or special-target scaling coverage",
        )
    return CoverageExpectation(
        required_layers=frozenset({"catalog_smoke", "family_matrix"}),
        reason="image transforms require transform-level size/channel/dtype matrix coverage",
    )


def _format_any_layers(groups: tuple[frozenset[str], ...]) -> list[list[str]]:
    """Format alternative layer requirements for JSON output."""
    return [sorted(group) for group in groups]


def _expectation_issues(name: str, layers: set[str], expectation: CoverageExpectation) -> list[str]:
    """Return unmet coverage-contract messages for one transform."""
    issues = [
        f"missing required benchmark coverage layer '{layer}'" for layer in sorted(expectation.required_layers - layers)
    ]
    issues.extend(
        f"expected at least one benchmark coverage layer from {', '.join(sorted(alternatives))}"
        for alternatives in expectation.required_any_layers
        if layers.isdisjoint(alternatives)
    )
    return issues


def _jsonable(value: Any) -> Any:
    """Return a stable JSON-compatible representation for benchmark metadata."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)


def _ordered(values: Iterable[Any], order: tuple[Any, ...] = ()) -> list[Any]:
    """Return values in project-defined order, then lexical order for unknowns."""
    value_set = set(values)
    ordered_values = [value for value in order if value in value_set]
    remaining = sorted(value_set - set(order), key=str)
    return [*ordered_values, *remaining]


def _case_name(case_id: str) -> str:
    """Return the transform/kernel name prefix from a pipe-delimited ASV case id."""
    return case_id.split("|", 1)[0]


def _route_targets(route: str) -> list[str]:
    """Return target names exercised by a catalog smoke route."""
    if route == "bboxes":
        return ["bboxes", "image"]
    if route == "crop_bbox":
        return ["bboxes", "cropping_bbox", "image"]
    if route == "mask":
        return ["image", "mask"]
    if route in {"metadata", "mixing"}:
        return ["image", "reference_metadata"]
    if route == "text":
        return ["image", "text_metadata"]
    if route == "volume":
        return ["mask3d", "volume"]
    return ["image"]


def _memory_targets(case_id: str) -> list[str]:
    """Return target names represented by a peak-memory case."""
    if "volume" in case_id:
        return ["mask3d", "volume"]
    if "batch" in case_id:
        return ["images"]
    if "copy_paste" in case_id or "mosaic" in case_id:
        return ["image", "reference_metadata"]
    return ["image"]


def _annotation_targets(name: str) -> list[str]:
    """Return target names represented by an annotation-scaling case."""
    targets = ["bboxes", "image"]
    if name.startswith("hbb_"):
        targets.append("mask")
    if name in HBB_KEYPOINT_TRANSFORMS:
        targets.extend(["keypoint_labels", "keypoints"])
    return sorted(targets)


def _special_target_targets(name: str) -> list[str]:
    """Return target names represented by a special-target matrix case."""
    targets = ["image", "mask"]
    if name in BBOX_SPECIAL_TARGET_TRANSFORMS:
        targets.extend(["bbox_labels", "bboxes"])
    if name == "random_crop_near_bbox":
        targets.append("cropping_bbox")
    return sorted(targets)


def _parse_image_matrix_case(case_id: str) -> dict[str, Any]:
    """Parse a size/channel/dtype image matrix case id."""
    name, size_name, channels, dtype_name = case_id.split("|")
    return {
        "channels": int(channels),
        "dtype": dtype_name,
        "matrix_name": name,
        "size": size_name,
    }


def _parse_annotation_case(case_id: str) -> dict[str, Any]:
    """Parse an annotation count case id."""
    name, count = case_id.split("|")
    return {
        "annotation_count": int(count),
        "annotation_type": "obb" if name.startswith("obb_") else "hbb",
        "matrix_name": name,
    }


def _parse_volume_case(case_id: str) -> dict[str, Any]:
    """Parse a volume size/dtype case id."""
    name, size_name, dtype_name = case_id.split("|")
    return {
        "dtype": dtype_name,
        "matrix_name": name,
        "volume_size": size_name,
    }


def _parse_pytorch_case(case_id: str) -> dict[str, Any]:
    """Parse an optional PyTorch tensor benchmark case id."""
    size_name, channels, dtype_name = case_id.split("|")
    return {
        "channels": int(channels),
        "dtype": dtype_name,
        "size": size_name,
    }


def _parse_batch_case(case_id: str) -> dict[str, Any]:
    """Parse a batch matrix benchmark case id."""
    name, target_route, size_name, channels, dtype_name, batch_size = case_id.split("|")
    targets = {
        "direct_images": ["images"],
        "direct_volumes": ["volumes"],
        "images": ["images"],
        "images_and_masks": ["images", "masks"],
        "volumes_and_masks3d": ["masks3d", "volumes"],
    }[target_route]
    scenario = {
        "batch_size": int(batch_size),
        "channels": int(channels),
        "dtype": dtype_name,
        "matrix_name": name,
        "target_route": target_route,
        "targets": targets,
    }
    if target_route in {"direct_volumes", "volumes_and_masks3d"}:
        scenario["volume_size"] = size_name
    else:
        scenario["size"] = size_name
    return scenario


def _parse_parameter_sensitivity_case(case_id: str) -> dict[str, Any]:
    """Parse a parameter-sensitivity benchmark case id."""
    name, parameter_scenario, size_name, channels, dtype_name = case_id.split("|")
    spec = PARAMETER_SENSITIVITY_TRANSFORMS[name]
    return {
        "channels": int(channels),
        "dtype": dtype_name,
        "matrix_name": name,
        "parameter_scenario": parameter_scenario,
        "parameter_values": _jsonable(spec.params),
        "size": size_name,
    }


def _catalog_smoke_scenario(case: Mapping[str, str], route: str, transform_name: str) -> dict[str, Any]:
    """Return scenario metadata for a catalog smoke case."""
    return {"scope": "compose", "targets": _route_targets(route)}


def _family_matrix_scenario(case: Mapping[str, str], route: str, transform_name: str) -> dict[str, Any]:
    """Return scenario metadata for an image family matrix case."""
    return {**_parse_image_matrix_case(case["case_id"]), "scope": "compose", "targets": ["image"]}


def _annotation_scaling_scenario(case: Mapping[str, str], route: str, transform_name: str) -> dict[str, Any]:
    """Return scenario metadata for an annotation-scaling case."""
    scenario = _parse_annotation_case(case["case_id"])
    return {**scenario, "scope": "compose", "targets": _annotation_targets(scenario["matrix_name"])}


def _target_matrix_scenario(case: Mapping[str, str], route: str, transform_name: str) -> dict[str, Any]:
    """Return scenario metadata for a special-target matrix case."""
    scenario = _parse_image_matrix_case(case["case_id"])
    return {**scenario, "scope": "compose", "targets": _special_target_targets(scenario["matrix_name"])}


def _batch_matrix_scenario(case: Mapping[str, str], route: str, transform_name: str) -> dict[str, Any]:
    """Return scenario metadata for a batch matrix case."""
    scenario = _parse_batch_case(case["case_id"])
    scope = "direct_batch" if scenario["target_route"].startswith("direct_") else "compose_batch"
    return {**scenario, "scope": scope}


def _parameter_sensitivity_scenario(case: Mapping[str, str], route: str, transform_name: str) -> dict[str, Any]:
    """Return scenario metadata for a parameter-sensitivity case."""
    return {
        **_parse_parameter_sensitivity_case(case["case_id"]),
        "scope": "compose_parameter_sensitivity",
        "targets": ["image"],
    }


def _reference_data_scenario(case: Mapping[str, str], route: str, transform_name: str) -> dict[str, Any]:
    """Return scenario metadata for a reference-data matrix case."""
    name, size_name = case["case_id"].split("|")
    targets = ["image", "text_metadata"] if name == "TextImage" else ["image", "reference_metadata"]
    return {"matrix_name": name, "scope": "compose", "size": size_name, "targets": targets}


def _volumetric_matrix_scenario(case: Mapping[str, str], route: str, transform_name: str) -> dict[str, Any]:
    """Return scenario metadata for a volumetric matrix case."""
    return {**_parse_volume_case(case["case_id"]), "scope": "compose", "targets": ["mask3d", "volume"]}


def _memory_scenario(case: Mapping[str, str], route: str, transform_name: str) -> dict[str, Any]:
    """Return scenario metadata for a peak-memory case."""
    case_id = case["case_id"]
    return {"memory_case": case_id, "scope": "memory", "targets": _memory_targets(case_id)}


def _pytorch_tensor_scenario(case: Mapping[str, str], route: str, transform_name: str) -> dict[str, Any]:
    """Return scenario metadata for an optional PyTorch tensor case."""
    targets = ["mask3d", "volume"] if transform_name == "ToTensor3D" else ["image", "images", "mask", "masks"]
    return {
        **_parse_pytorch_case(case["case_id"]),
        "batch_size": 8,
        "scope": "optional_pytorch",
        "targets": targets,
    }


ScenarioBuilder = Callable[[Mapping[str, str], str, str], dict[str, Any]]

SCENARIO_BUILDERS: Mapping[str, ScenarioBuilder] = {
    "annotation_scaling": _annotation_scaling_scenario,
    "batch_matrix": _batch_matrix_scenario,
    "catalog_smoke": _catalog_smoke_scenario,
    "direct_kernel": lambda case, _route, _transform_name: _direct_kernel_scenario(case),
    "family_matrix": _family_matrix_scenario,
    "memory": _memory_scenario,
    "parameter_sensitivity": _parameter_sensitivity_scenario,
    "pytorch_tensor": _pytorch_tensor_scenario,
    "reference_data": _reference_data_scenario,
    "target_matrix": _target_matrix_scenario,
    "volumetric_matrix": _volumetric_matrix_scenario,
}


def _case_scenario(case: Mapping[str, str], route: str, transform_name: str) -> dict[str, Any]:
    """Return reviewable scenario metadata for an ASV case."""
    layer = case["layer"]
    builder = SCENARIO_BUILDERS.get(layer)
    scenario = builder(case, route, transform_name) if builder is not None else {"scope": "unknown", "targets": []}
    return {"layer": layer, **scenario}


def _direct_kernel_scenario(case: Mapping[str, str]) -> dict[str, Any]:
    """Return scenario metadata for a direct functional-kernel ASV case."""
    benchmark = case["benchmark"]
    case_id = case["case_id"]
    if "GeometryAnnotation" in benchmark:
        scenario = _parse_annotation_case(case_id)
        return {
            **scenario,
            "kernel_group": "geometry_annotation",
            "scope": "functional_kernel",
            "targets": ["annotations"],
        }
    if "3D" in benchmark:
        scenario = _parse_volume_case(case_id)
        return {
            **scenario,
            "kernel_group": "volumetric",
            "scope": "functional_kernel",
            "targets": ["volume"],
        }
    scenario = _parse_image_matrix_case(case_id)
    kernel_group = "blur" if "Blur" in benchmark else "pixel" if "Pixel" in benchmark else "geometry_image"
    return {
        **scenario,
        "kernel_group": kernel_group,
        "scope": "functional_kernel",
        "targets": ["image"],
    }


def _annotate_asv_cases(cases: Iterable[dict[str, str]], route: str, transform_name: str) -> list[dict[str, Any]]:
    """Attach parsed scenario metadata to ASV case records."""
    return [
        {
            **case,
            "scenario": _case_scenario(case, route, transform_name),
        }
        for case in cases
    ]


def _scenario_contract(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return compact axes covered by a transform's ASV scenarios."""
    axes: dict[str, set[Any]] = {
        "annotation_counts": set(),
        "channels": set(),
        "configs": set(),
        "direct_kernel_groups": set(),
        "dtypes": set(),
        "layers": set(),
        "memory_cases": set(),
        "parameter_scenarios": set(),
        "scopes": set(),
        "sizes": set(),
        "targets": set(),
        "volume_sizes": set(),
    }
    batch_sizes: set[int] = set()
    case_count = 0
    for case in cases:
        case_count += 1
        axes["configs"].add(case["config"])
        scenario = case["scenario"]
        axes["layers"].add(scenario["layer"])
        for target in scenario.get("targets", []):
            axes["targets"].add(target)
        for key, axis_name in (
            ("annotation_count", "annotation_counts"),
            ("channels", "channels"),
            ("dtype", "dtypes"),
            ("kernel_group", "direct_kernel_groups"),
            ("memory_case", "memory_cases"),
            ("parameter_scenario", "parameter_scenarios"),
            ("scope", "scopes"),
            ("size", "sizes"),
            ("volume_size", "volume_sizes"),
        ):
            if key in scenario:
                axes[axis_name].add(scenario[key])
        if "batch_size" in scenario:
            batch_sizes.add(scenario["batch_size"])

    return {
        "annotation_counts": _ordered(axes["annotation_counts"]),
        "batch_sizes": _ordered(batch_sizes, BATCH_SIZE_ORDER),
        "case_count": case_count,
        "channels": _ordered(axes["channels"], CHANNEL_ORDER),
        "configs": _ordered(axes["configs"]),
        "direct_kernel_groups": _ordered(axes["direct_kernel_groups"]),
        "dtypes": _ordered(axes["dtypes"], DTYPE_ORDER),
        "layers": _ordered(axes["layers"]),
        "memory_cases": _ordered(axes["memory_cases"]),
        "parameter_scenarios": _ordered(axes["parameter_scenarios"]),
        "scopes": _ordered(axes["scopes"]),
        "sizes": _ordered(axes["sizes"], SIZE_ORDER),
        "targets": _ordered(axes["targets"]),
        "volume_sizes": _ordered(axes["volume_sizes"], VOLUME_SIZE_ORDER),
    }


def _axis_values(values: Iterable[Any], axis_name: str) -> list[Any]:
    """Return axis values in project-defined order."""
    return _ordered(values, AXIS_ORDERS.get(axis_name, ()))


def _scenario_axis_contracts(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return per-layer covered and intentionally skipped scenario axes."""
    cases_by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        cases_by_layer[case["layer"]].append(case)

    contracts: dict[str, dict[str, Any]] = {}
    for layer, layer_cases in sorted(cases_by_layer.items()):
        reference_axes = LAYER_AXIS_REFERENCES.get(layer)
        if reference_axes is None:
            continue

        layer_summary = _scenario_contract(layer_cases)
        covered: dict[str, list[Any]] = {}
        skipped: dict[str, list[Any]] = {}
        for axis_name, reference_values in reference_axes.items():
            covered_values = layer_summary[axis_name]
            if not covered_values:
                continue
            covered[axis_name] = covered_values
            skipped_values = _axis_values(set(reference_values) - set(covered_values), axis_name)
            if skipped_values:
                skipped[axis_name] = skipped_values

        contracts[layer] = {
            "covered": covered,
            "skip_reason": LAYER_SKIP_REASONS[layer] if skipped else "",
            "skipped": skipped,
        }

    return contracts


def _add_asv_case(
    cases: dict[str, list[dict[str, str]]],
    transform_name: str,
    *,
    benchmark: str,
    case_id: str,
    config: str = "default",
    layer: str,
) -> None:
    cases[transform_name].append(
        {
            "benchmark": benchmark,
            "case_id": case_id,
            "config": config,
            "layer": layer,
        },
    )


def _add_matrix_cases(
    cases: dict[str, list[dict[str, str]]],
    *,
    benchmark: str,
    case_ids: Iterable[str],
    layer: str,
    name_map: Mapping[str, str],
) -> None:
    for case_id in case_ids:
        _add_asv_case(
            cases,
            name_map[_case_name(case_id)],
            benchmark=benchmark,
            case_id=case_id,
            layer=layer,
        )


def _add_direct_kernel_cases(cases: dict[str, list[dict[str, str]]]) -> None:
    kernel_cases = {
        ASV_BENCHMARKS["direct_kernel_3d"]: FUNCTIONAL_3D_CASES,
        ASV_BENCHMARKS["direct_kernel_blur"]: FUNCTIONAL_BLUR_CASES,
        ASV_BENCHMARKS["direct_kernel_geometry_annotation"]: FUNCTIONAL_GEOMETRY_ANNOTATION_CASES,
        ASV_BENCHMARKS["direct_kernel_geometry_image"]: FUNCTIONAL_GEOMETRY_IMAGE_CASES,
        ASV_BENCHMARKS["direct_kernel_pixel"]: FUNCTIONAL_PIXEL_CASES,
    }
    cases_by_prefix: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for benchmark, case_ids in kernel_cases.items():
        for case_id in case_ids:
            cases_by_prefix[_case_name(case_id)].append((benchmark, case_id))
    for transform_name, prefixes in DIRECT_KERNEL_CASE_PREFIXES_BY_TRANSFORM.items():
        for prefix in prefixes:
            for benchmark, case_id in cases_by_prefix[prefix]:
                _add_asv_case(
                    cases,
                    transform_name,
                    benchmark=benchmark,
                    case_id=case_id,
                    layer="direct_kernel",
                )


def _benchmark_case_index() -> dict[str, list[dict[str, str]]]:
    cases: dict[str, list[dict[str, str]]] = defaultdict(list)
    for name in asv_case_ids():
        _add_asv_case(
            cases,
            name,
            benchmark=ASV_BENCHMARKS["catalog_smoke"],
            case_id=name,
            layer="catalog_smoke",
        )

    _add_matrix_cases(
        cases,
        benchmark=ASV_BENCHMARKS["family_matrix_geometry"],
        case_ids=GEOMETRY_CASES,
        layer="family_matrix",
        name_map=GEOMETRY_ALIAS_TO_TRANSFORM,
    )
    _add_matrix_cases(
        cases,
        benchmark=ASV_BENCHMARKS["family_matrix_pixel"],
        case_ids=PIXEL_CASES,
        layer="family_matrix",
        name_map=PIXEL_ALIAS_TO_TRANSFORM,
    )
    _add_matrix_cases(
        cases,
        benchmark=ASV_BENCHMARKS["annotation_scaling"],
        case_ids=ANNOTATION_CASES,
        layer="annotation_scaling",
        name_map=ANNOTATION_ALIAS_TO_TRANSFORM,
    )
    _add_matrix_cases(
        cases,
        benchmark=ASV_BENCHMARKS["target_matrix"],
        case_ids=SPECIAL_TARGET_CASES,
        layer="target_matrix",
        name_map=SPECIAL_TARGET_ALIAS_TO_TRANSFORM,
    )
    _add_matrix_cases(
        cases,
        benchmark=ASV_BENCHMARKS["volumetric_matrix"],
        case_ids=VOLUME_CASES,
        layer="volumetric_matrix",
        name_map=VOLUME_ALIAS_TO_TRANSFORM,
    )
    _add_matrix_cases(
        cases,
        benchmark=ASV_BENCHMARKS["batch_image"],
        case_ids=IMAGE_BATCH_CASES,
        layer="batch_matrix",
        name_map=BATCH_ALIAS_TO_TRANSFORM,
    )
    _add_matrix_cases(
        cases,
        benchmark=ASV_BENCHMARKS["batch_mask"],
        case_ids=MASK_BATCH_CASES,
        layer="batch_matrix",
        name_map=BATCH_ALIAS_TO_TRANSFORM,
    )
    _add_matrix_cases(
        cases,
        benchmark=ASV_BENCHMARKS["batch_random_tone_curve_direct_image"],
        case_ids=RANDOM_TONE_CURVE_DIRECT_IMAGE_CASES,
        layer="batch_matrix",
        name_map=BATCH_ALIAS_TO_TRANSFORM,
    )
    _add_matrix_cases(
        cases,
        benchmark=ASV_BENCHMARKS["batch_random_tone_curve_direct_volume"],
        case_ids=RANDOM_TONE_CURVE_DIRECT_VOLUME_CASES,
        layer="batch_matrix",
        name_map=BATCH_ALIAS_TO_TRANSFORM,
    )
    _add_matrix_cases(
        cases,
        benchmark=ASV_BENCHMARKS["batch_spatter_direct"],
        case_ids=SPATTER_DIRECT_CASES,
        layer="batch_matrix",
        name_map=BATCH_ALIAS_TO_TRANSFORM,
    )
    _add_matrix_cases(
        cases,
        benchmark=ASV_BENCHMARKS["batch_volume"],
        case_ids=VOLUME_BATCH_CASES,
        layer="batch_matrix",
        name_map=BATCH_ALIAS_TO_TRANSFORM,
    )
    _add_matrix_cases(
        cases,
        benchmark=ASV_BENCHMARKS["parameter_sensitivity"],
        case_ids=PARAMETER_SENSITIVITY_CASES,
        layer="parameter_sensitivity",
        name_map=PARAMETER_SENSITIVITY_ALIAS_TO_TRANSFORM,
    )
    for case_id in REFERENCE_CASES:
        _add_asv_case(
            cases,
            _case_name(case_id),
            benchmark=ASV_BENCHMARKS["reference_data"],
            case_id=case_id,
            layer="reference_data",
        )
    _add_direct_kernel_cases(cases)
    for transform_name, case_ids in MEMORY_CASES_BY_TRANSFORM.items():
        memory_benchmark = ASV_BENCHMARKS["memory_spatter"] if transform_name == "Spatter" else ASV_BENCHMARKS["memory"]
        for case_id in case_ids:
            _add_asv_case(
                cases,
                transform_name,
                benchmark=f"{memory_benchmark}.{case_id}",
                case_id=case_id,
                layer="memory",
            )
    for case_id in PYTORCH_IMAGE_CASES:
        _add_asv_case(
            cases,
            "ToTensorV2",
            benchmark=ASV_BENCHMARKS["pytorch_tensor_2d"],
            case_id=case_id,
            config="pytorch",
            layer="pytorch_tensor",
        )
    for case_id in PYTORCH_VOLUME_CASES:
        _add_asv_case(
            cases,
            "ToTensor3D",
            benchmark=ASV_BENCHMARKS["pytorch_tensor_3d"],
            case_id=case_id,
            config="pytorch",
            layer="pytorch_tensor",
        )
    return {
        name: sorted(items, key=lambda item: (item["layer"], item["benchmark"], item["case_id"]))
        for name, items in cases.items()
    }


def _family_labels(name: str, layers: Iterable[str]) -> list[str]:
    families = set(layers) - {"catalog_smoke", "optional"}
    if name in _mapped_names(GEOMETRY_ALIAS_TO_TRANSFORM, GEOMETRY_TRANSFORMS):
        families.add("geometry")
    if name in _mapped_names(PIXEL_ALIAS_TO_TRANSFORM, PIXEL_TRANSFORMS):
        families.add("pixel")
    if name in ALIAS_COVERAGE_TRANSFORMS:
        families.add("alias")
    return sorted(families)


def _transform_class_metadata(name: str) -> dict[str, str]:
    transform_cls = getattr(albumentations, name)
    return {
        "module": transform_cls.__module__,
        "public_api": f"albumentations.{name}",
        "qualname": transform_cls.__qualname__,
    }


def _declared_transform_methods(name: str, method_names: Iterable[str]) -> list[str]:
    """Return methods declared directly on a public transform class."""
    transform_cls = getattr(albumentations, name)
    return sorted(method_name for method_name in method_names if method_name in transform_cls.__dict__)


def _benchmark_spec_metadata(spec: Any) -> dict[str, Any]:
    return {
        "constructor_params": _jsonable(spec.params),
        "default_channels": spec.channels,
        "default_size": spec.size_name,
        "route": spec.route,
    }


def _performance_contract_entry(
    *,
    reason: str,
    required_layers: Iterable[str] = (),
    status: str,
    implementation_methods: Iterable[str] = (),
) -> dict[str, Any]:
    """Return a stable performance-contract entry for one behavior axis."""
    return {
        "implementation_methods": sorted(implementation_methods),
        "reason": reason,
        "required_layers": sorted(required_layers),
        "status": status,
    }


def _batch_performance_contract(name: str, layers: set[str]) -> dict[str, Any]:
    """Return batch-route performance expectations for one transform."""
    methods = _declared_transform_methods(name, BATCH_METHOD_NAMES)
    if name in PYTORCH_TENSOR_TRANSFORMS:
        return _performance_contract_entry(
            implementation_methods=methods,
            reason="optional tensor batch routes are measured in the dedicated PyTorch benchmark lane",
            required_layers=("pytorch_tensor",),
            status="covered_optional",
        )
    if "batch_matrix" in layers:
        return _performance_contract_entry(
            implementation_methods=methods,
            reason="batch-sensitive public routes have dedicated image, mask, or volume batch matrix cases",
            required_layers=("batch_matrix",),
            status="covered",
        )
    if methods:
        return _performance_contract_entry(
            implementation_methods=methods,
            reason=(
                "custom batch methods are inventoried for review; current release-critical evidence comes from "
                "catalog smoke, family matrices, direct kernels, and core batch dispatch until this route is promoted"
            ),
            status="tracked_without_dedicated_matrix",
        )
    return _performance_contract_entry(
        reason="transform does not declare a custom batch route that needs dedicated batch scaling evidence",
        status="not_required",
    )


def _annotation_performance_contract(name: str, route: str, layers: set[str]) -> dict[str, Any]:
    """Return annotation-scaling performance expectations for one transform."""
    methods = _declared_transform_methods(name, ANNOTATION_METHOD_NAMES)
    if "annotation_scaling" in layers:
        return _performance_contract_entry(
            implementation_methods=methods,
            reason="bbox, OBB, keypoint, and label-field scaling is measured by annotation matrix cases",
            required_layers=("annotation_scaling",),
            status="covered",
        )
    if "target_matrix" in layers:
        return _performance_contract_entry(
            implementation_methods=methods,
            reason="target-specialized crop/dropout route is measured by the special-target matrix",
            required_layers=("target_matrix",),
            status="covered_special_target",
        )
    if route in {"bboxes", "crop_bbox", "mask"}:
        return _performance_contract_entry(
            implementation_methods=methods,
            reason="target-specialized route must be covered by annotation scaling or target matrix evidence",
            required_layers=("annotation_scaling", "target_matrix"),
            status="missing_required_target_matrix",
        )
    if methods:
        return _performance_contract_entry(
            implementation_methods=methods,
            reason=(
                "annotation methods are inventoried for audit; representative scaling is covered by direct "
                "annotation kernels, target matrices, or core processor benchmarks where applicable"
            ),
            status="tracked_without_dedicated_scaling",
        )
    return _performance_contract_entry(
        reason="transform does not declare bbox or keypoint mutation methods",
        status="not_required",
    )


def _direct_kernel_performance_contract(name: str, layers: set[str]) -> dict[str, Any]:
    """Return direct functional-kernel expectations for one transform."""
    if "direct_kernel" in layers:
        return _performance_contract_entry(
            reason="shared hot functional kernels are benchmarked directly outside Compose",
            required_layers=("direct_kernel",),
            status="covered",
        )
    if "family_matrix" in layers:
        return _performance_contract_entry(
            reason="transform-level Compose matrix is the primary evidence; no dedicated shared kernel mapping exists",
            status="covered_by_compose_matrix",
        )
    return _performance_contract_entry(
        reason="no transform-specific direct functional-kernel benchmark is required by the current contract",
        status="not_required",
    )


def _parameter_performance_contract(name: str, layers: set[str]) -> dict[str, Any]:
    """Return parameter-sensitivity performance expectations for one transform."""
    if "parameter_sensitivity" in layers:
        return _performance_contract_entry(
            reason="runtime-sensitive constructor parameters have fixed typical and stress benchmark scenarios",
            required_layers=("parameter_sensitivity",),
            status="covered",
        )
    return _performance_contract_entry(
        reason="no nonlinear or parameter-dominated runtime axis is required by the current benchmark contract",
        status="not_required",
    )


def _memory_performance_contract(name: str, layers: set[str]) -> dict[str, Any]:
    """Return peak-memory performance expectations for one transform."""
    if "memory" in layers:
        return _performance_contract_entry(
            reason="allocation-heavy path has a selected peak-memory benchmark case",
            required_layers=("memory",),
            status="covered_advisory",
        )
    return _performance_contract_entry(
        reason="no dedicated peak-memory case is required by the current allocation-risk contract",
        status="not_required",
    )


def _performance_contract(name: str, route: str, layers: Iterable[str]) -> dict[str, Any]:
    """Return behavior-specific performance expectations for one transform."""
    layer_set = set(layers)
    return {
        "annotation": _annotation_performance_contract(name, route, layer_set),
        "batch": _batch_performance_contract(name, layer_set),
        "direct_kernel": _direct_kernel_performance_contract(name, layer_set),
        "memory": _memory_performance_contract(name, layer_set),
        "parameter_sensitivity": _parameter_performance_contract(name, layer_set),
    }


def _performance_contract_issues(layers: set[str], contract: Mapping[str, Any]) -> list[str]:
    """Return unmet behavior-specific performance-contract messages."""
    issues: list[str] = []
    for axis_name, axis_contract in contract.items():
        required_layers = set(axis_contract["required_layers"])
        if not required_layers:
            continue
        if axis_contract["status"].startswith("missing_"):
            missing = sorted(required_layers - layers)
        else:
            missing = sorted(required_layers - layers)
        if missing:
            issues.append(
                f"{axis_name} performance contract missing required layer(s): {', '.join(missing)}",
            )
    return issues


def _performance_contract_status_counts(transforms: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Return status counts for each behavior-specific performance axis."""
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for transform in transforms:
        for axis_name, axis_contract in transform["performance_contract"].items():
            counters[axis_name][axis_contract["status"]] += 1
    return {axis_name: dict(sorted(counter.items())) for axis_name, counter in sorted(counters.items())}


def coverage_details() -> dict[str, Any]:
    """Return per-transform benchmark coverage metadata."""
    specs = benchmark_specs()
    layer_sets = _coverage_layer_sets()
    asv_cases = _benchmark_case_index()
    transforms: list[dict[str, Any]] = []

    for name, spec in specs.items():
        layers = ["optional"] if not spec.benchmark else ["catalog_smoke"]
        for layer_name, transform_names in layer_sets.items():
            if name in transform_names:
                layers.append(layer_name)

        transform_cases = _annotate_asv_cases(asv_cases.get(name, []), spec.route, name)
        layer_set = set(layers)
        expectation = _coverage_expectation(name, spec.route)
        performance_contract = _performance_contract(name, spec.route, layers)
        contract_issues = [
            *_expectation_issues(name, layer_set, expectation),
            *_performance_contract_issues(layer_set, performance_contract),
        ]
        deep_layers = [layer for layer in layers if layer in DEEP_COVERAGE_LAYERS]
        transforms.append(
            {
                "asv_cases": transform_cases,
                "benchmark": spec.benchmark,
                "benchmark_spec": _benchmark_spec_metadata(spec),
                "class": _transform_class_metadata(name),
                "covered_by": ALIAS_COVERAGE_TRANSFORMS.get(name, ""),
                "coverage_contract": {
                    "issues": contract_issues,
                    "reason": expectation.reason,
                    "required_any_layers": _format_any_layers(expectation.required_any_layers),
                    "required_layers": sorted(expectation.required_layers),
                    "status": "ok" if not contract_issues else "missing",
                },
                "families": _family_labels(name, layers),
                "layers": layers,
                "name": name,
                "optional_reason": spec.reason,
                "performance_contract": performance_contract,
                "route": spec.route,
                "scenario_axis_contracts": _scenario_axis_contracts(transform_cases),
                "scenario_contract": _scenario_contract(transform_cases),
                "smoke_only": spec.benchmark and not deep_layers,
            },
        )

    smoke_only = sorted(item["name"] for item in transforms if item["smoke_only"])
    layer_counts = {
        layer_name: sum(1 for item in transforms if layer_name in item["layers"])
        for layer_name in ("catalog_smoke", "family_matrix", "annotation_scaling", "reference_data")
    }
    layer_counts.update(
        {
            layer_name: sum(1 for item in transforms if layer_name in item["layers"])
            for layer_name in (
                "alias_coverage",
                "batch_matrix",
                "target_matrix",
                "volumetric_matrix",
                "direct_kernel",
                "memory",
                "parameter_sensitivity",
                "pytorch_tensor",
                "optional",
            )
        },
    )
    contract_failures = sorted(item["name"] for item in transforms if item["coverage_contract"]["issues"])

    return {
        "contract_failures": contract_failures,
        "kind": "benchmark-coverage-detail",
        "layer_counts": dict(sorted(layer_counts.items())),
        "public_transforms": len(transforms),
        "schema_version": 5,
        "smoke_only_transforms": smoke_only,
        "summary": {
            "contract_failures": len(contract_failures),
            "deep_coverage_transforms": len(transforms) - len(smoke_only) - layer_counts["optional"],
            "optional_transforms": layer_counts["optional"],
            "performance_contract_status_counts": _performance_contract_status_counts(transforms),
            "smoke_only_transforms": len(smoke_only),
        },
        "transforms": transforms,
    }


def _spec_summary() -> dict[str, Any]:
    specs = benchmark_specs()
    route_counts = Counter(spec.route for spec in specs.values())
    runnable = [spec.name for spec in specs.values() if spec.benchmark]
    optional = {spec.name: spec.reason for spec in specs.values() if not spec.benchmark}
    details = coverage_details()
    return {
        "annotation_matrix_cases": len(ANNOTATION_CASES),
        "public_transforms": len(public_transform_names()),
        "coverage_depth": details["summary"],
        "coverage_layer_counts": details["layer_counts"],
        "full_matrix_cases": {
            "annotations": len(ANNOTATION_CASES),
            "batch_image": len(IMAGE_BATCH_CASES),
            "batch_mask": len(MASK_BATCH_CASES),
            "batch_volume": len(VOLUME_BATCH_CASES),
            "geometry": len(GEOMETRY_CASES),
            "parameter_sensitivity": len(PARAMETER_SENSITIVITY_CASES),
            "pixel": len(PIXEL_CASES),
            "reference_data": len(REFERENCE_CASES),
            "special_targets": len(SPECIAL_TARGET_CASES),
            "volumetric": len(VOLUME_CASES),
        },
        "direct_kernel_cases": {
            "blur": len(FUNCTIONAL_BLUR_CASES),
            "geometry_annotations": len(FUNCTIONAL_GEOMETRY_ANNOTATION_CASES),
            "geometry_images": len(FUNCTIONAL_GEOMETRY_IMAGE_CASES),
            "pixel": len(FUNCTIONAL_PIXEL_CASES),
            "volumetric": len(FUNCTIONAL_3D_CASES),
        },
        "memory_benchmarks": len(MEMORY_BENCHMARKS),
        "pytorch_tensor_benchmark_cases": len(PYTORCH_TENSOR_TRANSFORMS),
        "registered_specs": len(specs),
        "asv_cases": len(asv_case_ids()),
        "optional_cases": optional,
        "performance_contract_status_counts": details["summary"]["performance_contract_status_counts"],
        "route_counts": dict(sorted(route_counts.items())),
        "runnable_transforms": runnable,
    }


def _validate_public_registry(public_names: set[str], spec_names: set[str]) -> list[str]:
    errors: list[str] = []
    missing = sorted(public_names - spec_names)
    unexpected = sorted(spec_names - public_names)
    if missing:
        errors.append("Missing benchmark specs: " + ", ".join(missing))
    if unexpected:
        errors.append("Benchmark specs reference unknown transforms: " + ", ".join(unexpected))
    unknown_optional = sorted(
        set(OPTIONAL_BENCHMARK_TRANSFORMS) - public_names - unavailable_optional_transform_names(),
    )
    if unknown_optional:
        errors.append("Optional benchmark transform is not public: " + ", ".join(unknown_optional))
    return errors


def _validate_transform_construction(specs: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for spec in specs.values():
        if spec.route == "optional":
            continue
        try:
            instantiate_transform(spec)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{spec.name}: cannot instantiate benchmark transform: {exc}")
    return errors


def _validate_case_groups(groups: Mapping[str, tuple[str, ...]], message: str) -> list[str]:
    errors: list[str] = []
    for group, cases in groups.items():
        if not cases:
            errors.append(message.format(group=group))
    return errors


def _validate_coverage_layers(spec_names: set[str]) -> list[str]:
    layer_sets = _coverage_layer_sets()
    referenced = set().union(*layer_sets.values())
    unknown = sorted(referenced - spec_names - unavailable_optional_transform_names())
    if unknown:
        return ["Benchmark coverage layers reference unknown transforms: " + ", ".join(unknown)]
    return []


def _validate_coverage_contracts() -> list[str]:
    details = coverage_details()
    return [
        f"{item['name']}: {issue}" for item in details["transforms"] for issue in item["coverage_contract"]["issues"]
    ]


def _validate_asv_case_metadata() -> list[str]:
    details = coverage_details()
    errors: list[str] = []
    for item in details["transforms"]:
        case_layers = {case["layer"] for case in item["asv_cases"]}
        missing_case_layers = sorted((set(item["layers"]) & ASV_CASE_REQUIRED_LAYERS) - case_layers)
        errors.extend(
            f"{item['name']}: coverage layer '{layer}' has no ASV case metadata" for layer in missing_case_layers
        )
    return errors


def _validate_parameter_sensitivity_metadata() -> list[str]:
    errors: list[str] = []
    scenarios_by_transform: dict[str, set[str]] = defaultdict(set)
    for name, spec in PARAMETER_SENSITIVITY_TRANSFORMS.items():
        if not spec.params:
            errors.append(f"{name}: parameter-sensitivity benchmark must record constructor params")
        scenarios_by_transform[spec.public_transform].add(spec.parameter_scenario)

    for transform_name, scenarios in sorted(scenarios_by_transform.items()):
        if len(scenarios) < 2:
            errors.append(f"{transform_name}: parameter-sensitivity coverage needs typical and stress scenarios")
    return errors


def _validate_registry() -> list[str]:
    specs = benchmark_specs()
    errors = _validate_public_registry(set(public_transform_names()), set(specs))
    errors.extend(_validate_transform_construction(specs))
    errors.extend(_validate_coverage_layers(set(specs)))
    errors.extend(_validate_coverage_contracts())
    errors.extend(_validate_asv_case_metadata())
    errors.extend(_validate_parameter_sensitivity_metadata())
    errors.extend(
        _validate_case_groups(
            {
                "annotation": ANNOTATION_CASES,
                "batch_image": IMAGE_BATCH_CASES,
                "batch_mask": MASK_BATCH_CASES,
                "batch_volume": VOLUME_BATCH_CASES,
                "geometry": GEOMETRY_CASES,
                "parameter_sensitivity": PARAMETER_SENSITIVITY_CASES,
                "pixel": PIXEL_CASES,
                "reference_data": REFERENCE_CASES,
                "special_targets": SPECIAL_TARGET_CASES,
                "volumetric": VOLUME_CASES,
            },
            "Missing full-matrix benchmark cases for {group}",
        ),
    )
    required_direct_groups = {
        "blur": FUNCTIONAL_BLUR_CASES,
        "geometry_annotations": FUNCTIONAL_GEOMETRY_ANNOTATION_CASES,
        "geometry_images": FUNCTIONAL_GEOMETRY_IMAGE_CASES,
        "pixel": FUNCTIONAL_PIXEL_CASES,
        "volumetric": FUNCTIONAL_3D_CASES,
    }
    errors.extend(
        _validate_case_groups(
            required_direct_groups,
            "Missing direct functional-kernel benchmark cases for {group}",
        ),
    )
    return errors


def _validate_smoke() -> list[str]:
    errors: list[str] = []
    for spec in benchmark_specs().values():
        if not spec.benchmark:
            continue
        try:
            transform = make_compose(spec)
            data = make_data(spec)
            transform(**data)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{spec.name}: smoke route failed: {exc}")
    return errors


def check(*, run_smoke: bool, output: Path | None) -> int:
    """Run benchmark coverage validation."""
    errors = _validate_registry()
    if run_smoke:
        errors.extend(_validate_smoke())

    summary = _spec_summary()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(
            f"Benchmark coverage failed: {len(errors)} issue(s) across "
            f"{summary['public_transforms']} public transforms.",
            file=sys.stderr,
        )
        return 1

    print(
        "Benchmark coverage ok: "
        f"{summary['asv_cases']} ASV cases, "
        f"{len(summary['optional_cases'])} optional cases, "
        f"{summary['public_transforms']} public transforms accounted.",
    )
    return 0


def summary(output: Path | None) -> int:
    """Write or print benchmark coverage summary."""
    data = _spec_summary()
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    return 0


def details(output: Path | None) -> int:
    """Write or print per-transform benchmark coverage details."""
    data = coverage_details()
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    return 0


def _transform_index(detail: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return coverage detail records keyed by transform name."""
    return {str(item["name"]): dict(item) for item in detail.get("transforms", [])}


def _case_keys(item: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    """Return stable ASV case keys for one transform detail record."""
    return {(str(case["config"]), str(case["layer"]), str(case["case_id"])) for case in item.get("asv_cases", [])}


def _transform_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact transform metadata for added or removed diff entries."""
    return {
        "coverage_status": item.get("coverage_contract", {}).get("status", ""),
        "layers": item.get("layers", []),
        "name": item.get("name", ""),
        "route": item.get("route", ""),
    }


def _changed_transform(base_item: Mapping[str, Any], current_item: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a compact diff for one transform when benchmark coverage changed."""
    changes: dict[str, Any] = {}
    for field in ("route", "layers", "performance_contract", "scenario_axis_contracts"):
        if base_item.get(field) != current_item.get(field):
            changes[field] = {"base": base_item.get(field), "current": current_item.get(field)}

    base_contract = base_item.get("coverage_contract", {})
    current_contract = current_item.get("coverage_contract", {})
    if base_contract.get("status") != current_contract.get("status"):
        changes["coverage_status"] = {
            "base": base_contract.get("status", ""),
            "current": current_contract.get("status", ""),
        }

    base_cases = _case_keys(base_item)
    current_cases = _case_keys(current_item)
    added_cases = sorted(current_cases - base_cases)
    removed_cases = sorted(base_cases - current_cases)
    if added_cases or removed_cases:
        changes["asv_cases"] = {
            "added": [list(case) for case in added_cases],
            "removed": [list(case) for case in removed_cases],
        }

    if not changes:
        return None
    return {"changes": changes, "name": current_item["name"]}


def coverage_diff(base_detail: Mapping[str, Any], current_detail: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return machine-readable benchmark coverage drift between two detail artifacts."""
    current = coverage_details() if current_detail is None else dict(current_detail)
    base_index = _transform_index(base_detail)
    current_index = _transform_index(current)
    added_names = sorted(set(current_index) - set(base_index))
    removed_names = sorted(set(base_index) - set(current_index))
    changed = [
        item
        for name in sorted(set(base_index) & set(current_index))
        if (item := _changed_transform(base_index[name], current_index[name])) is not None
    ]

    return {
        "base_public_transforms": base_detail.get("public_transforms", len(base_index)),
        "base_schema_version": base_detail.get("schema_version"),
        "current_public_transforms": current.get("public_transforms", len(current_index)),
        "current_schema_version": current.get("schema_version"),
        "added_transforms": [_transform_summary(current_index[name]) for name in added_names],
        "changed_transforms": changed,
        "kind": "benchmark-coverage-diff",
        "removed_transforms": [_transform_summary(base_index[name]) for name in removed_names],
        "schema_version": 1,
        "summary": {
            "added_transforms": len(added_names),
            "changed_transforms": len(changed),
            "removed_transforms": len(removed_names),
            "status": "changed" if added_names or removed_names or changed else "ok",
        },
    }


def diff_details(base_detail_path: Path, output: Path | None) -> int:
    """Write or print benchmark coverage drift from a saved detail artifact."""
    data = coverage_diff(json.loads(base_detail_path.read_text()))
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("check", help="validate benchmark catalog coverage")
    check_parser.add_argument(
        "--no-smoke",
        action="store_true",
        help="validate registry shape without executing transform smoke routes",
    )
    check_parser.add_argument(
        "--output",
        type=Path,
        help="write a JSON coverage summary",
    )
    summary_parser = subparsers.add_parser("summary", help="emit benchmark coverage JSON summary")
    summary_parser.add_argument(
        "--output",
        type=Path,
        help="write summary JSON to this path instead of stdout",
    )
    details_parser = subparsers.add_parser("details", help="emit per-transform benchmark coverage JSON")
    details_parser.add_argument(
        "--output",
        type=Path,
        help="write details JSON to this path instead of stdout",
    )
    diff_parser = subparsers.add_parser("diff", help="compare saved benchmark coverage detail to current coverage")
    diff_parser.add_argument(
        "--base-detail",
        required=True,
        type=Path,
        help="previous benchmark-coverage-detail JSON to compare against current coverage",
    )
    diff_parser.add_argument(
        "--output",
        type=Path,
        help="write diff JSON to this path instead of stdout",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "check":
        return check(run_smoke=not args.no_smoke, output=args.output)
    if args.command == "summary":
        return summary(output=args.output)
    if args.command == "details":
        return details(output=args.output)
    if args.command == "diff":
        return diff_details(base_detail_path=args.base_detail, output=args.output)
    return 2


if __name__ == "__main__":
    sys.exit(main())
