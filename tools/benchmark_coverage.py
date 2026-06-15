"""Validate benchmark coverage for the public transform catalog."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

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
from benchmarks.test_family_matrix import (  # noqa: E402
    ANNOTATION_CASES,
    ANNOTATION_TRANSFORMS,
    GEOMETRY_CASES,
    GEOMETRY_TRANSFORMS,
    PIXEL_CASES,
    PIXEL_TRANSFORMS,
    REFERENCE_CASES,
    REFERENCE_TRANSFORMS,
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

GEOMETRY_ALIAS_TO_TRANSFORM = {
    "affine": "Affine",
    "center_crop": "CenterCrop",
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
    "random_crop": "RandomCrop",
    "resize": "Resize",
    "rotate": "Rotate",
    "safe_rotate": "SafeRotate",
    "smallest_max_size": "SmallestMaxSize",
    "thin_plate_spline": "ThinPlateSpline",
    "transpose": "Transpose",
    "water_refraction": "WaterRefraction",
}

PIXEL_ALIAS_TO_TRANSFORM = {
    "additive_noise": "AdditiveNoise",
    "auto_contrast": "AutoContrast",
    "clahe": "CLAHE",
    "equalize": "Equalize",
    "from_float": "FromFloat",
    "gauss_noise": "GaussNoise",
    "gaussian_blur": "GaussianBlur",
    "hue_saturation_value": "HueSaturationValue",
    "image_compression": "ImageCompression",
    "median_blur": "MedianBlur",
    "motion_blur": "MotionBlur",
    "multiplicative_noise": "MultiplicativeNoise",
    "normalize": "Normalize",
    "random_brightness_contrast": "RandomBrightnessContrast",
    "random_gamma": "RandomGamma",
    "rgb_shift": "RGBShift",
    "shot_noise": "ShotNoise",
    "superpixels": "Superpixels",
    "to_float": "ToFloat",
    "to_gray": "ToGray",
    "to_rgb": "ToRGB",
    "to_sepia": "ToSepia",
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

VOLUME_ALIAS_TO_TRANSFORM = {
    "center_crop3d": "CenterCrop3D",
    "coarse_dropout3d": "CoarseDropout3D",
    "cubic_symmetry": "CubicSymmetry",
    "grid_shuffle3d": "GridShuffle3D",
    "pad3d": "Pad3D",
    "pad_if_needed3d": "PadIfNeeded3D",
    "random_crop3d": "RandomCrop3D",
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
    },
)


def _mapped_names(mapping: Mapping[str, str], names: Iterable[str]) -> set[str]:
    """Map benchmark aliases to public transform names."""
    return {mapping[name] for name in names}


def _coverage_layer_sets() -> dict[str, set[str]]:
    geometry_matrix = _mapped_names(GEOMETRY_ALIAS_TO_TRANSFORM, GEOMETRY_TRANSFORMS)
    pixel_matrix = _mapped_names(PIXEL_ALIAS_TO_TRANSFORM, PIXEL_TRANSFORMS)
    annotation_scaling = _mapped_names(ANNOTATION_ALIAS_TO_TRANSFORM, ANNOTATION_TRANSFORMS)
    reference_data = set(REFERENCE_TRANSFORMS)
    volumetric_matrix = _mapped_names(VOLUME_ALIAS_TO_TRANSFORM, VOLUME_TRANSFORMS)

    return {
        "annotation_scaling": annotation_scaling,
        "direct_kernel": set(DIRECT_KERNEL_TRANSFORMS),
        "family_matrix": geometry_matrix | pixel_matrix,
        "memory": set(MEMORY_COVERED_TRANSFORMS),
        "reference_data": reference_data,
        "volumetric_matrix": volumetric_matrix,
    }


def coverage_details() -> dict[str, Any]:
    """Return per-transform benchmark coverage metadata."""
    specs = benchmark_specs()
    layer_sets = _coverage_layer_sets()
    transforms: list[dict[str, Any]] = []

    for name, spec in specs.items():
        layers = ["optional"] if not spec.benchmark else ["catalog_smoke"]
        for layer_name, transform_names in layer_sets.items():
            if name in transform_names:
                layers.append(layer_name)

        deep_layers = [layer for layer in layers if layer not in {"catalog_smoke", "optional"}]
        transforms.append(
            {
                "benchmark": spec.benchmark,
                "layers": layers,
                "name": name,
                "optional_reason": spec.reason,
                "route": spec.route,
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
            for layer_name in ("volumetric_matrix", "direct_kernel", "memory", "optional")
        },
    )

    return {
        "kind": "benchmark-coverage-detail",
        "layer_counts": dict(sorted(layer_counts.items())),
        "public_transforms": len(transforms),
        "smoke_only_transforms": smoke_only,
        "summary": {
            "deep_coverage_transforms": len(transforms) - len(smoke_only) - layer_counts["optional"],
            "optional_transforms": layer_counts["optional"],
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
            "geometry": len(GEOMETRY_CASES),
            "pixel": len(PIXEL_CASES),
            "reference_data": len(REFERENCE_CASES),
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
        "registered_specs": len(specs),
        "asv_cases": len(asv_case_ids()),
        "optional_cases": optional,
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
    unknown_optional = sorted(set(OPTIONAL_BENCHMARK_TRANSFORMS) - public_names)
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
    unknown = sorted(referenced - spec_names)
    if unknown:
        return ["Benchmark coverage layers reference unknown transforms: " + ", ".join(unknown)]
    return []


def _validate_registry() -> list[str]:
    specs = benchmark_specs()
    errors = _validate_public_registry(set(public_transform_names()), set(specs))
    errors.extend(_validate_transform_construction(specs))
    errors.extend(_validate_coverage_layers(set(specs)))
    errors.extend(
        _validate_case_groups(
            {
                "annotation": ANNOTATION_CASES,
                "geometry": GEOMETRY_CASES,
                "pixel": PIXEL_CASES,
                "reference_data": REFERENCE_CASES,
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "check":
        return check(run_smoke=not args.no_smoke, output=args.output)
    if args.command == "summary":
        return summary(output=args.output)
    if args.command == "details":
        return details(output=args.output)
    return 2


if __name__ == "__main__":
    sys.exit(main())
