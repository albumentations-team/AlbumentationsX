"""Select a bounded ASV benchmark regex."""

from __future__ import annotations

import argparse
from pathlib import Path

PROFILE_PATTERNS = {
    "release-core": (
        "TimeCorePipeline.time_single_transform_compose",
        "TimeCorePipeline.time_skip_transform_compose",
        "TimeCorePipeline.time_noop_compose",
        "TimeCorePipeline.time_noop_probability_compose",
        "TimeCorePipeline.time_multi_transform_compose",
        "TimeCorePipelineTargetProcessors.time_bbox_keypoint_processor_roundtrip",
        "TimeGeometricTransforms.time_horizontal_flip",
        "TimeGeometricTransforms.time_resize",
        "TimeGeometricTransforms.time_pad_if_needed",
        "TimeGeometricTransforms.time_affine",
        "TimeGeometricTransforms.peakmem_affine",
        "TimePixelTransforms.time_random_brightness_contrast",
        "TimePixelTransforms.time_gaussian_blur",
        "TimePixelTransforms.time_normalize",
        "TimePixelTransforms.peakmem_normalize",
        "TimeMixingTransforms.time_mosaic",
        "TimeMixingTransforms.peakmem_mosaic",
        "TimeVolumetricTransforms.time_center_crop3d",
        "TimeVolumetricTransforms.time_pad_if_needed3d",
        "TimeVolumetricTransforms.peakmem_center_crop3d",
        "TimeVolumetricTransforms.peakmem_pad_if_needed3d",
    ),
}

PATH_RULES: tuple[tuple[tuple[str, ...], frozenset[str]], ...] = (
    (
        (
            "albumentations/core/composition.py",
            "albumentations/core/random_utils.py",
            "albumentations/core/serialization.py",
            "albumentations/core/transforms_interface.py",
            "albumentations/core/utils.py",
        ),
        frozenset({"TimeBatch", "TimeCatalogTransformSmoke", "TimeComposeFullMatrix", "TimeCorePipeline"}),
    ),
    (
        (
            "albumentations/core/bbox_utils.py",
            "albumentations/core/keypoints_utils.py",
            "albumentations/core/label_manager.py",
        ),
        frozenset(
            {
                "TimeAnnotationTargets",
                "TimeFunctionalGeometryAnnotationKernels",
                "TimeSpecialTargetMatrix",
            },
        ),
    ),
    (
        (
            "albumentations/augmentations/crops/",
            "albumentations/augmentations/geometric/",
        ),
        frozenset(
            {
                "TimeAnnotationTargets",
                "TimeBatch",
                "TimeFunctionalGeometry",
                "TimeGeometryFullMatrix",
                "TimeParameterSensitivity",
                "TimeSpecialTargetMatrix",
            },
        ),
    ),
    (
        (
            "albumentations/augmentations/blur/",
            "albumentations/augmentations/pixel/",
            "albumentations/augmentations/spectrogram/",
        ),
        frozenset(
            {
                "PeakMemory",
                "TimeFunctionalBlurKernels",
                "TimeFunctionalPixelKernels",
                "TimeBatch",
                "TimeParameterSensitivity",
                "TimePixelFullMatrix",
            },
        ),
    ),
    (
        ("albumentations/augmentations/dropout/",),
        frozenset(
            {
                "PeakMemory",
                "TimeBatch",
                "TimeFunctionalPixelKernels",
                "TimeParameterSensitivity",
                "TimePixelFullMatrix",
                "TimeSpecialTargetMatrix",
            },
        ),
    ),
    (
        (
            "albumentations/augmentations/mixing/",
            "albumentations/augmentations/text/",
        ),
        frozenset({"PeakMemory", "TimeBatch", "TimeReferenceDataFullMatrix"}),
    ),
    (
        ("albumentations/augmentations/transforms3d/",),
        frozenset({"PeakMemory", "TimeBatch", "TimeFunctional3DKernels", "TimeVolumetricFullMatrix"}),
    ),
)

PATTERN_ORDER = (
    "Time",
    "PeakMemory",
    "TimeBatch",
    "TimeCatalogTransformSmoke",
    "TimeComposeFullMatrix",
    "TimeCorePipeline",
    "TimeTargetProcessorScaling",
    "TimeParameterSensitivity",
    "TimeGeometryFullMatrix",
    "TimePixelFullMatrix",
    "TimeAnnotationTargets",
    "TimeSpecialTargetMatrix",
    "TimeReferenceDataFullMatrix",
    "TimeVolumetricFullMatrix",
    "TimeFunctionalGeometry",
    "TimeFunctionalGeometryAnnotationKernels",
    "TimeFunctionalPixelKernels",
    "TimeFunctionalBlurKernels",
    "TimeFunctional3DKernels",
)


def _normalized_path(path: str) -> str:
    return path.strip().replace("\\", "/").removeprefix("./")


def _matches_prefix(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == prefix.removesuffix("/") or path.startswith(prefix) for prefix in prefixes)


def select_benchmark_patterns(changed_paths: list[str]) -> tuple[str, ...]:
    """Return ASV regex fragments for the changed repository paths."""
    patterns: set[str] = set()

    for raw_path in changed_paths:
        path = _normalized_path(raw_path)
        if not path:
            continue
        for prefixes, path_patterns in PATH_RULES:
            if _matches_prefix(path, prefixes):
                patterns.update(path_patterns)

    return tuple(pattern for pattern in PATTERN_ORDER if pattern in patterns)


def select_benchmark_regex(changed_paths: list[str]) -> str:
    """Return an ASV --bench regex for the changed repository paths."""
    return "|".join(select_benchmark_patterns(changed_paths))


def select_profile_patterns(profile: str, changed_paths: list[str] | None = None) -> tuple[str, ...]:
    """Return deterministic ASV patterns for a named evidence profile."""
    if profile in PROFILE_PATTERNS:
        if changed_paths is not None:
            raise ValueError(f"profile {profile!r} does not accept changed paths")
        return PROFILE_PATTERNS[profile]
    if profile == "changed":
        if changed_paths is None:
            raise ValueError("profile 'changed' requires changed paths")
        return select_benchmark_patterns(changed_paths)
    raise ValueError(f"unknown benchmark profile: {profile}")


def select_profile_regex(profile: str, changed_paths: list[str] | None = None) -> str:
    """Return one ASV regex for a named evidence profile."""
    return "|".join(select_profile_patterns(profile, changed_paths))


def _read_changed_paths(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("release-core", "changed"), required=True)
    parser.add_argument("paths", nargs="*", help="changed repository paths")
    parser.add_argument(
        "--changed-files",
        type=Path,
        help="text file containing one changed repository path per line",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.paths:
        raise SystemExit("positional paths are not supported; use --changed-files with --profile changed")
    if args.profile == "changed" and args.changed_files is None:
        raise SystemExit("--profile changed requires --changed-files")
    if args.profile != "changed" and args.changed_files is not None:
        raise SystemExit(f"--profile {args.profile} does not accept --changed-files")
    changed_paths = _read_changed_paths(args.changed_files) if args.changed_files is not None else None
    benchmark_regex = select_profile_regex(args.profile, changed_paths)
    if not benchmark_regex:
        raise SystemExit("No ASV benchmark family matches the changed paths; choose an explicit --bench filter.")
    print(benchmark_regex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
