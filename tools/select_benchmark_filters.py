"""Select an ASV benchmark regex for changed repository paths."""

from __future__ import annotations

import argparse
from pathlib import Path

BASELINE_PATTERNS = frozenset(
    {
        "TimeBatch",
        "TimeCatalogTransformSmoke",
        "TimeCorePipeline",
        "TimeFunctional",
    },
)

PATH_RULES: tuple[tuple[tuple[str, ...], frozenset[str]], ...] = (
    (
        (
            "albumentations/core/composition.py",
            "albumentations/core/random_utils.py",
            "albumentations/core/serialization.py",
            "albumentations/core/transforms_interface.py",
            "albumentations/core/utils.py",
        ),
        frozenset({"TimeBatch", "TimeCatalogTransformSmoke", "TimeCorePipeline"}),
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
    "TimeCorePipeline",
    "TimeFunctional",
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
    patterns = set(BASELINE_PATTERNS)
    normalized_paths = [_normalized_path(path) for path in changed_paths if _normalized_path(path)]

    for path in normalized_paths:
        for prefixes, path_patterns in PATH_RULES:
            if _matches_prefix(path, prefixes):
                patterns.update(path_patterns)

    return tuple(pattern for pattern in PATTERN_ORDER if pattern in patterns)


def select_benchmark_regex(changed_paths: list[str]) -> str:
    """Return an ASV --bench regex for the changed repository paths."""
    return "|".join(select_benchmark_patterns(changed_paths))


def _read_changed_paths(path: Path | None, positional_paths: list[str]) -> list[str]:
    if path is None:
        return positional_paths
    return [line for line in path.read_text().splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="changed repository paths")
    parser.add_argument(
        "--changed-files",
        type=Path,
        help="text file containing one changed repository path per line",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    changed_paths = _read_changed_paths(args.changed_files, args.paths)
    print(select_benchmark_regex(changed_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
