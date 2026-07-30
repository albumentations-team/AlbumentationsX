"""Reject new transform class names that use the ``Random`` prefix."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

LEGACY_RANDOM_TRANSFORM_NAMES = frozenset(
    {
        "RandomBrightnessContrast",
        "RandomCrop",
        "RandomCrop3D",
        "RandomCropFromBorders",
        "RandomCropNearBBox",
        "RandomFog",
        "RandomGamma",
        "RandomGravel",
        "RandomGridShuffle",
        "RandomRain",
        "RandomResizedCrop",
        "RandomRotate90",
        "RandomScale",
        "RandomShadow",
        "RandomSizedBBoxSafeCrop",
        "RandomSizedCrop",
        "RandomSnow",
        "RandomSunFlare",
        "RandomToneCurve",
    },
)


def check_source(source: str, *, filename: str = "<unknown>") -> list[tuple[int, str]]:
    """Return line-numbered transform naming violations in Python source."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    return [
        (
            node.lineno,
            f"New transform class '{node.name}' must not use the 'Random' prefix.",
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name.startswith("Random")
        and node.name not in LEGACY_RANDOM_TRANSFORM_NAMES
    ]


def check_file(path: Path) -> list[tuple[int, str]]:
    """Return transform naming violations in a Python file."""
    return check_source(path.read_text(encoding="utf-8"), filename=str(path))


def main() -> int:
    """Check files supplied by pre-commit."""
    parser = argparse.ArgumentParser(description="Check transform class naming conventions")
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()

    failed = False
    for filename in args.filenames:
        for line_number, message in check_file(Path(filename)):
            print(f"{filename}:{line_number}: {message}")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
