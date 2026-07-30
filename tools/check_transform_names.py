"""Reject new transform class names that use the ``Random`` prefix."""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "albumentations"

TRANSFORM_BASE_CLASS_NAMES = frozenset(
    {
        "BaseCompose",
        "BasicTransform",
        "DualTransform",
        "ImageOnlyTransform",
        "Transform3D",
    },
)

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
        "RandomOrder",
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


def _parse_source(source: str, filename: str) -> ast.Module | None:
    """Parse Python source, leaving syntax errors to the dedicated syntax checks."""
    try:
        return ast.parse(source, filename=filename)
    except SyntaxError:
        return None


def _base_class_name(base: ast.expr) -> str | None:
    """Return the unqualified name of a class base expression."""
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


def _find_transform_class_names(trees: Iterable[ast.AST]) -> frozenset[str]:
    """Find classes that transitively inherit from an Albumentations transform base."""
    class_bases: dict[str, set[str]] = {}
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = {_base_class_name(base) for base in node.bases}
                class_bases.setdefault(node.name, set()).update(base for base in bases if base is not None)

    transform_class_names = set(TRANSFORM_BASE_CLASS_NAMES)
    while derived_names := {
        class_name
        for class_name, base_names in class_bases.items()
        if class_name not in transform_class_names and not transform_class_names.isdisjoint(base_names)
    }:
        transform_class_names.update(derived_names)

    return frozenset(transform_class_names)


def find_package_transform_class_names() -> frozenset[str]:
    """Find transform classes across every production package location."""
    trees = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = _parse_source(path.read_text(encoding="utf-8"), str(path))
        if tree is not None:
            trees.append(tree)

    return _find_transform_class_names(trees)


def check_source(
    source: str,
    *,
    filename: str = "<unknown>",
    transform_class_names: frozenset[str] | None = None,
) -> list[tuple[int, str]]:
    """Return line-numbered transform naming violations in Python source."""
    tree = _parse_source(source, filename)
    if tree is None:
        return []

    known_transform_class_names = set(transform_class_names or ())
    known_transform_class_names.update(_find_transform_class_names([tree]))

    return [
        (
            node.lineno,
            f"New transform class '{node.name}' must not use the 'Random' prefix.",
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name.startswith("Random")
        and node.name in known_transform_class_names
        and node.name not in LEGACY_RANDOM_TRANSFORM_NAMES
    ]


def check_file(path: Path, *, transform_class_names: frozenset[str] | None = None) -> list[tuple[int, str]]:
    """Return transform naming violations in a Python file."""
    return check_source(
        path.read_text(encoding="utf-8"),
        filename=str(path),
        transform_class_names=transform_class_names,
    )


def main() -> int:
    """Check files supplied by pre-commit."""
    parser = argparse.ArgumentParser(description="Check transform class naming conventions")
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()

    transform_class_names = find_package_transform_class_names()
    failed = False
    for filename in args.filenames:
        for line_number, message in check_file(Path(filename), transform_class_names=transform_class_names):
            print(f"{filename}:{line_number}: {message}")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
