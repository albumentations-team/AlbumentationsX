"""Keep transform serialization derived from each transform's public constructor."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_IMPLEMENTATION = (Path("albumentations/core/transforms_interface.py"), "BasicTransform")
METHOD_NAME = "get_transform_init_args_names"


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def collect_errors(paths: Iterable[Path]) -> list[str]:
    """Return override violations for the supplied Python source files."""
    errors: list[str] = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue

        try:
            relative_path = path.resolve().relative_to(REPO_ROOT.resolve())
        except ValueError:
            relative_path = path

        for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            for member in class_node.body:
                if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) or member.name != METHOD_NAME:
                    continue
                if (relative_path, class_node.name) != BASE_IMPLEMENTATION:
                    errors.append(
                        f"{_display_path(path)}:{member.lineno}: {METHOD_NAME}() is inherited from "
                        "BasicTransform; do not override it",
                    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Reject transform init-argument overrides")
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()

    errors = collect_errors(Path(filename) for filename in args.filenames)
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
