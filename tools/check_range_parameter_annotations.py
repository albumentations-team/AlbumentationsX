"""Require public constructor range parameters to remain pairs, never scalars."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from pathlib import Path

APPROVED_RANGE_TYPE_ALIASES = {
    "AxisRanges3D",
    "MaskLengthRange",
    "PixelLengthRange",
    "PositiveAxisRanges3D",
}


def _annotation_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _slice_items(node: ast.expr) -> tuple[ast.expr, ...]:
    if isinstance(node, ast.Tuple):
        return node.elts
    return (node,)


def _unwrap_annotated(node: ast.expr) -> ast.expr:
    if isinstance(node, ast.Subscript) and _annotation_name(node.value) == "Annotated":
        return _slice_items(node.slice)[0]
    return node


def _is_pair_annotation(node: ast.expr) -> bool:
    node = _unwrap_annotated(node)
    if not isinstance(node, ast.Subscript) or _annotation_name(node.value) not in {"tuple", "Tuple"}:
        return False
    return len(_slice_items(node.slice)) == 2


def _is_pair_mapping_annotation(node: ast.expr) -> bool:
    if not isinstance(node, ast.Subscript) or _annotation_name(node.value) not in {"dict", "Dict"}:
        return False
    items = _slice_items(node.slice)
    return len(items) == 2 and _is_range_annotation(items[1])


def _is_none_annotation(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_approved_range_alias(node: ast.expr) -> bool:
    return _annotation_name(node) in APPROVED_RANGE_TYPE_ALIASES


def _is_range_annotation(node: ast.expr | None) -> bool:
    if node is None:
        return False
    node = _unwrap_annotated(node)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_range_annotation(node.left) and _is_range_annotation(node.right)
    return (
        _is_none_annotation(node)
        or _is_pair_annotation(node)
        or _is_pair_mapping_annotation(node)
        or _is_approved_range_alias(node)
    )


def _constructor_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ast.arg, ...]:
    return (*node.args.posonlyargs, *node.args.args[1:], *node.args.kwonlyargs)


def collect_errors(paths: Iterable[Path]) -> list[str]:
    """Return range-annotation violations in constructors below augmentations/."""
    errors: list[str] = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue

        for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            for member in class_node.body:
                if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) or member.name != "__init__":
                    continue
                invalid_parameters = (
                    parameter
                    for parameter in _constructor_parameters(member)
                    if parameter.arg.endswith("_range") and not _is_range_annotation(parameter.annotation)
                )
                errors.extend(
                    f"{path}:{parameter.lineno}: `{parameter.arg}` must describe a pair: "
                    "tuple[T, T], an optional pair form, or an axis-to-pair map"
                    for parameter in invalid_parameters
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check transform range-parameter annotations")
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()

    errors = collect_errors(Path(filename) for filename in args.filenames)
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
