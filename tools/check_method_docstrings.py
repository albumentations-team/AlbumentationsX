#!/usr/bin/env python3
"""Check that all public methods have docstrings, except for apply methods, getters, and setters."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

OPTIONAL_DOCSTRING_METHODS: set[str] = {
    "apply",
    "apply_to_images",
    "apply_to_volume",
    "apply_to_volumes",
    "apply_to_mask",
    "apply_to_masks",
    "apply_to_mask3d",
    "apply_to_masks3d",
    "apply_to_bboxes",
    "apply_to_keypoints",
    "get_params_dependent_on_data",
    "to_dict_private",
    "targets_as_params",
    "get_params",
    "get_transform_init_args",
}


class DocstringChecker(ast.NodeVisitor):
    """AST visitor to check for missing docstrings in methods."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.errors: list[str] = []
        self.current_class = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        """Visit class definition."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def _is_property_method(self, node: ast.FunctionDef) -> bool:
        """Check if method is a property getter, setter, or deleter."""
        for decorator in node.decorator_list:
            # Check for @property decorator
            if isinstance(decorator, ast.Name) and decorator.id == "property":
                return True
            # Check for @property_name.setter or @property_name.deleter
            if isinstance(decorator, ast.Attribute) and decorator.attr in ("setter", "deleter"):
                return True
        return False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        """Check if function/method has a docstring."""
        # Skip if not in a class (not a method)
        if not self.current_class:
            self.generic_visit(node)
            return

        # Skip private methods
        if node.name.startswith("_"):
            self.generic_visit(node)
            return

        # Skip methods in the optional list
        if node.name in OPTIONAL_DOCSTRING_METHODS:
            self.generic_visit(node)
            return

        # Skip property getters, setters, and deleters
        if self._is_property_method(node):
            self.generic_visit(node)
            return

        # Check if it's a method (has self/cls as first param)
        if not node.args.args or node.args.args[0].arg not in ("self", "cls"):
            self.generic_visit(node)
            return

        # Check for docstring
        if not ast.get_docstring(node):
            self.errors.append(
                f"{self.filepath}:{node.lineno}:{node.col_offset}: "
                f"D102 Missing docstring in public method "
                f"`{self.current_class}.{node.name}`",
            )

        self.generic_visit(node)


def check_file(filepath: Path) -> list[str]:
    """Check a single Python file for missing method docstrings."""
    try:
        with filepath.open(encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        return [f"{filepath}:1:1: Error reading file: {e}"]

    try:
        tree = ast.parse(content, filename=str(filepath))
    except SyntaxError as e:
        return [f"{filepath}:{e.lineno}:1: Syntax error: {e.msg}"]

    checker = DocstringChecker(filepath)
    checker.visit(tree)
    return checker.errors


def main() -> int:
    """Main entry point."""
    # Get all Python files from command line or find them
    if len(sys.argv) > 1:
        files = [Path(f) for f in sys.argv[1:] if f.endswith(".py")]
    else:
        # Find all Python files in albumentations directory
        files = list(Path("albumentations").rglob("*.py"))

    all_errors = []
    for filepath in files:
        if filepath.is_file():
            errors = check_file(filepath)
            all_errors.extend(errors)

    if all_errors:
        for error in sorted(all_errors):
            print(error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
