#!/usr/bin/env python3
"""Pre-commit hook to check that classes inheriting from BaseModel (like InitSchema)
do not have default values in their field definitions.

This enforces the albumentations coding guideline:
"We do not have ANY default values in the InitSchema class"
"""

import argparse
import ast
import sys
from pathlib import Path


class DefaultValueChecker(ast.NodeVisitor):
    def __init__(self):
        self.errors: list[tuple[str, int, str]] = []
        self.current_file = ""
        self.basemodel_classes: set[str] = set()
        self.class_inheritance: dict[str, list[str]] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition node to check for BaseModel inheritance."""
        # Track class inheritance
        base_names = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                # Handle cases like pydantic.BaseModel
                base_names.append(ast.unparse(base))

        self.class_inheritance[node.name] = base_names

        # Check if this class inherits from BaseModel (directly or indirectly)
        if self._inherits_from_basemodel(node.name):
            self.basemodel_classes.add(node.name)
            self._check_class_fields(node)

        self.generic_visit(node)

    def _inherits_from_basemodel(self, class_name: str) -> bool:
        """Check if a class inherits from BaseModel directly or indirectly."""
        if class_name not in self.class_inheritance:
            return False

        bases = self.class_inheritance[class_name]

        # Direct inheritance
        for base in bases:
            if base in ("BaseModel", "pydantic.BaseModel", "BaseTransformInitSchema"):
                return True

        # Indirect inheritance (recursive check)
        return any(base in self.class_inheritance and self._inherits_from_basemodel(base) for base in bases)

    def _check_class_fields(self, node: ast.ClassDef) -> None:
        """Check for default values in class field annotations."""
        for item in node.body:
            if isinstance(item, ast.AnnAssign):
                self._check_annotated_field(item, node.name)
            elif isinstance(item, ast.Assign):
                self._check_assigned_fields(item, node.name)

    def _check_annotated_field(self, item: ast.AnnAssign, class_name: str) -> None:
        if item.value is None:
            return

        field_name = ast.unparse(item.target) if hasattr(ast, "unparse") else str(item.target)
        if self._is_allowed_default_field(field_name) or self._is_discriminator_field(item):
            return

        if not self._is_field_without_default(item.value):
            self._add_error(field_name, class_name, item.lineno)

    def _check_assigned_fields(self, item: ast.Assign, class_name: str) -> None:
        for target in item.targets:
            if isinstance(target, ast.Name) and not self._is_allowed_default_field(target.id):
                self._add_error(target.id, class_name, item.lineno)

    def _is_field_without_default(self, value: ast.expr) -> bool:
        return (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "Field"
            and not value.args
            and all(keyword.arg != "default" for keyword in value.keywords)
        )

    def _add_error(self, field_name: str, class_name: str, line_number: int) -> None:
        self.errors.append(
            (
                self.current_file,
                line_number,
                f"Field '{field_name}' in BaseModel class '{class_name}' has a default value",
            ),
        )

    def _is_allowed_default_field(self, field_name: str) -> bool:
        """Check if a field is allowed to have default values."""
        # Allow private fields, class variables, and special methods
        if field_name.startswith("_"):
            return True

        # Allow specific field names that might legitimately have defaults
        allowed_fields = {
            "model_config",  # Pydantic config
            "strict",  # Core validation system field
            "__annotations__",
            "__module__",
            "__qualname__",
        }

        return field_name in allowed_fields

    def _is_discriminator_field(self, item: ast.AnnAssign) -> bool:
        """Check if this is a discriminator field for Pydantic discriminated unions."""
        if not item.annotation:
            return False

        # Check if the annotation is a Literal type
        annotation_str = ast.unparse(item.annotation) if hasattr(ast, "unparse") else str(item.annotation)

        # Look for Literal["some_value"] pattern
        if "Literal[" in annotation_str and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
            literal_value = item.value.value
            # Check if the literal value appears in the annotation
            if f'"{literal_value}"' in annotation_str or f"'{literal_value}'" in annotation_str:
                return True

        return False


def check_file(file_path: Path) -> list[tuple[str, int, str]]:
    """Check a single Python file for default values in BaseModel classes."""
    try:
        with file_path.open(encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(file_path))
        checker = DefaultValueChecker()
        checker.current_file = str(file_path)
        checker.visit(tree)

    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return []
    except (OSError, UnicodeDecodeError) as e:
        print(f"Error processing {file_path}: {e}")
        return []
    else:
        return checker.errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that BaseModel classes don't have default values",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Python files to check",
    )
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=[],
        help="Exclude files matching this pattern",
    )

    args = parser.parse_args()

    if not args.files:
        return 0

    all_errors = []

    for file_path in args.files:
        path = Path(file_path)

        # Skip non-Python files
        if path.suffix != ".py":
            continue

        # Skip excluded patterns
        skip = False
        for pattern in args.exclude_pattern:
            if pattern in str(path):
                skip = True
                break
        if skip:
            continue

        errors = check_file(path)
        all_errors.extend(errors)

    # Report errors
    if all_errors:
        print("❌ Found default values in BaseModel classes:")
        for file_path, line_no, message in all_errors:
            print(f"  {file_path}:{line_no}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
