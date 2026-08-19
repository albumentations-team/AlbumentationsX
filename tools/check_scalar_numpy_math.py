"""Reject NumPy math ufuncs when static syntax proves their inputs are Python scalars.

NumPy ufuncs are the right implementation for arrays and NumPy scalar contracts. For
Python ``float`` and ``int`` values they only add ufunc dispatch and scalar wrapping;
the equivalent :mod:`math` function is both clearer and substantially faster.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable
from pathlib import Path

NUMPY_MATH_TO_MATH = {
    "arccos": "acos",
    "arccosh": "acosh",
    "arcsin": "asin",
    "arcsinh": "asinh",
    "arctan": "atan",
    "arctan2": "atan2",
    "arctanh": "atanh",
    "cos": "cos",
    "cosh": "cosh",
    "deg2rad": "radians",
    "degrees": "degrees",
    "exp": "exp",
    "expm1": "expm1",
    "log": "log",
    "log10": "log10",
    "log1p": "log1p",
    "radians": "radians",
    "rad2deg": "degrees",
    "sin": "sin",
    "sinh": "sinh",
    "sqrt": "sqrt",
    "tan": "tan",
    "tanh": "tanh",
}

SCALAR_BUILTINS = {"abs", "float", "int", "max", "min", "round"}
SCALAR_TYPE_NAMES = {"float", "int"}
MAPPING_TYPE_NAMES = {"Mapping", "dict"}
SCALAR = "scalar"
SCALAR_MAPPING = "scalar_mapping"


def _attribute_chain(node: ast.expr) -> list[str] | None:
    """Return a dotted attribute chain, or ``None`` for dynamic expressions."""
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return list(reversed(parts))
    return None


def _slice_items(node: ast.expr) -> Iterable[ast.expr]:
    if isinstance(node, ast.Tuple):
        return node.elts
    return (node,)


class ScalarNumPyMathChecker(ast.NodeVisitor):
    """Track simple scalar dataflow without guessing about arrays or NumPy dtypes."""

    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}
        self.type_aliases: dict[str, str] = {}
        self.values: dict[str, str] = {}
        self.errors: list[tuple[int, str]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.aliases[alias.asname or alias.name] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            local_name = alias.asname or alias.name
            self.aliases[local_name] = f"{module}.{alias.name}" if module else alias.name

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._register_type_alias(node):
            return
        self.visit(node.value)
        value_kind = self._expression_kind(node.value)
        for target in node.targets:
            self._assign_kind(target, value_kind)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            annotation_kind = self._annotation_kind(node.annotation)
            if annotation_kind is not None:
                self.values[node.target.id] = annotation_kind
        if node.value is not None:
            self.visit(node.value)
            self._assign_kind(node.target, self._expression_kind(node.value))

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._assign_kind(node.target, self._iterated_value_kind(node.iter))
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        previous_values = self.values
        self.values = {}
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            if argument.annotation is None:
                continue
            annotation_kind = self._annotation_kind(argument.annotation)
            if annotation_kind is not None:
                self.values[argument.arg] = annotation_kind
        for statement in node.body:
            self.visit(statement)
        self.values = previous_values

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call) -> None:
        function_name = self._canonical_name(node.func)
        numpy_operation = self._numpy_operation(function_name)
        if numpy_operation is not None and self._has_scalar_arguments(node.args, numpy_operation):
            self.errors.append(
                (
                    node.lineno,
                    (
                        f"'{function_name}' receives Python scalar values; "
                        f"use 'math.{NUMPY_MATH_TO_MATH[numpy_operation]}' instead."
                    ),
                ),
            )
        self.generic_visit(node)

    def _register_type_alias(self, node: ast.Assign) -> bool:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return False
        annotation_kind = self._annotation_kind(node.value)
        if annotation_kind is None:
            return False
        self.type_aliases[node.targets[0].id] = annotation_kind
        return True

    def _annotation_kind(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            if node.id in SCALAR_TYPE_NAMES:
                return SCALAR
            return self.type_aliases.get(node.id)
        if isinstance(node, ast.Attribute):
            return SCALAR if node.attr in SCALAR_TYPE_NAMES else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left_kind = self._annotation_kind(node.left)
            right_kind = self._annotation_kind(node.right)
            return left_kind if left_kind == right_kind else None
        if isinstance(node, ast.Subscript):
            return self._subscript_annotation_kind(node)
        return None

    def _subscript_annotation_kind(self, node: ast.Subscript) -> str | None:
        base_name = self._canonical_name(node.value)
        if base_name is None or base_name.rsplit(".", 1)[-1] not in MAPPING_TYPE_NAMES:
            return None
        items = tuple(_slice_items(node.slice))
        if len(items) != 2:
            return None
        return SCALAR_MAPPING if self._annotation_kind(items[1]) == SCALAR else None

    def _assign_kind(self, target: ast.expr, value_kind: str | None) -> None:
        if value_kind is None:
            return
        if isinstance(target, ast.Name):
            self.values[target.id] = value_kind
        elif isinstance(target, (ast.Tuple, ast.List)) and value_kind == SCALAR:
            for item in target.elts:
                self._assign_kind(item, SCALAR)

    def _expression_kind(self, node: ast.expr) -> str | None:
        """Infer a scalar contract from local syntax without guessing about arrays."""
        for infer_kind in (
            self._literal_or_name_kind,
            self._attribute_or_subscript_kind,
            self._operation_kind,
            self._container_kind,
            self._call_kind,
        ):
            value_kind = infer_kind(node)
            if value_kind is not None:
                return value_kind
        return None

    def _literal_or_name_kind(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, (float, int)) and not isinstance(node.value, bool):
            return SCALAR
        if isinstance(node, ast.Name):
            return self.values.get(node.id)

        return None

    def _attribute_or_subscript_kind(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Attribute):
            canonical_name = self._canonical_name(node)
            return SCALAR if canonical_name in {"math.pi", "math.e", "numpy.pi", "numpy.e"} else None
        if isinstance(node, ast.Subscript):
            return SCALAR if self._expression_kind(node.value) == SCALAR_MAPPING else None

        return None

    def _operation_kind(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.UnaryOp):
            return self._expression_kind(node.operand)
        if isinstance(node, ast.BinOp):
            left_kind = self._expression_kind(node.left)
            right_kind = self._expression_kind(node.right)
            return SCALAR if left_kind == SCALAR and right_kind == SCALAR else None

        return None

    def _container_kind(self, node: ast.expr) -> str | None:
        if (
            isinstance(node, (ast.Tuple, ast.List))
            and node.elts
            and all(self._expression_kind(element) == SCALAR for element in node.elts)
        ):
            return SCALAR

        return None

    def _call_kind(self, node: ast.expr) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        function_name = self._canonical_name(node.func)
        if function_name in SCALAR_BUILTINS and all(self._expression_kind(arg) == SCALAR for arg in node.args):
            return SCALAR
        if function_name is not None and function_name.startswith("math."):
            return SCALAR if all(self._expression_kind(arg) == SCALAR for arg in node.args) else None
        numpy_operation = self._numpy_operation(function_name)
        if numpy_operation is not None and self._has_scalar_arguments(node.args, numpy_operation):
            return SCALAR
        return None

    def _iterated_value_kind(self, node: ast.expr) -> str | None:
        if (
            isinstance(node, (ast.Tuple, ast.List))
            and node.elts
            and all(self._expression_kind(element) == SCALAR for element in node.elts)
        ):
            return SCALAR
        return None

    def _canonical_name(self, node: ast.expr) -> str | None:
        chain = _attribute_chain(node)
        if not chain:
            return None
        root = self.aliases.get(chain[0], chain[0])
        return ".".join((root, *chain[1:]))

    @staticmethod
    def _numpy_operation(function_name: str | None) -> str | None:
        if function_name is None or not function_name.startswith("numpy."):
            return None
        operation = function_name.rsplit(".", 1)[-1]
        return operation if operation in NUMPY_MATH_TO_MATH else None

    def _has_scalar_arguments(self, arguments: list[ast.expr], operation: str) -> bool:
        required_arguments = 2 if operation == "arctan2" else 1
        return len(arguments) >= required_arguments and all(
            self._expression_kind(argument) == SCALAR for argument in arguments[:required_arguments]
        )


def check_file(path: Path) -> list[tuple[int, str]]:
    """Return scalar NumPy math calls in one Python file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    checker = ScalarNumPyMathChecker()
    checker.visit(tree)
    return checker.errors


def main() -> int:
    """Run the scalar NumPy math pre-commit check."""
    parser = argparse.ArgumentParser(description="Forbid NumPy math calls on statically proven Python scalars")
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
