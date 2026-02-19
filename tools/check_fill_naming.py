#!/usr/bin/env python3
"""Pre-commit hook to enforce fill/fill_mask naming conventions in function signatures.

Flags parameter names `fill_value` and `fill_mask_value` in function/method definitions.
These should be `fill` and `fill_mask` respectively.

Scope: only function/method parameter names — NOT local variables, keyword call arguments
to third-party APIs (e.g. np.full(fill_value=...)), or string literals.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

BANNED_PARAMS = {
    "fill_value": "fill",
    "fill_mask_value": "fill_mask",
}


class FillNamingChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[tuple[int, str]] = []

    def _check_arg(self, arg: ast.arg) -> None:
        for banned, replacement in BANNED_PARAMS.items():
            if arg.arg == banned:
                self.errors.append(
                    (
                        arg.lineno,
                        f"Parameter named '{banned}' should be '{replacement}'",
                    ),
                )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        all_args = node.args.args + node.args.posonlyargs + node.args.kwonlyargs
        if node.args.vararg:
            all_args.append(node.args.vararg)
        if node.args.kwarg:
            all_args.append(node.args.kwarg)
        for arg in all_args:
            self._check_arg(arg)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815


def check_file(path: Path) -> list[tuple[int, str]]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    checker = FillNamingChecker()
    checker.visit(tree)
    return checker.errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check fill/fill_mask naming convention")
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()

    failed = False
    for filename in args.filenames:
        path = Path(filename)
        errors = check_file(path)
        for lineno, msg in errors:
            print(f"{filename}:{lineno}: {msg}")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
