#!/usr/bin/env python3
"""Pre-commit hook to check that apply_* methods have no default argument values.

Enforces the coding guideline: "No default arguments in apply_xxx methods."
Default argument values in apply methods hide bugs where params are not properly
forwarded from get_params / get_params_dependent_on_data.

Allowed: self, *args, **kwargs, **params (keyword-only catch-alls)
Flagged: any named parameter with a default value
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

APPLY_PREFIXES = ("apply", "apply_to_")


def get_apply_methods_with_defaults(tree: ast.AST) -> list[tuple[int, str, str, str]]:
    """Return (lineno, class_name, method_name, param_name) for violations."""
    errors: list[tuple[int, str, str, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        class_name = node.name

        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            if not any(item.name.startswith(prefix) for prefix in APPLY_PREFIXES):
                continue

            args = item.args

            # Positional/regular args — defaults align to the END of args.args
            n_defaults = len(args.defaults)
            args_with_defaults = args.args[-n_defaults:] if n_defaults else []
            for arg, default in zip(args_with_defaults, args.defaults, strict=True):
                if arg.arg == "self":
                    continue
                errors.append(
                    (
                        item.lineno,
                        class_name,
                        item.name,
                        f"parameter '{arg.arg}' has default value '{ast.unparse(default)}'",
                    ),
                )

            # Keyword-only args — kw_defaults entries are None when no default is set
            for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
                if default is None:
                    continue
                errors.append(
                    (
                        item.lineno,
                        class_name,
                        item.name,
                        f"keyword-only parameter '{arg.arg}' has default value '{ast.unparse(default)}'",
                    ),
                )

    return errors


def check_file(path: Path) -> list[tuple[int, str, str, str]]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    return get_apply_methods_with_defaults(tree)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check apply_* methods have no default args")
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()

    failed = False
    for filename in args.filenames:
        path = Path(filename)
        errors = check_file(path)
        for lineno, class_name, method_name, msg in errors:
            print(f"{filename}:{lineno}: {class_name}.{method_name}: {msg}")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
