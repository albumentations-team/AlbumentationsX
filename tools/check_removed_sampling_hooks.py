"""Reject obsolete transform sampling hooks removed by the Compose executor rewrite."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

REMOVED_METHODS = frozenset({"get_params", "get_params_dependent_on_data"})


def find_removed_sampling_hooks(source: str, filename: str) -> list[str]:
    """Return diagnostics for obsolete method declarations in one Python source file."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    diagnostics: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for member in node.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name in REMOVED_METHODS:
                diagnostics += [
                    (
                        f"{filename}:{member.lineno}: {node.name}.{member.name} was removed; "
                        "implement sample_parameters(params, data, sampling) instead"
                    ),
                ]
    return diagnostics


def main() -> int:
    """Check every file supplied by pre-commit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()

    diagnostics = [
        diagnostic
        for filename in args.filenames
        for diagnostic in find_removed_sampling_hooks(Path(filename).read_text(encoding="utf-8"), filename)
    ]
    if diagnostics:
        print("\n".join(diagnostics))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
