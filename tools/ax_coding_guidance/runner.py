"""CLI and test-facing entry points for the unified AX guidance hook."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .diagnostics import Diagnostic
from .rules import run_all
from .source_index import SourceIndex


class InfrastructureError(RuntimeError):
    """Raised when the hook cannot build a complete source index."""


def run_sources(sources: dict[str, str]) -> list[Diagnostic]:
    """Run every AX rule against in-memory source files."""
    return run_all(SourceIndex.from_sources(sources))


def run_repo(root: Path, selected: set[str] | None = None) -> list[Diagnostic]:
    """Run every AX rule against the complete production package.

    ``selected`` filters displayed diagnostics only.  The index is always
    package-wide so a file cannot evade an inheritance or allowlist check by
    being invoked through a filename-filtered pre-commit call.
    """
    try:
        diagnostics = run_all(SourceIndex.from_repo(root))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise InfrastructureError(str(exc)) from exc
    if selected is None:
        return diagnostics
    return [diagnostic for diagnostic in diagnostics if diagnostic.path in selected]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check AlbumentationsX coding guidance")
    parser.add_argument("filenames", nargs="*", help="optional files whose diagnostics should be displayed")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    selected = None
    if args.filenames:
        selected = set()
        for filename in args.filenames:
            path = Path(filename)
            try:
                selected.add(path.resolve().relative_to(root.resolve()).as_posix())
            except ValueError:
                selected.add(path.as_posix())
    try:
        diagnostics = run_repo(root, selected)
    except InfrastructureError as exc:
        print(f"AXG infrastructure error: {exc}", file=sys.stderr)
        return 2
    for diagnostic in diagnostics:
        print(diagnostic.format())
    return 1 if diagnostics else 0


if __name__ == "__main__":
    raise SystemExit(main())
