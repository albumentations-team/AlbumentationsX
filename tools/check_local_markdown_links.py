"""Validate local inline Markdown links without making network requests."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
INLINE_LINK = re.compile(r"!?(?:\[[^\]]*\])\((?P<destination><[^>]*>|[^\s)]+)(?:\s+[^)]*)?\)")
REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*(?P<destination><[^>]*>|[^\s]+)")


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _destinations(lines: Iterable[str]) -> Iterator[tuple[int, str]]:
    in_fenced_block = False
    for line_number, line in enumerate(lines, start=1):
        if line.lstrip().startswith(("```", "~~~")):
            in_fenced_block = not in_fenced_block
            continue
        if in_fenced_block:
            continue
        for match in INLINE_LINK.finditer(line):
            yield line_number, match.group("destination")
        if match := REFERENCE_LINK.match(line):
            yield line_number, match.group("destination")


def _local_target(source: Path, destination: str) -> Path | None:
    destination = destination.strip("<>")
    parsed = urlsplit(destination)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    path_text = unquote(parsed.path)
    if path_text.startswith("/"):
        return (REPO_ROOT / path_text.lstrip("/")).resolve()
    return (source.parent / Path(path_text)).resolve()


def collect_errors(paths: Iterable[Path]) -> list[str]:
    """Return broken local-link errors for the supplied Markdown files."""
    errors: list[str] = []
    repository_root = REPO_ROOT.resolve()
    for path in paths:
        source = path.resolve()
        for line_number, destination in _destinations(source.read_text(encoding="utf-8").splitlines()):
            target = _local_target(source, destination)
            if target is None:
                continue
            try:
                target.relative_to(repository_root)
            except ValueError:
                errors.append(
                    f"{_display_path(source)}:{line_number}: local link escapes the repository: {destination}",
                )
                continue
            if not target.exists():
                errors.append(
                    f"{_display_path(source)}:{line_number}: local link target does not exist: {destination}",
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local Markdown link targets")
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()

    errors = collect_errors(Path(filename) for filename in args.filenames)
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
