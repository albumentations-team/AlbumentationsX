"""Reject forced staging of files that belong in the local-only workspace."""

from __future__ import annotations

import argparse
from pathlib import PurePosixPath

ALLOWED_PATHS = {PurePosixPath("_internal/.gitkeep")}


def collect_errors(filenames: tuple[str, ...]) -> list[str]:
    """Return violations for staged files below the local-only workspace."""
    errors: list[str] = []
    for filename in filenames:
        path = PurePosixPath(filename)
        if path.parts and path.parts[0] == "_internal" and path not in ALLOWED_PATHS:
            errors.append(f"{path}: _internal/ is local-only and must not be committed")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Reject staged _internal/ files")
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()

    errors = collect_errors(tuple(args.filenames))
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
