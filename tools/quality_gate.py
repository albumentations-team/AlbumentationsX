"""Canonical local quality gates for humans and coding agents."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from shutil import which

REPO_ROOT = Path(__file__).resolve().parents[1]

Command = tuple[str, ...]

FAST_CHECKS: tuple[Command, ...] = (
    ("ruff", "format", "--check", "albumentations", "tests", "tools"),
    ("ruff", "check", "albumentations", "tests", "tools", "--no-fix"),
    ("pre-commit", "run", "--all-files"),
    ("python", "-m", "tools.check_defaults"),
    ("pytest", "-q", "tests/test_core.py::test_compose"),
)

CHECK_GROUPS: dict[str, tuple[Command, ...]] = {
    "fast": FAST_CHECKS,
}


def resolve_command(command: Command) -> Command:
    executable = which(command[0])
    if executable is None:
        sys.stderr.write(f"Missing executable: {command[0]}\n")
        raise SystemExit(127)
    return (executable, *command[1:])


def run_checks(commands: tuple[Command, ...]) -> int:
    for command in commands:
        print("$ " + " ".join(command), flush=True)
        result = subprocess.run(resolve_command(command), cwd=REPO_ROOT, check=False)  # noqa: S603
        if result.returncode != 0:
            return result.returncode
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "group",
        nargs="?",
        default="fast",
        choices=sorted(CHECK_GROUPS),
        help="Quality-gate group to run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_checks(CHECK_GROUPS[args.group])


if __name__ == "__main__":
    sys.exit(main())
