"""Canonical local quality gates for humans, CI, and coding agents."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from shutil import which

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CommandSpec:
    """One quality command plus its deliberate environment overrides."""

    args: tuple[str, ...]
    environment: tuple[tuple[str, str], ...] = ()
    display: str | None = None


LINT_CHECKS = (
    CommandSpec(("ruff", "format", "--check", "albumentations", "benchmark", "tests", "tools")),
    CommandSpec(("ruff", "check", "albumentations", "benchmark", "tests", "tools", "--no-fix")),
)
MYPY_CHECKS = (CommandSpec(("pre-commit", "run", "mypy", "--all-files")),)
PYREFLY_CHECKS = (
    CommandSpec(
        (
            "pyrefly",
            "check",
            "--output-format=github",
            "--error",
            "unnecessary-type-conversion",
            "--error",
            "implicit-import",
            "--error",
            "redundant-cast",
        ),
    ),
)
CONTRACT_CHECKS = (
    CommandSpec(("python", "-m", "tools.ci_matrix", "check")),
    CommandSpec(("python", "-m", "tools.ci_shard", "check")),
    CommandSpec(("python", "-m", "tools.benchmark_coverage", "check")),
    CommandSpec(("python", "-m", "tools.performance_budget", "check")),
    CommandSpec(("python", "tools/verify_legal_integrity.py")),
    CommandSpec(("python", "-m", "tools.verify_regression_vectors", "--all")),
    CommandSpec(
        ("pre-commit", "run", "--all-files"),
        environment=(("SKIP", "mypy,pyrefly-check,ruff,ruff-format"),),
    ),
    CommandSpec(("python", "-m", "tools.check_defaults")),
)
SMOKE_CHECKS = (CommandSpec(("pytest", "-q", "tests/test_core.py::test_compose")),)

CHECK_GROUPS: dict[str, tuple[CommandSpec, ...]] = {
    "contracts": CONTRACT_CHECKS,
    "fast": (*LINT_CHECKS, *MYPY_CHECKS, *PYREFLY_CHECKS, *CONTRACT_CHECKS, *SMOKE_CHECKS),
    "lint": LINT_CHECKS,
    "mypy": MYPY_CHECKS,
    "pyrefly": PYREFLY_CHECKS,
    "smoke": SMOKE_CHECKS,
}


def resolve_command(command: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve the executable before passing a fixed argument vector to subprocess."""
    executable = which(command[0])
    if executable is None:
        sys.stderr.write(f"Missing executable: {command[0]}\n")
        raise SystemExit(127)
    return (executable, *command[1:])


def run_checks(commands: tuple[CommandSpec, ...]) -> int:
    """Run commands in order and stop at the first failure."""
    for command in commands:
        print(command.display or "$ " + " ".join(command.args), flush=True)
        environment = os.environ.copy()
        environment.update(command.environment)
        result = subprocess.run(  # noqa: S603 - executable is resolved from a fixed repository-owned command.
            resolve_command(command.args),
            cwd=REPO_ROOT,
            env=environment,
            check=False,
        )
        if result.returncode != 0:
            return result.returncode
    return 0


def _read_paths(path: Path, *, null_delimited: bool, markdown_only: bool = False) -> tuple[str, ...]:
    separator = b"\0" if null_delimited else b"\n"
    raw_paths = (item.decode("utf-8") for item in path.read_bytes().split(separator) if item)
    safe_paths: set[str] = set()
    for raw_path in raw_paths:
        candidate = PurePosixPath(raw_path.replace("\\", "/"))
        if candidate.is_absolute() or ".." in candidate.parts:
            continue
        repository_path = REPO_ROOT.joinpath(*candidate.parts)
        is_selected_type = not markdown_only or repository_path.suffix.casefold() in {".md", ".mdx"}
        if repository_path.is_file() and is_selected_type:
            safe_paths.add(candidate.as_posix())
    return tuple(sorted(safe_paths))


def markdown_checks(paths: tuple[str, ...]) -> tuple[CommandSpec, ...]:
    """Return changed-file Markdown checks without evaluating path text in a shell."""
    if not paths:
        return ()
    return (
        CommandSpec(
            ("pre-commit", "run", "--files", *paths),
            environment=(("SKIP", "mypy,pyrefly-check,ruff,ruff-format"),),
            display=f"$ pre-commit run --files <{len(paths)} Markdown files>",
        ),
    )


def scoped_contract_checks(paths: tuple[str, ...]) -> tuple[CommandSpec, ...]:
    """Scope filename-aware pre-commit hooks while preserving global contracts."""
    scoped: list[CommandSpec] = []
    for command in CONTRACT_CHECKS:
        if command.args == ("pre-commit", "run", "--all-files"):
            if not paths:
                continue
            scoped.append(
                CommandSpec(
                    ("pre-commit", "run", "--files", *paths),
                    environment=command.environment,
                    display=f"$ pre-commit run --files <{len(paths)} changed files>",
                ),
            )
        else:
            scoped.append(command)
    return tuple(scoped)


def parse_args() -> argparse.Namespace:
    """Parse the quality group and optional changed-file input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "group",
        nargs="?",
        default="fast",
        choices=[*sorted(CHECK_GROUPS), "markdown"],
        help="Quality-gate group to run.",
    )
    parser.add_argument("--paths-file", type=Path, help="Changed paths for the Markdown-only group.")
    parser.add_argument("--null", action="store_true", help="Read the paths file as NUL-delimited data.")
    return parser.parse_args()


def main() -> int:
    """Run the requested quality group."""
    args = parse_args()
    if args.group == "markdown":
        if args.paths_file is None:
            sys.stderr.write("markdown requires --paths-file\n")
            return 2
        paths = _read_paths(args.paths_file, null_delimited=args.null, markdown_only=True)
        return run_checks(markdown_checks(paths))
    if args.group == "contracts" and args.paths_file is not None:
        paths = _read_paths(args.paths_file, null_delimited=args.null)
        return run_checks(scoped_contract_checks(paths))
    if args.paths_file is not None:
        sys.stderr.write("--paths-file is only valid for the markdown and contracts groups\n")
        return 2
    return run_checks(CHECK_GROUPS[args.group])


if __name__ == "__main__":
    sys.exit(main())
