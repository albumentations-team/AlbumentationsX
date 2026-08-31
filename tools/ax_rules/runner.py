"""Run enabled AlbumentationsX repository rules from ``pyproject.toml``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TypeAlias

from tools import check_docstring_format, check_naming_conflicts
from tools.ax_coding_guidance.runner import InfrastructureError, run_repo

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


RuleCheck: TypeAlias = Callable[[Path, tuple[Path, ...]], int]


class ConfigurationError(ValueError):
    """Raised when the AX rule selection is invalid."""


def _run_coding_guidance(root: Path, _: tuple[Path, ...]) -> int:
    try:
        diagnostics = run_repo(root)
    except InfrastructureError as exc:
        print(f"AXG infrastructure error: {exc}", file=sys.stderr)
        return 2
    for diagnostic in diagnostics:
        print(diagnostic.format())
    return int(bool(diagnostics))


def _run_docstring_format(root: Path, filenames: tuple[Path, ...]) -> int:
    paths = docstring_paths(root, filenames)
    errors = check_docstring_format.collect_errors(paths, root=root)
    for error in errors:
        print(error)
    return int(bool(errors))


def _run_naming_conflicts(root: Path, _: tuple[Path, ...]) -> int:
    _, _, conflicts = check_naming_conflicts.find_conflicts(str(root / "albumentations"))
    if not conflicts:
        return 0
    check_naming_conflicts.print_conflicts(conflicts)
    return 1


def _run_public_transform_docstrings(root: Path, _: tuple[Path, ...]) -> int:
    import pytest

    return pytest.main(["-q", "tests/test_docstrings.py", "-k", "short_description_length"])


def _run_readme_transform_docs(root: Path, _: tuple[Path, ...]) -> int:
    from tools import make_transforms_docs

    try:
        make_transforms_docs.check_transform_docs(root / "README.md")
    except ValueError as exc:
        print(exc)
        return 1
    return 0


RULES: dict[str, RuleCheck] = {
    "coding-guidance": _run_coding_guidance,
    "docstring-format": _run_docstring_format,
    "naming-conflicts": _run_naming_conflicts,
    "public-transform-docstrings": _run_public_transform_docstrings,
    "readme-transform-docs": _run_readme_transform_docs,
}


def _read_rule_settings(root: Path) -> Mapping[str, Any]:
    config_path = root / "pyproject.toml"
    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"cannot read {config_path}: {exc}") from exc
    tool = data.get("tool", {})
    if not isinstance(tool, dict):
        raise ConfigurationError("[tool] must be a table")
    ax_rules = tool.get("ax-rules", {})
    if not isinstance(ax_rules, dict):
        raise ConfigurationError("[tool.ax-rules] must be a table")
    rules = ax_rules.get("rules")
    if rules is None:
        return {}
    if not isinstance(rules, dict):
        raise ConfigurationError("[tool.ax-rules.rules] must be a table of booleans")
    return rules


def enabled_rule_ids(root: Path, rules: Mapping[str, RuleCheck] = RULES) -> tuple[str, ...]:
    """Return the explicitly configured AX rule IDs."""
    settings = _read_rule_settings(root)
    unknown = sorted(set(settings) - set(rules))
    if unknown:
        raise ConfigurationError(f"unknown AX rule IDs: {', '.join(unknown)}")
    missing = sorted(set(rules) - set(settings))
    if missing:
        raise ConfigurationError(f"missing AX rule settings: {', '.join(missing)}")
    invalid = sorted(name for name, enabled in settings.items() if not isinstance(enabled, bool))
    if invalid:
        raise ConfigurationError(f"AX rule settings must be booleans: {', '.join(invalid)}")
    return tuple(name for name in rules if settings[name])


def _source_paths(root: Path, filenames: Iterable[str]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for filename in filenames:
        path = Path(filename)
        if not path.is_absolute():
            path = root / path
        try:
            relative = path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        if path.suffix == ".py" and relative.parts and relative.parts[0] != "tools":
            paths.append(path)
    return tuple(paths)


def docstring_paths(root: Path, filenames: tuple[Path, ...]) -> tuple[Path, ...]:
    """Return changed Python files or the repository's owned docstring sources."""
    if filenames:
        return filenames
    return tuple(
        path
        for source_dir in ("albumentations", "benchmark", "tests")
        if (root / source_dir).is_dir()
        for path in (root / source_dir).rglob("*.py")
        if ".asv" not in path.relative_to(root).parts
    )


def run(root: Path, filenames: Iterable[str] = (), rules: Mapping[str, RuleCheck] = RULES) -> int:
    """Run every enabled rule and return a pre-commit compatible status."""
    source_paths = _source_paths(root, filenames)
    try:
        selected = enabled_rule_ids(root, rules)
    except ConfigurationError as exc:
        print(f"AXR configuration error: {exc}", file=sys.stderr)
        return 2
    failures = [name for name in selected if rules[name](root, source_paths)]
    return int(bool(failures))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check configured AlbumentationsX repository rules")
    parser.add_argument("filenames", nargs="*", help="files supplied by pre-commit")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    return run(root, args.filenames)
