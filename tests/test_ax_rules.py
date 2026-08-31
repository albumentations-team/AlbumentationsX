"""Tests for the configurable AlbumentationsX rule dispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.ax_rules import runner


def write_rule_config(root: Path, rules: str) -> None:
    (root / "pyproject.toml").write_text(f"[tool.ax-rules.rules]\n{rules}", encoding="utf-8")


def passing_rule(_: Path, __: tuple[Path, ...]) -> int:
    return 0


def test_enabled_rule_ids_respect_pyproject(tmp_path: Path) -> None:
    write_rule_config(tmp_path, "first = false\nsecond = true\n")
    rules = {
        "first": passing_rule,
        "second": passing_rule,
    }
    assert runner.enabled_rule_ids(tmp_path, rules) == ("second",)


def test_enabled_rule_ids_reject_unknown_or_non_boolean_settings(tmp_path: Path) -> None:
    rules = {"known": passing_rule}
    write_rule_config(tmp_path, "unknown = true\n")
    with pytest.raises(runner.ConfigurationError, match="unknown AX rule IDs: unknown"):
        runner.enabled_rule_ids(tmp_path, rules)
    write_rule_config(tmp_path, 'known = "yes"\n')
    with pytest.raises(runner.ConfigurationError, match="AX rule settings must be booleans: known"):
        runner.enabled_rule_ids(tmp_path, rules)


def test_enabled_rule_ids_require_every_rule_to_be_configured(tmp_path: Path) -> None:
    write_rule_config(tmp_path, "first = true\n")
    rules = {
        "first": passing_rule,
        "second": passing_rule,
    }
    with pytest.raises(runner.ConfigurationError, match="missing AX rule settings: second"):
        runner.enabled_rule_ids(tmp_path, rules)


def test_repository_config_declares_every_ax_rule() -> None:
    root = Path(__file__).parents[1]
    assert runner.enabled_rule_ids(root) == tuple(runner.RULES)


def test_run_invokes_only_enabled_rules_with_pre_commit_python_files(tmp_path: Path) -> None:
    calls: list[tuple[str, tuple[Path, ...]]] = []

    def check(name: str):
        def run(_: Path, paths: tuple[Path, ...]) -> int:
            calls.append((name, paths))
            return 0

        return run

    write_rule_config(tmp_path, "first = true\nsecond = false\n")
    source = tmp_path / "albumentations" / "example.py"
    source.parent.mkdir()
    source.write_text("", encoding="utf-8")
    rules = {
        "first": check("first"),
        "second": check("second"),
    }
    assert runner.run(tmp_path, ("albumentations/example.py", "tools/helper.py"), rules) == 0
    assert calls == [("first", (source,))]


def test_docstring_paths_exclude_asv_environments_from_a_full_run(tmp_path: Path) -> None:
    source = tmp_path / "albumentations" / "example.py"
    asv_environment = tmp_path / "benchmark" / ".asv" / "env" / "dependency.py"
    source.parent.mkdir()
    asv_environment.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    asv_environment.write_text("", encoding="utf-8")
    assert runner.docstring_paths(tmp_path, ()) == (source,)


def test_pre_commit_uses_the_single_ax_rule_hook() -> None:
    config = (Path(__file__).parents[1] / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert config.count("id: check-ax-rules") == 1
    for retired_hook in (
        "check-ax-coding-guidance",
        "check-docstrings",
        "check-naming-conflicts",
        "check-public-transform-docstrings",
        "check-readme-transforms-docs",
    ):
        assert f"id: {retired_hook}" not in config
