"""Tests for canonical local and CI quality-gate groups."""

from __future__ import annotations

from pathlib import Path

from tools.quality_gate import CHECK_GROUPS, _read_paths, markdown_checks, scoped_contract_checks


def test_fast_group_contains_each_parallel_ci_group() -> None:
    fast = CHECK_GROUPS["fast"]

    for group in ("lint", "mypy", "pyrefly", "contracts", "smoke"):
        assert all(command in fast for command in CHECK_GROUPS[group])


def test_contracts_skip_checks_owned_by_parallel_jobs() -> None:
    pre_commit = next(command for command in CHECK_GROUPS["contracts"] if command.args[:2] == ("pre-commit", "run"))

    assert dict(pre_commit.environment)["SKIP"] == "mypy,pyrefly-check,ruff,ruff-format"


def test_contracts_run_applied_configuration_suite() -> None:
    commands = {command.args for command in CHECK_GROUPS["contracts"]}

    assert ("pytest", "-q", "tests/contracts") in commands


def test_read_paths_keeps_existing_markdown_without_directory_escape(tmp_path: Path) -> None:
    markdown = Path("docs/maintaining/ci-policy.md")
    paths_file = tmp_path / "paths"
    paths_file.write_bytes(f"{markdown}\0../outside.md\0tools/ci_plan.py\0missing.md\0".encode())

    assert _read_paths(paths_file, null_delimited=True, markdown_only=True) == (markdown.as_posix(),)


def test_markdown_command_does_not_use_a_shell() -> None:
    commands = markdown_checks(("README.md", "docs/maintaining/ci-policy.md"))

    assert commands[0].args == (
        "pre-commit",
        "run",
        "--files",
        "README.md",
        "docs/maintaining/ci-policy.md",
    )


def test_scoped_contracts_use_changed_files_without_dropping_global_checks() -> None:
    commands = scoped_contract_checks(("tools/ci_plan.py", ".github/workflows/pr.yml"))
    pre_commit = next(command for command in commands if command.args[:2] == ("pre-commit", "run"))

    assert pre_commit.args == (
        "pre-commit",
        "run",
        "--files",
        "tools/ci_plan.py",
        ".github/workflows/pr.yml",
    )
    assert ("python", "-m", "tools.ci_matrix", "check") in {command.args for command in commands}
