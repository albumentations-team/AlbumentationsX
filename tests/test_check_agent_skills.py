"""Tests for the repository agent-skill layout check."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import check_agent_skills


def write_skill(root: Path, name: str, frontmatter: str | None = None) -> None:
    skill_dir = root / ".agents" / "skills" / name
    skill_dir.mkdir(parents=True)
    if frontmatter is None:
        frontmatter = f"name: {name}\ndescription: Does one thing. Use when asked."
    (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n\n# Body\n", encoding="utf-8")


def link_skill(root: Path, name: str, target: str | None = None) -> None:
    claude_dir = root / ".claude" / "skills"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / name).symlink_to(target or f"../../.agents/skills/{name}")


def test_linked_valid_skill_passes(tmp_path: Path) -> None:
    write_skill(tmp_path, "benchmark")
    link_skill(tmp_path, "benchmark")
    assert check_agent_skills.collect_errors(tmp_path) == []


def test_repository_skills_pass() -> None:
    assert check_agent_skills.collect_errors(Path(__file__).parents[1]) == []


@pytest.mark.parametrize(
    ("frontmatter", "message"),
    [
        ("name: other\ndescription: Text.", "must equal the directory name"),
        ("name: skill\ndescription: Reviews docstrings: examples.", "invalid YAML frontmatter"),
        ("name: skill", "description is required"),
        (f"name: skill\ndescription: {'x' * 1025}", "the limit is 1024"),
    ],
)
def test_invalid_frontmatter_is_reported(tmp_path: Path, frontmatter: str, message: str) -> None:
    write_skill(tmp_path, "skill", frontmatter)
    link_skill(tmp_path, "skill")
    errors = check_agent_skills.collect_errors(tmp_path)
    assert len(errors) == 1
    assert message in errors[0]


def test_reserved_word_in_name_is_reported(tmp_path: Path) -> None:
    write_skill(tmp_path, "claude-helper")
    link_skill(tmp_path, "claude-helper")
    assert check_agent_skills.collect_errors(tmp_path) == [
        ".agents/skills/claude-helper/SKILL.md: name must not contain claude or anthropic",
    ]


def test_missing_claude_link_is_reported(tmp_path: Path) -> None:
    write_skill(tmp_path, "benchmark")
    assert check_agent_skills.collect_errors(tmp_path) == [
        ".claude/skills/benchmark: missing; run `ln -s ../../.agents/skills/benchmark .claude/skills/benchmark`",
    ]


def test_wrong_or_orphaned_links_are_reported(tmp_path: Path) -> None:
    write_skill(tmp_path, "benchmark")
    link_skill(tmp_path, "benchmark", "/elsewhere/benchmark")
    link_skill(tmp_path, "retired")
    assert check_agent_skills.collect_errors(tmp_path) == [
        ".claude/skills/benchmark: points to '/elsewhere/benchmark'; expected '../../.agents/skills/benchmark'",
        ".claude/skills/retired: no matching .agents/skills/retired; delete the link",
    ]


def test_skill_directory_inside_claude_is_reported(tmp_path: Path) -> None:
    write_skill(tmp_path, "benchmark")
    link_skill(tmp_path, "benchmark")
    (tmp_path / ".claude" / "skills" / "local").mkdir()
    assert check_agent_skills.collect_errors(tmp_path) == [
        ".claude/skills/local: must be a symlink to ../../.agents/skills/local; keep skill files in .agents/skills",
    ]
