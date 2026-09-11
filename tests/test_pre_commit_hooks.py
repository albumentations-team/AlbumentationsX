"""Tests for repository-wide non-guidance pre-commit checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import check_internal_workspace, check_local_markdown_links


def test_internal_workspace_rejects_everything_except_gitkeep() -> None:
    errors = check_internal_workspace.collect_errors(
        ("_internal/.gitkeep", "_internal/scratch/notes.md", "docs/guide.md"),
    )
    assert errors == ["_internal/scratch/notes.md: _internal/ is local-only and must not be committed"]


def test_local_markdown_links_ignore_external_urls_and_fenced_examples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(check_local_markdown_links, "REPO_ROOT", tmp_path)
    source = tmp_path / "docs/guide.md"
    target = tmp_path / "docs/target.md"
    source.parent.mkdir(parents=True)
    target.write_text("target", encoding="utf-8")
    source.write_text(
        "[Local](target.md#section)\n"
        "[Root-relative](/docs/target.md#section)\n"
        "[External](https://albumentations.ai/docs/)\n"
        "```markdown\n"
        "[Example](not-a-real-file.md)\n"
        "```\n",
        encoding="utf-8",
    )
    assert check_local_markdown_links.collect_errors((source,)) == []


def test_local_markdown_links_report_missing_and_escaping_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check_local_markdown_links, "REPO_ROOT", tmp_path)
    source = tmp_path / "docs/guide.md"
    source.parent.mkdir(parents=True)
    source.write_text("[Missing](missing.md)\n[Outside](../../outside.md)\n", encoding="utf-8")
    assert check_local_markdown_links.collect_errors((source,)) == [
        "docs/guide.md:1: local link target does not exist: missing.md",
        "docs/guide.md:2: local link escapes the repository: ../../outside.md",
    ]
