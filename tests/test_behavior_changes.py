"""Negative controls for release migration evidence and report synchronization."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.check_behavior_changes import (
    CATEGORIES,
    NOTES_PATH,
    POLICY_LINK_SOURCES,
    POLICY_PATH,
    check_repository,
    format_report,
)
from tools.check_local_markdown_links import INLINE_LINK
from tools.generate_correctness_report import generate_report

INVENTORY = """Release baseline: 2.6.0

[Policy](../maintaining/behavior-compatibility.md)

## Validation

### Input rejection

- Previous behavior: Invalid shapes were accepted on a skip.
- New behavior: Invalid shapes raise before sampling.
- Affected users: Callers passing mismatched targets.
- Migration: Align image and mask dimensions before calling Compose.
- First affected version: unreleased
- Contract: [Pipeline validation](../validation.md).
- Tests: [Validation](../../tests/contracts/test_validation_contract.py).

## Geometry

none

## Determinism

### Replay order

- Previous behavior: Repeated visits could be lost.
- New behavior: All recorded visits replay in order.
- Affected users: Callers replaying repeated selections.
- Migration: Capture a new replay record.
- First affected version: unreleased
- Contract: [Persistence](../determinism.md#choose-the-persistence-contract).
- Tests: [Replay](../../tests/test_composition_tracing.py).

## Constructor defaults

none

## Deprecations

none

## Regression vectors

none
"""


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a minimal policy and evidence tree for destructive negative controls."""
    notes = INVENTORY
    for match in INLINE_LINK.finditer(notes):
        destination = match["destination"].split("#", maxsplit=1)[0]
        path = (tmp_path / NOTES_PATH.parent / destination).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    for source in POLICY_LINK_SOURCES:
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        policy_link = os.path.relpath(tmp_path / POLICY_PATH, path.parent)
        notes_link = os.path.relpath(tmp_path / NOTES_PATH, path.parent)
        path.write_text(f"[Policy](<{policy_link}>)\n[Notes](<{notes_link}>)\n")
    (tmp_path / NOTES_PATH).write_text(notes)
    (tmp_path / POLICY_PATH).write_text("# Behavior policy\n")
    template = tmp_path / "docs/maintaining/correctness-report-template.md"
    with template.open("a") as file:
        file.write("\n".join(f"### {category}\n\nnone\n" for category in CATEGORIES))
    monkeypatch.setattr("tools.check_local_markdown_links.REPO_ROOT", tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("- Migration:", "- Missing migration:", "invalid or duplicate field"),
        ("- Previous behavior:", "- New behavior:", "duplicate field"),
        ("## Deprecations\n\nnone", "## Deprecations\n", "list changes or record exactly 'none'"),
        ("## Deprecations", "## Geometry", "each category exactly once"),
        ("Release baseline: 2.6.0", "", "Release baseline"),
        ("First affected version: unreleased", "First affected version: 2.6.0", "archive released entries"),
        ("../../tests/contracts/test_validation_contract.py", "../validation.md", "Tests must link within tests"),
        ("../validation.md", "https://example.com/validation.md", "Contract must link within docs"),
        ("../validation.md", "#validation", "Contract must name a repository file"),
        ("../validation.md", "../../../outside.md", "Contract must link within docs"),
    ],
)
def test_inventory_rejects_incomplete_or_inconsistent_evidence(repository, old, new, message) -> None:
    notes = repository / NOTES_PATH
    notes.write_text(notes.read_text().replace(old, new, 1))

    with pytest.raises(ValueError, match=message):
        check_repository(repository)


@pytest.mark.parametrize("source", POLICY_LINK_SOURCES)
def test_maintenance_documents_cannot_disconnect_policy(repository, source) -> None:
    path = repository / source
    path.write_text(path.read_text().replace("behavior-compatibility.md", "missing-policy.md"))

    with pytest.raises(ValueError, match=r"must link to docs/maintaining/behavior-compatibility\.md"):
        check_repository(repository)


def test_report_template_cannot_drop_a_category(repository) -> None:
    template = repository / "docs/maintaining/correctness-report-template.md"
    template.write_text(template.read_text().replace("### Deprecations\n\nnone", ""))

    with pytest.raises(ValueError, match="Report template must include category Deprecations"):
        check_repository(repository)


def test_missing_permanent_tests_block_report(repository) -> None:
    (repository / "tests/contracts/test_validation_contract.py").unlink()

    with pytest.raises(ValueError, match="missing Tests evidence"):
        format_report(repository, "https://github.com/example/project/blob/tested-sha", "2.6.1", preview=False)


def test_report_resolves_first_version_and_pins_all_evidence(repository) -> None:
    revision_url = "https://github.com/example/project/blob/tested-sha"
    report = format_report(repository, revision_url, "2.6.1", preview=False)

    assert "First affected version: 2.6.1" in report
    assert f"({revision_url}/docs/release-notes/unreleased.md#replay-order)" in report
    assert f"({revision_url}/tests/test_composition_tracing.py)" in report
    assert f"({revision_url}/docs/determinism.md#choose-the-persistence-contract)" in report
    assert "### Constructor defaults\n\nnone" in report
    assert "### Deprecations\n\nnone" in report
    assert "### Regression vectors\n\nnone" in report
    assert "../" not in report


@pytest.mark.parametrize("title", [' "Validation contract"', " 'Validation contract'", '   "Validation contract"'])
@pytest.mark.parametrize("angle_brackets", [False, True])
def test_report_preserves_link_titles_while_pinning_destinations(repository, title, angle_brackets) -> None:
    """Published evidence must stay navigable with link titles and angle delimiters."""
    notes = repository / NOTES_PATH
    destination = "<../validation.md>" if angle_brackets else "../validation.md"
    notes.write_text(notes.read_text().replace("../validation.md)", f"{destination}{title})"))
    revision_url = "https://github.com/example/project/blob/tested-sha"

    report = format_report(repository, revision_url, "2.6.1", preview=False)

    pinned = f"{revision_url}/docs/validation.md"
    if angle_brackets:
        pinned = f"<{pinned}>"
    assert f"[Pipeline validation]({pinned}{title})" in report
    assert "../validation.md" not in report


def test_preview_does_not_claim_an_unreleased_version_is_published(repository) -> None:
    report = format_report(repository, "https://github.com/example/project/blob/main", "2.6.0", preview=True)

    assert "First affected version: unreleased after 2.6.0 (local preview)" in report
    with pytest.raises(ValueError, match=r"version newer than baseline 2\.6\.0"):
        format_report(repository, "https://github.com/example/project/blob/main", "2.6.0", preview=False)


def test_report_generation_checks_migration_even_in_preview(repository, monkeypatch) -> None:
    (repository / "pyproject.toml").write_text('[project]\nversion = "2.6.1"\n')
    monkeypatch.setattr("tools.generate_correctness_report.REPO_ROOT", repository)
    notes = repository / NOTES_PATH
    notes.write_text(notes.read_text().replace("## Determinism", "## Missing category"))

    with pytest.raises(ValueError, match="each category exactly once"):
        generate_report(allow_missing_evidence=True)
