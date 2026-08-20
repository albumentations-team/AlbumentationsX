"""Removal-deadline contracts for code that announces its own retirement.

Several code paths tell users they will disappear in a named release, for
example the instance-binding permissive fallback in `Compose`. Nothing compares
those announcements against the package version, so a deadline can pass without
anyone noticing and the message keeps pointing users at a release that already
shipped.

This module parses every announcement in the source tree and fails once the
package reaches the announced version.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import albumentations as A

_ROOT = Path(A.__file__).resolve().parent
_SUFFIXES = frozenset({".py"})
_SKIP_PARTS = frozenset({"__pycache__"})

_ANNOUNCEMENT_PATTERNS = (
    re.compile(r"will be removed in\s+v?(?P<version>\d+\.\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"scheduled for removal in\s+v?(?P<version>\d+\.\d+(?:\.\d+)?)", re.IGNORECASE),
)

# Deadlines that have already passed and are awaiting a maintainer decision:
# either drop the fallback or restate the deadline. Removing an entry here
# without addressing the code turns the xfail into an unexpected pass, which
# fails the suite and prevents the marker from being forgotten.
_ACCEPTED_OVERDUE = frozenset(
    {
        ("core/composition.py", "2.3"),
    },
)


def _parse_version(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in raw.split("."))


def _package_version() -> tuple[int, ...]:
    return _parse_version(A.__version__)


def _iter_announcements() -> list[tuple[str, int, str]]:
    """Return (relative path, line number, announced version) for every announcement."""
    announcements: list[tuple[str, int, str]] = []
    for path in sorted(_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in _SUFFIXES:
            continue
        if set(path.parts) & _SKIP_PARTS:
            continue
        relative = path.relative_to(_ROOT).as_posix()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for pattern in _ANNOUNCEMENT_PATTERNS:
                match = pattern.search(line)
                if match:
                    announcements.append((relative, number, match.group("version")))
                    break
    return announcements


_ANNOUNCEMENTS = _iter_announcements()


def test_announcements_are_discoverable() -> None:
    """Guard the parser itself: silent zero matches would make the contract vacuous."""
    assert _ANNOUNCEMENTS, "no removal announcements found - the parser or the patterns are stale"


@pytest.mark.parametrize(
    ("relative_path", "line_number", "announced_raw"),
    [
        pytest.param(
            relative_path,
            line_number,
            announced_raw,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    f"{relative_path} announces removal in {announced_raw}, which has shipped. "
                    "Drop the code or restate the deadline, then remove this entry from _ACCEPTED_OVERDUE."
                ),
            ),
        )
        if (relative_path, announced_raw) in _ACCEPTED_OVERDUE
        else pytest.param(relative_path, line_number, announced_raw)
        for relative_path, line_number, announced_raw in _ANNOUNCEMENTS
    ],
    ids=[f"{relative_path}:{line_number}" for relative_path, line_number, _ in _ANNOUNCEMENTS],
)
def test_removal_deadline_has_not_passed(relative_path: str, line_number: int, announced_raw: str) -> None:
    """A path that announces removal in X.Y must be gone once X.Y ships."""
    announced = _parse_version(announced_raw)
    current = _package_version()
    assert current[:2] < announced[:2], (
        f"{relative_path}:{line_number} announces removal in {announced_raw}, "
        f"but the package is at {A.__version__}. Drop the code path or restate the deadline "
        f"so the message stops pointing users at a release that already shipped."
    )
