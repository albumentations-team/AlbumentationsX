"""Check that repository agent skills load in both Codex and Claude Code.

Skills live in ``.agents/skills/<name>/`` (read by Codex). Claude Code reads ``.claude/skills/<name>``, which must be a
relative symlink to the matching ``.agents/skills`` directory.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

SKILLS_DIR = Path(".agents/skills")
CLAUDE_SKILLS_DIR = Path(".claude/skills")
NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024  # Codex limit; Claude Code truncates description + when_to_use at 1536.
RESERVED_NAME_WORDS = ("claude", "anthropic")


def _frontmatter_errors(skill_dir: Path, root: Path) -> list[str]:
    skill_file = skill_dir / "SKILL.md"
    label = skill_file.relative_to(root).as_posix()
    if not skill_file.is_file():
        return [f"{label}: missing SKILL.md"]
    parts = skill_file.read_text(encoding="utf-8").split("---", maxsplit=2)
    if len(parts) < 3 or parts[0].strip():
        return [f"{label}: SKILL.md must start with YAML frontmatter between '---' lines"]
    try:
        frontmatter = yaml.safe_load(parts[1])
    except yaml.YAMLError as exc:
        return [f"{label}: invalid YAML frontmatter ({exc.__class__.__name__}); quote values that contain ': '"]
    if not isinstance(frontmatter, dict):
        return [f"{label}: frontmatter must be a mapping"]

    errors = []
    name = frontmatter.get("name")
    if name != skill_dir.name:
        errors.append(f"{label}: name {name!r} must equal the directory name {skill_dir.name!r}")
    elif len(name) > MAX_NAME_LENGTH or not NAME_PATTERN.fullmatch(name):
        errors.append(f"{label}: name must be lowercase-hyphen and at most {MAX_NAME_LENGTH} characters")
    elif any(word in name for word in RESERVED_NAME_WORDS):
        errors.append(f"{label}: name must not contain {' or '.join(RESERVED_NAME_WORDS)}")
    description = frontmatter.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append(f"{label}: description is required")
    elif len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(f"{label}: description has {len(description)} characters; the limit is {MAX_DESCRIPTION_LENGTH}")
    return errors


def _link_errors(root: Path, skill_names: set[str]) -> list[str]:
    claude_dir = root / CLAUDE_SKILLS_DIR
    entries = {entry.name: entry for entry in claude_dir.iterdir()} if claude_dir.is_dir() else {}
    errors = [
        f"{CLAUDE_SKILLS_DIR / name}: missing; run `ln -s ../../{SKILLS_DIR / name} {CLAUDE_SKILLS_DIR / name}`"
        for name in sorted(skill_names - set(entries))
    ]
    for name, entry in sorted(entries.items()):
        label = (CLAUDE_SKILLS_DIR / name).as_posix()
        expected = f"../../{(SKILLS_DIR / name).as_posix()}"
        if not entry.is_symlink():
            errors.append(f"{label}: must be a symlink to {expected}; keep skill files in {SKILLS_DIR}")
        elif name not in skill_names:
            errors.append(f"{label}: no matching {SKILLS_DIR / name}; delete the link")
        elif str(entry.readlink()) != expected:
            errors.append(f"{label}: points to {str(entry.readlink())!r}; expected {expected!r}")
    return errors


def collect_errors(root: Path) -> list[str]:
    """Return problems that would hide or break a repository skill in Codex or Claude Code."""
    skills_dir = root / SKILLS_DIR
    skill_dirs = sorted(path for path in skills_dir.iterdir() if path.is_dir()) if skills_dir.is_dir() else []
    errors = [error for skill_dir in skill_dirs for error in _frontmatter_errors(skill_dir, root)]
    errors.extend(_link_errors(root, {path.name for path in skill_dirs}))
    return errors
