"""Check the behavior-change inventory used by release reports."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

from packaging.version import Version

from tools.check_local_markdown_links import INLINE_LINK, collect_errors

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTES_PATH = Path("docs/release-notes/unreleased.md")
POLICY_PATH = Path("docs/maintaining/behavior-compatibility.md")
CATEGORIES = ("Validation", "Geometry", "Determinism", "Constructor defaults", "Deprecations", "Regression vectors")
FIELDS = (
    "Previous behavior",
    "New behavior",
    "Affected users",
    "Migration",
    "First affected version",
    "Contract",
    "Tests",
)
POLICY_LINK_SOURCES = (
    "docs/maintaining/support-policy.md",
    "docs/maintaining/release-process.md",
    "docs/maintaining/correctness-contracts.md",
    "docs/maintaining/correctness-report-template.md",
    "docs/maintaining/correctness-and-compatibility-report.md",
    NOTES_PATH.as_posix(),
)


def _change_fields(block: str) -> dict[str, str]:
    """Accept wrapped field values while rejecting duplicate, unknown, or missing labels."""
    title, _, body = block.partition("\n")
    fields = {"title": title.strip()}
    current_field = ""
    for line in body.splitlines():
        if line.startswith("- "):
            name, separator, value = line[2:].partition(": ")
            if not separator or name not in FIELDS or name in fields or not value.strip():
                raise ValueError(f"{title}: invalid or duplicate field {name!r}")
            fields[name] = value.strip()
            current_field = name
        elif line.startswith("  ") and current_field:
            fields[current_field] += " " + line.strip()
        elif line.strip():
            raise ValueError(f"{title}: expected a behavior-change field, got {line!r}")
    if missing := set(FIELDS) - fields.keys():
        raise ValueError(f"{title}: missing fields: {', '.join(sorted(missing))}")
    if fields["First affected version"] != "unreleased":
        raise ValueError(f"{title}: archive released entries; first affected version must be unreleased")
    return fields


def read_inventory(root: Path) -> tuple[str, dict[str, list[dict[str, str]]]]:
    """Read structured release notes, preserving explicit no-change categories."""
    notes = (root / NOTES_PATH).read_text(encoding="utf-8")
    baseline_match = re.search(r"^Release baseline: (\S+)$", notes, re.MULTILINE)
    if baseline_match is None:
        raise ValueError("Release notes must name their Release baseline")
    baseline = baseline_match[1]
    Version(baseline)
    sections = re.split(r"^## (.+)\n", notes, flags=re.MULTILINE)
    categories = sections[1::2]
    if len(categories) != len(CATEGORIES) or set(categories) != set(CATEGORIES):
        raise ValueError(f"Release notes must contain each category exactly once: {', '.join(CATEGORIES)}")
    inventory: dict[str, list[dict[str, str]]] = {}
    anchors: set[str] = set()
    for category, body in zip(categories, sections[2::2], strict=True):
        if body.strip() == "none":
            inventory[category] = []
            continue
        blocks = re.split(r"^### ", body, flags=re.MULTILINE)
        if blocks[0].strip() or len(blocks) == 1:
            raise ValueError(f"{category}: list changes or record exactly 'none'")
        changes = [_change_fields(block) for block in blocks[1:]]
        for change in changes:
            anchor = re.sub(r"[^a-z0-9 -]", "", change["title"].lower()).replace(" ", "-")
            if not anchor or anchor in anchors:
                raise ValueError(f"Duplicate or empty change anchor: {anchor!r}")
            anchors.add(anchor)
            change["anchor"] = anchor
        inventory[category] = changes
    return baseline, inventory


def _evidence_links(root: Path, change: dict[str, str]) -> None:
    """Require evidence files inside the contract and permanent-test source trees."""
    for field, directory, suffix in (("Contract", "docs", ".md"), ("Tests", "tests", ".py")):
        links = list(INLINE_LINK.finditer(change[field]))
        if not links:
            raise ValueError(f"{change['title']}: {field} must link to permanent repository evidence")
        for link in links:
            target = urlsplit(link["destination"].strip("<>"))
            if not target.path:
                raise ValueError(f"{change['title']}: {field} must name a repository file")
            path = (root / NOTES_PATH.parent / target.path).resolve()
            if target.scheme or target.netloc or not path.is_relative_to((root / directory).resolve()):
                raise ValueError(f"{change['title']}: {field} must link within {directory}/")
            if path.suffix != suffix or not path.is_file():
                raise ValueError(f"{change['title']}: missing {field} evidence: {target.path}")


def _links_to(source: Path, target: Path) -> bool:
    """Match a local documentation edge without depending on native path separators."""
    for link in INLINE_LINK.finditer(source.read_text(encoding="utf-8")):
        destination = urlsplit(link["destination"].strip("<>"))
        if (
            not destination.scheme
            and not destination.netloc
            and (source.parent / destination.path).resolve() == target.resolve()
        ):
            return True
    return False


def check_repository(root: Path = REPO_ROOT) -> tuple[str, dict[str, list[dict[str, str]]]]:
    """Reject missing migration evidence or disconnected maintenance documents."""
    baseline, inventory = read_inventory(root)
    for changes in inventory.values():
        for change in changes:
            _evidence_links(root, change)
    sources = [root / source for source in POLICY_LINK_SOURCES]
    for source in sources:
        if not _links_to(source, root / POLICY_PATH):
            raise ValueError(f"{source.relative_to(root).as_posix()} must link to {POLICY_PATH.as_posix()}")
    template = root / "docs/maintaining/correctness-report-template.md"
    template_text = template.read_text(encoding="utf-8")
    for category in CATEGORIES:
        if f"### {category}\n" not in template_text:
            raise ValueError(f"Report template must include category {category}")
    for source in (template, root / "docs/maintaining/correctness-and-compatibility-report.md"):
        if not _links_to(source, root / NOTES_PATH):
            raise ValueError(f"{source.relative_to(root).as_posix()} must link to {NOTES_PATH.as_posix()}")
    if errors := collect_errors([*sources, root / POLICY_PATH]):
        raise ValueError("\n".join(errors))
    return baseline, inventory


def format_report(root: Path, repository_url: str, version: str, *, preview: bool) -> str:
    """Render the checked inventory with evidence pinned to the tested revision."""
    baseline, inventory = check_repository(root)
    if Version(version) <= Version(baseline):
        if not preview:
            raise ValueError(f"Unreleased behavior changes require a version newer than baseline {baseline}")
        affected_version = f"unreleased after {baseline} (local preview)"
    else:
        affected_version = version

    def pin_links(text: str) -> str:
        """Keep Markdown labels and titles while binding destinations to the tested revision."""

        def replace_link(match: re.Match[str]) -> str:
            destination = match["destination"]
            target = urlsplit(destination.strip("<>"))
            path = (root / NOTES_PATH.parent / target.path).resolve().relative_to(root.resolve()).as_posix()
            fragment = f"#{target.fragment}" if target.fragment else ""
            pinned = f"{repository_url}/{path}{fragment}"
            if destination.startswith("<"):
                pinned = f"<{pinned}>"
            start = match.start("destination") - match.start()
            end = match.end("destination") - match.start()
            return f"{match[0][:start]}{pinned}{match[0][end:]}"

        return INLINE_LINK.sub(replace_link, text)

    lines = [
        f"[Behavior compatibility policy]({repository_url}/{POLICY_PATH.as_posix()}).",
        "",
        f"Release baseline: {baseline}. First affected version: {affected_version}.",
        "",
        "Each linked note records the previous behavior, new behavior, affected users, and migration.",
    ]
    for category, changes in inventory.items():
        lines.extend(("", f"### {category}", ""))
        if not changes:
            lines.append("none")
            continue
        lines.extend(("| Change and migration | Contract | Tests |", "| --- | --- | --- |"))
        for change in changes:
            note = f"[{change['title']}]({repository_url}/{NOTES_PATH.as_posix()}#{change['anchor']})"
            lines.append(f"| {note} | {pin_links(change['Contract'])} | {pin_links(change['Tests'])} |")
    return "\n".join(lines)


def main() -> int:
    """Report inventory violations as CLI failures without writing release artifacts."""
    try:
        check_repository()
    except (OSError, ValueError) as error:
        print(error)
        return 1
    print("Behavior policy, release notes, and report evidence are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
