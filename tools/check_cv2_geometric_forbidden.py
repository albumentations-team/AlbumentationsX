"""Pre-commit hook: forbid cv2.warpAffine, warpPerspective, copyMakeBorder, remap in albumentations.

These must use albucore equivalents (warp_affine, warp_perspective, copy_make_border, remap)
for multi-channel support. See docs/design/maybe_process_in_chunks_audit.md.

Allowlist:
- cv2.remap in functional.py: 2D data (mask/single channel); albucore.remap expects (H,W,C).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Forbidden in albumentations/ (use albucore instead). Keys for allowlist.
FORBIDDEN: list[tuple[str, str]] = [
    (r"cv2\.warpAffine\s*\(", "warpAffine"),
    (r"cv2\.warpPerspective\s*\(", "warpPerspective"),
    (r"cv2\.copyMakeBorder\s*\(", "copyMakeBorder"),
    (r"cv2\.remap\s*\(", "remap"),
]

# (path, forbidden_key) — allow forbidden_key in path. Key is the cv2 method name.
ALLOWLIST: list[tuple[str, str]] = [
    # 2D remap: remap_keypoints_via_mask (int16 mask), _distort_channel (single channel)
    ("albumentations/augmentations/geometric/functional.py", "remap"),
    ("albumentations/augmentations/pixel/functional.py", "remap"),
]


def path_matches_allowlist(filepath: str, forbidden_key: str) -> bool:
    """Check if (filepath, forbidden_key) is allowlisted."""
    norm = filepath.replace("\\", "/")
    for path_spec, allowed_key in ALLOWLIST:
        if norm != path_spec:
            continue
        if forbidden_key == allowed_key:
            return True
    return False


def _is_skippable_line(stripped: str) -> bool:
    """Skip docstring content (doctests, bullet points)."""
    return stripped.startswith(">>>") or bool(re.match(r"^[-*]\s", stripped))


def _scan_file(path: Path, root: Path) -> list[tuple[str, int, str]]:
    """Scan a single file for forbidden cv2 usage."""
    rel = path.relative_to(root).as_posix()
    if rel.startswith(("tools/", "docs/")):
        return []

    errors: list[tuple[str, int, str]] = []
    text = path.read_text()
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if _is_skippable_line(stripped):
            continue
        for pattern, key in FORBIDDEN:
            if re.search(pattern, line) and not path_matches_allowlist(rel, key):
                errors.append((rel, i, line.strip()))
    return errors


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    albumentations_dir = root / "albumentations"
    if not albumentations_dir.exists():
        return 0

    errors: list[tuple[str, int, str]] = []
    for path in albumentations_dir.rglob("*.py"):
        errors.extend(_scan_file(path, root))

    if errors:
        print("Forbidden cv2 usage (use albucore: warp_affine, warp_perspective, copy_make_border, remap):")
        for rel, line_no, content in errors:
            print(f"  {rel}:{line_no}: {content[:80]}{'...' if len(content) > 80 else ''}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
