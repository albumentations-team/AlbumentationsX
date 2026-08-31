import re
import sys
from collections.abc import Iterable
from pathlib import Path

DOCSTRING_PATTERN = re.compile(r'["\']{3}[\s\S]+?["\']{3}')
DASH_PATTERN = re.compile(r"---{2,}")
# Match exactly two backticks, not part of triple (```) - allows fenced code blocks
DOUBLE_BACKTICK_PATTERN = re.compile(r"(?<!`)``(?!`)")


def check_docstrings_for_dashes(file_path: str | Path) -> bool:
    with Path(file_path).open(encoding="utf-8") as file:
        content = file.read()
    return all(not DASH_PATTERN.search(match.group()) for match in DOCSTRING_PATTERN.finditer(content))


def check_docstrings_for_double_backticks(file_path: str | Path) -> bool:
    with Path(file_path).open(encoding="utf-8") as file:
        content = file.read()
    return all(not DOUBLE_BACKTICK_PATTERN.search(match.group()) for match in DOCSTRING_PATTERN.finditer(content))


def collect_errors(file_paths: Iterable[str | Path], *, root: Path | None = None) -> list[str]:
    """Return format errors for Python docstrings outside local tooling."""
    errors: list[str] = []
    for file_path in file_paths:
        path = Path(file_path)
        try:
            relative = path.resolve().relative_to(root.resolve()).as_posix() if root else str(path)
        except ValueError:
            continue
        if root and (not relative.endswith(".py") or relative.startswith("tools/")):
            continue
        if not check_docstrings_for_dashes(file_path):
            errors.append(
                f"Error in {relative}: According to Google Style docstrings, '---' should not be used "
                "to underline sections. Please refer to "
                "https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings"
            )
        if not check_docstrings_for_double_backticks(file_path):
            errors.append(
                f"Error in {relative}: Double backticks (``) in docstrings get incorrectly re-rendered "
                "on the website. Use single backticks (`) or other formatting instead."
            )
    return errors


def main() -> None:
    errors = collect_errors(sys.argv[1:])
    for error in errors:
        print(error)
    sys.exit(bool(errors))


if __name__ == "__main__":
    main()
