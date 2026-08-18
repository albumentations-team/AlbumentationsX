"""Reject attempts to disable mandatory complexity diagnostics."""

from __future__ import annotations

import argparse
import re
import tokenize
from pathlib import Path

MANDATORY_COMPLEXITY_RULES = {"C901", "PLR0912"}
NOQA_PATTERN = re.compile(r"#\s*(?:ruff:\s*)?noqa(?:\s*:\s*(?P<codes>[A-Z0-9,\s]+))?", re.IGNORECASE)
PER_FILE_IGNORES_PATTERN = re.compile(
    r'^lint\.per-file-ignores\."(?P<target>[^"]+)"\s*=\s*\[(?P<codes>.*?)\]',
    re.MULTILINE | re.DOTALL,
)
PER_FILE_IGNORES_SECTION_PATTERN = re.compile(
    r"^\[tool\.ruff\.lint\.per-file-ignores\]\s*$\n(?P<rules>.*?)(?=^\[|\Z)",
    re.MULTILINE | re.DOTALL,
)
PER_FILE_IGNORE_ENTRY_PATTERN = re.compile(
    r'^"(?P<target>[^"]+)"\s*=\s*\[(?P<codes>.*?)\]',
    re.MULTILINE | re.DOTALL,
)
GLOBAL_IGNORES_PATTERN = re.compile(
    r"^\s*(?:lint\.)?(?:extend-)?ignore\s*=\s*\[(?P<codes>.*?)\]",
    re.MULTILINE | re.DOTALL,
)
COMPLEXITY_LIMITS = {"max-complexity": 10, "max-branches": 12}


def _is_forbidden_rule(code: str) -> bool:
    return code.upper() in MANDATORY_COMPLEXITY_RULES


def _find_suppressed_rules(comment: str) -> tuple[str, ...]:
    match = NOQA_PATTERN.search(comment)
    if match is None:
        return ()

    codes = match.group("codes")
    if codes is None:
        return ("all diagnostics",)
    return tuple(code for code in re.findall(r"[A-Z]+\d+", codes.upper()) if _is_forbidden_rule(code))


def _collect_python_errors(path: Path) -> list[str]:
    errors: list[str] = []
    with path.open(encoding="utf-8") as source:
        tokens = tokenize.generate_tokens(source.readline)
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            suppressed_rules = _find_suppressed_rules(token.string)
            if suppressed_rules:
                rules = ", ".join(suppressed_rules)
                errors.append(f"{path}:{token.start[0]}: do not suppress mandatory complexity checks ({rules})")
    return errors


def _find_rules(config_value: str) -> tuple[str, ...]:
    return tuple(rule for rule in re.findall(r'["\']([A-Z]+\d+)["\']', config_value) if _is_forbidden_rule(rule))


def _collect_per_file_ignore_errors(path: Path, target: str, config_value: str) -> list[str]:
    if target.startswith("tests/") or not (rules := _find_rules(config_value)):
        return []
    return [f"{path}: do not ignore mandatory complexity checks for {target} ({', '.join(rules)})"]


def _collect_toml_errors(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    errors: list[str] = []

    errors.extend(
        f"{path}: do not ignore mandatory complexity checks ({', '.join(rules)})"
        for match in GLOBAL_IGNORES_PATTERN.finditer(source)
        if (rules := _find_rules(match.group("codes")))
    )

    for match in PER_FILE_IGNORES_PATTERN.finditer(source):
        errors.extend(_collect_per_file_ignore_errors(path, match.group("target"), match.group("codes")))

    for section in PER_FILE_IGNORES_SECTION_PATTERN.finditer(source):
        for match in PER_FILE_IGNORE_ENTRY_PATTERN.finditer(section.group("rules")):
            errors.extend(_collect_per_file_ignore_errors(path, match.group("target"), match.group("codes")))

    for line_number, line in enumerate(source.splitlines(), start=1):
        for setting, maximum in COMPLEXITY_LIMITS.items():
            match = re.match(rf"\s*{setting}\s*=\s*(\d+)\b", line)
            if match and int(match.group(1)) > maximum:
                errors.append(f"{path}:{line_number}: {setting} must not exceed {maximum}")
    return errors


def collect_errors(filenames: tuple[str | Path, ...]) -> list[str]:
    """Return mandatory-complexity-check suppressions in staged source or Ruff configuration."""
    errors: list[str] = []
    for filename in filenames:
        path = Path(filename)
        if path.suffix == ".py":
            errors.extend(_collect_python_errors(path))
        elif path.suffix == ".toml":
            errors.extend(_collect_toml_errors(path))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Reject suppression of mandatory complexity diagnostics")
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()

    errors = collect_errors(tuple(args.filenames))
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
