"""Reject attempts to disable mandatory complexity diagnostics."""

from __future__ import annotations

import argparse
import re
import tokenize
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

MANDATORY_COMPLEXITY_RULES = {"C901", "PLR0912"}
NOQA_PATTERN = re.compile(r"#\s*(?:ruff:\s*)?noqa(?:\s*:\s*(?P<codes>[A-Z0-9,\s]+))?", re.IGNORECASE)
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


def _find_rules(config_value: object) -> tuple[str, ...]:
    if not isinstance(config_value, list):
        return ()
    return tuple(rule for rule in config_value if isinstance(rule, str) and _is_forbidden_rule(rule))


def _collect_per_file_ignore_errors(path: Path, target: str, config_value: object) -> list[str]:
    if target.startswith("tests/") or not (rules := _find_rules(config_value)):
        return []
    return [f"{path}: do not ignore mandatory complexity checks for {target} ({', '.join(rules)})"]


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _collect_toml_errors(path: Path) -> list[str]:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    tool_config = _as_mapping(config.get("tool"))
    ruff_config = _as_mapping(tool_config.get("ruff"))
    lint_config = _as_mapping(ruff_config.get("lint"))
    errors: list[str] = []

    errors.extend(
        f"{path}: do not ignore mandatory complexity checks ({', '.join(rules)})"
        for setting in ("ignore", "extend-ignore")
        if (rules := _find_rules(lint_config.get(setting)))
    )

    per_file_ignores = _as_mapping(lint_config.get("per-file-ignores"))
    for target, rules in per_file_ignores.items():
        errors.extend(_collect_per_file_ignore_errors(path, target, rules))

    for section, setting in (("mccabe", "max-complexity"), ("pylint", "max-branches")):
        limit = _as_mapping(lint_config.get(section)).get(setting)
        if isinstance(limit, int) and limit > COMPLEXITY_LIMITS[setting]:
            errors.append(f"{path}: {setting} must not exceed {COMPLEXITY_LIMITS[setting]}")
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
