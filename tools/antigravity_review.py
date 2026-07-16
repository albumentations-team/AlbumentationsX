"""Validate Gemini CLI output before publishing an Antigravity PR review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class ReviewError(ValueError):
    """Raised when Gemini output does not contain a publishable review."""


def _error_type(payload: dict[str, Any]) -> str | None:
    error = payload.get("error")
    if error is None:
        return None
    if not isinstance(error, dict) or not isinstance(error.get("type"), str):
        return "UNKNOWN_ERROR"
    return error["type"]


def prepare_review(gemini_output: Path, review_path: Path) -> None:
    """Validate one Gemini JSON response before creating its review artifact."""
    payload = json.loads(gemini_output.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = "Gemini CLI output must be a JSON object"
        raise ReviewError(msg)
    error_type = _error_type(payload)
    if error_type is not None:
        msg = f"Gemini CLI returned an error instead of a complete review ({error_type})"
        raise ReviewError(msg)
    response = payload.get("response")
    if not isinstance(response, str) or not response.strip():
        msg = "Gemini CLI returned no publishable review (EMPTY_RESPONSE)"
        raise ReviewError(msg)
    review = response.strip()
    if review.splitlines()[0] != "## Antigravity Review":
        msg = "Gemini CLI response is missing the required Antigravity Review heading"
        raise ReviewError(msg)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(review + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Antigravity review sanitizer arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate Gemini output and report failures without echoing its payload."""
    args = parse_args(argv)
    try:
        prepare_review(args.input, args.output)
    except (OSError, json.JSONDecodeError, ReviewError) as error:
        print(f"Antigravity review error: {error}", file=sys.stderr)
        return 1
    print(f"Prepared Antigravity review artifact: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
