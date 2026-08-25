"""Convert ASV comparison text into compact JSON evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROW_RE = re.compile(r"^\|\s*(?P<change>[+-])\s*\|")


def _parse_table_row(line: str) -> dict[str, Any] | None:
    if not ROW_RE.match(line):
        return None

    parts = line.split("|", 5)
    if len(parts) < 6:
        return None

    benchmark = parts[5].rsplit("|", 1)[0].strip()
    ratio_text = parts[4].strip()
    try:
        ratio = float(ratio_text)
    except ValueError:
        ratio = None

    return {
        "after": parts[3].strip(),
        "before": parts[2].strip(),
        "benchmark": benchmark,
        "change": parts[1].strip(),
        "ratio": ratio,
    }


def _status_flags(lines: list[str]) -> dict[str, bool]:
    return {
        "changed_significantly": any("SOME BENCHMARKS HAVE CHANGED SIGNIFICANTLY" in line for line in lines),
        "performance_decreased": any("PERFORMANCE DECREASED" in line for line in lines),
        "performance_improved": any("PERFORMANCE INCREASED" in line for line in lines),
        "unchanged": any("BENCHMARKS NOT SIGNIFICANTLY CHANGED" in line for line in lines),
    }


def _sorted_rows(rows: list[dict[str, Any]], *, reverse: bool, max_items: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: row["ratio"] if isinstance(row["ratio"], float) else 0.0,
        reverse=reverse,
    )[:max_items]


def summarize_asv_output(path: Path, *, asv_exit_code: int | None, max_items: int) -> dict[str, Any]:
    """Summarize ASV comparison output."""
    if not path.exists():
        return {
            "schema_version": 1,
            "kind": "asv-comparison",
            "source": str(path),
            "missing": True,
            "asv_exit_code": asv_exit_code,
            "totals": {"improvements": 0, "regressions": 0, "changed": 0},
            "status": _status_flags([]),
            "improvements": [],
            "regressions": [],
        }

    lines = path.read_text(errors="replace").splitlines()
    rows = [row for line in lines if (row := _parse_table_row(line)) is not None]
    status = _status_flags(lines)
    regressions = [row for row in rows if row["change"] == "+"]
    improvements = [row for row in rows if row["change"] == "-"]
    return {
        "schema_version": 1,
        "kind": "asv-comparison",
        "source": str(path),
        "missing": not rows and not any(status.values()),
        "asv_exit_code": asv_exit_code,
        "totals": {
            "changed": len(rows),
            "improvements": len(improvements),
            "regressions": len(regressions),
        },
        "status": status,
        "improvements": _sorted_rows(improvements, reverse=False, max_items=max_items),
        "regressions": _sorted_rows(regressions, reverse=True, max_items=max_items),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="ASV continuous/compare text output.")
    parser.add_argument("--output", required=True, type=Path, help="JSON summary file to write.")
    parser.add_argument("--asv-exit-code", type=int, help="Exit code returned by the ASV comparison command.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write a missing-input summary instead of failing when the ASV text file is absent.",
    )
    parser.add_argument("--max-items", default=50, type=int, help="Maximum improvements/regressions to keep.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists() and not args.allow_missing:
        print(f"ASV output not found: {args.input}", file=sys.stderr)
        return 1

    summary = summarize_asv_output(args.input, asv_exit_code=args.asv_exit_code, max_items=args.max_items)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote ASV summary to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
