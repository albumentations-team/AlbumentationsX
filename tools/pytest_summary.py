"""Convert pytest JUnit XML output into a compact JSON summary."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _as_int(value: str | None) -> int:
    return int(float(value or 0))


def _as_float(value: str | None) -> float:
    return float(value or 0)


def _suite_summaries(root: ET.Element) -> list[dict[str, Any]]:
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))

    return [
        {
            "name": suite.attrib.get("name", ""),
            "tests": _as_int(suite.attrib.get("tests")),
            "failures": _as_int(suite.attrib.get("failures")),
            "errors": _as_int(suite.attrib.get("errors")),
            "skipped": _as_int(suite.attrib.get("skipped")),
            "time": _as_float(suite.attrib.get("time")),
        }
        for suite in suites
    ]


def summarize_junit(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "source": str(path),
            "missing": True,
            "totals": {
                "tests": 0,
                "failures": 0,
                "errors": 1,
                "skipped": 0,
                "time": 0.0,
                "passed": 0,
            },
            "suites": [],
        }

    root = ET.parse(path).getroot()  # noqa: S314
    suites = _suite_summaries(root)
    totals = {
        "tests": sum(suite["tests"] for suite in suites),
        "failures": sum(suite["failures"] for suite in suites),
        "errors": sum(suite["errors"] for suite in suites),
        "skipped": sum(suite["skipped"] for suite in suites),
        "time": round(sum(suite["time"] for suite in suites), 3),
    }
    totals["passed"] = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    return {"schema_version": 1, "source": str(path), "totals": totals, "suites": suites}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", required=True, type=Path, help="JUnit XML file produced by pytest.")
    parser.add_argument("--output", required=True, type=Path, help="JSON summary file to write.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summarize_junit(args.junit), indent=2, sort_keys=True) + "\n")
    print(f"Wrote pytest summary to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
