"""Tests for compact pytest JUnit summaries."""

from __future__ import annotations

import json
from pathlib import Path

from tools.pytest_summary import main, summarize_junit


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_summarize_junit_totals_from_testsuites(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(
        """
<testsuites>
  <testsuite name="unit" tests="3" failures="1" errors="0" skipped="1" time="1.25" />
  <testsuite name="property" tests="2" failures="0" errors="1" skipped="0" time="0.50" />
</testsuites>
""".strip(),
    )

    summary = summarize_junit(junit)

    assert summary["kind"] == "pytest-summary"
    assert summary["status"] == "ok"
    assert summary["missing"] is False
    assert summary["totals"] == {
        "tests": 5,
        "failures": 1,
        "errors": 1,
        "skipped": 1,
        "time": 1.75,
        "passed": 2,
    }


def test_pytest_summary_cli_fails_for_missing_junit_by_default(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"

    exit_code = main(["--junit", str(tmp_path / "missing.xml"), "--output", str(output)])

    assert exit_code == 1
    assert not output.exists()


def test_pytest_summary_cli_writes_missing_summary_when_allowed(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"

    exit_code = main(
        [
            "--junit",
            str(tmp_path / "missing.xml"),
            "--output",
            str(output),
            "--allow-incomplete",
        ],
    )

    assert exit_code == 0
    summary = _read_json(output)
    assert summary["status"] == "missing"
    assert summary["missing"] is True
    assert summary["totals"]["errors"] == 1


def test_pytest_summary_cli_writes_invalid_summary_when_allowed(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    output = tmp_path / "summary.json"
    junit.write_text("<testsuites><testsuite")

    exit_code = main(["--junit", str(junit), "--output", str(output), "--allow-incomplete"])

    assert exit_code == 0
    summary = _read_json(output)
    assert summary["status"] == "invalid"
    assert summary["missing"] is False
    assert summary["totals"]["errors"] == 1
