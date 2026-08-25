"""Tests for ASV comparison summary parsing."""

from __future__ import annotations

from textwrap import dedent

from tools.asv_summary import summarize_asv_output


def test_summarize_asv_output_parses_changed_rows_with_parameter_pipes(tmp_path):
    asv_output = tmp_path / "asv-continuous.txt"
    asv_output.write_text(
        dedent(
            """\
            | Change   | Before [abc] | After [def] |   Ratio | Benchmark (Parameter) |
            | -        | 68.1+/-0us   | 61.6+/-0us  |    0.9  | mod.Class.time('pad|small|float32') |
            | +        | 8.42+/-0us   | 14.3+/-0us  |    1.7  | mod.Class.time('cutout|small|float32') |
            SOME BENCHMARKS HAVE CHANGED SIGNIFICANTLY.
            PERFORMANCE DECREASED.
            """,
        ),
    )

    summary = summarize_asv_output(asv_output, asv_exit_code=1, max_items=10)

    assert summary["missing"] is False
    assert summary["asv_exit_code"] == 1
    assert summary["totals"] == {"changed": 2, "improvements": 1, "regressions": 1}
    assert summary["status"]["changed_significantly"] is True
    assert summary["status"]["performance_decreased"] is True
    assert summary["improvements"][0]["benchmark"] == "mod.Class.time('pad|small|float32')"
    assert summary["regressions"][0]["benchmark"] == "mod.Class.time('cutout|small|float32')"
    assert summary["regressions"][0]["ratio"] == 1.7


def test_summarize_asv_output_handles_missing_input(tmp_path):
    summary = summarize_asv_output(tmp_path / "missing.txt", asv_exit_code=None, max_items=10)

    assert summary["missing"] is True
    assert summary["totals"] == {"changed": 0, "improvements": 0, "regressions": 0}
    assert summary["regressions"] == []


def test_summarize_asv_output_marks_unrecognized_failure_as_missing(tmp_path):
    asv_output = tmp_path / "failed.txt"
    asv_output.write_text("ASV failed before producing a comparison table.\n")

    summary = summarize_asv_output(asv_output, asv_exit_code=1, max_items=10)

    assert summary["missing"] is True
    assert summary["asv_exit_code"] == 1
