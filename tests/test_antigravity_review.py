"""Tests for sanitizing Gemini output before publishing PR reviews."""

import json
from pathlib import Path

import pytest

from tools.antigravity_review import ReviewError, main, prepare_review


def test_prepare_review_rejects_empty_response_without_publishing_diagnostics(tmp_path: Path) -> None:
    gemini_output = tmp_path / "stdout.log"
    review_path = tmp_path / "review.md"
    gemini_output.write_text(
        json.dumps(
            {
                "response": "",
                "stats": {"models": {"gemini": {"tokens": {"total": 3869}}}},
                "error": {
                    "type": "INVALID_STREAM",
                    "message": "The model returned an empty response or malformed tool call.",
                },
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReviewError, match="INVALID_STREAM"):
        prepare_review(gemini_output, review_path)

    assert not review_path.exists()


def test_prepare_review_publishes_only_the_validated_response(tmp_path: Path) -> None:
    gemini_output = tmp_path / "stdout.log"
    review_path = tmp_path / "review.md"
    response = "## Antigravity Review\n\nNo actionable findings."
    gemini_output.write_text(
        json.dumps(
            {
                "response": response,
                "stats": {"models": {"gemini": {"tokens": {"total": 137}}}},
            },
        ),
        encoding="utf-8",
    )

    prepare_review(gemini_output, review_path)

    assert review_path.read_text(encoding="utf-8") == response + "\n"


def test_prepare_review_rejects_non_review_text(tmp_path: Path) -> None:
    gemini_output = tmp_path / "stdout.log"
    review_path = tmp_path / "review.md"
    gemini_output.write_text(json.dumps({"response": '{"error": "INVALID_STREAM"}'}), encoding="utf-8")

    with pytest.raises(ReviewError, match="heading"):
        prepare_review(gemini_output, review_path)

    assert not review_path.exists()


def test_main_reports_invalid_stream_without_echoing_diagnostics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gemini_output = tmp_path / "stdout.log"
    review_path = tmp_path / "review.md"
    gemini_output.write_text(
        json.dumps(
            {
                "response": "",
                "stats": {"private_diagnostics": "must not be echoed"},
                "error": {"type": "INVALID_STREAM"},
            },
        ),
        encoding="utf-8",
    )

    exit_code = main(["--input", str(gemini_output), "--output", str(review_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "INVALID_STREAM" in captured.err
    assert "private_diagnostics" not in captured.err


def test_prepare_review_rejects_partial_response_with_error(tmp_path: Path) -> None:
    gemini_output = tmp_path / "stdout.log"
    review_path = tmp_path / "review.md"
    gemini_output.write_text(
        json.dumps(
            {
                "response": "## Antigravity Review\n\nPartial output.",
                "error": {"type": "INVALID_STREAM"},
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReviewError, match="INVALID_STREAM"):
        prepare_review(gemini_output, review_path)

    assert not review_path.exists()
