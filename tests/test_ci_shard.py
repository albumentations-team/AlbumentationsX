"""Tests for duration-balanced CI sharding."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.ci_shard import assign_shards, validate_manifest, weights_from_junit


def test_assign_shards_is_balanced_deterministic_and_exact_once() -> None:
    files = ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py", "tests/test_d.py"]
    weights = {
        "tests/test_a.py": 8.0,
        "tests/test_b.py": 7.0,
        "tests/test_c.py": 2.0,
        "tests/test_d.py": 1.0,
    }

    shards = assign_shards(files, weights, 2)

    assert shards == (("tests/test_a.py", "tests/test_d.py"), ("tests/test_b.py", "tests/test_c.py"))
    assert sorted(path for shard in shards for path in shard) == sorted(files)


def test_assign_shards_places_unknown_files_without_omitting_them() -> None:
    shards = assign_shards(["tests/test_a.py", "tests/test_new.py"], {"tests/test_a.py": 3.0}, 2)

    assert sorted(path for shard in shards for path in shard) == ["tests/test_a.py", "tests/test_new.py"]


def test_assign_shards_rejects_invalid_count() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        assign_shards([], {}, 0)


def test_weights_from_junit_aggregates_by_existing_module(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    test_path = repo_root / "tests" / "test_sample.py"
    test_path.parent.mkdir(parents=True)
    test_path.write_text("def test_value(): pass\n", encoding="utf-8")
    junit = tmp_path / "junit.xml"
    junit.write_text(
        """<?xml version="1.0"?>
<testsuite>
  <testcase classname="tests.test_sample.TestValue" name="test_one" time="1.25" />
  <testcase classname="tests.test_sample" name="test_two" time="0.75" />
</testsuite>
""",
        encoding="utf-8",
    )

    assert weights_from_junit(junit, repo_root) == {"tests/test_sample.py": 2.0}


def test_validate_manifest_reports_stale_test_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    tests_dir = repo_root / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_current.py").write_text("def test_value(): pass\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version": 1, "weights_seconds": {"tests/test_removed.py": 1.0}}\n',
        encoding="utf-8",
    )

    assert validate_manifest(manifest, repo_root) == [
        "Manifest references missing test module: tests/test_removed.py",
    ]
