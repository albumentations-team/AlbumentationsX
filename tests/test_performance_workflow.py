"""Tests for the performance evidence workflow split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

PERFORMANCE_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "performance.yml"
REPO_ROOT = PERFORMANCE_WORKFLOW.parents[2]


def _performance_jobs() -> dict[str, dict[str, Any]]:
    workflow = yaml.safe_load(PERFORMANCE_WORKFLOW.read_text())
    return workflow["jobs"]


def _job_run_text(job: dict[str, Any]) -> str:
    return "\n".join(str(step.get("run", "")) for step in job["steps"] if isinstance(step, dict))


def test_performance_workflow_keeps_pr_evidence_job_fast() -> None:
    job = _performance_jobs()["benchmark_evidence"]
    run_text = _job_run_text(job)

    assert job["timeout-minutes"] == 10
    assert "asv --config asv.conf.json check --verbose" in run_text
    assert "asv --config asv.conf.json continuous" not in run_text
    assert "asv --config asv.conf.json run" not in run_text
    assert "benchmark-performance-budget.json" in run_text


def test_performance_workflow_keeps_full_asv_timing_opt_in() -> None:
    job = _performance_jobs()["asv_comparison"]
    run_text = _job_run_text(job)

    assert "run-performance" in job["if"]
    assert "asv --config asv.conf.json continuous" in run_text
    assert "asv --config asv.conf.json run" in run_text
    assert "benchmark-asv-evidence/changed-files.txt" in run_text
    assert "tools/select_benchmark_filters.py" in run_text


def test_asv_install_commands_are_separate_cpu_torch_steps() -> None:
    for config_name in ("asv.conf.json", "asv-pytorch.conf.json"):
        config = json.loads((REPO_ROOT / "benchmark" / config_name).read_text())
        commands = config["install_command"]

        assert isinstance(commands, list)
        assert len(commands) == 2
        assert "torch>=2.13.0" in commands[0]
        assert "https://download.pytorch.org/whl/cpu" in commands[0]
        assert config["build_command"] == "python -m pip wheel --no-deps -w {build_cache_dir} {build_dir}"
        assert "{wheel_file}" in commands[1]
        assert "&&" not in " ".join(commands)
