"""Tests for the performance evidence workflow split."""

from __future__ import annotations

import json
import re
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
    assert not re.search(r"asv --config asv\.conf\.json\s+run\b", run_text)
    assert "benchmark-performance-budget.json" in run_text


def test_performance_workflow_keeps_targeted_comparison_label_or_manual_gated() -> None:
    job = _performance_jobs()["asv_comparison"]
    run_text = _job_run_text(job)

    assert "run-performance" in job["if"]
    assert "asv --config asv.conf.json continuous" in run_text
    assert "targeted-performance-evidence/changed-files.txt" in run_text
    assert "--profile changed" in run_text
    assert "--profile stf-core" in run_text
    assert not re.search(r"asv --config asv\.conf\.json\s+run\b", run_text)


def test_performance_workflow_adds_bounded_pr_and_scheduled_comparisons() -> None:
    jobs = _performance_jobs()
    pr_job = jobs["pr_core_comparison"]
    scheduled_job = jobs["scheduled_core_comparison"]

    assert pr_job["needs"] == "benchmark_evidence"
    assert pr_job["timeout-minutes"] == 10
    assert "--profile pr-core" in _job_run_text(pr_job)
    assert "asv --config asv.conf.json continuous" in _job_run_text(pr_job)
    assert scheduled_job["if"] == "github.event_name == 'schedule'"
    assert scheduled_job["timeout-minutes"] == 20
    assert "--profile stf-core" in _job_run_text(scheduled_job)
    assert "git describe --tags --abbrev=0 --match '[0-9]*' HEAD^" in _job_run_text(scheduled_job)
    assert not re.search(r"HEAD\^!", PERFORMANCE_WORKFLOW.read_text())


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


def test_every_asv_job_explicitly_uses_the_cpu_torch_runtime() -> None:
    workflow = yaml.safe_load(PERFORMANCE_WORKFLOW.read_text())
    for job_name in ("benchmark_evidence", "pr_core_comparison", "asv_comparison", "scheduled_core_comparison"):
        setup_step = next(
            step for step in workflow["jobs"][job_name]["steps"] if step.get("uses") == "./.github/actions/setup-ci"
        )

        assert setup_step["with"] == {
            "python-version": "3.12",
            "dependency-group": "ci-benchmark",
            "runtime-profile": "torch-cpu",
        }
