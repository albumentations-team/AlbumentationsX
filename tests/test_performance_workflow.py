"""Contracts for release and explicit performance evidence workflows."""

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


def test_performance_workflow_has_no_routine_pr_asv_or_false_green_paths() -> None:
    text = PERFORMANCE_WORKFLOW.read_text()
    jobs = _performance_jobs()

    assert not {"benchmark_evidence", "pr_core_comparison"} & set(jobs)
    for retired in ("pr-core", "continue-on-error: true", "--allow-missing", "asv --config asv.conf.json check"):
        assert retired not in text


def test_targeted_comparison_is_explicitly_requested_and_can_reproduce_changed_families() -> None:
    job = _performance_jobs()["asv_comparison"]
    run_text = _job_run_text(job)

    assert "run-performance" in job["if"]
    assert "asv --config asv.conf.json continuous" in run_text
    assert "targeted-performance-evidence/changed-files.txt" in run_text
    assert "--profile changed" in run_text
    assert "--profile release-core" in run_text
    assert 'CANDIDATE_REF="${INPUT_CANDIDATE_REF:-HEAD}"' in run_text
    assert "git describe --tags --abbrev=0 --match '[0-9]*' \"$CANDIDATE_REF^\"" in run_text
    assert 'exit "$ASV_EXIT_CODE"' in run_text
    assert not re.search(r"asv --config asv\.conf\.json\s+run\b", run_text)


def test_scheduled_release_comparison_is_bounded_and_strict() -> None:
    job = _performance_jobs()["scheduled_core_comparison"]
    run_text = _job_run_text(job)

    assert job["if"] == "github.event_name == 'schedule'"
    assert job["timeout-minutes"] == 20
    assert "--profile release-core" in run_text
    assert "git describe --tags --abbrev=0 --match '[0-9]*' HEAD^" in run_text
    assert "--require-comparison" in run_text
    assert "--fail-on-release-blockers" in run_text
    assert not re.search(r"HEAD\^!", PERFORMANCE_WORKFLOW.read_text())


def test_pytorch_performance_is_manual_only() -> None:
    workflow_path = REPO_ROOT / ".github" / "workflows" / "pytorch-performance.yml"
    workflow = yaml.safe_load(workflow_path.read_text())
    trigger = workflow.get("on", workflow.get(True, {}))
    text = workflow_path.read_text()

    assert set(trigger) == {"workflow_dispatch"}
    assert 'CANDIDATE_REF="${INPUT_CANDIDATE_REF:-HEAD}"' in text
    assert "git describe --tags --abbrev=0 --match '[0-9]*' \"$CANDIDATE_REF^\"" in text
    assert "continue-on-error" not in text
    assert "--allow-missing" not in text
    assert 'exit "$ASV_EXIT_CODE"' in text


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
    for job_name in ("asv_comparison", "scheduled_core_comparison"):
        setup_step = next(
            step for step in workflow["jobs"][job_name]["steps"] if step.get("uses") == "./.github/actions/setup-ci"
        )

        assert setup_step["with"] == {
            "python-version": "3.12",
            "dependency-group": "ci-benchmark",
            "runtime-profile": "torch-cpu",
        }
