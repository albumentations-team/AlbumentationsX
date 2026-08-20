"""Tests for CI environment evidence collection."""

from __future__ import annotations

from tools import collect_test_environment


def test_torch_runtime_records_torch_free_environment(monkeypatch) -> None:
    monkeypatch.setattr(collect_test_environment.importlib.metadata, "distributions", lambda: ())
    monkeypatch.setattr(collect_test_environment.importlib.util, "find_spec", lambda _: None)

    assert collect_test_environment._torch_runtime() == {
        "accelerator_distributions": [],
        "cuda_version": None,
        "installed": False,
    }


def test_collect_environment_records_selected_ci_profile(monkeypatch) -> None:
    monkeypatch.setenv("ALBU_CI_DEPENDENCY_GROUP", "ci-test")
    monkeypatch.setenv("ALBU_CI_RUNTIME_PROFILE", "torch-cpu")

    environment = collect_test_environment.collect_environment("pytest")

    assert environment["command"] == "pytest"
    assert environment["ci_environment"] == {
        "dependency_group": "ci-test",
        "runtime_profile": "torch-cpu",
    }
