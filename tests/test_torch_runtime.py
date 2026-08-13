"""Unit tests for the CPU-only Torch runtime contract tool."""

from __future__ import annotations

from types import SimpleNamespace

from tools import torch_runtime


def test_runtime_errors_rejects_cuda_and_nvidia_distributions(monkeypatch) -> None:
    monkeypatch.setattr(torch_runtime.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(torch_runtime, "_load_torch", lambda: SimpleNamespace(version=SimpleNamespace(cuda="12.8")))
    monkeypatch.setattr(torch_runtime, "installed_accelerator_distributions", lambda: ("nvidia-cublas-cu12",))

    errors = torch_runtime.runtime_errors("cpu")

    assert "Torch reports CUDA '12.8'; CI requires the CPU-only build." in errors
    assert "CPU-only Torch runtime contains accelerator distributions: nvidia-cublas-cu12" in errors


def test_runtime_errors_accepts_absent_torch_without_accelerator_distributions(monkeypatch) -> None:
    monkeypatch.setattr(torch_runtime.importlib.util, "find_spec", lambda _: None)
    monkeypatch.setattr(torch_runtime, "installed_accelerator_distributions", lambda: ())

    assert torch_runtime.runtime_errors("absent") == []


def test_runtime_errors_rejects_torch_in_an_absent_runtime(monkeypatch) -> None:
    monkeypatch.setattr(torch_runtime.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(torch_runtime, "installed_accelerator_distributions", lambda: ())

    assert torch_runtime.runtime_errors("absent") == ["Torch is installed but the runtime must be Torch-free."]


def test_install_cpu_torch_uses_the_shared_cpu_index(monkeypatch, tmp_path) -> None:
    commands: list[list[str]] = []
    checked: list[tuple[object, str]] = []
    interpreter = tmp_path / "python"

    monkeypatch.setattr(torch_runtime, "_run", commands.append)
    monkeypatch.setattr(
        torch_runtime, "_check_interpreter_runtime", lambda path, expected: checked.append((path, expected))
    )

    torch_runtime.install_cpu_torch(interpreter)

    assert commands == [
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(interpreter),
            "torch>=2.13.0",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
    ]
    assert checked == [(interpreter, "cpu")]
