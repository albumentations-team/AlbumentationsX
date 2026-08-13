"""Install and verify the CPU-only Torch runtime used by AlbumentationsX CI."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Literal

TORCH_REQUIREMENT = "torch>=2.13.0"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
DEFAULT_OPENCV_REQUIREMENT = "opencv-python-headless>=5.0.0.93"
MISSING_TORCH_GUIDANCE = 'pip install "albumentationsx[headless]"'


def _normalize_distribution_name(name: str) -> str:
    return name.casefold().replace("_", "-")


def installed_accelerator_distributions() -> tuple[str, ...]:
    """Return installed CUDA and NVIDIA distribution names in normalized order."""
    names = {
        _normalize_distribution_name(distribution.metadata["Name"])
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    return tuple(sorted(name for name in names if name.startswith(("cuda", "nvidia-"))))


def _load_torch() -> ModuleType:
    import torch

    return torch


def runtime_errors(expected: Literal["absent", "cpu"]) -> list[str]:
    """Return contract violations for the selected Torch runtime state."""
    errors: list[str] = []
    torch_present = importlib.util.find_spec("torch") is not None

    if expected == "absent":
        if torch_present:
            errors.append("Torch is installed but the runtime must be Torch-free.")
    elif not torch_present:
        errors.append("Torch is not installed but the CPU runtime is required.")
    else:
        try:
            torch = _load_torch()
        except Exception as error:  # noqa: BLE001 - report the loader failure as runtime evidence.
            errors.append(f"Torch could not be imported: {error}")
        else:
            if torch.version.cuda is not None:
                errors.append(f"Torch reports CUDA {torch.version.cuda!r}; CI requires the CPU-only build.")

    accelerator_distributions = installed_accelerator_distributions()
    if accelerator_distributions:
        errors.append(
            "CPU-only Torch runtime contains accelerator distributions: " + ", ".join(accelerator_distributions),
        )

    return errors


def check_runtime(expected: Literal["absent", "cpu"]) -> None:
    """Raise an error when the current interpreter violates the Torch runtime contract."""
    errors = runtime_errors(expected)
    if errors:
        raise RuntimeError("\n".join(errors))


def _run(command: list[str]) -> None:
    print("+", shlex.join(command))
    subprocess.run(command, check=True)  # noqa: S603 - commands are constructed from fixed CLI arguments.


def _interpreter_path(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _check_interpreter_runtime(interpreter: Path, expected: Literal["absent", "cpu"]) -> None:
    _run([str(interpreter), str(Path(__file__).resolve()), "check", "--expect", expected])


def install_cpu_torch(interpreter: Path) -> None:
    """Install and verify the shared CPU-only Torch profile in one interpreter."""
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(interpreter),
            TORCH_REQUIREMENT,
            "--index-url",
            PYTORCH_CPU_INDEX,
        ],
    )
    _check_interpreter_runtime(interpreter, "cpu")


def _assert_missing_torch_import(interpreter: Path) -> None:
    result = subprocess.run(  # noqa: S603 - interpreter and command are constructed by this tool.
        [str(interpreter), "-I", "-c", "import albumentations"],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode == 0:
        msg = "AlbumentationsX imported without Torch."
        raise RuntimeError(msg)
    if MISSING_TORCH_GUIDANCE not in result.stderr:
        msg = "AlbumentationsX did not report the missing-Torch installation guidance."
        raise RuntimeError(msg)


def _run_import_smoke(interpreter: Path) -> None:
    _run(
        [
            str(interpreter),
            "-I",
            "-c",
            (
                "import albumentations as A; import numpy as np; "
                "A.HorizontalFlip(p=1)(image=np.zeros((8, 8, 3), dtype=np.uint8)); "
                "print(A.__version__)"
            ),
        ],
    )


def _resolve_wheel(wheel: Path) -> Path:
    wheels = sorted(wheel.parent.glob(wheel.name))
    if len(wheels) != 1:
        msg = f"Expected exactly one wheel matching {wheel}, found {len(wheels)}."
        raise RuntimeError(msg)
    return wheels[0]


def verify_install_contract(wheel: Path, python_version: str, opencv_requirement: str) -> None:
    """Prove the Torch-free wheel state and the CPU-Torch import state from a clean environment."""
    wheel = _resolve_wheel(wheel)
    with tempfile.TemporaryDirectory(prefix="albumentationsx-install-contract-") as temporary_directory:
        environment = Path(temporary_directory) / "environment"
        _run(["uv", "venv", "--python", python_version, str(environment)])
        interpreter = _interpreter_path(environment)
        _run(["uv", "pip", "install", "--python", str(interpreter), str(wheel), opencv_requirement])
        _check_interpreter_runtime(interpreter, "absent")
        _assert_missing_torch_import(interpreter)
        install_cpu_torch(interpreter)
        _run_import_smoke(interpreter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    check_parser = commands.add_parser("check", help="Validate a Torch runtime state.")
    check_parser.add_argument("--expect", choices=("absent", "cpu"), required=True)

    install_parser = commands.add_parser("install-cpu", help="Install the validated CPU-only Torch runtime.")
    install_parser.add_argument("--python", dest="interpreter", type=Path, required=True)

    contract_parser = commands.add_parser("install-contract", help="Validate a built wheel in a clean environment.")
    contract_parser.add_argument("--wheel", type=Path, required=True)
    contract_parser.add_argument("--python", dest="python_version", required=True)
    contract_parser.add_argument("--opencv", default=DEFAULT_OPENCV_REQUIREMENT)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "check":
        check_runtime(args.expect)
    elif args.command == "install-cpu":
        install_cpu_torch(args.interpreter)
    else:
        verify_install_contract(args.wheel, args.python_version, args.opencv)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
