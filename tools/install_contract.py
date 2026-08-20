"""Verify AlbumentationsX wheel behavior before and after a caller-provided Torch install."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_OPENCV_REQUIREMENT = "opencv-python-headless>=5.0.0.93"
MISSING_TORCH_GUIDANCE = 'pip install "albumentationsx[headless]"'


def _run(command: list[str]) -> None:
    print("+", shlex.join(command))
    subprocess.run(command, check=True)  # noqa: S603 - commands are constructed from fixed CLI arguments.


def _interpreter_path(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _resolve_wheel(wheel: Path) -> Path:
    wheels = sorted(wheel.parent.glob(wheel.name))
    if len(wheels) != 1:
        msg = f"Expected exactly one wheel matching {wheel}, found {len(wheels)}."
        raise RuntimeError(msg)
    return wheels[0]


def _assert_torch_is_absent(interpreter: Path) -> None:
    command = "import importlib.util; raise SystemExit(importlib.util.find_spec('torch') is not None)"
    result = subprocess.run(  # noqa: S603 - interpreter and command are constructed by this tool.
        [str(interpreter), "-I", "-c", command],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        msg = "Torch is installed but the clean wheel contract requires a Torch-free environment."
        raise RuntimeError(msg)


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


def prepare_install_contract(wheel: Path, python_version: str, opencv_requirement: str) -> Path:
    """Create a clean wheel environment and prove the package fails clearly without Torch."""
    wheel = _resolve_wheel(wheel)
    environment = Path(tempfile.mkdtemp(prefix="albumentationsx-install-contract-")) / "environment"
    _run(["uv", "venv", "--python", python_version, str(environment)])
    interpreter = _interpreter_path(environment)
    _run(["uv", "pip", "install", "--python", str(interpreter), str(wheel), opencv_requirement])
    _assert_torch_is_absent(interpreter)
    _assert_missing_torch_import(interpreter)
    return interpreter


def run_import_smoke(interpreter: Path) -> None:
    """Import AlbumentationsX and run a small transform after foundation installs CPU Torch."""
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


def _write_github_output(path: Path, name: str, value: str) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(f"{name}={value}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Create and validate a clean Torch-free wheel environment.")
    prepare.add_argument("--wheel", type=Path, required=True)
    prepare.add_argument("--python", dest="python_version", required=True)
    prepare.add_argument("--opencv", default=DEFAULT_OPENCV_REQUIREMENT)
    prepare.add_argument("--github-output", type=Path)

    smoke = commands.add_parser("smoke", help="Run the package import smoke check.")
    smoke.add_argument("--python", dest="interpreter", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        interpreter = prepare_install_contract(args.wheel, args.python_version, args.opencv)
        _write_github_output(_github_output_path(args.github_output), "python", str(interpreter))
    else:
        run_import_smoke(args.interpreter)
    return 0


def _github_output_path(argument: Path | None) -> Path:
    if argument is not None:
        return argument
    value = os.environ.get("GITHUB_OUTPUT")
    if not value:
        msg = "GITHUB_OUTPUT is required when --github-output is not provided."
        raise RuntimeError(msg)
    return Path(value)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
