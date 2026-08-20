"""Collect reproducible environment metadata for CI evidence artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from shutil import which
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

PACKAGE_NAMES = (
    "albumentationsx",
    "albucore",
    "numpy",
    "scipy",
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
    "pydantic",
    "torch",
)


def _package_versions() -> dict[str, str | None]:
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        distribution_name = distribution.metadata.get("Name")
        if isinstance(distribution_name, str):
            installed[distribution_name.lower()] = distribution.version

    return {package_name: installed.get(package_name.lower()) for package_name in PACKAGE_NAMES}


def _opencv_runtime_version() -> str | None:
    try:
        import cv2
    except ImportError:
        return None
    return cv2.__version__


def _albumentations_runtime_version() -> str | None:
    try:
        import albumentations
    except ImportError:
        return None
    return albumentations.__version__


def _torch_runtime() -> dict[str, Any]:
    accelerator_distributions = sorted(
        distribution.metadata["Name"]
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name", "").casefold().startswith(("cuda", "nvidia-"))
    )
    if importlib.util.find_spec("torch") is None:
        return {"cuda_version": None, "installed": False, "accelerator_distributions": accelerator_distributions}

    try:
        import torch
    except Exception as error:  # noqa: BLE001 - capture the loader error as evidence instead of failing diagnostics.
        return {
            "cuda_version": None,
            "installed": True,
            "accelerator_distributions": accelerator_distributions,
            "loader_error": str(error),
        }

    return {
        "cuda_version": torch.version.cuda,
        "installed": True,
        "accelerator_distributions": accelerator_distributions,
        "version": torch.__version__,
    }


def _git_commit() -> str | None:
    git_executable = which("git")
    if git_executable is None:
        return None

    result = subprocess.run(  # noqa: S603
        (git_executable, "rev-parse", "HEAD"),
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_environment(command: str | None = None) -> dict[str, Any]:
    package_versions = _package_versions()
    package_versions["albumentations"] = _albumentations_runtime_version()

    return {
        "schema_version": 1,
        "command": command,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "github_runner_os": os.environ.get("RUNNER_OS"),
            "github_runner_image": os.environ.get("IMAGEOS"),
        },
        "packages": package_versions,
        "ci_environment": {
            "dependency_group": os.environ.get("ALBU_CI_DEPENDENCY_GROUP"),
            "runtime_profile": os.environ.get("ALBU_CI_RUNTIME_PROFILE"),
        },
        "opencv_runtime_version": _opencv_runtime_version(),
        "torch_runtime": _torch_runtime(),
        "git_commit": _git_commit(),
        "uv_lock_sha256": _file_sha256(REPO_ROOT / "uv.lock"),
        "github": {
            "workflow": os.environ.get("GITHUB_WORKFLOW"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "sha": os.environ.get("GITHUB_SHA"),
            "ref": os.environ.get("GITHUB_REF"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="JSON file to write.")
    parser.add_argument("--command", help="Test or build command represented by this environment.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(collect_environment(args.command), indent=2, sort_keys=True) + "\n")
    print(f"Wrote environment evidence to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
