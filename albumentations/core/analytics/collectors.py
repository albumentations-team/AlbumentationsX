"""Data collectors for telemetry."""

from __future__ import annotations

import functools
import os
import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from albumentations import __version__ as albumentationsx_version
from albumentations.core.analytics.events import ComposeInitEvent

if TYPE_CHECKING:
    from albumentations.core.composition import Compose


def get_environment_info() -> dict[str, Any]:
    """Collect basic environment information without external dependencies.

    Returns:
        Dictionary with OS, CPU, GPU (if available), RAM, and environment type

    """
    return {
        "albumentationsx_version": albumentationsx_version,
        "python_version": f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}",
        "os": get_os_info(),
        "cpu": get_cpu_model(),
        "gpu": get_gpu_name(),
        "ram_gb": get_ram_size(),
        "environment": detect_environment(),
    }


def detect_environment() -> str:
    """Detect the runtime environment.

    Priority order: ci > colab > kaggle > docker > jupyter > local

    Returns:
        Environment name as string

    """
    # Check CI first
    if is_ci_environment():
        return "ci"

    # Check Colab
    if _check_module("google.colab"):
        return "colab"

    # Check Kaggle
    try:
        if Path("/kaggle/working").exists():
            return "kaggle"
    except OSError:
        pass

    # Check Docker
    try:
        if Path("/.dockerenv").exists() or Path("/proc/self/cgroup").is_file():
            return "docker"
    except OSError:
        pass

    # Check Jupyter
    if _check_jupyter():
        return "jupyter"

    return "local"


def _check_module(module_name: str) -> bool:
    """Check if a module is available."""
    try:
        import importlib.util

        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError):
        return False
    else:
        return spec is not None


def _check_jupyter() -> bool:
    """Check if running in Jupyter notebook."""
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            return False
    except (ImportError, NameError):
        return False
    else:
        return ipython.__class__.__name__ in ["ZMQInteractiveShell", "TerminalInteractiveShell"]


@functools.lru_cache(maxsize=1)
def _get_linux_os_info() -> str:
    """Get Linux OS information."""
    # Try to get distribution info
    try:
        if hasattr(platform, "freedesktop_os_release"):
            os_info = platform.freedesktop_os_release()

            if name := os_info.get("PRETTY_NAME", ""):
                return name
    except (OSError, AttributeError):
        pass

    # Fallback to /etc/os-release
    try:
        os_release_path = Path("/etc/os-release")
        if os_release_path.exists():
            with os_release_path.open() as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass

    return "Linux"


@functools.lru_cache(maxsize=1)
def get_os_info() -> str:
    """Get OS information in a simple format."""
    system = platform.system()

    if system == "Darwin":  # macOS
        version = platform.mac_ver()[0]
        return f"macOS {version}" if version else "macOS"
    if system == "Windows":
        # Simple Windows detection
        release = platform.release()
        version = platform.version()
        if release == "10" and version and version.startswith("10.0.22"):
            return "Windows 11"
        return f"Windows {release}" if release else "Windows"
    if system == "Linux":
        return _get_linux_os_info()

    # Other systems
    return f"{system} {platform.release()}"


@functools.lru_cache(maxsize=1)
def get_cpu_model() -> str:
    """Get CPU model name in a simplified way."""
    # First try platform.processor() - often gives good info
    processor = platform.processor()
    if processor and processor not in ["", "unknown", "arm", "arm64", "x86_64", "i386", "AMD64", "aarch64"]:
        return processor

    # Special handling for Apple Silicon on macOS
    if platform.system() == "Darwin":
        try:
            # Check for Apple Silicon
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],  # noqa: S607
                check=False,
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass

    # Fallback to machine architecture
    if machine := platform.machine():
        # Provide meaningful names for common architectures
        arch_names = {
            "arm64": "ARM64",
            "aarch64": "ARM64",
            "x86_64": "x86-64",
            "AMD64": "x86-64",
            "i386": "x86",
            "i686": "x86",
        }
        return arch_names.get(machine, machine)

    return "Unknown"


@functools.lru_cache(maxsize=1)
def get_gpu_name() -> str | None:
    """Get GPU name if torch is available and CUDA is accessible."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return None


def _get_ram_linux() -> float | None:
    """Get RAM size on Linux."""
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        with meminfo_path.open() as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 1)
    return None


def _get_ram_macos() -> float | None:
    """Get RAM size on macOS."""
    result = subprocess.run(
        ["sysctl", "-n", "hw.memsize"],  # noqa: S607
        check=False,
        capture_output=True,
        text=True,
        timeout=1,
    )
    if result.returncode == 0:
        bytes_val = int(result.stdout.strip())
        return round(bytes_val / (1024**3), 1)
    return None


def _get_ram_windows() -> float | None:
    """Get RAM size on Windows."""
    result = subprocess.run(
        ["wmic", "computersystem", "get", "TotalPhysicalMemory"],  # noqa: S607
        check=False,
        capture_output=True,
        text=True,
        timeout=1,
    )
    if result.returncode == 0:
        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:
            bytes_val = int(lines[1].strip())
            return round(bytes_val / (1024**3), 1)
    return None


@functools.lru_cache(maxsize=1)
def get_ram_size() -> float | None:
    """Get RAM size in GB without external dependencies."""
    try:
        system = platform.system()

        if system == "Linux":
            return _get_ram_linux()
        if system == "Darwin":  # macOS
            return _get_ram_macos()
        if system == "Windows":
            return _get_ram_windows()

    except (OSError, ValueError, subprocess.SubprocessError):
        pass

    return None


def is_ci_environment() -> bool:
    """Check if running in a CI/CD environment.

    Returns:
        True if any CI environment variable is detected

    """
    ci_env_vars = [
        "CI",
        "CONTINUOUS_INTEGRATION",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "JENKINS_HOME",
        "TRAVIS",
        "CIRCLECI",
        "BUILDKITE",
        "DRONE",
        "TEAMCITY_VERSION",
        "BITBUCKET_BUILD_NUMBER",
        "SEMAPHORE",
        "APPVEYOR",
        "CODEBUILD_BUILD_ID",
        "AZURE_PIPELINES_BUILD_ID",
        "TF_BUILD",
    ]
    return any(os.getenv(var) for var in ci_env_vars)


def is_pytest_running() -> bool:
    """Check if pytest is currently running.

    Returns:
        True if pytest is detected in the environment

    """
    return "PYTEST_CURRENT_TEST" in os.environ


def _extract_transforms_from_compose(transform: Any, transforms: list[str]) -> None:
    """Recursively extract transform names from a compose structure."""
    if hasattr(transform, "transforms") and transform.transforms:
        for t in transform.transforms:
            _extract_transform_names(t, transforms)
    elif hasattr(transform, "transforms_dict") and transform.transforms_dict:
        # For OneOf, SomeOf, etc.
        for t in transform.transforms_dict.values():
            if hasattr(t, "__iter__"):
                for sub_t in t:
                    _extract_transform_names(sub_t, transforms)
            else:
                _extract_transform_names(t, transforms)


def _extract_transform_names(transform: Any, transforms: list[str]) -> None:
    """Extract transform names from a single transform."""
    # Get the class name
    class_name = transform.__class__.__name__

    # Skip Lambda transforms
    if class_name == "Lambda":
        return

    # Add transform name
    transforms.append(class_name)

    # Handle nested structures
    compose_types = [
        "Compose",
        "ReplayCompose",
        "OneOf",
        "SomeOf",
        "Sequential",
        "SelectiveChannelTransform",
        "OneOrOther",
        "RandomOrder",
    ]
    if class_name in compose_types:
        _extract_transforms_from_compose(transform, transforms)


def _get_target_usage(compose: Compose) -> str:
    """Determine target usage from compose processors."""
    uses_keypoints = "keypoints" in compose.processors
    uses_bboxes = "bboxes" in compose.processors

    if uses_keypoints and uses_bboxes:
        return "bboxes_keypoints"
    if uses_bboxes:
        return "bboxes"
    if uses_keypoints:
        return "keypoints"
    return "None"


def collect_pipeline_info(compose: Compose) -> dict[str, Any]:
    """Collect information about the pipeline structure.

    Args:
        compose: The Compose instance to analyze

    Returns:
        Dictionary with transform names and target usage information

    """
    transforms: list[str] = []

    # Extract all transforms
    for transform in compose.transforms:
        _extract_transform_names(transform, transforms)

    # Determine target usage
    targets = _get_target_usage(compose)

    # Generate pipeline hash
    pipeline_hash = ComposeInitEvent.generate_pipeline_hash(transforms)

    return {
        "transforms": transforms,
        "targets": targets,
        "pipeline_hash": pipeline_hash,
    }
