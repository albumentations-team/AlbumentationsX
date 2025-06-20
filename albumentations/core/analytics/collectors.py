"""Data collectors for telemetry."""

from __future__ import annotations

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
    if is_ci_environment():
        return "ci"
    if is_colab_environment():
        return "colab"
    if is_kaggle_environment():
        return "kaggle"
    if is_docker_environment():
        return "docker"
    if is_jupyter_environment():
        return "jupyter"
    return "local"


def _try_freedesktop_os_release() -> str | None:
    """Try to get Linux info using platform.freedesktop_os_release()."""
    if not hasattr(platform, "freedesktop_os_release"):
        return None

    try:
        os_info = platform.freedesktop_os_release()
        name = os_info.get("PRETTY_NAME", "")
        if name:
            return name
        # Fallback to NAME and VERSION_ID
        name = os_info.get("NAME", "Linux")
        version = os_info.get("VERSION_ID", "")
        if version:
            return f"{name} {version}"
    except (OSError, AttributeError):
        pass
    return None


def _try_lsb_release() -> str | None:
    """Try to get Linux info using lsb_release command."""
    try:
        result = subprocess.run(
            ["lsb_release", "-d"],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
        if result.returncode == 0:
            # Parse "Description:\tUbuntu 22.04.3 LTS"
            desc = result.stdout.strip()
            if ":" in desc:
                return desc.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _try_os_release_file() -> str | None:
    """Try to get Linux info from /etc/os-release file."""
    try:
        os_release_path = Path("/etc/os-release")
        if os_release_path.exists():
            with os_release_path.open() as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return None


def _get_linux_os_info() -> str:
    """Get Linux distribution information."""
    # Try using platform.freedesktop_os_release() (Python 3.10+)
    info = _try_freedesktop_os_release()
    if info:
        return info

    # Try lsb_release command
    info = _try_lsb_release()
    if info:
        return info

    # Try reading /etc/os-release directly
    info = _try_os_release_file()
    if info:
        return info

    # Fallback to generic Linux
    return "Linux"


def get_os_info() -> str:
    """Get detailed OS information including version."""
    system = platform.system()

    if system == "Linux":
        return _get_linux_os_info()

    if system == "Darwin":  # macOS
        # Get macOS version
        version = platform.mac_ver()[0]
        if version:
            return f"macOS {version}"
        return "macOS"

    if system == "Windows":
        # Get Windows version
        version = platform.version()
        release = platform.release()
        if release == "10" and version and version.startswith("10.0.22"):
            return "Windows 11"
        if release:
            return f"Windows {release}"
        return "Windows"

    # For other systems, use platform info
    return f"{system} {platform.release()}"


def _get_macos_cpu_model() -> str | None:
    """Get CPU model for macOS systems."""
    # First try to get the specific chip model
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
        if result.returncode == 0 and result.stdout.strip():
            cpu_brand = result.stdout.strip()
            if cpu_brand:
                return cpu_brand
    except (OSError, subprocess.SubprocessError):
        pass

    # Check if it's Apple Silicon
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.arm64"],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
        if result.returncode == 0 and result.stdout.strip() == "1":
            # Try to get the specific model from system_profiler
            chip_info = _get_apple_silicon_chip_info()
            if chip_info:
                return chip_info
            # Fallback to architecture-based detection
            machine = platform.machine()
            if machine == "arm64":
                return "Apple Silicon"
    except (OSError, subprocess.SubprocessError):
        pass

    return None


def _get_apple_silicon_chip_info() -> str | None:
    """Get specific Apple Silicon chip information."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            output = result.stdout
            # Look for chip info in the output
            for line in output.split("\n"):
                if "Chip:" in line:
                    # Extract chip name (e.g., "Apple M1", "Apple M2 Pro")
                    chip = line.split(":", 1)[1].strip()
                    if chip:
                        return chip
                elif "Processor Name:" in line:
                    proc = line.split(":", 1)[1].strip()
                    if proc:
                        return proc
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _get_linux_cpu_model() -> str | None:
    """Get CPU model for Linux systems."""
    try:
        cpuinfo_path = Path("/proc/cpuinfo")
        if cpuinfo_path.exists():
            with cpuinfo_path.open() as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _get_windows_cpu_model() -> str | None:
    """Get CPU model for Windows systems."""
    try:
        result = subprocess.run(
            ["wmic", "cpu", "get", "name"],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                return lines[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def get_cpu_model() -> str:
    """Get CPU model name without external dependencies."""
    system = platform.system()

    # Special handling for macOS to detect Apple Silicon
    if system == "Darwin":
        macos_cpu = _get_macos_cpu_model()
        if macos_cpu:
            return macos_cpu

    # Try generic platform.processor() first
    processor = platform.processor()
    if processor and processor not in ["", "unknown", "arm", "arm64", "x86_64", "i386"]:
        return processor

    # OS-specific methods
    if system == "Linux":
        linux_cpu = _get_linux_cpu_model()
        if linux_cpu:
            return linux_cpu
    elif system == "Windows":
        windows_cpu = _get_windows_cpu_model()
        if windows_cpu:
            return windows_cpu

    # Fallback to machine type with better formatting
    machine = platform.machine()
    if machine:
        # Provide more meaningful names for common architectures
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


def get_gpu_name() -> str | None:
    """Get GPU name if torch is available and CUDA is accessible."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return None


def get_ram_size() -> float | None:  # noqa: C901
    """Get RAM size in GB without external dependencies."""
    try:
        if platform.system() == "Linux":
            # Try to read from /proc/meminfo
            meminfo_path = Path("/proc/meminfo")
            if meminfo_path.exists():
                with meminfo_path.open() as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            # Extract value in kB and convert to GB
                            kb = int(line.split()[1])
                            return kb / (1024 * 1024)
        elif platform.system() == "Darwin":  # macOS
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],  # noqa: S607
                check=False,
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                # Value is in bytes
                bytes_val = int(result.stdout.strip())
                return bytes_val / (1024**3)
        elif platform.system() == "Windows":
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
                    # Value is in bytes
                    bytes_val = int(lines[1].strip())
                    return bytes_val / (1024**3)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def is_colab_environment() -> bool:
    """Check if running in Google Colab."""
    try:
        # Use importlib to check if google.colab is available
        import importlib.util

        spec = importlib.util.find_spec("google.colab")
    except ImportError:
        return False
    else:
        return spec is not None


def is_kaggle_environment() -> bool:
    """Check if running in Kaggle."""
    return Path("/kaggle/working").exists()


def is_jupyter_environment() -> bool:
    """Check if running in Jupyter notebook."""
    try:
        # Check if IPython is available and get_ipython exists
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            return False

        # Check if it's a notebook environment
    except (ImportError, NameError):
        return False
    else:
        return ipython.__class__.__name__ in ["ZMQInteractiveShell", "TerminalInteractiveShell"]


def is_docker_environment() -> bool:
    """Check if running in Docker container."""
    return Path("/.dockerenv").exists() or Path("/proc/self/cgroup").is_file()


def is_ci_environment() -> bool:
    """Check if running in a CI/CD environment.

    Detects common CI environment variables used by various CI systems.

    Returns:
        True if any CI environment variable is detected

    """
    ci_env_vars = [
        "CI",  # Generic CI indicator
        "CONTINUOUS_INTEGRATION",  # Generic CI indicator
        "GITHUB_ACTIONS",  # GitHub Actions
        "GITLAB_CI",  # GitLab CI
        "JENKINS_HOME",  # Jenkins
        "TRAVIS",  # Travis CI
        "CIRCLECI",  # CircleCI
        "BUILDKITE",  # Buildkite
        "DRONE",  # Drone CI
        "TEAMCITY_VERSION",  # TeamCity
        "BITBUCKET_BUILD_NUMBER",  # Bitbucket Pipelines
        "SEMAPHORE",  # Semaphore CI
        "APPVEYOR",  # AppVeyor
        "CODEBUILD_BUILD_ID",  # AWS CodeBuild
        "AZURE_PIPELINES_BUILD_ID",  # Azure Pipelines
        "TF_BUILD",  # Azure DevOps
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
    # Check by class name to avoid circular imports
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

    # Generate targets string
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
