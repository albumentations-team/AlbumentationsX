import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def parse_requirement(req: str) -> tuple[str, str]:
    """Parse a requirement string into package name and version spec.

    Args:
        req: Requirement string like 'numpy>=1.24.4' or 'albucore==0.0.35'

    Returns:
        Tuple of (package_name, version_spec)

    """
    match = re.match(r"([a-zA-Z0-9_-]+)([>=<~!]+.+)", req)
    if match:
        return match[1], match[2]
    return req, ""


def normalize_package_name(name: str) -> str:
    """Normalize package name for comparison (conda vs pip naming)."""
    # Common package name mappings between pip and conda
    mappings = {
        "opencv-python-headless": "opencv-python-headless",
        "typing-extensions": "typing_extensions",
    }
    return mappings.get(name, name)


def versions_match(version1: str, version2: str) -> bool:
    """Check if two version specs are equivalent.

    Args:
        version1: First version spec (e.g., '>=1.10')
        version2: Second version spec (e.g., '>=1.10.0')

    Returns:
        True if versions match (considering .0 suffixes are equivalent)

    """
    # Direct match
    if version1 == version2:
        return True

    # Extract operator and version
    match1 = re.match(r"([>=<~!]+)(.+)", version1)
    match2 = re.match(r"([>=<~!]+)(.+)", version2)

    if not match1 or not match2:
        return False

    op1, ver1 = match1.groups()
    op2, ver2 = match2.groups()

    # Operators must match
    if op1 != op2:
        return False

    # Normalize versions by removing trailing .0 segments
    def normalize_version(ver: str) -> str:
        parts = ver.split(".")
        # Remove trailing zeros
        while len(parts) > 1 and parts[-1] == "0":
            parts.pop()
        return ".".join(parts)

    ver1_normalized = normalize_version(ver1)
    ver2_normalized = normalize_version(ver2)

    return ver1_normalized == ver2_normalized


def load_pip_dependencies() -> dict[str, str]:
    """Load dependencies from pyproject.toml.

    Returns:
        Dictionary mapping normalized package names to version specs

    """
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        msg = "Error: pyproject.toml not found"
        raise FileNotFoundError(msg)

    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)

    pip_deps = {}
    for dep in pyproject["project"]["dependencies"]:
        name, version = parse_requirement(dep)
        pip_deps[normalize_package_name(name)] = version

    return pip_deps


def load_conda_dependencies() -> dict[str, str]:
    """Load dependencies from conda meta.yaml.

    Returns:
        Dictionary mapping normalized package names to version specs

    """
    meta_yaml_path = Path("conda.recipe/meta.yaml")
    if not meta_yaml_path.exists():
        msg = "Error: conda.recipe/meta.yaml not found"
        raise FileNotFoundError(msg)

    with meta_yaml_path.open() as f:
        meta_yaml = f.read()

    run_section_match = re.search(r"run:\s*\n((?:    - .+\n)+)", meta_yaml)
    if not run_section_match:
        msg = "Error: Could not find 'run:' section in meta.yaml"
        raise ValueError(msg)

    run_deps_text = run_section_match[1]
    conda_deps = {}

    for line in run_deps_text.split("\n"):
        stripped_line = line.strip()
        if stripped_line.startswith("- ") and stripped_line != "- python":
            dep = stripped_line[2:].strip()
            name, version = parse_requirement(dep)
            conda_deps[normalize_package_name(name)] = version

    return conda_deps


def find_version_mismatches(pip_deps: dict[str, str], conda_deps: dict[str, str]) -> list[str]:
    """Find version mismatches between pip and conda dependencies.

    Args:
        pip_deps: Dictionary of pip dependencies
        conda_deps: Dictionary of conda dependencies

    Returns:
        List of error messages for mismatches

    """
    errors = []

    for pkg_name, pip_version in pip_deps.items():
        normalized_name = normalize_package_name(pkg_name)

        if normalized_name not in conda_deps:
            errors.append(f"Package '{pkg_name}' from pyproject.toml not found in meta.yaml run dependencies")
            continue

        conda_version = conda_deps[normalized_name]

        if not versions_match(pip_version, conda_version):
            errors.append(
                f"Version mismatch for '{pkg_name}':\n"
                f"  pyproject.toml: {pkg_name}{pip_version}\n"
                f"  meta.yaml:      {normalized_name}{conda_version}",
            )

    return errors


def check_conda_dependencies() -> int:
    """Check that conda meta.yaml uses correct versions from pyproject.toml."""
    try:
        pip_deps = load_pip_dependencies()
        conda_deps = load_conda_dependencies()
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        return 1

    errors = find_version_mismatches(pip_deps, conda_deps)

    if errors:
        print("Error: Dependency version mismatches found:\n")
        for error in errors:
            print(f"  {error}")
        print("\nPlease update conda.recipe/meta.yaml to match pyproject.toml")
        return 1

    print("✓ All conda dependencies match pyproject.toml")
    return 0


if __name__ == "__main__":
    sys.exit(check_conda_dependencies())
