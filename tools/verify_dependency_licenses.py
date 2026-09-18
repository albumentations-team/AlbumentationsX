"""Validate the reviewed runtime dependency registry and enrich CycloneDX SBOMs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from packaging.licenses import InvalidLicenseExpression, canonicalize_license_expression
from packaging.markers import InvalidMarker, Marker

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "legal" / "dependency-licenses.json"
REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^ ;\\]+)(?:\s*;\s*(?P<marker>.*?))?\s*\\?$",
)


def normalize_name(name: str) -> str:
    """Return the normalized Python distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _validate_component(component: Any, sources: Mapping[str, Any], previous_name: str, path: Path) -> str:
    """Validate one sorted registry component and return its name."""
    if not isinstance(component, Mapping):
        raise TypeError(f"{path}: every component must be an object")
    name = component.get("name")
    if not isinstance(name, str) or normalize_name(name) != name or name <= previous_name:
        raise ValueError(f"{path}: component names must be normalized, sorted, and unique")
    versions = component.get("reviewed_versions")
    if (
        not isinstance(versions, list)
        or not versions
        or not all(isinstance(version, str) and version for version in versions)
    ):
        raise ValueError(f"{path}: {name} needs reviewed_versions")
    expression = component.get("license_expression")
    if not isinstance(expression, str) or not expression:
        raise ValueError(f"{path}: {name} needs license_expression")
    try:
        canonicalize_license_expression(expression)
    except InvalidLicenseExpression as error:
        raise ValueError(f"{path}: {name} has an invalid SPDX license_expression") from error
    if not isinstance(component.get("notice"), str) or not component["notice"]:
        raise ValueError(f"{path}: {name} needs notice handling")
    source = component.get("evidence_source")
    if not isinstance(source, str) or source not in sources:
        raise ValueError(f"{path}: {name} has an unknown evidence_source")
    identifiers = component.get("metadata_identifiers")
    valid_identifiers = (
        isinstance(identifiers, list)
        and bool(identifiers)
        and all(isinstance(identifier, str) and identifier for identifier in identifiers)
    )
    if identifiers is not None and not valid_identifiers:
        raise TypeError(f"{path}: {name} metadata_identifiers must be a non-empty string list")
    if component.get("decision") != "accepted":
        raise ValueError(f"{path}: {name} must have an explicit accepted decision")
    return name


def load_registry(path: Path) -> dict[str, Any]:
    """Load and validate the reviewed dependency registry."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")
    sources = data.get("evidence_sources")
    components = data.get("components")
    if not isinstance(sources, Mapping) or not isinstance(components, list):
        raise TypeError(f"{path}: evidence_sources and components are required")
    previous_name = ""
    for component in components:
        previous_name = _validate_component(component, sources, previous_name, path)
    return data


def registry_by_name(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Index registry entries by normalized distribution name."""
    return {component["name"]: component for component in registry["components"]}


def review_error(entry: Mapping[str, Any] | None, name: str, version: str) -> str | None:
    """Return the registry error for an unreviewed dependency name or version."""
    if entry is None:
        return f"{name}=={version} is absent from the reviewed dependency registry"
    if version not in entry["reviewed_versions"]:
        return f"{name}=={version} is not a reviewed version in the dependency registry"
    return None


def exported_requirements(paths: Iterable[Path]) -> set[tuple[str, str, str | None]]:
    """Read pinned components and their optional markers from uv's requirements export."""
    requirements: set[tuple[str, str, str | None]] = set()
    for path in paths:
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.split(" #", maxsplit=1)[0].strip()
            if not line or line.startswith(("#", "--hash=")):
                continue
            match = REQUIREMENT.fullmatch(line)
            if match is None:
                raise ValueError(f"{path}:{line_number}: unsupported requirements line")
            requirements.add(
                (
                    normalize_name(match["name"]),
                    match["version"],
                    match["marker"].strip() if match["marker"] else None,
                ),
            )
    return requirements


def requirement_components(paths: Iterable[Path]) -> set[tuple[str, str]]:
    """Return every pinned component, including platform-specific lock entries."""
    return {(name, version) for name, version, _ in exported_requirements(paths)}


def active_requirement_components(paths: Iterable[Path]) -> set[tuple[str, str]]:
    """Return locked components whose PEP 508 markers apply to this environment."""
    active: set[tuple[str, str]] = set()
    for name, version, marker in exported_requirements(paths):
        try:
            applies = marker is None or Marker(marker).evaluate()
        except InvalidMarker as error:
            raise ValueError(f"invalid requirement marker {marker!r}") from error
        if applies:
            active.add((name, version))
    return active


def check_requirements(registry: Mapping[str, Any], paths: Iterable[Path]) -> list[str]:
    """Report exported packages that have no reviewed registry entry."""
    entries = registry_by_name(registry)
    errors = []
    for name, version in sorted(requirement_components(paths)):
        if error := review_error(entries.get(name), name, version):
            errors.append(error)
    return errors


def _component_key(component: Mapping[str, Any]) -> tuple[str, str] | None:
    name = component.get("name")
    version = component.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        return None
    return normalize_name(name), version


def enrich_sbom(registry: Mapping[str, Any], path: Path) -> list[str]:
    """Add reviewed SPDX expressions to every dependency component in an SBOM."""
    sbom = json.loads(path.read_text(encoding="utf-8"))
    components = sbom.get("components")
    if not isinstance(components, list):
        return [f"{path}: CycloneDX document has no components list"]

    entries = registry_by_name(registry)
    errors: list[str] = []
    for component in components:
        if not isinstance(component, dict):
            errors.append(f"{path}: CycloneDX component is not an object")
            continue
        key = _component_key(component)
        if key is None:
            errors.append(f"{path}: CycloneDX component has no name and version")
            continue
        name, version = key
        entry = entries.get(name)
        if error := review_error(entry, name, version):
            errors.append(f"{path}: {error}")
            continue
        component["licenses"] = [
            {
                "acknowledgement": "declared",
                "expression": entry["license_expression"],
            }
        ]
    if not errors:
        path.write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return errors


def license_identifiers(component: Mapping[str, Any]) -> set[str]:
    """Return SPDX expressions, IDs, and classifier names declared in an SBOM component."""
    identifiers: set[str] = set()
    licenses = component.get("licenses")
    if not isinstance(licenses, list):
        return identifiers
    for item in licenses:
        if not isinstance(item, Mapping):
            continue
        expression = item.get("expression")
        if isinstance(expression, str):
            identifiers.add(expression)
        license_data = item.get("license")
        if isinstance(license_data, Mapping):
            for key in ("id", "name"):
                value = license_data.get(key)
                if isinstance(value, str):
                    identifiers.add(value)
    return identifiers


def normalize_license_identifier(identifier: str) -> str:
    """Normalize insignificant case and whitespace differences in license metadata."""
    return " ".join(identifier.casefold().split())


def check_license_evidence(registry: Mapping[str, Any], requirements: Iterable[Path], path: Path) -> list[str]:
    """Compare installed-distribution license metadata with reviewed identifiers."""
    sbom = json.loads(path.read_text(encoding="utf-8"))
    components = sbom.get("components")
    if not isinstance(components, list):
        return [f"{path}: CycloneDX document has no components list"]
    required = active_requirement_components(requirements)
    required_names = {name for name, _ in required}
    entries = registry_by_name(registry)
    errors: list[str] = []
    observed: set[tuple[str, str]] = set()
    for component in components:
        if not isinstance(component, Mapping):
            continue
        key = _component_key(component)
        if key is None:
            continue
        name, version = key
        if key not in required:
            if name in required_names:
                errors.append(f"{path}: installed {name}=={version} does not match the locked export")
            continue
        observed.add(key)
        entry = entries.get(name)
        if error := review_error(entry, name, version):
            errors.append(f"{path}: {error}")
            continue
        expected = {
            normalize_license_identifier(identifier)
            for identifier in entry.get("metadata_identifiers", [entry["license_expression"]])
        }
        identifiers = {normalize_license_identifier(identifier) for identifier in license_identifiers(component)}
        if not expected.intersection(identifiers):
            errors.append(f"{path}: {name}=={version} has no matching reviewed license metadata")
    for name, version in sorted(required - observed):
        errors.append(f"{path}: {name}=={version} is absent from installed dependency license evidence")
    return errors


def check(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    """Run checks requested by the command-line arguments."""
    registry = load_registry(args.registry)
    errors = check_requirements(registry, args.requirements)
    if args.license_sbom is not None:
        errors.extend(check_license_evidence(registry, args.requirements, args.license_sbom))
    if args.write_sbom:
        if args.sbom is None:
            errors.append("--write-sbom requires --sbom")
        else:
            errors.extend(enrich_sbom(registry, args.sbom))
    elif args.sbom is not None:
        errors.append("--sbom requires --write-sbom")
    return registry, errors


def main(argv: list[str] | None = None) -> int:
    """Run registry validation and optional requirement/SBOM checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--requirements", type=Path, action="append", default=[])
    parser.add_argument("--sbom", type=Path)
    parser.add_argument("--license-sbom", type=Path)
    parser.add_argument("--write-sbom", action="store_true")
    args = parser.parse_args(argv)

    try:
        registry, errors = check(args)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors = [str(error)]

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Dependency license registry is valid: {len(registry['components'])} reviewed components.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
