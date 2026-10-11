"""Extend reviewed versions only when hash-pinned distribution license evidence is unchanged."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.licenses import InvalidLicenseExpression, canonicalize_license_expression

from tools.verify_dependency_licenses import (
    check_requirements,
    load_registry,
    normalize_name,
    registry_by_name,
    requirement_components,
)

LICENSE_NAME = re.compile(r"^(licen[cs]e|notice|copying|copyright|authors)(?:[._-]|$)", re.IGNORECASE)
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


def download_artifact(artifact: dict[str, Any], cache: Path) -> bytes:
    """Read a PyPI artifact and verify its lockfile SHA-256, including cached bytes."""
    url = artifact["url"]
    parsed = urllib.parse.urlsplit(url)
    digest = artifact["hash"]
    if (
        parsed.scheme != "https"
        or parsed.netloc != "files.pythonhosted.org"
        or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
    ):
        raise ValueError(f"Untrusted distribution URL or digest: {url}")
    path = cache / digest.removeprefix("sha256:")
    if path.exists():
        content = path.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=30) as response:
            if urllib.parse.urlsplit(response.geturl()).netloc != "files.pythonhosted.org":
                raise ValueError(f"Unexpected artifact redirect: {url}")
            content = response.read(MAX_ARTIFACT_BYTES + 1)
    if len(content) > MAX_ARTIFACT_BYTES or "sha256:" + hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"Distribution digest or size mismatch: {url}")
    cache.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
    return content


def _archive_files(content: bytes, *, wheel: bool) -> dict[str, bytes]:
    """Read only metadata and legal files, without extracting or executing an archive."""
    files = {}
    if wheel or zipfile.is_zipfile(io.BytesIO(content)):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            for member in archive.infolist():
                if not member.is_dir() and _evidence_path(member.filename):
                    if (
                        stat.S_ISLNK(member.external_attr >> 16)
                        or member.file_size > MAX_EVIDENCE_BYTES
                        or member.filename in files
                    ):
                        raise ValueError(f"Non-regular, duplicate or oversized evidence: {member.filename}")
                    files[member.filename] = archive.read(member)
    else:
        with tarfile.open(fileobj=io.BytesIO(content)) as archive:
            for member in archive.getmembers():
                if _evidence_path(member.name):
                    if not member.isfile() or member.size > MAX_EVIDENCE_BYTES or member.name in files:
                        raise ValueError(f"Non-regular, duplicate or oversized evidence: {member.name}")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"Unreadable evidence: {member.name}")
                    files[member.name] = stream.read()
    return files


def _evidence_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if path.startswith("/") or ".." in parts or "\\" in path:
        raise ValueError(f"Unsafe archive path: {path}")
    return (
        path.endswith(("/METADATA", "/PKG-INFO"))
        or ".dist-info/licenses/" in path
        or any(LICENSE_NAME.match(part) for part in parts)
    )


def _license_metadata(message: Any, entry: dict[str, Any]) -> dict[str, list[str]]:
    fields = {field: sorted(message.get_all(field, [])) for field in ("License", "License-Expression", "License-File")}
    fields["Classifier"] = sorted(
        value for value in message.get_all("Classifier", []) if value.startswith("License ::")
    )
    accepted = {canonicalize_license_expression(entry["license_expression"]), *entry.get("metadata_identifiers", [])}
    observed = set(fields["Classifier"])
    for value in fields["License"] + fields["License-Expression"]:
        try:
            expression = canonicalize_license_expression(value)
        except InvalidLicenseExpression:
            if value in fields["License-Expression"]:
                raise ValueError(f"Invalid License-Expression: {value}") from None
        else:
            if expression not in accepted:
                raise ValueError(f"License metadata disagrees with the reviewed registry: {value}")
            observed.add(expression)
    if not observed & accepted:
        raise ValueError(f"Missing reviewed license metadata for {entry['name']}")
    return fields


def artifact_evidence(package: dict[str, Any], artifact: dict[str, Any], cache: Path, entry: dict[str, Any]) -> dict:
    """Fingerprint declared license files and all packaged licenses and notices."""
    wheel = urllib.parse.urlsplit(artifact["url"]).path.endswith(".whl")
    files = _archive_files(download_artifact(artifact, cache), wheel=wheel)
    metadata_paths = [
        path
        for path in files
        if (wheel and path.endswith(".dist-info/METADATA"))
        or (not wheel and len(PurePosixPath(path).parts) == 2 and path.endswith("/PKG-INFO"))
    ]
    if len(metadata_paths) != 1:
        raise ValueError(f"Missing or ambiguous distribution metadata: {artifact['url']}")
    metadata_path = metadata_paths[0]
    message = BytesParser().parsebytes(files[metadata_path])
    if normalize_name(message.get("Name", "")) != package["name"] or message.get("Version") != package["version"]:
        raise ValueError(f"Distribution identity mismatch: {artifact['url']}")
    metadata = _license_metadata(message, entry)
    root = metadata_path.rsplit("/", 1)[0] + ("/licenses/" if wheel else "/")
    declared = metadata["License-File"]
    if not declared or any(not files.get(root + path, b"").strip() for path in declared):
        raise ValueError(f"Missing declared license files: {artifact['url']}")
    licenses = {}
    for path, data in files.items():
        if path.endswith(("/METADATA", "/PKG-INFO")):
            continue
        relative = path
        if path.startswith(root):
            relative = path.removeprefix(root)
        elif not wheel:
            relative = path.split("/", 1)[-1]
        licenses[relative] = hashlib.sha256(data).hexdigest()
    evidence = {"kind": "wheel" if wheel else "sdist", "metadata": metadata, "files": licenses}
    signature = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    return {"url": artifact["url"], "sha256": artifact["hash"], "signature": signature, **evidence}


def package_evidence(package: dict[str, Any], cache: Path, entry: dict[str, Any]) -> list[dict]:
    """Check every distribution pinned for one package, including platform variants."""
    if package.get("source") != {"registry": "https://pypi.org/simple"}:
        raise ValueError(f"Only PyPI distributions can be reviewed automatically: {package['name']}")
    artifacts = [*package.get("wheels", [])]
    if package.get("sdist"):
        artifacts.append(package["sdist"])
    if not artifacts:
        raise ValueError(f"No pinned distributions for {package['name']}=={package['version']}")
    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(lambda artifact: artifact_evidence(package, artifact, cache, entry), artifacts))


def review_updates(
    registry: dict[str, Any],
    base_lock: dict[str, Any],
    head_lock: dict[str, Any],
    base_requirements: Path,
    head_requirements: Path,
    cache: Path,
) -> tuple[dict[str, Any], list[dict]]:
    """Return an atomic registry update and evidence; reject every unresolved change."""
    if errors := check_requirements(registry, [base_requirements]):
        raise ValueError("Baseline registry is not reviewed: " + "; ".join(errors))
    base_components = requirement_components([base_requirements])
    head_components = requirement_components([head_requirements])
    if {name for name, _ in base_components} != {name for name, _ in head_components}:
        raise ValueError("Base dependency names changed; a maintainer license review is required")
    updated = copy.deepcopy(registry)
    entries = registry_by_name(updated)
    base_packages = {(p["name"], p["version"]): p for p in base_lock["package"]}
    head_packages = {(p["name"], p["version"]): p for p in head_lock["package"]}
    reports = []
    for name, version in sorted(head_components):
        entry = entries[name]
        if version in entry["reviewed_versions"]:
            continue
        baseline = []
        for key in sorted(base_components):
            if key[0] == name:
                baseline.extend(package_evidence(base_packages[key], cache, entry))
        candidate = package_evidence(head_packages[name, version], cache, entry)
        signatures = {item["signature"] for item in baseline}
        if any(item["signature"] not in signatures for item in candidate):
            raise ValueError(f"License files or metadata changed for {name}=={version}; maintainer review required")
        entry["reviewed_versions"].append(version)
        reports.append({"name": name, "version": version, "baseline": baseline, "candidate": candidate})
    if errors := check_requirements(updated, [head_requirements]):
        raise ValueError("Updated registry is incomplete: " + "; ".join(errors))
    if reports:
        updated["reviewed_on"] = datetime.now(UTC).date().isoformat()
    return updated, reports


def export_requirements(project: Path, lock: Path, destination: Path) -> None:
    """Export an untrusted lockfile as data using the trusted project manifest, offline."""
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("uv executable is required")
    with tempfile.TemporaryDirectory(prefix="dependency-license-export-") as directory:
        root = Path(directory)
        shutil.copyfile(project, root / "pyproject.toml")
        shutil.copyfile(lock, root / "uv.lock")
        with destination.open("w", encoding="utf-8") as stream:
            subprocess.run(  # noqa: S603 - fixed export flags; no package code is executed.
                [uv, "export", "--offline", "--frozen", "--no-dev", "--no-emit-project", "--directory", str(root)],
                check=True,
                stdout=stream,
            )


def write_registry(path: Path, registry: dict[str, Any]) -> None:
    """Preserve the registry's compact component layout."""
    components = registry["components"]
    text = json.dumps({**registry, "components": []}, indent=2)
    rows = ",\n".join("    " + json.dumps(component, ensure_ascii=False) for component in components)
    path.write_text(text.replace('"components": []', '"components": [\n' + rows + "\n  ]") + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-lock", required=True, type=Path)
    parser.add_argument("--head-lock", required=True, type=Path)
    parser.add_argument("--project", default=Path("pyproject.toml"), type=Path)
    parser.add_argument("--registry", default=Path("legal/dependency-licenses.json"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        with tempfile.TemporaryDirectory(prefix="dependency-license-review-") as directory:
            base_requirements, head_requirements = (Path(directory) / name for name in ("base.txt", "head.txt"))
            export_requirements(args.project, args.base_lock, base_requirements)
            export_requirements(args.project, args.head_lock, head_requirements)
            registry, reports = review_updates(
                load_registry(args.registry),
                tomllib.loads(args.base_lock.read_text()),
                tomllib.loads(args.head_lock.read_text()),
                base_requirements,
                head_requirements,
                args.cache,
            )
        write_registry(args.output, registry)
        args.evidence.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
        print(f"Verified unchanged license evidence for {len(reports)} version update(s).")
    except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
