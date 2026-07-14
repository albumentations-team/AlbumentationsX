"""Verify license, CLA provenance, and distribution artifact integrity."""

from __future__ import annotations

import argparse
import base64
import hashlib
import sys
import tarfile
import zipfile
from collections.abc import Mapping
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 uses the project dependency
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]

SPDX_LICENSE = "AGPL-3.0-only"
FIRST_ONLY_RELEASE = "2.3.2"
PUBLISHED_OR_LATER_VERSIONS = frozenset(
    {
        "2.1.3",
        "2.2.0",
        "2.2.1",
        "2.2.2",
        "2.2.3",
        "2.2.4",
        "2.2.5",
        "2.2.6",
        "2.3.0",
        "2.3.1",
    },
)

AGPL_TEXT_MARKER = b"                    GNU AFFERO GENERAL PUBLIC LICENSE\n"
AGPL_TEXT_SHA256 = "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"
MIT_208_SHA256 = "bea4dc8e93ae2784bccd45f1cdba53da97b99646bca390c7d725e17b72dc2180"
OFL_11_SHA256 = "21b459dcbf31a1933546fb2b2511bfcab7c51c500838cef480f2266d8aba93b3"
LIBERATION_SERIF_BOLD_SHA256 = "d754ba427cfe0bca54ae052384baa8f842da5bd6550ad4da024ac441e7a7d5ce"
CLA_V1_INITIAL_SHA256 = "0318da3ff5c1d7b6e67ab0affa59e97b7e64902ae591e0c2d9e39a6299f835e9"
CLA_V1_FORMATTED_SHA256 = "d3ce911a802d2cea06f4deeb406d5667155305f01ba3ef0bd550d41d363b50ed"
CLA_V2_SHA256 = "cf25a9fedf2fbc0d6f796f9e3bfebf0f5ce133177c8e7865522f614fa682d878"
CLA_V1_GIST_URL = "https://gist.github.com/ternaus/df31e11d8a3180ba5520f72b72d57198"
CLA_V1_GIST_REVISION = "3115a7364f5ab8a58a7e7ffa51dfdf1ec8a5b006"

REQUIRED_LICENSE_FILES = (
    "LICENSE",
    "LICENSE_HISTORY.md",
    "THIRD_PARTY_NOTICES.md",
    "LICENSES/MIT-Albumentations-2.0.8.txt",
)
SOURCE_ONLY_NOTICE_FILES = (
    "LICENSES/OFL-1.1.txt",
    "tests/files/LiberationSerif-Bold.ttf",
)


def sha256(data: bytes) -> str:
    """Return the lowercase SHA-256 digest for data."""
    return hashlib.sha256(data).hexdigest()


def _read_required_files(repo_root: Path) -> dict[str, bytes]:
    return {relative: (repo_root / relative).read_bytes() for relative in REQUIRED_LICENSE_FILES}


def _check_project_metadata(repo_root: Path) -> list[str]:
    errors: list[str] = []
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    project = pyproject.get("project", {})
    if project.get("license") != SPDX_LICENSE:
        errors.append(f"pyproject project.license must be {SPDX_LICENSE!r}")

    project_version = project.get("version")
    if not isinstance(project_version, str) or not project_version:
        errors.append("pyproject project.version must be a non-empty string")
    elif project_version in PUBLISHED_OR_LATER_VERSIONS:
        errors.append(
            f"pyproject version {project_version} is already published with AGPL-3.0-or-later metadata; "
            f"AGPL-3.0-only starts at {FIRST_ONLY_RELEASE}",
        )

    configured_license_files = set(project.get("license-files", []))
    if configured_license_files != set(REQUIRED_LICENSE_FILES):
        errors.append("pyproject project.license-files does not match the four required artifact notices")

    build_excludes = set(pyproject.get("tool", {}).get("hatch", {}).get("build", {}).get("exclude", []))
    required_excludes = ("CLA.md", "legal", "LICENSES/OFL-1.1.txt")
    errors.extend(
        f"pyproject hatch build exclusions must contain {excluded_path!r}"
        for excluded_path in required_excludes
        if excluded_path not in build_excludes
    )

    conda_metadata = (repo_root / "conda.recipe/meta.yaml").read_text()
    required_conda_lines = (
        "license: AGPL-3.0-only",
        "- LICENSE",
        "- LICENSE_HISTORY.md",
        "- THIRD_PARTY_NOTICES.md",
        "- LICENSES/MIT-Albumentations-2.0.8.txt",
    )
    errors.extend(
        f"conda metadata is missing {required_line!r}"
        for required_line in required_conda_lines
        if required_line not in conda_metadata
    )

    manifest = (repo_root / "MANIFEST.in").read_text()
    required_manifest_lines = (
        "include LICENSES/MIT-Albumentations-2.0.8.txt",
        "exclude CLA.md",
        "exclude LICENSES/OFL-1.1.txt",
        "prune legal",
    )
    errors.extend(
        f"MANIFEST.in is missing {required_line!r}"
        for required_line in required_manifest_lines
        if required_line not in manifest
    )
    return errors


def _check_license_texts(repo_root: Path) -> list[str]:
    errors: list[str] = []
    license_bytes = (repo_root / "LICENSE").read_bytes()
    if license_bytes.count(AGPL_TEXT_MARKER) != 1:
        errors.append("LICENSE must contain exactly one canonical AGPL text marker")
    else:
        agpl_text = license_bytes[license_bytes.index(AGPL_TEXT_MARKER) :]
        if sha256(agpl_text) != AGPL_TEXT_SHA256:
            errors.append("the complete AGPL text below the repository lead-in changed")

    lead_in_end = license_bytes.find(AGPL_TEXT_MARKER)
    license_lead_in = " ".join(license_bytes[:lead_in_end].decode().split())
    required_lead_in_phrases = (
        "code, tests, documentation, and other copyrightable material",
        "AGPL-3.0-only",
        "The AGPL permits commercial use subject to its terms.",
        "scope-specific permissions",
        "only when an executed agreement or order form expressly says so",
    )
    errors.extend(
        f"LICENSE lead-in is missing {required_phrase!r}"
        for required_phrase in required_lead_in_phrases
        if required_phrase not in license_lead_in
    )

    mit_bytes = (repo_root / "LICENSES/MIT-Albumentations-2.0.8.txt").read_bytes()
    if sha256(mit_bytes) != MIT_208_SHA256:
        errors.append("legacy Albumentations 2.0.8 MIT text changed")

    ofl_bytes = (repo_root / "LICENSES/OFL-1.1.txt").read_bytes()
    if sha256(ofl_bytes) != OFL_11_SHA256:
        errors.append("Liberation Serif Bold OFL-1.1 notice or license text changed")

    font_bytes = (repo_root / "tests/files/LiberationSerif-Bold.ttf").read_bytes()
    if sha256(font_bytes) != LIBERATION_SERIF_BOLD_SHA256:
        errors.append("LiberationSerif-Bold.ttf changed; review its provenance and notice before updating the hash")
    return errors


def _check_history_and_notices(repo_root: Path) -> list[str]:
    history = " ".join((repo_root / "LICENSE_HISTORY.md").read_text().split())
    required_history_phrases = (
        "4d2cf04b6635663275a747333754410ef255e54c",
        "c1720fbab8209450328ef2e68f0ddc0c4806f7a8",
        "068f0ec5a6a49e0b0f8b138ea0dcdc1d60cdcc21",
        "337dc65588a032e2bd878462cdc7c5cdb099c6b6",
        "2.1.3",
        "2.3.1",
        FIRST_ONLY_RELEASE,
        "AGPL-3.0-or-later",
        "AGPL-3.0-only",
        "No Retroactive Relicensing",
        "prospectively",
        "never be rebuilt or republished",
    )
    errors = [
        f"LICENSE_HISTORY.md is missing {phrase!r}" for phrase in required_history_phrases if phrase not in history
    ]

    notices = " ".join((repo_root / "THIRD_PARTY_NOTICES.md").read_text().split())
    required_notice_phrases = (
        "Copyright (c) 2017 Vladimir Iglovikov, Alexander Buslaev, Alexander Parinov,",
        MIT_208_SHA256,
        "tests/files/LiberationSerif-Bold.ttf",
        LIBERATION_SERIF_BOLD_SHA256,
        "Digitized data copyright (c) 2010 Google Corporation",
        "Copyright (c) 2012 Red Hat, Inc.",
        "Arimo, Tinos and Cousine",
        "Reserved Font Name Liberation",
        "SIL Open Font License, Version 1.1",
        "not relicensed under the repository default",
    )
    errors.extend(
        f"THIRD_PARTY_NOTICES.md is missing {phrase!r}" for phrase in required_notice_phrases if phrase not in notices
    )
    return errors


def _check_cla_archive(repo_root: Path) -> list[str]:
    errors: list[str] = []
    archive = repo_root / "legal/cla/archive"
    cla_v2 = (repo_root / "CLA.md").read_bytes()
    archived_v2 = (archive / "CLA-v2.0-2026-07-14.md").read_bytes()
    formatted_v1 = (archive / "CLA-v1-09b767a.md").read_bytes()
    encoded_initial_v1 = b"".join((archive / "CLA-v1-c1720fb.md.base64").read_bytes().split())
    initial_v1 = base64.b64decode(encoded_initial_v1, validate=True)

    expected_digests = {
        "CLA.md": (cla_v2, CLA_V2_SHA256),
        "archived CLA v2.0": (archived_v2, CLA_V2_SHA256),
        "formatted CLA v1": (formatted_v1, CLA_V1_FORMATTED_SHA256),
        "initial CLA v1": (initial_v1, CLA_V1_INITIAL_SHA256),
    }
    for label, (content, expected_digest) in expected_digests.items():
        if sha256(content) != expected_digest:
            errors.append(f"{label} does not match its immutable SHA-256 identifier")

    if cla_v2 != archived_v2:
        errors.append("root CLA.md is not byte-identical to the archived Version 2.0 text")

    manifest = (archive / "MANIFEST.md").read_text()
    required_manifest_values = (
        CLA_V1_INITIAL_SHA256,
        CLA_V1_FORMATTED_SHA256,
        CLA_V2_SHA256,
        CLA_V1_GIST_URL,
        CLA_V1_GIST_REVISION,
    )
    errors.extend(
        f"CLA manifest is missing {required_value}"
        for required_value in required_manifest_values
        if required_value not in manifest
    )

    normalized_cla = " ".join(cla_v2.decode().split())
    required_v2_phrases = (
        "exact legal entity",
        "scope and covered period",
        "Prior GitHub username(s) or other submission aliases",
        "any terms other than AGPL-3.0-only",
        "first or simultaneously",
        "natural author's personal rights",
        "Additional Individual Representations",
        "Additional Entity Representations",
        "Contributor Consequential-Damages Waiver",
    )
    errors.extend(
        f"CLA Version 2.0 is missing {phrase!r}" for phrase in required_v2_phrases if phrase not in normalized_cla
    )

    contributing = (repo_root / "CONTRIBUTING.md").read_text()
    if "A Version 1 signature does **not** accept\nVersion 2.0" not in contributing:
        errors.append("CONTRIBUTING.md must require Version 1 signers to reaccept Version 2.0")
    return errors


def _check_public_copy(repo_root: Path) -> list[str]:
    normalized_readme = " ".join((repo_root / "README.md").read_text().split())
    stale_phrases = (
        "Free for open-source projects",
        "For proprietary/commercial use",
        "bypassing the open-source requirements",
    )
    errors = [
        f"README contains stale categorical license wording: {phrase!r}"
        for phrase in stale_phrases
        if phrase in normalized_readme
    ]
    if "The AGPL permits commercial use subject to its terms." not in normalized_readme:
        errors.append("README must state that the AGPL permits commercial use subject to its terms")
    return errors


def collect_source_errors(repo_root: Path = REPO_ROOT) -> list[str]:
    """Collect all source-tree legal-integrity violations."""
    required_source_files = (*REQUIRED_LICENSE_FILES, *SOURCE_ONLY_NOTICE_FILES)
    errors = [
        f"missing required license or provenance file: {relative_path}"
        for relative_path in required_source_files
        if not (repo_root / relative_path).is_file()
    ]
    if errors:
        return errors

    errors.extend(_check_project_metadata(repo_root))
    errors.extend(_check_license_texts(repo_root))
    errors.extend(_check_history_and_notices(repo_root))
    errors.extend(_check_cla_archive(repo_root))
    errors.extend(_check_public_copy(repo_root))
    return errors


def _zip_members(artifact: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(artifact) as archive:
        return {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}


def _tar_members(artifact: Path) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    with tarfile.open(artifact, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is not None:
                members[member.name] = extracted.read()
    return members


def read_artifact_members(artifact: Path) -> dict[str, bytes]:
    """Read regular files from a wheel, zip file, or source tarball."""
    if zipfile.is_zipfile(artifact):
        return _zip_members(artifact)
    if tarfile.is_tarfile(artifact):
        return _tar_members(artifact)
    raise ValueError(f"unsupported distribution artifact: {artifact}")


def _matching_members(members: Mapping[str, bytes], relative_path: str) -> list[str]:
    normalized = relative_path.replace("\\", "/")
    return [name for name in members if name == normalized or name.endswith(f"/{normalized}")]


def _metadata_errors(artifact: Path, metadata: bytes, label: str) -> list[str]:
    message = BytesParser().parsebytes(metadata, headersonly=True)
    errors: list[str] = []

    license_expressions = message.get_all("License-Expression", [])
    if license_expressions != [SPDX_LICENSE]:
        expression_summary = ", ".join(license_expressions) if license_expressions else "missing"
        errors.append(
            f"{artifact.name}: {label} License-Expression must be {SPDX_LICENSE!r}, found {expression_summary!r}",
        )

    license_files = message.get_all("License-File", [])
    if len(license_files) != len(REQUIRED_LICENSE_FILES) or set(license_files) != set(REQUIRED_LICENSE_FILES):
        errors.append(
            f"{artifact.name}: {label} License-File entries must be exactly {', '.join(REQUIRED_LICENSE_FILES)}",
        )

    versions = message.get_all("Version", [])
    if len(versions) != 1:
        errors.append(f"{artifact.name}: {label} must contain exactly one Version field")
    elif versions[0] in PUBLISHED_OR_LATER_VERSIONS:
        errors.append(
            f"{artifact.name}: {label} reuses published version {versions[0]}; "
            f"AGPL-3.0-only artifacts start at {FIRST_ONLY_RELEASE}",
        )
    return errors


def _wheel_metadata_errors(artifact: Path, members: Mapping[str, bytes]) -> list[str]:
    metadata_members = [
        name
        for name in members
        if PurePosixPath(name).name == "METADATA" and PurePosixPath(name).parent.name.endswith(".dist-info")
    ]
    if len(metadata_members) != 1:
        return [f"{artifact.name}: expected exactly one wheel .dist-info/METADATA file, found {len(metadata_members)}"]
    return _metadata_errors(artifact, members[metadata_members[0]], "wheel METADATA")


def _sdist_metadata_errors(artifact: Path, members: Mapping[str, bytes]) -> list[str]:
    metadata_members = [
        name for name in members if PurePosixPath(name).name == "PKG-INFO" and len(PurePosixPath(name).parts) == 2
    ]
    if len(metadata_members) != 1:
        return [f"{artifact.name}: expected exactly one root sdist PKG-INFO file, found {len(metadata_members)}"]
    return _metadata_errors(artifact, members[metadata_members[0]], "sdist PKG-INFO")


def _forbidden_artifact_errors(artifact: Path, members: Mapping[str, bytes]) -> list[str]:
    errors: list[str] = []
    for member_name in members:
        member_path = PurePosixPath(member_name)
        if member_path.name == "CLA.md" or "legal/cla" in member_name:
            errors.append(f"{artifact.name}: inbound CLA material leaked into artifact as {member_name}")
        if member_path.name == "LiberationSerif-Bold.ttf" or member_name.endswith("LICENSES/OFL-1.1.txt"):
            errors.append(f"{artifact.name}: source-only OFL test asset leaked into artifact as {member_name}")
        if artifact.suffix != ".whl" and member_name.endswith((".whl", ".tar.gz")):
            errors.append(f"{artifact.name}: nested distribution artifact leaked into sdist as {member_name}")
    return errors


def collect_artifact_errors(artifact: Path, expected_files: Mapping[str, bytes]) -> list[str]:
    """Collect metadata, notice-content, and source-only exclusion errors."""
    members = read_artifact_members(artifact)
    if artifact.suffix == ".whl":
        errors = _wheel_metadata_errors(artifact, members)
    else:
        errors = _sdist_metadata_errors(artifact, members)

    for relative_path, expected_bytes in expected_files.items():
        matches = _matching_members(members, relative_path)
        if len(matches) != 1:
            errors.append(f"{artifact.name}: expected one copy of {relative_path}, found {len(matches)}")
            continue
        if members[matches[0]] != expected_bytes:
            errors.append(f"{artifact.name}: {relative_path} is not byte-identical to the source file")

    errors.extend(_forbidden_artifact_errors(artifact, members))
    return errors


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        nargs="*",
        type=Path,
        default=(),
        help="Built wheel and sdist paths to verify after the source-tree checks.",
    )
    return parser.parse_args()


def main() -> int:
    """Run source and optional artifact checks."""
    args = parse_args()
    errors = collect_source_errors(REPO_ROOT)
    if not errors and args.artifacts:
        expected_files = _read_required_files(REPO_ROOT)
        for artifact in args.artifacts:
            if not artifact.is_file():
                errors.append(f"artifact does not exist: {artifact}")
                continue
            errors.extend(collect_artifact_errors(artifact, expected_files))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    artifact_count = len(args.artifacts)
    print(f"Legal integrity verified: source tree and {artifact_count} artifact(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
