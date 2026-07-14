"""Tests for license and CLA provenance verification."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

from tools.verify_legal_integrity import (
    FIRST_ONLY_RELEASE,
    REPO_ROOT,
    REQUIRED_LICENSE_FILES,
    collect_artifact_errors,
    collect_source_errors,
)


def _expected_files() -> dict[str, bytes]:
    return {relative: (REPO_ROOT / relative).read_bytes() for relative in REQUIRED_LICENSE_FILES}


def _distribution_metadata(
    *,
    version: str = FIRST_ONLY_RELEASE,
    license_expression: str = "AGPL-3.0-only",
    license_files: tuple[str, ...] = REQUIRED_LICENSE_FILES,
) -> bytes:
    lines = [
        "Metadata-Version: 2.4",
        "Name: albumentationsx",
        f"Version: {version}",
        f"License-Expression: {license_expression}",
        *(f"License-File: {relative_path}" for relative_path in license_files),
        "",
    ]
    return "\n".join(lines).encode()


def _write_wheel(
    path: Path,
    files: dict[str, bytes],
    *,
    include_cla: bool = False,
    include_source_only_asset: bool = False,
    metadata: bytes | None = None,
) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for relative_path, content in files.items():
            archive.writestr(f"albumentationsx-{FIRST_ONLY_RELEASE}.dist-info/licenses/{relative_path}", content)
        archive.writestr(
            f"albumentationsx-{FIRST_ONLY_RELEASE}.dist-info/METADATA",
            metadata or _distribution_metadata(),
        )
        if include_cla:
            archive.writestr("CLA.md", b"inbound agreement")
        if include_source_only_asset:
            archive.writestr("LICENSES/OFL-1.1.txt", b"source-only")


def _write_sdist(
    path: Path,
    files: dict[str, bytes],
    *,
    include_nested_artifact: bool = False,
    metadata: bytes | None = None,
) -> None:
    root = f"albumentationsx-{FIRST_ONLY_RELEASE}"
    with tarfile.open(path, "w:gz") as archive:
        for relative_path, content in files.items():
            info = tarfile.TarInfo(f"{root}/{relative_path}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        pkg_info = metadata or _distribution_metadata()
        info = tarfile.TarInfo(f"{root}/PKG-INFO")
        info.size = len(pkg_info)
        archive.addfile(info, io.BytesIO(pkg_info))
        if include_nested_artifact:
            nested_artifact = b"old wheel"
            info = tarfile.TarInfo(f"{root}/unexpected/albumentationsx-2.3.1-py3-none-any.whl")
            info.size = len(nested_artifact)
            archive.addfile(info, io.BytesIO(nested_artifact))


def test_source_legal_integrity() -> None:
    assert collect_source_errors() == []


def test_wheel_and_sdist_accept_exact_required_files(tmp_path: Path) -> None:
    expected_files = _expected_files()
    wheel = tmp_path / f"albumentationsx-{FIRST_ONLY_RELEASE}-py3-none-any.whl"
    sdist = tmp_path / f"albumentationsx-{FIRST_ONLY_RELEASE}.tar.gz"
    _write_wheel(wheel, expected_files)
    _write_sdist(sdist, expected_files)

    assert collect_artifact_errors(wheel, expected_files) == []
    assert collect_artifact_errors(sdist, expected_files) == []


def test_artifact_rejects_missing_notice(tmp_path: Path) -> None:
    expected_files = _expected_files()
    incomplete_files = expected_files.copy()
    incomplete_files.pop("THIRD_PARTY_NOTICES.md")
    wheel = tmp_path / f"albumentationsx-{FIRST_ONLY_RELEASE}-py3-none-any.whl"
    _write_wheel(wheel, incomplete_files)

    errors = collect_artifact_errors(wheel, expected_files)

    assert errors == [f"{wheel.name}: expected one copy of THIRD_PARTY_NOTICES.md, found 0"]


def test_artifact_rejects_cla(tmp_path: Path) -> None:
    expected_files = _expected_files()
    wheel = tmp_path / f"albumentationsx-{FIRST_ONLY_RELEASE}-py3-none-any.whl"
    _write_wheel(wheel, expected_files, include_cla=True)

    errors = collect_artifact_errors(wheel, expected_files)

    assert errors == [f"{wheel.name}: inbound CLA material leaked into artifact as CLA.md"]


def test_artifact_rejects_source_only_ofl_asset(tmp_path: Path) -> None:
    expected_files = _expected_files()
    wheel = tmp_path / f"albumentationsx-{FIRST_ONLY_RELEASE}-py3-none-any.whl"
    _write_wheel(wheel, expected_files, include_source_only_asset=True)

    errors = collect_artifact_errors(wheel, expected_files)

    assert errors == [f"{wheel.name}: source-only OFL test asset leaked into artifact as LICENSES/OFL-1.1.txt"]


def test_sdist_rejects_nested_distribution_artifact(tmp_path: Path) -> None:
    expected_files = _expected_files()
    sdist = tmp_path / f"albumentationsx-{FIRST_ONLY_RELEASE}.tar.gz"
    _write_sdist(sdist, expected_files, include_nested_artifact=True)

    errors = collect_artifact_errors(sdist, expected_files)

    assert errors == [
        f"{sdist.name}: nested distribution artifact leaked into sdist as "
        f"albumentationsx-{FIRST_ONLY_RELEASE}/unexpected/albumentationsx-2.3.1-py3-none-any.whl",
    ]


def test_wheel_rejects_wrong_license_expression(tmp_path: Path) -> None:
    expected_files = _expected_files()
    wheel = tmp_path / f"albumentationsx-{FIRST_ONLY_RELEASE}-py3-none-any.whl"
    metadata = _distribution_metadata(license_expression="AGPL-3.0-or-later")
    _write_wheel(wheel, expected_files, metadata=metadata)

    errors = collect_artifact_errors(wheel, expected_files)

    assert errors == [
        f"{wheel.name}: wheel METADATA License-Expression must be 'AGPL-3.0-only', found 'AGPL-3.0-or-later'",
    ]


def test_wheel_rejects_missing_license_file_metadata(tmp_path: Path) -> None:
    expected_files = _expected_files()
    wheel = tmp_path / f"albumentationsx-{FIRST_ONLY_RELEASE}-py3-none-any.whl"
    declared_license_files = tuple(
        relative_path for relative_path in REQUIRED_LICENSE_FILES if relative_path != "THIRD_PARTY_NOTICES.md"
    )
    _write_wheel(wheel, expected_files, metadata=_distribution_metadata(license_files=declared_license_files))

    errors = collect_artifact_errors(wheel, expected_files)

    assert errors == [
        f"{wheel.name}: wheel METADATA License-File entries must be exactly LICENSE, "
        "LICENSE_HISTORY.md, THIRD_PARTY_NOTICES.md, LICENSES/MIT-Albumentations-2.0.8.txt",
    ]


def test_sdist_rejects_wrong_license_expression(tmp_path: Path) -> None:
    expected_files = _expected_files()
    sdist = tmp_path / f"albumentationsx-{FIRST_ONLY_RELEASE}.tar.gz"
    metadata = _distribution_metadata(license_expression="AGPL-3.0-or-later")
    _write_sdist(sdist, expected_files, metadata=metadata)

    errors = collect_artifact_errors(sdist, expected_files)

    assert errors == [
        f"{sdist.name}: sdist PKG-INFO License-Expression must be 'AGPL-3.0-only', found 'AGPL-3.0-or-later'",
    ]


def test_sdist_rejects_missing_license_file_metadata(tmp_path: Path) -> None:
    expected_files = _expected_files()
    sdist = tmp_path / f"albumentationsx-{FIRST_ONLY_RELEASE}.tar.gz"
    declared_license_files = tuple(
        relative_path for relative_path in REQUIRED_LICENSE_FILES if relative_path != "THIRD_PARTY_NOTICES.md"
    )
    _write_sdist(sdist, expected_files, metadata=_distribution_metadata(license_files=declared_license_files))

    errors = collect_artifact_errors(sdist, expected_files)

    assert errors == [
        f"{sdist.name}: sdist PKG-INFO License-File entries must be exactly LICENSE, "
        "LICENSE_HISTORY.md, THIRD_PARTY_NOTICES.md, LICENSES/MIT-Albumentations-2.0.8.txt",
    ]


def test_artifact_rejects_reuse_of_published_or_later_version(tmp_path: Path) -> None:
    expected_files = _expected_files()
    wheel = tmp_path / "albumentationsx-2.3.1-py3-none-any.whl"
    _write_wheel(wheel, expected_files, metadata=_distribution_metadata(version="2.3.1"))

    errors = collect_artifact_errors(wheel, expected_files)

    assert errors == [
        f"{wheel.name}: wheel METADATA reuses published version 2.3.1; "
        f"AGPL-3.0-only artifacts start at {FIRST_ONLY_RELEASE}",
    ]
