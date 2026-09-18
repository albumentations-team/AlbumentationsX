"""Tests for the dependency license registry verifier."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools import verify_dependency_licenses
from tools.verify_dependency_licenses import check_license_evidence, check_requirements, enrich_sbom, load_registry


def _registry(tmp_path: Path) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_sources": {"test": "test evidence"},
                "components": [
                    {
                        "name": "example-package",
                        "reviewed_versions": ["1.0"],
                        "license_expression": "MIT",
                        "evidence_source": "test",
                        "decision": "accepted",
                        "notice": "none",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_requirements_require_reviewed_versions(tmp_path: Path) -> None:
    registry = load_registry(_registry(tmp_path))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example-package==1.1\nnew-package==2.0\n", encoding="utf-8")

    assert check_requirements(registry, [requirements]) == [
        "example-package==1.1 is not a reviewed version in the dependency registry",
        "new-package==2.0 is absent from the reviewed dependency registry",
    ]


def test_requirements_reject_unsupported_lines(tmp_path: Path) -> None:
    registry = load_registry(_registry(tmp_path))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example-package == 1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"requirements\.txt:1: unsupported requirements line"):
        check_requirements(registry, [requirements])


def test_export_runtime_requirements_uses_repository_root(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(command: list[str], *, check: bool, cwd: Path) -> None:
        calls.append({"command": command, "check": check, "cwd": cwd})

    monkeypatch.setattr(verify_dependency_licenses.shutil, "which", lambda _: "/usr/local/bin/uv")
    monkeypatch.setattr(verify_dependency_licenses.subprocess, "run", fake_run)

    with verify_dependency_licenses.export_runtime_requirements() as paths:
        assert [path.name for path in paths] == ["runtime-requirements.txt", "all-runtime-requirements.txt"]

    assert len(calls) == 2
    assert all(call["check"] is True and call["cwd"] == verify_dependency_licenses.REPO_ROOT for call in calls)
    assert "--all-extras" in calls[1]["command"]


def test_main_reports_uv_export_failure(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fail_export() -> None:
        raise subprocess.CalledProcessError(2, ["uv", "export"])

    monkeypatch.setattr(verify_dependency_licenses, "export_runtime_requirements", fail_export)

    assert verify_dependency_licenses.main(["--export-runtime"]) == 1
    assert "ERROR: Command '['uv', 'export']' returned non-zero exit status 2." in capsys.readouterr().err


def test_enrich_sbom_writes_the_reviewed_expression(tmp_path: Path) -> None:
    registry = load_registry(_registry(tmp_path))
    sbom = tmp_path / "sbom.json"
    sbom.write_text(
        json.dumps({"components": [{"name": "Example_Package", "version": "1.0"}]}),
        encoding="utf-8",
    )

    assert enrich_sbom(registry, sbom) == []
    licenses = json.loads(sbom.read_text(encoding="utf-8"))["components"][0]["licenses"]
    assert licenses == [{"acknowledgement": "declared", "expression": "MIT"}]


def test_enrich_sbom_rejects_an_unreviewed_version(tmp_path: Path) -> None:
    registry = load_registry(_registry(tmp_path))
    sbom = tmp_path / "sbom.json"
    sbom.write_text(
        json.dumps({"components": [{"name": "Example_Package", "version": "1.1"}]}),
        encoding="utf-8",
    )

    assert enrich_sbom(registry, sbom) == [
        f"{sbom}: example-package==1.1 is not a reviewed version in the dependency registry",
    ]


def test_license_evidence_requires_every_active_locked_component(tmp_path: Path) -> None:
    registry = load_registry(_registry(tmp_path))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "example-package==1.0 ; sys_platform != 'never' # active\nmissing-package==1.0 ; sys_platform == 'never'\n",
        encoding="utf-8",
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"components": []}), encoding="utf-8")

    assert check_license_evidence(registry, [requirements], evidence) == [
        f"{evidence}: example-package==1.0 is absent from installed dependency license evidence",
    ]


def test_license_evidence_rejects_an_unreviewed_version(tmp_path: Path) -> None:
    registry = load_registry(_registry(tmp_path))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example-package==1.1\n", encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "components": [
                    {
                        "name": "example-package",
                        "version": "1.1",
                        "licenses": [{"expression": "MIT"}],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    assert check_license_evidence(registry, [requirements], evidence) == [
        f"{evidence}: example-package==1.1 is not a reviewed version in the dependency registry",
    ]


def test_license_evidence_normalizes_identifier_case(tmp_path: Path) -> None:
    registry = load_registry(_registry(tmp_path))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example-package==1.0\n", encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "components": [
                    {
                        "name": "example-package",
                        "version": "1.0",
                        "licenses": [{"expression": "mit"}],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    assert check_license_evidence(registry, [requirements], evidence) == []


def test_registry_rejects_empty_reviewed_versions(tmp_path: Path) -> None:
    path = _registry(tmp_path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    registry["components"][0]["reviewed_versions"] = []
    path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match=r"example-package needs reviewed_versions"):
        load_registry(path)


def test_registry_rejects_invalid_spdx_expression(tmp_path: Path) -> None:
    path = _registry(tmp_path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    registry["components"][0]["license_expression"] = "MIT OR"
    path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match=r"example-package has an invalid SPDX license_expression"):
        load_registry(path)
