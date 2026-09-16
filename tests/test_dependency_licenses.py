"""Tests for the dependency license registry verifier."""

from __future__ import annotations

import json
from pathlib import Path

from tools.verify_dependency_licenses import check_requirements, enrich_sbom, load_registry


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


def test_requirements_reject_an_unreviewed_package(tmp_path: Path) -> None:
    registry = load_registry(_registry(tmp_path))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example-package==1.1\nnew-package==2.0\n", encoding="utf-8")

    assert check_requirements(registry, [requirements]) == [
        "new-package==2.0 is absent from the reviewed dependency registry"
    ]


def test_enrich_sbom_writes_the_reviewed_expression(tmp_path: Path) -> None:
    registry = load_registry(_registry(tmp_path))
    sbom = tmp_path / "sbom.json"
    sbom.write_text(json.dumps({"components": [{"name": "Example_Package", "version": "1.1"}]}), encoding="utf-8")

    assert enrich_sbom(registry, sbom) == []
    assert json.loads(sbom.read_text(encoding="utf-8"))["components"][0]["licenses"] == [
        {"acknowledgement": "declared", "expression": "MIT"}
    ]
