"""Regressions for distribution evidence review and registry-only publication."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import tarfile
import tomllib
import zipfile
from pathlib import Path
from typing import Self

import pytest
import yaml

from tools import sync_dependency_licenses as sync
from tools import update_dependency_licenses as update
from tools.release_bundle import GitHubClient


def _registry() -> dict:
    return {
        "schema_version": 1,
        "reviewed_on": "2026-10-01",
        "evidence_sources": {"pypi": "Pinned distribution evidence"},
        "components": [
            {
                "name": "example-package",
                "reviewed_versions": ["1.0"],
                "license_expression": "MIT",
                "evidence_source": "pypi",
                "decision": "accepted",
                "notice": "Separately installed dependency.",
            }
        ],
    }


def _package(
    cache: Path,
    version: str,
    *,
    wheel: bool = True,
    license_text: bytes = b"The reviewed MIT license",
    extra_files: dict[str, bytes] | None = None,
    metadata: str | None = None,
) -> dict:
    root = f"example_package-{version}.dist-info" if wheel else f"example_package-{version}"
    metadata = metadata or f"Name: example-package\nVersion: {version}\nLicense: MIT\nLicense-File: LICENSE\n"
    files = {
        f"{root}/{'METADATA' if wheel else 'PKG-INFO'}": metadata.encode(),
        f"{root}/{'licenses/' if wheel else ''}LICENSE": license_text,
        "example_package/__init__.py" if wheel else f"{root}/setup.py": b"raise AssertionError('must not execute')",
        **(extra_files or {}),
    }
    stream = io.BytesIO()
    if wheel:
        with zipfile.ZipFile(stream, "w") as archive:
            for path, data in files.items():
                archive.writestr(path, data)
    else:
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for path, data in files.items():
                member = tarfile.TarInfo(path)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    content = stream.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    cache.mkdir(exist_ok=True)
    (cache / digest).write_bytes(content)
    artifact = {
        "url": f"https://files.pythonhosted.org/example-{version}.{'whl' if wheel else 'tar.gz'}",
        "hash": f"sha256:{digest}",
    }
    return {
        "name": "example-package",
        "version": version,
        "source": {"registry": "https://pypi.org/simple"},
        **({"wheels": [artifact]} if wheel else {"sdist": artifact}),
    }


def _review(tmp_path: Path, baseline: dict, candidate: dict, registry: dict | None = None) -> tuple[dict, list]:
    base = tmp_path / "base.txt"
    head = tmp_path / "head.txt"
    base.write_text("example-package==1.0\n", encoding="utf-8")
    head.write_text("example-package==1.1\n", encoding="utf-8")
    return update.review_updates(
        registry or _registry(), {"package": [baseline]}, {"package": [candidate]}, base, head, tmp_path / "cache"
    )


@pytest.mark.parametrize("wheel", [False, True])
def test_unchanged_distribution_evidence_extends_only_exact_versions(tmp_path: Path, wheel: bool) -> None:
    registry = _registry()
    baseline = _package(tmp_path / "cache", "1.0", wheel=wheel)
    candidate = _package(tmp_path / "cache", "1.1", wheel=wheel)

    updated, reports = _review(tmp_path, baseline, candidate, registry)

    assert updated["components"][0]["reviewed_versions"] == ["1.0", "1.1"]
    assert registry == _registry()
    assert reports[0]["baseline"][0]["signature"] == reports[0]["candidate"][0]["signature"]
    assert reports[0]["candidate"][0]["sha256"] != reports[0]["baseline"][0]["sha256"]


@pytest.mark.parametrize("wheel", [False, True])
@pytest.mark.parametrize("change", ["license", "notice", "metadata", "missing", "identity", "path"])
def test_changed_or_incomplete_evidence_stops_review(tmp_path: Path, wheel: bool, change: str) -> None:
    cache = tmp_path / "cache"
    baseline = _package(cache, "1.0", wheel=wheel)
    root = "example_package-1.1.dist-info/licenses" if wheel else "example_package-1.1"
    changes = {
        "license": {"license_text": b"New redistribution terms"},
        "notice": {"extra_files": {f"{root}/NOTICE": b"New third party attribution"}},
        "metadata": {"metadata": "Name: example-package\nVersion: 1.1\nLicense: Apache-2.0\nLicense-File: LICENSE\n"},
        "missing": {"metadata": "Name: example-package\nVersion: 1.1\nLicense: MIT\nLicense-File: missing.txt\n"},
        "identity": {"metadata": "Name: other-package\nVersion: 1.1\nLicense: MIT\nLicense-File: LICENSE\n"},
        "path": {"extra_files": {"../NOTICE": b"Escaping the package root"}},
    }
    candidate = _package(cache, "1.1", wheel=wheel, **changes[change])
    registry = _registry()

    with pytest.raises(ValueError):
        _review(tmp_path, baseline, candidate, registry)

    assert registry == _registry()


def test_every_platform_wheel_is_checked_and_failed_review_is_atomic(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    baseline = _package(cache, "1.0")
    candidate = _package(cache, "1.1")
    changed_wheel = _package(cache, "1.1", license_text=b"Different wheel license")
    candidate["wheels"].extend(changed_wheel["wheels"])
    registry = _registry()

    with pytest.raises(ValueError, match="License files or metadata changed"):
        _review(tmp_path, baseline, candidate, registry)

    assert registry == _registry()


@pytest.mark.parametrize("source", [{"git": "https://example.com/repo"}, {"registry": "https://example.com/simple"}])
def test_non_pypi_sources_require_manual_review(tmp_path: Path, source: dict) -> None:
    baseline = _package(tmp_path / "cache", "1.0")
    candidate = _package(tmp_path / "cache", "1.1")
    candidate["source"] = source

    with pytest.raises(ValueError, match="Only PyPI distributions"):
        _review(tmp_path, baseline, candidate)


def test_new_base_dependency_cannot_be_added_by_automation(tmp_path: Path) -> None:
    baseline = _package(tmp_path / "cache", "1.0")
    candidate = _package(tmp_path / "cache", "1.1")
    base = tmp_path / "base.txt"
    head = tmp_path / "head.txt"
    base.write_text("example-package==1.0\n", encoding="utf-8")
    head.write_text("example-package==1.1\nnew-package==2.0 ; sys_platform == 'never'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Base dependency names changed"):
        update.review_updates(
            _registry(), {"package": [baseline]}, {"package": [candidate]}, base, head, tmp_path / "cache"
        )


def test_cached_distribution_bytes_must_match_the_pinned_digest(tmp_path: Path) -> None:
    package = _package(tmp_path, "1.0")
    artifact = package["wheels"][0]
    (tmp_path / artifact["hash"].removeprefix("sha256:")).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="digest or size mismatch"):
        update.download_artifact(artifact, tmp_path)


@pytest.mark.parametrize("url", ["http://files.pythonhosted.org/example.whl", "https://example.com/example.whl"])
def test_distribution_downloads_reject_untrusted_urls_before_network_access(tmp_path: Path, url: str) -> None:
    with pytest.raises(ValueError, match="Untrusted distribution URL"):
        update.download_artifact({"url": url, "hash": "sha256:" + "0" * 64}, tmp_path)


def test_duplicate_license_archive_members_are_rejected(tmp_path: Path) -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("package.dist-info/licenses/LICENSE", b"first")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("package.dist-info/licenses/LICENSE", b"second")

    with pytest.raises(ValueError, match="duplicate or oversized evidence"):
        update._archive_files(stream.getvalue(), wheel=True)


def _pull_request() -> dict:
    return {
        "number": 137,
        "state": "open",
        "user": {"login": "dependabot[bot]"},
        "head": {"sha": "head", "ref": "dependabot/uv/group", "repo": {"full_name": "owner/repo"}},
        "base": {"sha": "base", "ref": "main"},
        "changed_files": 1,
    }


class _API:
    def __init__(self, *, pull: dict | None = None, files: list | None = None, race: bool = False) -> None:
        self.pull = pull or _pull_request()
        self.files = files if files is not None else [{"filename": "uv.lock", "status": "modified"}]
        self.race = race
        self.writes = []

    def get_json(self, path: str) -> dict | list:
        if "/files?" in path:
            return self.files
        return copy.deepcopy(self.pull)

    def post_json(self, path: str, payload: dict) -> dict:
        self.writes.append((path, payload))
        if path == "/graphql":
            if self.race:
                return {"errors": [{"message": "Expected branch head oid did not match"}]}
            return {"data": {"createCommitOnBranch": {"commit": {"oid": "commits-sha"}}}}
        return {"sha": path.rsplit("/", 1)[-1] + "-sha"}


@pytest.mark.parametrize("change", ["fork", "human", "closed", "retargeted", "base", "branch", "scope", "removed"])
def test_privileged_registry_update_rejects_ineligible_prs(change: str) -> None:
    pull = _pull_request()
    files = [{"filename": "uv.lock", "status": "modified"}]
    if change == "fork":
        pull["head"]["repo"]["full_name"] = "fork/repo"
    elif change == "human":
        pull["user"]["login"] = "human"
    elif change == "closed":
        pull["state"] = "closed"
    elif change == "retargeted":
        pull["base"]["ref"] = "other"
    elif change == "base":
        pull["base"]["sha"] = "new-base"
    elif change == "branch":
        pull["head"]["ref"] = "untrusted/branch"
    elif change == "scope":
        files[0]["filename"] = "tools/update_dependency_licenses.py"
    else:
        files[0]["status"] = "removed"
    api = _API(pull=pull, files=files)

    with pytest.raises(ValueError):
        sync.validate_pull_request(api, "owner/repo", 137, "base")

    assert api.writes == []


def test_registry_publication_preserves_tree_and_dispatches_normal_pr_checks() -> None:
    api = _API()

    assert sync.publish_registry(api, "owner/repo", _pull_request(), "verified registry") == "commits-sha"

    assert api.writes[0][0] == "/graphql"
    commit = api.writes[0][1]["variables"]["input"]
    assert commit["branch"] == {"repositoryNameWithOwner": "owner/repo", "branchName": "dependabot/uv/group"}
    assert commit["expectedHeadOid"] == "head"
    assert commit["fileChanges"] == {
        "additions": [
            {"path": "legal/dependency-licenses.json", "contents": base64.b64encode(b"verified registry").decode()}
        ]
    }
    assert api.writes[1] == (
        "/repos/owner/repo/actions/workflows/pr.yml/dispatches",
        {"ref": "dependabot/uv/group", "inputs": {"pull-request": "137"}},
    )


def test_branch_race_rejects_write_and_does_not_dispatch() -> None:
    api = _API(race=True)

    with pytest.raises(ValueError, match="Expected branch head oid did not match"):
        sync.publish_registry(api, "owner/repo", _pull_request(), "verified registry")

    assert not any("/dispatches" in path for path, _ in api.writes)


def test_stale_review_cannot_create_commits() -> None:
    changed = _pull_request()
    changed["head"]["sha"] = "advanced"
    api = _API(pull=changed)

    with pytest.raises(ValueError, match="changed during license review"):
        sync.publish_registry(api, "owner/repo", _pull_request(), "verified registry")

    assert api.writes == []


def test_github_client_dispatch_accepts_empty_204_response(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []

    class Response:
        status = 204

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    def urlopen(request: object, timeout: int) -> Response:
        requests.append(request)
        assert timeout == 30
        return Response()

    monkeypatch.setattr("tools.release_bundle.urllib.request.urlopen", urlopen)
    api = GitHubClient("test-token")

    assert api.post_json("/dispatches", {"ref": "branch", "inputs": {"pull-request": "137"}}) == {}
    assert requests[0].get_method() == "POST"
    assert json.loads(requests[0].data) == {"ref": "branch", "inputs": {"pull-request": "137"}}


def test_registry_sync_workflow_executes_trusted_code_without_pr_checkout_or_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / ".github/workflows/dependency-license-sync.yml"
    workflow = yaml.safe_load(path.read_text())
    assert workflow[True] == {"workflow_run": {"workflows": ["PR"], "types": ["requested"]}}
    job = workflow["jobs"]["synchronize"]
    checkout = job["steps"][0]
    assert checkout["with"] == {"ref": "${{ github.sha }}", "persist-credentials": False}
    assert "github.event.workflow_run.event == 'pull_request'" in job["if"]
    assert "github.event.workflow_run.actor.login == 'dependabot[bot]'" in job["if"]
    assert job["permissions"] == {"contents": "write", "actions": "write", "pull-requests": "read"}
    assert not any("download-artifact" in step.get("uses", "") for step in job["steps"])


@pytest.mark.parametrize("change", ["event", "actor", "path", "fork", "associations"])
def test_workflow_run_resolution_rejects_privilege_escalation_and_recursion(change: str) -> None:
    run = {
        "event": "pull_request",
        "path": ".github/workflows/pr.yml",
        "actor": {"login": "dependabot[bot]"},
        "head_repository": {"full_name": "owner/repo"},
        "pull_requests": [{"number": 137}],
        "head_sha": "head",
    }
    if change == "event":
        run["event"] = "workflow_dispatch"
    elif change == "actor":
        run["actor"]["login"] = "human"
    elif change == "path":
        run["path"] = ".github/workflows/untrusted.yml"
    elif change == "fork":
        run["head_repository"]["full_name"] = "fork/repo"
    else:
        run["pull_requests"] = []

    with pytest.raises(ValueError, match="workflow run is eligible"):
        sync.run_pull_request(_API(pull=run), "owner/repo", 137)


def _lock_text(package: dict) -> str:
    artifact = package["wheels"][0]
    return (
        '[[package]]\nname = "example-package"\n'
        f'version = "{package["version"]}"\n'
        '[package.source]\nregistry = "https://pypi.org/simple"\n'
        f'[[package.wheels]]\nurl = "{artifact["url"]}"\nhash = "{artifact["hash"]}"\n'
    )


@pytest.mark.parametrize("changed_license", [False, True])
def test_pr_sync_publishes_only_after_real_distribution_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_license: bool
) -> None:
    root = tmp_path / "trusted-base"
    (root / "legal").mkdir(parents=True)
    output = tmp_path / "output"
    output.mkdir()
    baseline = _package(output / "distributions", "1.0")
    candidate = _package(
        output / "distributions",
        "1.1",
        license_text=b"Changed terms" if changed_license else b"The reviewed MIT license",
    )
    registry = _registry()
    trusted_registry = copy.deepcopy(registry)
    (root / "legal/dependency-licenses.json").write_text(json.dumps(registry), encoding="utf-8")
    (root / "uv.lock").write_text(_lock_text(baseline), encoding="utf-8")

    class DataAPI(_API):
        def get_json(self, path: str) -> dict | list:
            if "/contents/" in path:
                content = json.dumps(registry) if "dependency-licenses.json" in path else _lock_text(candidate)
                return {"encoding": "base64", "type": "file", "content": base64.b64encode(content.encode()).decode()}
            return super().get_json(path)

    def export(project: Path, lock: Path, destination: Path) -> None:
        assert project == root / "pyproject.toml"
        version = tomllib.loads(lock.read_text())["package"][0]["version"]
        destination.write_text(f"example-package=={version}\n", encoding="utf-8")

    monkeypatch.setattr(sync, "export_requirements", export)
    api = DataAPI()
    if changed_license:
        with pytest.raises(ValueError, match="License files or metadata changed"):
            sync.sync_pull_request(api, "owner/repo", 137, "base", root, output, "head")
        assert api.writes == []
    else:
        assert sync.sync_pull_request(api, "owner/repo", 137, "base", root, output, "head") == "commits-sha"
        addition = api.writes[0][1]["variables"]["input"]["fileChanges"]["additions"][0]
        written = json.loads(base64.b64decode(addition["contents"]))
        assert written["components"][0]["reviewed_versions"] == ["1.0", "1.1"]
        evidence = json.loads((output / "evidence.json").read_text())
        assert evidence[0]["candidate"][0]["sha256"] == candidate["wheels"][0]["hash"]
        assert api.writes[-1][0].endswith("/pr.yml/dispatches")
        registry = written
        api.pull["head"]["sha"] = "commits-sha"
        written_count = len(api.writes)
        assert sync.sync_pull_request(api, "owner/repo", 137, "base", root, output, "head") is None
        assert api.writes[written_count:] == [
            (
                "/repos/owner/repo/actions/workflows/pr.yml/dispatches",
                {"ref": "dependabot/uv/group", "inputs": {"pull-request": "137"}},
            )
        ]
    assert json.loads((root / "legal/dependency-licenses.json").read_text()) == trusted_registry
