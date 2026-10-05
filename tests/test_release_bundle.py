"""Tests for immutable release-bundle creation and provenance resolution."""

from __future__ import annotations

import shutil
import subprocess
import urllib.error
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tools.release_bundle import (
    REQUIRED_CHECKS,
    BundleError,
    GitHubResponse,
    finalize_bundle,
    release_metadata,
    resolve_artifact,
    source_digest,
    verify_bundle,
)
from tools.release_tag import event_pull_requests, tag_merged_release


def _run_git(repository: Path, *args: str) -> None:
    git_executable = shutil.which("git")
    if git_executable is None:
        pytest.fail("git executable is required for release-bundle tests")
    subprocess.run([git_executable, *args], cwd=repository, check=True, capture_output=True)  # noqa: S603


def _repository(tmp_path: Path, *, version: str = "2.3.3") -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _run_git(repository, "init")
    _run_git(repository, "config", "user.email", "ci@example.com")
    _run_git(repository, "config", "user.name", "CI")
    (repository / "pyproject.toml").write_text(
        f'[project]\nname = "albumentationsx"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (repository / "source.py").write_text("VALUE = 137\n", encoding="utf-8")
    _run_git(repository, "add", "pyproject.toml", "source.py")
    _run_git(repository, "commit", "-m", "initial")
    return repository


def _bundle(repository: Path, *, version: str = "2.3.3") -> Path:
    bundle = repository.parent / "release-bundle"
    dist = bundle / "dist"
    evidence = bundle / "evidence"
    public = bundle / "public"
    dist.mkdir(parents=True)
    evidence.mkdir()
    public.mkdir()
    (dist / f"albumentationsx-{version}-py3-none-any.whl").write_bytes(b"wheel")
    (dist / f"albumentationsx-{version}.tar.gz").write_bytes(b"sdist")
    (evidence / "pytest-summary-release.json").write_text('{"status":"ok"}\n', encoding="utf-8")
    (public / f"albumentationsx-{version}-sbom.cdx.json").write_text("{}\n", encoding="utf-8")
    (public / f"albumentationsx-{version}-correctness-compatibility-report.md").write_text(
        "# Report\n",
        encoding="utf-8",
    )
    return bundle


def test_source_digest_is_independent_of_commit_identity(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    first_digest = source_digest(repository)

    _run_git(repository, "commit", "--allow-empty", "-m", "same tree")

    assert source_digest(repository) == first_digest


def test_source_digest_changes_with_tracked_content(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    first_digest = source_digest(repository)
    (repository / "source.py").write_text("VALUE = 138\n", encoding="utf-8")
    _run_git(repository, "add", "source.py")
    _run_git(repository, "commit", "-m", "change source")

    assert source_digest(repository) != first_digest


def test_release_metadata_binds_version_and_source_digest(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    metadata = release_metadata(repository, expected_tag="v2.3.3")

    assert metadata.version == "2.3.3"
    assert metadata.source_digest.startswith("sha256:")
    assert metadata.artifact_name == f"release-bundle-2.3.3-{metadata.source_digest.removeprefix('sha256:')}"


def test_release_metadata_rejects_tag_version_mismatch(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    with pytest.raises(BundleError, match=r"does not match project\.version"):
        release_metadata(repository, expected_tag="2.3.4")


def test_finalize_and_verify_bundle(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    bundle = _bundle(repository)
    now = datetime(2026, 7, 15, 12, tzinfo=UTC)

    manifest = finalize_bundle(
        bundle,
        repository,
        checks=REQUIRED_CHECKS,
        now=now,
        retention_days=90,
    )
    verified = verify_bundle(
        bundle,
        expected_version="2.3.3",
        expected_source_digest=source_digest(repository),
        now=now + timedelta(days=1),
    )

    assert verified == manifest
    assert manifest["status"] == "ok"
    assert manifest["checks"] == dict.fromkeys(sorted(REQUIRED_CHECKS), "ok")
    assert set(manifest["publish_files"]) == {
        "dist/albumentationsx-2.3.3-py3-none-any.whl",
        "dist/albumentationsx-2.3.3.tar.gz",
        "public/SHA256SUMS.txt",
        "public/albumentationsx-2.3.3-correctness-compatibility-report.md",
        "public/albumentationsx-2.3.3-sbom.cdx.json",
    }


def test_verify_bundle_rejects_payload_tampering(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    bundle = _bundle(repository)
    finalize_bundle(bundle, repository, checks=REQUIRED_CHECKS)
    wheel = next((bundle / "dist").glob("*.whl"))
    wheel.write_bytes(b"different wheel")

    with pytest.raises(BundleError, match="digest mismatch"):
        verify_bundle(
            bundle,
            expected_version="2.3.3",
            expected_source_digest=source_digest(repository),
        )


def test_verify_bundle_accepts_known_legacy_upload_artifact_omission(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    bundle = _bundle(repository)
    gitignore = bundle / "dist" / ".gitignore"
    gitignore.write_text("*", encoding="utf-8")
    manifest = finalize_bundle(bundle, repository, checks=REQUIRED_CHECKS)
    gitignore.unlink()

    verified = verify_bundle(
        bundle,
        expected_version="2.3.3",
        expected_source_digest=source_digest(repository),
    )

    assert verified == manifest


def test_verify_bundle_rejects_legacy_omission_with_unexpected_digest(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    bundle = _bundle(repository)
    gitignore = bundle / "dist" / ".gitignore"
    gitignore.write_text("unexpected", encoding="utf-8")
    finalize_bundle(bundle, repository, checks=REQUIRED_CHECKS)
    gitignore.unlink()

    with pytest.raises(BundleError, match=r"missing or is not regular: dist/\.gitignore"):
        verify_bundle(
            bundle,
            expected_version="2.3.3",
            expected_source_digest=source_digest(repository),
        )


def test_verify_bundle_rejects_legacy_omission_for_another_version(tmp_path: Path) -> None:
    repository = _repository(tmp_path, version="2.3.4")
    bundle = _bundle(repository, version="2.3.4")
    gitignore = bundle / "dist" / ".gitignore"
    gitignore.write_text("*", encoding="utf-8")
    finalize_bundle(bundle, repository, checks=REQUIRED_CHECKS)
    gitignore.unlink()

    with pytest.raises(BundleError, match=r"missing or is not regular: dist/\.gitignore"):
        verify_bundle(
            bundle,
            expected_version="2.3.4",
            expected_source_digest=source_digest(repository),
        )


def test_verify_bundle_rejects_expired_manifest(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    bundle = _bundle(repository)
    now = datetime(2026, 7, 15, 12, tzinfo=UTC)
    finalize_bundle(bundle, repository, checks=REQUIRED_CHECKS, now=now, retention_days=1)

    with pytest.raises(BundleError, match="expired"):
        verify_bundle(
            bundle,
            expected_version="2.3.3",
            expected_source_digest=source_digest(repository),
            now=now + timedelta(days=2),
        )


def test_finalize_bundle_requires_complete_check_set(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    bundle = _bundle(repository)

    with pytest.raises(BundleError, match="Missing successful release check"):
        finalize_bundle(bundle, repository, checks=REQUIRED_CHECKS - {"performance_core"})


class FakeGitHubAPI:
    """Fixture-backed GitHub API for artifact provenance tests."""

    def __init__(self, responses: dict[str, dict[str, Any] | list[dict[str, Any]]]) -> None:
        self.responses = responses

    def get_json(self, path: str) -> dict[str, Any] | list[dict[str, Any]]:
        return self.responses[path]


class FakeTagAPI(FakeGitHubAPI):
    """Record tag writes and emulate missing GitHub references."""

    def __init__(self, responses: dict[str, GitHubResponse]) -> None:
        super().__init__(responses)
        self.writes: list[tuple[str, dict[str, str]]] = []

    def get_json(self, path: str) -> GitHubResponse:
        if "/git/ref/tags/" in path and path not in self.responses:
            raise urllib.error.HTTPError(path, 404, "Not Found", None, None)
        return super().get_json(path)

    def post_json(self, path: str, payload: dict[str, str]) -> GitHubResponse:
        self.writes.append((path, payload))
        response = {"object": {"type": "commit", "sha": payload["sha"]}}
        self.responses[path.removesuffix("refs") + "ref/" + payload["ref"].removeprefix("refs/")] = response
        return response


def _artifact_responses(
    *,
    merged_at: str | None = "2026-07-15T12:00:00Z",
) -> dict[str, dict[str, Any] | list[dict[str, Any]]]:
    artifact_name = "release-bundle-2.3.3-digest"
    return {
        "/repos/albumentations-team/AlbumentationsX/actions/artifacts?name=" + artifact_name + "&per_page=100": {
            "artifacts": [
                {
                    "id": 137,
                    "name": artifact_name,
                    "expired": False,
                    "created_at": "2026-07-15T11:00:00Z",
                    "workflow_run": {"id": 173},
                },
            ],
        },
        "/repos/albumentations-team/AlbumentationsX/actions/runs/173": {
            "id": 173,
            "event": "pull_request",
            "status": "completed",
            "conclusion": "success",
            "path": ".github/workflows/pr.yml",
            "pull_requests": [{"number": 305, "base": {"ref": "main"}}],
        },
        "/repos/albumentations-team/AlbumentationsX/pulls/305": {
            "number": 305,
            "merged_at": merged_at,
            "base": {"ref": "main"},
        },
    }


def _head_commit_fallback_responses(
    *,
    merged_at: str | None = "2026-07-15T12:00:00Z",
    base_ref: str = "main",
) -> dict[str, dict[str, Any] | list[dict[str, Any]]]:
    responses = _artifact_responses()
    run_path = "/repos/albumentations-team/AlbumentationsX/actions/runs/173"
    run = responses[run_path]
    assert isinstance(run, dict)
    run["head_sha"] = "abc137"
    run["pull_requests"] = []
    responses["/repos/albumentations-team/AlbumentationsX/commits/abc137/pulls"] = [
        {
            "number": 305,
            "merged_at": merged_at,
            "base": {"ref": base_ref},
        },
    ]
    return responses


def test_resolve_artifact_requires_successful_merged_pr_run() -> None:
    resolved = resolve_artifact(
        FakeGitHubAPI(_artifact_responses()),
        repository="albumentations-team/AlbumentationsX",
        artifact_name="release-bundle-2.3.3-digest",
        workflow_path=".github/workflows/pr.yml",
        default_branch="main",
    )

    assert resolved.artifact_id == 137
    assert resolved.run_id == 173


def test_resolve_artifact_uses_head_commit_when_run_omits_pull_requests() -> None:
    resolved = resolve_artifact(
        FakeGitHubAPI(_head_commit_fallback_responses()),
        repository="albumentations-team/AlbumentationsX",
        artifact_name="release-bundle-2.3.3-digest",
        workflow_path=".github/workflows/pr.yml",
        default_branch="main",
    )

    assert resolved.artifact_id == 137
    assert resolved.run_id == 173


@pytest.mark.parametrize(
    ("merged_at", "base_ref"),
    [
        (None, "main"),
        ("2026-07-15T12:00:00Z", "release"),
    ],
)
def test_resolve_artifact_rejects_head_commit_without_merged_default_branch_pr(
    merged_at: str | None,
    base_ref: str,
) -> None:
    with pytest.raises(BundleError, match="No unexpired release bundle"):
        resolve_artifact(
            FakeGitHubAPI(_head_commit_fallback_responses(merged_at=merged_at, base_ref=base_ref)),
            repository="albumentations-team/AlbumentationsX",
            artifact_name="release-bundle-2.3.3-digest",
            workflow_path=".github/workflows/pr.yml",
            default_branch="main",
        )


def test_resolve_artifact_rejects_unmerged_pr_run() -> None:
    with pytest.raises(BundleError, match="No unexpired release bundle"):
        resolve_artifact(
            FakeGitHubAPI(_artifact_responses(merged_at=None)),
            repository="albumentations-team/AlbumentationsX",
            artifact_name="release-bundle-2.3.3-digest",
            workflow_path=".github/workflows/pr.yml",
            default_branch="main",
        )


def test_resolve_artifact_does_not_replace_reported_prs_with_head_commit_associations() -> None:
    responses = _artifact_responses(merged_at=None)
    run_path = "/repos/albumentations-team/AlbumentationsX/actions/runs/173"
    run = responses[run_path]
    assert isinstance(run, dict)
    run["head_sha"] = "abc137"
    responses["/repos/albumentations-team/AlbumentationsX/commits/abc137/pulls"] = [
        {
            "number": 306,
            "merged_at": "2026-07-15T12:00:00Z",
            "base": {"ref": "main"},
        },
    ]

    with pytest.raises(BundleError, match="No unexpired release bundle"):
        resolve_artifact(
            FakeGitHubAPI(responses),
            repository="albumentations-team/AlbumentationsX",
            artifact_name="release-bundle-2.3.3-digest",
            workflow_path=".github/workflows/pr.yml",
            default_branch="main",
        )


@pytest.fixture
def merged_release(tmp_path: Path) -> tuple[Path, FakeTagAPI, str]:
    repository = _repository(tmp_path, version="2.3.2")
    base_commit = release_metadata(repository).source_commit
    project = repository / "pyproject.toml"
    project.write_text(project.read_text().replace('"2.3.2"', '"2.3.3"'), encoding="utf-8")
    _run_git(repository, "commit", "-am", "bump version")
    metadata = release_metadata(repository)
    responses = _artifact_responses()
    artifacts = responses.pop(
        "/repos/albumentations-team/AlbumentationsX/actions/artifacts?name=release-bundle-2.3.3-digest&per_page=100",
    )
    assert isinstance(artifacts, dict)
    artifacts["artifacts"][0]["name"] = metadata.artifact_name
    responses[
        f"/repos/albumentations-team/AlbumentationsX/actions/artifacts?name={metadata.artifact_name}&per_page=100"
    ] = artifacts
    responses["/repos/albumentations-team/AlbumentationsX/pulls/305"] = {
        "number": 305,
        "merged_at": "2026-07-15T12:00:00Z",
        "merge_commit_sha": metadata.source_commit,
        "base": {"ref": "main", "sha": base_commit},
    }
    (repository / "source.py").write_text("VALUE = 138\n", encoding="utf-8")
    _run_git(repository, "commit", "-am", "later main change")
    _run_git(repository, "update-ref", "refs/remotes/origin/main", "HEAD")
    return repository, FakeTagAPI(responses), metadata.source_commit


def _tag_release(repository: Path, api: FakeTagAPI) -> str | None:
    return tag_merged_release(
        api,
        repository="albumentations-team/AlbumentationsX",
        pull_number=305,
        default_branch="main",
        source_root=repository,
    )


def test_release_tag_pins_verified_merge_after_main_advances(merged_release: tuple[Path, FakeTagAPI, str]) -> None:
    repository, api, merge_commit = merged_release
    assert release_metadata(repository).source_commit != merge_commit

    assert _tag_release(repository, api) == "2.3.3"
    assert api.writes == [
        ("/repos/albumentations-team/AlbumentationsX/git/refs", {"ref": "refs/tags/2.3.3", "sha": merge_commit}),
    ]
    assert _tag_release(repository, api) == "2.3.3"
    assert len(api.writes) == 1


@pytest.mark.parametrize("target_type", ["commit", "tag"])
def test_release_tag_preserves_existing_matching_tag(
    merged_release: tuple[Path, FakeTagAPI, str],
    target_type: str,
) -> None:
    repository, api, merge_commit = merged_release
    api.responses["/repos/albumentations-team/AlbumentationsX/git/ref/tags/2.3.3"] = {
        "object": {"type": target_type, "sha": merge_commit},
    }
    if target_type == "tag":
        api.responses[f"/repos/albumentations-team/AlbumentationsX/git/tags/{merge_commit}"] = {
            "object": {"type": "commit", "sha": merge_commit},
        }

    assert _tag_release(repository, api) == "2.3.3"
    assert api.writes == []


def test_release_tag_refuses_to_move_conflicting_tag(merged_release: tuple[Path, FakeTagAPI, str]) -> None:
    repository, api, _ = merged_release
    api.responses["/repos/albumentations-team/AlbumentationsX/git/ref/tags/2.3.3"] = {
        "object": {"type": "commit", "sha": "different-commit"},
    }

    with pytest.raises(BundleError, match="refusing to move"):
        _tag_release(repository, api)
    assert api.writes == []


@pytest.mark.parametrize(
    "blocker",
    ["unmerged", "other-branch", "unchanged-version", "failed-run", "missing-bundle", "expired-bundle"],
)
def test_release_tag_requires_version_bump_and_successful_merged_bundle(
    merged_release: tuple[Path, FakeTagAPI, str],
    blocker: str,
) -> None:
    repository, api, merge_commit = merged_release
    pull = api.responses["/repos/albumentations-team/AlbumentationsX/pulls/305"]
    assert isinstance(pull, dict)
    if blocker == "unmerged":
        pull["merged_at"] = None
    elif blocker == "other-branch":
        pull["base"]["ref"] = "release"
    elif blocker == "unchanged-version":
        pull["base"]["sha"] = merge_commit
    elif blocker == "failed-run":
        run = api.responses["/repos/albumentations-team/AlbumentationsX/actions/runs/173"]
        assert isinstance(run, dict)
        run["conclusion"] = "failure"
    else:
        for path, response in api.responses.items():
            if "/actions/artifacts?" in path:
                assert isinstance(response, dict)
                if blocker == "missing-bundle":
                    response["artifacts"] = []
                else:
                    response["artifacts"][0]["expired"] = True

    assert _tag_release(repository, api) is None
    assert api.writes == []


@pytest.mark.parametrize("missing_associations", [False, True])
def test_release_tag_finds_pr_after_workflow_completion(missing_associations: bool) -> None:
    responses = _head_commit_fallback_responses() if missing_associations else _artifact_responses()

    assert event_pull_requests(
        FakeGitHubAPI(responses),
        "albumentations-team/AlbumentationsX",
        {"workflow_run": {"id": 173}},
    ) == [305]


def test_release_tag_finds_pr_from_merge_event() -> None:
    assert event_pull_requests(
        FakeGitHubAPI({}), "albumentations-team/AlbumentationsX", {"pull_request": {"number": 305}}
    ) == [305]


@pytest.mark.parametrize(("field", "value"), [("conclusion", "failure"), ("event", "push"), ("path", "other.yml")])
def test_release_tag_ignores_other_workflow_runs(field: str, value: str) -> None:
    responses = _artifact_responses()
    run = responses["/repos/albumentations-team/AlbumentationsX/actions/runs/173"]
    assert isinstance(run, dict)
    run[field] = value

    assert (
        event_pull_requests(
            FakeGitHubAPI(responses), "albumentations-team/AlbumentationsX", {"workflow_run": {"id": 173}}
        )
        == []
    )
