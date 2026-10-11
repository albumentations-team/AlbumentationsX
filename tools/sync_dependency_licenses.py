"""Publish verified license registry updates to same-repository Dependabot pull requests."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path
from typing import Any, Protocol

from tools.release_bundle import REPOSITORY_PATTERN, GitHubAPI, GitHubClient, GitHubResponse
from tools.update_dependency_licenses import export_requirements, review_updates, write_registry
from tools.verify_dependency_licenses import load_registry

REGISTRY_PATH = "legal/dependency-licenses.json"


class RegistryAPI(GitHubAPI, Protocol):
    """GitHub reads, commit creation, compare-and-swap and workflow dispatch."""

    def post_json(self, path: str, payload: dict[str, Any]) -> GitHubResponse:
        """Create a GitHub resource."""


def _object(response: GitHubResponse) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise TypeError("GitHub endpoint did not return an object")
    return response


def _read_file(api: GitHubAPI, repository: str, path: str, commit: str) -> bytes:
    response = _object(api.get_json(f"/repos/{repository}/contents/{path}?ref={commit}"))
    if response.get("encoding") != "base64" or response.get("type") != "file":
        raise ValueError(f"GitHub did not return a regular base64 file: {path}")
    return base64.b64decode(response["content"], validate=False)


def run_pull_request(api: GitHubAPI, repository: str, run_id: int) -> tuple[int, str]:
    """Resolve only a same-repository Dependabot PR run; dispatches cannot recurse."""
    if REPOSITORY_PATTERN.fullmatch(repository) is None or run_id <= 0:
        raise ValueError("Invalid repository or workflow run ID")
    run = _object(api.get_json(f"/repos/{repository}/actions/runs/{run_id}"))
    if (
        run["event"] != "pull_request"
        or run["path"] != ".github/workflows/pr.yml"
        or run["actor"]["login"] != "dependabot[bot]"
        or run["head_repository"]["full_name"] != repository
        or len(run["pull_requests"]) != 1
    ):
        raise ValueError("Only an associated same-repository Dependabot PR workflow run is eligible")
    return run["pull_requests"][0]["number"], run["head_sha"]


def validate_pull_request(api: GitHubAPI, repository: str, number: int, base_sha: str) -> dict[str, Any]:
    """Reject fork, non-bot, closed, retargeted or expanded-scope pull requests."""
    if REPOSITORY_PATTERN.fullmatch(repository) is None or number <= 0:
        raise ValueError("Invalid repository or pull-request number")
    pull = _object(api.get_json(f"/repos/{repository}/pulls/{number}"))
    if (
        pull["state"] != "open"
        or pull["user"]["login"] != "dependabot[bot]"
        or pull["head"]["repo"]["full_name"] != repository
        or not pull["head"]["ref"].startswith("dependabot/uv/")
        or pull["base"]["sha"] != base_sha
        or pull["base"]["ref"] != "main"
    ):
        raise ValueError("Only an open same-repository Dependabot uv PR against the checked-out base is eligible")
    files = api.get_json(f"/repos/{repository}/pulls/{number}/files?per_page=100")
    if (
        not isinstance(files, list)
        or len(files) != pull["changed_files"]
        or not files
        or any(item["filename"] not in {"uv.lock", REGISTRY_PATH} or item["status"] != "modified" for item in files)
    ):
        raise ValueError("The dependency update must change only uv.lock and the license registry")
    return pull


def publish_registry(api: RegistryAPI, repository: str, pull: dict[str, Any], content: str) -> str:
    """Make one registry-only commit without overwriting a concurrent branch update."""
    prefix = f"/repos/{repository}"
    head = pull["head"]["sha"]
    current = _object(api.get_json(f"{prefix}/pulls/{pull['number']}"))
    if current["state"] != "open" or current["head"]["sha"] != head or current["base"]["sha"] != pull["base"]["sha"]:
        raise ValueError("Pull request changed during license review; refusing to publish")
    response = _object(
        api.post_json(
            "/graphql",
            {
                "query": (
                    "mutation($input: CreateCommitOnBranchInput!) {"
                    " createCommitOnBranch(input: $input) { commit { oid } } }"
                ),
                "variables": {
                    "input": {
                        "branch": {"repositoryNameWithOwner": repository, "branchName": pull["head"]["ref"]},
                        "expectedHeadOid": head,
                        "message": {"headline": "chore(deps): record unchanged dependency license evidence"},
                        "fileChanges": {
                            "additions": [
                                {"path": REGISTRY_PATH, "contents": base64.b64encode(content.encode()).decode()}
                            ],
                        },
                    },
                },
            },
        ),
    )
    if response.get("errors"):
        raise ValueError("Registry commit rejected: " + json.dumps(response["errors"]))
    commit = response["data"]["createCommitOnBranch"]["commit"]["oid"]
    dispatch_pr_checks(api, repository, pull)
    return commit


def dispatch_pr_checks(api: RegistryAPI, repository: str, pull: dict[str, Any]) -> None:
    """Run the ordinary checks on the current dependency-update branch."""
    api.post_json(
        f"/repos/{repository}/actions/workflows/pr.yml/dispatches",
        {"ref": pull["head"]["ref"], "inputs": {"pull-request": str(pull["number"])}},
    )


def sync_pull_request(
    api: RegistryAPI, repository: str, number: int, base_sha: str, root: Path, output: Path, expected_head: str
) -> str | None:
    """Review the untrusted lock as data, then publish only the computed registry."""
    pull = validate_pull_request(api, repository, number, base_sha)
    output.mkdir(parents=True, exist_ok=True)
    head_lock = output / "head-uv.lock"
    lock_bytes = _read_file(api, repository, "uv.lock", pull["head"]["sha"])
    retry = pull["head"]["sha"] != expected_head
    if retry and lock_bytes != _read_file(api, repository, "uv.lock", expected_head):
        raise ValueError("Workflow run is stale; the pull request lockfile has changed")
    head_lock.write_bytes(lock_bytes)
    with tempfile.TemporaryDirectory(prefix="dependency-license-sync-") as directory:
        base_requirements, head_requirements = (Path(directory) / name for name in ("base.txt", "head.txt"))
        export_requirements(root / "pyproject.toml", root / "uv.lock", base_requirements)
        export_requirements(root / "pyproject.toml", head_lock, head_requirements)
        registry, reports = review_updates(
            load_registry(root / REGISTRY_PATH),
            tomllib.loads((root / "uv.lock").read_text()),
            tomllib.loads(head_lock.read_text()),
            base_requirements,
            head_requirements,
            output / "distributions",
        )
    (output / "evidence.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    existing = json.loads(_read_file(api, repository, REGISTRY_PATH, pull["head"]["sha"]))
    if {key: value for key, value in existing.items() if key != "reviewed_on"} == {
        key: value for key, value in registry.items() if key != "reviewed_on"
    }:
        if retry:
            dispatch_pr_checks(api, repository, pull)
        return None
    if retry:
        raise ValueError("Stale workflow head does not contain the verified registry update")
    if existing != load_registry(root / REGISTRY_PATH):
        raise ValueError("PR registry differs from both the trusted baseline and the verified update")
    path = output / "dependency-licenses.json"
    write_registry(path, registry)
    return publish_registry(api, repository, pull, path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--root", default=Path.cwd(), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        client = GitHubClient(os.environ.get("GITHUB_TOKEN", ""))
        number, head = run_pull_request(client, args.repository, args.run_id)
        commit = sync_pull_request(client, args.repository, number, args.base_sha, args.root, args.output, head)
        print(f"Published registry update at {commit}." if commit else "License registry is already synchronized.")
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        subprocess.CalledProcessError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
