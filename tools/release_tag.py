"""Pin release tags to merged version-bump PRs with successful release bundles."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any, Protocol

from tools.release_bundle import (
    DEFAULT_WORKFLOW_PATH,
    REPOSITORY_PATTERN,
    BundleError,
    BundleNotFoundError,
    GitHubAPI,
    GitHubClient,
    GitHubResponse,
    _git,
    _is_merged_into,
    _run_has_required_status,
    release_metadata,
    resolve_artifact,
)


class TagAPI(GitHubAPI, Protocol):
    """GitHub reads and tag creation used by release automation."""

    def post_json(self, path: str, payload: dict[str, str]) -> GitHubResponse:
        """Create a GitHub resource."""


def event_pull_requests(api: GitHubAPI, repository: str, event: dict[str, Any]) -> list[int]:
    """Find the PR from a merge event or a successful PR workflow completion."""
    if REPOSITORY_PATTERN.fullmatch(repository) is None:
        msg = f"Invalid GitHub repository name: {repository!r}"
        raise BundleError(msg)
    if "pull_request" in event:
        return [event["pull_request"]["number"]]
    run = api.get_json(f"/repos/{repository}/actions/runs/{event['workflow_run']['id']}")
    if not isinstance(run, dict) or not _run_has_required_status(run, DEFAULT_WORKFLOW_PATH):
        return []
    pulls = run.get("pull_requests") or api.get_json(f"/repos/{repository}/commits/{run['head_sha']}/pulls")
    if not isinstance(pulls, list):
        msg = "GitHub PR associations are not a JSON array"
        raise BundleError(msg)
    return [pull["number"] for pull in pulls]


def _ensure_tag(api: TagAPI, repository: str, version: str, commit: str) -> None:
    ref_path = f"/repos/{repository}/git/ref/tags/{urllib.parse.quote(version, safe='')}"
    try:
        existing = api.get_json(ref_path)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        api.post_json(f"/repos/{repository}/git/refs", {"ref": f"refs/tags/{version}", "sha": commit})
        print(f"Created release tag {version} at {commit}")
        return
    if not isinstance(existing, dict):
        msg = "GitHub tag response is not a JSON object"
        raise BundleError(msg)
    target = existing["object"]
    while target["type"] == "tag":
        annotated = api.get_json(f"/repos/{repository}/git/tags/{target['sha']}")
        if not isinstance(annotated, dict):
            msg = "GitHub annotated tag response is not a JSON object"
            raise BundleError(msg)
        target = annotated["object"]
    if target["type"] != "commit" or target["sha"] != commit:
        msg = f"Release tag {version} already points to {target['sha']}; refusing to move it to {commit}"
        raise BundleError(msg)
    print(f"Release tag {version} already points to {commit}")


def tag_merged_release(
    api: TagAPI,
    *,
    repository: str,
    pull_number: int,
    default_branch: str,
    source_root: Path,
) -> str | None:
    """Create a version tag only when the merged tree has a verified PR bundle."""
    pull = api.get_json(f"/repos/{repository}/pulls/{pull_number}")
    if not isinstance(pull, dict) or not _is_merged_into(pull, default_branch):
        return None
    merge_commit = pull["merge_commit_sha"]
    base_commit = pull["base"]["sha"]
    if any(
        not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        for commit in (merge_commit, base_commit)
    ):
        msg = "Merged PR must identify full merge and base commit SHAs"
        raise BundleError(msg)
    _git(source_root, "merge-base", "--is-ancestor", merge_commit, f"origin/{default_branch}")
    base_project = tomllib.loads(_git(source_root, "show", f"{base_commit}:pyproject.toml").decode("utf-8"))
    merged_project = tomllib.loads(_git(source_root, "show", f"{merge_commit}:pyproject.toml").decode("utf-8"))
    if merged_project["project"]["version"] == base_project["project"]["version"]:
        return None
    _git(source_root, "checkout", "--detach", merge_commit)
    metadata = release_metadata(source_root)
    try:
        resolve_artifact(
            api,
            repository=repository,
            artifact_name=metadata.artifact_name,
            default_branch=default_branch,
        )
    except BundleNotFoundError:
        print(f"PR #{pull_number}: waiting for a successful release bundle for {metadata.version}")
        return None
    _ensure_tag(api, repository, metadata.version, metadata.source_commit)
    return metadata.version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--default-branch", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--event-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        api = GitHubClient(
            os.environ.get("GITHUB_TOKEN", ""), os.environ.get("GITHUB_API_URL", "https://api.github.com")
        )
        event = json.loads(args.event_path.read_text(encoding="utf-8"))
        for pull_number in event_pull_requests(api, args.repository, event):
            tag_merged_release(
                api,
                repository=args.repository,
                pull_number=pull_number,
                default_branch=args.default_branch,
                source_root=args.source_root,
            )
    except (BundleError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Release tag error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
