"""Create, verify, and resolve immutable AlbumentationsX release bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

SCHEMA_VERSION = 1
MANIFEST_NAME = "release-manifest.json"
DEFAULT_RETENTION_DAYS = 90
DEFAULT_WORKFLOW_PATH = ".github/workflows/pr.yml"
REQUIRED_CHECKS = frozenset(
    {
        "clean_install",
        "correctness",
        "legal",
        "lock",
        "package",
        "performance_core",
        "report",
        "security",
    },
)
MANIFEST_KEYS = {
    "artifact_name",
    "artifacts",
    "checks",
    "expires_at",
    "generated_at",
    "publish_files",
    "release_input_digest",
    "schema_version",
    "source_commit",
    "status",
    "version",
}
REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
GitHubResponse = dict[str, Any] | list[dict[str, Any]]


class BundleError(ValueError):
    """Raised when release-bundle identity or integrity validation fails."""


@dataclass(frozen=True)
class ReleaseMetadata:
    """Source identity used to name and validate one release bundle."""

    version: str
    source_digest: str
    source_commit: str
    artifact_name: str


@dataclass(frozen=True)
class ResolvedArtifact:
    """GitHub artifact and workflow run selected for release delivery."""

    artifact_id: int
    run_id: int
    artifact_name: str


class GitHubAPI(Protocol):
    """Minimal interface required by the provenance resolver."""

    def get_json(self, path: str) -> GitHubResponse:
        """Return one GitHub REST response as a JSON object or array."""


class GitHubClient:
    """Small authenticated GitHub REST client for release artifact lookup."""

    def __init__(self, token: str, api_url: str = "https://api.github.com") -> None:
        if not token:
            msg = "GITHUB_TOKEN or GH_TOKEN is required to resolve a release bundle"
            raise BundleError(msg)
        self.token = token
        self.api_url = api_url.rstrip("/")

    def get_json(self, path: str) -> GitHubResponse:
        """Fetch a GitHub REST endpoint without exposing the token in arguments or logs."""
        request = urllib.request.Request(
            f"{self.api_url}{path}",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "User-Agent": "albumentationsx-release-bundle",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
        if not isinstance(data, (dict, list)):
            msg = f"GitHub API endpoint {path} did not return a JSON object or array"
            raise BundleError(msg)
        return data


def _git(repository: Path, *args: str) -> bytes:
    git_executable = shutil.which("git")
    if git_executable is None:
        msg = "git executable is required for release source identity"
        raise BundleError(msg)
    result = subprocess.run(  # noqa: S603
        [git_executable, *args],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        msg = f"git {' '.join(args)} failed: {stderr or f'exit code {result.returncode}'}"
        raise BundleError(msg)
    return result.stdout


def _ensure_clean_tracked_tree(repository: Path) -> None:
    git_executable = shutil.which("git")
    if git_executable is None:
        msg = "git executable is required for release source identity"
        raise BundleError(msg)
    for args in (("diff", "--quiet", "--"), ("diff", "--cached", "--quiet", "--")):
        result = subprocess.run(  # noqa: S603
            [git_executable, *args],
            cwd=repository,
            check=False,
            capture_output=True,
        )
        if result.returncode == 1:
            msg = "Tracked repository content changed after checkout; refusing to compute a release source digest"
            raise BundleError(msg)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            msg = f"git {' '.join(args)} failed: {stderr or f'exit code {result.returncode}'}"
            raise BundleError(msg)


def _tracked_records(repository: Path) -> list[tuple[bytes, bytes]]:
    records: list[tuple[bytes, bytes]] = []
    for record in _git(repository, "ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        if not separator:
            msg = "git ls-files returned a malformed tracked-file record"
            raise BundleError(msg)
        fields = metadata.split()
        if len(fields) != 3 or fields[2] != b"0":
            msg = "Release source digest requires an index without unmerged entries"
            raise BundleError(msg)
        mode = fields[0]
        if mode not in {b"100644", b"100755"}:
            decoded_path = os.fsdecode(raw_path)
            msg = f"Unsupported tracked file mode {mode.decode()} for {decoded_path}"
            raise BundleError(msg)
        records.append((raw_path, mode))
    if not records:
        msg = "Release source digest requires at least one tracked file"
        raise BundleError(msg)
    return sorted(records)


def source_digest(repository: Path) -> str:
    """Return a commit-independent SHA-256 digest of the clean tracked tree."""
    repository = repository.resolve()
    _ensure_clean_tracked_tree(repository)
    digest = hashlib.sha256()
    for raw_path, mode in _tracked_records(repository):
        path = repository / os.fsdecode(raw_path)
        if path.is_symlink() or not path.is_file():
            msg = f"Tracked release input must be a regular file: {os.fsdecode(raw_path)}"
            raise BundleError(msg)
        file_size = path.stat().st_size
        digest.update(mode)
        digest.update(b"\0")
        digest.update(len(raw_path).to_bytes(8, "big"))
        digest.update(raw_path)
        digest.update(file_size.to_bytes(8, "big"))
        with path.open("rb") as tracked_file:
            while chunk := tracked_file.read(1024 * 1024):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _project_version(repository: Path) -> str:
    with (repository / "pyproject.toml").open("rb") as pyproject_file:
        data = tomllib.load(pyproject_file)
    project = data.get("project")
    if not isinstance(project, dict):
        msg = "pyproject.toml is missing a [project] table"
        raise BundleError(msg)
    version = project.get("version")
    if not isinstance(version, str) or not version:
        msg = "pyproject.toml is missing a non-empty project.version"
        raise BundleError(msg)
    return version


def release_metadata(repository: Path, expected_tag: str | None = None) -> ReleaseMetadata:
    """Return release identity for a clean repository and optionally validate its tag."""
    repository = repository.resolve()
    version = _project_version(repository)
    if expected_tag is not None and expected_tag.removeprefix("v") != version:
        msg = f"Release tag {expected_tag!r} does not match project.version {version!r}"
        raise BundleError(msg)
    digest = source_digest(repository)
    digest_hex = digest.removeprefix("sha256:")
    source_commit = _git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    return ReleaseMetadata(
        version=version,
        source_digest=digest,
        source_commit=source_commit,
        artifact_name=f"release-bundle-{version}-{digest_hex}",
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        while chunk := artifact_file.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _single_file(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1 or not matches[0].is_file() or matches[0].is_symlink():
        msg = f"Release bundle requires exactly one regular {label}; found {len(matches)}"
        raise BundleError(msg)
    return matches[0]


def _required_publish_files(bundle_dir: Path, version: str) -> tuple[Path, ...]:
    dist_dir = bundle_dir / "dist"
    public_dir = bundle_dir / "public"
    evidence_dir = bundle_dir / "evidence"
    wheel = _single_file(dist_dir, "*.whl", "wheel")
    sdist = _single_file(dist_dir, "*.tar.gz", "source distribution")
    sbom = public_dir / f"albumentationsx-{version}-sbom.cdx.json"
    report = public_dir / f"albumentationsx-{version}-correctness-compatibility-report.md"
    for path, label in ((sbom, "SBOM"), (report, "correctness report")):
        if not path.is_file() or path.is_symlink():
            msg = f"Release bundle is missing its regular {label}: {path.relative_to(bundle_dir)}"
            raise BundleError(msg)
    evidence_files = [path for path in evidence_dir.rglob("*") if path.is_file() and not path.is_symlink()]
    if not evidence_files:
        msg = "Release bundle requires diagnostic evidence"
        raise BundleError(msg)
    checksums = public_dir / "SHA256SUMS.txt"
    return wheel, sdist, sbom, report, checksums


def _checksum_text(paths: tuple[Path, ...]) -> str:
    basenames = [path.name for path in paths]
    if len(basenames) != len(set(basenames)):
        msg = "Published release files must have unique basenames"
        raise BundleError(msg)
    return "".join(f"{_hash_file(path).removeprefix('sha256:')}  {path.name}\n" for path in paths)


def _payload_files(bundle_dir: Path) -> dict[str, Path]:
    payload: dict[str, Path] = {}
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_symlink():
            msg = f"Release bundle must not contain symlinks: {path.relative_to(bundle_dir)}"
            raise BundleError(msg)
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        relative = path.relative_to(bundle_dir).as_posix()
        payload[relative] = path
    return payload


def _utc_now(value: datetime | None) -> datetime:
    resolved = value or datetime.now(tz=timezone.utc)
    if resolved.tzinfo is None:
        msg = "Release bundle timestamps must be timezone-aware"
        raise BundleError(msg)
    return resolved.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def finalize_bundle(
    bundle_dir: Path,
    repository: Path,
    *,
    checks: frozenset[str] | set[str],
    now: datetime | None = None,
    retention_days: int = DEFAULT_RETENTION_DAYS,
) -> dict[str, Any]:
    """Create checksums and the manifest after every release preflight check succeeds."""
    missing_checks = REQUIRED_CHECKS - set(checks)
    unknown_checks = set(checks) - REQUIRED_CHECKS
    if missing_checks:
        msg = "Missing successful release check(s): " + ", ".join(sorted(missing_checks))
        raise BundleError(msg)
    if unknown_checks:
        msg = "Unknown release check(s): " + ", ".join(sorted(unknown_checks))
        raise BundleError(msg)
    if retention_days <= 0:
        msg = "retention_days must be positive"
        raise BundleError(msg)

    bundle_dir = bundle_dir.resolve()
    repository = repository.resolve()
    metadata = release_metadata(repository)
    manifest_path = bundle_dir / MANIFEST_NAME
    manifest_path.unlink(missing_ok=True)
    wheel, sdist, sbom, report, checksums = _required_publish_files(bundle_dir, metadata.version)
    checksums.parent.mkdir(parents=True, exist_ok=True)
    checksums.write_text(_checksum_text((wheel, sdist, sbom, report)), encoding="utf-8")
    publish_paths = (wheel, sdist, sbom, report, checksums)
    payload = _payload_files(bundle_dir)
    generated_at = _utc_now(now)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "version": metadata.version,
        "status": "ok",
        "source_commit": metadata.source_commit,
        "release_input_digest": metadata.source_digest,
        "artifact_name": metadata.artifact_name,
        "generated_at": _format_timestamp(generated_at),
        "expires_at": _format_timestamp(generated_at + timedelta(days=retention_days)),
        "checks": dict.fromkeys(sorted(REQUIRED_CHECKS), "ok"),
        "publish_files": sorted(path.relative_to(bundle_dir).as_posix() for path in publish_paths),
        "artifacts": {relative: _hash_file(path) for relative, path in sorted(payload.items())},
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest


def _parse_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        msg = f"Release manifest {label} must be an ISO-8601 string"
        raise BundleError(msg)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        msg = f"Release manifest {label} is not a valid ISO-8601 timestamp"
        raise BundleError(msg) from error
    if parsed.tzinfo is None:
        msg = f"Release manifest {label} must be timezone-aware"
        raise BundleError(msg)
    return parsed.astimezone(timezone.utc)


def _validated_artifacts(bundle_dir: Path, artifacts: Any) -> dict[str, str]:
    if not isinstance(artifacts, dict) or not artifacts:
        msg = "Release manifest artifacts must be a non-empty mapping"
        raise BundleError(msg)
    validated: dict[str, str] = {}
    for raw_path, raw_digest in artifacts.items():
        if not isinstance(raw_path, str) or not isinstance(raw_digest, str):
            msg = "Release manifest artifact paths and digests must be strings"
            raise BundleError(msg)
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts or raw_path != relative.as_posix():
            msg = f"Release manifest contains an unsafe artifact path: {raw_path!r}"
            raise BundleError(msg)
        path = bundle_dir.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            msg = f"Release manifest artifact is missing or is not regular: {raw_path}"
            raise BundleError(msg)
        actual_digest = _hash_file(path)
        if actual_digest != raw_digest:
            msg = f"Release artifact digest mismatch for {raw_path}: {actual_digest} != {raw_digest}"
            raise BundleError(msg)
        validated[raw_path] = raw_digest
    return validated


def _read_manifest(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        msg = f"Could not read release manifest: {error}"
        raise BundleError(msg) from error
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        msg = "Release manifest fields do not match schema version 1"
        raise BundleError(msg)
    return manifest


def _validate_manifest_freshness(manifest: dict[str, Any], now: datetime | None) -> None:
    generated_at = _parse_timestamp(manifest["generated_at"], "generated_at")
    expires_at = _parse_timestamp(manifest["expires_at"], "expires_at")
    if expires_at <= generated_at:
        msg = "Release manifest expires_at must be later than generated_at"
        raise BundleError(msg)
    if _utc_now(now) > expires_at:
        msg = f"Release bundle expired at {_format_timestamp(expires_at)}"
        raise BundleError(msg)


def _validate_manifest_identity(
    manifest: dict[str, Any],
    *,
    expected_version: str,
    expected_source_digest: str,
    now: datetime | None,
) -> None:
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["status"] != "ok":
        msg = "Release manifest is not a successful schema version 1 bundle"
        raise BundleError(msg)
    if manifest["version"] != expected_version:
        msg = f"Release manifest version {manifest['version']!r} does not match {expected_version!r}"
        raise BundleError(msg)
    if manifest["release_input_digest"] != expected_source_digest:
        msg = "Release manifest source digest does not match the released tag"
        raise BundleError(msg)
    expected_artifact_name = f"release-bundle-{expected_version}-{expected_source_digest.removeprefix('sha256:')}"
    if manifest["artifact_name"] != expected_artifact_name:
        msg = "Release manifest artifact name does not match its version and source digest"
        raise BundleError(msg)
    if manifest["checks"] != dict.fromkeys(sorted(REQUIRED_CHECKS), "ok"):
        msg = "Release manifest does not record every required check as successful"
        raise BundleError(msg)
    _validate_manifest_freshness(manifest, now)


def _validate_manifest_payload(bundle_dir: Path, manifest: dict[str, Any], version: str) -> None:
    validated_artifacts = _validated_artifacts(bundle_dir, manifest["artifacts"])
    actual_payload = set(_payload_files(bundle_dir))
    if actual_payload != set(validated_artifacts):
        msg = "Release bundle payload does not exactly match the manifest"
        raise BundleError(msg)
    wheel, sdist, sbom, report, checksums = _required_publish_files(bundle_dir, version)
    expected_publish_files = sorted(
        path.relative_to(bundle_dir).as_posix() for path in (wheel, sdist, sbom, report, checksums)
    )
    if manifest["publish_files"] != expected_publish_files:
        msg = "Release manifest publish_files do not match the required release artifacts"
        raise BundleError(msg)
    if checksums.read_text(encoding="utf-8") != _checksum_text((wheel, sdist, sbom, report)):
        msg = "Release checksum manifest does not match the publishable files"
        raise BundleError(msg)


def verify_bundle(
    bundle_dir: Path,
    *,
    expected_version: str,
    expected_source_digest: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify that a downloaded bundle is complete, current, and byte-identical to its manifest."""
    bundle_dir = bundle_dir.resolve()
    manifest = _read_manifest(bundle_dir)
    _validate_manifest_identity(
        manifest,
        expected_version=expected_version,
        expected_source_digest=expected_source_digest,
        now=now,
    )
    _validate_manifest_payload(bundle_dir, manifest, expected_version)
    return manifest


def _is_merged_into(pull_request: dict[str, Any], default_branch: str) -> bool:
    pull_base = pull_request.get("base", {})
    return (
        pull_request.get("merged_at") is not None
        and isinstance(pull_base, dict)
        and pull_base.get("ref") == default_branch
    )


def _run_has_required_status(run: dict[str, Any], workflow_path: str) -> bool:
    return (
        run.get("event") == "pull_request"
        and run.get("status") == "completed"
        and run.get("conclusion") == "success"
        and run.get("path") == workflow_path
    )


def _summary_identifies_merged_pr(
    api: GitHubAPI,
    repository: str,
    summary: Any,
    default_branch: str,
) -> bool:
    if not isinstance(summary, dict):
        return False
    base = summary.get("base", {})
    number = summary.get("number")
    if not isinstance(base, dict) or base.get("ref") != default_branch or not isinstance(number, int):
        return False
    pull_request = api.get_json(f"/repos/{repository}/pulls/{number}")
    return isinstance(pull_request, dict) and _is_merged_into(pull_request, default_branch)


def _head_commit_identifies_merged_pr(
    api: GitHubAPI,
    repository: str,
    head_sha: Any,
    default_branch: str,
) -> bool:
    if not isinstance(head_sha, str) or not head_sha:
        return False
    pull_requests = api.get_json(f"/repos/{repository}/commits/{head_sha}/pulls")
    return isinstance(pull_requests, list) and any(
        isinstance(pull_request, dict) and _is_merged_into(pull_request, default_branch)
        for pull_request in pull_requests
    )


def _run_matches_release_policy(
    api: GitHubAPI,
    repository: str,
    run_id: int,
    *,
    workflow_path: str,
    default_branch: str,
) -> bool:
    run = api.get_json(f"/repos/{repository}/actions/runs/{run_id}")
    if not isinstance(run, dict) or not _run_has_required_status(run, workflow_path):
        return False
    pull_requests = run.get("pull_requests", [])
    if isinstance(pull_requests, list) and any(
        _summary_identifies_merged_pr(api, repository, summary, default_branch) for summary in pull_requests
    ):
        return True
    return _head_commit_identifies_merged_pr(api, repository, run.get("head_sha"), default_branch)


def resolve_artifact(
    api: GitHubAPI,
    *,
    repository: str,
    artifact_name: str,
    workflow_path: str = DEFAULT_WORKFLOW_PATH,
    default_branch: str,
) -> ResolvedArtifact:
    """Resolve the newest unexpired bundle from a successful merged PR run."""
    if REPOSITORY_PATTERN.fullmatch(repository) is None:
        msg = f"Invalid GitHub repository name: {repository!r}"
        raise BundleError(msg)
    encoded_name = urllib.parse.quote(artifact_name, safe="")
    response = api.get_json(f"/repos/{repository}/actions/artifacts?name={encoded_name}&per_page=100")
    if not isinstance(response, dict):
        msg = "GitHub artifacts response is not a JSON object"
        raise BundleError(msg)
    artifacts = response.get("artifacts", [])
    if not isinstance(artifacts, list):
        msg = "GitHub artifacts response does not contain an artifacts array"
        raise BundleError(msg)
    candidates = sorted(
        (
            artifact
            for artifact in artifacts
            if isinstance(artifact, dict) and artifact.get("name") == artifact_name and artifact.get("expired") is False
        ),
        key=lambda artifact: str(artifact.get("created_at", "")),
        reverse=True,
    )
    for artifact in candidates:
        artifact_id = artifact.get("id")
        workflow_run = artifact.get("workflow_run", {})
        run_id = workflow_run.get("id") if isinstance(workflow_run, dict) else None
        if not isinstance(artifact_id, int) or not isinstance(run_id, int):
            continue
        if _run_matches_release_policy(
            api,
            repository,
            run_id,
            workflow_path=workflow_path,
            default_branch=default_branch,
        ):
            return ResolvedArtifact(artifact_id=artifact_id, run_id=run_id, artifact_name=artifact_name)
    msg = f"No unexpired release bundle {artifact_name!r} comes from a successful merged PR run"
    raise BundleError(msg)


def _write_github_outputs(path: Path | None, values: Mapping[str, str | int]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as output_file:
        for name, value in values.items():
            output_file.write(f"{name}={value}\n")


def _metadata_command(args: argparse.Namespace) -> int:
    metadata = release_metadata(args.source_root, expected_tag=args.expected_tag)
    values = {
        "package_version": metadata.version,
        "source_digest": metadata.source_digest,
        "source_digest_hex": metadata.source_digest.removeprefix("sha256:"),
        "source_commit": metadata.source_commit,
        "artifact_name": metadata.artifact_name,
    }
    _write_github_outputs(args.github_output, values)
    print(json.dumps(values, sort_keys=True))
    return 0


def _finalize_command(args: argparse.Namespace) -> int:
    manifest = finalize_bundle(
        args.bundle_dir,
        args.source_root,
        checks=set(args.check),
        retention_days=args.retention_days,
    )
    _write_github_outputs(
        args.github_output,
        {
            "package_version": str(manifest["version"]),
            "source_digest": str(manifest["release_input_digest"]),
            "artifact_name": str(manifest["artifact_name"]),
        },
    )
    print(f"Created {args.bundle_dir / MANIFEST_NAME} for {manifest['artifact_name']}")
    return 0


def _verify_command(args: argparse.Namespace) -> int:
    manifest = verify_bundle(
        args.bundle_dir,
        expected_version=args.expected_version,
        expected_source_digest=args.expected_source_digest,
    )
    print(f"Verified release bundle {manifest['artifact_name']}")
    return 0


def _resolve_command(args: argparse.Namespace) -> int:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN", "")
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    resolved = resolve_artifact(
        GitHubClient(token, api_url),
        repository=args.repository,
        artifact_name=args.artifact_name,
        workflow_path=args.workflow_path,
        default_branch=args.default_branch,
    )
    _write_github_outputs(
        args.github_output,
        {
            "artifact_id": resolved.artifact_id,
            "artifact_name": resolved.artifact_name,
            "run_id": resolved.run_id,
        },
    )
    print(f"Resolved {resolved.artifact_name} from successful PR workflow run {resolved.run_id}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata_parser = subparsers.add_parser("metadata", help="Compute release version and tracked-tree identity.")
    metadata_parser.add_argument("--source-root", type=Path, default=Path.cwd())
    metadata_parser.add_argument("--expected-tag")
    metadata_parser.add_argument("--github-output", type=Path)
    metadata_parser.set_defaults(func=_metadata_command)

    finalize_parser = subparsers.add_parser("finalize", help="Create release checksums and manifest.")
    finalize_parser.add_argument("--bundle-dir", type=Path, required=True)
    finalize_parser.add_argument("--source-root", type=Path, default=Path.cwd())
    finalize_parser.add_argument("--check", action="append", default=[])
    finalize_parser.add_argument("--retention-days", type=int, default=DEFAULT_RETENTION_DAYS)
    finalize_parser.add_argument("--github-output", type=Path)
    finalize_parser.set_defaults(func=_finalize_command)

    verify_parser = subparsers.add_parser("verify", help="Verify a downloaded release bundle.")
    verify_parser.add_argument("--bundle-dir", type=Path, required=True)
    verify_parser.add_argument("--expected-version", required=True)
    verify_parser.add_argument("--expected-source-digest", required=True)
    verify_parser.set_defaults(func=_verify_command)

    resolve_parser = subparsers.add_parser("resolve", help="Find a bundle from a successful merged PR run.")
    resolve_parser.add_argument("--repository", required=True)
    resolve_parser.add_argument("--artifact-name", required=True)
    resolve_parser.add_argument("--workflow-path", default=DEFAULT_WORKFLOW_PATH)
    resolve_parser.add_argument("--default-branch", required=True)
    resolve_parser.add_argument("--github-output", type=Path)
    resolve_parser.set_defaults(func=_resolve_command)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return args.func(args)
    except (BundleError, OSError, TypeError, json.JSONDecodeError) as error:
        print(f"Release bundle error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
