"""Build and consume deterministic duration-balanced pytest file shards."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from defusedxml.ElementTree import parse as parse_xml

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]


def discover_test_files(repo_root: Path = REPO_ROOT) -> tuple[str, ...]:
    """Return every pytest test module as a repository-relative POSIX path."""
    return tuple(sorted(path.relative_to(repo_root).as_posix() for path in (repo_root / "tests").rglob("test_*.py")))


def _default_weight(weights: dict[str, float]) -> float:
    positive_weights = [weight for weight in weights.values() if weight > 0]
    return statistics.median(positive_weights) if positive_weights else 1.0


def assign_shards(
    test_files: tuple[str, ...] | list[str],
    weights: dict[str, float],
    shard_count: int,
) -> tuple[tuple[str, ...], ...]:
    """Assign each test file exactly once using deterministic greedy balancing."""
    if shard_count < 1:
        msg = "shard_count must be at least 1"
        raise ValueError(msg)

    files = tuple(sorted(set(test_files)))
    fallback_weight = _default_weight(weights)
    weighted_files = sorted(files, key=lambda path: (-weights.get(path, fallback_weight), path))
    shards: list[list[str]] = [[] for _ in range(shard_count)]
    totals = [0.0] * shard_count

    for path in weighted_files:
        shard_index = min(range(shard_count), key=lambda index: (totals[index], len(shards[index]), index))
        shards[shard_index].append(path)
        totals[shard_index] += weights.get(path, fallback_weight)

    return tuple(tuple(sorted(shard)) for shard in shards)


def _testcase_path(classname: str, repo_root: Path) -> str | None:
    parts = classname.split(".")
    for length in range(len(parts), 0, -1):
        candidate = Path(*parts[:length]).with_suffix(".py")
        if (repo_root / candidate).is_file():
            return candidate.as_posix()
    return None


def weights_from_junit(junit_path: Path, repo_root: Path = REPO_ROOT) -> dict[str, float]:
    """Aggregate xdist JUnit testcase durations by test module."""
    root = parse_xml(junit_path).getroot()
    weights: dict[str, float] = {}
    for testcase in root.iter("testcase"):
        path = _testcase_path(testcase.attrib.get("classname", ""), repo_root)
        if path is None:
            continue
        try:
            duration = float(testcase.attrib.get("time", "0"))
        except ValueError:
            duration = 0.0
        weights[path] = weights.get(path, 0.0) + max(duration, 0.0)
    return {path: round(weight, 6) for path, weight in sorted(weights.items())}


def build_manifest(junit_path: Path, source: str, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Build the committed shard-weight manifest from a successful JUnit file."""
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "weights_seconds": weights_from_junit(junit_path, repo_root),
    }


def load_manifest(path: Path) -> dict[str, float]:
    """Load and validate a shard-weight manifest."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        msg = f"{path} has an unsupported shard manifest schema"
        raise ValueError(msg)
    raw_weights = data.get("weights_seconds")
    if not isinstance(raw_weights, dict):
        msg = f"{path} does not contain a weights_seconds mapping"
        raise TypeError(msg)

    weights: dict[str, float] = {}
    for test_path, raw_weight in raw_weights.items():
        if not isinstance(test_path, str) or not isinstance(raw_weight, (int, float)) or raw_weight < 0:
            msg = f"{path} contains an invalid weight entry: {test_path!r}={raw_weight!r}"
            raise ValueError(msg)
        weights[test_path] = float(raw_weight)
    return weights


def validate_manifest(path: Path, repo_root: Path = REPO_ROOT) -> list[str]:
    """Return manifest integrity issues."""
    try:
        weights = load_manifest(path)
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        return [str(error)]

    discovered = set(discover_test_files(repo_root))
    stale = sorted(set(weights) - discovered)
    issues = [f"Manifest references missing test module: {test_path}" for test_path in stale]
    shards = assign_shards(tuple(discovered), weights, 2)
    assigned = [test_path for shard in shards for test_path in shard]
    if sorted(assigned) != sorted(discovered) or len(assigned) != len(set(assigned)):
        issues.append("Shard assignment does not cover every discovered test module exactly once.")
    return issues


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _add_common_manifest_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "ci" / "test-durations.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    update_parser = subparsers.add_parser("update", help="Regenerate weights from JUnit evidence.")
    update_parser.add_argument("--junit", type=Path, required=True)
    update_parser.add_argument("--source", required=True)
    _add_common_manifest_argument(update_parser)

    select_parser = subparsers.add_parser("select", help="Emit one NUL-delimited shard.")
    _add_common_manifest_argument(select_parser)
    select_parser.add_argument("--shard-count", type=int, required=True)
    select_parser.add_argument("--shard-index", type=int, required=True)
    select_parser.add_argument("--exclude-file", action="append", default=[])

    check_parser = subparsers.add_parser("check", help="Validate the committed manifest.")
    _add_common_manifest_argument(check_parser)
    return parser.parse_args()


def _select(args: argparse.Namespace) -> int:
    weights = load_manifest(args.manifest)
    excluded = set(args.exclude_file)
    test_files = tuple(path for path in discover_test_files() if path not in excluded)
    shards = assign_shards(test_files, weights, args.shard_count)
    if args.shard_index < 0 or args.shard_index >= len(shards):
        print(f"shard-index must be between 0 and {len(shards) - 1}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(b"\0".join(path.encode() for path in shards[args.shard_index]) + b"\0")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "update":
        _write_manifest(args.manifest, build_manifest(args.junit, args.source))
        return 0
    if args.command == "select":
        return _select(args)

    issues = validate_manifest(args.manifest)
    if issues:
        print("CI shard manifest validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("CI shard manifest validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
