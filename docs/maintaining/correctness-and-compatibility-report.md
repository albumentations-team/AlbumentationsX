# Correctness & Compatibility Report

Every official release should publish a concise Correctness & Compatibility
Report alongside the wheel, sdist, SBOM, and checksum manifest. The report is a
public downstream-facing summary of what was tested, what is guaranteed, and
which limitations are known.

## Required Sections

1. Release identity: version, date, tag, commit, GitHub Release, and PyPI
   release where available.
2. Compatibility matrix: Python versions, operating systems, and dependency
   sets tested.
3. Test summary covering unit, regression, property, serialization/replay, 3D,
   bbox/keypoint, and OBB tests. Record separate results for the [validation contract](../validation.md)
   the [2D geometry contract](../geometry.md), and the [determinism and replay contract](../determinism.md).
4. Correctness contracts: determinism, dtype and shape, target alignment,
   label integrity, [2D geometry](../geometry.md), volume semantics, and
   serialization behavior.
5. Known limitations: dependency-sensitive exactness, advisory optional-extra
   coverage, and platform caveats.
6. Performance summary: benchmark status, coverage-contract status,
   performance-budget status, dedicated PyTorch Tensor benchmark evidence, ASV
   comparison triage where available, and memory observations.
7. Security posture: runtime audit, workflow audit, OpenSSF Scorecard, CodeQL
   status, SBOM, checksums, and PyPI provenance.
8. Reproducibility details: CI runs, lockfile hash, environment artifacts, and
   commands.
9. Behavior changes and migration guidance from the checked [release-note inventory](../release-notes/unreleased.md),
   following the [behavior compatibility policy](behavior-compatibility.md). Each change links its migration, contract,
   and tests at the tested revision. Every category explicitly lists its changes or `none`; the first affected version
   is the version being released.

Reports must avoid internal funding or grant context. They should describe the
library guarantees in language useful to downstream users.

## Local Generation

```bash
uv run python tools/generate_correctness_report.py \
  --allow-missing-evidence \
  --output _internal/correctness-report-dry-run.md
```

The version-bump release preflight should omit `--allow-missing-evidence` and
provide at least one environment JSON artifact, passing pytest summary JSON,
passing `validation-contract.json`, `geometry-contract.json`, and `determinism-contract.json` suite results,
benchmark coverage summary/detail JSON,
benchmark performance-budget JSON, and security JSON evidence. Local dry runs may use `--allow-missing-evidence` when
those artifacts are not available.

`--allow-missing-evidence` selects a local preview: it permits missing test artifacts and a package version at or below
the inventory baseline. Release preflight omits the flag and enforces both checks.

The determinism artifact must cover the [composition audit](determinism-audit.md), including real spawned DataLoader
workers with and without persistence. A missing, empty, failing, or skipped determinism suite blocks the release
report. The report pins documentation to the tested source revision and links the CI run from environment evidence.

The report also runs `tools.check_behavior_changes`. Missing migration fields, disconnected policy documents, and
missing contract/test links block generation, including local previews. A release version at or below the inventory's
baseline blocks a release report. Local previews may retain `unreleased` and are not publication evidence.
