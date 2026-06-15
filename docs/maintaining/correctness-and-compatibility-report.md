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
3. Test summary: unit, regression, property, serialization/replay, 3D,
   bbox/keypoint, and OBB coverage.
4. Correctness contracts: determinism, dtype and shape, target alignment,
   label integrity, volume semantics, and serialization behavior.
5. Known limitations: dependency-sensitive exactness, advisory optional-extra
   coverage, and platform caveats.
6. Performance summary: benchmark status, coverage-contract status, optional
   PyTorch tensor benchmark evidence, and memory observations.
7. Security posture: runtime audit, workflow audit, OpenSSF Scorecard, CodeQL
   status, SBOM, checksums, and PyPI provenance.
8. Reproducibility details: CI runs, lockfile hash, environment artifacts, and
   commands.

Reports must avoid internal funding or grant context. They should describe the
library guarantees in language useful to downstream users.

## Local Generation

```bash
uv run python tools/generate_correctness_report.py \
  --allow-missing-evidence \
  --output _internal/correctness-report-dry-run.md
```

Release workflows should omit `--allow-missing-evidence` and provide at least
one environment JSON artifact, benchmark coverage summary/detail JSON, and
security JSON evidence. Local dry runs may use `--allow-missing-evidence` when
those artifacts are not available.
