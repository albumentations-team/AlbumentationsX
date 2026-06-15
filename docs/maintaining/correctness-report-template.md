# AlbumentationsX Correctness & Compatibility Report: `<version>`

Release date: `<date>`
GitHub Release: `<url>`
PyPI: `<url>`
Commit: `<sha>`

## Compatibility

Supported Python versions: 3.10, 3.11, 3.12, 3.13, 3.14

| OS | Python | Dependency Set | Result |
| --- | --- | --- | --- |
| `ubuntu-latest` | 3.10, 3.11, 3.12, 3.13, 3.14 | `locked-latest` | `<result>` |
| `windows-latest` | 3.10, 3.11, 3.12, 3.13, 3.14 | `locked-latest` | `<result>` |
| `macos-latest` | 3.10, 3.11, 3.12, 3.13, 3.14 | `locked-latest` | `<result>` |
| `ubuntu-latest` | 3.10 | `declared-minimum` | `<result>` |
| `ubuntu-latest` | 3.14 | `optional-extras` | `<result>` |
| `ubuntu-latest` | `3.15-dev` | `pre-release-probe` | `<result>` |

## Correctness Coverage

- Unit tests: `<count/result>`
- Golden regression vectors: `<count/result>`
- Property-based invariant tests: `<profile/count/result>`
- Serialization and ReplayCompose checks: `<result>`
- Bbox/keypoint/OBB checks: `<result>`
- Volumetric checks: `<result>`

## Guaranteed Contracts

- Fixed-seed Compose pipelines are deterministic for tested transforms.
- ReplayCompose reproduces tested transform parameters and outputs.
- Image outputs preserve documented dtype and channel semantics.
- Bbox and keypoint label fields remain aligned with surviving annotations.
- Compose-level compatibility checks fail at pipeline creation where applicable.

## Known Limitations

- Exact pixel values for selected OpenCV-backed interpolation paths may vary
  across upstream OpenCV versions.
- Optional extras are smoke-tested, not exhaustively cross-product tested.

## Performance

| Area | Status |
| --- | --- |
| Core Compose overhead | `<status>` |
| Geometric transforms | `<status>` |
| Pixel transforms | `<status>` |
| Memory checks | `<status>` |

## Security And Release Integrity

- pip-audit runtime dependency scan: `<status>`
- GitHub Actions hardening audit: `<status>`
- OpenSSF Scorecard: `<link/status>`
- CodeQL: `<link/status>`
- SBOM: `<link>`
- SHA256 checksums: `<link>`
- PyPI provenance: `<link>`

## Reproducibility

- CI workflow run: `<url>`
- Release workflow run: `<url>`
- Lockfile hash: `<hash>`
- Environment summary artifacts: `<url>`
