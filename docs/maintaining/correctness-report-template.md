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

- Unit and functional tests: `<count/result>`
- Public transform-like API coverage routes: `<covered>/<total>`
- Parameterized transform sweep coverage: `<count/result>`
- Golden regression vectors: `<manifest-count>/<registered-contract-count>/<result>`
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
- Golden vectors are compatibility sentinels, not a replacement for the full
  parameterized transform, functional, annotation, and serialization test
  suites.
- Optional extras are smoke-tested, not exhaustively cross-product tested.

## Performance

| Area | Status |
| --- | --- |
| Catalog ASV smoke cases | `<count/result>` |
| Deep per-transform coverage | `<count/result>` |
| Smoke-only transforms | `<count/list or link>` |
| Alias-covered transforms | `<count/list or link>` |
| Full-matrix geometry cases | `<count/result>` |
| Full-matrix pixel cases | `<count/result>` |
| Special target matrix cases | `<count/result>` |
| Direct functional-kernel cases | `<count/result>` |
| Annotation/reference/volumetric cases | `<count/result>` |
| Core Compose overhead | `<status>` |
| Memory checks | `<status>` |
| ASV comparison summary | `<status/link>` |
| ASV comparison refs | `<baseline-ref> -> <candidate-ref>` |
| Benchmark coverage diff | `<ok/changed/unavailable/link>` |
| Performance contract audit | `<batch/annotation/direct-kernel/parameter/memory counts>` |
| Performance budget | `<ok/triage_required/release_blocked/link>` |
| Material runtime regressions | `<none/list with justification>` |
| Material memory regressions | `<none/list with justification>` |

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
- Version-bump preflight run: `<url>`
- Release delivery workflow run: `<url>`
- Lockfile hash: `<hash>`
- Environment summary artifacts: `<url>`
