# CI Policy

AlbumentationsX CI separates fast pull-request signal from slower confidence
checks. The current required PR matrix remains intentionally broad while the
new nightly, release-candidate, and performance lanes collect evidence without
destabilizing contributor workflows.

## Required Pull-Request Gates

- Full test matrix on `ubuntu-latest`, `windows-latest`, and `macos-latest`.
- Python 3.10, 3.11, 3.12, 3.13, and 3.14.
- Locked latest dependency set from `uv.lock`.
- Pre-commit hooks and project-specific checks.
- `tools.ci_matrix check`.

## Scheduled And Release Gates

- Lower-bound dependency testing on Ubuntu and Python 3.10.
- Golden regression vectors.
- Property tests with the `ci-fast` profile on PR and broader profiles in
  nightly/release workflows.
- Performance smoke and ASV artifact generation.
- Security checks, including runtime dependency audit, GitHub Actions
  hardening audit, and OpenSSF Scorecard.

## Evidence Artifacts

Each CI lane that runs tests should upload:

- environment JSON from `tools/collect_test_environment.py`;
- JUnit XML from pytest where practical;
- compact pytest summary JSON from `tools/pytest_summary.py`;
- benchmark or security JSON where those checks run.

Release reports are generated from these artifacts when available. Local dry
runs may use `--allow-missing-evidence`, but release workflows should provide
environment evidence.
