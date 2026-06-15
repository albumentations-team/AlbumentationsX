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
- Manual release-candidate verification runs the full test suite across the
  supported OS/Python matrix with the release Hypothesis profile.
- Performance benchmark import checks, catalog coverage validation, direct
  functional-kernel coverage validation, ASV before/after comparison for
  performance-sensitive changes, manual baseline/candidate comparison through
  `workflow_dispatch`, and scheduled ASV artifact generation.
- Security checks, including runtime dependency audit, GitHub Actions
  hardening audit, and OpenSSF Scorecard.

## Performance Regression Policy

The performance workflow is advisory on GitHub-hosted runners until scheduled
data establishes reliable blocking thresholds. It still produces review
evidence: changes to transform hot paths, functional kernels, parameter
generation, or core pipeline code should compare the changed state against a
baseline with ASV. Pull requests do this automatically when they touch runtime
or benchmark code, using a bounded catalog/core/direct-kernel comparison so the
advisory lane stays timely. Release tasks, large branches, and local
investigations can use the manual workflow inputs to compare an explicit
`baseline_ref`, `candidate_ref`, and optional ASV `bench_filter`.

Material slowdowns, initially interpreted as more than about 5% on a
representative benchmark case, should be treated as release-relevant. The
change should either recover the regression or document the reason the slower
behavior is an intentional tradeoff, such as a correctness fix, broader target
support, or lower memory use.

The benchmark coverage layers and release acceptance criteria are defined in
`docs/maintaining/performance-coverage.md`.

## Evidence Artifacts

Each CI lane that runs tests should upload:

- environment JSON from `tools/collect_test_environment.py`;
- JUnit XML from pytest where practical;
- compact pytest summary JSON from `tools/pytest_summary.py`;
- benchmark, ASV comparison, or security JSON where those checks run.

Release reports are generated from these artifacts when available. Local dry
runs may use `--allow-missing-evidence`, but release workflows should provide
environment evidence.
