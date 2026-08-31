# CI policy

AlbumentationsX uses the shortest CI graph that can expose a real regression.
The pull-request workflow always starts its router and its five pre-commit
partitions. Product, package, and policy jobs run only when the changed paths
can affect their result. Unknown paths select the conservative profile.

The implementation lives in [`.github/workflows/pr.yml`](../../.github/workflows/pr.yml).
`tools/ci_plan.py` owns path routing. Direct job names are the branch-protection
contexts; there are no aggregate gate aliases.

## Required pull-request contexts

Require these repository-owned contexts and the external `license/cla` check:

- `PR / PR plan`
- `PR / Pre-commit / Ruff`
- `PR / Pre-commit / Ruff format`
- `PR / Pre-commit / mypy`
- `PR / Pre-commit / Pyrefly`
- `PR / Pre-commit / Other hooks`

For a runtime or shared-test change, also require every visible
`Compatibility (<OS>, Python <version>)` context and `PyTorch tests`. Package
and policy contexts are required only when their path rule selects them. CodeQL,
Antigravity review, and performance workflows are advisory because their path
filters or explicit triggers can correctly skip them.

## Pre-commit ownership

Every configured hook runs through pre-commit exactly once. The four expensive
hooks have their own parallel jobs; the fifth job runs every remaining hook:

```bash
pre-commit run ruff --all-files --show-diff-on-failure
pre-commit run ruff-format --all-files --show-diff-on-failure
pre-commit run mypy --all-files --show-diff-on-failure
pre-commit run pyrefly-check --all-files --show-diff-on-failure
SKIP=ruff,ruff-format,mypy,pyrefly-check pre-commit run --all-files --show-diff-on-failure
```

CI does not invoke Ruff, mypy, or Pyrefly directly. This keeps local and remote
quality rules identical. The `Other hooks` partition owns repository contracts,
lock freshness, benchmark catalog validation, regression vectors, legal checks,
Markdown links, generated documentation, syntax checks, and secret detection.

## Product-test routing

| Changed area | Selected jobs |
| --- | --- |
| Runtime source, shared test infrastructure, dependency metadata, CI router, or unknown path | Complete 3 × 5 compatibility matrix |
| Core runtime or PyTorch source; dedicated PyTorch tests; dependency metadata | Dedicated CPU-only PyTorch tests |
| Isolated non-PyTorch test module | Targeted tests on the boundary platforms |
| Workflow-only change | GitHub Actions hardening audit |
| Legal, package, or dependency metadata | Source legal check, package build/metadata check, and relevant dependency audit |
| Version-only release bump | Release preflight only, outside always-run pre-commit jobs |

The matrix is fifteen distinct jobs: Ubuntu, Windows, and macOS on Python 3.10,
3.11, 3.12, 3.13, and 3.14. Each job runs the complete non-PyTorch suite with
two xdist workers and `--hypothesis-profile=ci-fast`. There are no Windows shard
aliases and no duplicate primary suite.

PyTorch-only files are excluded from the base suite and run once in a dedicated
Ubuntu/Python 3.12 CPU-Torch job. This prevents a marker-specific regression
from being silently absent while avoiding duplicate base-suite cases.

Draft PRs keep the five pre-commit partitions and defer routed jobs. The
`full-ci` label overrides that deferral.

## Removed work

Codecov is not part of AlbumentationsX CI. The repository has no Codecov action,
token, XML artifact, `pytest-cov` dependency, coverage configuration, or
coverage-only test run. Benchmark catalog coverage is unrelated: it verifies
that public transform families have owned performance cases and remains a
pre-commit contract.

The workflow also has no `Fast checks`, `Correctness`, or `Security and policy`
aggregation jobs. A failing leaf is already the actionable result; a second job
only delays the final status and hides the failing owner.

## Release performance evidence

Scheduled and release ASV comparisons run the fixed `release-core` profile. It has
21 named runtime and peak-memory cases at `256×256`, RGB, and `uint8`; target
processor evidence uses exactly 10 bounding boxes and 10 keypoints. The profile
adds representative transform families rather than a size, channel, or
annotation-count matrix, so weekly wall time stays bounded.

The Performance workflow runs timed ASV only when a maintainer applies the
`run-performance` label or starts it manually. That route is for reproducing a
specific report, not for routine PR timing. Scheduled and release-preflight
comparisons register an ASV machine, compare a real baseline with a candidate,
and fail if the comparison evidence is missing. The manual PyTorch Tensor ASV
workflow has no schedule.

For a PR that changes a hot path or claims a performance or memory result, the
author runs the affected `512×512` and/or `1024×1024` benchmark cases locally
and records the following in the PR description:

1. Command and exact ASV filter.
2. Baseline and candidate SHAs.
3. Hardware, operating system, Python, dependency versions, and CPU-thread
   settings.
4. Selected larger-input cases and their before/after result.

No remote workflow expands to larger input sizes implicitly. A maintainer may
use `run-performance` only when a remote reproduction is useful.

## CodeQL and environments

CodeQL Python runs for Python, stub, or CodeQL-configuration paths. CodeQL
Actions runs only for `.github/actions/**` and `.github/workflows/**`; both also
run weekly and on manual dispatch. Direct pushes to `main` are blocked by the
ruleset, except for these path-scoped CodeQL alert refreshes.

Each job selects a locked dependency group through `setup-ci`. Jobs that import
AlbumentationsX also select `torch-cpu`; static package, security, and typing
jobs stay Torch-free. CI never syncs the broad `dev` group.

## Local validation

```bash
uv lock --check
uv run python -m tools.ci_matrix check
pre-commit run --all-files --show-diff-on-failure
```

Run the relevant test module or ASV filter in addition to these commands. Do
not describe an unrun command as evidence.
