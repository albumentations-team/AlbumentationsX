# CI policy

AlbumentationsX pull-request CI optimizes for merge-ready wall time while
preserving the supported compatibility guarantee. The workflow always starts,
classifies the changed paths, and runs only the checks that can produce useful
signal for those paths. Unknown paths fail closed to the complete profile.

The implementation lives in `.github/workflows/pr.yml`. Routing is owned by
`tools/ci_plan.py`; stable gate aggregation is owned by `tools/ci_gate.py`.

## Required pull-request gates

Repository rules should require these repository-owned jobs, not conditional
leaf jobs:

- `PR / PR plan`
- `PR / Fast checks`
- `PR / Correctness`
- `PR / Security and policy`

The plan job always reports. Each aggregate gate also always reports and fails
closed when a selected leaf is missing, skipped, cancelled, or failed. A leaf
that the router did not select may be skipped without leaving the pull request
waiting for a status that will never arrive.

Keep the external CLA check and GitHub's stable code-scanning results rule when
they are part of repository policy. Do not require dynamic CodeQL leaf names
such as `Analyze (actions)` or `Analyze (python)`. Do not require advisory ASV
or AI-review jobs.

Direct pushes to `main` are outside the maintenance policy and should be
blocked by the repository ruleset. Repository workflows use pull-request,
scheduled, manual, and release events instead of `push` events.

## Path-based selection

The router applies additive domains: a mixed change receives the union of the
relevant checks. These are the normal profiles:

| Changed paths | Required work | Work intentionally skipped |
| --- | --- | --- |
| Ordinary Markdown | Changed-file Markdown hooks | pytest, typing, packaging, security audits |
| CI policy Markdown | Markdown hooks and repository contracts | product pytest |
| Runtime source | Ruff, mypy, Pyrefly, contracts, full 3 × 5 compatibility, one coverage lane | package builds and workflow audit |
| Isolated test module | Changed module on the primary and OS/version boundary lanes | full product matrix |
| Shared pytest infrastructure | Full 3 × 5 compatibility | unrelated package and workflow policy jobs |
| PyTorch source or tests | Dedicated CPU-only PyTorch job plus relevant base checks | Torch installation in ordinary compatibility jobs |
| Dependency metadata or lockfile | Primary suite, PyTorch, dependency audit, legal/package checks, clean install matrix | ASV timing comparison |
| Packaging or legal inputs | Source legal verification, wheel/sdist verification, metadata check, clean installs when relevant | product compatibility matrix |
| `.github/**` | Repository contracts and `zizmor` | product pytest unless the PR workflow/router itself changed |
| Benchmark source or tooling | Benchmark contracts and advisory ASV evidence | product pytest |
| Unknown path | Complete conservative profile | Nothing |

Draft pull requests run the selected fast feedback only. Moving a pull request
to ready-for-review starts its selected correctness and policy work. The
`full-ci` label overrides draft routing and selects the complete conservative
profile.

## Compatibility and wall time

Runtime changes retain every supported operating-system and Python pair:

- Ubuntu, Windows, and macOS;
- Python 3.10, 3.11, 3.12, 3.13, and 3.14;
- the locked `ci-test` dependency profile.

Compatibility jobs do not collect coverage and do not install PyTorch. Branch
coverage runs once on Ubuntu and Python 3.12. PyTorch-marked tests run once in a
dedicated CPU-only environment.

Windows 3.11, 3.12, and 3.13 are split into two duration-balanced test-file
shards because the measured baseline exceeded two minutes. The committed
weights live in `ci/test-durations.json`; `tools/ci_shard.py` assigns every
discovered test file exactly once and gives new files a deterministic fallback
weight. Other compatibility pairs stay unsharded.

All pytest lanes use two xdist workers with the `worksteal` scheduler. The
workflow cancels obsolete runs when a newer commit updates the same pull
request.

The operational targets are:

| Pull-request class | Required p50 | Required p95 |
| --- | ---: | ---: |
| Ordinary Markdown only | 20s | 45s |
| Workflow, legal, or benchmark infrastructure | 30s | 60s |
| Runtime Python | 90s | 120s |
| Dependency, packaging, or PyTorch | 120s | 180s |

Measure from the latest pull-request event until every required gate is final,
including queue time. Two-minute parallel jobs are acceptable; an irrelevant
job or a missing required status is not.

## Purpose-specific environments

CI jobs sync one locked dependency group through
`.github/actions/setup-ci/action.yml`:

- `ci-test`: base pytest suite and optional test libraries;
- `ci-quality`: Ruff, pre-commit, the isolated mypy hook, and repository contracts;
- `ci-types`: Pyrefly and standalone typing tools;
- `ci-pytorch`: base test dependencies before the CPU-only Torch install;
- `ci-security`: pip-audit and zizmor;
- `ci-package`: build, twine, and legal verifier tests;
- `ci-benchmark`: ASV;
- `ci-release`: version-bump preflight, final distributions, release evidence,
  and bundle tooling.

The contributor-facing `dev` group includes all of these capabilities plus
normal PyTorch packages. CI must not sync the broad `dev` group.

## ASV performance evidence

ASV means **airspeed velocity**, the Python benchmark framework used under
`benchmark/`. It is unrelated to Python abstract syntax trees.

The lightweight `ASV benchmark evidence` job validates benchmark catalog
coverage and benchmark-suite importability. It runs only when runtime,
benchmark, performance-workflow, or benchmark-policy files change. Actual
before/after timing runs for pull requests require the `run-performance` label;
scheduled and manual workflows can also run them.

Performance jobs remain advisory while hosted-runner variance is being
measured. If performance becomes blocking, introduce one stable always-reported
`Performance` gate with documented thresholds. Never make a conditional ASV
leaf globally required.

## Security, legal, and release coverage

The pull-request workflow routes dependency audit, workflow audit, source legal
verification, artifact verification, and clean install smoke tests by path.
Any valid `project.version` increase also selects the complete profile and the
release preflight that creates the final publishable bundle. The later
`release: published` workflow only verifies and delivers that bundle; it does
not repeat the release checks.
The scheduled Security workflow still runs dependency audit, `zizmor`, and
OpenSSF Scorecard evidence independently of pull-request routing.

Nightly and release-candidate workflows run the complete 3 × 5 base suite.
They also retain lower-bound, property/regression, optional-extra, PyTorch,
performance, and release artifact evidence where appropriate.

CodeQL is managed by GitHub default setup. Do not add an advanced CodeQL
workflow while default setup is enabled, because duplicate configurations can
reject SARIF uploads. Selective language analysis should be handled as a
separate default-to-advanced migration.

## Antigravity pull-request reviews

`Antigravity PR Checks` is advisory and outside merge-ready wall time. It runs
for non-draft, same-repository pull requests whose paths select source, tests,
workflows, legal policy, or unknown high-risk changes. Ordinary Markdown and
dependency-only changes do not invoke the model.

The `pull_request_target` workflow checks out only the trusted base revision.
Gemini receives pull-request metadata and diff as untrusted review data, has
read-only repository tools, and has no shell or pull-request write token. A
separate publisher job posts the one-day review artifact.

Configure these repository variables:

| Name | Value |
| --- | --- |
| `ANTIGRAVITY_GCP_PROJECT_ID` | `albumentations` |
| `ANTIGRAVITY_GCP_LOCATION` | `global` |
| `ANTIGRAVITY_GCP_SERVICE_ACCOUNT` | `antigravity-pr-review@albumentations.iam.gserviceaccount.com` |
| `ANTIGRAVITY_GCP_WIF_PROVIDER` | `projects/663083315901/locations/global/workloadIdentityPools/github-actions/providers/albumentationsx-pr-review` |

The Workload Identity provider condition must match repository ID
`1005218687`, owner ID `57894582`, event `pull_request_target`, base branch
`main`, and workflow ref
`albumentations-team/AlbumentationsX/.github/workflows/antigravity-pr-checks.yml@refs/heads/main`.

## Local validation and evidence

Use the same repository-owned commands locally:

```bash
uv run python -m tools.ci_matrix check
uv run python -m tools.ci_shard check
uv run python -m tools.quality_gate fast
```

Test lanes should retain JUnit XML and compact summaries where practical.
Always-run evidence steps may pass `--allow-incomplete` to
`tools/pytest_summary.py` so early failures still produce explicit evidence.
Release reports remain strict and require complete test, environment,
benchmark, performance-budget, and security evidence.

The design rationale, measured baseline, and future impact-selection criteria
are recorded in `docs/maintaining/ci-greenfield-plan.md`.
