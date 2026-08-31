# Greenfield pull-request CI

**Status:** Implemented

**Decision date:** 2026-08-31

## Decision

Pull-request CI keeps independent compatibility contracts and all pre-commit
rules, while deleting repeated work that did not add a distinct signal.

- Five parallel jobs run every configured pre-commit hook exactly once. Ruff,
  Ruff format, mypy, and Pyrefly run through their pre-commit hook IDs; the
  fifth partition owns the remaining hooks.
- Runtime, shared-test, dependency, CI-router, and unknown changes run the full
  3 × 5 compatibility matrix: Ubuntu, Windows, and macOS on Python 3.10–3.14.
  Each pair is one visible job.
- The dedicated CPU-only PyTorch job owns PyTorch-marked tests once. Base matrix
  jobs exclude those files.
- Codecov, `pytest-cov`, coverage XML, the coverage-only test run, the duplicate
  primary suite, clean-install PR matrix, and aggregate gate jobs are removed.
- A direct leaf job is the merge signal. The ruleset requires direct pre-commit
  contexts plus the plan; it must not wait for retired aggregate aliases.
- Routine PR ASV timing is removed. `run-performance` and manual dispatch are
  explicit reproductions. Weekly and release `release-core` comparisons provide
  recurring runtime and memory evidence.

## Why this graph is smaller

The prior coverage-only and primary jobs repeated the same non-PyTorch product
suite already covered by the compatibility matrix. They did not enforce an
independent repository policy. A Codecov upload is not a test assertion, so it
cannot make that duplicate execution useful.

The aggregate jobs also had no independent check. They started after the leaf
that already contained the actionable failure. Requiring direct job contexts
therefore shortens the final status and makes a cancellation or timeout visible
at its owner.

The compatibility matrix remains intact because an operating system and Python
version pair is an execution contract, not a duplicate. It has 15 independent
jobs and no sharded aliases.

## Pre-commit partition

```bash
pre-commit run ruff --all-files --show-diff-on-failure
pre-commit run ruff-format --all-files --show-diff-on-failure
pre-commit run mypy --all-files --show-diff-on-failure
pre-commit run pyrefly-check --all-files --show-diff-on-failure
SKIP=ruff,ruff-format,mypy,pyrefly-check pre-commit run --all-files --show-diff-on-failure
```

This preserves mypy, Pyrefly, and every other configured hook without creating
direct tool invocations that drift from contributor validation.

## Release and large-input evidence

`release-core` is a fixed 21-case profile at RGB `uint8` `256×256`. It includes
runtime and peak-memory cases across Compose, target processing, geometric,
pixel, mixing, and volumetric paths. The target-processor case uses 10 bounding
boxes and 10 keypoints. The benchmark policy defines the full evidence contract.

Weekly main and release preflight runs compare a real baseline with a candidate.
They fail if comparison evidence is missing. The manual PyTorch Tensor workflow
has no schedule.

When a PR changes a hot path or makes a performance/memory claim, the author
runs the affected `512×512` and/or `1024×1024` catalog cases locally. The PR
description must contain the command, ASV filter, baseline and candidate SHAs,
environment, selected cases, and results. Hosted `run-performance` is for
explicit reproduction; it never expands to a large-input matrix implicitly.

## Completion conditions

- `pre-commit run --all-files --show-diff-on-failure` passes locally.
- `python -m tools.ci_matrix check` validates workflow, matrix, dependency, and
  runtime-profile contracts.
- No tracked workflow, manifest, lockfile, or dependency group contains Codecov,
  `pytest-cov`, a coverage XML job, an aggregate gate, or the retired routine PR
  ASV jobs.
- The `release-core` selector returns exactly 21 fixed cases and includes runtime
  plus peak-memory evidence.
- Maintained CI, benchmark, and release documentation describe the implemented
  graph and local large-input PR evidence.
