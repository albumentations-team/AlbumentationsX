# Performance coverage

AlbumentationsX maintains two complementary performance contracts:

1. The benchmark catalog maps public transform families to owned benchmark
   routes. `tools.benchmark_coverage` validates that mapping.
2. The fixed `release-core` ASV profile supplies recurring runtime and
   peak-memory regression evidence for release decisions.

Catalog coverage answers “does this public family have an owned benchmark?” It
does not report Python test coverage and does not replace a baseline-to-candidate
timing comparison.

## Catalog contract

Run the catalog check locally or through the `check-benchmark-coverage`
pre-commit hook:

```bash
uv run python -m tools.benchmark_coverage check
uv run python -m tools.benchmark_coverage details --output benchmark-coverage-detail.json
```

Public image transforms need a family-matrix route beyond catalog smoke. Target,
batch, mixing, volumetric, Tensor, and functional-kernel routes are explicit
where their public behavior requires them. The catalog retains matrices for
larger sizes, multiple channel counts, dtypes, and annotation scaling as local
diagnostics. Those axes are not scheduled by default.

Use `tools/benchmark_coverage.py details` when reviewing a transform addition or
benchmark-family change. It identifies the owning class, layer, route, required
coverage contract, aliases, and missing layers. Fix missing ownership instead of
adding an unrelated benchmark case.

## Release routine profile

`release-core` is the small, stable ASV profile for weekly and release evidence. It
runs 21 named runtime and peak-memory cases at `256×256`, RGB, and `uint8`.
The target-processor case uses exactly 10 bounding boxes and 10 keypoints.

The profile covers Compose behavior, target processors, geometric, pixel,
mixing, and volumetric transforms. The `release-core` selector below is the
exact case list. Do not add a second input-size, channel-count, dtype, or
annotation-count axis to this profile. Add a
representative transform only when it covers a distinct core path.

```bash
uv run python tools/select_benchmark_filters.py --profile release-core
```

The scheduled Performance workflow and release preflight run ASV `continuous`
against a baseline and candidate. They save raw ASV output, parsed summary,
catalog coverage detail, environment evidence, and a performance-budget result.
Release preflight requires a comparison and fails on release-blocking regressions.

## Local large-input evidence

For a PR that changes a hot path or makes a performance or memory claim, run
only the affected `512×512` and/or `1024×1024` catalog cases locally. Record the
command, filter, baseline SHA, candidate SHA, environment, and results in the PR
description. A benchmark without the exact baseline and candidate is diagnostic,
not a regression claim.

The `run-performance` label and manual Performance workflow exist for explicit
remote reproduction. They do not run on ordinary PRs and never expand to the
full catalog by default. A blank manual filter selects `release-core`; an explicit
filter selects a larger or changed-family investigation.

## Interpretation

ASV timing is sensitive to host load and environment. Compare the same route,
input, and execution mode on the same controlled environment before treating a
number as a regression. Peak-memory cases measure allocations from the public
`Compose` route; they do not establish a general memory bound for all inputs.

Do not use Codecov, test coverage percentage, a catalog-only report, or a
single-revision ASV result as performance evidence. They answer different
questions.
