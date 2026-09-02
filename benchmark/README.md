# AlbumentationsX benchmarks

The benchmark catalog measures public AlbumentationsX execution paths with ASV.
It has two uses:

- `release-core` is the fixed 21-case `256×256` runtime and peak-memory profile
  for weekly and release comparisons.
- The remaining catalog supplies focused local diagnostics, including `512×512`,
  `1024×1024`, channel, dtype, and annotation-scaling cases.

The catalog is not a replacement for correctness tests. It measures only the
named public route and its configured input.

## Run the fixed release profile

From the repository root:

```bash
filter="$(uv run python tools/select_benchmark_filters.py --profile release-core)"
uv run --group ci-benchmark asv --config benchmark/asv.conf.json continuous \
  --bench "$filter" <baseline-ref> <candidate-ref>
```

`release-core` uses RGB `uint8` `256×256` images. Its annotation route uses 10
bounding boxes and 10 keypoints. The full case list and evidence contract are in
[`docs/maintaining/performance-coverage.md`](../docs/maintaining/performance-coverage.md).

## Run local PR evidence

If a PR changes a hot path or claims a performance or memory result, select only
the affected full-matrix cases. For example, a Compose change can use the local
full-matrix class:

```bash
uv run --group ci-benchmark asv --config benchmark/asv.conf.json continuous \
  --bench TimeComposeFullMatrix <baseline-ref> <candidate-ref>
```

For annotation-count behavior, use `TimeTargetProcessorScaling`. Geometry,
pixel, mixing, and volumetric families have their own full-matrix classes where
the larger input or parameter axis is meaningful.

Record the command, exact ASV filter, baseline SHA, candidate SHA, environment,
and results in the PR description. Do not report a one-revision run as a speed
comparison.

## Explicit hosted-runner reproduction

The Performance workflow runs only when a maintainer applies `run-performance`
to a PR or starts it manually. On a labeled PR it selects changed benchmark
families. On manual dispatch, an empty filter uses `release-core`; provide an exact
filter to reproduce a larger or full-family investigation.

Weekly and release preflight comparisons use `release-core`. The manual Tensor ASV
workflow is separate and unscheduled.

## Catalog validation

```bash
uv run python -m tools.benchmark_coverage check
uv run python -m tools.benchmark_coverage details --output benchmark-coverage-detail.json
```

Add benchmark ownership when a public transform family needs a new route. Do not
add a duplicate case only to increase a benchmark count. See
[`docs/maintaining/performance-coverage.md`](../docs/maintaining/performance-coverage.md)
for the catalog and evidence policy.
