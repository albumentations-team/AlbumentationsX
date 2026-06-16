# Performance Coverage

This document defines the maintenance standard for saying that
AlbumentationsX performance is covered across the transform catalog. The
benchmark suite is evidence for maintainers and release reviewers. It is not a
runtime API surface.

## Coverage Standard

Performance coverage has multiple layers. A transform catalog is considered
covered only when all required layers are present, validated by tooling, and
included in CI or release evidence.

### L0: Catalog Inventory

- Discover public concrete `BasicTransform` subclasses exported by
  `albumentations`.
- Require every public transform to have benchmark accounting.
- Record optional transforms explicitly with a reason instead of silently
  skipping them.
- Record each transform's expected coverage contract and fail validation when
  required layers are missing.

The machine-readable source is `benchmark/benchmarks/catalog.py`. Validate it
with:

```bash
uv run python -m tools.benchmark_coverage check
```

Inspect per-transform coverage depth with:

```bash
uv run python -m tools.benchmark_coverage details --output benchmark-coverage-detail.json
```

The detail artifact is the auditable transform-by-transform evidence file. For
each public transform it records:

- public class metadata: module, qualname, and `albumentations.<Transform>`
  export path
- benchmark route and constructor parameters used by the catalog smoke path
- coverage layers and required coverage contract
- family labels such as geometry, pixel, reference-data, volumetric, direct
  kernel, memory, alias, or optional PyTorch
- exact ASV benchmark class, config, and case IDs that measure the transform

Validate coverage policy and benchmark stability classes with:

```bash
uv run python -m tools.performance_budget check
```

### L1: Catalog Smoke

- Run one valid `Compose` path for every runnable public transform.
- Use deterministic inputs and explicit parameter overrides for transforms that
  need non-default constructor arguments or auxiliary metadata.
- Keep optional PyTorch tensor transforms accounted separately unless the
  benchmark environment installs the optional dependency.

This layer catches public catalog drift and verifies that every runnable
transform has at least one measured user-facing route.

Alias transforms that intentionally warn at construction are benchmarked
through their canonical implementation and recorded with an `alias_coverage`
layer in the per-transform detail artifact. The catalog smoke path still
verifies that the public alias constructs and executes.

`ToTensorV2` and `ToTensor3D` are not part of the default headless ASV suite.
They are covered by the dedicated PyTorch ASV config,
`benchmark/asv-pytorch.conf.json`, and appear in coverage details under the
`pytorch_tensor` layer. They must not be treated as uncovered simply because
the default ASV environment avoids torch.

### L2: Family Matrices

Normal image transforms must have transform-level matrix coverage beyond the
catalog smoke path. Direct functional-kernel coverage is useful diagnostic
evidence, but it does not replace user-facing `Compose` coverage for public
transforms.

- image sizes: `256x256`, `512x512`, `1024x1024`
- channels: `1`, `3`, `5` where supported
- dtypes: `uint8`, `float32` where supported
- targets: image, mask, bbox, keypoint, volume, and reference metadata where
  applicable

The current family matrix covers:

- geometry: crop, resize, pad, symmetry, rotate, affine, perspective,
  distortions, spline, scale, and refraction paths
- pixel: pointwise, LUT-like, blur/filter, noise, normalization,
  dtype-conversion, color, compression, and superpixel paths
- annotations: HBB, OBB, keypoints, masks, label fields, and bbox-safe crops
- special targets: bbox-safe crops, near-bbox crop metadata, constrained
  dropout, non-empty-mask crop, and mask dropout over size/channel/dtype
  variants
- alias coverage: warning aliases such as `ShiftScaleRotate`,
  `TimeReverse`, `TimeMasking`, and `FrequencyMasking` mapped to their
  canonical benchmarked transforms
- reference data: mixing, domain adaptation, overlay, copy-paste, mosaic, and
  text metadata transforms
- volumetric data: public 3D transforms over volume size and dtype variants
- optional tensor data: PyTorch tensor conversion paths for 2D and 3D terminal
  transforms in the optional PyTorch ASV lane

### L3: Direct Functional Kernels

Shared functional kernels must be benchmarked directly, outside `Compose`, so
reviewers can separate raw kernel changes from pipeline overhead. This layer
covers:

- 2D geometry image kernels such as flips, resize, padding, perspective warp,
  and remap
- geometry annotation kernels for bbox and keypoint transforms
- pixel kernels such as gamma, multiply/add, weighted blending, equalize,
  auto-contrast, HSV shifting, RGB matrix transforms, dropout, compression,
  solarize, and posterize
- blur/filter kernels such as box blur, convolution, defocus, zoom blur, mode
  filter, and glass blur
- 3D kernels such as crop, pad, cutout, cube symmetry, and tile swapping

Direct-kernel coverage is summarized under `direct_kernel_cases` in the JSON
output from `tools.benchmark_coverage`. Per-transform layer membership is
published in the `benchmark-coverage-detail.json` evidence artifact.

### L3b: Core Pipeline

Core `Compose` and processor behavior must be benchmarked separately from
transform kernels. This layer covers:

- single-transform `Compose`
- multi-transform `Compose`
- `ReplayCompose`
- `p=0` skip dispatch
- `additional_targets` routing
- image batch routing through `images`
- `Compose` setup time for simple and multi-transform pipelines
- bbox/keypoint processor setup
- bbox/keypoint processor round-trips using a no-op transform

### L4: Target Scaling

Annotation and metadata scaling must be measured separately from simple image
paths:

- bbox/keypoint counts: `10`, `100`, and `1000` for stable scaling paths
- direct affine bbox/keypoint kernels: `10` and `100`; larger affine scaling is
  covered through Compose-level annotation benchmarks to avoid noisy numerical
  warnings in the low-level kernel
- reference metadata: representative copy-paste, overlay, mosaic, adaptation,
  and text cases
- labels: at least one path must exercise label-field routing

### L5: Memory

Peak-memory checks must cover allocation-heavy paths:

- large resize
- large affine
- large normalize
- batch image pipeline
- mosaic
- copy-paste
- volume padding

Memory results from shared runners are advisory until enough scheduled data
exists to set reliable blocking thresholds.

### L6: Regression Governance

Benchmark existence is not enough. A performance-sensitive change must produce
before/after evidence against a baseline.

Initial triage thresholds:

- more than about 5 percent slower on a representative case: investigate and
  mention in review
- more than about 10 percent slower on a representative case: release-relevant
  unless recovered or explicitly justified
- material memory growth: investigate, especially for large image, batch,
  reference-data, and volumetric paths

Accepted slowdowns should have a maintainer-visible reason, such as a
correctness fix, broader dtype/channel/target support, lower memory use,
security hardening, or simpler behavior that removes a known maintenance risk.

ASV comparison summaries are classified by `tools/performance_budget.py` into
machine-readable budget evidence:

- `ok`: coverage contracts pass and no regression triage is required
- `triage_required`: at least one benchmark changed beyond the warning budget
  or an unclassified benchmark changed
- `release_blocked`: a stable release-critical benchmark regressed beyond the
  blocking budget or benchmark coverage contracts failed
- `missing_comparison`: a strict release check required before/after evidence
  but no ASV comparison summary was provided

Initial benchmark classes:

- stable release-blocking classes: core pipeline, geometry/pixel/volumetric
  matrices, annotation/special-target scaling, and direct functional kernels
- advisory classes: catalog smoke, reference-data matrices, legacy
  representative benchmarks, optional PyTorch tensor benchmarks, and
  peak-memory checks

Advisory does not mean optional. It means the evidence must be reviewed, but
shared-runner noise or optional dependency setup should not automatically block
a release until scheduled history proves the benchmark is stable enough.

## CI And Release Evidence

Pull requests:

- required: benchmark coverage validation through `tools.benchmark_coverage`
- required: ASV suite importability where the performance workflow runs
- advisory: ASV before/after comparison on GitHub-hosted runners when runtime
  or benchmark code changes; PR comparison starts with catalog smoke, core
  pipeline, and direct functional-kernel benchmarks, then adds changed-family
  matrix benchmarks through `tools/select_benchmark_filters.py`
- benchmark infrastructure changes rely on ASV importability and benchmark
  coverage validation in PR; full-suite comparison remains scheduled or manual
- selected PR benchmark filter and changed-file evidence are uploaded with
  benchmark artifacts when a comparison runs
- performance-budget evidence classifies coverage status, benchmark stability,
  triage items, and release-blocking regressions
- optional PyTorch ASV is not run on every pull request because installing
  torch can dominate feedback time; it is run on `main`, scheduled, and manual
  performance workflows

Nightly and scheduled runs:

- full ASV evidence on `main`
- optional PyTorch tensor ASV evidence
- environment JSON
- benchmark coverage JSON
- ASV result artifacts

Release candidates:

- benchmark coverage JSON
- per-transform benchmark coverage detail JSON
- performance-budget JSON
- ASV evidence
- ASV comparison summary JSON when a baseline/candidate comparison is run
- correctness report performance summary
- documented triage for material regressions

Manual investigations:

```bash
cd benchmark
uv tool run --from asv asv --config asv.conf.json continuous \
  --factor 1.05 \
  --split \
  --show-stderr \
  <baseline-ref> \
  <candidate-ref>
```

Manual workflow runs may pass `bench_filter` to scope a comparison or leave it
empty to compare the full suite.

Optional PyTorch tensor benchmarks can be run locally with:

```bash
cd benchmark
uv tool run --from asv asv --config asv-pytorch.conf.json run --quick --show-stderr
```

The raw ASV comparison text is the source artifact. The compact JSON summary
used by release reports can be generated with:

```bash
uv run python tools/asv_summary.py \
  --input benchmark-evidence/asv-continuous.txt \
  --output benchmark-evidence/benchmark-asv-summary.json
```

Classify the comparison against the release budget with:

```bash
uv run python tools/performance_budget.py summarize \
  --coverage-summary benchmark-evidence/benchmark-coverage.json \
  --coverage-detail benchmark-evidence/benchmark-coverage-detail.json \
  --asv-summary benchmark-evidence/benchmark-asv-summary.json \
  --output benchmark-evidence/benchmark-performance-budget.json
```

## Acceptance Criteria

The project can claim catalog-wide performance coverage when all of the
following are true:

- `uv run python -m tools.benchmark_coverage check` passes.
- ASV importability passes for the benchmark suite.
- Every public transform is either runnable in the catalog smoke layer or
  explicitly accounted as optional with a dedicated benchmark lane.
- Every transform's benchmark coverage contract is satisfied in
  `benchmark-coverage-detail.json`.
- `benchmark-coverage-detail.json` exposes class metadata, benchmark route,
  constructor parameters, family labels, and ASV case IDs for each public
  transform.
- The per-transform detail artifact is published and has zero smoke-only
  runnable transforms.
- Every normal image transform has transform-level size/channel/dtype matrix
  coverage unless it is an explicitly documented alias.
- Direct functional kernels have non-empty coverage in each required group.
- Annotation, reference-data, volumetric, batch, and memory paths are included
  in benchmark evidence.
- Core pipeline evidence includes dispatch, setup, `additional_targets`,
  image batch routing, and bbox/keypoint processor overhead.
- Optional PyTorch tensor paths are included in scheduled or release-adjacent
  benchmark evidence.
- Performance-budget evidence is published and has no coverage-contract
  failures.
- Release evidence compares the candidate against an appropriate baseline or
  documents why comparison evidence is unavailable.
- Material runtime or memory regressions are either fixed or justified in
  release-visible notes.
