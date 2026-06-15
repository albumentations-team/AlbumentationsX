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

The machine-readable source is `benchmark/benchmarks/catalog.py`. Validate it
with:

```bash
uv run python -m tools.benchmark_coverage check
```

### L1: Catalog Smoke

- Run one valid `Compose` path for every runnable public transform.
- Use deterministic inputs and explicit parameter overrides for transforms that
  need non-default constructor arguments or auxiliary metadata.
- Keep optional PyTorch tensor transforms accounted separately unless the
  benchmark environment installs the optional dependency.

This layer catches public catalog drift and verifies that every runnable
transform has at least one measured user-facing route.

### L2: Family Matrices

Hot transform families must have richer matrix coverage than the catalog smoke
path:

- image sizes: `256x256`, `512x512`, `1024x1024`
- channels: `1`, `3`, `5` where supported
- dtypes: `uint8`, `float32` where supported
- targets: image, mask, bbox, keypoint, volume, and reference metadata where
  applicable

The current family matrix covers:

- geometry: crop, resize, pad, symmetry, rotate, affine, perspective,
  distortions, spline, and refraction paths
- pixel: pointwise, LUT-like, blur/filter, noise, normalization,
  dtype-conversion, color, compression, and superpixel paths
- annotations: HBB, OBB, keypoints, masks, label fields, and bbox-safe crops
- reference data: mixing, domain adaptation, overlay, copy-paste, mosaic, and
  text metadata transforms
- volumetric data: public 3D transforms over volume size and dtype variants

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
output from `tools.benchmark_coverage`.

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

## CI And Release Evidence

Pull requests:

- required: benchmark coverage validation through `tools.benchmark_coverage`
- required: ASV suite importability where the performance workflow runs
- advisory: ASV before/after comparison on GitHub-hosted runners when runtime
  or benchmark code changes; PR comparison is bounded to catalog smoke, core
  pipeline, and direct functional-kernel benchmarks to keep feedback timely

Nightly and scheduled runs:

- full ASV evidence on `main`
- environment JSON
- benchmark coverage JSON
- ASV result artifacts

Release candidates:

- benchmark coverage JSON
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

The raw ASV comparison text is the source artifact. The compact JSON summary
used by release reports can be generated with:

```bash
uv run python tools/asv_summary.py \
  --input benchmark-evidence/asv-continuous.txt \
  --output benchmark-evidence/benchmark-asv-summary.json
```

## Acceptance Criteria

The project can claim catalog-wide performance coverage when all of the
following are true:

- `uv run python -m tools.benchmark_coverage check` passes.
- ASV importability passes for the benchmark suite.
- Every public transform is either runnable in the catalog smoke layer or
  explicitly accounted as optional.
- Hot transform families have size/channel/dtype matrix coverage.
- Direct functional kernels have non-empty coverage in each required group.
- Annotation, reference-data, volumetric, batch, and memory paths are included
  in benchmark evidence.
- Release evidence compares the candidate against an appropriate baseline or
  documents why comparison evidence is unavailable.
- Material runtime or memory regressions are either fixed or justified in
  release-visible notes.
