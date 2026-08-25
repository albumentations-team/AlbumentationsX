# Benchmark Suite

AlbumentationsX uses ASV for scheduled and release performance evidence. The
benchmarks are separate from unit tests and should not be imported by tests.

## Quick Comparison

```bash
cd benchmark
BASELINE_REF="$(git describe --tags --abbrev=0 --match '[0-9]*' HEAD^)"
uv tool run --from asv asv --config asv.conf.json continuous \
  --quick --show-stderr \
  --attribute repeat=1 --attribute number=1 \
  "$BASELINE_REF" HEAD
```

For task-level before/after comparisons, use the same baseline/candidate shape
as CI:

```bash
cd benchmark
uv tool run --from asv asv --config asv.conf.json continuous \
  --factor 1.05 \
  --split \
  --show-stderr \
  <baseline-ref> \
  <candidate-ref>
```

PyTorch Tensor routes are benchmarked with a separate ASV config so terminal
conversion and accepted Tensor-native capability matrices remain independently
reviewable:

```bash
cd benchmark
BASELINE_REF="$(git describe --tags --abbrev=0 --match '[0-9]*' HEAD^)"
uv tool run --from asv asv --config asv-pytorch.conf.json continuous \
  --quick --show-stderr \
  --attribute repeat=1 --attribute number=1 \
  "$BASELINE_REF" HEAD
```

The GitHub performance workflow can also be run manually with
`workflow_dispatch` inputs:

- `baseline_ref`: git ref or SHA for the before state.
- `candidate_ref`: git ref or SHA for the after state; defaults to the selected
  workflow ref when omitted.
- `bench_filter`: optional ASV `--bench` regular expression for manual
  baseline/candidate comparisons.

Every runtime pull request runs a bounded `pr-core` comparison over the core
pipeline. A PR labeled `run-performance` selects the `changed` profile from
changed files. The Tuesday schedule compares the previous release with `main`
using `stf-core`, which combines catalog smoke, core pipeline routes, and
selected memory sentinels. Manual comparisons with an empty `bench_filter` also
select `stf-core`; a full-family run requires an explicit regex.

Resolve the named profiles locally with:

```bash
uv run python tools/select_benchmark_filters.py --profile pr-core
uv run python tools/select_benchmark_filters.py --profile stf-core
uv run python tools/select_benchmark_filters.py --profile changed --changed-files changed-files.txt
```

## Matrix

The baseline image matrix follows the project performance rule:

- 256 x 256 x 1
- 256 x 256 x 3
- 256 x 256 x 5
- 512 x 512 x 1
- 512 x 512 x 3
- 512 x 512 x 5
- 1024 x 1024 x 1
- 1024 x 1024 x 3
- 1024 x 1024 x 5

Benchmarks cover public Compose paths, core pipeline dispatch/setup paths,
direct functional kernels, image and mask batch routes,
representative volumetric transforms, annotation scaling paths, selected
parameter-sensitive stress cases, memory checks, and reference-data transform
paths. GitHub-hosted runner results are advisory until enough scheduled data
exists to set reliable blocking thresholds.

## Catalog Coverage

The transform catalog has a machine-readable benchmark registry in
`benchmark/benchmarks/catalog.py`. The registry discovers public
`BasicTransform` subclasses exported by `albumentations`, assigns each transform
to a valid benchmark route, and records explicit setup parameters for transforms
that need non-default constructor arguments or auxiliary data.

Run the coverage check with:

```bash
uv run python -m tools.benchmark_coverage check
```

Emit the JSON summary that CI uploads with benchmark artifacts with:

```bash
uv run python -m tools.benchmark_coverage summary --output benchmark-coverage.json
```

Emit the per-transform detail artifact with:

```bash
uv run python -m tools.benchmark_coverage details --output benchmark-coverage-detail.json
```

Compare a saved detail artifact against the current transform catalog with:

```bash
uv run python -m tools.benchmark_coverage diff \
  --base-detail benchmark-coverage-detail-main.json \
  --output benchmark-coverage-diff.json
```

The detail artifact is intended for review, not only for automation. Each
record includes the public class module/qualname, catalog benchmark route,
constructor parameters, family labels, required coverage contract, and exact
ASV case IDs for the transform's smoke, family-matrix, direct-kernel, memory,
reference-data, volumetric, target, or dedicated PyTorch Tensor evidence.
Each ASV case also includes parsed scenario metadata, and each transform record
includes a `scenario_contract` summary of covered sizes, channels, dtypes,
annotation counts, batch sizes, targets, memory cases, direct-kernel groups,
parameter-sensitivity scenarios, and benchmark scopes.
The detail artifact also includes `scenario_axis_contracts`, which records
covered axes, intentionally skipped standard axes, and the policy reason for
each benchmark layer. Parameter-sensitivity cases record the exact constructor
values used for each stress scenario.
Each transform also has a `performance_contract` section for batch,
annotation, direct-kernel, parameter-sensitivity, and memory expectations. This
section records required benchmark layers, directly declared implementation
methods, and the reason a dedicated layer is covered, advisory, tracked for
audit, or not required.
The diff artifact is intended for PR and release review. It reports added or
removed public transforms and coverage changes such as layer, ASV case, or
axis-contract, performance-contract, or benchmark-policy drift.

Summarize ASV before/after comparison text for release reports with:

```bash
uv run python tools/asv_summary.py \
  --input benchmark-evidence/asv-continuous.txt \
  --output benchmark-evidence/benchmark-asv-summary.json
```

Classify coverage and comparison evidence against the performance budget with:

```bash
uv run python tools/performance_budget.py summarize \
  --coverage-summary benchmark-evidence/benchmark-coverage.json \
  --coverage-detail benchmark-evidence/benchmark-coverage-detail.json \
  --asv-summary benchmark-evidence/benchmark-asv-summary.json \
  --output benchmark-evidence/benchmark-performance-budget.json
```

The default ASV environment installs the `headless` and `text` extras so the
suite can run OpenCV-backed transforms and `TextImage` without GUI packages.
The check fails when a public transform is missing from coverage accounting,
when a configured transform no longer exists, when a benchmark transform cannot
be constructed, or when the representative smoke route no longer runs. The ASV
suite includes `TimeCatalogTransformSmoke`, which executes one valid
`Compose` path for every runnable public transform. Explicit terminal Tensor
transforms are accounted separately and benchmarked by the dedicated
`asv-pytorch.conf.json` lane. The detail artifact records the expected coverage
contract and actual coverage layers for each public transform.

Catalog smoke coverage is not the whole performance policy. It proves that
every transform has at least one measured route. Hot families still need richer
benchmarks over the standard image-size and channel matrix, annotation-heavy
paths, volumetric paths, batch inputs, direct functional kernels, and selected
memory checks.

The current ASV suite includes these production coverage layers:

- Catalog smoke: one runnable `Compose` path for every public transform except
  explicit terminal Tensor transforms, which have a dedicated matrix.
- Full-matrix geometry: representative crop, resize, pad, symmetry, rotate,
  affine, perspective, distortion, spline, and refraction transforms across
  size, channel, and dtype matrices.
- Full-matrix pixel: representative pointwise, LUT-like, blur/filter, noise,
  normalization, dtype-conversion, color, compression, and superpixel
  transforms across their supported size, channel, and dtype matrices.
- Annotation scaling: HBB bboxes, OBB bboxes, keypoints, masks, label-field
  routing, and bbox-safe crop paths at 10, 100, and 1000 annotations.
- Special target matrix: bbox-safe crops, near-bbox crop metadata,
  constrained dropout, non-empty-mask crop, and mask dropout over
  size/channel/dtype variants.
- Reference-data paths: mixing, domain adaptation, overlay, copy-paste,
  mosaic, and text metadata transforms.
- Volumetric paths: all public 3D transforms over size and dtype variants.
- Alias coverage: warning aliases mapped to their canonical benchmarked
  implementation while catalog smoke still validates public construction.
- Dedicated PyTorch Tensor paths: `ToTensorV2`, `ToTensor3D`, and accepted
  Tensor-native capabilities such as `Transpose(image=...)` in a separate ASV
  environment. The Tensor-native `Transpose` matrix currently covers C=1 and
  C=3 image Tensors only.
- Core pipeline: single-transform Compose, multi-transform Compose,
  ReplayCompose, `p=0` skip dispatch, `additional_targets`, image batches,
  Compose setup, and bbox/keypoint processor overhead.
- Batch matrix: `images` and `images` plus `masks` routes over bounded size,
  channel, dtype, and batch-size variants.
- Parameter sensitivity: representative transforms whose runtime changes
  materially with constructor parameters, such as blur kernel size, dropout
  hole count, grid map resolution, JPEG quality, and superpixel segment count.
- Direct functional kernels: shared geometry, annotation, pixel, blur/filter,
  and 3D kernels benchmarked outside `Compose` to identify whether changes come
  from raw kernels or pipeline overhead.
- Memory checks: allocation-heavy resize, affine, normalize, large float32
  median blur, batch pipeline, mosaic, copy-paste, and volume padding paths.

Any task that changes transform hot paths, functional kernels, parameter
generation, or core pipeline code should include before/after benchmark
evidence. This applies whether the change is reviewed as a pull request,
prepared as a release task, or checked locally before merging a larger branch.
A material slowdown, initially treated as more than about 5% on a
representative case, needs either a code change to recover the regression or a
clear maintainer-visible reason why the tradeoff is intentional.
The raw ASV comparison text remains the authoritative artifact; the JSON
summary is a compact index for release reports and quick review. The
performance-budget JSON is the policy artifact that distinguishes advisory
triage from stable release-blocking regressions.
