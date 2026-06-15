# Benchmark Suite

AlbumentationsX uses ASV for scheduled and release performance evidence. The
benchmarks are separate from unit tests and should not be imported by tests.

## Quick Run

```bash
cd benchmark
uv tool run --from asv asv --config asv.conf.json run --quick --show-stderr
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

The GitHub performance workflow can also be run manually with
`workflow_dispatch` inputs:

- `baseline_ref`: git ref or SHA for the before state.
- `candidate_ref`: git ref or SHA for the after state; defaults to the selected
  workflow ref when omitted.

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

Benchmarks cover public Compose paths, selected direct transform paths where
useful, image batches, representative volumetric transforms, and one
reference-data transform smoke path. GitHub-hosted runner results are advisory
until enough scheduled data exists to set reliable blocking thresholds.

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

The default ASV environment installs the `headless` and `text` extras so the
suite can run OpenCV-backed transforms and `TextImage` without GUI packages.
The check fails when a public transform is missing from coverage accounting,
when a configured transform no longer exists, when a benchmark transform cannot
be constructed, or when the representative smoke route no longer runs. The ASV
suite includes `TimeCatalogTransformSmoke`, which executes one valid
`Compose` path for every runnable public transform. Optional PyTorch tensor
transforms are accounted separately because the default performance environment
installs the headless package extras.

Catalog smoke coverage is not the whole performance policy. It proves that
every transform has at least one measured route. Hot families still need richer
benchmarks over the standard image-size and channel matrix, annotation-heavy
paths, volumetric paths, batch inputs, direct functional kernels, and selected
memory checks.

Any task that changes transform hot paths, functional kernels, parameter
generation, or core pipeline code should include before/after benchmark
evidence. This applies whether the change is reviewed as a pull request,
prepared as a release task, or checked locally before merging a larger branch.
A material slowdown, initially treated as more than about 5% on a
representative case, needs either a code change to recover the regression or a
clear maintainer-visible reason why the tradeoff is intentional.
