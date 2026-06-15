# Benchmark Suite

AlbumentationsX uses ASV for scheduled and release performance evidence. The
benchmarks are separate from unit tests and should not be imported by tests.

## Quick Run

```bash
cd benchmark
uv tool run --from asv asv --config asv.conf.json run --quick --show-stderr
```

For pull-request comparisons, use the same base/head shape as CI:

```bash
cd benchmark
uv tool run --from asv asv --config asv.conf.json continuous \
  --factor 1.05 \
  --split \
  --show-stderr \
  <base-commit> \
  <head-commit>
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

Benchmarks cover public Compose paths, selected direct transform paths where
useful, image batches, representative volumetric transforms, and one
reference-data transform smoke path. GitHub-hosted runner results are advisory
until enough scheduled data exists to set reliable blocking thresholds.

Pull requests that change transform hot paths, functional kernels, parameter
generation, or core pipeline code should include before/after benchmark
evidence. A material slowdown, initially treated as more than about 5% on a
representative case, needs either a code change to recover the regression or a
clear maintainer-visible reason why the tradeoff is intentional.
