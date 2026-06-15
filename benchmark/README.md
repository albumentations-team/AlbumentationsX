# Benchmark Suite

AlbumentationsX uses ASV for scheduled and release performance evidence. The
benchmarks are separate from unit tests and should not be imported by tests.

## Quick Run

```bash
uv tool run --from asv asv --config "$(pwd)/benchmark/asv.conf.json" run --quick --show-stderr
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
