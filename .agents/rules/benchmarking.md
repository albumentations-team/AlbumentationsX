---
description: Benchmarking requirements for performance-sensitive changes
applies_to: albumentations/**/*.py
always_apply: false
---

# Benchmarking Requirements

Use the `performance-optimization` skill and its canonical reference before changing a hot route. Use the `benchmark`
skill when the task needs measurements.

**Every timing runs with exactly one CPU thread, including quick ad-hoc checks.** Production is AX under a PyTorch
DataLoader with one thread per worker, so a multi-threaded timing measures the wrong thing. Before importing numerical
libraries set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and
`NUMEXPR_NUM_THREADS` to `1`; then call `cv2.setNumThreads(0)`, `torch.set_num_threads(1)`, and
`torch.set_num_interop_threads(1)`. Time thread counts above one only on an explicit user request. Never write a bare
`timeit` loop; load the `benchmark` skill first.

Benchmark when a change can alter an executed runtime path, and select only the routes and axes that can falsify its
performance claim. Compare baseline and candidate at the same revision boundary and in the same environment. Include
the public `Compose` route as well as the direct kernel when both are affected.

For pixel arithmetic, conversion, layout, or backend-routing changes, the required image matrix is 256, 512, and 1024
pixels by 1, 3, and 5 channels where supported; grayscale Compose inputs remain `(H, W, 1)`. For a different proposal,
vary the controlling axis instead—for example label density, random-output size, or routing-threshold boundaries—and
state why the matrix is bounded.

Report every measured before/after cell, the baseline revision, environment, and any regression above 5%. Do not add a
benchmark merely because an adjacent file changed.
