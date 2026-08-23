---
description: Benchmarking requirements for performance-sensitive changes
applies_to: albumentations/**/*.py
always_apply: false
---

# Benchmarking Requirements

Use the `performance-optimization` skill and its canonical reference before changing a hot route. Use the `benchmark`
skill when the task needs measurements.

Benchmark when a change can alter an executed runtime path, and select only the routes and axes that can falsify its
performance claim. Compare baseline and candidate at the same revision boundary and in the same environment. Include
the public `Compose` route as well as the direct kernel when both are affected.

For pixel arithmetic, conversion, layout, or backend-routing changes, the required image matrix is 256, 512, and 1024
pixels by 1, 3, and 5 channels where supported; grayscale Compose inputs remain `(H, W, 1)`. For a different proposal,
vary the controlling axis instead—for example label density, random-output size, or routing-threshold boundaries—and
state why the matrix is bounded.

Report every measured before/after cell, the baseline revision, environment, and any regression above 5%. Do not add a
benchmark merely because an adjacent file changed.
