---
description: Testing conventions for transform tests
applies_to: tests/**/*.py
always_apply: false
---

# Testing Conventions

## Prefer Standard Transform Calls Over Direct Method Calls

Use `transform(image=image)` / `transform(images=images)` through `Compose`, not direct calls like `transform.apply()` or `transform.apply_to_images()`.

This exercises the full pipeline (parameter generation, dtype handling, Compose routing) and catches integration issues.

```python
# CORRECT - use transform call interface
transform = A.Compose([A.MedianBlur(blur_range=(3, 7), p=1.0)])

# Single image
result = transform(image=image)["image"]

# Batch of images
batch_result = transform(images=images)["images"]

# INCORRECT - avoid direct method calls in tests
result = transform.apply(image, kernel=3)
batch_result = transform.apply_to_images(images, kernel=3)
```

## Parallel Test Execution

Use `pytest-xdist` to run tests in parallel:

```bash
# Run tests using all available CPU cores
pytest -n auto

# Run tests using a specific number of workers
pytest -n 4
```

Ensure tests are independent and don't share mutable global state so they can run safely in parallel.

## Use the Generated Target Cluster

Register every public constructor mode once in `tests/helpers/transform_cases.py`. A `DualTransform` case automatically
collects against applicable core profiles for image/mask, HBB, OBB, keypoints, and volume/`mask3d`. One explicit primary
mode per class collects against extended dtype, channel, batch, empty-target, layout, and read-only profiles.

- Put reusable input workloads and structural assertions in `tests/helpers/target_profiles.py`.
- Put transform-required metadata in the case `context_factory`.
- Declare `required_targets` when sampling needs a non-empty mask or bbox collection.
- Derive applicability from `_targets`, `_supported_bbox_types`, and genuine channel-count capabilities.
- Do not add hard-coded transform lists, compatibility adapters, class-name branches, or target smoke tests whose only
  assertion is successful execution.
- Keep focused tests for exact geometry, interpolation, sampling, validation, empty representation, and metamorphic
  semantics that shared profiles cannot express.

See `docs/design/transform-target-contracts.md` and run:

```bash
uv run pytest -n auto -q tests/contracts/test_target_cluster_contract.py
```
