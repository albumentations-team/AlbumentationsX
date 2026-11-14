# Unreleased behavior changes

Release baseline: 2.6.2

Follow the [behavior compatibility policy](../maintaining/behavior-compatibility.md) when updating this inventory.
Earlier changes shipped in [2.6.2](2.6.2.md), [2.6.1](2.6.1.md), and [2.6.0](2.6.0.md).

## Validation

none

## Geometry

none

## Determinism

### Sparse SaltAndPepper replay payloads

- Previous behavior: ReplayCompose stored two boolean masks for sampled salt and pepper positions.
- New behavior: Samples affecting up to 16% of spatial positions store flat indices and a salt count. Dense samples
  still store masks, and existing mask-based records remain readable. The draw and pixel values are preserved in
  the tested cases.
- Affected users: Pipelines that inspect sampled parameters directly or pass new sparse records to workers on 2.6.2.
- Migration: Treat replay parameters as opaque and use 2.6.3 or newer on workers that read new sparse records.
  For example, replay with `A.ReplayCompose.replay(record, image=image)` on an upgraded worker.
- First affected version: unreleased
- Contract: [Determinism and replay](../determinism.md).
- Tests: [SaltAndPepper replay parity](../../tests/test_augmentations.py).

### CPU NumPy to Tensor storage

- Previous behavior: ToTensorV2 and ToTensor3D could wrap a read-only NumPy buffer when it was already contiguous.
- New behavior: The shared bridge copies read-only buffers before wrapping them. Terminal transforms still return
  contiguous tensors, and compatible writable arrays may share storage.
- Affected users: Pipelines that convert read-only images, masks, or volumes to CPU tensors.
- Migration: Existing transforms preserve values and layouts. Reuse the returned tensor as the transformed output,
  for example `A.ToTensorV2()(image=image)["image"]`. Do not depend on storage aliasing.
- First affected version: unreleased
- Contract: [Conversion and storage](../design/numpy-tensor-routing.md#conversion-and-storage).
- Tests: [2D tensor storage](../../tests/test_pytorch.py), [3D tensor storage](../../tests/transforms3d/test_pytorch.py).

## Constructor defaults

none

## Deprecations

### ModeFilter removal

- Previous behavior: `A.ModeFilter` replaced each pixel with the most frequent value in its neighborhood per channel.
- New behavior: Loading a saved pipeline, replay, or applied configuration containing the removed transform
  raises `ValueError` for an unknown transform.
- Affected users: Pipelines that call `A.ModeFilter` or load saved records that contain it.
- Migration: Remove the transform from the pipeline and regenerate its saved records. Use `albumentationsx==2.6.2`
  to reproduce existing pipelines and saved records that require this filter.
- First affected version: unreleased
- Contract: [Replay and deserialization validation](../validation.md).
- Tests: [Public constructor coverage](../../tests/contracts/test_constructor_parameter_coverage.py),
  [unknown transform validation](../../tests/contracts/test_validation_contract.py).

## Regression vectors

none
