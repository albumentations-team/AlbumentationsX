# Unreleased behavior changes

Release baseline: 2.6.1

Follow the [behavior compatibility policy](../maintaining/behavior-compatibility.md) when updating this inventory.
Earlier changes shipped in [2.6.1](2.6.1.md) and [2.6.0](2.6.0.md).

## Validation

### Vector-field input names and shapes

- Previous behavior: Compose had no built-in vector-field targets. Permissive pipelines could pass through undeclared
  keywords such as `gradient`; strict pipelines required those keywords to be declared.
- New behavior: `gradient`, `displacement`, `vector_field`, `normal`, and their `3d` variants are recognized
  spatial targets. Vector-field inputs must have the documented component count, layout, and dtype, and must match
  the spatial grid when shape checks are enabled. Strict Compose rejects `None` fields.
- Affected users: Pipelines adding dense vector annotations and permissive pipelines using a new target name for
  unrelated data.
- Migration: Pass dense fields as float16 or float32 with the documented component axis. Move unrelated metadata
  into `user_data`, for example `pipeline(image=image, user_data={"gradient": metadata})`. Omit absent fields
  in strict pipelines.
- First affected version: unreleased
- Contract: [Vector-field layouts and dtypes](../vector-fields.md#layouts-and-dtypes).
- Tests: [Vector-field validation](../../tests/test_vector_field_targets.py).

## Geometry

### Dense fields follow sampled geometry

- Previous behavior: Dense vectors had no dedicated geometry. Aliasing them to images or masks treated components as
  scalar channels.
- New behavior: Supported transforms share the sampled geometry with the field and update its components according
  to its meaning. Interpolation renormalizes finite neighbors, and out-of-bounds locations become NaN. Dropout keeps
  fields as-is. CopyAndPaste, Mosaic, OverlayElements, Morphological, and PixelSpread skip their entire node when a
  vector field is present.
- Affected users: Optical-flow, surface-normal, local-vector, and gradient pipelines for images and volumes.
- Migration: Use a semantic target, for example `pipeline(image=image, displacement=flow)` for flow in pixel
  coordinates. Check each transform's declared support before adding fields to an existing pipeline.
- First affected version: unreleased
- Contract: [Vector-field geometry and supported routes](../vector-fields.md).
- Tests: [Field geometry](../../tests/functional/test_vector_field_geometry.py),
  [routing, replay, and Tensor conversion](../../tests/test_vector_field_targets.py).

## Determinism

### Reduced-precision CUDA normalization

- Previous behavior: InputNormalizer used PyTorch operations with a full-size FP32 intermediate before casting.
- New behavior: Eligible CUDA inputs use fused Triton kernels that write FP16 or BF16 directly. Fixed normalization
  retains FP32 arithmetic. Per-sample modes use tiled FP32 statistics, so rounding can differ from PyTorch.
- Affected users: InputNormalizer on Linux with Triton and CUDA compute capability 8.0 or later, for eligible
  combinations of layout, input dtype, and reduced-precision output.
- Migration: Existing calls select the route automatically. Use tolerance-based comparisons for per-sample
  normalization. Pin 2.6.1 when reproducing exact outputs from that release.
- First affected version: unreleased
- Contract: [CUDA input normalization](../maintaining/correctness-contracts.md#cuda-input-normalization).
- Tests: [InputNormalizer CUDA parity and fallback](../../tests/test_batch_normalization.py).

## Constructor defaults

none

## Deprecations

none

## Regression vectors

none
