# Unreleased behavior changes

Release baseline: 2.6.4

Follow the [behavior compatibility policy](../maintaining/behavior-compatibility.md) when updating this inventory.
Earlier changes shipped in [2.6.4](2.6.4.md), [2.6.3](2.6.3.md), [2.6.2](2.6.2.md), [2.6.1](2.6.1.md),
and [2.6.0](2.6.0.md).

## Validation

none

## Geometry

### Affine supports raster targets with different sizes

- Previous behavior: Affine required a common height and width even with `is_check_shapes=False`. Its applied
  configuration also stored percentage translations in pixels and saved shear with the opposite sign.
- New behavior: With shape checks disabled, Affine shares the sampled parameters across raster targets and computes
  the warp for each height and width. Bounding boxes, keypoints, and vector fields still require a common size.
  New applied configurations preserve percentage translations and the sampled shear in constructor units.
- Affected users: Pipelines that augment rasters of different sizes in one call or reconstruct Affine from applied
  configurations with percentage translation or shear.
- Migration: Set `is_check_shapes=False` for mixed raster sizes. Recapture applied configurations that used percentage
  translation or shear if you need to reproduce the original transform.
- First affected version: unreleased
- Contract: [Affine on rasters of different sizes](../geometry.md#affine-on-rasters-of-different-sizes).
- Tests: [Geometry contracts](../../tests/contracts/test_geometry_contract.py).

### CPU tensor fallback exposes overlapping storage to PyTorch

- Previous behavior: A NumPy fallback could return a CPU tensor that shared memory with its input while PyTorch
  treated the two tensors as separate storages.
- New behavior: Albucore 0.3.10 preserves the original Tensor storage when the fallback reuses a compatible view.
  PyTorch can detect overlapping reads and writes, including a copy from a transposed view into its input.
- Affected users: CPU Tensor pipelines that write results back into input storage.
- Migration: Copy from `result.clone()` when a transformed view overlaps the destination.
- First affected version: unreleased
- Contract: [Conversion and storage](../design/numpy-tensor-routing.md#conversion-and-storage).
- Tests: [CPU Tensor fallback storage](../../tests/test_tensor_bridge.py).

### Exact geometry preserves vector field components

- Previous behavior: Exact symmetries and whole-sample translations could turn a partially non-finite vector entirely
  into NaN, normalize surface normals, or round displacement components through endpoint arithmetic.
- New behavior: Exact symmetries permute components and apply sign changes in the input dtype. Whole-sample translations
  preserve overlapping vectors as supplied. Both preserve partial non-finite vectors and normal lengths.
- Affected users: Training pipelines that used crop, pad, flips, or quarter turns to normalize or sanitize vector fields.
- Migration: Prepare unit normals and replace partially non-finite vectors with all-NaN vectors when loading data if
  your model requires those conventions.
- First affected version: unreleased
- Contract: [Exact vector field geometry](../vector-fields.md#exact-geometry).
- Tests: [Vector field geometry](../../tests/functional/test_vector_field_geometry.py).

## Determinism

none

## Constructor defaults

none

## Deprecations

none

## Regression vectors

none
