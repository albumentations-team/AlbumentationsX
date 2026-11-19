# Unreleased behavior changes

Release baseline: 2.6.5

Follow the [behavior compatibility policy](../maintaining/behavior-compatibility.md) when updating this inventory.
Earlier changes shipped in [2.6.5](2.6.5.md), [2.6.4](2.6.4.md), [2.6.3](2.6.3.md), [2.6.2](2.6.2.md), [2.6.1](2.6.1.md),
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

### Dropout inpainting honors every nonzero mask value

- Previous behavior: Converting a numeric mask to `uint8` could turn a nonzero value such as `256` into zero,
  leaving that pixel unchanged.
- New behavior: Dropout inpainting treats every nonzero mask value as foreground.
- Affected users: Direct callers of `fill_masked_holes` with a numeric dropout mask and an inpainting fill.
- Migration: Pass the same mask to use the corrected behavior. Cast the mask to `uint8` before calling the function
  to reproduce the old pixel selection.
- First affected version: unreleased
- Contract: [Interpolation, borders, and fill](../geometry.md#interpolation-borders-and-fill).
- Tests: [Dropout inpainting tests](../../tests/functional/test_dropout.py).

## Determinism

none

## Constructor defaults

none

## Deprecations

none

## Regression vectors

none
