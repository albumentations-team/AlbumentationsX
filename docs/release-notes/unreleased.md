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

### OpticalDistortion maps fisheye annotations in continuous coordinates

- Previous behavior: OpticalDistortion rasterized boxes to masks and could return original coordinates for a box
  whose entire support left the frame. Keypoints used a raster mask or an inverse pixel grid.
- New behavior: For a full-resolution fisheye map with constant border and a coefficient in `[-0.3, 0.3]`,
  OpticalDistortion inverts the radial map at keypoints and along sampled box edges. It refits boxes to their
  visible contours and removes annotations that leave the frame.
- Affected users: Pipelines that pass keypoints, HBBs or OBBs to `OpticalDistortion(mode="fisheye")` in this range.
- Migration: Update cached annotations or expected boxes and keypoints using the new transform. Pin the previous
  version when reproducing an augmentation run that relied on raster-rounded coordinates or retained boxes.
- First affected version: unreleased
- Contract: [OpticalDistortion annotation geometry](../geometry.md#opticaldistortion-annotation-geometry).
- Tests: [OpticalDistortion annotation tests](../../tests/test_optical_distortion_annotations.py).

## Determinism

none

## Constructor defaults

none

## Deprecations

none

## Regression vectors

none
