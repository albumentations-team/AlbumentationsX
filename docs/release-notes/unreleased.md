# Unreleased behavior changes

Release baseline: 2.6.5

Follow the [behavior compatibility policy](../maintaining/behavior-compatibility.md) when updating this inventory.
Earlier changes shipped in [2.6.5](2.6.5.md), [2.6.4](2.6.4.md), [2.6.3](2.6.3.md), [2.6.2](2.6.2.md), [2.6.1](2.6.1.md),
and [2.6.0](2.6.0.md).

## Validation

### Clear observations after rejected trace snapshot targets

- Previous behavior: After an observed call, rejecting an unknown snapshot target in `Compose.run_with_trace()` or
  `ReplayCompose.run_with_trace()` left the preceding sample's applied params and config visible through leaf getters.
- New behavior: This rejection clears the caller's completed observation, so `get_applied_params()` and
  `get_applied_config()` return empty dictionaries. Earlier returned results, other callers, and an active outer
  invocation retain their data. The error and RNG continuation are unchanged. This corrects stale observation exposure.
- Affected users: Callers inspecting applied params or config after catching an unknown trace snapshot-target error,
  including nested Compose and ReplayCompose pipelines.
- Migration: No pipeline configuration change is needed. Save the observed configuration before the next call, for example
  `previous_config = leaf.get_applied_config()` after a successful observed call.
- First affected version: unreleased
- Contract: [Invocation observation ownership](../design/compose-greenfield-architecture.md#permanent-contracts).
- Tests: [Trace rejection and retained artifacts](../../tests/test_composition_tracing.py),
  [caller and active-invocation isolation](../../tests/test_compose_reentrancy.py).

### Compose keeps bound instances aligned when a transform drops boxes

- Previous behavior: A transform that removed bbox rows could leave extra mask rows or orphan keypoints when called
  inside a nested container or with `check_each_transform=False`.
- New behavior: Compose aligns masks and grouped keypoints to the surviving bbox IDs before the next transform.
- Affected users: Pipelines with `instance_binding` and transforms that remove bbox rows directly.
- Migration: Keep the existing instance binding and bbox settings.
- First affected version: unreleased
- Contract: [Instance binding](../design/instance_binding.md#2-single-positional-alignment-invariant).
- Tests: [Instance binding tests](../../tests/test_instance_binding.py).

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
  OpticalDistortion uses the inverse radial map for keypoints and sampled box edges when the map is monotone in
  the frame. It fits boxes to the visible contour and applies `BboxParams.min_visibility` to the retained area
  fraction. `KeypointParams.remove_invisible` controls whether out-of-frame keypoints and their labels are kept.
- Affected users: Pipelines that pass keypoints, HBBs or OBBs to `OpticalDistortion(mode="fisheye")` in this range.
- Migration: Update cached annotations or expected boxes and keypoints using the new transform. Pin the previous
  version when reproducing an augmentation run that relied on raster-rounded coordinates or retained boxes.
- First affected version: unreleased
- Contract: [OpticalDistortion annotation geometry](../geometry.md#opticaldistortion-annotation-geometry).
- Tests: [OpticalDistortion annotation tests](../../tests/test_optical_distortion_annotations.py).

### GridElasticDeform handles cells without an invertible homography

- Previous behavior: Collapsed cells could produce image and annotation results from a non-invertible warp.
- New behavior: Cells with collinear destination corners stay at their source rectangles. Source rectangles without
  area contribute no warp. Images and annotations use the same recovered grid.
- Affected users: Pipelines and saved replays in which a fine grid or a large displacement collapses a cell.
- Migration: Regenerate expected outputs for samples with collapsed cells.
- First affected version: unreleased
- Contract: [Grid elastic deformation](../geometry.md).
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
