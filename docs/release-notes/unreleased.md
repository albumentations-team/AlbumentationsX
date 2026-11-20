# Unreleased behavior changes

Release baseline: 2.6.6

Follow the [behavior compatibility policy](../maintaining/behavior-compatibility.md) when updating this inventory.
Earlier changes shipped in [2.6.6](2.6.6.md), [2.6.5](https://github.com/albumentations-team/AlbumentationsX/releases/tag/2.6.5),
[2.6.4](https://github.com/albumentations-team/AlbumentationsX/releases/tag/2.6.4), [2.6.3](https://github.com/albumentations-team/AlbumentationsX/releases/tag/2.6.3),
[2.6.2](2.6.2.md), [2.6.1](2.6.1.md), and [2.6.0](2.6.0.md).

## Validation

### OpenCV import requires version 5.1 or later

- Previous behavior: Importing `albumentations` did not reject OpenCV versions below 5.1, and the OpenCV extras
  allowed older wheels.
- New behavior: Import raises an actionable `ImportError` for native OpenCV versions below 5.1 or PyPI wheels
  below 5.1.0.95. The OpenCV extras require wheels at or above 5.1.0.95.
- Affected users: Installations that reuse OpenCV 4.x or an older 5.1 wheel.
- Migration: Update the selected OpenCV variant to 5.1 or later; PyPI wheels must be at least 5.1.0.95. For
  example, install `albumentationsx[headless]` or the matching GUI or contrib extra.
- First affected version: unreleased
- Contract: [OpenCV runtime minimum](../maintaining/support-policy.md#opencv-policy).
- Tests: [OpenCV import requirements](../../tests/test_import_requirements.py).

### Compose keeps bound instances aligned when a transform drops boxes

- Previous behavior: Filtering bbox rows during preprocessing or inside a nested container could leave masks or
  grouped keypoints attached to removed or different instances, especially with `check_each_transform=False`.
- New behavior: Compose aligns masks and grouped keypoints to surviving bbox IDs at input and transform boundaries,
  including nested execution and deferred filtering.
- Affected users: Pipelines with `instance_binding` where input filtering, transforms, or bbox processors remove
  rows.
- Migration: No configuration change is required. Regenerate cached outputs or expectations that captured
  misaligned instances.
- First affected version: unreleased
- Contract: [Instance binding](../design/instance_binding.md#2-single-positional-alignment-invariant).
- Tests: [Instance binding tests](../../tests/test_instance_binding.py).

## Geometry

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
