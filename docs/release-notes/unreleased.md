# Unreleased

- `LongestMaxSize`, `SmallestMaxSize`, and `LetterBox` now refit oriented
  bounding boxes when rounding the output dimensions produces different
  scales along the two axes. For example, resizing from `7 × 11` to `3 × 4`
  can change the angle and side lengths of an oriented box. Regenerate
  cached annotations for affected samples to match the corrected geometry.
  For oriented boxes, use `BboxParams(bbox_type="obb", ...)` in `Compose`.
  [#490](https://github.com/albumentations-team/AlbumentationsX/issues/490)
- `CropAndPad(keep_size=True)`, `RandomSizedCrop`, `RandomResizedCrop`, and
  `RandomSizedBBoxSafeCrop` now refit oriented boxes when resizing the
  intermediate frame by different amounts on the two axes. Recompute cached
  boxes for affected augmentations to pick up the corrected angles and extents.
  [#490](https://github.com/albumentations-team/AlbumentationsX/issues/490)
- `CropAndPad` now keeps one pixel on each axis that a crop request would
  otherwise empty, before applying padding or resizing. For example,
  `px=-64` with `keep_size=False` on a `7 × 9` frame now returns its center pixel
  instead of raising a crop-window error. Refresh cached crops and annotations
  if the combined crop amounts can exhaust an axis.
  [#490](https://github.com/albumentations-team/AlbumentationsX/issues/490)
- `Perspective` fits oriented boxes in output pixels and shares one output
  homography across rasters and annotations. Refresh cached oriented boxes
  from rectangular frames to pick up the corrected geometry. Existing replay
  records can be reused. [#490](https://github.com/albumentations-team/AlbumentationsX/issues/490)
- Remap distortions now rasterize oriented boxes from their rotated corners
  and preserve their geometry on an exact identity map. Refresh cached OBB
  annotations from affected distortion pipelines. [#490](https://github.com/albumentations-team/AlbumentationsX/issues/490)
- When `ReplayCompose` applies a recorded `RandomScale`, `LongestMaxSize`, or
  `SmallestMaxSize` to a differently sized frame, it now maps keypoints and
  oriented boxes using the current frame dimensions. Existing replay records
  can be reused. Refresh any cached expected coordinates for cross-size replay.
  [#490](https://github.com/albumentations-team/AlbumentationsX/issues/490)
- `ReplayCompose` now maps oriented boxes and keypoints from the current frame
  when replaying `Resize` or `LetterBox` on a different input size. The record
  still determines the output size and LetterBox padding. Refresh cached
  annotations from affected replays. [#490](https://github.com/albumentations-team/AlbumentationsX/issues/490)
- Custom transforms that override `apply_with_params` should move parameter
  preparation to `sample_parameters` and apply the resulting values in target
  handlers such as `apply` and `apply_to_mask`. The framework now uses private
  dispatch for sampled parameters.
- Fixed `GridElasticDeform` failing on wide images with horizontal bounding
  boxes. Affected pipelines can reuse their existing replay records.
  [#490](https://github.com/albumentations-team/AlbumentationsX/issues/490)
- Fixed delayed release of input arrays and CPU tensors after `Compose` calls,
  including calls that raise an exception. Optional-channel images, masks,
  batches and volumes no longer depend on cyclic garbage collection for release.
  This addresses [#596](https://github.com/albumentations-team/AlbumentationsX/issues/596).
- **Custom sampler compatibility:** `SamplingContext.invocation` and the
  `invocation=` constructor argument have been removed. Use `sampling.py_random`,
  `sampling.random_generator`, the sampling hook's `data` and `targets`, or
  `SamplingContext.from_seed()` / `from_owner()`. See the
  [migration guide](../contributing/coding_guidelines.md#samplingcontext-compatibility-and-migration).
