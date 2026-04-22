# Instance Binding

## Problem

Albumentations treats `masks`, `bboxes`, and `keypoints` as independent targets. When bbox
filtering removes an instance (e.g., after a crop makes it too small), the corresponding mask
and keypoints stay behind, causing index misalignment. Pose estimation models also expect a
fixed number of keypoints per surviving instance, but `filter_keypoints` may drop individual
out-of-bounds keypoints independently.

## Solution

A `Compose(..., instance_binding=[...])` parameter plus a structured `instances` input format.
Users pass `instances` as a list of dicts, each representing one object. `Compose` unpacks
them into flat arrays for the existing pipeline, then repacks the survivors into the same
format.

The pipeline is held together by **two structural invariants** that are enforced by the code's
construction, not by per-transform discipline. Wrong states are unrepresentable; there is no
"rule" to follow.

## Structural contracts

Both contracts are zero-cost at runtime: one comes from a `NewType` brand, the other from a
single resync hook in `Compose`'s transform loop.

### 1. Canonical 4-D mask shape (`StackedMasks4D`)

Every value bound to `data["masks"]` while the pipeline is running is a `StackedMasks4D` with
shape `(N, H, W, C)`. `StackedMasks4D` is a `typing.NewType` declared in
`albumentations.core.type_definitions`:

```python
StackedMasks4D = NewType("StackedMasks4D", np.ndarray)
```

`NewType` is identity at runtime (zero overhead). Static checkers treat it as distinct from
`np.ndarray`, so any function annotated `-> StackedMasks4D` cannot return a raw ndarray
without going through the single construction site:

- `albumentations.core.composition._make_stacked_masks(rows: list[np.ndarray]) -> StackedMasks4D`

This factory normalizes per-instance `(H, W)` or `(H, W, C)` arrays into a single
`(N, H, W, C)` stack and brands the result. It is the **only** place the trailing channel
dimension is added or the brand is minted.

Constructed at:
- `Compose._unpack_masks` (preprocess) — turns the user's instance dicts into `data["masks"]`.
- `Compose._init_empty_instance_data` — supplies a zero-row 4-D placeholder when no instances
  are provided.
- `Mosaic.apply_to_masks` and `CopyAndPaste.apply_to_masks` — preserve the brand on output.

Because no other code can construct a `StackedMasks4D`, **`data["masks"]` cannot be 3-D**.
Every `apply_to_masks` override (Affine, Crop, Flip, Rotate, Perspective, ToTensorV2, …) takes
`StackedMasks4D` and returns `StackedMasks4D`. Their bodies index `masks.shape[3]`
unconditionally — there is no `ndim == 4` branch to forget.

mypy enforces this on
`albumentations.{core.composition, augmentations.mixing.*, augmentations.geometric.{transforms,flip}, augmentations.crops.transforms, pytorch.transforms}`.
Returning a raw `np.ndarray` from any `apply_to_masks` override breaks CI.

### 2. Row-alignment via `Compose._resync_masks_to_bboxes`

Between every transform, when `instance_binding` is active, the following holds:

- `len(data["masks"]) == len(data["bboxes"])`
- the `_bbox_instance_id` column equals `range(N)` (id == row position)
- the `_kp_instance_id` column points at the same row positions

This is maintained by a single chokepoint in the transform loop,
`Compose._resync_masks_to_bboxes`, which runs after each transform's processor postprocess:

1. Read `bbox_ids = data["bboxes"][:, _bbox_instance_id_column]`.
2. If they already equal `range(N)`, return immediately (the common case for non-mixing
   transforms, microseconds).
3. Otherwise, remap `_kp_instance_id` through the same `old → new` table and rewrite the
   `_bbox_instance_id` column to `range(N)`.
4. If `len(masks) != N`, fancy-index `data["masks"] = masks[bbox_ids]` to realign rows.

Because the invariant is restored before any next transform runs, no `apply_to_masks`,
`apply_to_bboxes`, or `apply_to_keypoints` override needs to handle filtering or reorder
artifacts. **Mixing transforms emit `[surviving; pasted]` in row order and stop there**;
they do not maintain `_bbox_instance_id` themselves. The Compose hook does the rebasing once,
in one place.

This kills the entire class of "id-as-position desync" bugs (e.g. `IndexError` from
`Compose._repack_mask_into` after `CopyAndPaste + Perspective`): there is nothing to desync
because the rebase happens between every pair of transforms.

### Consequence: a single repack path

Because the invariants above always hold going into `_repack_instances`, the repack path
collapses to one branch — `bbox_row_idx == mask_row_idx == row_idx` for every surviving
instance. There are no fallback branches, no per-id lookups, no "what if `len(masks)`
disagrees with `len(bboxes)`" cases.

## Instance Dict Schema

```python
{
    "mask": np.ndarray,                      # (H, W) binary mask
    "bbox": np.ndarray,                      # (4,) or (5+,) per coord format
    "keypoints": np.ndarray,                 # (K, C) e.g. (K, 2) for xy
    "bbox_labels": {"class": "cat", ...},    # scalars, keys = BboxParams.label_fields
    "keypoint_labels": {"name": [...], ...}, # lists len==K, keys = KeypointParams.label_fields
}
```

Which keys are required depends on `instance_binding`:

- `"masks"` or `"mask"` → requires `"mask"` key
- `"bboxes"` → requires `"bbox"` key
- `"keypoints"` → requires `"keypoints"` key (use a zero-row array such as
  `np.empty((0, 2))` for no points)

`bbox_labels` / `keypoint_labels` are present when the corresponding `Params.label_fields`
are non-empty. They use nested dicts to avoid name collisions (both bbox and keypoint can
have a `"class"` field).

## Compose Parameter

```python
A.Compose(
    transforms,
    bbox_params=...,
    keypoint_params=...,
    instance_binding=["masks", "bboxes", "keypoints"],
)
```

Valid targets: `"mask"`, `"masks"`, `"bboxes"`, `"keypoints"`. Minimum 2. `"mask"` and
`"masks"` are mutually exclusive.

## Preprocessing (Unpack)

1. Pop `data["instances"]` list.
2. Build `data["masks"]` via `_make_stacked_masks([inst["mask"] for inst in instances])` →
   shape `(N, H, W, C)` (canonical), or concat as channels → `data["mask"]` `(H, W, C)`.
3. Stack bboxes → `data["bboxes"]` `(N, 4+)`.
4. Concatenate keypoints → `data["keypoints"]` `(M, C)`.
5. Flatten `bbox_labels` → `data[field_name]` as list of N scalars.
6. Flatten `keypoint_labels` → `data[field_name]` as concatenated list of M values.
7. Inject hidden label fields: `_bbox_instance_id = [0..N-1]`,
   `_kp_instance_id = [0, 0, …, 1, 1, …]` (one per keypoint row, repeated by parent
   instance's bbox id).

The processor `preprocess` then `numpy.hstack`s these label columns onto the bbox / keypoint
arrays, so `_bbox_instance_id` lives in the last column of `data["bboxes"]` mid-pipeline.

## Postprocessing (Repack)

1. Processor `postprocess` runs (filter, format convert, split label fields back out).
2. `Compose._resync_masks_to_bboxes` runs — guarantees the row-alignment invariant.
3. `Compose._repack_instances` walks `range(N)` once: every row index is simultaneously the
   bbox row, the mask row, and the keypoint group id.
4. Hidden fields and flat arrays are removed; `data["instances"]` is the final list of
   instance dicts.

## Anchor

- When `"bboxes"` is bound, bbox filtering (`min_area`, `min_visibility`, etc.) drives
  instance removal.
- When `"bboxes"` is not bound, nothing drives removal; binding only enforces correspondence.

## Keypoint Behavior

When keypoints are bound, `remove_invisible` and `check_each_transform` on `KeypointParams`
are forced to `False`. Keypoints may land outside image bounds; the only removal is via
instance survival (driven by their parent bbox).

## Where to look in the code

| Concern                                  | File                                                                          |
|------------------------------------------|-------------------------------------------------------------------------------|
| `StackedMasks4D` brand                   | `albumentations/core/type_definitions.py`                                     |
| `_make_stacked_masks` factory            | `albumentations/core/composition.py`                                          |
| Transform loop & resync hook             | `albumentations/core/composition.py` (`Compose._resync_masks_to_bboxes`)      |
| Mosaic mask assembly (always 4-D)        | `albumentations/augmentations/mixing/functional.py` (`assemble_mosaic_instance_masks_stack`) |
| CopyAndPaste mask paste (always 4-D)     | `albumentations/augmentations/mixing/transforms.py` (`CopyAndPaste.apply_to_masks`) |
| Property-based regression tests          | `tests/test_instance_binding.py` (`TestPipelineInvariantsHypothesis`)         |
