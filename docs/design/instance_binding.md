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

The pipeline is held together by **two structural invariants** enforced by the code's
construction, not by per-transform discipline. Wrong states are unrepresentable; there is no
"rule" to follow.

## Structural contracts

Both contracts are zero-cost at runtime: one comes from a `NewType` brand, the other from a
single resync hook in `Compose`'s transform loop plus a mirror-drop in
`check_data_post_transform`.

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

### 2. Single positional-alignment invariant

After every transform's `__call__` returns AND after `check_data_post_transform` runs, the
following hold simultaneously when `instance_binding` is active:

1. `len(data["masks"]) == len(data["bboxes"])` (when both bound).
2. Row `i` of masks describes the same instance as row `i` of bboxes.
3. `data["keypoints"][:, -1]` (the `_instance_id` column) contains only ids present in
   `data["bboxes"][:, -1]`. No orphan keypoints.
4. Going into the next transform, `data["bboxes"][:, -1] == arange(N)`.

Violating (1) or (3) raises `RuntimeError` immediately from `_resync_instance_ids`. In legacy
mode (`Compose(..., strict_instance_invariant=False)`, kept for one minor version) it
downgrades to a `UserWarning` and falls through to the previous permissive behavior.

#### How the invariant is enforced (one chokepoint, three layers)

```mermaid
flowchart TB
  subgraph T [transform t.__call__]
    direction TB
    GP[get_params_dependent_on_data computes shared keep_mask] --> AB[apply_to_bboxes uses keep_mask]
    GP --> AM[apply_to_masks uses same keep_mask]
    GP --> AK[apply_to_keypoints filters by surviving _instance_id]
    AB --> OUT[t OUT: positional 1:1 always]
    AM --> OUT
    AK --> OUT
  end
  OUT --> CHK[check_data_post_transform]
  subgraph CHK_DETAIL [check_data_post_transform]
    direction TB
    PRE[Pre-filter realignment for legacy in-transform filters]
    BPF[BboxProcessor.filter_with_keep_mask]
    MIRROR[Mirror keep_mask: masks positional, keypoints by surviving id]
    PRE --> BPF --> MIRROR
  end
  CHK --> RES[_resync_instance_ids]
  RES --> ASSERT[assert len(masks) == len(bboxes); rebase ids = arange(N)]
  ASSERT --> NEXT[next transform]
```

##### Layer 1 — shared survival decision in transforms that filter (Phase 2)

Mosaic and CopyAndPaste compute a single `keep_mask` (or surviving id set) inside
`get_params_dependent_on_data` and ferry it through `params` to all three apply methods. This
removes the historical bug where `apply_to_bboxes` filtered N→K bboxes but `apply_to_masks`
emitted all N masks.

- `Mosaic.get_params_dependent_on_data` calls `_compute_mosaic_survival(...)` once. The
  resulting `keep_mask`, `surviving_instance_ids`, and `filtered_bboxes` are read by
  `apply_to_{bboxes,masks,keypoints}`.
- A per-cell pre-pass `_filter_cell_masks_to_surviving_bboxes` runs *before*
  `remap_mosaic_instance_label_ids` so each cell's mask stack is already aligned with its
  surviving bboxes; otherwise concatenation across cells would produce orphan mask rows.
- `CopyAndPaste.apply_to_bboxes` re-stamps `_bbox_instance_id = arange(N)` at output;
  `apply_to_keypoints` calls `_restamp_keypoint_ids` with the matching old→new table. The
  legacy "sparse-id positional" layout no longer exists.

##### Layer 2 — mirror-drop in the processor layer (Phase 3 / 3b)

`BboxProcessor.filter_with_keep_mask` returns `(filtered_bboxes, keep_mask: np.ndarray[bool])`.
The legacy `BboxProcessor.filter` and module-level `filter_bboxes` are kept as one-line
wrappers around this primitive (public API preserved).

`Compose.check_data_post_transform` delegates to `_bbox_filter_with_mirror` whenever the
`BboxProcessor` is in `self.check_each_transform` and `instance_binding` is active. That
helper does two stages:

1. **Pre-filter realignment.** Some transforms (CoarseDropout, Crop with `min_area`, etc.)
   drop bbox rows inside their own `apply_to_bboxes` without touching the mask stack. At this
   point `len(masks) > len(bboxes)` and the surviving bboxes still carry their original
   `_bbox_instance_id` in the last column. That id is the row index into the still-id-indexed
   mask stack, so we fancy-index masks down to the surviving id set and drop keypoints whose
   `_kp_instance_id` is no longer present. This collapses the legacy "id-indexed masks +
   sparse ids" layout into the positional layout the rest of the method assumes.
2. **BboxProcessor filter + post-filter mirror.** Standard `filter_with_keep_mask` on bboxes;
   mirror the resulting `keep_mask` positionally onto masks and by surviving id onto
   keypoints.

##### Layer 3 — `_resync_instance_ids` (Phase 4)

Once Layers 1 and 2 are in place, the resync's only job is to:

1. Assert `len(masks) == len(bboxes)` (raises `RuntimeError` in strict mode, `UserWarning` in
   legacy mode).
2. Translate `_kp_instance_id` from old bbox ids to the new positional ids.
3. Stamp `_bbox_instance_id = arange(N)` so the next transform sees a dense namespace.

The pre-2.2.2 snapshot machinery (`_snapshot_pre_processor_bbox_ids`,
`_mask_positions_for_surviving_ids`) is gone. The resync no longer encodes any recovery branch
for dual mask layouts; that case is impossible by construction now.

### Consequence: a single repack path

Because the invariants above always hold going into `_repack_instances`, the repack path
collapses to one branch — `bbox_row_idx == mask_row_idx == row_idx` for every surviving
instance. There are no fallback branches, no per-id lookups, no "what if `len(masks)`
disagrees with `len(bboxes)`" cases.

## Id namespace

There is one logical instance-id namespace (last column of both bboxes and keypoints
arrays). Inside the `data` dict it ferries under two distinct keys because the bbox
per-row id list and the keypoint per-row id list can have different lengths and both
must coexist between unpack and per-processor preprocess:

- `_BBOX_INSTANCE_ID = "_bbox_instance_id"` — one entry per bbox row.
- `_KP_INSTANCE_ID = "_kp_instance_id"` — one entry per keypoint row, valued at the parent
  bbox's id.

Both are members of the constant set `_INSTANCE_ID_FERRY_KEYS` so `_clean_params_dict` and
similar utilities filter them out as a group rather than naming each one.

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

## Compose Parameters

```python
A.Compose(
    transforms,
    bbox_params=...,
    keypoint_params=...,
    instance_binding=["masks", "bboxes", "keypoints"],
    strict_instance_invariant=True,  # default since 2.2.2
)
```

Valid binding targets: `"mask"`, `"masks"`, `"bboxes"`, `"keypoints"`. Minimum 2. `"mask"` and
`"masks"` are mutually exclusive.

`strict_instance_invariant=True` (default) raises `RuntimeError` from `_resync_instance_ids`
on contract violations. Setting it to `False` downgrades to a `UserWarning` and falls back to
legacy permissive behavior; this escape hatch exists for one minor version so users with
custom transforms that violate the row-alignment contract have time to migrate.

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
2. `Compose._resync_instance_ids` runs — asserts the row-alignment invariant and rebases ids.
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

## Authoring custom DualTransforms

If you write a `DualTransform` whose `apply_to_bboxes` drops rows (min-area culling,
out-of-frame removal, etc.), your `apply_to_masks` MUST drop the corresponding rows. The
canonical pattern is to compute the keep-mask once in `get_params_dependent_on_data` and ferry
it to both apply methods via `params` (see `Mosaic` / `CopyAndPaste`).

The default `BasicTransform.apply_to_masks` is total — it preserves alignment for transforms
whose `apply_to_mask` doesn't drop rows. The `BboxProcessor` mirror handles the case where the
processor's visibility/area filter is the SOLE drop cause. Transform-internal filters need
their own shared keep-mask plumbing; otherwise `_resync_instance_ids` raises.

## Where to look in the code

| Concern                                  | File                                                                          |
|------------------------------------------|-------------------------------------------------------------------------------|
| `StackedMasks4D` brand                   | `albumentations/core/type_definitions.py`                                     |
| `_make_stacked_masks` factory            | `albumentations/core/composition.py`                                          |
| Resync + assert hook                     | `albumentations/core/composition.py` (`Compose._resync_instance_ids`)         |
| Bbox-processor mirror                    | `albumentations/core/composition.py` (`Compose._bbox_filter_with_mirror`)     |
| `filter_bboxes_with_mask` primitive      | `albumentations/core/bbox_utils.py`                                           |
| Mosaic shared keep-mask                  | `albumentations/augmentations/mixing/transforms.py` (`Mosaic._compute_mosaic_survival`) |
| CopyAndPaste dense-id output             | `albumentations/augmentations/mixing/transforms.py` (`CopyAndPaste.apply_to_bboxes`)    |
| `apply_to_masks` row-alignment contract  | `albumentations/core/transforms_interface.py` (`BasicTransform.apply_to_masks`)         |
| Property-based invariant fuzz            | `tests/test_instance_binding.py` (`TestPipelineInvariantsHypothesis`)         |
| Contract test for broken transforms      | `tests/test_instance_binding.py` (`TestStructuralInvariantContract`)          |
