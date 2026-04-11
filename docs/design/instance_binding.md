# Instance Binding

## Problem

Albumentations treats `masks`, `bboxes`, and `keypoints` as independent targets. When bbox
filtering removes an instance (e.g., after a crop makes it too small), the corresponding mask
and keypoints stay behind, causing index misalignment. Pose estimation models also expect a
fixed number of keypoints per surviving instance, but `filter_keypoints` may drop individual
out-of-bounds keypoints independently.

## Solution

A new `instance_binding` parameter on `Compose` plus a structured `instances` input format.
Users pass `instances` as a list of dicts, each representing one object. Compose unpacks them
into flat arrays for the existing pipeline, then repacks the survivors into the same format.

No transform changes are required.

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
- `"keypoints"` → requires `"keypoints"` key (use a zero-row array such as `np.empty((0, 2))` for no points)

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
2. Stack masks → `data["masks"]` `(N,H,W)` or concat as channels → `data["mask"]` `(H,W,C)`.
3. Stack bboxes → `data["bboxes"]` `(N,4+)`.
4. Concatenate keypoints → `data["keypoints"]` `(M,C)`.
5. Flatten `bbox_labels` → `data[field_name]` as list of N scalars.
6. Flatten `keypoint_labels` → `data[field_name]` as concatenated list of M values.
7. Inject hidden label fields: `_bbox_instance_id = [0..N-1]`,
   `_kp_instance_id = [0,0,..,1,1,..]` (one per keypoint row).

Then the existing processor `preprocess` horizontally stacks (`numpy.hstack`) these label
columns onto the arrays.

## Postprocessing (Repack)

1. Processor `postprocess` runs (filter, format convert, split label fields back out).
2. Read surviving `_bbox_instance_id` values → determine which instances survived.
3. Index `masks` and `keypoints` by surviving IDs.
4. Rebuild instance dicts with masks, bboxes, keypoints, and label dicts.
5. Remove flat arrays and hidden fields from `data`; set `data["instances"]`.

## Anchor

- When `"bboxes"` is bound, bbox filtering (min_area, min_visibility, etc.) drives
  instance removal.
- When `"bboxes"` is not bound, nothing drives removal; binding only enforces
  correspondence.

## Keypoint Behavior

When keypoints are bound, `remove_invisible` and `check_each_transform` on
`KeypointParams` are forced to `False`. Keypoints may land outside image bounds; the only
removal is via instance survival.
