# Mixing metadata format

Examples of the metadata contract summarized in the `mixing-transforms` skill.

## Metadata format: `list[dict]`

All mixing transforms receive auxiliary data as `list[dict]` under a `metadata_key`. Each dict is
one item (one full image for Mosaic, one object instance for CopyAndPaste). This is consistent
across transforms.

```python
mosaic_metadata = [
    {"image": img1, "mask": mask1, "bboxes": bboxes1, "bbox_labels": {...}},
    {"image": img2, ...},
]

copy_paste_metadata = [
    {"image": src_img, "mask": obj_mask, "bbox": [x1, y1, x2, y2], "bbox_labels": {"class_id": 3}},
    {"image": src_img, "mask": obj_mask2, "bbox_labels": {"class_id": 7}},
]
```


## Label fields: `bbox_labels` and `keypoint_labels` (dicts)

All mixing transforms use the same wrapper dict convention for labels:

- `bbox_labels`: `dict[str, Any]` — maps each label field name (as declared in
  `BboxParams.label_fields`) to its value(s) for this item.
- `keypoint_labels`: `dict[str, Any]` — maps each label field name (as declared in
  `KeypointParams.label_fields`) to its value(s) for this item.

For **CopyAndPaste** (one object per dict), values are scalars (one bbox, one object):

```python
{
    "image": src_image,
    "mask": obj_mask,
    "bbox": [10, 20, 50, 80],  # same coord_format as BboxParams
    "bbox_labels": {
        "class_id": 3,
        "is_crowd": 0,
    },
    "keypoints": [[25, 40]],  # same coord_format as KeypointParams
    "keypoint_labels": {
        "joint_name": "left_eye",
    },
}
```

For **Mosaic** (one full image per dict), values are lists — one entry per bbox/keypoint:

```python
{
    "image": img,
    "bboxes": [[10, 20, 50, 80], [5, 5, 30, 30]],
    "bbox_labels": {
        "class_id": [3, 7],
        "is_crowd": [0, 1],
    },
    "keypoints": [[25, 40], [60, 70]],
    "keypoint_labels": {
        "joint_name": ["left_eye", "nose"],
    },
}
```

**Key rule**: the dict keys in `bbox_labels` / `keypoint_labels` must exactly match what is
declared in `BboxParams(label_fields=[...])` and `KeypointParams(label_fields=[...])`.


## Coordinates use the same format as `BboxParams` / `KeypointParams`

Bboxes and keypoints in metadata dicts must use the **same `coord_format`** as declared in `Compose`.
The processor's `preprocess()` converts them to the internal albumentations format — no manual
conversion needed.

```python
# BboxParams declared with coord_format='pascal_voc'
# → bboxes in metadata must also be pascal_voc [x_min, y_min, x_max, y_max]
copy_paste_metadata = [
    {"image": img, "mask": m, "bbox": [10, 20, 50, 80], "bbox_labels": {"class_id": 3}},
]
```


## `metadata_key` pattern

Every mixing transform exposes `metadata_key: str` in its constructor and lists it in
`targets_as_params`. This ensures `Compose` validates that the key is present.

```python
@property
def targets_as_params(self) -> list[str]:
    return [self.metadata_key]
```
