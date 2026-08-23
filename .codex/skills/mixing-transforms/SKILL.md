---
name: mixing-transforms
description: Policy for AlbumentationsX transforms that combine multiple images or objects. Use when implementing, reviewing, or using Mosaic, CopyAndPaste, OverlayElements, HistogramMatching, PixelDistributionAdaptation, or other mixing transforms.
---

# Mixing Transforms Policy

Apply this skill when implementing, reviewing, or using transforms that combine data from multiple
images: `Mosaic`, `CopyAndPaste`, `OverlayElements`, `HistogramMatching`, `PixelDistributionAdaptation`, etc.

---

## 1. The caller owns the donor pool

The caller supplies donor records under `metadata_key`; a transform never accesses a dataset, loader, or global donor
source. A normal Mosaic call supplies the needed donors, and every valid supplied donor is used.

Mosaic handles caller mistakes deterministically within that pool:

- if there are more valid donors than visible cells need, it samples only the surplus away with the invocation's
  `SamplingContext` RNG;
- if there are fewer, it fills remaining cells by replicating the primary item.

This preserves user control over the candidate set while keeping the public transform usable with an oversized or
undersized list.

```python
# CORRECT — caller provides candidate donor records
donors = [dataset[random.choice(indices)] for _ in range(n)]
result = transform(image=image, mosaic_metadata=donors)

# INCORRECT — transform reaches into a dataset on its own
result = MosaicWithSampling(dataset=dataset)(image=image)
```

---

## 2. Metadata format: `list[dict]`

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

---

## 3. Label fields: `bbox_labels` and `keypoint_labels` (dicts)

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
    "bbox": [10, 20, 50, 80],        # same coord_format as BboxParams
    "bbox_labels": {
        "class_id": 3,
        "is_crowd": 0,
    },
    "keypoints": [[25, 40]],         # same coord_format as KeypointParams
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

---

## 4. Coordinates use the same format as `BboxParams` / `KeypointParams`

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

---

## 5. `metadata_key` pattern

Every mixing transform exposes `metadata_key: str` in its constructor and lists it in
`targets_as_params`. This ensures `Compose` validates that the key is present.

```python
@property
def targets_as_params(self) -> list[str]:
    return [self.metadata_key]
```

---

## 6. Empty metadata is transform-specific

Do not impose a universal no-op rule on mixing transforms. `CopyAndPaste` returns no-op parameters when it has no
usable donor. `Mosaic` instead creates its remaining visible cells from replicated primary data, so empty or missing
metadata can still produce a mosaic. State this behavior in the transform's public docstring and test it explicitly.

```python
# CopyAndPaste: no donor means no change.
if not usable_donors:
    return self._no_op_params()

# Mosaic: remaining cells use copies of the primary item.
final_items = [primary, *usable_donors, *replicated_primary_items]
```
