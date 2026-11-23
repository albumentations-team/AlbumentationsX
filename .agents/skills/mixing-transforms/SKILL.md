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

## 2. Metadata contract

Read [the metadata format examples](references/metadata-format.md) before writing or reviewing donor records.

- Every mixing transform takes auxiliary data as `list[dict]` under a constructor `metadata_key`, and lists that key in
  `targets_as_params` so `Compose` validates its presence. One dict is one full image for Mosaic and one object for
  CopyAndPaste.
- `bbox_labels` and `keypoint_labels` are dicts whose keys exactly match `BboxParams.label_fields` and
  `KeypointParams.label_fields`. Values are scalars for one object (CopyAndPaste) and per-row lists for a full image
  (Mosaic).
- Donor bboxes and keypoints use the same `coord_format` as the `Compose` processors; `preprocess()` converts them, so
  never convert manually.

---

## 3. Empty metadata is transform-specific

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

## 4. Vector field inputs

`CopyAndPaste`, `Mosaic`, and `OverlayElements` do not define how mixing transforms vector fields. If any canonical
field or `additional_targets` alias is present, the entire node is skipped before sampling or reading donor metadata,
and the trace reports `skipped_unsupported_target`. [Vector fields](../../../docs/vector-fields.md) owns this policy;
keep it and `tests/test_vector_field_targets.py` synchronized when changing one of these transforms.
