# Correctness Contracts

This document defines the vocabulary used by the existing transform tests,
regression tests, property tests, and public release reports. It distinguishes
behavior AlbumentationsX guarantees from behavior that is tested structurally
or intentionally left unspecified.

Changes to these guarantees follow the [behavior compatibility policy](behavior-compatibility.md), including migration
evidence and deliberate regression-epoch updates.

The mature verification model is layered:

- parameterized transform sweeps are the primary broad coverage mechanism for
  construction, target routing, dtype/shape behavior, masks, serialization, and
  common parameter sets;
- focused unit and functional tests cover exact algorithms, annotation
  geometry, OBB behavior, 3D behavior, optional targets, and edge cases;
- property tests exercise cross-cutting invariants over generated inputs;
- golden vectors are compact compatibility sentinels for behavior that should
  not drift silently.

Golden vectors are not intended to mirror every transform test. A large golden
dump would be expensive to review and brittle across supported dependency
versions. New public transform-like APIs must instead have a real coverage
route: a parameterized sweep route, an explicit focused test route, a
composition/tensor route, or a deliberate golden compatibility contract.

## Contract Dimensions

- Transform class and family: image-only, dual 2D, 3D, mixing/reference-data,
  or composition helper.
- Target support: image, images, mask, masks, bboxes, keypoints, volume,
  mask3d, and additional targets.
- Shape contract: NumPy targets are channel-last with an explicit channel dimension inside `Compose`; plain CPU Tensor
  targets are channel-first (`CHW`, `NCHW`, or `CDHW`). Public optional-channel ranks are restored after dispatch.
  The complete table and dtype boundary are in [the CPU Tensor routing design](../design/numpy-tensor-routing.md).
- Dtype contract: outputs preserve documented dtype semantics for the tested
  target.
- Channel contract: transforms document whether arbitrary channels, RGB only,
  or a narrower channel set is supported.
- Randomness contract: fixed seeds reproduce tested outputs.
- Label contract: bbox and keypoint labels stay aligned with surviving
  annotations.
- Geometry and bounds contract. The [2D geometry contract](../geometry.md)
  describes coordinate systems, target alignment, and the bounds enforced by
  the configured annotation filters.
- Serialization contract: supported transforms round-trip through
  serialization and replay checks.

## Box-aware RandomResizedCrop sampling

When `RandomResizedCrop.bbox_preservation` is `"one_box"`, the sampled crop contains one uniformly
selected box across active bbox fields. The `"all_boxes"` mode contains every box. With no boxes, the
transform uses ordinary random-resized-crop sampling. The configured `scale` and aspect-ratio bounds
are relaxed only when needed to satisfy the selected box constraint; `BboxParams` filters still apply
after the crop. The [permanent transform tests](../../tests/test_random_resized_crop.py) cover both
selection modes, aliases, replay, and geometry.

## Macenko stain estimation

`MacenkoNormalizer` fits its optical-density plane from the two eigenvectors associated with the
largest eigenvalues. OpenCV returns these eigenvectors as rows. On the
[synthetic known-stain test](../../tests/functional/test_functional.py), the recovered hematoxylin
and eosin vectors each remain within five degrees of their known directions.

## 3D Orientation Label Mappings

An orientation-reversing `Transform3D` may emit a transform-name event that activates configured semantic mappings.
The augmented output must match a manual annotation of the transformed volume:

- `mask3d` and every target aliased to `mask3d` receive the spatial transform, then a simultaneous semantic-label
  mapping. A swap such as `2: 3, 3: 2` uses the source mask for both assignments.
- `KeypointParams.label_mapping` remaps configured keypoint label fields in the row that already holds the coordinate
  transformed by `Transform3D`. It never moves coordinate rows to express a label mapping.
- Semantic-label remapping does not add, remove, or reorder instance rows. When `instance_binding` is active, masks,
  bounding boxes, and keypoint groups retain their positional alignment.

`semantic_mask_label_mappings` and `KeypointParams.label_mapping` are independent configuration surfaces. A transform
event activates each mapping only when the caller configured it for that target type.

## Settings loading

Package import loads settings from defaults, then merges every entry of `settings.json` only when its JSON root is
an object, then applies environment overrides. Object values and unknown keys are preserved without validation or
coercion. Non-object roots contribute no general settings.

One compatibility exception preserves telemetry opt-outs in legacy arrays accepted by the previous dictionary
merge. If the entire array converts to a dictionary and its final `telemetry` value is falsey (`false`, `0`, `null`,
`""`, `[]`, or `{}`), telemetry is set to Boolean `false` in memory. The last duplicate key wins; other entries are
ignored. Missing or truthy telemetry values do not change defaults. If any element prevents conversion, the entire
array is ignored, including a valid opt-out prefix. Loading never partially applies an array. Prefer an object such
as `{"telemetry": false}` for settings; this exception does not restore general array settings.

`ALBUMENTATIONS_NO_TELEMETRY` and `ALBUMENTATIONS_OFFLINE` independently disable telemetry when set to `1` or
case-insensitive `true`, overriding file settings. Other environment values do not undo a stored opt-out. Without an
object setting, preserved legacy opt-out, or disabling environment override, telemetry defaults to `true`.
Loading never rewrites the file or creates a missing settings file. Missing files, malformed JSON, and `OSError`
raised while opening or reading an existing file retain defaults before environment overrides; other errors keep
their existing propagation behavior. The [settings import tests](../../tests/test_settings_import.py) cover these
boundaries in isolated processes, simulate the caught read error, and check Compose/client delivery with a transport
spy and an enabled positive control without contacting a telemetry endpoint.

## Stability Modes

Golden regression cases declare one stability mode:

- `exact`: exact array values are part of the tested contract.
- `tolerance`: array values are compared with documented tolerances.
- `digest`: compact digest and sentinel metadata are compared.
- `structural`: shape, dtype, value range, and annotation invariants are
  checked, but exact pixels are not promised.

OpenCV-backed interpolation and platform-sensitive effects should avoid exact
byte-level promises unless the behavior is known to be stable across supported
dependency versions.

## Coverage Route Registry

`tests/regression/transform_contracts.py` maintains the public transform
coverage route registry. It fails when a public transform-like API is neither
covered by the established transform sweeps nor assigned to an explicit route
such as composition, PyTorch conversion, custom transform behavior, or a golden
compatibility contract. This registry is an ownership check, not a substitute
for the underlying tests.

## Unspecified Behavior

AlbumentationsX does not promise behavior for unsupported target combinations,
invalid input arrays, direct calls that bypass documented Compose contracts, or
optional dependency cross-products that are not listed in the support policy.
