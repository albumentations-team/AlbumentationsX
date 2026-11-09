# Unreleased behavior changes

Release baseline: 2.6.0

These changes are in the development tree after 2.6.0. Release preflight assigns the first affected published version.
Follow the [behavior compatibility policy](../maintaining/behavior-compatibility.md) when updating this inventory.
Earlier custom-sampler and input-lifetime changes shipped in [2.6.0](2.6.0.md).

Implementation: [validation #620](https://github.com/albumentations-team/AlbumentationsX/pull/620),
[geometry #621](https://github.com/albumentations-team/AlbumentationsX/pull/621), and
[determinism #622](https://github.com/albumentations-team/AlbumentationsX/pull/622).

## Validation

### Constructor and graph errors

- Previous behavior: Some schema errors could warn and substitute defaults; duplicate arguments could be overwritten.
  Invalid probabilities, aliases, processor settings, and selector branches failed inconsistently.
- New behavior: Invalid declared parameters raise without a default fallback in both strict modes. Wrong kinds raise
  TypeError and invalid values raise ValueError. Graph validation rejects unknown alias targets, missing coordinate
  formats, all-zero OneOf weights, and OneOrOther with zero or more than two branches.
- Affected users: Custom InitSchema validators, invalid configurations, and callers catching only ValueError. Unknown
  keywords in permissive mode and one-branch OneOrOther still warn.
- Migration: Correct declared arguments and use supported branches and aliases, for example
  `A.Compose([A.HorizontalFlip()], additional_targets={"right": "image"})`. Handle TypeError for incorrect kinds.
- First affected version: unreleased
- Contract: [Pipeline validation](../validation.md).
- Tests: [Constructor and graph validation](../../tests/contracts/test_validation_contract.py).

### Root inputs before sampling

- Previous behavior: A skipped call could accept malformed inputs or reject them after random draws. Frame IDs could
  hide missing bbox coordinates; empty rows could bypass label-count checks.
- New behavior: Root and traced calls, including p=0, reject invalid containers, ranks, alias sources, shapes,
  coordinates, frame rows, and label counts before drawing randomness. Valid skipped inputs retain their original
  objects, including strided images. Positional sample arguments raise TypeError rather than KeyError.
- Affected users: Pipelines relying on skips to bypass validation, malformed annotations, retry loops, or earlier
  exception types.
- Migration: Align spatial targets and pass named arguments, for example `pipeline(image=image, mask=mask)` with
  `mask.shape[:2] == image.shape[:2]`. Correct rows and labels; refresh seeded expectations after rejected calls.
- First affected version: unreleased
- Contract: [Validation boundaries](../validation.md), [failure and stream
  consumption](../determinism.md#failure-and-stream-consumption).
- Tests: [Input rejection and RNG preservation](../../tests/contracts/test_validation_contract.py).

### Serialized records and worker errors

- Previous behavior: Incomplete records could produce incidental KeyError or accept missing applied parameters. Worker
  seed lookup could silently ignore AttributeError.
- New behavior: Malformed record kinds raise TypeError. Missing fields, unknown classes, and missing Lambda bindings
  raise actionable ValueError. Worker RNG lookup errors propagate instead of silently selecting another stream. These
  are validation corrections.
- Affected users: Hand-edited records, custom transform registrations, exception handlers, and failing worker
  integrations.
- Migration: Preserve the complete result's `replay` value and import custom transforms before loading. For named
  Lambda transforms use `A.from_dict(payload, nonserializable={"custom_flip": custom_flip})`. Correct worker lookup
  errors.
- First affected version: unreleased
- Contract: [Replay and validation errors](../validation.md), [worker seeding](../determinism.md).
- Tests: [Malformed records and Lambda bindings](../../tests/contracts/test_validation_contract.py), [worker
  seeding](../../tests/test_per_worker_seed.py).

## Geometry

### Oriented boxes in output pixels

- Previous behavior: Rounded resize and crop-resize could scale box sides without refitting rotated corners.
  Perspective could fit boxes in normalized coordinates; remaps could use axis-aligned support.
- New behavior: LongestMaxSize, SmallestMaxSize, LetterBox, CropAndPad, RandomSizedCrop, RandomResizedCrop,
  RandomSizedBBoxSafeCrop, Perspective, and remap distortions correctly fit oriented boxes in output pixels. Identity
  remaps preserve fractional oriented-box geometry.
- Affected users: Oriented boxes on rectangular frames, rounded or anisotropic resizes, perspective, and distortions.
- Migration: Recompute cached annotations with `A.BboxParams(bbox_type="obb", coord_format="cxcywh")` in Compose. For
  example, resizing a 7-by-11 frame to 3-by-4 can change the box's angle as well as its side lengths.
- First affected version: unreleased
- Contract: [2D geometry](../geometry.md), [bounding boxes](../design/bounding_boxes.md).
- Tests: [Oriented-box containment and remaps](../../tests/contracts/test_geometry_contract.py).

### Shared perspective and elastic geometry

- Previous behavior: Perspective target paths could derive different output mappings. GridElasticDeform could
  recompute cell mappings per target and fail on wide images with horizontal boxes.
- New behavior: Perspective shares its output homography across rasters and annotations. GridElasticDeform shares
  sampled cell matrices, handles wide-image boxes, and reconstructs matrices from older mesh records. These correct
  synchronized-target geometry.
- Affected users: Synchronized image, mask, bbox, and keypoint targets with Perspective or GridElasticDeform.
- Migration: Refresh cached transformed targets together, for example `pipeline(image=image, mask=mask,
  bboxes=boxes)`. Existing geometry replay records remain readable; recheck expected coordinates and pixels.
- First affected version: unreleased
- Contract: [Shared geometry and replay](../geometry.md).
- Tests: [Perspective and elastic-grid alignment](../../tests/contracts/test_geometry_contract.py).

### Exhaustive crops retain a pixel

- Previous behavior: CropAndPad could raise a crop-window error when combined crop amounts exhausted an axis.
- New behavior: CropAndPad correctly retains one center pixel on each exhausted axis before padding or resizing.
- Affected users: CropAndPad configurations whose combined crop amounts can empty the input.
- Migration: Refresh cached crops and annotations. For example, `A.CropAndPad(px=-64, keep_size=False, p=1.0)` on a
  7-by-9 frame now produces a 1-by-1 crop instead of raising.
- First affected version: unreleased
- Contract: [Crop geometry](../geometry.md).
- Tests: [Single-pixel crops and annotation mapping](../../tests/contracts/test_geometry_contract.py).

### Resize replay uses the current frame

- Previous behavior: Resize-family replay could map annotations using recording-frame dimensions; an empty target
  could supply the wrong resize shape.
- New behavior: RandomScale, LongestMaxSize, SmallestMaxSize, Resize, and LetterBox correctly map keypoints and
  oriented boxes from the current input frame. Recorded output sizes and LetterBox padding remain authoritative. Shape
  discovery uses a nonempty target.
- Affected users: Cross-size resize replay and inputs containing empty and nonempty spatial targets.
- Migration: Reuse records with `A.ReplayCompose.replay(record, image=new_image, bboxes=new_boxes)` and annotations
  for the new frame. Refresh cached expected coordinates.
- First affected version: unreleased
- Contract: [Resize and replay](../geometry.md).
- Tests: [Cross-size replay and nonempty targets](../../tests/contracts/test_geometry_contract.py).

### Direct bbox calls use normalized horizontal boxes

- Previous behavior: Direct bbox-capable calls could lose the normalized horizontal-box interpretation without a
  processor, including older CropAndPad records.
- New behavior: The call boundary correctly defaults unprocessed bbox targets and aliases to normalized horizontal
  boxes. Explicit BboxParams still selects the configured type.
- Affected users: Direct transforms with bboxes or bbox aliases, and older records without a bbox type.
- Migration: Pass normalized horizontal boxes to direct calls, for example `A.HorizontalFlip(p=1.0)(image=image,
  bboxes=normalized_boxes)`. Use Compose with explicit BboxParams for other formats or oriented boxes.
- First affected version: unreleased
- Contract: [BBox call boundary](../geometry.md).
- Tests: [Direct HBB calls, aliases, and older replay](../../tests/contracts/test_geometry_contract.py).

### Box-aware RandomResizedCrop sampling

- Previous behavior: RandomResizedCrop sampled its crop without considering bounding boxes, so small or
  separated objects could be cropped out.
- New behavior: `bbox_preservation="one_box"` requires the crop to contain one uniformly selected box
  across active bbox fields; `"all_boxes"` requires it to contain every box. The default `None` keeps
  ordinary sampling. Scale and aspect-ratio bounds are relaxed only as needed to meet the selected box
  constraint.
- Affected users: Detection pipelines that need RandomResizedCrop to retain at least one object or every
  annotated object.
- Migration: Set `bbox_preservation="one_box"` or `"all_boxes"` on RandomResizedCrop when the crop must
  retain boxes. Existing calls using the default need no change.
- First affected version: unreleased
- Contract: [Box-aware RandomResizedCrop sampling](../maintaining/correctness-contracts.md#box-aware-randomresizedcrop-sampling).
- Tests: [RandomResizedCrop tests](../../tests/test_random_resized_crop.py).

### Macenko stain-plane estimation

- Previous behavior: MacenkoNormalizer treated OpenCV's row-oriented eigenvectors as columns and fitted
  the projection plane from the wrong vectors.
- New behavior: MacenkoNormalizer uses the two row eigenvectors associated with the largest eigenvalues.
  The recovered hematoxylin and eosin vectors now match the largest-principal-component plane.
- Affected users: MacenkoNormalizer and HEStain with `method="macenko"`; their recovered stain vectors
  and augmented outputs change.
- Migration: No API change is required. Re-run Macenko pipelines to use the corrected stain basis and regenerate any cached normalized images.
- First affected version: unreleased
- Contract: [Macenko stain estimation](../maintaining/correctness-contracts.md#macenko-stain-estimation).
- Tests: [Known-stain recovery test](../../tests/functional/test_functional.py).

### Custom sampled dispatch hooks

- Previous behavior: An apply_with_params override could intercept framework dispatch and diverge from replay or
  tracing.
- New behavior: Framework dispatch is private and the old sampled-dispatch override is rejected. This intentional
  customization API change requires sample_parameters and target handlers such as apply and apply_to_mask.
- Affected users: Custom transforms overriding apply_with_params or accessing dispatch state directly.
- Migration: Move preparation into `sample_parameters(self, params, data, targets, sampling)` and return
  SampledParams. Consume its values in `apply(self, img, **params)` or another target handler; follow the [sampling
  example](../contributing/coding_guidelines.md).
- First affected version: unreleased
- Contract: [Custom transform hooks](../contributing/coding_guidelines.md).
- Tests: [Sampling boundaries](../../tests/test_sampling_context.py), [dispatch
  guidance](../../tests/test_ax_coding_guidance.py).

## Determinism

### Ordered repeated replay visits

- Previous behavior: SomeOf replay could collapse repeated children and lose per-visit parameters; RandomOrder could
  lose execution order. A reused skipped leaf could retain its earlier observation.
- New behavior: Replay correctly preserves ordered applied and skipped visits, individual parameters, nested
  selectors, channel scopes, aliases, and annotations. Skipped leaves clear prior observations. Dense parameter
  storage grows with the visits performed. Direct replay skips use skipped_replay in traces; skipped Compose and
  selective-channel subtrees can use skipped_probability.
- Affected users: ReplayCompose with repeated or reordered selections, nested selectors, and trace-status consumers.
- Migration: Capture a new result's `replay` value, then use `A.ReplayCompose.replay(record, image=image)`. Older
  records retain their original applied-child interpretation; missing historical visits cannot be reconstructed.
  Refresh affected trace expectations.
- First affected version: unreleased
- Contract: [Determinism and persistence](../determinism.md#choose-the-persistence-contract).
- Tests: [Ordered replay and trace contracts](../../tests/test_composition_tracing.py), [determinism
  properties](../../tests/property/test_determinism.py).

## Constructor defaults

none

## Deprecations

none

## Regression vectors

none
