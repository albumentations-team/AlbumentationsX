# Coding Guidelines

Apply these standards to new and revised code within the scope of a change. The repository hooks enforce the
mechanical rules; the linked skills and design documents explain decisions that require review.

## Code Style and Formatting

### Line Length

- Maximum line length is **120 characters** (enforced by ruff).
- **Never** add `E501` to `pyproject.toml` or `# noqa: E501` inline suppression to work around long lines.
- Long strings (docstrings, comments, expressions) must be split across multiple lines at a word or operator boundary.
- For docstrings, wrap to the next line — the Google docstring format allows multi-line short descriptions.

### Code Complexity

- Ruff enforces McCabe complexity (`C901`, limit 10), return count (`PLR0911`), branch count (`PLR0912`, limit 12), and statement count (`PLR0915`).
- **Never** suppress these with `# noqa: C901`, `# noqa: PLR0911`, `# noqa: PLR0912`, `# noqa: PLR0915`, or any other inline suppression.
- **Never** raise the limit in `pyproject.toml`.
- **Fix**: Extract private helper methods that each own a single concern. A function over the limit is a signal it is doing too many things and should be split.

### Pre-commit Hooks

We use pre-commit hooks to maintain consistent code quality. These hooks automatically check and format your code before each commit.

- Install pre-commit if you haven't already:

  ```bash
  pip install pre-commit
  pre-commit install
  ```

- The hooks will run automatically on `git commit`. To run manually:

  ```bash
  uv run pre-commit run --all-files
  ```

- Pyrefly runs through the official pre-commit hook in system mode, using the same `uv` environment as CI.

AX repository checks run through one configurable hook: `pre-commit run check-ax-rules --all-files`. Its enabled
rules are listed in `[tool.ax-rules.rules]` in `pyproject.toml`; set an entry to `false` to disable it. On commit,
each enabled rule runs only for the files in its own scope; `uv run python -m tools.ax_rules` without filenames runs
every enabled rule. The `coding-guidance` rule emits `AXG001`–`AXG025` diagnostics for transform API, sampling, schema, naming,
performance-shape, documentation, bbox propagation, and target-specific sampling contracts. The public
`BboxParams.__init__(bbox_type="hbb")` compatibility default is intentional; all internal transform, processor, and
functional calls must pass `bbox_type` explicitly. Keep design judgment and benchmark interpretation in the relevant
review skill rather than duplicating these mechanical checks. `AGENTS.md`, `.codex/rules/`, and `.codex/skills/` point
to this document and the hook instead of restating the AXG catalog.

- Before handing off Python changes, run the fast local quality gate:

  ```bash
  pre-commit run --all-files --show-diff-on-failure
  ```

- Changes to support metadata, CI workflows, release docs, or correctness-report
  templates must keep the machine-readable support matrix in sync:

  ```bash
  uv run python -m tools.ci_matrix check
  uv run python -m tools.ci_shard check
  ```

### Python Version and Type Hints

- Use Python 3.10+ features and syntax
- Always include type hints for all functions

## Naming Conventions

### Variable Names

- Avoid unclear or single-letter variable names when a descriptive name improves readability:

  ```python
  # Correct - descriptive
  rot90_count = C4_GROUP_ELEMENT_TO_K[group_element]
  return np.rot90(img, rot90_count)

  # Avoid - unclear
  k = C4_GROUP_ELEMENT_TO_K[group_element]
  return np.rot90(img, k)
  ```

- Note: Ruff does not flag single-letter names; this is a manual style preference.

### Transform Names

- Avoid adding "Random" prefix to new transforms

  ```python
  # Correct
  class Brightness(ImageOnlyTransform):

  # Incorrect
  class RandomBrightness(ImageOnlyTransform):
  ```

### Parameter Naming

- Use `_range` suffix for interval parameters:

  ```python
  # Correct
  brightness_range: tuple[float, float]
  shadow_intensity_range: tuple[float, float]

  # Incorrect
  brightness_limit: tuple[float, float]
  shadow_intensity: tuple[float, float]
  ```

### Standard Parameter Names

For transforms that handle gaps or boundaries, use these consistent names:

- `border_mode`: Specifies how to handle gaps, not `mode` or `pad_mode`
- `fill`: Defines how to fill holes (pixel value or method), not `fill_value`, `cval`, `fill_color`, `pad_value`, `pad_cval`, `value`, `color`
- `fill_mask`: Same as `fill` but for mask filling, not `fill_mask_value`, `fill_mask_color`, `fill_mask_cval`

## Parameter Types and Ranges

### Parameter Definitions

- Prefer range parameters over fixed values:

  ```python
  # Correct
  def __init__(self, brightness_range: tuple[float, float] = (-0.2, 0.2)):

  # Avoid
  def __init__(self, brightness: float = 0.2):
  ```

### Avoid Union Types for Parameters

- Don't use `Union[float, tuple[float, float]]` for parameters
- Instead, always use ranges where sampling is needed:

  ```python
  # Correct
  scale_range: tuple[float, float] = (0.5, 1.5)

  # Avoid
  scale: float | tuple[float, float] = 1.0
  ```

- For fixed values, use same value for both range ends:

  ```python
  brightness_range = (0.1, 0.1)  # Fixed brightness of 0.1
  ```

## Transform Design Principles

### Relative Parameters

- Prefer parameters that are relative to image dimensions rather than fixed pixel values:

  ```python
  # Correct - relative to image size
  def __init__(self, crop_size_range: tuple[float, float] = (0.1, 0.3)):
      # crop_size will be fraction of min(height, width)

  # Avoid - fixed pixel values
  def __init__(self, crop_size_range: tuple[int, int] = (32, 96)):
      # crop_size will be fixed regardless of image size
  ```

### Data Type Consistency

- Ensure transforms produce consistent results regardless of input data type
- Use provided decorators to handle type conversions:
  - `@uint8_io`: For transforms that work with uint8 images
  - `@float32_io`: For transforms that work with float32 images

The decorators will:

- Pass through images that are already in the target type without conversion
- Convert other types as needed and convert back after processing

Apply the decorator to the functional operation that owns the dtype-specific kernel. Keep transform application
methods as dispatchers; do not add manual conversion or a forwarding wrapper just to attach a decorator.

### Image Value Ranges and `@clipped`

- Image transforms preserve `[0, 255]` for `uint8` and `[0, 1]` for `float32`, unless the transform explicitly sets
  `_preserves_input_image_range = False`.
- Use `@clipped` when every route of a functional image operation can leave this range. When only a known float32 mode
  can overshoot, branch on that mode and call `albucore.clip(..., inplace=True)` there. Do not clip masks or annotations.
- Do not add a forwarding wrapper around a one-line call merely to attach a decorator. Add a function only when it
  owns a real image operation and separates image policy from mask or annotation semantics.

### Channel Flexibility

Support arbitrary channel counts unless the operation requires a specific layout, such as RGB. Validate that
transform-specific requirement before dispatching to the functional operation. Keep channel arithmetic in the
functional layer.

### Image and Volume Shape Invariants

`Compose` normalizes optional channel axes before dispatch. The canonical layout depends on the caller's container:

| Target family | NumPy inside `Compose` | CPU Tensor inside `Compose` |
| --- | --- | --- |
| image or mask | `(H, W, C)` | `(C, H, W)` |
| images or masks | `(N, H, W, C)` | `(N, C, H, W)` |
| volume or mask3d | `(D, H, W, C)` | `(C, D, H, W)` |

- Do not add compatibility branches for channel-free inputs in `apply_*` methods or functional kernels used by
  `Compose`; the root normalizes them before dispatch and restores the public rank when the channel remains singleton.
- A handler without `torch.Tensor` in its primary input annotation receives a canonical NumPy view through the base
  fallback. Keep its existing NumPy implementation channel-last; do not duplicate bridge logic in the transform.
- A handler that accepts Tensor input declares `torch.Tensor` in that annotation and supports the canonical Tensor
  layouts above. The annotation is its runtime contract.
- It is fine to branch on `img.ndim` when selecting image, batch, and volume semantics. Do not use rank to guess where
  a channel axis is.

### Handling Auxiliary Data via Metadata

When a transform requires complex or variable auxiliary data beyond simple configuration parameters (e.g., additional images and labels for `Mosaic`, extra images for domain adaptation transforms like `FDA` or `HistogramMatching`), **do not pass this data directly through the `__init__` constructor**.

Instead, follow this preferred pattern:

1. **Pass the auxiliary data** within the main `data` dictionary provided to the transform's `__call__` method, using a descriptive key (e.g., `mosaic_metadata`, `copy_paste_metadata`).
2. **Declare this key** in the transform's `targets_as_params` property. This signals to `Compose` that the key should be extracted and forwarded to `sample_parameters`.
3. **Access the data** inside `sample_parameters` using `data.get("your_metadata_key")`.
4. **Define empty metadata deliberately**. A transform may no-op or use a documented fallback; it must not reach into a
   dataset or another global donor source to fill the gap.

`targets_as_params` is also the complete declaration needed for CPU Tensor fallback. Do not add Tensor-specific routing
properties, flags, target lists, or adapters to a concrete transform. The base adapter converts direct Tensor parameters
to NumPy views and recognizes standard target fields inside donor records.

Passing data via `__init__` couples the transform instance to specific data, making it less reusable and potentially breaking serialization or pipeline composition.

### Mixing Transforms: Additional Rules

Mixing transforms (`Mosaic`, `CopyAndPaste`, etc.) combine data from multiple images and require additional
conventions beyond the general metadata pattern.

#### Donor sampling and ownership

The caller supplies the candidate donor pool; a mixing transform never reaches into a dataset or another global donor
source. A normal Mosaic call uses every valid supplied donor. When the supplied pool exceeds the number of additional
cells, Mosaic chooses the required subset with the call-local `SamplingContext`; when it is smaller, Mosaic replicates
the primary item to fill the remaining cells.

```python
# Correct — caller owns the candidate pool.
result = transform(image=image, mosaic_metadata=donors)

# Incorrect — the transform owns a dataset and discovers donors itself.
result = TransformThatSamplesInternally(dataset=dataset)(image=image)
```

Keeping the pool with the caller preserves deterministic control, class-balanced selection, hard-example mining, and
curriculum strategies. The transform may only select from that supplied pool when it has more candidates than cells.

#### Metadata format: `list[dict]`

All mixing transforms use `list[dict]` as metadata: one dictionary per full image for `Mosaic` or per object instance
for `CopyAndPaste`.

#### Label fields in metadata

All mixing transforms use `bbox_labels` and `keypoint_labels` wrapper dictionaries for label fields:

- `bbox_labels` maps every field declared in `BboxParams.label_fields` to its value or values.
- `keypoint_labels` maps every field declared in `KeypointParams.label_fields` to its value or values.

For **CopyAndPaste** (one object per dictionary), values are scalars:

```python
{
    "image": src_image,
    "mask": object_mask,
    "bbox": [10, 20, 50, 80],
    "bbox_labels": {"class_id": 3, "is_crowd": 0},
    "keypoints": [[25, 40]],
    "keypoint_labels": {"joint_name": "left_eye"},
}
```

For **Mosaic** (one full image per dictionary), values are lists — one entry per bounding box or keypoint:

```python
{
    "image": image,
    "bboxes": [[10, 20, 50, 80], [5, 5, 30, 30]],
    "bbox_labels": {"class_id": [3, 7], "is_crowd": [0, 1]},
    "keypoints": [[25, 40]],
    "keypoint_labels": {"joint_name": ["left_eye"]},
}
```

The keys inside `bbox_labels` and `keypoint_labels` must exactly match the corresponding `label_fields` declaration.

#### Coordinates use the enclosing processor format

Bounding boxes and keypoints in metadata use the same `coord_format` declared by the enclosing `Compose`. The
processor converts them to its internal format; do not convert them manually before passing metadata.

**Mosaic selection example:**

```python
def select_mosaic_items(
    candidates: list[dict[str, Any]],
    *,
    primary: dict[str, Any],
    num_additional_cells: int,
    sampling: SamplingContext,
) -> list[dict[str, Any]]:
    if len(candidates) > num_additional_cells:
        return sampling.py_random.sample(candidates, num_additional_cells)

    replicas = [copy.deepcopy(primary) for _ in range(num_additional_cells - len(candidates))]
    return [*candidates, *replicas]
```

`CopyAndPaste` and Mosaic intentionally differ for empty metadata: `CopyAndPaste` is a no-op, while Mosaic may fill
its cells with independent primary-item replicas.

See `docs/design/mosaic.md` for Mosaic's full processor, sampling, and label-encoding contract.

## Random Number Generation

### SamplingContext owns call-local randomness

`Compose` passes one `SamplingContext` to every `sample_parameters` call. Use it for every random draw and for the
applied-configuration record. Do not read RNG state from the transform instance or use module-level random functions:

```python
from albumentations.core.invocation import SamplingContext
from albumentations.core.transform_params import SampledParams, TargetSet


def sample_parameters(
    self,
    params: dict[str, Any],
    data: dict[str, Any],
    targets: TargetSet,
    sampling: SamplingContext,
) -> SampledParams:
    brightness = sampling.py_random.uniform(*self.brightness_range)
    height, width = targets.require_aligned_spatial_shape(2)
    noise = sampling.random_generator.uniform(-1, 1, size=(height, width))
    sampling.applied_overrides["brightness_range"] = brightness
    return SampledParams(params={"brightness": brightness, "noise": noise})
```

Use `sampling.py_random` for a few scalar choices. Use `sampling.random_generator` for array-valued draws. Generate
all stochastic parameters before `apply`, then return a `SampledParams`. Values that apply to every target go in
`SampledParams.params`; representation-dependent values belong in `TargetParams` entries addressed by actual target key.
Use `targets` and its descriptors for channel, dtype, layout, topology, or target-content decisions. Do not derive
those decisions from the first-target `params["shape"]` field.

## Transform Development

### Method Definitions

- Don't use default arguments in `apply_xxx` methods:

  ```python
  # Correct
  def apply_to_mask(self, mask: np.ndarray, fill_mask: int) -> np.ndarray:

  # Incorrect
  def apply_to_mask(self, mask: np.ndarray, fill_mask: int = 0) -> np.ndarray:
  ```

#### Declare consumed sampled parameters explicitly

Every transform `apply*` method must name each sampled parameter it reads, including execution parameters such as
`shape` and `bbox_type`. Keep `**params` only to forward parameters to another handler; do not read it with
`params[...]` or `params.get(...)`. The `coding-guidance` rule in `check-ax-rules` enforces this rule.

#### Keep `apply*` Methods Thin

`apply`, `apply_to_images`, and other `apply*` methods are transform-layer policy and dispatch points, not places to
implement pixel kernels. Keep a transform-specific runtime input check here, so the transform's supported contract is
visible next to its public application method. Then select a target-specific functional operation and pass sampled
parameters. Image arithmetic, gradient or mask construction, dtype routing, clipping, and kernel-selection branches
belong in a named helper in the functional layer (or Albucore when the operation is reusable there).

Each concrete transform `apply*` body is limited to 20 code-bearing physical lines. Its signature, docstring, blank
lines, and standalone comments do not count; a line that contains code and an inline comment does. Only base
infrastructure classes whose names begin with `Base` (for example, `BaseCrop` and `BaseMaxSizeTransform`) are excluded.
Name a non-public base class `BaseX`, not `X`. This rule
does not apply to `Compose` orchestration such as `apply_in_invocation`; it is enforced by the unified
`coding-guidance` rule in `check-ax-rules` (`AXG003`).

### Parameter Generation

#### Using sample_parameters

This method receives explicit execution parameters, preprocessed data, and invocation-local target descriptors for parameter
generation:

```python
def sample_parameters(
    self,
    params: dict[str, Any],
    data: dict[str, Any],
    targets: TargetSet,
    sampling: SamplingContext,
) -> SampledParams:
    height, width = targets.require_aligned_spatial_shape(2)

    crop_size = min(height, width) // 2
    center_x = width // 2
    center_y = height // 2

    return SampledParams(params={"crop_size": crop_size, "center": (center_x, center_y)})
```

The method receives:

- `params`: Execution parameters such as interpolation and fill values
- `data`: The full preprocessed invocation, including auxiliary values outside the active targets
- `targets`: The ordered `TargetSet`; use `require_aligned_spatial_shape(2)` or `require_aligned_spatial_shape(3)` for synchronized geometry
- `sampling`: Call-local Python and NumPy RNG streams plus the applied-configuration sink

Never return a plain dictionary. For target-specific materialization, group by a transform-defined compatibility key and
return `TargetParams(targets=..., params=..., requirements=...)`. The core stores the resulting schema in replay and
rejects legacy flat payloads.

### Parameter Validation with `InitSchema`

Each transform that introduces constructor inputs or changes their annotations must include a non-empty `InitSchema`
that inherits from `BaseTransformInitSchema`. A transform that only repeats an inherited constructor contract and
explicitly forwards those inputs does not need a new schema. The schema is responsible for:

- Validating input parameters before `__init__` execution
- Converting parameter types if needed
- Ensuring consistent parameter handling

#### No Default Values in InitSchema

**InitSchema classes must not contain default values for their fields.** This ensures that all transform parameters are explicitly provided and validated at initialization time.

```python
# Correct - no default values in InitSchema
class MyTransform(ImageOnlyTransform):
    class InitSchema(BaseTransformInitSchema):
        brightness_range: tuple[float, float]
        contrast_range: tuple[float, float]

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        contrast_range: tuple[float, float] = (0.8, 1.2),
        p: float = 0.5,
    ):
        # Default values go in __init__, not InitSchema
        super().__init__(p=p)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range


# Incorrect - default values in InitSchema
class MyTransform(ImageOnlyTransform):
    class InitSchema(BaseTransformInitSchema):
        brightness_range: tuple[float, float] = (0.8, 1.2)  # ❌ No defaults in InitSchema
        contrast_range: tuple[float, float] = (0.8, 1.2)  # ❌ No defaults in InitSchema
```

The rule is enforced by the `coding-guidance` rule in `check-ax-rules` (`AXG001`). It checks nested and
module-level `*InitSchema` classes, follows imported schema inheritance, and does not exempt discriminator fields.
If a transform only repeats inherited constructor inputs and explicitly forwards them, it does not need an empty schema.
When it adds an input or changes an annotation, its non-empty schema must declare that input (`AXG020`).
Compatibility aliases that deliberately preserve a parent validator's permissive boundary may retain an empty schema;
they must not introduce new constructor fields.

Sampling overrides must return `SampledParams` rather than a flat dictionary (`AXG021`), and must not derive
target-sensitive values from first-target `params["shape"]` or legacy shape helpers (`AXG022`). Application methods
use semantic parameter names; target routing belongs in `TargetParams` entries rather than names such as `volume_noise_map`
(`AXG023`). When choosing among canonical image targets, samplers use `TargetSet` rather than selecting `image`,
`images`, or `volume` from `data` (`AXG024`).

#### No `get_transform_init_args_names` Override

**Do not override `get_transform_init_args_names()`.** The base class reads the concrete transform's public `__init__`
signature. Parent-only implementation fields are deliberately excluded because they are not part of the public
serialization or applied-configuration boundary. Overriding this method can cause constructor mismatches.

### Batch Performance (`apply_to_images`)

NumPy image batches inside transform execution have shape `(N, H, W, C)`. Native Tensor handlers receive
`(N, C, H, W)`. Both have rank four; do not repeat that check inside `apply_to_images`.

Keep `apply_to_images` as a short delegation method. Put the complete batch image operation—including empty-batch
handling, shared setup, routing between native and per-image kernels, and clipping—in one functional helper so it has a
single direct correctness and benchmark boundary.

Override `apply_to_images` only when measurement shows an advantage over the default per-image loop. In the
functional helper, compare shared setup, direct batch indexing, and a preallocated loop. Do not construct kernels or
read sampled values from `**params` inside the transform method.

Keep batch and channel axes distinct. Reshaping `(N, H, W, 1)` into `(H, W, N)` adds a transpose and may require a
copy; OpenCV processes those channels sequentially. Use the
[benchmark skill](../../.codex/skills/benchmark/SKILL.md) for the required direct and Compose measurements.

### Coordinate Systems

#### Image Center Calculations

The center point calculation differs slightly between targets:

- For images, masks, and keypoints:

  ```python
  # Correct - using helper function
  from albumentations.augmentations.geometric.functional import center

  center_x, center_y = center(image_shape)  # Returns ((width-1)/2, (height-1)/2)

  # Incorrect - manual calculation might miss the -1
  center_x = width / 2  # Wrong!
  center_y = height / 2  # Wrong!
  ```

- For bounding boxes:

  ```python
  # Correct - using helper function
  from albumentations.augmentations.geometric.functional import center_bbox

  center_x, center_y = center_bbox(image_shape)  # Returns (width/2, height/2)

  # Incorrect - using wrong center calculation
  center_x, center_y = center(image_shape)  # Wrong for bboxes!
  ```

This small difference is crucial for pixel-perfect accuracy. Always use the appropriate helper functions:

- `center()` for image, mask, and keypoint transformations
- `center_bbox()` for bounding box transformations

### Serialization Compatibility

- Ensure transforms work with both tuples and lists for range parameters
- Test serialization/deserialization with JSON and YAML formats

## Documentation

### Docstrings

Use Google-style docstrings. Concrete transform `apply*` methods inherit their documentation from the class and base
interface; do not repeat it on each override.

- The first paragraph is a 120–160 character web/search preview. Explain the effect and when to use it. Wrap at a word
  boundary within the 120-character line limit, with no blank line inside the paragraph.
- Put parameter names and units in `Args`, return details in `Returns`, and supported targets and dtypes in their own
  sections. Generic preservation claims and directions to read other sections do not belong in the preview.
- Use `See also` for two to four related transforms, one per bullet with a selection hint. Keep cross-links reciprocal.
- Use `Note` bullets for factual details. Put recommendations in `See also`.

Use the [docstring review skill](../../.codex/skills/docstring-deep-dive/SKILL.md) to review reader-facing quality.

### Examples in Docstrings

Every transform class descending from `ImageOnlyTransform`, `DualTransform`, or `Transform3D` needs an `Examples`
section. Use doctest syntax: `>>>` for statements, `...` for continuations, and no prefix for output.

Include imports, sample data, transform construction, a public call, and result access. Demonstrate the supported
targets: images for image-only transforms; images, masks, bboxes, keypoints, and their labels for dual transforms;
volumes, 3D masks, and keypoint labels where supported for 3D transforms. Use tuple ranges for sampled parameters.
Base-class examples should implement a working subclass and execute it through Compose.

Here's an example for a `DualTransform`:

```python
"""
Examples:
    >>> import numpy as np
    >>> import albumentations as A
    >>> rng = np.random.default_rng(137)
    >>> image = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> mask = rng.integers(0, 2, (100, 100), dtype=np.uint8)
    >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
    >>> bbox_labels = [1, 2]
    >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
    >>> keypoint_labels = ['left_eye', 'right_eye']
    >>>
    >>> transform = A.Compose([
    ...     A.HorizontalFlip(p=1.0),
    ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
    ...    keypoint_params=A.KeypointParams(
    ...        coord_format='xy',
    ...        label_fields=['keypoint_labels'],
    ...        label_mapping={'HorizontalFlip': {
    ...            'keypoint_labels': {'left_eye': 'right_eye', 'right_eye': 'left_eye'},
    ...        }},
    ...    ))
    >>>
    >>> transformed = transform(
    ...     image=image,
    ...     mask=mask,
    ...     bboxes=bboxes,
    ...     bbox_labels=bbox_labels,
    ...     keypoints=keypoints,
    ...     keypoint_labels=keypoint_labels
    ... )
    >>>
    >>> transformed_image = transformed['image']
    >>> transformed_mask = transformed['mask']
    >>> transformed_bboxes = transformed['bboxes']
    >>> transformed_keypoints = transformed['keypoints']
    >>> transformed_bbox_labels = transformed['bbox_labels']
    >>> transformed_keypoint_labels = transformed['keypoint_labels']
"""
```

A base-class example can start with identity maps so the reader can verify the remapping before adding distortion:

```python
from typing import Any

import cv2
import numpy as np

import albumentations as A
from albumentations.augmentations.geometric.distortion import BaseDistortion
from albumentations.core.invocation import SamplingContext
from albumentations.core.transform_params import SampledParams, TargetSet


class IdentityDistortion(BaseDistortion):
    def sample_parameters(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
        targets: TargetSet,
        sampling: SamplingContext,
    ) -> SampledParams:
        height, width = targets.require_aligned_spatial_shape(2)
        map_y, map_x = np.mgrid[:height, :width].astype(np.float32)
        return SampledParams(params={"map_x": map_x, "map_y": map_y})


image = np.random.default_rng(137).integers(0, 256, (100, 100, 3), dtype=np.uint8)
transform = A.Compose(
    [
        IdentityDistortion(
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            keypoint_remapping_method="mask",
            p=1.0,
        ),
    ]
)
transformed_image = transform(image=image)["image"]
np.testing.assert_array_equal(transformed_image, image)
```

### Comments

- Add comments for complex logic
- Explain why, not what (the code shows what)
- Keep comments up to date with code changes

## Performance Optimization

Use [Performance Optimization](../../.codex/skills/performance-optimization/SKILL.md) and its synchronized reference
for runtime audits. The workflow covers removable work, memory traffic, vectorization, grouped reductions, LUTs,
random generation, backend selection, Albucore ownership, and aliasing.

Treat each candidate as a hypothesis. Use the [benchmark skill](../../.codex/skills/benchmark/SKILL.md) to compare
baseline and candidate on the affected public routes, and record every measured cell and any regressions.

### Updating Transform Documentation

When adding a new transform or changing its target contract, update the generated transform table:

```bash
python -m tools.make_transforms_docs make
```

Then update the relevant README section with the generated result and confirm that the documented targets (image, mask,
bounding boxes, keypoints, and so on) match the public contract.

## Testing

### Test Coverage

- Write tests for all new functionality
- Include edge cases and error conditions
- Ensure reproducibility with fixed random seeds

### Test Organization

- Place tests in the appropriate module under `tests/`
- Follow existing test patterns and naming conventions
- Use pytest fixtures when appropriate

## Code Review Guidelines

Before submitting your PR:

1. Name the changed contract and run the smallest validation that can disprove it.
2. Run the relevant pre-commit hooks and report their actual results.
3. Add broader tests, benchmarks, or type checks only when the changed route requires them.
4. Update public documentation when the public contract changed.

Use `.codex/skills/validate-and-fix/SKILL.md` for the decision-based validation workflow.

## Getting Help

If you have questions about these guidelines:

1. Join our [Discord community](https://discord.gg/e6zHCXTvaN)
2. Open a GitHub [issue](https://github.com/albumentations-team/AlbumentationsX/issues)
3. Ask in your [pull request](https://github.com/albumentations-team/AlbumentationsX/pulls)
