# Coding Guidelines

This document outlines the coding standards and best practices for contributing to AlbumentationsX.

## Important Note About Guidelines

These guidelines define the required standard for code that enters or is revised in the AlbumentationsX codebase.

**For new contributions:**

- All new code must follow these guidelines
- All revised code must follow these guidelines within the scope of the change
- Pull requests that introduce non-conforming patterns will not be accepted

Apply the same standard when changing existing code: do not preserve a non-conforming pattern merely because it is
already present.

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

The repository-specific deterministic rules run through one package-wide hook:
`pre-commit run check-ax-coding-guidance --all-files`. It emits `AXG001`–`AXG024` diagnostics for transform API,
sampling, schema, naming, performance-shape, documentation, bbox propagation, and target-specific sampling contracts. The public
`BboxParams.__init__(bbox_type="hbb")` compatibility default is intentional; all internal transform, processor, and
functional calls must pass `bbox_type` explicitly. Keep design judgment and benchmark interpretation in the relevant
review skill rather than duplicating these mechanical checks. `AGENTS.md`, `.codex/rules/`, and `.codex/skills/` point to
this document and the hook instead of restating the AXG catalog.

- Before handing off Python changes, run the fast local quality gate:

  ```bash
  uv run python -m tools.quality_gate fast
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

```python
@uint8_io  # If input is uint8 => use as is; if float32 => convert to uint8, process, convert back
def apply(self, img: np.ndarray, **params) -> np.ndarray:
    # img is guaranteed to be uint8
    # if input was float32 => result will be converted back to float32
    # if input was uint8 => result will stay uint8
    return cv2.blur(img, (3, 3))


@float32_io  # If input is float32 => use as is; if uint8 => convert to float32, process, convert back
def apply(self, img: np.ndarray, **params) -> np.ndarray:
    # img is guaranteed to be float32 in range [0, 1]
    # if input was uint8 => result will be converted back to uint8
    # if input was float32 => result will stay float32
    return img * 0.5


# Avoid - manual type conversion
def apply(self, img: np.ndarray, **params) -> np.ndarray:
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    result = cv2.blur(img, (3, 3))
    if img.dtype != np.uint8:
        result = result.astype(np.float32) / 255
    return result
```

### Image Value Ranges and `@clipped`

- Image transforms preserve `[0, 255]` for `uint8` and `[0, 1]` for `float32`, unless the transform explicitly sets
  `_preserves_input_image_range = False`.
- Use `@clipped` when every route of a functional image operation can leave this range. When only a known float32 mode
  can overshoot, branch on that mode and call `albucore.clip(..., inplace=True)` there. Do not clip masks or annotations.
- Do not add a forwarding wrapper around a one-line call merely to attach a decorator. Add a function only when it
  owns a real image operation and separates image policy from mask or annotation semantics.

### Channel Flexibility

- Support arbitrary number of channels unless specifically constrained:

  ```python
  # Correct - works with any number of channels
  def apply(self, img: np.ndarray, **params) -> np.ndarray:
      # img shape is (H, W, C), works for any C
      return img * self.factor


  # Also correct - explicitly requires RGB
  def apply(self, img: np.ndarray, **params) -> np.ndarray:
      if img.shape[-1] != 3:
          raise ValueError("Transform requires RGB image")
      return rgb_to_hsv(img)  # RGB-specific processing
  ```

### Image and Volume Shape Invariants

`Compose` normalizes optional channel axes before dispatch. The canonical layout depends on the caller's container:

| Target family | NumPy inside `Compose` | CPU Tensor inside `Compose` |
| --- | --- | --- |
| image or mask | `(H, W, C)` | `(C, H, W)` |
| images or masks | `(N, H, W, C)` | `(N, C, H, W)` |
| volume or mask3d | `(D, H, W, C)` | `(C, D, H, W)` |

- Do not add compatibility branches for channel-free inputs in `apply_*` methods or functional kernels used by
  `Compose`; the root normalizes them before dispatch and restores the public rank when the channel remains singleton.
- A transform with no complete Tensor route receives a canonical NumPy view through the base fallback. Keep its existing
  NumPy implementation channel-last; do not duplicate bridge logic in the transform.
- A Tensor-aware transform must support the complete leaf lifecycle and the canonical Tensor layouts above. It may
  decline a case and let the base fallback run the established NumPy path.
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
properties to a concrete transform. The base adapter converts direct Tensor parameters to NumPy views and recognizes
standard target fields inside donor records.

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
`params[...]` or `params.get(...)`. The `check-ax-coding-guidance` pre-commit hook enforces this rule.

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
`check-ax-coding-guidance` pre-commit hook (`AXG003`).

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

    image = data.get("image")
    mask = data.get("mask")
    bboxes = data.get("bboxes")
    keypoints = data.get("keypoints")

    # Example: Calculate parameters based on image size
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

Use this method when you need to:

- Calculate parameters based on image dimensions
- Access target data for parameter generation
- Ensure transform parameters are appropriate for the input data

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

  ```python
  # Correct - full parameter validation
  class RandomGravel(ImageOnlyTransform):
      class InitSchema(BaseTransformInitSchema):
        slant_range: Annotated[tuple[float, float], AfterValidator(nondecreasing)]
        brightness_coefficient: float = Field(gt=0, le=1)


    def __init__(self, slant_range: tuple[float, float], brightness_coefficient: float, p: float = 0.5):
        super().__init__(p=p)
        self.slant_range = slant_range
        self.brightness_coefficient = brightness_coefficient
  ```

  ```python
  # Incorrect - missing InitSchema
  class RandomGravel(ImageOnlyTransform):
      def __init__(self, slant_range: tuple[float, float], brightness_coefficient: float, p: float = 0.5):
          super().__init__(p=p)
          self.slant_range = slant_range
          self.brightness_coefficient = brightness_coefficient
  ```

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

The rule is enforced by the unified `check-ax-coding-guidance` pre-commit hook (`AXG001`). It checks nested and
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

Images in batch mode are always `(N, H, W, C)`. Never check `ndim == 4` — it's always true.

Keep `apply_to_images` as a short delegation method. Put the complete batch image operation—including empty-batch
handling, shared setup, routing between native and per-image kernels, and clipping—in one functional helper so it has a
single direct correctness and benchmark boundary.

Override `apply_to_images` when you can do better than the default per-image loop:

1. **Pre-compute expensive setup once** (kernels, LUTs, gradient maps):

```python
def apply_to_images(self, images: ImageType, *args: Any, **params: Any) -> ImageType:
    kernel = create_kernel(params["size"])  # once per batch
    return self._apply_to_batch(images, lambda img: convolve(img, kernel))
```

2. **Direct 4D indexing** for simple array ops:

```python
result = images.copy()
result[:, :, :, channels] = fill  # vectorized across N
```

3. **Pre-allocated loop** to avoid repeated allocations:

```python
result = np.empty_like(images)
for i, image in enumerate(images):
    result[i] = self.apply(image, **params)
```

> **Anti-pattern**: Do NOT reshape `(N,H,W,1)` to `(H,W,N)` to call a cv2 function once — transpose yields non-contiguous memory (requiring a full copy), and cv2 processes channels sequentially so an N-channel call is not faster than N single-channel calls. Benchmarks show 2–4× regression.

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

- Use Google-style docstrings
- **Transform apply methods:** Do not add docstrings to `apply`, `apply_to_image`, `apply_to_mask`, or other `apply_to_*` methods in transform classes. The transform class docstring and the base interface in `transforms_interface` are sufficient.
- **First paragraph (120–160 characters):** A **useful short description** — an elevator pitch: what the function or transform does, how it works in one sentence, and when to use it. This is the web/search preview. Paragraphs are separated by blank lines; there is no blank line within a paragraph. So the first paragraph occupies **two lines of text with no blank line between them** (not "line, blank line, line"). **Line limit 120 chars** ⇒ the first paragraph must be two lines (no single line over 120; do not use `# noqa: E501`). Wrap at a word boundary. Do **not** list parameter names ("Parameters: x, y, z" or "Params: ...") in the first paragraph — that belongs in Args. Do **not** use "Preserves X" boilerplate (e.g. "Preserves channel count", "preserves dtype and channels") in the first paragraph — describe effect and when to use it instead. Do not put "Targets: ...", "Same shape", "Used by X", return type (e.g. "Returns np.ndarray"), or "Supports uint8/float32" (or Image types) in the first paragraph — return type belongs in Returns; dtype/target support has a separate Image types section, and all transforms support uint8 and float32 unless noted. Both length (120–160) and usefulness matter for discoverability.
- **Similar transforms / See also:** Use a **bullet list** (`-` per item) listing 2–4 related transforms with brief when-to-use hints, so users discover more than a limited set (e.g. RandomResizedCrop, ColorJitter). **One transform per bullet** — do not combine multiple transforms in one bullet. When you add transform X to transform Y's See also, update X's docstring to mention Y (reciprocal cross-links).
- **Note:** Use a **bullet list** (`-` per point). Note is **pure info** only — no call-to-action (e.g. no "Explore other transforms…" or "Consider using…"); put discoverability in See also.
- Include type information, parameter descriptions, and examples:

  ```python
  def transform(self, image: np.ndarray) -> np.ndarray:
      """Apply brightness transformation to the image.

      Args:
          image: Input image in RGB format.

      Returns:
          Transformed image.

      Examples:
          >>> transform = Brightness(brightness_range=(-0.2, 0.2))
          >>> transformed = transform(image=image)
      """
  ```

### Examples in Docstrings

Every transform class that is a descendant of `ImageOnlyTransform`, `DualTransform`, or `Transform3D` **must** include a comprehensive Examples section in its docstring. The examples should follow these guidelines:

1. **Section Naming**: The section should be titled "Examples" (not "Example").

2. **Jupyter Notebook Format**: Examples should mimic Jupyter notebook format, using `>>>` for code lines and no prefix for output lines.

3. **Comprehensiveness**: Examples should be fully reproducible, including:
   - Initialization of sample data
   - Creation of transform(s)
   - Application of transform(s)
   - Retrieving results from the transform

4. **Target-Specific Requirements**:
   - For `ImageOnlyTransform`: Pass and demonstrate transformation of image data. Including how to get all transformed targets.
   - For `DualTransform`: Pass and demonstrate transformation of image, mask, bboxes, keypoints, bbox_labels, class_labels (where supported). Including how to get all transformed targets including bbox_labels and keypoints_labels
   - For `Transform3D`: Pass and demonstrate transformation of volume and mask3d data. Including how to get all transformed targets.
    Including keypoint_labels

5. **For Base Classes**: Examples for base classes should show:
   - How to initialize a custom transform that inherits from the base class
   - How to use the custom transform as part of a Compose pipeline

6. **Parameter Examples**: When a parameter accepts both a single value and a tuple of values (to be sampled from), always use a tuple in the example.

Here's an example for a `DualTransform`:

```python
"""
Examples:
    >>> import numpy as np
    >>> import albumentations as A
    >>> # Prepare sample data
    >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
    >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
    >>> bbox_labels = [1, 2]
    >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
    >>> keypoint_labels = [0, 1]
    >>>
    >>> # Define transform with parameters as tuples when possible
    >>> transform = A.Compose([
    ...     A.HorizontalFlip(p=1.0),
    ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
    ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
    >>>
    >>> # Apply the transform
    >>> transformed = transform(
    ...     image=image,
    ...     mask=mask,
    ...     bboxes=bboxes,
    ...     bbox_labels=bbox_labels,
    ...     keypoints=keypoints,
    ...     keypoint_labels=keypoint_labels
    ... )
    >>>
    >>> # Get the transformed data
    >>> transformed_image = transformed['image']  # Horizontally flipped image
    >>> transformed_mask = transformed['mask']    # Horizontally flipped mask
    >>> transformed_bboxes = transformed['bboxes']  # Horizontally flipped bounding boxes
    >>> transformed_keypoints = transformed['keypoints']  # Horizontally flipped keypoints
"""
```

Examples for a base class showing custom implementation:

```python
"""
Examples:
    # Example of a custom distortion subclass
    >>> import numpy as np
    >>> import albumentations as A
    >>>
    >>> class CustomDistortion(A.BaseDistortion):
    ...     def __init__(self, *args, **kwargs):
    ...         super().__init__(*args, **kwargs)
    ...         # Add custom parameters here
    ...
    ...     def sample_parameters(self, params, data, targets, sampling):
    ...         height, width = targets.require_aligned_spatial_shape(2)
    ...         # Generate distortion maps
    ...         map_x = np.zeros((height, width), dtype=np.float32)
    ...         map_y = np.zeros((height, width), dtype=np.float32)
    ...         # Apply your custom distortion logic here
    ...         # ...
    ...         return SampledParams(params={"map_x": map_x, "map_y": map_y})
    >>>
    >>> # Prepare sample data
    >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
    >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
    >>> bbox_labels = [1, 2]
    >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
    >>> keypoint_labels = [0, 1]
    >>>
    >>> # Apply the custom distortion
    >>> transform = A.Compose([
    ...     CustomDistortion(
    ...         interpolation=A.cv2.INTER_LINEAR,
    ...         mask_interpolation=A.cv2.INTER_NEAREST,
    ...         keypoint_remapping_method="mask",
    ...         p=1.0
    ...     )
    ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
    ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
    >>>
    >>> # Apply the transform
    >>> transformed = transform(
    ...     image=image,
    ...     mask=mask,
    ...     bboxes=bboxes,
    ...     bbox_labels=bbox_labels,
    ...     keypoints=keypoints,
    ...     keypoint_labels=keypoint_labels
    ... )
    >>>
    >>> # Get results
    >>> transformed_image = transformed['image']
    >>> transformed_mask = transformed['mask']
    >>> transformed_bboxes = transformed['bboxes']
    >>> transformed_keypoints = transformed['keypoints']
"""
```

### Comments

- Add comments for complex logic
- Explain why, not what (the code shows what)
- Keep comments up to date with code changes

## Performance Optimization Checklist

Runtime performance is an evidence problem, not a list of automatic rewrites. For functional-layer code, `apply`
methods, and core pipeline paths, use the techniques below to form candidates, then measure them on the affected public
route. The repo-local `performance-optimization` skill provides the full workflow: delete-first review, backend
comparison, correctness baseline, and representative benchmarks. Use the `benchmark` skill to record the baseline,
candidate, public route, and every relevant measured cell.

### 1. Eliminate Python Loops Over Pixels

Python `for y in range(h): for x in range(w):` loops are much slower than vectorized NumPy. Prefer:

- `np.mgrid` / `np.meshgrid` plus broadcasting for grid computations;
- `np.einsum` for weighted sums over control points; and
- scatter updates via fancy indexing (`arr[ys, xs] = values`) instead of per-pixel assignment.

### 2. Eliminate Per-Label Full-Array Scans

When integer labels are dense and non-negative, `np.bincount` can be a grouped reduction instead of constructing one
full-image mask per label:

```python
counts = np.bincount(labels, minlength=num_labels)
weighted_sums = np.bincount(labels, weights=values, minlength=num_labels)
means = weighted_sums / counts
```

This can replace `for label: mask = labels == label` for component sizes, sums, means, histograms, and cluster-centre
updates. Benchmark small label counts too: sparse IDs require remapping, `weights=` changes accumulation precision, and
`np.unique(..., return_inverse=True)` can cost more than the scans it replaces.

### 3. Consider `cv2.LUT` / `sz_lut` for uint8 Pixel-Wise Transforms

For a uint8 operation of the form `f(pixel) -> pixel`, compare a 256-entry LUT applied with
`sz_lut(image, lut, inplace=...)` to direct array arithmetic. LUTs often win, but the complete operation and the input
layout decide the result.

An operation that is literally a bit mask is a common exception:

```python
# Often fast for posterization-style bit masks.
mask = ~np.uint8(2 ** (8 - num_bits) - 1)
result = image & mask
```

For multichannel bit masks, benchmark broadcasted `image & masks` against a preallocated per-channel implementation.
Broadcasting small per-channel masks can create slow strided operations.

### 4. Vectorize LUT and Array Construction

Replace Python list comprehensions with NumPy vectorized equivalents when constructing arrays in a hot path:

```python
# Slow
lut = np.array([max_value - i if i >= threshold else i for i in range(256)])

# Fast
indices = np.arange(256, dtype=np.uint8)
lut = np.where(indices >= threshold, max_value - indices, indices)
```

### 5. Use `out=` for In-Place Operations

Avoid unnecessary temporaries on image-sized arrays:

```python
# Allocates a temporary.
result = np.clip(image + noise, 0, 1)

# Reuses the result buffer.
result = image + noise
np.clip(result, 0, 1, out=result)
```

Useful functions with `out=` include `np.clip`, `np.multiply`, `np.add`, and `np.divide`. Only use in-place mutation when
the public contract permits it.

### 6. Avoid Float64 Waste

NumPy defaults to float64. Specify `dtype=np.float32` for float arrays when float32 is sufficient, including
`np.arange`, `np.linspace`, `np.zeros`, `np.ones`, `np.full`, and mesh-grid inputs.

### 7. Fuse Multi-Step Operations

Replace chains of temporary allocations with a fused helper where its semantics match:

```python
# Two temporaries.
result = image + alpha * (image - blurred)

# A fused weighted sum.
result = add_weighted(image, 1.0 + alpha, blurred, -alpha)
```

### 8. Choose Backends by Benchmark

Compare NumPy, OpenCV, NumKong, StringZilla, and LUT implementations that express the complete operation. Include
dispatch, conversion, contiguity, clipping, and allocation costs, and benchmark the exact dtype, shape, and channel
matrix before switching.

- Use scalar NumPy bitwise operations, such as `image & np.uint8(mask)`, for a scalar mask.
- Use `cv2.bitwise_*` when both operands are dense contiguous arrays and `dst=` can reuse output.
- Do not allocate a full image-sized mask merely to call OpenCV; that allocation often loses.
- For a single-source Euclidean distance field, direct coordinate math with `np.arange(..., dtype=np.float32)` and
  `cv2.sqrt(..., dst=...)` may be simpler and faster than `cv2.distanceTransform`.
- For sparse multi-channel replacement, copy plus masked assignment can beat nested `np.where`; measure the density
  threshold.

### 9. Preallocate Outside Loops

Move repeated allocations outside loops and reset reusable buffers:

```python
# Slow — allocates every iteration.
for item in items:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, ...)

# Reuse one allocation.
mask = np.zeros((height, width), dtype=np.uint8)
for item in items:
    mask[:] = 0
    cv2.fillPoly(mask, ...)
```

### 10. Skip Redundant Work in Hot Paths

- Guard `np.ascontiguousarray` with `if not array.flags["C_CONTIGUOUS"]`.
- Use `first_result[np.newaxis]` instead of `np.array([first_result])` for a batch of one.
- Precompute loop-invariant expressions, for example `inverse_squared_step = 1.0 / (step * step)`.
- Prefer `np.where(mask)` to `np.argwhere(mask)` when a tuple of one-dimensional index arrays is enough.

### 11. Compare and Vectorize Random Number Generation

Use Python `random.Random` for a few scalar choices and a NumPy `Generator` for arrays as starting candidates. When
multiple generators can express the workload, benchmark them while preserving transform-local seeded isolation and
replay behaviour; a global OpenCV RNG can be ineligible even if its kernel is fast.

Replace per-element Python RNG loops with one NumPy call when an array draw is appropriate:

```python
# Slow
steps = [1 + sampling.py_random.uniform(*limit) for _ in range(n)]

# Fast
steps = (1 + sampling.random_generator.uniform(*limit, size=n)).tolist()
```

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
