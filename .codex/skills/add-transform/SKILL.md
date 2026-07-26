---
name: add-transform
description: Full checklist for adding a new transform to AlbumentationsX. Use when the user asks to add, implement, or create a new transform/augmentation.
---

# Add Transform

Follow this checklist in order. Do not skip steps.

## 1. Choose the right module

Put the transform in the most specific matching subpackage:
- `albumentations/augmentations/geometric/` — spatial transforms (flip, rotate, warp, etc.)
- `albumentations/augmentations/pixel/` — pixel-level (color, brightness, noise, etc.)
- `albumentations/augmentations/dropout/` — masking/dropout
- `albumentations/augmentations/blur/` — blurring
- `albumentations/augmentations/crops/` — cropping
- `albumentations/augmentations/mixing/` — multi-image mixing
- `albumentations/augmentations/transforms3d/` — 3D/volume
- `albumentations/augmentations/other/` — everything else

## 2. Functional layer first

Read `../performance-optimization/SKILL.md` and its required reference completely before implementing the functional
kernel.

Add the pure function in the corresponding `functional.py` file (no class state, no RNG):

```python
def my_transform(img: np.ndarray, param1: float, param2: int) -> np.ndarray:
    ...
```

- Accept `np.ndarray`, return `np.ndarray`
- No randomness — all random values come from `get_params` / `get_params_dependent_on_data`
- Delete redundant work and full-array passes before selecting a backend
- Compare applicable NumPy, OpenCV, NumKong, StringZilla, and LUT implementations
- Consider `np.bincount` for repeated reductions over dense non-negative integer labels
- Move a reusable atomic image operation into Albucore instead of duplicating it locally
- Use in-place operations only when ownership and aliasing make mutation safe
- Use `@uint8_io` / `@float32_io` decorators if dtype conversion is needed

## 3. Write the transform class

- Do not add docstrings to `apply` or `apply_to_*` methods; the transform class docstring and `transforms_interface` are sufficient.

```python
class MyTransform(DualTransform):  # or ImageOnlyTransform / NoOp
    """First paragraph (120–160 chars): elevator pitch — what the transform does, how it works in one sentence, when to use it. No "Parameters: x, y", "Targets:", return type, or "Supports uint8/float32"; no "Used by X". Two lines, wrap at 120.

    More detail about what the transform does.

    Args:
        param_range: (min, max) tuple controlling X. Default: (0.1, 0.3).
        fill: Padding value for image. Default: 0.
        fill_mask: Padding value for masks. Default: 0.
        p: Probability. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        >>> bbox_labels = [1]
        >>> keypoints = np.array([[20, 30]], dtype=np.float32)
        >>> keypoint_labels = [0]
        >>>
        >>> transform = A.Compose([
        ...     A.MyTransform(param_range=(0.1, 0.3), p=1.0)
        ... ], bbox_params=A.BboxParams(coord_format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(coord_format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> result = transform(
        ...     image=image, mask=mask,
        ...     bboxes=bboxes, bbox_labels=bbox_labels,
        ...     keypoints=keypoints, keypoint_labels=keypoint_labels,
        ... )
    """

    class InitSchema(BaseTransformInitSchema):
        param_range: Annotated[tuple[float, float], AfterValidator(nondecreasing)]
        # NO default values here (except discriminator fields)

    def __init__(self, param_range: tuple[float, float], p: float = 0.5):
        super().__init__(p=p)
        self.param_range = param_range

    def apply(self, img: ImageType, param1: float, **params: Any) -> ImageType:
        # NO default values for param1 here
        return fpixel.my_transform(img, param1)

    def get_params(self) -> dict[str, Any]:
        return {
            "param1": self.py_random.uniform(*self.param_range),
        }
```

### Critical rules:
- **NO `get_transform_init_args_names()` override** — the base class reads the concrete transform's public `__init__`
  signature. Do not expose parent-only implementation fields.
- **NO "Random" prefix** in the class name
- **Parameter ranges** use `_range` suffix: `brightness_range`, not `brightness_limit`
- **`fill` not `fill_value`**, **`fill_mask` not `fill_mask_value`**
- **`border_mode`** not `mode` or `pad_mode`
- **NO default values in `InitSchema`** (except Pydantic discriminator fields)
- **Range parameters are always `tuple[T, T]`**, never `T | tuple[T, T]` — no union with a scalar. Users always pass a tuple.
- **NO default argument values in `apply_*` methods** (other than `self`, `**params`)
- **All randomness in `get_params` or `get_params_dependent_on_data`**, never in `apply_*`
- Use **`self.py_random`** for simple random ops, **`self.random_generator`** only when numpy arrays needed
- **Never** use `np.random.*` or `random.*` module directly
- Prefer **relative parameters** (fractions of image size) over fixed pixel values
- Use **`ImageType`** for image/mask/volume type hints, `np.ndarray` only for bboxes/keypoints
- **Use descriptive variable names** — avoid single-letter or generic names like `x`, `y`, `dx`, `dy`, `cx`, `cy`. Prefer `pixel_cols`, `norm_x`, `center_col`, `run_starts`, `col_x`, etc. Names should read like documentation.
- **Images and volumes under Compose always have channels** — images are `(H, W, C)`, image batches are
  `(N, H, W, C)`, volumes are `(D, H, W, C)`, and volume batches are `(N, D, H, W, C)`.
- **Grayscale under Compose is `(H, W, 1)`**, not `(H, W)`. Do not add functional-layer compatibility branches for
  2D grayscale images in code reached through `Compose`.
- Branch on `ndim` to distinguish image vs batch vs volume paths when needed, not to infer whether channels exist.
- **Helper functions belong in `functional.py`**, never in the transform class file.

## 4. Add batch optimization (`apply_to_images`)

Override `apply_to_images` only if you can beat the default per-image loop. Priority patterns:

**Pre-compute expensive setup once per batch** (kernels, LUTs, gradient maps):
```python
def apply_to_images(self, images: ImageType, *args: Any, **params: Any) -> ImageType:
    kernel = create_kernel(params["size"])  # once, not N times
    return self._apply_to_batch(images, lambda img: convolve(img, kernel))
```

**Direct 4D indexing** for simple array ops:
```python
def apply_to_images(self, images: ImageType, channels_to_drop: list[int], **params: Any) -> ImageType:
    result = images.copy()
    result[:, :, :, channels_to_drop] = self.fill
    return result
```

**Pre-allocated loop** as fallback when params vary per image:
```python
def apply_to_images(self, images: ImageType, *args: Any, **params: Any) -> ImageType:
    result = np.empty_like(images)
    for i, image in enumerate(images):
        result[i] = self.apply(image, **params)
    return result
```

> **DO NOT** reshape `(N,H,W,1)` to `(H,W,N)` to call cv2 once — this is 2–4× slower in practice (transpose → non-contiguous copy + cv2 sequential channel processing).

## 5. Export the transform

Add to `albumentations/__init__.py`:
```python
from albumentations.augmentations.<module>.transforms import MyTransform
```

Add to `albumentations/augmentations/<module>/__init__.py` if one exists.

## 6. Write tests

Add to `tests/test_transforms.py` or `tests/test_<category>.py`:

```python
@pytest.mark.parametrize(
    ("param_range", "expected_..."),
    [
        ((0.1, 0.3), ...),
        ((0.5, 0.8), ...),
    ],
)
def test_my_transform(param_range, expected_...):
    image = TestDataFactory.create_image((100, 100, 3), dtype=np.uint8, seed=137)
    aug = A.MyTransform(param_range=param_range, p=1.0)
    result = aug(image=image)
    # use np.testing assertions, not plain assert
    np.testing.assert_...
```

Register the transform in `tests/helpers/transform_cases.py`:

- Add at least one named `TransformContractCase`.
- Give every configurable public constructor parameter except `p` and `strict` a non-default case. A singleton `Literal`
  equal to its default is non-configurable and needs no artificial mode.
- Add distinct cases for mutually exclusive fields or behaviorally different modes.
- Select or add a deterministic factory in `tests/helpers/contract_data.py` for masks, bboxes, keypoints, volumes,
  batches, or custom metadata.
- Use `ReplayProfile.EXACT` only when `applied_config` resolves all randomness required to reproduce every supplied
  target; otherwise use `RUNNABLE`.
- Do not add another class/parameter inventory, compatibility adapter, broad skip, or coverage exemption.

If the transform samples constructor fields, write the realized values to `self.applied_config`. Clear any original
policy field that becomes mutually exclusive with the realized value. If a convenience alias emits the canonical
constructor's state, declare `_applied_replay_class`.

Check edge cases: uint8, float32, single channel, multichannel.

## 7. Verify checklist

- [ ] No `get_transform_init_args_names()` override (derived from the concrete public `__init__`)
- [ ] No "Random" prefix in class name
- [ ] `_range` suffix on range params
- [ ] `fill` / `fill_mask` (not `fill_value` / `fill_mask_value`)
- [ ] No defaults in `InitSchema`
- [ ] No defaults in `apply_*` method args
- [ ] All random ops in `get_params` / `get_params_dependent_on_data`
- [ ] Using `self.py_random` or `self.random_generator` (not `np.random` / `random`)
- [ ] `ImageType` for image type hints
- [ ] Custom `apply_to_images` if expensive setup can be shared across batch
- [ ] Docstring has `Args`, `Targets`, `Image types`, `Examples` sections
- [ ] Examples section uses plural "Examples" (not "Example")
- [ ] Exported in `albumentations/__init__.py`
- [ ] Tests added (parametrized, seed=137, `np.testing` assertions)
- [ ] Named cases added to `tests/helpers/transform_cases.py`
- [ ] Every configurable public constructor parameter has a non-default case
- [ ] Applied configuration passes `uv run pytest -q tests/contracts`
- [ ] Constructor dict/JSON/YAML round trips pass in `tests/test_serialization.py`
- [ ] Pre-commit passes: `pre-commit run --all-files`
- [ ] Tests pass: `uv run pytest -m "not slow"`
