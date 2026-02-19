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

Add the pure function in the corresponding `functional.py` file (no class state, no RNG):

```python
def my_transform(img: np.ndarray, param1: float, param2: int) -> np.ndarray:
    ...
```

- Accept `np.ndarray`, return `np.ndarray`
- No randomness — all random values come from `get_params` / `get_params_dependent_on_data`
- Prefer `cv2` over numpy for performance (see benchmarking rules)
- Use `cv2.LUT` for lookup-based pixel ops (fastest)
- Use `@uint8_io` / `@float32_io` decorators if dtype conversion is needed

## 3. Write the transform class

```python
class MyTransform(DualTransform):  # or ImageOnlyTransform / NoOp
    """One-line summary.

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

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("param_range",)
```

### Critical rules:
- **NO "Random" prefix** in the class name
- **Parameter ranges** use `_range` suffix: `brightness_range`, not `brightness_limit`
- **`fill` not `fill_value`**, **`fill_mask` not `fill_mask_value`**
- **`border_mode`** not `mode` or `pad_mode`
- **NO default values in `InitSchema`** (except Pydantic discriminator fields)
- **NO default argument values in `apply_*` methods** (other than `self`, `**params`)
- **All randomness in `get_params` or `get_params_dependent_on_data`**, never in `apply_*`
- Use **`self.py_random`** for simple random ops, **`self.random_generator`** only when numpy arrays needed
- **Never** use `np.random.*` or `random.*` module directly
- Prefer **relative parameters** (fractions of image size) over fixed pixel values
- Use **`ImageType`** for image/mask/volume type hints, `np.ndarray` only for bboxes/keypoints

## 4. Export the transform

Add to `albumentations/__init__.py`:
```python
from albumentations.augmentations.<module>.transforms import MyTransform
```

Add to `albumentations/augmentations/<module>/__init__.py` if one exists.

## 5. Write tests

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

Also add it to the parametrized lists in `tests/utils.py`:
- `get_dual_transforms()` if it's a `DualTransform`
- `get_image_only_transforms()` if it's `ImageOnlyTransform`

Check edge cases: uint8, float32, single channel, multichannel.

## 6. Verify checklist

- [ ] No "Random" prefix in class name
- [ ] `_range` suffix on range params
- [ ] `fill` / `fill_mask` (not `fill_value` / `fill_mask_value`)
- [ ] No defaults in `InitSchema`
- [ ] No defaults in `apply_*` method args
- [ ] All random ops in `get_params` / `get_params_dependent_on_data`
- [ ] Using `self.py_random` or `self.random_generator` (not `np.random` / `random`)
- [ ] `ImageType` for image type hints
- [ ] Docstring has `Args`, `Targets`, `Image types`, `Examples` sections
- [ ] Examples section uses plural "Examples" (not "Example")
- [ ] Exported in `albumentations/__init__.py`
- [ ] Tests added (parametrized, seed=137, `np.testing` assertions)
- [ ] Pre-commit passes: `pre-commit run --all-files`
- [ ] Tests pass: `uv run pytest -m "not slow"`
