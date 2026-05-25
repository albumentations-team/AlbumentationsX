---
name: review-transform
description: Run the full shared Codex review checklist against a transform. Use when the user asks to review, audit, or check a transform for correctness, performance, or API consistency.
---

# Review Transform

Run these checks in order. Report issues with severity: 🔴 Critical, 🟡 Important, 🟢 Suggestion.

## 1. Dead Code (🔴 Critical)

- Any methods defined but never called within the class or externally
- Unused imports at the top of the file
- Unreachable branches (`if False:`, conditions that can never be true)

## 2. Correctness

- Mathematical/logical errors in the transform
- Off-by-one errors in coordinate handling
- Incorrect dtype preservation (uint8 in → uint8 out, float32 in → float32 out)
- BBox/keypoint coordinate correctness after spatial transforms
- Do not request 2D grayscale compatibility for Compose paths: images and volumes are channel-last with explicit channels
  (`(H,W,C)`, `(N,H,W,C)`, `(D,H,W,C)`, `(N,D,H,W,C)`), and grayscale is `(H,W,1)`.
- **Never auto-detect bbox type from column count** — type comes from `BboxParams.bbox_type`
- For OBB: never use raw `cv2.minAreaRect` output; use `cv2.boxPoints` then `polygons_to_obb`

## 3. API Consistency (🔴 Critical)

- [ ] **No "Random" prefix** in class name
- [ ] **Range params use `_range` suffix**: `brightness_range`, not `brightness_limit`
- [ ] **`fill` not `fill_value`**, **`fill_mask` not `fill_mask_value`**
- [ ] **`border_mode`** not `mode` or `pad_mode`
- [ ] **No default values in `InitSchema`** (except Pydantic discriminator fields)
- [ ] **No default values in `apply_*` method args** (other than `self`, `**params`)
- [ ] All `InitSchema` fields use `Annotated[...]` validators where applicable
- [ ] **No `get_transform_init_args_names()` override** — the base class auto-detects from `__init__` via MRO

## 4. Random Number Generation (🔴 Critical)

- [ ] All randomness lives in `get_params` or `get_params_dependent_on_data`, NOT in `apply_*`
- [ ] Uses `self.py_random` for simple ops (faster)
- [ ] Uses `self.random_generator` only when numpy arrays are needed
- [ ] **No `np.random.*` or `random.*` module-level calls** anywhere in the class

## 5. Type Safety (🔴 Critical)

- [ ] All methods have complete type hints
- [ ] `ImageType` used for image/mask/volume params and return types (not `np.ndarray`)
- [ ] `np.ndarray` used for bboxes and keypoints only
- [ ] No unsafe type conversions or missing dtype handling

## 6. Performance (🟡 Important)

Priority order to check:
1. **`cv2.LUT`** used for pixel lookup operations (fastest)
2. **`albucore.resize` not `cv2.resize`** for image resizing (handles 5+ channels, INTER_AREA, etc.)
3. **`cv2` over numpy** for image ops where applicable
4. **Vectorized numpy** instead of Python loops
5. **In-place ops** where safe (avoid unnecessary `.copy()`)
6. No repeated array allocations in tight loops
7. Expensive computations cached in `get_params` / `get_params_dependent_on_data`

### Batch Optimization Checks

- [ ] **Custom `apply_to_images`** if expensive setup (kernels, LUTs, gradient maps) can be computed once per batch
- [ ] **No redundant `ndim == 4` checks** on images — they're always 4D in batch context
- [ ] **No 2D grayscale branches** in Compose functional paths — grayscale images are `(H,W,1)`
- [ ] **No reshape trick**: Do NOT reshape `(N,H,W,1)` to `(H,W,N)` for cv2 — 2–4× slower due to non-contiguous copy + sequential channel processing

Flag any violations with a concrete speedup suggestion.

## 7. Documentation (🟡 Important)

- [ ] Docstring has `Args`, `Targets`, `Image types` sections
- [ ] `Examples` section present (plural, not "Example")
- [ ] Examples follow the standard pattern with image, mask, bboxes, keypoints
- [ ] Examples use `A.Compose` with `BboxParams` and `KeypointParams`
- [ ] No `---` sequences in docstring (pre-commit will catch this but check anyway)

## 8. Test Coverage (🟡 Important)

- [ ] Transform appears in `get_dual_transforms()` or `get_image_only_transforms()` in `tests/utils.py`
- [ ] Tested with uint8 and float32
- [ ] Tested with 1, 3, and N channels (if applicable)
- [ ] Edge cases covered (empty bboxes, zero-area regions, etc.)
- [ ] Tests use `seed=137` (not 42)
- [ ] Tests use `np.testing` assertions (not plain `assert`)

## 9. Code Quality (🟢 Suggestion)

- [ ] No unused imports
- [ ] No overly complex logic that could be simplified
- [ ] Relative parameters (fractions) preferred over fixed pixel values
- [ ] Consistent style with similar existing transforms

## Reporting Format

```
## Review: <TransformName>

### 🔴 Critical
- **Dead code**: `_unused_method` is never called (line 42)
- **API**: Parameter `fill_value` should be `fill`

### 🟡 Important
- **Performance**: Use `cv2.LUT` instead of numpy indexing for pixel mapping (5-10x faster)
- **Docs**: Missing `Examples` section in docstring

### 🟢 Suggestions
- Consider using relative `noise_range` instead of absolute pixel values
```
