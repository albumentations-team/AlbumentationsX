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
- Do not request 2D grayscale compatibility for Compose paths: images and volume data are channel-last with explicit
  channels (`(H,W,C)`, `(N,H,W,C)`, `(D,H,W,C)`), and grayscale is `(H,W,1)`.
- **Never auto-detect bbox type from column count** — type comes from `BboxParams.bbox_type`
- For OBB: never use raw `cv2.minAreaRect` output; use `cv2.boxPoints` then `polygons_to_obb`

## 3. Mechanical AX contracts (🔴 Critical)

Run `pre-commit run check-ax-coding-guidance --all-files` first and report its exact `AXG` diagnostic. Keep the
deterministic contract and its exceptions in `docs/contributing/coding_guidelines.md`.

Use human review for what the hook cannot decide: whether constructor fields express a coherent public API, whether
validators preserve intended semantics, and whether a proposed functional operation belongs in Albucore.

## 4. Sampling design (🔴 Critical)

Review the sampling design and replay contract here. Use the AX guidance hook for deterministic violations.

## 5. Type Safety (🔴 Critical)

- [ ] All methods have complete type hints
- [ ] `ImageType` used for image/mask/volume params and return types (not `np.ndarray`)
- [ ] `np.ndarray` used for bboxes and keypoints only
- [ ] No unsafe type conversions or missing dtype handling

## 6. Performance (🟡 Important)

Read `../performance-optimization/SKILL.md` and its required reference completely before reviewing this section.

Priority order to check:
1. Delete redundant work, full-array passes, conversions, and copies.
2. Vectorize loops when the benchmark supports the resulting memory layout.
3. Use grouped reductions such as `np.bincount` instead of repeated per-label full-array masks when labels are dense.
4. Compare LUT, NumPy, OpenCV, NumKong, and StringZilla implementations where applicable.
5. Compare Python, NumPy, and OpenCV random generation without breaking seeded isolation or replay.
6. Use `albucore.resize`, not `cv2.resize`, for image resizing.
7. Move reusable atomic image operations into Albucore.
8. Use in-place operations where ownership and aliasing make them safe.
9. Cache expensive setup in `sample_parameters` or once per batch.

### Batch Optimization Checks

- [ ] **Custom `apply_to_images`** if expensive setup (kernels, LUTs, gradient maps) can be computed once per batch
- [ ] **No redundant `ndim == 4` checks** on images — they're always 4D in batch context
- [ ] **No 2D grayscale branches** in Compose functional paths — grayscale images are `(H,W,1)`
- [ ] **No reshape trick**: Do NOT reshape `(N,H,W,1)` to `(H,W,N)` for cv2 — 2–4× slower due to non-contiguous copy + sequential channel processing

Flag any violations with a concrete speedup suggestion.

## 7. Documentation (🟡 Important)

- [ ] Docstring has `Args`, `Targets`, `Image types` sections
- [ ] Examples follow the standard pattern with image, mask, bboxes, keypoints
- [ ] Examples use `A.Compose` with `BboxParams` and `KeypointParams`

## 8. Test Coverage (🟡 Important)

- [ ] Transform has named cases in `tests/helpers/transform_cases.py`
- [ ] Every configurable public constructor parameter except `p` and `strict` has a non-default case
- [ ] Behaviorally distinct and mutually exclusive modes have separate cases
- [ ] Special targets or metadata use a deterministic factory, not a generic-test class-name branch
- [ ] Every registered mode collects against all applicable core profiles in the generated target cluster
- [ ] Target applicability comes from declared targets, bbox types, prerequisites, and channel capabilities
- [ ] Target profiles and runners contain no transform-class lists, class-name skips, or duplicated constructor kwargs
- [ ] Transform-required metadata lives in the case `context_factory`; target-dependent sampling uses `required_targets`
- [ ] Realized `applied_config` survives strict JSON, public reconstruction, and fresh-data execution
- [ ] Conflicting stochastic-policy fields are cleared when sampled fields are emitted
- [ ] Exact replay is asserted only when the applied configuration captures all relevant randomness
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
- **AXG diagnostic**: report the exact hook rule and location (for example, `AXG008`)

### 🟡 Important
- **Performance**: Use `cv2.LUT` instead of numpy indexing for pixel mapping (5-10x faster)
- **Docs**: Example does not explain the transform's target semantics

### 🟢 Suggestions
- Consider using relative `noise_range` instead of absolute pixel values
```
