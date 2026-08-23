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

Run `pre-commit run check-ax-coding-guidance --all-files` first. Report a failing `AXG` diagnostic exactly; otherwise
record that the hook passed. Keep the deterministic contract and its exceptions in
`docs/contributing/coding_guidelines.md`.

Use human review for what the hook cannot decide: whether constructor fields express a coherent public API, whether
validators preserve intended semantics, and whether a proposed functional operation belongs in Albucore.

## 4. Sampling design (🔴 Critical)

Review the policy and replay boundary, not the hook's syntax checks:

- What is sampled per invocation, and which constructor fields remain policy rather than realized state?
- Does target-dependent sampling declare its required data and produce the same result for the same invocation seed?
- Does `applied_config` record every realized constructor value needed for reconstruction and clear a conflicting source
  policy field?
- Is `ReplayProfile.EXACT` claimed only when the emitted configuration captures every output-changing random value?
- Does the reconstructed transform execute on fresh equivalent data without mutating caller-owned inputs?

Read `docs/design/applied-config-replay-contracts.md` when a finding concerns constructor serialization, applied
configuration, or `ReplayCompose` rather than local sampling logic.

## 5. Type Safety (🔴 Critical)

- [ ] All methods have complete type hints
- [ ] `ImageType` used for image/mask/volume params and return types (not `np.ndarray`)
- [ ] `np.ndarray` used for bboxes and keypoints only
- [ ] No unsafe type conversions or missing dtype handling

## 6. Performance (🟡 Important)

Read `../performance-optimization/SKILL.md` and its required reference completely before reviewing this section.

Report only an applicable missed candidate: work that can be deleted, an existing Albucore primitive, an unsafe
allocation or conversion, an unexamined backend, or setup that can be shared across a batch. Tie the finding to the
affected public route and benchmark evidence; do not request an optimization merely because its pattern is generally
faster.

## 7. Documentation (🟡 Important)

Use `docstring-deep-dive` for public-docstring quality. Review whether the description lets a user choose the transform,
and whether each example demonstrates the targets and configuration that the transform actually supports.

## 8. Test Coverage (🟡 Important)

- Does every public constructor mode have a named registry case, including mutually exclusive behavior?
- Do declared targets, bbox types, metadata prerequisites, and genuine channel restrictions produce the expected
  generated target-profile pairs without class-name branches or skips?
- Does a focused test state a stronger semantic property than the generated cluster already covers?
- Does replay cross strict JSON, reconstruct publicly, and execute on fresh data at the declared strength?

Use `docs/design/transform-target-contracts.md` and `docs/design/applied-config-replay-contracts.md` for the exact
registry and profile contracts.

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
- **Performance**: The public Compose route has no baseline comparison for the proposed LUT candidate
- **Docs**: Example does not explain the transform's target semantics

### 🟢 Suggestions
- Consider using relative `noise_range` instead of absolute pixel values
```
