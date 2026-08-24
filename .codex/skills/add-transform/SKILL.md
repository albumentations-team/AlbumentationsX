---
name: add-transform
description: Full checklist for adding a new transform to AlbumentationsX. Use when the user asks to add, implement, or create a new transform/augmentation.
---

# Add Transform

Use this workflow for architectural choices and coverage. Pre-commit owns fixed API and style conventions; read its
diagnostic instead of maintaining a parallel checklist here.

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
def my_transform(img: np.ndarray, param1: float, param2: int) -> np.ndarray: ...
```

- Keep the functional layer deterministic; define stochastic behavior at the transform sampling boundary.
- Delete redundant work and full-array passes before selecting a backend
- Compare applicable NumPy, OpenCV, NumKong, StringZilla, and LUT implementations
- Consider `np.bincount` for repeated reductions over dense non-negative integer labels
- Move a reusable atomic image operation into Albucore instead of duplicating it locally
- Use in-place operations only when ownership and aliasing make mutation safe
- Use `@uint8_io` / `@float32_io` decorators if dtype conversion is needed

## 3. Write the transform class

- Define sampling and replay behavior at the `sample_parameters` boundary. Implement the greenfield
  `sample_parameters(params, data, targets, sampling) -> SampledParams` contract;
  return `SampledParams.shared_only(...)` for shared values and use actual-key `TargetParams` entries
  for representation-dependent values. The coding guidance document describes the complete contract and the hook reports
  mechanical violations.
- Use relative parameters where users should transfer a policy across image sizes.
- Use `ImageType` for image, mask, and volume signatures; reserve `np.ndarray` for bboxes and keypoints.
- Compose inputs always have a channel dimension: `(H, W, C)`, `(N, H, W, C)`, and `(D, H, W, C)`; grayscale is
  `(H, W, 1)`. Do not add Compose-path compatibility branches for two-dimensional grayscale data.
- Keep reusable pixel arithmetic in `functional.py`, not in a transform class.

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

Export it through `albumentations/__init__.py` and the relevant augmentation package initializer.

## 6. Write tests

Register the transform in `tests/helpers/transform_cases.py`:

- Add at least one named `TransformContractCase`.
- Give every configurable public constructor parameter except `p` and `strict` a non-default case. A singleton `Literal`
  equal to its default is non-configurable and needs no artificial mode.
- Add distinct cases for mutually exclusive fields or behaviorally different modes.
- Select a primary target factory and add transform-required metadata through `context_factory`; do not combine standard
  targets and external metadata in a second transform-specific data inventory.
- Declare `required_targets` when parameter sampling needs a non-empty mask or bbox collection.
- Use `ReplayProfile.EXACT` only when `applied_config` resolves all randomness required to reproduce every supplied
  target; otherwise use `RUNNABLE`.
- Do not add another class/parameter inventory, compatibility adapter, broad skip, or coverage exemption.

Every registered `DualTransform` mode automatically collects against applicable core profiles from
`tests/helpers/target_profiles.py`. Confirm the new mode covers each declared target, bbox type, and volume path. Add a
new profile only when the same workload should apply to a cluster of transforms; profiles must contain no transform
class inventories or constructor kwargs. Keep exact geometry, sampling, validation, and metamorphic semantics in
focused tests.

If the transform samples constructor fields, write the realized values to `sampling.applied_overrides`.
Clear any original policy field that becomes mutually exclusive with the realized value. If a convenience alias emits
the canonical constructor's state, declare `_applied_replay_class`.

Check edge cases: uint8, float32, single channel, multichannel.

## 7. Verify

- Every configurable public constructor parameter has a non-default contract case and every `DualTransform` mode reaches
  the applicable core target profiles.
- Transform context and target prerequisites are declared on the case without runner branches.
- Applied configuration survives strict JSON, reconstruction, and fresh-data execution.
- Run the focused tests, `tests/contracts`, the required benchmark matrix, and the relevant pre-commit hooks.
