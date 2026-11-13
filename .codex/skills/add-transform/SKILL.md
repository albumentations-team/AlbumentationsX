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
- `albumentations/augmentations/medical/` — medical acquisition artifacts and histology stain transforms
- `albumentations/augmentations/transforms3d/` — 3D/volume
- `albumentations/augmentations/other/` — everything else

## 2. Functional layer first

Read `../performance-optimization/SKILL.md` and its required reference completely before implementing the functional
kernel.

Implement the operation in the corresponding functional module, without transform state or invocation sampling.

- Keep the functional layer deterministic; define stochastic behavior at the transform sampling boundary.
- Use the performance workflow to compare kernels, remove redundant work, and check Albucore ownership.
- Use a concrete Albucore router when its semantics match; otherwise keep working dtype, scaling, rounding, clipping,
  and output restoration explicit in the operation owner.

## 3. Write the transform class

- Set supported parameter bounds in Pydantic `InitSchema`; follow
  [initialization-only validation](../../../docs/contributing/coding_guidelines.md#initialization-only-validation)
  for every execution path. Before implementation, define recovery for sample-dependent constraints, including
  missing annotations and impossible geometry. Errors belong at initialization; training execution must continue
  with documented, target-consistent results.
- Read [Generated Transform Target Contracts](../../../docs/design/transform-target-contracts.md) before declaring
  transform support. For every concrete public transform, its `Targets:` docstring, effective `_targets`, and active
  dispatch keys describe the same behavior. A working inherited handler counts; a stub does not. Declare `images` and
  `masks` independently because batch support does not follow from `image` or `mask`.
- For spatial transforms, read [Vector fields](../../../docs/vector-fields.md) before declaring field targets. Check
  `gradient`, `displacement`, `vector_field`, and `normal` separately: covectors, finite endpoint offsets, local
  vectors, and unit normals obey different component rules. Repeat the check for 3D targets and for 2D transforms
  applied to volumes. A field target belongs in `_targets` only when the handler preserves that meaning. Keep field
  input dtypes at float16/float32; the raster image dtype list does not apply to fields. Use the field target profiles
  and exact geometry tests rather than relying on output shape alone.
- `_targets` is the only source for active dispatch. Do not infer targets from method names or add an automatic route
  for `user_data`; it is passthrough unless a custom transform explicitly declares and implements that target.
- Read [Instance and Frame Binding](../../../docs/design/instance_binding.md) when a transform supports batches or
  structured per-frame annotations. `Compose(frame_binding=...)` owns those relationships; a leaf transform should use
  the shared dispatch path instead of unpacking frame dictionaries itself.
- Define sampling and replay behavior at the `sample_parameters` boundary. Implement the greenfield
  `sample_parameters(params, data, targets, sampling) -> SampledParams` contract;
  return `SampledParams(params={...})` for values used by every target and use actual-key `TargetParams` entries
  for representation-dependent values. The coding guidance document describes the complete contract and the hook reports
  mechanical violations.
- Use relative parameters where users should transfer a policy across image sizes.
- Use explicit `NDArray[...]` dtype annotations for images, masks, and volumes: a concrete dtype where known, or `NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]` for the three supported image dtypes. Reserve `np.ndarray` for bboxes and keypoints.
- NumPy inputs inside transform execution have explicit channel-last layouts: `(H, W, C)`, `(N, H, W, C)`, and
  `(D, H, W, C)`; grayscale is `(H, W, 1)`. Compose normalizes public channel-free inputs before dispatch. Native
  Tensor handlers use the channel-first layouts in [NumPy and Tensor routing](../../../docs/design/numpy-tensor-routing.md).
- Keep reusable pixel arithmetic in `functional.py`, not in a transform class.

## 4. Add batch optimization (`apply_to_images`)

Override `apply_to_images` only when measurement shows an advantage over the default per-image loop. Keep the
method as a thin dispatcher with explicit sampled parameters. Put the complete batch operation in a functional
helper: shared kernel or LUT setup, direct batch indexing, empty-batch handling, and any preallocated per-image loop.

Keep batch and channel axes distinct; do not reshape `(N, H, W, 1)` into `(H, W, N)` to make one OpenCV call.
Use [Benchmark](../benchmark/SKILL.md) to compare the direct operation and public Compose route.

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

Check raster edge cases: uint8, float32, single channel, and multichannel. For geometric fields, check both supported
floating dtypes, NaN support, component layouts, and the semantics-specific transform rule instead of uint8 cases.

## 7. Verify

- Every configurable public constructor parameter has a non-default contract case and every `DualTransform` mode reaches
  the applicable core target profiles.
- Transform context and target prerequisites are declared on the case without runner branches.
- Applied configuration survives strict JSON, reconstruction, and fresh-data execution.
- Run the focused tests, `tests/contracts`, the required benchmark matrix, and the relevant pre-commit hooks.
