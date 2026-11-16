# Unreleased behavior changes

Release baseline: 2.6.3

Follow the [behavior compatibility policy](../maintaining/behavior-compatibility.md) when updating this inventory.
Earlier changes shipped in [2.6.3](2.6.3.md), [2.6.2](2.6.2.md), [2.6.1](2.6.1.md), and [2.6.0](2.6.0.md).

## Validation

### ChannelDropout accepts mean fill

- Previous behavior: `ChannelDropout` accepted a numeric constant for `fill`.
- New behavior: `fill="mean"` replaces each selected channel with its own spatial mean. Images and volumes in a
  batch compute their means independently. Integer storage rounds the mean to the nearest integer.
- Affected users: Pipelines modeling missing channels while retaining their average intensity.
- Migration: Use `A.ChannelDropout(fill="mean")` to opt in. Use 2.6.4 or newer to load configurations that use this
  fill value. Numeric constants and the default fill keep their existing behavior.
- First affected version: unreleased
- Contract: [Target-specific parameter sampling](../design/target-specific-parameter-sampling.md).
- Tests: [ChannelDropout mean fill](../../tests/test_channel_dropout.py).

## Geometry

### Supported transforms accept targets at different resolutions

- Previous behavior: Some transforms rejected mixed-resolution inputs through `additional_targets` even when
  `is_check_shapes=False`. Some also rejected the same sample only during replay.
- New behavior: Supported symmetries, fixed padding and resize, morphology, and dropout can apply to each raster
  at its own resolution. Transforms that compute shared geometry from an input shape still require alignment.
- Affected users: Pipelines that augment images, masks, or volumes at different resolutions together.
- Migration: Use `A.Compose([A.D4()], additional_targets={"low": "image"}, is_check_shapes=False)` or
  `ReplayCompose` with the same options. Keep bounding boxes and keypoints in the primary raster's coordinates.
- First affected version: unreleased
- Contract: [Geometry across targets](../geometry.md#geometry-is-shared-across-targets).
- Tests: [Mixed-resolution Compose and replay](../../tests/test_sampled_params.py),
  [shared D4 symmetry](../../tests/test_flip_masks_comprehensive.py).

## Determinism

### Replay uses the current primary annotation frame

- Previous behavior: Replaying some symmetries on a differently sized sample could use the recorded raster
  dimensions for keypoints. An empty primary batch could fail annotation processing or let an alias supply the replay frame.
- New behavior: Supported transforms retain the recorded operation and compute annotation coordinates from the current
  primary image or volume dimensions. An empty batch with known dimensions still defines the primary frame.
- Affected users: Pipelines that apply one recorded symmetry or fixed resize to compatible inputs of different sizes.
- Migration: Pass the existing record and the new primary image or volume to `A.ReplayCompose.replay(record, ...)`.
  Keep any accompanying annotation coordinates in the new primary frame. Dense recorded parameters still require
  compatible input shapes.
- First affected version: unreleased
- Contract: [Replay contracts](../determinism.md#choose-the-persistence-contract).
- Tests: [Replay annotation dimensions](../../tests/test_sampled_params.py).

## Constructor defaults

none

## Deprecations

none

## Regression vectors

none
