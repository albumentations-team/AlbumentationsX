# Correctness Contracts

This document defines the vocabulary used by regression tests, property tests,
and public release reports. It distinguishes behavior AlbumentationsX
guarantees from behavior that is tested structurally or intentionally left
unspecified.

## Contract Dimensions

- Transform class and family: image-only, dual 2D, 3D, mixing/reference-data,
  or composition helper.
- Target support: image, images, mask, masks, bboxes, keypoints, volume,
  volumes, mask3d, masks3d, and additional targets.
- Shape contract: Compose-routed images and volumes are channel-last with an
  explicit channel dimension.
- Dtype contract: outputs preserve documented dtype semantics for the tested
  target.
- Channel contract: transforms document whether arbitrary channels, RGB only,
  or a narrower channel set is supported.
- Randomness contract: fixed seeds reproduce tested outputs.
- Label contract: bbox and keypoint labels stay aligned with surviving
  annotations.
- Bounds contract: geometric outputs remain finite and inside image or volume
  bounds after Compose filtering.
- Serialization contract: supported transforms round-trip through
  serialization and replay checks.

## Stability Modes

Golden regression cases declare one stability mode:

- `exact`: exact array values are part of the tested contract.
- `tolerance`: array values are compared with documented tolerances.
- `digest`: compact digest and sentinel metadata are compared.
- `structural`: shape, dtype, value range, and annotation invariants are
  checked, but exact pixels are not promised.

OpenCV-backed interpolation and platform-sensitive effects should avoid exact
byte-level promises unless the behavior is known to be stable across supported
dependency versions.

## Unspecified Behavior

AlbumentationsX does not promise behavior for unsupported target combinations,
invalid input arrays, direct calls that bypass documented Compose contracts, or
optional dependency cross-products that are not listed in the support policy.
