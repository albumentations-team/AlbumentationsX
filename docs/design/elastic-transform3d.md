# ElasticTransform3D: bounded smooth volume deformation

**Status:** implemented

## Overview

`ElasticTransform3D` applies one smooth XYZ pull field to a volume, `mask3d`, and XYZ keypoints. It samples compact cubic B-spline coefficient planes, expands them into one normalized `(D, H, W, 3)` sampling grid, and calls Albucore `remap3d` once for each raster target.

Use it when a label remains valid under small, smooth changes to volumetric anatomy or acquisition geometry. It changes voxel-index coordinates only. Voxel spacing, orientation metadata, and 3D bounding boxes are outside this transform's contract.

## Problem Statement

Applying the 2D `ElasticTransform` map to every depth slice leaves depth unchanged. A true 3D deformation needs each displacement component to depend on depth, height, and width.

Generating dense 3D random noise and smoothing it pays for full-resolution random generation, a 3D filter, intermediate fields, and resampling. `ElasticTransform3D` instead builds its field from three compact orthogonal control planes. The representation cannot model arbitrary local three-way variation. That limit keeps the field smooth, bounded, and inexpensive to construct.

## Design Principles

### One compact field defines all geometry

Each invocation samples one magnitude and three coefficient planes: XY, XZ, and YZ. The transform averages their contributions, so identical sampled geometry drives every volume, mask, and keypoint target.

### The deformation remains bounded and invertible

`displacement_range=(low, high)` is relative to the shortest positive voxel-center span. For sampled magnitude `m`, every control vector has radius at most `m * shortest_span`. The constructor accepts only configurations satisfying:

```text
2 * high * sqrt((rows - 3)^2 + (columns - 3)^2) < 0.75
```

The bound is strict. It keeps the continuous pull map in the supported no-fold regime and permits a fixed-point inverse for keypoints.

### The hot path avoids redundant dense work

The implementation samples `2 * rows * columns` values per control plane, expands only 2D planes, and adds each result directly into the final sampling grid. It does not create dense 3D random noise, run 3D smoothing, allocate a separate dense displacement volume, create a meshgrid, or resample once per plane.

### Tensor input follows the measured faster route

NumPy volumes use `(D, H, W, C)`. CPU Tensor `volume` and `mask3d` inputs with one channel use `(C, D, H, W)` and `(D, H, W)` directly. Multi-channel Tensor volumes use Compose's single NumPy bridge and return in `(C, D, H, W)` layout because that route is faster or equally fast on the measured matrix.

The current routing decision was measured on 2026-08-28 on an Apple M4 Max with Torch restricted to one CPU thread. Three ABBA pairs covered the standard `(8, 64, 64)` and `(16, 128, 128)` volumes; C=1, C=3, and C=5; and `uint8` and `float32`. C=1 stayed within 4% of the bridge. C=3 included a 14% native slowdown, and C=5 was 20–41% slower natively, so only C=1 stays direct. The Tensor ASV lane tracks the standard volume cases. A routing decision for a larger volume requires a separate ABBA run.

## Implementation

### Public API

```python
A.ElasticTransform3D(
    displacement_range=(0.02, 0.05),
    control_grid_shape=(7, 7),
    interpolation=cv2.INTER_LINEAR,
    mask_interpolation=cv2.INTER_NEAREST,
    border_mode=cv2.BORDER_CONSTANT,
    fill=0,
    fill_mask=0,
    p=0.5,
)
```

`control_grid_shape=(rows, columns)` contains cubic B-spline coefficients. Both dimensions must be at least four. `interpolation` selects volume resampling, while the cubic field basis remains fixed.

### Field construction and coordinate convention

For output coordinate `q = (x, y, z)`, the source coordinate is `S(q) = q + d(q)`. Positive displacement samples from a larger source coordinate. Let `V_xy`, `V_xz`, and `V_yz` be the three plane fields embedded in XYZ vector space. The final displacement is:

```text
d(z, y, x) = mean(V_xy(y, x), V_xz(z, x), V_yz(z, y))
```

where the mean includes all three planes.

The sampler converts the compact planes to Albucore's normalized `(x, y, z)` grid once and passes it to every raster target. `ReplayCompose` persists only the compact sampler data and rebuilds the grid after a JSON round trip.

### Targets and persistence

- `volume` uses `interpolation`, `border_mode`, and `fill`.
- `mask3d` uses `mask_interpolation`, `border_mode`, and `fill_mask`; channel-less integer masks retain their dtype.
- Keypoints use a bounded fixed-point inverse of `S`. Accepted rows have forward residual at most `1e-3` voxel units, and trailing keypoint columns remain unchanged.
- `ReplayCompose` stores compact coefficient planes and the input spatial shape. It reproduces the geometry for the same shape and raises `ValueError` for another shape.
- Applied configuration records the realized magnitude as `displacement_range=(m, m)` and samples fresh coefficient planes when reconstructed.

## Testing Strategy

Focused tests cover constructor validation, the strict topology boundary, every supported raster dtype path, fill behavior, target alignment, keypoint inversion, exact identity, seeded execution, strict-JSON replay, applied-configuration reconstruction, additional targets, and native Tensor output parity.

The permanent ASV coverage has two layers:

- `TimeVolumetricFullMatrix` runs the public Compose route over the standard volume size and dtype matrix.
- `TimeFunctional3DKernels` measures complete grid construction plus `remap3d`, so a field-only speedup cannot hide a slower resampling route.

External MONAI and TorchIO transforms use dense Gaussian and full 3D B-spline fields, respectively. Their public parameters do not express this transform's bounded orthogonal-plane field exactly. Compare them only under a documented matched-workload protocol; do not call a diagnostic timing result semantic equivalence.

## References

- [Issue #327: ElasticTransform3D design](https://github.com/albumentations-team/AlbumentationsX/issues/327)
- [Bounded 2D ElasticTransform](elastic-transform.md)
- [Applied configuration and replay contracts](applied-config-replay-contracts.md)
- [Torch CPU backend and Tensor-native Compose](torch-cpu-backend-migration.md)
- [Albucore](https://github.com/albumentations-team/albucore)
