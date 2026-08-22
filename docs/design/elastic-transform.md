# Bounded 2D ElasticTransform

**Status:** implemented

`ElasticTransform` applies one bounded, smooth XY pull field to an image and
all synchronized targets. The field is sampled from a compact cubic B-spline
coefficient lattice. Map construction therefore depends on the lattice and the
output shape, not on a full-resolution random-noise sample.

True 3D elastic deformation is a separate follow-up. The current transform
applies its 2D XY map to every depth slice. A future 3D transform will wait for
a tested dense 3D remap primitive in Albucore.

## Public contract

```python
A.ElasticTransform(
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

`displacement_range` is relative to the shorter distance between the first and
last pixel centers. If the sampled magnitude is `m`, every coefficient has a
radius of at most `m * min(H - 1, W - 1)` pixels.

`control_grid_shape` is the number of cubic B-spline coefficients, not the
number of interpolation anchors. Each dimension must be at least four. A grid
with `(rows, columns)` coefficients has `(rows - 3, columns - 3)` knot spans.

The constructor rejects configurations whose conservative topology bound is
not strict:

```text
2 * displacement_range[1]
  * sqrt((rows - 3)^2 + (columns - 3)^2) < 0.75
```

This is a breaking contract. The former dense Gaussian-field parameters and
Elastic-specific map-resolution and keypoint-policy parameters are not part of
the constructor and are not aliased.

## Field construction

The transform builds a target-to-source pull map:

```text
S(q) = q + d(q)
```

For each spatial axis, the coefficient lattice is evaluated over uniform knot
spans. With `t` in `[0, 1]`, the four cardinal cubic B-spline weights are:

```text
B0(t) = (1 - t)^3 / 6
B1(t) = (3t^3 - 6t^2 + 4) / 6
B2(t) = (-3t^3 + 3t^2 + 3t + 1) / 6
B3(t) = t^3 / 6
```

Every pixel uses the tensor product of four weights in each axis. The weights
are non-negative and sum to one, so the dense displacement stays inside the
convex hull of the sampled coefficient vectors. This gives the hard displacement
bound without a clipping pass. The field is C2 at internal knots.

Coefficients are sampled uniformly by area in a disk. The scalar magnitude
uses the invocation's Python random stream; the coefficient lattice uses the
invocation's NumPy generator. No process-global RNG is used.

## Targets and annotations

- `image` uses `interpolation`, `border_mode`, and `fill`.
- `mask` and `mask3d` use `mask_interpolation`, `border_mode`, and `fill_mask`.
- `images` and `volume` reuse one XY map for every member or depth slice.
- HBB and OBB use the existing raster-support route with constant-zero
  annotation exterior.
- Keypoints use a bounded fixed-point inverse of the continuous pull map. The
  result is accepted only when the forward residual is at most `1e-3` pixel.
  Z coordinates and extra keypoint columns are preserved.

The topology bound is a sufficient injectivity condition. Runtime code does
not run a dense fold detector or negate the field as an inverse approximation.

## Persistence

Constructor serialization keeps the stochastic policy: the original
`displacement_range` and public fields are serialized, so a reconstructed
transform samples a new coefficient lattice.

Applied configuration records the realized magnitude as `(m, m)`. It remains
constructor-valid after strict JSON transport, but it intentionally does not
freeze the sampled lattice. This is the same distinction as other stochastic
transforms that sample new geometry on each call.

`ReplayCompose` stores the compact `control_coefficients` lattice and the
recorded spatial shape. It does not store dense maps. Replay reproduces all
targets exactly for the recorded `(H, W)` shape and raises `ValueError` for a
different spatial shape.

## Implementation boundary

`BaseRemapTransform` owns synchronized raster, batch, volume, mask3d, and bbox
dispatch. `ElasticTransform` owns cubic coefficient sampling and its continuous
keypoint inverse. Other remap transforms retain `BaseDistortion`'s independent
map-resolution and direct/mask keypoint policies.

The functional layer builds one pair of float32 maps per invocation. It uses
the same maps for every raster target, avoiding repeated random generation and
repeated field expansion.

## Verification

The focused suite covers constructor and topology validation, scalar agreement
with the cubic basis, constant fields, coefficient bounds, strict-JSON replay,
applied-config reconstruction, fixed-point inverse accuracy, volume slice
consistency, and exact identity. Shared distortion, OBB, serialization, and
target-contract suites cover integration with the rest of the library.

Performance evidence must compare the exact baseline and exact candidate head on
the direct and `Compose` routes for 256, 512, and 1024-pixel inputs, 1, 3, and 5
channels, and both `uint8` and `float32`. The benchmark must retain raw cells and
state any regressions above the repository's five-percent threshold.

The 2D cutover benchmark ran on 2026-08-22 with Python 3.12.7, NumPy 2.5.2,
OpenCV 5.0.0, and OpenCV threads pinned to one.
It compared `3b31262` with the candidate head across 108 direct, `Compose`,
batch, and volume cells. The candidate was faster in every cell: the minimum
speedup was `1.07x`, the median was `1.79x`, and the maximum was `5.28x`.

| Route | Target | Shape | Channels | Dtype | Baseline | Cubic B-spline | Speedup |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| direct | image | 512x512 | 3 | uint8 | 3.504 ms | 0.784 ms | 4.47x |
| `Compose` | image | 512x512 | 3 | uint8 | 3.301 ms | 0.746 ms | 4.42x |
| direct | volume, D=16 | 512x512 | 3 | uint8 | 14.591 ms | 8.116 ms | 1.80x |
| `Compose` | volume, D=16 | 512x512 | 3 | uint8 | 14.125 ms | 8.234 ms | 1.72x |

## Related work

- [Issue #453](https://github.com/albumentations-team/AlbumentationsX/issues/453)
- [Issue #327: 3D ElasticTransform](https://github.com/albumentations-team/AlbumentationsX/issues/327)
- [Applied configuration and replay contracts](applied-config-replay-contracts.md)
- [Bounding box processing](bounding_boxes.md)
