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

<details>
<summary>Full 108-cell matrix</summary>

Each row reports the median wall time per transform call. The baseline is commit `3b31262`; the candidate is the PR head `fe471a9`.

| Route | Target | Size | Channels | Dtype | Baseline (ms) | Cubic B-spline (ms) | Speedup |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| compose | image | 256 | 1 | float32 | 0.677 | 0.211 | 3.215x |
| compose | image | 256 | 1 | uint8 | 0.700 | 0.342 | 2.051x |
| compose | image | 256 | 3 | float32 | 0.782 | 0.392 | 1.998x |
| compose | image | 256 | 3 | uint8 | 0.750 | 0.409 | 1.833x |
| compose | image | 256 | 5 | float32 | 0.944 | 0.437 | 2.163x |
| compose | image | 256 | 5 | uint8 | 1.177 | 0.608 | 1.935x |
| compose | image | 512 | 1 | float32 | 3.252 | 0.640 | 5.084x |
| compose | image | 512 | 1 | uint8 | 3.434 | 0.790 | 4.348x |
| compose | image | 512 | 3 | float32 | 3.904 | 1.447 | 2.697x |
| compose | image | 512 | 3 | uint8 | 3.301 | 0.746 | 4.423x |
| compose | image | 512 | 5 | float32 | 4.257 | 1.952 | 2.180x |
| compose | image | 512 | 5 | uint8 | 4.423 | 2.076 | 2.130x |
| compose | image | 1024 | 1 | float32 | 13.749 | 2.710 | 5.073x |
| compose | image | 1024 | 1 | uint8 | 13.070 | 3.580 | 3.650x |
| compose | image | 1024 | 3 | float32 | 14.508 | 5.095 | 2.848x |
| compose | image | 1024 | 3 | uint8 | 14.174 | 3.411 | 4.156x |
| compose | image | 1024 | 5 | float32 | 16.787 | 7.819 | 2.147x |
| compose | image | 1024 | 5 | uint8 | 18.202 | 9.514 | 1.913x |
| compose | images | 256 | 1 | float32 | 1.593 | 0.890 | 1.790x |
| compose | images | 256 | 1 | uint8 | 1.935 | 1.341 | 1.442x |
| compose | images | 256 | 3 | float32 | 3.092 | 2.294 | 1.348x |
| compose | images | 256 | 3 | uint8 | 1.999 | 1.277 | 1.566x |
| compose | images | 256 | 5 | float32 | 4.527 | 3.013 | 1.503x |
| compose | images | 256 | 5 | uint8 | 4.805 | 4.345 | 1.106x |
| compose | images | 512 | 1 | float32 | 7.073 | 3.311 | 2.136x |
| compose | images | 512 | 1 | uint8 | 8.507 | 4.814 | 1.767x |
| compose | images | 512 | 3 | float32 | 14.686 | 10.364 | 1.417x |
| compose | images | 512 | 3 | uint8 | 8.576 | 4.557 | 1.882x |
| compose | images | 512 | 5 | float32 | 22.339 | 16.981 | 1.316x |
| compose | images | 512 | 5 | uint8 | 19.976 | 15.387 | 1.298x |
| compose | images | 1024 | 1 | float32 | 37.633 | 15.566 | 2.418x |
| compose | images | 1024 | 1 | uint8 | 38.125 | 17.570 | 2.170x |
| compose | images | 1024 | 3 | float32 | 63.554 | 38.476 | 1.652x |
| compose | images | 1024 | 3 | uint8 | 38.063 | 19.383 | 1.964x |
| compose | images | 1024 | 5 | float32 | 83.360 | 60.409 | 1.380x |
| compose | images | 1024 | 5 | uint8 | 83.034 | 70.844 | 1.172x |
| compose | volume | 256 | 1 | float32 | 2.601 | 1.795 | 1.449x |
| compose | volume | 256 | 1 | uint8 | 3.343 | 2.733 | 1.223x |
| compose | volume | 256 | 3 | float32 | 5.237 | 3.826 | 1.369x |
| compose | volume | 256 | 3 | uint8 | 3.294 | 2.364 | 1.393x |
| compose | volume | 256 | 5 | float32 | 8.408 | 7.054 | 1.192x |
| compose | volume | 256 | 5 | uint8 | 9.603 | 8.057 | 1.192x |
| compose | volume | 512 | 1 | float32 | 11.553 | 5.516 | 2.095x |
| compose | volume | 512 | 1 | uint8 | 15.180 | 8.975 | 1.691x |
| compose | volume | 512 | 3 | float32 | 28.977 | 22.215 | 1.304x |
| compose | volume | 512 | 3 | uint8 | 14.125 | 8.234 | 1.715x |
| compose | volume | 512 | 5 | float32 | 37.478 | 34.966 | 1.072x |
| compose | volume | 512 | 5 | uint8 | 36.724 | 30.872 | 1.190x |
| compose | volume | 1024 | 1 | float32 | 59.723 | 31.400 | 1.902x |
| compose | volume | 1024 | 1 | uint8 | 67.590 | 34.144 | 1.980x |
| compose | volume | 1024 | 3 | float32 | 121.190 | 83.986 | 1.443x |
| compose | volume | 1024 | 3 | uint8 | 63.355 | 40.462 | 1.566x |
| compose | volume | 1024 | 5 | float32 | 160.429 | 119.308 | 1.345x |
| compose | volume | 1024 | 5 | uint8 | 164.051 | 133.911 | 1.225x |
| direct | image | 256 | 1 | float32 | 0.700 | 0.256 | 2.736x |
| direct | image | 256 | 1 | uint8 | 0.849 | 0.313 | 2.717x |
| direct | image | 256 | 3 | float32 | 0.846 | 0.396 | 2.137x |
| direct | image | 256 | 3 | uint8 | 0.828 | 0.290 | 2.851x |
| direct | image | 256 | 5 | float32 | 1.139 | 0.464 | 2.455x |
| direct | image | 256 | 5 | uint8 | 1.186 | 0.642 | 1.848x |
| direct | image | 512 | 1 | float32 | 3.436 | 0.651 | 5.280x |
| direct | image | 512 | 1 | uint8 | 3.305 | 1.002 | 3.300x |
| direct | image | 512 | 3 | float32 | 3.966 | 1.445 | 2.744x |
| direct | image | 512 | 3 | uint8 | 3.504 | 0.784 | 4.470x |
| direct | image | 512 | 5 | float32 | 4.554 | 2.246 | 2.027x |
| direct | image | 512 | 5 | uint8 | 4.728 | 2.116 | 2.234x |
| direct | image | 1024 | 1 | float32 | 11.830 | 2.901 | 4.078x |
| direct | image | 1024 | 1 | uint8 | 12.939 | 3.903 | 3.315x |
| direct | image | 1024 | 3 | float32 | 14.499 | 5.717 | 2.536x |
| direct | image | 1024 | 3 | uint8 | 14.621 | 3.702 | 3.949x |
| direct | image | 1024 | 5 | float32 | 18.411 | 7.820 | 2.354x |
| direct | image | 1024 | 5 | uint8 | 19.540 | 10.445 | 1.871x |
| direct | images | 256 | 1 | float32 | 1.560 | 0.965 | 1.616x |
| direct | images | 256 | 1 | uint8 | 2.075 | 1.274 | 1.630x |
| direct | images | 256 | 3 | float32 | 3.381 | 2.501 | 1.352x |
| direct | images | 256 | 3 | uint8 | 1.960 | 1.133 | 1.731x |
| direct | images | 256 | 5 | float32 | 4.291 | 3.395 | 1.264x |
| direct | images | 256 | 5 | uint8 | 4.846 | 4.143 | 1.170x |
| direct | images | 512 | 1 | float32 | 6.866 | 3.442 | 1.995x |
| direct | images | 512 | 1 | uint8 | 9.031 | 5.026 | 1.797x |
| direct | images | 512 | 3 | float32 | 14.488 | 10.941 | 1.324x |
| direct | images | 512 | 3 | uint8 | 8.402 | 4.487 | 1.873x |
| direct | images | 512 | 5 | float32 | 20.966 | 16.886 | 1.242x |
| direct | images | 512 | 5 | uint8 | 19.947 | 15.853 | 1.258x |
| direct | images | 1024 | 1 | float32 | 34.992 | 16.180 | 2.163x |
| direct | images | 1024 | 1 | uint8 | 34.776 | 18.976 | 1.833x |
| direct | images | 1024 | 3 | float32 | 58.221 | 41.078 | 1.417x |
| direct | images | 1024 | 3 | uint8 | 40.933 | 21.690 | 1.887x |
| direct | images | 1024 | 5 | float32 | 84.910 | 61.644 | 1.377x |
| direct | images | 1024 | 5 | uint8 | 85.132 | 67.559 | 1.260x |
| direct | volume | 256 | 1 | float32 | 2.583 | 1.687 | 1.531x |
| direct | volume | 256 | 1 | uint8 | 3.264 | 2.488 | 1.312x |
| direct | volume | 256 | 3 | float32 | 5.752 | 4.424 | 1.300x |
| direct | volume | 256 | 3 | uint8 | 3.336 | 2.469 | 1.351x |
| direct | volume | 256 | 5 | float32 | 7.989 | 6.986 | 1.143x |
| direct | volume | 256 | 5 | uint8 | 9.660 | 8.041 | 1.201x |
| direct | volume | 512 | 1 | float32 | 11.566 | 5.877 | 1.968x |
| direct | volume | 512 | 1 | uint8 | 15.611 | 9.075 | 1.720x |
| direct | volume | 512 | 3 | float32 | 29.129 | 24.429 | 1.192x |
| direct | volume | 512 | 3 | uint8 | 14.591 | 8.116 | 1.798x |
| direct | volume | 512 | 5 | float32 | 38.354 | 32.866 | 1.167x |
| direct | volume | 512 | 5 | uint8 | 36.365 | 31.171 | 1.167x |
| direct | volume | 1024 | 1 | float32 | 59.490 | 29.149 | 2.041x |
| direct | volume | 1024 | 1 | uint8 | 72.858 | 34.003 | 2.143x |
| direct | volume | 1024 | 3 | float32 | 102.932 | 84.712 | 1.215x |
| direct | volume | 1024 | 3 | uint8 | 63.920 | 41.879 | 1.526x |
| direct | volume | 1024 | 5 | float32 | 145.964 | 118.726 | 1.229x |
| direct | volume | 1024 | 5 | uint8 | 160.523 | 132.524 | 1.211x |

</details>

## Related work

- [Issue #453](https://github.com/albumentations-team/AlbumentationsX/issues/453)
- [Issue #327: 3D ElasticTransform](https://github.com/albumentations-team/AlbumentationsX/issues/327)
- [Applied configuration and replay contracts](applied-config-replay-contracts.md)
- [Bounding box processing](bounding_boxes.md)
