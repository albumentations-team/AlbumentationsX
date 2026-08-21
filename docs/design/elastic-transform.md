# Bounded 2D ElasticTransform

**Status:** Implemented
**Scope:** 2D `ElasticTransform`, including its XY application to volumes
**Related issues:** [#453](https://github.com/albumentations-team/AlbumentationsX/issues/453), [#327](https://github.com/albumentations-team/AlbumentationsX/issues/327)

## What users get

`ElasticTransform` applies a smooth, bounded pull deformation to an image and
its synchronized targets. The deformation is sampled from a small control grid,
so sampling cost does not grow with image area. The same XY map is used for every
image in a batch and every depth slice in a volume.

The public constructor is:

```python
A.ElasticTransform(
    displacement_range=(0.02, 0.05),
    control_grid_shape=(5, 5),
    interpolation=cv2.INTER_LINEAR,
    mask_interpolation=cv2.INTER_NEAREST,
    border_mode=cv2.BORDER_CONSTANT,
    fill=0,
    fill_mask=0,
    p=0.5,
)
```

The displacement range is relative to the shorter span between the first and
last pixel centers. For an image with shape `(H, W)`, a sampled magnitude `m`
allows control-vector radius `m * min(H - 1, W - 1)` pixels. The constructor
rejects a range/grid pair that violates the strict topology bound:

```text
2 * displacement_range[1]
  * sqrt((rows - 1)^2 + (columns - 1)^2) < 1
```

This is a deliberate breaking contract. The former Gaussian-field parameters
(`alpha`, `sigma`, `approximate`, `same_dxdy`, `noise_distribution`), Elastic's
map-resolution controls, and the raster keypoint policy are not accepted or
aliased.

## Geometry

The transform builds a target-to-source pull map:

```text
S(q) = q + d(q)
```

`q = (x, y)` is an output coordinate and `S(q)` is the source coordinate passed
to the raster remapper. Control vectors are stored in `(row, column, (dx, dy))`
order. Their anchors include both image edges:

```text
x_j = j * (W - 1) / (columns - 1)
y_i = i * (H - 1) / (rows - 1)
```

The dense field is endpoint-aligned bilinear interpolation of the four enclosing
vectors. This is not `cv2.resize`'s half-pixel convention. The expansion is
independent of image and mask interpolation; those settings apply only when
sampling raster targets from `S`.

Each control vector is uniform by area in a disk. If `U` and `V` are independent
uniform samples, the vector is:

```text
radius = R * sqrt(U)
angle = 2 * pi * V
vector = radius * (cos(angle), sin(angle))
```

The transform samples the scalar magnitude with the invocation's Python random
stream and the control grid with its NumPy generator. A zero range takes an
exact identity path: no control grid, dense maps, or remap call are allocated.

## Targets and annotations

- `image` uses `interpolation`, `border_mode`, and `fill`.
- `mask` and `mask3d` use `mask_interpolation`, `border_mode`, and `fill_mask`.
- `images` and `volume` reuse one XY map for all members or depth slices.
- HBB and OBB use the existing distortion bbox route: rasterize support,
  nearest-neighbor remap with constant-zero support, then recover the enclosing
  box. Raster border modes do not create annotation support outside the source.
- Keypoints are inverted analytically per control cell. The solver evaluates the
  bilinear map, solves its quadratic/linear degeneracies, and accepts only a
  candidate whose forward residual is at most `1e-3` pixel. It preserves Z and
  all extra columns; points without a valid inverse receive the existing
  out-of-domain sentinel.

The strict topology bound is a conservative sufficient condition for an
orientation-preserving, injective pull map. Runtime code does not run a per-call
fold detector; constructor validation establishes the supported region.

## Persistence contracts

The three persistence routes intentionally preserve different state:

| Route | Stored state | Result |
| --- | --- | --- |
| Constructor serialization | Original `displacement_range` and all public fields | A reconstructed transform keeps its stochastic policy. |
| `applied_config` | Realized scalar as `(m, m)` plus public fields | `Compose.from_applied_transforms()` is runnable, but samples a new control grid. |
| `ReplayCompose` | Compact control grid, realized magnitude, and original spatial shape | The same sampled geometry replays exactly for the same `(H, W)`. |

Applied records contain only strict-JSON-safe values. They do not contain dense
`map_x`/`map_y` arrays. Replay on a different spatial shape raises `ValueError`
instead of silently rescaling the recorded field.

## Implementation boundary

`BaseRemapTransform` owns common raster, batch, volume, mask3d, and bbox dispatch.
The public `BaseDistortion` subclass retains map-resolution sampling and the
direct/mask keypoint policy needed by the other distortion families. Elastic has
its own constructor and continuous keypoint inverse and does not inherit those
fields.

The dense control-grid expansion and absolute-map construction live in the
geometric functional layer. Keeping map construction separate from dispatch
ensures one map per invocation and keeps concrete `apply*` methods focused on
policy and routing.

3D elastic deformation remains a separate design. Issue #327 may reuse the
relative magnitude and replay vocabulary, but it needs a generic dense 3D remap
primitive and its own performance and annotation contract.

## Verification

The focused suite checks constructor validation, endpoint anchors, vector bounds,
strict JSON replay, applied-config reconstruction, analytic inverse accuracy,
volume slice consistency, and exact identity. The shared distortion, OBB,
serialization, and target-contract suites cover the non-Elastic base-class split
and public transform integration.

Performance comparisons are reported as a breaking default-to-default cutover.
The representative matrix includes direct and `Compose` routes for 256, 512,
and 1024-pixel inputs, 1/3/5 channels, `uint8`/`float32`, image batches, and
volumes. OpenCV/BLAS thread counts are pinned, and raw cells are retained with
the benchmark artifact.

## References

- [Issue #453](https://github.com/albumentations-team/AlbumentationsX/issues/453)
- [Applied configuration and replay contracts](applied-config-replay-contracts.md)
- [Bounding box processing](bounding_boxes.md)
- [Issue #327: 3D ElasticTransform](https://github.com/albumentations-team/AlbumentationsX/issues/327)
