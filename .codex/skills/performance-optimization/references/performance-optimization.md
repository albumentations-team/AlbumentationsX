# Performance Optimization Workflow

Use this workflow whenever implementing, reviewing, or profiling runtime code in Albucore or AlbumentationsX. Its
order matters: remove work before making the remaining work faster. Treat every proposed optimization as a hypothesis
until correctness tests and representative benchmarks support it.

This file is canonical at `albucore/docs/performance-optimization.md`. AlbumentationsX keeps a synchronized fallback
copy under `.codex/skills/performance-optimization/references/` so its performance skill can load the guide without a
sibling Albucore checkout.

## Run the optimization pass in this order

### 1. Delete work first

Look for work that does not need to happen:

- redundant validation, conversions, clipping, copies, reshapes, and contiguity fixes;
- repeated calculations whose inputs do not change;
- identity operations and empty inputs that can return early;
- temporary arrays whose result is immediately overwritten;
- setup repeated per image that can run once per batch;
- duplicate kernels already available in Albucore.

Use `np.empty` instead of zero-initialized storage when every element will be overwritten. Hoist invariant setup out
of loops. Deleting a full-array pass usually beats making that pass faster.

### 2. Reduce memory traffic and allocations

Count full-array reads and writes, not only Python statements. Image kernels are often limited by memory bandwidth.

- Fuse compatible arithmetic and reductions.
- Use one-dimensional coordinate arrays, broadcasting, or `np.ogrid` instead of materializing full `meshgrid`
  coordinates.
- Reuse output buffers when ownership is clear.
- Avoid full-array `astype`, `copy`, and `ascontiguousarray` calls unless the selected backend earns their cost.
- Keep working arrays `float32` unless correctness requires wider precision.
- Preallocate outside repeated loops and reset only the data that must change.
- Prefer a view when downstream code accepts its strides and ownership.
- Choose sparse coordinates, rectangles, or indices instead of a dense image-sized mask when the affected set is
  small; use dense array operations when the set is large.

Measure peak memory or allocation counts when the change targets allocations. A lower wall time with much higher peak
memory is not an unconditional win.

### 3. Vectorize the right dimension

Inspect loops over pixels, labels, channels, images, frames, depth slices, boxes, and keypoints.

- Replace pixel loops with array operations.
- Batch random draws and repeated setup.
- Vectorize across independent annotations when the temporary arrays remain bounded.
- Keep a preallocated loop when broadcasting creates expensive strided access or a much larger temporary.
- Specialize common channel layouts only when the benchmark supports the branch.

`np.vectorize` and `np.frompyfunc` still execute Python callables element by element. They do not satisfy this
vectorization pass.

Vectorization can lose on tiny arrays, high-channel strided views, or operations with large broadcasted temporaries.
Benchmark the original loop as a real candidate.

### 4. Use `bincount` for dense integer-label reductions

`np.bincount` is a segmented reduction. For non-negative integer label `labels[i]`, it can compute per-label counts
and weighted sums in one compiled pass:

```python
counts = np.bincount(labels, minlength=num_labels)
sum_x = np.bincount(labels, weights=x_coordinates, minlength=num_labels)
sum_y = np.bincount(labels, weights=y_coordinates, minlength=num_labels)

nonempty = counts > 0
center_x = sum_x[nonempty] / counts[nonempty]
center_y = sum_y[nonempty] / counts[nonempty]
```

This replaces code that constructs `labels == label` for every label. The old pattern scans `N` elements `K` times
and allocates `K` boolean masks. The grouped form performs a small fixed number of `O(N + K)` passes.

Consider `bincount` for:

- component sizes and per-component sums or means;
- superpixel statistics and cluster-center updates;
- class or label histograms;
- confusion matrices after mapping a label pair to one integer bin;
- repeated reductions over dense region IDs.

Check these constraints before adopting it:

- labels must be non-negative integers;
- output size follows the largest label, so sparse large IDs need remapping;
- `weights=` returns floating-point sums and may change accumulation precision;
- negative or ignored labels need filtering or an explicit offset;
- `np.unique(..., return_inverse=True)` can compress sparse IDs, but sorting and remapping may erase the gain;
- a few labels, a small input, or a one-time scan may favor the existing implementation.

For reductions other than counts and sums, also benchmark `np.minimum.at`, `np.maximum.at`, `np.logical_or.at`, or
sorting followed by `ufunc.reduceat`. Repeated-index `ufunc.at` operations preserve scatter semantics but can be slower
than `bincount`; they are candidates, not automatic replacements.

Benchmark label cardinality and density, including the common small-label case. In AlbumentationsX audits,
`bincount` greatly accelerated SLIC and superpixel means, did not improve component point sampling, and slowed the
common 4-by-4 bbox-grid case. Use it as a candidate, never as a blanket rewrite.

### 5. Check whether the operation is a LUT

For a true pointwise mapping from uint8 input values to output values, compare:

- Albucore's LUT router;
- OpenCV LUT;
- StringZilla translation;
- NumPy indexing or direct arithmetic.

Include shared and per-channel tables, contiguous and non-contiguous input, and channels above OpenCV's common
four-channel limit. A direct bitwise operation can beat every LUT for a bit mask.

Keep floating-point LUT tables `float32`. A float64 table can force a full float64 output and an expensive conversion
back to float32.

### 6. Compare complete backend implementations

For each atomic operation, benchmark viable implementations end to end:

- NumPy;
- OpenCV;
- NumKong;
- StringZilla;
- a LUT route;
- a small Python or `math` implementation when the data is scalar or tiny.

Include dispatch, dtype conversion, contiguity, allocation, clipping, and output-shape repair in the timing. A fast
kernel that requires a full-size copy can lose at the public API.

Route only where evidence wins. Thresholds may depend on dtype, rank, size, channel count, contiguity, scalar versus
per-channel parameters, and in-place versus allocating mode. Existing Albucore routes demonstrate that no backend is
globally fastest:

- NumKong wins selected reductions, moments, blends, scales, and small distance matrices;
- NumPy wins other float32 and high-channel paths;
- OpenCV wins many dense image operations but has channel and layout constraints;
- StringZilla is competitive for selected uint8 translation paths.

### 7. Compare random-generation paths

Choose random generation by workload and contract:

- Python `random.Random` is a strong candidate for a few scalar parameters or choices.
- NumPy `Generator` is the primary candidate for vectors, masks, noise fields, shuffles, and distributions.
- OpenCV random fill can be benchmarked when it writes directly into an existing dense buffer.

Measure generation and required post-processing together. Compare direct dtype generation, vectorized draws, bounds,
temporary casts, and in-place scaling.

For sparse dropout or impulse noise, compare generating a dense random mask with sampling sparse flat indices and
scattering values. For channel-shared noise, do not generate independent channel data that will be discarded.

Preserve seeded behavior, per-transform generator isolation, thread safety, and replay semantics. OpenCV's global RNG
or a different NumPy dtype path may be faster while changing those contracts. Treat a changed random sequence as an
explicit compatibility decision, not an invisible optimization.

### 8. Move reusable atomic operations into Albucore

AlbumentationsX owns transform policy, parameter sampling, target dispatch, and annotation semantics. Albucore owns
reusable array kernels and benchmark-driven backend routing.

Move an operation to Albucore when it:

- appears in multiple transforms or has clear reuse beyond one transform;
- has stable array semantics independent of augmentation policy;
- benefits from dtype, layout, channel, size, or backend routing;
- can expose a small typed API with direct correctness tests and microbenchmarks.

Do not move transform-specific sampling or target semantics. Avoid parallel local helpers that implement the same
atomic operation differently.

### 9. Make safe operations in-place

Prefer `out=`, OpenCV `dst=`, and Albucore `inplace=True` when all aliasing conditions hold:

- the input may be modified under the public contract;
- no later calculation needs the original values;
- source and destination overlap is supported by the backend;
- shared arrays or views do not make the mutation observable elsewhere;
- dtype and shape already match the destination.

In-place is not automatically faster. It can trigger copies for non-contiguous views, block fusion, or corrupt
read-after-write calculations. Benchmark both modes and test aliasing explicitly.

### 10. Specialize and route only after the simpler pass

After removing work and comparing straight implementations, consider routing by:

- uint8 versus float32;
- scalar versus per-channel parameters;
- 1, 3, 4, and high-channel inputs;
- image, batch, volume, and batch-of-volumes rank;
- contiguous versus strided input;
- small versus large arrays;
- dense versus sparse labels;
- allocating versus in-place output.

Keep the number of branches justified by durable benchmark regions. Do not add a threshold for noisy single-machine
measurements.

### 11. Measure dispatch and pipeline overhead

An isolated kernel speedup may disappear behind transform dispatch, conversion decorators, target routing, or
`Compose`. Benchmark the layer users execute.

For AlbumentationsX, measure both the direct functional call and the transform through `Compose`. For batch-capable
transforms, compare per-image dispatch with `apply_to_images` and include setup reuse.

For Albucore, benchmark the public router, not only its private backend. Include non-square inputs, supported dtypes,
1/3/high-channel cases, and relevant batch or volume ranks.

## Reuse existing Albucore routes

Check these shared entry points before writing a local kernel:

- `albucore.apply_uint8_lut` and `albucore.sz_lut` for benchmark-routed uint8 tables;
- `albucore.stats` for `sum`, `mean`, `std`, and `mean_std` across images, batches, and volumes;
- Albucore arithmetic and weighted routers for add, multiply, normalize, power, blend, and fused multiply-add;
- Albucore geometric and resize functions for channel-safe OpenCV routing;
- `pairwise_distances_squared` for the routed small-set NumKong and larger NumPy paths.

The stats API already routes selected uint8 and low-channel reductions through NumKong while retaining NumPy or
OpenCV where they win. Reimplementing these reductions inside AlbumentationsX loses both the routing and its benchmark
coverage.

Use `docs/numkong-performance.md` for current NumKong route evidence and
`docs/performance-regressions-plan.md` for known router regressions. Regenerate or extend the relevant benchmark when a
route changes; do not update a threshold from an isolated microbenchmark alone.

## Preserve behavior before measuring speed

Create a correctness baseline before editing:

- exact output or a justified numerical tolerance;
- shape, dtype, range, and channel preservation;
- empty and degenerate inputs;
- annotations and label semantics;
- mutation and aliasing behavior;
- deterministic and replay behavior for seeded transforms.

Use a copied reference implementation or regression vectors when replacing an algorithmic inner loop. Performance
does not justify an unreviewed output change.

## Benchmark every candidate, including rejected ones

Use the repository benchmark skill and policy. At minimum:

- run old and new code on the same machine and environment;
- control OpenCV and BLAS threads;
- include warmup and enough repetitions for stable timing;
- report versions, dtype, shape, channels, parameters, and allocation mode;
- cover the full required size and channel matrix;
- compare direct and routed/public execution;
- investigate any cell that regresses by more than the project threshold;
- retain a short record of rejected candidates and the cases where they lost.

Vary the dimension that controls the proposed optimization. For `bincount`, sweep label count and density. For LUTs,
sweep table layout and channels. For random generation, sweep output size and dtype. For backend routing, sweep both
sides of every proposed threshold.

## Required review record

Every performance task should report:

- work deleted or avoided;
- full-array passes, copies, conversions, and allocations removed;
- loops reviewed and vectorization decisions;
- grouped integer-label reductions reviewed, including `bincount`;
- LUT applicability and candidates;
- random-generation candidates;
- NumPy, OpenCV, NumKong, StringZilla, and Python candidates that apply;
- reusable atoms moved or proposed for Albucore;
- safe in-place opportunities;
- correctness evidence;
- benchmark matrix, wins, regressions, and rejected candidates.
