# Float16 support for transforms and Compose

**Status:** current-source full suites pass on macOS ARM64. Linux full suites pass only for the October 30 source snapshot; current-source Linux correctness remains open. Performance and working-memory acceptance remain open on both platforms.
**Decisions recorded:** 2026-10-12 through 2026-10-17.
**Initial baseline:** AlbumentationsX `1ab3d3d944a2ddad361a13bd34c447824e78f221`, albucore 0.3.0.
**Published dependency:** [Albucore 0.3.2](https://pypi.org/project/albucore/0.3.2/) provides the explicit output API used by this branch. AX pins this release in Python and Conda metadata; CI installs the published wheel.

Support NumPy `float16` and eager CPU Torch `float16` throughout public transforms and composition. Cover 2D images,
3D volumes, collections, masks, additional targets, and reference/donor data. Preserve float16 outputs, including
`Normalize` and `ToFloat`. Compare old and new implementations at the same dtype through the full public `Compose`
call. Same-dtype results determine acceptance; float16 versus float32 is mandatory diagnostic evidence.

Preserve frozen baseline snapshots so a failure can still be attributed to its source and dependency change.
The coordinated Core/AX pair now uses the published Albucore 0.3.2 wheel.

## Acceptance environment

Run final correctness, performance, and memory acceptance on Python 3.11, the lowest supported Python minor shared
with Albucore. Resolve the newest stable third-party dependencies that support Python 3.11 and each target platform.
The profile recorded on 2026-10-31 used NumPy 2.4.6, OpenCV 5.0.0.93, PyTorch 2.15.0 on macOS ARM64 and
`2.15.0+cpu` on Linux x86-64, NumKong 7.8.6, and StringZilla 5.3.2. Re-resolve before future acceptance runs and
record the complete package inventory and imported Core/AX source paths. A newer Python or stale project lock does not
replace this profile. The full compatibility matrix on Python 3.12–3.14 remains additional evidence.

Preserve the relationship between input and output types:

- Use `overload` at public entry points with distinct NumPy/Tensor or dtype-dependent return contracts. A NumPy
  argument must infer a NumPy result, and a Tensor argument must infer a Tensor result.
- Use a `TypeVar` when one generic signature accurately expresses the preserved type. Do not promise preservation
  of arbitrary NumPy subclasses when an operation only preserves dtype and container.
- Keep an honest input union in a shared NumPy/Tensor implementation where AX reads that annotation for native
  Tensor dispatch. Overloads describe the public call contracts; they do not replace the implementation annotation.
- NumPy-only helpers retain NumPy types. Do not broaden every helper to accept Tensor or cast a Tensor result to a
  NumPy image type. Declaring Tensor only in `Annotated` metadata does not provide a truthful static type.

Verify inferred return types for concrete NumPy and Tensor inputs and for union-typed inputs with mypy and Pyrefly.

## Public dtype and target contract

| Operation or target | Float16 behavior |
| --- | --- |
| `image`, `images`, `volume`, `volumes` | Accept float16 and preserve it through dtype-preserving transforms. |
| `mask`, `masks`, `mask3d`, `masks3d` | Accept float16 alongside existing mask dtypes; preserve dtype and the configured mask semantics. |
| `Normalize`, all normalization modes | Float16 input returns float16; uint8 and float32 retain their existing float32 output. Honor `clip_range`. |
| `ToFloat` | Float16 input remains float16; existing uint8 and float32 results remain float32. |
| `FromFloat` | Accept float16 input and retain the existing explicit uint8 output contract. |
| `ToTensorV2`, `ToTensor3D` | NumPy float16 becomes Torch float16 with the existing target layout. |
| Bboxes, keypoints, transform matrices, remap coordinates | Retain their existing computation precision; Tensor bbox/keypoint targets remain float32. |
| Reference images and donor records | Support float16 data under the existing `targets_as_params` and instance/frame binding contracts. Output follows the primary target dtype. |
| `Lambda` and user callbacks | Forward float16 data through the existing callback API; caller-defined output semantics remain the callback's responsibility. |

Normalized floating images use the existing `[0, 1]` convention. `Normalize` may produce signed or out-of-range values
under its own contract. Retain float32's existing special-value policy; adding float16 must not introduce a new
full-array finiteness scan at the Compose boundary.

For standard `Normalize`, artificially tiny caller-supplied `std` values are outside the representative workload.
Do not add a float16-specific exception, automatic widening, or saturation policy for that case. Retain ordinary
NumPy/Torch arithmetic behavior and the explicit `clip_range` contract. Per-image normalization must still handle
constant images and near-zero measured variance through its existing numerical safeguards.

CPU Tensor rules remain plain, strided tensors with `requires_grad=False`. Preserve each target's container and its
existing layout. NumPy uses HWC/NHWC/DHWC/NDHWC; Torch uses CHW/NCHW/CDHW/NCDHW. Optional grayscale channel axes,
mixed NumPy/Tensor calls, read-only NumPy inputs, positive-stride Tensor views, and additional-target aliases retain
their current behavior. CUDA, MPS, bfloat16, autograd, and `torch.compile` are outside this plan.

For masks with categorical labels, nearest-neighbor resampling and copying must preserve represented labels exactly,
except for explicitly configured fill or dropout values. Existing integer masks must never be converted to float16:
half precision cannot represent every integer above 2048. Continuous masks retain their configured interpolation.

### Use the same channel rule for images and masks

Normalize optional channel axes by target layout, using the same rule for images and masks:

| Public input | Internal representation | Output when the channel count remains one |
| --- | --- | --- |
| NumPy image or mask `(H, W)` | `(H, W, 1)` | `(H, W)` |
| Torch image or mask `(H, W)` through Compose | `(1, H, W)` | `(H, W)` |
| NumPy image or mask `(H, W, 1)` | `(H, W, 1)` | `(H, W, 1)` |
| Torch image or mask `(1, H, W)` | `(1, H, W)` | `(1, H, W)` |

Apply the corresponding rule to collections and volumes using their existing target layouts. Adding or removing
the singleton channel axis is a view operation; it must not copy pixel data or change dtype. Record whether this
invocation added the axis, and remove only that axis when it still has size one. Avoid unrestricted `squeeze`, which
would also remove spatial, depth, or collection dimensions of size one. If a transform creates multiple channels,
retain the resulting channel axis.

Root Compose already owns this normalization and restoration for images and masks. Keep that single boundary;
nested nodes and leaves must not repeat preparation of canonical inputs. For direct NumPy transform calls,
standalone composition nodes, and geometric helpers that accept `(H, W)` outside Compose, adapt the input in the
owning functional layer before calling albucore: pass an `(H, W, 1)` view and restore `(H, W)` afterward. Reuse the
shape adaptation where it is shared; do not add a second graph traversal or teach albucore to infer omitted channels.
Direct Tensor leaves retain their existing canonical-layout requirement; Compose handles their optional axes.

Channel normalization changes representation only. Masks continue through mask handlers, preserve their dtype and
represented labels, and use the configured mask interpolation. A mask's channels do not make it an RGB image.

## What the dependency provides and what AX still owns

Each AX functional owner keeps transform-specific conversion, rounding, clipping, and dtype restoration explicit.
Core primitives alone do not prove per-transform Compose correctness or performance. Current local test runs and CI
use the published Albucore wheel.

The development branch records Albucore 0.3.2 in Python and Conda metadata. Its lock selects NumKong 7.8.7 and
StringZilla 5.3.2. Changed distributions are reviewed through `tools/update_dependency_licenses.py` and the existing
license registry. Torch remains an externally selected runtime; the `ci-torch-cpu` validation profile selects 2.15.0.

The source currently has these integration boundaries:

| Boundary | Current implementation | Remaining acceptance |
| --- | --- | --- |
| NumPy and CPU Tensor image types | Shared image/volume aliases and Tensor dtype acceptance include float16; annotations retain their existing precision. | Keep the full supported target/dtype matrix green on both platforms. |
| Compose preparation and restoration | Root Compose owns optional-channel adaptation, container preservation and output restoration. | Complete old/new timing and strict memory checks through full Compose. |
| Pixel, medical, conversion and terminal operations | Current functional owners preserve or explicitly restore the declared output dtype. | Finish same-dtype speed and memory matrices for the affected families. |
| Geometry, crops and volume routes | Current source retains explicit ranks/layouts, coordinate precision and route-specific kernels. | Complete coordinate, border, batch and volume timing/memory acceptance on both platforms. |
| Mixing, references and annotations | AX keeps sampling, donor/reference policy and annotation precision. | Finish the affected target, auxiliary-input and full-path acceptance cases. |
| Shared test inventory | The existing transform registry and target profiles cover the public transform set. | Keep its dtype-sensitive constructor modes and Compose routes current. |

Keep augmentation policy, sampling, target dispatch, and annotation processing in AX. Keep reusable array kernels and
backend selection in albucore. Read [NumPy/Tensor routing](numpy-tensor-routing.md) and
[generated target contracts](transform-target-contracts.md) before changing these boundaries.

### Bound dependency work and keep blocked routes visible

Small, focused albucore corrections needed for compatibility, correctness, or the performance gate belong to this
work. Verify them in albucore and consume the fix through a published release, updating AX's exact dependency pin,
lockfile, reviewed license versions, and affected checks. Validate the coordinated Core/AX pair and retain source
provenance for dependency and behavior changes.

New kernels in NumKong, OpenCV, or PyTorch are separate work items. Record the affected public route, reproduction or
paired timing evidence, and the upstream issue/work links. A portable route using an existing dependency kernel is
acceptable when the full AX route passes correctness, same-dtype performance and working-memory validation. Claim a
native accelerated route only after the required dependency release exists. Continue independent operation families;
do not disable required dtype support or claim completion while required routes remain blocked.

### Initial package findings

The first Albucore 0.3.0 integration exposed grayscale-channel and type-consumer issues. The current local Core/AX
source pair now passes the full ARM64 and Linux suites plus actual-stub mypy and Pyrefly checks. These results apply to
the local source pair; they do not establish compatibility with the published 0.3.1 wheel.

## Estimate memory traffic before choosing a route

For `N` image elements, float16 storage uses `2N` bytes and float32 uses `4N`. This halves input storage, but does not
by itself halve execution time. One float16-to-float32 conversion and one output conversion back to float16 move at
least `12N` bytes in total, before the kernel reads or writes anything. A separate float32 input, float32 result, and
float16 result can coexist with the original half input and occupy `12N` bytes before backend scratch allocations.

| RGB image | Float16 array | Float32 array | Traffic for widening and restoration alone |
| --- | ---: | ---: | ---: |
| 256 x 256 x 3 | 0.375 MiB | 0.75 MiB | 2.25 MiB |
| 512 x 512 x 3 | 1.5 MiB | 3 MiB | 9 MiB |
| 1024 x 1024 x 3 | 6 MiB | 12 MiB | 36 MiB |

These are byte-count estimates, assuming materialized casts and excluding caches, reuse, and kernel scratch space.
For crops, flips, and other cheap routes, begin with dtype-independent views or native operations. For expensive
geometry, color conversion, filtering, and reductions, compare native half with float32 working buffers. Include
both casts and all allocations in timing. A universal float16-to-float32 wrapper cannot satisfy the speed contract
without route-specific evidence.

When an existing uint8 route already uses `uint8 -> float32 -> uint8`, compare it with an operation-specific
`uint8 -> float16 -> uint8` implementation. This optimization is in scope. Select the half working route when the
complete public call, including both conversions, is faster and the agreed correctness checks pass. Preserve uint8
input/output, range, target semantics, and input ownership. Benchmark against the existing uint8 route on the same
logical input and parameters, independently of the float16-input versus float32-input comparison. Uint8 image values
may differ from the previous float32-working result by at most one intensity level per element, only with confirmed
full-route acceleration. Categorical masks, reflections, and copying remain exact. Native uint8 routes that already
avoid a floating conversion are not candidates solely because float16 support is being added.

## Implement one family at a time

1. **Verify the dependency upgrade on existing dtypes.** Run the existing uint8/float32 integration and regression
   suites against the installed PyPI package. Verify lock consistency, Conda parity, license evidence, CI profiles,
   and built artifact metadata. Fix the two-dimensional channel handling and type incompatibilities without weakening
   checks or changing the supported public dtype contract. Finish this stage with passing existing tests, mypy,
   Pyrefly, and applicable quality checks. Record its source revision and dependency versions as the baseline while
   implementing the coordinated API and float16 changes.
2. **Establish types and boundary coverage.** Extend shared NumPy aliases and Tensor dtype sets. Add float16 fixtures
   for every spatial target family and metadata source. Validate acceptance before RNG consumption and preserve
   rejection of unsupported device, layout, grad, and annotation inputs. No-op, root skip, and skipped leaves must
   preserve object identity where the current contract promises it.
3. **Close cheap and conversion routes.** Cover `NoOp`, flips, transpose, rotations by right angles, slicing/crops,
   temporal selection, `ToFloat`, `FromFloat`, `Normalize`, and NumPy-to-Tensor terminals. These establish output dtype,
   layout, exact-copy semantics, and input ownership before adding numerical kernels.
4. **Close pixel families.** Cover arithmetic/contrast, color-space and channel operations, blur/sharpening, noise,
   weather, compression, dropout, medical color/stain operations, and spectrogram masking. Audit explicit
   `dtype == np.float32` branches, hardcoded float32 result arrays, NumPy promotion, and direct backend calls.
   Generate noise and statistics at the precision their algorithms need;
   restore output dtype in the owning operation. Existing uint8-only algorithms may quantize normalized float inputs
   according to their existing semantics, with explicit precision and timing evidence.
5. **Close spatial and volume families.** Cover resize, affine/perspective, distortions/remap, padding, 3D resampling,
   and volumetric artifacts. Retain float32 coordinate grids and transformation matrices. Verify all interpolation,
   border, channel, batch, and mask modes. Reuse albucore's spatial routes where their contract matches.
6. **Close mixing and all composition nodes.** Exercise reference matching, overlays, Mosaic, CopyAndPaste, selective
   channels, and instance/frame targets. Cover `Compose`, nested `Compose`, `Sequential`, `OneOf`, `SomeOf`,
   `RandomOrder`, `OneOrOther`, `SelectiveChannelTransform`, and `ReplayCompose`. Include mixed target containers and
   repeated native/fallback leaves. Use the same configured-graph dispatcher and invocation context.
7. **Close documentation and release gates.** Update public image-type sections, normalization/conversion examples,
   routing documentation, benchmark workloads, and any deterministic dtype checks that own the changed contract.
   Delete superseded special cases after proving their replacement. Re-run the affected checks and inspect the final
   diff for redundant wrappers, duplicate kernels, narrated comments, and weak tests.

Each family closes only when its correctness and performance cells pass. Track a failure under its concrete public
class, constructor mode, target, container, dtype, shape, and backend. Keep upstream capability requests linked to the
affected route. Do not hide unfinished routes behind blanket skips or declare complete support from a passing import.

## Preserve Compose's execution model

Root Compose validates and repairs representation once. Nested nodes do not establish another dtype or Tensor
boundary. Leaves select an existing native handler or the existing NumPy bridge; skipped leaves perform no conversion.
Do not widen an entire pipeline at the root: every dtype-preserving leaf must satisfy the float16 contract when
called directly as well as through Compose.

For CPU Torch inputs, the existing `Tensor -> NumPy -> Tensor` leaf route is accepted. Bridge views must retain
float16, and a dtype-preserving operation must return a float16 Tensor in the original target layout. The complete
Compose call includes layout changes, required copies, kernel working buffers and Tensor restoration. Compare old and
new calls at the same dtype, graph and equivalent transform controls. Compare half against float32 separately as a
diagnostic. Use identical recorded parameters for numerical checks, and include actual sampling
in full-call timings. A native Torch implementation is required only when correctness or measured performance needs
it; do not implement every operation twice solely to avoid the existing bridge.

Keep sampled parameters, bridges, restoration metadata, replay records, and processor sessions in `InvocationContext`.
Do not cache an input dtype or working buffer on a configured transform. Preserve seeded sampling, sample-keyed
`invocation_seed`, applied-configuration capture, tracing, reentrancy, and worker-seed behavior.

Require reproducible sampling and exact parameter replay within each dtype at fixed source/dependency revisions and
call configuration. The same seed does not have to produce identical parameters across float16 and float32.
Float16 may use a faster random-generation route when its distribution and invocation-local RNG contracts pass
validation. Existing uint8 and float32 random sequences must remain unchanged, including accelerated uint8 routes
that use half working buffers. Production replay keeps its existing materialization checks; this work adds no
guarantee that a record captured in one dtype can be replayed in another.

When a dtype-preserving operation on float16 input needs float32 computation, restore float16 before returning.
Reuse a concrete Albucore operation when it owns that conversion. Add a helper only when it owns a real shared
operation. No dtype flag, second traversal, or parallel backend registry is needed for this contract.

## Prove correctness through the shared test inventory

Extend the existing case registry and target profiles. Cover every built-in public transform and its dtype-sensitive
constructor modes, including conversion transforms, terminals, and 3D-only transforms. Make a missing registered
class or applicable float16 target pair fail coverage. Run core target modes broadly; reserve expensive representation
profiles for primary cases as the current harness does. Do not multiply every existing axis indiscriminately.

Required evidence:

- Assert output dtype, container, shape, range where promised, finite values for finite in-domain workloads, and input
  non-mutation. Exercise one and five channels, non-square inputs, batches, volumes, read-only and strided arrays.
- Build the numerical reference by rounding input values to float16, widening those represented values to float32,
  executing the float32 operation with the same recorded parameters, and rounding its output to float16. This keeps
  input quantization, independent random draws, and final output quantization out of the discrepancy being measured.
- Require exact results for indexing/copying, categorical masks, skipped paths, and replay within one dtype.
- Check repeated seeded execution and captured-parameter replay independently within float16. Preserve existing
  uint8/float32 seeded regression expectations. Cross-dtype numerical comparisons use identical recorded parameters
  rather than assuming that identical seeds imply identical draws; retain production replay requirement checks.
- Cover omitted and explicit singleton channels separately through Compose and supported direct NumPy calls,
  including D4 and standalone composition nodes. Assert that adaptation shares storage, preserves dtype, restores
  only the added axis, and retains explicit channels. Include spatial/depth/collection dimensions of size one and
  a channel-changing image transform to detect unrestricted squeezing or stale restoration state.
- Arithmetic, filtering, normalization, and interpolation may differ slightly from that rounded reference. Define
  and document bounds for each operation family from quantization and numerical conditioning before accepting a
  route; derive them independently of the candidate's observed errors. `rtol=2e-3`, `atol=1e-3` on unit-scale values
  are initial probe bounds, not a blanket acceptance rule. Use absolute error near zero and scaled bounds for
  normalized signed outputs. Keep indexing, reflections, copying, and categorical mask results exact. Test constant
  inputs, near-zero variance, reductions large enough to overflow half accumulation, and values near clipping or
  rounding boundaries.
- Float32 `Sharpen(method="kernel")` uses `rtol=0`, `atol=1e-7` for the singleton-channel constant-image reference;
  Linux OpenCV differed from direct scalar multiplication by `5.96e-8`. Uint8 and float16 cases remain exact.
- For an existing uint8-to-float32-to-uint8 route, require a confirmed full-route speedup and a maximum absolute
  uint8 image error of one intensity level per element against the previous implementation. Compute differences in
  a signed or wider dtype so uint8 subtraction cannot wrap. The allowance never applies to categorical masks,
  copying, or reflections. Preserve existing float32 regression expectations and keep uint8 coverage strict under
  this bounded-error contract; do not relax unrelated comparisons or use average error to hide larger pixel errors.
- Check half-to-uint8 output against float32 conversion of the same represented half input; retain the existing
  rounding and clipping contract. Do not introduce an approximate path silently.
- Exercise auxiliary reference/donor images, bindings, empty annotations, mixed image/mask dtypes, and full replay,
  tracing, reentrancy, DataLoader, and terminal-transform rules.

Reuse `tests/contracts/test_target_cluster_contract.py`, `tests/contracts/test_image_range_contract.py`,
`tests/test_tensor_compose.py`, `tests/test_tensor_dataloader.py`, the composition/replay suites, and the affected
2D/3D family tests. Add permanent tests only for behaviors the existing matrix cannot express.

## Measure the complete public route

Extend the existing [ASV catalog](../../benchmark/README.md) with paired float16/float32 workloads. Compare the
candidate's half and float32 routes on the same machine and dependencies; compare candidate float32 and uint8 with
the pre-change revision to catch regressions. Preserve the fixed uint8 release profile.

For changed image kernels use 256/512/1024 spatial sizes and 1/3/5 channels where supported. Add one non-square and
strided representative. For batches start with four images; for 3D start with a `16 x 128 x 128` volume and two-volume
batch. Expand only along an axis that could change routing or invalidate the conclusion. Measure NumPy and CPU Torch
separately, direct functional calls, applied leaves, and complete Compose pipelines.

Compose workloads include an empty graph, root skip, probabilistic skip, one cheap applied transform, a numerical
transform, a mixed geometry/pixel/normalization chain, native/fallback Tensor transitions, annotation processors,
replay, tracing, and concurrent calls. Compare equivalent effects and verify each route's agreed output dtype. Choose
controls exactly representable in half where practical; verify that both dtypes execute the intended branch. Full
Compose timings include parameter and random-field generation in each dtype. Fixed-parameter or replay timings
provide controlled kernel/dispatch evidence separately; they do not replace the full sampling-inclusive comparison.

Use one CPU thread per process, OpenCV internal threading disabled, and Torch intra/inter-op threads set to one.
Set BLAS/OpenMP thread controls before imports. Record effective settings, CPU/OS, package versions, source SHAs,
shape, dtype, mode, parameters, warmup, repetitions, median, and timing dispersion. The required acceptance platforms
are native macOS ARM64 and Linux x86_64. Speed, memory, and correctness must pass independently on each. Fix and
record the exact CPU models and dependency versions before measuring.

The migration gate applies independently on each platform to old/new implementations at the same dtype. An individual
operation may regress by up to 5% only with substantial code simplification, preserved accuracy, no working-memory
growth and no sustained slowdown of the full AX path. Timing noise requires a focused repeat. Float16/float32
comparisons remain diagnostic. Report every measured cell and retain losing candidates; an isolated kernel win or a
half-sized returned array cannot establish Compose throughput.

Measure and record peak memory for the paired float16/float32 public workloads, including working buffers and bridge
copies. Run memory measurements separately from timing so instrumentation does not change the speed comparison.
Reuse the existing benchmark facilities and state the measured metric, method, absolute peak, and difference between
dtypes. Returned-array storage alone is not a peak-memory measurement. Old/new working memory must not grow at the
same dtype. Attribute fixed resident code pages separately from working buffers. Half/float32 memory comparisons are
diagnostic; newly enabled routes still need explicit peak/retained measurements. Do not claim a peak-memory reduction
solely from half-sized input or output arrays.

## Completion conditions

- [x] A compatible Albucore 0.3.2 wheel is published and selected by the AX exact pin, lock and Conda recipe.
- [ ] Every applicable built-in transform, constructor mode, target family, and composition node has float16 coverage.
- [ ] NumPy and CPU Torch outputs satisfy the agreed dtype, precision, mask, layout, ownership, and replay contracts.
- [ ] Float32 regression expectations remain valid; accelerated uint8 float round trips satisfy the agreed maximum
  pixel error of one. Other uint8 routes and categorical masks remain exact; bbox/keypoint precision stays unchanged.
- [ ] Same-dtype old/new full AX paths pass on macOS ARM64 and Linux x86_64; half/float32 diagnostics are recorded.
- [ ] Old/new working memory does not grow at the same dtype; fixed code pages and half/float32 diagnostics are reported separately.
- [ ] Sampling is reproducible and parameter replay is exact within each dtype; existing uint8/float32 random
  sequences remain unchanged. Numerical comparisons use identical recorded parameters across dtypes.
- [ ] Documentation describes supported routes accurately; known upstream gaps remain explicit until resolved.
- [ ] Superseded code is removed, final changes pass the unslop review, and focused quality checks and tests pass.

Scope, output dtype, channel handling, Tensor bridging, coordinated delivery, validation platforms, family-specific
float16 tolerances, dependency-work boundaries and working-memory acceptance follow the agreed Core/AX architecture
contract. Reproducibility and exact replay remain required within each dtype. Per-route correctness, speed, numerical
bounds and backend capability require measurement. Select the compatible final release pin before publication.
This document records implementation work, not completed support.
