# Torch CPU Backend and Tensor-Native Compose Plan

**Status:** In progress

**Decision:** `torch` becomes a required dependency. The existing `Compose` accepts both NumPy arrays and CPU
`torch.Tensor` values. One execution engine routes each compatible *segment* to NumPy/OpenCV/NumKong or Torch based on
full-path benchmark evidence. A Tensor input may use a faster NumPy segment and return to Tensor form. A NumPy input
may use a faster Torch segment and return to NumPy form, unless an existing explicit terminal `ToTensorV2` or
`ToTensor3D` transform requests Tensor output.

**Primary workload:** a PyTorch training process in which Torch is already imported. Steady-state `DataLoader`
throughput is a milestone integration gate. Individual transform pull requests use direct and `Compose` benchmarks
without rebuilding and timing a complete loader for every matrix cell.

## Current implementation status

The repository is in the foundation stage. `torch` is now a required dependency. `Compose` accepts CPU Tensor image,
mask, bbox, and keypoint targets, preserves their public representation through the existing dispatch flow, and rejects
accelerators, autograd inputs, mixed spatial representations, and legacy terminal Tensor transforms. Bbox and keypoint
processors still use one centralized NumPy bridge internally and restore their `float32` Tensor output at the public
boundary. A private probe and a `DataLoader` collation test cover this plumbing.

`NoOp` accepts every supported Tensor target. The measured `Transpose` capability accepts C=1 and C=3 Tensor images,
plus `mask`, `masks`, and `mask3d`; it keeps Tensor bbox and keypoint entry and exit through the central processor
bridge. Its Tensor `images` and `volume` routes remain unsupported. Every accepted Tensor-image pipeline requires each
supplied spatial target to be a Tensor; the prior image-Tensor plus NumPy-target combination is no longer accepted.

The repository also contains `python -m tools.tensor_compose_benchmark`. It records raw model-ready `Compose` and
optional steady-state `DataLoader` measurements as JSON. Run it after a change to Compose, a bridge, batching, or
collation. Do not run it for every individual transform capability.

The first pilot, `HorizontalFlip`, was rejected on the current macOS ARM development host. A contiguous `C,H,W` RGB
Tensor must become a strided `H,W,C` view before it reaches the existing OpenCV helper. Across the 256, 512, and 1024
matrix, that route was much slower than the NumPy baseline. `torch.flip` also lost on the required 1- and 3-channel
cells. Torch won on some contiguous 5-channel cells, but a partial win cannot accept a route that regresses common
RGB inputs. No PyTorch primitive is missing, so this is a retained-backend decision rather than an upstream request.

`VerticalFlip` was also rejected: `torch.flip` loses on the required 256-pixel C=1 and C=3 cells, even though it wins
selected larger cells. The route therefore remains NumPy/Albucore rather than creating a partial Tensor capability.

The first 3D investigation rejected `CenterCrop3D`, `RandomCrop3D`, and `CubicSymmetry` as native CPU Tensor routes.
For no-padding crops, direct `Compose(volume=Tensor)` was 1.40x to 1.67x the NumPy route over the small and medium
volume matrix (C=1, C=3, C=5; `uint8` and `float32`). Reusing the NumPy crop helper through the shared bridge was
slower still. A correctness-equivalent `CubicSymmetry` prototype reduced all 48 symmetries to one Tensor permutation
and at most one flip or clone, but it still lost on required common cells; only the medium C=5 `float32` subset won.
`Pad3D` with `torch.nn.functional.pad` likewise lost by 1.10x to 3.43x outside the one medium C=5 `float32` win. The
needed Torch operations exist, so no upstream feature request is open. These transforms remain explicitly rejected
until a future CPU release or a different full-path implementation passes every required cell.

The initial integration run measured workers 0 and 2 successfully. The 8-worker run did not finish on this macOS ARM
host and produced no artifact. Repeat the 0, 2, and 8 worker suite on the stable Linux x86-64 benchmark host before
marking Phase 1 complete. The route result must include the generated JSON, not only a summary.

## Problem and goal

Today, a caller that trains a PyTorch model normally augments a NumPy array and converts it to Tensor at the end. This
adds representation boundaries and may copy data. Some Torch CPU kernels can replace NumPy, OpenCV, or NumKong work.
Others cannot match their speed or semantics. A global replacement would regress part of the matrix.

The goal is to remove only the boundaries and helpers that full measurements prove safe. The result preserves the
current NumPy contracts and adds a direct Tensor input form for training pipelines. It does not require all work to use
one backend.

## Decisions

### One Compose and two representations

There is one public `Compose`, one transform hierarchy, and one parameter-sampling and replay flow. `apply`,
`apply_to_images`, and `apply_to_volume` decide target policy and dispatch. Reusable pixel arithmetic remains in the
functional layer or Albucore.

The planner may choose either representation for any compatible contiguous segment:

```mermaid
flowchart LR
    NI["NumPy input"] --> NS1["NumPy / OpenCV / NumKong segment"]
    NS1 --> NT["Measured bridge"]
    NT --> TS["Torch segment"]
    TS --> NO["NumPy output"]

    TI["Tensor input"] --> TS1["Torch segment"]
    TS1 --> TN["Measured bridge"]
    TN --> NS["NumPy / OpenCV / NumKong segment"]
    NS --> TO["Tensor output"]
```

The diagram shows possible boundaries, not a required alternation. The planner groups adjacent helpers with the same
backend. It must not convert around every helper: a Tensor → NumPy → Tensor crossing is paid once per accepted NumPy
segment, and a NumPy → Tensor → NumPy crossing is paid once per accepted Torch segment.

The first segment may use either backend. No helper may add an ad hoc `.numpy()`, `torch.from_numpy()`, or layout
conversion. The central bridge and capability registry own every crossing. The registry records why a route is
eligible, rejected, or blocked.

### Tensor boundary contract

The public CPU Tensor contract is channel first:

| Compose target | Tensor shape | Meaning |
|---|---|---|
| `image` | `C, H, W` | One 2D image |
| `images` | `C, L, H, W` | Sequence of images; `L=N` for an image sequence and `L=T` for video frames |
| `volume` | `C, D, H, W` | One medical or scientific volume |

`images` and `volume` deliberately share the four-axis pattern `C, L, H, W`. The target key gives the second axis its
meaning. `images` selects framewise image semantics; `volume` selects volume and 3D-transform semantics. No `video`
or `videos` target is introduced by this work.

Normal `DataLoader` collation adds the outer batch axis. A 3D video model receives `B, C, T, H, W`; a volume model
receives `B, C, D, H, W`. This matches the input convention of `torch.nn.Conv3d` and common PyTorch video models.

The logical axes are fixed. There is no public `data_format`, `channel_first`, or `channel_last` parameter.
`torch.channels_last` and `torch.channels_last_3d` are physical-memory-format candidates only at valid batched model
boundaries. They do not alter the public Tensor axis contract.

NumPy callers retain the current explicit-channel, channel-last contracts: `H,W,C`, `N,H,W,C`, and `D,H,W,C`.

### Annotation Tensor boundary

When the caller supplies an image Tensor, every supplied spatial target also uses Tensor form. This removes the
last public representation boundary before `DataLoader` collation. Annotation axes follow PyTorch training
conventions rather than the image channel convention:

| Compose target | Tensor shape | Dtype | Meaning |
|---|---|---|---|
| `mask` | `H,W` | `uint8`, `int64`, `float32`, or `bool` | One semantic, regression, or binary mask |
| `masks` | `N,H,W` | `uint8`, `int64`, `float32`, or `bool` | `N` instance masks |
| `mask3d` | `D,H,W` | `uint8`, `int64`, `float32`, or `bool` | One volumetric mask |
| `bboxes` | `N,K` | `float32` | `N` boxes in the configured current coordinate format; the existing processor defines columns after coordinates |
| `keypoints` | `N,K` | `float32` | `N` keypoints in the configured current coordinate format; the existing processor defines extra columns |

`mask` has no channel axis because a training mask usually stores one label or value per pixel. `masks` adds an
instance axis, not an image-channel axis. If a model needs a channel-first target, the dataset or collate function
creates that model-specific form after `Compose`.

The implementation keeps one transform hierarchy. A route may bridge Tensor annotations to the current NumPy
processor or helper when that full route is faster, then return Tensor. The bridge owns conversion, layout, and
ownership checks. No transform-specific helper may convert an annotation silently.

### CPU-only boundary

The first stage accepts CPU tensors with `requires_grad=False`.

- `Compose` exposes no `device` parameter and never calls `.to(device)`.
- CUDA, MPS, XPU, and other accelerator tensors are rejected during capability validation.
- The pipeline does not construct an autograd graph, manage streams, set global Torch thread settings, or change the
  caller's grad mode, RNG state, deterministic-algorithm setting, or default dtype.
- `DataLoader(pin_memory=True)` and host-to-device transfer remain training-code responsibilities after `Compose`.

Existing `ToTensorV2` and `ToTensor3D` remain explicit NumPy-to-Tensor terminal transforms. The CPU routing work does
not change their legacy NumPy behavior. A Tensor-native pipeline does not contain either transform because its input is
already Tensor. Tensor capability validation reports a clear error when a Tensor-input pipeline still contains one and
asks the caller to remove it.

Accelerator support is a later project phase. It reuses the Tensor and capability contracts only after the CPU routes
have passed their performance gate.

## Bidirectional segment planner

### Capability declaration

Each transform family and reusable functional helper receives a private capability record. It declares:

- supported targets, dtypes, ranks, channel counts, layouts, strides, and parameter modes;
- exact output shape, dtype, range, mutation, aliasing, RNG, replay, and annotation behavior;
- available implementations: current backend, Torch, or both;
- valid entry and exit bridges, including all required layout views and copies;
- supported predecessor and successor capability IDs; and
- benchmark cells and status: `accepted`, `partial_route`, `existing_backend`, `rejected`, or `blocked_upstream`.

`Compose` builds the route from the ordered transforms and their target set. Nested `OneOf`, `SomeOf`, and `Sequential`
branches must all have a valid plan before execution. A plan must not change probabilities, parameter sampling, replay,
or transform order.

### Bridges are operations, not bookkeeping

On a supported CPU array/Tensor, `torch.from_numpy(array)` and `tensor.numpy()` can share storage. That does not make a
route free. Torch dispatch, reference lifetime, unsupported strides, read-only storage, layout views, and a required
contiguous copy all affect wall time and ownership.

A Tensor NumPy segment commonly needs these steps:

```text
C,L,H,W Tensor → layout view for the NumPy helper → NumPy helper → output layout view → Tensor
```

Any `.contiguous()`, `np.ascontiguousarray()`, dtype cast, or allocation is O(n). The benchmark includes it. If the
NumPy helper only accepts a layout requiring a copy, the planner uses that route only when the entire segment remains
at least as fast as its baseline and preserves the stated ownership contract.

The analogous rule applies to NumPy callers entering Torch. A fast Torch kernel does not qualify if its bridge,
layout conversion, and return conversion make the whole route slower.

### Backend decision rule

For every matrix cell:

1. Keep the current backend when it is faster or is the only implementation with the required semantics.
2. Route a compatible segment through Torch when the full route is faster or tied within calibrated noise and every
   other contract is equal.
3. At a measured tie, prefer Torch when it removes a boundary or extends a proven adjacent Torch segment.
4. Route a Tensor segment through NumPy/OpenCV/NumKong when that complete Tensor → NumPy → Tensor route is faster
   and the enclosing Tensor `Compose` route passes its direct performance gate.
5. Split a segment when a required operation loses. A later win cannot compensate for a repeatable regression.

The input representation does not force the backend. It determines only the input and output representation seen by
the caller.

## Performance invariant and evidence

### Acceptance condition

> No accepted route may show a repeatable CPU slowdown in any required benchmark cell.

Torch is imported before both baseline and candidate timing. Cold import time, wheel size, and install time are tracked
as dependency diagnostics; they do not decide a hot training-path route.

For NumPy input, the baseline is the current public NumPy `Compose` or functional call. Each transform capability has
two blocking comparisons that do not construct a `DataLoader`:

1. **Direct Compose gate:** run the same transform chain and sampled parameters with pre-created, logically identical
   NumPy and CPU Tensor inputs. Input construction and output normalization for value comparison stay outside the timed
   block. `Compose(image=tensor)` must be no slower than `Compose(image=array)` in every accepted cell. Measure both a
   shared-storage channel-first Tensor view and a native contiguous channel-first Tensor where either is supported.
2. **Model-ready output gate:** compare NumPy `Compose` plus the existing explicit terminal `ToTensorV2` or
   `ToTensor3D` conversion with direct Tensor `Compose`. This is a controlled `Compose` benchmark with pre-created
   inputs; it does not include collation or `DataLoader` workers.

The direct gate detects Tensor-dispatch or planner overhead. The model-ready output gate measures the cost to produce
the Tensor that the model consumes. A candidate includes every bridge and layout operation that occurs inside its
timed path.

A transform PR reports every direct helper, segment, direct-Compose, and model-ready-output cell. The output report
records raw before/after measurements rather than aggregates alone.

### Benchmark cadence

`DataLoader` and collation measure shared integration behavior. Repeating them for every transform would multiply the
runtime without isolating that transform's kernel or dispatch cost. Use this cadence:

| Change | Required benchmark |
|---|---|
| `Compose`, preprocessing, postprocessing, shape handling, common bridge, or planner | Direct microbenchmarks, `Compose`, and representative `DataLoader`/collation cases |
| One transform or functional family gains Tensor capability | Direct functional, ordered segment, direct `Compose`, and model-ready output; no `DataLoader` |
| Several accepted capabilities form a new mixed-backend segment | Ordered segment and `Compose`; add one representative `DataLoader` case only when a shared boundary or batch behavior changed |
| Milestone, release candidate, or scheduled performance run | Full representative `DataLoader` suite for NumPy, Tensor, and mixed routes |

This keeps per-transform feedback short while preserving an integration gate for changes that can affect the whole
training input pipeline.

### Required matrix

Every affected 2D route covers the repository matrix where supported:

| Resolution | Channels | Typical workload |
|---|---:|---|
| 256 × 256 | 1, 3, 5 | Classification and RGBD/multispectral training |
| 512 × 512 | 1, 3, 5 | Detection and segmentation |
| 1024 × 1024 | 1, 3, 5 | High-resolution, medical, and satellite workloads |

Run `uint8` and `float32` whenever the public function supports them. Also cover positive and negative strides,
read-only storage, non-contiguous Tensor inputs, image sequences, video-through-`images`, masks and target
annotations where applicable, and the selected parameter axes.

When the integration suite runs, it holds batch size, workers, persistent workers, prefetching, pinning, collation, and
output consumption constant. It compares decoded NumPy samples, Tensor samples, mixed backend segments, and 0, 2, and
8 workers where the workload is relevant.

### Correctness and ownership gates

An accepted route preserves:

- target alignment for image, mask, bbox, OBB, keypoint, volume, and metadata;
- shape, axis meaning, dtype, value range, exact integer behavior, and documented floating tolerance;
- seeded RNG behavior, replay records, transform probability, and constructor serialization;
- caller-visible mutation and aliasing behavior; and
- output representation: backend routing returns the input representation. Existing explicit terminal `ToTensorV2`
  and `ToTensor3D` transforms retain their documented NumPy-to-Tensor behavior.

A Tensor route may use a NumPy segment internally. This boundary is planner-controlled and reported in the route
artifact. It is not an implicit fallback hidden inside a helper.

## Implementation phases

### Phase 0 — dependency and baselines

1. Add the validated Torch floor to runtime dependencies. Keep TorchVision optional.
2. Update lockfile, CI, platform/Python support checks, SBOM/security inputs, and environment reporting.
3. Measure baseline variance on a stable Linux x86-64 machine. Calibrate a repeatability threshold no larger than 1%.
4. Add the route-result JSON schema: environment, input representation, helper/segment IDs, every bridge, raw samples,
   correctness result, memory result, and decision.

Exit: a default install includes Torch, and the benchmark runner can distinguish a real regression from noise.

### Phase 1 — prove the complete Compose Tensor lifecycle once

The first runtime pull request changes the framework plumbing. It does not add Tensor implementations to every public
transform.

1. Define Array-or-Tensor internal types and the fixed Tensor target contracts.
2. Extend `Compose` input checks, shape extraction, preprocessing, target routing, postprocessing, and output repair for
   CPU Tensor input.
3. Pass Tensor values through `BasicTransform.__call__`, `apply_with_params`, and the existing `apply_*` dispatch.
4. Use a minimal private probe transform to prove that the Tensor arriving at `apply` or `apply_to_images` also leaves
   `Compose` as Tensor. Test empty pipelines, `p=0`, nested compositions, additional targets, masks, annotations,
   replay, and failure before parameter sampling.
5. Reject accelerator tensors, `requires_grad=True`, and Tensor-input pipelines containing `ToTensorV2` or
   `ToTensor3D` with actionable errors.
6. Preserve every NumPy test and benchmark result.
7. Run the representative `DataLoader` and collation suite once for this shared lifecycle change.

Exit: the full `Compose` wrapper carries supported CPU Tensor targets from input to a transform and from its result to
the caller. No public transform is considered Tensor-capable until its own capability PR lands.

### Phase 2 — add the central bridge and the first two capabilities

1. Build one bridge API for Tensor/NumPy conversion, axis adapters, ownership checks, and route diagnostics.
2. Add the capability record and single-transform routing needed by the first pilots. Do not build a global optimizer
   before real capabilities exist.
3. Prove one Tensor input that uses a faster NumPy/OpenCV helper and returns Tensor.
4. Prove one NumPy input that uses a faster or tied Torch helper and returns NumPy.
5. Run direct, `Compose`, model-ready-output, and one representative `DataLoader` benchmark for each bridge direction.

Exit: both bridge directions preserve correctness and pass their full-path performance gates. Adjacent accepted
capabilities can then be grouped into longer segments without changing transform order.

### Phase 3 — add Tensor support one transform family at a time

Each transform-family pull request follows the same bounded workflow:

1. Freeze the current NumPy contract and save reference outputs for every target, dtype, channel count, and parameter
   mode in scope.
2. Identify the best Tensor route. It may call a Torch helper or cross once into an existing NumPy/OpenCV/NumKong
   helper and return to Tensor.
3. Add the Tensor implementation to the existing functional and `apply_*` flow. Keep reusable arithmetic in the
   functional layer or Albucore; do not add a parallel transform class.
4. Add correctness tests for Tensor input and confirm that NumPy input remains unchanged.
5. Run direct functional, ordered-segment, direct `Compose`, and model-ready-output benchmarks for NumPy and Tensor.
   Do not run `DataLoader` for this transform PR unless it changes shared bridge, planner, batching, or collation code.
6. Accept, narrow, reject, or mark the capability `blocked_upstream`.
7. After the Tensor path is accepted, test whether the same Torch helper should serve NumPy input. Route NumPy through
   Torch only when the full NumPy → Torch → NumPy path is no slower.

Land one operation family, capability record, tests, benchmark artifact, and routing change per pull request.

### Phase 4 — consolidate segments and run milestones

As accepted capabilities accumulate, group adjacent compatible operations so each NumPy/Tensor boundary is paid once
per segment. A segment change runs its ordered-chain and `Compose` benchmarks. Add a representative `DataLoader` case
only when the segment changes shared boundaries or batch behavior.

Run the complete `DataLoader` suite on scheduled milestones and before release. It covers NumPy input, Tensor input,
mixed backend segments, video-through-`images`, volume, and the supported worker matrix.

### Phase 5 — audit retained backends and prepare accelerators

After the CPU registry is complete, publish the retained NumPy, OpenCV, NumKong, Albucore, and Torch routes with their
evidence. Removing a dependency requires a separate audit showing no remaining required or faster route.

Accelerator work later adds device and stream capability, transfer planning, synchronization-aware benchmarks, and
RNG/autograd policy. It does not change the public Tensor axis contract or create another composition API.

## Missing Torch operations

An unavailable primitive, missing dtype/stride support, incorrect semantics, or unavoidable losing copy blocks only its
candidate. Open the request where the missing capability belongs:

- PyTorch for a general Tensor operator, CPU dispatch, dtype, stride, correctness, or performance gap;
- Albucore for a reusable image-processing helper or shared bridge contract;
- TorchVision for a vision-specific operator that does not belong in core Torch; or
- AlbumentationsX for transform semantics and local routing.

The issue or pull request includes:

- a minimal input and expected result;
- dtype, shape, stride, layout, and device;
- a correctness reproducer;
- a self-contained benchmark; and
- the affected capability IDs and fallback decision.

The registry marks that candidate `blocked_upstream` and the next candidate proceeds. A missing operation never blocks
unrelated migration work.

## Completion criteria

The CPU program is complete when:

- Torch is a required dependency on supported platforms;
- `Compose` accepts the declared CPU Tensor forms and returns Tensor output for Tensor input;
- the route planner can use accepted NumPy and Torch segments in either input representation;
- every accepted route has permanent correctness tests and full per-cell performance evidence;
- every accepted Tensor route passes both the direct Tensor-Compose and model-ready-output performance gates;
- Compose/planner milestones and release candidates pass the representative `DataLoader` suite;
- no accepted route has a repeatable CPU regression;
- all missing primitives have an upstream artifact or explicit retained-backend decision; and
- the audit explains every retained backend and leaves a reusable capability contract for future accelerator work.

## References

- [PyTorch `Conv2d`](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Conv2d.html)
- [PyTorch `Conv3d`](https://docs.pytorch.org/docs/2.13/generated/torch.nn.Conv3d.html)
- [PyTorch tensor memory formats](https://docs.pytorch.org/docs/2.13/tensor_attributes.html)
- [TorchVision Tensor conventions](https://docs.pytorch.org/vision/stable/transforms.html)
- [PyTorchVideo data convention](https://pytorchvideo.readthedocs.io/en/latest/data.html)
- [Kornia `VideoSequential`](https://kornia.readthedocs.io/en/v0.6.3/_modules/kornia/augmentation/container/video.html)
- [MONAI transform conventions](https://docs.monai.io/en/0.9.1/transforms.html)
- [TorchIO transform input](https://docs.torchio.org/latest/transforms/)
