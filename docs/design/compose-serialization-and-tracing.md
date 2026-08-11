# Compose Serialization and Execution Tracing

**Status:** Implemented

This document defines the shared graph contract for portable composition policies and opt-in per-step observation.
It applies to `Compose`, `ReplayCompose`, and the nested composition containers.

## Overview

A composition is an ordered tree. The order and each child position are meaningful both when the pipeline is saved and
when it executes. The implementation has two deliberately separate projections of that tree:

- a portable constructor policy for `to_dict`, `save`, `from_dict`, `load`, and composition operators;
- a runtime execution trace returned only by `run_with_trace`.

The two projections share child order and class names. They do not share lifetime: trace options, callbacks, records,
snapshots, timings, RNG progress, processors, and worker state are never serialized.

## Portable constructor policy

`BaseCompose._get_reconstruction_kwargs()` is the single constructor-policy projection. `to_dict_private()` adds the
recursive `transforms` list, while composition operators reuse the same kwargs to make a new instance. A subclass
extends the method only with policy it owns.

`Compose` serializes all of its behaviour-affecting policy, including defaults and `None` values:

| Group | Fields |
|---|---|
| Target processing | `bbox_params`, `keypoint_params`, `additional_targets`, `semantic_mask_label_mappings`, `instance_binding`, `strict_instance_invariant` |
| Validation and output | `is_check_shapes`, `strict`, `save_applied_params` |
| Random and pixel policy | `p`, `seed`, `mask_interpolation` |
| Operational policy | `telemetry` |

`SomeOf` and `RandomOrder` add `n` and `replace`; `SelectiveChannelTransform` adds `channels`; `ReplayCompose` adds
`save_key`. `OneOrOther.first` and `second` are construction aliases: the canonical graph always stores
`transforms`.

Processor policy is represented as detached plain dictionaries. The constructor rebuilds processors, selection
weights, available keys, internal label maps, and runtime RNG objects. Those derived values must not be copied into a
portable pipeline definition.

### Transport codec

`to_dict()` emits JSON/YAML-safe values. Mapping keys that JSON would coerce to strings are encoded as ordered pairs
with a private codec-version marker and decoded before construction. A raw mapping that has the exact marker shape
escapes that private key, so it cannot be mistaken for codec metadata. This keeps integer label mappings in
`KeypointParams` and semantic-mask policy equivalent across dict, JSON, and YAML round trips.

The portable payload is a complete current-format definition. Runtime tracing has no effect on its shape.

## Runtime trace API

```python
trace = transform.run_with_trace(
    image=image,
    options=A.TraceOptions(snapshot_targets=("image", "bboxes")),
)

output = trace.data
records = trace.records
```

`ReplayCompose.replay_with_trace(saved_augmentations, image=image)` returns the same `TraceResult` type while replaying
recorded parameters.

`TraceOptions` is validated before the pipeline samples its first probability or parameter. `snapshot_targets` names
targets that the pipeline can process; duplicate, unknown, and string-as-a-sequence configurations fail immediately.
`TraceOptions` and `TraceRecord` are immutable values; `TraceResult` deliberately carries the normal mutable output
dictionary. None of these runtime objects is serialized or retained in a DataLoader-pickled pipeline.

| Option | Effect |
|---|---|
| `snapshot_targets` | Own copies of just these targets on executed leaf records |
| `include_timing` | Per-leaf `elapsed_ns` timings |
| `observer` | Synchronous callback for each completed record |
| `collect_records=False` | Observer-only mode; the returned `records` tuple is empty |

Tracing is off by default. `Compose.__call__` does not allocate a trace context, create records, call a callback, copy
targets, or read a clock.

## Concurrent invocation contract

One configured `Compose` or `ReplayCompose` instance is reentrant through serialized execution. A process-local
reentrant lock covers each complete `__call__()` and `run_with_trace()` invocation, including worker RNG
synchronization, probability and parameter sampling, Tensor route selection, preprocessing, transform dispatch,
processor work, tracing, restoration, replay finalization, and cleanup. Overlapping callers therefore observe results
consistent with one serial order and cannot replace another call's grayscale, instance-binding, applied parameter, or
representation bookkeeping.

This contract does not provide parallel augmentation from one pipeline object. Lock acquisition order determines
which overlapping caller receives the next draw from a seeded shared RNG stream. Applications that require parallel
CPU augmentation should use independent pipeline instances, as normal multi-process `DataLoader` workers do.

The Tensor route plan is call-local rather than serialized policy or mutable pipeline state. Runtime locks are also
excluded from pickle payloads and recreated after unpickling, so worker processes never share synchronization objects.

## Trace records and paths

A `TraceRecord` contains:

- `node_path`: tuple of child indices, such as `()` for the root and `(0, 2)` for its third grandchild;
- `event_index`: chronological visit number in one call;
- `occurrence_index`: visit number for the same structural path, needed by `SomeOf(replace=True)`;
- `class_fullname` and `node_kind` (`composition` or `leaf`);
- `status` (`applied`, `skipped_probability`, `skipped_selection`, or `skipped_replay`);
- detached sampled `params` when a leaf applied;
- an optional target `snapshot`; and
- optional `elapsed_ns`.

Paths are derived from the current ordered `transforms` tree; they are not stored as another serialized field. A path
therefore survives dict/JSON/YAML reconstruction of the same graph, but it intentionally changes after a structural
edit such as inserting or reordering a transform.

Applied leaves are recorded after the composition's per-transform filtering and instance resynchronization boundary.
Skipped subtrees also receive lightweight records, allowing a debugger to map every decision back to the serialized
tree. The trace consumes the existing RNG stream only; a normal and traced pipeline with the same seed have equal
final outputs and equal continuation on the next call.

Nested `Compose` nodes run their own preprocess and postprocess boundaries while tracing, so shape validation,
grayscale normalization, and nested policy match an ordinary pipeline call.

## Snapshot and timing ownership

Metadata-only records contain no target arrays. A snapshot copies each requested value at emission time: NumPy arrays,
Torch tensors, and nested Python values do not alias later pipeline state or the final result. The snapshot represents
the normalized post-step working state; `TraceResult.data` is the normal public postprocessed result.
Torch snapshots use `clone()` on the tensor's current device, so requesting them for CUDA targets consumes device
memory; request only the targets required for diagnosis.

Timing is opt-in. It covers the leaf execution and required post-step boundary, while observer work and the actual
snapshot copy are outside the measured interval. Snapshot cost is intentionally a separate benchmarked mode.

`SelectiveChannelTransform` executes children on selected channels. When snapshots are requested, tracing reconstructs
the corresponding full image for the record; metadata-only tracing does not retain that image-sized state.

## Testing and performance gates

`tests/contracts/test_composition_serialization_contracts.py` discovers every public concrete `BaseCompose` subclass,
requires every portable constructor parameter to be present in its canonical payload, and verifies a non-default
witness through dict, JSON, and YAML. It also requires `BboxParams` and `KeypointParams` transport projections to
include every public constructor field.

`tests/test_composition_tracing.py` covers nested control flow, repeated selections, skipped nodes, snapshots,
filtering, timing, observer-only mode, replay, RNG continuation, serialization-stable paths, and
`save_applied_params` parity.

`tests/test_compose_reentrancy.py` uses explicit thread events to force overlapping call attempts and covers normal
execution, tracing, replay, Tensor/NumPy restoration, grayscale layout, applied parameters, instance binding, and the
seeded serial order.

When the relevant core code or either test changes, the scoped `Check Compose policy and tracing contracts`
pre-commit hook runs both suites before a commit is created.

`TimeCorePipeline` remains the normal `Compose` performance baseline. `TimeCorePipelineTracing` measures metadata,
snapshot, and timing modes separately over the standard 3 sizes × 3 channel counts. A repeatable regression above 5%
in any normal-path cell requires investigation rather than averaging it away.

## References

- `albumentations/core/composition.py`
- `albumentations/core/serialization.py`
- `albumentations/core/tracing.py`
- `tests/contracts/test_composition_serialization_contracts.py`
- `tests/test_composition_tracing.py`
- `benchmark/benchmarks/test_core_pipeline.py`
