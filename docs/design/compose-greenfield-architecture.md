# Greenfield Compose architecture

## Decision

`Compose` is constructed once and applied once per training sample, often millions of times. Its performance budget is
therefore the warmed cost of `pipeline(**sample)`, not the cost of `Compose.__init__()`. Construction may validate,
normalize, allocate, and compile any configuration-derived state that makes later calls faster.

Root `p=0` is a correctness contract, not a representative benchmark. `save_applied_params`, tracing, and replay are
observed routes with separate budgets; they must not add branches, allocations, or state retention to ordinary
training calls.

Transform subclasses implement the single supported sampling hook, `sample_parameters`.

## Configured graph

Construction freezes `transforms` as a tuple and derives immutable edge descriptors. Each descriptor holds the child
node and its precomputed bbox/keypoint effects, including whether its parent must perform annotation finalization.
Graph operators and deserialization construct a new graph and therefore a new plan. Runtime code must not scan the
configured graph to rediscover this policy.

The construction phase is the place to add further static information when it removes work from every call:

- target-effect masks and annotation-filter points;
- supported Tensor and root-boundary routes;
- selector and nested-container policy;
- stable serialization and trace paths;
- root RNG ownership and process-local synchronization.

Configured graph nodes contain policy only. Per-call parameters, processor sessions, trace state, grayscale repair,
and instance-binding state belong to the invocation.

## Invocation and sampling

One public `Compose` call creates one `InvocationContext`. The root is responsible for input validation,
preprocessing, graph execution, postprocessing, and restoration of public Tensor and grayscale layouts. Configured
nested `Compose`, `OneOf`, `SomeOf`, `RandomOrder`, `OneOrOther`, `SelectiveChannelTransform`, and `Sequential`
nodes receive that invocation explicitly. Generic, observed, and replay paths use `apply_in_invocation`; the ordinary
compiled path calls its dedicated direct executor. Neither opens another public boundary.

Built-in samplers receive randomness explicitly:

```python
def sample_parameters(
    self,
    inputs: TransformSamplingInput,
    sampling: SamplingContext,
) -> TransformParameterPlan:
    value = sampling.py_random.uniform(0.0, 1.0)
    return TransformParameterPlan.shared_only({"value": value})
```

`SamplingContext` exposes the invocation-local Python and NumPy streams plus `applied_overrides`. Built-in sampler
code must use these streams, never `self.py_random` or `self.random_generator`. For an ordinary call, the reusable
sampling context is pointed at a no-op override sink: samplers can keep the same interface without allocating or
retaining an applied-configuration dictionary. Observed calls provide a real sink and build the durable replay record.
`apply_*` receives realized parameters and target data only; it does not generate randomness.
Target-sensitive samplers use `inputs.targets` and return `TargetParameterGroup` entries keyed by actual target names;
they must not route through target-named fields such as `volume_noise_map`.

## Ordinary hot path

The ordinary executor has one root boundary and a direct loop over compiled children. It must avoid work that cannot
affect the output of that call:

- no applied-parameter records, snapshots, timing, or trace records;
- no `ContextVar` lookup at built-in child edges or built-in sampler RNG access;
- no per-child `nullcontext`, trace scope, or timer check;
- no annotation shape calculation, clipping, filtering, or instance resynchronization for image-only edges;
- no processor session or label-manager allocation when annotation targets are absent;
- no repeated Tensor conversion or grayscale normalization inside the graph.

An eligible ordinary graph receives an explicit, unactivated invocation: built-in children do not need a `ContextVar`
lookup because they receive all call-local state as an argument. Its all-`BasicTransform` case uses a direct loop over
the compiled children. Other configured graphs use the same compiled edge descriptors and explicit invocation, while
retaining only the policy their containers require.

## Annotation schedule

The plan decides at construction which edges may alter bboxes or keypoints and whether a child container already
finalizes them. `check_each_transform=True` filters only after an edge that makes the relevant target dirty. If no
configured processor needs filtering after an edge, the executor returns before constructing shape metadata.

Processor sessions are call-local. Bbox survival produces one keep mapping that synchronizes bound masks, keypoints,
and instance identifiers before the next dependent node. Root postprocessing performs the remaining conversion and
label restoration once.

## RNG and concurrency

The root RNG owner creates its lock during construction. Locks and thread-local reservation state are excluded from
pickle state and recreated in the receiving process. The short reservation lock protects only stream allocation;
validation, arrays, transforms, tracing, and postprocessing run outside it.

An explicit `invocation_seed` creates isolated sample-keyed streams and does not advance the configured worker
stream. Concurrent users of one configured pipeline receive independent invocation streams. A failed call neither
publishes observation state nor leaves an active invocation behind.

## Observed routes

Applied-parameter recording, tracing, and replay follow the same configured order and annotation schedule as ordinary
execution. They add only their requested artifacts:

- `save_applied_params` records constructor-valid realized policy;
- metadata tracing records node status without snapshots unless requested;
- timing exists only when requested;
- replay replaces sampling with recorded parameters and never resamples.

These routes are correctness-critical, but their allocations and branches are intentionally outside the ordinary
training path.

## Permanent contracts

Tests must prove all of the following:

- configured graphs are immutable and serialization round-trips behavior-affecting policy;
- configured nested containers share one invocation, while a public `Compose` called by custom code creates an
  independent root invocation;
- built-in samplers access RNG only through `SamplingContext`;
- concurrent calls, first use, pickle/unpickle, DataLoader workers, and explicit invocation seeds are deterministic
  and race-free;
- validation or execution failure cannot expose stale observed parameters;
- ordinary, applied-parameter, trace, and replay routes agree on decisions and transformed output for the same seed;
- traces expose the configured nested structure and post-filter target state;
- bbox/keypoint filtering, instance binding, Tensor bridging, and grayscale normalization occur at exactly the
  intended root or compiled-edge boundary;
- output values, dtype, shape, aliases, and target alignment preserve public contracts.

## Acceptance measurements

Compare a warmed, preconstructed pipeline against its baseline in the same process, Python environment, hardware,
image shape, dtype, and thread settings. Report every measured cell, including direct and Compose routes affected by a
change. Constructor time may be shown as context but cannot justify a regression in application time.

The primary cells are ordinary stochastic and deterministic training pipelines, including a realistic pixel-heavy
pipeline. Add bbox/keypoint, nested-selector, Tensor, trace, and replay cells when their route changes. Investigate
common-path regressions above 5%; do not hide one behind an average.
