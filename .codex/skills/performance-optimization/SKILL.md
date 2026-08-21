---
name: performance-optimization
description: Systematic performance audit for AlbumentationsX runtime code. Use whenever implementing, reviewing, profiling, or optimizing transforms, functional kernels, apply methods, random generation, reductions, label maps, dtype conversions, batch paths, allocation-heavy code, backend routing, or code that may belong in Albucore.
---

# Performance Optimization

Read `references/performance-optimization.md` completely before inspecting or changing runtime code. It is the
in-repository fallback copy of Albucore's canonical `docs/performance-optimization.md`, so the guide remains available
when the Albucore checkout is absent.

When changing the guide, update the canonical Albucore document first and keep the bundled copy synchronized.
When both repositories are available, verify exact synchronization with
`cmp -s ../albucore/docs/performance-optimization.md .codex/skills/performance-optimization/references/performance-optimization.md`.
A mismatch blocks completion.

## Workflow

1. Establish correctness and performance baselines before editing.
2. Run every stage of the guide's optimization pass. Do not stop after finding the first plausible improvement.
3. Treat every backend, vectorization, LUT, random-generation, `bincount`, and in-place proposal as a benchmark
   hypothesis.
4. Check the repository boundary. Move a reusable image-processing atom into Albucore instead of duplicating it in
   transforms.
5. Read `../benchmark/SKILL.md` completely and benchmark both the isolated kernel and the `Compose` route on its
   required size, channel, and dtype matrix.
6. Add correctness tests before accepting a faster path. Preserve shapes, dtypes, values, annotations, aliasing, and
   seeded-replay contracts unless the change explicitly documents a compatibility break.
7. Run the project validation workflow after the final implementation.

## Torch Runtime Contract

Torch is already an externally managed, soft-required AlbumentationsX runtime dependency: package metadata does not
select a CPU, CUDA, or MPS build, but importing `albumentations` requires an installed Torch runtime. Do not reject a
Torch-backed implementation on the grounds that it would introduce a new runtime dependency. Benchmark the complete
route, including NumPy/Tensor bridges, layout conversions, allocations, and return conversion.

## Compose Execution Changes

For `composition.py`, `transforms_interface.py`, or invocation-state work, keep configured graph policy separate from
per-call state. Sampled parameters, applied records, processor sessions, Tensor bridge metadata, grayscale repair, and
instance-binding bookkeeping belong to `InvocationContext`; a configured transform or `Compose` must not become a
mailbox for any of them.

- Reserve a default DataLoader seed only at the root. The reservation lock may cover the counter and configured RNG
  source; array conversion, preprocessing, transform execution, tracing, and finalization must remain outside it.
- Preserve one root prepare/finalize boundary. Nested `Compose` nodes run focused additional-target and shape checks
  without opening a second processor, grayscale, Tensor, or restoration boundary.
- Keep `BaseCompose._apply_child()` as the sole configured-graph dispatcher. `run_with_trace()` may attach an
  invocation-local observer, but it must not introduce a second node-selection or execution traversal.
- Schedule bbox/keypoint filtering from declared target effects. Image-only nodes must not pay for shape discovery,
  clipping, filtering, or instance re-alignment; a final conversion must not repeat a filter already current for the
  invocation.
- Optimize repeated application before construction. `Compose.__init__()` may compile the graph and eagerly prepare
  root-owned execution state when that removes work from the per-sample path. Report construction cost only as
  context; do not reject a call-time speedup because pipeline construction becomes slower.
- Keep `invocation_seed` sample-keyed and side-effect free with respect to the worker stream.
- Measure root skip, no-op, probabilistic no-op, always-applied cheap transform, applied-configuration capture, trace,
  Tensor, processor, and concurrent-call routes. Include the complete before/after cells in the handoff.
- Exercise `tests/test_compose_reentrancy.py`, `tests/test_per_worker_seed.py`,
  `tests/test_composition_tracing.py`, and the affected Tensor, processor, replay, and instance-binding suites.

## AlbumentationsX Boundary

Keep transform policy, parameter sampling, target dispatch, and annotation semantics in AlbumentationsX.

Keep transform `apply*` methods as thin policy and dispatch methods. A method may validate its transform-specific
runtime input contract, select a functional operation, and forward sampled parameters, but pixel arithmetic,
temporary-array construction, clipping, dtype routing, and kernel-selection branches belong in the functional helper
that owns the operation. The repository's pre-commit hook limits transform `apply*`
bodies to 20 code-bearing lines (excluding signatures, docstrings, blank lines, and standalone comments). Base
infrastructure classes whose names begin with `Base` and `Compose` orchestration are excluded; a non-public base class
must use that prefix.

Propose an Albucore primitive when an operation:

- is useful to more than one transform or image-processing caller;
- has stable array semantics independent of transform policy;
- benefits from dtype, channel, contiguity, or backend routing;
- can be tested and benchmarked as an atomic operation.

Do not create a local helper merely to avoid coordinating an Albucore change when the operation satisfies these
conditions.

If investigation identifies an Albucore defect, pause AlbumentationsX changes and open an Albucore Issue and PR
before resuming.

Do not add a forwarding wrapper around a one-line call merely to attach a decorator. Use `@clipped` when every route
of an image operation can leave the public range; branch and call `albucore.clip(..., inplace=True)` when only a known
float32 mode (such as cubic interpolation) can. A separate function is justified only when it owns a real image
operation and keeps that image policy distinct from masks or annotations.

## Required Handoff

Report:

- work deleted or avoided;
- full-array passes, copies, conversions, and allocations removed;
- vectorization and grouped-reduction candidates evaluated;
- LUT, random-generator, and backend candidates compared;
- safe in-place opportunities taken or rejected with a reason;
- operations moved or proposed for Albucore;
- correctness evidence and benchmark matrix;
- regressions, compatibility changes, and rejected candidates.
