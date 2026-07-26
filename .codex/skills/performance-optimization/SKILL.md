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

## AlbumentationsX Boundary

Keep transform policy, parameter sampling, target dispatch, and annotation semantics in AlbumentationsX.

Propose an Albucore primitive when an operation:

- is useful to more than one transform or image-processing caller;
- has stable array semantics independent of transform policy;
- benefits from dtype, channel, contiguity, or backend routing;
- can be tested and benchmarked as an atomic operation.

Do not create a local helper merely to avoid coordinating an Albucore change when the operation satisfies these
conditions.

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
