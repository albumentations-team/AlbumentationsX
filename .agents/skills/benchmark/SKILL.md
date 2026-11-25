---
name: benchmark
description: Measures AlbumentationsX runtime changes with paired baseline and candidate benchmarks on the affected routes. Use when claiming or checking a speedup or regression, running ASV, timing a transform or functional kernel, or comparing branches ("benchmark this", "is it faster", "замерь").
---

# Benchmark a runtime change

Benchmark when a change can alter executed work in a functional operation, sampling, target dispatch, or Compose.
Documentation and mechanical refactors with no plausible runtime effect need no timing run.

Read [Performance Optimization](../performance-optimization/SKILL.md) and its required reference before selecting
candidates. Define the changed route and the workload dimension that could falsify the proposed speedup.

## One thread, always

**Every timing runs with exactly one CPU thread, including quick ad-hoc checks.** Production is AX under a PyTorch
DataLoader with one thread per worker, so a multi-threaded timing measures the wrong thing. Before importing numerical
libraries set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and
`NUMEXPR_NUM_THREADS` to `1`; then call `cv2.setNumThreads(0)`, `torch.set_num_threads(1)`, and
`torch.set_num_interop_threads(1)`. Time thread counts above one only on an explicit user request. Never write a bare
`timeit` loop; load the `benchmark` skill first.

## Acceptance profile

Final Core/AX acceptance, including float16 acceptance, uses Python 3.11 and the newest stable third-party releases
that support Python 3.11 and the target platform. OpenCV must be at least 5.1.0.95. Re-resolve before each acceptance
run and record the exact Python, NumPy, OpenCV, PyTorch, NumKong, StringZilla, and Albucore versions together with the
imported Albucore/AX source paths. A run from a stale lock, a newer Python, or an older environment is not acceptance
evidence.

On macOS ARM64, run NumKong benchmarks outside the agent sandbox (request elevated permissions) and verify `nk.get_capabilities()["neon"]` is true in the same process. The Albucore router harness refuses native timing when NEON detection fails; `--allow-restricted-isa` is diagnostic only. Never force capabilities to hide a failed hardware probe.

## Select the workload

For pixel arithmetic, dtype conversion, layout, or backend routing, use the nine combinations of square sizes
`256`, `512`, and `1024` with `1`, `3`, and `5` channels. Skip explicitly unsupported channel counts. Keep grayscale
NumPy inputs `(H, W, 1)`.

For other changes, vary the controlling axis: label count and density, random-output size, annotation count, or both
sides of a routing threshold. Explain why the chosen matrix covers the claim.

- Measure the direct functional call and the public Compose route when both are affected.
- For batch changes, compare per-image dispatch with the batch route at batch sizes `4`, `8`, and `16`.
- For dtype routing, measure uint8 and float32. A change confined to one dtype may time that dtype and use correctness
  tests to verify the other dtype's wrapper round-trip.
- For Compose changes, include root skip, no-op, probabilistic no-op, an always-applied cheap leaf, applied-parameter
  capture, trace, Tensor, processors, and concurrent calls. For annotation changes, include empty, single, and dense
  annotations alongside image-only inputs.
- Construction time and retained allocations are optional context when RNG or graph ownership changes. Judge the
  result by repeated-call performance; a measured call-time gain can justify slower construction.

## Use the existing catalog

Start with [benchmark/README.md](../../../benchmark/README.md) for ASV commands and
[Performance Coverage](../../../docs/maintaining/performance-coverage.md) for case ownership.

`release-core` supplies the fixed weekly and release profile. `changed` selects affected families for requested PR
comparisons. Resolve these profiles with `tools/select_benchmark_filters.py`. Use ASV `continuous` with explicit
baseline and candidate refs. An empty selection does not authorize a full-catalog run; choose an explicit regex for
a larger investigation.

If the catalog cannot express the question, keep a focused local measurement under `_internal/`. Save matching
before/after cells with shape, dtype, channels, parameters, allocation mode, elapsed time, and iteration count.

## Compare and report

1. Run baseline and candidate on the same machine and environment, back-to-back, with warm-up and exactly one
   CPU thread per process. Follow the thread controls in the required performance guide. Benchmark additional
   thread counts only when the user explicitly requests a thread-scaling experiment.
2. Use at least 100 iterations for fast functions. For slow functions, choose enough repetitions for stable timing,
   aiming for more than one second per cell.
3. Verify correctness, seeded behavior, and aliasing before accepting a faster path.
4. Report every before/after cell, speedup, exact revisions, command or ASV filter, environment, and rejected candidates.
5. Investigate any regression above 5%; rework it or explain the measured trade-off.

A one-revision timing is a baseline observation, not evidence of a speedup.
