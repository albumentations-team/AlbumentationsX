---
name: benchmark
description: Measure AlbumentationsX runtime changes with paired baseline and candidate benchmarks on the affected routes.
---

# Benchmark a runtime change

Benchmark when a change can alter executed work in a functional operation, sampling, target dispatch, or Compose.
Documentation and mechanical refactors with no plausible runtime effect need no timing run.

Read [Performance Optimization](../performance-optimization/SKILL.md) and its required reference before selecting
candidates. Define the changed route and the workload dimension that could falsify the proposed speedup.

Use Python 3.11 and the newest stable third-party releases that support Python 3.11 and the target platform for
final acceptance benchmarks. The recorded profile uses PyTorch 2.15.0 (`2.15.0+cpu` on Linux), NumPy 2.4.6,
OpenCV 5.0.0.93, NumKong 7.8.6, and StringZilla 5.3.2. Re-resolve and record package versions and imported
Albucore/AX paths before a new acceptance run. A run from a stale lock or environment is not acceptance evidence.

On macOS ARM64, run NumKong benchmarks outside the Codex sandbox with `sandbox_permissions="require_escalated"` and verify `nk.get_capabilities()["neon"]` is true in the same process. The Albucore router harness refuses native timing when NEON detection fails; `--allow-restricted-isa` is diagnostic only. Never force capabilities to hide a failed hardware probe.

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
