# Performance coverage

AlbumentationsX maintains two complementary performance contracts:

1. The benchmark catalog maps public transform families to owned benchmark
   routes. `tools.benchmark_coverage` validates that mapping.
2. The fixed `release-core` ASV profile supplies recurring runtime and
   peak-memory regression evidence for release decisions.

Catalog coverage answers “does this public family have an owned benchmark?” It
does not report Python test coverage and does not replace a baseline-to-candidate
timing comparison.

## Catalog contract

Run the catalog check locally or through the `check-benchmark-coverage`
pre-commit hook:

```bash
uv run python -m tools.benchmark_coverage check
uv run python -m tools.benchmark_coverage details --output benchmark-coverage-detail.json
```

Public image transforms need a family-matrix route beyond catalog smoke. Target,
batch, mixing, volumetric, Tensor, and functional-kernel routes are explicit
where their public behavior requires them. The catalog retains matrices for
larger sizes, multiple channel counts, dtypes, and annotation scaling as local
diagnostics. Those axes are not scheduled by default.

Use `tools/benchmark_coverage.py details` when reviewing a transform addition or
benchmark-family change. It identifies the owning class, layer, route, required
coverage contract, aliases, and missing layers. Fix missing ownership instead of
adding an unrelated benchmark case.

## Release routine profile

`release-core` is the small, stable ASV profile for weekly and release evidence. It
runs 21 named runtime and peak-memory cases at `256×256`, RGB, and `uint8`.
The target-processor case uses exactly 10 bounding boxes and 10 keypoints.

The profile covers Compose behavior, target processors, geometric, pixel,
mixing, and volumetric transforms. The `release-core` selector below is the
exact case list. Do not add a second input-size, channel-count, dtype, or
annotation-count axis to this profile. Add a
representative transform only when it covers a distinct core path.

```bash
uv run python tools/select_benchmark_filters.py --profile release-core
```

The scheduled Performance workflow and release preflight run ASV `continuous`
against a baseline and candidate. They save raw ASV output, parsed summary,
catalog coverage detail, environment evidence, and a performance-budget result.
Release preflight requires a comparison and fails on release-blocking regressions.

## Memory regression gates

Memory has three separate checks:

| Check | Failure it detects | Execution |
|---|---|---|
| Reference lifetime | A completed call retains arrays through an ownership cycle | Normal pytest compatibility matrix |
| Repeated-call RSS | Native allocations or allocator retention accumulate across calls | Linux Python 3.11 PR and release-candidate jobs, weekly performance, release preflight |
| Working peak RSS | One call allocates excessive output or temporary storage | The same Linux memory command |

`tests/test_invocation_lifetime.py` disables cyclic GC and checks success, failure,
optional channels, aliases, batches, volumes, Tensor inputs, retained sampling
views, and DataLoader workers. `tests/contracts/test_transform_lifetime.py` runs
every primary mode from the existing transform registry while keeping its
configured pipeline alive. Inputs and returned arrays must be released after
the caller discards them. Live outputs, replay/trace artifacts and exception
tracebacks remain legitimate owners and have separate lifetime tests.

Run the Linux checks with the normal CPU Torch environment:

```bash
uv run python tools/check_memory.py --output _internal/memory-budget.json
```

The command runs each workload three times in fresh processes with one compute
thread and writes every sample, source hash and dependency version. An incomplete
run fails. The JSON is included in CI evidence and the publishable release bundle.

The RSS cases reproduce 100 warmup and 800 measured calls on fixed or varying
float32 HW inputs, explicit invocation seeds, image plus mask, HWC1 inputs and
direct Blur. Automatic GC remains enabled; full collection occurs only at the
measurement endpoints. The measured loop performs no memory reads or logging.
Post-GC median growth is limited to 16 MiB, or 32 MiB for image plus mask. These
budgets allow approximately two maximum-sized sample buffers plus allocator
noise; the deterministic lifetime tests still require immediate release.
Control growth and excess relative to HWC1/direct Blur are reported alongside
the absolute growth budget. No `malloc_trim` is used.

The peak cases cover float32/float16 Blur, RGB normalization, batch normalization,
CPU Tensor normalization and volume padding. Backend initialization uses small
inputs before measurement. The worker resets its Linux RSS high-water mark after
imports and input setup, then reads the kernel peak after one public Compose call.
This uses the documented
[clear_refs reset](https://www.kernel.org/doc/html/v6.12/filesystems/proc.html).
The budget is two expected output buffers plus 8 MiB of fixed backend allowance;
output storage must also match the case's expected shape/dtype byte count.
This captures native allocations that Python-only allocation tracing can miss.

These are bounds for the registered workloads. Add a case when a new allocation
pattern, dtype conversion or target route needs its own budget. Keep ASV memory
comparisons as additional triage evidence; they do not replace the blocking
lifetime, repeated-call and working-peak checks.

## Local large-input evidence

For a PR that changes a hot path or makes a performance or memory claim, run
only the affected `512×512` and/or `1024×1024` catalog cases locally. Record the
command, filter, baseline SHA, candidate SHA, environment, and results in the PR
description. A benchmark without the exact baseline and candidate is diagnostic,
not a regression claim.

The `run-performance` label and manual Performance workflow exist for explicit
remote reproduction. They do not run on ordinary PRs and never expand to the
full catalog by default. A blank manual filter selects `release-core`; an explicit
filter selects a larger or changed-family investigation.

## Interpretation

ASV timing is sensitive to host load and environment. Compare the same route,
input, and execution mode on the same controlled environment before treating a
number as a regression. Peak-memory cases measure allocations from the public
`Compose` route; they do not establish a general memory bound for all inputs.

Do not use Codecov, test coverage percentage, a catalog-only report, or a
single-revision ASV result as performance evidence. They answer different
questions.
