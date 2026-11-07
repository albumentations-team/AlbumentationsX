# Determinism audit and evidence

The [public determinism and replay contract](../determinism.md) defines the supported guarantees. This audit maps
the composition implementation and permanent tests to [#491](https://github.com/albumentations-team/AlbumentationsX/issues/491).
It covers random sampling and sample identity; it does not promise cross-platform numerical equality for every kernel.

## Composition audit

Every configured child is dispatched through `BaseCompose._apply_child` with the enclosing `InvocationContext`.
Transform samplers receive its `SamplingContext`. No audited composition samples from a separate instance generator.
Standalone containers open a root invocation when there is no enclosing invocation.

| Public composition | Random decisions | Shared path |
| --- | --- | --- |
| `Compose` | Root probability and leaf sampling | `_execute_root`, `_apply_children` |
| `ReplayCompose` | Ordinary calls use Compose sources; replay uses stored decisions | `_execute_root`, `apply_in_invocation` |
| `OneOf` | Probability and weighted child choice | `apply_in_invocation` |
| `SomeOf` | Probability and child indices, including replacement | `apply_in_invocation`, `_get_indices` |
| `RandomOrder` | SomeOf selection with sampled execution order | Inherited `apply_in_invocation`, `_get_indices` |
| `OneOrOther` | First/second branch choice | `apply_in_invocation` |
| `Sequential` | Container and child probabilities | `apply_in_invocation` |
| `SelectiveChannelTransform` | Container and child probabilities for selected channels | `apply_in_invocation` |

Sources: [composition](../../albumentations/core/composition.py),
[invocation and sampling state](../../albumentations/core/invocation.py), and
[worker seed derivation](../../albumentations/core/random_utils.py).

## Permanent coverage

| Guarantee | Evidence |
| --- | --- |
| Fixed-seed sequences and ordinary/trace agreement for all eight nested composition types | `tests/property/test_determinism.py` |
| Sample keys survive reordering and concurrent calls for all eight types | `tests/property/test_determinism.py` |
| Explicit seeds leave the configured stream untouched; failures preserve exact draw consumption | `tests/test_sampling_context.py` |
| Concurrent execution isolates generators, observations, processors, and target restoration | `tests/test_compose_reentrancy.py` |
| Worker-count-independent samples, including persistent workers and epoch keys | `tests/test_dataloader_determinism.py` with real `spawn` workers at counts 0, 1, and 2 |
| Ordinary workers reproduce a fixed loader setup, advance across epochs, and differ when worker count changes | `tests/test_dataloader_determinism.py` |
| Pickle/unpickle rebuilds process-local RNG state | `tests/test_per_worker_seed.py` pickle contracts; spawned-worker tests |
| Constructor persistence, ReplayCompose, applied configurations, and tracing | `tests/contracts/test_composition_serialization_contracts.py`, `tests/test_composition_tracing.py`, and `tests/property/test_applied_config_replay.py` |
| Prohibited global sampling, instance RNG access, and sampler-created independent sources are rejected | `tests/test_ax_coding_guidance.py` negative controls for `AXG005` and `AXG027` |

The real-worker tests reproduce [#81](https://github.com/albumentations-team/AlbumentationsX/issues/81): ordinary
augmentations may change when worker count changes. Stable sample-keyed `invocation_seed` values reproduce the same
samples across worker counts, loader seeds, and persistence settings. The test includes epoch identity in requests
sent to the workers, so persistent dataset copies do not depend on parent-process attribute updates.

## CI and report evidence

The [PR workflow](../../.github/workflows/pr.yml) runs the base/property contracts in compatibility jobs and the
real-worker suite in its PyTorch job. The [nightly](../../.github/workflows/nightly.yml) and
[release-candidate](../../.github/workflows/release-candidate.yml) PyTorch jobs run the same worker suite.
The repository's `check-ax-rules` pre-commit hook enforces the RNG source checks.

Version-bump release preflight produces `junit-determinism-contract.xml` and summarizes it as
`determinism-contract.json`. The [Correctness & Compatibility Report](correctness-and-compatibility-report.md)
requires a nonempty passing determinism suite with no skipped tests. Its contract and audit links use the tested
source revision; environment artifacts identify the Python/dependency profile, commit, and CI run.

Run the report's determinism suite locally with:

```bash
uv run pytest -q \
  tests/test_sampling_context.py tests/test_compose_reentrancy.py \
  tests/test_composition_tracing.py tests/test_dataloader_determinism.py \
  tests/contracts/test_composition_serialization_contracts.py \
  tests/property/test_determinism.py tests/property/test_applied_config_replay.py \
  tests/test_per_worker_seed.py::test_unpickled_compose_resets_runtime_context \
  tests/test_per_worker_seed.py::test_unpickled_basic_transform_resets_runtime_context \
  --hypothesis-profile=ci-fast \
  --junitxml=_internal/determinism-contract.xml
```

## Verification record: 2026-11-07

Tested source: [e7db973](https://github.com/albumentations-team/AlbumentationsX/commit/e7db973df5d40963ea102b3510111d8d6822fe7a).
This record is added separately from the tested implementation. The local profile was macOS ARM64, Python 3.11.15,
NumPy 2.4.6, OpenCV 5.0.0.93, Albucore 0.3.5, and CPU PyTorch 2.15.0, installed from the project lockfile.

| Check | Local result |
| --- | --- |
| Determinism suite listed above, collected from the full, final focused, and real-worker JUnit results | 142 passed, 0 skipped |
| Full project suite, excluding the separately executed worker module | 25,107 passed, 34 skipped |
| Real spawned-worker suite | 5 passed, 0 skipped |
| Final focused guidance, report, routing, and generative checks | 116 passed |
| All six Python examples in the public contract | Passed, including the spawn example |
| All pre-commit hooks, including mypy and Pyrefly | Passed |
| CI matrix/shard consistency and workflow security audit | Passed |

[Hosted PR CI evidence](https://github.com/albumentations-team/AlbumentationsX/actions/workflows/pr.yml?query=branch%3Acodex%2Fdeterministic-pipeline-contract)
records the Linux, Windows, and macOS results separately from this local run. Raw generated reports, environment
metadata, JUnit, and JSON evidence belong outside Git; release preflight retains them with its generated report.
