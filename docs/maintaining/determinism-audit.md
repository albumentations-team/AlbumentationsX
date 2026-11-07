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

Raw generated reports and local logs belong outside Git. A milestone handoff should link the passing PR/CI run and
retain the environment, JUnit, and JSON evidence alongside the generated report. It must distinguish a local run
from hosted Linux/Windows/macOS CI results.
