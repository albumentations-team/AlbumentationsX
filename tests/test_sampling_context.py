"""Contract tests for explicit transform sampling state."""

from __future__ import annotations

import ast
import random
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import albumentations as A
import albumentations.core.invocation as invocation_module
from albumentations.core.invocation import SamplingContext
from albumentations.core.random_utils import _derive_invocation_seed
from albumentations.core.transform_params import SampledParams, TargetSet


def _core_modules() -> tuple[Path, ...]:
    core_root = Path(__file__).parents[1] / "albumentations" / "core"
    return core_root / "composition.py", core_root / "transforms_interface.py"


def test_augmentation_code_does_not_read_rng_from_transform_instances() -> None:
    """Keep all transform sampling on the explicit call-local SamplingContext API."""
    augmentation_root = Path(__file__).parents[1] / "albumentations" / "augmentations"
    violations: list[str] = []

    for source_path in sorted(augmentation_root.rglob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr in {"py_random", "random_generator"}
            ):
                violations.append(f"{source_path.relative_to(augmentation_root.parent)}:{node.lineno}:{node.attr}")

    assert violations == [], "Transform sampling must use SamplingContext:\n" + "\n".join(violations)


def test_execution_entry_points_require_explicit_invocations() -> None:
    """Keep configured graph dispatch explicit instead of falling back to dynamic execution state."""
    violations: list[str] = []
    removed_helpers = {"should_apply", "get_indices", "select_branch_index"}

    for source_path in _core_modules():
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name in removed_helpers:
                violations.append(f"{source_path.name}:{node.lineno}:{node.name} is obsolete")
            if node.name != "apply_in_invocation":
                continue

            invocation_arg = next(
                (argument for argument in (*node.args.posonlyargs, *node.args.args) if argument.arg == "invocation"),
                None,
            )
            if invocation_arg is None or ast.unparse(invocation_arg.annotation) != "InvocationContext":
                violations.append(f"{source_path.name}:{node.lineno}: invocation must be required")
            if any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "get_current_invocation"
                for child in ast.walk(node)
            ):
                violations.append(f"{source_path.name}:{node.lineno}: dynamic invocation fallback")

    assert violations == [], "Configured dispatch must stay explicit:\n" + "\n".join(violations)


@pytest.mark.parametrize("hook_name", ("get_params", "get_params_dependent_on_data"))
def test_declaring_a_removed_sampling_hook_fails_without_an_invocation(hook_name: str) -> None:
    """Fail when legacy code is defined rather than adding a branch to transform execution."""
    with pytest.raises(TypeError, match=rf"LegacyTransform defines removed sampling hook\(s\): {hook_name}"):
        type("LegacyTransform", (A.ImageOnlyTransform,), {hook_name: lambda self: {}})


class _DrawProbe(A.ImageOnlyTransform):
    """Expose draws to an independent generator oracle without storing input data."""

    def __init__(self) -> None:
        super().__init__(p=1)
        self.draws: list[tuple[float, float]] = []
        self.fail = False

    def sample_parameters(
        self, params: dict, data: dict, targets: TargetSet, sampling: SamplingContext
    ) -> SampledParams:
        python_value = sampling.py_random.random()
        numpy_value = float(sampling.random_generator.random())
        self.draws.append((python_value, numpy_value))
        if self.fail:
            raise RuntimeError("after sampler draws")
        return SampledParams(params={})

    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        return image


@pytest.mark.parametrize("invocation_seed", [None, 137])
@pytest.mark.parametrize(("probability", "force_apply"), [(1, False), (0.5, False), (0.5, True)])
def test_root_and_sampler_draws_match_independent_generators(
    invocation_seed: int | None,
    probability: float,
    force_apply: bool,
) -> None:
    probe = _DrawProbe()
    pipeline = A.Compose([probe], p=probability, seed=137)
    python_oracle, numpy_oracle = random.Random(137), np.random.default_rng(137)
    expected = []
    for _ in range(8):
        if invocation_seed is not None:
            python_oracle, numpy_oracle = random.Random(137), np.random.default_rng(137)
        applies = force_apply or probability == 1 or python_oracle.random() < probability
        if applies:
            expected.append((python_oracle.random(), float(numpy_oracle.random())))
        pipeline(image=np.ones((17, 19), np.float32), force_apply=force_apply, invocation_seed=invocation_seed)
    assert probe.draws == expected
    if invocation_seed is None:
        assert pipeline.py_random.getstate() == python_oracle.getstate()
        assert pipeline.random_generator.bit_generator.state == numpy_oracle.bit_generator.state
    else:
        assert pipeline.py_random.getstate() == random.Random(137).getstate()
        assert pipeline.random_generator.bit_generator.state == np.random.default_rng(137).bit_generator.state


@pytest.mark.parametrize("manual", [False, True])
def test_explicit_call_does_not_advance_configured_stream(manual: bool) -> None:
    probe = _DrawProbe()
    pipeline = A.Compose([probe], seed=137)
    if manual:
        pipeline.set_random_state(np.random.default_rng(137), random.Random(137))
    image = np.ones((17, 19), np.float32)
    pipeline(image=image)
    pipeline(image=image, invocation_seed=271)
    pipeline(image=image)
    python_oracle, numpy_oracle = random.Random(137), np.random.default_rng(137)
    expected = [(python_oracle.random(), float(numpy_oracle.random())) for _ in range(2)]
    assert [probe.draws[0], probe.draws[2]] == expected
    assert probe.draws[1] == (random.Random(271).random(), np.random.default_rng(271).random())


@pytest.mark.parametrize("failure", ["before_draw", "after_probability", "after_sampler"])
def test_errors_preserve_exact_draw_consumption_and_retry(failure: str) -> None:
    probe = _DrawProbe()
    probability = 0.5 if failure == "after_probability" else 1
    pipeline = A.Compose([probe], p=probability, seed=137, save_applied_params=True)
    image = np.ones((17, 19), np.float32)
    python_oracle, numpy_oracle = random.Random(137), np.random.default_rng(137)
    if failure == "before_draw":
        data = {"image": image, "frame_annotations": [{}]}
    elif failure == "after_probability":
        data = {"image": image, "mask": np.ones((2, 3), np.uint8)}
        assert python_oracle.random() < probability
    else:
        data = {"image": image}
        probe.fail = True
        python_oracle.random()
        numpy_oracle.random()
    with pytest.raises((ValueError, RuntimeError)):
        pipeline(**data)
    assert invocation_module.get_current_invocation() is None
    assert probe.get_applied_params() == {}
    assert pipeline.py_random.getstate() == python_oracle.getstate()
    assert pipeline.random_generator.bit_generator.state == numpy_oracle.bit_generator.state
    probe.fail = False
    applies = probability == 1 or python_oracle.random() < probability
    expected = (python_oracle.random(), float(numpy_oracle.random())) if applies else None
    before_retry = len(probe.draws)
    pipeline(image=image)
    assert probe.draws[before_retry:] == ([] if expected is None else [expected])
    assert pipeline.py_random.getstate() == python_oracle.getstate()
    assert pipeline.random_generator.bit_generator.state == numpy_oracle.bit_generator.state


@pytest.fixture
def generator_allocations(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    counts = {"python": 0, "numpy": 0}
    original_numpy = np.random.default_rng
    original_python = random.Random

    def numpy_factory(*args: Any, **kwargs: Any) -> np.random.Generator:
        counts["numpy"] += 1
        return original_numpy(*args, **kwargs)

    class CountedRandom(original_python):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            counts["python"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(np.random, "default_rng", numpy_factory)
    monkeypatch.setattr(random, "Random", CountedRandom)
    return counts


@pytest.mark.parametrize("access_order", [("python",), ("numpy",), ("python", "numpy"), ("numpy", "python")])
@pytest.mark.parametrize("factory", ["seed", "owner_cold", "owner_warm"])
def test_standalone_generators_keep_initialization_boundaries(
    factory: str,
    access_order: tuple[str, ...],
    generator_allocations: dict,
) -> None:
    owner = A.Blur(p=1)
    if factory == "owner_warm":
        owner.set_random_seed(137)
    before = generator_allocations.copy()
    sampling = SamplingContext.from_seed(137) if factory == "seed" else SamplingContext.from_owner(owner)
    assert generator_allocations == before
    for access in access_order:
        generator = sampling.py_random if access == "python" else sampling.random_generator
        assert generator is (sampling.py_random if access == "python" else sampling.random_generator)
        expected = (
            {
                "python": int("python" in access_order[: access_order.index(access) + 1]),
                "numpy": int("numpy" in access_order[: access_order.index(access) + 1]),
            }
            if factory == "seed"
            else {"python": 2, "numpy": 1}
            if factory == "owner_cold"
            else {"python": 0, "numpy": 0}
        )
        assert generator_allocations == {key: before[key] + expected[key] for key in before}
    if factory != "seed":
        again = SamplingContext.from_owner(owner)
        after = generator_allocations.copy()
        assert again.py_random is sampling.py_random
        assert again.random_generator is sampling.random_generator
        assert generator_allocations == after


@pytest.mark.parametrize("first_access", ["python", "numpy"])
def test_thread_stream_materialization_and_reuse(first_access: str, generator_allocations: dict) -> None:
    owner = A.Compose([], seed=137)
    before = generator_allocations.copy()

    def in_worker() -> None:
        sampling = SamplingContext.from_owner(owner)
        assert generator_allocations == before
        derived_seed = _derive_invocation_seed(
            base_seed=137,
            runtime_context=None,
            invocation_index=1,
            random_generator=None,
            py_random=None,
            manual=False,
        )
        # Construct the oracle before recording constructor deltas.
        python_oracle, numpy_oracle = random.Random(derived_seed), np.random.default_rng(derived_seed)
        initial = generator_allocations.copy()
        if first_access == "python":
            assert sampling.py_random.random() == python_oracle.random()
        else:
            assert sampling.random_generator.random() == numpy_oracle.random()
        assert generator_allocations == {key: initial[key] + int(key == first_access) for key in initial}
        assert sampling.py_random.random() == python_oracle.random()
        assert sampling.random_generator.random() == numpy_oracle.random()
        assert generator_allocations == {key: initial[key] + 1 for key in initial}
        again = SamplingContext.from_owner(owner)
        after = generator_allocations.copy()
        assert again.py_random is sampling.py_random
        assert again.random_generator is sampling.random_generator
        assert generator_allocations == after

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(in_worker).result(timeout=10)


@pytest.mark.parametrize("invocation_seed", [None, 137])
@pytest.mark.parametrize("prime", [False, True])
def test_invocation_and_sampling_share_generators_without_reallocation(
    invocation_seed: int | None,
    prime: bool,
    generator_allocations: dict,
) -> None:
    owner = A.Compose([], seed=137)
    before = generator_allocations.copy()
    context = owner._create_invocation_context(
        collect_applied=False,
        invocation_seed=invocation_seed,
        prime_py_random=prime,
    )
    sampling = context.sampling_context({})
    assert generator_allocations == {
        "python": before["python"] + int(invocation_seed is not None and prime),
        "numpy": before["numpy"],
    }
    assert sampling.py_random is context.py_random
    assert generator_allocations["numpy"] == before["numpy"]
    assert sampling.random_generator is context.random_generator
    assert generator_allocations == {key: before[key] + int(invocation_seed is not None) for key in before}
    assert context.sampling_context({}) is sampling


@pytest.mark.parametrize("route", ["root_zero", "leaf_zero", "noop"])
def test_deterministic_calls_do_not_materialize_rng(route: str, generator_allocations: dict) -> None:
    pipeline = A.Compose(
        [A.Blur(p=0) if route == "leaf_zero" else A.NoOp(p=1)],
        p=0 if route == "root_zero" else 1,
        seed=137,
    )
    before = generator_allocations.copy()
    for invocation_seed in (None, 137):
        pipeline(image=np.ones((17, 19), np.float32), invocation_seed=invocation_seed)
    assert generator_allocations == before


@pytest.mark.parametrize("invocation_seed", [None, 137])
@pytest.mark.parametrize("probability", [0.5, 1])
def test_probability_only_calls_do_not_allocate_sampler_state(
    invocation_seed: int | None, probability: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_state(*args: Any, **kwargs: Any) -> None:
        pytest.fail("probability-only execution allocated sampler state")

    monkeypatch.setattr(invocation_module, "_InvocationRandomState", unexpected_state, raising=False)
    pipeline = A.Compose([A.NoOp(p=0.5)], p=probability, seed=137)
    image = np.ones((17, 19), np.float32)
    for _ in range(100):
        assert pipeline(image=image, invocation_seed=invocation_seed)["image"] is image


@pytest.mark.parametrize("invocation_seed", [None, 137])
def test_root_streams_created_before_sampling_are_shared(
    invocation_seed: int | None, generator_allocations: dict
) -> None:
    context = A.Compose([], seed=137)._create_invocation_context(collect_applied=False, invocation_seed=invocation_seed)
    python_stream, numpy_stream = context.py_random, context.random_generator
    before = generator_allocations.copy()
    sampling = context.sampling_context({})
    assert sampling.py_random is python_stream
    assert sampling.random_generator is numpy_stream
    assert generator_allocations == before


@pytest.mark.parametrize("invocation_seed", [None, 137])
def test_numpy_only_sampler_does_not_create_isolated_python_rng(
    invocation_seed: int | None,
    generator_allocations: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def sample(
        transform: _DrawProbe, params: dict, data: dict, targets: TargetSet, sampling: SamplingContext
    ) -> SampledParams:
        sampling.random_generator.random()
        return SampledParams(params={})

    monkeypatch.setattr(_DrawProbe, "sample_parameters", sample)
    pipeline = A.Compose([_DrawProbe()], seed=137)
    before = generator_allocations.copy()
    pipeline(image=np.ones((17, 19), np.float32), invocation_seed=invocation_seed)
    assert generator_allocations == {
        "python": before["python"],
        "numpy": before["numpy"] + int(invocation_seed is not None),
    }


def test_direct_call_initializes_owner_pair_once(generator_allocations: dict) -> None:
    transform = A.Blur(p=1)
    assert generator_allocations == {"python": 0, "numpy": 0}
    transform(image=np.ones((17, 19, 1), np.float32))
    assert generator_allocations == {"python": 2, "numpy": 1}
    transform(image=np.ones((17, 19, 1), np.float32))
    assert generator_allocations == {"python": 2, "numpy": 1}


@pytest.mark.parametrize("observed", [False, True])
def test_sampling_view_reuse_and_override_isolation(observed: bool) -> None:
    visits = []
    original = A.Blur.sample_parameters

    def sample(
        transform: A.Blur, params: dict, data: dict, targets: TargetSet, sampling: SamplingContext
    ) -> SampledParams:
        if observed:
            assert sampling.applied_overrides == {}
        else:
            value = np.ones((2, 3))
            reference = weakref.ref(value)
            sampling.applied_overrides["discard_probe"] = value
            sampling.applied_overrides.update(discard_probe=value)
            del value
            assert reference() is None
        visits.append((id(sampling), sampling.applied_overrides))
        return original(transform, params, data, targets, sampling)

    pipeline = A.Compose([A.Blur(p=1), A.Blur(p=1)], seed=137, save_applied_params=observed)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(A.Blur, "sample_parameters", sample)
        first = pipeline(image=np.ones((17, 19), np.float32))
        if observed:
            records = first["applied_transforms"].copy()
        pipeline(image=np.ones((17, 19), np.float32))
    assert visits[0][0] == visits[1][0]
    assert visits[2][0] == visits[3][0]
    if observed:
        assert all(sink is not other for index, (_, sink) in enumerate(visits) for _, other in visits[index + 1 :])
        assert first["applied_transforms"] == records
        pipeline.p = 0
        pipeline(image=np.ones((17, 19), np.float32))
        assert all(transform.get_applied_config() == {} for transform in pipeline.transforms)
    else:
        assert all(sink is visits[0][1] for _, sink in visits)


def test_direct_sampler_from_seed_preserves_draws_and_overrides() -> None:
    transform = A.Blur(blur_range=(3, 7), p=1)
    image = np.ones((17, 19, 1), np.float32)
    sink = {}
    sampling = SamplingContext.from_seed(137, sink)
    sampled = transform.sample_parameters({}, {"image": image}, transform._build_target_set({"image": image}), sampling)
    oracle = random.Random(137)
    kernel = oracle.randrange(3, 8, 2)
    assert sampled.params == {"kernel": kernel}
    assert sampling.applied_overrides is sink
    assert sink
