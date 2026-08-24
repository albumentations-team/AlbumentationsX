"""Regression coverage for reentrant Compose execution on one configured pipeline."""

from __future__ import annotations

import random
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock
from typing import Any, TypeVar

import numpy as np
import pytest
import torch

import albumentations as A
import albumentations.core.composition as composition_module
import albumentations.core.invocation as invocation_module
from albumentations.core.invocation import SamplingContext
from albumentations.core.transform_params import TransformParameterPlan, TransformSamplingInput
from albumentations.core.transforms_interface import ImageOnlyTransform

T = TypeVar("T")


class _BlockingNumpyOnlyProbe(ImageOnlyTransform):
    """Hold one fallback call inside transform dispatch until the test releases it."""

    def __init__(self, marker_range: tuple[int, int] = (0, 0)) -> None:
        super().__init__(p=1.0)
        self.marker_range = marker_range
        self.first_entered = Event()
        self.release_first = Event()
        self.second_entered = Event()

    def sample_parameters(
        self,
        inputs: TransformSamplingInput,
        sampling: SamplingContext,
    ) -> TransformParameterPlan:
        marker = int(inputs.data["image"][0, 0, 0])
        sampling.applied_overrides["marker_range"] = (marker, marker)
        return TransformParameterPlan.shared_only({"marker": marker})

    def apply(self, image: np.ndarray, marker: int, **params: Any) -> np.ndarray:
        if marker == 1:
            self.first_entered.set()
            assert self.release_first.wait(timeout=5)
        else:
            self.second_entered.set()
        return image


class _BlockingRandomProbe(_BlockingNumpyOnlyProbe):
    """Record the active Python stream while two calls overlap."""

    def __init__(self) -> None:
        super().__init__()
        self.py_random_ids: list[int] = []
        self._ids_lock = Lock()

    def sample_parameters(
        self,
        inputs: TransformSamplingInput,
        sampling: SamplingContext,
    ) -> TransformParameterPlan:
        with self._ids_lock:
            self.py_random_ids.append(id(sampling.py_random))
        return super().sample_parameters(inputs, sampling)


class _NumpyRandomMarker(ImageOnlyTransform):
    """Expose one NumPy-sampled value through the image for worker-stream tests."""

    def sample_parameters(
        self,
        inputs: TransformSamplingInput,
        sampling: SamplingContext,
    ) -> TransformParameterPlan:
        del inputs
        return TransformParameterPlan.shared_only(
            {"marker": int(sampling.random_generator.integers(np.iinfo(np.int64).max, dtype=np.int64))},
        )

    def apply(self, image: np.ndarray, marker: int, **params: Any) -> np.ndarray:
        del params
        return np.full_like(image, marker)


class _BlockingDeterministicFlip(A.HorizontalFlip):
    """Force two context-free deterministic direct calls to overlap."""

    def __init__(self) -> None:
        super().__init__(p=1.0)
        self.first_entered = Event()
        self.release_first = Event()
        self.second_entered = Event()

    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        if image.shape[0] == 5:
            self.first_entered.set()
            assert self.release_first.wait(timeout=5)
        else:
            self.second_entered.set()
        return super().apply(image, **params)


class _ExplicitExternalSampler(ImageOnlyTransform):
    """Use the supported sampling contract from outside the built-in augmentation package."""

    def __init__(self) -> None:
        super().__init__(p=1.0)

    def sample_parameters(
        self,
        inputs: TransformSamplingInput,
        sampling: SamplingContext,
    ) -> TransformParameterPlan:
        del inputs
        return TransformParameterPlan.shared_only({"offset": sampling.py_random.randint(1, 7)})

    def apply(self, image: np.ndarray, offset: int, **params: Any) -> np.ndarray:
        del params
        return image + np.uint8(offset)


class _ContextOnlyRootProbabilityCompose(A.Compose):
    """Fail if a root probability decision reads the configured object directly."""

    def should_apply(self, force_apply: bool = False) -> bool:
        msg = "root probability must use its invocation context"
        raise AssertionError(msg)


class _CallIndependentCompose(ImageOnlyTransform):
    """Call a separately configured public Compose from an outer transform."""

    def __init__(self, inner: A.Compose) -> None:
        super().__init__(p=1.0)
        self.inner = inner

    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        return self.inner(image=image)["image"]


class _CallIndependentTraceCompose(_CallIndependentCompose):
    """Exercise the traced public entry point from an outer transform."""

    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        return self.inner.run_with_trace(image=image).data["image"]


class _FailAfterSampling(ImageOnlyTransform):
    """Raise after sampling so a failed invocation cannot publish partial state."""

    def __init__(self, marker: int = 137) -> None:
        super().__init__(p=1.0)
        self.marker = marker

    def sample_parameters(
        self,
        inputs: TransformSamplingInput,
        sampling: SamplingContext,
    ) -> TransformParameterPlan:
        del inputs
        sampling.applied_overrides["marker"] = self.marker
        return TransformParameterPlan.shared_only({"marker": self.marker})

    def apply(self, image: np.ndarray, marker: int, **params: Any) -> np.ndarray:
        del image, marker, params
        raise RuntimeError("intentional execution failure")


def _wait(event: Event) -> None:
    assert event.wait(timeout=5)


def _submit(
    executor: ThreadPoolExecutor,
    function: Callable[[], T],
) -> Future[T]:
    return executor.submit(function)


def test_compose_overlaps_tensor_fallback_calls_and_keeps_applied_params_isolated() -> None:
    probe = _BlockingNumpyOnlyProbe()
    compose = A.Compose([probe], save_applied_params=True, strict=True)

    first_image = torch.full((3, 5, 7), 1, dtype=torch.uint8)
    first_mask = torch.full((5, 7), 11, dtype=torch.uint8)
    second_image = torch.full((3, 5, 7), 2, dtype=torch.uint8)

    def run_first() -> tuple[dict[str, Any], dict[str, Any]]:
        result = compose(image=first_image, mask=first_mask)
        return result, probe.applied_config

    def run_second() -> tuple[dict[str, Any], dict[str, Any]]:
        result = compose(image=second_image)
        return result, probe.applied_config

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = _submit(executor, run_first)
        _wait(probe.first_entered)
        second_future = _submit(executor, run_second)
        _wait(probe.second_entered)

        probe.release_first.set()
        (first_result, first_observation) = first_future.result(timeout=5)
        (second_result, second_observation) = second_future.result(timeout=5)

    assert isinstance(first_result["image"], torch.Tensor)
    assert isinstance(first_result["mask"], torch.Tensor)
    assert isinstance(second_result["image"], torch.Tensor)
    torch.testing.assert_close(first_result["image"], first_image)
    torch.testing.assert_close(first_result["mask"], first_mask)
    torch.testing.assert_close(second_result["image"], second_image)
    assert first_result["applied_transforms"][0][1] == {"marker_range": (1, 1)}
    assert second_result["applied_transforms"][0][1] == {"marker_range": (2, 2)}
    assert first_observation == {"marker_range": (1, 1)}
    assert second_observation == {"marker_range": (2, 2)}


def test_plain_compose_discards_child_observations() -> None:
    transform = A.HorizontalFlip(p=1.0)
    transform(image=np.zeros((5, 7, 3), dtype=np.uint8))
    assert transform.get_applied_params()

    A.Compose([transform])(image=np.zeros((5, 7, 3), dtype=np.uint8))

    assert transform.get_applied_params() == {}
    assert transform.get_applied_config() == {}


def test_all_skipped_compose_discards_child_observations_without_a_context() -> None:
    transform = A.HorizontalFlip(p=1.0)
    transform(image=np.zeros((5, 7, 3), dtype=np.uint8))
    assert transform.get_applied_params()
    transform.p = 0.0

    A.Compose([transform])(image=np.zeros((5, 7, 3), dtype=np.uint8))

    assert transform.get_applied_params() == {}
    assert transform.get_applied_config() == {}


def test_deterministic_direct_transform_keeps_caller_local_observation() -> None:
    transform = _BlockingDeterministicFlip()
    first_image = np.full((5, 7, 3), 1, dtype=np.uint8)
    second_image = np.full((6, 7, 3), 2, dtype=np.uint8)

    def run(image: np.ndarray) -> tuple[tuple[int, ...], np.ndarray]:
        result = transform(image=image)
        return transform.get_applied_params()["shared"]["shape"], result["image"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = _submit(executor, lambda: run(first_image))
        _wait(transform.first_entered)
        second_future = _submit(executor, lambda: run(second_image))
        _wait(transform.second_entered)
        transform.release_first.set()
        first_params, first_result = first_future.result(timeout=5)
        second_params, second_result = second_future.result(timeout=5)

    assert first_params == first_image.shape
    assert second_params == second_image.shape
    np.testing.assert_array_equal(first_result, np.flip(first_image, axis=1))
    np.testing.assert_array_equal(second_result, np.flip(second_image, axis=1))


def test_compose_concurrent_threads_have_private_random_streams() -> None:
    probe = _BlockingRandomProbe()
    compose = A.Compose([probe], seed=137)
    first_image = np.full((5, 7, 3), 1, dtype=np.uint8)
    second_image = np.full((5, 7, 3), 2, dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = _submit(executor, lambda: compose(image=first_image))
        _wait(probe.first_entered)
        second_future = _submit(executor, lambda: compose(image=second_image))
        _wait(probe.second_entered)

        probe.release_first.set()
        first_future.result(timeout=5)
        second_future.result(timeout=5)

    assert len(probe.py_random_ids) == 2
    assert len(set(probe.py_random_ids)) == 2


def test_manual_numpy_random_state_seeds_worker_streams() -> None:
    """A worker must preserve the manual NumPy source instead of deriving both streams from Python RNG state."""
    image = np.zeros((1, 1, 1), dtype=np.int64)

    def make_compose(numpy_seed: int) -> A.Compose:
        compose = A.Compose([_NumpyRandomMarker(p=1.0)])
        compose.set_random_state(np.random.default_rng(numpy_seed), random.Random(137))
        return compose

    first = make_compose(11)
    second = make_compose(29)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first_marker = int(executor.submit(lambda: first(image=image)["image"][0, 0, 0]).result(timeout=5))
        second_marker = int(executor.submit(lambda: second(image=image)["image"][0, 0, 0]).result(timeout=5))

    assert first_marker != second_marker


def test_root_probability_uses_the_invocation_random_stream() -> None:
    compose = _ContextOnlyRootProbabilityCompose([A.NoOp(p=1.0)], p=0.5, seed=137)

    result = compose(image=np.zeros((5, 7, 3), dtype=np.uint8))

    assert result["image"].shape == (5, 7, 3)


def test_unactivated_compose_preserves_shape_validation_for_explicit_channel_targets() -> None:
    """Mismatched explicit-channel targets must not bypass the root shape contract."""
    compose = A.Compose([A.NoOp(p=1.0)])

    with pytest.raises(ValueError, match="Height and Width of image, mask or masks should be equal"):
        compose(
            image=np.zeros((5, 7, 3), dtype=np.uint8),
            mask=np.zeros((4, 7, 1), dtype=np.uint8),
        )


def test_public_compose_called_inside_a_transform_opens_an_independent_root() -> None:
    image = np.full((7, 9, 3), 127, dtype=np.uint8)
    inner = A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.2, 0.2), contrast_limit=(0, 0), p=1.0)], seed=137)
    outer = A.Compose([A.NoOp(p=0.5), _CallIndependentCompose(inner)], seed=271)

    expected = A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=(0.2, 0.2), contrast_limit=(0, 0), p=1.0)],
        seed=137,
    )(image=image)["image"]
    actual = outer(image=image)["image"]

    np.testing.assert_array_equal(actual, expected)


def test_deterministic_public_compose_called_inside_a_traced_transform_is_independent() -> None:
    """A public inner root must not inherit the outer trace through the ordinary executor."""
    image = np.full((7, 9, 3), 127, dtype=np.uint8)
    inner = A.Compose([A.HorizontalFlip(p=1.0)])
    outer = A.Compose([_CallIndependentCompose(inner)])

    traced = outer.run_with_trace(image=image)

    assert [record.node_path for record in traced.records if record.node_kind == "leaf"] == [(0,)]


def test_compose_initializes_its_root_random_source_at_construction() -> None:
    """A configured root has no first-call RNG initialization work to race."""
    compose = A.Compose([A.NoOp(p=1.0)], seed=137)
    assert hasattr(compose, "_seed_lock")
    assert compose._rng_initialized

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            invocation_module,
            "_get_runtime_rng_context",
            lambda _seed: pytest.fail("Compose must not initialize root RNG during a call"),
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = _submit(executor, compose._ensure_configured_random_sources)
            second = _submit(executor, compose._ensure_configured_random_sources)
            first.result(timeout=5)
            second.result(timeout=5)


@pytest.mark.parametrize(
    "container",
    [
        A.Compose([A.NoOp(p=1.0)]),
        A.OneOf([A.NoOp(p=1.0)], p=1.0),
        A.SomeOf([A.NoOp(p=1.0)], n=1, p=1.0),
        A.RandomOrder([A.NoOp(p=1.0)], n=1, p=1.0),
        A.OneOrOther(A.NoOp(p=1.0), A.NoOp(p=1.0), p=1.0),
        A.SelectiveChannelTransform([A.NoOp(p=1.0)], channels=(0, 1, 2), p=1.0),
        A.Sequential([A.NoOp(p=1.0)], p=1.0),
    ],
)
def test_configured_containers_receive_invocation_without_contextvar_lookup(
    container: A.BaseCompose,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured edges use their explicit invocation rather than rediscovering it."""
    invocation = container._new_invocation_context()
    image = np.zeros((5, 7, 3), dtype=np.uint8)

    with invocation:
        monkeypatch.setattr(
            composition_module,
            "get_current_invocation",
            lambda: pytest.fail("configured container performed a ContextVar lookup"),
        )
        result = container.apply_in_invocation(invocation, force_apply=False, image=image)

    np.testing.assert_array_equal(result["image"], image)


def test_unactivated_compiled_graph_matches_observed_container_execution() -> None:
    """The ordinary recursive loop preserves selector and leaf decisions exactly."""

    def make_compose(*, save_applied_params: bool) -> A.Compose:
        return A.Compose(
            [
                A.OneOf([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], p=1.0),
                A.SomeOf([A.NoOp(p=0.5), A.RandomRotate90(p=1.0)], n=1, p=1.0),
                A.RandomOrder([A.NoOp(p=0.5), A.NoOp(p=1.0)], n=1, p=1.0),
                A.OneOrOther(A.NoOp(p=1.0), A.Transpose(p=1.0), p=0.5),
                A.Sequential([A.RandomBrightnessContrast(p=0.5), A.NoOp(p=1.0)], p=1.0),
            ],
            p=0.75,
            seed=137,
            save_applied_params=save_applied_params,
            strict=True,
        )

    image = np.arange(17 * 19 * 3, dtype=np.uint8).reshape(17, 19, 3)
    ordinary = make_compose(save_applied_params=False)
    observed = make_compose(save_applied_params=True)

    assert ordinary._unactivated_compiled_graph
    ordinary_result = ordinary(image=image.copy())
    observed_result = observed(image=image.copy())

    np.testing.assert_array_equal(ordinary_result["image"], observed_result["image"])


def test_external_transform_using_sampling_context_uses_the_ordinary_executor() -> None:
    """A current custom transform receives the same fast route as built-in transforms."""
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    ordinary = A.Compose([_ExplicitExternalSampler()], seed=137, strict=True)
    observed = A.Compose([_ExplicitExternalSampler()], seed=137, save_applied_params=True, strict=True)

    assert ordinary._unactivated_compiled_graph
    ordinary_result = ordinary(image=image.copy())
    observed_result = observed(image=image.copy())

    np.testing.assert_array_equal(ordinary_result["image"], observed_result["image"])


def test_public_trace_called_inside_a_transform_opens_an_independent_root() -> None:
    image = np.full((7, 9, 3), 127, dtype=np.uint8)
    inner = A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.2, 0.2), contrast_limit=(0, 0), p=1.0)], seed=137)
    outer = A.Compose([A.NoOp(p=0.5), _CallIndependentTraceCompose(inner)], seed=271)

    expected = A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=(0.2, 0.2), contrast_limit=(0, 0), p=1.0)],
        seed=137,
    )(image=image)["image"]
    actual = outer(image=image)["image"]

    np.testing.assert_array_equal(actual, expected)


def test_failed_compose_invocation_does_not_publish_partial_observation() -> None:
    transform = _FailAfterSampling()

    with pytest.raises(RuntimeError, match="intentional execution failure"):
        A.Compose([transform], save_applied_params=True)(image=np.zeros((5, 7, 3), dtype=np.uint8))

    assert transform.get_applied_params() == {}
    assert transform.get_applied_config() == {}


def test_input_validation_failure_clears_previous_observation() -> None:
    transform = A.RandomBrightnessContrast(p=1.0)
    compose = A.Compose(
        [transform],
        additional_targets={"image2": "image"},
        save_applied_params=True,
    )
    image = np.zeros((5, 7, 3), dtype=np.uint8)

    compose(image=image)
    assert transform.get_applied_config()

    with pytest.raises(ValueError, match="requires canonical target"):
        compose(image2=image)

    assert transform.get_applied_params() == {}
    assert transform.get_applied_config() == {}


def test_root_probability_skip_clears_previous_observation() -> None:
    transform = A.HorizontalFlip(p=1.0)
    transform(image=np.zeros((5, 7, 3), dtype=np.uint8))
    assert transform.get_applied_params()

    A.Compose([transform], p=0.0)(image=np.zeros((5, 7, 3), dtype=np.uint8))

    assert transform.get_applied_params() == {}


def test_unactivated_compose_does_not_publish_leaf_observation() -> None:
    """An unactivated ordinary root keeps Compose calls from exposing child-local state."""
    transform = A.NoOp(p=1.0)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    transform(image=image)
    assert transform.get_applied_params()

    A.Compose([transform], p=1.0)(image=image)

    assert transform.get_applied_params() == {}


def test_traced_compose_overlaps_tensor_fallback_calls() -> None:
    probe = _BlockingNumpyOnlyProbe()
    compose = A.Compose([probe], strict=True)

    first_image = torch.full((3, 5, 7), 1, dtype=torch.uint8)
    first_mask = torch.full((5, 7), 11, dtype=torch.uint8)
    second_image = torch.full((3, 5, 7), 2, dtype=torch.uint8)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = _submit(executor, lambda: compose.run_with_trace(image=first_image, mask=first_mask).data)
        _wait(probe.first_entered)
        second_future = _submit(executor, lambda: compose.run_with_trace(image=second_image).data)
        _wait(probe.second_entered)

        probe.release_first.set()
        first_result = first_future.result(timeout=5)
        second_result = second_future.result(timeout=5)

    assert isinstance(first_result["image"], torch.Tensor)
    assert isinstance(first_result["mask"], torch.Tensor)
    assert isinstance(second_result["image"], torch.Tensor)
    torch.testing.assert_close(first_result["image"], first_image)
    torch.testing.assert_close(first_result["mask"], first_mask)
    torch.testing.assert_close(second_result["image"], second_image)


def test_traced_parent_and_direct_nested_compose_calls_overlap() -> None:
    probe = _BlockingNumpyOnlyProbe()
    nested = A.Compose([probe], strict=True)
    compose = A.Compose([nested], strict=True)
    first_image = np.full((5, 7, 3), 1, dtype=np.uint8)
    second_image = np.full((5, 7, 3), 2, dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = _submit(executor, lambda: compose.run_with_trace(image=first_image).data)
        _wait(probe.first_entered)
        second_future = _submit(executor, lambda: nested.run_with_trace(image=second_image).data)
        _wait(probe.second_entered)

        probe.release_first.set()
        first_result = first_future.result(timeout=5)
        second_result = second_future.result(timeout=5)

    np.testing.assert_array_equal(first_result["image"], first_image)
    np.testing.assert_array_equal(second_result["image"], second_image)


def test_compose_overlaps_grayscale_and_instance_binding_bookkeeping() -> None:
    probe = _BlockingNumpyOnlyProbe()
    compose = A.Compose(
        [probe],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        instance_binding=("masks", "bboxes"),
        strict=True,
    )

    first_image = np.full((5, 7), 1, dtype=np.uint8)
    second_image = np.full((5, 7), 2, dtype=np.uint8)
    first_instances = [
        {
            "mask": np.full((5, 7), 3, dtype=np.uint8),
            "bbox": np.array([0, 0, 7, 5], dtype=np.float32),
        },
        {
            "mask": np.full((5, 7), 4, dtype=np.uint8),
            "bbox": np.array([0, 0, 7, 5], dtype=np.float32),
        },
    ]
    second_instances = [
        {
            "mask": np.full((5, 7), 5, dtype=np.uint8),
            "bbox": np.array([0, 0, 7, 5], dtype=np.float32),
        },
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = _submit(executor, lambda: compose(image=first_image, instances=first_instances))
        _wait(probe.first_entered)
        second_future = _submit(executor, lambda: compose(image=second_image, instances=second_instances))
        _wait(probe.second_entered)

        probe.release_first.set()
        first_result = first_future.result(timeout=5)
        second_result = second_future.result(timeout=5)

    assert first_result["image"].shape == first_image.shape
    assert second_result["image"].shape == second_image.shape
    assert len(first_result["instances"]) == 2
    assert len(second_result["instances"]) == 1
    for actual, expected in zip(first_result["instances"], first_instances, strict=True):
        np.testing.assert_array_equal(actual["mask"], expected["mask"])
        np.testing.assert_array_equal(actual["bbox"], expected["bbox"])
    np.testing.assert_array_equal(second_result["instances"][0]["mask"], second_instances[0]["mask"])
    np.testing.assert_array_equal(second_result["instances"][0]["bbox"], second_instances[0]["bbox"])


def test_compose_uses_a_fresh_label_processor_session_per_call() -> None:
    probe = _BlockingNumpyOnlyProbe()
    compose = A.Compose(
        [probe],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
        strict=True,
    )
    image = np.full((5, 7, 3), 1, dtype=np.uint8)
    bboxes = np.array([[0, 0, 7, 5]], dtype=np.float32)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = _submit(executor, lambda: compose(image=image, bboxes=bboxes, labels=["cat"]))
        _wait(probe.first_entered)
        second_future = _submit(
            executor,
            lambda: compose(image=np.full_like(image, 2), bboxes=bboxes, labels=["dog"]),
        )
        _wait(probe.second_entered)

        probe.release_first.set()
        first_result = first_future.result(timeout=5)
        second_result = second_future.result(timeout=5)

    assert first_result["labels"] == ["cat"]
    assert second_result["labels"] == ["dog"]
    assert compose._configured_processors["bboxes"].label_manager.metadata == {}
