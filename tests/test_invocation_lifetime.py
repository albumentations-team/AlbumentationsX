"""Completed calls release temporary buffers and contexts without cyclic collection."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

import numpy as np
import pytest
import torch

import albumentations as A
import albumentations.core.composition as composition_module
import albumentations.core.invocation as invocation_module
from albumentations.core.invocation import SamplingContext
from albumentations.core.transform_params import SampledParams, TargetSet


def _watch(value: Any, references: list[weakref.ReferenceType[Any]]) -> None:
    if isinstance(value, (np.ndarray, torch.Tensor)):
        references.append(weakref.ref(value))
        base = value.base if isinstance(value, np.ndarray) else value._base
        if base is not None:
            _watch(base, references)
    elif isinstance(value, dict):
        for item in value.values():
            _watch(item, references)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _watch(item, references)


def _assert_released(references: list[weakref.ReferenceType[Any]]) -> None:
    assert references
    assert not gc.isenabled()
    assert all(reference() is None for reference in references)
    assert invocation_module.get_current_invocation() is None


@pytest.fixture
def refcount_only() -> Iterator[None]:
    """Guard the oracle against both automatic and explicit production collection."""
    was_enabled = gc.isenabled()
    invocation_module.clear_completed_observation()
    gc.collect()
    gc.disable()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(gc, "collect", lambda *args, **kwargs: pytest.fail("collection inside lifetime oracle"))
            yield
            assert not gc.isenabled()
    finally:
        invocation_module.clear_completed_observation()
        if was_enabled:
            gc.enable()
        else:
            gc.disable()
        gc.collect()


@pytest.fixture
def temporary_references(monkeypatch: pytest.MonkeyPatch) -> list[weakref.ReferenceType[Any]]:
    """Observe contexts, restoration values, sampler inputs and array parameters without owning them."""
    references: list[weakref.ReferenceType[Any]] = []
    original_context = invocation_module.InvocationContext
    original_restoration = composition_module.ChannelRestorationState
    original_sample = A.BasicTransform._sample_parameters

    class WatchedInvocation(original_context):
        __slots__ = ("__weakref__",)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            references.append(weakref.ref(self))

    def restoration(*args: Any, **kwargs: Any) -> Any:
        state = original_restoration(*args, **kwargs)
        _watch(state.original_value, references)
        _watch(state.normalized_value, references)
        return state

    def sample(transform: A.BasicTransform, *args: Any, **kwargs: Any) -> tuple[Any, SampledParams]:
        _watch(args, references)
        _watch(kwargs, references)
        sink, sampled = original_sample(transform, *args, **kwargs)
        _watch(sampled.params, references)
        for target_params in sampled.target_params:
            _watch(target_params.params, references)
        return sink, sampled

    monkeypatch.setattr(invocation_module, "InvocationContext", WatchedInvocation)
    original_random_state = getattr(invocation_module, "_InvocationRandomState", None)
    if original_random_state is not None:

        class WatchedRandomState(original_random_state):
            __slots__ = ("__weakref__",)

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                references.append(weakref.ref(self))

        monkeypatch.setattr(invocation_module, "_InvocationRandomState", WatchedRandomState)
    monkeypatch.setattr(composition_module, "ChannelRestorationState", restoration)
    monkeypatch.setattr(A.BasicTransform, "_sample_parameters", sample)
    return references


def _call_target(
    pipeline: Any,
    target: str = "image",
    shape: tuple[int, ...] = (17, 19),
    *,
    tensor: bool = False,
    support_tensor: bool = False,
    invocation_seed: int | None = None,
    dtype: Any = np.float32,
) -> list[weakref.ReferenceType[Any]]:
    data: dict[str, Any] = {target: np.full(shape, 1, dtype=dtype)}
    if tensor:
        data[target] = torch.from_numpy(data[target])
    if target in {"mask", "masks"}:
        data["image"] = torch.ones((1, 17, 19)) if support_tensor else np.ones((17, 19, 1), np.float32)
    elif target in {"mask3d", "masks3d"}:
        data["volume"] = np.ones((3, 17, 19, 1), np.float32)
    references: list[weakref.ReferenceType[Any]] = []
    _watch(data, references)
    result = pipeline(
        **data, **({} if isinstance(pipeline, A.BasicTransform) else {"invocation_seed": invocation_seed})
    )
    output = result[target]
    assert type(output) is type(data[target])
    assert output.shape == shape
    assert output.dtype == data[target].dtype
    if tensor:
        assert torch.equal(output, torch.ones_like(output))
        assert torch.equal(data[target], torch.ones_like(data[target]))
    else:
        np.testing.assert_array_equal(output, np.ones(shape, dtype=dtype))
        np.testing.assert_array_equal(data[target], np.ones(shape, dtype=dtype))
    return references


@pytest.mark.usefixtures("refcount_only")
@pytest.mark.parametrize("invocation_seed", [None, 137])
@pytest.mark.parametrize(
    ("target", "shape"),
    [
        ("image", (17, 19)),
        ("images", (2, 17, 19)),
        ("mask", (17, 19)),
        ("masks", (2, 17, 19)),
        ("volume", (3, 17, 19)),
        ("volumes", (2, 3, 17, 19)),
        ("mask3d", (3, 17, 19)),
        ("masks3d", (2, 3, 17, 19)),
    ],
)
def test_numpy_optional_channels_release_buffers(
    target: str,
    shape: tuple[int, ...],
    invocation_seed: int | None,
    temporary_references: list,
) -> None:
    pipeline = A.Compose([A.Blur(p=1)], seed=137)
    references = _call_target(pipeline, target, shape, invocation_seed=invocation_seed)
    _assert_released(references + temporary_references)


@pytest.mark.usefixtures("refcount_only")
@pytest.mark.parametrize("invocation_seed", [None, 137])
@pytest.mark.parametrize(
    ("target", "shape", "tensor", "support_tensor"),
    [
        ("image", (17, 19), True, False),
        ("mask", (17, 19), True, False),
        ("mask", (17, 19), False, True),
        ("masks", (2, 17, 19), True, True),
        ("mask3d", (3, 17, 19), True, False),
        ("masks3d", (2, 3, 17, 19), True, False),
    ],
)
def test_tensor_optional_channels_release_buffers(
    target: str,
    shape: tuple[int, ...],
    tensor: bool,
    support_tensor: bool,
    invocation_seed: int | None,
    temporary_references: list,
) -> None:
    pipeline = A.Compose([A.Blur(p=1)], seed=137)
    references = _call_target(
        pipeline,
        target,
        shape,
        tensor=tensor,
        support_tensor=support_tensor,
        invocation_seed=invocation_seed,
    )
    _assert_released(references + temporary_references)


@pytest.mark.usefixtures("refcount_only")
@pytest.mark.parametrize("invocation_seed", [None, 137])
@pytest.mark.parametrize(
    ("direct", "tensor", "shape"),
    [
        (False, False, (17, 19, 1)),
        (False, False, (17, 19, 3)),
        (False, True, (1, 17, 19)),
        (True, False, (17, 19)),
        (True, False, (17, 19, 1)),
    ],
)
def test_explicit_channel_and_direct_routes_release_contexts(
    direct: bool,
    tensor: bool,
    shape: tuple[int, ...],
    invocation_seed: int | None,
    temporary_references: list,
) -> None:
    pipeline = A.Blur(p=1) if direct else A.Compose([A.Blur(p=1)], seed=137)
    references = _call_target(pipeline, shape=shape, tensor=tensor, invocation_seed=invocation_seed)
    invocation_module.clear_completed_observation()
    assert temporary_references
    _assert_released(references + temporary_references)


@pytest.mark.usefixtures("refcount_only")
@pytest.mark.parametrize("invocation_seed", [None, 137])
def test_primed_probability_releases_buffers(invocation_seed: int | None, temporary_references: list) -> None:
    pipeline = A.Compose([A.Blur(p=1)], p=0.5, seed=137)
    _assert_released(_call_target(pipeline, invocation_seed=invocation_seed) + temporary_references)


@pytest.mark.usefixtures("refcount_only")
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("shape", [(1, 19), (17, 1), (17, 19, 1), (17, 19, 3)])
def test_degenerate_and_explicit_shapes_preserve_values(dtype: Any, shape: tuple[int, ...]) -> None:
    _assert_released(_call_target(A.Compose([A.Blur(p=1)], seed=137), shape=shape, dtype=dtype))


@pytest.mark.usefixtures("refcount_only")
def test_int16_mask_and_additional_target_release(temporary_references: list) -> None:
    def call() -> list:
        image = np.ones((17, 19), np.float32)
        alias = np.full((17, 19), 0.5, np.float32)
        mask = np.full((17, 19), 137, np.int16)
        result = A.Compose([A.Blur(p=1)], additional_targets={"other": "image"}, seed=137)(
            image=image,
            other=alias,
            mask=mask,
        )
        np.testing.assert_array_equal(result["image"], image)
        np.testing.assert_array_equal(result["other"], alias)
        assert result["mask"] is mask
        return [weakref.ref(image), weakref.ref(alias), weakref.ref(mask)]

    _assert_released(call() + temporary_references)


@pytest.mark.usefixtures("refcount_only")
@pytest.mark.parametrize("layout", ["readonly", "strided"])
@pytest.mark.parametrize("transform", ["blur", "noop"])
def test_storage_owners_and_internal_copies_release(layout: str, transform: str, temporary_references: list) -> None:
    def call() -> list:
        owner = np.ones((34, 38), np.float32) if layout == "strided" else np.ones((17, 19), np.float32)
        image = owner[::2, ::2] if layout == "strided" else owner
        if layout == "readonly":
            image.flags.writeable = False
        pipeline = A.Compose([A.Blur(p=1) if transform == "blur" else A.NoOp(p=1)], seed=137)
        result = pipeline(image=image)
        np.testing.assert_array_equal(result["image"], np.ones((17, 19), np.float32))
        np.testing.assert_array_equal(owner, np.ones_like(owner))
        if transform == "noop" and layout == "readonly":
            assert result["image"] is image
        if layout == "strided":
            # Root preprocessing already copies strided inputs on the baseline.
            assert not np.shares_memory(result["image"], owner)
        return [weakref.ref(owner), weakref.ref(image)]

    _assert_released(call() + temporary_references)


class _NumpyOnlySampler(A.ImageOnlyTransform):
    def sample_parameters(
        self, params: dict, data: dict, targets: TargetSet, sampling: SamplingContext
    ) -> SampledParams:
        return SampledParams(params={"sample": sampling.random_generator.random((2, 3))})

    def apply(self, image: np.ndarray, sample: np.ndarray, **params: Any) -> np.ndarray:
        assert sample.shape == (2, 3)
        return image


@pytest.mark.usefixtures("refcount_only")
@pytest.mark.parametrize("kind", ["blur", "noise", "crop", "numpy"])
@pytest.mark.parametrize("invocation_seed", [None, 137])
def test_scalar_array_and_geometry_samplers_release(
    kind: str,
    invocation_seed: int | None,
    temporary_references: list,
) -> None:
    def call() -> list:
        transforms = {
            "blur": lambda: A.Blur(p=1),
            "noise": lambda: A.GaussNoise(p=1),
            "crop": lambda: A.RandomCrop(height=11, width=13, p=1),
            "numpy": lambda: _NumpyOnlySampler(p=1),
        }
        image = np.full((17, 19), 0.5, np.float32)
        result = A.Compose([transforms[kind]()], seed=137)(image=image, invocation_seed=invocation_seed)
        assert result["image"].shape == ((11, 13) if kind == "crop" else (17, 19))
        assert result["image"].dtype == image.dtype
        assert np.isfinite(result["image"]).all()
        np.testing.assert_array_equal(image, np.full((17, 19), 0.5, np.float32))
        return [weakref.ref(image)]

    _assert_released(call() + temporary_references)


class _Fail(A.ImageOnlyTransform):
    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        raise RuntimeError("lifetime probe failure")


@pytest.mark.usefixtures("refcount_only")
@pytest.mark.parametrize("failure", ["apply", "postprocess", "before_sampling"])
@pytest.mark.parametrize("invocation_seed", [None, 137])
def test_exception_frames_do_not_retain_buffers(
    failure: str,
    invocation_seed: int | None,
    temporary_references: list,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = A.Compose([A.Blur(p=1), *([_Fail(p=1)] if failure == "apply" else [])], seed=137)
    if failure == "postprocess":

        def fail_postprocess(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("lifetime probe failure")

        monkeypatch.setattr(pipeline, "postprocess", fail_postprocess)

    def call() -> list:
        image = np.ones((17, 19), np.float32)
        try:
            pipeline(
                image=image,
                invocation_seed=invocation_seed,
                **({"frame_annotations": [{}]} if failure == "before_sampling" else {}),
            )
        except (ValueError, RuntimeError) as error:
            message = str(error)
        else:
            pytest.fail("expected an execution failure")
        assert "frame_annotations" in message or message == "lifetime probe failure"
        return [weakref.ref(image)]

    _assert_released(call() + temporary_references)


@pytest.mark.usefixtures("refcount_only")
@pytest.mark.parametrize("route", ["empty", "root_zero", "leaf_zero", "noop", "root_skip"])
def test_unchanged_routes_keep_exact_identity(route: str, temporary_references: list) -> None:
    def call() -> list:
        image = np.ones((17, 19), np.float32)
        transforms = (
            [] if route == "empty" else [A.NoOp(p=1) if route == "noop" else A.Blur(p=0 if route == "leaf_zero" else 1)]
        )
        probability = {"root_zero": 0, "root_skip": 0.01}.get(route, 1)
        result = A.Compose(transforms, p=probability, seed=137)(image=image)
        assert result["image"] is image
        return [weakref.ref(image)]

    _assert_released(call() + temporary_references)


@pytest.mark.usefixtures("refcount_only")
def test_live_unchanged_output_is_a_legitimate_owner() -> None:
    def call() -> tuple[list, dict]:
        image = np.ones((17, 19), np.float32)
        return [weakref.ref(image)], A.Compose([A.NoOp(p=1)])(image=image)

    references, result = call()
    assert references[0]() is result["image"]
    np.testing.assert_array_equal(result["image"], np.ones((17, 19), np.float32))
    del result
    _assert_released(references)


@pytest.mark.usefixtures("refcount_only")
def test_oracle_detects_a_deliberate_cycle() -> None:
    class Owner:
        pass

    def call() -> weakref.ReferenceType:
        owner = Owner()
        owner.self_reference = owner
        return weakref.ref(owner)

    reference = call()
    assert reference() is not None
    del reference().self_reference
    _assert_released([reference])


@pytest.mark.usefixtures("refcount_only")
def test_configured_nested_graph_has_one_boundary(temporary_references: list, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = A.Compose.preprocess

    def preprocess(pipeline: A.Compose, *args: Any, **kwargs: Any) -> None:
        calls.append(pipeline.invocation_key)
        original(pipeline, *args, **kwargs)

    monkeypatch.setattr(A.Compose, "preprocess", preprocess)
    pipeline = A.Compose([A.Compose([A.OneOf([A.Blur(p=1)], p=1)])], seed=137)
    _assert_released(_call_target(pipeline) + temporary_references)
    assert calls == [pipeline.invocation_key]


@pytest.mark.usefixtures("refcount_only")
@pytest.mark.parametrize("inner_error", [False, True])
def test_public_reentry_releases_inner_buffers_before_outer_finishes(
    inner_error: bool, temporary_references: list
) -> None:
    inner = A.Compose([A.Blur(p=1), *([_Fail(p=1)] if inner_error else [])], seed=137)

    class Reenter(A.ImageOnlyTransform):
        def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
            outer = invocation_module.get_current_invocation()

            def call_inner() -> list:
                inner_image = np.ones((17, 19), np.float32)
                message = None
                try:
                    inner(image=inner_image)
                except RuntimeError as error:
                    message = str(error)
                assert message == ("lifetime probe failure" if inner_error else None)
                return [weakref.ref(inner_image)]

            references = call_inner()
            assert all(reference() is None for reference in references)
            assert invocation_module.get_current_invocation() is outer
            return image

    pipeline = A.Compose([A.Blur(p=1), Reenter(p=1)], seed=137)
    _assert_released(_call_target(pipeline) + temporary_references)


@pytest.mark.usefixtures("refcount_only")
@pytest.mark.parametrize("mode", ["applied", "metadata", "snapshot", "timing", "replay"])
def test_observed_artifacts_survive_later_calls_and_release_after_discard(
    mode: str, temporary_references: list
) -> None:
    def call() -> list:
        image = np.arange(17 * 19, dtype=np.float32).reshape(17, 19) / (17 * 19)
        pipeline_cls = A.ReplayCompose if mode == "replay" else A.Compose
        pipeline = pipeline_cls([A.Blur(p=1)], seed=137, **({"save_applied_params": True} if mode == "applied" else {}))
        if mode in {"metadata", "snapshot", "timing"}:
            options = (
                A.TraceOptions(snapshot_targets=("image",))
                if mode == "snapshot"
                else A.TraceOptions(include_timing=mode == "timing")
            )
            trace = pipeline.run_with_trace(image=image, options=options)
            result = trace.data
            assert trace.records[-1].status == "applied"
        else:
            result = pipeline(image=image)
        expected = result["image"].copy()
        pipeline(image=np.ones((17, 19), np.float32))
        try:
            pipeline(image=image, frame_annotations=[{}])
        except ValueError:
            pass
        else:
            pytest.fail("expected invalid frame annotations")
        np.testing.assert_array_equal(result["image"], expected)
        if mode == "replay":
            np.testing.assert_array_equal(A.ReplayCompose.replay(result["replay"], image=image)["image"], expected)
        elif mode == "applied":
            replay = A.Compose.from_applied_transforms(result["applied_transforms"])
            np.testing.assert_array_equal(replay(image=image)["image"], expected)
        elif mode == "snapshot":
            snapshots = [record.snapshot for record in trace.records if record.snapshot is not None]
            assert snapshots
            np.testing.assert_array_equal(snapshots[-1]["image"], expected[..., None])
        references = []
        _watch(image, references)
        invocation_module.clear_completed_observation()
        return references

    _assert_released(call() + temporary_references)


@pytest.mark.usefixtures("refcount_only")
@pytest.mark.parametrize("binding", ["processors", "instances", "frames"])
def test_processor_and_binding_states_release(binding: str, temporary_references: list) -> None:
    def call() -> list:
        image = np.ones((17, 19), np.float32)
        mask = np.ones((17, 19), np.uint8)
        options: dict[str, Any] = {"bbox_params": A.BboxParams(coord_format="pascal_voc", label_fields=["label"])}
        if binding == "instances":
            options["instance_binding"] = ["masks", "bboxes"]
            data = {
                "image": image,
                "instances": [{"mask": mask, "bbox": [2, 3, 10, 12], "bbox_labels": {"label": "cat"}}],
            }
        elif binding == "frames":
            options["frame_binding"] = ["images", "frame_annotations"]
            data = {
                "images": np.stack([image, image]),
                "frame_annotations": [{"mask": mask, "bboxes": [[2, 3, 10, 12]], "label": ["cat"]}, {}],
            }
        else:
            options["keypoint_params"] = A.KeypointParams(coord_format="xy")
            data = {"image": image, "mask": mask, "bboxes": [[2, 3, 10, 12]], "label": ["cat"], "keypoints": [[4, 5]]}
        references = []
        _watch(data, references)
        result = A.Compose([A.Blur(p=1)], seed=137, **options)(**data)
        if binding == "instances":
            assert result["instances"][0]["bbox_labels"]["label"] == "cat"
            np.testing.assert_array_equal(result["instances"][0]["mask"], mask)
        elif binding == "frames":
            assert result["frame_annotations"][0]["label"] == ["cat"]
            np.testing.assert_array_equal(result["frame_annotations"][0]["mask"], mask)
            assert result["frame_annotations"][1] == {}
        else:
            np.testing.assert_allclose(result["bboxes"], [[2, 3, 10, 12]])
            np.testing.assert_array_equal(result["keypoints"], [[4, 5]])
            assert result["label"] == ["cat"]
        return references

    _assert_released(call() + temporary_references)


@pytest.mark.usefixtures("refcount_only")
def test_repeated_calls_do_not_accumulate_buffers(temporary_references: list) -> None:
    pipeline = A.Compose([A.Blur(p=1)], seed=137)
    references = []
    for _ in range(32):
        references.extend(_call_target(pipeline))
        _assert_released(references + temporary_references)


@pytest.mark.usefixtures("refcount_only")
@pytest.mark.parametrize("invocation_seed", [None, 137])
def test_overlapping_calls_release_independent_inputs(invocation_seed: int | None, temporary_references: list) -> None:
    barrier = Barrier(2)

    class Overlap(A.Blur):
        def sample_parameters(
            self, params: dict, data: dict, targets: TargetSet, sampling: SamplingContext
        ) -> SampledParams:
            barrier.wait(timeout=10)
            return super().sample_parameters(params, data, targets, sampling)

    pipeline = A.Compose([Overlap(p=1)], seed=137)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_call_target, pipeline, invocation_seed=invocation_seed) for _ in range(2)]
        references = [reference for future in futures for reference in future.result(timeout=15)]
    del futures
    _assert_released(references + temporary_references)


@pytest.mark.usefixtures("refcount_only")
@pytest.mark.parametrize("factory", ["seed", "owner"])
@pytest.mark.parametrize("access", ["python", "numpy", "both"])
def test_standalone_sampling_releases_call_context(factory: str, access: str, temporary_references: list) -> None:
    def call() -> list:
        sampling = (
            invocation_module.SamplingContext.from_seed(137)
            if factory == "seed"
            else invocation_module.SamplingContext.from_owner(A.Blur(p=1))
        )
        if access in {"python", "both"}:
            assert 0 <= sampling.py_random.random() < 1
        if access in {"numpy", "both"}:
            assert 0 <= sampling.random_generator.random() < 1
        return temporary_references.copy()

    references = call()
    assert all(reference() is None for reference in references)


@pytest.mark.usefixtures("refcount_only")
def test_retained_sampling_view_does_not_own_execution_state(temporary_references: list) -> None:
    handles = []

    class KeepSampling(A.Blur):
        def sample_parameters(
            self, params: dict, data: dict, targets: TargetSet, sampling: SamplingContext
        ) -> SampledParams:
            handles.append(sampling)
            return super().sample_parameters(params, data, targets, sampling)

    pipeline = A.Compose([KeepSampling(p=1)], seed=137)
    references = _call_target(pipeline)
    # The retained view legitimately owns its RNG state, but not its invocation or buffers.
    assert all(
        reference() is None or isinstance(reference(), invocation_module._InvocationRandomState)
        for reference in temporary_references
    )
    _assert_released(references)
    assert 0 <= handles[0].py_random.random() < 1
    assert 0 <= handles[0].random_generator.random() < 1
    handles.clear()
    _assert_released(temporary_references)
