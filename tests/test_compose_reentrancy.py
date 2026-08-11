"""Threaded regression coverage for the public Compose reentrancy contract."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any, Literal

import numpy as np
import pytest
import torch

import albumentations as A


class _BlockingAppliedConfigTransform(A.ImageOnlyTransform):
    """Hold one invocation inside transform dispatch while detecting a second invocation."""

    def __init__(self, marker: int = 0, p: float = 1.0):
        super().__init__(p=p)
        self.marker = marker
        self.first_entered = Event()
        self.second_entered = Event()
        self.release_first = Event()

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        marker = int(np.asarray(data["image"]).reshape(-1)[0])
        self.applied_config = {"marker": marker}
        return {"marker": marker}

    def apply(self, image: np.ndarray, marker: int, **params: Any) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("Compose must bridge Tensor input before NumPy-only dispatch")
        if marker == 1:
            self.first_entered.set()
            if not self.release_first.wait(timeout=10):
                raise TimeoutError("First Compose invocation was not released")
        elif marker == 2:
            self.second_entered.set()
        return image


def _invoke(
    pipeline: A.Compose,
    mode: Literal["call", "trace"],
    data: dict[str, Any],
) -> tuple[dict[str, Any], tuple[A.TraceRecord, ...]]:
    if mode == "trace":
        traced = pipeline.run_with_trace(**data)
        return traced.data, traced.records
    return pipeline(**data), ()


def _run_overlapping_calls(
    pipeline: A.Compose,
    probe: _BlockingAppliedConfigTransform,
    first_mode: Literal["call", "trace"],
    first_data: dict[str, Any],
    second_data: dict[str, Any],
) -> tuple[
    tuple[dict[str, Any], tuple[A.TraceRecord, ...]],
    tuple[dict[str, Any], tuple[A.TraceRecord, ...]],
]:
    second_attempted = Event()

    def run_second() -> tuple[dict[str, Any], tuple[A.TraceRecord, ...]]:
        second_attempted.set()
        return _invoke(pipeline, "call", second_data)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(_invoke, pipeline, first_mode, first_data)
        second_future = None
        try:
            if not probe.first_entered.wait(timeout=5):
                raise TimeoutError("First Compose invocation did not reach transform dispatch")
            second_future = executor.submit(run_second)
            if not second_attempted.wait(timeout=5):
                raise TimeoutError("Second Compose invocation did not start")
            second_overlapped = probe.second_entered.wait(timeout=0.25)
        finally:
            probe.release_first.set()

        first_result = first_future.result(timeout=5)
        if second_future is None:
            raise RuntimeError("Second Compose invocation was not submitted")
        second_result = second_future.result(timeout=5)

    assert not second_overlapped, "Calls to one Compose instance must be serialized"
    return first_result, second_result


@pytest.mark.parametrize("compose_type", [A.Compose, A.ReplayCompose])
@pytest.mark.parametrize("first_mode", ["call", "trace"])
def test_overlapping_calls_preserve_tensor_grayscale_and_applied_params(
    compose_type: type[A.Compose],
    first_mode: Literal["call", "trace"],
) -> None:
    probe = _BlockingAppliedConfigTransform(p=1.0)
    pipeline = compose_type([probe], save_applied_params=True, telemetry=False)
    tensor_image = torch.ones((1, 8, 8), dtype=torch.uint8)
    tensor_mask = torch.ones((8, 8), dtype=torch.uint8)
    numpy_image = np.full((8, 8), 2, dtype=np.uint8)

    (first_result, first_records), (second_result, _) = _run_overlapping_calls(
        pipeline,
        probe,
        first_mode,
        {"image": tensor_image, "mask": tensor_mask},
        {"image": numpy_image},
    )

    assert isinstance(first_result["image"], torch.Tensor)
    assert isinstance(first_result["mask"], torch.Tensor)
    assert first_result["image"].shape == tensor_image.shape
    assert first_result["mask"].shape == tensor_mask.shape
    torch.testing.assert_close(first_result["image"], tensor_image)
    torch.testing.assert_close(first_result["mask"], tensor_mask)
    assert isinstance(second_result["image"], np.ndarray)
    assert second_result["image"].shape == numpy_image.shape
    np.testing.assert_array_equal(second_result["image"], numpy_image)
    assert first_result["applied_transforms"][0][1]["marker"] == 1
    assert second_result["applied_transforms"][0][1]["marker"] == 2
    if compose_type is A.ReplayCompose:
        assert first_result["replay"]["applied"]
        assert second_result["replay"]["applied"]
    if first_mode == "trace":
        leaf_records = [record for record in first_records if record.node_kind == "leaf"]
        assert leaf_records[0].params is not None
        assert leaf_records[0].params["marker"] == 1


def test_overlapping_calls_preserve_instance_binding_bookkeeping() -> None:
    probe = _BlockingAppliedConfigTransform(p=1.0)
    pipeline = A.Compose(
        [probe],
        bbox_params=A.BboxParams(coord_format="pascal_voc"),
        instance_binding=["masks", "bboxes"],
        telemetry=False,
    )

    def make_instances(count: int) -> list[dict[str, Any]]:
        return [
            {
                "mask": np.full((16, 16), index + 1, dtype=np.uint8),
                "bbox": np.array([1, 1, 8, 8], dtype=np.float32),
            }
            for index in range(count)
        ]

    first_instances = make_instances(1)
    second_instances = make_instances(2)
    (first_result, _), (second_result, _) = _run_overlapping_calls(
        pipeline,
        probe,
        "call",
        {"image": np.ones((16, 16, 3), dtype=np.uint8), "instances": first_instances},
        {"image": np.full((16, 16, 3), 2, dtype=np.uint8), "instances": second_instances},
    )

    assert len(first_result["instances"]) == 1
    assert len(second_result["instances"]) == 2
    np.testing.assert_array_equal(first_result["instances"][0]["mask"], first_instances[0]["mask"])
    for actual, expected in zip(second_result["instances"], second_instances, strict=True):
        np.testing.assert_array_equal(actual["mask"], expected["mask"])


def test_overlapping_seeded_calls_match_lock_acquisition_order() -> None:
    first_image = np.full((16, 16, 3), 100, dtype=np.uint8)
    first_image[0, 0] = 1
    second_image = np.full((16, 16, 3), 150, dtype=np.uint8)
    second_image[0, 0] = 2

    concurrent_probe = _BlockingAppliedConfigTransform(p=1.0)
    concurrent_pipeline = A.Compose(
        [concurrent_probe, A.RandomBrightnessContrast(p=1.0)],
        seed=137,
        telemetry=False,
    )
    (first_result, _), (second_result, _) = _run_overlapping_calls(
        concurrent_pipeline,
        concurrent_probe,
        "call",
        {"image": first_image.copy()},
        {"image": second_image.copy()},
    )

    serial_probe = _BlockingAppliedConfigTransform(p=1.0)
    serial_probe.release_first.set()
    serial_pipeline = A.Compose(
        [serial_probe, A.RandomBrightnessContrast(p=1.0)],
        seed=137,
        telemetry=False,
    )
    expected_first = serial_pipeline(image=first_image.copy())
    expected_second = serial_pipeline(image=second_image.copy())

    np.testing.assert_array_equal(first_result["image"], expected_first["image"])
    np.testing.assert_array_equal(second_result["image"], expected_second["image"])
