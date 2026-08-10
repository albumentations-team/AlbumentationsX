import copy

import numpy as np
import pytest

import albumentations as A


def test_run_with_trace_reports_the_executed_leaf_at_its_structural_path() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    pipeline = A.Compose([A.HorizontalFlip(p=1.0)], seed=137)

    trace_result = pipeline.run_with_trace(image=image)

    np.testing.assert_array_equal(trace_result.data["image"], image[:, ::-1])
    leaf_records = [record for record in trace_result.records if record.node_kind == "leaf"]
    assert len(leaf_records) == 1
    assert leaf_records[0].node_path == (0,)
    assert leaf_records[0].class_fullname == "HorizontalFlip"
    assert leaf_records[0].status == "applied"
    assert leaf_records[0].snapshot is None


def test_run_with_trace_matches_normal_execution_for_nested_stochastic_compositions() -> None:
    def make_pipeline() -> A.Compose:
        return A.Compose(
            [
                A.OneOf([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], p=1.0),
                A.SomeOf([A.RandomRotate90(p=1.0), A.Transpose(p=0.0)], n=2, replace=True, p=1.0),
                A.Sequential([A.InvertImg(p=1.0)], p=1.0),
                A.SelectiveChannelTransform([A.InvertImg(p=1.0)], channels=(1,), p=1.0),
            ],
            seed=137,
        )

    image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)

    expected = make_pipeline()(image=image)
    traced = make_pipeline().run_with_trace(image=image)

    np.testing.assert_array_equal(traced.data["image"], expected["image"])
    leaf_records = [record for record in traced.records if record.node_kind == "leaf"]
    assert {record.node_path for record in leaf_records} <= {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)}
    assert any(record.status == "skipped_selection" for record in leaf_records)
    assert all(record.event_index == index for index, record in enumerate(traced.records))


def test_replay_with_trace_replays_output_and_trace_tree() -> None:
    image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    pipeline = A.ReplayCompose(
        [A.OneOf([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], p=1.0)],
        seed=137,
    )

    original = pipeline.run_with_trace(image=image)
    replayed = A.ReplayCompose.replay_with_trace(original.data["replay"], image=image)

    np.testing.assert_array_equal(replayed.data["image"], original.data["image"])
    assert [(record.node_path, record.node_kind, record.status) for record in replayed.records] == [
        (record.node_path, record.node_kind, record.status) for record in original.records
    ]


def test_trace_snapshots_are_post_step_owned_copies() -> None:
    image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    pipeline = A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], seed=137)

    traced = pipeline.run_with_trace(image=image, options=A.TraceOptions(snapshot_targets=("image",)))

    leaf_records = [record for record in traced.records if record.node_kind == "leaf"]
    assert len(leaf_records) == 2
    np.testing.assert_array_equal(leaf_records[0].snapshot["image"], image[:, ::-1])
    np.testing.assert_array_equal(leaf_records[1].snapshot["image"], image[::-1, ::-1])
    assert not np.shares_memory(leaf_records[0].snapshot["image"], traced.data["image"])
    assert not np.shares_memory(leaf_records[0].snapshot["image"], leaf_records[1].snapshot["image"])


def test_trace_records_repeated_structural_nodes_by_occurrence() -> None:
    pipeline = A.Compose(
        [A.SomeOf([A.InvertImg(p=1.0)], n=2, replace=True, p=1.0)],
        seed=137,
    )

    traced = pipeline.run_with_trace(image=np.arange(12, dtype=np.uint8).reshape(2, 2, 3))

    leaf_records = [record for record in traced.records if record.node_kind == "leaf"]
    assert [record.node_path for record in leaf_records] == [(0, 0), (0, 0)]
    assert [record.occurrence_index for record in leaf_records] == [0, 1]


def test_trace_consumes_the_same_rng_stream_as_normal_execution() -> None:
    def make_pipeline() -> A.Compose:
        return A.Compose(
            [
                A.OneOf([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], p=1.0),
                A.RandomBrightnessContrast(p=1.0),
            ],
            seed=137,
        )

    image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    normal = make_pipeline()
    traced = make_pipeline()

    expected_first = normal(image=image)
    actual_first = traced.run_with_trace(image=image)
    expected_second = normal(image=image)
    actual_second = traced(image=image)

    np.testing.assert_array_equal(actual_first.data["image"], expected_first["image"])
    np.testing.assert_array_equal(actual_second["image"], expected_second["image"])


def test_trace_snapshots_include_all_synchronized_public_targets() -> None:
    image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    mask = np.arange(3 * 5, dtype=np.uint8).reshape(3, 5)
    volume = np.arange(2 * 3 * 5, dtype=np.uint8).reshape(2, 3, 5)
    mask3d = np.arange(2 * 3 * 5, dtype=np.uint8).reshape(2, 3, 5)
    snapshot_targets = ("image", "image2", "mask", "bboxes", "keypoints", "volume", "mask3d")
    pipeline = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams("pascal_voc"),
        keypoint_params=A.KeypointParams("xy"),
        additional_targets={"image2": "image"},
        seed=137,
    )

    traced = pipeline.run_with_trace(
        image=image,
        image2=image.copy(),
        mask=mask,
        bboxes=[[1.0, 0.0, 3.0, 2.0]],
        keypoints=[[1.0, 1.0]],
        volume=volume,
        mask3d=mask3d,
        options=A.TraceOptions(snapshot_targets=snapshot_targets),
    )

    leaf_record = next(record for record in traced.records if record.node_kind == "leaf")
    assert set(leaf_record.snapshot) == set(snapshot_targets)
    np.testing.assert_array_equal(leaf_record.snapshot["image"], image[:, ::-1])
    np.testing.assert_array_equal(leaf_record.snapshot["image2"], image[:, ::-1])
    np.testing.assert_array_equal(leaf_record.snapshot["mask"].squeeze(-1), mask[:, ::-1])
    np.testing.assert_array_equal(leaf_record.snapshot["volume"].squeeze(-1), volume[:, :, ::-1])
    np.testing.assert_array_equal(leaf_record.snapshot["mask3d"].squeeze(-1), mask3d[:, :, ::-1])
    assert leaf_record.snapshot["bboxes"].shape == (1, 4)
    assert leaf_record.snapshot["keypoints"].shape == (1, 5)
    assert not np.shares_memory(leaf_record.snapshot["image"], traced.data["image"])


def test_trace_options_reject_invalid_snapshot_configuration() -> None:
    with pytest.raises(TypeError, match="not a string"):
        A.TraceOptions(snapshot_targets="image")

    pipeline = A.Compose([A.NoOp(p=1.0)], seed=137)
    with pytest.raises(ValueError, match="Unknown trace snapshot targets: missing"):
        pipeline.run_with_trace(
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            options=A.TraceOptions(snapshot_targets=("missing",)),
        )


def test_trace_reports_every_node_when_the_root_is_skipped() -> None:
    pipeline = A.Compose(
        [A.OneOf([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], p=1.0)],
        p=0.0,
        seed=137,
    )

    traced = pipeline.run_with_trace(image=np.zeros((2, 2, 3), dtype=np.uint8))

    assert [(record.node_path, record.status) for record in traced.records] == [
        ((), "skipped_probability"),
        ((0,), "skipped_probability"),
        ((0, 0), "skipped_probability"),
        ((0, 1), "skipped_probability"),
    ]


def test_trace_and_normal_execution_preserve_applied_transform_records_in_every_composition() -> None:
    def make_pipeline() -> A.Compose:
        return A.Compose(
            [
                A.OneOrOther(A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), p=1.0),
                A.SelectiveChannelTransform([A.InvertImg(p=1.0)], channels=(1,), p=1.0),
            ],
            save_applied_params=True,
            seed=137,
        )

    image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    expected = make_pipeline()(image=image)
    traced = make_pipeline().run_with_trace(image=image)

    np.testing.assert_array_equal(traced.data["image"], expected["image"])
    assert traced.data["applied_transforms"] == expected["applied_transforms"]


def test_trace_paths_cover_nested_compose_and_random_order() -> None:
    pipeline = A.Compose(
        [
            A.Compose(
                [A.RandomOrder([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], n=2, p=1.0)],
                p=1.0,
            ),
        ],
        seed=137,
    )

    traced = pipeline.run_with_trace(image=np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3))

    leaf_records = [record for record in traced.records if record.node_kind == "leaf"]
    assert {record.node_path for record in leaf_records} == {(0, 0, 0), (0, 0, 1)}
    assert all(record.status == "applied" for record in leaf_records)


def test_trace_paths_and_output_survive_portable_serialization() -> None:
    pipeline = A.Compose(
        [
            A.OneOf([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], p=1.0),
            A.SomeOf([A.RandomRotate90(p=1.0)], n=2, replace=True, p=1.0),
            A.SelectiveChannelTransform([A.InvertImg(p=1.0)], channels=[1], p=1.0),
        ],
        seed=137,
    )
    restored = A.from_dict(A.to_dict(pipeline))
    image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)

    original_trace = pipeline.run_with_trace(image=image)
    restored_trace = restored.run_with_trace(image=image)

    np.testing.assert_array_equal(restored_trace.data["image"], original_trace.data["image"])
    assert [
        (record.node_path, record.occurrence_index, record.node_kind, record.status)
        for record in restored_trace.records
    ] == [
        (record.node_path, record.occurrence_index, record.node_kind, record.status)
        for record in original_trace.records
    ]


def test_trace_snapshot_observes_bbox_filtering_at_the_leaf_boundary() -> None:
    pipeline = A.Compose(
        [A.Crop(x_min=0, y_min=0, x_max=2, y_max=2, p=1.0)],
        bbox_params=A.BboxParams("pascal_voc"),
        seed=137,
    )

    traced = pipeline.run_with_trace(
        image=np.zeros((5, 5, 3), dtype=np.uint8),
        bboxes=[[3.0, 3.0, 4.0, 4.0]],
        options=A.TraceOptions(snapshot_targets=("bboxes",)),
    )

    leaf_record = next(record for record in traced.records if record.node_kind == "leaf")
    assert leaf_record.snapshot["bboxes"].shape == (0, 4)
    assert traced.data["bboxes"].shape == (0, 4)


def test_timed_trace_preserves_output_and_reports_leaf_duration() -> None:
    image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    expected = A.Compose([A.HorizontalFlip(p=1.0)], seed=137)(image=image)
    traced = A.Compose([A.HorizontalFlip(p=1.0)], seed=137).run_with_trace(
        image=image,
        options=A.TraceOptions(include_timing=True),
    )

    np.testing.assert_array_equal(traced.data["image"], expected["image"])
    leaf_record = next(record for record in traced.records if record.node_kind == "leaf")
    assert leaf_record.elapsed_ns is not None
    assert leaf_record.elapsed_ns >= 0


def test_observer_only_trace_emits_records_without_retaining_them() -> None:
    observed = []
    pipeline = A.Compose([A.HorizontalFlip(p=1.0)], seed=137)

    traced = pipeline.run_with_trace(
        image=np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        options=A.TraceOptions(observer=observed.append, collect_records=False),
    )

    assert traced.records == ()
    assert [(record.node_path, record.node_kind) for record in observed] == [((0,), "leaf"), ((), "composition")]


def test_trace_preserves_instance_binding_output_and_snapshots() -> None:
    def make_pipeline() -> A.Compose:
        return A.Compose(
            [A.HorizontalFlip(p=1.0)],
            bbox_params=A.BboxParams("pascal_voc"),
            instance_binding=("masks", "bboxes"),
            seed=137,
        )

    image = np.zeros((6, 8, 3), dtype=np.uint8)
    first_mask = np.zeros((6, 8), dtype=np.uint8)
    first_mask[:, :3] = 1
    second_mask = np.zeros((6, 8), dtype=np.uint8)
    second_mask[:, 4:] = 1
    instances = [
        {"mask": first_mask, "bbox": np.array([0.0, 0.0, 3.0, 6.0], dtype=np.float32)},
        {"mask": second_mask, "bbox": np.array([4.0, 0.0, 8.0, 6.0], dtype=np.float32)},
    ]

    expected = make_pipeline()(image=image, instances=copy.deepcopy(instances))
    traced = make_pipeline().run_with_trace(
        image=image,
        instances=copy.deepcopy(instances),
        options=A.TraceOptions(snapshot_targets=("masks", "bboxes")),
    )

    assert len(traced.data["instances"]) == len(expected["instances"])
    for actual_instance, expected_instance in zip(traced.data["instances"], expected["instances"], strict=True):
        np.testing.assert_array_equal(actual_instance["mask"], expected_instance["mask"])
        np.testing.assert_array_equal(actual_instance["bbox"], expected_instance["bbox"])

    leaf_record = next(record for record in traced.records if record.node_kind == "leaf")
    assert leaf_record.snapshot["masks"].shape[0] == 2
    assert leaf_record.snapshot["bboxes"].shape[0] == 2
