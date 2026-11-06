import copy
import json
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

import albumentations as A
from tests.helpers.contract_assertions import assert_contract_values_equal


def _trace_record_values(records: tuple[A.TraceRecord, ...]) -> list[dict[str, Any]]:
    return [
        {
            "node_path": record.node_path,
            "event_index": record.event_index,
            "occurrence_index": record.occurrence_index,
            "class_fullname": record.class_fullname,
            "node_kind": record.node_kind,
            "status": record.status,
            "params": None if record.params is None else dict(record.params),
            "snapshot": None if record.snapshot is None else dict(record.snapshot),
            "elapsed_ns": record.elapsed_ns,
        }
        for record in records
    ]


@pytest.mark.parametrize("root_type", [A.Compose, A.ReplayCompose])
@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
@pytest.mark.parametrize("prior_traced", [False, True], ids=["ordinary-prior", "traced-prior"])
def test_trace_unknown_target_clears_previous_observation(
    root_type: type[A.Compose],
    nested: bool,
    prior_traced: bool,
) -> None:
    def make_pipeline() -> tuple[A.Compose, A.RandomBrightnessContrast]:
        leaf = A.RandomBrightnessContrast(p=1.0)
        child = A.Compose([leaf]) if nested else leaf
        return root_type([child], seed=137, save_applied_params=not prior_traced), leaf

    def run_prior(pipeline: A.Compose) -> tuple[dict[str, Any], tuple[A.TraceRecord, ...]]:
        if prior_traced:
            traced = pipeline.run_with_trace(image=image, options=A.TraceOptions(snapshot_targets=("image",)))
            return traced.data, traced.records
        return pipeline(image=image), ()

    image = np.arange(45, dtype=np.uint8).reshape(3, 5, 3)
    expected_image = image.copy()
    pipeline, leaf = make_pipeline()
    prior_data, prior_records = run_prior(pipeline)
    prior_params = leaf.get_applied_params()
    prior_config = leaf.get_applied_config()
    assert prior_params
    assert prior_config
    prior_artifacts = {"data": prior_data, "params": prior_params, "config": prior_config}
    expected_artifacts = copy.deepcopy(prior_artifacts)
    expected_records = copy.deepcopy(_trace_record_values(prior_records))
    callbacks: list[A.TraceRecord] = []

    with pytest.raises(ValueError, match=r"^Unknown trace snapshot targets: missing$"):
        pipeline.run_with_trace(
            image=image,
            options=A.TraceOptions(snapshot_targets=("missing",), observer=callbacks.append),
        )

    assert leaf.get_applied_params() == {}
    assert leaf.get_applied_config() == {}
    assert callbacks == []
    assert_contract_values_equal(image, expected_image)
    assert_contract_values_equal(prior_artifacts, expected_artifacts)
    assert_contract_values_equal(_trace_record_values(prior_records), expected_records)

    retry = pipeline.run_with_trace(image=image)
    retry_params = leaf.get_applied_params()
    retry_config = leaf.get_applied_config()
    assert retry_params
    assert retry_config
    reference, reference_leaf = make_pipeline()
    run_prior(reference)
    expected_retry = reference.run_with_trace(image=image)

    assert_contract_values_equal(retry.data["image"], expected_retry.data["image"])
    assert_contract_values_equal(retry_params, reference_leaf.get_applied_params())
    assert_contract_values_equal(retry_config, reference_leaf.get_applied_config())
    assert_contract_values_equal(image, expected_image)
    assert_contract_values_equal(prior_artifacts, expected_artifacts)
    assert_contract_values_equal(_trace_record_values(prior_records), expected_records)


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


def test_replay_preserves_each_occurrence_of_one_reused_transform() -> None:
    """Repeated visits record separate parameter samples instead of overwriting by object id."""
    transform = A.RandomBrightnessContrast(p=1.0)
    pipeline = A.ReplayCompose([transform, transform], seed=137)
    image = np.full((3, 3, 3), 128, dtype=np.uint8)

    original = pipeline(image=image)
    replayed = A.ReplayCompose.replay(original["replay"], image=image)

    np.testing.assert_array_equal(replayed["image"], original["image"])
    first, second = original["replay"]["transforms"]
    assert first["params"] != second["params"]


@pytest.mark.parametrize("container", [A.SomeOf, A.RandomOrder])
@pytest.mark.parametrize("replace", [False, True])
def test_replay_preserves_selected_child_visits_and_their_parameters(container, replace) -> None:
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    for seed in range(137, 145):
        pipeline = A.ReplayCompose(
            [
                container(
                    [A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=1.0)],
                    n=4 if replace else 2,
                    replace=replace,
                    p=1.0,
                )
            ],
            seed=seed,
        )
        original = pipeline.run_with_trace(image=image)
        saved = json.loads(json.dumps(original.data["replay"]))
        replayed = A.ReplayCompose.replay_with_trace(saved, image=image)
        np.testing.assert_array_equal(replayed.data["image"], original.data["image"])
        second_replay = A.ReplayCompose.replay(replayed.data["replay"], image=image)
        np.testing.assert_array_equal(second_replay["image"], original.data["image"])
        expected_paths = [record.node_path for record in original.records if record.node_kind == "leaf"]
        assert [record.node_path for record in replayed.records if record.node_kind == "leaf"] == expected_paths


def test_replay_preserves_repeated_nested_selections_and_channel_scopes() -> None:
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    mask = np.arange(5 * 7, dtype=np.uint8).reshape(5, 7)
    for seed in range(137, 145):
        pipeline = A.ReplayCompose(
            [
                A.SomeOf(
                    [
                        A.OneOf([A.HorizontalFlip(p=1), A.VerticalFlip(p=1)], p=0.7),
                        A.SelectiveChannelTransform([A.RandomBrightnessContrast(p=1)], channels=(0,), p=1),
                        A.OneOrOther(A.HorizontalFlip(p=1), A.VerticalFlip(p=1), p=0.5),
                        A.Compose([A.HorizontalFlip(p=1)], p=0.3),
                    ],
                    n=5,
                    replace=True,
                    p=1,
                )
            ],
            seed=seed,
        )
        original = pipeline(image=image, mask=mask)
        saved = json.loads(json.dumps(original["replay"]))
        replayed = A.ReplayCompose.replay(saved, image=image, mask=mask)
        for target in ("image", "mask"):
            np.testing.assert_array_equal(replayed[target], original[target])


@pytest.mark.parametrize("alias", ["other", "replay_params"])
def test_replay_visit_parameters_use_root_additional_target_policy(alias: str) -> None:
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    pipeline = A.ReplayCompose(
        [A.SomeOf([A.HorizontalFlip(p=1)], n=3, replace=True)],
        additional_targets={alias: "image"},
        seed=137,
    )
    data = {"image": image, alias: image.copy()}
    original = pipeline(**data)
    replayed = A.ReplayCompose.replay(original["replay"], **data)
    np.testing.assert_array_equal(replayed[alias], original[alias])


def test_nested_replay_containers_preserve_outer_visits() -> None:
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    pipeline = A.ReplayCompose(
        [A.HorizontalFlip(p=1), A.ReplayCompose([A.VerticalFlip(p=1)]), A.RandomBrightnessContrast(p=1)], seed=137
    )
    original = pipeline(image=image)
    replayed = A.ReplayCompose.replay(original["replay"], image=image)
    np.testing.assert_array_equal(replayed["image"], original["image"])


def test_repeated_replay_stores_each_sampled_noise_field_once() -> None:
    def arrays(value):
        if isinstance(value, np.ndarray):
            return [value]
        if isinstance(value, dict):
            return [array for child in value.values() for array in arrays(child)]
        if isinstance(value, (list, tuple)):
            return [array for child in value for array in arrays(child)]
        return []

    image = np.full((12, 14, 3), 137, dtype=np.uint8)
    pipeline = A.ReplayCompose([A.SomeOf([A.GaussNoise(p=1)], n=3, replace=True)], seed=137)
    original = pipeline(image=image)
    assert len(arrays(original["replay"])) == 3
    replayed = A.ReplayCompose.replay(original["replay"], image=image)
    np.testing.assert_array_equal(replayed["image"], original["image"])


@pytest.mark.parametrize(
    ("visits", "error", "message"),
    [
        ("invalid", TypeError, "visits must be a list"),
        ([{"index": True, "execution": {}}], TypeError, "integer index"),
        ([{"index": 137, "execution": {}}], ValueError, "outside the configured child pool"),
        ([{"index": 0, "execution": {"applied": "yes"}}], TypeError, "boolean applied flag"),
        ([{"index": 0, "execution": {"applied": True, "params": None}}], TypeError, "sampled parameters"),
        ([{"index": 0, "execution": {"applied": False}}], ValueError, "params"),
        ([{"index": 0, "execution": {"applied": False, "params": 137}}], TypeError, "dict or None"),
    ],
)
def test_replay_rejects_malformed_child_visit_metadata(visits, error, message) -> None:
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    pipeline = A.ReplayCompose([A.SomeOf([A.HorizontalFlip(p=1)], n=2, replace=True)], seed=137)
    saved = pipeline(image=image)["replay"]
    saved["transforms"][0]["params"]["visits"] = visits
    with pytest.raises(error, match=message):
        A.ReplayCompose.replay(saved, image=image)


@pytest.mark.parametrize("container_name", ["OneOf", "OneOrOther", "SomeOf"])
def test_replay_preserves_selected_containers_when_their_leaves_skip(container_name: str) -> None:
    selected = A.Sequential([A.HorizontalFlip(p=0)], p=1)
    if container_name == "OneOrOther":
        container = A.OneOrOther(selected, A.VerticalFlip(p=1), p=1)
    elif container_name == "SomeOf":
        container = A.SomeOf([selected], n=1, p=1)
    else:
        container = A.OneOf([selected], p=1)
    pipeline = A.ReplayCompose([container], seed=137)
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    original = pipeline.run_with_trace(image=image)
    replayed = A.ReplayCompose.replay_with_trace(original.data["replay"], image=image)
    assert [(record.node_path, record.node_kind, record.status == "applied") for record in replayed.records] == [
        (record.node_path, record.node_kind, record.status == "applied") for record in original.records
    ]
    np.testing.assert_array_equal(replayed.data["image"], original.data["image"])


def test_reused_forced_leaf_clears_observation_when_its_next_visit_skips() -> None:
    flip = A.HorizontalFlip(p=0)
    pipeline = A.ReplayCompose([A.OneOrOther(flip, A.NoOp(p=1), p=1), flip], seed=137, save_applied_params=True)
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    original = pipeline.run_with_trace(image=image)
    leaves = [record for record in original.records if record.node_kind == "leaf"]
    assert leaves[-1].status == "skipped_probability"
    assert len(original.data["applied_transforms"]) == 1
    replayed = A.ReplayCompose.replay_with_trace(original.data["replay"], image=image)
    np.testing.assert_array_equal(replayed.data["image"], original.data["image"])


@pytest.mark.parametrize("container_name", ["SomeOf", "RandomOrder"])
def test_replay_accepts_legacy_records_without_child_visit_metadata(container_name: str) -> None:
    path = Path(__file__).parent / "files" / "replay-someof-without-visits.json"
    saved = json.loads(path.read_text())
    saved["transforms"][0]["__class_fullname__"] = container_name
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    replayed = A.ReplayCompose.replay(saved, image=image)
    np.testing.assert_array_equal(replayed["image"], image[:, ::-1])


def test_replay_keeps_annotations_aligned_through_reordered_repeated_geometry() -> None:
    data = {
        "image": np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3),
        "mask": np.arange(5 * 7, dtype=np.uint8).reshape(5, 7),
        "bboxes": np.array([[1, 1, 6, 4]], dtype=np.float32),
        "labels": ["object"],
        "keypoints": np.array([[2, 3]], dtype=np.float32),
    }
    pipeline = A.ReplayCompose(
        [
            A.RandomOrder(
                [A.Crop(x_min=0, y_min=0, x_max=5, y_max=4, p=1), A.HorizontalFlip(p=1)], n=4, replace=True, p=1
            )
        ],
        bbox_params=A.BboxParams("pascal_voc", label_fields=["labels"]),
        keypoint_params=A.KeypointParams("xy", remove_invisible=False),
        seed=137,
    )
    original = pipeline(**data)
    replayed = A.ReplayCompose.replay(original["replay"], **data)
    for target in data:
        np.testing.assert_array_equal(replayed[target], original[target])


def test_trace_matches_nested_replay_compose_execution() -> None:
    def make_pipeline() -> A.Compose:
        return A.Compose(
            [A.ReplayCompose([A.HorizontalFlip(p=1.0)], p=1.0)],
            seed=137,
        )

    image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)

    expected = make_pipeline()(image=image)
    traced = make_pipeline().run_with_trace(image=image)

    np.testing.assert_array_equal(traced.data["image"], expected["image"])
    assert traced.data["replay"] == expected["replay"]


@pytest.mark.parametrize("traced", [False, True], ids=["ordinary", "traced"])
def test_trace_validates_nested_additional_target_sources(traced: bool) -> None:
    leaf = A.RandomBrightnessContrast(p=1.0)
    pipeline = A.Compose(
        [A.Compose([leaf], additional_targets={"image2": "image"}, p=1.0)],
        seed=137,
        save_applied_params=True,
    )
    image = np.arange(45, dtype=np.uint8).reshape(3, 5, 3)
    pipeline(image=image, image2=image.copy())
    assert leaf.get_applied_params()
    assert leaf.get_applied_config()

    call = pipeline.run_with_trace if traced else pipeline
    with pytest.raises(ValueError, match="Additional target 'image2' requires canonical target 'image'"):
        call(image2=image)

    assert leaf.get_applied_params() == {}
    assert leaf.get_applied_config() == {}


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


@pytest.mark.parametrize("root_type", [A.Compose, A.ReplayCompose])
def test_trace_options_reject_invalid_snapshot_configuration(root_type: type[A.Compose]) -> None:
    with pytest.raises(TypeError, match="not a string"):
        A.TraceOptions(snapshot_targets="image")

    leaf = A.NoOp(p=1.0)
    pipeline = root_type([leaf], seed=137)
    with pytest.raises(ValueError, match="Unknown trace snapshot targets: missing"):
        pipeline.run_with_trace(
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            options=A.TraceOptions(snapshot_targets=("missing",)),
        )

    assert leaf.get_applied_params() == {}
    assert leaf.get_applied_config() == {}


@pytest.mark.parametrize("root_type", [A.Compose, A.ReplayCompose])
def test_trace_options_constructor_failure_preserves_previous_observation(root_type: type[A.Compose]) -> None:
    leaf = A.RandomBrightnessContrast(p=1.0)
    pipeline = root_type([leaf], seed=137, save_applied_params=True)
    pipeline(image=np.arange(45, dtype=np.uint8).reshape(3, 5, 3))
    prior_params = copy.deepcopy(leaf.get_applied_params())
    prior_config = copy.deepcopy(leaf.get_applied_config())
    assert prior_params
    assert prior_config

    with pytest.raises(TypeError, match="not a string"):
        A.TraceOptions(snapshot_targets="image")

    assert_contract_values_equal(leaf.get_applied_params(), prior_params)
    assert_contract_values_equal(leaf.get_applied_config(), prior_config)


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


def test_trace_forces_nested_one_of_selected_by_one_or_other() -> None:
    def make_pipeline() -> A.Compose:
        return A.Compose(
            [
                A.OneOrOther(
                    A.OneOf([A.HorizontalFlip(p=1.0)], p=0.0),
                    A.NoOp(p=1.0),
                    p=1.0,
                ),
            ],
            seed=137,
        )

    image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    expected = make_pipeline()(image=image)
    traced = make_pipeline().run_with_trace(image=image)

    np.testing.assert_array_equal(traced.data["image"], expected["image"])


def test_trace_runs_nested_compose_preprocess() -> None:
    pipeline = A.Compose(
        [A.Compose([A.NoOp(p=1.0)], is_check_shapes=True, p=1.0)],
        is_check_shapes=False,
        seed=137,
    )
    data = {
        "image": np.zeros((3, 5, 3), dtype=np.uint8),
        "mask": np.zeros((2, 5), dtype=np.uint8),
    }

    with pytest.raises(ValueError, match="Height and Width"):
        pipeline(**data)

    with pytest.raises(ValueError, match="Height and Width"):
        pipeline.run_with_trace(**data)


def test_trace_reuses_the_root_prepare_and_finalize_boundary_for_nested_compose() -> None:
    nested = A.Compose([A.NoOp(p=1.0)], seed=137)
    pipeline = A.Compose([nested], seed=137)
    image = np.zeros((3, 5, 3), dtype=np.uint8)

    with mock.patch.object(nested, "preprocess", wraps=nested.preprocess) as preprocess:
        with mock.patch.object(nested, "postprocess", wraps=nested.postprocess) as postprocess:
            traced = pipeline.run_with_trace(image=image)

    np.testing.assert_array_equal(traced.data["image"], image)
    preprocess.assert_not_called()
    postprocess.assert_not_called()


def test_replay_trace_marks_skipped_some_of_as_skipped_replay() -> None:
    image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    original = A.ReplayCompose(
        [A.SomeOf([A.HorizontalFlip(p=1.0)], n=1, p=0.0)],
        seed=137,
    ).run_with_trace(image=image)

    replayed = A.ReplayCompose.replay_with_trace(original.data["replay"], image=image)

    np.testing.assert_array_equal(replayed.data["image"], original.data["image"])
    assert [(record.node_path, record.status) for record in replayed.records] == [
        ((0,), "skipped_replay"),
        ((0, 0), "skipped_replay"),
        ((), "applied"),
    ]


def test_trace_keeps_selective_channel_post_transform_replacement() -> None:
    class ReplacingPostTransformCompose(A.Compose):
        def check_data_post_transform(self, data: dict[str, object]) -> dict[str, object]:
            return {**data, "post_transform_marker": "preserved"}

    pipeline = ReplacingPostTransformCompose(
        [A.SelectiveChannelTransform([A.InvertImg(p=1.0)], channels=(1,), p=1.0)],
        seed=137,
    )

    traced = pipeline.run_with_trace(image=np.arange(12, dtype=np.uint8).reshape(2, 2, 3))

    assert traced.data["post_transform_marker"] == "preserved"


def test_trace_reconstructs_selective_snapshots_through_nested_compositions() -> None:
    def make_pipeline() -> A.Compose:
        return A.Compose(
            [
                A.SelectiveChannelTransform(
                    [A.Compose([A.InvertImg(p=1.0)], p=1.0)],
                    channels=(1,),
                    p=1.0,
                ),
            ],
            seed=137,
        )

    image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    expected = make_pipeline()(image=image)
    traced = make_pipeline().run_with_trace(image=image, options=A.TraceOptions(snapshot_targets=("image",)))

    np.testing.assert_array_equal(traced.data["image"], expected["image"])
    leaf_record = next(record for record in traced.records if record.node_kind == "leaf")
    assert leaf_record.node_path == (0, 0, 0)
    np.testing.assert_array_equal(leaf_record.snapshot["image"], expected["image"])


def test_replay_trace_marks_skipped_sequential_as_skipped_replay() -> None:
    image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    original = A.ReplayCompose(
        [A.Sequential([A.HorizontalFlip(p=1.0)], p=0.0)],
        seed=137,
    ).run_with_trace(image=image)

    replayed = A.ReplayCompose.replay_with_trace(original.data["replay"], image=image)

    np.testing.assert_array_equal(replayed.data["image"], original.data["image"])
    assert [(record.node_path, record.status) for record in replayed.records] == [
        ((0,), "skipped_replay"),
        ((0, 0), "skipped_replay"),
        ((), "applied"),
    ]
