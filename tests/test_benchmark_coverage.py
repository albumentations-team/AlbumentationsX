from __future__ import annotations

import copy
from typing import Any

import pytest

from tools import benchmark_coverage
from tools.benchmark_coverage import coverage_details, coverage_diff


def _coverage_for(transform_name: str) -> dict[str, Any]:
    details = coverage_details()
    transforms = {item["name"]: item for item in details["transforms"]}
    return transforms[transform_name]


def test_benchmark_coverage_details_account_for_every_public_transform() -> None:
    details = coverage_details()

    assert details["schema_version"] == 5
    assert details["public_transforms"] == len(details["transforms"])
    assert (
        details["layer_counts"]["catalog_smoke"] + details["layer_counts"]["optional"] == details["public_transforms"]
    )
    assert details["summary"]["contract_failures"] == 0
    assert details["contract_failures"] == []
    assert details["summary"]["performance_contract_status_counts"]["batch"]["covered"] == 13
    assert details["summary"]["performance_contract_status_counts"]["parameter_sensitivity"]["covered"] == 8


def test_benchmark_coverage_details_expose_deep_hot_path_layers() -> None:
    resize = _coverage_for("Resize")

    assert resize["smoke_only"] is False
    assert resize["class"] == {
        "module": "albumentations.augmentations.geometric.resize",
        "public_api": "albumentations.Resize",
        "qualname": "Resize",
    }
    assert resize["benchmark_spec"]["constructor_params"] == {"height": 128, "width": 128}
    assert resize["benchmark_spec"]["route"] == "image"
    assert {"annotation_scaling", "direct_kernel", "family_matrix", "geometry", "memory"}.issubset(
        resize["families"],
    )
    assert {"annotation_scaling", "catalog_smoke", "direct_kernel", "family_matrix", "memory"}.issubset(
        resize["layers"],
    )
    assert {
        ("catalog_smoke", "Resize"),
        ("family_matrix", "resize|small|1|uint8"),
        ("direct_kernel", "resize|small|1|uint8"),
        ("memory", "peakmem_resize_large_rgb"),
    }.issubset({(case["layer"], case["case_id"]) for case in resize["asv_cases"]})
    assert resize["scenario_contract"]["sizes"] == ["small", "medium", "large"]
    assert resize["scenario_contract"]["channels"] == [1, 3, 5]
    assert resize["scenario_contract"]["dtypes"] == ["uint8", "float32"]
    assert resize["scenario_contract"]["annotation_counts"] == [10, 100, 1000]
    assert resize["scenario_contract"]["batch_sizes"] == [4, 8]
    assert resize["scenario_axis_contracts"]["family_matrix"]["skipped"] == {}
    assert resize["scenario_axis_contracts"]["batch_matrix"]["skipped"] == {"sizes": ["large"]}
    assert resize["performance_contract"]["annotation"]["status"] == "covered"
    assert resize["performance_contract"]["batch"]["status"] == "covered"
    assert resize["performance_contract"]["direct_kernel"]["status"] == "covered"
    assert resize["performance_contract"]["memory"]["status"] == "covered_advisory"
    assert resize["performance_contract"]["parameter_sensitivity"]["status"] == "not_required"
    assert resize["performance_contract"]["batch"]["required_layers"] == ["batch_matrix"]
    assert {"bboxes", "image", "images", "masks"}.issubset(resize["scenario_contract"]["targets"])
    assert "peakmem_resize_large_rgb" in resize["scenario_contract"]["memory_cases"]
    assert {"geometry_annotation", "geometry_image"}.issubset(
        resize["scenario_contract"]["direct_kernel_groups"],
    )


def test_benchmark_coverage_details_map_volumetric_matrix_to_public_transforms() -> None:
    center_crop3d = _coverage_for("CenterCrop3D")

    assert center_crop3d["smoke_only"] is False
    assert "volumetric_matrix" in center_crop3d["layers"]


def test_benchmark_coverage_details_map_batch_matrix_to_public_transforms() -> None:
    horizontal_flip = _coverage_for("HorizontalFlip")

    assert horizontal_flip["smoke_only"] is False
    assert "batch_matrix" in horizontal_flip["layers"]
    assert horizontal_flip["performance_contract"]["batch"]["status"] == "covered"
    assert set(horizontal_flip["performance_contract"]["batch"]["implementation_methods"]) == {
        "apply_to_images",
        "apply_to_masks",
        "apply_to_masks3d",
        "apply_to_volumes",
    }
    assert {"images", "masks", "masks3d", "volumes"}.issubset(horizontal_flip["scenario_contract"]["targets"])
    assert {
        ("batch_matrix", "horizontal_flip|images|small|1|uint8|4"),
        ("batch_matrix", "horizontal_flip|images_and_masks|small|1|uint8|4"),
        ("batch_matrix", "horizontal_flip|volumes_and_masks3d|small|1|uint8|2"),
    }.issubset({(case["layer"], case["case_id"]) for case in horizontal_flip["asv_cases"]})


def test_benchmark_coverage_details_map_spatter_modes_to_batch_matrix() -> None:
    spatter = _coverage_for("Spatter")

    assert spatter["performance_contract"]["batch"]["status"] == "covered"
    assert spatter["scenario_contract"]["channels"] == [3]
    assert spatter["scenario_contract"]["dtypes"] == ["uint8", "float32"]
    assert spatter["scenario_contract"]["batch_sizes"] == [2, 4, 8, 16]
    assert spatter["scenario_contract"]["sizes"] == ["small", "medium", "large"]
    assert {"compose_batch", "direct_batch"}.issubset(spatter["scenario_contract"]["scopes"])
    assert "peakmem_spatter_batch_large_rgb" in spatter["scenario_contract"]["memory_cases"]
    assert (
        "memory",
        "benchmarks.test_batch_matrix.PeakMemorySpatterBatchMatrix.peakmem_spatter_batch_large_rgb",
    ) in {(case["layer"], case["benchmark"]) for case in spatter["asv_cases"]}
    assert {
        "spatter_mud|direct_images|large|3|float32|16",
        "spatter_mud|images|small|3|uint8|4",
        "spatter_rain|direct_images|small|3|uint8|2",
        "spatter_mud|images|medium|3|float32|8",
        "spatter_rain|images|small|3|uint8|4",
        "spatter_rain|images|large|3|float32|16",
    }.issubset({case["case_id"] for case in spatter["asv_cases"] if case["layer"] == "batch_matrix"})


def test_benchmark_coverage_details_map_parameter_sensitivity_to_public_transforms() -> None:
    blur = _coverage_for("Blur")
    superpixels = _coverage_for("Superpixels")

    assert "parameter_sensitivity" in blur["layers"]
    assert blur["performance_contract"]["parameter_sensitivity"]["status"] == "covered"
    assert set(blur["scenario_contract"]["parameter_scenarios"]) == {"kernel_3", "kernel_15"}
    assert blur["scenario_contract"]["sizes"] == ["small", "medium", "large"]
    assert blur["scenario_contract"]["channels"] == [1, 3, 5]
    assert blur["scenario_contract"]["dtypes"] == ["uint8", "float32"]
    assert blur["scenario_axis_contracts"]["parameter_sensitivity"]["skipped"] == {
        "channels": [1, 5],
        "sizes": ["large"],
    }
    assert ("parameter_sensitivity", "blur_kernel_15|kernel_15|medium|3|float32") in {
        (case["layer"], case["case_id"]) for case in blur["asv_cases"]
    }
    assert any(
        case["scenario"].get("parameter_values", {}).get("blur_range") == [15, 15]
        for case in blur["asv_cases"]
        if case["layer"] == "parameter_sensitivity"
    )
    assert "parameter_sensitivity" in superpixels["layers"]
    assert superpixels["performance_contract"]["parameter_sensitivity"]["status"] == "covered"
    assert set(superpixels["scenario_contract"]["parameter_scenarios"]) == {"segments_32", "segments_128"}
    assert superpixels["scenario_contract"]["dtypes"] == ["uint8"]
    assert superpixels["scenario_axis_contracts"]["family_matrix"]["skipped"] == {
        "channels": [1, 5],
        "dtypes": ["float32"],
    }


def test_benchmark_coverage_details_map_expanded_pixel_matrix_to_public_transforms() -> None:
    random_rain = _coverage_for("RandomRain")

    assert random_rain["smoke_only"] is False
    assert "family_matrix" in random_rain["layers"]


def test_benchmark_coverage_details_require_family_matrix_for_image_transforms() -> None:
    details = coverage_details()

    missing_family_matrix = [
        item["name"]
        for item in details["transforms"]
        if item["route"] == "image" and "alias_coverage" not in item["layers"] and "family_matrix" not in item["layers"]
    ]

    assert missing_family_matrix == []


def test_benchmark_coverage_details_map_crop_and_dropout_matrix_to_public_transforms() -> None:
    random_resized_crop = _coverage_for("RandomResizedCrop")
    channel_dropout = _coverage_for("ChannelDropout")

    assert random_resized_crop["smoke_only"] is False
    assert channel_dropout["smoke_only"] is False
    assert "family_matrix" in random_resized_crop["layers"]
    assert "family_matrix" in channel_dropout["layers"]


def test_benchmark_coverage_details_map_special_target_matrix_to_public_transforms() -> None:
    bbox_safe_crop = _coverage_for("BBoxSafeRandomCrop")
    mask_dropout = _coverage_for("MaskDropout")

    assert bbox_safe_crop["smoke_only"] is False
    assert mask_dropout["smoke_only"] is False
    assert "target_matrix" in bbox_safe_crop["layers"]
    assert "target_matrix" in mask_dropout["layers"]


def test_benchmark_coverage_details_explain_warning_aliases() -> None:
    shift_scale_rotate = _coverage_for("ShiftScaleRotate")

    assert shift_scale_rotate["smoke_only"] is False
    assert shift_scale_rotate["covered_by"] == "Affine"
    assert shift_scale_rotate["families"] == ["alias", "alias_coverage"]
    assert "alias_coverage" in shift_scale_rotate["layers"]
    assert shift_scale_rotate["asv_cases"] == [
        {
            "benchmark": "benchmarks.test_catalog_smoke.TimeCatalogTransformSmoke.time_transform_compose",
            "case_id": "ShiftScaleRotate",
            "config": "default",
            "layer": "catalog_smoke",
            "scenario": {
                "layer": "catalog_smoke",
                "scope": "compose",
                "targets": ["image"],
            },
        },
    ]
    assert shift_scale_rotate["scenario_contract"]["case_count"] == 1
    assert shift_scale_rotate["scenario_contract"]["targets"] == ["image"]
    assert shift_scale_rotate["performance_contract"]["direct_kernel"]["status"] == "not_required"


@pytest.mark.pytorch
def test_benchmark_coverage_details_keep_optional_transforms_explicit() -> None:
    to_tensor = _coverage_for("ToTensorV2")

    assert to_tensor["benchmark"] is False
    assert to_tensor["class"] == {
        "module": "albumentations.pytorch.transforms",
        "public_api": "albumentations.ToTensorV2",
        "qualname": "ToTensorV2",
    }
    assert to_tensor["layers"] == ["optional", "pytorch_tensor"]
    assert to_tensor["families"] == ["pytorch_tensor"]
    assert to_tensor["coverage_contract"]["status"] == "ok"
    assert to_tensor["coverage_contract"]["required_layers"] == ["optional", "pytorch_tensor"]
    assert to_tensor["performance_contract"]["batch"]["status"] == "covered_optional"
    assert "PyTorch" in str(to_tensor["optional_reason"])
    assert {
        ("pytorch", "pytorch_tensor", "small|1|uint8"),
        ("pytorch", "pytorch_tensor", "large|5|float32"),
    }.issubset({(case["config"], case["layer"], case["case_id"]) for case in to_tensor["asv_cases"]})
    assert to_tensor["scenario_contract"]["batch_sizes"] == [8]
    assert to_tensor["scenario_contract"]["configs"] == ["pytorch"]
    assert to_tensor["scenario_contract"]["targets"] == ["image", "images", "mask", "masks"]


def test_registry_allows_optional_transforms_only_when_their_dependency_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optional = set(benchmark_coverage.OPTIONAL_BENCHMARK_TRANSFORMS)
    spec_names = set(benchmark_coverage.benchmark_specs()) - optional
    public_names = set(benchmark_coverage.public_transform_names()) - optional

    monkeypatch.setattr(benchmark_coverage, "unavailable_optional_transform_names", lambda: optional)
    assert benchmark_coverage._validate_public_registry(public_names, spec_names) == []
    assert benchmark_coverage._validate_coverage_layers(spec_names) == []

    monkeypatch.setattr(benchmark_coverage, "unavailable_optional_transform_names", set)
    registry_errors = benchmark_coverage._validate_public_registry(public_names, spec_names)
    layer_errors = benchmark_coverage._validate_coverage_layers(spec_names)

    assert registry_errors == ["Optional benchmark transform is not public: ToTensor3D, ToTensorV2"]
    assert layer_errors == ["Benchmark coverage layers reference unknown transforms: ToTensor3D, ToTensorV2"]


def test_benchmark_coverage_details_include_reviewable_case_metadata_for_all_layers() -> None:
    details = coverage_details()

    for transform in details["transforms"]:
        assert transform["class"]["module"].startswith("albumentations.")
        assert transform["class"]["public_api"] == f"albumentations.{transform['name']}"
        assert transform["benchmark_spec"]["route"] == transform["route"]
        assert transform["asv_cases"] or transform["layers"] == ["optional"]
        assert transform["scenario_contract"]["case_count"] == len(transform["asv_cases"])
        assert set(transform["scenario_contract"]["layers"]).issubset(transform["layers"])
        assert set(transform["scenario_axis_contracts"]).issubset(transform["layers"])
        assert set(transform["performance_contract"]) == {
            "annotation",
            "batch",
            "direct_kernel",
            "memory",
            "parameter_sensitivity",
        }
        for axis_contract in transform["performance_contract"].values():
            assert axis_contract["reason"]
            assert isinstance(axis_contract["implementation_methods"], list)
            assert set(axis_contract["required_layers"]).issubset(transform["layers"])
        for axis_contract in transform["scenario_axis_contracts"].values():
            if axis_contract["skipped"]:
                assert axis_contract["skip_reason"]
        for case in transform["asv_cases"]:
            assert case["benchmark"]
            assert case["case_id"]
            assert case["config"] in {"default", "pytorch"}
            assert case["layer"] in transform["layers"]
            assert case["scenario"]["layer"] == case["layer"]
            assert case["scenario"]["scope"]
            assert isinstance(case["scenario"]["targets"], list)


def test_benchmark_coverage_details_track_batch_and_annotation_audit_paths() -> None:
    auto_contrast = _coverage_for("AutoContrast")
    crop_and_pad = _coverage_for("CropAndPad")

    assert auto_contrast["performance_contract"]["batch"] == {
        "implementation_methods": ["apply_to_images", "apply_to_volumes"],
        "reason": (
            "custom batch methods are inventoried for review; current release-critical evidence comes from "
            "catalog smoke, family matrices, direct kernels, and core batch dispatch until this route is promoted"
        ),
        "required_layers": [],
        "status": "tracked_without_dedicated_matrix",
    }
    assert crop_and_pad["performance_contract"]["annotation"]["status"] == "tracked_without_dedicated_scaling"
    assert crop_and_pad["performance_contract"]["annotation"]["implementation_methods"] == [
        "apply_to_bboxes",
        "apply_to_keypoints",
    ]


def test_benchmark_coverage_diff_reports_catalog_and_case_drift() -> None:
    base = copy.deepcopy(coverage_details())
    base["transforms"] = [item for item in base["transforms"] if item["name"] != "Resize"]
    base["transforms"].append(
        {
            "asv_cases": [],
            "coverage_contract": {"status": "ok"},
            "layers": ["catalog_smoke"],
            "name": "RemovedTransform",
            "route": "image",
            "scenario_axis_contracts": {},
        },
    )
    for item in base["transforms"]:
        if item["name"] == "Blur":
            item["layers"].remove("parameter_sensitivity")
            item["asv_cases"] = [case for case in item["asv_cases"] if case["layer"] != "parameter_sensitivity"]
            item["scenario_axis_contracts"].pop("parameter_sensitivity")
            break

    diff = coverage_diff(base)

    assert diff["kind"] == "benchmark-coverage-diff"
    assert diff["summary"]["status"] == "changed"
    assert "Resize" in {item["name"] for item in diff["added_transforms"]}
    assert "RemovedTransform" in {item["name"] for item in diff["removed_transforms"]}
    blur_diff = next(item for item in diff["changed_transforms"] if item["name"] == "Blur")
    assert set(blur_diff["changes"]["layers"]["current"]) == {
        "catalog_smoke",
        "direct_kernel",
        "family_matrix",
        "parameter_sensitivity",
    }
    assert blur_diff["changes"]["asv_cases"]["added"]
