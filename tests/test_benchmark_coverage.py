from __future__ import annotations

from typing import Any

from tools.benchmark_coverage import coverage_details


def _coverage_for(transform_name: str) -> dict[str, Any]:
    details = coverage_details()
    transforms = {item["name"]: item for item in details["transforms"]}
    return transforms[transform_name]


def test_benchmark_coverage_details_account_for_every_public_transform() -> None:
    details = coverage_details()

    assert details["public_transforms"] == len(details["transforms"])
    assert (
        details["layer_counts"]["catalog_smoke"] + details["layer_counts"]["optional"] == details["public_transforms"]
    )


def test_benchmark_coverage_details_expose_deep_hot_path_layers() -> None:
    resize = _coverage_for("Resize")

    assert resize["smoke_only"] is False
    assert {"annotation_scaling", "catalog_smoke", "direct_kernel", "family_matrix", "memory"}.issubset(
        resize["layers"],
    )


def test_benchmark_coverage_details_map_volumetric_matrix_to_public_transforms() -> None:
    center_crop3d = _coverage_for("CenterCrop3D")

    assert center_crop3d["smoke_only"] is False
    assert "volumetric_matrix" in center_crop3d["layers"]


def test_benchmark_coverage_details_map_expanded_pixel_matrix_to_public_transforms() -> None:
    random_rain = _coverage_for("RandomRain")

    assert random_rain["smoke_only"] is False
    assert "family_matrix" in random_rain["layers"]


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
    assert "alias_coverage" in shift_scale_rotate["layers"]


def test_benchmark_coverage_details_keep_optional_transforms_explicit() -> None:
    to_tensor = _coverage_for("ToTensorV2")

    assert to_tensor["benchmark"] is False
    assert to_tensor["layers"] == ["optional"]
    assert "PyTorch" in str(to_tensor["optional_reason"])
