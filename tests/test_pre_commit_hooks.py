"""Tests for repository-specific pre-commit policy hooks."""

from __future__ import annotations

from pathlib import Path

from tools import (
    check_internal_workspace,
    check_local_markdown_links,
    check_quality_suppressions,
    check_range_parameter_annotations,
    check_removed_sampling_hooks,
    check_transform_init_args_override,
)


def test_quality_suppression_hook_rejects_inline_complexity_suppressions(tmp_path: Path) -> None:
    source = tmp_path / "example.py"
    source.write_text("def transform():  # noqa: C901, PLR0912 - temporary\n    pass\n", encoding="utf-8")

    assert check_quality_suppressions.collect_errors((source,)) == [
        f"{source}:1: do not suppress mandatory complexity checks (C901, PLR0912)",
    ]


def test_quality_suppression_hook_rejects_relaxed_production_limits(tmp_path: Path) -> None:
    config = tmp_path / "pyproject.toml"
    config.write_text(
        '[tool.ruff.lint]\nignore = ["C901"]\n\n'
        "[tool.ruff.lint.pylint]\nmax-branches = 13\n\n"
        'lint.per-file-ignores."albumentations/example.py" = ["PLR0912"]\n\n'
        "[tool.ruff.lint.per-file-ignores]\n"
        '"albumentations/other.py" = ["C901"]\n',
        encoding="utf-8",
    )

    assert check_quality_suppressions.collect_errors((config,)) == [
        f"{config}: do not ignore mandatory complexity checks (C901)",
        f"{config}: do not ignore mandatory complexity checks for albumentations/example.py (PLR0912)",
        f"{config}: do not ignore mandatory complexity checks for albumentations/other.py (C901)",
        f"{config}:5: max-branches must not exceed 12",
    ]


def test_quality_suppression_hook_keeps_test_complexity_policy(tmp_path: Path) -> None:
    config = tmp_path / "pyproject.toml"
    config.write_text('lint.per-file-ignores."tests/*" = ["C901", "PLR0912"]\n', encoding="utf-8")

    assert check_quality_suppressions.collect_errors((config,)) == []


def test_internal_workspace_rejects_everything_except_gitkeep() -> None:
    errors = check_internal_workspace.collect_errors(
        ("_internal/.gitkeep", "_internal/scratch/notes.md", "docs/guide.md"),
    )

    assert errors == ["_internal/scratch/notes.md: _internal/ is local-only and must not be committed"]


def test_transform_init_args_override_allows_only_the_base_implementation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(check_transform_init_args_override, "REPO_ROOT", tmp_path)
    base = tmp_path / "albumentations/core/transforms_interface.py"
    invalid = tmp_path / "albumentations/augmentations/example.py"
    base.parent.mkdir(parents=True)
    invalid.parent.mkdir(parents=True)
    base.write_text(
        "class BasicTransform:\n    def get_transform_init_args_names(self):\n        return ()\n",
        encoding="utf-8",
    )
    invalid.write_text(
        "class Example:\n    def get_transform_init_args_names(self):\n        return ()\n",
        encoding="utf-8",
    )

    assert check_transform_init_args_override.collect_errors((base,)) == []
    assert check_transform_init_args_override.collect_errors((invalid,)) == [
        (
            "albumentations/augmentations/example.py:2: get_transform_init_args_names() is inherited from "
            "BasicTransform; do not override it"
        ),
    ]


def test_range_parameter_annotations_require_pair_types(tmp_path: Path) -> None:
    valid = tmp_path / "valid.py"
    invalid = tmp_path / "invalid.py"
    valid.write_text(
        "class Transform:\n"
        "    def __init__(\n"
        "        self,\n"
        "        blur_range: tuple[float, float],\n"
        "        size_range: tuple[int, int] | tuple[float, float],\n"
        "        optional_range: tuple[int, int] | None,\n"
        "        axis_range: dict[str, tuple[float, float]],\n"
        "        optional_axis_range: dict[str, tuple[float, float] | None],\n"
        "        named_range: PixelLengthRange | None,\n"
        "        mask_range: MaskLengthRange,\n"
        "        axis_ranges: AxisRanges3D,\n"
        "        positive_axis_ranges: PositiveAxisRanges3D,\n"
        "    ):\n"
        "        pass\n",
        encoding="utf-8",
    )
    invalid.write_text(
        "class Transform:\n    def __init__(self, blur_range: float | tuple[float, float]):\n        pass\n",
        encoding="utf-8",
    )

    assert check_range_parameter_annotations.collect_errors((valid,)) == []
    assert check_range_parameter_annotations.collect_errors((invalid,)) == [
        f"{invalid}:2: `blur_range` must describe a pair: tuple[T, T], an optional pair form, or an axis-to-pair map",
    ]


def test_removed_sampling_hooks_rejects_only_method_declarations() -> None:
    source = """
class Current:
    def sample_parameters(self, params, data, sampling):
        return params


class Obsolete:
    def get_params(self):
        return {}

    async def get_params_dependent_on_data(self, params, data):
        return params
"""

    assert check_removed_sampling_hooks.find_removed_sampling_hooks(source, "example.py") == [
        ("example.py:8: Obsolete.get_params was removed; implement sample_parameters(params, data, sampling) instead"),
        (
            "example.py:11: Obsolete.get_params_dependent_on_data was removed; "
            "implement sample_parameters(params, data, sampling) instead"
        ),
    ]


def test_local_markdown_links_ignore_external_urls_and_fenced_examples(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(check_local_markdown_links, "REPO_ROOT", tmp_path)
    source = tmp_path / "docs/guide.md"
    target = tmp_path / "docs/target.md"
    source.parent.mkdir(parents=True)
    target.write_text("target", encoding="utf-8")
    source.write_text(
        "[Local](target.md#section)\n"
        "[Root-relative](/docs/target.md#section)\n"
        "[External](https://albumentations.ai/docs/)\n"
        "```markdown\n"
        "[Example](not-a-real-file.md)\n"
        "```\n",
        encoding="utf-8",
    )

    assert check_local_markdown_links.collect_errors((source,)) == []


def test_local_markdown_links_report_missing_and_escaping_targets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(check_local_markdown_links, "REPO_ROOT", tmp_path)
    source = tmp_path / "docs/guide.md"
    source.parent.mkdir(parents=True)
    source.write_text("[Missing](missing.md)\n[Outside](../../outside.md)\n", encoding="utf-8")

    assert check_local_markdown_links.collect_errors((source,)) == [
        "docs/guide.md:1: local link target does not exist: missing.md",
        "docs/guide.md:2: local link escapes the repository: ../../outside.md",
    ]
