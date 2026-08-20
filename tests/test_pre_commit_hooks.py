"""Tests for repository-specific pre-commit policy hooks."""

from __future__ import annotations

from pathlib import Path

from tools import (
    check_apply_method_length,
    check_internal_workspace,
    check_local_markdown_links,
    check_no_defaults_in_schemas,
    check_range_parameter_annotations,
    check_removed_sampling_hooks,
    check_transform_init_args_override,
)


def test_apply_method_length_hook_excludes_docstrings_and_comments(tmp_path: Path) -> None:
    source = tmp_path / "thin.py"
    source.write_text(
        "class Thin:\n"
        "    def apply(self, image):\n"
        '        """A deliberately long docstring.\n'
        "\n"
        "        It does not count toward the body limit.\n"
        '        """\n'
        "        # Nor does this standalone comment.\n"
        "        result = image  # This is code and must count.\n"
        "        return result\n",
        encoding="utf-8",
    )

    assert check_apply_method_length.collect_errors((source,)) == []


def test_apply_method_length_hook_rejects_long_transform_method(tmp_path: Path) -> None:
    source = tmp_path / "long.py"
    source_text = (
        "class Long:\n"
        "    def apply(self, image):\n"
        "        value = image\n" + "".join("        value = value\n" for _ in range(18)) + "        return value\n"
    )
    source.write_text(source_text, encoding="utf-8")

    assert check_apply_method_length.collect_errors((source,)) == []

    source.write_text(
        source_text.replace("        return value\n", "        value = value\n        return value\n"), encoding="utf-8"
    )

    assert check_apply_method_length.collect_errors((source,)) == [
        (
            f"{source}:2: Long.apply has 21 code-bearing body lines; limit is 20. "
            "Move image arithmetic and routing into a functional helper."
        ),
    ]


def test_apply_method_length_hook_excludes_compose_orchestration(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(check_apply_method_length, "REPO_ROOT", tmp_path)
    source = tmp_path / "albumentations/core/composition.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class Compose:\n"
        "    def apply_in_invocation(self, data):\n"
        "        value = data\n" + "".join("        value = value\n" for _ in range(15)) + "        return value\n",
        encoding="utf-8",
    )

    assert check_apply_method_length.collect_errors((source,)) == []


def test_apply_method_length_hook_excludes_base_classes(tmp_path: Path) -> None:
    source = tmp_path / "base.py"
    method_body = (
        "        value = image\n" + "".join("        value = value\n" for _ in range(20)) + "        return value\n"
    )
    source.write_text(
        "class BaseTransform:\n"
        "    def apply(self, image):\n"
        + method_body
        + "\nclass BaseMaxSizeTransform:\n"
        + "    def apply(self, image):\n"
        + method_body
        + "\nclass BaseTensorTransform:\n"
        + "    def apply(self, image):\n"
        + method_body,
        encoding="utf-8",
    )

    assert check_apply_method_length.collect_errors((source,)) == []


def test_apply_method_length_hook_requires_base_prefix(tmp_path: Path) -> None:
    source = tmp_path / "non_concrete.py"
    source.write_text(
        "class MaxSizeTransform:\n"
        "    def apply(self, image):\n"
        "        value = image\n" + "".join("        value = value\n" for _ in range(20)) + "        return value\n",
        encoding="utf-8",
    )

    assert check_apply_method_length.collect_errors((source,)) == [
        (
            f"{source}:2: MaxSizeTransform.apply has 22 code-bearing body lines; limit is 20. "
            "Move image arithmetic and routing into a functional helper."
        ),
    ]


def test_schema_default_hook_rejects_default_factory(tmp_path: Path) -> None:
    source = tmp_path / "schema.py"
    source.write_text(
        "from pydantic import BaseModel, Field\n"
        "class Schema(BaseModel):\n"
        "    values: list[int] = Field(default_factory=list)\n",
        encoding="utf-8",
    )

    assert check_no_defaults_in_schemas.check_file(source) == [
        (str(source), 3, "Field 'values' in BaseModel class 'Schema' has a default value"),
    ]


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
