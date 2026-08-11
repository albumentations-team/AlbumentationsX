"""Contract tests for explicit transform sampling state."""

from __future__ import annotations

import ast
from pathlib import Path


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
