"""Vector-field code may import only the Albucore names that keep NaN exactly.

Fields are the only AX data containing NaN. Albucore guarantees NaN exactness for `remap`, `remap3d` and the
`albucore.tensor_bridge` layout helpers only; every other Albucore operation is image, mask and volume only.
See docs/vector-fields.md.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "albumentations"
TENSOR_BRIDGE_NAMES = frozenset(
    {
        "numpy_to_tensor_layout",
        "numpy_to_tensor_storage",
        "tensor_to_numpy_channel_last",
        "tensor_to_numpy_storage",
    },
)
ALLOWED_ALBUCORE_NAMES = {
    "augmentations/geometric/_functional_vector_fields.py": frozenset({"albucore.remap", "albucore.remap3d"}),
    "core/tensor.py": frozenset({"albucore.TensorLayout"})
    | {f"albucore.tensor_bridge.{n}" for n in TENSOR_BRIDGE_NAMES},
}


def _imported_albucore_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "albucore":
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names if alias.name.split(".")[0] == "albucore")
    return names


@pytest.mark.parametrize("relative_path", sorted(ALLOWED_ALBUCORE_NAMES))
def test_vector_field_modules_import_only_nan_safe_albucore_names(relative_path: str) -> None:
    imported = _imported_albucore_names(PACKAGE_ROOT / relative_path)

    assert imported, f"{relative_path} no longer imports Albucore; update this boundary test"
    assert imported <= ALLOWED_ALBUCORE_NAMES[relative_path], sorted(imported - ALLOWED_ALBUCORE_NAMES[relative_path])
