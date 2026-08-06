"""Tests for the scalar NumPy math pre-commit checker."""

from __future__ import annotations

from pathlib import Path

from tools.check_scalar_numpy_math import check_file


def test_checker_flags_python_float_values(tmp_path: Path) -> None:
    source_path = tmp_path / "scalar.py"
    source_path.write_text(
        """\
import numpy as np
from collections.abc import Mapping

AxisAngles = Mapping[str, float]

def make_matrix(angle: float, angles: AxisAngles) -> None:
    radians = np.deg2rad(angle)
    np.sin(radians)
    np.cos(angles[\"x\"])
""",
    )

    errors = check_file(source_path)

    assert [line_number for line_number, _ in errors] == [7, 8, 9]
    assert all("math." in message for _, message in errors)


def test_checker_permits_numpy_arrays_and_unknown_numpy_scalars(tmp_path: Path) -> None:
    source_path = tmp_path / "arrays.py"
    source_path.write_text(
        """\
import numpy as np

values = np.arange(10, dtype=np.float32)
np.sin(values)
percentile = np.percentile(values, 50)
np.cos(percentile)
""",
    )

    assert check_file(source_path) == []
