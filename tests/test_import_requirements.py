import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("native_version", "major_minor", "wheel_version", "expected_error"),
    [
        ("5.0.0", (5, 0), "5.0.0.93", "requires OpenCV 5.1"),
        ("5.1.0", (5, 1), "5.1.0.94", r"Found OpenCV 5\.1\.0\.94"),
        ("5.1.0", (5, 1), "5.1.0.95", "requires PyTorch"),
        ("5.1-dev", (5, 1), None, "requires PyTorch"),
    ],
)
def test_import_checks_opencv_before_torch(monkeypatch, native_version, major_minor, wheel_version, expected_error):
    major, minor = major_minor
    cv2 = SimpleNamespace(
        __version__=native_version,
        getVersionMajor=lambda: major,
        getVersionMinor=lambda: minor,
    )
    if wheel_version is not None:
        cv2.version = SimpleNamespace(opencv_version=wheel_version)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "torch", None)
    initializer = Path(__file__).parents[1] / "albumentations" / "__init__.py"

    with pytest.raises(ImportError, match=expected_error):
        runpy.run_path(str(initializer))
