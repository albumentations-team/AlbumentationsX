"""PyTorch packaging and import-contract regression tests."""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest

import albumentations


def test_import_without_torch_names_the_installation_extra() -> None:
    original_import = __import__

    def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", import_without_torch):
        with pytest.raises(ImportError) as error:
            importlib.reload(albumentations)

    importlib.reload(albumentations)

    assert 'pip install "albumentationsx[headless]"' in str(error.value)


def test_import_propagates_torch_loader_errors() -> None:
    original_import = __import__

    def import_with_broken_torch(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("simulated Torch loader failure")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", import_with_broken_torch):
        with pytest.raises(ImportError, match="simulated Torch loader failure") as error:
            importlib.reload(albumentations)

    importlib.reload(albumentations)

    assert "Install the PyTorch build" not in str(error.value)
