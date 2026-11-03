"""Observe array owners without extending their lifetime."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch

from albumentations.core.invocation import clear_completed_observation


def watch_array_references(value: Any, references: list[weakref.ReferenceType[Any]]) -> None:
    if isinstance(value, (np.ndarray, torch.Tensor)):
        references.append(weakref.ref(value))
        base = value.base if isinstance(value, np.ndarray) else value._base
        if base is not None:
            watch_array_references(base, references)
    elif isinstance(value, dict):
        for item in value.values():
            watch_array_references(item, references)
    elif isinstance(value, (tuple, list)):
        for item in value:
            watch_array_references(item, references)


@pytest.fixture
def refcount_only() -> Iterator[None]:
    """Require reference counting to release buffers without production GC calls."""
    was_enabled = gc.isenabled()
    clear_completed_observation()
    gc.collect()
    gc.disable()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(gc, "collect", lambda *args, **kwargs: pytest.fail("collection inside lifetime oracle"))
            yield
            assert not gc.isenabled()
    finally:
        clear_completed_observation()
        if was_enabled:
            gc.enable()
        else:
            gc.disable()
        gc.collect()
