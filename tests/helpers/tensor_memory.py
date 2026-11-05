"""CPU tensor allocator measurements, excluding inputs created before profiling."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypeVar

import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


_Result = TypeVar("_Result")


def measure_peak_tensor_bytes(operation: Callable[[], _Result], path: Path) -> tuple[_Result, int]:
    """Return the result and peak live bytes added during the operation."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profile:
        result = operation()
    profile.export_chrome_trace(str(path))
    trace = json.loads(path.read_text())
    events = sorted(
        (event for event in trace["traceEvents"] if event.get("name") == "[memory]"),
        key=lambda event: event["ts"],
    )
    live = peak = 0
    for event in events:
        live += int(event["args"]["Bytes"])
        peak = max(peak, live)
    return result, peak
