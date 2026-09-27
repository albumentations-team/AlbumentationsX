"""Frame ownership carried through ordinary target handlers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import torch

if TYPE_CHECKING:
    from .composition import Compose


ValuesType = TypeVar("ValuesType", bound=np.ndarray | torch.Tensor)


@dataclass(frozen=True, slots=True)
class TargetResult(Generic[ValuesType]):
    """Return transformed values and their explicit row ownership."""

    values: ValuesType
    frame_ids: np.ndarray | None = None
    instance_ids: np.ndarray | None = None
    source_frame_ids: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class RowIds:
    """Keep row ownership without retaining a previous handler's pixel arrays."""

    frame_ids: np.ndarray
    instance_ids: np.ndarray | None = None


@dataclass(slots=True)
class FrameBinding:
    """Own flattened annotations and reconstruction metadata for one Compose invocation."""

    frames: list[dict[str, Any]]
    sources: np.ndarray
    policy: Compose
    routes: dict[str, str] = field(default_factory=dict)
    rows: dict[str, RowIds] = field(default_factory=dict)
    frame_columns: dict[str, int] = field(default_factory=dict)
    instance_frames: np.ndarray | None = None
    handlers: dict[object, dict[str, Callable[..., Any]]] = field(default_factory=dict)

    @staticmethod
    def empty_parameters(names: Iterable[str]) -> dict[str, None]:
        """Resolve absent binding metadata without adding it to sampled or replayed parameters."""
        return dict.fromkeys(
            {"frame_ids", "instance_ids", "instance_frame_ids", "frame_id_column", "source_frame_ids"}.intersection(
                names
            )
        )

    def functions(self, key: object, configured: Mapping[str, Callable[..., Any]]) -> dict[str, Callable[..., Any]]:
        """Resolve flattened collections without changing the reusable transform's aliases."""
        if key not in self.handlers:
            functions = dict(configured)
            for name, canonical in self.routes.items():
                handler = configured.get(canonical)
                if handler is None:
                    functions.pop(name, None)
                else:
                    functions[name] = handler
            self.handlers[key] = functions
        return self.handlers[key]

    def parameters(self, name: str) -> dict[str, Any]:
        """Supply row IDs and annotation columns as ordinary handler parameters."""
        if name == "images" or self.routes.get(name) == "images":
            return {"source_frame_ids": self.sources}
        if name in self.frame_columns:
            return {
                "frame_id_column": self.frame_columns[name],
                "instance_frame_ids": self.instance_frames,
            }
        row = self.rows.get(name)
        if row is None:
            return {}
        return {
            "frame_ids": row.frame_ids,
            "instance_ids": row.instance_ids,
            "instance_frame_ids": self.instance_frames,
        }

    def accept(self, results: Mapping[str, TargetResult[Any]]) -> None:
        """Store ownership only after all handlers have consumed the incoming IDs."""
        for name, result in results.items():
            if result.source_frame_ids is not None:
                if name == "images":
                    self.sources = result.source_frame_ids
            elif result.frame_ids is not None:
                self.rows[name] = RowIds(result.frame_ids, result.instance_ids)
                if result.instance_ids is not None:
                    self.instance_frames = result.frame_ids
