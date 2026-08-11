"""Runtime value objects for observing one composition execution."""

import copy
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

__all__ = ["TraceOptions", "TraceRecord", "TraceResult"]


@dataclass(frozen=True, slots=True)
class TraceOptions:
    """Configure one traced composition call with snapshots, timing, retention, and an optional observer, without
    changing output, random state, or portable policy.

    Args:
        snapshot_targets (Sequence[str]): Target names copied into executed-leaf records.
        include_timing (bool): Whether records include execution time in nanoseconds.
        observer (Callable[[TraceRecord], None] | None): Synchronous callback invoked for each completed record.
        collect_records (bool): Whether to retain records in the returned result.

    """

    snapshot_targets: Sequence[str] = ()
    include_timing: bool = False
    observer: Callable[["TraceRecord"], None] | None = field(default=None, compare=False, repr=False)
    collect_records: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.snapshot_targets, str):
            raise TypeError("snapshot_targets must be a sequence of target names, not a string")
        snapshot_targets = tuple(self.snapshot_targets)
        if len(snapshot_targets) != len(set(snapshot_targets)):
            raise ValueError("snapshot_targets must not contain duplicates")
        if any(not isinstance(target, str) for target in snapshot_targets):
            raise TypeError("snapshot_targets must contain only strings")
        if not self.collect_records and self.observer is None:
            raise ValueError("observer-only tracing requires an observer")
        object.__setattr__(self, "snapshot_targets", snapshot_targets)


@dataclass(frozen=True, slots=True)
class TraceRecord:
    """Describe a completed composition or leaf visit with a structural path, outcome, detached parameters,
    snapshots, and optional duration for diagnosis.
    """

    node_path: tuple[int, ...]
    event_index: int
    occurrence_index: int
    class_fullname: str
    node_kind: str
    status: str
    params: Mapping[str, Any] | None
    snapshot: Mapping[str, Any] | None
    elapsed_ns: int | None


@dataclass(frozen=True, slots=True)
class TraceResult:
    """Bundle normal public pipeline output with immutable execution records, keeping observation opt-in without
    changing the ordinary Compose result contract.
    """

    data: dict[str, Any]
    records: tuple[TraceRecord, ...]


class _TraceContext:
    """Manage per-call trace ordering, detached values, and target snapshots privately, keeping mutable bookkeeping
    out of the ordinary Compose hot path.
    """

    def __init__(self, options: TraceOptions):
        self.options = options
        self.records: list[TraceRecord] = []
        self._event_index = 0
        self._occurrences: dict[tuple[int, ...], int] = {}

    @property
    def needs_snapshot(self) -> bool:
        return bool(self.options.snapshot_targets)

    def emit(
        self,
        *,
        node_path: tuple[int, ...],
        class_fullname: str,
        node_kind: str,
        status: str,
        params: dict[str, Any] | None = None,
        data: Mapping[str, Any] | None = None,
        elapsed_ns: int | None = None,
    ) -> None:
        """Create and publish one completed node record with ordered occurrence metadata, keeping requested parameter
        and target copies detached from pipeline state.

        Args:
            node_path (tuple[int, ...]): Structural child-index path of the visited node.
            class_fullname (str): Serialized class identifier of the visited node.
            node_kind (str): Whether the node is a composition or a leaf transform.
            status (str): Applied or skipped execution outcome.
            params (dict[str, Any] | None): Applied leaf parameters to detach into the record.
            data (Mapping[str, Any] | None): Post-step working targets available for requested snapshots.
            elapsed_ns (int | None): Opt-in elapsed leaf duration in nanoseconds.

        """
        occurrence_index = self._occurrences.get(node_path, 0)
        self._occurrences[node_path] = occurrence_index + 1
        snapshot = self._snapshot(data) if data is not None and self.needs_snapshot else None
        record = TraceRecord(
            node_path=node_path,
            event_index=self._event_index,
            occurrence_index=occurrence_index,
            class_fullname=class_fullname,
            node_kind=node_kind,
            status=status,
            params=MappingProxyType(copy.deepcopy(params)) if params is not None else None,
            snapshot=snapshot,
            elapsed_ns=elapsed_ns,
        )
        self._event_index += 1
        if self.options.collect_records:
            self.records.append(record)
        if self.options.observer is not None:
            self.options.observer(record)

    def finish(self, data: dict[str, Any]) -> TraceResult:
        """Return normal public output and retained immutable records, completing this trace context without extra
        data copies in observer-only or metadata modes.

        Args:
            data (dict[str, Any]): Final public targets produced by the traced pipeline call.

        Returns:
            TraceResult: Final output paired with records retained by this context.

        """
        return TraceResult(data=data, records=tuple(self.records))

    def _snapshot(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                target: self._copy_snapshot_value(data[target])
                for target in self.options.snapshot_targets
                if target in data
            },
        )

    @staticmethod
    def _copy_snapshot_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, torch.Tensor):
            return value.clone()
        return copy.deepcopy(value)


class _ExecutionTrace:
    """Attach an opt-in observer to ordinary graph execution so metadata and snapshots describe
    decisions without a duplicate interpreter.

    The session carries structural paths and the snapshot view for the current
    execution branch.  It contains no transform decisions: Compose nodes make
    those decisions exactly once in their normal execution methods and report
    the resulting events here.
    """

    def __init__(self, options: TraceOptions):
        self.context = _TraceContext(options)
        self._node_path: tuple[int, ...] = ()
        self._snapshot_data_factory: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None

    @property
    def node_path(self) -> tuple[int, ...]:
        """Return the active composition path so trace events can be joined with the configured
        graph node that was executing when each event was recorded.
        """
        return self._node_path

    def child_path(self, index: int) -> tuple[int, ...]:
        """Append `index` to the active path so sibling records remain stable even if a selector
        leaves neighboring configured branches unexecuted.
        """
        return (*self._node_path, index)

    @contextmanager
    def node_scope(self, path: tuple[int, ...]) -> Iterator[None]:
        """Set `path` while one configured node dispatches so its leaf, composition, and skip records
        share the same structural graph identity.
        """
        previous_path = self._node_path
        self._node_path = path
        try:
            yield
        finally:
            self._node_path = previous_path

    @contextmanager
    def snapshot_scope(self, factory: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> Iterator[None]:
        """Map nested working data to the public target view so trace snapshots describe values at
        the root boundary instead of internal aliases.
        """
        previous_factory = self._snapshot_data_factory
        self._snapshot_data_factory = factory
        try:
            yield
        finally:
            self._snapshot_data_factory = previous_factory

    def snapshot_data(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
        """Adapt data using the active snapshot scope so observer callbacks receive public target
        names rather than aliases used by nested graph execution.
        """
        return data if self._snapshot_data_factory is None else self._snapshot_data_factory(data)

    def emit_leaf(
        self,
        *,
        class_fullname: str,
        status: str,
        params: dict[str, Any] | None,
        data: Mapping[str, Any] | None,
        elapsed_ns: int | None,
    ) -> None:
        """Publish one leaf result at the active path, adding parameters, snapshots, and timing only
        when enabled by the requested trace options.
        """
        self.context.emit(
            node_path=self._node_path,
            class_fullname=class_fullname,
            node_kind="leaf",
            status=status,
            params=params,
            data=None if data is None else self.snapshot_data(data),
            elapsed_ns=elapsed_ns,
        )

    def emit_composition(self, *, class_fullname: str, status: str) -> None:
        """Publish one composition result at the active path so completed containers can be
        reconstructed with their configured child records.
        """
        self.context.emit(
            node_path=self._node_path,
            class_fullname=class_fullname,
            node_kind="composition",
            status=status,
        )

    def finish(self, data: dict[str, Any]) -> TraceResult:
        """Return immutable trace output after root finalization so final data has the same public
        representation and annotation processing as an ordinary Compose call.
        """
        return self.context.finish(data)
