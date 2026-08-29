"""Call-local execution state for transforms and composition containers."""

from __future__ import annotations

import copy
import os
import random
import threading
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Protocol

import numpy as np

from albumentations.core.label_manager import LabelManager
from albumentations.core.random_utils import (
    _derive_effective_seed,
    _derive_invocation_seed,
    _get_runtime_rng_context,
    _restore_runtime_rng_state,
    _RuntimeRngContext,
    _should_sync_runtime_rng,
)


class InvocationOwner(Protocol):
    """Defines an identity for call-local sampling and processor state. It keeps transient execution values off
    the reusable configured graph node that owns this key.
    """

    invocation_key: object


@dataclass(slots=True)
class _ReservedRandomStreams:
    """Holds Python and NumPy streams for an invocation or execution thread. It creates generators lazily so
    deterministic pipeline routes allocate no random state.
    """

    random_generator: np.random.Generator | None = None
    py_random: random.Random | None = None
    random_generator_factory: Callable[[], np.random.Generator] | None = None
    py_random_factory: Callable[[], random.Random] | None = None
    epoch: int | None = None

    def get_random_generator(self) -> np.random.Generator:
        """Returns the NumPy generator, creating it only when a sampler needs direct array-valued randomness rather than
        scalar choices.
        """
        if self.random_generator is None:
            if self.random_generator_factory is None:
                msg = "random-stream lease has no NumPy generator"
                raise RuntimeError(msg)
            self.random_generator = self.random_generator_factory()
        return self.random_generator

    def get_py_random(self) -> random.Random:
        """Returns the invocation's Python generator, constructing it from the reserved factory only when probability
        or scalar parameter sampling needs a random choice.
        """
        if self.py_random is None:
            if self.py_random_factory is None:
                msg = "random-stream lease has no Python generator"
                raise RuntimeError(msg)
            self.py_random = self.py_random_factory()
        return self.py_random


class InvocationRngOwner:
    """Owns configured seed sources for transforms and Compose nodes. It resolves thread-local streams without
    retaining generators that belong to active calls.

    A configured object owns the source pair for its configuring thread and derives one private pair for every other
    execution thread. An :class:`InvocationContext` only resolves the source for the active call; it never stores an
    active generator on the configured object.
    """

    def _initialize_invocation_rng(self, seed: int | None) -> None:
        """Initialize an owner's seed fields and lock during construction so concurrent first calls cannot race and
        sampling state stays outside the configured graph.

        A pipeline is constructed once and called many times during training. Creating
        the lock in the constructor keeps first-call initialization race-free without
        putting lock installation on the execution path. Generators and thread-local
        streams remain deferred until this object becomes a sampling root.
        """
        self.seed = seed
        self._base_seed = seed
        self._manual_random_state = False
        self._rng_context: _RuntimeRngContext | None = None
        self._rng_initialized = False
        self._seed_lock = threading.Lock()

    def _ensure_configured_random_sources(self) -> None:
        """Create locks and sources only when this object supplies root randomness, leaving configured
        graph children lightweight until they are invoked directly.

        Child graph nodes resolve randomness through the active
        :class:`InvocationContext` before reaching this method, so their dormant
        owner stays allocation-free for the lifetime of an ordinary Compose graph.
        """
        if self._rng_initialized:
            return

        with self._seed_lock:
            if self._rng_initialized:
                return
            runtime_context = _get_runtime_rng_context(self._base_seed)
            effective_seed = runtime_context.effective_seed if runtime_context else self._base_seed
            self._configured_random_generator = np.random.default_rng(effective_seed)
            self._configured_py_random = random.Random(effective_seed)
            self._rng_context = runtime_context
            self._invocation_index = 1
            self._rng_epoch = 0
            self._concurrent_random_generator = copy.deepcopy(self._configured_random_generator)
            self._concurrent_py_random = copy.copy(self._configured_py_random)
            self._rng_process_id = os.getpid()
            self._rng_source_thread_id = threading.get_ident()
            self._thread_rng_streams = threading.local()
            self._rng_initialized = True

    def _ensure_rng_lock(self) -> None:
        """Recreate a process-local seed lock after deserialization so future generator replacement and stream
        reservation stay synchronized across callers.
        """
        if not hasattr(self, "_seed_lock"):
            self._seed_lock = threading.Lock()

    @property
    def random_generator(self) -> np.random.Generator:
        """Returns the active invocation NumPy generator or configured source, preserving the familiar interface
        without sharing active call state.
        """
        invocation = get_current_invocation()
        if invocation is not None:
            return invocation.random_generator
        self._ensure_configured_random_sources()
        return self._configured_random_generator

    @random_generator.setter
    def random_generator(self, value: np.random.Generator) -> None:
        self._ensure_configured_random_sources()
        with self._seed_lock:
            self._configured_random_generator = value
            self._invalidate_thread_random_sources()

    @property
    def py_random(self) -> random.Random:
        """Returns the active invocation Python generator or configured source, preserving the familiar interface
        without sharing active call state.
        """
        invocation = get_current_invocation()
        if invocation is not None:
            return invocation.py_random
        self._ensure_configured_random_sources()
        return self._configured_py_random

    @py_random.setter
    def py_random(self, value: random.Random) -> None:
        self._ensure_configured_random_sources()
        with self._seed_lock:
            self._configured_py_random = value
            self._invalidate_thread_random_sources()

    def _invalidate_thread_random_sources(self) -> None:
        """Invalidates cached thread streams after a generator replacement, so the next call uses supplied manual
        state rather than an earlier reservation.
        """
        self._ensure_rng_lock()
        self._manual_random_state = True
        self._invocation_index = 1
        self._rng_epoch += 1
        configured_random_generator = self._configured_random_generator
        if configured_random_generator is not None:
            self._concurrent_random_generator = copy.deepcopy(configured_random_generator)
        configured_py_random = self._configured_py_random
        if configured_py_random is not None:
            self._concurrent_py_random = copy.copy(configured_py_random)
        self._rng_process_id = os.getpid()
        self._rng_source_thread_id = threading.get_ident()
        self._thread_rng_streams = threading.local()

    def _create_invocation_context(
        self,
        *,
        collect_applied: bool,
        root_key: object | None = None,
        invocation_seed: int | None = None,
        prime_py_random: bool = False,
    ) -> InvocationContext:
        """Creates a call-local context with lazy streams and primes Python randomness only for a root probability
        decision that runs before activation.
        """
        streams: _ReservedRandomStreams | None = None
        if prime_py_random:
            streams = (
                _ReservedRandomStreams(
                    random_generator_factory=lambda: np.random.default_rng(invocation_seed),
                    py_random=random.Random(invocation_seed),
                )
                if invocation_seed is not None
                else self._reserve_invocation_streams()
            )
        return InvocationContext(
            seed=invocation_seed,
            random_stream_reserver=None if invocation_seed is not None else self._reserve_invocation_streams,
            collect_applied=collect_applied,
            root_key=root_key,
            _py_random=None if streams is None else streams.get_py_random(),
            _reserved_random_streams=streams,
        )

    def create_sampling_context(self, applied_overrides: Any | None = None) -> SamplingContext:
        """Create a direct sampler view backed by an owner's configured stream so tests and utilities can exercise
        sampling without a full transform call.

        This is for tests and internal utilities that intentionally sample without
        invoking a transform. Normal transform execution receives its context from
        `apply_in_invocation`.
        """
        invocation = self._create_invocation_context(collect_applied=False)
        return invocation.sampling_context({} if applied_overrides is None else applied_overrides)

    def _reserve_invocation_streams(self) -> _ReservedRandomStreams:
        """Returns this execution thread's random-stream pair without steady-state locks, ensuring concurrent threads
        never share mutable generators.

        The thread that configured this owner advances its original stream, preserving serial seeded behavior. Other
        threads initialize a private derived stream once. Compose calls are synchronous, so one thread cannot overlap
        two root invocations; concurrent threads therefore never share a mutable generator.
        """
        self._ensure_configured_random_sources()
        self._sync_runtime_random_state()
        local = self._thread_rng_streams
        streams = getattr(local, "streams", None)
        if streams is not None and streams.epoch == self._rng_epoch:
            return streams

        with self._seed_lock:
            streams = getattr(local, "streams", None)
            if streams is not None and streams.epoch == self._rng_epoch:
                return streams

            if threading.get_ident() == self._rng_source_thread_id:
                configured_random_generator = self._configured_random_generator
                configured_py_random = self._configured_py_random
                if configured_random_generator is None or configured_py_random is None:
                    msg = "configured random-stream pair is missing"
                    raise RuntimeError(msg)
                streams = _ReservedRandomStreams(
                    random_generator=configured_random_generator,
                    py_random=configured_py_random,
                    epoch=self._rng_epoch,
                )
            else:
                if self._manual_random_state or self._base_seed is None:
                    numpy_seed = int(
                        self._concurrent_random_generator.integers(
                            np.iinfo(np.uint64).max,
                            dtype=np.uint64,
                        ),
                    )
                    python_seed = self._concurrent_py_random.getrandbits(64)
                else:
                    seed = _derive_invocation_seed(
                        base_seed=self._base_seed,
                        runtime_context=self._rng_context,
                        invocation_index=self._invocation_index,
                        random_generator=None,
                        py_random=None,
                        manual=False,
                    )
                    numpy_seed = seed
                    python_seed = seed
                streams = _ReservedRandomStreams(
                    random_generator_factory=lambda: np.random.default_rng(numpy_seed),
                    py_random_factory=lambda: random.Random(python_seed),
                    epoch=self._rng_epoch,
                )
            self._invocation_index += 1
            local.streams = streams
        return streams

    def set_random_state(
        self,
        random_generator: np.random.Generator,
        py_random: random.Random,
        *,
        runtime_context: _RuntimeRngContext | None = None,
        manual: bool = True,
    ) -> None:
        """Sets explicit Python and NumPy generators for future call-local streams, using caller-provided state rather
        than automatic DataLoader worker seeding.
        """
        self._set_random_state(
            random_generator,
            py_random,
            runtime_context=runtime_context,
            manual=manual,
        )

    def _set_random_state(
        self,
        random_generator: np.random.Generator,
        py_random: random.Random,
        *,
        runtime_context: _RuntimeRngContext | None,
        manual: bool,
    ) -> None:
        self._ensure_rng_lock()
        with self._seed_lock:
            if not self._rng_initialized:
                self._invocation_index = 0
                self._rng_epoch = 0
            self._configured_random_generator = random_generator
            self._configured_py_random = py_random
            self._rng_context = runtime_context
            self._manual_random_state = manual
            # The configured thread owns stream zero; derived thread sources start at a distinct index.
            self._invocation_index = 1
            self._rng_epoch += 1
            self._concurrent_random_generator = copy.deepcopy(random_generator)
            self._concurrent_py_random = copy.copy(py_random)
            self._rng_process_id = os.getpid()
            self._rng_source_thread_id = threading.get_ident()
            self._thread_rng_streams = threading.local()
            self._rng_initialized = True

    def set_random_seed(self, seed: int | None) -> None:
        """Sets the configured seed for reproducible call-local streams. A DataLoader worker seed is combined when
        execution crosses into a worker process.
        """
        self.seed = seed
        self._base_seed = seed
        runtime_context = _get_runtime_rng_context(seed)
        effective_seed = runtime_context.effective_seed if runtime_context else seed
        self._set_random_state(
            np.random.default_rng(effective_seed),
            random.Random(effective_seed),
            runtime_context=runtime_context,
            manual=False,
        )

    def _sync_runtime_random_state(self) -> None:
        """Refreshes automatic sources at a DataLoader process boundary while leaving manual generators unchanged for
        caller control.
        """
        if not self._rng_initialized:
            return

        current_process_id = os.getpid()
        if getattr(self, "_rng_process_id", None) == current_process_id:
            return
        self._rng_process_id = current_process_id
        runtime_context = _get_runtime_rng_context(self._base_seed)
        if runtime_context is None or not _should_sync_runtime_rng(
            manual=self._manual_random_state,
            current_context=self._rng_context,
            runtime_context=runtime_context,
        ):
            return
        self._set_random_state(
            np.random.default_rng(runtime_context.effective_seed),
            random.Random(runtime_context.effective_seed),
            runtime_context=runtime_context,
            manual=False,
        )

    def _get_effective_seed(self, base_seed: int | None) -> int | None:
        """Returns a configured seed combined with an active worker seed, matching the effective source automatic
        Compose execution uses inside that worker.
        """
        runtime_context = _get_runtime_rng_context(base_seed)
        return _derive_effective_seed(base_seed, None) if runtime_context is None else runtime_context.effective_seed

    def _get_invocation_pickle_state(self) -> dict[str, Any]:
        """Returns serializable configured state without process-local locks or thread reservations, which cannot
        safely cross a pickle or DataLoader worker boundary.
        """
        state = self.__dict__.copy()
        state.pop("_seed_lock", None)
        state.pop("_thread_rng_streams", None)
        state.pop("_concurrent_random_generator", None)
        return state

    def _restore_invocation_pickle_state(self, state: dict[str, Any]) -> None:
        """Restores configured state from pickle and recreates locks and thread reservations so receiving workers have
        no inherited active state.
        """
        self.__dict__.update(state)
        self._seed_lock = threading.Lock()
        if self._rng_initialized:
            self._invocation_index = 1
            self._rng_epoch = 0
            self._thread_rng_streams = threading.local()
            configured_py_random = self._configured_py_random
            if configured_py_random is None:
                msg = "pickled invocation owner has no configured Python random stream"
                raise RuntimeError(msg)
            configured_random_generator = self._configured_random_generator
            if configured_random_generator is None:
                msg = "pickled invocation owner has no configured NumPy random stream"
                raise RuntimeError(msg)
            self._concurrent_random_generator = copy.deepcopy(configured_random_generator)
            self._concurrent_py_random = copy.copy(configured_py_random)
        _restore_runtime_rng_state(self)


@dataclass(slots=True)
class TransformInvocationState:
    """Stores parameters and realized configuration from one transform call. It keeps replay and observation data off
    the configured transform object.
    """

    params: dict[Any, Any] | None = None
    applied_config: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ChannelRestorationState:
    """Keep original object identity with the normalized optional-channel value."""

    canonical_target: str
    original_value: Any
    normalized_value: Any


@dataclass(slots=True)
class ComposeInvocationState:
    """Stores grayscale normalization and instance-binding details for one Compose call. It prevents temporary target
    metadata from leaking into another sample.
    """

    channel_restorations: dict[str, ChannelRestorationState] = field(default_factory=dict)
    instance_count: int | None = None
    repack_after_processors: bool = False


@dataclass(slots=True)
class InvocationObservation:
    """Keeps caller-local snapshots from an observing call, exposing detached parameters without retaining configured
    transforms or shared state.
    """

    first_transform_key: object | None = None
    first_transform_state: TransformInvocationState | None = None
    transforms: dict[object, TransformInvocationState] | None = None

    def get_transform_state(self, transform: InvocationOwner) -> TransformInvocationState | None:
        """Returns observation state for `transform` through its stable identity. It retains no configured owner and
        later execution cannot mutate the published snapshot.
        """
        if self.first_transform_key is transform.invocation_key:
            return self.first_transform_state
        return None if self.transforms is None else self.transforms.get(transform.invocation_key)


@dataclass(slots=True)
class InvocationContext:
    """Owns mutable state for one augmentation call, including streams, parameters, target sessions, and normalization
    decisions. Configured graphs remain reusable.

    The context contains the root random streams, per-transform sampled values, cloned target processors, and
    normalization state. Configured Compose and BasicTransform objects remain reusable across calls.
    """

    seed: int | None = None
    random_stream_reserver: Callable[[], _ReservedRandomStreams] | None = None
    collect_applied: bool = False
    root_key: object | None = None
    has_tensor_inputs: bool | None = None
    _py_random: random.Random | None = None
    _random_generator: np.random.Generator | None = None
    _reserved_random_streams: _ReservedRandomStreams | None = None
    _first_transform_key: object | None = None
    _first_transform_state: TransformInvocationState | None = None
    _transform_states: dict[object, TransformInvocationState] | None = None
    _first_compose_key: object | None = None
    _first_compose_state: ComposeInvocationState | None = None
    _compose_states: dict[object, ComposeInvocationState] | None = None
    _processor_sessions: dict[int, dict[str, Any]] | None = None
    _active_processors: dict[str, Any] | None = None
    _active_processors_by_id: dict[int, Any] | None = None
    _filtered_processor_ids: set[int] | None = None
    _sampling_context: SamplingContext | None = None
    trace_session: Any | None = None
    _activation_token: Token[InvocationContext | None] | None = None

    def _random_streams(self) -> _ReservedRandomStreams:
        """Resolves this invocation's stream pair only when code samples randomness. Deterministic execution therefore
        avoids NumPy and Python generator allocation.
        """
        if self._reserved_random_streams is None:
            self._reserved_random_streams = (
                _ReservedRandomStreams(
                    random_generator_factory=lambda: np.random.default_rng(self.seed),
                    py_random_factory=lambda: random.Random(self.seed),
                )
                if self.random_stream_reserver is None
                else self.random_stream_reserver()
            )
        return self._reserved_random_streams

    @property
    def random_generator(self) -> np.random.Generator:
        """Returns this invocation's NumPy stream and creates it only for array-valued random sampling, so deterministic
        routes retain a zero-allocation fast path.
        """
        if self._random_generator is None:
            self._random_generator = self._random_streams().get_random_generator()
        return self._random_generator

    @property
    def py_random(self) -> random.Random:
        """Returns this invocation's Python stream and creates it only for probability or scalar sampling, so
        deterministic routes retain a zero-allocation fast path.
        """
        if self._py_random is None:
            self._py_random = self._random_streams().get_py_random()
        return self._py_random

    def transform_state(self, transform: InvocationOwner) -> TransformInvocationState:
        """Returns mutable call-local state for `transform` using a compact first slot. It grows into a mapping only
        when one invocation observes multiple transform nodes.
        """
        key = transform.invocation_key
        if self._first_transform_key is key:
            if self._first_transform_state is None:
                msg = "first transform key requires a state"
                raise RuntimeError(msg)
            return self._first_transform_state
        if self._first_transform_key is None:
            state = TransformInvocationState()
            self._first_transform_key = key
            self._first_transform_state = state
            return state
        if self._transform_states is None:
            first_state = self._first_transform_state
            if first_state is None:
                msg = "first transform key requires a state"
                raise RuntimeError(msg)
            self._transform_states = {self._first_transform_key: first_state}
        return self._transform_states.setdefault(key, TransformInvocationState())

    def get_transform_state(self, transform: InvocationOwner) -> TransformInvocationState | None:
        """Returns state without allocating a record for skipped leaves, keeping observing invocations cheap when
        callers only inspect the active transform.
        """
        key = transform.invocation_key
        if self._first_transform_key is key:
            return self._first_transform_state
        return None if self._transform_states is None else self._transform_states.get(key)

    def compose_state(self, compose: InvocationOwner) -> ComposeInvocationState:
        """Returns normalization and instance-binding state for `compose` using one compact slot, adding a mapping only
        when nested composition nodes need state.
        """
        key = compose.invocation_key
        if self._first_compose_key is key:
            if self._first_compose_state is None:
                msg = "first compose key requires a state"
                raise RuntimeError(msg)
            return self._first_compose_state
        if self._first_compose_key is None:
            state = ComposeInvocationState()
            self._first_compose_key = key
            self._first_compose_state = state
            return state
        if self._compose_states is None:
            first_state = self._first_compose_state
            if first_state is None:
                msg = "first compose key requires a state"
                raise RuntimeError(msg)
            self._compose_states = {self._first_compose_key: first_state}
        return self._compose_states.setdefault(key, ComposeInvocationState())

    def get_compose_state(self, compose: InvocationOwner) -> ComposeInvocationState | None:
        """Returns existing state for `compose` without allocating on no-op paths, so postprocessing can distinguish
        real grayscale normalization from an untouched input.
        """
        key = compose.invocation_key
        if self._first_compose_key is key:
            return self._first_compose_state
        return None if self._compose_states is None else self._compose_states.get(key)

    def processors(self, configured_processors: dict[str, Any]) -> dict[str, Any]:
        """Returns per-call processor sessions cloned from the configured mapping. This isolates mutable label encoders
        and metadata from other calls sharing the pipeline.

        Processor policy and target aliases are immutable during execution. Label encoders and metadata are not: each
        session gets a fresh LabelManager so categorical labels from one sample cannot affect another sample.
        """
        configured_id = id(configured_processors)
        if self._processor_sessions is None:
            self._processor_sessions = {}
        sessions = self._processor_sessions.get(configured_id)
        if sessions is not None:
            return sessions

        sessions = {}
        for name, processor in configured_processors.items():
            session = copy.copy(processor)
            session.label_manager = LabelManager()
            sessions[name] = session
        self._processor_sessions[configured_id] = sessions
        return sessions

    def activate_processors(self, configured_processors: dict[str, Any]) -> dict[str, Any]:
        """Create and activate per-call annotation sessions at the root so leaves share mutable labels
        without persistent processor injection through the configured graph.

        Configured processors remain policy owned by the root Compose. Leaves retrieve
        only these call-local sessions, so building a graph never injects mutable
        annotation state into every transform node.
        """
        sessions = self.processors(configured_processors)
        self._active_processors = sessions
        self._active_processors_by_id = {
            id(configured_processor): sessions[name] for name, configured_processor in configured_processors.items()
        }
        return sessions

    def get_processor(self, name: str) -> Any | None:
        """Return the active annotation processor for this invocation, keeping each leaf detached
        from root configuration and sessions owned by other callers.
        """
        return None if self._active_processors is None else self._active_processors.get(name)

    def get_processor_session(self, configured_processor: object) -> Any | None:
        """Return a call-local session for this policy identity, letting nested containers filter
        annotations without mutating persistent root configuration.
        """
        if self._active_processors_by_id is None:
            return None
        return self._active_processors_by_id.get(id(configured_processor))

    def sampling_context(self, applied_overrides: Any) -> SamplingContext:
        """Reuse an explicit sampling view while replacing its policy sink, so samplers see call-local streams without
        allocating a new context per leaf.

        The view exposes call-local RNG streams without requiring transform samplers
        to read a ContextVar. Its policy sink is replaced before each sampler so
        observation remains isolated while ordinary calls keep discarding policy.
        """
        context = self._sampling_context
        if context is None:
            context = SamplingContext(invocation=self, applied_overrides=applied_overrides)
            self._sampling_context = context
        else:
            context.applied_overrides = applied_overrides
        return context

    def mark_processor_filtered(self, configured_processor: object) -> None:
        """Remember that this policy filtered after a node so root finalization avoids scanning,
        clipping, and relabeling the same target data a second time in one call.

        The configured processor identity is stable policy; the call-local session is
        intentionally not. Recording the policy key lets the root reuse its last
        per-node filtering decision instead of scanning the same annotations again.
        """
        if self._filtered_processor_ids is None:
            self._filtered_processor_ids = set()
        self._filtered_processor_ids.add(id(configured_processor))

    def was_processor_filtered(self, configured_processor: object) -> bool:
        """Report whether this invocation already filtered one policy so root finalization skips
        duplicate clipping, label encoding, and target traversal work.
        """
        return self._filtered_processor_ids is not None and id(configured_processor) in self._filtered_processor_ids

    def observation(self) -> InvocationObservation:
        """Builds a detached observation from transform state without retaining configured transforms or processor
        sessions after the completed invocation.
        """
        return InvocationObservation(
            first_transform_key=self._first_transform_key,
            first_transform_state=self._first_transform_state,
            transforms=self._transform_states,
        )

    def __enter__(self) -> None:
        """Publishes this call-local context for dynamic transform access and saves its reset token, keeping configured
        graph nodes free of active execution data.
        """
        self._activation_token = _CURRENT_INVOCATION.set(self)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Publishes completed observations or clears stale caller data, then resets this context even when transform
        execution raises before normal postprocessing.
        """
        try:
            if exc_type is None and self.collect_applied:
                _LAST_COMPLETED_OBSERVATION.set(self.observation())
            elif _LAST_COMPLETED_OBSERVATION.get() is not None:
                _LAST_COMPLETED_OBSERVATION.set(None)
        finally:
            token = self._activation_token
            if token is None:
                msg = "invocation context manager exited before entering"
                raise RuntimeError(msg)
            _CURRENT_INVOCATION.reset(token)
            self._activation_token = None


@dataclass(slots=True)
class SamplingContext:
    """Carry one invocation's random streams and realized-policy sink, so samplers receive call-local state explicitly
    instead of reading a reusable configured object.
    """

    invocation: InvocationContext
    applied_overrides: Any

    @classmethod
    def from_seed(cls, seed: int | None, applied_overrides: Any | None = None) -> SamplingContext:
        """Build an isolated sampling view from a fixed seed, making direct sampler tests and small utilities
        deterministic without configuring an owner.
        """
        invocation = InvocationContext(seed=seed)
        return cls(invocation=invocation, applied_overrides={} if applied_overrides is None else applied_overrides)

    @classmethod
    def from_owner(cls, owner: InvocationRngOwner, applied_overrides: Any | None = None) -> SamplingContext:
        """Build a direct sampler view that advances an owner's configured stream, preserving its seeded sequence
        without a full transform application.
        """
        return owner.create_sampling_context(applied_overrides)

    @property
    def random_generator(self) -> np.random.Generator:
        """Expose the invocation-local NumPy generator for array-valued sampling, creating its reserved stream only
        when a sampler needs it.
        """
        return self.invocation.random_generator

    @property
    def py_random(self) -> random.Random:
        """Expose the invocation-local Python generator for probabilities and scalar choices, creating its reserved
        stream only when a sampler needs it.
        """
        return self.invocation.py_random


_CURRENT_INVOCATION: ContextVar[InvocationContext | None] = ContextVar("albumentations_invocation", default=None)
_LAST_COMPLETED_OBSERVATION: ContextVar[InvocationObservation | None] = ContextVar(
    "albumentations_completed_observation",
    default=None,
)


def get_current_invocation() -> InvocationContext | None:
    """Returns the invocation active in this thread or task, letting transform RNG and state access resolve call-local
    values without configured-object mutation.
    """
    return _CURRENT_INVOCATION.get()


def get_completed_transform_state(transform: InvocationOwner) -> TransformInvocationState | None:
    """Returns detached state from the caller's recent observing root call. Ordinary Compose execution intentionally
    discards observation data and returns no state.
    """
    observation = _LAST_COMPLETED_OBSERVATION.get()
    return None if observation is None else observation.get_transform_state(transform)


def publish_completed_transform_state(transform: InvocationOwner, state: TransformInvocationState) -> None:
    """Publishes a direct deterministic leaf's caller-local result after application without activating a full
    execution context when no state is read during dispatch.

    The fast path has no sampler, configured processors, or dynamic random access, so the state is already isolated in
    the synchronous direct call and can be published only after it returns.
    """
    _LAST_COMPLETED_OBSERVATION.set(
        InvocationObservation(
            first_transform_key=transform.invocation_key,
            first_transform_state=state,
        ),
    )


def clear_completed_observation() -> None:
    """Clears this caller's observation after a non-observing root call, so child parameter access never describes an
    earlier sample instead of the latest result.
    """
    if _LAST_COMPLETED_OBSERVATION.get() is not None:
        _LAST_COMPLETED_OBSERVATION.set(None)
