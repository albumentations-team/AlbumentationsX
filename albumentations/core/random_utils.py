"""Private helpers for runtime random seed synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

_UINT32_MODULUS: Final = 1 << 32


@dataclass(frozen=True, slots=True)
class _RuntimeRngContext:
    """Runtime-only RNG context derived from the current DataLoader worker and effective seed used
    to rebuild copied RNG state in worker processes.
    """

    worker_seed: int
    effective_seed: int


def _get_torch_worker_seed() -> int | None:
    """Return PyTorch's current DataLoader worker seed inside worker processes, or None when
    PyTorch is unavailable or the call happens outside DataLoader workers.
    """
    try:
        import torch
        import torch.utils.data
    except ImportError:
        return None

    try:
        if torch.utils.data.get_worker_info() is None:
            return None
        return torch.initial_seed() % _UINT32_MODULUS
    except AttributeError:
        return None


def _derive_effective_seed(base_seed: int | None, worker_seed: int | None) -> int | None:
    """Derive the runtime seed from the user-provided base seed and optional DataLoader worker
    seed while preserving None as unseeded outside worker processes.
    """
    if worker_seed is None:
        return base_seed
    if base_seed is None:
        return worker_seed
    return (base_seed + worker_seed) % _UINT32_MODULUS


def _get_runtime_rng_context(base_seed: int | None) -> _RuntimeRngContext | None:
    """Build a runtime RNG context for the current PyTorch DataLoader worker so copied pipeline
    RNG state can be replaced exactly once per worker seed.
    """
    worker_seed = _get_torch_worker_seed()
    if worker_seed is None:
        return None

    effective_seed = _derive_effective_seed(base_seed, worker_seed)
    if effective_seed is None:
        return None

    return _RuntimeRngContext(worker_seed=worker_seed, effective_seed=effective_seed)


def _should_sync_runtime_rng(
    *,
    manual: bool,
    current_context: _RuntimeRngContext | None,
    runtime_context: _RuntimeRngContext | None,
) -> bool:
    """Return whether RNG state should be rebuilt for the active runtime context while preserving
    parent-propagated effective seeds for children in the same worker.
    """
    if manual or runtime_context is None:
        return False
    if current_context is None:
        return True
    return current_context.worker_seed != runtime_context.worker_seed


def _restore_runtime_rng_state(target: Any) -> None:
    """Restore runtime RNG bookkeeping after unpickling so objects resynchronize against the
    active DataLoader worker seed on their first post-unpickle call.
    """
    target_state = target.__dict__
    target_state.setdefault("_base_seed", getattr(target, "seed", None))
    target_state.setdefault("_manual_random_state", False)
    target_state["_rng_context"] = None
