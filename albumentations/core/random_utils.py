"""Private helpers for runtime random seed synchronization."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch

_UINT32_MODULUS: Final = 1 << 32
_UINT64_MASK: Final = (1 << 64) - 1
_SPLITMIX64_INCREMENT: Final = 0x9E3779B97F4A7C15


@dataclass(frozen=True, slots=True)
class _RuntimeRngContext:
    """Runtime-only RNG context derived from the current DataLoader worker and effective seed used
    to rebuild copied RNG state in worker processes.
    """

    worker_seed: int
    effective_seed: int


def _get_torch_worker_seed() -> int | None:
    """Return PyTorch's current DataLoader worker seed inside worker processes, or None when
    the call happens outside DataLoader workers.
    """
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


def _derive_invocation_seed(
    *,
    base_seed: int | None,
    runtime_context: _RuntimeRngContext | None,
    invocation_index: int,
    random_generator: np.random.Generator | None,
    py_random: random.Random | None,
    manual: bool,
) -> int:
    """Derives a call-local seed from configured, worker, and reservation state, preserving manual and unseeded
    generator contracts.

    Automatic streams mix the configured base seed, active DataLoader worker seed, and monotonically reserved
    invocation index. Explicit `set_random_state` streams reserve one seed from the user-supplied NumPy generator
    under the owner's short seed lock. Unseeded automatic streams reserve entropy from the configured Python stream.
    """
    if manual:
        if random_generator is None:
            msg = "manual invocation seed derivation requires a NumPy generator"
            raise RuntimeError(msg)
        return int(random_generator.integers(0, _UINT64_MASK, dtype=np.uint64))

    effective_seed = runtime_context.effective_seed if runtime_context is not None else base_seed
    if effective_seed is None:
        if py_random is None:
            msg = "unseeded invocation seed derivation requires a Python random stream"
            raise RuntimeError(msg)
        return py_random.getrandbits(64)
    if invocation_index == 0:
        return effective_seed

    value = (effective_seed + _SPLITMIX64_INCREMENT * (invocation_index + 1)) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK
