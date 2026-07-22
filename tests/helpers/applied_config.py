"""Public-path contract runner for applied transform configurations."""

from __future__ import annotations

import copy
import json
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np

import albumentations as A
from albumentations.core.serialization import SERIALIZABLE_REGISTRY

if TYPE_CHECKING:
    from tests.helpers.transform_cases import TransformContractCase


class ReplayProfile(Enum):
    """Strength of the semantic assertion made after runnable replay."""

    RUNNABLE = auto()
    EXACT = auto()


class ContractLevel(Enum):
    """Observable stage of the applied-configuration contract."""

    EMISSION = 1
    TRANSPORT = 2
    RECONSTRUCTION = 3
    EXECUTION = 4
    EQUIVALENCE = 5


class AppliedConfigContractError(AssertionError):
    """Failure with enough context to reproduce one contract case."""

    def __init__(
        self,
        *,
        level: ContractLevel,
        case: TransformContractCase,
        seed: int,
        applied_record: Any = None,
        transported_record: Any = None,
        detail: str,
    ) -> None:
        message = "\n".join(
            (
                f"Level {level.value} {level.name.lower()} failed",
                f"case: {case.case_id}",
                f"seed: {seed}",
                f"transform: {case.transform_cls.__name__}",
                f"init_kwargs: {dict(case.init_kwargs)!r}",
                f"applied_record: {applied_record!r}",
                f"transported_record: {transported_record!r}",
                f"detail: {detail}",
            ),
        )
        super().__init__(message)
        self.level = level


@dataclass(frozen=True)
class AppliedConfigContractResult:
    """Artifacts returned to focused regression tests."""

    applied_record: list[tuple[str, dict[str, Any]]]
    transported_record: list[list[Any]]
    original_result: dict[str, Any]
    replay_result: dict[str, Any]


def _raise_contract_error(
    *,
    level: ContractLevel,
    case: TransformContractCase,
    seed: int,
    applied_record: Any,
    transported_record: Any,
    detail: str,
    cause: BaseException | None = None,
) -> None:
    error = AppliedConfigContractError(
        level=level,
        case=case,
        seed=seed,
        applied_record=applied_record,
        transported_record=transported_record,
        detail=detail,
    )
    if cause is None:
        raise error
    raise error from cause


def _assert_emission(
    case: TransformContractCase,
    seed: int,
    transform: A.BasicTransform,
    applied_record: list[tuple[str, dict[str, Any]]],
) -> None:
    if len(applied_record) != 1:
        _raise_contract_error(
            level=ContractLevel.EMISSION,
            case=case,
            seed=seed,
            applied_record=applied_record,
            transported_record=None,
            detail=f"expected exactly one applied transform, got {len(applied_record)}",
        )

    class_name, config = applied_record[0]
    if class_name not in SERIALIZABLE_REGISTRY:
        _raise_contract_error(
            level=ContractLevel.EMISSION,
            case=case,
            seed=seed,
            applied_record=applied_record,
            transported_record=None,
            detail=f"{class_name!r} is absent from SERIALIZABLE_REGISTRY",
        )
    if "p" not in config:
        _raise_contract_error(
            level=ContractLevel.EMISSION,
            case=case,
            seed=seed,
            applied_record=applied_record,
            transported_record=None,
            detail="applied configuration does not contain p",
        )

    replay_cls = SERIALIZABLE_REGISTRY[class_name]
    invalid_keys = set(config) - replay_cls._get_valid_config_keys()
    if invalid_keys:
        _raise_contract_error(
            level=ContractLevel.EMISSION,
            case=case,
            seed=seed,
            applied_record=applied_record,
            transported_record=None,
            detail=f"unknown constructor keys: {sorted(invalid_keys)}",
        )
    if transform.applied_config != config:
        _raise_contract_error(
            level=ContractLevel.EMISSION,
            case=case,
            seed=seed,
            applied_record=applied_record,
            transported_record=None,
            detail="captured configuration differs from transform.applied_config",
        )


def _assert_execution_result(
    case: TransformContractCase,
    seed: int,
    source_data: dict[str, Any],
    result: dict[str, Any],
    applied_record: Any,
    transported_record: Any,
) -> None:
    target_keys = set(source_data) - case.metadata_keys
    missing = target_keys - set(result)
    if missing:
        _raise_contract_error(
            level=ContractLevel.EXECUTION,
            case=case,
            seed=seed,
            applied_record=applied_record,
            transported_record=transported_record,
            detail=f"replay result is missing targets: {sorted(missing)}",
        )

    for key in target_keys:
        value = result[key]
        if not isinstance(value, np.ndarray):
            continue
        if value.dtype == np.dtype("O"):
            _raise_contract_error(
                level=ContractLevel.EXECUTION,
                case=case,
                seed=seed,
                applied_record=applied_record,
                transported_record=transported_record,
                detail=f"target {key!r} has object dtype",
            )
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            _raise_contract_error(
                level=ContractLevel.EXECUTION,
                case=case,
                seed=seed,
                applied_record=applied_record,
                transported_record=transported_record,
                detail=f"target {key!r} contains non-finite values",
            )


def _find_input_mutation(current: Any, snapshot: Any, path: str) -> str | None:
    """Return the first changed fixture path, limited to stable transport types."""
    if isinstance(snapshot, np.ndarray):
        if not isinstance(current, np.ndarray):
            return f"{path} changed type from ndarray to {type(current).__name__}"
        if current.dtype != snapshot.dtype or current.shape != snapshot.shape:
            return (
                f"{path} changed array metadata from {snapshot.dtype}/{snapshot.shape} "
                f"to {current.dtype}/{current.shape}"
            )
        if not np.array_equal(current, snapshot):
            return f"{path} array contents changed"
        return None

    if isinstance(snapshot, dict):
        if not isinstance(current, dict):
            return f"{path} changed type from dict to {type(current).__name__}"
        if current.keys() != snapshot.keys():
            return f"{path} keys changed from {sorted(snapshot)} to {sorted(current)}"
        for key in snapshot:
            mutation = _find_input_mutation(current[key], snapshot[key], f"{path}.{key}")
            if mutation is not None:
                return mutation
        return None

    if isinstance(snapshot, (list, tuple)):
        if not isinstance(current, type(snapshot)):
            return f"{path} changed type from {type(snapshot).__name__} to {type(current).__name__}"
        if len(current) != len(snapshot):
            return f"{path} length changed from {len(snapshot)} to {len(current)}"
        for index, (current_item, snapshot_item) in enumerate(zip(current, snapshot, strict=True)):
            mutation = _find_input_mutation(current_item, snapshot_item, f"{path}[{index}]")
            if mutation is not None:
                return mutation
        return None

    if isinstance(snapshot, (str, bytes, int, float, bool, type(None))) and current != snapshot:
        return f"{path} changed from {snapshot!r} to {current!r}"
    return None


def _assert_input_unchanged(
    case: TransformContractCase,
    seed: int,
    current: dict[str, Any],
    snapshot: dict[str, Any],
    applied_record: Any,
    transported_record: Any,
) -> None:
    mutation = _find_input_mutation(current, snapshot, "input")
    if mutation is not None:
        _raise_contract_error(
            level=ContractLevel.EXECUTION,
            case=case,
            seed=seed,
            applied_record=applied_record,
            transported_record=transported_record,
            detail=f"transform mutated its caller-owned fixture: {mutation}",
        )


def _assert_repeated_capture_is_independent(
    case: TransformContractCase,
    seed: int,
    pipeline: A.Compose,
    transform: A.BasicTransform,
    applied_record: list[tuple[str, dict[str, Any]]],
) -> None:
    first_record_snapshot = copy.deepcopy(applied_record)
    second_data = case.data_factory(np.random.default_rng(seed + 1))
    second_snapshot = copy.deepcopy(second_data)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        second_result = pipeline(**second_data)
    if applied_record != first_record_snapshot:
        _raise_contract_error(
            level=ContractLevel.EMISSION,
            case=case,
            seed=seed,
            applied_record=applied_record,
            transported_record=None,
            detail="a later application mutated the previously captured record",
        )
    _assert_emission(case, seed, transform, second_result["applied_transforms"])
    _assert_input_unchanged(case, seed, second_data, second_snapshot, applied_record, None)


def _assert_exact_targets(
    case: TransformContractCase,
    seed: int,
    original_result: dict[str, Any],
    replay_result: dict[str, Any],
    applied_record: Any,
    transported_record: Any,
) -> None:
    comparable_keys = (set(original_result) & set(replay_result)) - {"applied_transforms"} - case.metadata_keys
    try:
        for key in comparable_keys:
            original = original_result[key]
            replayed = replay_result[key]
            if isinstance(original, np.ndarray):
                np.testing.assert_array_equal(replayed, original, err_msg=f"target {key!r} differs")
            else:
                assert replayed == original, f"target {key!r} differs: {replayed!r} != {original!r}"
    except (AssertionError, ValueError) as exc:
        _raise_contract_error(
            level=ContractLevel.EQUIVALENCE,
            case=case,
            seed=seed,
            applied_record=applied_record,
            transported_record=transported_record,
            detail=str(exc),
            cause=exc,
        )


def run_applied_config_contract(case: TransformContractCase, seed: int) -> AppliedConfigContractResult:
    """Run one case through capture, JSON transport, reconstruction, and execution."""
    original_data = case.data_factory(np.random.default_rng(seed))
    replay_data = case.data_factory(np.random.default_rng(seed))
    original_snapshot = copy.deepcopy(original_data)
    replay_snapshot = copy.deepcopy(replay_data)
    applied_record: list[tuple[str, dict[str, Any]]] | None = None
    transported_record: list[list[Any]] | None = None

    try:
        transform = case.transform_cls(**copy.deepcopy(dict(case.init_kwargs)), p=1.0)
        pipeline = A.Compose(
            [transform],
            save_applied_params=True,
            seed=seed,
            strict=True,
            **copy.deepcopy(dict(case.compose_kwargs)),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            original_result = pipeline(**original_data)
        applied_record = original_result["applied_transforms"]
        _assert_emission(case, seed, transform, applied_record)
        _assert_input_unchanged(case, seed, original_data, original_snapshot, applied_record, None)
        _assert_repeated_capture_is_independent(case, seed, pipeline, transform, applied_record)
    except AppliedConfigContractError:
        raise
    except Exception as exc:
        _raise_contract_error(
            level=ContractLevel.EMISSION,
            case=case,
            seed=seed,
            applied_record=applied_record,
            transported_record=None,
            detail=f"{type(exc).__name__}: {exc}",
            cause=exc,
        )

    try:
        transported_record = json.loads(json.dumps(applied_record, allow_nan=False))
    except (TypeError, ValueError) as exc:
        _raise_contract_error(
            level=ContractLevel.TRANSPORT,
            case=case,
            seed=seed,
            applied_record=applied_record,
            transported_record=None,
            detail=f"{type(exc).__name__}: {exc}",
            cause=exc,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            replay = A.Compose.from_applied_transforms(
                transported_record,
                **copy.deepcopy(dict(case.compose_kwargs)),
            )
    except Exception as exc:
        _raise_contract_error(
            level=ContractLevel.RECONSTRUCTION,
            case=case,
            seed=seed,
            applied_record=applied_record,
            transported_record=transported_record,
            detail=f"{type(exc).__name__}: {exc}",
            cause=exc,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            replay_result = replay(**replay_data)
        _assert_input_unchanged(case, seed, replay_data, replay_snapshot, applied_record, transported_record)
        _assert_execution_result(case, seed, original_snapshot, replay_result, applied_record, transported_record)
    except AppliedConfigContractError:
        raise
    except Exception as exc:
        _raise_contract_error(
            level=ContractLevel.EXECUTION,
            case=case,
            seed=seed,
            applied_record=applied_record,
            transported_record=transported_record,
            detail=f"{type(exc).__name__}: {exc}",
            cause=exc,
        )

    if case.replay_profile is ReplayProfile.EXACT:
        _assert_exact_targets(case, seed, original_result, replay_result, applied_record, transported_record)

    return AppliedConfigContractResult(applied_record, transported_record, original_result, replay_result)
