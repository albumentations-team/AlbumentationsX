"""Negative controls that prove each harness stage is active."""

from typing import Any, Literal

import numpy as np
import pytest
from pydantic import model_validator
from typing_extensions import Self

from albumentations.core.invocation import SamplingContext
from albumentations.core.transform_params import TransformParameterPlan, TransformSamplingInput
from albumentations.core.transforms_interface import BaseTransformInitSchema, ImageOnlyTransform
from albumentations.core.type_definitions import ImageType
from tests.helpers.applied_config import (
    AppliedConfigContractError,
    ContractLevel,
    ReplayProfile,
    run_applied_config_contract,
)
from tests.helpers.transform_cases import TransformContractCase


class CrossFieldConflict(ImageOnlyTransform):
    class InitSchema(BaseTransformInitSchema):
        choice: str | None
        choices: tuple[str, ...] | None

        @model_validator(mode="after")
        def validate_exclusive_fields(self) -> Self:
            if self.choice is not None and self.choices is not None:
                raise ValueError("choice and choices are mutually exclusive")
            return self

    def __init__(
        self,
        choice: str | None = None,
        choices: tuple[str, ...] | None = None,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.choice = choice
        self.choices = choices

    def sample_parameters(
        self,
        inputs: TransformSamplingInput,
        sampling: SamplingContext,
    ) -> TransformParameterPlan:
        sampling.applied_overrides["choice"] = "a"
        return TransformParameterPlan.shared_only({})

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        return img


class UnknownEmittedKey(ImageOnlyTransform):
    def sample_parameters(
        self,
        inputs: TransformSamplingInput,
        sampling: SamplingContext,
    ) -> TransformParameterPlan:
        sampling.applied_overrides["unknown_key"] = 137
        return TransformParameterPlan.shared_only({})

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        return img


class NonJsonValue(ImageOnlyTransform):
    class InitSchema(BaseTransformInitSchema):
        payload: Any

    def __init__(self, payload: Any = None, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.payload = payload

    def sample_parameters(
        self,
        inputs: TransformSamplingInput,
        sampling: SamplingContext,
    ) -> TransformParameterPlan:
        sampling.applied_overrides["payload"] = {1, 2}
        return TransformParameterPlan.shared_only({})

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        return img


class MutatesPreviousRecord(ImageOnlyTransform):
    class InitSchema(BaseTransformInitSchema):
        history: list[int] | None

    def __init__(self, history: list[int] | None = None, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.history = [] if history is None else history

    def sample_parameters(
        self,
        inputs: TransformSamplingInput,
        sampling: SamplingContext,
    ) -> TransformParameterPlan:
        self.history.append(len(self.history) + 1)
        sampling.applied_overrides["history"] = self.history
        return TransformParameterPlan.shared_only({})

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        return img


class ReconstructsButCannotRun(ImageOnlyTransform):
    class InitSchema(BaseTransformInitSchema):
        mode: Literal["capture", "replay"]

    def __init__(self, mode: Literal["capture", "replay"] = "capture", p: float = 1.0) -> None:
        super().__init__(p=p)
        self.mode = mode

    def sample_parameters(
        self,
        inputs: TransformSamplingInput,
        sampling: SamplingContext,
    ) -> TransformParameterPlan:
        sampling.applied_overrides["mode"] = "replay"
        return TransformParameterPlan.shared_only({})

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        if self.mode == "replay":
            raise RuntimeError("replay mode deliberately cannot execute")
        return img


class ReconstructsWithDifferentOutput(ImageOnlyTransform):
    class InitSchema(BaseTransformInitSchema):
        mode: Literal["capture", "replay"]

    def __init__(self, mode: Literal["capture", "replay"] = "capture", p: float = 1.0) -> None:
        super().__init__(p=p)
        self.mode = mode

    def sample_parameters(
        self,
        inputs: TransformSamplingInput,
        sampling: SamplingContext,
    ) -> TransformParameterPlan:
        sampling.applied_overrides["mode"] = "replay"
        return TransformParameterPlan.shared_only({})

    def apply(self, img: ImageType, **params: Any) -> ImageType:
        return np.zeros_like(img) if self.mode == "replay" else img


class MutatesInput(ImageOnlyTransform):
    def apply(self, img: ImageType, **params: Any) -> ImageType:
        img[...] = 0
        return img


@pytest.mark.parametrize(
    ("case", "level"),
    [
        (
            TransformContractCase(
                case_id="negative-cross-field-conflict",
                transform_cls=CrossFieldConflict,
                init_kwargs={"choices": ("a", "b")},
            ),
            ContractLevel.RECONSTRUCTION,
        ),
        (
            TransformContractCase(
                case_id="negative-unknown-key",
                transform_cls=UnknownEmittedKey,
            ),
            ContractLevel.EMISSION,
        ),
        (
            TransformContractCase(
                case_id="negative-non-json",
                transform_cls=NonJsonValue,
            ),
            ContractLevel.TRANSPORT,
        ),
        (
            TransformContractCase(
                case_id="negative-stale-record",
                transform_cls=MutatesPreviousRecord,
            ),
            ContractLevel.EMISSION,
        ),
        (
            TransformContractCase(
                case_id="negative-broken-execution",
                transform_cls=ReconstructsButCannotRun,
            ),
            ContractLevel.EXECUTION,
        ),
        (
            TransformContractCase(
                case_id="negative-input-mutation",
                transform_cls=MutatesInput,
            ),
            ContractLevel.EXECUTION,
        ),
        (
            TransformContractCase(
                case_id="negative-output-mismatch",
                transform_cls=ReconstructsWithDifferentOutput,
                replay_profile=ReplayProfile.EXACT,
            ),
            ContractLevel.EQUIVALENCE,
        ),
    ],
    ids=lambda value: value.case_id if isinstance(value, TransformContractCase) else None,
)
def test_negative_control_fails_at_intended_level(case: TransformContractCase, level: ContractLevel) -> None:
    with pytest.raises(AppliedConfigContractError) as raised:
        run_applied_config_contract(case, seed=137)

    assert raised.value.level is level
    assert f"Level {level.value}" in str(raised.value)
