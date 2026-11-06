"""Transform constructor validation."""

from collections.abc import Callable
from inspect import Signature, signature
from typing import Any
from warnings import warn

from pydantic import BaseModel, ValidationError


class ValidatedTransformMeta(type):
    """Validate constructor arguments with the transform's InitSchema before calling __init__."""

    @staticmethod
    def _process_init_parameters(
        init_signature: Signature,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str], bool]:
        param_names = list(init_signature.parameters)[1:]
        strict = kwargs.get("strict", False)
        if not isinstance(strict, bool):
            raise TypeError(f"strict must be a bool, got {type(strict).__name__}")
        known_kwargs = {key: value for key, value in kwargs.items() if key in param_names and key != "strict"}
        try:
            bound = init_signature.bind_partial(None, *args, **known_kwargs)
        except TypeError as error:
            raise TypeError(f"Invalid transform arguments ({error}). Check the constructor signature.") from error
        bound.apply_defaults()
        arguments = {key: value for key, value in bound.arguments.items() if key in param_names and key != "strict"}
        return arguments, param_names, strict

    @staticmethod
    def _validate_parameters(
        schema_cls: type[BaseModel],
        full_kwargs: dict[str, Any],
        strict: bool,
    ) -> dict[str, Any]:
        try:
            config = schema_cls(**full_kwargs, strict=strict)
        except ValidationError as error:
            errors = error.errors(include_url=False, include_input=False, include_context=False)
            error_type = TypeError if all(item["type"].endswith("_type") for item in errors) else ValueError
            raise error_type(str(error)) from error
        return config.model_dump(exclude={"strict"})

    def __new__(cls: type[Any], name: str, bases: tuple[type, ...], dct: dict[str, Any]) -> type[Any]:
        if "InitSchema" in dct and issubclass(dct["InitSchema"], BaseModel):
            original_init: Callable[..., Any] | None = dct.get("__init__")
            if original_init is None:
                msg = "__init__ not found in class definition"
                raise ValueError(msg)

            original_sig = signature(original_init)

            def custom_init(self: Any, *args: Any, **kwargs: Any) -> None:
                full_kwargs, param_names, strict = cls._process_init_parameters(original_sig, args, kwargs)
                validated_kwargs = cls._validate_parameters(dct["InitSchema"], full_kwargs, strict)
                invalid_args = [key for key in kwargs if key not in param_names and key != "strict"]
                if invalid_args:
                    message = (
                        f"Argument(s) '{', '.join(invalid_args)}' are not valid for transform {name}. "
                        "Remove them or use the documented parameter names."
                    )
                    if strict:
                        raise ValueError(message)
                    warn(message, stacklevel=2)
                original_init(self, **validated_kwargs)
                self.invalid_args = invalid_args

            custom_init_with_signature: Any = custom_init
            custom_init_with_signature.__signature__ = original_sig
            custom_init.__doc__ = original_init.__doc__

            dct["__init__"] = custom_init

        return super().__new__(cls, name, bases, dct)
