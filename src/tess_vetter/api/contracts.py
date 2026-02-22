"""Minimal API-layer contracts for callable input typing."""

from __future__ import annotations

import inspect
from typing import Any

from pydantic import BaseModel


def _object_schema(*, allows_extra: bool) -> dict[str, Any]:
    return {"type": "object", "properties": {}, "additionalProperties": allows_extra}


def opaque_object_schema() -> dict[str, Any]:
    """Return a minimal object-shaped schema for intentionally opaque payloads."""
    return {"type": "object"}


def model_input_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Build validation JSON schema from a Pydantic model contract."""
    return model.model_json_schema(mode="validation")


def model_output_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Build serialization JSON schema from a Pydantic model contract."""
    return model.model_json_schema(mode="serialization")


def callable_input_schema_from_signature(fn: object) -> dict[str, Any]:
    """Build a deterministic minimal input schema from callable signature truth."""
    if not callable(fn):
        return _object_schema(allows_extra=True)

    try:
        signature = inspect.signature(inspect.unwrap(fn))
    except (TypeError, ValueError):
        return _object_schema(allows_extra=True)

    required: list[str] = []
    properties: dict[str, Any] = {}
    allows_extra = False

    for parameter in signature.parameters.values():
        if parameter.name in {"self", "cls"}:
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            allows_extra = True
            continue

        properties[parameter.name] = {}
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)

    schema = _object_schema(allows_extra=allows_extra)
    schema["properties"] = {name: properties[name] for name in sorted(properties)}
    if required:
        schema["required"] = sorted(set(required))
    return schema


__all__ = [
    "callable_input_schema_from_signature",
    "model_input_schema",
    "model_output_schema",
    "opaque_object_schema",
]
