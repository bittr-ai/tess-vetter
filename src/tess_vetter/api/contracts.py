"""Minimal API-layer contracts for callable input typing."""

from __future__ import annotations

import inspect
from typing import Any


def callable_input_schema_from_signature(fn: object) -> dict[str, Any]:
    """Build a deterministic minimal input schema from callable signature truth."""
    if not callable(fn):
        return {"type": "object", "properties": {}, "additionalProperties": True}

    try:
        signature = inspect.signature(inspect.unwrap(fn))
    except (TypeError, ValueError):
        return {"type": "object", "properties": {}, "additionalProperties": True}

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

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {name: properties[name] for name in sorted(properties)},
        "additionalProperties": allows_extra,
    }
    if required:
        schema["required"] = sorted(set(required))
    return schema


__all__ = ["callable_input_schema_from_signature"]
