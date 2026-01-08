"""Local error taxonomy for bittr-tess-vetter.

This library is domain-only and must not depend on any platform runtime or tool
framework. We keep a small, stable error enum/envelope that downstream
applications can translate into their own error formats.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorType(str, Enum):
    CACHE_MISS = "CACHE_MISS"
    INVALID_REF = "INVALID_REF"
    INVALID_DATA = "INVALID_DATA"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: ErrorType
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


def make_error(error_type: ErrorType, message: str, **context: Any) -> ErrorEnvelope:
    return ErrorEnvelope(type=error_type, message=message, context=dict(context))
