"""Operation specification models for code-mode adapters."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_OPERATION_ID_RE = re.compile(r"^code_mode\.([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)$")
_VERSION_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


class SafetyClass(str, Enum):
    """High-level safety classification for an operation."""

    SAFE = "safe"
    GUARDED = "guarded"
    RESTRICTED = "restricted"


class OperationAvailability(str, Enum):
    """Availability of an operation adapter at runtime."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class SafetyRequirements(BaseModel):
    """Execution constraints required to safely run an operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    needs_network: bool = False
    needs_filesystem: bool = False
    needs_secrets: bool = False
    requires_human_review: bool = False


class OperationCitation(BaseModel):
    """Citation metadata for an operation's provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    url: str | None = None


class OperationExample(BaseModel):
    """Example invocation payloads for an operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)


class OperationSpec(BaseModel):
    """Declarative operation schema for code-mode wrappers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    version: str = "1.0"
    description: str = ""
    availability: OperationAvailability = OperationAvailability.AVAILABLE

    deprecated: bool = False
    replaced_by: str | None = None

    tier_tags: tuple[str, ...] = Field(default_factory=tuple)

    safety_class: SafetyClass = SafetyClass.SAFE
    safety_requirements: SafetyRequirements = Field(default_factory=SafetyRequirements)

    input_json_schema: dict[str, Any] = Field(default_factory=dict)
    output_json_schema: dict[str, Any] = Field(default_factory=dict)

    examples: tuple[OperationExample, ...] = Field(default_factory=tuple)
    citations: tuple[OperationCitation, ...] = Field(default_factory=tuple)

    @property
    def tier(self) -> str:
        """Return tier parsed from `id` (`code_mode.<tier>.<name>`)."""
        match = _OPERATION_ID_RE.match(self.id)
        if match is None:
            raise ValueError(f"Invalid operation id: {self.id}")
        return match.group(1)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if _OPERATION_ID_RE.match(value) is None:
            raise ValueError("id must match 'code_mode.<tier>.<snake_case>'")
        return value

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        if _VERSION_RE.match(value) is None:
            raise ValueError("version must match 'major.minor'")
        return value

    @field_validator("tier_tags")
    @classmethod
    def _normalize_tier_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        # Keep tags deterministic regardless of input ordering.
        return tuple(sorted({tag for tag in value if tag}))

    @field_validator("replaced_by")
    @classmethod
    def _validate_replaced_by(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if _OPERATION_ID_RE.match(value) is None:
            raise ValueError("replaced_by must match 'code_mode.<tier>.<snake_case>'")
        return value

    @model_validator(mode="after")
    def _validate_deprecation(self) -> OperationSpec:
        if self.replaced_by is not None and not self.deprecated:
            raise ValueError("replaced_by requires deprecated=True")
        if self.replaced_by is not None and self.replaced_by == self.id:
            raise ValueError("replaced_by cannot equal id")
        return self


__all__ = [
    "OperationAvailability",
    "OperationCitation",
    "OperationExample",
    "OperationSpec",
    "SafetyClass",
    "SafetyRequirements",
]
