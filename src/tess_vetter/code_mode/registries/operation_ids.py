"""Deterministic operation id helpers for code-mode registries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

OPERATION_NAMESPACE: Final[str] = "code_mode"
OperationTier = Literal["golden_path", "primitive", "internal"]

_ALLOWED_TIERS: Final[tuple[OperationTier, ...]] = ("golden_path", "primitive", "internal")
_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")
_SNAKE_BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(r"(?<!^)(?=[A-Z])")
_NON_IDENT_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9_]+")


@dataclass(frozen=True, slots=True)
class OperationIdParts:
    """Structured operation id parts."""

    namespace: str
    tier: OperationTier
    name: str

    @property
    def id(self) -> str:
        """Return deterministic string form."""
        return build_operation_id(tier=self.tier, name=self.name)


def _validate_tier(tier: str) -> OperationTier:
    if tier not in _ALLOWED_TIERS:
        joined = ", ".join(_ALLOWED_TIERS)
        raise ValueError(f"tier must be one of: {joined}")
    return tier  # type: ignore[return-value]


def _validate_name(name: str) -> str:
    if _NAME_RE.match(name) is None:
        raise ValueError("operation name must be snake_case: ^[a-z][a-z0-9_]*$")
    return name


def normalize_operation_name(name: str) -> str:
    """Normalize export/symbol names to canonical snake_case operation names."""
    normalized = _SNAKE_BOUNDARY_RE.sub("_", name).lower()
    normalized = normalized.replace("-", "_")
    normalized = _NON_IDENT_RE.sub("_", normalized).strip("_")
    if not normalized:
        normalized = "export"
    if not normalized[0].isalpha():
        normalized = f"fn_{normalized}"
    return normalized


def build_operation_id(*, tier: OperationTier, name: str) -> str:
    """Build a canonical operation id: ``code_mode.<tier>.<snake_case>``."""
    normalized_tier = _validate_tier(tier)
    normalized_name = _validate_name(name)
    return f"{OPERATION_NAMESPACE}.{normalized_tier}.{normalized_name}"


def parse_operation_id(operation_id: str) -> OperationIdParts:
    """Parse ``code_mode.<tier>.<snake_case>`` into structured parts."""
    parts = operation_id.split(".")
    if len(parts) != 3:
        raise ValueError("operation id must match 'code_mode.<tier>.<snake_case>'")

    namespace, tier, name = parts
    if namespace != OPERATION_NAMESPACE:
        raise ValueError("operation id must start with 'code_mode.'")

    return OperationIdParts(
        namespace=namespace,
        tier=_validate_tier(tier),
        name=_validate_name(name),
    )


def validate_operation_id(operation_id: str) -> str:
    """Validate an operation id and return the original value on success."""
    parse_operation_id(operation_id)
    return operation_id


def is_valid_operation_id(operation_id: str) -> bool:
    """Return True when ``operation_id`` is canonical and valid."""
    try:
        parse_operation_id(operation_id)
    except ValueError:
        return False
    return True


__all__ = [
    "OPERATION_NAMESPACE",
    "OperationIdParts",
    "OperationTier",
    "build_operation_id",
    "is_valid_operation_id",
    "normalize_operation_name",
    "parse_operation_id",
    "validate_operation_id",
]
