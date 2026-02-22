"""Tier assignment policy helpers for API symbols."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from tess_vetter.code_mode.registries.operation_ids import OperationTier

# Keep this intentionally curated and small: explicit golden-path overrides only.
DEFAULT_GOLDEN_PATH_SYMBOLS: Final[frozenset[str]] = frozenset(
    {
        "vet_candidate",
        "run_periodogram",
    }
)

_PRIMITIVE_MODULE_TOKENS: Final[tuple[str, ...]] = (
    "primitives",
    "primitive",
)
_SNAKE_BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(r"(?<!^)(?=[A-Z])")
_NON_IDENT_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9_]+")


@dataclass(frozen=True, slots=True)
class ApiSymbol:
    """Normalized symbol identity from the public API surface."""

    module: str
    name: str



def normalize_api_symbol(symbol: str | ApiSymbol) -> ApiSymbol:
    """Normalize string or ``ApiSymbol`` input into a structured symbol."""
    if isinstance(symbol, ApiSymbol):
        return symbol

    raw = symbol.strip()
    if not raw:
        raise ValueError("symbol must be non-empty")

    if ":" in raw:
        module, name = raw.rsplit(":", 1)
    elif "." in raw:
        module, name = raw.rsplit(".", 1)
    else:
        module, name = "", raw

    return ApiSymbol(module=module, name=name)


def _normalize_policy_name(name: str) -> str:
    normalized = _SNAKE_BOUNDARY_RE.sub("_", name).lower()
    normalized = normalized.replace("-", "_")
    normalized = _NON_IDENT_RE.sub("_", normalized).strip("_")
    return normalized


def tier_for_api_symbol(
    symbol: str | ApiSymbol,
    *,
    golden_path_symbols: frozenset[str] = DEFAULT_GOLDEN_PATH_SYMBOLS,
) -> OperationTier:
    """Assign tier for an API symbol using lightweight policy heuristics.

    Policy precedence:
    1) explicit golden-path symbol names
    2) primitive module naming hints
    3) internal fallback
    """
    normalized = normalize_api_symbol(symbol)
    normalized_name = _normalize_policy_name(normalized.name)

    if normalized_name in golden_path_symbols:
        return "golden_path"

    module_lc = normalized.module.lower()
    if any(token in module_lc for token in _PRIMITIVE_MODULE_TOKENS):
        return "primitive"

    return "internal"


__all__ = [
    "ApiSymbol",
    "DEFAULT_GOLDEN_PATH_SYMBOLS",
    "normalize_api_symbol",
    "tier_for_api_symbol",
]
