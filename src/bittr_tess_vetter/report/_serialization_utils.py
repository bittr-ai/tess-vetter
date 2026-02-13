"""Pure serialization and scalar coercion helpers for report payloads."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _scrub_non_finite(obj: Any) -> Any:
    """Replace NaN/Inf float values with None for JSON safety (RFC 8259)."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _scrub_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_non_finite(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_scrub_non_finite(v) for v in obj)
    return obj


def _normalize_for_hash(obj: Any) -> Any:
    """Normalize payload for deterministic hashing."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _normalize_for_hash(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_for_hash(v) for v in obj]
    if isinstance(obj, tuple):
        return [_normalize_for_hash(v) for v in obj]
    return obj


def _canonical_sha256(payload: dict[str, Any]) -> str:
    normalized = _normalize_for_hash(payload)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _coerce_finite_float(value: Any, *, scale: float = 1.0) -> float | None:
    """Best-effort float coercion with finite guard."""
    try:
        out = float(value) * scale
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _coerce_int(value: Any) -> int | None:
    """Best-effort integer coercion for scalar summary fields."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
