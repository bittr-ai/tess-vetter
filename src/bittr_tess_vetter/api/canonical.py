"""Canonical JSON serialization utilities (host-facing).

This module provides deterministic JSON encoding suitable for hashing and
provenance tracking. It is intentionally strict:
- keys are sorted
- floats are quantized
- NaN/Inf are rejected
- output is compact (no whitespace)
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime
from typing import Any

import numpy as np

FLOAT_DECIMAL_PLACES = 10


def _quantize_float(value: float) -> int | float:
    if math.isnan(value):
        raise ValueError("NaN is not allowed in canonical JSON")
    if math.isinf(value):
        raise ValueError("Inf is not allowed in canonical JSON")

    rounded = round(float(value), FLOAT_DECIMAL_PLACES)
    if rounded == 0.0:
        return 0
    if float(rounded).is_integer():
        return int(rounded)
    return float(rounded)


def _normalize(obj: Any) -> Any:
    if obj is None or obj is True or obj is False:
        return obj

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, (int,)) and not isinstance(obj, bool):
        return int(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (float,)) and not isinstance(obj, bool):
        return _quantize_float(obj)

    if isinstance(obj, (np.floating,)):
        return _quantize_float(float(obj))

    if isinstance(obj, (str,)):
        return obj

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, (np.ndarray,)):
        # Reject NaN/Inf early (tests require this).
        if np.issubdtype(obj.dtype, np.floating):
            if np.any(np.isnan(obj)):
                raise ValueError("NaN is not allowed in canonical JSON")
            if np.any(np.isinf(obj)):
                raise ValueError("Inf is not allowed in canonical JSON")
        return _normalize(obj.tolist())

    if isinstance(obj, (list, tuple)):
        return [_normalize(x) for x in obj]

    if isinstance(obj, (set, frozenset)):
        normalized_items = [_normalize(x) for x in obj]
        try:
            return sorted(normalized_items)
        except TypeError:
            # Mixed types: fall back to string sort for determinism.
            return sorted(normalized_items, key=lambda x: str(x))

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out[str(k)] = _normalize(v)
        # json.dumps(sort_keys=True) will handle ordering.
        return out

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class CanonicalEncoder(json.JSONEncoder):
    """JSONEncoder with deterministic defaults for supported types."""

    def default(self, o: Any) -> Any:  # noqa: D401
        return _normalize(o)


def canonical_json(obj: Any) -> bytes:
    """Serialize to canonical JSON bytes (UTF-8, compact, sorted keys)."""
    normalized = _normalize(obj)
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        cls=CanonicalEncoder,
    )
    return payload.encode("utf-8")


def canonical_hash(obj: Any) -> str:
    """SHA-256 hex digest of canonical_json(obj)."""
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def canonical_hash_prefix(obj: Any, *, length: int = 12) -> str:
    """Prefix of the SHA-256 canonical hash."""
    if int(length) < 1 or int(length) > 64:
        raise ValueError("length must be between 1 and 64")
    return canonical_hash(obj)[: int(length)]
