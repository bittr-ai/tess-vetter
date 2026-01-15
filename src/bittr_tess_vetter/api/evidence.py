"""Evidence-friendly conversion helpers for the public API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import numpy as np

from bittr_tess_vetter.api.types import CheckResult


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(v) for v in value]
    # Best-effort fallback: keep it stable/serializable.
    return str(value)


def checks_to_evidence_items(checks: list[CheckResult]) -> list[dict[str, Any]]:
    """Convert CheckResult objects to JSON-serializable evidence-like dicts.

    The canonical CheckResult uses status-based semantics (ok/skipped/error).
    For backward compatibility, the `passed` property is derived from status:
    - status="ok" -> passed=True
    - status="error" -> passed=False
    - status="skipped" -> passed=None
    """
    items: list[dict[str, Any]] = []
    for c in checks:
        # The .details property combines metrics/flags/notes/raw for backward compat
        details = dict(c.details)
        # All results are metrics-only in the new schema
        metrics_only = True
        items.append(
            {
                "id": c.id,
                "name": c.name,
                "passed": c.passed,  # Derived from status via property
                "confidence": c.confidence,
                "metrics_only": metrics_only,
                "details": _jsonable(details),
            }
        )
    return items
