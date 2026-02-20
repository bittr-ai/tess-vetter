"""Transit recovery primitives API surface (host-facing).

This module provides a stable facade for low-level recovery helpers used by
host applications and orchestration layers.
"""

from __future__ import annotations

from tess_vetter.recovery.primitives import (  # noqa: F401
    detrend_for_recovery,
    estimate_rotation_period,
)

__all__ = [
    "estimate_rotation_period",
    "detrend_for_recovery",
]
