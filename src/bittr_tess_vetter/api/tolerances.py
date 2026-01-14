"""Tolerance checking utilities for the public API.

Delegates to `bittr_tess_vetter.utils.tolerances`.
"""

from __future__ import annotations

from bittr_tess_vetter.utils.tolerances import (  # noqa: F401
    HARMONIC_RATIOS,
    ToleranceResult,
    _check_default_tolerance,
    _check_depth_tolerance,
    _check_period_tolerance,
    _check_t0_tolerance,
    _format_ratio,
    check_tolerance,
)

__all__ = [
    "ToleranceResult",
    "HARMONIC_RATIOS",
    "check_tolerance",
    "_check_period_tolerance",
    "_check_t0_tolerance",
    "_check_depth_tolerance",
    "_check_default_tolerance",
    "_format_ratio",
]
