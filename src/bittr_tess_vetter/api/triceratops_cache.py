"""TRICERATOPS cache + helper API facade (host-facing).

This module exposes a small, reusable surface area around TRICERATOPS caching
and helper routines that host applications may need for:
- warming TRILEGAL downloads (slow external dependency)
- persisting TRICERATOPS Target objects between runs
- estimating durations when duration_hours is missing

The underlying implementation lives in `bittr_tess_vetter.validation.triceratops_fpp`,
but hosts should import from `bittr_tess_vetter.api.*` only.
"""

from __future__ import annotations

from bittr_tess_vetter.validation.triceratops_fpp import (  # noqa: F401
    CalculateFppInput,
    FppResult,
    _estimate_transit_duration as estimate_transit_duration,
    _load_cached_triceratops_target as load_cached_triceratops_target,
    _prefetch_trilegal_csv as prefetch_trilegal_csv,
    _save_cached_triceratops_target as save_cached_triceratops_target,
)


def get_disposition(fpp: float, nfpp: float) -> str:
    """Map TRICERATOPS FPP/NFPP into a coarse disposition label.

    Community threshold for statistical validation is typically FPP < 0.01.
    We also use NFPP (nearby false-positive probability) to indicate when the
    signal may be coming from a nearby source even if overall FPP is low.

    Labels are stable strings intended for UI/reporting, not as a hard verdict.
    """
    if fpp < 0.01:
        if nfpp < 0.001:
            return "VALIDATED"
        return "LIKELY_PLANET_NEARBY_UNCERTAIN"
    if fpp < 0.05:
        return "LIKELY_PLANET"
    if fpp < 0.5:
        return "INCONCLUSIVE"
    if fpp < 0.9:
        return "LIKELY_FP"
    return "FALSE_POSITIVE"
