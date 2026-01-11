"""Ephemeris reliability-regime diagnostics for the public API.

Delegates to `bittr_tess_vetter.validation.ephemeris_reliability` to avoid
duplicating numerical logic.
"""

from __future__ import annotations

from bittr_tess_vetter.validation.ephemeris_reliability import (  # noqa: F401
    AblationResult,
    EphemerisReliabilityRegimeResult,
    PeriodNeighborhoodResult,
    compute_reliability_regime_numpy,
)

__all__ = [
    "AblationResult",
    "EphemerisReliabilityRegimeResult",
    "PeriodNeighborhoodResult",
    "compute_reliability_regime_numpy",
]

