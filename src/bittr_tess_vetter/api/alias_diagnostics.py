"""Alias diagnostics for the public API.

This module re-exports metrics-only alias computations from
`bittr_tess_vetter.validation.alias_diagnostics` to provide stable imports for
host applications.
"""

from __future__ import annotations

import numpy as np

from bittr_tess_vetter.api.types import Candidate, LightCurve
from bittr_tess_vetter.validation.alias_diagnostics import (  # noqa: F401
    HARMONIC_LABELS,
    AliasClass,
    HarmonicPowerSummary,
    HarmonicScore,
    PhaseShiftEvent,
    classify_alias,
    compute_harmonic_scores,
    compute_secondary_significance,
    detect_phase_shift_events,
    summarize_harmonic_power,
)

__all__ = [
    "AliasClass",
    "HARMONIC_LABELS",
    "HarmonicPowerSummary",
    "HarmonicScore",
    "PhaseShiftEvent",
    "classify_alias",
    "compute_harmonic_scores",
    "compute_secondary_significance",
    "detect_phase_shift_events",
    "harmonic_power_summary",
    "summarize_harmonic_power",
]


def harmonic_power_summary(
    lc: LightCurve,
    candidate: Candidate,
) -> HarmonicPowerSummary:
    """Compute compact alias summary near P, P/2, and 2P."""
    internal = lc.to_internal()
    valid = np.asarray(internal.valid_mask, dtype=np.bool_)
    time = np.asarray(internal.time, dtype=np.float64)[valid]
    flux = np.asarray(internal.flux, dtype=np.float64)[valid]
    flux_err = np.asarray(internal.flux_err, dtype=np.float64)[valid]
    eph = candidate.ephemeris
    return summarize_harmonic_power(
        time=time,
        flux=flux,
        flux_err=flux_err,
        base_period=eph.period_days,
        base_t0=eph.t0_btjd,
        duration_hours=eph.duration_hours,
    )
