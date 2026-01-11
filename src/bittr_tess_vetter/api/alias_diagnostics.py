"""Alias diagnostics for the public API.

This module re-exports metrics-only alias computations from
`bittr_tess_vetter.validation.alias_diagnostics` to provide stable imports for
host applications.
"""

from __future__ import annotations

from bittr_tess_vetter.validation.alias_diagnostics import (  # noqa: F401
    AliasClass,
    HARMONIC_LABELS,
    HarmonicScore,
    PhaseShiftEvent,
    classify_alias,
    compute_harmonic_scores,
    compute_secondary_significance,
    detect_phase_shift_events,
)

__all__ = [
    "AliasClass",
    "HARMONIC_LABELS",
    "HarmonicScore",
    "PhaseShiftEvent",
    "classify_alias",
    "compute_harmonic_scores",
    "compute_secondary_significance",
    "detect_phase_shift_events",
]
