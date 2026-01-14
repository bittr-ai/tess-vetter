"""Ephemeris matching diagnostics for the public API.

This module exposes deterministic ephemeris-matching primitives and a simple
index format. It delegates all numerical logic to
`bittr_tess_vetter.validation.ephemeris_match` so host apps can rely on stable
imports from `bittr_tess_vetter.api.*` while implementation details stay in
the metrics-only layer.
"""

from __future__ import annotations

from bittr_tess_vetter.validation.ephemeris_match import (  # noqa: F401
    EphemerisEntry,
    EphemerisIndex,
    EphemerisMatch,
    EphemerisMatchResult,
    MatchClass,
    build_index_from_csv,
    classify_matches,
    compute_harmonic_match,
    compute_match_score,
    load_index,
    run_ephemeris_matching,
    save_index,
    wrap_t0,
)

__all__ = [
    "EphemerisEntry",
    "EphemerisIndex",
    "EphemerisMatch",
    "EphemerisMatchResult",
    "MatchClass",
    "build_index_from_csv",
    "classify_matches",
    "compute_harmonic_match",
    "compute_match_score",
    "load_index",
    "run_ephemeris_matching",
    "save_index",
    "wrap_t0",
]
