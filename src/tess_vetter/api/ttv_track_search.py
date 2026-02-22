"""TTV track search public API.

This provides a stable API wrapper for the internal TTV track search.
It is designed for "production hardening" use cases:
  - Deterministic results via explicit random seed
  - Shape/unit validation
  - Optional lightweight flux normalization
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

from tess_vetter.api.references import (
    AGOL_2005,
    HOLMAN_MURRAY_2005,
    STEFFEN_AGOL_2006,
    cite,
    cites,
)
from tess_vetter.api.types import Candidate, LightCurve
from tess_vetter.transit.ttv_track_search import (
    TTVSearchBudget,
    TTVTrackSearchResult,
)
from tess_vetter.transit.ttv_track_search import (
    estimate_search_cost as _estimate_search_cost,
)
from tess_vetter.transit.ttv_track_search import (
    identify_observing_windows as _identify_observing_windows,
)
from tess_vetter.transit.ttv_track_search import (
    run_ttv_track_search as _run_ttv_track_search,
)
from tess_vetter.transit.ttv_track_search import (
    should_run_ttv_search as _should_run_ttv_search,
)

__all__ = [
    "TTV_SEARCH_COST_SCHEMA_VERSION",
    "TTV_TRACK_SEARCH_RESULT_SCHEMA_VERSION",
    "TTVSearchCostEstimatePayload",
    "TTVTrackHypothesisPayload",
    "TTVTrackCandidatePayload",
    "TTVTrackSearchResultPayload",
    "TTVSearchBudget",
    "TTVTrackSearchResult",
    "estimate_search_cost",
    "identify_observing_windows",
    "run_ttv_track_search",
    "run_ttv_track_search_for_candidate",
    "to_ttv_track_search_result_payload",
    "should_run_ttv_search",
]

TTV_SEARCH_COST_SCHEMA_VERSION = 1
TTV_TRACK_SEARCH_RESULT_SCHEMA_VERSION = 1


class TTVSearchCostEstimatePayload(TypedDict):
    schema_version: int
    n_windows: int
    period_steps: int
    tracks_per_period: int
    theoretical_total_tracks: int
    budget_limited_tracks: int
    estimated_seconds: float
    will_hit_budget: bool


class TTVTrackHypothesisPayload(TypedDict):
    track_id: str
    base_period_days: float
    base_t0_btjd: float
    window_offsets_days: list[float]
    score: float
    score_improvement: float
    n_transits_matched: int


class TTVTrackCandidatePayload(TypedDict):
    best_track: TTVTrackHypothesisPayload
    alternative_tracks: list[TTVTrackHypothesisPayload]
    periodic_score: float
    per_transit_residuals: list[float]
    runtime_seconds: float
    provenance: dict[str, Any]


class TTVTrackSearchResultPayload(TypedDict):
    schema_version: int
    candidates: list[TTVTrackCandidatePayload]
    n_periods_searched: int
    n_tracks_evaluated: int
    runtime_seconds: float
    budget_exhausted: bool
    provenance: dict[str, Any]


def to_ttv_track_search_result_payload(
    result: TTVTrackSearchResult,
) -> TTVTrackSearchResultPayload:
    data = result.to_dict()
    return {
        "schema_version": TTV_TRACK_SEARCH_RESULT_SCHEMA_VERSION,
        "candidates": data["candidates"],
        "n_periods_searched": data["n_periods_searched"],
        "n_tracks_evaluated": data["n_tracks_evaluated"],
        "runtime_seconds": data["runtime_seconds"],
        "budget_exhausted": data["budget_exhausted"],
        "provenance": data["provenance"],
    }


def identify_observing_windows(
    time_btjd: np.ndarray,
    *,
    gap_threshold_days: float = 5.0,
) -> list[tuple[float, float]]:
    return _identify_observing_windows(time_btjd, gap_threshold_days=gap_threshold_days)


def should_run_ttv_search(
    time_btjd: np.ndarray,
    *,
    min_baseline_days: float = 100.0,
    min_windows: int = 3,
    gap_threshold_days: float = 5.0,
) -> bool:
    return _should_run_ttv_search(
        time_btjd,
        min_baseline_days=min_baseline_days,
        min_windows=min_windows,
        gap_threshold_days=gap_threshold_days,
    )


def estimate_search_cost(
    time_btjd: np.ndarray,
    *,
    period_steps: int,
    n_offset_steps: int,
    max_tracks_per_period: int,
    budget: TTVSearchBudget,
    gap_threshold_days: float = 5.0,
) -> TTVSearchCostEstimatePayload:
    data = _estimate_search_cost(
        time_btjd,
        period_steps=period_steps,
        n_offset_steps=n_offset_steps,
        max_tracks_per_period=max_tracks_per_period,
        budget=budget,
        gap_threshold_days=gap_threshold_days,
    )
    return {
        "schema_version": TTV_SEARCH_COST_SCHEMA_VERSION,
        "n_windows": int(data["n_windows"]),
        "period_steps": int(data["period_steps"]),
        "tracks_per_period": int(data["tracks_per_period"]),
        "theoretical_total_tracks": int(data["theoretical_total_tracks"]),
        "budget_limited_tracks": int(data["budget_limited_tracks"]),
        "estimated_seconds": float(data["estimated_seconds"]),
        "will_hit_budget": bool(data["will_hit_budget"]),
    }


@cites(
    cite(HOLMAN_MURRAY_2005, "TTVs as a probe of additional planets"),
    cite(AGOL_2005, "timing-based detection sensitivity"),
    cite(STEFFEN_AGOL_2006, "TTV detection methodology and developments"),
)
def run_ttv_track_search(
    time_btjd: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    period_span_fraction: float = 0.005,
    period_steps: int = 50,
    max_offset_days: float = 0.1,
    n_offset_steps: int = 5,
    max_tracks_per_period: int = 200,
    min_score_improvement: float = 2.0,
    gap_threshold_days: float = 5.0,
    budget: TTVSearchBudget | None = None,
    random_seed: int = 42,
    normalize_flux: bool = True,
) -> TTVTrackSearchResult:
    """Run a TTV track search around a known ephemeris.

    This function implements a RIVERS-like "track search" that can recover transit
    signals with large transit timing variations (TTVs) that periodic-only BLS/TLS
    can miss. It searches over bounded timing offsets per observing window and
    scores each offset "track" by the improvement of a simple box transit model
    vs. a periodic-only model.

    Args:
        time_btjd: Time array in Barycentric TESS Julian Date (BTJD = BJD - 2457000).
        flux: Flux array. Will be normalized to ~1 by median if normalize_flux=True.
        flux_err: Flux uncertainty array. Scaled consistently with flux normalization.
        period_days: Base orbital period in days to search around.
        t0_btjd: Base transit epoch in BTJD.
        duration_hours: Transit duration in hours.
        period_span_fraction: Search +/- this fraction around period_days.
        period_steps: Number of period grid points.
        max_offset_days: Max per-window timing offset (days).
        n_offset_steps: Grid resolution for per-window offsets.
        max_tracks_per_period: Max offset tracks evaluated per period.
        min_score_improvement: Minimum improvement over periodic to report.
        gap_threshold_days: Gap size threshold for window splitting.
        budget: Optional compute budget.
        random_seed: Seed for deterministic grid sampling.
        normalize_flux: If True, divides flux and flux_err by nanmedian(flux).
            Default is True.

    Returns:
        TTVTrackSearchResult containing:
            - candidates: List of TTVTrackCandidate objects with best tracks found.
            - n_periods_searched: Number of period values searched.
            - n_tracks_evaluated: Total number of track hypotheses evaluated.
            - runtime_seconds: Total search runtime.
            - budget_exhausted: Whether the search was terminated due to budget.

    Example:
        >>> import numpy as np
        >>> from tess_vetter.api import run_ttv_track_search
        >>> result = run_ttv_track_search(
        ...     time_btjd=time,
        ...     flux=flux,
        ...     flux_err=flux_err,
        ...     period_days=3.5,
        ...     t0_btjd=1850.0,
        ...     duration_hours=2.5,
        ... )
        >>> print(f"Found {len(result.candidates)} candidate(s)")
    """
    time_btjd = np.asarray(time_btjd, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    if normalize_flux:
        m = float(np.nanmedian(flux)) if flux.size else 1.0
        if np.isfinite(m) and m != 0.0:
            flux = flux / m
            flux_err = flux_err / abs(m)
    return _run_ttv_track_search(
        time_btjd,
        flux,
        flux_err,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        period_span_fraction=period_span_fraction,
        period_steps=period_steps,
        max_offset_days=max_offset_days,
        n_offset_steps=n_offset_steps,
        max_tracks_per_period=max_tracks_per_period,
        min_score_improvement=min_score_improvement,
        gap_threshold_days=gap_threshold_days,
        budget=budget,
        random_seed=random_seed,
    )


@cites(
    cite(HOLMAN_MURRAY_2005, "TTVs as a probe of additional planets"),
    cite(AGOL_2005, "timing-based detection sensitivity"),
    cite(STEFFEN_AGOL_2006, "TTV detection methodology and developments"),
)
def run_ttv_track_search_for_candidate(
    lc: LightCurve,
    candidate: Candidate,
    *,
    period_span_fraction: float = 0.005,
    period_steps: int = 50,
    max_offset_days: float = 0.1,
    n_offset_steps: int = 5,
    max_tracks_per_period: int = 200,
    min_score_improvement: float = 2.0,
    gap_threshold_days: float = 5.0,
    budget: TTVSearchBudget | None = None,
    random_seed: int = 42,
    normalize_flux: bool = True,
) -> TTVTrackSearchResult:
    """Run TTV track search using a Candidate's ephemeris.

    This is a convenience wrapper around run_ttv_track_search that extracts
    the ephemeris parameters (period, t0, duration) from a Candidate object
    and applies them to a LightCurve.

    Args:
        lc: LightCurve object containing time, flux, and flux_err arrays.
            Invalid data points (NaN/Inf) are automatically masked.
        candidate: Candidate object containing the ephemeris to search around.
            The ephemeris provides period_days, t0_btjd, and duration_hours.
        period_span_fraction: Search +/- this fraction around period_days.
        period_steps: Number of period grid points.
        max_offset_days: Max per-window timing offset (days).
        n_offset_steps: Grid resolution for per-window offsets.
        max_tracks_per_period: Max offset tracks evaluated per period.
        min_score_improvement: Minimum improvement over periodic to report.
        gap_threshold_days: Gap size threshold for window splitting.
        budget: Optional compute budget.
        random_seed: Seed for deterministic grid sampling.
        normalize_flux: If True, divides flux and flux_err by nanmedian(flux).
            Default is True.

    Returns:
        TTVTrackSearchResult containing candidate tracks and search metadata.
        See run_ttv_track_search for full result structure.

    Example:
        >>> from tess_vetter.api import (
        ...     LightCurve, Candidate, Ephemeris,
        ...     run_ttv_track_search_for_candidate
        ... )
        >>> lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        >>> candidate = Candidate(
        ...     ephemeris=Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=2.5),
        ...     depth_ppm=500,
        ... )
        >>> result = run_ttv_track_search_for_candidate(lc, candidate)
        >>> for track_candidate in result.candidates:
        ...     print(f"Score improvement: {track_candidate.best_track.score_improvement:.3f}")
    """
    internal = lc.to_internal()
    mask = internal.valid_mask
    time = internal.time[mask]
    flux = internal.flux[mask]
    flux_err = internal.flux_err[mask]
    eph = candidate.ephemeris
    return run_ttv_track_search(
        time,
        flux,
        flux_err,
        period_days=eph.period_days,
        t0_btjd=eph.t0_btjd,
        duration_hours=eph.duration_hours,
        period_span_fraction=period_span_fraction,
        period_steps=period_steps,
        max_offset_days=max_offset_days,
        n_offset_steps=n_offset_steps,
        max_tracks_per_period=max_tracks_per_period,
        min_score_improvement=min_score_improvement,
        gap_threshold_days=gap_threshold_days,
        budget=budget,
        random_seed=random_seed,
        normalize_flux=normalize_flux,
    )
