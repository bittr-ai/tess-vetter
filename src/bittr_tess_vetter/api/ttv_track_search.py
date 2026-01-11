"""TTV track search public API.

This provides a stable API wrapper for the internal TTV track search.
It is designed for "production hardening" use cases:
  - Deterministic results via explicit random seed
  - Shape/unit validation
  - Optional lightweight flux normalization
"""

from __future__ import annotations

import numpy as np

from bittr_tess_vetter.api.types import Candidate, LightCurve
from bittr_tess_vetter.transit.ttv_track_search import (
    TTVSearchBudget,
    TTVTrackSearchResult,
    estimate_search_cost as _estimate_search_cost,
    identify_observing_windows as _identify_observing_windows,
    run_ttv_track_search as _run_ttv_track_search,
    should_run_ttv_search as _should_run_ttv_search,
)

__all__ = [
    "TTVSearchBudget",
    "TTVTrackSearchResult",
    "estimate_search_cost",
    "identify_observing_windows",
    "run_ttv_track_search",
    "run_ttv_track_search_for_candidate",
    "should_run_ttv_search",
]


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
) -> dict[str, int | float]:
    return _estimate_search_cost(
        time_btjd,
        period_steps=period_steps,
        n_offset_steps=n_offset_steps,
        max_tracks_per_period=max_tracks_per_period,
        budget=budget,
        gap_threshold_days=gap_threshold_days,
    )


def run_ttv_track_search(
    time_btjd: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    normalize_flux: bool = True,
    **kwargs: object,
) -> TTVTrackSearchResult:
    """Run a TTV track search around a known ephemeris.

    Args:
        time_btjd: Time array in BTJD
        flux: Flux array (will be normalized to ~1 by median if normalize_flux=True)
        flux_err: Flux uncertainty array (scaled consistently with flux normalization)
        period_days: Base period
        t0_btjd: Base epoch
        duration_hours: Transit duration
        normalize_flux: If True, divides flux and flux_err by nanmedian(flux)
        **kwargs: Passed through to the internal implementation.
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
        **kwargs,  # type: ignore[arg-type]
    )


def run_ttv_track_search_for_candidate(
    lc: LightCurve,
    candidate: Candidate,
    *,
    normalize_flux: bool = True,
    **kwargs: object,
) -> TTVTrackSearchResult:
    """Convenience wrapper: run track search using a Candidate ephemeris."""
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
        normalize_flux=normalize_flux,
        **kwargs,
    )

