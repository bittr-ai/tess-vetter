"""TTV track search public API.

This provides a stable API wrapper for the internal TTV track search.
It is designed for "production hardening" use cases:
  - Deterministic results via explicit random seed
  - Shape/unit validation
  - Optional lightweight flux normalization
"""

from __future__ import annotations

import numpy as np

from bittr_tess_vetter.api.references import (
    AGOL_2005,
    HOLMAN_MURRAY_2005,
    STEFFEN_AGOL_2006,
    cite,
    cites,
)
from bittr_tess_vetter.api.types import Candidate, LightCurve
from bittr_tess_vetter.transit.ttv_track_search import (
    TTVSearchBudget,
    TTVTrackSearchResult,
)
from bittr_tess_vetter.transit.ttv_track_search import (
    estimate_search_cost as _estimate_search_cost,
)
from bittr_tess_vetter.transit.ttv_track_search import (
    identify_observing_windows as _identify_observing_windows,
)
from bittr_tess_vetter.transit.ttv_track_search import (
    run_ttv_track_search as _run_ttv_track_search,
)
from bittr_tess_vetter.transit.ttv_track_search import (
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
    normalize_flux: bool = True,
    **kwargs: object,
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
        normalize_flux: If True, divides flux and flux_err by nanmedian(flux).
            Default is True.
        **kwargs: Additional parameters passed to the internal implementation,
            including:
            - budget (TTVSearchBudget): Runtime/evaluation budget controls.
            - n_offset_steps (int): Number of timing offset steps per window.
            - max_tracks_per_period (int): Maximum tracks to evaluate per period.

    Returns:
        TTVTrackSearchResult containing:
            - candidates: List of TTVTrackCandidate objects with best tracks found.
            - n_periods_searched: Number of period values searched.
            - n_tracks_evaluated: Total number of track hypotheses evaluated.
            - runtime_seconds: Total search runtime.
            - budget_exhausted: Whether the search was terminated due to budget.

    Example:
        >>> import numpy as np
        >>> from bittr_tess_vetter.api import run_ttv_track_search
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
        **kwargs,  # type: ignore[arg-type]
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
    normalize_flux: bool = True,
    **kwargs: object,
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
        normalize_flux: If True, divides flux and flux_err by nanmedian(flux).
            Default is True.
        **kwargs: Additional parameters passed to run_ttv_track_search,
            including budget controls and search parameters.

    Returns:
        TTVTrackSearchResult containing candidate tracks and search metadata.
        See run_ttv_track_search for full result structure.

    Example:
        >>> from bittr_tess_vetter.api import (
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
        normalize_flux=normalize_flux,
        **kwargs,
    )
