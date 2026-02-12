"""Core assembly logic for LC-only vetting reports.

Implements build_report() and its private helpers for computing
LC summary stats, plot-ready arrays, and assembling ReportData.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

from bittr_tess_vetter.api.lc_only import vet_lc_only
from bittr_tess_vetter.api.types import (
    Candidate,
    Ephemeris,
    LightCurve,
    StellarParams,
    VettingBundleResult,
)
from bittr_tess_vetter.compute.transit import (
    detect_transit,
    fold_transit,
    measure_depth,
)
from bittr_tess_vetter.report._data import (
    FullLCPlotData,
    LCSummary,
    PhaseFoldedPlotData,
    ReportData,
)
from bittr_tess_vetter.validation.base import (
    count_transits,
    get_in_transit_mask,
    get_out_of_transit_mask,
)

logger = logging.getLogger(__name__)

# Default Phase 1 checks (V03 excluded by default)
_DEFAULT_ENABLED = {"V01", "V02", "V04", "V05", "V13", "V15"}


def build_report(
    lc: LightCurve,
    candidate: Candidate,
    *,
    stellar: StellarParams | None = None,
    tic_id: int | None = None,
    toi: str | None = None,
    include_v03: bool = False,
    bin_minutes: float = 30.0,
    check_config: dict[str, dict[str, Any]] | None = None,
    max_lc_points: int = 50_000,
    max_phase_points: int = 10_000,
) -> ReportData:
    """Assemble LC-only report data packet.

    Runs vetting checks via vet_lc_only(), computes LC summary stats
    and SNR, assembles plot-ready arrays, and returns a ReportData.

    Args:
        lc: Light curve data.
        candidate: Transit candidate (provides ephemeris + depth).
        stellar: Stellar params (enables V03 if include_v03=True).
        tic_id: TIC identifier for display.
        toi: TOI designation for display.
        include_v03: If True, include V03 duration consistency.
            Only meaningful when stellar is not None.
        bin_minutes: Phase-fold bin width in minutes.
        check_config: Per-check config dicts (e.g., {"V01": {...}}).
        max_lc_points: Downsample full LC arrays if longer than this.
            Prevents bloated JSON for multi-sector LCs.
        max_phase_points: Downsample phase-folded raw arrays if longer
            than this.  Near-transit points (within the duration-based
            display window) are preserved at full resolution.

    Returns:
        ReportData with all Phase 1 contents populated.
    """
    ephemeris = candidate.ephemeris

    # 1. Determine enabled checks
    enabled = set(_DEFAULT_ENABLED)
    if include_v03:
        enabled.add("V03")

    # 2. Run checks
    results = vet_lc_only(
        lc,
        ephemeris,
        stellar=stellar,
        enabled=enabled,
        config=check_config,
    )

    # 3. Build bundle
    bundle = VettingBundleResult.from_checks(results)

    # 4. Compute LC summary + SNR
    lc_summary = _compute_lc_summary(lc, ephemeris)

    # 5. Compute full-LC plot arrays
    full_lc = _build_full_lc_plot_data(lc, ephemeris, max_lc_points)

    # 6. Compute phase-folded plot arrays
    phase_folded = _build_phase_folded_plot_data(
        lc, ephemeris, bin_minutes, max_phase_points
    )

    # 7. Assemble ReportData
    return ReportData(
        tic_id=tic_id,
        toi=toi,
        candidate=candidate,
        stellar=stellar,
        lc_summary=lc_summary,
        checks={r.id: r for r in results},
        bundle=bundle,
        full_lc=full_lc,
        phase_folded=phase_folded,
        checks_run=[r.id for r in results],
    )


def _compute_lc_summary(
    lc: LightCurve,
    ephemeris: Ephemeris,
) -> LCSummary:
    """Compute LC vital signs. ~1ms for typical TESS LC."""
    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)
    flux_err = (
        np.asarray(lc.flux_err, dtype=np.float64)
        if lc.flux_err is not None
        else np.zeros_like(flux)
    )

    n_points = len(time)

    # Valid mask: finite time + flux + flux_err, intersected with user valid_mask
    valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    if lc.valid_mask is not None:
        valid = valid & np.asarray(lc.valid_mask, dtype=np.bool_)

    n_valid = int(np.sum(valid))
    gap_fraction = 1.0 - n_valid / n_points if n_points > 0 else 0.0

    time_valid = time[valid]
    flux_valid = flux[valid]
    flux_err_valid = flux_err[valid]

    # Time baseline
    duration_days = float(np.ptp(time_valid)) if n_valid > 1 else 0.0

    # Median cadence
    if n_valid > 1:
        diffs = np.diff(np.sort(time_valid))
        cadence_seconds = float(np.median(diffs) * 86400.0)
    else:
        cadence_seconds = 0.0

    period = ephemeris.period_days
    t0 = ephemeris.t0_btjd
    dur_h = ephemeris.duration_hours

    # Transit counts
    n_transits = count_transits(time_valid, period, t0, dur_h, min_points=3)

    # In-transit mask (on valid data)
    in_mask = get_in_transit_mask(time_valid, period, t0, dur_h)
    n_in_transit_total = int(np.sum(in_mask))

    # Out-of-transit mask (on valid data) for scatter stats
    oot_mask = get_out_of_transit_mask(time_valid, period, t0, dur_h, buffer_factor=2.0)
    flux_oot = flux_valid[oot_mask]

    if len(flux_oot) > 1:
        flux_std_ppm = float(np.std(flux_oot, ddof=1) * 1e6)
        median_oot = float(np.median(flux_oot))
        flux_mad_ppm = float(1.4826 * np.median(np.abs(flux_oot - median_oot)) * 1e6)
    else:
        flux_std_ppm = 0.0
        flux_mad_ppm = 0.0

    # SNR + depth via detect_transit
    snr = 0.0
    depth_ppm = 0.0
    depth_err_ppm: float | None = None

    try:
        tc = detect_transit(time_valid, flux_valid, flux_err_valid, period, t0, dur_h)
        snr = float(tc.snr)
        depth_ppm = float(tc.depth * 1e6)

        # Compute depth_err separately since TransitCandidate doesn't expose it
        in_mask_depth = get_in_transit_mask(time_valid, period, t0, dur_h)
        _, d_err = measure_depth(flux_valid, in_mask_depth)
        depth_err_ppm = float(d_err * 1e6)
    except Exception as exc:
        warnings.warn(
            f"detect_transit failed, using snr=0.0: {exc}",
            stacklevel=2,
        )

    return LCSummary(
        n_points=n_points,
        n_valid=n_valid,
        n_transits=n_transits,
        n_in_transit_total=n_in_transit_total,
        duration_days=duration_days,
        cadence_seconds=cadence_seconds,
        flux_std_ppm=flux_std_ppm,
        flux_mad_ppm=flux_mad_ppm,
        gap_fraction=gap_fraction,
        snr=snr,
        depth_ppm=depth_ppm,
        depth_err_ppm=depth_err_ppm,
    )


def _downsample_preserving_transits(
    time: np.ndarray,
    flux: np.ndarray,
    transit_mask: np.ndarray,
    max_points: int,
) -> tuple[list[float], list[float], list[bool]]:
    """Downsample LC while keeping all in-transit points."""
    in_transit_idx = np.where(transit_mask)[0]
    oot_idx = np.where(~transit_mask)[0]

    n_oot_budget = max_points - len(in_transit_idx)
    if n_oot_budget <= 0 or len(oot_idx) <= n_oot_budget:
        # No downsampling needed or all points are in-transit
        return time.tolist(), flux.tolist(), transit_mask.tolist()

    # Evenly sample OOT points
    step = max(1, len(oot_idx) // n_oot_budget)
    sampled_oot = oot_idx[::step][:n_oot_budget]

    keep = np.sort(np.concatenate([in_transit_idx, sampled_oot]))
    return time[keep].tolist(), flux[keep].tolist(), transit_mask[keep].tolist()


def _build_full_lc_plot_data(
    lc: LightCurve,
    ephemeris: Ephemeris,
    max_points: int,
) -> FullLCPlotData:
    """Build plot-ready arrays for full light curve panel."""
    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)

    # Apply valid mask if present
    valid = np.isfinite(time) & np.isfinite(flux)
    if lc.valid_mask is not None:
        valid = valid & np.asarray(lc.valid_mask, dtype=np.bool_)

    time_v = time[valid]
    flux_v = flux[valid]

    # Transit mask on valid data
    transit_mask = get_in_transit_mask(
        time_v, ephemeris.period_days, ephemeris.t0_btjd, ephemeris.duration_hours
    )

    # Downsample if needed
    if len(time_v) > max_points:
        t_list, f_list, m_list = _downsample_preserving_transits(
            time_v, flux_v, transit_mask, max_points
        )
    else:
        t_list = time_v.tolist()
        f_list = flux_v.tolist()
        m_list = transit_mask.tolist()

    return FullLCPlotData(time=t_list, flux=f_list, transit_mask=m_list)


def _downsample_phase_preserving_transit(
    phase: np.ndarray,
    flux: np.ndarray,
    max_points: int,
    near_transit_half_phase: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample phase-folded data while keeping near-transit points.

    Points within ±near_transit_half_phase of transit center (phase=0)
    are always kept.  Remaining baseline points are evenly thinned to
    fit within *max_points*.

    Args:
        phase: Phase values (centered on 0).
        flux: Flux values.
        max_points: Maximum number of output points.
        near_transit_half_phase: Half-width of the near-transit
            preservation window in phase units.  Defaults to 0.1 for
            backward compatibility, but callers should pass a
            duration-based value (e.g. 3 * transit_duration_phase).
    """
    near_transit = np.abs(phase) < near_transit_half_phase
    near_idx = np.where(near_transit)[0]
    far_idx = np.where(~near_transit)[0]

    n_far_budget = max_points - len(near_idx)
    if n_far_budget <= 0:
        # Near-transit points alone exceed the budget; keep only those
        # (dropping all far-from-transit baseline points).
        return phase[near_idx], flux[near_idx]

    if len(far_idx) <= n_far_budget:
        # No downsampling needed — everything fits.
        return phase, flux

    step = max(1, len(far_idx) // n_far_budget)
    sampled_far = far_idx[::step][:n_far_budget]

    keep = np.sort(np.concatenate([near_idx, sampled_far]))
    return phase[keep], flux[keep]


def _build_phase_folded_plot_data(
    lc: LightCurve,
    ephemeris: Ephemeris,
    bin_minutes: float,
    max_phase_points: int = 10_000,
) -> PhaseFoldedPlotData:
    """Build plot-ready arrays for phase-folded transit panel.

    Uses a duration-based display window of ±3 transit durations
    (in phase units) centered on the transit midpoint.  This follows
    standard practice in TESS/Kepler transit papers (e.g., Guerrero+2021,
    Ricker+2015) where phase-folded plots zoom to the transit region
    with enough pre/post-transit baseline for visual context around
    ingress and egress.

    The factor of 3 provides ~1 full transit duration of baseline on
    each side of the transit (since the transit itself occupies ±0.5
    duration), which is sufficient for triage without wasting data on
    the vast orbital baseline.  A floor of ±0.015 phase prevents
    degenerate windows for very short-duration transits.

    Bins are computed only within this display window, not across the
    full orbit.  Raw points use a two-zone strategy: full-resolution
    inside the display window, heavily downsampled outside.
    """
    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)

    # Apply valid mask if present
    valid = np.isfinite(time) & np.isfinite(flux)
    if lc.valid_mask is not None:
        valid = valid & np.asarray(lc.valid_mask, dtype=np.bool_)

    time_v = time[valid]
    flux_v = flux[valid]

    # Transit duration in phase units
    transit_duration_phase = (ephemeris.duration_hours / 24.0) / ephemeris.period_days

    # Display window: ±3 transit durations, floored at ±0.015 phase.
    # 3x gives ~1 full duration of baseline on each side of the transit.
    half_window = max(3.0 * transit_duration_phase, 0.015)
    # Cap at ±0.5 (full orbit) — can't exceed physical range.
    half_window = min(half_window, 0.5)
    phase_range = (-half_window, half_window)

    # Phase-fold using existing function (returns sorted arrays)
    phase, flux_folded = fold_transit(
        time_v, flux_v, ephemeris.period_days, ephemeris.t0_btjd
    )

    # Bin only within the display window (transit-centric region).
    # Binning the full orbit wastes compute and produces irrelevant bins.
    bin_centers, bin_flux, bin_err = _bin_phase_data(
        phase, flux_folded, ephemeris.period_days, bin_minutes,
        phase_range=phase_range,
    )

    # Downsample raw phase-folded points if needed.
    # Near-transit window for preservation matches the display window.
    if len(phase) > max_phase_points:
        phase, flux_folded = _downsample_phase_preserving_transit(
            phase, flux_folded, max_phase_points,
            near_transit_half_phase=half_window,
        )

    return PhaseFoldedPlotData(
        phase=phase.tolist(),
        flux=flux_folded.tolist(),
        bin_centers=bin_centers,
        bin_flux=bin_flux,
        bin_err=bin_err,
        bin_minutes=bin_minutes,
        transit_duration_phase=transit_duration_phase,
        phase_range=phase_range,
    )


def _bin_phase_data(
    phase: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    bin_minutes: float,
    phase_range: tuple[float, float] | None = None,
) -> tuple[list[float], list[float], list[float | None]]:
    """Bin phase-folded data by phase.

    Args:
        phase: Phase values (centered on 0).
        flux: Flux values.
        period_days: Orbital period in days.
        bin_minutes: Bin size in minutes.
        phase_range: If provided, only bin data within this phase window.
            Bins are placed to cover the window; data outside is ignored.
            If None, bins span the full range of the input data.

    Returns:
        Tuple of (bin_centers, bin_flux, bin_err) lists.
        bin_err entries are None for single-point bins.
    """
    if len(phase) == 0:
        return [], [], []

    # Convert bin size from minutes to phase units
    bin_phase = (bin_minutes / 60.0 / 24.0) / period_days

    # Determine bin edges — either from phase_range or data extent
    if phase_range is not None:
        phase_min, phase_max = phase_range
        # Filter data to the window for binning
        in_window = (phase >= phase_min) & (phase <= phase_max)
        phase = phase[in_window]
        flux = flux[in_window]
        if len(phase) == 0:
            return [], [], []
    else:
        phase_min = float(np.min(phase))
        phase_max = float(np.max(phase))

    n_bins = max(1, int(np.ceil((phase_max - phase_min) / bin_phase)))
    bin_edges = np.linspace(phase_min, phase_max, n_bins + 1)

    # Bin the data
    centers: list[float] = []
    fluxes: list[float] = []
    errors: list[float | None] = []

    for i in range(n_bins):
        if i == n_bins - 1:
            # Include right edge in last bin
            mask = (phase >= bin_edges[i]) & (phase <= bin_edges[i + 1])
        else:
            mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])

        n_in_bin = int(np.sum(mask))
        if n_in_bin > 0:
            centers.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))
            fluxes.append(float(np.mean(flux[mask])))
            # Standard error of the mean
            if n_in_bin > 1:
                errors.append(float(np.std(flux[mask], ddof=1) / np.sqrt(n_in_bin)))
            else:
                errors.append(None)

    return centers, fluxes, errors
