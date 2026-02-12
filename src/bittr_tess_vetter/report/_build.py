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
    phase_folded = _build_phase_folded_plot_data(lc, ephemeris, bin_minutes)

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


def _build_phase_folded_plot_data(
    lc: LightCurve,
    ephemeris: Ephemeris,
    bin_minutes: float,
) -> PhaseFoldedPlotData:
    """Build plot-ready arrays for phase-folded transit panel."""
    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)

    # Apply valid mask if present
    valid = np.isfinite(time) & np.isfinite(flux)
    if lc.valid_mask is not None:
        valid = valid & np.asarray(lc.valid_mask, dtype=np.bool_)

    time_v = time[valid]
    flux_v = flux[valid]

    # Phase-fold using existing function (returns sorted arrays)
    phase, flux_folded = fold_transit(
        time_v, flux_v, ephemeris.period_days, ephemeris.t0_btjd
    )

    # Bin the phase-folded data
    bin_centers, bin_flux, bin_err = _bin_phase_data(
        phase, flux_folded, ephemeris.period_days, bin_minutes
    )

    return PhaseFoldedPlotData(
        phase=phase.tolist(),
        flux=flux_folded.tolist(),
        bin_centers=bin_centers,
        bin_flux=bin_flux,
        bin_err=bin_err,
        bin_minutes=bin_minutes,
    )


def _bin_phase_data(
    phase: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    bin_minutes: float,
) -> tuple[list[float], list[float], list[float | None]]:
    """Bin phase-folded data by phase.

    Args:
        phase: Phase values (centered on 0).
        flux: Flux values.
        period_days: Orbital period in days.
        bin_minutes: Bin size in minutes.

    Returns:
        Tuple of (bin_centers, bin_flux, bin_err) lists.
        bin_err entries are None for single-point bins.
    """
    if len(phase) == 0:
        return [], [], []

    # Convert bin size from minutes to phase units
    bin_phase = (bin_minutes / 60.0 / 24.0) / period_days

    # Determine bin edges
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
