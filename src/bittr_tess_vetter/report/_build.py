"""Core assembly logic for LC-only vetting reports.

Implements build_report() and its private helpers for computing
LC summary stats, plot-ready arrays, and assembling ReportData.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from bittr_tess_vetter.api.alias_diagnostics import harmonic_power_summary
from bittr_tess_vetter.api.lc_only import vet_lc_only
from bittr_tess_vetter.api.timing import timing_series
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
    AliasHarmonicSummaryData,
    FullLCPlotData,
    LCSummary,
    LocalDetrendDiagnosticPlotData,
    LocalDetrendWindowData,
    OddEvenPhasePlotData,
    OOTContextPlotData,
    PerTransitStackPlotData,
    PhaseFoldedPlotData,
    ReportData,
    SecondaryScanPlotData,
    SecondaryScanQuality,
    SecondaryScanRenderHints,
    TransitTimingPlotData,
    TransitWindowData,
)
from bittr_tess_vetter.validation.base import (
    count_transits,
    get_in_transit_mask,
    get_out_of_transit_mask,
)

logger = logging.getLogger(__name__)

# Default Phase 1 checks (V03 excluded by default)
_DEFAULT_ENABLED = {"V01", "V02", "V04", "V05", "V13", "V15"}


def _validate_build_inputs(
    ephemeris: Ephemeris,
    bin_minutes: float,
    max_lc_points: int,
    max_phase_points: int,
    max_transit_windows: int,
    max_points_per_window: int,
    max_timing_points: int = 200,
) -> None:
    """Validate numeric inputs for build_report().

    Raises ValueError for non-finite, non-positive, or otherwise
    invalid parameter values.  Centralised here so that any future
    entry point can reuse the same guards.
    """
    # Ephemeris — must be finite; period + duration must also be positive
    if not math.isfinite(ephemeris.period_days) or ephemeris.period_days <= 0:
        raise ValueError(
            f"period_days must be finite and positive, got {ephemeris.period_days}"
        )
    if not math.isfinite(ephemeris.duration_hours) or ephemeris.duration_hours <= 0:
        raise ValueError(
            f"duration_hours must be finite and positive, got {ephemeris.duration_hours}"
        )
    if not math.isfinite(ephemeris.t0_btjd):
        raise ValueError(f"t0_btjd must be finite, got {ephemeris.t0_btjd}")

    # Bin width
    if not math.isfinite(bin_minutes) or bin_minutes <= 0:
        raise ValueError(
            f"bin_minutes must be finite and positive, got {bin_minutes}"
        )

    # Point budgets
    def _validate_point_budget(name: str, value: Any) -> None:
        # Require integer-like values and explicitly reject booleans.
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")

    _validate_point_budget("max_lc_points", max_lc_points)
    _validate_point_budget("max_phase_points", max_phase_points)
    _validate_point_budget("max_transit_windows", max_transit_windows)
    _validate_point_budget("max_points_per_window", max_points_per_window)
    _validate_point_budget("max_timing_points", max_timing_points)


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
    include_additional_plots: bool = True,
    max_transit_windows: int = 24,
    max_points_per_window: int = 300,
    max_timing_points: int = 200,
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
        include_additional_plots: If True, include per-transit stack,
            odd/even phase, and secondary scan payloads.
        max_transit_windows: Max number of transit windows to include
            in the per-transit stack panel.
        max_points_per_window: Max points per per-transit window.
        max_timing_points: Max per-epoch points for timing series.

    Returns:
        ReportData with all Phase 1 contents populated.
    """
    ephemeris = candidate.ephemeris

    # 0. Validate inputs
    _validate_build_inputs(
        ephemeris,
        bin_minutes,
        max_lc_points,
        max_phase_points,
        max_transit_windows,
        max_points_per_window,
        max_timing_points,
    )

    # 1. Determine enabled checks
    enabled = set(_DEFAULT_ENABLED)
    if include_v03:
        if stellar is None:
            logger.warning(
                "include_v03=True but stellar is None; disabling V03"
            )
        else:
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
        lc, ephemeris, bin_minutes, candidate.depth_ppm, max_phase_points
    )

    per_transit_stack: PerTransitStackPlotData | None = None
    local_detrend: LocalDetrendDiagnosticPlotData | None = None
    oot_context: OOTContextPlotData | None = None
    timing_diag: TransitTimingPlotData | None = None
    alias_diag: AliasHarmonicSummaryData | None = None
    odd_even_phase: OddEvenPhasePlotData | None = None
    secondary_scan: SecondaryScanPlotData | None = None
    if include_additional_plots:
        per_transit_stack = _build_per_transit_stack_plot_data(
            lc,
            ephemeris,
            max_windows=max_transit_windows,
            max_points_per_window=max_points_per_window,
        )
        odd_even_phase = _build_odd_even_phase_plot_data(
            lc,
            ephemeris,
            bin_minutes=bin_minutes,
            max_points=max_phase_points,
        )
        local_detrend = _build_local_detrend_diagnostic_plot_data(
            lc,
            ephemeris,
            max_windows=max_transit_windows,
            max_points_per_window=max_points_per_window,
        )
        oot_context = _build_oot_context_plot_data(
            lc,
            ephemeris,
            max_points=max_phase_points,
        )
        timing_diag = _build_timing_series_plot_data(
            lc,
            candidate,
            max_points=max_timing_points,
        )
        alias_diag = _build_alias_harmonic_summary_data(
            lc,
            candidate,
        )
        secondary_scan = _build_secondary_scan_plot_data(
            lc,
            ephemeris,
            bin_minutes=bin_minutes,
            max_points=max_phase_points,
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
        per_transit_stack=per_transit_stack,
        local_detrend=local_detrend,
        oot_context=oot_context,
        timing_series=timing_diag,
        alias_summary=alias_diag,
        odd_even_phase=odd_even_phase,
        secondary_scan=secondary_scan,
        checks_run=[r.id for r in results],
    )


def _compute_lc_summary(
    lc: LightCurve,
    ephemeris: Ephemeris,
) -> LCSummary:
    """Compute LC vital signs. ~1ms for typical TESS LC."""
    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)
    has_flux_err = lc.flux_err is not None
    flux_err = (
        np.asarray(lc.flux_err, dtype=np.float64)
        if has_flux_err
        else np.full_like(flux, np.nan)
    )

    n_points = len(time)

    # Valid mask: finite time + flux (+ flux_err when provided),
    # intersected with user valid_mask
    valid = np.isfinite(time) & np.isfinite(flux)
    if has_flux_err:
        valid = valid & np.isfinite(flux_err)
    if lc.valid_mask is not None:
        valid = valid & np.asarray(lc.valid_mask, dtype=np.bool_)

    n_valid = int(np.sum(valid))
    gap_fraction = 1.0 - n_valid / n_points if n_points > 0 else 0.0

    time_valid = time[valid]
    flux_valid = flux[valid]

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

    # Build flux_err for detect_transit: use provided errors, or estimate
    # a constant per-point uncertainty from OOT robust MAD scatter.
    if has_flux_err:
        flux_err_valid = flux_err[valid]
    elif flux_mad_ppm > 0.0:
        flux_err_valid = np.full(n_valid, flux_mad_ppm / 1e6)
    else:
        # Fallback: use std-based scatter (less robust but non-zero)
        flux_err_valid = np.full(n_valid, max(flux_std_ppm / 1e6, 1e-10))

    # SNR + depth via detect_transit
    snr = 0.0
    depth_ppm = 0.0
    depth_err_ppm: float | None = None

    try:
        tc = detect_transit(time_valid, flux_valid, flux_err_valid, period, t0, dur_h)
        snr = float(tc.snr)
        depth_ppm = float(tc.depth * 1e6)

        # Compute depth_err separately since TransitCandidate doesn't expose it
        _, d_err = measure_depth(flux_valid, in_mask)
        depth_err_ppm = float(d_err * 1e6)
    except Exception as exc:
        logger.warning("detect_transit failed, using snr=0.0: %s", exc, exc_info=True)

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
    """Downsample LC to at most *max_points*, prioritizing in-transit points.

    Hard cap: output never exceeds *max_points*.  If in-transit points
    alone exceed the budget, they are uniformly thinned to fit.
    """
    in_transit_idx = np.where(transit_mask)[0]
    oot_idx = np.where(~transit_mask)[0]

    if len(time) <= max_points:
        return time.tolist(), flux.tolist(), transit_mask.tolist()

    if len(in_transit_idx) >= max_points:
        # In-transit alone exceeds budget — uniformly thin them
        pick = np.round(np.linspace(0, len(in_transit_idx) - 1, max_points)).astype(int)
        keep = np.sort(in_transit_idx[pick])
        return time[keep].tolist(), flux[keep].tolist(), transit_mask[keep].tolist()

    # Keep all in-transit; evenly sample OOT to fill remaining budget.
    # Use linspace to pick evenly-spaced indices across the full OOT
    # array, avoiding the start-biased truncation of [::step][:budget].
    n_oot_budget = max_points - len(in_transit_idx)
    pick = np.round(np.linspace(0, len(oot_idx) - 1, n_oot_budget)).astype(int)
    sampled_oot = oot_idx[pick]

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
    """Downsample phase-folded data to at most *max_points*.

    Near-transit points (within ±near_transit_half_phase) are prioritized.
    Hard cap: output never exceeds *max_points*.  If near-transit points
    alone exceed the budget, they are uniformly thinned to fit.

    Args:
        phase: Phase values (centered on 0).
        flux: Flux values.
        max_points: Maximum number of output points.
        near_transit_half_phase: Half-width of the near-transit
            preservation window in phase units.  Defaults to 0.1 for
            backward compatibility, but callers should pass a
            duration-based value (e.g. 3 * transit_duration_phase).
    """
    if len(phase) <= max_points:
        return phase, flux

    near_transit = np.abs(phase) < near_transit_half_phase
    near_idx = np.where(near_transit)[0]
    far_idx = np.where(~near_transit)[0]

    if len(near_idx) >= max_points:
        # Near-transit alone exceeds budget — uniformly thin them
        pick = np.round(np.linspace(0, len(near_idx) - 1, max_points)).astype(int)
        keep = np.sort(near_idx[pick])
        return phase[keep], flux[keep]

    # Keep all near-transit; evenly sample far points to fill budget.
    # Use linspace to pick evenly-spaced indices across the full far
    # array.  The old [::step][:budget] approach was biased toward the
    # start of the sorted-by-phase array, which systematically clipped
    # positive-phase (right-side) points and caused visible asymmetry
    # in the phase-folded plot.
    n_far_budget = max_points - len(near_idx)
    if len(far_idx) <= n_far_budget:
        return phase, flux

    pick = np.round(np.linspace(0, len(far_idx) - 1, n_far_budget)).astype(int)
    sampled_far = far_idx[pick]

    keep = np.sort(np.concatenate([near_idx, sampled_far]))
    return phase[keep], flux[keep]


def _build_phase_folded_plot_data(
    lc: LightCurve,
    ephemeris: Ephemeris,
    bin_minutes: float,
    depth_ppm: float | None = None,
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

    y_range_suggested = _suggest_flux_y_range(flux_folded)
    depth_reference_flux = _depth_ppm_to_flux(depth_ppm)

    return PhaseFoldedPlotData(
        phase=phase.tolist(),
        flux=flux_folded.tolist(),
        bin_centers=bin_centers,
        bin_flux=bin_flux,
        bin_err=bin_err,
        bin_minutes=bin_minutes,
        transit_duration_phase=transit_duration_phase,
        phase_range=phase_range,
        y_range_suggested=y_range_suggested,
        depth_reference_flux=depth_reference_flux,
    )


def _suggest_flux_y_range(
    flux: np.ndarray,
    *,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    padding_fraction: float = 0.1,
) -> tuple[float, float] | None:
    """Suggest a robust y-axis range for phase-folded display.

    Uses percentile clipping plus symmetric padding to keep shallow transits
    visible in the presence of outliers. This is display metadata only.
    """
    finite = flux[np.isfinite(flux)]
    if len(finite) < 3:
        return None

    lo, hi = np.percentile(finite, [lower_percentile, upper_percentile])
    lo_f = float(lo)
    hi_f = float(hi)
    if not np.isfinite(lo_f) or not np.isfinite(hi_f):
        return None

    span = hi_f - lo_f
    if span <= 0.0:
        # Degenerate case: nearly constant flux; add tiny guard band.
        pad = max(abs(lo_f), 1.0) * 1e-6
        return (lo_f - pad, hi_f + pad)

    pad = span * max(padding_fraction, 0.0)
    return (lo_f - pad, hi_f + pad)


def _depth_ppm_to_flux(depth_ppm: float | None) -> float | None:
    """Convert positive depth in ppm to normalized flux reference level."""
    if depth_ppm is None or not np.isfinite(depth_ppm) or depth_ppm <= 0.0:
        return None
    return float(1.0 - (depth_ppm / 1e6))


def _get_valid_time_flux(lc: LightCurve) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, mask-filtered time/flux arrays."""
    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)
    valid = np.isfinite(time) & np.isfinite(flux)
    if lc.valid_mask is not None:
        valid = valid & np.asarray(lc.valid_mask, dtype=np.bool_)
    return time[valid], flux[valid]


def _thin_evenly(arr: np.ndarray, max_points: int) -> np.ndarray:
    """Evenly thin a 1D array to at most max_points elements."""
    if len(arr) <= max_points:
        return arr
    pick = np.round(np.linspace(0, len(arr) - 1, max_points)).astype(int)
    return arr[pick]


def _build_per_transit_stack_plot_data(
    lc: LightCurve,
    ephemeris: Ephemeris,
    *,
    max_windows: int,
    max_points_per_window: int,
) -> PerTransitStackPlotData:
    """Build per-transit window payload for small-multiples panel."""
    time, flux = _get_valid_time_flux(lc)
    if len(time) == 0:
        return PerTransitStackPlotData(windows=[], window_half_hours=0.0, max_windows=max_windows)

    period = ephemeris.period_days
    t0 = ephemeris.t0_btjd
    duration_hours = ephemeris.duration_hours
    half_window_days = 3.0 * duration_hours / 24.0
    half_window_hours = 3.0 * duration_hours

    n_start = int(np.ceil((np.min(time) - t0) / period))
    n_end = int(np.floor((np.max(time) - t0) / period))

    epochs = np.arange(n_start, n_end + 1, dtype=int)
    if len(epochs) == 0:
        return PerTransitStackPlotData(windows=[], window_half_hours=half_window_hours, max_windows=max_windows)

    if len(epochs) > max_windows:
        epochs = _thin_evenly(epochs, max_windows)

    windows: list[TransitWindowData] = []
    for n in epochs:
        t_mid = t0 + float(n) * period
        in_window = np.abs(time - t_mid) <= half_window_days
        if not np.any(in_window):
            continue
        t_w = time[in_window]
        f_w = flux[in_window]

        if len(t_w) > max_points_per_window:
            pick = np.round(np.linspace(0, len(t_w) - 1, max_points_per_window)).astype(int)
            t_w = t_w[pick]
            f_w = f_w[pick]

        dt_hours = (t_w - t_mid) * 24.0
        in_mask = np.abs(dt_hours) <= (duration_hours / 2.0)

        windows.append(
            TransitWindowData(
                epoch=int(n),
                t_mid_btjd=float(t_mid),
                dt_hours=dt_hours.tolist(),
                flux=f_w.tolist(),
                in_transit_mask=in_mask.tolist(),
            )
        )

    return PerTransitStackPlotData(
        windows=windows,
        window_half_hours=float(half_window_hours),
        max_windows=int(max_windows),
    )


def _build_odd_even_phase_plot_data(
    lc: LightCurve,
    ephemeris: Ephemeris,
    *,
    bin_minutes: float,
    max_points: int,
) -> OddEvenPhasePlotData:
    """Build odd/even phase-fold comparison payload."""
    time, flux = _get_valid_time_flux(lc)
    if len(time) == 0:
        return OddEvenPhasePlotData(
            phase_range=(-0.1, 0.1),
            odd_phase=[],
            odd_flux=[],
            even_phase=[],
            even_flux=[],
            odd_bin_centers=[],
            even_bin_centers=[],
            odd_bin_flux=[],
            even_bin_flux=[],
            bin_minutes=bin_minutes,
        )

    period = ephemeris.period_days
    t0 = ephemeris.t0_btjd
    duration_phase = (ephemeris.duration_hours / 24.0) / period
    half_window = min(max(3.0 * duration_phase, 0.015), 0.5)
    phase_range = (-half_window, half_window)

    phase, flux_folded = fold_transit(time, flux, period, t0)
    # Transit index/parity for each point.
    epoch = np.floor((time - t0 + period / 2.0) / period).astype(int)
    sort_idx = np.argsort(((time - t0) / period + 0.5) % 1.0 - 0.5)
    parity_sorted = epoch[sort_idx] % 2

    odd_mask = parity_sorted == 1
    even_mask = ~odd_mask

    odd_phase = phase[odd_mask]
    odd_flux = flux_folded[odd_mask]
    even_phase = phase[even_mask]
    even_flux = flux_folded[even_mask]

    if len(odd_phase) > max_points:
        odd_phase, odd_flux = _downsample_phase_preserving_transit(
            odd_phase, odd_flux, max_points, near_transit_half_phase=half_window
        )
    if len(even_phase) > max_points:
        even_phase, even_flux = _downsample_phase_preserving_transit(
            even_phase, even_flux, max_points, near_transit_half_phase=half_window
        )

    odd_centers, odd_bin_flux, _ = _bin_phase_data(
        odd_phase, odd_flux, period, bin_minutes, phase_range=phase_range
    )
    even_centers, even_bin_flux, _ = _bin_phase_data(
        even_phase, even_flux, period, bin_minutes, phase_range=phase_range
    )

    return OddEvenPhasePlotData(
        phase_range=phase_range,
        odd_phase=odd_phase.tolist(),
        odd_flux=odd_flux.tolist(),
        even_phase=even_phase.tolist(),
        even_flux=even_flux.tolist(),
        odd_bin_centers=odd_centers,
        even_bin_centers=even_centers,
        odd_bin_flux=odd_bin_flux,
        even_bin_flux=even_bin_flux,
        bin_minutes=bin_minutes,
    )


def _build_local_detrend_diagnostic_plot_data(
    lc: LightCurve,
    ephemeris: Ephemeris,
    *,
    max_windows: int,
    max_points_per_window: int,
) -> LocalDetrendDiagnosticPlotData:
    """Build local baseline diagnostic windows around each observed transit."""
    time, flux = _get_valid_time_flux(lc)
    if len(time) == 0:
        return LocalDetrendDiagnosticPlotData(
            windows=[],
            window_half_hours=0.0,
            max_windows=max_windows,
            baseline_method="linear_oot_fit",
        )

    period = ephemeris.period_days
    t0 = ephemeris.t0_btjd
    duration_hours = ephemeris.duration_hours
    half_window_days = 3.0 * duration_hours / 24.0
    half_window_hours = 3.0 * duration_hours
    in_half_days = (duration_hours / 24.0) / 2.0

    n_start = int(np.ceil((np.min(time) - t0) / period))
    n_end = int(np.floor((np.max(time) - t0) / period))
    epochs = np.arange(n_start, n_end + 1, dtype=int)
    if len(epochs) == 0:
        return LocalDetrendDiagnosticPlotData(
            windows=[],
            window_half_hours=float(half_window_hours),
            max_windows=max_windows,
            baseline_method="linear_oot_fit",
        )
    if len(epochs) > max_windows:
        epochs = _thin_evenly(epochs, max_windows)

    windows: list[LocalDetrendWindowData] = []
    for n in epochs:
        t_mid = t0 + float(n) * period
        in_window = np.abs(time - t_mid) <= half_window_days
        if not np.any(in_window):
            continue

        t_w = time[in_window]
        f_w = flux[in_window]
        if len(t_w) > max_points_per_window:
            pick = np.round(np.linspace(0, len(t_w) - 1, max_points_per_window)).astype(int)
            t_w = t_w[pick]
            f_w = f_w[pick]

        dt_days = t_w - t_mid
        dt_hours = dt_days * 24.0
        in_transit = np.abs(dt_days) <= in_half_days
        oot = ~in_transit

        baseline = np.full_like(f_w, np.nan, dtype=np.float64)
        if int(np.sum(oot)) >= 2:
            x = dt_days[oot]
            y = f_w[oot]
            try:
                coeff = np.polyfit(x, y, deg=1)
                baseline = np.polyval(coeff, dt_days)
            except Exception:
                baseline[:] = np.nanmedian(y)
        elif len(f_w) > 0:
            baseline[:] = np.nanmedian(f_w)

        windows.append(
            LocalDetrendWindowData(
                epoch=int(n),
                t_mid_btjd=float(t_mid),
                dt_hours=dt_hours.tolist(),
                flux=f_w.tolist(),
                baseline_flux=baseline.tolist(),
                in_transit_mask=in_transit.tolist(),
            )
        )

    return LocalDetrendDiagnosticPlotData(
        windows=windows,
        window_half_hours=float(half_window_hours),
        max_windows=int(max_windows),
        baseline_method="linear_oot_fit",
    )


def _build_oot_context_plot_data(
    lc: LightCurve,
    ephemeris: Ephemeris,
    *,
    max_points: int,
) -> OOTContextPlotData:
    """Build out-of-transit flux distribution and scatter summary payload."""
    time, flux = _get_valid_time_flux(lc)
    if len(time) == 0:
        return OOTContextPlotData(
            flux_sample=[],
            flux_residual_ppm_sample=[],
            sample_indices=[],
            hist_centers=[],
            hist_counts=[],
            median_flux=None,
            std_ppm=None,
            mad_ppm=None,
            robust_sigma_ppm=None,
            n_oot_points=0,
        )

    oot_mask = get_out_of_transit_mask(
        time,
        ephemeris.period_days,
        ephemeris.t0_btjd,
        ephemeris.duration_hours,
        buffer_factor=2.0,
    )
    flux_oot = flux[oot_mask]
    n_oot = int(len(flux_oot))
    if n_oot == 0:
        return OOTContextPlotData(
            flux_sample=[],
            flux_residual_ppm_sample=[],
            sample_indices=[],
            hist_centers=[],
            hist_counts=[],
            median_flux=None,
            std_ppm=None,
            mad_ppm=None,
            robust_sigma_ppm=None,
            n_oot_points=0,
        )

    idx = np.arange(n_oot, dtype=int)
    if n_oot > max_points:
        pick = np.round(np.linspace(0, n_oot - 1, max_points)).astype(int)
        flux_sample = flux_oot[pick]
        sample_idx = idx[pick]
    else:
        flux_sample = flux_oot
        sample_idx = idx

    median = float(np.nanmedian(flux_oot))
    mad = float(np.nanmedian(np.abs(flux_oot - median)))
    robust_sigma = float(1.4826 * mad)
    std = float(np.nanstd(flux_oot, ddof=1)) if n_oot > 1 else 0.0

    # Histogram in flux residual ppm for stable scale across targets.
    residual_ppm = (flux_oot - median) * 1e6
    counts, edges = np.histogram(residual_ppm, bins=40)
    centers = (edges[:-1] + edges[1:]) / 2.0
    residual_sample_ppm = (flux_sample - median) * 1e6

    return OOTContextPlotData(
        flux_sample=flux_sample.tolist(),
        flux_residual_ppm_sample=residual_sample_ppm.tolist(),
        sample_indices=sample_idx.tolist(),
        hist_centers=centers.tolist(),
        hist_counts=counts.astype(int).tolist(),
        median_flux=median,
        std_ppm=std * 1e6,
        mad_ppm=mad * 1e6,
        robust_sigma_ppm=robust_sigma * 1e6,
        n_oot_points=n_oot,
    )


def _build_timing_series_plot_data(
    lc: LightCurve,
    candidate: Candidate,
    *,
    max_points: int,
) -> TransitTimingPlotData:
    """Build timing diagnostics payload from API-level timing series."""
    series = timing_series(lc, candidate, min_snr=2.0)
    points = series.points
    if len(points) > max_points:
        pick = np.round(np.linspace(0, len(points) - 1, max_points)).astype(int)
        points = [points[int(i)] for i in pick]

    return TransitTimingPlotData(
        epochs=[int(p.epoch) for p in points],
        oc_seconds=[float(p.oc_seconds) for p in points],
        snr=[float(p.snr) for p in points],
        rms_seconds=series.rms_seconds,
        periodicity_sigma=series.periodicity_sigma,
        linear_trend_sec_per_epoch=series.linear_trend_sec_per_epoch,
    )


def _build_alias_harmonic_summary_data(
    lc: LightCurve,
    candidate: Candidate,
) -> AliasHarmonicSummaryData:
    """Build compact harmonic summary payload from API diagnostics."""
    summary = harmonic_power_summary(lc, candidate)
    harmonics = summary.harmonics
    return AliasHarmonicSummaryData(
        harmonic_labels=[str(h.harmonic) for h in harmonics],
        periods=[float(h.period) for h in harmonics],
        scores=[float(h.score) for h in harmonics],
        best_harmonic=str(summary.best_harmonic),
        best_ratio_over_p=float(summary.best_ratio_over_p),
    )


def _build_secondary_scan_plot_data(
    lc: LightCurve,
    ephemeris: Ephemeris,
    *,
    bin_minutes: float,
    max_points: int,
) -> SecondaryScanPlotData:
    """Build full-orbit phase scan payload for secondary-eclipse triage."""
    time, flux = _get_valid_time_flux(lc)
    if len(time) == 0:
        return SecondaryScanPlotData(
            phase=[],
            flux=[],
            bin_centers=[],
            bin_flux=[],
            bin_err=[],
            bin_minutes=bin_minutes,
            primary_phase=0.0,
            secondary_phase=0.5,
            strongest_dip_phase=None,
            strongest_dip_flux=None,
            quality=SecondaryScanQuality(
                n_raw_points=0,
                n_bins=0,
                n_bins_with_error=0,
                phase_coverage_fraction=0.0,
                largest_phase_gap=1.0,
                is_degraded=True,
                flags=["NO_POINTS"],
            ),
            render_hints=SecondaryScanRenderHints(
                style_mode="degraded",
                connect_bins=False,
                max_connect_phase_gap=0.02,
                show_error_bars=False,
                error_bar_stride=1,
                raw_marker_opacity=0.18,
                binned_marker_size=5.0,
                binned_line_width=0.0,
            ),
        )

    period = ephemeris.period_days
    phase, flux_folded = fold_transit(time, flux, period, ephemeris.t0_btjd)

    if len(phase) > max_points:
        pick = np.round(np.linspace(0, len(phase) - 1, max_points)).astype(int)
        phase = phase[pick]
        flux_folded = flux_folded[pick]

    centers, bflux, berr = _bin_phase_data(
        phase,
        flux_folded,
        period,
        bin_minutes,
        phase_range=(-0.5, 0.5),
    )

    strongest_dip_phase: float | None = None
    strongest_dip_flux: float | None = None
    if centers and bflux:
        dur_phase = (ephemeris.duration_hours / 24.0) / period
        # Ignore immediate primary window when searching strongest non-zero-phase dip.
        mask = np.array([abs(c) > max(dur_phase, 0.01) for c in centers], dtype=bool)
        if np.any(mask):
            bflux_arr = np.asarray(bflux, dtype=np.float64)[mask]
            centers_arr = np.asarray(centers, dtype=np.float64)[mask]
            idx = int(np.argmin(bflux_arr))
            strongest_dip_flux = float(bflux_arr[idx])
            strongest_dip_phase = float(centers_arr[idx])

    quality = _compute_secondary_scan_quality(
        phase=phase,
        bin_err=berr,
        period_days=period,
        bin_minutes=bin_minutes,
    )
    render_hints = _secondary_scan_render_hints(quality, period_days=period, bin_minutes=bin_minutes)

    return SecondaryScanPlotData(
        phase=phase.tolist(),
        flux=flux_folded.tolist(),
        bin_centers=centers,
        bin_flux=bflux,
        bin_err=berr,
        bin_minutes=bin_minutes,
        primary_phase=0.0,
        secondary_phase=0.5,
        strongest_dip_phase=strongest_dip_phase,
        strongest_dip_flux=strongest_dip_flux,
        quality=quality,
        render_hints=render_hints,
    )


def _compute_secondary_scan_quality(
    *,
    phase: np.ndarray,
    bin_err: list[float | None],
    period_days: float,
    bin_minutes: float,
) -> SecondaryScanQuality:
    """Compute robust quality metrics for full-orbit secondary scan."""
    n_raw_points = int(len(phase))
    n_bins = int(len(bin_err))
    n_bins_with_error = int(sum(e is not None for e in bin_err))

    if n_raw_points == 0:
        return SecondaryScanQuality(
            n_raw_points=0,
            n_bins=n_bins,
            n_bins_with_error=n_bins_with_error,
            phase_coverage_fraction=0.0,
            largest_phase_gap=1.0,
            is_degraded=True,
            flags=["NO_POINTS"],
        )

    finite_phase = np.sort(phase[np.isfinite(phase)])
    if len(finite_phase) == 0:
        return SecondaryScanQuality(
            n_raw_points=n_raw_points,
            n_bins=n_bins,
            n_bins_with_error=n_bins_with_error,
            phase_coverage_fraction=0.0,
            largest_phase_gap=1.0,
            is_degraded=True,
            flags=["NON_FINITE_PHASE"],
        )

    # Phase-gap metric (include domain boundaries).
    extended = np.concatenate([np.array([-0.5]), finite_phase, np.array([0.5])])
    largest_phase_gap = float(np.max(np.diff(extended)))

    # Occupied-bin fraction over full orbit.
    bin_phase = (bin_minutes / 60.0 / 24.0) / period_days
    n_bins_total = max(1, int(np.ceil(1.0 / bin_phase)))
    bin_edges = np.linspace(-0.5, 0.5, n_bins_total + 1)
    idx = np.digitize(finite_phase, bin_edges) - 1
    idx = np.clip(idx, 0, n_bins_total - 1)
    occupied = int(np.sum(np.bincount(idx, minlength=n_bins_total) > 0))
    phase_coverage_fraction = float(occupied / n_bins_total)

    flags: list[str] = []
    if n_bins < 30:
        flags.append("LOW_BIN_COUNT")
    if phase_coverage_fraction < 0.6:
        flags.append("LOW_PHASE_COVERAGE")
    if largest_phase_gap > 0.12:
        flags.append("LARGE_PHASE_GAP")

    is_degraded = len(flags) > 0
    return SecondaryScanQuality(
        n_raw_points=n_raw_points,
        n_bins=n_bins,
        n_bins_with_error=n_bins_with_error,
        phase_coverage_fraction=phase_coverage_fraction,
        largest_phase_gap=largest_phase_gap,
        is_degraded=is_degraded,
        flags=flags,
    )


def _secondary_scan_render_hints(
    quality: SecondaryScanQuality,
    *,
    period_days: float,
    bin_minutes: float,
) -> SecondaryScanRenderHints:
    """Derive deterministic rendering hints from quality metrics."""
    bin_phase = (bin_minutes / 60.0 / 24.0) / period_days
    max_connect_phase_gap = float(max(3.0 * bin_phase, 0.01))

    show_error_bars = quality.n_bins_with_error > 0 and quality.n_bins <= 500
    error_bar_stride = max(1, int(np.ceil(max(1, quality.n_bins) / 180)))

    if quality.is_degraded:
        return SecondaryScanRenderHints(
            style_mode="degraded",
            connect_bins=False,
            max_connect_phase_gap=max_connect_phase_gap,
            show_error_bars=False,
            error_bar_stride=error_bar_stride,
            raw_marker_opacity=0.18,
            binned_marker_size=5.0,
            binned_line_width=0.0,
        )

    return SecondaryScanRenderHints(
        style_mode="normal",
        connect_bins=True,
        max_connect_phase_gap=max_connect_phase_gap,
        show_error_bars=show_error_bars,
        error_bar_stride=error_bar_stride,
        raw_marker_opacity=0.24,
        binned_marker_size=3.0,
        binned_line_width=1.1,
    )


def _bin_phase_data(
    phase: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    bin_minutes: float,
    phase_range: tuple[float, float] | None = None,
) -> tuple[list[float], list[float], list[float | None]]:
    """Bin phase-folded data by phase.

    Uses vectorized np.digitize + np.bincount for O(n_points + n_bins)
    performance instead of per-bin masking.

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

    # Vectorized binning via digitize (returns 1-based bin indices)
    idx = np.digitize(phase, bin_edges) - 1
    # Clamp right-edge points into last bin (digitize puts them at n_bins)
    idx = np.clip(idx, 0, n_bins - 1)

    counts = np.bincount(idx, minlength=n_bins)[:n_bins]
    sums = np.bincount(idx, weights=flux, minlength=n_bins)[:n_bins]
    sumsq = np.bincount(idx, weights=flux * flux, minlength=n_bins)[:n_bins]

    # Only emit bins that have data
    occupied = counts > 0
    occ_idx = np.where(occupied)[0]

    bin_centers_arr = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    centers: list[float] = bin_centers_arr[occ_idx].tolist()
    mean = sums[occupied] / counts[occupied]
    fluxes: list[float] = mean.tolist()

    # SEM: sqrt(sample_variance / n) where sample_variance uses ddof=1
    errors: list[float | None] = []
    occ_counts = counts[occupied]
    occ_sumsq = sumsq[occupied]
    occ_sums = sums[occupied]
    for i in range(len(occ_counts)):
        n = int(occ_counts[i])
        if n > 1:
            var = (occ_sumsq[i] - occ_sums[i] ** 2 / n) / (n - 1)
            errors.append(float(np.sqrt(max(var, 0.0) / n)))
        else:
            errors.append(None)

    return centers, fluxes, errors
