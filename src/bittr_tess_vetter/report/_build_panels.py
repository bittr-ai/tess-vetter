"""Panel and diagnostic payload builders for report assembly."""

from __future__ import annotations

from typing import Any

import numpy as np

from bittr_tess_vetter.compute.transit import fold_transit
from bittr_tess_vetter.report._data import (
    AliasHarmonicSummaryData,
    FullLCPlotData,
    LCFPSignals,
    LCRobustnessData,
    LCRobustnessEpochMetrics,
    LCRobustnessMetrics,
    LCRobustnessRedNoiseMetrics,
    LocalDetrendDiagnosticPlotData,
    LocalDetrendWindowData,
    OddEvenPhasePlotData,
    OOTContextPlotData,
    PerTransitStackPlotData,
    PhaseFoldedPlotData,
    SecondaryScanPlotData,
    SecondaryScanQuality,
    SecondaryScanRenderHints,
    TransitTimingPlotData,
    TransitWindowData,
)
from bittr_tess_vetter.validation.base import (
    get_in_transit_mask,
    get_out_of_transit_mask,
    measure_transit_depth,
)
from bittr_tess_vetter.validation.report_bridge import (
    compute_alias_diagnostics,
    compute_timing_series,
)

from ._build_utils import (
    _bin_phase_data,
    _depth_ppm_to_flux,
    _downsample_phase_preserving_transit,
    _downsample_preserving_transits,
    _get_valid_time_flux,
    _get_valid_time_flux_quality,
    _red_noise_beta,
    _suggest_flux_y_range,
    _thin_evenly,
    _to_internal_lightcurve,
)


def _build_full_lc_plot_data(
    lc: Any,
    ephemeris: Any,
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
    lc: Any,
    ephemeris: Any,
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


def _build_per_transit_stack_plot_data(
    lc: Any,
    ephemeris: Any,
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
    lc: Any,
    ephemeris: Any,
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
    lc: Any,
    ephemeris: Any,
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
    lc: Any,
    ephemeris: Any,
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
    lc: Any,
    candidate: Any,
    *,
    max_points: int,
) -> tuple[TransitTimingPlotData, TransitTimingPlotData]:
    """Build timing diagnostics payload from bridge timing series."""
    eph = candidate.ephemeris
    series = compute_timing_series(
        _to_internal_lightcurve(lc),
        period_days=eph.period_days,
        t0_btjd=eph.t0_btjd,
        duration_hours=eph.duration_hours,
        min_snr=2.0,
    )
    points_full = list(series.points)
    summary_series = _to_timing_plot_data(
        points=points_full,
        rms_seconds=series.rms_seconds,
        periodicity_score=series.periodicity_score,
        linear_trend_sec_per_epoch=series.linear_trend_sec_per_epoch,
    )

    points_plot = points_full
    if len(points_plot) > max_points:
        outliers = [p for p in points_plot if p.is_outlier]
        non_outliers = [p for p in points_plot if not p.is_outlier]

        if len(outliers) >= max_points:
            # Keep the strongest anomalies when outliers exceed display budget.
            outliers_sorted = sorted(outliers, key=lambda p: abs(p.oc_seconds), reverse=True)
            points_plot = sorted(outliers_sorted[:max_points], key=lambda p: p.epoch)
        else:
            n_keep_non_outliers = max_points - len(outliers)
            if len(non_outliers) > n_keep_non_outliers and n_keep_non_outliers > 0:
                pick = np.round(
                    np.linspace(0, len(non_outliers) - 1, n_keep_non_outliers)
                ).astype(int)
                non_outliers = [non_outliers[int(i)] for i in pick]
            points_plot = sorted([*outliers, *non_outliers], key=lambda p: p.epoch)

    return (
        _to_timing_plot_data(
            points=points_plot,
            rms_seconds=series.rms_seconds,
            periodicity_score=series.periodicity_score,
            linear_trend_sec_per_epoch=series.linear_trend_sec_per_epoch,
        ),
        summary_series,
    )


def _to_timing_plot_data(
    *,
    points: list[Any],
    rms_seconds: float | None,
    periodicity_score: float | None,
    linear_trend_sec_per_epoch: float | None,
) -> TransitTimingPlotData:
    return TransitTimingPlotData(
        epochs=[int(p.epoch) for p in points],
        oc_seconds=[float(p.oc_seconds) for p in points],
        snr=[float(p.snr) for p in points],
        rms_seconds=rms_seconds,
        periodicity_score=periodicity_score,
        linear_trend_sec_per_epoch=linear_trend_sec_per_epoch,
    )


def _build_alias_harmonic_summary_data(
    lc: Any,
    candidate: Any,
) -> AliasHarmonicSummaryData:
    """Build compact harmonic summary payload from bridge diagnostics."""
    eph = candidate.ephemeris
    internal_lc = _to_internal_lightcurve(lc)
    alias_diagnostics = compute_alias_diagnostics(
        internal_lc,
        period_days=eph.period_days,
        t0_btjd=eph.t0_btjd,
        duration_hours=eph.duration_hours,
    )
    return AliasHarmonicSummaryData(
        harmonic_labels=[str(label) for label in alias_diagnostics.harmonic_labels],
        periods=[float(period) for period in alias_diagnostics.periods],
        scores=[float(score) for score in alias_diagnostics.scores],
        harmonic_depth_ppm=[
            float(depth_ppm)
            for depth_ppm in alias_diagnostics.harmonic_depth_ppm
        ],
        best_harmonic=str(alias_diagnostics.best_harmonic),
        best_ratio_over_p=float(alias_diagnostics.best_ratio_over_p),
        classification=str(alias_diagnostics.classification)
        if alias_diagnostics.classification is not None
        else None,
        phase_shift_event_count=(
            int(alias_diagnostics.phase_shift_event_count)
            if alias_diagnostics.phase_shift_event_count is not None
            else None
        ),
        phase_shift_peak_sigma=(
            float(alias_diagnostics.phase_shift_peak_sigma)
            if alias_diagnostics.phase_shift_peak_sigma is not None
            else None
        ),
        secondary_significance=(
            float(alias_diagnostics.secondary_significance)
            if alias_diagnostics.secondary_significance is not None
            else None
        ),
    )


def _build_lc_robustness_data(
    lc: Any,
    candidate: Any,
    *,
    checks: dict[str, Any],
    max_epochs: int,
    baseline_window_mult: float = 6.0,
) -> LCRobustnessData:
    """Build LC robustness payload (LC-only, deterministic)."""
    time, flux, quality = _get_valid_time_flux_quality(lc)
    eph = candidate.ephemeris
    if len(time) == 0:
        return LCRobustnessData(
            version="1.0",
            baseline_window_mult=float(baseline_window_mult),
            per_epoch=[],
            robustness=LCRobustnessMetrics(
                n_epochs_measured=0,
                loto_snr_min=None,
                loto_snr_max=None,
                loto_snr_mean=None,
                loto_depth_ppm_min=None,
                loto_depth_ppm_max=None,
                loto_depth_shift_ppm_max=None,
                dominance_index=None,
            ),
            red_noise=LCRobustnessRedNoiseMetrics(beta_30m=None, beta_60m=None, beta_duration=None),
            fp_signals=LCFPSignals(
                odd_even_depth_diff_sigma=None,
                secondary_depth_sigma=None,
                phase_0p5_bin_depth_ppm=None,
                v_shape_metric=None,
                asymmetry_sigma=None,
            ),
        )

    period = eph.period_days
    t0 = eph.t0_btjd
    duration_days = eph.duration_hours / 24.0
    half_window_days = baseline_window_mult * duration_days
    in_half_days = duration_days / 2.0

    if len(time) > 1:
        cadence_days = float(np.median(np.diff(np.sort(time))))
        if cadence_days <= 0 or not np.isfinite(cadence_days):
            cadence_days = None
    else:
        cadence_days = None

    epoch_idx = np.floor((time - t0 + period / 2.0) / period).astype(int)
    unique_epochs = np.unique(epoch_idx)

    per_epoch: list[LCRobustnessEpochMetrics] = []
    for ep in unique_epochs:
        t_mid = t0 + float(ep) * period
        local_window = np.abs(time - t_mid) <= half_window_days
        if not np.any(local_window):
            continue

        t_local = time[local_window]
        f_local = flux[local_window]
        q_local = quality[local_window] if quality is not None else None
        dt_local = t_local - t_mid
        in_transit = np.abs(dt_local) <= in_half_days
        n_in = int(np.sum(in_transit))
        n_total = int(len(t_local))

        local_oot = ~in_transit
        n_oot_local = int(np.sum(local_oot))

        # Baseline arrays default to local window; optionally fall back to global OOT.
        t_base = t_local
        f_base = f_local
        dt_base = dt_local
        in_base = in_transit
        oot_base = local_oot
        n_oot_base = n_oot_local
        q_base = q_local
        if n_oot_base < 5:
            global_oot = ~get_in_transit_mask(time, period, t0, eph.duration_hours)
            t_base = time[global_oot]
            f_base = flux[global_oot]
            q_base = quality[global_oot] if quality is not None else None
            dt_base = t_base - t_mid
            in_base = np.abs(dt_base) <= in_half_days
            oot_base = ~in_base
            n_oot_base = int(np.sum(oot_base))

        t_mid_measured = float(np.median(t_local[in_transit])) if n_in >= 3 else None

        depth_ppm: float | None = None
        depth_err_ppm: float | None = None
        if n_in >= 3 and n_oot_base >= 5:
            d, d_err = measure_transit_depth(f_base, in_base, oot_base)
            if np.isfinite(d) and np.isfinite(d_err):
                depth_ppm = float(d * 1e6)
                depth_err_ppm = float(d_err * 1e6)

        baseline_level: float | None = None
        baseline_slope: float | None = None
        oot_scatter_ppm: float | None = None
        oot_mad_ppm: float | None = None
        in_outliers = 0
        oot_outliers = 0
        if n_oot_base >= 3:
            x = dt_base[oot_base]
            y = f_base[oot_base]
            try:
                coeff = np.polyfit(x, y, deg=1)
                baseline_slope = float(coeff[0])
                baseline_level = float(coeff[1])
                model_base = np.polyval(coeff, dt_base)
                model_local = np.polyval(coeff, dt_local)
            except Exception:
                baseline_level = float(np.median(y))
                baseline_slope = 0.0
                model_base = np.full_like(dt_base, baseline_level, dtype=np.float64)
                model_local = np.full_like(dt_local, baseline_level, dtype=np.float64)

            oot_resid = f_base[oot_base] - model_base[oot_base]
            oot_scatter_ppm = float(np.std(oot_resid, ddof=1) * 1e6) if len(oot_resid) > 1 else 0.0
            mad = float(np.median(np.abs(oot_resid - np.median(oot_resid)))) if len(oot_resid) > 0 else 0.0
            robust_sigma = max(1.4826 * mad, 1e-10)
            oot_mad_ppm = float(1.4826 * mad * 1e6)
            threshold = 4.0 * robust_sigma
            in_outliers = int(np.sum(np.abs(f_local[in_transit] - model_local[in_transit]) > threshold)) if n_in > 0 else 0
            oot_outliers = int(np.sum(np.abs(oot_resid) > threshold))

        quality_in_nonzero: int | None = None
        quality_oot_nonzero: int | None = None
        if q_base is not None:
            quality_in_nonzero = int(np.sum(q_local[in_transit] != 0)) if (q_local is not None and n_in > 0) else 0
            quality_oot_nonzero = int(np.sum(q_base[oot_base] != 0)) if n_oot_base > 0 else 0

        if cadence_days is not None and cadence_days > 0:
            expected_points = max((2.0 * half_window_days) / cadence_days, 1.0)
            coverage = float(min(n_total / expected_points, 1.0))
        else:
            coverage = 1.0

        per_epoch.append(
            LCRobustnessEpochMetrics(
                epoch_index=int(ep),
                t_mid_expected_btjd=float(t_mid),
                t_mid_measured_btjd=t_mid_measured,
                time_coverage_fraction=coverage,
                n_points_total=n_total,
                n_in_transit=n_in,
                n_oot=n_oot_base,
                depth_ppm=depth_ppm,
                depth_err_ppm=depth_err_ppm,
                baseline_level=baseline_level,
                baseline_slope_per_day=baseline_slope,
                oot_scatter_ppm=oot_scatter_ppm,
                oot_mad_ppm=oot_mad_ppm,
                in_transit_outlier_count=in_outliers,
                oot_outlier_count=oot_outliers,
                quality_in_transit_nonzero=quality_in_nonzero,
                quality_oot_nonzero=quality_oot_nonzero,
            )
        )

    per_epoch = sorted(per_epoch, key=lambda x: x.epoch_index)
    if len(per_epoch) > max_epochs:
        pick = np.round(np.linspace(0, len(per_epoch) - 1, max_epochs)).astype(int)
        per_epoch = [per_epoch[int(i)] for i in pick]

    robustness = _build_lc_robustness_metrics(per_epoch)
    red_noise = _build_lc_robustness_red_noise(time, flux, eph, cadence_days=cadence_days)
    fp_signals = _build_lc_robustness_fp_signals(checks, time, flux, eph)

    return LCRobustnessData(
        version="1.0",
        baseline_window_mult=float(baseline_window_mult),
        per_epoch=per_epoch,
        robustness=robustness,
        red_noise=red_noise,
        fp_signals=fp_signals,
    )


def _build_lc_robustness_metrics(
    per_epoch: list[LCRobustnessEpochMetrics],
) -> LCRobustnessMetrics:
    """Build leave-one-transit-out robustness summary."""
    depths = np.array(
        [m.depth_ppm for m in per_epoch if m.depth_ppm is not None and m.depth_err_ppm and m.depth_err_ppm > 0],
        dtype=np.float64,
    )
    errs = np.array(
        [m.depth_err_ppm for m in per_epoch if m.depth_ppm is not None and m.depth_err_ppm and m.depth_err_ppm > 0],
        dtype=np.float64,
    )
    n = int(len(depths))
    if n < 3:
        return LCRobustnessMetrics(
            n_epochs_measured=n,
            loto_snr_min=None,
            loto_snr_max=None,
            loto_snr_mean=None,
            loto_depth_ppm_min=None,
            loto_depth_ppm_max=None,
            loto_depth_shift_ppm_max=None,
            dominance_index=None,
        )

    w = 1.0 / np.maximum(errs**2, 1e-12)
    full_depth = float(np.sum(depths * w) / np.sum(w))
    full_err = float(np.sqrt(1.0 / np.sum(w)))
    full_snr = abs(full_depth) / max(full_err, 1e-12)

    loto_depths: list[float] = []
    loto_snrs: list[float] = []
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        if int(np.sum(keep)) < 2:
            continue
        w_k = w[keep]
        d_k = depths[keep]
        d = float(np.sum(d_k * w_k) / np.sum(w_k))
        e = float(np.sqrt(1.0 / np.sum(w_k)))
        loto_depths.append(d)
        loto_snrs.append(abs(d) / max(e, 1e-12))

    if len(loto_depths) == 0:
        return LCRobustnessMetrics(
            n_epochs_measured=n,
            loto_snr_min=None,
            loto_snr_max=None,
            loto_snr_mean=None,
            loto_depth_ppm_min=None,
            loto_depth_ppm_max=None,
            loto_depth_shift_ppm_max=None,
            dominance_index=None,
        )

    loto_d_arr = np.asarray(loto_depths, dtype=np.float64)
    loto_s_arr = np.asarray(loto_snrs, dtype=np.float64)
    shift_max = float(np.max(np.abs(loto_d_arr - full_depth)))
    dominance = float(max((full_snr - float(np.min(loto_s_arr))) / max(full_snr, 1e-12), 0.0))
    return LCRobustnessMetrics(
        n_epochs_measured=n,
        loto_snr_min=float(np.min(loto_s_arr)),
        loto_snr_max=float(np.max(loto_s_arr)),
        loto_snr_mean=float(np.mean(loto_s_arr)),
        loto_depth_ppm_min=float(np.min(loto_d_arr)),
        loto_depth_ppm_max=float(np.max(loto_d_arr)),
        loto_depth_shift_ppm_max=shift_max,
        dominance_index=dominance,
    )


def _build_lc_robustness_red_noise(
    time: np.ndarray,
    flux: np.ndarray,
    ephemeris: Any,
    *,
    cadence_days: float | None,
) -> LCRobustnessRedNoiseMetrics:
    oot_mask = get_out_of_transit_mask(
        time, ephemeris.period_days, ephemeris.t0_btjd, ephemeris.duration_hours, buffer_factor=2.0
    )
    if not np.any(oot_mask):
        return LCRobustnessRedNoiseMetrics(beta_30m=None, beta_60m=None, beta_duration=None)
    t_oot = time[oot_mask]
    f_oot = flux[oot_mask]
    resid = f_oot - np.median(f_oot)
    beta_30m = _red_noise_beta(resid, t_oot, bin_size_days=30.0 / 1440.0)
    beta_60m = _red_noise_beta(resid, t_oot, bin_size_days=60.0 / 1440.0)
    beta_dur = _red_noise_beta(resid, t_oot, bin_size_days=max(ephemeris.duration_hours, 0.5) / 24.0)
    return LCRobustnessRedNoiseMetrics(beta_30m=beta_30m, beta_60m=beta_60m, beta_duration=beta_dur)


def _build_lc_robustness_fp_signals(
    checks: dict[str, Any],
    time: np.ndarray,
    flux: np.ndarray,
    ephemeris: Any,
) -> LCFPSignals:
    def _metric(check_id: str, key: str) -> float | None:
        check = checks.get(check_id)
        if check is None:
            return None
        metrics = getattr(check, "metrics", None) or {}
        val = metrics.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    phase = ((time - ephemeris.t0_btjd) % ephemeris.period_days) / ephemeris.period_days
    half_dur_phase = ((ephemeris.duration_hours / 24.0) / ephemeris.period_days) / 2.0
    in_secondary = np.abs(phase - 0.5) <= half_dur_phase
    oot = ((phase > 0.15) & (phase < 0.35)) | ((phase > 0.65) & (phase < 0.85))
    phase_0p5_ppm: float | None = None
    if int(np.sum(in_secondary)) >= 3 and int(np.sum(oot)) >= 10:
        sec = float(np.mean(flux[in_secondary]))
        base = float(np.mean(flux[oot]))
        phase_0p5_ppm = float((base - sec) * 1e6)

    odd_even_sigma = _metric("V01", "delta_sigma")
    if odd_even_sigma is None:
        odd_even_sigma = _metric("V01", "depth_diff_sigma")
    return LCFPSignals(
        odd_even_depth_diff_sigma=odd_even_sigma,
        secondary_depth_sigma=_metric("V02", "secondary_depth_sigma"),
        phase_0p5_bin_depth_ppm=phase_0p5_ppm,
        v_shape_metric=_metric("V05", "tflat_ttotal_ratio"),
        asymmetry_sigma=_metric("V15", "asymmetry_sigma"),
    )


def _build_secondary_scan_plot_data(
    lc: Any,
    ephemeris: Any,
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
    phase_full, flux_full = fold_transit(time, flux, period, ephemeris.t0_btjd)

    centers, bflux, berr = _bin_phase_data(
        phase_full,
        flux_full,
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

    # Downsample only the raw scatter display series.
    phase_plot = phase_full
    flux_plot = flux_full
    if len(phase_plot) > max_points:
        pick = np.round(np.linspace(0, len(phase_plot) - 1, max_points)).astype(int)
        phase_plot = phase_plot[pick]
        flux_plot = flux_plot[pick]

    quality = _compute_secondary_scan_quality(
        phase=phase_full,
        bin_err=berr,
        period_days=period,
        bin_minutes=bin_minutes,
    )
    render_hints = _secondary_scan_render_hints(quality, period_days=period, bin_minutes=bin_minutes)

    return SecondaryScanPlotData(
        phase=phase_plot.tolist(),
        flux=flux_plot.tolist(),
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
