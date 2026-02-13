"""Core orchestration and input validation for report build assembly."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from bittr_tess_vetter.compute.transit import detect_transit, measure_depth
from bittr_tess_vetter.report._data import CheckExecutionState, LCSummary, ReportData
from bittr_tess_vetter.validation.base import (
    count_transits,
    get_in_transit_mask,
    get_out_of_transit_mask,
)
from bittr_tess_vetter.validation.report_bridge import run_lc_checks
from bittr_tess_vetter.validation.result_schema import VettingBundleResult

from ._build_panels import (
    _build_alias_harmonic_summary_data,
    _build_full_lc_plot_data,
    _build_lc_robustness_data,
    _build_local_detrend_diagnostic_plot_data,
    _build_odd_even_phase_plot_data,
    _build_oot_context_plot_data,
    _build_per_transit_stack_plot_data,
    _build_phase_folded_plot_data,
    _build_secondary_scan_plot_data,
    _build_timing_series_plot_data,
)
from ._build_utils import _to_internal_lightcurve

logger = logging.getLogger(__name__)

# Default LC checks (V03 excluded by default)
_DEFAULT_ENABLED = {"V01", "V02", "V04", "V05", "V13", "V15"}


def _validate_build_inputs(
    ephemeris: Any,
    bin_minutes: float,
    max_lc_points: int,
    max_phase_points: int,
    max_transit_windows: int,
    max_points_per_window: int,
    max_timing_points: int = 200,
    max_lc_robustness_epochs: int = 128,
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
    _validate_point_budget("max_lc_robustness_epochs", max_lc_robustness_epochs)


def build_report(
    lc: Any,
    candidate: Any,
    *,
    stellar: Any | None = None,
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
    include_lc_robustness: bool = True,
    max_lc_robustness_epochs: int = 128,
) -> ReportData:
    """Assemble LC-only report data packet.

    Runs LC checks via validation.report_bridge.run_lc_checks(), computes LC summary stats
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
        include_lc_robustness: If True, compute LC robustness data.
        max_lc_robustness_epochs: Max per-epoch windows in lc_robustness payload.

    Returns:
        ReportData with all LC report contents populated.
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
        max_lc_robustness_epochs,
    )

    # 1. Determine enabled checks
    enabled = set(_DEFAULT_ENABLED)
    v03_disabled_reason: str | None = None
    if include_v03:
        if stellar is None:
            v03_disabled_reason = "stellar is required to enable V03"
            logger.warning(
                "include_v03=True but stellar is None; disabling V03"
            )
        else:
            enabled.add("V03")

    # 2. Run checks
    internal_lc = _to_internal_lightcurve(lc)
    results = run_lc_checks(
        internal_lc,
        period_days=ephemeris.period_days,
        t0_btjd=ephemeris.t0_btjd,
        duration_hours=ephemeris.duration_hours,
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

    per_transit_stack = None
    local_detrend = None
    oot_context = None
    timing_diag = None
    timing_summary_source = None
    alias_diag = None
    lc_robustness_data = None
    odd_even_phase = None
    secondary_scan = None
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
        timing_diag, timing_summary_source = _build_timing_series_plot_data(
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

    if include_lc_robustness:
        lc_robustness_data = _build_lc_robustness_data(
            lc,
            candidate,
            checks={r.id: r for r in results},
            max_epochs=max_lc_robustness_epochs,
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
        timing_summary_series=timing_summary_source,
        alias_summary=alias_diag,
        lc_robustness=lc_robustness_data,
        odd_even_phase=odd_even_phase,
        secondary_scan=secondary_scan,
        check_execution=CheckExecutionState(
            v03_requested=include_v03,
            v03_enabled="V03" in enabled,
            v03_disabled_reason=v03_disabled_reason,
        ),
        checks_run=[r.id for r in results],
    )


def _compute_lc_summary(
    lc: Any,
    ephemeris: Any,
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
