"""Transit recovery pipeline for active stars (pure core).

This module contains the recovery algorithm as a pure function suitable for
host applications (e.g., MCP servers):
- array-in / dict-out
- no cache access
- no network
- no file I/O
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from bittr_tess_vetter.compute.transit import get_transit_mask
from bittr_tess_vetter.recovery.primitives import (
    _trapezoid_model,
    detrend_for_recovery,
    estimate_rotation_period,
    fit_trapezoid,
    stack_transits,
)


def recover_transit_timeseries(
    *,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
    rotation_period: float | None = None,
    n_harmonics: int = 3,
    detection_threshold: float = 5.0,
    detrend_method: Literal["harmonic", "wotan_biweight", "wotan_gp"] = "harmonic",
    output_mode: Literal["window", "full_phase", "both"] = "window",
    full_phase_bins: int = 200,
    tic_id: int = 0,
    sectors_used: list[int] | None = None,
) -> dict[str, Any]:
    """Recover a known transit signal by removing stellar variability and stacking transits."""
    if len(time) != len(flux) or len(time) != len(flux_err):
        raise ValueError("time/flux/flux_err must have the same length")
    if len(time) < 10:
        raise ValueError("Insufficient data points for recovery")
    if period <= 0 or duration_hours <= 0:
        raise ValueError("period and duration_hours must be positive")

    sort_idx = np.argsort(time)
    time = np.asarray(time, dtype=np.float64)[sort_idx]
    flux = np.asarray(flux, dtype=np.float64)[sort_idx]
    flux_err = np.asarray(flux_err, dtype=np.float64)[sort_idx]

    rotation_period_estimated = False
    if rotation_period is None:
        rot_period, rot_snr = estimate_rotation_period(time, flux)
        if rot_snr < 3.0:
            rot_period = 5.0
        rotation_period_estimated = True
    else:
        rot_period = float(rotation_period)

    variability_amplitude_ppm = float(np.std(flux) * 1e6)
    duration_hours_wide = float(duration_hours) * 1.5
    transit_mask = get_transit_mask(time, float(period), float(t0), float(duration_hours_wide))

    detrended_flux = detrend_for_recovery(
        time,
        flux,
        transit_mask,
        method=str(detrend_method),
        rotation_period=float(rot_period),
        n_harmonics=int(n_harmonics),
    )

    oot_mask = ~transit_mask
    median_oot = float(np.median(detrended_flux[oot_mask])) if np.any(oot_mask) else 1.0
    if median_oot > 0:
        detrended_flux = detrended_flux / median_oot

    residual_scatter_ppm = (
        float(np.std(detrended_flux[oot_mask]) * 1e6) if np.any(oot_mask) else 0.0
    )

    stacked = stack_transits(
        time,
        detrended_flux,
        flux_err,
        float(period),
        float(t0),
        float(duration_hours),
    )

    initial_depth = 0.003
    initial_duration_phase = float(duration_hours) / 24.0 / float(period)
    fit = fit_trapezoid(
        stacked.phase,
        stacked.flux,
        stacked.flux_err,
        initial_depth=initial_depth,
        initial_duration_phase=initial_duration_phase,
    )
    if not fit.converged:
        raise RuntimeError("Transit fit failed to converge. Check ephemeris parameters.")

    snr_val = float(fit.depth / fit.depth_err) if fit.depth_err > 0 else 0.0
    detected = snr_val >= float(detection_threshold)
    model_flux = _trapezoid_model(stacked.phase, fit.depth, fit.duration_phase, fit.ingress_ratio)

    full_phase_stack = None
    full_phase_model = None
    if output_mode in ("full_phase", "both"):
        full_phase_stack = stack_transits(
            time,
            detrended_flux,
            flux_err,
            float(period),
            float(t0),
            float(duration_hours),
            phase_bins=int(full_phase_bins),
            phase_min=0.0,
            phase_max=1.0,
        )
        full_phase_model = _trapezoid_model(
            full_phase_stack.phase,
            fit.depth,
            fit.duration_phase,
            fit.ingress_ratio,
        )

    duration_hours_measured = float(fit.duration_phase * float(period) * 24.0)
    baseline_days = float(time[-1] - time[0])
    n_points_in_transit = int(np.sum(stacked.n_points_per_bin))

    result: dict[str, Any] = {
        "detected": bool(detected),
        "snr": round(snr_val, 2),
        "depth_ppm": round(float(fit.depth) * 1e6, 1),
        "depth_err_ppm": round(float(fit.depth_err) * 1e6, 1),
        "duration_hours_measured": round(duration_hours_measured, 2),
        "n_transits": int(stacked.n_transits),
        "n_points_in_transit": int(n_points_in_transit),
        "chi2": round(float(fit.chi2), 2),
        "reduced_chi2": round(float(fit.reduced_chi2), 2),
        "stacked_phase": stacked.phase.tolist(),
        "stacked_flux": stacked.flux.tolist(),
        "stacked_flux_err": stacked.flux_err.tolist(),
        "stacked_model": model_flux.tolist(),
        "output_mode": str(output_mode),
        "detrend_method": str(detrend_method),
        "rotation_period_used": round(float(rot_period), 3),
        "rotation_period_estimated": bool(rotation_period_estimated),
        "variability_amplitude_ppm": round(float(variability_amplitude_ppm), 0),
        "residual_scatter_ppm": round(float(residual_scatter_ppm), 0),
        "sectors_used": list(sectors_used or []),
        "baseline_days": round(float(baseline_days), 1),
        "tic_id": int(tic_id),
    }

    if full_phase_stack is not None:
        result["recovered_phase_full"] = full_phase_stack.phase.tolist()
        result["recovered_flux_full"] = full_phase_stack.flux.tolist()
        result["recovered_flux_err_full"] = full_phase_stack.flux_err.tolist()
        if full_phase_model is not None:
            result["recovered_model_full"] = full_phase_model.tolist()

    if not detected:
        result["depth_upper_limit_ppm"] = round(3.0 * float(fit.depth_err) * 1e6, 1)

    return result
