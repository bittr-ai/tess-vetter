"""Transit timing measurement primitives.

Pure-compute functions for measuring individual transit times and
computing TTV (Transit Timing Variation) statistics:
- measure_single_transit: Fit trapezoid to single transit window
- measure_all_transit_times: Measure all transits in a light curve
- compute_ttv_statistics: Compute O-C residuals and TTV metrics

All functions use pure numpy/scipy with no I/O operations.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from scipy.optimize import minimize

from tess_vetter.transit.result import (
    TransitTime,
    TransitTimingPoint,
    TransitTimingSeries,
    TTVResult,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TransitMeasurementDetails(TypedDict):
    tc: float
    tc_err: float
    depth: float
    duration_hours_measured: float
    snr: float
    snr_method: str
    converged: bool
    n_window_points: int
    n_valid_points: int
    window_factor_used: float
    reject_reason: str | None
    stages: list[dict[str, object]]


def _robust_scatter_mad(values: NDArray[np.float64]) -> float:
    if values.size == 0:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    sigma = mad * 1.4826
    if np.isfinite(sigma) and sigma > 0.0:
        return sigma
    std = float(np.std(finite))
    return std if np.isfinite(std) and std > 0.0 else 0.0


def _compute_snr_with_fallback(
    *,
    depth_fit: float,
    residuals: NDArray[np.float64],
    f_valid: NDArray[np.float64],
    t_valid: NDArray[np.float64],
    tc_fit: float,
    duration_fit: float,
) -> tuple[float, str]:
    rms_residual = float(np.std(residuals))
    snr_residual = (
        float(depth_fit / rms_residual)
        if np.isfinite(rms_residual) and rms_residual > 0.0
        else 0.0
    )

    phase_from_center = np.abs(t_valid - tc_fit)
    out_transit_mask = phase_from_center > (duration_fit * 0.75)
    snr_oot = 0.0
    if int(np.sum(out_transit_mask)) >= 8:
        oot_sigma = _robust_scatter_mad(f_valid[out_transit_mask])
        if np.isfinite(oot_sigma) and oot_sigma > 0.0:
            snr_oot = float(depth_fit / oot_sigma)

    # Use OOT fallback when residual-based SNR is unstable or strongly pessimistic.
    if snr_oot > 0.0 and (snr_residual <= 0.0 or snr_residual < (0.75 * snr_oot)):
        return float(snr_oot), "oot_mad_fallback"
    return float(max(snr_residual, 0.0)), "residual"


def _trapezoid_model_with_center(
    time: NDArray[np.float64],
    t_center: float,
    depth: float,
    duration_days: float,
    ingress_ratio: float,
) -> NDArray[np.float64]:
    """Compute trapezoid transit model centered at t_center.

    Args:
        time: Time array, in days (e.g., BTJD)
        t_center: Mid-transit time, in days
        depth: Fractional transit depth
        duration_days: Total transit duration, in days
        ingress_ratio: Ingress/egress as fraction of duration

    Returns:
        Model flux array
    """
    dt = np.abs(time - t_center)

    half_duration = duration_days / 2.0
    ingress_duration = half_duration * ingress_ratio
    flat_duration = half_duration - ingress_duration

    model = np.ones_like(time, dtype=np.float64)

    in_flat = dt < flat_duration
    model[in_flat] = 1.0 - depth

    in_transition = (dt >= flat_duration) & (dt < half_duration)
    if np.any(in_transition):
        frac = (dt[in_transition] - flat_duration) / max(ingress_duration, 1e-10)
        model[in_transition] = 1.0 - depth * (1.0 - frac)

    return model


def _estimate_tc_error(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    tc: float,
    depth: float,
    duration: float,
    ingress_ratio: float,
    delta: float = 0.0001,
) -> float:
    """Estimate mid-transit time uncertainty from chi2 curvature."""

    def chi2_for_tc(t_center: float) -> float:
        model = _trapezoid_model_with_center(time, t_center, depth, duration, ingress_ratio)
        residuals = (flux - model) / flux_err
        return float(np.sum(residuals**2))

    chi2_center = chi2_for_tc(tc)
    chi2_plus = chi2_for_tc(tc + delta)
    chi2_minus = chi2_for_tc(tc - delta)

    curvature = (chi2_plus - 2 * chi2_center + chi2_minus) / (delta**2)

    if curvature <= 0:
        return duration * 0.1

    tc_err = float(np.sqrt(2.0 / curvature))
    return min(tc_err, duration * 0.5)


def _measure_single_transit_detailed(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    t_center_expected: float,
    duration_hours: float,
    *,
    window_factors: tuple[float, ...] = (2.0, 4.0, 6.0),
) -> TransitMeasurementDetails:
    duration_days = duration_hours / 24.0
    final_reason = "no_successful_stage"
    stage_rows: list[dict[str, object]] = []

    for stage_index, window_factor in enumerate(window_factors):
        half_window = duration_days * float(window_factor)

        mask = np.abs(time - t_center_expected) < half_window
        n_window_points = int(np.sum(mask))
        stage_row: dict[str, object] = {
            "stage_index": int(stage_index),
            "window_factor": float(window_factor),
            "n_window_points": int(n_window_points),
        }
        if n_window_points < 10:
            final_reason = "insufficient_points"
            stage_row["reason"] = final_reason
            stage_rows.append(stage_row)
            continue

        t_window = time[mask]
        f_window = flux[mask]
        f_err_window = flux_err[mask]

        valid = np.isfinite(f_window) & np.isfinite(f_err_window) & (f_err_window > 0)
        n_valid_points = int(np.sum(valid))
        stage_row["n_valid_points"] = int(n_valid_points)
        if n_valid_points < 10:
            final_reason = "insufficient_valid_points"
            stage_row["reason"] = final_reason
            stage_rows.append(stage_row)
            continue

        t_valid = t_window[valid]
        f_valid = f_window[valid]
        f_err_valid = f_err_window[valid]

        phase_from_center = np.abs(t_valid - t_center_expected)
        in_transit_mask = phase_from_center < duration_days / 2.0
        out_transit_mask = phase_from_center > duration_days * 0.75

        if np.sum(in_transit_mask) > 0 and np.sum(out_transit_mask) > 0:
            baseline = float(np.median(f_valid[out_transit_mask]))
            transit_flux = float(np.median(f_valid[in_transit_mask]))
            initial_depth = max(0.0001, baseline - transit_flux)
        else:
            initial_depth = 0.002

        def chi_squared(
            params: NDArray[np.float64],
            t_valid: NDArray[np.float64] = t_valid,
            f_valid: NDArray[np.float64] = f_valid,
            f_err_valid: NDArray[np.float64] = f_err_valid,
        ) -> float:
            tc, depth, duration, ingress_ratio = params
            model = _trapezoid_model_with_center(t_valid, tc, depth, duration, ingress_ratio)
            residuals = (f_valid - model) / f_err_valid
            return float(np.sum(residuals**2))

        x0 = np.array([t_center_expected, initial_depth, duration_days, 0.2], dtype=np.float64)

        tc_bound = duration_days * max(0.5, float(window_factor) / 2.0)
        duration_low_factor = max(0.25, 0.5 - 0.1 * float(stage_index))
        duration_high_factor = min(3.0, 2.0 + 0.5 * float(stage_index))
        bounds = [
            (t_center_expected - tc_bound, t_center_expected + tc_bound),
            (1e-5, 0.5),
            (duration_days * duration_low_factor, duration_days * duration_high_factor),
            (0.05, 0.45),
        ]

        result = minimize(chi_squared, x0, bounds=bounds, method="L-BFGS-B")
        tc_fit, depth_fit, duration_fit, ingress_ratio_fit = result.x
        dof = int(np.sum(valid)) - 4
        converged = bool(result.success) and dof > 0
        stage_row["converged"] = bool(converged)
        stage_row["dof"] = int(dof)
        if not converged:
            final_reason = "optimizer_failed"
            stage_row["reason"] = final_reason
            stage_rows.append(stage_row)
            continue

        tc_err = _estimate_tc_error(
            t_valid,
            f_valid,
            f_err_valid,
            tc_fit,
            depth_fit,
            duration_fit,
            ingress_ratio_fit,
        )
        model = _trapezoid_model_with_center(
            t_valid, tc_fit, depth_fit, duration_fit, ingress_ratio_fit
        )
        residuals = f_valid - model
        snr, snr_method = _compute_snr_with_fallback(
            depth_fit=float(depth_fit),
            residuals=residuals,
            f_valid=f_valid,
            t_valid=t_valid,
            tc_fit=float(tc_fit),
            duration_fit=float(duration_fit),
        )
        stage_row["snr"] = float(snr)
        stage_row["snr_method"] = snr_method
        stage_rows.append(stage_row)

        return {
            "tc": float(tc_fit),
            "tc_err": float(tc_err),
            "depth": float(depth_fit),
            "duration_hours_measured": float(duration_fit * 24.0),
            "snr": float(snr),
            "snr_method": snr_method,
            "converged": True,
            "n_window_points": int(n_window_points),
            "n_valid_points": int(n_valid_points),
            "window_factor_used": float(window_factor),
            "reject_reason": None,
            "stages": stage_rows,
        }

    return {
        "tc": float(t_center_expected),
        "tc_err": float(duration_days),
        "depth": 0.0,
        "duration_hours_measured": float(duration_hours),
        "snr": 0.0,
        "snr_method": "none",
        "converged": False,
        "n_window_points": 0,
        "n_valid_points": 0,
        "window_factor_used": float(window_factors[-1]) if len(window_factors) > 0 else 2.0,
        "reject_reason": final_reason,
        "stages": stage_rows,
    }


def measure_single_transit(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    t_center_expected: float,
    duration_hours: float,
    window_factor: float = 2.0,
) -> tuple[float, float, float, float, float, bool]:
    """Fit trapezoid model to a single transit window."""
    details = _measure_single_transit_detailed(
        time=time,
        flux=flux,
        flux_err=flux_err,
        t_center_expected=t_center_expected,
        duration_hours=duration_hours,
        window_factors=(float(window_factor),),
    )
    return (
        float(details["tc"]),
        float(details["tc_err"]),
        float(details["depth"]),
        float(details["duration_hours_measured"]),
        float(details["snr"]),
        bool(details["converged"]),
    )


def measure_all_transit_times_with_diagnostics(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period: float,
    t0: float,
    duration_hours: float,
    min_snr: float = 2.0,
) -> tuple[list[TransitTime], dict[str, object]]:
    """Measure transit times and return per-epoch diagnostics."""
    time_min = float(np.min(time))
    time_max = float(np.max(time))

    epoch_min = int(np.floor((time_min - t0) / period)) - 1
    epoch_max = int(np.ceil((time_max - t0) / period)) + 1

    transit_times: list[TransitTime] = []
    epoch_rows: list[dict[str, object]] = []
    reject_counts: Counter[str] = Counter()
    snr_method_counts: Counter[str] = Counter()

    for epoch in range(epoch_min, epoch_max + 1):
        t_center_expected = t0 + epoch * period

        if t_center_expected < time_min - period * 0.5:
            continue
        if t_center_expected > time_max + period * 0.5:
            continue

        details = _measure_single_transit_detailed(
            time=time,
            flux=flux,
            flux_err=flux_err,
            t_center_expected=t_center_expected,
            duration_hours=duration_hours,
        )
        converged = bool(details["converged"])
        snr = float(details["snr"])
        accepted = bool(converged and snr >= min_snr)

        reject_reason: str | None = None
        if accepted:
            transit_times.append(
                TransitTime(
                    epoch=epoch,
                    tc=float(details["tc"]),
                    tc_err=float(details["tc_err"]),
                    depth_ppm=float(details["depth"]) * 1e6,
                    duration_hours=float(details["duration_hours_measured"]),
                    snr=float(snr),
                )
            )
            snr_method_counts[str(details["snr_method"])] += 1
        else:
            reject_reason = (
                "snr_below_threshold"
                if converged and snr < min_snr
                else str(details["reject_reason"] or "rejected")
            )
            reject_counts[reject_reason] += 1

        epoch_rows.append(
            {
                "epoch": int(epoch),
                "t_center_expected": float(t_center_expected),
                "converged": bool(converged),
                "accepted": bool(accepted),
                "reject_reason": reject_reason,
                "snr": float(snr),
                "snr_method": str(details["snr_method"]),
                "window_factor_used": float(details["window_factor_used"]),
                "n_window_points": int(details["n_window_points"]),
                "n_valid_points": int(details["n_valid_points"]),
                "stages": details["stages"],
            }
        )

    diagnostics: dict[str, object] = {
        "attempted_epochs": int(len(epoch_rows)),
        "accepted_epochs": int(len(transit_times)),
        "reject_counts": dict(reject_counts),
        "snr_method_counts": dict(snr_method_counts),
        "epoch_details": epoch_rows,
    }
    return transit_times, diagnostics


def measure_all_transit_times(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period: float,
    t0: float,
    duration_hours: float,
    min_snr: float = 2.0,
) -> list[TransitTime]:
    """Measure mid-transit times for all transits in the light curve."""
    transit_times, _ = measure_all_transit_times_with_diagnostics(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=period,
        t0=t0,
        duration_hours=duration_hours,
        min_snr=min_snr,
    )
    return transit_times


def _flag_outliers(
    transit_times: list[TransitTime],
    o_minus_c_seconds: NDArray[np.float64],
    expected_duration_hours: float,
    oc_sigma_threshold: float = 3.0,
    duration_diff_threshold: float = 0.5,
) -> list[TransitTime]:
    """Flag outlier transits based on O-C and duration criteria."""
    if len(transit_times) == 0:
        return []

    median_oc = float(np.median(o_minus_c_seconds))
    mad_oc = float(np.median(np.abs(o_minus_c_seconds - median_oc)))
    sigma_oc = mad_oc * 1.4826 if mad_oc > 0 else 1.0

    flagged_times: list[TransitTime] = []

    for i, tt in enumerate(transit_times):
        is_outlier = False
        reasons: list[str] = []

        oc_deviation = abs(o_minus_c_seconds[i] - median_oc)
        if oc_deviation > oc_sigma_threshold * sigma_oc:
            is_outlier = True
            reasons.append(
                f"O-C={o_minus_c_seconds[i]:.0f}s exceeds {oc_sigma_threshold:.1f}*sigma"
            )

        if expected_duration_hours > 0:
            duration_diff = abs(tt.duration_hours - expected_duration_hours)
            relative_diff = duration_diff / expected_duration_hours
            if relative_diff > duration_diff_threshold:
                is_outlier = True
                reasons.append(
                    f"duration={tt.duration_hours:.2f}h differs {relative_diff * 100:.0f}% from expected"
                )

        outlier_reason = "; ".join(reasons) if reasons else None
        flagged_times.append(
            TransitTime(
                epoch=tt.epoch,
                tc=tt.tc,
                tc_err=tt.tc_err,
                depth_ppm=tt.depth_ppm,
                duration_hours=tt.duration_hours,
                snr=tt.snr,
                is_outlier=is_outlier,
                outlier_reason=outlier_reason,
            )
        )

    return flagged_times


def compute_ttv_statistics(
    transit_times: list[TransitTime],
    period: float,
    t0: float,
    expected_duration_hours: float | None = None,
) -> TTVResult:
    """Compute TTV statistics from measured transit times."""
    if len(transit_times) == 0:
        return TTVResult(
            transit_times=[],
            o_minus_c=[],
            rms_seconds=0.0,
            periodicity_sigma=0.0,
            n_transits=0,
            linear_trend=None,
        )

    epochs = np.array([t.epoch for t in transit_times], dtype=np.float64)
    observed_times = np.array([t.tc for t in transit_times], dtype=np.float64)
    expected_times = t0 + epochs * period

    o_minus_c_days = observed_times - expected_times
    o_minus_c_seconds = o_minus_c_days * 86400.0

    rms_seconds = float(np.sqrt(np.mean(o_minus_c_seconds**2)))

    if expected_duration_hours is not None:
        flagged_times = _flag_outliers(transit_times, o_minus_c_seconds, expected_duration_hours)
    else:
        median_duration = float(np.median([t.duration_hours for t in transit_times]))
        flagged_times = _flag_outliers(transit_times, o_minus_c_seconds, median_duration)

    linear_trend = None
    if len(transit_times) >= 5:
        coeffs = np.polyfit(epochs, o_minus_c_seconds, 1)
        linear_trend = float(coeffs[0])

    periodicity_sigma = _compute_periodicity_significance(epochs, o_minus_c_seconds)

    return TTVResult(
        transit_times=flagged_times,
        o_minus_c=o_minus_c_seconds.tolist(),
        rms_seconds=rms_seconds,
        periodicity_sigma=periodicity_sigma,
        n_transits=len(flagged_times),
        linear_trend=linear_trend,
    )


def build_timing_series(
    transit_times: list[TransitTime],
    period: float,
    t0: float,
) -> TransitTimingSeries:
    """Build per-epoch timing diagnostics from measured transit times."""
    if len(transit_times) == 0:
        return TransitTimingSeries(
            points=[],
            n_points=0,
            rms_seconds=None,
            periodicity_score=None,
            linear_trend_sec_per_epoch=None,
        )

    expected_duration = float(np.median([t.duration_hours for t in transit_times]))
    ttv = compute_ttv_statistics(
        transit_times=transit_times,
        period=period,
        t0=t0,
        expected_duration_hours=expected_duration,
    )

    points: list[TransitTimingPoint] = []
    for tt, oc in zip(ttv.transit_times, ttv.o_minus_c, strict=False):
        points.append(
            TransitTimingPoint(
                epoch=int(tt.epoch),
                tc_btjd=float(tt.tc),
                tc_err_days=float(tt.tc_err),
                oc_seconds=float(oc),
                snr=float(tt.snr),
                depth_ppm=float(tt.depth_ppm),
                duration_hours=float(tt.duration_hours),
                is_outlier=bool(tt.is_outlier),
                outlier_reason=tt.outlier_reason,
            )
        )

    return TransitTimingSeries(
        points=points,
        n_points=len(points),
        rms_seconds=float(ttv.rms_seconds),
        periodicity_score=float(ttv.periodicity_sigma),
        linear_trend_sec_per_epoch=(
            float(ttv.linear_trend) if ttv.linear_trend is not None else None
        ),
    )


def _compute_periodicity_significance(
    epochs: NDArray[np.float64],
    o_minus_c: NDArray[np.float64],
) -> float:
    """Compute heuristic periodicity score in O-C residuals."""
    if len(epochs) < 8:
        return 0.0

    from scipy import signal

    coeffs = np.polyfit(epochs, o_minus_c, 1)
    detrended = o_minus_c - np.polyval(coeffs, epochs)

    if np.std(detrended) < 1e-10:
        return 0.0

    n_epochs = len(epochs)
    min_period = 2.0
    max_period = float(n_epochs) / 2.0

    if max_period <= min_period:
        return 0.0

    periods = np.linspace(min_period, max_period, 100, dtype=np.float64)
    angular_freqs = 2.0 * np.pi / periods

    try:
        power = signal.lombscargle(
            epochs.astype(np.float64),
            detrended.astype(np.float64),
            angular_freqs,
            normalize=True,
        )
    except (ValueError, ZeroDivisionError):
        return 0.0

    best_power = float(np.max(power))

    median_power = float(np.median(power))
    mad = float(np.median(np.abs(power - median_power)))
    sigma = mad * 1.4826

    if sigma <= 0:
        return 0.0

    significance = (best_power - median_power) / sigma
    return max(0.0, significance)
