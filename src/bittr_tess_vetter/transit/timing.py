"""Transit timing measurement primitives.

Pure-compute functions for measuring individual transit times and
computing TTV (Transit Timing Variation) statistics:
- measure_single_transit: Fit trapezoid to single transit window
- measure_all_transit_times: Measure all transits in a light curve
- compute_ttv_statistics: Compute O-C residuals and TTV metrics

All functions use pure numpy/scipy with no I/O operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from bittr_tess_vetter.transit.result import (
    TransitTime,
    TransitTimingPoint,
    TransitTimingSeries,
    TTVResult,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    # Time from transit center
    dt = np.abs(time - t_center)

    # Calculate ingress/egress and flat bottom durations
    half_duration = duration_days / 2.0
    ingress_duration = half_duration * ingress_ratio
    flat_duration = half_duration - ingress_duration

    model = np.ones_like(time, dtype=np.float64)

    # In flat bottom
    in_flat = dt < flat_duration
    model[in_flat] = 1.0 - depth

    # In ingress/egress
    in_transition = (dt >= flat_duration) & (dt < half_duration)
    if np.any(in_transition):
        frac = (dt[in_transition] - flat_duration) / max(ingress_duration, 1e-10)
        model[in_transition] = 1.0 - depth * (1.0 - frac)

    return model


def measure_single_transit(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    t_center_expected: float,
    duration_hours: float,
    window_factor: float = 2.0,
) -> tuple[float, float, float, float, float, bool]:
    """Fit trapezoid model to a single transit window.

    Extracts data around the expected transit center and fits a trapezoid
    model to measure the actual mid-transit time.

    Args:
        time: Time array, in days
        flux: Normalized flux array (median ~1.0)
        flux_err: Flux uncertainties
        t_center_expected: Expected mid-transit time, in days
        duration_hours: Expected transit duration, in hours
        window_factor: Factor to extend window beyond duration (default 2x)

    Returns:
        Tuple of (tc, tc_err, depth, duration_hours_measured, snr, converged)
        - tc: Measured mid-transit time, in days
        - tc_err: Uncertainty on tc, in days
        - depth: Measured fractional depth
        - duration_hours_measured: Measured duration, in hours
        - snr: Signal-to-noise ratio
        - converged: Whether the fit converged
    """
    duration_days = duration_hours / 24.0
    half_window = duration_days * window_factor

    # Extract transit window
    mask = np.abs(time - t_center_expected) < half_window
    if np.sum(mask) < 10:
        # Insufficient data in window
        return (
            t_center_expected,
            duration_days,
            0.0,
            duration_hours,
            0.0,
            False,
        )

    t_window = time[mask]
    f_window = flux[mask]
    f_err_window = flux_err[mask]

    # Filter out NaN/inf
    valid = np.isfinite(f_window) & np.isfinite(f_err_window) & (f_err_window > 0)
    if np.sum(valid) < 10:
        return (
            t_center_expected,
            duration_days,
            0.0,
            duration_hours,
            0.0,
            False,
        )

    t_valid = t_window[valid]
    f_valid = f_window[valid]
    f_err_valid = f_err_window[valid]

    # Initial guesses
    # Estimate depth from in-transit vs out-of-transit flux
    phase_from_center = np.abs(t_valid - t_center_expected)
    in_transit_mask = phase_from_center < duration_days / 2.0
    out_transit_mask = phase_from_center > duration_days * 0.75

    if np.sum(in_transit_mask) > 0 and np.sum(out_transit_mask) > 0:
        baseline = float(np.median(f_valid[out_transit_mask]))
        transit_flux = float(np.median(f_valid[in_transit_mask]))
        initial_depth = max(0.0001, baseline - transit_flux)
    else:
        initial_depth = 0.002  # Default 2000 ppm

    def chi_squared(params: NDArray[np.float64]) -> float:
        tc, depth, duration, ingress_ratio = params
        model = _trapezoid_model_with_center(t_valid, tc, depth, duration, ingress_ratio)
        residuals = (f_valid - model) / f_err_valid
        return float(np.sum(residuals**2))

    # Initial parameters: [t_center, depth, duration_days, ingress_ratio]
    x0 = np.array([t_center_expected, initial_depth, duration_days, 0.2], dtype=np.float64)

    # Bounds
    tc_bound = duration_days * 0.5  # Allow shift up to half a duration
    bounds = [
        (t_center_expected - tc_bound, t_center_expected + tc_bound),  # tc
        (1e-5, 0.5),  # depth
        (duration_days * 0.5, duration_days * 2.0),  # duration
        (0.05, 0.45),  # ingress_ratio
    ]

    result = minimize(chi_squared, x0, bounds=bounds, method="L-BFGS-B")

    tc_fit, depth_fit, duration_fit, ingress_ratio_fit = result.x
    dof = int(np.sum(valid)) - 4
    converged = bool(result.success) and dof > 0

    # Estimate tc uncertainty from chi2 curvature
    tc_err = _estimate_tc_error(
        t_valid,
        f_valid,
        f_err_valid,
        tc_fit,
        depth_fit,
        duration_fit,
        ingress_ratio_fit,
    )

    # Compute SNR
    model = _trapezoid_model_with_center(
        t_valid, tc_fit, depth_fit, duration_fit, ingress_ratio_fit
    )
    residuals = f_valid - model
    rms = float(np.std(residuals))
    snr = depth_fit / rms if rms > 0 else 0.0

    duration_hours_measured = duration_fit * 24.0

    return (tc_fit, tc_err, depth_fit, duration_hours_measured, snr, converged)


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
    """Estimate mid-transit time uncertainty from chi2 curvature.

    Args:
        time: Time array, in days
        flux: Flux array
        flux_err: Flux uncertainties
        tc: Best-fit mid-transit time, in days
        depth: Best-fit depth
        duration: Best-fit duration, in days
        ingress_ratio: Best-fit ingress ratio
        delta: Perturbation for numerical derivative, in days

    Returns:
        Estimated 1-sigma uncertainty on tc, in days
    """

    def chi2_for_tc(t_center: float) -> float:
        model = _trapezoid_model_with_center(time, t_center, depth, duration, ingress_ratio)
        residuals = (flux - model) / flux_err
        return float(np.sum(residuals**2))

    # Compute second derivative (curvature) at best-fit tc
    chi2_center = chi2_for_tc(tc)
    chi2_plus = chi2_for_tc(tc + delta)
    chi2_minus = chi2_for_tc(tc - delta)

    curvature = (chi2_plus - 2 * chi2_center + chi2_minus) / (delta**2)

    if curvature <= 0:
        # Return a fallback uncertainty based on duration
        return duration * 0.1

    # sigma = 1/sqrt(curvature/2) for delta_chi2 = 1
    tc_err = float(np.sqrt(2.0 / curvature))

    # Clamp to reasonable range
    return min(tc_err, duration * 0.5)


def measure_all_transit_times(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period: float,
    t0: float,
    duration_hours: float,
    min_snr: float = 2.0,
) -> list[TransitTime]:
    """Measure mid-transit times for all transits in the light curve.

    Loops over all transit epochs covered by the data and fits each
    individual transit to extract timing measurements.

    Args:
        time: Time array, in days
        flux: Normalized flux array (median ~1.0)
        flux_err: Flux uncertainties
        period: Orbital period, in days
        t0: Reference transit epoch, in days (e.g., BTJD)
        duration_hours: Transit duration, in hours
        min_snr: Minimum SNR to include a transit (default 2.0)

    Returns:
        List of TransitTime measurements for successfully measured transits
    """
    # Find all transit epochs in the data
    time_min = float(np.min(time))
    time_max = float(np.max(time))

    # Calculate epoch range
    epoch_min = int(np.floor((time_min - t0) / period)) - 1
    epoch_max = int(np.ceil((time_max - t0) / period)) + 1

    transit_times: list[TransitTime] = []

    for epoch in range(epoch_min, epoch_max + 1):
        t_center_expected = t0 + epoch * period

        # Skip if outside data range
        if t_center_expected < time_min - period * 0.5:
            continue
        if t_center_expected > time_max + period * 0.5:
            continue

        # Measure this transit
        tc, tc_err, depth, dur_measured, snr, converged = measure_single_transit(
            time, flux, flux_err, t_center_expected, duration_hours
        )

        # Only include if converged and meets SNR threshold
        if converged and snr >= min_snr:
            transit_times.append(
                TransitTime(
                    epoch=epoch,
                    tc=tc,
                    tc_err=tc_err,
                    depth_ppm=depth * 1e6,
                    duration_hours=dur_measured,
                    snr=snr,
                )
            )

    return transit_times


def _flag_outliers(
    transit_times: list[TransitTime],
    o_minus_c_seconds: NDArray[np.float64],
    expected_duration_hours: float,
    oc_sigma_threshold: float = 3.0,
    duration_diff_threshold: float = 0.5,
) -> list[TransitTime]:
    """Flag outlier transits based on O-C and duration criteria.

    Args:
        transit_times: List of measured transit times
        o_minus_c_seconds: O-C residuals, in seconds
        expected_duration_hours: Expected transit duration, in hours
        oc_sigma_threshold: Flag if |O-C| > threshold * MAD (default 3.0)
        duration_diff_threshold: Flag if |dur - expected|/expected > threshold (default 0.5)

    Returns:
        New list of TransitTime objects with outlier flags set
    """
    if len(transit_times) == 0:
        return []

    # Compute median absolute deviation of O-C
    median_oc = float(np.median(o_minus_c_seconds))
    mad_oc = float(np.median(np.abs(o_minus_c_seconds - median_oc)))
    # Convert MAD to sigma-equivalent (MAD * 1.4826 for normal distribution)
    sigma_oc = mad_oc * 1.4826 if mad_oc > 0 else 1.0

    flagged_times: list[TransitTime] = []

    for i, tt in enumerate(transit_times):
        is_outlier = False
        reasons: list[str] = []

        # Check O-C outlier
        oc_deviation = abs(o_minus_c_seconds[i] - median_oc)
        if oc_deviation > oc_sigma_threshold * sigma_oc:
            is_outlier = True
            reasons.append(
                f"O-C={o_minus_c_seconds[i]:.0f}s exceeds {oc_sigma_threshold:.1f}*sigma"
            )

        # Check duration outlier
        if expected_duration_hours > 0:
            duration_diff = abs(tt.duration_hours - expected_duration_hours)
            relative_diff = duration_diff / expected_duration_hours
            if relative_diff > duration_diff_threshold:
                is_outlier = True
                reasons.append(
                    f"duration={tt.duration_hours:.2f}h differs {relative_diff * 100:.0f}% from expected"
                )

        # Create new TransitTime with outlier flags
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
    """Compute TTV statistics from measured transit times.

    Calculates O-C (observed minus calculated) residuals and computes
    summary statistics including RMS, periodicity, and linear trends.
    Also flags outliers based on O-C and duration criteria.

    Args:
        transit_times: List of measured transit times
        period: Orbital period, in days (for expected times)
        t0: Reference epoch, in days (for expected times)
        expected_duration_hours: Expected transit duration for outlier detection, in hours

    Returns:
        TTVResult containing O-C residuals, TTV statistics, and outlier flags
    """
    if len(transit_times) == 0:
        return TTVResult(
            transit_times=[],
            o_minus_c=[],
            rms_seconds=0.0,
            periodicity_sigma=0.0,
            n_transits=0,
            linear_trend=None,
        )

    # Compute expected times
    epochs = np.array([t.epoch for t in transit_times], dtype=np.float64)
    observed_times = np.array([t.tc for t in transit_times], dtype=np.float64)
    expected_times = t0 + epochs * period

    # O-C residuals in days, then convert to seconds
    o_minus_c_days = observed_times - expected_times
    o_minus_c_seconds = o_minus_c_days * 86400.0  # days to seconds

    # RMS of O-C
    rms_seconds = float(np.sqrt(np.mean(o_minus_c_seconds**2)))

    # Flag outliers if expected duration is provided
    if expected_duration_hours is not None:
        flagged_times = _flag_outliers(transit_times, o_minus_c_seconds, expected_duration_hours)
    else:
        # Flag only based on O-C (use median duration as expected)
        median_duration = float(np.median([t.duration_hours for t in transit_times]))
        flagged_times = _flag_outliers(transit_times, o_minus_c_seconds, median_duration)

    # Fit linear trend if enough transits
    linear_trend = None
    if len(transit_times) >= 5:
        # Fit: O-C = a * epoch + b
        coeffs = np.polyfit(epochs, o_minus_c_seconds, 1)
        linear_trend = float(coeffs[0])  # seconds per epoch

    # Check for periodicity
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
        # This is a heuristic score from LS power contrast, not a calibrated sigma.
        periodicity_score=float(ttv.periodicity_sigma),
        linear_trend_sec_per_epoch=(
            float(ttv.linear_trend) if ttv.linear_trend is not None else None
        ),
    )


def _compute_periodicity_significance(
    epochs: NDArray[np.float64],
    o_minus_c: NDArray[np.float64],
) -> float:
    """Compute heuristic periodicity score in O-C residuals.

    Uses Lomb-Scargle periodogram to search for periodic TTVs.

    Args:
        epochs: Transit epoch numbers
        o_minus_c: O-C residuals in seconds

    Returns:
        Heuristic score of best periodic signal from LS power contrast.
        This is not a calibrated false-alarm probability or sigma.
    """
    if len(epochs) < 8:
        return 0.0

    from scipy import signal

    # Remove linear trend first
    coeffs = np.polyfit(epochs, o_minus_c, 1)
    detrended = o_minus_c - np.polyval(coeffs, epochs)

    if np.std(detrended) < 1e-10:
        return 0.0

    # Search for periods from 2 to N/2 epochs
    n_epochs = len(epochs)
    min_period = 2.0
    max_period = float(n_epochs) / 2.0

    if max_period <= min_period:
        return 0.0

    # Generate period grid
    periods = np.linspace(min_period, max_period, 100, dtype=np.float64)
    angular_freqs = 2.0 * np.pi / periods

    # Compute Lomb-Scargle periodogram
    try:
        power = signal.lombscargle(
            epochs.astype(np.float64),
            detrended.astype(np.float64),
            angular_freqs,
            normalize=True,
        )
    except (ValueError, ZeroDivisionError):
        return 0.0

    # Find best peak
    best_power = float(np.max(power))

    # Convert to significance using MAD-based noise estimate
    median_power = float(np.median(power))
    mad = float(np.median(np.abs(power - median_power)))
    sigma = mad * 1.4826  # MAD to sigma conversion

    if sigma <= 0:
        return 0.0

    significance = (best_power - median_power) / sigma
    return max(0.0, significance)
