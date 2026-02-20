"""ModShift / signal uniqueness metrics.

Independent implementation based on published algorithms from the Kepler
Robovetter. Computes properly-scaled Fred (red/white noise ratio) and
MS1-MS6 uniqueness metrics for TESS transit candidate vetting.

Algorithm references:
    Thompson, S.E., et al. (2018). ApJS, 235, 38.
    "Planetary Candidates Observed by Kepler. VIII."

    Coughlin, J.L., et al. (2016). ApJS, 224, 12.
    "Planetary Candidates Observed by Kepler. VII."

See also:
    Kunimoto, M., et al. (2025). AJ, 170, 280.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.special import erfcinv

logger = logging.getLogger(__name__)


def _weighted_mean(y: np.ndarray, dy: np.ndarray) -> float:
    """Inverse-variance weighted mean.

    Args:
        y: Data values.
        dy: Uncertainties on data values (must be positive).

    Returns:
        Weighted mean of y.
    """
    w = 1.0 / (dy**2)
    return float(np.sum(w * y) / np.sum(w))


def _weighted_std(y: np.ndarray, dy: np.ndarray) -> float:
    """Inverse-variance weighted standard deviation.

    Uses N-1 correction for unbiased estimation.

    Args:
        y: Data values.
        dy: Uncertainties on data values (must be positive).

    Returns:
        Weighted standard deviation of y.
    """
    if len(y) < 2:
        return np.nan
    w = 1.0 / (dy**2)
    n = len(w)
    mean = np.sum(w * y) / np.sum(w)
    variance = np.sum(w * (y - mean) ** 2) / ((n - 1) * np.sum(w) / n)
    return float(np.sqrt(variance))


def _phasefold(t: np.ndarray, period: float, epoch: float) -> np.ndarray:
    """Phase-fold times to [-0.5, 0.5] with transit at 0.

    Args:
        t: Time array.
        period: Orbital period in days.
        epoch: Transit epoch in days.

    Returns:
        Phase array in range [-0.5, 0.5].
    """
    phase = np.mod(t - epoch, period) / period
    phase[phase > 0.5] -= 1.0
    return phase


def _compute_mes_series_binned(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    epoch: float,
    qtran: float,
    n_bins: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Compute MES series using phase binning for efficiency.

    This is an O(N + n_bins) approximation of the O(N^2) naive algorithm.
    For each point, we look up the pre-computed bin at its phase.

    Args:
        time: Time array (days).
        flux: Flux array (normalized).
        flux_err: Flux uncertainty array.
        period: Orbital period (days).
        epoch: Transit epoch (days).
        qtran: Transit duration as fraction of period.
        n_bins: Number of phase bins.

    Returns:
        Tuple of (MES_series, dep_series, err_series, zpt, n_in).
    """
    n = len(time)
    phase = _phasefold(time, period, epoch)

    # Identify in-transit and out-of-transit points
    in_tran = np.abs(phase) < qtran
    out_tran = ~in_tran

    n_in = int(np.sum(in_tran))
    n_out = int(np.sum(out_tran))

    if n_out < 3:
        # Not enough out-of-transit data
        return (
            np.full(n, np.nan),
            np.full(n, np.nan),
            np.full(n, np.nan),
            np.nan,
            n_in,
        )

    # Zero point from out-of-transit flux
    zpt = _weighted_mean(flux[out_tran], flux_err[out_tran])

    # Convert phase to [0, 1] for binning
    phase_positive = phase.copy()
    phase_positive[phase_positive < 0] += 1.0

    # Bin edges and centers
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Half transit width in phase units (for window around each bin)
    half_tran_phase = 0.5 * qtran

    # Assign each point to a bin
    bin_indices = np.clip(np.floor(phase_positive * n_bins).astype(int), 0, n_bins - 1)

    # For each bin, compute the weighted mean depth
    bin_depths = np.zeros(n_bins)
    bin_errors = np.zeros(n_bins)
    bin_n_transits = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        center = bin_centers[b]

        # Circular distance in phase
        dist_to_center = np.abs(phase_positive - center)
        dist_to_center = np.minimum(dist_to_center, 1.0 - dist_to_center)

        # Points within half a transit duration of this phase
        in_window = dist_to_center < half_tran_phase

        if np.sum(in_window) < 1:
            bin_depths[b] = np.nan
            bin_errors[b] = np.nan
            bin_n_transits[b] = 0
            continue

        window_flux = flux[in_window]
        window_err = flux_err[in_window]
        window_time = time[in_window]

        # Depth at this phase
        bin_depths[b] = zpt - _weighted_mean(window_flux, window_err)

        # Count unique epochs contributing
        epochs = np.round((window_time - window_time[0]) / period)
        bin_n_transits[b] = len(np.unique(epochs))

        # Error estimate (simplified)
        w = 1.0 / (window_err**2)
        bin_errors[b] = 1.0 / np.sqrt(np.sum(w))

    # Map bin values back to each point
    dep_series = bin_depths[bin_indices]
    err_series = bin_errors[bin_indices]
    n_transit_series = bin_n_transits[bin_indices]

    # Estimate white and red noise for error scaling
    sig_w = _weighted_std(flux[out_tran], flux_err[out_tran])

    # Simple red noise estimate from binned residuals (Hartman & Bakos 2016)
    # We use the scatter of bin depths as a proxy
    valid_bins = ~np.isnan(bin_depths)
    if np.sum(valid_bins) > 2:
        # Red noise from scatter of depth measurements in non-transit bins
        non_tran_bins = np.abs(bin_centers - 0.5) > 2 * qtran  # Far from secondary
        non_tran_bins &= np.abs(bin_centers) > 2 * qtran  # Far from primary (handles wrap)
        non_tran_bins &= valid_bins
        sig_r = np.std(bin_depths[non_tran_bins]) if np.sum(non_tran_bins) > 2 else sig_w
    else:
        sig_r = sig_w

    # Combined error: white + red noise
    n_per_bin = np.maximum(n_transit_series, 1)
    combined_err = np.sqrt((sig_w**2 / n_in) + (sig_r**2 / n_per_bin))

    # MES series
    with np.errstate(divide="ignore", invalid="ignore"):
        mes_series = dep_series / combined_err

    return mes_series, dep_series, err_series, zpt, n_in


def _compute_false_alarm_thresholds(qtran: float, n_tce: int) -> tuple[float, float]:
    """Compute false alarm thresholds FA1 and FA2.

    From Thompson et al. 2018, using complementary error function.

    Args:
        qtran: Transit duration as fraction of period.
        n_tce: Number of TCEs (threshold crossing events) to account for.

    Returns:
        Tuple of (FA1, FA2).
    """
    # FA1: Multi-TCE aware
    fa1_arg = qtran * (1.0 / n_tce)
    fa1_arg = np.clip(fa1_arg, 1e-15, 2.0 - 1e-15)  # erfcinv domain
    fa1 = float(np.sqrt(2) * erfcinv(fa1_arg))

    # FA2: Single TCE
    fa2_arg = np.clip(qtran, 1e-15, 2.0 - 1e-15)
    fa2 = float(np.sqrt(2) * erfcinv(fa2_arg))

    return fa1, fa2


def _find_signal_peaks(
    mes_series: np.ndarray,
    phase: np.ndarray,
    qtran: float,
) -> tuple[float, float, float, float, float, float]:
    """Find primary, secondary, tertiary, and positive signal peaks.

    Args:
        mes_series: MES time series.
        phase: Phase array in [-0.5, 0.5].
        qtran: Transit duration as fraction of period.

    Returns:
        Tuple of (sig_pri, sig_sec, sig_ter, sig_pos, phs_pri, phs_sec).
    """
    # Primary: max MES in transit window (near phase 0)
    in_tran = np.abs(phase) < qtran
    if not np.any(in_tran):
        return np.nan, np.nan, np.nan, np.nan, 0.0, 0.5

    valid_pri = in_tran & np.isfinite(mes_series)
    if not np.any(valid_pri):
        return np.nan, np.nan, np.nan, np.nan, 0.0, 0.5

    arg_pri = np.argmax(mes_series[valid_pri])
    indices_pri = np.where(valid_pri)[0]
    sig_pri = float(mes_series[indices_pri[arg_pri]])
    phs_pri = float(phase[indices_pri[arg_pri]])

    # Secondary: max MES at least 2 durations from primary
    # Use circular distance in phase
    dist_from_pri = np.abs(phase - phs_pri)
    dist_from_pri = np.minimum(dist_from_pri, 1.0 - dist_from_pri)
    mask_sec = dist_from_pri >= 2 * qtran
    mask_sec &= np.isfinite(mes_series)

    if not np.any(mask_sec):
        sig_sec = 0.0
        phs_sec = 0.5  # Default to opposite phase
    else:
        arg_sec = np.argmax(mes_series[mask_sec])
        indices_sec = np.where(mask_sec)[0]
        sig_sec = float(mes_series[indices_sec[arg_sec]])
        phs_sec = float(phase[indices_sec[arg_sec]])

    # Tertiary: max MES at least 2 durations from both primary and secondary
    dist_from_sec = np.abs(phase - phs_sec)
    dist_from_sec = np.minimum(dist_from_sec, 1.0 - dist_from_sec)
    mask_ter = (dist_from_pri >= 2 * qtran) & (dist_from_sec >= 2 * qtran)
    mask_ter &= np.isfinite(mes_series)

    sig_ter = 0.0 if not np.any(mask_ter) else float(np.nanmax(mes_series[mask_ter]))

    # Positive: max of NEGATIVE MES (inverted transits) at least 3 durations
    # from primary and secondary
    mask_pos = (dist_from_pri >= 3 * qtran) & (dist_from_sec >= 3 * qtran)
    mask_pos &= np.isfinite(mes_series)

    sig_pos = 0.0 if not np.any(mask_pos) else float(np.nanmax(-mes_series[mask_pos]))

    return sig_pri, sig_sec, sig_ter, sig_pos, phs_pri, phs_sec


def _compute_fred(
    dep_series: np.ndarray,
    err_series: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    phase: np.ndarray,
    qtran: float,
    phs_pri: float,
    phs_sec: float,
    n_in: int,
) -> float:
    """Compute Fred (red/white noise ratio scaled by sqrt(n_in)).

    Args:
        dep_series: Depth series.
        err_series: Error on depth series.
        flux: Raw flux array.
        flux_err: Flux uncertainties.
        phase: Phase array.
        qtran: Transit duration as fraction of period.
        phs_pri: Primary phase.
        phs_sec: Secondary phase.
        n_in: Number of in-transit points.

    Returns:
        Fred value.
    """
    # Mask out primary (within 2 qtran) and secondary (within qtran)
    dist_from_pri = np.abs(phase - phs_pri)
    dist_from_pri = np.minimum(dist_from_pri, 1.0 - dist_from_pri)
    dist_from_sec = np.abs(phase - phs_sec)
    dist_from_sec = np.minimum(dist_from_sec, 1.0 - dist_from_sec)

    mask_pri = dist_from_pri < 2 * qtran
    mask_sec = dist_from_sec < qtran
    non_pri_sec = ~mask_pri & ~mask_sec & np.isfinite(dep_series)

    if np.sum(non_pri_sec) < 3:
        return np.nan

    # Red noise = scatter of depth measurements
    red_noise = _weighted_std(dep_series[non_pri_sec], err_series[non_pri_sec])

    # White noise = scatter of raw flux
    white_noise = _weighted_std(flux[non_pri_sec], flux_err[non_pri_sec])

    if white_noise <= 0 or not np.isfinite(white_noise):
        return np.nan

    # Fred: red/white scaled by sqrt(n_in)
    fred = float(np.sqrt(n_in) * red_noise / white_noise)

    return fred


def _compute_chases(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    phase: np.ndarray,
    period: float,
    epoch: float,
    qtran: float,
    frac: float = 0.5,
) -> float:
    """Compute CHASES metric (local event uniqueness).

    For each transit epoch, measures distance to nearest significant nearby event.

    Args:
        time: Time array.
        flux: Flux array.
        flux_err: Flux uncertainty array.
        phase: Phase array.
        period: Orbital period.
        epoch: Transit epoch.
        qtran: Transit duration as fraction of period.
        frac: Fraction threshold for significant event.

    Returns:
        Median CHASES value (0-1, higher is better).
    """
    n = len(time)
    if n < 10:
        return np.nan

    # Identify transit epochs
    epochs = np.round((time - epoch) / period)
    unique_epochs = np.unique(epochs)
    n_transits = len(unique_epochs)

    if n_transits < 3:
        return np.nan

    # Compute per-point SES (single event statistic)
    # Simplified: use (zpt - flux) / flux_err as proxy
    out_tran = np.abs(phase) > qtran
    if np.sum(out_tran) < 3:
        return np.nan

    zpt = _weighted_mean(flux[out_tran], flux_err[out_tran])
    ses = (zpt - flux) / flux_err

    chases_values = []

    for ep in unique_epochs:
        # Points in this epoch's transit window
        in_epoch = epochs == ep
        in_tran_epoch = in_epoch & (np.abs(phase) < qtran)

        if not np.any(in_tran_epoch):
            continue

        # Get the transit time and SES for this epoch
        transit_time = np.mean(time[in_tran_epoch])
        transit_ses = np.mean(ses[in_tran_epoch])

        if not np.isfinite(transit_ses) or transit_ses <= 0:
            chases_values.append(1.0)
            continue

        # Look for nearby significant events (1-10% of period away)
        in_epoch_all = epochs == ep
        nearby = in_epoch_all & (np.abs(phase) > qtran) & (np.abs(phase) < 0.1)
        significant = nearby & (np.abs(ses) > frac * transit_ses)

        if np.any(significant):
            distances = np.abs(time[significant] - transit_time)
            min_dist = np.min(distances)
            # Normalize by 10% of period
            chases_values.append(min_dist / (0.1 * period))
        else:
            chases_values.append(1.0)  # No nearby event = good

    if len(chases_values) == 0:
        return np.nan

    return float(np.median(chases_values))


def _compute_chi(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    phase: np.ndarray,
    period: float,
    epoch: float,
    qtran: float,
    overall_mes: float,
) -> float:
    """Compute CHI metric (transit consistency).

    Measures whether individual transit depths are consistent with expected.

    Args:
        time: Time array.
        flux: Flux array.
        flux_err: Flux uncertainty array.
        phase: Phase array.
        period: Orbital period.
        epoch: Transit epoch.
        qtran: Transit duration as fraction of period.
        overall_mes: Overall MES from primary signal.

    Returns:
        CHI value (higher is better for planets).
    """
    # Identify transit epochs
    epochs = np.round((time - epoch) / period)
    unique_epochs = np.unique(epochs)
    n_transits = len(unique_epochs)

    if n_transits < 3:
        return np.nan

    # Compute per-transit SES
    out_tran = np.abs(phase) > qtran
    if np.sum(out_tran) < 3:
        return np.nan

    zpt = _weighted_mean(flux[out_tran], flux_err[out_tran])

    ses_values = []
    expected_ses_values = []

    for ep in unique_epochs:
        in_tran_epoch = (epochs == ep) & (np.abs(phase) < qtran)
        if np.sum(in_tran_epoch) < 2:
            continue

        ep_flux = flux[in_tran_epoch]
        ep_err = flux_err[in_tran_epoch]

        # Observed depth and error for this transit
        ep_depth = zpt - _weighted_mean(ep_flux, ep_err)
        w = 1.0 / (ep_err**2)
        ep_depth_err = 1.0 / np.sqrt(np.sum(w))

        if ep_depth_err <= 0 or not np.isfinite(ep_depth_err):
            continue

        # Observed SES
        ses_obs = ep_depth / ep_depth_err

        # Expected SES based on overall depth
        # (simplified: assume constant depth across transits)
        ses_exp = overall_mes / np.sqrt(n_transits)

        if np.isfinite(ses_obs) and np.isfinite(ses_exp) and ses_exp > 0:
            ses_values.append(ses_obs)
            expected_ses_values.append(ses_exp)

    if len(ses_values) < 3:
        return np.nan

    ses_obs_arr = np.array(ses_values)
    ses_exp_arr = np.array(expected_ses_values)

    # Chi-squared statistic
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.sum((ses_obs_arr - ses_exp_arr) ** 2 / np.abs(ses_exp_arr))

    n_dof = len(ses_values) - 1
    if n_dof < 1 or chi2 <= 0:
        return np.nan

    # CHI = MES / sqrt(chi2 / (N-1))
    chi = overall_mes / np.sqrt(chi2 / n_dof)

    return float(chi)


def run_modshift_uniqueness(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
    n_tce: int = 20000,
) -> dict[str, Any]:
    """Compute ModShift signal uniqueness metrics.

    Independent implementation based on published algorithms from Thompson et al.
    (2018) and Coughlin et al. (2016). Computes properly-scaled Fred and MS1-MS6.

    Args:
        time: Time array in days (BTJD).
        flux: Normalized flux array.
        flux_err: Flux uncertainty array.
        period: Orbital period in days.
        t0: Transit epoch in days (BTJD).
        duration_hours: Transit duration in hours.
        n_tce: Number of TCEs for false alarm threshold calculation.

    Returns:
        Dictionary containing:
        - Signal significances: sig_pri, sig_sec, sig_ter, sig_pos
        - Fred and thresholds: fred, fa1, fa2
        - MS metrics: ms1, ms2, ms3, ms4, ms5, ms6
        - Bonus metrics: med_chases, chi
        - Provenance: n_in, n_out, n_transits
        - Status: status ("ok" or "error"), warnings list
    """
    warnings: list[str] = []

    # Input validation
    if len(time) < 10:
        return {
            "sig_pri": np.nan,
            "sig_sec": np.nan,
            "sig_ter": np.nan,
            "sig_pos": np.nan,
            "fred": np.nan,
            "fa1": np.nan,
            "fa2": np.nan,
            "ms1": np.nan,
            "ms2": np.nan,
            "ms3": np.nan,
            "ms4": np.nan,
            "ms5": np.nan,
            "ms6": np.nan,
            "med_chases": np.nan,
            "chi": np.nan,
            "n_in": 0,
            "n_out": 0,
            "n_transits": 0,
            "status": "error",
            "warnings": ["INSUFFICIENT_DATA"],
        }

    # Ensure arrays are numpy arrays and handle NaN/Inf
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64)

    # Filter out invalid points
    valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    valid &= flux_err > 0

    if np.sum(valid) < 10:
        return {
            "sig_pri": np.nan,
            "sig_sec": np.nan,
            "sig_ter": np.nan,
            "sig_pos": np.nan,
            "fred": np.nan,
            "fa1": np.nan,
            "fa2": np.nan,
            "ms1": np.nan,
            "ms2": np.nan,
            "ms3": np.nan,
            "ms4": np.nan,
            "ms5": np.nan,
            "ms6": np.nan,
            "med_chases": np.nan,
            "chi": np.nan,
            "n_in": 0,
            "n_out": 0,
            "n_transits": 0,
            "status": "error",
            "warnings": ["INSUFFICIENT_VALID_DATA"],
        }

    time = time[valid]
    flux = flux[valid]
    flux_err = flux_err[valid]

    # Transit duration as fraction of period
    duration_days = duration_hours / 24.0
    qtran = duration_days / period

    if qtran >= 0.5:
        warnings.append("DURATION_TOO_LONG")
        qtran = 0.49  # Clamp to avoid issues

    # Count transits
    epochs = np.round((time - t0) / period)
    n_transits = len(np.unique(epochs))

    if n_transits < 3:
        warnings.append("FEW_TRANSITS")

    # Compute phase
    phase = _phasefold(time, period, t0)

    # Compute MES series using binned method
    mes_series, dep_series, err_series, _zpt, n_in = _compute_mes_series_binned(
        time, flux, flux_err, period, t0, qtran
    )

    n_out = len(time) - n_in

    if n_out < 3:
        warnings.append("INSUFFICIENT_OUT_OF_TRANSIT")
        return {
            "sig_pri": np.nan,
            "sig_sec": np.nan,
            "sig_ter": np.nan,
            "sig_pos": np.nan,
            "fred": np.nan,
            "fa1": np.nan,
            "fa2": np.nan,
            "ms1": np.nan,
            "ms2": np.nan,
            "ms3": np.nan,
            "ms4": np.nan,
            "ms5": np.nan,
            "ms6": np.nan,
            "med_chases": np.nan,
            "chi": np.nan,
            "n_in": n_in,
            "n_out": n_out,
            "n_transits": n_transits,
            "status": "error",
            "warnings": warnings,
        }

    # Find signal peaks
    sig_pri, sig_sec, sig_ter, sig_pos, phs_pri, phs_sec = _find_signal_peaks(
        mes_series, phase, qtran
    )

    # Compute Fred
    fred = _compute_fred(
        dep_series,
        err_series,
        flux,
        flux_err,
        phase,
        qtran,
        phs_pri,
        phs_sec,
        n_in,
    )

    # Compute false alarm thresholds
    fa1, fa2 = _compute_false_alarm_thresholds(qtran, n_tce)

    # Compute MS metrics
    # MS1: Primary uniqueness (Fred-normalized)
    if np.isfinite(fred) and fred > 0:
        ms1 = sig_pri / fred - fa1
    else:
        ms1 = np.nan
        warnings.append("FRED_INVALID")

    # MS2: Primary vs tertiary
    ms2 = sig_pri - sig_ter - fa2

    # MS3: Primary vs positive
    ms3 = sig_pri - sig_pos - fa2

    # MS4: Secondary significance (Fred-normalized)
    ms4 = sig_sec / fred - fa1 if np.isfinite(fred) and fred > 0 else np.nan

    # MS5: Secondary vs tertiary
    ms5 = sig_sec - sig_ter - fa2

    # MS6: Secondary vs positive
    ms6 = sig_sec - sig_pos - fa2

    # Compute CHASES
    med_chases = _compute_chases(time, flux, flux_err, phase, period, t0, qtran)

    # Compute CHI
    chi = _compute_chi(time, flux, flux_err, phase, period, t0, qtran, sig_pri)

    # Determine overall status
    status = "ok"
    if not np.isfinite(sig_pri):
        status = "error"
        warnings.append("PRIMARY_SIGNAL_INVALID")

    return {
        "sig_pri": float(sig_pri) if np.isfinite(sig_pri) else np.nan,
        "sig_sec": float(sig_sec) if np.isfinite(sig_sec) else np.nan,
        "sig_ter": float(sig_ter) if np.isfinite(sig_ter) else np.nan,
        "sig_pos": float(sig_pos) if np.isfinite(sig_pos) else np.nan,
        "fred": float(fred) if np.isfinite(fred) else np.nan,
        "fa1": float(fa1) if np.isfinite(fa1) else np.nan,
        "fa2": float(fa2) if np.isfinite(fa2) else np.nan,
        "ms1": float(ms1) if np.isfinite(ms1) else np.nan,
        "ms2": float(ms2) if np.isfinite(ms2) else np.nan,
        "ms3": float(ms3) if np.isfinite(ms3) else np.nan,
        "ms4": float(ms4) if np.isfinite(ms4) else np.nan,
        "ms5": float(ms5) if np.isfinite(ms5) else np.nan,
        "ms6": float(ms6) if np.isfinite(ms6) else np.nan,
        "med_chases": float(med_chases) if np.isfinite(med_chases) else np.nan,
        "chi": float(chi) if np.isfinite(chi) else np.nan,
        "n_in": n_in,
        "n_out": n_out,
        "n_transits": n_transits,
        "status": status,
        "warnings": warnings,
    }


__all__ = ["run_modshift_uniqueness"]
