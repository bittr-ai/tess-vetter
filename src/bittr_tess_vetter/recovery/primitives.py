"""Recovery primitives for active star transit detection.

Pure-compute functions for:
- Rotation period estimation (Lomb-Scargle)
- Stellar variability removal (harmonic subtraction)
- Transit stacking (phase-fold and bin)
- Trapezoid fitting (least-squares optimization)
- Detection significance computation

All functions use pure numpy/scipy with no I/O operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import signal
from scipy.optimize import minimize

from bittr_tess_vetter.recovery.result import StackedTransit, TrapezoidFit

if TYPE_CHECKING:
    from numpy.typing import NDArray


def estimate_rotation_period(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    min_period: float = 0.5,
    max_period: float = 30.0,
    known_period: float | None = None,
    n_periods: int = 2000,
) -> tuple[float, float]:
    """Estimate stellar rotation period using Lomb-Scargle periodogram.

    Args:
        time: Time array, in days (e.g., BTJD)
        flux: Normalized flux array (median ~1.0)
        min_period: Minimum period to search, in days
        max_period: Maximum period to search, in days
        known_period: Optional known rotation period for validation, in days
        n_periods: Number of period grid points

    Returns:
        Tuple of (best_period, snr) - period in days and detection SNR.
        SNR < 3 indicates no significant detection.
    """
    # Generate period grid (log-spaced for efficiency)
    periods = np.logspace(np.log10(min_period), np.log10(max_period), n_periods, dtype=np.float64)

    # Center flux around zero for Lomb-Scargle
    flux_centered = flux - np.mean(flux)

    # Convert periods to angular frequencies
    angular_frequencies = 2.0 * np.pi / periods

    # Compute Lomb-Scargle periodogram
    power = signal.lombscargle(
        time,
        flux_centered,
        angular_frequencies,
        normalize=True,
    )

    # Find best period
    best_idx = int(np.argmax(power))
    best_period = float(periods[best_idx])

    # If known_period provided, check if detected period is consistent
    if known_period is not None:
        tolerance = 0.1 * known_period  # 10% tolerance
        if abs(best_period - known_period) > tolerance:
            # Check harmonics (rotation signal may appear at half or double)
            for harmonic in [0.5, 2.0]:
                if abs(best_period - known_period * harmonic) < tolerance:
                    best_period = known_period
                    break
            else:
                # If harmonics don't match either, use known period
                # if the power at known period is reasonable
                known_idx = int(np.argmin(np.abs(periods - known_period)))
                if power[known_idx] > 0.3 * power[best_idx]:
                    best_period = known_period

    # Compute SNR using MAD-based noise estimate
    snr = _estimate_snr(power[best_idx], power)

    return best_period, snr


def _estimate_snr(peak_power: float, powers: NDArray[np.float64]) -> float:
    """Estimate signal-to-noise ratio for a peak.

    Uses median absolute deviation (MAD) as robust noise estimate.

    Args:
        peak_power: Power of the peak
        powers: Full power spectrum

    Returns:
        Estimated SNR (clamped to non-negative)
    """
    median_power = float(np.median(powers))
    mad = float(np.median(np.abs(powers - median_power)))
    # MAD to standard deviation conversion factor for Gaussian
    sigma = mad * 1.4826

    if sigma <= 0:
        # Cap at 999.0 to prevent misleading Infinity values for spurious detections
        return 999.0 if peak_power > median_power else 0.0

    snr = (peak_power - median_power) / sigma
    # Cap at 999 to prevent misleading extreme values
    return min(999.0, max(0.0, snr))


def detrend_for_recovery(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    transit_mask: NDArray[np.bool_],
    method: str = "harmonic",
    rotation_period: float | None = None,
    n_harmonics: int = 3,
    window_length: float = 0.5,
) -> NDArray[np.float64]:
    """Detrend light curve for transit recovery.

    Supports multiple detrending methods to handle different types of
    stellar variability. Choose based on the nature of the variability:

    Methods:
    - "harmonic": Fourier series subtraction (existing method).
        Best when rotation period is known and variability is quasi-sinusoidal.
        Requires rotation_period parameter.
    - "wotan_biweight": Robust biweight filter using wotan.
        Fast, good general-purpose method. Works well for most cases.
    - "wotan_gp": Gaussian process detrending using wotan.
        Slower but handles complex, non-periodic variability well.

    Args:
        time: Time array, in days
        flux: Normalized flux array (median ~1.0)
        transit_mask: Boolean mask where True = in-transit points to exclude
        method: Detrending method ("harmonic", "wotan_biweight", "wotan_gp")
        rotation_period: Stellar rotation period, in days. Required for "harmonic".
        n_harmonics: Number of harmonics to fit (1-5). Only used with "harmonic".
        window_length: Window length in days for wotan methods. Default 0.5 days.

    Returns:
        Detrended flux array (normalized, median ~1.0)

    Raises:
        ValueError: If method is "harmonic" and rotation_period is None.
        ImportError: If wotan method requested but wotan is not installed.
    """
    if method == "harmonic":
        if rotation_period is None:
            raise ValueError("rotation_period is required for harmonic detrending method")
        return remove_stellar_variability(time, flux, rotation_period, transit_mask, n_harmonics)

    elif method.startswith("wotan_"):
        from bittr_tess_vetter.compute.detrend import WOTAN_AVAILABLE, wotan_flatten

        if not WOTAN_AVAILABLE:
            raise ImportError(
                f"wotan is required for {method} detrending. Install with: pip install wotan"
            )

        # Extract wotan method name: "wotan_biweight" -> "biweight", "wotan_gp" -> "gp"
        wotan_method = method.replace("wotan_", "")

        # Map "gp" to wotan's actual GP method name if needed
        # wotan uses 'gp' for Gaussian process
        gp_kernel_size: float | None = None
        if wotan_method == "gp":
            # GP detrending typically needs a longer window and kernel_size
            window_length = max(window_length, 1.0)
            # kernel_size is the GP kernel width in days - typically 0.5-1.0 for stellar variability
            gp_kernel_size = 0.5

        # Call wotan_flatten with transit mask
        # wotan_flatten returns just the flattened flux when return_trend=False
        result = wotan_flatten(
            time,
            flux,
            window_length=window_length,
            method=wotan_method,  # type: ignore[arg-type]
            transit_mask=transit_mask,
            return_trend=False,
            kernel_size=gp_kernel_size,  # Only used for GP method
        )
        # Cast since return_trend=False returns NDArray, not tuple
        return result  # type: ignore[return-value]

    else:
        raise ValueError(
            f"Unknown detrend method: {method}. "
            "Supported methods: 'harmonic', 'wotan_biweight', 'wotan_gp'"
        )


def remove_stellar_variability(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    rotation_period: float,
    transit_mask: NDArray[np.bool_],
    n_harmonics: int = 3,
) -> NDArray[np.float64]:
    """Remove stellar variability using harmonic model.

    Fits a Fourier series to out-of-transit data and subtracts from all data:
        flux(t) = sum_{k=1}^{n_harmonics} [A_k * sin(2*pi*k*t/P) + B_k * cos(2*pi*k*t/P)]

    Args:
        time: Time array, in days
        flux: Normalized flux array (median ~1.0)
        rotation_period: Stellar rotation period, in days
        transit_mask: Boolean mask where True = in-transit points
        n_harmonics: Number of harmonics to fit (1-5)

    Returns:
        Detrended flux array (normalized, median ~1.0)
    """
    # Use only out-of-transit data for fitting
    oot_mask = ~transit_mask
    oot_time = time[oot_mask]
    oot_flux = flux[oot_mask]

    if len(oot_time) < 2 * n_harmonics + 1:
        # Not enough out-of-transit data, return original
        return flux.copy()

    # Build design matrix for out-of-transit data
    n_params = 2 * n_harmonics
    design_matrix = np.zeros((len(oot_time), n_params), dtype=np.float64)

    for k in range(1, n_harmonics + 1):
        phase = 2.0 * np.pi * k * oot_time / rotation_period
        design_matrix[:, 2 * (k - 1)] = np.sin(phase)
        design_matrix[:, 2 * (k - 1) + 1] = np.cos(phase)

    # Solve linear least squares (fit to flux - 1.0 since flux is normalized)
    coeffs, _, _, _ = np.linalg.lstsq(design_matrix, oot_flux - 1.0, rcond=None)

    # Build full design matrix for all times
    design_matrix_full = np.zeros((len(time), n_params), dtype=np.float64)
    for k in range(1, n_harmonics + 1):
        phase = 2.0 * np.pi * k * time / rotation_period
        design_matrix_full[:, 2 * (k - 1)] = np.sin(phase)
        design_matrix_full[:, 2 * (k - 1) + 1] = np.cos(phase)

    # Compute model and divide out (multiplicative detrending preserves transit depth)
    model = design_matrix_full @ coeffs + 1.0

    # Avoid division by very small values
    model = np.clip(model, 0.5, 2.0)
    detrended_flux = flux / model

    return detrended_flux


def stack_transits(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period: float,
    t0: float,
    duration_hours: float,
    phase_bins: int = 100,
    phase_min: float | None = None,
    phase_max: float | None = None,
) -> StackedTransit:
    """Phase-fold and bin transits for improved SNR.

    Args:
        time: Time array, in days
        flux: Detrended flux array (normalized)
        flux_err: Flux uncertainties
        period: Orbital period, in days
        t0: Reference transit epoch, in days (e.g., BTJD)
        duration_hours: Transit duration, in hours
        phase_bins: Number of phase bins for stacking
        phase_min: Minimum phase for binning (0-1). If None, uses a transit-centered window.
        phase_max: Maximum phase for binning (0-1). If None, uses a transit-centered window.

    Returns:
        StackedTransit with binned phase, flux, errors, and transit count
    """
    duration_days = duration_hours / 24.0

    # Calculate phase (0 to 1)
    phase = ((time - t0) / period) % 1.0

    # Shift to center transit at phase 0.5
    phase = (phase + 0.5) % 1.0

    # Define phase bins.
    # Default: transit-centered window (~3x transit duration) to maximize SNR for recovery.
    if phase_min is None or phase_max is None:
        transit_phase_range = 3.0 * duration_days / period
        phase_min_eff = max(0.35, 0.5 - transit_phase_range)
        phase_max_eff = min(0.65, 0.5 + transit_phase_range)
    else:
        phase_min_eff = float(phase_min)
        phase_max_eff = float(phase_max)

    bin_edges = np.linspace(phase_min_eff, phase_max_eff, phase_bins + 1, dtype=np.float64)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Bin the data using inverse-variance weighting
    binned_flux = np.zeros(phase_bins, dtype=np.float64)
    binned_flux_err = np.zeros(phase_bins, dtype=np.float64)
    n_points = np.zeros(phase_bins, dtype=np.int32)

    # Prevent division by zero in weights
    safe_flux_err = np.maximum(flux_err, 1e-10)

    for i in range(phase_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        n_in_bin = int(np.sum(mask))

        if n_in_bin > 0:
            weights = 1.0 / safe_flux_err[mask] ** 2
            total_weight = np.sum(weights)
            binned_flux[i] = float(np.sum(flux[mask] * weights) / total_weight)
            binned_flux_err[i] = float(1.0 / np.sqrt(total_weight))
            n_points[i] = n_in_bin
        else:
            binned_flux[i] = 1.0  # Default to normalized baseline
            binned_flux_err[i] = 1.0  # Large uncertainty
            n_points[i] = 0

    # Count distinct transits
    transit_numbers = np.floor((time - t0) / period).astype(int)
    n_transits = len(np.unique(transit_numbers))

    return StackedTransit(
        phase=bin_centers,
        flux=binned_flux,
        flux_err=binned_flux_err,
        n_points_per_bin=n_points,
        n_transits=n_transits,
    )


def _trapezoid_model(
    phase: NDArray[np.float64],
    depth: float,
    duration: float,
    ingress_ratio: float,
    t_center: float = 0.5,
) -> NDArray[np.float64]:
    """Compute trapezoid transit model.

    Args:
        phase: Phase values (0 to 1, transit at 0.5)
        depth: Fractional transit depth
        duration: Total transit duration in phase units
        ingress_ratio: Ingress/egress as fraction of duration
        t_center: Phase of transit center

    Returns:
        Model flux array
    """
    # Phase from transit center
    dp = np.abs(phase - t_center)

    # Calculate ingress/egress and flat bottom durations
    half_duration = duration / 2.0
    ingress_duration = half_duration * ingress_ratio
    flat_duration = half_duration - ingress_duration

    model = np.ones_like(phase)

    # In flat bottom
    in_flat = dp < flat_duration
    model[in_flat] = 1.0 - depth

    # In ingress/egress
    in_transition = (dp >= flat_duration) & (dp < half_duration)
    if np.any(in_transition):
        frac = (dp[in_transition] - flat_duration) / max(ingress_duration, 1e-10)
        model[in_transition] = 1.0 - depth * (1.0 - frac)

    return model


def fit_trapezoid(
    phase: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    initial_depth: float = 0.01,
    initial_duration_phase: float = 0.02,
    duration_constraint_factor: float = 2.0,
) -> TrapezoidFit:
    """Fit trapezoid transit model to stacked light curve.

    Uses L-BFGS-B optimization to minimize chi-squared.

    Args:
        phase: Phase values (0 to 1, transit at 0.5)
        flux: Binned flux values
        flux_err: Binned flux uncertainties
        initial_depth: Initial depth estimate
        initial_duration_phase: Initial duration in phase units
        duration_constraint_factor: Constrains duration to be within this factor
            of the initial value (e.g., 2.0 means duration can vary from
            initial/2 to initial*2). Set to 0 to use unconstrained bounds.

    Returns:
        TrapezoidFit with best-fit parameters and uncertainties
    """
    # Filter out bins with no data (flux_err = 1.0 indicates empty bin)
    valid = flux_err < 0.99
    if np.sum(valid) < 5:
        # Not enough valid bins
        return TrapezoidFit(
            depth=0.0,
            depth_err=1.0,
            duration_phase=initial_duration_phase,
            ingress_ratio=0.2,
            chi2=float("inf"),
            reduced_chi2=float("inf"),
            converged=False,
        )

    phase_valid = phase[valid]
    flux_valid = flux[valid]
    flux_err_valid = flux_err[valid]

    def chi_squared(params: NDArray[np.float64]) -> float:
        depth, duration, ingress_ratio = params
        model = _trapezoid_model(phase_valid, depth, duration, ingress_ratio)
        residuals = (flux_valid - model) / flux_err_valid
        return float(np.sum(residuals**2))

    # Initial parameters
    x0 = np.array([initial_depth, initial_duration_phase, 0.2], dtype=np.float64)

    # Constrain duration to be within factor of initial value
    # This prevents the optimizer from fitting unrealistically wide transits
    # when there's residual stellar variability in the stacked data
    if duration_constraint_factor > 0:
        min_dur = max(0.005, initial_duration_phase / duration_constraint_factor)
        max_dur = min(0.2, initial_duration_phase * duration_constraint_factor)
    else:
        # Unconstrained mode (original behavior)
        min_dur = 0.005
        max_dur = 0.2

    # Bounds: depth (1e-5 to 0.5), duration (constrained), ingress_ratio (0.05 to 0.45)
    bounds = [(1e-5, 0.5), (min_dur, max_dur), (0.05, 0.45)]

    result = minimize(chi_squared, x0, bounds=bounds, method="L-BFGS-B")

    depth, duration, ingress_ratio = result.x
    chi2 = float(result.fun)
    dof = int(np.sum(valid)) - 3  # 3 free parameters

    # Estimate depth uncertainty from Hessian approximation
    # Perturb depth and measure chi2 change
    depth_err = _estimate_depth_error(
        phase_valid, flux_valid, flux_err_valid, depth, duration, ingress_ratio
    )

    reduced_chi2 = chi2 / max(dof, 1)

    return TrapezoidFit(
        depth=float(depth),
        depth_err=depth_err,
        duration_phase=float(duration),
        ingress_ratio=float(ingress_ratio),
        chi2=chi2,
        reduced_chi2=reduced_chi2,
        converged=bool(result.success),
    )


def _estimate_depth_error(
    phase: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    depth: float,
    duration: float,
    ingress_ratio: float,
    delta: float = 0.001,
) -> float:
    """Estimate depth uncertainty from curvature of chi-squared.

    Args:
        phase: Phase values
        flux: Flux values
        flux_err: Flux uncertainties
        depth: Best-fit depth
        duration: Best-fit duration
        ingress_ratio: Best-fit ingress ratio
        delta: Perturbation for numerical derivative

    Returns:
        Estimated 1-sigma uncertainty on depth
    """

    def chi2_for_depth(d: float) -> float:
        model = _trapezoid_model(phase, d, duration, ingress_ratio)
        residuals = (flux - model) / flux_err
        return float(np.sum(residuals**2))

    # Compute second derivative (curvature) at best-fit depth
    chi2_center = chi2_for_depth(depth)
    chi2_plus = chi2_for_depth(depth + delta)
    chi2_minus = chi2_for_depth(depth - delta)

    # Second derivative: d2chi2/ddepth2 = (chi2+ - 2*chi2_0 + chi2-) / delta^2
    curvature = (chi2_plus - 2 * chi2_center + chi2_minus) / (delta**2)

    if curvature <= 0:
        # Curvature should be positive at minimum; return large uncertainty
        return abs(depth) * 0.5

    # sigma = 1/sqrt(curvature/2) for chi2 = chi2_min + 1
    depth_err = float(np.sqrt(2.0 / curvature))

    return depth_err


def compute_detection_snr(depth: float, depth_err: float) -> float:
    """Compute detection signal-to-noise ratio.

    Args:
        depth: Measured transit depth
        depth_err: Depth uncertainty

    Returns:
        SNR = depth / depth_err (capped at 999.0)
    """
    if depth_err <= 0:
        # Cap at 999.0 to prevent misleading Infinity values for spurious detections
        return 999.0 if depth > 0 else 0.0
    # Cap at 999 to prevent misleading extreme values
    return min(999.0, depth / depth_err)


def count_transits(
    time: NDArray[np.float64],
    period: float,
    t0: float,
) -> int:
    """Count the number of distinct transits in the data.

    Args:
        time: Time array, in days
        period: Orbital period, in days
        t0: Reference transit epoch, in days (e.g., BTJD)

    Returns:
        Number of distinct transit epochs covered by data
    """
    transit_numbers = np.floor((time - t0) / period).astype(int)
    return len(np.unique(transit_numbers))
