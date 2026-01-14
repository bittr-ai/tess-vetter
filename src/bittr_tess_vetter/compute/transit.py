"""Transit detection and measurement compute operations.

This module provides pure numpy-based functions for transit analysis:
- detect_transit: Fit box model and measure transit parameters
- measure_depth: Calculate transit depth from in/out of transit flux
- get_transit_mask: Create boolean mask for in-transit points
- fold_transit: Phase-fold light curve around transit

All functions are pure compute operations safe for sandbox execution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from bittr_tess_vetter.domain.detection import TransitCandidate

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Maximum reasonable SNR to prevent overflow issues downstream
MAX_SNR = 1e6


def get_transit_mask(
    time: NDArray[np.float64],
    period: float,
    t0: float,
    duration_hours: float,
) -> NDArray[np.bool_]:
    """Create boolean mask for points in transit.

    Identifies data points that fall within the transit window,
    accounting for periodicity.

    Args:
        time: Time array in BTJD (float64)
        period: Orbital period in days
        t0: Reference epoch (mid-transit time) in BTJD
        duration_hours: Transit duration in hours

    Returns:
        Boolean array where True indicates in-transit points

    Example:
        >>> time = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        >>> mask = get_transit_mask(time, period=1.0, t0=0.5, duration_hours=4.8)
        >>> mask  # True at t=0.5 and t=1.5
    """
    duration_days = float(duration_hours) / 24.0
    # Calculate phase relative to t0 (range -0.5 to 0.5)
    phase = ((time - t0) / period + 0.5) % 1.0 - 0.5

    # Transit occurs when |phase| < duration/(2*period)
    half_duration_phase = duration_days / (2.0 * period)

    return np.abs(phase) < half_duration_phase


def measure_depth(
    flux: NDArray[np.float64],
    in_transit_mask: NDArray[np.bool_],
) -> tuple[float, float]:
    """Measure transit depth from in/out of transit flux.

    Calculates depth as the fractional decrease in flux during transit
    relative to out-of-transit baseline.

    Args:
        flux: Normalized flux array (float64, median ~1.0)
        in_transit_mask: Boolean mask where True = in transit

    Returns:
        Tuple of (depth, depth_err) where:
        - depth: Fractional transit depth (positive value)
        - depth_err: Uncertainty on depth

    Raises:
        ValueError: If no in-transit or out-of-transit points

    Example:
        >>> flux = np.array([1.0, 1.0, 0.99, 1.0, 1.0])
        >>> mask = np.array([False, False, True, False, False])
        >>> depth, err = measure_depth(flux, mask)
        >>> depth  # ~0.01
    """
    out_transit_mask = ~in_transit_mask

    n_in = np.sum(in_transit_mask)
    n_out = np.sum(out_transit_mask)

    if n_in == 0:
        raise ValueError("No in-transit points found")
    if n_out == 0:
        raise ValueError("No out-of-transit points found")

    # Calculate mean flux in and out of transit
    flux_in = flux[in_transit_mask]
    flux_out = flux[out_transit_mask]

    mean_in = np.mean(flux_in)
    mean_out = np.mean(flux_out)

    # Depth is the fractional decrease
    # depth = (F_out - F_in) / F_out
    depth = (mean_out - mean_in) / mean_out

    # Error propagation: standard error of the means
    std_in = np.std(flux_in, ddof=1) if n_in > 1 else 0.0
    std_out = np.std(flux_out, ddof=1) if n_out > 1 else 0.0

    sem_in = std_in / np.sqrt(n_in)
    sem_out = std_out / np.sqrt(n_out)

    # Propagate errors for depth = (mean_out - mean_in) / mean_out
    # Using partial derivatives:
    # d(depth)/d(mean_in) = -1/mean_out
    # d(depth)/d(mean_out) = mean_in/mean_out^2
    if mean_out != 0:
        depth_err = np.sqrt((sem_in / mean_out) ** 2 + (mean_in * sem_out / mean_out**2) ** 2)
    else:
        depth_err = float("nan")

    return float(depth), float(depth_err)


def fold_transit(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    period: float,
    t0: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Phase-fold light curve around transit.

    Converts time series to phase-folded representation centered on transit.

    Args:
        time: Time array in BTJD (float64)
        flux: Flux array (float64)
        period: Orbital period in days
        t0: Reference epoch (mid-transit time) in BTJD

    Returns:
        Tuple of (phase, flux_folded) where:
        - phase: Phase array in range [-0.5, 0.5] centered on transit
        - flux_folded: Flux values (same as input, just reordered with phase)

    Example:
        >>> time = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        >>> flux = np.array([1.0, 0.99, 1.0, 0.99, 1.0])
        >>> phase, flux_folded = fold_transit(time, flux, period=1.0, t0=0.5)
    """
    # Calculate phase centered on t0 (range -0.5 to 0.5)
    phase = ((time - t0) / period + 0.5) % 1.0 - 0.5

    # Sort by phase for cleaner visualization
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = flux[sort_idx]

    return phase_sorted.astype(np.float64), flux_sorted.astype(np.float64)


def detect_transit(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period: float,
    t0: float,
    duration_hours: float,
) -> TransitCandidate:
    """Detect transit by fitting box model and measuring parameters.

    Fits a simple box-shaped transit model to measure depth and calculate
    signal-to-noise ratio (SNR).

    Args:
        time: Time array in BTJD (float64)
        flux: Normalized flux array (float64, median ~1.0)
        flux_err: Flux uncertainty array (float64)
        period: Orbital period in days (from BLS)
        t0: Reference epoch (mid-transit time) in BTJD (from BLS)
        duration_hours: Transit duration in hours

    Returns:
        TransitCandidate with measured parameters

    Raises:
        ValueError: If insufficient data points or invalid parameters

    Example:
        >>> time = np.linspace(0, 10, 1000)
        >>> flux = np.ones_like(time)
        >>> flux[490:510] = 0.99  # Inject transit
        >>> flux_err = np.ones_like(time) * 0.001
        >>> candidate = detect_transit(time, flux, flux_err, period=5.0, t0=5.0, duration_hours=2.4)
    """
    # Validate inputs
    if len(time) < 10:
        raise ValueError("Insufficient data points for transit detection")
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    if duration_hours <= 0:
        raise ValueError(f"Duration_hours must be positive, got {duration_hours}")

    duration_days = float(duration_hours) / 24.0
    if duration_days >= period:
        raise ValueError(f"Duration_days ({duration_days}) must be less than period ({period})")

    # Get transit mask
    in_transit_mask = get_transit_mask(time, period, t0, duration_hours)

    n_in_transit = np.sum(in_transit_mask)
    if n_in_transit < 3:
        raise ValueError(f"Insufficient in-transit points ({n_in_transit}), need at least 3")

    n_out_transit = np.sum(~in_transit_mask)
    if n_out_transit < 10:
        raise ValueError(f"Insufficient out-of-transit points ({n_out_transit}), need at least 10")

    # Measure depth
    depth, depth_err = measure_depth(flux, in_transit_mask)

    # Ensure depth is positive (we're measuring a dip)
    if depth < 0:
        # Transit should be a dip, but data suggests brightening
        # This is a data quality indicator - may suggest:
        # - Wrong t0/period (phase offset)
        # - Anti-transit (secondary eclipse at wrong phase)
        # - Data quality issues
        logger.warning(
            f"Negative transit depth detected ({depth:.6f}), using absolute value. "
            "This may indicate incorrect ephemeris or data quality issues."
        )
        depth = abs(depth)

    # Calculate SNR
    # SNR = depth / noise_in_bin
    # where noise_in_bin accounts for averaging in-transit points

    # Get the typical photometric scatter
    out_transit_flux = flux[~in_transit_mask]
    scatter = np.std(out_transit_flux)

    # Number of transits in the data
    # Use np.ptp (peak-to-peak) which is safer than time[-1] - time[0]
    # as it doesn't assume sorted time array
    _ = np.ptp(time)  # data_span unused but kept for documentation

    # Total number of in-transit points
    total_in_transit = n_in_transit  # Already summed across all transits

    # Effective noise per transit
    # SNR scales as depth / (scatter / sqrt(n_in_transit_per_transit * n_transits))
    # Simplified: SNR = depth * sqrt(total_in_transit) / scatter

    # Guard against very small scatter values to prevent overflow
    min_scatter = 1e-15
    if scatter > min_scatter:
        snr = depth * np.sqrt(total_in_transit) / scatter
    elif scatter > 0:
        # Scatter is extremely small but non-zero
        logger.warning(f"Very small scatter ({scatter:.2e}) detected, capping SNR to {MAX_SNR}")
        snr = MAX_SNR if depth > 0 else 0.0
    else:
        snr = MAX_SNR if depth > 0 else 0.0

    # Ensure SNR is non-negative and capped to prevent downstream issues
    snr = max(0.0, min(snr, MAX_SNR))

    return TransitCandidate(
        period=period,
        t0=t0,
        duration_hours=duration_hours,
        depth=depth,
        snr=snr,
    )
