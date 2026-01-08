"""Transit vetting primitives for eclipsing binary detection.

Pure-compute functions for odd/even depth comparison to detect
diluted eclipsing binaries:
- split_odd_even: Separate transits by epoch parity
- compare_odd_even_depths: Measure depth difference and significance
- compute_odd_even_result: Full odd/even analysis

All functions use pure numpy/scipy with no I/O operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bittr_tess_vetter.transit.result import OddEvenResult

if TYPE_CHECKING:
    from numpy.typing import NDArray


def split_odd_even(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period: float,
    t0: float,
    duration_hours: float,
) -> tuple[
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    int,
    int,
]:
    """Separate transit data by epoch parity (odd vs even).

    Extracts in-transit data points and separates them into odd and even
    epoch groups based on floor((t - t0) / period).

    Args:
        time: Time array, in days
        flux: Normalized flux array (median ~1.0)
        flux_err: Flux uncertainties
        period: Orbital period, in days
        t0: Reference transit epoch, in days (e.g., BTJD)
        duration_hours: Transit duration, in hours

    Returns:
        Tuple of ((odd_time, odd_flux, odd_flux_err),
                  (even_time, even_flux, even_flux_err),
                  n_odd_transits, n_even_transits)
    """
    duration_days = duration_hours / 24.0
    half_width = duration_days * 0.75  # Slightly wider to capture full transit

    # Calculate phase (0 to 1, with transit at phase 0)
    phase = ((time - t0) / period) % 1.0

    # In-transit mask (near phase 0 or 1)
    in_transit = (phase < half_width / period) | (phase > 1.0 - half_width / period)

    # Calculate epoch number for each point using round (since we're near transit center)
    # This correctly assigns points to the epoch whose transit they belong to
    epochs = np.round((time - t0) / period).astype(np.int64)

    # Separate by parity
    is_odd = (epochs % 2) != 0
    is_even = ~is_odd

    odd_mask = in_transit & is_odd
    even_mask = in_transit & is_even

    # Extract data
    odd_time = time[odd_mask]
    odd_flux = flux[odd_mask]
    odd_flux_err = flux_err[odd_mask]

    even_time = time[even_mask]
    even_flux = flux[even_mask]
    even_flux_err = flux_err[even_mask]

    # Count unique transits
    n_odd_transits = len(np.unique(epochs[odd_mask])) if np.any(odd_mask) else 0
    n_even_transits = len(np.unique(epochs[even_mask])) if np.any(even_mask) else 0

    return (
        (odd_time, odd_flux, odd_flux_err),
        (even_time, even_flux, even_flux_err),
        n_odd_transits,
        n_even_transits,
    )


def _measure_transit_depth(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period: float,
    t0: float,
    duration_hours: float,
) -> tuple[float, float, int]:
    """Measure weighted mean transit depth for a set of in-transit points.

    Uses inverse-variance weighting to combine measurements.

    Args:
        time: Time array of in-transit points, in days
        flux: Flux array of in-transit points
        flux_err: Flux errors of in-transit points
        period: Orbital period, in days
        t0: Reference epoch, in days
        duration_hours: Transit duration, in hours

    Returns:
        Tuple of (depth_ppm, depth_err_ppm, n_points)
    """
    if len(flux) < 5:
        return 0.0, float("inf"), len(flux)

    # Filter to finite values
    valid = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    if np.sum(valid) < 5:
        return 0.0, float("inf"), int(np.sum(valid))

    f = flux[valid]
    f_err = flux_err[valid]

    # Estimate baseline from out-of-transit (assuming median is ~1.0)
    # For in-transit data, we use 1.0 as the baseline
    baseline = 1.0

    # Transit depth for each point: 1 - flux
    depths = baseline - f
    weights = 1.0 / f_err**2

    # Weighted mean
    weighted_depth = float(np.sum(depths * weights) / np.sum(weights))
    weighted_err = float(1.0 / np.sqrt(np.sum(weights)))

    # Convert to ppm
    depth_ppm = weighted_depth * 1e6
    depth_err_ppm = weighted_err * 1e6

    return depth_ppm, depth_err_ppm, int(np.sum(valid))


def compare_odd_even_depths(
    odd_flux: NDArray[np.float64],
    odd_flux_err: NDArray[np.float64],
    even_flux: NDArray[np.float64],
    even_flux_err: NDArray[np.float64],
) -> tuple[float, float, float, float, float]:
    """Compare transit depths between odd and even epochs.

    Args:
        odd_flux: In-transit flux values for odd epochs
        odd_flux_err: Flux errors for odd epochs
        even_flux: In-transit flux values for even epochs
        even_flux_err: Flux errors for even epochs

    Returns:
        Tuple of (depth_odd_ppm, depth_even_ppm, depth_diff_ppm,
                  diff_err_ppm, significance_sigma)
    """
    # Measure depths for each group
    # Filter valid data
    odd_valid = np.isfinite(odd_flux) & np.isfinite(odd_flux_err) & (odd_flux_err > 0)
    even_valid = np.isfinite(even_flux) & np.isfinite(even_flux_err) & (even_flux_err > 0)

    n_odd = int(np.sum(odd_valid))
    n_even = int(np.sum(even_valid))

    if n_odd < 3 or n_even < 3:
        return 0.0, 0.0, 0.0, float("inf"), 0.0

    # Calculate weighted mean depth for odd epochs
    odd_depths = 1.0 - odd_flux[odd_valid]
    odd_weights = 1.0 / odd_flux_err[odd_valid] ** 2
    depth_odd = float(np.sum(odd_depths * odd_weights) / np.sum(odd_weights))
    err_odd = float(1.0 / np.sqrt(np.sum(odd_weights)))

    # Calculate weighted mean depth for even epochs
    even_depths = 1.0 - even_flux[even_valid]
    even_weights = 1.0 / even_flux_err[even_valid] ** 2
    depth_even = float(np.sum(even_depths * even_weights) / np.sum(even_weights))
    err_even = float(1.0 / np.sqrt(np.sum(even_weights)))

    # Convert to ppm
    depth_odd_ppm = depth_odd * 1e6
    depth_even_ppm = depth_even * 1e6

    # Compute difference and its uncertainty
    depth_diff = abs(depth_odd - depth_even)
    diff_err = np.sqrt(err_odd**2 + err_even**2)

    depth_diff_ppm = depth_diff * 1e6
    diff_err_ppm = diff_err * 1e6

    # Significance in sigma
    significance = depth_diff / diff_err if diff_err > 0 else 0.0

    return depth_odd_ppm, depth_even_ppm, depth_diff_ppm, diff_err_ppm, significance


def compute_odd_even_result(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period: float,
    t0: float,
    duration_hours: float,
    relative_threshold_percent: float = 10.0,
) -> OddEvenResult:
    """Perform full odd/even depth comparison analysis.

    Splits transits by epoch parity, measures depths, and computes
    the significance of any difference. Uses relative depth difference
    to determine if the signal is suspicious (potential eclipsing binary).

    Real eclipsing binaries show 50-100% relative depth differences between
    primary and secondary eclipses. Confirmed planets typically show <5%
    relative difference. We flag as suspicious only when the relative
    depth difference exceeds the threshold (default 10%).

    Args:
        time: Time array, in days
        flux: Normalized flux array (median ~1.0)
        flux_err: Flux uncertainties
        period: Orbital period, in days
        t0: Reference transit epoch, in days (e.g., BTJD)
        duration_hours: Transit duration, in hours
        relative_threshold_percent: Relative depth difference threshold in percent
            for flagging as suspicious (default 10.0%). Real EBs show 50-100%,
            planets show <5%.

    Returns:
        OddEvenResult with depth comparison and significance
    """
    # Split by parity
    (
        (odd_time, odd_flux, odd_flux_err),
        (
            even_time,
            even_flux,
            even_flux_err,
        ),
        n_odd,
        n_even,
    ) = split_odd_even(time, flux, flux_err, period, t0, duration_hours)

    # Handle case with insufficient data
    if n_odd < 1 or n_even < 1:
        return OddEvenResult(
            depth_odd_ppm=0.0,
            depth_even_ppm=0.0,
            depth_diff_ppm=0.0,
            relative_depth_diff_percent=0.0,
            significance_sigma=0.0,
            is_suspicious=False,
            n_odd=n_odd,
            n_even=n_even,
        )

    # Compare depths
    (
        depth_odd_ppm,
        depth_even_ppm,
        depth_diff_ppm,
        diff_err_ppm,
        significance,
    ) = compare_odd_even_depths(odd_flux, odd_flux_err, even_flux, even_flux_err)

    # Compute relative depth difference as percentage
    # Use average depth as denominator to avoid division by zero
    avg_depth_ppm = (depth_odd_ppm + depth_even_ppm) / 2.0
    if avg_depth_ppm > 0:
        relative_depth_diff_percent = (depth_diff_ppm / avg_depth_ppm) * 100.0
    else:
        relative_depth_diff_percent = 0.0

    # Flag as suspicious if RELATIVE difference exceeds threshold
    # Real EBs have 50-100% relative differences, planets have <5%
    # Using 10% threshold avoids false positives on confirmed planets
    is_suspicious = bool(relative_depth_diff_percent > relative_threshold_percent)

    return OddEvenResult(
        depth_odd_ppm=depth_odd_ppm,
        depth_even_ppm=depth_even_ppm,
        depth_diff_ppm=depth_diff_ppm,
        relative_depth_diff_percent=relative_depth_diff_percent,
        significance_sigma=significance,
        is_suspicious=is_suspicious,
        n_odd=n_odd,
        n_even=n_even,
    )
