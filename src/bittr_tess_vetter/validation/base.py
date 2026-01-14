"""Array-in/array-out primitives for vetting computations.

This module provides low-level utilities used throughout `bittr-tess-vetter`:
- phase folding
- in/out-of-transit masks
- sigma clipping
- simple depth measurement
- secondary eclipse search (as a measurement, not a policy decision)

All policy (pass/warn/reject, dispositions, guardrails) is applied by the host.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-fold a light curve.

    Args:
        time: Time array (BTJD)
        flux: Flux array
        period: Orbital period in days
        t0: Reference epoch (BTJD)

    Returns:
        Tuple of (phase, flux) where phase is in [-0.5, 0.5]
    """
    phase = ((time - t0) / period) % 1.0
    # Shift to [-0.5, 0.5] for transit at phase 0
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    return phase, flux


def get_in_transit_mask(
    time: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
    buffer_factor: float = 1.0,
) -> np.ndarray:
    """Get boolean mask for in-transit points.

    Args:
        time: Time array (BTJD)
        period: Orbital period in days
        t0: Reference epoch (BTJD)
        duration_hours: Transit duration in hours
        buffer_factor: Multiply duration by this factor (default: 1.0)

    Returns:
        Boolean mask (True for in-transit points)
    """
    time = np.asarray(time, dtype=np.float64)
    finite_time = np.isfinite(time)
    phase, _ = phase_fold(time, time, period, t0)
    half_dur = (duration_hours / 24.0 / period) * buffer_factor / 2.0
    # Keep strict inequality to preserve boundary behavior expected by existing tests.
    result: np.ndarray[tuple[Any, ...], np.dtype[Any]] = finite_time & (np.abs(phase) < half_dur)
    return result


def get_out_of_transit_mask(
    time: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
    buffer_factor: float = 2.0,
) -> np.ndarray:
    """Get boolean mask for out-of-transit points.

    Excludes points within buffer_factor * duration of transit center.

    Args:
        time: Time array (BTJD)
        period: Orbital period in days
        t0: Reference epoch (BTJD)
        duration_hours: Transit duration in hours
        buffer_factor: Points within this * duration are excluded (default: 2.0)

    Returns:
        Boolean mask (True for out-of-transit points)
    """
    time = np.asarray(time, dtype=np.float64)
    finite_time = np.isfinite(time)
    in_mask = get_in_transit_mask(time, period, t0, duration_hours, buffer_factor)
    return finite_time & (~in_mask)


def bin_phase_curve(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin a phase-folded light curve.

    Args:
        phase: Phase array (typically [-0.5, 0.5])
        flux: Flux array
        n_bins: Number of bins (default: 100)

    Returns:
        Tuple of (bin_centers, bin_means, bin_stds)
    """
    bin_edges = np.linspace(phase.min(), phase.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_means = np.zeros(n_bins)
    bin_stds = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_means[i] = np.nanmean(flux[mask])
            bin_stds[i] = np.nanstd(flux[mask])
        else:
            bin_means[i] = np.nan
            bin_stds[i] = np.nan

    return bin_centers, bin_means, bin_stds


def sigma_clip(
    values: np.ndarray,
    sigma: float = 3.0,
    max_iters: int = 5,
) -> np.ndarray:
    """Sigma-clip an array, returning mask of valid values.

    Args:
        values: Input array
        sigma: Number of standard deviations for clipping (default: 3.0)
        max_iters: Maximum iterations (default: 5)

    Returns:
        Boolean mask (True for valid/unclipped values)
    """
    mask = np.isfinite(values)

    for _ in range(max_iters):
        if not np.any(mask):
            break

        clipped = values[mask]
        med = np.median(clipped)
        std = np.std(clipped)

        if std == 0:
            break

        new_mask = mask & (np.abs(values - med) < sigma * std)

        if np.array_equal(new_mask, mask):
            break

        mask = new_mask

    result: np.ndarray[tuple[Any, ...], np.dtype[Any]] = mask
    return result


def measure_transit_depth(
    flux: np.ndarray,
    in_transit_mask: np.ndarray,
    out_of_transit_mask: np.ndarray,
) -> tuple[float, float]:
    """Measure transit depth from in/out-of-transit flux.

    Args:
        flux: Flux array
        in_transit_mask: Boolean mask for in-transit points
        out_of_transit_mask: Boolean mask for out-of-transit points

    Returns:
        Tuple of (depth, depth_uncertainty)
    """
    flux = np.asarray(flux, dtype=np.float64)
    in_flux = flux[np.asarray(in_transit_mask, dtype=bool)]
    out_flux = flux[np.asarray(out_of_transit_mask, dtype=bool)]

    in_flux = in_flux[np.isfinite(in_flux)]
    out_flux = out_flux[np.isfinite(out_flux)]

    if in_flux.size == 0 or out_flux.size == 0:
        return 0.0, 1.0

    in_median = np.nanmedian(in_flux)
    out_median = np.nanmedian(out_flux)

    depth = out_median - in_median

    # Uncertainty from scatter
    in_std = np.nanstd(in_flux) / np.sqrt(max(1, len(in_flux)))
    out_std = np.nanstd(out_flux) / np.sqrt(max(1, len(out_flux)))
    depth_err = np.sqrt(in_std**2 + out_std**2)

    return depth, depth_err


def count_transits(
    time: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
    min_points: int = 3,
) -> int:
    """Count the number of observable transits in the data.

    Args:
        time: Time array (BTJD)
        period: Orbital period in days
        t0: Reference epoch (BTJD)
        duration_hours: Transit duration in hours
        min_points: Minimum points required to count as covered (default: 3)

    Returns:
        Number of transits with sufficient data coverage
    """
    time = np.asarray(time, dtype=np.float64)
    time = time[np.isfinite(time)]
    if time.size == 0:
        return 0

    t_min, t_max = time.min(), time.max()

    # Find first transit after t_min
    n_start = int(np.ceil((t_min - t0) / period))
    n_end = int(np.floor((t_max - t0) / period))

    count = 0
    half_dur = duration_hours / 24.0 / 2.0

    for n in range(n_start, n_end + 1):
        t_mid = t0 + n * period
        t_start = t_mid - half_dur
        t_end = t_mid + half_dur

        # Check if we have data during this transit
        in_window = (time >= t_start) & (time <= t_end)
        if np.sum(in_window) >= min_points:
            count += 1

    return count


def get_odd_even_transit_indices(
    time: np.ndarray,
    period: float,
    t0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Get transit orbit numbers for each time point (odd/even).

    Args:
        time: Time array (BTJD)
        period: Orbital period in days
        t0: Reference epoch (BTJD)

    Returns:
        Tuple of (orbit_numbers, is_odd_mask) where:
        - orbit_numbers: Integer orbit number for each point
        - is_odd_mask: Boolean mask (True for odd orbit numbers)
    """
    time = np.asarray(time, dtype=np.float64)
    finite = np.isfinite(time)
    orbit_numbers = np.zeros(time.shape, dtype=int)
    orbit_numbers[finite] = np.round((time[finite] - t0) / period).astype(int)
    is_odd = (orbit_numbers % 2 == 1) & finite
    return orbit_numbers, is_odd


def search_secondary_eclipse(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
    phase_offset: float = 0.5,
) -> tuple[float, float, float]:
    """Search for secondary eclipse at a given phase offset.

    Args:
        time: Time array (BTJD)
        flux: Flux array
        period: Orbital period in days
        t0: Reference epoch (BTJD)
        duration_hours: Duration to search in hours
        phase_offset: Phase offset from primary (default: 0.5)

    Returns:
        Tuple of (depth, depth_err, snr) for secondary eclipse
    """
    # Shift t0 to secondary eclipse position
    t0_secondary = t0 + phase_offset * period

    in_mask = get_in_transit_mask(time, period, t0_secondary, duration_hours)
    out_mask = get_out_of_transit_mask(
        time, period, t0_secondary, duration_hours, buffer_factor=3.0
    )

    depth, depth_err = measure_transit_depth(flux, in_mask, out_mask)

    snr = depth / depth_err if depth_err > 0 else 0.0

    return depth, depth_err, snr
