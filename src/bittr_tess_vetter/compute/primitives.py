"""Pure-compute astronomy primitives for sandbox injection.

This module contains ONLY numpy/scipy operations - no I/O, no network.
It is injected into the sandbox via additional_scope.

CRITICAL: This module must remain pure compute:
- NO file I/O (open, Path, etc.)
- NO network access
- NO imports of astropy, lightkurve, batman
- ONLY numpy and scipy dependencies

These functions are designed to work with pre-validated numpy arrays
that have been loaded and cleaned in the handler layer.
"""

from __future__ import annotations

import logging
import types

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal

logger = logging.getLogger(__name__)


def periodogram(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    periods: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Lomb-Scargle periodogram.

    Uses scipy.signal.lombscargle for the core computation.

    Parameters
    ----------
    time : np.ndarray
        Time array in days (BTJD format expected). Shape (n,).
    flux : np.ndarray
        Normalized flux array (median ~1.0). Shape (n,).
    periods : np.ndarray
        Period grid to evaluate in days. Shape (m,).

    Returns
    -------
    np.ndarray
        Power spectrum values at each period. Shape (m,).
        Normalized to have values between 0 and 1.

    Notes
    -----
    - The flux is mean-subtracted before computing the periodogram
    - Angular frequencies are computed as 2*pi/period
    - The power is normalized using the variance of the flux
    """
    # Validate inputs
    if len(time) != len(flux):
        raise ValueError(f"time and flux must have same length: {len(time)} vs {len(flux)}")
    if len(periods) == 0:
        return np.array([], dtype=np.float64)
    if len(time) < 3:
        raise ValueError("Need at least 3 data points for periodogram")

    # Remove NaN values
    mask = np.isfinite(time) & np.isfinite(flux)
    time_clean = time[mask]
    flux_clean = flux[mask]

    if len(time_clean) < 3:
        raise ValueError("Need at least 3 finite data points for periodogram")

    # Mean-subtract the flux
    flux_centered = flux_clean - np.mean(flux_clean)

    # If the flux has (near) zero variance, Lomb-Scargle normalization can
    # produce divide-by-zero warnings and non-finite power. Treat this as
    # "no periodic signal" and return a zero spectrum.
    flux_var = float(np.var(flux_centered))
    if not np.isfinite(flux_var) or flux_var <= 0.0:
        return np.zeros_like(periods, dtype=np.float64)

    # Validate periods are all positive to prevent overflow in angular frequency
    if np.any(periods <= 0):
        raise ValueError("All periods must be positive")

    # Guard against very small periods that could cause overflow
    min_safe_period = 1e-10  # Approximately 0.86 milliseconds
    if np.any(periods < min_safe_period):
        raise ValueError(f"Periods below {min_safe_period} days may cause numerical overflow")

    # Convert periods to angular frequencies
    # scipy.signal.lombscargle expects angular frequencies
    angular_freqs = 2.0 * np.pi / periods

    # Compute the Lomb-Scargle periodogram
    # scipy.signal.lombscargle returns unnormalized power
    power = signal.lombscargle(time_clean, flux_centered, angular_freqs, normalize=True)

    result: NDArray[np.float64] = power.astype(np.float64)
    return result


def fold(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    period: float,
    t0: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Phase-fold a light curve.

    Parameters
    ----------
    time : np.ndarray
        Time array in days. Shape (n,).
    flux : np.ndarray
        Flux array. Shape (n,).
    period : float
        Orbital period in days.
    t0 : float
        Reference epoch (time of transit center) in days.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (phase, flux) where phase is in range [0, 1).
        Both arrays have shape (n,) and are sorted by phase.

    Notes
    -----
    - Phase 0.0 corresponds to time t0
    - The output is sorted by phase for easier plotting
    """
    if period <= 0:
        raise ValueError(f"Period must be positive: {period}")
    if len(time) != len(flux):
        raise ValueError(f"time and flux must have same length: {len(time)} vs {len(flux)}")

    # Compute phase
    phase = ((time - t0) / period) % 1.0

    # Sort by phase
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = flux[sort_idx]

    return phase_sorted.astype(np.float64), flux_sorted.astype(np.float64)


def detrend(
    flux: NDArray[np.float64],
    window: int = 101,
) -> NDArray[np.float64]:
    """Simple median detrending using a sliding window.

    Removes long-term trends by dividing by a median filter.

    Parameters
    ----------
    flux : np.ndarray
        Flux array to detrend. Shape (n,).
    window : int, optional
        Window size for median filter. Must be odd. Default is 101.

    Returns
    -------
    np.ndarray
        Detrended flux array. Shape (n,).

    Notes
    -----
    - Uses scipy.ndimage.median_filter for the sliding median
    - NaN values are preserved in the output
    - The window size should be larger than transit duration
      but smaller than stellar variability timescales
    """
    if window < 1:
        raise ValueError(f"Window must be positive: {window}")

    # Ensure window is odd for symmetric filtering
    if window % 2 == 0:
        window += 1

    # Handle NaN values by replacing temporarily
    nan_mask = ~np.isfinite(flux)
    flux_filled = flux.copy()

    if np.any(nan_mask):
        # Replace NaN with local median where possible
        median_val = np.nanmedian(flux)
        flux_filled[nan_mask] = median_val

    # Apply median filter
    trend = ndimage.median_filter(flux_filled, size=window, mode="reflect")

    # Divide to detrend (multiplicative detrending)
    # Use relative tolerance for near-zero trend values to avoid numerical instability
    # A trend value near zero indicates problematic data - replace with NaN and warn
    near_zero_threshold = 1e-10
    near_zero_mask = np.abs(trend) < near_zero_threshold

    if np.any(near_zero_mask):
        logger.warning(
            f"Found {np.sum(near_zero_mask)} trend values near zero "
            f"(|trend| < {near_zero_threshold}), setting detrended values to NaN"
        )

    trend_safe = np.where(near_zero_mask, np.nan, trend)
    detrended = flux / trend_safe

    # Restore NaN values
    detrended[nan_mask] = np.nan

    return detrended.astype(np.float64)


def box_model(
    phase: NDArray[np.float64],
    depth: float,
    duration: float,
) -> NDArray[np.float64]:
    """Generate a simple box transit model.

    Creates a box-shaped transit model centered at phase 0.0.

    Parameters
    ----------
    phase : np.ndarray
        Phase array in range [0, 1). Shape (n,).
        Phase 0.0 is transit center.
    depth : float
        Transit depth as fractional flux decrease (e.g., 0.01 for 1% depth).
        Must be positive.
    duration : float
        Transit duration as fraction of orbital period (e.g., 0.02 for 2%).
        Must be positive and less than 0.5.

    Returns
    -------
    np.ndarray
        Model flux array. Shape (n,).
        Values are 1.0 out of transit and (1.0 - depth) in transit.

    Notes
    -----
    - Transit is centered at phase 0.0 (and equivalently at phase 1.0)
    - No limb darkening or ingress/egress ramps (pure box shape)
    - For more realistic models, use batman in the handler layer
    """
    if depth < 0:
        raise ValueError(f"Depth must be non-negative: {depth}")
    if duration <= 0 or duration >= 0.5:
        raise ValueError(f"Duration must be in (0, 0.5): {duration}")

    # Initialize model at unity (out of transit)
    model = np.ones_like(phase, dtype=np.float64)

    # Transit is centered at phase 0 (and wraps around at phase 1)
    half_duration = duration / 2.0

    # In-transit mask: phase near 0 or near 1 (wrapped)
    # Phase 0 is center, so transit spans [-half_duration, +half_duration]
    # Which in [0,1) phase space means [0, half_duration) and (1-half_duration, 1)
    in_transit = (phase < half_duration) | (phase > (1.0 - half_duration))

    # Apply transit depth
    model[in_transit] = 1.0 - depth

    return model


def create_primitives_namespace(**funcs: object) -> types.SimpleNamespace:
    """Create a SimpleNamespace with primitive functions.

    Uses SimpleNamespace instead of a class to pass sandbox type validation.
    This pattern allows extensions (astro, finance, etc.) to expose functions
    with dot-notation (e.g., astro.periodogram) without modifying any host application.

    Args:
        **funcs: Functions to expose in the namespace.

    Returns:
        SimpleNamespace with functions as attributes.
    """
    import types

    return types.SimpleNamespace(**funcs)


# Module-level namespace for sandbox injection
# Uses SimpleNamespace to pass sandbox type validation while preserving
# the astro.periodogram(...) API
astro = create_primitives_namespace(
    periodogram=periodogram,
    fold=fold,
    detrend=detrend,
    box_model=box_model,
)


# Keep class for backwards compatibility and type hints
class AstroPrimitives:
    """Namespace class exposing pure-compute astronomy primitives.

    Note: For sandbox injection, use `astro` (SimpleNamespace) instead.
    This class is retained for backwards compatibility and documentation.

    Available functions:
    - astro.periodogram(time, flux, periods) -> power spectrum
    - astro.fold(time, flux, period, t0) -> (phase, flux)
    - astro.detrend(flux, window) -> detrended flux
    - astro.box_model(phase, depth, duration) -> model flux
    """

    periodogram = staticmethod(periodogram)
    fold = staticmethod(fold)
    detrend = staticmethod(detrend)
    box_model = staticmethod(box_model)


__all__ = [
    "periodogram",
    "fold",
    "detrend",
    "box_model",
    "AstroPrimitives",
    "astro",
]
