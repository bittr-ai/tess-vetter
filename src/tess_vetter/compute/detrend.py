"""Detrending and normalization utilities for light curve processing.

This module provides pure-compute functions for light curve detrending:
- median_detrend: Rolling median filter detrending
- normalize_flux: Normalize flux to median = 1.0
- sigma_clip: Outlier rejection via sigma clipping
- flatten: Flatten long-term trends for transit search
- wotan_flatten: Transit-aware detrending using wotan library

All functions use numpy and scipy only - safe for sandbox injection.
Wotan integration provides transit-preserving detrending for active stars.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from scipy.ndimage import median_filter

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Optional wotan import for transit-aware detrending
try:
    from wotan import flatten as _wotan_flatten

    WOTAN_AVAILABLE = True
except ImportError:
    _wotan_flatten = None
    WOTAN_AVAILABLE = False

logger = logging.getLogger(__name__)

# Supported wotan methods for type hints
WotanMethod = Literal[
    "biweight",
    "median",
    "mean",
    "trim_mean",
    "huber",
    "hampel",
    "rspline",
    "hspline",
    "pspline",
    "lowess",
    "supersmoother",
    "gp",
]


def median_detrend(
    flux: NDArray[np.float64],
    window: int = 101,
) -> NDArray[np.float64]:
    """Apply rolling median filter detrending to flux data.

    Removes long-term trends by dividing out a smoothed median baseline.
    This is useful for removing instrumental systematics while preserving
    short-duration transit signals.

    Args:
        flux: Input flux array (1D, float64)
        window: Window size for median filter in data points.
            Must be odd. Default is 101 points (~3.4 hours for 2-min cadence).

    Returns:
        Detrended flux array with same shape as input.
        Values are flux / median_baseline, so median should be ~1.0.

    Raises:
        ValueError: If window is not a positive odd integer.

    Example:
        >>> flux = np.array([1.0, 1.001, 0.99, 1.002, 1.0])
        >>> detrended = median_detrend(flux, window=3)
    """
    if window < 1:
        raise ValueError(f"window must be positive, got {window}")
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")

    # Handle NaN values by interpolating before filtering
    flux_clean = flux.copy()
    nan_mask = np.isnan(flux_clean)

    if np.any(nan_mask):
        # Linear interpolation over NaN gaps
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) > 1:
            flux_clean[nan_mask] = np.interp(
                np.where(nan_mask)[0], valid_idx, flux_clean[valid_idx]
            )
        elif len(valid_idx) == 1:
            flux_clean[nan_mask] = flux_clean[valid_idx[0]]
        else:
            # All NaN - return as-is
            return flux.copy()

    # Apply median filter to get baseline
    baseline = median_filter(flux_clean, size=window, mode="reflect")

    # Avoid division by near-zero values - use relative tolerance
    near_zero_threshold = 1e-10
    near_zero_mask = np.abs(baseline) < near_zero_threshold

    if np.any(near_zero_mask):
        logger.warning(
            f"Found {np.sum(near_zero_mask)} baseline values near zero "
            f"(|baseline| < {near_zero_threshold}), setting to NaN"
        )

    baseline = np.where(near_zero_mask, np.nan, baseline)

    # Detrend by dividing out baseline
    detrended: NDArray[np.float64] = flux / baseline

    # Restore original NaN positions
    detrended[nan_mask] = np.nan

    return detrended


def normalize_flux(
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalize flux to have median = 1.0.

    Standard normalization convention for TESS light curves.
    Both flux and flux_err are divided by the median flux value.

    Args:
        flux: Input flux array (1D, float64)
        flux_err: Input flux error array (1D, float64)

    Returns:
        Tuple of (normalized_flux, normalized_flux_err) with median ~1.0

    Raises:
        ValueError: If flux and flux_err have different shapes.

    Example:
        >>> flux = np.array([1000.0, 1001.0, 999.0, 1002.0])
        >>> flux_err = np.array([10.0, 10.0, 10.0, 10.0])
        >>> norm_flux, norm_err = normalize_flux(flux, flux_err)
        >>> np.isclose(np.median(norm_flux), 1.0)
        True
    """
    if flux.shape != flux_err.shape:
        raise ValueError(
            f"flux and flux_err must have same shape, got {flux.shape} and {flux_err.shape}"
        )

    # Compute median ignoring NaN values
    median = np.nanmedian(flux)

    # Use relative tolerance for near-zero median to prevent numerical instability
    near_zero_threshold = 1e-10
    if median == 0 or np.isnan(median) or np.abs(median) < near_zero_threshold:
        # Cannot normalize - return copies
        if np.abs(median) < near_zero_threshold and not np.isnan(median):
            logger.warning(f"Cannot normalize: median flux ({median:.2e}) is near zero")
        return flux.copy(), flux_err.copy()

    norm_flux = flux / median
    norm_flux_err = flux_err / median

    return norm_flux, norm_flux_err


def sigma_clip(
    flux: NDArray[np.float64],
    sigma: float = 5.0,
) -> NDArray[np.bool_]:
    """Create a boolean mask for outlier rejection via sigma clipping.

    Uses MAD (Median Absolute Deviation) for robust scatter estimation,
    which is less sensitive to outliers than standard deviation.

    Args:
        flux: Input flux array (1D, float64)
        sigma: Number of standard deviations for clipping threshold.
            Default is 5.0 (conservative for transit signals).

    Returns:
        Boolean mask where True indicates valid (non-outlier) points.
        Shape matches input flux.

    Raises:
        ValueError: If sigma is not positive.

    Example:
        >>> flux = np.array([1.0, 1.001, 10.0, 1.002, 1.0])  # 10.0 is outlier
        >>> mask = sigma_clip(flux, sigma=3.0)
        >>> mask
        array([ True,  True, False,  True,  True])
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    # Compute robust statistics ignoring NaN
    median = np.nanmedian(flux)

    # Use MAD (Median Absolute Deviation) for robust scatter estimation
    # MAD * 1.4826 approximates standard deviation for normal distributions
    mad = np.nanmedian(np.abs(flux - median))
    robust_std = mad * 1.4826

    if robust_std == 0 or np.isnan(robust_std):
        # No variation - all points are valid (or all NaN)
        return ~np.isnan(flux)

    # Compute deviation from median
    deviation = np.abs(flux - median)

    # Points within sigma * robust_std are valid
    valid_mask = deviation <= (sigma * robust_std)

    # NaN values are not valid
    valid_mask = valid_mask & ~np.isnan(flux)

    return np.asarray(valid_mask, dtype=np.bool_)


def bin_median_trend(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    *,
    bin_hours: float,
    min_bin_points: int = 10,
) -> NDArray[np.float64]:
    """Compute a piecewise median trend by time binning.

    This is a lightweight detrending primitive: partition the time series into
    bins of fixed width (in hours), compute the median flux per bin, and then
    linearly interpolate the bin medians back onto the original time grid.

    Args:
        time: Time array in days (BTJD, float64)
        flux: Flux array (float64)
        bin_hours: Bin width in hours (must be > 0)
        min_bin_points: Minimum points required per bin to accept its median.
            Bins with fewer points are ignored.

    Returns:
        Trend array (float64) with same shape as input.

    Raises:
        ValueError: If bin_hours <= 0, or if there are insufficient populated bins.
    """
    if float(bin_hours) <= 0:
        raise ValueError("bin_hours must be > 0")
    if time.shape != flux.shape:
        raise ValueError(f"time and flux must have same shape, got {time.shape} and {flux.shape}")

    bin_width_days = float(bin_hours) / 24.0
    t0 = float(np.nanmin(time))
    bin_index = np.floor((time - t0) / bin_width_days).astype(int)
    n_bins = int(np.nanmax(bin_index)) + 1
    if n_bins < 3:
        raise ValueError("Not enough bins for detrending")

    medians = np.full(n_bins, np.nan, dtype=np.float64)
    centers = np.full(n_bins, np.nan, dtype=np.float64)
    for b in range(n_bins):
        m = bin_index == b
        if int(np.sum(m)) < int(min_bin_points):
            continue
        medians[b] = float(np.nanmedian(flux[m]))
        centers[b] = float(np.nanmedian(time[m]))

    ok = np.isfinite(medians) & np.isfinite(centers)
    if int(np.sum(ok)) < 3:
        raise ValueError("Too few populated bins for detrending")

    trend = np.interp(time, centers[ok], medians[ok])
    trend = np.where(np.isfinite(trend), trend, float(np.nanmedian(flux)))
    return trend.astype(np.float64)


def flatten(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    window_length: float = 0.5,
) -> NDArray[np.float64]:
    """Flatten long-term trends in light curve for transit search.

    Uses a running median filter with window size specified in days.
    This is similar to median_detrend but with time-based windowing,
    making it more appropriate when cadence varies or for specifying
    physically meaningful detrending timescales.

    Args:
        time: Time array in days (BTJD, float64)
        flux: Flux array (float64)
        window_length: Window length in days for the smoothing filter.
            Default is 0.5 days (12 hours), good for preserving transits
            while removing longer-term stellar variability.

    Returns:
        Flattened flux array with same shape as input.
        Values are flux / trend, so median should be ~1.0.

    Raises:
        ValueError: If time and flux have different shapes.
        ValueError: If window_length is not positive.

    Example:
        >>> time = np.linspace(0, 10, 1000)  # 10 days of data
        >>> flux = 1.0 + 0.01 * np.sin(2 * np.pi * time / 5)  # 5-day variation
        >>> flat = flatten(time, flux, window_length=1.0)
    """
    if time.shape != flux.shape:
        raise ValueError(f"time and flux must have same shape, got {time.shape} and {flux.shape}")
    if window_length <= 0:
        raise ValueError(f"window_length must be positive, got {window_length}")

    if len(time) == 0:
        return flux.copy()

    # Validate time array is sorted in ascending order
    if len(time) > 1:
        time_diffs = np.diff(time)
        if np.any(time_diffs < 0):
            raise ValueError(
                "Time array must be sorted in ascending order. "
                f"Found {np.sum(time_diffs < 0)} negative time differences."
            )

    # Estimate typical cadence from time array
    # Default to 2-min cadence
    default_dt: float = 1.0 / (24.0 * 30.0)
    if len(time) > 1:
        # Use median of time differences to handle gaps
        dt: float = float(np.median(np.diff(time)))
        if dt <= 0:
            # This should not happen after the sorting check, but keep as safety
            logger.warning("Median time difference is <= 0, defaulting to 2-min cadence")
            dt = default_dt
    else:
        dt = default_dt

    # Convert window_length from days to number of points
    window_points = int(np.ceil(window_length / dt))

    # Ensure window is at least 3 and odd
    window_points = max(3, window_points)
    if window_points % 2 == 0:
        window_points += 1

    # Use median_detrend with the computed window size
    return median_detrend(flux, window=window_points)


def wotan_flatten(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    window_length: float = 0.5,
    method: WotanMethod = "biweight",
    transit_mask: NDArray[np.bool_] | None = None,
    break_tolerance: float = 0.5,
    edge_cutoff: float = 0.0,
    return_trend: bool = False,
    kernel_size: float | None = None,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Flatten light curve using wotan's robust detrending methods.

    Wotan provides transit-preserving detrending that is critical for
    active stars where standard median filtering can distort transit signals.
    The transit_mask parameter allows excluding known transit times from
    the trend calculation.

    Args:
        time: Time array in days (BTJD, float64)
        flux: Flux array (float64), should be normalized (median ~1.0)
        window_length: Window length in days for the smoothing filter.
            Default is 0.5 days (12 hours). Choose larger than expected
            transit duration to preserve transit shape.
        method: Detrending method. Options:
            - "biweight" (default): Robust, good for most cases
            - "median": Simple, fast, less robust
            - "huber": Robust M-estimator
            - "hampel": Very robust to outliers
            - "trim_mean": Trimmed mean (moderate robustness)
            - "rspline", "hspline", "pspline": Spline methods
            - "lowess": Locally weighted regression
            - "supersmoother": Friedman's supersmoother
        transit_mask: Boolean array where True = in-transit point to mask.
            These points are excluded from the trend fit but included in
            the flattened output. Critical for preserving transit depth.
        break_tolerance: Maximum gap (days) before treating as segment break.
            Data gaps larger than this trigger independent trend fitting
            per segment. Default 0.5 days handles TESS orbit gaps.
        edge_cutoff: Time (days) to trim from segment edges where trend
            estimates are unreliable. Default 0.0 (no trimming).
        return_trend: If True, return (flattened_flux, trend) tuple.
            Default False returns only flattened flux.
        kernel_size: Length scale for Gaussian Process kernel (days).
            Only used when method="gp". Default None uses wotan's default.
            Typical values: 0.5-1.0 days for stellar variability.

    Returns:
        If return_trend=False: Flattened flux array with same shape as input.
        If return_trend=True: Tuple of (flattened_flux, trend_array).

    Raises:
        ImportError: If wotan is not installed (install with `pip install wotan`
            or `pip install tess-vetter[fit]`).
        ValueError: If time and flux have different shapes.
        ValueError: If window_length is not positive.

    Example:
        >>> import numpy as np
        >>> time = np.linspace(0, 10, 1000)
        >>> flux = 1.0 + 0.01 * np.sin(2 * np.pi * time / 2.0)  # Stellar variation
        >>> # Create transit mask (True = in transit)
        >>> transit_mask = np.abs((time - 5.0) % 3.0) < 0.1
        >>> flat = wotan_flatten(time, flux, window_length=0.5,
        ...                      method="biweight", transit_mask=transit_mask)

    Note:
        For active stars like AU Mic, use method="biweight" or "huber"
        with transit_mask from known ephemeris. For multi-sector data,
        set break_tolerance appropriately for TESS data gaps (~0.5 days).
    """
    if not WOTAN_AVAILABLE:
        raise ImportError("wotan is required for wotan_flatten(). Install with: pip install wotan")

    if time.shape != flux.shape:
        raise ValueError(f"time and flux must have same shape, got {time.shape} and {flux.shape}")
    if window_length <= 0:
        raise ValueError(f"window_length must be positive, got {window_length}")

    if len(time) == 0:
        if return_trend:
            return flux.copy(), np.ones_like(flux)
        return flux.copy()

    # Validate time array is sorted
    if len(time) > 1:
        time_diffs = np.diff(time)
        if np.any(time_diffs < 0):
            raise ValueError(
                "Time array must be sorted in ascending order. "
                f"Found {np.sum(time_diffs < 0)} negative time differences."
            )

    # Validate transit_mask shape if provided
    if transit_mask is not None and transit_mask.shape != flux.shape:
        raise ValueError(
            f"transit_mask must have same shape as flux, got {transit_mask.shape} and {flux.shape}"
        )

    # Call wotan's flatten function
    # Note: wotan expects transit_mask where True = mask (exclude from fit)
    assert _wotan_flatten is not None  # Guaranteed by WOTAN_AVAILABLE check

    # Build kwargs, only pass kernel_size if it's set (for GP method)
    flatten_kwargs: dict[str, Any] = {
        "window_length": window_length,
        "method": method,
        "break_tolerance": break_tolerance,
        "edge_cutoff": edge_cutoff,
        "return_trend": True,  # Always get trend for consistency
        "mask": transit_mask,  # wotan uses 'mask' parameter
    }
    if kernel_size is not None:
        flatten_kwargs["kernel_size"] = kernel_size

    result = _wotan_flatten(time, flux, **flatten_kwargs)

    flattened, trend = result

    # Handle any NaN values in the trend (can happen at edges)
    # Replace NaN trend values with 1.0 (no detrending)
    nan_trend = np.isnan(trend)
    if np.any(nan_trend):
        logger.debug(f"Found {np.sum(nan_trend)} NaN values in wotan trend, replacing with 1.0")
        trend = np.where(nan_trend, 1.0, trend)
        flattened = flux / trend

    if return_trend:
        return np.asarray(flattened, dtype=np.float64), np.asarray(trend, dtype=np.float64)
    return np.asarray(flattened, dtype=np.float64)


def flatten_with_wotan(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    window_length: float = 0.5,
    method: WotanMethod | str = "biweight",
    transit_mask: NDArray[np.bool_] | None = None,
    break_tolerance: float = 0.5,
    fallback_on_error: bool = True,
) -> NDArray[np.float64]:
    """Flatten light curve using wotan with automatic fallback to median filter.

    This is a convenience wrapper that tries wotan first and falls back
    to the basic median filter if wotan is unavailable or fails.

    Args:
        time: Time array in days (BTJD, float64)
        flux: Flux array (float64)
        window_length: Window length in days for smoothing. Default 0.5 days.
        method: Wotan detrending method. Default "biweight".
        transit_mask: Boolean mask where True = in-transit points to exclude
            from trend fitting. Only used if wotan is available.
        break_tolerance: Maximum gap (days) before segment break. Default 0.5.
        fallback_on_error: If True (default), fall back to median filter
            when wotan fails. If False, raise the exception.

    Returns:
        Flattened flux array with same shape as input.

    Example:
        >>> # Will use wotan if available, otherwise fall back to median
        >>> flat = flatten_with_wotan(time, flux, transit_mask=transit_mask)
    """
    if WOTAN_AVAILABLE:
        try:
            result = wotan_flatten(
                time,
                flux,
                window_length=window_length,
                method=method,  # type: ignore[arg-type]
                transit_mask=transit_mask,
                break_tolerance=break_tolerance,
                return_trend=False,
            )
            # Cast needed because mypy doesn't narrow Union based on return_trend
            return cast("NDArray[np.float64]", result)
        except Exception as e:
            if not fallback_on_error:
                raise
            logger.warning(f"wotan_flatten failed ({e}), falling back to median filter")

    # Fallback to basic flatten (median filter)
    if transit_mask is not None:
        logger.warning("transit_mask ignored: wotan not available, using basic median filter")
    return flatten(time, flux, window_length=window_length)
