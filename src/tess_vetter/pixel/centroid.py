"""Centroid shift analysis for transit detection in TPF data.

This module provides tools for computing flux-weighted centroids
and detecting shifts between in-transit and out-of-transit cadences.
Significant centroid shifts can indicate background eclipsing binaries
or other false positive scenarios.

References:
    - Bryson et al. (2013) - Pixel-level centroid analysis for Kepler
    - Bryson et al. (2010) - PRF methodology for sub-pixel centroid determination
    - Batalha et al. (2010) - Pre-spectroscopic false positive elimination
    - Higgins & Bell (2022) - TESS-specific localization methodology
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from tess_vetter.pixel.cadence_mask import default_cadence_mask

# =============================================================================
# Constants
# =============================================================================

# TESS pixel scale in arcseconds per pixel
TESS_PIXEL_SCALE_ARCSEC = 21.0

# TESS saturation threshold (typical for 2-min cadence, e-/s)
TESS_SATURATION_THRESHOLD = 150000.0

# Minimum cadence requirements
MIN_IN_TRANSIT_CADENCES = 5
MIN_OUT_TRANSIT_CADENCES = 20
WARN_IN_TRANSIT_CADENCES = 10
WARN_OUT_TRANSIT_CADENCES = 50


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class CentroidShiftConfig:
    """Configuration for V08 centroid shift check.

    Attributes:
        centroid_method: Centroid aggregation method ("mean", "median", "huber").
            Default: "median" for robustness to outliers.
        significance_method: Method for computing significance ("analytic", "bootstrap", "permutation").
            Default: "bootstrap" for proper uncertainty quantification.
        n_bootstrap: Number of bootstrap iterations. Default: 1000.
        min_in_transit_cadences: Minimum in-transit cadences required. Default: 5.
        min_out_transit_cadences: Minimum out-of-transit cadences required. Default: 20.
        pixel_scale_arcsec: TESS pixel scale. Default: 21.0.
        saturation_threshold: Flux threshold for saturation warning. Default: 150000.0.
        outlier_sigma: Sigma threshold for outlier rejection. Default: 3.0.
    """

    centroid_method: Literal["mean", "median", "huber"] = "median"
    significance_method: Literal["analytic", "bootstrap", "permutation"] = "bootstrap"
    n_bootstrap: int = 1000
    min_in_transit_cadences: int = MIN_IN_TRANSIT_CADENCES
    min_out_transit_cadences: int = MIN_OUT_TRANSIT_CADENCES
    pixel_scale_arcsec: float = TESS_PIXEL_SCALE_ARCSEC
    saturation_threshold: float = TESS_SATURATION_THRESHOLD
    outlier_sigma: float = 3.0


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class CentroidResult:
    """Result of centroid shift analysis.

    Attributes:
        centroid_shift_pixels: Distance between in-transit and out-of-transit
            centroids in pixel units.
        significance_sigma: Statistical significance of the shift in units
            of standard deviations.
        in_transit_centroid: Flux-weighted centroid (x, y) during transit
            in pixel coordinates.
        out_of_transit_centroid: Flux-weighted centroid (x, y) out of transit
            in pixel coordinates.
        n_in_transit_cadences: Number of cadences used for in-transit centroid.
        n_out_transit_cadences: Number of cadences used for out-of-transit centroid.
        centroid_shift_arcsec: Shift distance in arcseconds.
        pixel_scale_arcsec: Pixel scale used for conversion.
        shift_uncertainty_pixels: Bootstrap or analytic standard error of shift.
        shift_ci_lower_pixels: 95% confidence interval lower bound (bootstrap only).
        shift_ci_upper_pixels: 95% confidence interval upper bound (bootstrap only).
        in_transit_centroid_se: Standard error of in-transit centroid (se_x, se_y).
        out_of_transit_centroid_se: Standard error of out-of-transit centroid.
        centroid_method: Method used for centroid computation.
        significance_method: Method used for significance computation.
        n_bootstrap: Number of bootstrap iterations (if applicable).
        saturation_risk: True if saturation may affect centroid reliability.
        max_flux_fraction: Maximum flux as fraction of saturation threshold.
        n_outliers_rejected: Number of outlier cadences rejected.
        warnings: List of warning messages.
    """

    centroid_shift_pixels: float
    significance_sigma: float
    in_transit_centroid: tuple[float, float]  # (x, y) pixels
    out_of_transit_centroid: tuple[float, float]
    n_in_transit_cadences: int
    n_out_transit_cadences: int
    # New fields
    centroid_shift_arcsec: float = 0.0
    pixel_scale_arcsec: float = TESS_PIXEL_SCALE_ARCSEC
    shift_uncertainty_pixels: float = np.nan
    shift_ci_lower_pixels: float = np.nan
    shift_ci_upper_pixels: float = np.nan
    in_transit_centroid_se: tuple[float, float] = (np.nan, np.nan)
    out_of_transit_centroid_se: tuple[float, float] = (np.nan, np.nan)
    centroid_method: str = "mean"
    significance_method: str = "analytic"
    n_bootstrap: int = 0
    saturation_risk: bool = False
    max_flux_fraction: float = 0.0
    n_outliers_rejected: int = 0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TransitParams:
    """Transit ephemeris parameters.

    Attributes:
        period: Orbital period in days.
        t0: Reference transit epoch in BTJD (Barycentric TESS Julian Date).
        duration: Transit duration in hours.
    """

    period: float  # days
    t0: float  # BTJD
    duration: float  # hours


# Window policy configurations
# Each policy defines k_in (in-transit window multiplier) and k_buffer (buffer exclusion)
WINDOW_POLICIES: dict[str, dict[str, float]] = {
    "v1": {
        "k_in": 1.0,  # in_transit_window = k_in * duration
        "k_buffer": 0.5,  # exclude k_buffer * duration around transit edges
    },
}


# =============================================================================
# Saturation Detection
# =============================================================================


def detect_saturation_risk(
    tpf_data: NDArray[np.floating],
    saturation_threshold: float = TESS_SATURATION_THRESHOLD,
) -> tuple[bool, float]:
    """Detect if TPF may contain saturation.

    Saturated stars have centroids biased toward bleed trails, making
    centroid shift analysis unreliable.

    Parameters
    ----------
    tpf_data : ndarray
        Target pixel file data with shape (time, rows, cols).
    saturation_threshold : float
        Flux threshold above which saturation is suspected.

    Returns
    -------
    tuple[bool, float]
        (is_saturated, max_flux_fraction) where max_flux_fraction is
        the maximum observed flux divided by the saturation threshold.
    """
    max_flux = float(np.nanmax(tpf_data))
    frac = max_flux / saturation_threshold if saturation_threshold > 0 else 0.0
    return (frac > 1.0, frac)


# =============================================================================
# Robust Centroid Estimation
# =============================================================================


def _compute_flux_weighted_centroid_single_frame(
    frame: NDArray[np.floating],
) -> tuple[float, float]:
    """Compute flux-weighted centroid for a single frame.

    Parameters
    ----------
    frame : ndarray
        2D array with shape (rows, cols).

    Returns
    -------
    tuple[float, float]
        Flux-weighted centroid (x, y) = (col, row) in pixel coordinates.
        Returns (nan, nan) if flux is zero/negative or all NaN.
    """
    # Handle negative or zero total flux
    total_flux = np.nansum(frame)
    if total_flux <= 0 or not np.isfinite(total_flux):
        return (np.nan, np.nan)

    # Create coordinate grids
    n_rows, n_cols = frame.shape
    row_coords, col_coords = np.meshgrid(
        np.arange(n_rows, dtype=np.float64),
        np.arange(n_cols, dtype=np.float64),
        indexing="ij",
    )

    # Compute flux-weighted centroid
    # Replace NaN with 0 for weighting
    flux_weights = np.where(np.isfinite(frame), frame, 0.0)
    flux_weights = np.maximum(flux_weights, 0.0)  # Ignore negative flux

    weight_sum = np.sum(flux_weights)
    if weight_sum <= 0:
        return (np.nan, np.nan)

    centroid_row = np.sum(row_coords * flux_weights) / weight_sum
    centroid_col = np.sum(col_coords * flux_weights) / weight_sum

    # Return as (x, y) = (col, row) following image convention
    return (float(centroid_col), float(centroid_row))


def _compute_per_cadence_centroids(
    tpf_data: NDArray[np.floating],
    mask: NDArray[np.bool_],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute centroids for each cadence specified by mask.

    Parameters
    ----------
    tpf_data : ndarray
        Target pixel file data with shape (time, rows, cols).
    mask : ndarray
        Boolean mask selecting which cadences to use.

    Returns
    -------
    tuple[ndarray, ndarray]
        (centroids_x, centroids_y) arrays containing valid centroid values.
        NaN values are filtered out.
    """
    centroids_x: list[float] = []
    centroids_y: list[float] = []

    for i in np.where(mask)[0]:
        frame = tpf_data[i]
        cx, cy = _compute_flux_weighted_centroid_single_frame(frame)
        if np.isfinite(cx) and np.isfinite(cy):
            centroids_x.append(cx)
            centroids_y.append(cy)

    return (
        np.array(centroids_x, dtype=np.float64),
        np.array(centroids_y, dtype=np.float64),
    )


def _reject_outliers(
    centroids_x: NDArray[np.floating],
    centroids_y: NDArray[np.floating],
    sigma: float = 3.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], int]:
    """Reject outlier centroids using sigma clipping.

    Parameters
    ----------
    centroids_x : ndarray
        X (column) centroid values.
    centroids_y : ndarray
        Y (row) centroid values.
    sigma : float
        Number of standard deviations for clipping.

    Returns
    -------
    tuple[ndarray, ndarray, int]
        (filtered_x, filtered_y, n_rejected) where n_rejected is the
        number of outliers removed.
    """
    if len(centroids_x) < 5:
        # Not enough points for reliable outlier detection
        return centroids_x, centroids_y, 0

    # Compute median and MAD for robust statistics
    med_x = np.median(centroids_x)
    med_y = np.median(centroids_y)

    # MAD-based robust standard deviation
    mad_x = 1.4826 * np.median(np.abs(centroids_x - med_x))
    mad_y = 1.4826 * np.median(np.abs(centroids_y - med_y))

    # Avoid division by zero
    if mad_x <= 0:
        mad_x = np.std(centroids_x)
    if mad_y <= 0:
        mad_y = np.std(centroids_y)

    # Create mask for valid points
    valid_x = np.abs(centroids_x - med_x) <= sigma * mad_x
    valid_y = np.abs(centroids_y - med_y) <= sigma * mad_y
    valid = valid_x & valid_y

    n_rejected = len(centroids_x) - np.sum(valid)

    return centroids_x[valid], centroids_y[valid], int(n_rejected)


def robust_centroid_estimate(
    tpf_data: NDArray[np.floating],
    mask: NDArray[np.bool_],
    method: Literal["mean", "median", "huber"] = "median",
    outlier_sigma: float = 3.0,
) -> tuple[float, float, float, float, int]:
    """Compute robust centroid with uncertainty from per-cadence measurements.

    This function computes centroids for each cadence individually, then
    aggregates them using a robust method (median or Huber M-estimator).
    This approach is more robust to outliers from cosmic rays, bad pixels,
    and pointing jitter than computing a single centroid from stacked frames.

    Parameters
    ----------
    tpf_data : ndarray
        Target pixel file data with shape (time, rows, cols).
    mask : ndarray
        Boolean mask selecting which cadences to use.
    method : {"mean", "median", "huber"}
        Aggregation method for per-cadence centroids.
    outlier_sigma : float
        Sigma threshold for outlier rejection before aggregation.

    Returns
    -------
    tuple[float, float, float, float, int]
        (centroid_x, centroid_y, se_x, se_y, n_rejected) where se_x, se_y
        are the standard errors of the mean/median centroids.
    """
    # Get per-cadence centroids
    centroids_x, centroids_y = _compute_per_cadence_centroids(tpf_data, mask)

    if len(centroids_x) < 3:
        return (np.nan, np.nan, np.nan, np.nan, 0)

    # Reject outliers
    centroids_x, centroids_y, n_rejected = _reject_outliers(
        centroids_x, centroids_y, sigma=outlier_sigma
    )

    if len(centroids_x) < 3:
        return (np.nan, np.nan, np.nan, np.nan, n_rejected)

    n = len(centroids_x)

    if method == "median":
        cx = float(np.median(centroids_x))
        cy = float(np.median(centroids_y))
        # MAD-based standard error for median
        mad_x = 1.4826 * np.median(np.abs(centroids_x - cx))
        mad_y = 1.4826 * np.median(np.abs(centroids_y - cy))
        # SE of median is approximately MAD / sqrt(n) * sqrt(pi/2)
        se_x = float(mad_x / np.sqrt(n) * np.sqrt(np.pi / 2))
        se_y = float(mad_y / np.sqrt(n) * np.sqrt(np.pi / 2))
    elif method == "huber":
        # Use Huber M-estimator with iteratively reweighted least squares
        # approach with Huber weights
        try:

            def huber_location(data: NDArray[np.floating], c: float = 1.35) -> float:
                """Compute Huber M-estimator of location."""
                # Initial estimate: median
                med = np.median(data)
                mad = 1.4826 * np.median(np.abs(data - med))
                if mad == 0:
                    return float(med)
                # Standardize data
                z = (data - med) / mad
                # Huber weights
                weights = np.where(np.abs(z) <= c, 1.0, c / np.abs(z))
                return float(np.sum(weights * data) / np.sum(weights))

            cx = huber_location(centroids_x)
            cy = huber_location(centroids_y)
            # MAD-based standard error
            mad_x = 1.4826 * np.median(np.abs(centroids_x - cx))
            mad_y = 1.4826 * np.median(np.abs(centroids_y - cy))
            se_x = float(mad_x / np.sqrt(n))
            se_y = float(mad_y / np.sqrt(n))
        except (ImportError, ValueError):
            # Fall back to median
            cx = float(np.median(centroids_x))
            cy = float(np.median(centroids_y))
            se_x = float(np.std(centroids_x, ddof=1) / np.sqrt(n))
            se_y = float(np.std(centroids_y, ddof=1) / np.sqrt(n))
    else:  # mean
        cx = float(np.mean(centroids_x))
        cy = float(np.mean(centroids_y))
        se_x = float(np.std(centroids_x, ddof=1) / np.sqrt(n))
        se_y = float(np.std(centroids_y, ddof=1) / np.sqrt(n))

    return (cx, cy, se_x, se_y, n_rejected)


def _compute_flux_weighted_centroid(
    tpf_data: NDArray[np.floating],
    mask: NDArray[np.bool_],
) -> tuple[float, float]:
    """Compute flux-weighted centroid for cadences specified by mask.

    This is the legacy function for backward compatibility. It pools all
    cadences into a single mean frame before computing centroid.

    Parameters
    ----------
    tpf_data : ndarray
        Target pixel file data with shape (time, rows, cols).
    mask : ndarray
        Boolean mask selecting which cadences to use.

    Returns
    -------
    tuple[float, float]
        Flux-weighted centroid (x, y) in pixel coordinates.
        Returns (nan, nan) if no valid cadences or all flux is zero/negative.
    """
    if not np.any(mask):
        return (np.nan, np.nan)

    # Select cadences and compute mean flux per pixel
    selected_frames = tpf_data[mask]
    # nanmean warns when a pixel is all-NaN across selected frames, which is
    # expected for some masked/flagged pixels; suppress and treat those pixels
    # as 0-weight downstream.
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        mean_flux = np.nanmean(selected_frames, axis=0)
    mean_flux = np.where(np.isfinite(mean_flux), mean_flux, 0.0)

    return _compute_flux_weighted_centroid_single_frame(mean_flux)


# =============================================================================
# Transit Mask Computation
# =============================================================================


def _get_transit_masks(
    time: NDArray[np.floating],
    transit_params: TransitParams,
    k_in: float,
    k_buffer: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Compute in-transit and out-of-transit masks.

    Parameters
    ----------
    time : ndarray
        Time array in BTJD.
    transit_params : TransitParams
        Transit ephemeris parameters.
    k_in : float
        Multiplier for in-transit window (window = k_in * duration).
    k_buffer : float
        Multiplier for buffer zone around transit (excluded from out-of-transit).

    Returns
    -------
    tuple[ndarray, ndarray]
        Boolean masks for in-transit and out-of-transit cadences.
    """
    # Convert duration from hours to days
    duration_days = transit_params.duration / 24.0

    # In-transit window half-width
    in_transit_half_width = (k_in * duration_days) / 2.0

    # Buffer zone half-width (excludes ingress/egress)
    buffer_half_width = duration_days / 2.0 + k_buffer * duration_days

    # Compute phase relative to transit center
    # Phase is in range [-0.5, 0.5) with 0 at transit center
    phase = ((time - transit_params.t0) / transit_params.period) % 1.0
    phase = np.where(phase > 0.5, phase - 1.0, phase)

    # Time from nearest transit center in days
    time_from_transit = phase * transit_params.period

    # In-transit mask: within k_in * duration of transit center
    in_transit_mask = np.abs(time_from_transit) <= in_transit_half_width

    # Out-of-transit mask: outside buffer zone
    out_of_transit_mask = np.abs(time_from_transit) > buffer_half_width

    return in_transit_mask.astype(bool), out_of_transit_mask.astype(bool)


# =============================================================================
# Significance Estimation
# =============================================================================


def _compute_shift_significance_bootstrap(
    tpf_data: NDArray[np.floating],
    in_transit_mask: NDArray[np.bool_],
    out_of_transit_mask: NDArray[np.bool_],
    observed_shift: float,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
    centroid_method: Literal["mean", "median", "huber"] = "median",
    outlier_sigma: float = 3.0,
) -> tuple[float, float, float, float]:
    """Compute significance of centroid shift using bootstrap resampling.

    This function uses stratified bootstrap resampling to estimate the
    null distribution of centroid shifts and compute confidence intervals.

    Parameters
    ----------
    tpf_data : ndarray
        Target pixel file data with shape (time, rows, cols).
    in_transit_mask : ndarray
        Boolean mask for in-transit cadences.
    out_of_transit_mask : ndarray
        Boolean mask for out-of-transit cadences.
    observed_shift : float
        Observed centroid shift in pixels.
    n_bootstrap : int
        Number of bootstrap iterations.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.
    centroid_method : str
        Method for centroid aggregation.
    outlier_sigma : float
        Sigma threshold for outlier rejection.

    Returns
    -------
    tuple[float, float, float, float]
        (significance_sigma, shift_se, ci_lower, ci_upper) where:
        - significance_sigma: Significance in standard deviations
        - shift_se: Standard error of the shift
        - ci_lower: 95% CI lower bound (2.5th percentile)
        - ci_upper: 95% CI upper bound (97.5th percentile)
    """
    _rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

    # Get per-cadence centroids for in-transit and out-of-transit
    in_centroids_x, in_centroids_y = _compute_per_cadence_centroids(tpf_data, in_transit_mask)
    out_centroids_x, out_centroids_y = _compute_per_cadence_centroids(tpf_data, out_of_transit_mask)

    # Reject outliers
    in_centroids_x, in_centroids_y, _ = _reject_outliers(
        in_centroids_x, in_centroids_y, sigma=outlier_sigma
    )
    out_centroids_x, out_centroids_y, _ = _reject_outliers(
        out_centroids_x, out_centroids_y, sigma=outlier_sigma
    )

    n_in = len(in_centroids_x)
    n_out = len(out_centroids_x)

    if n_in < 3 or n_out < 3:
        return (np.nan, np.nan, np.nan, np.nan)

    # Stack into 2D arrays for easier resampling
    in_centroids = np.column_stack([in_centroids_x, in_centroids_y])
    out_centroids = np.column_stack([out_centroids_x, out_centroids_y])

    bootstrap_shifts: list[float] = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        in_sample_idx = _rng.choice(n_in, size=n_in, replace=True)
        out_sample_idx = _rng.choice(n_out, size=n_out, replace=True)

        in_sample = in_centroids[in_sample_idx]
        out_sample = out_centroids[out_sample_idx]

        # Compute centroid aggregates
        if centroid_method == "median":
            in_cx = np.median(in_sample[:, 0])
            in_cy = np.median(in_sample[:, 1])
            out_cx = np.median(out_sample[:, 0])
            out_cy = np.median(out_sample[:, 1])
        else:  # mean
            in_cx = np.mean(in_sample[:, 0])
            in_cy = np.mean(in_sample[:, 1])
            out_cx = np.mean(out_sample[:, 0])
            out_cy = np.mean(out_sample[:, 1])

        # Compute shift
        shift = np.sqrt((in_cx - out_cx) ** 2 + (in_cy - out_cy) ** 2)
        bootstrap_shifts.append(float(shift))

    if len(bootstrap_shifts) < 10:
        return (np.nan, np.nan, np.nan, np.nan)

    bootstrap_arr = np.array(bootstrap_shifts)

    # Standard error from bootstrap distribution
    shift_se = float(np.std(bootstrap_arr, ddof=1))

    # 95% confidence interval
    ci_lower = float(np.percentile(bootstrap_arr, 2.5))
    ci_upper = float(np.percentile(bootstrap_arr, 97.5))

    # Convert the bootstrap spread into an uncertainty on the observed shift.
    # Note: A standard bootstrap over (in, out) samples estimates the sampling
    # distribution around the observed shift (not a "null" centered at 0), so
    # using (observed - mean_bootstrap) yields ~0 significance even for real shifts.
    # For a "sigma" style metric, use shift / SE.
    if shift_se > 0 and np.isfinite(shift_se) and np.isfinite(observed_shift):
        significance = float(observed_shift / shift_se)
        significance = float(np.clip(significance, 0.0, 8.0))
    else:
        significance = 0.0

    return (significance, shift_se, ci_lower, ci_upper)


def _compute_shift_significance_permutation(
    tpf_data: NDArray[np.floating],
    in_transit_mask: NDArray[np.bool_],
    out_of_transit_mask: NDArray[np.bool_],
    observed_shift: float,
    n_permutations: int = 1000,
    rng: np.random.Generator | None = None,
    centroid_method: Literal["mean", "median", "huber"] = "median",
    outlier_sigma: float = 3.0,
) -> tuple[float, float, float, float, float]:
    """Compute centroid shift significance via label permutation.

    This estimates a *null* distribution by randomizing which cadences are
    treated as "in-transit" vs "out-of-transit" while preserving group sizes.

    Returns:
        (significance_sigma, p_value, null_sigma, null_ci_lower, null_ci_upper)
    """
    _rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

    in_centroids_x, in_centroids_y = _compute_per_cadence_centroids(tpf_data, in_transit_mask)
    out_centroids_x, out_centroids_y = _compute_per_cadence_centroids(tpf_data, out_of_transit_mask)

    # Reject outliers (same as bootstrap path).
    in_centroids_x, in_centroids_y, _ = _reject_outliers(
        in_centroids_x, in_centroids_y, sigma=outlier_sigma
    )
    out_centroids_x, out_centroids_y, _ = _reject_outliers(
        out_centroids_x, out_centroids_y, sigma=outlier_sigma
    )

    n_in = len(in_centroids_x)
    n_out = len(out_centroids_x)
    if n_in < 3 or n_out < 3:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    pool = np.concatenate(
        [
            np.column_stack([in_centroids_x, in_centroids_y]),
            np.column_stack([out_centroids_x, out_centroids_y]),
        ],
        axis=0,
    )
    n_total = pool.shape[0]
    if n_permutations < 10 or n_total < (n_in + n_out):
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    perm_shifts = np.empty(n_permutations, dtype=np.float64)
    idx = np.arange(n_total)

    for i in range(n_permutations):
        perm = _rng.permutation(idx)
        in_idx = perm[:n_in]
        out_idx = perm[n_in : n_in + n_out]
        in_sample = pool[in_idx]
        out_sample = pool[out_idx]

        if centroid_method == "median":
            in_cx = float(np.median(in_sample[:, 0]))
            in_cy = float(np.median(in_sample[:, 1]))
            out_cx = float(np.median(out_sample[:, 0]))
            out_cy = float(np.median(out_sample[:, 1]))
        else:
            in_cx = float(np.mean(in_sample[:, 0]))
            in_cy = float(np.mean(in_sample[:, 1]))
            out_cx = float(np.mean(out_sample[:, 0]))
            out_cy = float(np.mean(out_sample[:, 1]))

        perm_shifts[i] = np.sqrt((in_cx - out_cx) ** 2 + (in_cy - out_cy) ** 2)

    ge = int(np.sum(perm_shifts >= observed_shift))
    p_value = float((ge + 1) / (n_permutations + 1))
    significance = float(norm.isf(p_value)) if 0 < p_value < 1 else (8.0 if p_value == 0 else 0.0)
    significance = float(np.clip(significance, 0.0, 8.0))

    null_sigma = float(np.std(perm_shifts, ddof=1))
    ci_lower = float(np.percentile(perm_shifts, 2.5))
    ci_upper = float(np.percentile(perm_shifts, 97.5))
    return (significance, p_value, null_sigma, ci_lower, ci_upper)


def _compute_shift_significance_analytic(
    tpf_data: NDArray[np.floating],
    in_transit_mask: NDArray[np.bool_],
    out_of_transit_mask: NDArray[np.bool_],
    in_centroid: tuple[float, float],
    out_centroid: tuple[float, float],
    observed_shift: float,
    in_centroid_se: tuple[float, float] | None = None,
    out_centroid_se: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Compute significance of centroid shift using analytic formula.

    Uses error propagation assuming the centroid measurements are
    independent and normally distributed.

    Parameters
    ----------
    tpf_data : ndarray
        Target pixel file data with shape (time, rows, cols).
    in_transit_mask : ndarray
        Boolean mask for in-transit cadences.
    out_of_transit_mask : ndarray
        Boolean mask for out-of-transit cadences.
    in_centroid : tuple[float, float]
        In-transit centroid (x, y).
    out_centroid : tuple[float, float]
        Out-of-transit centroid (x, y).
    observed_shift : float
        Observed centroid shift in pixels.
    in_centroid_se : tuple[float, float], optional
        Pre-computed standard errors for in-transit centroid.
    out_centroid_se : tuple[float, float], optional
        Pre-computed standard errors for out-of-transit centroid.

    Returns
    -------
    tuple[float, float]
        (significance_sigma, shift_se) where shift_se is the standard
        error of the shift.
    """
    n_in = int(np.sum(in_transit_mask))
    n_out = int(np.sum(out_of_transit_mask))

    if n_in == 0 or n_out == 0:
        return (np.nan, np.nan)

    # Use pre-computed standard errors if provided
    if in_centroid_se is not None and out_centroid_se is not None:
        se_in_x, se_in_y = in_centroid_se
        se_out_x, se_out_y = out_centroid_se
    else:
        # Compute variance of centroids for each group from per-cadence scatter
        def compute_centroid_variance(
            data: NDArray[np.floating], mask: NDArray[np.bool_]
        ) -> tuple[float, float]:
            """Compute variance of centroid across cadences."""
            centroids_x, centroids_y = _compute_per_cadence_centroids(data, mask)
            n_cadences = len(centroids_x)

            if n_cadences < 2:
                return (np.nan, np.nan)

            var_x = float(np.var(centroids_x, ddof=1))
            var_y = float(np.var(centroids_y, ddof=1))
            return (var_x, var_y)

        var_in_x, var_in_y = compute_centroid_variance(tpf_data, in_transit_mask)
        var_out_x, var_out_y = compute_centroid_variance(tpf_data, out_of_transit_mask)

        if any(np.isnan(v) for v in [var_in_x, var_in_y, var_out_x, var_out_y]):
            return (np.nan, np.nan)

        # Standard error of the mean centroid
        se_in_x = np.sqrt(var_in_x / n_in)
        se_in_y = np.sqrt(var_in_y / n_in)
        se_out_x = np.sqrt(var_out_x / n_out)
        se_out_y = np.sqrt(var_out_y / n_out)

    # Combined standard error for the difference
    se_diff_x = np.sqrt(se_in_x**2 + se_out_x**2)
    se_diff_y = np.sqrt(se_in_y**2 + se_out_y**2)

    # Difference in centroids
    diff_x = in_centroid[0] - out_centroid[0]
    diff_y = in_centroid[1] - out_centroid[1]

    # Significance of the shift using error propagation
    # For distance d = sqrt(dx^2 + dy^2), we use:
    # sigma_d^2 = (dx/d)^2 * sigma_dx^2 + (dy/d)^2 * sigma_dy^2
    if observed_shift <= 0:
        return (0.0, 0.0)

    sigma_shift_sq = (diff_x / observed_shift) ** 2 * se_diff_x**2 + (
        diff_y / observed_shift
    ) ** 2 * se_diff_y**2

    sigma_shift = float(np.sqrt(sigma_shift_sq))

    if sigma_shift <= 0:
        return (np.nan, sigma_shift)

    significance = float(observed_shift / sigma_shift)
    return (significance, sigma_shift)


# =============================================================================
# Confidence Computation
# =============================================================================


def compute_data_quality_confidence(
    n_in: int,
    n_out: int,
    has_warnings: bool = False,
    saturation_risk: bool = False,
) -> tuple[float, list[str]]:
    """Compute confidence adjustment based on data quality.

    Parameters
    ----------
    n_in : int
        Number of in-transit cadences.
    n_out : int
        Number of out-of-transit cadences.
    has_warnings : bool
        Whether other warnings are present.
    saturation_risk : bool
        Whether saturation may affect reliability.

    Returns
    -------
    tuple[float, list[str]]
        (confidence, warnings) where confidence is in [0, 1].
    """
    warnings: list[str] = []

    # Check minimum requirements
    if n_in < MIN_IN_TRANSIT_CADENCES:
        warnings.append(f"low_n_in_transit ({n_in} < {MIN_IN_TRANSIT_CADENCES})")
        base = 0.2
    elif n_out < MIN_OUT_TRANSIT_CADENCES:
        warnings.append(f"low_n_out_transit ({n_out} < {MIN_OUT_TRANSIT_CADENCES})")
        base = 0.3
    elif n_in < WARN_IN_TRANSIT_CADENCES:
        warnings.append(f"marginal_n_in_transit ({n_in} < {WARN_IN_TRANSIT_CADENCES})")
        base = 0.5
    elif n_in < 20:
        base = 0.7
    else:
        base = 0.85

    # Degrade for out-of-transit
    if n_out < WARN_OUT_TRANSIT_CADENCES and base > 0.3:
        warnings.append(f"marginal_n_out_transit ({n_out} < {WARN_OUT_TRANSIT_CADENCES})")
        base *= 0.9

    # Degrade for saturation
    if saturation_risk:
        warnings.append("saturation_risk")
        base *= 0.8

    # Degrade for other warnings
    if has_warnings:
        base *= 0.9

    return (min(0.95, base), warnings)


# =============================================================================
# Main Function
# =============================================================================


def compute_centroid_shift(
    tpf_data: NDArray[np.floating],
    time: NDArray[np.floating],
    transit_params: TransitParams,
    window_policy_version: str = "v1",
    significance_method: Literal["analytic", "bootstrap", "permutation"] = "bootstrap",
    n_bootstrap: int = 1000,
    bootstrap_seed: int | None = None,
    centroid_method: Literal["mean", "median", "huber"] = "median",
    outlier_sigma: float = 3.0,
    config: CentroidShiftConfig | None = None,
) -> CentroidResult:
    """Compute centroid shift between in-transit and out-of-transit cadences.

    This function computes the flux-weighted centroid during transit and
    outside of transit, then calculates the distance between them and
    the statistical significance of any shift.

    The improved implementation (v2) includes:
    - Robust centroid estimation via per-cadence median aggregation
    - Bootstrap confidence intervals for uncertainty quantification
    - Minimum cadence requirements with confidence degradation
    - Saturation detection and flagging
    - Outlier rejection for cosmic rays and bad pixels
    - Output in both pixels and arcseconds

    Parameters
    ----------
    tpf_data : ndarray
        Target pixel file flux data with shape (time, rows, cols).
        Values should be in electrons/s or similar flux units.
    time : ndarray
        Time array in BTJD (Barycentric TESS Julian Date).
        Must have the same length as tpf_data's first dimension.
    transit_params : TransitParams
        Transit ephemeris parameters (period, t0, duration).
    window_policy_version : str, optional
        Version of window policy to use. Default is "v1".
        - "v1": k_in=1.0, k_buffer=0.5
    significance_method : {"analytic", "bootstrap", "permutation"}, optional
        Method for computing significance. Default is "bootstrap".
        - "analytic": Uses error propagation from centroid scatter.
        - "bootstrap": Uses bootstrap resampling (more robust).
        - "permutation": Uses label permutation to estimate a null p-value.
    n_bootstrap : int, optional
        Number of bootstrap iterations if using bootstrap method.
        Default is 1000.
    bootstrap_seed : int, optional
        Random seed for bootstrap reproducibility.
    centroid_method : {"mean", "median", "huber"}, optional
        Method for aggregating per-cadence centroids. Default is "median".
        - "mean": Standard weighted mean (legacy behavior).
        - "median": Robust to outliers (recommended).
        - "huber": Huber M-estimator for robust location.
    outlier_sigma : float, optional
        Sigma threshold for outlier rejection. Default is 3.0.
    config : CentroidShiftConfig, optional
        Full configuration object (overrides other parameters).

    Returns
    -------
    CentroidResult
        Result containing centroid shift, significance, uncertainties,
        and diagnostic metadata.

    Raises
    ------
    ValueError
        If window_policy_version is unknown or input shapes are invalid.

    References
    ----------
    - Bryson et al. (2013) - Pixel-level centroid methodology
    - Batalha et al. (2010) - Centroid-based false positive elimination
    - Higgins & Bell (2022) - TESS localization methods

    Examples
    --------
    >>> import numpy as np
    >>> from tess_vetter.pixel.centroid import (
    ...     compute_centroid_shift, TransitParams
    ... )
    >>> # Create synthetic TPF data
    >>> tpf = np.random.rand(100, 11, 11) * 1000
    >>> time = np.linspace(0, 10, 100)
    >>> params = TransitParams(period=2.5, t0=1.25, duration=3.0)
    >>> result = compute_centroid_shift(tpf, time, params)
    >>> print(f"Shift: {result.centroid_shift_pixels:.4f} pixels")
    >>> print(f"Shift: {result.centroid_shift_arcsec:.2f} arcsec")
    """
    # Apply config if provided
    if config is not None:
        significance_method = config.significance_method
        n_bootstrap = config.n_bootstrap
        centroid_method = config.centroid_method
        outlier_sigma = config.outlier_sigma
        pixel_scale = config.pixel_scale_arcsec
        saturation_threshold = config.saturation_threshold
    else:
        pixel_scale = TESS_PIXEL_SCALE_ARCSEC
        saturation_threshold = TESS_SATURATION_THRESHOLD

    # Validate inputs
    if tpf_data.ndim != 3:
        raise ValueError(f"tpf_data must be 3D (time, rows, cols), got shape {tpf_data.shape}")

    if len(time) != tpf_data.shape[0]:
        raise ValueError(
            f"time length ({len(time)}) must match tpf_data first dimension ({tpf_data.shape[0]})"
        )

    # Filter out cadences that cannot support centroiding (non-finite time or no finite pixels).
    # Quality flags are not available at this layer (TPFData has only time+flux), so we apply
    # a conservative, data-driven cadence mask.
    cadence_mask = default_cadence_mask(
        time=time,
        flux=tpf_data,
        quality=np.zeros(int(time.shape[0]), dtype=np.int32),
        require_finite_pixels=True,
    )
    n_total = int(time.shape[0])
    n_used = int(np.sum(cadence_mask))
    warnings_dropped = n_total - n_used if n_used < n_total else 0
    if n_used < 3:
        # Keep behavior consistent with the rest of the module: return NaNs with warnings rather than crash.
        return CentroidResult(
            centroid_shift_pixels=np.nan,
            significance_sigma=np.nan,
            in_transit_centroid=(np.nan, np.nan),
            out_of_transit_centroid=(np.nan, np.nan),
            n_in_transit_cadences=0,
            n_out_transit_cadences=0,
            centroid_shift_arcsec=np.nan,
            pixel_scale_arcsec=pixel_scale,
            shift_uncertainty_pixels=np.nan,
            saturation_risk=False,
            max_flux_fraction=0.0,
            n_outliers_rejected=0,
            warnings=(f"dropped_invalid_cadences ({warnings_dropped} of {n_total})",)
            if warnings_dropped
            else ("insufficient_valid_cadences",),
        )

    if warnings_dropped:
        # Defer policy interpretation to downstream guardrails, but record the drop count.
        # (Useful to explain centroid instability on real TESS data with flagged cadences / NaNs.)
        warnings: list[str] = [f"dropped_invalid_cadences ({warnings_dropped} of {n_total})"]
    else:
        warnings = []

    tpf_data = tpf_data[cadence_mask]
    time = time[cadence_mask]

    if window_policy_version not in WINDOW_POLICIES:
        raise ValueError(
            f"Unknown window_policy_version: {window_policy_version}. "
            f"Available: {list(WINDOW_POLICIES.keys())}"
        )

    # Detect saturation
    saturation_risk, max_flux_fraction = detect_saturation_risk(tpf_data, saturation_threshold)

    # Get window policy parameters
    policy = WINDOW_POLICIES[window_policy_version]
    k_in = policy["k_in"]
    k_buffer = policy["k_buffer"]

    # Compute transit masks
    in_transit_mask, out_of_transit_mask = _get_transit_masks(time, transit_params, k_in, k_buffer)

    n_in_transit = int(np.sum(in_transit_mask))
    n_out_transit = int(np.sum(out_of_transit_mask))

    # Check minimum cadence requirements
    _, cadence_warnings = compute_data_quality_confidence(
        n_in_transit, n_out_transit, saturation_risk=saturation_risk
    )
    warnings.extend(cadence_warnings)

    # Compute centroids using robust method
    if centroid_method in ("median", "huber"):
        # Use robust per-cadence aggregation
        in_cx, in_cy, in_se_x, in_se_y, in_rejected = robust_centroid_estimate(
            tpf_data, in_transit_mask, method=centroid_method, outlier_sigma=outlier_sigma
        )
        out_cx, out_cy, out_se_x, out_se_y, out_rejected = robust_centroid_estimate(
            tpf_data, out_of_transit_mask, method=centroid_method, outlier_sigma=outlier_sigma
        )
        n_outliers_rejected = in_rejected + out_rejected
        in_centroid = (in_cx, in_cy)
        out_centroid = (out_cx, out_cy)
        in_centroid_se = (in_se_x, in_se_y)
        out_centroid_se = (out_se_x, out_se_y)
    else:
        # Legacy mean method
        in_centroid = _compute_flux_weighted_centroid(tpf_data, in_transit_mask)
        out_centroid = _compute_flux_weighted_centroid(tpf_data, out_of_transit_mask)
        in_centroid_se = (np.nan, np.nan)
        out_centroid_se = (np.nan, np.nan)
        n_outliers_rejected = 0

    # Handle edge cases where centroids cannot be computed
    if np.isnan(in_centroid[0]) or np.isnan(out_centroid[0]):
        return CentroidResult(
            centroid_shift_pixels=np.nan,
            significance_sigma=np.nan,
            in_transit_centroid=in_centroid,
            out_of_transit_centroid=out_centroid,
            n_in_transit_cadences=n_in_transit,
            n_out_transit_cadences=n_out_transit,
            centroid_shift_arcsec=np.nan,
            pixel_scale_arcsec=pixel_scale,
            shift_uncertainty_pixels=np.nan,
            in_transit_centroid_se=in_centroid_se,
            out_of_transit_centroid_se=out_centroid_se,
            centroid_method=centroid_method,
            significance_method=significance_method,
            n_bootstrap=n_bootstrap if significance_method == "bootstrap" else 0,
            saturation_risk=saturation_risk,
            max_flux_fraction=max_flux_fraction,
            n_outliers_rejected=n_outliers_rejected,
            warnings=tuple(warnings),
        )

    # Compute shift distance
    shift = float(
        np.sqrt((in_centroid[0] - out_centroid[0]) ** 2 + (in_centroid[1] - out_centroid[1]) ** 2)
    )
    shift_arcsec = shift * pixel_scale

    # Compute significance
    if significance_method == "bootstrap":
        rng = np.random.default_rng(bootstrap_seed) if bootstrap_seed is not None else None
        significance, shift_se, ci_lower, ci_upper = _compute_shift_significance_bootstrap(
            tpf_data,
            in_transit_mask,
            out_of_transit_mask,
            shift,
            n_bootstrap=n_bootstrap,
            rng=rng,
            centroid_method=centroid_method,
            outlier_sigma=outlier_sigma,
        )
    elif significance_method == "permutation":
        rng = np.random.default_rng(bootstrap_seed) if bootstrap_seed is not None else None
        significance, p_value, shift_se, ci_lower, ci_upper = (
            _compute_shift_significance_permutation(
                tpf_data,
                in_transit_mask,
                out_of_transit_mask,
                shift,
                n_permutations=n_bootstrap,
                rng=rng,
                centroid_method=centroid_method,
                outlier_sigma=outlier_sigma,
            )
        )
        if np.isfinite(p_value):
            warnings.append(f"permutation_p={p_value:.3g}")
    else:  # analytic
        significance, shift_se = _compute_shift_significance_analytic(
            tpf_data,
            in_transit_mask,
            out_of_transit_mask,
            in_centroid,
            out_centroid,
            shift,
            in_centroid_se=in_centroid_se if centroid_method != "mean" else None,
            out_centroid_se=out_centroid_se if centroid_method != "mean" else None,
        )
        ci_lower = np.nan
        ci_upper = np.nan

    return CentroidResult(
        centroid_shift_pixels=shift,
        significance_sigma=significance,
        in_transit_centroid=in_centroid,
        out_of_transit_centroid=out_centroid,
        n_in_transit_cadences=n_in_transit,
        n_out_transit_cadences=n_out_transit,
        centroid_shift_arcsec=shift_arcsec,
        pixel_scale_arcsec=pixel_scale,
        shift_uncertainty_pixels=shift_se,
        shift_ci_lower_pixels=ci_lower,
        shift_ci_upper_pixels=ci_upper,
        in_transit_centroid_se=in_centroid_se,
        out_of_transit_centroid_se=out_centroid_se,
        centroid_method=centroid_method,
        significance_method=significance_method,
        n_bootstrap=n_bootstrap if significance_method == "bootstrap" else 0,
        saturation_risk=saturation_risk,
        max_flux_fraction=max_flux_fraction,
        n_outliers_rejected=n_outliers_rejected,
        warnings=tuple(warnings),
    )
