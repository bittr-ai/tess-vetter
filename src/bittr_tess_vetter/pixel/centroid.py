"""Centroid shift analysis for transit detection in TPF data.

This module provides tools for computing flux-weighted centroids
and detecting shifts between in-transit and out-of-transit cadences.
Significant centroid shifts can indicate background eclipsing binaries
or other false positive scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


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
    """

    centroid_shift_pixels: float
    significance_sigma: float
    in_transit_centroid: tuple[float, float]  # (x, y) pixels
    out_of_transit_centroid: tuple[float, float]
    n_in_transit_cadences: int
    n_out_transit_cadences: int


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


def _compute_flux_weighted_centroid(
    tpf_data: NDArray[np.floating],
    mask: NDArray[np.bool_],
) -> tuple[float, float]:
    """Compute flux-weighted centroid for cadences specified by mask.

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
    mean_flux = np.nanmean(selected_frames, axis=0)

    # Handle negative or zero total flux
    total_flux = np.nansum(mean_flux)
    if total_flux <= 0 or not np.isfinite(total_flux):
        return (np.nan, np.nan)

    # Create coordinate grids
    n_rows, n_cols = mean_flux.shape
    row_coords, col_coords = np.meshgrid(
        np.arange(n_rows, dtype=np.float64),
        np.arange(n_cols, dtype=np.float64),
        indexing="ij",
    )

    # Compute flux-weighted centroid
    # Replace NaN with 0 for weighting
    flux_weights = np.where(np.isfinite(mean_flux), mean_flux, 0.0)
    flux_weights = np.maximum(flux_weights, 0.0)  # Ignore negative flux

    weight_sum = np.sum(flux_weights)
    if weight_sum <= 0:
        return (np.nan, np.nan)

    centroid_row = np.sum(row_coords * flux_weights) / weight_sum
    centroid_col = np.sum(col_coords * flux_weights) / weight_sum

    # Return as (x, y) = (col, row) following image convention
    return (float(centroid_col), float(centroid_row))


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


def _compute_shift_significance_bootstrap(
    tpf_data: NDArray[np.floating],
    in_transit_mask: NDArray[np.bool_],
    out_of_transit_mask: NDArray[np.bool_],
    observed_shift: float,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    """Compute significance of centroid shift using bootstrap resampling.

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

    Returns
    -------
    float
        Significance in units of standard deviations.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_in = int(np.sum(in_transit_mask))
    n_out = int(np.sum(out_of_transit_mask))

    if n_in == 0 or n_out == 0:
        return np.nan

    # Get all cadence indices
    all_indices = np.where(in_transit_mask | out_of_transit_mask)[0]
    n_total = len(all_indices)

    if n_total < 2:
        return np.nan

    bootstrap_shifts: list[float] = []

    for _ in range(n_bootstrap):
        # Randomly assign cadences to "in-transit" and "out-of-transit"
        shuffled = rng.permutation(all_indices)
        fake_in = shuffled[:n_in]
        fake_out = shuffled[n_in : n_in + n_out]

        # Create masks
        fake_in_mask = np.zeros(len(tpf_data), dtype=bool)
        fake_out_mask = np.zeros(len(tpf_data), dtype=bool)
        fake_in_mask[fake_in] = True
        fake_out_mask[fake_out] = True

        # Compute centroids
        in_centroid = _compute_flux_weighted_centroid(tpf_data, fake_in_mask)
        out_centroid = _compute_flux_weighted_centroid(tpf_data, fake_out_mask)

        if np.isnan(in_centroid[0]) or np.isnan(out_centroid[0]):
            continue

        # Compute shift
        shift = np.sqrt(
            (in_centroid[0] - out_centroid[0]) ** 2 + (in_centroid[1] - out_centroid[1]) ** 2
        )
        bootstrap_shifts.append(shift)

    if len(bootstrap_shifts) < 10:
        return np.nan

    # Compute significance as (observed - mean) / std
    bootstrap_arr = np.array(bootstrap_shifts)
    mean_shift = np.mean(bootstrap_arr)
    std_shift = np.std(bootstrap_arr, ddof=1)

    if std_shift <= 0:
        return np.nan

    significance = (observed_shift - mean_shift) / std_shift
    return float(significance)


def _compute_shift_significance_analytic(
    tpf_data: NDArray[np.floating],
    in_transit_mask: NDArray[np.bool_],
    out_of_transit_mask: NDArray[np.bool_],
    in_centroid: tuple[float, float],
    out_centroid: tuple[float, float],
    observed_shift: float,
) -> float:
    """Compute significance of centroid shift using analytic formula.

    Uses error propagation assuming photon noise dominates.
    The standard error of the centroid is approximately:
    sigma_centroid ~ sigma_flux / (sqrt(N) * total_flux) * PSF_width

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

    Returns
    -------
    float
        Significance in units of standard deviations.
    """
    n_in = int(np.sum(in_transit_mask))
    n_out = int(np.sum(out_of_transit_mask))

    if n_in == 0 or n_out == 0:
        return np.nan

    # Compute variance of centroids for each group
    def compute_centroid_variance(
        data: NDArray[np.floating], mask: NDArray[np.bool_]
    ) -> tuple[float, float]:
        """Compute variance of centroid across cadences."""
        selected = data[mask]
        n_cadences = selected.shape[0]

        if n_cadences < 2:
            return (np.nan, np.nan)

        centroids_x = []
        centroids_y = []

        for i in range(n_cadences):
            frame = selected[i]
            total_flux = np.nansum(frame)
            if total_flux <= 0 or not np.isfinite(total_flux):
                continue

            n_rows, n_cols = frame.shape
            row_coords, col_coords = np.meshgrid(
                np.arange(n_rows, dtype=np.float64),
                np.arange(n_cols, dtype=np.float64),
                indexing="ij",
            )

            flux_weights = np.where(np.isfinite(frame), frame, 0.0)
            flux_weights = np.maximum(flux_weights, 0.0)
            weight_sum = np.sum(flux_weights)

            if weight_sum <= 0:
                continue

            cx = np.sum(col_coords * flux_weights) / weight_sum
            cy = np.sum(row_coords * flux_weights) / weight_sum
            centroids_x.append(cx)
            centroids_y.append(cy)

        if len(centroids_x) < 2:
            return (np.nan, np.nan)

        var_x = float(np.var(centroids_x, ddof=1))
        var_y = float(np.var(centroids_y, ddof=1))
        return (var_x, var_y)

    var_in_x, var_in_y = compute_centroid_variance(tpf_data, in_transit_mask)
    var_out_x, var_out_y = compute_centroid_variance(tpf_data, out_of_transit_mask)

    if any(np.isnan(v) for v in [var_in_x, var_in_y, var_out_x, var_out_y]):
        return np.nan

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
        return 0.0

    sigma_shift_sq = (diff_x / observed_shift) ** 2 * se_diff_x**2 + (
        diff_y / observed_shift
    ) ** 2 * se_diff_y**2

    sigma_shift = np.sqrt(sigma_shift_sq)

    if sigma_shift <= 0:
        return np.nan

    significance = observed_shift / sigma_shift
    return float(significance)


def compute_centroid_shift(
    tpf_data: NDArray[np.floating],
    time: NDArray[np.floating],
    transit_params: TransitParams,
    window_policy_version: str = "v1",
    significance_method: Literal["analytic", "bootstrap"] = "analytic",
    n_bootstrap: int = 1000,
    bootstrap_seed: int | None = None,
) -> CentroidResult:
    """Compute centroid shift between in-transit and out-of-transit cadences.

    This function computes the flux-weighted centroid during transit and
    outside of transit, then calculates the distance between them and
    the statistical significance of any shift.

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
    significance_method : {"analytic", "bootstrap"}, optional
        Method for computing significance. Default is "analytic".
        - "analytic": Uses error propagation from centroid scatter.
        - "bootstrap": Uses bootstrap resampling (slower but more robust).
    n_bootstrap : int, optional
        Number of bootstrap iterations if using bootstrap method.
        Default is 1000.
    bootstrap_seed : int, optional
        Random seed for bootstrap reproducibility.

    Returns
    -------
    CentroidResult
        Result containing centroid shift, significance, and metadata.

    Raises
    ------
    ValueError
        If window_policy_version is unknown or input shapes are invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from bittr_tess_vetter.pixel.centroid import (
    ...     compute_centroid_shift, TransitParams
    ... )
    >>> # Create synthetic TPF data
    >>> tpf = np.random.rand(100, 11, 11) * 1000
    >>> time = np.linspace(0, 10, 100)
    >>> params = TransitParams(period=2.5, t0=1.25, duration=3.0)
    >>> result = compute_centroid_shift(tpf, time, params)
    >>> print(f"Shift: {result.centroid_shift_pixels:.4f} pixels")
    """
    # Validate inputs
    if tpf_data.ndim != 3:
        raise ValueError(f"tpf_data must be 3D (time, rows, cols), got shape {tpf_data.shape}")

    if len(time) != tpf_data.shape[0]:
        raise ValueError(
            f"time length ({len(time)}) must match tpf_data first dimension ({tpf_data.shape[0]})"
        )

    if window_policy_version not in WINDOW_POLICIES:
        raise ValueError(
            f"Unknown window_policy_version: {window_policy_version}. "
            f"Available: {list(WINDOW_POLICIES.keys())}"
        )

    # Get window policy parameters
    policy = WINDOW_POLICIES[window_policy_version]
    k_in = policy["k_in"]
    k_buffer = policy["k_buffer"]

    # Compute transit masks
    in_transit_mask, out_of_transit_mask = _get_transit_masks(time, transit_params, k_in, k_buffer)

    n_in_transit = int(np.sum(in_transit_mask))
    n_out_transit = int(np.sum(out_of_transit_mask))

    # Compute centroids
    in_centroid = _compute_flux_weighted_centroid(tpf_data, in_transit_mask)
    out_centroid = _compute_flux_weighted_centroid(tpf_data, out_of_transit_mask)

    # Handle edge cases where centroids cannot be computed
    if np.isnan(in_centroid[0]) or np.isnan(out_centroid[0]):
        return CentroidResult(
            centroid_shift_pixels=np.nan,
            significance_sigma=np.nan,
            in_transit_centroid=in_centroid,
            out_of_transit_centroid=out_centroid,
            n_in_transit_cadences=n_in_transit,
            n_out_transit_cadences=n_out_transit,
        )

    # Compute shift distance
    shift = np.sqrt(
        (in_centroid[0] - out_centroid[0]) ** 2 + (in_centroid[1] - out_centroid[1]) ** 2
    )

    # Compute significance
    if significance_method == "bootstrap":
        rng = np.random.default_rng(bootstrap_seed) if bootstrap_seed is not None else None
        significance = _compute_shift_significance_bootstrap(
            tpf_data,
            in_transit_mask,
            out_of_transit_mask,
            shift,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
    else:  # analytic
        significance = _compute_shift_significance_analytic(
            tpf_data,
            in_transit_mask,
            out_of_transit_mask,
            in_centroid,
            out_centroid,
            shift,
        )

    return CentroidResult(
        centroid_shift_pixels=float(shift),
        significance_sigma=float(significance),
        in_transit_centroid=in_centroid,
        out_of_transit_centroid=out_centroid,
        n_in_transit_cadences=n_in_transit,
        n_out_transit_cadences=n_out_transit,
    )
