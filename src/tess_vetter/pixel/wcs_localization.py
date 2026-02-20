"""WCS-aware transit source localization with bootstrap uncertainty.

This module provides algorithms for localizing the source of a transit signal
in TESS Target Pixel Files using WCS coordinates and difference imaging.

Key features:
- Difference image centroid computation (in-transit vs out-of-transit)
- Bootstrap resampling for uncertainty ellipse estimation
- WCS-aware coordinate transforms (pixel <-> sky)
- Angular distance computation to reference sources (e.g., Gaia neighbors)
- Localization verdict with machine-readable rationale

The primary entry point is `localize_transit_source()`, which combines all
the above into a complete localization analysis.

Example:
    from tess_vetter.pixel.wcs_localization import localize_transit_source

    result = localize_transit_source(
        tpf_fits=tpf_data,
        period=5.0,
        t0=1345.0,
        duration_hours=4.0,
        reference_sources=[
            {"name": "TIC 123456789", "ra": 120.0, "dec": -50.0},
            {"name": "Gaia DR3 ...", "ra": 120.001, "dec": -50.001},
        ],
        bootstrap_draws=500,
        bootstrap_seed=42,
    )

    print(result.verdict)  # LocalizationVerdict.ON_TARGET
    print(result.distances_to_sources)  # {"TIC 123456789": 1.2, "Gaia DR3 ...": 15.3}

References:
    - Twicken et al. 2018 (2018PASP..130f4502T): difference image centroid offsets (Kepler DV)
    - Bryson et al. 2013 (2013PASP..125..889B): background false positive localization diagnostics
    - Greisen & Calabretta 2002 (2002A&A...395.1061G), Calabretta & Greisen 2002 (2002A&A...395.1077C):
      FITS WCS conventions used for sky↔pixel transforms (via astropy.wcs)
    - Astropy Collaboration 2013 (2013A&A...558A..33A): astropy.wcs implementation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.optimize import minimize

from tess_vetter.pixel.cadence_mask import default_cadence_mask
from tess_vetter.pixel.wcs_utils import (
    compute_pixel_scale,
    compute_source_distances,
    get_reference_source_pixel_positions,
    pixel_to_world,
)

if TYPE_CHECKING:
    from tess_vetter.pixel.tpf_fits import TPFFitsData

logger = logging.getLogger(__name__)


class LocalizationVerdict(str, Enum):
    """Machine-readable localization verdict.

    Values:
        ON_TARGET: Centroid is consistent with the target position.
        OFF_TARGET: Centroid is significantly offset from target.
        AMBIGUOUS: Cannot distinguish between multiple nearby sources.
        INVALID: Localization failed due to data quality issues.
    """

    ON_TARGET = "ON_TARGET"
    OFF_TARGET = "OFF_TARGET"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID = "INVALID"


@dataclass(frozen=True)
class LocalizationResult:
    """WCS-aware localization result with uncertainty quantification.

    All angular measurements are in arcseconds unless otherwise noted.
    All coordinates follow the TESS convention (row, col) = (y, x).

    Attributes:
        centroid_pixel_rc: Centroid position in pixel coordinates (row, col).
        centroid_sky_ra: Centroid Right Ascension in degrees.
        centroid_sky_dec: Centroid Declination in degrees.
        uncertainty_semimajor_arcsec: Semi-major axis of uncertainty ellipse.
        uncertainty_semiminor_arcsec: Semi-minor axis of uncertainty ellipse.
        uncertainty_pa_deg: Position angle of uncertainty ellipse (E of N).
        distances_to_sources: Angular distances from centroid to each reference
            source, in arcseconds. Keys are source names.
        wcs_source_positions: Pixel positions of reference sources, as (row, col).
        n_bootstrap_draws: Number of bootstrap draws used.
        bootstrap_seed: Random seed for reproducibility.
        verdict: Machine-readable localization verdict.
        verdict_rationale: List of reasons supporting the verdict.
        warnings: List of warning messages (saturation, instability, etc.).
        extra: Additional diagnostic information.
    """

    centroid_pixel_rc: tuple[float, float]
    centroid_sky_ra: float
    centroid_sky_dec: float
    uncertainty_semimajor_arcsec: float
    uncertainty_semiminor_arcsec: float
    uncertainty_pa_deg: float
    distances_to_sources: dict[str, float]
    wcs_source_positions: dict[str, tuple[float, float]]
    n_bootstrap_draws: int
    bootstrap_seed: int
    verdict: LocalizationVerdict
    verdict_rationale: list[str]
    warnings: list[str]
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "centroid_pixel_rc": list(self.centroid_pixel_rc),
            "centroid_sky_ra": self.centroid_sky_ra,
            "centroid_sky_dec": self.centroid_sky_dec,
            "uncertainty_semimajor_arcsec": self.uncertainty_semimajor_arcsec,
            "uncertainty_semiminor_arcsec": self.uncertainty_semiminor_arcsec,
            "uncertainty_pa_deg": self.uncertainty_pa_deg,
            "distances_to_sources": self.distances_to_sources,
            "wcs_source_positions": {
                name: list(pos) for name, pos in self.wcs_source_positions.items()
            },
            "n_bootstrap_draws": self.n_bootstrap_draws,
            "bootstrap_seed": self.bootstrap_seed,
            "verdict": self.verdict.value,
            "verdict_rationale": self.verdict_rationale,
            "warnings": self.warnings,
            "extra": self.extra,
        }


# =============================================================================
# Transit mask computation
# =============================================================================


def _compute_transit_mask(
    time: np.ndarray[Any, np.dtype[np.floating[Any]]],
    period: float,
    t0: float,
    duration_days: float,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Compute boolean mask indicating in-transit times.

    Args:
        time: Array of observation times (BTJD).
        period: Orbital period in days.
        t0: Transit epoch in BTJD.
        duration_days: Full transit duration in days.

    Returns:
        Boolean mask where True indicates in-transit cadences.
    """
    phase = ((time - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    half_duration_phase = (duration_days / 2) / period
    mask: np.ndarray[Any, np.dtype[np.bool_]] = np.abs(phase) <= half_duration_phase
    return mask


def _compute_out_of_transit_mask(
    time: np.ndarray[Any, np.dtype[np.floating[Any]]],
    period: float,
    t0: float,
    duration_days: float,
    oot_margin_mult: float = 1.5,
    oot_window_mult: float | None = 10.0,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Compute boolean mask indicating out-of-transit times with margin.

    Excludes times within oot_margin_mult * duration of transit center.

    By default this uses a *local window* around each transit to avoid
    biasing the baseline with long-term stellar/instrument variability.

    Args:
        time: Array of observation times (BTJD).
        period: Orbital period in days.
        t0: Transit epoch in BTJD.
        duration_days: Full transit duration in days.
        oot_margin_mult: Multiplier for exclusion zone around transit.
        oot_window_mult: Window half-width in units of duration for selecting
            out-of-transit points near each transit. If None, uses all cadences
            outside the exclusion zone (global baseline).

    Returns:
        Boolean mask where True indicates safe out-of-transit cadences.
    """
    phase = ((time - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    exclusion_phase = (duration_days * oot_margin_mult / 2) / period
    abs_phase = np.abs(phase)
    if oot_window_mult is None:
        mask: np.ndarray[Any, np.dtype[np.bool_]] = abs_phase > exclusion_phase
        return mask

    window_phase = (duration_days * float(oot_window_mult) / 2.0) / period
    window_phase = float(min(0.5, max(window_phase, exclusion_phase)))
    mask = (abs_phase > exclusion_phase) & (abs_phase <= window_phase)
    return mask


# =============================================================================
# Centroid computation
# =============================================================================


def _flux_weighted_centroid(
    image: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> tuple[float, float]:
    """Compute flux-weighted centroid of an image.

    Args:
        image: 2D flux array.

    Returns:
        (row, col) centroid coordinates.
    """
    img = np.asarray(image, dtype=np.float64)
    img = np.where(np.isfinite(img), img, 0.0)

    # Difference images can contain negative values (systematics/background).
    # For localization, we only want the positive "dimming" signal.
    img = np.clip(img, 0.0, None)

    if img.size == 0:
        return (float("nan"), float("nan"))

    # Peak pixel (used for a robustness refinement pass below).
    r_peak, c_peak = np.unravel_index(int(np.argmax(img)), img.shape)

    total = float(np.sum(img))
    if total <= 0:
        # Fall back to brightest pixel if there is no positive mass.
        # This avoids NaNs in real TESS data where the sum can cancel.
        return (float(r_peak), float(c_peak))

    n_rows, n_cols = img.shape
    row_grid, col_grid = np.mgrid[0:n_rows, 0:n_cols]

    row_centroid = float(np.sum(row_grid * img) / total)
    col_centroid = float(np.sum(col_grid * img) / total)

    # Robustness refinement:
    # In real TESS data, difference images can include low-level positive structure
    # (e.g., background gradients/PRF wings) that pulls the flux-weighted centroid
    # away from the true transit signal peak. If the centroid is far from the peak,
    # recompute using only the high-signal pixels near the peak.
    try:
        dist_to_peak = float(np.hypot(row_centroid - float(r_peak), col_centroid - float(c_peak)))
    except Exception:
        dist_to_peak = 0.0

    if dist_to_peak > 0.75:
        peak = float(np.max(img))
        if peak > 0:
            # Use the high-signal core of the difference image to reduce centroid bias.
            # Adapt the threshold so we include at least a couple of pixels (avoid single-pixel noise spikes).
            mask = None
            for frac in (0.5, 0.3, 0.2):
                m = img >= (float(frac) * peak)
                if int(np.sum(m)) >= 2:
                    mask = m
                    break
            if mask is not None:
                img2 = np.where(mask, img, 0.0)
                total2 = float(np.sum(img2))
                if total2 > 0:
                    row_centroid = float(np.sum(row_grid * img2) / total2)
                    col_centroid = float(np.sum(col_grid * img2) / total2)

    return (row_centroid, col_centroid)


def _gaussian_fit_centroid(
    image: np.ndarray[Any, np.dtype[np.floating[Any]]],
    initial_guess: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Fit a 2D Gaussian to an image and return the center.

    Args:
        image: 2D flux array.
        initial_guess: Optional (row, col) initial guess. If None, uses
            flux-weighted centroid as initial guess.

    Returns:
        (row, col) Gaussian center coordinates.
    """
    img = np.asarray(image, dtype=np.float64)
    img = np.where(np.isfinite(img), img, 0.0)
    img = np.clip(img, 0.0, None)

    n_rows, n_cols = img.shape
    row_grid, col_grid = np.mgrid[0:n_rows, 0:n_cols]

    # Get initial guess from flux-weighted centroid
    if initial_guess is None:
        initial_guess = _flux_weighted_centroid(img)

    if not np.isfinite(initial_guess[0]) or not np.isfinite(initial_guess[1]):
        return (float("nan"), float("nan"))

    # Flatten for optimization
    row_flat = row_grid.ravel()
    col_flat = col_grid.ravel()
    img_flat = img.ravel()

    # Initial parameters: [amplitude, row_center, col_center, sigma]
    amplitude_guess = float(np.max(img))
    sigma_guess = 1.5

    def gaussian_residual(params: np.ndarray) -> float:
        amp, r0, c0, sigma = params
        if sigma <= 0:
            return 1e10
        model = amp * np.exp(-((row_flat - r0) ** 2 + (col_flat - c0) ** 2) / (2 * sigma**2))
        return float(np.sum((img_flat - model) ** 2))

    x0 = np.array([amplitude_guess, initial_guess[0], initial_guess[1], sigma_guess])

    try:
        result = minimize(
            gaussian_residual,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 1000},
        )
        if result.success:
            _, row_center, col_center, _ = result.x
            return (float(row_center), float(col_center))
    except Exception:
        pass

    # Fall back to flux-weighted centroid
    return initial_guess


# =============================================================================
# Difference image computation
# =============================================================================


def compute_difference_image_centroid(
    *,
    tpf_fits: TPFFitsData,
    period: float,
    t0: float,
    duration_hours: float,
    oot_margin_mult: float = 1.5,
    oot_window_mult: float | None = 10.0,
    method: Literal["centroid", "gaussian_fit"] = "centroid",
) -> tuple[tuple[float, float], np.ndarray[Any, np.dtype[np.floating[Any]]]]:
    """Compute difference image and centroid in pixel coordinates.

    The difference image is computed as:
        diff = median(out_of_transit) - median(in_transit)

    Since transit causes dimming, the difference image shows positive flux
    where the transit is localized.

    Args:
        tpf_fits: TPF FITS data with flux cube and WCS.
        period: Orbital period in days.
        t0: Transit epoch in BTJD.
        duration_hours: Full transit duration in hours.
        oot_margin_mult: Multiplier for out-of-transit exclusion zone.
        method: Centroid method - "centroid" for flux-weighted, "gaussian_fit"
            for Gaussian fit.

    Returns:
        Tuple of:
            - (row, col) centroid of the difference image
            - 2D difference image array

    Raises:
        ValueError: If insufficient in-transit or out-of-transit data.
    """
    cadence_mask = default_cadence_mask(
        time=tpf_fits.time, flux=tpf_fits.flux, quality=tpf_fits.quality
    )
    time = tpf_fits.time[cadence_mask]
    flux = tpf_fits.flux[cadence_mask]
    duration_days = duration_hours / 24.0

    # Compute masks
    in_transit_mask = _compute_transit_mask(time, period, t0, duration_days)
    out_of_transit_mask = _compute_out_of_transit_mask(
        time, period, t0, duration_days, oot_margin_mult, oot_window_mult
    )

    n_in_transit = int(np.sum(in_transit_mask))
    n_out_of_transit = int(np.sum(out_of_transit_mask))

    if n_in_transit < 3:
        raise ValueError(
            f"Insufficient in-transit data points ({n_in_transit}). "
            "Check ephemeris against time array coverage."
        )

    if n_out_of_transit < 3:
        raise ValueError(
            f"Insufficient out-of-transit data points ({n_out_of_transit}). "
            "Transit duration may be too long relative to period."
        )

    # Compute median images (NaN-robust)
    in_transit_image = np.nanmedian(flux[in_transit_mask], axis=0)
    out_of_transit_image = np.nanmedian(flux[out_of_transit_mask], axis=0)

    # Difference: out - in (transit causes dimming)
    difference_image = out_of_transit_image - in_transit_image

    # Compute centroid
    if method == "gaussian_fit":
        centroid = _gaussian_fit_centroid(difference_image)
    else:
        centroid = _flux_weighted_centroid(difference_image)

    return (centroid, difference_image.astype(np.float64))


# =============================================================================
# Bootstrap uncertainty estimation
# =============================================================================


def _fit_ellipse_to_points(
    points: np.ndarray[Any, np.dtype[np.floating[Any]]],
    pixel_scale_arcsec: float,
) -> tuple[float, float, float]:
    """Fit an uncertainty ellipse to a set of 2D points.

    Uses the covariance matrix eigendecomposition to determine the
    ellipse parameters.

    Args:
        points: Array of shape (N, 2) with (row, col) coordinates.
        pixel_scale_arcsec: Pixel scale in arcseconds per pixel.

    Returns:
        Tuple of (semimajor_arcsec, semiminor_arcsec, pa_deg).
        Position angle is measured East of North.
    """
    if len(points) < 3:
        return (float("nan"), float("nan"), float("nan"))

    # Remove any NaN points
    valid_mask = np.all(np.isfinite(points), axis=1)
    points = points[valid_mask]

    if len(points) < 3:
        return (float("nan"), float("nan"), float("nan"))

    # Compute covariance matrix
    cov = np.cov(points.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Semi-axes in pixels (1-sigma)
    semimajor_px = float(np.sqrt(eigenvalues[0]))
    semiminor_px = float(np.sqrt(eigenvalues[1]))

    # Convert to arcseconds
    semimajor_arcsec = semimajor_px * pixel_scale_arcsec
    semiminor_arcsec = semiminor_px * pixel_scale_arcsec

    # Position angle (angle of major axis from +col direction, measured CCW)
    # In TESS convention, +row is approximately +Dec, +col is approximately -RA
    # Position angle is typically measured E of N (CCW from +Dec)
    major_axis = eigenvectors[:, 0]  # (row_component, col_component)
    pa_rad = np.arctan2(major_axis[1], major_axis[0])  # angle from +row (N)
    pa_deg = float(np.degrees(pa_rad))

    # Normalize PA to [0, 180) since ellipse is symmetric
    pa_deg = pa_deg % 180.0

    return (semimajor_arcsec, semiminor_arcsec, pa_deg)


def bootstrap_centroid_uncertainty(
    *,
    tpf_fits: TPFFitsData,
    period: float,
    t0: float,
    duration_hours: float,
    n_draws: int = 500,
    seed: int | None = None,
    oot_margin_mult: float = 1.5,
    oot_window_mult: float | None = 10.0,
    method: Literal["centroid", "gaussian_fit"] = "centroid",
) -> tuple[
    np.ndarray[Any, np.dtype[np.floating[Any]]],
    float,
    float,
    float,
]:
    """Estimate centroid uncertainty via bootstrap resampling.

    Resamples the in-transit cadences with replacement, recomputes the
    difference image and centroid, and fits an ellipse to the distribution
    of centroids.

    Args:
        tpf_fits: TPF FITS data with flux cube and WCS.
        period: Orbital period in days.
        t0: Transit epoch in BTJD.
        duration_hours: Full transit duration in hours.
        n_draws: Number of bootstrap draws.
        seed: Random seed for reproducibility. If None, uses random seed.
        oot_margin_mult: Multiplier for out-of-transit exclusion zone.
        oot_window_mult: Window half-width in units of duration for selecting
            out-of-transit points near each transit. If None, uses a global baseline.
        method: Centroid method for each bootstrap draw.

    Returns:
        Tuple of:
            - centroids array of shape (n_draws, 2) with (row, col) for each draw
            - semimajor_arcsec: Semi-major axis of uncertainty ellipse
            - semiminor_arcsec: Semi-minor axis of uncertainty ellipse
            - pa_deg: Position angle of ellipse (E of N)
    """
    rng = np.random.default_rng(seed)
    cadence_mask = default_cadence_mask(
        time=tpf_fits.time, flux=tpf_fits.flux, quality=tpf_fits.quality
    )
    time = tpf_fits.time[cadence_mask]
    flux = tpf_fits.flux[cadence_mask]
    duration_days = duration_hours / 24.0

    # Get pixel scale for converting to arcseconds
    pixel_scale_arcsec = compute_pixel_scale(tpf_fits.wcs)

    # Compute masks
    in_transit_mask = _compute_transit_mask(time, period, t0, duration_days)
    out_of_transit_mask = _compute_out_of_transit_mask(
        time, period, t0, duration_days, oot_margin_mult, oot_window_mult
    )

    # Get indices
    in_transit_indices = np.where(in_transit_mask)[0]
    out_of_transit_indices = np.where(out_of_transit_mask)[0]

    n_in = len(in_transit_indices)
    n_out = len(out_of_transit_indices)

    if n_in < 3 or n_out < 3:
        # Cannot bootstrap with insufficient data
        empty_centroids = np.full((n_draws, 2), np.nan)
        return (empty_centroids.astype(np.float64), float("nan"), float("nan"), float("nan"))

    # Fixed out-of-transit median (don't resample OOT)
    out_of_transit_image = np.nanmedian(flux[out_of_transit_mask], axis=0)

    # Bootstrap resampling of in-transit cadences
    centroids = np.zeros((n_draws, 2), dtype=np.float64)

    for i in range(n_draws):
        # Resample in-transit indices with replacement
        resampled_indices = rng.choice(in_transit_indices, size=n_in, replace=True)
        in_transit_image = np.nanmedian(flux[resampled_indices], axis=0)

        # Compute difference image
        difference_image = out_of_transit_image - in_transit_image

        # Compute centroid
        if method == "gaussian_fit":
            centroid = _gaussian_fit_centroid(difference_image)
        else:
            centroid = _flux_weighted_centroid(difference_image)

        centroids[i, 0] = centroid[0]
        centroids[i, 1] = centroid[1]

    # Fit ellipse to centroid distribution
    semimajor, semiminor, pa = _fit_ellipse_to_points(centroids, pixel_scale_arcsec)

    return (centroids, semimajor, semiminor, pa)


# =============================================================================
# Reference source distances
# =============================================================================


def compute_reference_source_distances(
    *,
    centroid_sky: tuple[float, float],
    reference_sources: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute angular distances from centroid to reference sources.

    Args:
        centroid_sky: (RA, Dec) of the centroid in degrees.
        reference_sources: List of dicts with 'name', 'ra', 'dec' keys.

    Returns:
        Dictionary mapping source names to distances in arcseconds.
    """
    return compute_source_distances(
        centroid_ra_deg=centroid_sky[0],
        centroid_dec_deg=centroid_sky[1],
        reference_sources=reference_sources,
    )


# =============================================================================
# Saturation detection
# =============================================================================


def _detect_saturation(
    flux: np.ndarray[Any, np.dtype[np.floating[Any]]],
    near_max_fraction: float = 0.999,
    min_pixels_near_max: int = 3,
    min_cadence_fraction: float = 0.02,
) -> bool:
    """Detect if significant portion of pixels may be saturated.

    Args:
        flux: 3D flux array (n_cadences, n_rows, n_cols).
        near_max_fraction: Per-cadence threshold as a fraction of that cadence's
            maximum pixel value. Pixels within this fraction are considered
            "near max" and may indicate clipping/bleed.
        min_pixels_near_max: Minimum number of pixels near the cadence max to
            consider that cadence "plateaued".
        min_cadence_fraction: Fraction of cadences that must be plateaued to
            trigger a saturation warning.

    Returns:
        True if saturation is suspected.
    """
    return bool(
        _saturation_plateau_metrics(
            flux,
            near_max_fraction=near_max_fraction,
            min_pixels_near_max=min_pixels_near_max,
            min_cadence_fraction=min_cadence_fraction,
        )["suspected"]
    )


def _saturation_plateau_metrics(
    flux: np.ndarray[Any, np.dtype[np.floating[Any]]],
    *,
    near_max_fraction: float = 0.999,
    min_pixels_near_max: int = 3,
    min_cadence_fraction: float = 0.02,
) -> dict[str, float | bool]:
    """Return diagnostics for plateau-based saturation heuristic.

    The main warning signal is `suspected=True`, which indicates that an
    unusual fraction of cadences have multiple pixels at (or extremely near)
    the per-cadence maximum, consistent with saturation/bleed.

    Returns a small dict so callers can log/tune thresholds without inspecting
    the full flux cube.
    """
    # The earlier implementation used a per-cadence max percentile heuristic,
    # which can false-positive on stable, unsaturated stars (max values are
    # naturally concentrated). Here we look for a *plateau* signature: multiple
    # pixels per cadence sitting at (or extremely near) the cadence maximum,
    # which is a common consequence of saturation/bleed.
    diagnostics: dict[str, float | bool] = {
        "suspected": False,
        "near_max_fraction": float(near_max_fraction),
        "min_pixels_near_max": float(min_pixels_near_max),
        "min_cadence_fraction": float(min_cadence_fraction),
        "cadence_fraction_plateaued": float("nan"),
        "median_pixels_near_max": float("nan"),
        "n_cadences": float(getattr(flux, "shape", [0])[0] if hasattr(flux, "shape") else 0),
    }

    if flux.ndim != 3:
        return diagnostics

    n_cadences = flux.shape[0]
    if n_cadences < 10:
        diagnostics["n_cadences"] = float(n_cadences)
        return diagnostics
    diagnostics["n_cadences"] = float(n_cadences)

    flat = flux.reshape(n_cadences, -1).astype(np.float64, copy=False)
    flat = np.where(np.isfinite(flat), flat, np.nan)

    # Per-cadence maximum pixel.
    cad_max = np.nanmax(flat, axis=1)
    if not np.isfinite(np.nanmedian(cad_max)):
        return diagnostics

    # Count pixels near the per-cadence maximum.
    # Guard against cad_max==0 producing a trivial near-max mask.
    denom = np.where(cad_max > 0, cad_max, np.nan)
    near = flat >= (denom[:, None] * float(near_max_fraction))
    count_near = np.nansum(near, axis=1)

    plateaued = count_near >= int(min_pixels_near_max)
    frac_plateaued = float(np.sum(plateaued)) / float(n_cadences)

    diagnostics["cadence_fraction_plateaued"] = float(frac_plateaued)
    diagnostics["median_pixels_near_max"] = float(np.nanmedian(count_near))
    diagnostics["suspected"] = bool(frac_plateaued >= float(min_cadence_fraction))
    return diagnostics


# =============================================================================
# Verdict determination
# =============================================================================


def _determine_verdict(
    *,
    centroid_pixel_rc: tuple[float, float],
    distances_to_sources: dict[str, float],
    uncertainty_semimajor_arcsec: float,
    pixel_scale_arcsec: float,
    warnings: list[str],
    reference_sources: list[dict[str, Any]],
) -> tuple[LocalizationVerdict, list[str]]:
    """Determine localization verdict and rationale.

    Args:
        centroid_pixel_rc: Centroid position in pixels.
        distances_to_sources: Distances to reference sources in arcsec.
        uncertainty_semimajor_arcsec: 1-sigma uncertainty in arcsec.
        pixel_scale_arcsec: Pixel scale in arcseconds.
        warnings: List of warnings accumulated so far.
        reference_sources: List of reference sources.

    Returns:
        Tuple of (verdict, rationale_list).
    """
    rationale: list[str] = []

    # Check for invalid conditions
    if not np.isfinite(centroid_pixel_rc[0]) or not np.isfinite(centroid_pixel_rc[1]):
        return (LocalizationVerdict.INVALID, ["Centroid computation failed"])

    if not np.isfinite(uncertainty_semimajor_arcsec):
        rationale.append("Uncertainty estimation failed (using 1-pixel fallback)")

    # Check for saturation warning
    if "saturation_suspected" in warnings:
        rationale.append("Saturation suspected - centroid may be unreliable")

    # Find the target (typically the first reference source or one named "target")
    target_name: str | None = None
    for src in reference_sources:
        name = src.get("name", "")
        if "target" in name.lower() or src == reference_sources[0]:
            target_name = name
            break

    if target_name is None and reference_sources:
        target_name = reference_sources[0].get("name")

    # Get distance to target
    target_distance = (
        distances_to_sources.get(target_name, float("nan")) if target_name else float("nan")
    )

    # Define thresholds
    # 1 TESS pixel = ~21 arcsec
    one_pixel_arcsec = pixel_scale_arcsec
    # Within 1 sigma of target = ON_TARGET
    # More than 2 sigma from target = OFF_TARGET
    # Between 1-2 sigma = AMBIGUOUS

    if not np.isfinite(target_distance):
        rationale.append("Could not compute distance to target")
        if warnings:
            return (LocalizationVerdict.INVALID, rationale)
        return (LocalizationVerdict.AMBIGUOUS, rationale)

    # Use uncertainty for verdict (default to 1 pixel if uncertainty invalid)
    sigma = (
        uncertainty_semimajor_arcsec
        if np.isfinite(uncertainty_semimajor_arcsec)
        else one_pixel_arcsec
    )

    # Check if centroid is within uncertainty of target
    if target_distance <= sigma:
        rationale.append(
            f'Centroid within 1-sigma ({sigma:.1f}") of target ({target_distance:.1f}")'
        )
        verdict = LocalizationVerdict.ON_TARGET
    elif target_distance <= 2 * sigma:
        rationale.append(
            f'Centroid within 2-sigma ({2 * sigma:.1f}") of target ({target_distance:.1f}")'
        )
        verdict = LocalizationVerdict.AMBIGUOUS
    else:
        rationale.append(
            f'Centroid >2-sigma ({2 * sigma:.1f}") from target ({target_distance:.1f}")'
        )
        verdict = LocalizationVerdict.OFF_TARGET

    # Check for ambiguity with other sources
    other_sources_within_sigma = []
    for name, dist in distances_to_sources.items():
        if name != target_name and dist <= sigma:
            other_sources_within_sigma.append((name, dist))

    if other_sources_within_sigma:
        rationale.append(
            f"Other source(s) within 1-sigma: {', '.join(n for n, _ in other_sources_within_sigma)}"
        )
        # If multiple sources within sigma, verdict is ambiguous
        if verdict == LocalizationVerdict.ON_TARGET and len(other_sources_within_sigma) > 0:
            # Check if other source is closer than target
            for name, dist in other_sources_within_sigma:
                if dist < target_distance:
                    rationale.append(f'Source {name} ({dist:.1f}") is closer than target')
                    verdict = LocalizationVerdict.AMBIGUOUS
                    break

    # If uncertainty is larger than separation to nearest other source, flag ambiguity
    if len(distances_to_sources) > 1:
        min_other_dist = (
            min(d for n, d in distances_to_sources.items() if n != target_name)
            if any(n != target_name for n in distances_to_sources)
            else float("inf")
        )

        if np.isfinite(sigma) and min_other_dist < sigma:
            rationale.append(
                f'Cannot disambiguate - uncertainty ({sigma:.1f}") exceeds nearest neighbor separation ({min_other_dist:.1f}")'
            )
            if verdict == LocalizationVerdict.ON_TARGET:
                verdict = LocalizationVerdict.AMBIGUOUS

    # Avoid over-confident OFF_TARGET claims within ~1 TESS pixel of the target unless a
    # specific alternative hypothesis is clearly closer. This addresses cases where
    # small WCS/centroid systematics can produce ~1-pixel apparent offsets.
    if verdict == LocalizationVerdict.OFF_TARGET and target_distance <= one_pixel_arcsec:
        margin_arcsec = 0.25 * one_pixel_arcsec
        closest_other_name: str | None = None
        closest_other_dist = float("inf")
        for name, dist in distances_to_sources.items():
            if name == target_name:
                continue
            if float(dist) < closest_other_dist:
                closest_other_dist = float(dist)
                closest_other_name = name

        if (closest_other_name is None) or not (
            np.isfinite(closest_other_dist)
            and (closest_other_dist + margin_arcsec) < target_distance
        ):
            rationale.append(
                f'OFF_TARGET downgraded: centroid is within 1 pixel ({one_pixel_arcsec:.1f}") of target '
                "and no clearly closer alternative source is present."
            )
            verdict = LocalizationVerdict.AMBIGUOUS

    return (verdict, rationale)


# =============================================================================
# Main localization function
# =============================================================================


def localize_transit_source(
    *,
    tpf_fits: TPFFitsData,
    period: float,
    t0: float,
    duration_hours: float,
    reference_sources: list[dict[str, Any]],
    bootstrap_draws: int = 500,
    bootstrap_seed: int | None = None,
    oot_margin_mult: float = 1.5,
    oot_window_mult: float | None = 10.0,
    method: Literal["centroid", "gaussian_fit"] = "centroid",
) -> LocalizationResult:
    """Full WCS-aware transit source localization with uncertainty.

    This is the main entry point for transit localization. It:
    1. Computes the difference image (out-of-transit - in-transit)
    2. Finds the centroid of the difference image
    3. Converts to sky coordinates using WCS
    4. Bootstraps the in-transit frames to estimate uncertainty
    5. Computes distances to reference sources (e.g., Gaia neighbors)
    6. Determines a verdict (ON_TARGET, OFF_TARGET, AMBIGUOUS, INVALID)

    Args:
        tpf_fits: TPF FITS data with flux cube and WCS.
        period: Orbital period in days.
        t0: Transit epoch in BTJD.
        duration_hours: Full transit duration in hours.
        reference_sources: List of reference sources, each a dict with:
            - "name": Source identifier (e.g., "TIC 123456789", "Gaia DR3 ...")
            - "ra": Right Ascension in degrees
            - "dec": Declination in degrees
        bootstrap_draws: Number of bootstrap draws for uncertainty. Set to 0
            for fast mode (no uncertainty estimation).
        bootstrap_seed: Random seed for reproducibility. If None, a random
            seed is generated and recorded.
        oot_margin_mult: Multiplier for out-of-transit exclusion zone.
        oot_window_mult: Window half-width in units of duration for selecting
            out-of-transit points near each transit. If None, uses a global baseline.
        method: Centroid method - "centroid" for flux-weighted, "gaussian_fit"
            for Gaussian fit.

    Returns:
        LocalizationResult with centroid, uncertainty, distances, and verdict.

    Example:
        >>> result = localize_transit_source(
        ...     tpf_fits=tpf_data,
        ...     period=5.0,
        ...     t0=1345.0,
        ...     duration_hours=4.0,
        ...     reference_sources=[
        ...         {"name": "Target", "ra": 120.0, "dec": -50.0},
        ...         {"name": "Neighbor", "ra": 120.003, "dec": -50.0},
        ...     ],
        ...     bootstrap_seed=42,
        ... )
        >>> print(result.verdict)
        LocalizationVerdict.ON_TARGET
    """
    warnings: list[str] = []
    extra: dict[str, Any] = {}

    cadence_mask = default_cadence_mask(
        time=tpf_fits.time, flux=tpf_fits.flux, quality=tpf_fits.quality
    )
    n_total = int(tpf_fits.time.shape[0])
    n_used = int(np.sum(cadence_mask))
    extra["n_cadences_total"] = n_total
    extra["n_cadences_used"] = n_used
    extra["n_cadences_dropped"] = n_total - n_used
    if n_used < n_total:
        warnings.append(f"dropped_bad_cadences:{n_total - n_used}")

    # Use a cadence-filtered view for downstream computations.
    tpf_used = tpf_fits
    if n_used < n_total:
        from tess_vetter.pixel.tpf_fits import TPFFitsData

        tpf_used = TPFFitsData(
            ref=tpf_fits.ref,
            time=tpf_fits.time[cadence_mask],
            flux=tpf_fits.flux[cadence_mask],
            flux_err=None if tpf_fits.flux_err is None else tpf_fits.flux_err[cadence_mask],
            wcs=tpf_fits.wcs,
            aperture_mask=tpf_fits.aperture_mask,
            quality=tpf_fits.quality[cadence_mask],
            camera=tpf_fits.camera,
            ccd=tpf_fits.ccd,
            meta=tpf_fits.meta,
        )

    # Generate seed if not provided
    if bootstrap_seed is None:
        bootstrap_seed = int(np.random.default_rng().integers(0, 2**31))

    # Get pixel scale
    pixel_scale_arcsec = compute_pixel_scale(tpf_used.wcs)

    # Check for saturation
    saturation = _saturation_plateau_metrics(tpf_used.flux)
    extra["saturation"] = saturation
    if bool(saturation.get("suspected")):
        warnings.append("saturation_suspected")

    # Compute difference image and centroid
    try:
        centroid_pixel_rc, diff_image = compute_difference_image_centroid(
            tpf_fits=tpf_used,
            period=period,
            t0=t0,
            duration_hours=duration_hours,
            oot_margin_mult=oot_margin_mult,
            oot_window_mult=oot_window_mult,
            method=method,
        )
    except ValueError as e:
        # Return invalid result
        return LocalizationResult(
            centroid_pixel_rc=(float("nan"), float("nan")),
            centroid_sky_ra=float("nan"),
            centroid_sky_dec=float("nan"),
            uncertainty_semimajor_arcsec=float("nan"),
            uncertainty_semiminor_arcsec=float("nan"),
            uncertainty_pa_deg=float("nan"),
            distances_to_sources={},
            wcs_source_positions={},
            n_bootstrap_draws=0,
            bootstrap_seed=bootstrap_seed,
            verdict=LocalizationVerdict.INVALID,
            verdict_rationale=[f"Difference image computation failed: {e}"],
            warnings=warnings,
            extra={"error": str(e)},
        )

    # -------------------------------------------------------------------------
    # Difference-image quality / significance metrics (metrics-only)
    # -------------------------------------------------------------------------
    duration_days = duration_hours / 24.0
    in_transit_mask = _compute_transit_mask(tpf_used.time, period, t0, duration_days)
    out_of_transit_mask = _compute_out_of_transit_mask(
        tpf_used.time, period, t0, duration_days, oot_margin_mult, oot_window_mult
    )

    n_in_transit = int(np.sum(in_transit_mask))
    n_out_of_transit = int(np.sum(out_of_transit_mask))
    extra["n_in_transit"] = n_in_transit
    extra["n_out_of_transit"] = n_out_of_transit
    extra["baseline_mode"] = "global" if oot_window_mult is None else "local"

    try:
        diff_median = float(np.nanmedian(diff_image))
        diff_abs = np.abs(diff_image - diff_median)
        diff_mad = float(np.nanmedian(diff_abs))
        diff_sigma_robust = 1.4826 * diff_mad
    except Exception:
        diff_median = float("nan")
        diff_sigma_robust = float("nan")

    extra["diff_image_median"] = diff_median
    extra["diff_image_robust_sigma"] = diff_sigma_robust

    try:
        peak_flat_index = int(np.nanargmax(diff_image))
        peak_row, peak_col = np.unravel_index(peak_flat_index, diff_image.shape)
        peak_value = float(diff_image[peak_row, peak_col])
    except Exception:
        peak_row, peak_col = -1, -1
        peak_value = float("nan")

    extra["diff_peak_pixel_rc"] = [int(peak_row), int(peak_col)]
    extra["diff_peak_value"] = peak_value
    if (
        np.isfinite(peak_value)
        and np.isfinite(diff_median)
        and np.isfinite(diff_sigma_robust)
        and diff_sigma_robust > 0
    ):
        extra["diff_peak_snr"] = float((peak_value - diff_median) / diff_sigma_robust)
    else:
        extra["diff_peak_snr"] = float("nan")

    # Pixel-level depth significance at the peak difference pixel.
    if peak_row >= 0 and peak_col >= 0 and n_in_transit >= 3 and n_out_of_transit >= 3:
        in_pix = tpf_used.flux[in_transit_mask, peak_row, peak_col]
        out_pix = tpf_used.flux[out_of_transit_mask, peak_row, peak_col]
        in_pix = in_pix[np.isfinite(in_pix)]
        out_pix = out_pix[np.isfinite(out_pix)]
        if in_pix.size >= 3 and out_pix.size >= 3:
            in_med = float(np.nanmedian(in_pix))
            out_med = float(np.nanmedian(out_pix))
            depth = out_med - in_med

            def _robust_std(x: np.ndarray) -> float:
                m = float(np.nanmedian(x))
                mad = float(np.nanmedian(np.abs(x - m)))
                return 1.4826 * mad

            in_std = _robust_std(in_pix)
            out_std = _robust_std(out_pix)
            depth_err = float(
                np.sqrt(
                    (in_std / np.sqrt(max(1, in_pix.size))) ** 2
                    + (out_std / np.sqrt(max(1, out_pix.size))) ** 2
                )
            )
            extra["peak_pixel_depth"] = float(depth)
            extra["peak_pixel_depth_err"] = float(depth_err)
            extra["peak_pixel_depth_sigma"] = (
                float(depth / depth_err) if depth_err > 0 else float("nan")
            )
        else:
            extra["peak_pixel_depth"] = float("nan")
            extra["peak_pixel_depth_err"] = float("nan")
            extra["peak_pixel_depth_sigma"] = float("nan")
    else:
        extra["peak_pixel_depth"] = float("nan")
        extra["peak_pixel_depth_err"] = float("nan")
        extra["peak_pixel_depth_sigma"] = float("nan")

    # Convert centroid to sky coordinates
    try:
        centroid_sky_ra, centroid_sky_dec = pixel_to_world(
            tpf_used.wcs,
            row=centroid_pixel_rc[0],
            col=centroid_pixel_rc[1],
            origin=0,
        )
    except Exception as e:
        centroid_sky_ra = float("nan")
        centroid_sky_dec = float("nan")
        warnings.append(f"WCS transform failed: {e}")

    # Bootstrap uncertainty estimation
    if bootstrap_draws > 0:
        centroids_array, semimajor, semiminor, pa = bootstrap_centroid_uncertainty(
            tpf_fits=tpf_used,
            period=period,
            t0=t0,
            duration_hours=duration_hours,
            n_draws=bootstrap_draws,
            seed=bootstrap_seed,
            oot_margin_mult=oot_margin_mult,
            oot_window_mult=oot_window_mult,
            method=method,
        )
        extra["bootstrap_centroid_mean_rc"] = [
            float(np.nanmean(centroids_array[:, 0])),
            float(np.nanmean(centroids_array[:, 1])),
        ]
        extra["bootstrap_centroid_std_rc"] = [
            float(np.nanstd(centroids_array[:, 0])),
            float(np.nanstd(centroids_array[:, 1])),
        ]
    else:
        semimajor = float("nan")
        semiminor = float("nan")
        pa = float("nan")

    # Compute distances to reference sources
    distances_to_sources = compute_reference_source_distances(
        centroid_sky=(centroid_sky_ra, centroid_sky_dec),
        reference_sources=reference_sources,
    )

    # Get pixel positions of reference sources
    wcs_source_positions = get_reference_source_pixel_positions(
        tpf_used.wcs,
        reference_sources,
        origin=0,
    )

    # Record extra info
    extra["pixel_scale_arcsec"] = pixel_scale_arcsec
    extra["method"] = method

    # Determine verdict
    verdict, rationale = _determine_verdict(
        centroid_pixel_rc=centroid_pixel_rc,
        distances_to_sources=distances_to_sources,
        uncertainty_semimajor_arcsec=semimajor,
        pixel_scale_arcsec=pixel_scale_arcsec,
        warnings=warnings,
        reference_sources=reference_sources,
    )

    return LocalizationResult(
        centroid_pixel_rc=centroid_pixel_rc,
        centroid_sky_ra=centroid_sky_ra,
        centroid_sky_dec=centroid_sky_dec,
        uncertainty_semimajor_arcsec=semimajor,
        uncertainty_semiminor_arcsec=semiminor,
        uncertainty_pa_deg=pa,
        distances_to_sources=distances_to_sources,
        wcs_source_positions=wcs_source_positions,
        n_bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
        verdict=verdict,
        verdict_rationale=rationale,
        warnings=warnings,
        extra=extra,
    )


__all__ = [
    "LocalizationResult",
    "LocalizationVerdict",
    "bootstrap_centroid_uncertainty",
    "compute_difference_image_centroid",
    "compute_reference_source_distances",
    "localize_transit_source",
]
