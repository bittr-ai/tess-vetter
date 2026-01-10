"""Aperture family depth curve analysis for blend detection.

This module provides algorithms for computing transit depth as a function of
aperture size. A depth that increases with aperture radius indicates contamination
from a nearby source (blend), while a flat depth curve indicates the transit
is consistent with the target star.

Key types:
- ApertureFamilyResult: dataclass with depths by radius, slope, blend indicator
- compute_aperture_family_depth_curve: main analysis function

Usage:
    from bittr_tess_vetter.pixel.aperture_family import (
        ApertureFamilyResult,
        compute_aperture_family_depth_curve,
    )

    result = compute_aperture_family_depth_curve(
        tpf_fits=tpf_data,
        period=5.0,
        t0=1345.0,
        duration_hours=4.0,
    )

    if result.blend_indicator == "increasing":
        print("Potential blend detected!")

References:
    - Twicken et al. 2018 (2018PASP..130f4502T): pixel-level DV diagnostics and contamination context
    - Bryson et al. 2013 (2013PASP..125..889B): pixel-level background false positive diagnostics
    - Torres et al. 2011 (2011ApJ...727...24T): blend scenario interpretation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import stats

from bittr_tess_vetter.pixel.cadence_mask import default_cadence_mask

if TYPE_CHECKING:
    from bittr_tess_vetter.pixel.tpf_fits import TPFFitsData

logger = logging.getLogger(__name__)

# Default aperture radii in pixels
DEFAULT_RADII_PX = [1.5, 2.0, 2.5, 3.0, 3.5]

# Threshold for slope significance to classify as blend (2-sigma)
SLOPE_SIGNIFICANCE_THRESHOLD = 2.0

# Minimum number of in-transit points for reliable depth measurement
MIN_IN_TRANSIT_POINTS = 5

# Minimum number of out-of-transit points for reliable baseline
MIN_OUT_OF_TRANSIT_POINTS = 10


@dataclass(frozen=True)
class ApertureFamilyResult:
    """Result from aperture family depth curve analysis.

    Attributes:
        depths_by_radius_ppm: Transit depth (ppm) for each aperture radius.
        depth_uncertainties_ppm: Uncertainty in depth (ppm) for each radius.
        depth_slope_ppm_per_pixel: Linear slope of depth vs radius (ppm/pixel).
        depth_slope_significance: Significance of the slope (slope / uncertainty).
        blend_indicator: Classification of depth trend:
            - "consistent": Flat depth curve (no blend)
            - "increasing": Depth increases with radius (likely blend)
            - "decreasing": Depth decreases with radius (rare, edge effects)
            - "unstable": Depths have high scatter or insufficient data
        recommended_aperture_px: Recommended aperture radius (smallest with stable depth).
        warnings: List of warning messages.
        evidence_summary: Summary dict for evidence packet integration.
    """

    depths_by_radius_ppm: dict[float, float]
    depth_uncertainties_ppm: dict[float, float]
    depth_slope_ppm_per_pixel: float
    depth_slope_significance: float
    blend_indicator: Literal["consistent", "increasing", "decreasing", "unstable"]
    recommended_aperture_px: float
    warnings: list[str]
    evidence_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "depths_by_radius_ppm": self.depths_by_radius_ppm,
            "depth_uncertainties_ppm": self.depth_uncertainties_ppm,
            "depth_slope_ppm_per_pixel": self.depth_slope_ppm_per_pixel,
            "depth_slope_significance": self.depth_slope_significance,
            "blend_indicator": self.blend_indicator,
            "recommended_aperture_px": self.recommended_aperture_px,
            "warnings": self.warnings,
            "evidence_summary": self.evidence_summary,
        }


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

    Args:
        time: Array of observation times (BTJD).
        period: Orbital period in days.
        t0: Transit epoch in BTJD.
        duration_days: Full transit duration in days.
        oot_margin_mult: Multiplier for exclusion zone around transit.
        oot_window_mult: Window half-width in units of duration for selecting
            out-of-transit points near each transit. If None, uses a global baseline.

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


def _create_circular_aperture_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius: float,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Create a circular aperture mask.

    Args:
        shape: (n_rows, n_cols) shape of the mask.
        center: (row, col) center of the aperture.
        radius: Radius of the aperture in pixels.

    Returns:
        Boolean mask where True indicates pixels inside the aperture.
    """
    n_rows, n_cols = shape
    row_indices, col_indices = np.ogrid[:n_rows, :n_cols]

    # Distance from center (using center of each pixel)
    center_row, center_col = center
    distance = np.sqrt((row_indices - center_row) ** 2 + (col_indices - center_col) ** 2)

    mask: np.ndarray[Any, np.dtype[np.bool_]] = distance <= radius
    return mask


def _extract_aperture_lightcurve(
    flux: np.ndarray[Any, np.dtype[np.floating[Any]]],
    aperture_mask: np.ndarray[Any, np.dtype[np.bool_]],
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    """Extract summed light curve from flux cube using aperture mask.

    Args:
        flux: 3D flux array (n_cadences, n_rows, n_cols).
        aperture_mask: 2D boolean mask.

    Returns:
        1D array of summed flux values (n_cadences,).
    """
    # Sum flux within aperture for each cadence
    # flux has shape (n_cadences, n_rows, n_cols)
    # aperture_mask has shape (n_rows, n_cols)
    # Use nansum so a single NaN pixel does not poison the cadence sum.
    summed = np.nansum(flux[:, aperture_mask], axis=1)
    return summed.astype(np.float64)


def _measure_transit_depth(
    flux: np.ndarray[Any, np.dtype[np.floating[Any]]],
    in_transit_mask: np.ndarray[Any, np.dtype[np.bool_]],
    out_of_transit_mask: np.ndarray[Any, np.dtype[np.bool_]],
) -> tuple[float, float]:
    """Measure transit depth and uncertainty from a light curve.

    Uses median baseline and median in-transit flux for robustness.
    Uncertainty is estimated from standard error of the in-transit points.

    Args:
        flux: 1D light curve array.
        in_transit_mask: Boolean mask for in-transit times.
        out_of_transit_mask: Boolean mask for out-of-transit times.

    Returns:
        Tuple of (depth_ppm, depth_uncertainty_ppm).
    """
    in_transit_flux = flux[in_transit_mask]
    out_of_transit_flux = flux[out_of_transit_mask]

    # Check for sufficient data
    if len(in_transit_flux) < MIN_IN_TRANSIT_POINTS:
        return (float("nan"), float("nan"))
    if len(out_of_transit_flux) < MIN_OUT_OF_TRANSIT_POINTS:
        return (float("nan"), float("nan"))

    # Compute baseline (median out-of-transit)
    baseline = float(np.nanmedian(out_of_transit_flux))
    if baseline <= 0:
        return (float("nan"), float("nan"))

    # Compute in-transit median
    in_transit_median = float(np.nanmedian(in_transit_flux))

    # Depth = (baseline - in_transit) / baseline
    depth_frac = (baseline - in_transit_median) / baseline
    depth_ppm = depth_frac * 1_000_000.0

    # Estimate uncertainty using standard error of in-transit median
    # Using MAD-based estimate for robustness
    mad_in_transit = float(np.nanmedian(np.abs(in_transit_flux - in_transit_median)))
    # Standard deviation estimate from MAD
    sigma_in_transit = 1.4826 * mad_in_transit
    # Standard error of median
    se_median = sigma_in_transit / np.sqrt(len(in_transit_flux))
    # Propagate to depth uncertainty
    depth_uncertainty_ppm = (se_median / baseline) * 1_000_000.0

    return (depth_ppm, depth_uncertainty_ppm)


def _classify_blend_indicator(
    slope: float,
    slope_significance: float,
    n_valid_depths: int,
) -> Literal["consistent", "increasing", "decreasing", "unstable"]:
    """Classify the blend indicator based on depth slope.

    Args:
        slope: Linear slope of depth vs radius (ppm/pixel).
        slope_significance: Significance of the slope (slope / uncertainty).
        n_valid_depths: Number of apertures with valid depth measurements.

    Returns:
        Blend indicator classification.
    """
    # Check for insufficient data
    if n_valid_depths < 3:
        return "unstable"

    # Check significance
    if np.abs(slope_significance) < SLOPE_SIGNIFICANCE_THRESHOLD:
        return "consistent"

    # Significant slope
    if slope > 0:
        return "increasing"
    else:
        return "decreasing"


def compute_aperture_family_depth_curve(
    *,
    tpf_fits: TPFFitsData,
    period: float,
    t0: float,
    duration_hours: float,
    radii_px: list[float] | None = None,
    center: tuple[float, float] | None = None,
    oot_margin_mult: float = 1.5,
    oot_window_mult: float | None = 10.0,
) -> ApertureFamilyResult:
    """Compute transit depth as a function of aperture size.

    This function measures the transit depth using circular apertures of
    increasing radius. A depth that increases with aperture size indicates
    the transit signal includes flux from a nearby blended source.

    Args:
        tpf_fits: TPF FITS data with flux cube and WCS.
        period: Orbital period in days.
        t0: Transit epoch in BTJD.
        duration_hours: Full transit duration in hours.
        radii_px: List of aperture radii to test (pixels). Defaults to
            [1.5, 2.0, 2.5, 3.0, 3.5].
        center: (row, col) center for apertures. If None, uses WCS target
            position or stamp center.
        oot_margin_mult: Multiplier for out-of-transit exclusion zone.
        oot_window_mult: Window half-width in units of duration for selecting
            out-of-transit points near each transit. If None, uses a global baseline.

    Returns:
        ApertureFamilyResult with depth curve and blend indicator.

    Example:
        >>> result = compute_aperture_family_depth_curve(
        ...     tpf_fits=tpf_data,
        ...     period=5.0,
        ...     t0=1345.0,
        ...     duration_hours=4.0,
        ... )
        >>> print(result.blend_indicator)
        'consistent'
    """
    warnings: list[str] = []

    # Use default radii if not specified
    if radii_px is None:
        radii_px = DEFAULT_RADII_PX.copy()

    # Determine aperture center
    if center is None:
        # Prefer FITS target coordinates (if present) mapped through WCS, otherwise fall back
        # to stamp center.
        n_rows, n_cols = tpf_fits.flux.shape[1:3]
        center = ((n_rows - 1) / 2.0, (n_cols - 1) / 2.0)
        try:
            ra_obj = tpf_fits.meta.get("RA_OBJ")
            dec_obj = tpf_fits.meta.get("DEC_OBJ")
            if ra_obj is not None and dec_obj is not None:
                from bittr_tess_vetter.pixel.wcs_utils import world_to_pixel

                r, c = world_to_pixel(tpf_fits.wcs, float(ra_obj), float(dec_obj), origin=0)
                if np.isfinite(r) and np.isfinite(c):
                    center = (float(r), float(c))
        except Exception:
            pass

    cadence_mask = default_cadence_mask(time=tpf_fits.time, flux=tpf_fits.flux, quality=tpf_fits.quality)
    if int(np.sum(cadence_mask)) < int(tpf_fits.time.shape[0]):
        n_dropped = int(tpf_fits.time.shape[0]) - int(np.sum(cadence_mask))
        warnings.append(f"Dropped {n_dropped} cadences (quality!=0 or non-finite)")

    time = tpf_fits.time[cadence_mask]
    flux = tpf_fits.flux[cadence_mask]
    duration_days = duration_hours / 24.0

    # Compute transit masks
    in_transit_mask = _compute_transit_mask(time, period, t0, duration_days)
    out_of_transit_mask = _compute_out_of_transit_mask(
        time, period, t0, duration_days, oot_margin_mult, oot_window_mult
    )

    n_in_transit = int(np.sum(in_transit_mask))
    n_out_of_transit = int(np.sum(out_of_transit_mask))

    if n_in_transit < MIN_IN_TRANSIT_POINTS:
        warnings.append(f"Insufficient in-transit data ({n_in_transit} < {MIN_IN_TRANSIT_POINTS})")

    if n_out_of_transit < MIN_OUT_OF_TRANSIT_POINTS:
        warnings.append(
            f"Insufficient out-of-transit data ({n_out_of_transit} < {MIN_OUT_OF_TRANSIT_POINTS})"
        )

    # Measure depths for each aperture radius
    depths_by_radius: dict[float, float] = {}
    uncertainties_by_radius: dict[float, float] = {}
    stamp_shape = (flux.shape[1], flux.shape[2])

    for radius in radii_px:
        # Create aperture mask
        aperture_mask = _create_circular_aperture_mask(stamp_shape, center, radius)

        # Check aperture has pixels
        n_pixels = int(np.sum(aperture_mask))
        if n_pixels == 0:
            warnings.append(f"Aperture radius {radius} px has no pixels")
            depths_by_radius[radius] = float("nan")
            uncertainties_by_radius[radius] = float("nan")
            continue

        # Extract light curve for this aperture
        lc = _extract_aperture_lightcurve(flux, aperture_mask)

        # Measure depth
        depth_ppm, uncertainty_ppm = _measure_transit_depth(
            lc, in_transit_mask, out_of_transit_mask
        )

        depths_by_radius[radius] = depth_ppm
        uncertainties_by_radius[radius] = uncertainty_ppm

    # Fit linear trend to depth vs radius
    valid_radii = []
    valid_depths = []

    for r in radii_px:
        d = depths_by_radius.get(r, float("nan"))
        if np.isfinite(d):
            valid_radii.append(r)
            valid_depths.append(d)

    n_valid = len(valid_radii)

    if n_valid >= 2:
        # Linear regression
        result = stats.linregress(valid_radii, valid_depths)
        slope_ppm_per_pixel = float(result.slope)  # type: ignore[arg-type]
        slope_uncertainty = float(result.stderr)  # type: ignore[arg-type]

        if slope_uncertainty > 0:
            slope_significance = slope_ppm_per_pixel / slope_uncertainty
        else:
            slope_significance = float("nan")
    else:
        slope_ppm_per_pixel = float("nan")
        slope_significance = float("nan")
        warnings.append("Insufficient valid depths for slope estimation")

    # Classify blend indicator
    blend_indicator = _classify_blend_indicator(slope_ppm_per_pixel, slope_significance, n_valid)

    # Determine recommended aperture (smallest with stable depth)
    recommended_aperture = min(valid_radii) if valid_radii else (radii_px[0] if radii_px else 2.0)

    # Build evidence summary
    evidence_summary: dict[str, Any] = {
        "n_apertures_tested": len(radii_px),
        "n_valid_depths": n_valid,
        "depth_slope_ppm_per_pixel": slope_ppm_per_pixel,
        "depth_slope_significance": slope_significance,
        "blend_indicator": blend_indicator,
        "center_row": center[0],
        "center_col": center[1],
        "n_in_transit": n_in_transit,
        "n_out_of_transit": n_out_of_transit,
    }

    return ApertureFamilyResult(
        depths_by_radius_ppm=depths_by_radius,
        depth_uncertainties_ppm=uncertainties_by_radius,
        depth_slope_ppm_per_pixel=slope_ppm_per_pixel,
        depth_slope_significance=slope_significance,
        blend_indicator=blend_indicator,
        recommended_aperture_px=recommended_aperture,
        warnings=warnings,
        evidence_summary=evidence_summary,
    )


__all__ = [
    "ApertureFamilyResult",
    "compute_aperture_family_depth_curve",
    "DEFAULT_RADII_PX",
    "SLOPE_SIGNIFICANCE_THRESHOLD",
]
