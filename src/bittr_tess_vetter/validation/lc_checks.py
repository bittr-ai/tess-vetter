"""Vetter check implementations for transit candidate validation.

This module implements the 10 vetting checks (V01-V10) organized by tier:
- LC-only (V01-V05): Use only light curve data, always available
- Catalog (V06-V07): Require catalog cross-matching (local cache)
- Pixel (V08-V10): Require TPF/FFI data (deferred to v2)

Each check returns a VetterCheckResult with pass/fail, confidence, and details.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from bittr_tess_vetter.domain.detection import VetterCheckResult

if TYPE_CHECKING:
    from bittr_tess_vetter.domain.detection import TransitCandidate
    from bittr_tess_vetter.domain.lightcurve import LightCurveData
    from bittr_tess_vetter.domain.target import StellarParameters, Target

logger = logging.getLogger(__name__)


# =============================================================================
# Tier 1: LC-Only Checks (V01-V05)
# =============================================================================


def check_odd_even_depth(
    lightcurve: LightCurveData,
    period: float,
    t0: float,
    duration_hours: float,
) -> VetterCheckResult:
    """V01: Compare depth of odd vs even transits.

    Detects eclipsing binaries masquerading as planets at 2x the true period.
    If odd and even depths differ significantly, likely an EB.

    Args:
        lightcurve: Light curve data
        period: Orbital period in days
        t0: Reference epoch (BTJD)
        duration_hours: Transit duration in hours

    Returns:
        VetterCheckResult with pass if depths are consistent
    """
    time = lightcurve.time[lightcurve.valid_mask]
    flux = lightcurve.flux[lightcurve.valid_mask]

    duration_days = duration_hours / 24.0

    # Calculate transit number for each point
    phase = ((time - t0) / period) % 1
    transit_num = np.floor((time - t0) / period).astype(int)

    # In-transit mask
    in_transit = (phase < duration_days / period / 2) | (phase > 1 - duration_days / period / 2)

    # Separate odd and even transits
    odd_mask = in_transit & (transit_num % 2 == 1)
    even_mask = in_transit & (transit_num % 2 == 0)

    odd_flux = flux[odd_mask]
    even_flux = flux[even_mask]

    if len(odd_flux) < 5 or len(even_flux) < 5:
        # Insufficient data for comparison
        return VetterCheckResult(
            id="V01",
            name="odd_even_depth",
            passed=True,  # Assume pass if not enough data
            confidence=0.3,
            details={
                "n_odd_points": len(odd_flux),
                "n_even_points": len(even_flux),
                "note": "Insufficient data for odd/even comparison",
            },
        )

    # Calculate depths as deviation from median
    out_of_transit = ~in_transit
    baseline = np.median(flux[out_of_transit]) if np.any(out_of_transit) else 1.0

    odd_depth = 1.0 - np.median(odd_flux) / baseline
    even_depth = 1.0 - np.median(even_flux) / baseline

    # Error in depth estimate (simplified)
    odd_err = np.std(odd_flux) / np.sqrt(len(odd_flux)) / baseline
    even_err = np.std(even_flux) / np.sqrt(len(even_flux)) / baseline

    # Combined uncertainty
    combined_err = np.sqrt(odd_err**2 + even_err**2)

    # Significance of depth difference
    depth_diff_sigma = abs(odd_depth - even_depth) / combined_err if combined_err > 0 else 0.0

    # Pass if difference < 3 sigma
    passed = depth_diff_sigma < 3.0

    # Confidence based on number of transits and sigma
    confidence = min(0.95, 0.5 + 0.1 * min(len(odd_flux), len(even_flux)) / 10)
    if depth_diff_sigma > 2.0:
        confidence *= 0.7  # Lower confidence if marginal

    return VetterCheckResult(
        id="V01",
        name="odd_even_depth",
        passed=passed,
        confidence=round(confidence, 3),
        details={
            "odd_depth": round(odd_depth, 6),
            "even_depth": round(even_depth, 6),
            "depth_diff_sigma": round(depth_diff_sigma, 2),
            "n_odd_points": len(odd_flux),
            "n_even_points": len(even_flux),
        },
    )


def check_secondary_eclipse(
    lightcurve: LightCurveData,
    period: float,
    t0: float,
) -> VetterCheckResult:
    """V02: Search for secondary eclipse at phase 0.5.

    Presence of secondary eclipse indicates hot planet (thermal emission)
    or eclipsing binary. Significant secondary suggests EB.

    Args:
        lightcurve: Light curve data
        period: Orbital period in days
        t0: Reference epoch (BTJD)

    Returns:
        VetterCheckResult with details on secondary eclipse search
    """
    time = lightcurve.time[lightcurve.valid_mask]
    flux = lightcurve.flux[lightcurve.valid_mask]

    # Calculate phase
    phase = ((time - t0) / period) % 1

    # Define regions
    # Secondary around phase 0.5 (using 0.40-0.60 = 10% half-width)
    # Widened from 5% to catch eccentric orbit EBs where secondary
    # can occur at phase 0.4-0.6 rather than exactly 0.5
    secondary_mask = (phase > 0.40) & (phase < 0.60)
    # Out of transit/eclipse (0.15-0.35 and 0.65-0.85)
    # Adjusted baseline regions to avoid overlap with wider secondary window
    baseline_mask = ((phase > 0.15) & (phase < 0.35)) | ((phase > 0.65) & (phase < 0.85))

    secondary_flux = flux[secondary_mask]
    baseline_flux = flux[baseline_mask]

    if len(secondary_flux) < 10 or len(baseline_flux) < 10:
        return VetterCheckResult(
            id="V02",
            name="secondary_eclipse",
            passed=True,
            confidence=0.3,
            details={
                "n_secondary_points": len(secondary_flux),
                "n_baseline_points": len(baseline_flux),
                "note": "Insufficient data for secondary eclipse search",
            },
        )

    # Calculate secondary depth
    baseline_median = np.median(baseline_flux)
    secondary_median = np.median(secondary_flux)
    secondary_depth = 1.0 - secondary_median / baseline_median

    # Uncertainty
    secondary_std = np.std(secondary_flux) / np.sqrt(len(secondary_flux))
    secondary_depth_sigma = abs(secondary_depth) / (secondary_std / baseline_median)

    # A deep secondary (>3 sigma and >50% of primary-like depth) suggests EB
    # For planets, secondary should be very shallow (<<1%)
    significant_secondary = secondary_depth_sigma > 3.0 and secondary_depth > 0.005

    passed = not significant_secondary

    confidence = 0.8 if len(secondary_flux) > 50 else 0.5 + 0.006 * len(secondary_flux)

    return VetterCheckResult(
        id="V02",
        name="secondary_eclipse",
        passed=passed,
        confidence=round(min(confidence, 0.95), 3),
        details={
            "secondary_depth": round(secondary_depth, 6),
            "secondary_depth_sigma": round(secondary_depth_sigma, 2),
            "baseline_flux": round(baseline_median, 6),
            "n_secondary_points": len(secondary_flux),
            "significant_secondary": significant_secondary,
        },
    )


def check_duration_consistency(
    period: float,
    duration_hours: float,
    stellar: StellarParameters | None,
) -> VetterCheckResult:
    """V03: Check transit duration vs stellar density expectation.

    Transit duration depends on stellar density. Unphysical durations
    (too long or too short) indicate false positive.

    Expected: T_dur ∝ P^(1/3) / ρ_star^(1/3)

    For a central transit (b=0) of a Sun-like star:
        T_dur ~ 13 hours * (P/year)^(1/3)

    The stellar density correction is critical for non-solar-type hosts:
    - M-dwarfs have ~10x solar density -> durations ~50% shorter
    - Giants have ~0.01x solar density -> durations ~5x longer

    Args:
        period: Orbital period in days
        duration_hours: Transit duration in hours
        stellar: Stellar parameters (optional but strongly recommended)

    Returns:
        VetterCheckResult with duration consistency analysis
    """
    # Base expected duration assuming solar-type host (fallback)
    period_years = period / 365.25
    expected_duration_solar = 13.0 * (period_years ** (1.0 / 3.0))

    # Apply stellar density correction if parameters available
    rho_star = None
    density_corrected = False

    if stellar is not None and stellar.has_minimum_params():
        rho_star = stellar.stellar_density_solar()
        if rho_star is not None and rho_star > 0:
            # Apply stellar density correction: T_dur ∝ ρ^(-1/3)
            # For solar density rho=1, no change
            # For M-dwarf (rho~10), duration ~ 0.46x solar
            # For giant (rho~0.01), duration ~ 4.6x solar
            expected_duration_hours = expected_duration_solar * (rho_star ** (-1.0 / 3.0))
            density_corrected = True
        else:
            expected_duration_hours = expected_duration_solar
    else:
        expected_duration_hours = expected_duration_solar

    # Allow factor of 3 uncertainty in either direction
    # (accounts for impact parameter, eccentricity, etc.)
    ratio = (
        duration_hours / expected_duration_hours if expected_duration_hours > 0 else float("inf")
    )
    passed = 0.3 < ratio < 3.0

    # Marginal if ratio is between 0.2-0.3 or 3-5
    marginal = (0.2 < ratio < 0.3) or (3.0 < ratio < 5.0)

    # Confidence depends on whether we have stellar parameters
    if density_corrected:
        confidence = 0.85  # High confidence with stellar density correction
    elif stellar is not None:
        confidence = 0.5  # Moderate confidence with some stellar info
    else:
        confidence = 0.2  # Low confidence without stellar info

    if marginal:
        confidence *= 0.7

    # Build detailed result
    details = {
        "duration_hours": round(duration_hours, 4),
        "expected_duration_hours": round(expected_duration_hours, 4),
        "expected_duration_solar": round(expected_duration_solar, 4),
        "duration_ratio": round(ratio, 3),
        "density_corrected": density_corrected,
        "period_days": round(period, 4),
    }

    if rho_star is not None:
        details["stellar_density_solar"] = round(rho_star, 4)
    if stellar is not None:
        details["stellar_radius"] = stellar.radius
        details["stellar_mass"] = stellar.mass

    if not density_corrected:
        details["note"] = (
            "Duration check uses solar-type assumption. "
            "Provide stellar parameters for accurate M-dwarf/giant assessment."
        )

    return VetterCheckResult(
        id="V03",
        name="duration_consistency",
        passed=passed,
        confidence=round(confidence, 3),
        details=details,
    )


def check_depth_stability(
    lightcurve: LightCurveData,
    period: float,
    t0: float,
    duration_hours: float,
) -> VetterCheckResult:
    """V04: Check depth consistency across individual transits.

    Variable depth suggests blended eclipsing binary or systematic issues.
    Real planets have consistent depths.

    Args:
        lightcurve: Light curve data
        period: Orbital period in days
        t0: Reference epoch (BTJD)
        duration_hours: Transit duration in hours

    Returns:
        VetterCheckResult with depth stability metrics
    """
    time = lightcurve.time[lightcurve.valid_mask]
    flux = lightcurve.flux[lightcurve.valid_mask]

    duration_days = duration_hours / 24.0

    # Calculate transit number for each point
    transit_num = np.floor((time - t0) / period).astype(int)
    phase = ((time - t0) / period) % 1

    # In-transit mask
    in_transit = (phase < duration_days / period / 2) | (phase > 1 - duration_days / period / 2)

    # Out-of-transit baseline
    baseline = np.median(flux[~in_transit]) if np.any(~in_transit) else 1.0

    # Measure depth for each transit
    unique_transits = np.unique(transit_num[in_transit])
    individual_depths = []

    for tn in unique_transits:
        transit_mask = in_transit & (transit_num == tn)
        transit_flux = flux[transit_mask]
        if len(transit_flux) >= 3:
            depth = 1.0 - np.median(transit_flux) / baseline
            if depth > 0:  # Only count actual dips
                individual_depths.append(depth)

    if len(individual_depths) < 2:
        return VetterCheckResult(
            id="V04",
            name="depth_stability",
            passed=True,
            confidence=0.3,
            details={
                "n_transits_measured": len(individual_depths),
                "note": "Insufficient transits for depth stability check",
            },
        )

    depths = np.array(individual_depths)
    mean_depth = np.mean(depths)
    std_depth = np.std(depths)
    rms_scatter = std_depth / mean_depth if mean_depth > 0 else 0

    # Pass if RMS scatter < 30% of mean depth
    passed = bool(rms_scatter < 0.3)

    # Confidence increases with number of transits
    confidence = min(0.95, 0.5 + 0.1 * len(depths))
    if rms_scatter > 0.2:
        confidence *= 0.8

    return VetterCheckResult(
        id="V04",
        name="depth_stability",
        passed=passed,
        confidence=round(confidence, 3),
        details={
            "mean_depth": round(mean_depth, 6),
            "std_depth": round(std_depth, 6),
            "rms_scatter": round(rms_scatter, 4),
            "n_transits_measured": len(depths),
            "individual_depths": [round(d, 6) for d in depths[:10]],  # First 10
        },
    )


def check_v_shape(
    lightcurve: LightCurveData,
    period: float,
    t0: float,
    duration_hours: float,
) -> VetterCheckResult:
    """V05: Distinguish U-shaped (planet) vs V-shaped (grazing EB) transits.

    Planets have flat-bottomed U-shaped transits. Grazing eclipsing binaries
    show V-shaped transits with no flat bottom.

    Args:
        lightcurve: Light curve data
        period: Orbital period in days
        t0: Reference epoch (BTJD)
        duration_hours: Transit duration in hours

    Returns:
        VetterCheckResult with shape analysis
    """
    time = lightcurve.time[lightcurve.valid_mask]
    flux = lightcurve.flux[lightcurve.valid_mask]

    duration_days = duration_hours / 24.0

    # Calculate phase centered on transit
    phase = ((time - t0) / period + 0.5) % 1 - 0.5  # -0.5 to 0.5, transit at 0

    # Define regions: ingress, flat bottom, egress
    half_dur = duration_days / period / 2

    ingress_mask = (phase > -half_dur) & (phase < -half_dur / 2)
    bottom_mask = (phase > -half_dur / 4) & (phase < half_dur / 4)
    egress_mask = (phase > half_dur / 2) & (phase < half_dur)
    baseline_mask = (abs(phase) > half_dur * 1.5) & (abs(phase) < 0.25)

    ingress_flux = flux[ingress_mask]
    bottom_flux = flux[bottom_mask]
    egress_flux = flux[egress_mask]
    baseline_flux = flux[baseline_mask]

    if len(bottom_flux) < 5 or len(baseline_flux) < 10:
        return VetterCheckResult(
            id="V05",
            name="v_shape",
            passed=True,
            confidence=0.3,
            details={
                "n_bottom_points": len(bottom_flux),
                "n_baseline_points": len(baseline_flux),
                "note": "Insufficient data for V-shape analysis",
            },
        )

    baseline_median = np.median(baseline_flux)
    bottom_median = np.median(bottom_flux)

    # For U-shape: bottom should be at max depth (flat)
    # For V-shape: bottom similar to ingress/egress
    depth_bottom = 1.0 - bottom_median / baseline_median

    # Calculate average depth during ingress/egress
    edge_flux = (
        np.concatenate([ingress_flux, egress_flux])
        if len(ingress_flux) > 0 or len(egress_flux) > 0
        else np.array([])
    )
    if len(edge_flux) > 3:
        depth_edge = 1.0 - np.median(edge_flux) / baseline_median
    else:
        depth_edge = depth_bottom * 0.5  # Assume half depth at edges

    # V-shape ratio: how much deeper is bottom vs edges
    # U-shape: ratio >> 1 (bottom much deeper)
    # V-shape: ratio ~ 1 (bottom similar to edges)
    shape_ratio = depth_bottom / depth_edge if depth_edge > 0 and depth_bottom > 0 else 2.0

    # Pass if shape_ratio > 1.3 (U-shaped, bottom is significantly deeper)
    passed = shape_ratio > 1.3

    confidence = 0.7 if len(bottom_flux) > 20 else 0.4 + 0.015 * len(bottom_flux)

    return VetterCheckResult(
        id="V05",
        name="v_shape",
        passed=passed,
        confidence=round(min(confidence, 0.9), 3),
        details={
            "depth_bottom": round(depth_bottom, 6),
            "depth_edge": round(depth_edge, 6),
            "shape_ratio": round(shape_ratio, 3),
            "shape": "U-shaped" if passed else "V-shaped",
            "n_bottom_points": len(bottom_flux),
            "n_edge_points": len(edge_flux),
        },
    )


# =============================================================================
# Tier 2: Catalog Checks (V06-V07)
# =============================================================================


def check_nearby_eb_search(
    target: Target | None,
) -> VetterCheckResult:
    """V06: Search for nearby eclipsing binaries in catalogs.

    STUB IMPLEMENTATION (v0/v1):
    This check is currently stubbed and always returns a low-confidence pass.
    Full implementation requires local catalog cache with Gaia DR3 and TIC
    cross-matching capabilities, which is planned for a future release.

    When fully implemented, this check will:
    - Query Gaia DR3 and TIC catalogs for known EBs within 2 arcmin
    - Check if any contaminating EB could produce the observed signal
    - Account for flux dilution from nearby bright sources

    Current behavior:
    - Returns passed=True with confidence=0.2 (low confidence)
    - Logs a warning to alert users the check is not performing validation
    - Sets deferred=True in details to indicate stub status

    Args:
        target: Target with position and catalog info

    Returns:
        VetterCheckResult with nearby EB search results (stub: always passes)
    """
    logger.warning(
        "V06 (Nearby EB Search) is stubbed: returning low-confidence pass. "
        "Full catalog cross-matching not yet implemented."
    )

    if target is None or not target.has_position():
        return VetterCheckResult(
            id="V06",
            name="nearby_eb_search",
            passed=True,
            confidence=0.2,
            details={
                "note": "Target position unavailable for nearby EB search",
                "deferred": True,
                "stub": True,
            },
        )

    # In v0/v1, catalog checks are deferred (no network queries)
    # Return a placeholder result indicating check was not performed
    return VetterCheckResult(
        id="V06",
        name="nearby_eb_search",
        passed=True,
        confidence=0.2,
        details={
            "ra": target.ra,
            "dec": target.dec,
            "note": "Catalog check deferred (no local cache available)",
            "deferred": True,
            "stub": True,
        },
    )


def check_known_fp_match(
    target: Target | None,
) -> VetterCheckResult:
    """V07: Check against known false positive catalog.

    STUB IMPLEMENTATION (v0/v1):
    This check is currently stubbed and always returns a low-confidence pass.
    Full implementation requires local catalog cache with TESS FP catalog
    and ExoFOP cross-matching capabilities, planned for a future release.

    When fully implemented, this check will:
    - Cross-reference target TIC ID against TESS Community FP catalog
    - Check ExoFOP disposition history for known false positives
    - Flag targets with previous FP classifications

    Current behavior:
    - Returns passed=True with confidence=0.2 (low confidence)
    - Logs a warning to alert users the check is not performing validation
    - Sets deferred=True in details to indicate stub status

    Args:
        target: Target with TIC ID

    Returns:
        VetterCheckResult with FP match results (stub: always passes)
    """
    logger.warning(
        "V07 (Known FP Match) is stubbed: returning low-confidence pass. "
        "FP catalog cross-matching not yet implemented."
    )

    if target is None:
        return VetterCheckResult(
            id="V07",
            name="known_fp_match",
            passed=True,
            confidence=0.2,
            details={
                "note": "Target info unavailable for FP catalog check",
                "deferred": True,
                "stub": True,
            },
        )

    # In v0/v1, catalog checks are deferred
    return VetterCheckResult(
        id="V07",
        name="known_fp_match",
        passed=True,
        confidence=0.2,
        details={
            "tic_id": target.tic_id,
            "note": "Catalog check deferred (no local cache available)",
            "deferred": True,
            "stub": True,
        },
    )


# =============================================================================
# Tier 3: Pixel Checks (V08-V10) - Deferred to v2
# =============================================================================


def check_centroid_shift() -> VetterCheckResult:
    """V08: Compare in-transit vs out-of-transit centroid position.

    DEFERRED TO v2:
    This check requires Target Pixel File (TPF) data which is not yet
    supported. Full implementation is planned for v2.

    When fully implemented, this check will:
    - Extract MOM_CENTR1/MOM_CENTR2 columns from TPF
    - Compare in-transit vs out-of-transit centroid positions
    - Flag significant centroid shifts indicating background EB
    - Use PRF fitting for sub-pixel centroid determination

    Current behavior:
    - Returns passed=True with confidence=0.1 (very low confidence)
    - Logs a warning to alert users the check is not performing validation
    - Sets deferred=True in details to indicate stub status

    Significant centroid shift indicates background eclipsing binary
    contaminating the photometric aperture.

    Returns:
        VetterCheckResult (stub: always passes with very low confidence)
    """
    logger.warning(
        "V08 (Centroid Shift) is deferred to v2: returning low-confidence pass. "
        "Requires TPF data not yet supported."
    )
    return VetterCheckResult(
        id="V08",
        name="centroid_shift",
        passed=True,
        confidence=0.1,
        details={
            "note": "Pixel-level check deferred to v2 (requires TPF)",
            "deferred": True,
        },
    )


def check_pixel_level_lc(
    tpf_data: np.ndarray | None = None,
    time: np.ndarray | None = None,
    period: float | None = None,
    t0: float | None = None,
    duration_hours: float | None = None,
    target_pixel: tuple[int, int] | None = None,
    concentration_threshold: float = 0.7,
    proximity_radius: int = 1,
) -> VetterCheckResult:
    """V09: Extract and analyze light curves from individual pixels.

    Analyzes light curves from individual TPF pixels to determine if the
    transit signal originates from the target star or a nearby source.

    When TPF data is provided, this check will:
    - Extract light curves from each pixel in the TPF
    - Compute transit depth in each pixel
    - Create a depth map to locate the transit source
    - Flag if maximum depth pixel is NOT at the target star location

    When TPF data is NOT provided:
    - Returns passed=True with confidence=0.1 (very low confidence)
    - Logs a warning to alert users the check is not performing validation
    - Sets deferred=True in details to indicate stub status

    Args:
        tpf_data: TPF flux data with shape (time, rows, cols). Optional.
        time: Time array in BTJD. Required if tpf_data provided.
        period: Orbital period in days. Required if tpf_data provided.
        t0: Reference transit epoch in BTJD. Required if tpf_data provided.
        duration_hours: Transit duration in hours. Required if tpf_data provided.
        target_pixel: Expected target pixel (row, col). Default: TPF center.
        concentration_threshold: Min target/max depth ratio to pass (default 0.7).
        proximity_radius: Max pixel distance for on-target (default 1).

    Returns:
        VetterCheckResult with pixel-level analysis results.
    """
    # If no TPF data, return stub result
    if tpf_data is None:
        logger.warning("V09 (Pixel-Level LC): No TPF data provided. Returning low-confidence pass.")
        return VetterCheckResult(
            id="V09",
            name="pixel_level_lc",
            passed=True,
            confidence=0.1,
            details={
                "note": "Pixel-level check requires TPF data",
                "deferred": True,
            },
        )

    # Validate required parameters when TPF is provided
    if time is None or period is None or t0 is None or duration_hours is None:
        logger.warning(
            "V09 (Pixel-Level LC): TPF data provided but missing ephemeris. "
            "Returning low-confidence pass."
        )
        return VetterCheckResult(
            id="V09",
            name="pixel_level_lc",
            passed=True,
            confidence=0.1,
            details={
                "note": "TPF provided but missing time/period/t0/duration_hours",
                "deferred": True,
            },
        )

    # Import here to avoid circular imports
    from bittr_tess_vetter.validation.checks_pixel import check_pixel_level_lc_with_tpf

    return check_pixel_level_lc_with_tpf(
        tpf_data=tpf_data,
        time=time,
        period=period,
        t0=t0,
        duration_hours=duration_hours,
        target_pixel=target_pixel,
        concentration_threshold=concentration_threshold,
        proximity_radius=proximity_radius,
    )


def check_aperture_dependence() -> VetterCheckResult:
    """V10: Measure transit depth vs aperture size.

    DEFERRED TO v2:
    This check requires Target Pixel File (TPF) data which is not yet
    supported. Full implementation is planned for v2.

    When fully implemented, this check will:
    - Extract light curves using multiple aperture sizes
    - Measure transit depth as a function of aperture radius
    - Flag if depth varies significantly with aperture
    - Detect flux contamination from nearby sources

    Current behavior:
    - Returns passed=True with confidence=0.1 (very low confidence)
    - Logs a warning to alert users the check is not performing validation
    - Sets deferred=True in details to indicate stub status

    Depth that varies with aperture size indicates contamination from
    nearby sources that are included in larger apertures.

    Returns:
        VetterCheckResult (stub: always passes with very low confidence)
    """
    logger.warning(
        "V10 (Aperture Dependence) is deferred to v2: returning low-confidence pass. "
        "Requires TPF data not yet supported."
    )
    return VetterCheckResult(
        id="V10",
        name="aperture_dependence",
        passed=True,
        confidence=0.1,
        details={
            "note": "Pixel-level check deferred to v2 (requires TPF)",
            "deferred": True,
        },
    )


# =============================================================================
# Validation Suite Runner
# =============================================================================


def run_all_checks(
    lightcurve: LightCurveData,
    candidate: TransitCandidate,
    stellar: StellarParameters | None = None,
    target: Target | None = None,
    tpf_data: np.ndarray | None = None,
    tpf_time: np.ndarray | None = None,
    target_pixel: tuple[int, int] | None = None,
) -> list[VetterCheckResult]:
    """Run all 10 vetting checks on a transit candidate.

    Runs checks in order V01-V10, with appropriate handling for
    deferred checks (catalog and pixel).

    Args:
        lightcurve: Light curve data
        candidate: Transit candidate with period, t0, duration, depth
        stellar: Stellar parameters (optional, improves V03)
        target: Full target info (optional, required for V06-V07)
        tpf_data: TPF flux data (time, rows, cols). Optional for V09.
        tpf_time: Time array for TPF data. Required if tpf_data provided.
        target_pixel: Expected target pixel (row, col). Default: TPF center.

    Returns:
        List of 10 VetterCheckResult objects
    """
    results = []

    # V01-V05: LC-only checks (always run)
    results.append(
        check_odd_even_depth(lightcurve, candidate.period, candidate.t0, candidate.duration_hours)
    )
    results.append(check_secondary_eclipse(lightcurve, candidate.period, candidate.t0))
    results.append(check_duration_consistency(candidate.period, candidate.duration_hours, stellar))
    results.append(
        check_depth_stability(lightcurve, candidate.period, candidate.t0, candidate.duration_hours)
    )
    results.append(
        check_v_shape(lightcurve, candidate.period, candidate.t0, candidate.duration_hours)
    )

    # V06-V07: Catalog checks (deferred in v0/v1)
    results.append(check_nearby_eb_search(target))
    results.append(check_known_fp_match(target))

    # V08-V10: Pixel checks
    results.append(check_centroid_shift())
    results.append(
        check_pixel_level_lc(
            tpf_data=tpf_data,
            time=tpf_time,
            period=candidate.period,
            t0=candidate.t0,
            duration_hours=candidate.duration_hours,
            target_pixel=target_pixel,
        )
    )
    results.append(check_aperture_dependence())

    return results
