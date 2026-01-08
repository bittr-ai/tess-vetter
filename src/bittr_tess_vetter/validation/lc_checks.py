"""Vetter check implementations for transit candidate validation.

This module implements the 10 vetting checks (V01-V10) organized by tier:
- LC-only (V01-V05): Use only light curve data, always available
- Catalog (V06-V07): Require catalog cross-matching (local cache)
- Pixel (V08-V10): Require TPF/FFI data (deferred to v2)

Each check returns a VetterCheckResult with pass/fail, confidence, and details.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bittr_tess_vetter.domain.detection import VetterCheckResult

if TYPE_CHECKING:
    from bittr_tess_vetter.domain.detection import TransitCandidate
    from bittr_tess_vetter.domain.lightcurve import LightCurveData
    from bittr_tess_vetter.domain.target import StellarParameters, Target

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class OddEvenConfig:
    """Configuration for V01 odd/even depth check.

    Attributes:
        sigma_threshold: Significance threshold for delta_sigma (default 3.0)
        rel_diff_threshold: Relative difference threshold (default 0.5 = 50%)
        min_transits_per_parity: Minimum transits needed per odd/even group
        min_points_in_transit_per_epoch: Minimum in-transit points per epoch
        min_points_in_transit_per_parity: Minimum total in-transit points per parity
        baseline_window_mult: Local baseline window as multiple of duration
        use_red_noise_inflation: Whether to apply red noise inflation to uncertainties
    """

    sigma_threshold: float = 3.0
    rel_diff_threshold: float = 0.5
    min_transits_per_parity: int = 2
    min_points_in_transit_per_epoch: int = 5
    min_points_in_transit_per_parity: int = 20
    baseline_window_mult: float = 6.0
    use_red_noise_inflation: bool = True


# =============================================================================
# Tier 1: LC-Only Checks (V01-V05)
# =============================================================================


def _robust_std(arr: np.ndarray) -> float:
    """Compute robust standard deviation using MAD.

    Uses median absolute deviation (MAD) scaled to match standard deviation
    for normally distributed data.

    Args:
        arr: Input array

    Returns:
        Robust estimate of standard deviation
    """
    if len(arr) < 2:
        return 0.0
    mad = np.median(np.abs(arr - np.median(arr)))
    # Scale factor for normal distribution: MAD * 1.4826 ≈ std
    return float(mad * 1.4826)


def _compute_red_noise_inflation(
    oot_residuals: np.ndarray,
    oot_time: np.ndarray,
    bin_size_days: float,
) -> tuple[float, bool]:
    """Compute red noise inflation factor from OOT residuals.

    Bins the residuals and compares observed scatter to expected white noise.

    Args:
        oot_residuals: Out-of-transit flux residuals (flux - median)
        oot_time: Time array corresponding to residuals
        bin_size_days: Bin size in days

    Returns:
        Tuple of (inflation_factor, success_flag)
    """
    if len(oot_residuals) < 10 or bin_size_days <= 0:
        return 1.0, False

    # Sort by time for binning
    sort_idx = np.argsort(oot_time)
    sorted_residuals = oot_residuals[sort_idx]
    sorted_time = oot_time[sort_idx]

    # Create time bins
    t_min, t_max = sorted_time[0], sorted_time[-1]
    time_span = t_max - t_min
    if time_span < bin_size_days * 2:
        return 1.0, False

    n_bins = max(3, int(time_span / bin_size_days))
    bin_edges = np.linspace(t_min, t_max, n_bins + 1)

    # Compute bin means
    bin_means = []
    bin_counts = []
    for i in range(n_bins):
        mask = (sorted_time >= bin_edges[i]) & (sorted_time < bin_edges[i + 1])
        if np.sum(mask) >= 3:
            bin_means.append(np.mean(sorted_residuals[mask]))
            bin_counts.append(np.sum(mask))

    if len(bin_means) < 3:
        return 1.0, False

    bin_means_arr = np.array(bin_means)
    bin_counts_arr = np.array(bin_counts)

    # Observed scatter of bin means
    observed_scatter = np.std(bin_means_arr)

    # Expected white noise scatter: std / sqrt(n_per_bin)
    point_scatter = _robust_std(sorted_residuals)
    avg_per_bin = np.mean(bin_counts_arr)
    expected_scatter = point_scatter / np.sqrt(avg_per_bin) if avg_per_bin > 0 else point_scatter

    if expected_scatter <= 0:
        return 1.0, False

    ratio = observed_scatter / expected_scatter
    return float(max(1.0, ratio)), True


def _compute_confidence(
    n_odd_transits: int,
    n_even_transits: int,
    delta_sigma: float,
    sigma_threshold: float,
    has_warnings: bool,
) -> float:
    """Compute confidence score for odd/even check.

    Args:
        n_odd_transits: Number of odd transits with sufficient data
        n_even_transits: Number of even transits with sufficient data
        delta_sigma: Significance of depth difference
        sigma_threshold: Threshold for failing
        has_warnings: Whether warnings were issued (degrades confidence)

    Returns:
        Confidence score in [0, 1]
    """
    n_min = min(n_odd_transits, n_even_transits)

    # Base confidence from transit count
    if n_min <= 1:
        base = 0.2
    elif n_min <= 3:
        base = 0.5
    elif n_min <= 7:
        base = 0.7
    else:
        base = 0.85

    # Adjust for proximity to threshold
    if delta_sigma > 0.8 * sigma_threshold:
        # Near threshold - reduce confidence
        base *= 0.85
    elif delta_sigma < 1.0 and n_min >= 4:
        # Strong pass with good N - boost slightly
        base = min(0.95, base * 1.1)

    # Degrade if warnings present
    if has_warnings:
        base *= 0.9

    return round(min(0.95, base), 3)


def check_odd_even_depth(
    lightcurve: LightCurveData,
    period: float,
    t0: float,
    duration_hours: float,
    config: OddEvenConfig | None = None,
) -> VetterCheckResult:
    """V01: Compare depth of odd vs even transits.

    Detects eclipsing binaries masquerading as planets at 2x the true period.
    If odd and even depths differ significantly, likely an EB.

    This implementation uses per-epoch depth estimates with local baselines,
    which is more robust to baseline drift and correlated noise than pooling
    all in-transit points.

    Decision rule (dual threshold):
        FAIL if delta_sigma >= sigma_threshold AND rel_diff >= rel_diff_threshold
        PASS otherwise

    Args:
        lightcurve: Light curve data
        period: Orbital period in days
        t0: Reference epoch (BTJD)
        duration_hours: Transit duration in hours
        config: Optional configuration overrides

    Returns:
        VetterCheckResult with pass if depths are consistent

    References:
        - Thompson et al. 2018, ApJS 235, 38 (Kepler Robovetter odd/even test)
        - Pont et al. 2006, MNRAS 373, 231 (correlated noise in transit photometry)
    """
    if config is None:
        config = OddEvenConfig()

    time = lightcurve.time[lightcurve.valid_mask]
    flux = lightcurve.flux[lightcurve.valid_mask]

    duration_days = duration_hours / 24.0
    warnings: list[str] = []

    # Calculate epoch index and parity for each point
    # Offset by half period so epoch boundaries fall BETWEEN transits, not AT transit centers.
    # This ensures each epoch fully contains exactly one transit.
    # epoch 0: centered on transit at t0
    # epoch 1: centered on transit at t0 + period
    # etc.
    epoch = np.floor((time - t0 + period / 2) / period).astype(int)

    # Phase distance from transit center (0 = at center, 0.5 = at anti-transit)
    phase = ((time - t0) / period) % 1
    phase_dist = np.minimum(phase, 1 - phase)  # Distance to nearest transit center

    # In-transit mask: within half-duration of transit center
    half_dur_phase = 0.5 * (duration_days / period)
    in_transit = phase_dist < half_dur_phase

    # Get unique epochs
    unique_epochs = np.unique(epoch)

    # Per-epoch depth extraction with local baselines
    epoch_data: dict[int, dict[str, float]] = {}

    for ep in unique_epochs:
        epoch_mask = epoch == ep
        epoch_in_transit = epoch_mask & in_transit
        n_in = np.sum(epoch_in_transit)

        if n_in < config.min_points_in_transit_per_epoch:
            continue

        # Define local OOT baseline window around epoch center
        epoch_center = t0 + ep * period
        baseline_half_window = config.baseline_window_mult * duration_days
        local_window = (time >= epoch_center - baseline_half_window) & (
            time <= epoch_center + baseline_half_window
        )
        local_oot = local_window & ~in_transit

        n_oot = np.sum(local_oot)
        if n_oot < 5:
            # Fall back to global OOT if local is too sparse
            local_oot = ~in_transit
            n_oot = np.sum(local_oot)
            if n_oot < 10:
                continue

        # Compute local baseline
        baseline_flux = flux[local_oot]
        baseline = float(np.median(baseline_flux))

        if baseline <= 0:
            continue

        # Compute depth for this epoch
        in_flux = flux[epoch_in_transit]
        depth_k = 1.0 - float(np.median(in_flux)) / baseline

        # Compute uncertainty: robust_std(oot) / sqrt(n_in) / baseline
        oot_scatter = _robust_std(baseline_flux)
        sigma_k = oot_scatter / np.sqrt(n_in) / baseline if n_in > 0 else float("inf")

        # Optional red noise inflation
        if config.use_red_noise_inflation and n_oot >= 20:
            oot_residuals = baseline_flux - baseline
            oot_time = time[local_oot]
            inflation, success = _compute_red_noise_inflation(
                oot_residuals, oot_time, duration_days / 2
            )
            if success:
                sigma_k *= inflation
            elif n_oot < 30:
                warnings.append(f"Epoch {ep}: insufficient OOT for red noise estimation")

        epoch_data[int(ep)] = {
            "depth": depth_k,
            "sigma": sigma_k,
            "n_in": int(n_in),
            "baseline": baseline,
        }

    # Separate odd and even epochs
    odd_epochs = {k: v for k, v in epoch_data.items() if k % 2 == 1}
    even_epochs = {k: v for k, v in epoch_data.items() if k % 2 == 0}

    n_odd_transits = len(odd_epochs)
    n_even_transits = len(even_epochs)

    # Total in-transit points per parity (for legacy compatibility)
    n_odd_points = sum(v["n_in"] for v in odd_epochs.values())
    n_even_points = sum(v["n_in"] for v in even_epochs.values())

    # Check minimum data requirements
    insufficient_data = False
    if n_odd_transits < config.min_transits_per_parity:
        warnings.append(
            f"Only {n_odd_transits} odd transit(s), need {config.min_transits_per_parity}"
        )
        insufficient_data = True
    if n_even_transits < config.min_transits_per_parity:
        warnings.append(
            f"Only {n_even_transits} even transit(s), need {config.min_transits_per_parity}"
        )
        insufficient_data = True
    if n_odd_points < config.min_points_in_transit_per_parity:
        warnings.append(
            f"Only {n_odd_points} odd in-transit points, need "
            f"{config.min_points_in_transit_per_parity}"
        )
        insufficient_data = True
    if n_even_points < config.min_points_in_transit_per_parity:
        warnings.append(
            f"Only {n_even_points} even in-transit points, need "
            f"{config.min_points_in_transit_per_parity}"
        )
        insufficient_data = True

    if insufficient_data:
        # Cannot reject with insufficient data - return low-confidence pass
        return VetterCheckResult(
            id="V01",
            name="odd_even_depth",
            passed=True,
            confidence=0.2,
            details={
                # Legacy keys
                "odd_depth": 0.0,
                "even_depth": 0.0,
                "depth_diff_sigma": 0.0,
                "n_odd_points": n_odd_points,
                "n_even_points": n_even_points,
                # New keys
                "n_odd_transits": n_odd_transits,
                "n_even_transits": n_even_transits,
                "depth_odd_ppm": 0.0,
                "depth_even_ppm": 0.0,
                "depth_err_odd_ppm": 0.0,
                "depth_err_even_ppm": 0.0,
                "delta_ppm": 0.0,
                "delta_sigma": 0.0,
                "rel_diff": 0.0,
                "warnings": warnings,
                "method": "per_epoch_median",
                "epoch_depths_odd_ppm": [],
                "epoch_depths_even_ppm": [],
            },
        )

    # Aggregate odd depths
    odd_depths = np.array([v["depth"] for v in odd_epochs.values()])
    odd_sigmas = np.array([v["sigma"] for v in odd_epochs.values()])
    median_odd = float(np.median(odd_depths))
    # Aggregate uncertainty: median(sigma_k) / sqrt(n_transits)
    sigma_odd = float(np.median(odd_sigmas)) / np.sqrt(n_odd_transits)

    # Aggregate even depths
    even_depths = np.array([v["depth"] for v in even_epochs.values()])
    even_sigmas = np.array([v["sigma"] for v in even_epochs.values()])
    median_even = float(np.median(even_depths))
    sigma_even = float(np.median(even_sigmas)) / np.sqrt(n_even_transits)

    # Compute delta and significance
    delta = median_odd - median_even
    sigma_delta = np.sqrt(sigma_odd**2 + sigma_even**2)
    delta_sigma = abs(delta) / sigma_delta if sigma_delta > 0 else 0.0

    # Relative difference
    eps = 1e-10
    max_depth = max(abs(median_odd), abs(median_even), eps)
    rel_diff = abs(delta) / max_depth

    # Decision rule: FAIL if BOTH thresholds exceeded
    passed = not (delta_sigma >= config.sigma_threshold and rel_diff >= config.rel_diff_threshold)

    # Confidence
    confidence = _compute_confidence(
        n_odd_transits, n_even_transits, delta_sigma, config.sigma_threshold, len(warnings) > 0
    )

    # Convert to ppm for output
    depth_odd_ppm = median_odd * 1e6
    depth_even_ppm = median_even * 1e6
    depth_err_odd_ppm = sigma_odd * 1e6
    depth_err_even_ppm = sigma_even * 1e6
    delta_ppm = delta * 1e6

    # Cap epoch depths arrays to 20 elements
    epoch_depths_odd_ppm = [round(d * 1e6, 1) for d in odd_depths[:20]]
    epoch_depths_even_ppm = [round(d * 1e6, 1) for d in even_depths[:20]]

    return VetterCheckResult(
        id="V01",
        name="odd_even_depth",
        passed=passed,
        confidence=confidence,
        details={
            # Legacy keys (fractional depths)
            "odd_depth": round(median_odd, 6),
            "even_depth": round(median_even, 6),
            "depth_diff_sigma": round(delta_sigma, 2),
            "n_odd_points": n_odd_points,
            "n_even_points": n_even_points,
            # New keys (ppm and extended diagnostics)
            "n_odd_transits": n_odd_transits,
            "n_even_transits": n_even_transits,
            "depth_odd_ppm": round(depth_odd_ppm, 1),
            "depth_even_ppm": round(depth_even_ppm, 1),
            "depth_err_odd_ppm": round(depth_err_odd_ppm, 1),
            "depth_err_even_ppm": round(depth_err_even_ppm, 1),
            "delta_ppm": round(delta_ppm, 1),
            "delta_sigma": round(delta_sigma, 2),
            "rel_diff": round(rel_diff, 3),
            "warnings": warnings,
            "method": "per_epoch_median",
            "epoch_depths_odd_ppm": epoch_depths_odd_ppm,
            "epoch_depths_even_ppm": epoch_depths_even_ppm,
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
