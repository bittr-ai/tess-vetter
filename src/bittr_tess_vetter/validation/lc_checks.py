"""Light-curve-only vetting checks (V01-V05).

This module implements the LC-only tier of vetting checks:
- V01 odd_even_depth: Compare depth of odd vs even transits (detect EBs at 2x period)
- V02 secondary_eclipse: Search for secondary eclipse at phase 0.5
- V03 duration_consistency: Check transit duration vs stellar density expectation
- V04 depth_stability: Check depth consistency across individual transits
- V05 v_shape: Distinguish U-shaped (planet) vs V-shaped (grazing EB) transits

Each check returns a VetterCheckResult with metrics, confidence, and details.
All checks run in metrics-only mode (`passed=None`) so host applications can
apply policy/guardrails externally.

For catalog checks (V06-V07), see `validation.checks_catalog`.
For pixel checks (V08-V10), see `validation.checks_pixel`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bittr_tess_vetter.domain.detection import VetterCheckResult

if TYPE_CHECKING:
    from bittr_tess_vetter.domain.lightcurve import LightCurveData
    from bittr_tess_vetter.domain.target import StellarParameters

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class OddEvenConfig:
    """Configuration for V01 odd/even depth check.

    Attributes:
        min_transits_per_parity: Minimum transits needed per odd/even group
        min_points_in_transit_per_epoch: Minimum in-transit points per epoch
        min_points_in_transit_per_parity: Minimum total in-transit points per parity
        baseline_window_mult: Local baseline window as multiple of duration
        baseline_window_max_fraction_of_period: Cap baseline window to this fraction of period
            to avoid spanning adjacent transits for short-period candidates (default 0.45)
        use_red_noise_inflation: Whether to apply red noise inflation to uncertainties
    """

    min_transits_per_parity: int = 2
    min_points_in_transit_per_epoch: int = 5
    min_points_in_transit_per_parity: int = 20
    baseline_window_mult: float = 6.0
    baseline_window_max_fraction_of_period: float = 0.45
    use_red_noise_inflation: bool = True


@dataclass
class VShapeConfig:
    """Configuration for V05 transit shape check.

    Uses trapezoid model fitting to extract tF/tT ratio (flat-bottom to total
    duration), the standard shape discriminant in the literature.

    Attributes:
        min_points_in_transit: Minimum total in-transit points (default 10)
        min_transit_coverage: Minimum fraction of transit phases with data (default 0.6)
        n_bootstrap: Number of bootstrap iterations for uncertainty (default 100)
        bootstrap_ci: Confidence interval for bootstrap (default 0.68 = 1-sigma)
    """

    min_points_in_transit: int = 10
    min_transit_coverage: float = 0.6
    n_bootstrap: int = 100
    bootstrap_ci: float = 0.68


@dataclass
class SecondaryEclipseConfig:
    """Configuration for V02 secondary eclipse check.

    Attributes:
        secondary_center: Center of secondary search window in phase (default 0.5)
        secondary_half_width: Half-width of search window in phase units (default 0.15)
            This covers phase 0.35-0.65, widened from 0.10 to catch eccentric orbit EBs.
        baseline_half_width: Half-width of adjacent baseline windows (default 0.15)
        min_secondary_points: Minimum points in secondary window (default 10)
        min_baseline_points: Minimum points in baseline windows (default 10)
        min_secondary_events: Minimum distinct orbital cycles with secondary data (default 2)
        min_phase_coverage: Minimum phase coverage fraction for reliable result (default 0.3)
        use_red_noise_inflation: Whether to apply red noise inflation (default True)
        default_inflation: Fallback inflation factor when estimation fails (default 1.5)
        n_coverage_bins: Number of bins for phase coverage calculation (default 20)

    References:
        - Coughlin & Lopez-Morales 2012, AJ 143, 39 (secondary eclipse methodology)
        - Thompson et al. 2018, ApJS 235, 38 (Robovetter significant secondary test)
        - Pont et al. 2006, MNRAS 373, 231 (red noise inflation)
        - Santerne et al. 2013, A&A 557, A139 (eccentric orbit secondary offsets)
    """

    secondary_center: float = 0.5
    secondary_half_width: float = 0.15
    baseline_half_width: float = 0.15
    min_secondary_points: int = 10
    min_baseline_points: int = 10
    min_secondary_events: int = 2
    min_phase_coverage: float = 0.3
    use_red_noise_inflation: bool = True
    default_inflation: float = 1.5
    n_coverage_bins: int = 20


@dataclass
class DepthStabilityConfig:
    """Configuration for V04 depth stability check.

    Attributes:
        min_transits_for_confidence: Minimum transits for meaningful scatter (default 3)
        min_points_per_epoch: Minimum in-transit points per epoch (default 5)
        baseline_window_mult: Local baseline window as multiple of duration (default 6.0)
        outlier_sigma: MAD-based outlier flagging threshold (default 4.0)
        use_red_noise_inflation: Whether to apply red noise inflation (default True)

    References:
        - Thompson et al. 2018, ApJS 235, 38 (depth consistency tests)
        - Pont et al. 2006, MNRAS 373, 231 (correlated noise)
        - Wang & Espinoza 2023, arXiv:2311.02154 (per-transit depth fitting)
    """

    min_transits_for_confidence: int = 3
    min_points_per_epoch: int = 5
    baseline_window_mult: float = 6.0
    outlier_sigma: float = 4.0
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
    has_warnings: bool,
) -> float:
    """Compute confidence score for odd/even check.

    Args:
        n_odd_transits: Number of odd transits with sufficient data
        n_even_transits: Number of even transits with sufficient data
        delta_sigma: Significance of depth difference
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

    # Mild boost for strong N when delta_sigma is small (typically consistent odd/even).
    if delta_sigma < 1.0 and n_min >= 4:
        base = min(0.95, base * 1.05)

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

    Args:
        lightcurve: Light curve data
        period: Orbital period in days
        t0: Reference epoch (BTJD)
        duration_hours: Transit duration in hours
        config: Optional configuration overrides

    Returns:
        VetterCheckResult with odd/even depth metrics (metrics-only; no pass/fail)

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

    # Guardrail: if duration is a large fraction of the orbit, in/out-of-transit
    # separation becomes ill-defined and per-epoch baselines will be unreliable.
    if duration_days >= 0.5 * period:
        warnings.append("duration_too_long_relative_to_period")
        warnings.append("insufficient_data_for_odd_even_check")
        return VetterCheckResult(
            id="V01",
            name="odd_even_depth",
            passed=None,
            confidence=0.2,
            details={
                # Back-compat keys
                "odd_depth": 0.0,
                "even_depth": 0.0,
                "depth_diff_sigma": 0.0,
                "n_odd_points": 0,
                "n_even_points": 0,
                # New keys
                "n_odd_transits": 0,
                "n_even_transits": 0,
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
                "_metrics_only": True,
            },
        )

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
    global_oot_fallback_count = 0
    epochs_processed = 0

    for ep in unique_epochs:
        epoch_mask = epoch == ep
        epoch_in_transit = epoch_mask & in_transit
        n_in = np.sum(epoch_in_transit)

        if n_in < config.min_points_in_transit_per_epoch:
            continue

        # Define local OOT baseline window around epoch center
        # Cap baseline window to avoid spanning adjacent transits for short periods
        epoch_center = t0 + ep * period
        baseline_half_window = min(
            config.baseline_window_mult * duration_days,
            config.baseline_window_max_fraction_of_period * period,
        )
        local_window = (time >= epoch_center - baseline_half_window) & (
            time <= epoch_center + baseline_half_window
        )
        local_oot = local_window & ~in_transit

        n_oot = np.sum(local_oot)
        used_global_fallback = False
        if n_oot < 5:
            # Fall back to global OOT if local is too sparse
            local_oot = ~in_transit
            n_oot = np.sum(local_oot)
            used_global_fallback = True
            if n_oot < 10:
                continue

        epochs_processed += 1
        if used_global_fallback:
            global_oot_fallback_count += 1

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

    # Warn if global OOT fallback used for ≥50% of epochs
    if epochs_processed > 0 and global_oot_fallback_count >= epochs_processed * 0.5:
        warnings.append(
            f"odd_even_baseline_fallback_global_oot: {global_oot_fallback_count}/{epochs_processed} "
            "epochs used global OOT baseline (local window too sparse)"
        )

    # Separate odd and even epochs
    odd_epochs = {k: v for k, v in epoch_data.items() if k % 2 == 1}
    even_epochs = {k: v for k, v in epoch_data.items() if k % 2 == 0}

    n_odd_transits = len(odd_epochs)
    n_even_transits = len(even_epochs)

    # Total in-transit points per parity
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
        # Metrics-only: insufficient data to interpret odd/even behavior.
        warnings.append("insufficient_data_for_odd_even_check")
        return VetterCheckResult(
            id="V01",
            name="odd_even_depth",
            passed=None,
            confidence=0.2,
            details={
                # Back-compat keys
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
                "_metrics_only": True,
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

    # Metrics-only: host applications decide policy based on returned metrics.
    passed: bool | None = None

    # Confidence
    confidence = _compute_confidence(
        n_odd_transits, n_even_transits, delta_sigma, len(warnings) > 0
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
            "_metrics_only": True,
        },
    )


def check_secondary_eclipse(
    lightcurve: LightCurveData,
    period: float,
    t0: float,
    config: SecondaryEclipseConfig | None = None,
) -> VetterCheckResult:
    """V02: Search for secondary eclipse at phase 0.5.

    Presence of secondary eclipse indicates hot planet (thermal emission)
    or eclipsing binary. Significant secondary suggests EB.

    This implementation uses:
    - Local baseline windows adjacent to the secondary (not global)
    - Widened search window (phase 0.35-0.65) to catch eccentric orbit EBs
    - Red noise inflation for uncertainty estimation
    - Phase coverage metric and event counting
    - Graduated confidence based on data quality

    Args:
        lightcurve: Light curve data
        period: Orbital period in days
        t0: Reference epoch (BTJD)
        config: Optional configuration overrides

    Returns:
        VetterCheckResult with details on secondary eclipse search

    References:
        - Coughlin & Lopez-Morales 2012, AJ 143, 39 (secondary eclipse methodology)
        - Thompson et al. 2018, ApJS 235, 38 (Robovetter significant secondary test)
        - Pont et al. 2006, MNRAS 373, 231 (red noise inflation)
        - Santerne et al. 2013, A&A 557, A139 (eccentric orbit secondary offsets)
    """
    if config is None:
        config = SecondaryEclipseConfig()

    time = lightcurve.time[lightcurve.valid_mask]
    flux = lightcurve.flux[lightcurve.valid_mask]

    warnings: list[str] = []

    # Calculate phase
    phase = ((time - t0) / period) % 1

    # Define regions with configurable widths
    # Secondary window: center +/- half_width (default: 0.35-0.65)
    sec_lo = config.secondary_center - config.secondary_half_width
    sec_hi = config.secondary_center + config.secondary_half_width
    secondary_mask = (phase > sec_lo) & (phase < sec_hi)

    # Adjacent baseline windows (before and after secondary)
    # Before: sec_lo - baseline_half_width to sec_lo
    # After: sec_hi to sec_hi + baseline_half_width
    baseline_before_lo = sec_lo - config.baseline_half_width
    baseline_after_hi = sec_hi + config.baseline_half_width

    # Avoid wrapping around transit at phase 0 (exclude 0-0.1 and 0.9-1.0)
    baseline_before_mask = (phase > max(0.10, baseline_before_lo)) & (phase < sec_lo)
    baseline_after_mask = (phase > sec_hi) & (phase < min(0.90, baseline_after_hi))
    baseline_mask = baseline_before_mask | baseline_after_mask

    secondary_flux = flux[secondary_mask]
    baseline_flux = flux[baseline_mask]
    secondary_time = time[secondary_mask]
    baseline_time = time[baseline_mask]

    n_secondary_points = len(secondary_flux)
    n_baseline_points = len(baseline_flux)

    # Check minimum data requirements
    if n_secondary_points < config.min_secondary_points:
        warnings.append(
            f"Only {n_secondary_points} secondary points, need {config.min_secondary_points}"
        )
    if n_baseline_points < config.min_baseline_points:
        warnings.append(
            f"Only {n_baseline_points} baseline points, need {config.min_baseline_points}"
        )

    if (
        n_secondary_points < config.min_secondary_points
        or n_baseline_points < config.min_baseline_points
    ):
        return VetterCheckResult(
            id="V02",
            name="secondary_eclipse",
            passed=None,
            confidence=0.3,
            details={
                "n_secondary_points": n_secondary_points,
                "n_baseline_points": n_baseline_points,
                "secondary_depth_ppm": 0.0,
                "secondary_depth_err_ppm": 0.0,
                "secondary_depth_sigma": 0.0,
                "secondary_phase_coverage": 0.0,
                "n_secondary_events_effective": 0,
                "warnings": warnings,
                "note": "Insufficient data for secondary eclipse search",
                "_metrics_only": True,
            },
        )

    # Count distinct secondary events (orbital cycles with data in secondary window)
    secondary_epochs = np.floor((secondary_time - t0) / period).astype(int)
    n_secondary_events = len(np.unique(secondary_epochs))

    if n_secondary_events < config.min_secondary_events:
        warnings.append(
            f"Only {n_secondary_events} secondary event(s), need {config.min_secondary_events}"
        )

    # Compute phase coverage within secondary window
    secondary_phases = phase[secondary_mask]
    coverage_bins = np.linspace(sec_lo, sec_hi, config.n_coverage_bins + 1)
    coverage_counts = np.histogram(secondary_phases, bins=coverage_bins)[0]
    n_covered_bins = np.sum(coverage_counts > 0)
    phase_coverage = n_covered_bins / config.n_coverage_bins

    if phase_coverage < config.min_phase_coverage:
        warnings.append(f"Phase coverage {phase_coverage:.2f} < {config.min_phase_coverage}")

    # Calculate secondary depth using local baseline
    baseline_median = float(np.median(baseline_flux))
    secondary_median = float(np.median(secondary_flux))

    if baseline_median <= 0:
        return VetterCheckResult(
            id="V02",
            name="secondary_eclipse",
            passed=None,
            confidence=0.2,
            details={
                "n_secondary_points": n_secondary_points,
                "n_baseline_points": n_baseline_points,
                "warnings": warnings + ["Invalid baseline median <= 0"],
                "note": "Invalid baseline flux",
                "_metrics_only": True,
            },
        )

    secondary_depth = 1.0 - secondary_median / baseline_median

    # Uncertainty estimation with red noise inflation
    baseline_scatter = _robust_std(baseline_flux)
    secondary_err_base = baseline_scatter / np.sqrt(n_secondary_points) / baseline_median

    inflation = 1.0
    if config.use_red_noise_inflation and n_baseline_points >= 20:
        baseline_residuals = baseline_flux - baseline_median
        inflation, rn_success = _compute_red_noise_inflation(
            baseline_residuals, baseline_time, period / 10
        )
        if not rn_success:
            inflation = config.default_inflation
            warnings.append("Red noise estimation failed, using default inflation")
    elif config.use_red_noise_inflation:
        inflation = config.default_inflation
        warnings.append("Insufficient baseline for red noise, using default inflation")

    secondary_err = secondary_err_base * inflation
    secondary_depth_sigma = abs(secondary_depth) / secondary_err if secondary_err > 0 else 0.0

    # Metrics-only: host applications decide policy based on returned metrics.
    passed: bool | None = None

    # Confidence degradation model
    # Base confidence from phase coverage and event count
    if phase_coverage >= 0.7 and n_secondary_events >= 5:
        base_confidence = 0.85
    elif phase_coverage >= 0.5 and n_secondary_events >= 3:
        base_confidence = 0.7
    elif (
        phase_coverage >= config.min_phase_coverage
        and n_secondary_events >= config.min_secondary_events
    ):
        base_confidence = 0.55
    else:
        base_confidence = 0.4

    # If significance is high, confidence increases (still metrics-only; no policy applied here).
    if secondary_depth_sigma >= 5.0:
        base_confidence = min(0.9, base_confidence + 0.1)
    elif secondary_depth_sigma >= 3.0:
        base_confidence = min(0.85, base_confidence + 0.05)

    # Degrade if warnings
    if warnings:
        base_confidence *= 0.9

    confidence = round(min(0.95, max(0.2, base_confidence)), 3)

    # Convert to ppm for output
    secondary_depth_ppm = secondary_depth * 1e6
    secondary_err_ppm = secondary_err * 1e6

    return VetterCheckResult(
        id="V02",
        name="secondary_eclipse",
        passed=passed,
        confidence=confidence,
        details={
            # Back-compat keys
            "secondary_depth": round(secondary_depth, 6),
            "secondary_depth_sigma": round(secondary_depth_sigma, 2),
            "baseline_flux": round(baseline_median, 6),
            "n_secondary_points": n_secondary_points,
            # New keys
            "secondary_depth_ppm": round(secondary_depth_ppm, 1),
            "secondary_depth_err_ppm": round(secondary_err_ppm, 2),
            "secondary_phase_coverage": round(phase_coverage, 3),
            "n_secondary_events_effective": n_secondary_events,
            "n_baseline_points": n_baseline_points,
            "red_noise_inflation": round(inflation, 2),
            "search_window": [round(sec_lo, 3), round(sec_hi, 3)],
            "warnings": warnings,
            "_metrics_only": True,
        },
    )


def check_duration_consistency(
    period: float,
    duration_hours: float,
    stellar: StellarParameters | None,
) -> VetterCheckResult:
    """V03: Check transit duration vs stellar density expectation.

    Transit duration depends on stellar density. Large mismatches between the
    observed duration and a simple expectation can indicate host/parameter
    mismatch or a non-planet scenario, but this function is metrics-only and
    does not apply policy.

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

    ratio = (
        duration_hours / expected_duration_hours if expected_duration_hours > 0 else float("inf")
    )
    warnings: list[str] = []

    # Confidence depends on whether we have stellar parameters
    if density_corrected:
        confidence = 0.85  # High confidence with stellar density correction
    elif stellar is not None:
        confidence = 0.5  # Moderate confidence with some stellar info
    else:
        confidence = 0.2  # Low confidence without stellar info

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
        passed=None,
        confidence=round(confidence, 3),
        details={**details, "warnings": warnings, "_metrics_only": True},
    )


def check_depth_stability(
    lightcurve: LightCurveData,
    period: float,
    t0: float,
    duration_hours: float,
    config: DepthStabilityConfig | None = None,
) -> VetterCheckResult:
    """V04: Check depth consistency across individual transits.

    Variable depth suggests blended eclipsing binary or systematic issues.
    Real planets have consistent depths.

    This implementation uses:
    - Per-transit box depth fitting with local baselines
    - Chi-squared ratio metric (observed vs expected scatter)
    - Red noise inflation for uncertainty estimation
    - Outlier epoch detection and flagging
    - Graduated confidence based on N_transits

    Args:
        lightcurve: Light curve data
        period: Orbital period in days
        t0: Reference epoch (BTJD)
        duration_hours: Transit duration in hours
        config: Optional configuration overrides

    Returns:
        VetterCheckResult with depth stability metrics

    References:
        - Thompson et al. 2018, ApJS 235, 38 (depth consistency tests)
        - Pont et al. 2006, MNRAS 373, 231 (correlated noise)
        - Wang & Espinoza 2023, arXiv:2311.02154 (per-transit depth fitting)
    """
    if config is None:
        config = DepthStabilityConfig()

    time = lightcurve.time[lightcurve.valid_mask]
    flux = lightcurve.flux[lightcurve.valid_mask]

    duration_days = duration_hours / 24.0
    warnings: list[str] = []

    # Guardrail: if duration is a large fraction of the orbit, in/out-of-transit
    # separation becomes ill-defined, so depth stability is not meaningful.
    if duration_days >= 0.5 * period:
        warnings.append("duration_too_long_relative_to_period")
        return VetterCheckResult(
            id="V04",
            name="depth_stability",
            passed=None,
            confidence=0.2,
            details={
                "n_transits_measured": 0,
                "depths_ppm": [],
                "depth_scatter_ppm": 0.0,
                "expected_scatter_ppm": 0.0,
                "chi2_reduced": 0.0,
                "warnings": warnings,
                "note": "Duration too long relative to period for depth stability check",
                "_metrics_only": True,
            },
        )

    # Calculate epoch index for each point (same logic as odd/even)
    epoch = np.floor((time - t0 + period / 2) / period).astype(int)

    # Phase distance from transit center
    phase = ((time - t0) / period) % 1
    phase_dist = np.minimum(phase, 1 - phase)

    # In-transit mask
    half_dur_phase = 0.5 * (duration_days / period)
    in_transit = phase_dist < half_dur_phase

    unique_epochs = np.unique(epoch)

    # Per-epoch depth extraction with local baselines
    epoch_depths: list[float] = []
    epoch_sigmas: list[float] = []
    epoch_indices: list[int] = []
    global_oot_fallback_count = 0
    epochs_processed = 0

    for ep in unique_epochs:
        epoch_mask = epoch == ep
        epoch_in_transit = epoch_mask & in_transit
        n_in = int(np.sum(epoch_in_transit))

        if n_in < config.min_points_per_epoch:
            continue

        # Define local OOT baseline window
        epoch_center = t0 + ep * period
        baseline_half_window = config.baseline_window_mult * duration_days
        local_window = (time >= epoch_center - baseline_half_window) & (
            time <= epoch_center + baseline_half_window
        )
        local_oot = local_window & ~in_transit

        n_oot = int(np.sum(local_oot))
        used_global_fallback = False
        if n_oot < 5:
            # Fall back to global OOT
            local_oot = ~in_transit
            n_oot = int(np.sum(local_oot))
            used_global_fallback = True
            if n_oot < 10:
                continue

        epochs_processed += 1
        if used_global_fallback:
            global_oot_fallback_count += 1

        # Compute local baseline
        baseline_flux = flux[local_oot]
        baseline = float(np.median(baseline_flux))

        if baseline <= 0:
            continue

        # Compute depth for this epoch
        in_flux = flux[epoch_in_transit]
        depth_k = 1.0 - float(np.median(in_flux)) / baseline

        # Skip if not a real dip
        if depth_k <= 0:
            continue

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

        epoch_depths.append(depth_k)
        epoch_sigmas.append(sigma_k)
        epoch_indices.append(int(ep))

    # Warn if global OOT fallback used for >=50% of epochs
    if epochs_processed > 0 and global_oot_fallback_count >= epochs_processed * 0.5:
        warnings.append(
            f"depth_stability_baseline_fallback: {global_oot_fallback_count}/{epochs_processed} "
            "epochs used global OOT baseline"
        )

    n_transits = len(epoch_depths)

    if n_transits < 2:
        return VetterCheckResult(
            id="V04",
            name="depth_stability",
            passed=None,
            confidence=0.3,
            details={
                "n_transits_measured": n_transits,
                "depths_ppm": [],
                "depth_scatter_ppm": 0.0,
                "expected_scatter_ppm": 0.0,
                "chi2_reduced": 0.0,
                "warnings": warnings,
                "note": "Insufficient transits for depth stability check",
                "_metrics_only": True,
            },
        )

    depths_arr = np.array(epoch_depths)
    sigmas_arr = np.array(epoch_sigmas)

    mean_depth = float(np.mean(depths_arr))
    median_depth = float(np.median(depths_arr))
    std_depth = float(np.std(depths_arr))

    rms_scatter = std_depth / mean_depth if mean_depth > 0 else 0.0

    # Compute expected scatter from individual uncertainties
    # Expected: sqrt(sum(sigma_k^2)) / N
    expected_scatter = float(np.sqrt(np.sum(sigmas_arr**2))) / n_transits if n_transits > 0 else 0.0

    # Chi-squared: sum((depth_k - mean)^2 / sigma_k^2)
    if np.all(sigmas_arr > 0):
        residuals = depths_arr - mean_depth
        chi2 = float(np.sum((residuals / sigmas_arr) ** 2))
        dof = n_transits - 1  # 1 parameter (mean)
        chi2_reduced = chi2 / dof if dof > 0 else 0.0
    else:
        chi2 = 0.0
        chi2_reduced = 0.0
        warnings.append("Some sigma values are zero, chi2 unreliable")

    # Outlier detection using MAD
    outlier_epochs: list[int] = []
    if n_transits >= config.min_transits_for_confidence:
        mad = float(np.median(np.abs(depths_arr - median_depth)))
        mad_scale = mad * 1.4826  # Scale to std
        if mad_scale > 0:
            outlier_mask = np.abs(depths_arr - median_depth) > config.outlier_sigma * mad_scale
            outlier_epochs = [epoch_indices[i] for i in range(n_transits) if outlier_mask[i]]
            if outlier_epochs:
                warnings.append(f"Outlier epochs detected: {outlier_epochs}")

    # Metrics-only: host applications decide policy based on returned metrics.
    passed: bool | None = None

    # Graduated confidence by N_transits
    if n_transits < config.min_transits_for_confidence:
        base_confidence = 0.35
    elif n_transits < 5:
        base_confidence = 0.55
    elif n_transits < 10:
        base_confidence = 0.7
    elif n_transits < 20:
        base_confidence = 0.8
    else:
        base_confidence = 0.85

    # Note: do not use chi2 thresholds for policy decisions here; confidence reflects
    # data quantity/quality (n_transits, outliers, warnings), not pass/fail.

    # Degrade if outliers or warnings
    if outlier_epochs:
        base_confidence *= 0.85
    if warnings and base_confidence > 0.5:
        base_confidence *= 0.95

    confidence = round(min(0.95, max(0.2, base_confidence)), 3)

    # Convert to ppm
    depths_ppm = [round(d * 1e6, 1) for d in depths_arr[:20]]  # Cap at 20
    depth_scatter_ppm = std_depth * 1e6
    expected_scatter_ppm = expected_scatter * 1e6
    mean_depth_ppm = mean_depth * 1e6

    return VetterCheckResult(
        id="V04",
        name="depth_stability",
        passed=passed,
        confidence=confidence,
        details={
            # Legacy keys
            "mean_depth": round(mean_depth, 6),
            "std_depth": round(std_depth, 6),
            "rms_scatter": round(rms_scatter, 4),
            "n_transits_measured": n_transits,
            "individual_depths": [round(d, 6) for d in depths_arr[:10]],
            # New keys
            "mean_depth_ppm": round(mean_depth_ppm, 1),
            "depths_ppm": depths_ppm,
            "depth_scatter_ppm": round(depth_scatter_ppm, 1),
            "expected_scatter_ppm": round(expected_scatter_ppm, 2),
            "chi2_reduced": round(chi2_reduced, 2),
            "outlier_epochs": outlier_epochs,
            "warnings": warnings,
            "method": "per_epoch_local_baseline",
            "_metrics_only": True,
        },
    )


def _trapezoid_model(
    phase: np.ndarray,
    t_flat_phase: float,
    t_total_phase: float,
    depth: float,
) -> np.ndarray:
    """Symmetric trapezoid transit model.

    Args:
        phase: Phase array centered on transit (0 = mid-transit)
        t_flat_phase: Flat-bottom duration in phase units
        t_total_phase: Total transit duration in phase units
        depth: Transit depth (fractional)

    Returns:
        Model flux array (1.0 = baseline, 1.0 - depth = bottom)
    """
    half_flat = t_flat_phase / 2
    half_total = t_total_phase / 2

    flux = np.ones_like(phase)

    # Pure V-shape case: no flat bottom
    if half_flat <= 0:
        # Linear from baseline to depth at center
        in_transit = np.abs(phase) < half_total
        if np.any(in_transit):
            # Linear ramp: depth at center, 0 at edges
            flux[in_transit] = 1 - depth * (1 - np.abs(phase[in_transit]) / half_total)
        return flux

    # Flat bottom region
    flat_mask = np.abs(phase) < half_flat
    flux[flat_mask] = 1 - depth

    # Ingress/egress slopes
    if half_total > half_flat:
        slope_width = half_total - half_flat
        ingress_mask = (phase < -half_flat) & (phase > -half_total)
        egress_mask = (phase > half_flat) & (phase < half_total)

        # Ingress: goes from 1 at -half_total to (1-depth) at -half_flat
        if np.any(ingress_mask):
            frac = (-phase[ingress_mask] - half_flat) / slope_width
            flux[ingress_mask] = (1 - depth) + depth * frac

        # Egress: goes from (1-depth) at +half_flat to 1 at +half_total
        if np.any(egress_mask):
            frac = (phase[egress_mask] - half_flat) / slope_width
            flux[egress_mask] = (1 - depth) + depth * frac

    return flux


def _fit_trapezoid_grid_search(
    phase: np.ndarray,
    flux: np.ndarray,
    t_total_phase: float,
    n_grid: int = 20,
) -> tuple[float, float, float]:
    """Fit trapezoid model using grid search over tF/tT ratio.

    Args:
        phase: Phase array centered on transit
        flux: Normalized flux array
        t_total_phase: Total transit duration in phase units (from input ephemeris)
        n_grid: Number of grid points for tF/tT ratio search

    Returns:
        Tuple of (best_tflat_ttotal_ratio, best_depth, min_chi2)
    """
    # Grid of tF/tT ratios from 0 (V-shape) to 1 (box)
    tflat_ttotal_ratios = np.linspace(0, 1, n_grid)

    best_ratio = 0.5
    best_depth = 0.001
    min_chi2 = float("inf")

    for ratio in tflat_ttotal_ratios:
        t_flat_phase = ratio * t_total_phase

        # Estimate depth by linear least squares for this shape:
        # model = 1 - depth * shape(phase), with shape in [0, 1]
        shape = 1.0 - _trapezoid_model(phase, t_flat_phase, t_total_phase, depth=1.0)
        y = 1.0 - flux
        denom = float(np.sum(shape**2))
        if denom <= 0:
            continue
        depth_estimate = float(np.sum(y * shape) / denom)
        if not np.isfinite(depth_estimate) or depth_estimate <= 0:
            depth_estimate = 0.001

        model = 1.0 - depth_estimate * shape
        residuals = flux - model
        chi2 = float(np.sum(residuals**2))

        if chi2 < min_chi2:
            min_chi2 = chi2
            best_ratio = ratio
            best_depth = depth_estimate

    return float(best_ratio), float(best_depth), float(min_chi2)


def _compute_v_shape_confidence(
    n_in_transit: int,
    transit_coverage: float,
    has_warnings: bool,
) -> float:
    """Compute confidence score for V-shape check.

    Args:
        n_in_transit: Number of in-transit points
        transit_coverage: Fraction of transit phases with data
        has_warnings: Whether warnings were issued
    Returns:
        Confidence score in [0, 1]
    """
    # Base confidence from data quantity
    if n_in_transit < 10:
        base = 0.2
    elif n_in_transit < 30:
        base = 0.5
    elif n_in_transit < 100:
        base = 0.7
    else:
        base = 0.85

    # Adjust for coverage
    if transit_coverage >= 0.8:
        base = min(0.95, base * 1.1)
    elif transit_coverage < 0.6:
        base *= 0.8

    # Degrade if warnings
    if has_warnings:
        base *= 0.9

    return round(min(0.95, base), 3)


def check_v_shape(
    lightcurve: LightCurveData,
    period: float,
    t0: float,
    duration_hours: float,
    config: VShapeConfig | None = None,
) -> VetterCheckResult:
    """V05: Distinguish U-shaped (planet) vs V-shaped (grazing EB) transits.

    Uses trapezoid model fitting to extract tF/tT ratio (flat-bottom to total
    duration), the standard shape discriminant in the literature.

    Args:
        lightcurve: Light curve data
        period: Orbital period in days
        t0: Reference epoch (BTJD)
        duration_hours: Transit duration in hours
        config: Optional configuration overrides

    Returns:
        VetterCheckResult with shape analysis including tF/tT ratio

    References:
        [1] Seager & Mallen-Ornelas 2003, ApJ 585, 1038 (2003ApJ...585.1038S)
            Section 3: Transit shape parameters tF/tT and impact parameter b
        [2] Kipping 2010, MNRAS 407, 301 (arXiv:1004.3819)
            Transit duration expressions and T14/T23 definitions
        [3] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T)
            Section 3.1: Not Transit-Like (V-shape) metric in DR25 Robovetter
        [4] Prsa et al. 2011, AJ 141, 83 (2011AJ....141...83P)
            EB morphology classification; V-shape vs U-shape distinction
    """
    if config is None:
        config = VShapeConfig()

    time = lightcurve.time[lightcurve.valid_mask]
    flux = lightcurve.flux[lightcurve.valid_mask]

    duration_days = duration_hours / 24.0
    warnings: list[str] = []

    # Guardrail: if duration is a large fraction of the orbit, the concept of a
    # localized transit shape becomes ill-defined.
    if duration_days >= 0.5 * period:
        warnings.append("duration_too_long_relative_to_period")
        return VetterCheckResult(
            id="V05",
            name="v_shape",
            passed=None,
            confidence=0.2,
            details={
                # Legacy keys
                "depth_bottom": 0.0,
                "depth_edge": 0.0,
                "shape_ratio": 2.0,
                "n_bottom_points": 0,
                "n_edge_points": 0,
                # New keys
                "status": "invalid_duration",
                "t_flat_hours": 0.0,
                "t_total_hours": duration_hours,
                "tflat_ttotal_ratio": 0.5,
                "tflat_ttotal_ratio_err": 0.5,
                "shape_metric_uncertainty": 0.5,
                "transit_coverage": 0.0,
                "n_in_transit": 0,
                "n_baseline": 0,
                "warnings": warnings,
                "method": "trapezoid_grid_search",
                "_metrics_only": True,
            },
        )

    # Calculate phase centered on transit (-0.5 to 0.5, transit at 0)
    phase = ((time - t0) / period + 0.5) % 1 - 0.5

    # Duration in phase units
    t_total_phase = duration_days / period
    half_dur_phase = t_total_phase / 2

    # Define in-transit and baseline masks
    in_transit_mask = np.abs(phase) < half_dur_phase * 1.2  # Slight buffer
    baseline_mask = (np.abs(phase) > half_dur_phase * 1.5) & (np.abs(phase) < 0.25)

    n_in_transit = int(np.sum(in_transit_mask))
    n_baseline = int(np.sum(baseline_mask))

    # Compute transit coverage: fraction of transit phase bins with data
    n_phase_bins = 20
    phase_bins = np.linspace(-half_dur_phase, half_dur_phase, n_phase_bins + 1)
    bins_with_data = 0
    for i in range(n_phase_bins):
        bin_mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
        if np.sum(bin_mask) >= 1:
            bins_with_data += 1
    transit_coverage = bins_with_data / n_phase_bins

    # Check minimum data requirements
    insufficient_data = False
    if n_in_transit < config.min_points_in_transit:
        warnings.append(
            f"Only {n_in_transit} in-transit points, need {config.min_points_in_transit}"
        )
        insufficient_data = True
    if n_baseline < 10:
        warnings.append(f"Only {n_baseline} baseline points, need 10")
        insufficient_data = True
    if transit_coverage < config.min_transit_coverage:
        warnings.append(
            f"Transit coverage {transit_coverage:.2f} below minimum {config.min_transit_coverage}"
        )
        insufficient_data = True

    if insufficient_data:
        # Metrics-only: insufficient data to interpret transit shape.
        return VetterCheckResult(
            id="V05",
            name="v_shape",
            passed=None,
            confidence=0.2,
            details={
                # Legacy keys
                "depth_bottom": 0.0,
                "depth_edge": 0.0,
                "shape_ratio": 2.0,
                "n_bottom_points": 0,
                "n_edge_points": 0,
                # New keys
                "status": "insufficient_data",
                "t_flat_hours": 0.0,
                "t_total_hours": duration_hours,
                "tflat_ttotal_ratio": 0.5,
                "tflat_ttotal_ratio_err": 0.5,
                "shape_metric_uncertainty": 0.5,
                "transit_coverage": round(transit_coverage, 3),
                "n_in_transit": n_in_transit,
                "n_baseline": n_baseline,
                "warnings": warnings,
                "method": "trapezoid_grid_search",
                "_metrics_only": True,
            },
        )

    # Normalize flux using baseline
    baseline_flux = flux[baseline_mask]
    baseline_median = float(np.median(baseline_flux))
    normalized_flux = flux / baseline_median

    # Select in-transit data for fitting
    in_transit_phase = phase[in_transit_mask]
    in_transit_flux = normalized_flux[in_transit_mask]

    # Fit trapezoid model using grid search
    tflat_ttotal_ratio, depth, _ = _fit_trapezoid_grid_search(
        in_transit_phase, in_transit_flux, t_total_phase
    )

    # Bootstrap uncertainty estimation
    rng = np.random.default_rng(42)
    bootstrap_ratios: list[float] = []

    for _ in range(config.n_bootstrap):
        # Resample in-transit points with replacement
        indices = rng.choice(len(in_transit_phase), size=len(in_transit_phase), replace=True)
        boot_phase = in_transit_phase[indices]
        boot_flux = in_transit_flux[indices]

        # Fit on bootstrap sample
        boot_ratio, _, _ = _fit_trapezoid_grid_search(boot_phase, boot_flux, t_total_phase)
        bootstrap_ratios.append(boot_ratio)

    # Compute uncertainty from bootstrap distribution
    if len(bootstrap_ratios) > 10:
        lower_pct = (1 - config.bootstrap_ci) / 2 * 100
        upper_pct = (1 + config.bootstrap_ci) / 2 * 100
        lower_bound = float(np.percentile(bootstrap_ratios, lower_pct))
        upper_bound = float(np.percentile(bootstrap_ratios, upper_pct))
        tflat_ttotal_ratio_err = (upper_bound - lower_bound) / 2
    else:
        tflat_ttotal_ratio_err = 0.2  # Default uncertainty

    # Convert to physical units
    t_flat_hours = tflat_ttotal_ratio * duration_hours
    t_total_hours = duration_hours

    # Compute depth in ppm
    depth_ppm = depth * 1e6

    # Compute confidence
    confidence = _compute_v_shape_confidence(n_in_transit, transit_coverage, len(warnings) > 0)

    half_dur = duration_days / period / 2
    ingress_mask = (phase > -half_dur) & (phase < -half_dur / 2)
    bottom_mask = (phase > -half_dur / 4) & (phase < half_dur / 4)
    egress_mask = (phase > half_dur / 2) & (phase < half_dur)

    bottom_flux = flux[bottom_mask]
    ingress_flux = flux[ingress_mask]
    egress_flux = flux[egress_mask]

    if len(bottom_flux) > 0:
        depth_bottom = 1.0 - float(np.median(bottom_flux)) / baseline_median
    else:
        depth_bottom = depth

    edge_flux_arr = (
        np.concatenate([ingress_flux, egress_flux])
        if len(ingress_flux) > 0 or len(egress_flux) > 0
        else np.array([])
    )

    if len(edge_flux_arr) > 3:
        depth_edge = 1.0 - float(np.median(edge_flux_arr)) / baseline_median
    else:
        depth_edge = depth_bottom * 0.5

    # Legacy shape_ratio (retained for stable outputs)
    shape_ratio = depth_bottom / depth_edge if depth_edge > 0 and depth_bottom > 0 else 2.0

    passed: bool | None = None
    return VetterCheckResult(
        id="V05",
        name="v_shape",
        passed=passed,
        confidence=confidence,
        details={
            # Legacy keys (preserved for backward compatibility)
            "depth_bottom": round(depth_bottom, 6),
            "depth_edge": round(depth_edge, 6),
            "shape_ratio": round(shape_ratio, 3),
            "n_bottom_points": len(bottom_flux),
            "n_edge_points": len(edge_flux_arr),
            # New keys
            "status": "ok",
            "t_flat_hours": round(t_flat_hours, 4),
            "t_total_hours": round(t_total_hours, 4),
            "tflat_ttotal_ratio": round(tflat_ttotal_ratio, 4),
            "tflat_ttotal_ratio_err": round(tflat_ttotal_ratio_err, 4),
            "shape_metric_uncertainty": round(tflat_ttotal_ratio_err, 4),
            "depth_ppm": round(depth_ppm, 1),
            "transit_coverage": round(transit_coverage, 3),
            "n_in_transit": n_in_transit,
            "n_baseline": n_baseline,
            "warnings": warnings,
            "method": "trapezoid_grid_search",
            "_metrics_only": True,
        },
    )
