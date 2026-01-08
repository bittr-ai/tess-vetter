"""Base framework for transit candidate vetting checks.

This module provides:
- VetterCheck: Abstract base class defining the interface for all vetting checks
- CheckConfig: Configuration dataclass for check parameters
- CheckID: Enum of standard check identifiers (V01-V13)
- VetterRegistry: Registry for managing vetting check instances
- Aggregation logic to combine check results into Verdict (PASS/WARN/REJECT)
- Common utilities for vetter implementations

Design decisions:
- Each check is a class that encapsulates its configuration and logic
- Checks receive a TransitCandidate and optional LightCurveData
- All checks return VetterCheckResult with standardized fields
- Confidence values indicate how certain we are about the result
- Aggregation follows tiered logic: LC failures are weighted more heavily

Check Tiers:
- Tier 1 (LC-only): V01-V05 - Always available, use only light curve data
- Tier 2 (Catalog): V06-V07 - Require local catalog cache
- Tier 3 (Pixel): V08-V10 - Require TPF data (deferred to v2)
- Tier 4 (Advanced): V11-V13 - Additional validation checks
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from bittr_tess_vetter.domain.detection import (
    Disposition,
    TransitCandidate,
    ValidationResult,
    Verdict,
    VetterCheckResult,
)

if TYPE_CHECKING:
    from bittr_tess_vetter.domain.lightcurve import LightCurveData
    from bittr_tess_vetter.domain.target import StellarParameters


# ---------------------------------------------------------------------------
# Check ID Registry
# ---------------------------------------------------------------------------


class CheckID(str, Enum):
    """Standard vetting check IDs.

    CANONICAL MAPPING (matches lc_checks.py and handlers/validate.py):

    LC-only checks (V01-V05): Always available, use only light curve data
    - V01: Odd-Even Depth comparison (detect EBs at 2x period)
    - V02: Secondary Eclipse search (detect EBs via phase 0.5 eclipse)
    - V03: Duration Consistency (check vs stellar density expectation)
    - V04: Depth Stability (consistent depth across transits)
    - V05: V-Shape analysis (U-shaped planet vs V-shaped grazing EB)

    Catalog checks (V06-V07): Require local catalog cache
    - V06: Nearby EB Search (check for known EBs in aperture)
    - V07: Known FP Match (cross-reference FP catalogs)

    Pixel checks (V08-V10): Require TPF data (deferred to v2)
    - V08: Centroid Shift (in-transit vs out-of-transit centroid)
    - V09: Pixel-Level LC (per-pixel light curves)
    - V10: Aperture Dependence (depth vs aperture size)

    Advanced checks (V11-V13): Additional validation checks
    - V11: Stellar Density Check (compare transit-derived vs TIC density)
    - V12: Single Transit Check (flag single-transit candidates)
    - V13: Ephemeris Stability Check (verify linear ephemeris)
    """

    # Tier 1: LC-only checks
    V01_ODD_EVEN_DEPTH = "V01"
    V02_SECONDARY_ECLIPSE = "V02"
    V03_DURATION_CONSISTENCY = "V03"
    V04_DEPTH_STABILITY = "V04"
    V05_V_SHAPE = "V05"

    # Tier 2: Catalog checks
    V06_NEARBY_EB_SEARCH = "V06"
    V07_KNOWN_FP_MATCH = "V07"

    # Tier 3: Pixel checks (deferred)
    V08_CENTROID_SHIFT = "V08"
    V09_PIXEL_LEVEL_LC = "V09"
    V10_APERTURE_DEPENDENCE = "V10"

    # Tier 4: Advanced checks
    V11_STELLAR_DENSITY = "V11"
    V12_SINGLE_TRANSIT = "V12"
    V13_EPHEMERIS_STABILITY = "V13"


# ---------------------------------------------------------------------------
# Check Names (Human-Readable)
# ---------------------------------------------------------------------------

CHECK_NAMES: dict[str, str] = {
    "V01": "odd_even_depth",
    "V02": "secondary_eclipse",
    "V03": "duration_consistency",
    "V04": "depth_stability",
    "V05": "v_shape",
    "V06": "nearby_eb_search",
    "V07": "known_fp_match",
    "V08": "centroid_shift",
    "V09": "pixel_level_lc",
    "V10": "aperture_dependence",
    "V11": "stellar_density",
    "V12": "single_transit",
    "V13": "ephemeris_stability",
}

# Tier membership for checks
LC_ONLY_CHECKS = {"V01", "V02", "V03", "V04", "V05"}
CATALOG_CHECKS = {"V06", "V07"}
PIXEL_CHECKS = {"V08", "V09", "V10"}
ADVANCED_CHECKS = {"V11", "V12", "V13"}


# ---------------------------------------------------------------------------
# VetterResult Factory
# ---------------------------------------------------------------------------

# Re-export VetterCheckResult as VetterResult for convenience
VetterResult = VetterCheckResult


def make_result(
    check_id: str,
    passed: bool,
    confidence: float,
    details: dict[str, Any] | None = None,
) -> VetterResult:
    """Factory function to create a VetterResult.

    Args:
        check_id: Check identifier (V01-V13)
        passed: Whether the check passed
        confidence: Confidence in the result (0-1)
        details: Check-specific details

    Returns:
        VetterResult instance

    Example:
        result = make_result(
            "V01",
            passed=True,
            confidence=0.95,
            details={"odd_depth": 0.0012, "even_depth": 0.0011}
        )
    """
    name = CHECK_NAMES.get(check_id, "unknown")
    return VetterResult(
        id=check_id,
        name=name,
        passed=passed,
        confidence=min(1.0, max(0.0, confidence)),  # Clamp to [0, 1]
        details=details or {},
    )


# ---------------------------------------------------------------------------
# Check Configuration
# ---------------------------------------------------------------------------


@dataclass
class CheckConfig:
    """Configuration for a vetting check.

    Attributes:
        enabled: Whether this check should be run.
        threshold: Primary threshold for pass/fail decision.
        additional: Check-specific additional parameters.
    """

    enabled: bool = True
    threshold: float | None = None
    additional: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# VetterCheck Abstract Base Class
# ---------------------------------------------------------------------------


class VetterCheck(ABC):
    """Abstract base class for transit candidate vetting checks.

    Each vetting check evaluates a specific aspect of a transit candidate
    to determine if it is consistent with a genuine planetary transit.

    Attributes:
        id: Unique identifier (V01, V02, etc.) - class attribute
        name: Human-readable name for the check - class attribute
        config: Configuration parameters for the check - instance attribute

    Example:
        class OddEvenDepthCheck(VetterCheck):
            id = "V01"
            name = "odd_even_depth"

            @classmethod
            def _default_config(cls) -> CheckConfig:
                return CheckConfig(threshold=3.0)  # 3-sigma

            def run(self, candidate, lightcurve=None, stellar=None):
                # Compare odd vs even transit depths
                odd_depth, even_depth = self._measure_odd_even(lightcurve, candidate)
                diff_sigma = abs(odd_depth - even_depth) / uncertainty
                passed = diff_sigma < self.config.threshold
                return make_result(
                    self.id,
                    passed=passed,
                    confidence=0.95,
                    details={"odd_depth": odd_depth, "even_depth": even_depth}
                )
    """

    # Class attributes to be overridden by subclasses
    id: ClassVar[str]  # e.g., "V01"
    name: ClassVar[str]  # e.g., "odd_even_depth"

    def __init__(self, config: CheckConfig | None = None) -> None:
        """Initialize the vetting check.

        Args:
            config: Optional configuration. If None, uses default config.
        """
        self.config = config or self._default_config()

    @classmethod
    @abstractmethod
    def _default_config(cls) -> CheckConfig:
        """Return the default configuration for this check.

        Subclasses must implement this to provide sensible defaults.

        Returns:
            CheckConfig with default values for this check.
        """
        ...

    @abstractmethod
    def run(
        self,
        candidate: TransitCandidate,
        lightcurve: LightCurveData | None = None,
        stellar: StellarParameters | None = None,
    ) -> VetterCheckResult:
        """Execute the vetting check on a transit candidate.

        Args:
            candidate: Transit candidate parameters to evaluate.
            lightcurve: Optional light curve data for LC-based checks.
            stellar: Optional stellar parameters for physics-based checks.

        Returns:
            VetterCheckResult with pass/fail status and details.
        """
        ...

    @property
    def tier(self) -> str:
        """Get the tier for this check (lc_only, catalog, pixel, or advanced)."""
        if self.id in LC_ONLY_CHECKS:
            return "lc_only"
        if self.id in CATALOG_CHECKS:
            return "catalog"
        if self.id in PIXEL_CHECKS:
            return "pixel"
        if self.id in ADVANCED_CHECKS:
            return "advanced"
        return "unknown"

    def __repr__(self) -> str:
        """Return string representation of the check."""
        return f"{self.__class__.__name__}(id={self.id!r}, name={self.name!r})"


# ---------------------------------------------------------------------------
# Vetter Registry
# ---------------------------------------------------------------------------


class VetterRegistry:
    """Registry for vetting checks.

    Allows dynamic registration and lookup of vetting checks.
    Supports filtering by tier (LC-only, catalog, pixel).

    Example:
        registry = VetterRegistry()
        registry.register(OddEvenDepthCheck())
        registry.register(SecondaryEclipseCheck())

        # Get all LC-only checks
        lc_checks = registry.get_lc_only_checks()

        # Run all checks
        results = registry.run_all(candidate, lightcurve)
    """

    def __init__(self) -> None:
        self._checks: dict[str, VetterCheck] = {}

    def register(self, check: VetterCheck) -> None:
        """Register a vetting check.

        Args:
            check: VetterCheck instance to register
        """
        self._checks[check.id] = check

    def get(self, check_id: str) -> VetterCheck | None:
        """Get a check by ID.

        Args:
            check_id: Check identifier (V01-V13)

        Returns:
            VetterCheck instance or None if not found
        """
        return self._checks.get(check_id)

    def get_lc_only_checks(self) -> list[VetterCheck]:
        """Get all LC-only checks (V01-V05).

        Returns:
            List of registered LC-only checks, sorted by ID
        """
        return [c for cid, c in sorted(self._checks.items()) if cid in LC_ONLY_CHECKS]

    def get_catalog_checks(self) -> list[VetterCheck]:
        """Get all catalog checks (V06-V07).

        Returns:
            List of registered catalog checks, sorted by ID
        """
        return [c for cid, c in sorted(self._checks.items()) if cid in CATALOG_CHECKS]

    def get_pixel_checks(self) -> list[VetterCheck]:
        """Get all pixel checks (V08-V10).

        Returns:
            List of registered pixel checks, sorted by ID
        """
        return [c for cid, c in sorted(self._checks.items()) if cid in PIXEL_CHECKS]

    def get_all_checks(self) -> list[VetterCheck]:
        """Get all registered checks.

        Returns:
            List of all registered checks, sorted by ID
        """
        return [c for _, c in sorted(self._checks.items())]

    def list_ids(self) -> list[str]:
        """List all registered check IDs.

        Returns:
            Sorted list of check IDs
        """
        return sorted(self._checks.keys())

    def run_checks(
        self,
        checks: list[VetterCheck],
        candidate: TransitCandidate,
        lightcurve: LightCurveData | None = None,
        stellar: StellarParameters | None = None,
    ) -> list[VetterCheckResult]:
        """Run a list of checks on a candidate.

        Args:
            checks: List of VetterCheck instances to run
            candidate: Transit candidate to evaluate
            lightcurve: Optional light curve data
            stellar: Optional stellar parameters

        Returns:
            List of VetterCheckResult from each check
        """
        results = []
        for check in checks:
            if check.config.enabled:
                result = check.run(candidate, lightcurve, stellar)
                results.append(result)
        return results

    def run_lc_only(
        self,
        candidate: TransitCandidate,
        lightcurve: LightCurveData,
        stellar: StellarParameters | None = None,
    ) -> list[VetterCheckResult]:
        """Run all LC-only checks (V01-V05).

        Args:
            candidate: Transit candidate to evaluate
            lightcurve: Light curve data (required for LC checks)
            stellar: Optional stellar parameters

        Returns:
            List of VetterCheckResult from LC-only checks
        """
        return self.run_checks(self.get_lc_only_checks(), candidate, lightcurve, stellar)


# Global registry instance
_registry = VetterRegistry()


def register_check(check: VetterCheck) -> VetterCheck:
    """Register a check with the global registry.

    Args:
        check: VetterCheck instance to register

    Returns:
        The same check (for decorator-style usage)
    """
    _registry.register(check)
    return check


def get_check(check_id: str) -> VetterCheck | None:
    """Get a check from the global registry.

    Args:
        check_id: Check identifier (V01-V13)

    Returns:
        VetterCheck instance or None
    """
    return _registry.get(check_id)


def get_registry() -> VetterRegistry:
    """Get the global registry instance.

    Returns:
        The global VetterRegistry singleton
    """
    return _registry


# ---------------------------------------------------------------------------
# Verdict Aggregation
# ---------------------------------------------------------------------------


@dataclass
class AggregationConfig:
    """Configuration for verdict aggregation.

    Controls how individual check results are combined into a final verdict.

    Attributes:
        lc_failure_threshold: Number of LC failures to trigger REJECT (default: 2)
        mixed_failure_threshold: Combined failures (LC + catalog) for REJECT
        low_confidence_threshold: Below this, treat as marginal (default: 0.5)
        warn_on_low_confidence: Whether to WARN on low-confidence passes
    """

    lc_failure_threshold: int = 2
    mixed_failure_threshold: int = 2  # e.g., 1 LC + 1 catalog
    low_confidence_threshold: float = 0.5
    warn_on_low_confidence: bool = True


DEFAULT_AGGREGATION_CONFIG = AggregationConfig()


def compute_verdict(
    lc_checks: list[VetterCheckResult],
    catalog_checks: list[VetterCheckResult] | None = None,
    config: AggregationConfig | None = None,
) -> Verdict:
    """Aggregate check results into a final verdict.

    Logic (from specification):
    - REJECT: >= 2 LC failures, OR (>= 1 catalog failure AND >= 1 LC failure)
    - WARN: 1 LC failure, OR 1 catalog failure, OR any low-confidence pass
    - PASS: All checks passed with good confidence

    Args:
        lc_checks: Results from LC-only checks (V01-V05)
        catalog_checks: Results from catalog checks (V06-V07), optional
        config: Aggregation configuration

    Returns:
        Verdict enum value (PASS, WARN, or REJECT)

    Example:
        verdict = compute_verdict(lc_checks, catalog_checks)
        if verdict == Verdict.REJECT:
            print("Candidate rejected")
    """
    cfg = config or DEFAULT_AGGREGATION_CONFIG
    catalog_checks = catalog_checks or []

    # Count failures
    lc_failures = sum(1 for c in lc_checks if not c.passed)
    cat_failures = sum(1 for c in catalog_checks if not c.passed)

    # Check for low-confidence passes
    all_checks = lc_checks + catalog_checks
    low_confidence_passes = [
        c for c in all_checks if c.passed and c.confidence < cfg.low_confidence_threshold
    ]

    # REJECT conditions
    if lc_failures >= cfg.lc_failure_threshold:
        return Verdict.REJECT

    if cat_failures >= 1 and lc_failures >= 1:
        return Verdict.REJECT

    if (lc_failures + cat_failures) >= cfg.mixed_failure_threshold:
        return Verdict.REJECT

    # WARN conditions
    if lc_failures == 1 or cat_failures == 1:
        return Verdict.WARN

    if cfg.warn_on_low_confidence and low_confidence_passes:
        return Verdict.WARN

    # PASS
    return Verdict.PASS


def compute_disposition(
    verdict: Verdict,
    checks: list[VetterCheckResult],
) -> Disposition:
    """Compute final disposition from verdict and checks.

    Args:
        verdict: Aggregated verdict
        checks: All check results

    Returns:
        Disposition (PLANET, FALSE_POSITIVE, or UNCERTAIN)
    """
    if verdict == Verdict.REJECT:
        return Disposition.FALSE_POSITIVE

    if verdict == Verdict.WARN:
        return Disposition.UNCERTAIN

    # For PASS, check confidence levels
    mean_confidence = sum(c.confidence for c in checks) / len(checks) if checks else 0.0

    if mean_confidence < 0.7:
        return Disposition.UNCERTAIN

    return Disposition.PLANET


def generate_summary(
    disposition: Disposition,
    verdict: Verdict,
    checks: list[VetterCheckResult],
) -> str:
    """Generate human-readable summary of validation results.

    Args:
        disposition: Final disposition
        verdict: Aggregated verdict
        checks: All check results

    Returns:
        Summary string describing the validation outcome
    """
    n_passed = sum(1 for c in checks if c.passed)
    n_failed = len(checks) - n_passed
    failed_names = [c.name for c in checks if not c.passed]

    if verdict == Verdict.PASS:
        return f"Passed all {len(checks)} vetting checks. Candidate is a likely planet."

    if verdict == Verdict.WARN:
        if failed_names:
            return (
                f"Passed {n_passed}/{len(checks)} checks. "
                f"Marginal on: {', '.join(failed_names)}. "
                "Requires further investigation."
            )
        # Warn due to low confidence
        return "Passed all checks but with low confidence. Requires further investigation."

    # REJECT
    return (
        f"Failed {n_failed}/{len(checks)} checks: {', '.join(failed_names)}. Likely false positive."
    )


def aggregate_results(
    lc_checks: list[VetterCheckResult],
    catalog_checks: list[VetterCheckResult] | None = None,
    config: AggregationConfig | None = None,
) -> ValidationResult:
    """Full aggregation pipeline: verdict + disposition + summary.

    This is the main entry point for combining check results into
    a complete ValidationResult.

    Args:
        lc_checks: Results from LC-only checks
        catalog_checks: Results from catalog checks, optional
        config: Aggregation configuration

    Returns:
        Complete ValidationResult with verdict, disposition, and summary

    Example:
        result = aggregate_results(lc_checks, catalog_checks)
        print(f"Verdict: {result.verdict}")
        print(f"Disposition: {result.disposition}")
        print(f"Summary: {result.summary}")
    """
    catalog_checks = catalog_checks or []
    all_checks = lc_checks + catalog_checks

    verdict = compute_verdict(lc_checks, catalog_checks, config)
    disposition = compute_disposition(verdict, all_checks)
    summary = generate_summary(disposition, verdict, all_checks)

    return ValidationResult(
        disposition=disposition,
        verdict=verdict,
        checks=all_checks,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Common Utilities for Vetter Implementations
# ---------------------------------------------------------------------------


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
    phase, _ = phase_fold(time, time, period, t0)
    half_dur = (duration_hours / 24.0 / period) * buffer_factor / 2.0
    result: np.ndarray[tuple[Any, ...], np.dtype[Any]] = np.abs(phase) < half_dur
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
    return ~get_in_transit_mask(time, period, t0, duration_hours, buffer_factor)


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
    in_flux = flux[in_transit_mask]
    out_flux = flux[out_of_transit_mask]

    if len(in_flux) == 0 or len(out_flux) == 0:
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
    if len(time) == 0:
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
    orbit_numbers = np.round((time - t0) / period).astype(int)
    is_odd = orbit_numbers % 2 == 1
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
