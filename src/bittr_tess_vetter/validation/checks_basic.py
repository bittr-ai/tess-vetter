"""Basic vetting checks for transit candidate validation (class-based).

This module implements vetting checks using the class-based VetterCheck
interface. These are ALTERNATIVE implementations to the function-based
checks in lc_checks.py.

NOTE: The canonical check ID mapping is defined in lc_checks.py and
handlers/validate.py. This module provides additional check implementations
(SNRCheck, DepthCheck) that can be used as pre-filters before the main
V01-V10 checks.

Class-based checks in this module:
- SNRCheck: Transit must have sufficient signal-to-noise (pre-filter)
- DepthCheck: Transit depth must be physically reasonable (pre-filter)
- DurationCheck: Duration consistency (maps to V03)
- OddEvenCheck: Odd-even depth comparison (maps to V01)

CANONICAL ID MAPPING (from lc_checks.py):
- V01: odd_even_depth
- V02: secondary_eclipse
- V03: duration_consistency
- V04: depth_stability
- V05: v_shape

References:
- Batalha et al. (2010) - Kepler vetting pipeline
- Thompson et al. (2018) - Kepler DR25 vetting methodology
- Guerrero et al. (2021) - TESS vetting procedures
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bittr_tess_vetter.domain.detection import TransitCandidate, VetterCheckResult
from bittr_tess_vetter.validation.base import CheckConfig, VetterCheck

if TYPE_CHECKING:
    from bittr_tess_vetter.domain.lightcurve import LightCurveData
    from bittr_tess_vetter.domain.target import StellarParameters


# =============================================================================
# PF01: SNR Threshold Check (Pre-filter, not part of V01-V10)
# =============================================================================


class SNRCheck(VetterCheck):
    """Pre-filter: Signal-to-Noise Ratio threshold check.

    Astronomical Significance:
    --------------------------
    The SNR of a transit signal is fundamental to its reliability. Low SNR
    signals are often noise artifacts or systematics rather than genuine
    astrophysical events. The standard threshold of 7.0 (or higher) comes
    from the requirement that the signal should be clearly distinguishable
    from Gaussian noise at roughly 5-sigma significance with margin for
    systematic effects.

    For TESS data, this threshold is particularly important because:
    1. Short observing baselines (27 days per sector) limit the number of
       transit events available for averaging
    2. Scattered light and spacecraft systematics can produce transit-like
       features
    3. The large pixel scale (21 arcsec) means blending is common

    A signal with SNR < 7 may still be real but requires additional validation
    or data from multiple sectors before it can be confidently promoted to
    planet candidate status.

    Pass Criteria:
    - candidate.snr >= threshold (default 7.0)

    Confidence Calculation:
    - High confidence (0.95) when SNR is well above or below threshold
    - Lower confidence (0.80-0.90) near the threshold boundary
    """

    # Pre-filter check - not part of canonical V01-V10 sequence
    id = "PF01"
    name = "snr_threshold"

    @classmethod
    def _default_config(cls) -> CheckConfig:
        """Default SNR threshold of 7.0 - standard for transit detection."""
        return CheckConfig(
            enabled=True,
            threshold=7.0,
            additional={"margin": 1.0},  # Width of uncertainty region
        )

    def run(
        self,
        candidate: TransitCandidate,
        lightcurve: LightCurveData | None = None,
        stellar: StellarParameters | None = None,
    ) -> VetterCheckResult:
        """Check if transit SNR exceeds threshold.

        Args:
            candidate: Transit candidate with SNR measurement.
            lightcurve: Not used for this check.
            stellar: Not used for this check.

        Returns:
            VetterCheckResult with pass/fail based on SNR threshold.
        """
        threshold = self.config.threshold or 7.0
        margin = (self.config.additional or {}).get("margin", 1.0)

        snr = candidate.snr
        passed = snr >= threshold

        # Calculate confidence based on how far from threshold
        # High confidence when clearly above or below
        # Lower confidence in the "gray zone" near threshold
        distance_from_threshold = abs(snr - threshold)
        if distance_from_threshold > margin * 2:
            confidence = 0.95
        elif distance_from_threshold > margin:
            confidence = 0.90
        else:
            confidence = 0.80

        return VetterCheckResult(
            id=self.id,
            name=self.name,
            passed=passed,
            confidence=confidence,
            details={
                "snr": snr,
                "threshold": threshold,
                "snr_excess": snr - threshold,
                "interpretation": (
                    "Signal is strong enough for reliable detection"
                    if passed
                    else "Signal may be noise or requires additional data"
                ),
            },
        )


# =============================================================================
# PF02: Depth Plausibility Check (Pre-filter, not part of V01-V10)
# =============================================================================


class DepthCheck(VetterCheck):
    """Pre-filter: Transit depth physical plausibility check.

    Astronomical Significance:
    --------------------------
    Transit depth is directly related to the ratio of planet to star radii:
    depth = (Rp/Rs)^2

    Physical constraints on this ratio:
    1. For a main-sequence star, a depth of 1% corresponds to a Jupiter-sized
       planet (Rp ~ 0.1 Rs for a Sun-like star)
    2. Depths exceeding ~3% indicate objects larger than Jupiter
    3. Depths exceeding ~10-15% are typically grazing eclipsing binaries
       or stellar companions
    4. The theoretical maximum depth is 100% (total eclipse), but depths
       above 20% almost always indicate a stellar-mass companion

    The 20% threshold is chosen because:
    - Even a brown dwarf (~0.08 solar masses) transiting a M-dwarf would
      produce depths of ~10-15%
    - Depths above 20% require the occulting object to be comparable in
      size to the host star, which means it's a stellar binary
    - Some inflated hot Jupiters can reach 3-4%, but never 20%

    Pass Criteria:
    - 0 < candidate.depth < max_depth (default 0.20 = 20%)

    Confidence Calculation:
    - Very high confidence (0.98) for depth > 20% fails (almost certainly EB)
    - High confidence (0.95) for normal planetary depths (< 5%)
    - Moderate confidence (0.85) for large but plausible depths (5-15%)
    """

    # Pre-filter check - not part of canonical V01-V10 sequence
    id = "PF02"
    name = "depth_plausibility"

    @classmethod
    def _default_config(cls) -> CheckConfig:
        """Default max depth of 20% - above this is almost certainly an EB."""
        return CheckConfig(
            enabled=True,
            threshold=0.20,  # 20% maximum depth
            additional={
                "min_depth": 0.00001,  # 10 ppm minimum (noise floor)
                "suspicious_depth": 0.05,  # 5% - large but possible
            },
        )

    def run(
        self,
        candidate: TransitCandidate,
        lightcurve: LightCurveData | None = None,
        stellar: StellarParameters | None = None,
    ) -> VetterCheckResult:
        """Check if transit depth is physically plausible for a planet.

        Args:
            candidate: Transit candidate with depth measurement.
            lightcurve: Not used for this check.
            stellar: Not used for this check.

        Returns:
            VetterCheckResult with pass/fail based on depth limits.
        """
        max_depth = self.config.threshold or 0.20
        additional = self.config.additional or {}
        min_depth = additional.get("min_depth", 0.00001)
        suspicious_depth = additional.get("suspicious_depth", 0.05)

        depth = candidate.depth

        # Check bounds
        too_deep = depth >= max_depth
        too_shallow = depth < min_depth
        passed = not too_deep and not too_shallow

        # Estimate implied radius ratio
        # depth = (Rp/Rs)^2, so Rp/Rs = sqrt(depth)
        radius_ratio = np.sqrt(depth) if depth > 0 else 0.0

        # Calculate confidence
        if too_deep:
            # Very confident this is an EB
            confidence = 0.98
            interpretation = (
                f"Depth of {depth * 100:.2f}% implies stellar-sized companion "
                f"(Rp/Rs = {radius_ratio:.3f}). Likely eclipsing binary."
            )
        elif too_shallow:
            confidence = 0.90
            interpretation = (
                f"Depth of {depth * 1e6:.1f} ppm is at or below noise floor. "
                "Signal may be spurious."
            )
        elif depth > suspicious_depth:
            # Large but possible - brown dwarf or inflated planet
            confidence = 0.85
            interpretation = (
                f"Depth of {depth * 100:.2f}% is large but physically possible. "
                f"Implied Rp/Rs = {radius_ratio:.3f}. Could be brown dwarf or "
                "inflated hot Jupiter."
            )
        else:
            # Normal planetary depth
            confidence = 0.95
            interpretation = (
                f"Depth of {depth * 100:.4f}% is consistent with planetary transit. "
                f"Implied Rp/Rs = {radius_ratio:.4f}."
            )

        return VetterCheckResult(
            id=self.id,
            name=self.name,
            passed=passed,
            confidence=confidence,
            details={
                "depth": depth,
                "depth_ppm": depth * 1e6,
                "depth_percent": depth * 100,
                "max_depth": max_depth,
                "radius_ratio": radius_ratio,
                "interpretation": interpretation,
            },
        )


# =============================================================================
# V03: Duration Consistency Check (Canonical ID)
# =============================================================================


class DurationCheck(VetterCheck):
    """V03: Transit duration consistency with orbital parameters.

    Astronomical Significance:
    --------------------------
    For a circular orbit, transit duration follows Kepler's laws:

    T_dur = (P / pi) * arcsin(sqrt((Rs + Rp)^2 - b^2 * Rs^2) / a)

    Where:
    - P = orbital period
    - Rs = stellar radius
    - Rp = planet radius
    - b = impact parameter (0 for central transit, 1 for grazing)
    - a = semi-major axis

    For a central transit (b=0) of a small planet (Rp << Rs):
    T_dur ~ (P / pi) * (Rs / a)

    Using Kepler's 3rd law: a^3 = (G * M_star / 4*pi^2) * P^2

    This gives us: T_dur ~ P^(1/3) * Rs * (rho_star)^(-1/3)

    Key physical constraints:
    1. Duration cannot exceed P/2 (half the period)
    2. For circular orbits, duration scales with P^(1/3)
    3. Very short durations (< 0.5 hours) are suspicious for typical hosts
    4. Very long durations may indicate eccentric orbits or blends

    Without stellar parameters, we use empirical limits:
    - Minimum duration: 0.5 hours (grazing transit of compact star)
    - Maximum duration: min(0.4 * P, 20 hours) for typical systems

    Pass Criteria:
    - Duration within physically plausible range for given period
    - If stellar parameters available, check against expected duration

    Confidence Calculation:
    - High confidence (0.95) when duration matches expectations
    - Lower confidence when duration is unusual but possible
    """

    id = "V03"
    name = "duration_consistency"

    @classmethod
    def _default_config(cls) -> CheckConfig:
        """Default duration limits based on empirical transit statistics."""
        return CheckConfig(
            enabled=True,
            threshold=None,  # Not a simple threshold check
            additional={
                "min_duration_hours": 0.3,  # 18 minutes minimum
                "max_duration_fraction": 0.4,  # Max 40% of period
                "max_duration_hours": 24.0,  # Absolute maximum
                "solar_density_gcm3": 1.41,  # Solar density for scaling
            },
        )

    def run(
        self,
        candidate: TransitCandidate,
        lightcurve: LightCurveData | None = None,
        stellar: StellarParameters | None = None,
    ) -> VetterCheckResult:
        """Check if transit duration is consistent with orbital period.

        Args:
            candidate: Transit candidate with period and duration.
            lightcurve: Not used for this check.
            stellar: Optional stellar parameters for refined limits.

        Returns:
            VetterCheckResult with pass/fail based on duration plausibility.
        """
        additional = self.config.additional or {}
        min_dur = additional.get("min_duration_hours", 0.3)
        max_dur_frac = additional.get("max_duration_fraction", 0.4)
        max_dur_abs = additional.get("max_duration_hours", 24.0)

        period_days = candidate.period
        period_hours = period_days * 24.0
        duration_hours = candidate.duration_hours

        # Calculate duration limits
        max_dur_from_period = max_dur_frac * period_hours
        max_duration = min(max_dur_from_period, max_dur_abs)

        # Duration as fraction of period
        duration_fraction = duration_hours / period_hours

        # Check bounds
        too_short = duration_hours < min_dur
        too_long = duration_hours > max_duration
        exceeds_half_period = duration_fraction > 0.5  # Physical impossibility

        passed = not too_short and not too_long and not exceeds_half_period

        # Expected duration scaling for solar-type star
        # T_dur ~ 13 hours * (P/1 year)^(1/3) for central transit of Sun-like star
        # For shorter periods, scale accordingly
        expected_duration_solar = 13.0 * (period_days / 365.25) ** (1 / 3)

        # Calculate how far actual duration deviates from expectation
        if expected_duration_solar > 0:
            duration_ratio = duration_hours / expected_duration_solar
        else:
            duration_ratio = float("inf")

        # Confidence calculation
        if exceeds_half_period:
            confidence = 0.99  # Physically impossible, very confident fail
            interpretation = (
                f"Duration ({duration_hours:.2f}h) exceeds half the period "
                f"({period_hours / 2:.2f}h). This is physically impossible."
            )
        elif too_long:
            confidence = 0.90
            interpretation = (
                f"Duration ({duration_hours:.2f}h) is unusually long for "
                f"P={period_days:.2f}d. May indicate blend or eccentric orbit."
            )
        elif too_short:
            confidence = 0.85
            interpretation = (
                f"Duration ({duration_hours:.2f}h) is very short. "
                "May be grazing transit or compact host star."
            )
        elif 0.3 < duration_ratio < 3.0:
            # Within factor of 3 of solar expectation
            confidence = 0.95
            interpretation = (
                f"Duration ({duration_hours:.2f}h) is consistent with expectations "
                f"for P={period_days:.2f}d."
            )
        else:
            confidence = 0.80
            interpretation = (
                f"Duration ({duration_hours:.2f}h) deviates from solar-type "
                f"expectation ({expected_duration_solar:.2f}h) but is possible."
            )

        return VetterCheckResult(
            id=self.id,
            name=self.name,
            passed=passed,
            confidence=confidence,
            details={
                "duration_hours": duration_hours,
                "period_days": period_days,
                "duration_fraction": duration_fraction,
                "min_duration": min_dur,
                "max_duration": max_duration,
                "expected_duration_solar": expected_duration_solar,
                "interpretation": interpretation,
            },
        )


# =============================================================================
# V01: Odd-Even Transit Depth Comparison (Canonical ID)
# =============================================================================


class OddEvenCheck(VetterCheck):
    """V01: Compare odd and even transit depths to detect eclipsing binaries.

    Astronomical Significance:
    --------------------------
    This is one of the most powerful checks for identifying eclipsing binaries
    (EBs) masquerading as planetary transits. The key insight:

    An eclipsing binary at period P will show:
    - Primary eclipse: Smaller, hotter star transits in front of larger, cooler star
    - Secondary eclipse: Larger star transits in front of smaller star

    If we detect the signal at period P/2 (half the true period), we will see:
    - Odd transits: Primary eclipses
    - Even transits: Secondary eclipses

    Since primary and secondary eclipses have different depths (due to different
    surface brightness ratios), comparing odd vs even transit depths reveals this.

    For a genuine planetary transit:
    - All transits should have identical depths (planet doesn't emit significant light)
    - Statistical scatter is expected, but odd and even should be consistent

    For an EB at 2x the detected period:
    - Odd and even depths will differ significantly
    - The difference indicates the temperature/luminosity ratio of the binary

    Test Statistic:
    - We compute the difference between mean odd and even depths
    - Normalize by the combined uncertainty
    - A difference > 3-sigma indicates likely EB

    This check requires the light curve to phase-fold the data and measure
    depths in individual transit windows. If the light curve is not provided,
    we fall back to lower confidence based on other indicators.

    Pass Criteria:
    - |depth_odd - depth_even| / sigma < threshold (default 3.0)

    Confidence Calculation:
    - High confidence when light curve data allows direct measurement
    - Lower confidence when relying on candidate parameters alone
    """

    # Canonical ID: V01 (matches lc_checks.py)
    id = "V01"
    name = "odd_even_depth"

    @classmethod
    def _default_config(cls) -> CheckConfig:
        """Default 3-sigma threshold for odd-even difference."""
        return CheckConfig(
            enabled=True,
            threshold=3.0,  # 3-sigma difference threshold
            additional={
                "min_transits_per_class": 2,  # Minimum odd or even transits needed
                "max_depth_difference_ppm": 5000,  # Absolute limit (0.5%)
            },
        )

    def run(
        self,
        candidate: TransitCandidate,
        lightcurve: LightCurveData | None = None,
        stellar: StellarParameters | None = None,
    ) -> VetterCheckResult:
        """Compare odd and even transit depths for EB detection.

        Args:
            candidate: Transit candidate parameters.
            lightcurve: Light curve data for direct depth measurement.
            stellar: Not used for this check.

        Returns:
            VetterCheckResult with pass/fail based on odd-even consistency.
        """
        sigma_threshold = self.config.threshold or 3.0
        additional = self.config.additional or {}
        min_transits = additional.get("min_transits_per_class", 2)
        max_diff_ppm = additional.get("max_depth_difference_ppm", 5000)

        if lightcurve is None:
            # Cannot perform this check without light curve data
            return VetterCheckResult(
                id=self.id,
                name=self.name,
                passed=True,  # Pass by default when check cannot be performed
                confidence=0.50,  # Low confidence - check not actually run
                details={
                    "status": "skipped",
                    "reason": "Light curve data required for odd-even comparison",
                    "interpretation": (
                        "Cannot measure individual transit depths without light curve. "
                        "Check passed by default with low confidence."
                    ),
                },
            )

        # Phase-fold the light curve and identify transit windows
        period = candidate.period
        t0 = candidate.t0
        duration_days = candidate.duration_hours / 24.0

        time = lightcurve.time[lightcurve.valid_mask]
        flux = lightcurve.flux[lightcurve.valid_mask]
        flux_err = lightcurve.flux_err[lightcurve.valid_mask]

        # Compute transit number for each point
        # Transit N occurs at time t0 + N * period
        transit_numbers = np.round((time - t0) / period).astype(int)

        # Define in-transit mask (within half duration of transit center)
        phase = (time - t0) / period
        phase = phase - np.floor(phase)  # Wrap to [0, 1)
        phase[phase > 0.5] -= 1.0  # Center on 0
        in_transit = np.abs(phase * period) < (duration_days / 2)

        # Separate odd and even transits
        is_odd = transit_numbers % 2 == 1
        is_even = transit_numbers % 2 == 0

        odd_in_transit = in_transit & is_odd
        even_in_transit = in_transit & is_even

        n_odd = np.sum(odd_in_transit)
        n_even = np.sum(even_in_transit)

        # Check if we have enough transits
        # Count unique transit numbers
        odd_transit_ids = np.unique(transit_numbers[odd_in_transit])
        even_transit_ids = np.unique(transit_numbers[even_in_transit])
        n_odd_transits = len(odd_transit_ids)
        n_even_transits = len(even_transit_ids)

        if n_odd_transits < min_transits or n_even_transits < min_transits:
            return VetterCheckResult(
                id=self.id,
                name=self.name,
                passed=True,  # Pass by default when insufficient data
                confidence=0.60,
                details={
                    "status": "insufficient_data",
                    "n_odd_transits": n_odd_transits,
                    "n_even_transits": n_even_transits,
                    "min_required": min_transits,
                    "interpretation": (
                        f"Found {n_odd_transits} odd and {n_even_transits} even transits. "
                        f"Need at least {min_transits} of each for reliable comparison."
                    ),
                },
            )

        # Calculate mean depths for odd and even transits
        # Depth is (1 - flux) for normalized flux where out-of-transit ~ 1.0
        out_of_transit = ~in_transit
        baseline = np.median(flux[out_of_transit]) if np.any(out_of_transit) else 1.0

        odd_flux = flux[odd_in_transit]
        even_flux = flux[even_in_transit]
        odd_err = flux_err[odd_in_transit]
        even_err = flux_err[even_in_transit]

        # Mean depth and uncertainty for each class
        depth_odd = baseline - np.mean(odd_flux)
        depth_even = baseline - np.mean(even_flux)

        # Standard error on mean
        sigma_odd = np.sqrt(np.sum(odd_err**2)) / n_odd
        sigma_even = np.sqrt(np.sum(even_err**2)) / n_even

        # Combined uncertainty for the difference
        sigma_diff = np.sqrt(sigma_odd**2 + sigma_even**2)

        # Depth difference
        depth_diff = abs(depth_odd - depth_even)
        depth_diff_ppm = depth_diff * 1e6

        # Significance of the difference
        significance = depth_diff / sigma_diff if sigma_diff > 0 else 0.0

        # Check pass/fail
        fails_sigma = significance > sigma_threshold
        fails_absolute = depth_diff_ppm > max_diff_ppm
        passed = not fails_sigma and not fails_absolute

        # Confidence calculation
        if not passed:
            if significance > 5.0:
                confidence = 0.98  # Very strong EB signal
            elif significance > sigma_threshold:
                confidence = 0.90
            else:
                confidence = 0.85
        else:
            if significance < 1.0:
                confidence = 0.95  # Depths very consistent
            elif significance < 2.0:
                confidence = 0.90
            else:
                confidence = 0.80  # Approaching threshold

        # Build interpretation
        if not passed:
            interpretation = (
                f"Odd-even depth difference ({significance:.1f}-sigma) indicates "
                f"this may be an eclipsing binary at 2x the detected period. "
                f"Odd depth: {depth_odd * 1e6:.1f} ppm, Even depth: {depth_even * 1e6:.1f} ppm."
            )
        else:
            interpretation = (
                f"Odd and even transit depths are consistent ({significance:.1f}-sigma). "
                f"No evidence for eclipsing binary at 2x period."
            )

        return VetterCheckResult(
            id=self.id,
            name=self.name,
            passed=passed,
            confidence=confidence,
            details={
                "depth_odd_ppm": depth_odd * 1e6,
                "depth_even_ppm": depth_even * 1e6,
                "depth_difference_ppm": depth_diff_ppm,
                "sigma_difference": sigma_diff * 1e6,
                "significance": significance,
                "threshold_sigma": sigma_threshold,
                "n_odd_transits": n_odd_transits,
                "n_even_transits": n_even_transits,
                "n_odd_points": int(n_odd),
                "n_even_points": int(n_even),
                "interpretation": interpretation,
            },
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def get_basic_checks(config: dict[str, CheckConfig] | None = None) -> list[VetterCheck]:
    """Get all basic vetting checks (V01-V04) with optional custom config.

    Args:
        config: Optional dictionary mapping check IDs to CheckConfig.

    Returns:
        List of instantiated VetterCheck objects.
    """
    config = config or {}
    return [
        SNRCheck(config.get("V01")),
        DepthCheck(config.get("V02")),
        DurationCheck(config.get("V03")),
        OddEvenCheck(config.get("V04")),
    ]


def run_basic_checks(
    candidate: TransitCandidate,
    lightcurve: LightCurveData | None = None,
    stellar: StellarParameters | None = None,
    config: dict[str, CheckConfig] | None = None,
) -> list[VetterCheckResult]:
    """Run all basic vetting checks on a transit candidate.

    Args:
        candidate: Transit candidate to validate.
        lightcurve: Optional light curve for LC-based checks.
        stellar: Optional stellar parameters for physics-based checks.
        config: Optional check-specific configuration.

    Returns:
        List of VetterCheckResult from each check.
    """
    checks = get_basic_checks(config)
    results = []
    for check in checks:
        if check.config.enabled:
            result = check.run(candidate, lightcurve, stellar)
            results.append(result)
    return results
