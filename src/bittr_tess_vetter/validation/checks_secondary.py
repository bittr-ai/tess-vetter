"""Secondary vetting checks for transit candidate validation.

This module provides vetter checks V05-V07:
- V05: Secondary Eclipse Check - Looks for secondary eclipse at phase 0.5 (EB indicator)
- V06: Centroid Check - Verifies transit occurs on target star (not contamination)
- V07: Contamination Check - Accounts for flux dilution from nearby sources

Each check inherits from VetterCheck and returns VetterCheckResult.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from bittr_tess_vetter.domain.detection import TransitCandidate, VetterCheckResult

if TYPE_CHECKING:
    from bittr_tess_vetter.domain.lightcurve import LightCurveData
    from bittr_tess_vetter.domain.target import Target


class VetterCheck(ABC):
    """Abstract base class for transit candidate vetting checks.

    Each vetter check evaluates a specific aspect of a transit candidate
    to help distinguish genuine planets from false positives.

    Subclasses must implement the run() method which returns a VetterCheckResult.
    """

    @property
    @abstractmethod
    def check_id(self) -> str:
        """Return the check ID (e.g., 'V05')."""
        ...

    @property
    @abstractmethod
    def check_name(self) -> str:
        """Return the human-readable check name."""
        ...

    @abstractmethod
    def run(
        self,
        candidate: TransitCandidate,
        light_curve: LightCurveData,
        target: Target | None = None,
    ) -> VetterCheckResult:
        """Execute the vetting check.

        Args:
            candidate: Transit candidate parameters to evaluate.
            light_curve: Light curve data for analysis.
            target: Optional target information with stellar parameters.

        Returns:
            VetterCheckResult with pass/fail status, confidence, and details.
        """
        ...


class SecondaryEclipseCheck(VetterCheck):
    """V05: Check for secondary eclipse at phase 0.5.

    Eclipsing binaries (EBs) typically show a secondary eclipse when the
    smaller (or cooler) star passes behind the primary. This occurs at
    orbital phase 0.5 for circular orbits.

    Detection of significant flux decrease at phase 0.5 suggests the
    signal is from an EB rather than a transiting planet.

    Thresholds:
        - Secondary depth > 20% of primary depth: FAIL (likely EB)
        - Secondary depth > 10% of primary depth: WARN (possible EB)
        - Secondary depth < 10% of primary depth: PASS
    """

    SECONDARY_FAIL_RATIO = 0.20  # Fail if secondary > 20% of primary
    SECONDARY_WARN_RATIO = 0.10  # Warn if secondary > 10% of primary
    # Widened from 0.05 to 0.10 to catch eccentric orbit EBs
    # where secondary can occur at phase 0.4-0.6 rather than exactly 0.5
    PHASE_HALF_WIDTH = 0.10  # Search window around phase 0.5

    @property
    def check_id(self) -> str:
        return "V05"

    @property
    def check_name(self) -> str:
        return "Secondary Eclipse"

    def run(
        self,
        candidate: TransitCandidate,
        light_curve: LightCurveData,
        target: Target | None = None,
    ) -> VetterCheckResult:
        """Check for secondary eclipse at phase 0.5.

        Args:
            candidate: Transit candidate with period and t0.
            light_curve: Light curve data to search for secondary.
            target: Not used for this check.

        Returns:
            VetterCheckResult indicating whether secondary eclipse was detected.
        """
        # Extract valid data
        mask = light_curve.valid_mask
        time = light_curve.time[mask]
        flux = light_curve.flux[mask]

        if len(time) < 10:
            # Insufficient data
            return VetterCheckResult(
                id=self.check_id,
                name=self.check_name,
                passed=True,
                confidence=0.3,
                details={
                    "reason": "insufficient_data",
                    "n_valid_points": len(time),
                },
            )

        # Compute orbital phase (0 = primary transit, 0.5 = secondary)
        phase = ((time - candidate.t0) / candidate.period) % 1.0

        # Find points near phase 0.5 (secondary eclipse location)
        secondary_mask = np.abs(phase - 0.5) < self.PHASE_HALF_WIDTH

        # Find points away from both transits (out-of-transit baseline)
        primary_mask = phase < (candidate.duration_days / candidate.period / 2)
        primary_mask |= phase > (1.0 - candidate.duration_days / candidate.period / 2)
        oot_mask = ~secondary_mask & ~primary_mask

        if np.sum(secondary_mask) < 5 or np.sum(oot_mask) < 10:
            # Not enough points to measure secondary
            return VetterCheckResult(
                id=self.check_id,
                name=self.check_name,
                passed=True,
                confidence=0.4,
                details={
                    "reason": "insufficient_phase_coverage",
                    "n_secondary_points": int(np.sum(secondary_mask)),
                    "n_oot_points": int(np.sum(oot_mask)),
                },
            )

        # Measure secondary eclipse depth
        baseline_flux = np.median(flux[oot_mask])
        secondary_flux = np.median(flux[secondary_mask])
        secondary_depth = baseline_flux - secondary_flux

        # Compute uncertainty from scatter
        baseline_std = np.std(flux[oot_mask])
        secondary_snr = secondary_depth / baseline_std if baseline_std > 0 else 0.0

        # Compare to primary transit depth
        primary_depth = candidate.depth
        depth_ratio = secondary_depth / primary_depth if primary_depth > 0 else 0.0

        # Determine pass/fail
        if depth_ratio > self.SECONDARY_FAIL_RATIO and secondary_snr > 3.0:
            # Strong secondary eclipse detected - likely EB
            passed = False
            confidence = min(0.95, 0.7 + 0.1 * secondary_snr)
            reason = "secondary_eclipse_detected"
        elif depth_ratio > self.SECONDARY_WARN_RATIO and secondary_snr > 2.0:
            # Marginal secondary - could be EB or planet with thermal emission
            passed = True
            confidence = 0.6
            reason = "marginal_secondary"
        else:
            # No significant secondary - consistent with planet
            passed = True
            confidence = min(0.9, 0.7 + 0.05 * len(time) / 100)
            reason = "no_secondary_detected"

        return VetterCheckResult(
            id=self.check_id,
            name=self.check_name,
            passed=passed,
            confidence=round(confidence, 3),
            details={
                "reason": reason,
                "secondary_depth": round(float(secondary_depth), 8),
                "primary_depth": round(float(primary_depth), 8),
                "depth_ratio": round(float(depth_ratio), 4),
                "secondary_snr": round(float(secondary_snr), 2),
                "n_secondary_points": int(np.sum(secondary_mask)),
                "n_oot_points": int(np.sum(oot_mask)),
            },
        )


class CentroidCheck(VetterCheck):
    """V06: Verify transit occurs on the target star.

    Centroid motion during transit can indicate that the eclipse is
    actually occurring on a nearby star that is blended with the target.
    This is a common source of false positives in TESS due to large pixels.

    Note: Full centroid analysis requires difference imaging or PRF fitting.
    This implementation is a placeholder that returns passed=True with
    low confidence when centroid data is unavailable.
    """

    # Centroid shift threshold in arcseconds
    CENTROID_SHIFT_THRESHOLD = 2.0  # arcsec

    @property
    def check_id(self) -> str:
        return "V06"

    @property
    def check_name(self) -> str:
        return "Centroid"

    def run(
        self,
        candidate: TransitCandidate,
        light_curve: LightCurveData,
        target: Target | None = None,
    ) -> VetterCheckResult:
        """Check for centroid motion during transit.

        Args:
            candidate: Transit candidate parameters.
            light_curve: Light curve data (would need centroid columns).
            target: Target information with position.

        Returns:
            VetterCheckResult. Returns passed=True with low confidence
            if centroid data is unavailable.
        """
        # Check if we have position information
        has_position = target is not None and target.has_position()

        # Placeholder: TESS SPOC light curves don't include centroid columns
        # Full implementation would require:
        # 1. MOM_CENTR1, MOM_CENTR2 columns from TPF
        # 2. Comparison of in-transit vs out-of-transit centroid positions
        # 3. Statistical test for significant centroid shift

        # For now, return passed with low confidence (data unavailable)
        return VetterCheckResult(
            id=self.check_id,
            name=self.check_name,
            passed=True,
            confidence=0.3,
            details={
                "reason": "centroid_data_unavailable",
                "has_target_position": has_position,
                "note": "Full centroid analysis requires TPF/PRF fitting",
                "placeholder": True,
            },
        )


class ContaminationCheck(VetterCheck):
    """V07: Account for flux dilution from nearby sources.

    Contamination (also called blending or dilution) occurs when flux from
    nearby stars falls within the photometric aperture. This dilutes the
    observed transit depth, making it appear shallower than the true depth.

    If contamination is high, the true transit depth could be much deeper,
    potentially indicating an eclipsing binary rather than a planet.

    Thresholds:
        - Contamination > 50%: FAIL (true depth could be 2x+ observed)
        - Contamination > 20%: WARN (significant dilution)
        - Contamination < 20%: PASS

    Note on MAX_CORRECTED_DEPTH:
        The threshold for "too deep" corrected depth depends on stellar type.
        Hot Jupiters around M-dwarfs can have depths up to 10-12% due to
        the small stellar radius. The default 5% is conservative for Sun-like
        stars but may cause false rejections for M-dwarf targets.
    """

    CONTAMINATION_FAIL_THRESHOLD = 0.50  # Fail if > 50% flux from neighbors
    CONTAMINATION_WARN_THRESHOLD = 0.20  # Warn if > 20% flux from neighbors

    # Default max corrected depth (5%) - can be overridden for M-dwarfs
    # M-dwarf hot Jupiters can reach 10-12% depth
    MAX_CORRECTED_DEPTH_DEFAULT = 0.05
    MAX_CORRECTED_DEPTH_MDWARF = 0.15  # Allow deeper for M-dwarfs (Teff < 4000K)

    @property
    def check_id(self) -> str:
        return "V07"

    @property
    def check_name(self) -> str:
        return "Contamination"

    def run(
        self,
        candidate: TransitCandidate,
        light_curve: LightCurveData,
        target: Target | None = None,
    ) -> VetterCheckResult:
        """Check contamination level and compute corrected transit depth.

        Args:
            candidate: Transit candidate with observed depth.
            light_curve: Light curve data.
            target: Target with StellarParameters.contamination value.

        Returns:
            VetterCheckResult with contamination assessment.
        """
        # Check if we have contamination data
        if target is None or target.stellar.contamination is None:
            return VetterCheckResult(
                id=self.check_id,
                name=self.check_name,
                passed=True,
                confidence=0.4,
                details={
                    "reason": "contamination_data_unavailable",
                    "note": "TIC contamination ratio not available",
                },
            )

        contamination = target.stellar.contamination
        observed_depth = candidate.depth

        # Compute corrected depth accounting for dilution
        # observed_depth = true_depth * (1 - contamination)
        # true_depth = observed_depth / (1 - contamination)
        if contamination >= 1.0:
            # All flux from neighbors - cannot determine true depth
            return VetterCheckResult(
                id=self.check_id,
                name=self.check_name,
                passed=False,
                confidence=0.9,
                details={
                    "reason": "fully_contaminated",
                    "contamination_ratio": round(float(contamination), 4),
                    "note": "Target flux completely dominated by neighbors",
                },
            )

        corrected_depth = observed_depth / (1.0 - contamination)

        # Determine stellar-type dependent max depth threshold
        # M-dwarfs (Teff < 4000K) can have deeper planetary transits
        is_mdwarf = False
        if target.stellar.teff is not None and target.stellar.teff < 4000:
            max_corrected_depth = self.MAX_CORRECTED_DEPTH_MDWARF
            is_mdwarf = True
        else:
            max_corrected_depth = self.MAX_CORRECTED_DEPTH_DEFAULT

        # Determine pass/fail
        if contamination > self.CONTAMINATION_FAIL_THRESHOLD:
            # High contamination - true depth could be very deep
            passed = False
            confidence = 0.85
            reason = "high_contamination"
        elif corrected_depth > max_corrected_depth:
            # Corrected depth too deep for planet (stellar-type adjusted)
            passed = False
            confidence = 0.80
            reason = "corrected_depth_too_deep"
        elif contamination > self.CONTAMINATION_WARN_THRESHOLD:
            # Moderate contamination - flag but pass
            passed = True
            confidence = 0.65
            reason = "moderate_contamination"
        else:
            # Low contamination - minimal impact on transit depth
            passed = True
            confidence = 0.90
            reason = "low_contamination"

        # Compute dilution factor (how much deeper is true depth)
        dilution_factor = 1.0 / (1.0 - contamination)

        return VetterCheckResult(
            id=self.check_id,
            name=self.check_name,
            passed=passed,
            confidence=round(confidence, 3),
            details={
                "reason": reason,
                "contamination_ratio": round(float(contamination), 4),
                "observed_depth": round(float(observed_depth), 8),
                "corrected_depth": round(float(corrected_depth), 8),
                "dilution_factor": round(float(dilution_factor), 4),
                "max_corrected_depth_threshold": round(max_corrected_depth, 4),
                "is_mdwarf_target": is_mdwarf,
            },
        )


# Registry of secondary checks for easy access
SECONDARY_CHECKS: dict[str, type[VetterCheck]] = {
    "V05": SecondaryEclipseCheck,
    "V06": CentroidCheck,
    "V07": ContaminationCheck,
}


def run_secondary_checks(
    candidate: TransitCandidate,
    light_curve: LightCurveData,
    target: Target | None = None,
) -> list[VetterCheckResult]:
    """Run all secondary vetting checks on a candidate.

    Args:
        candidate: Transit candidate to evaluate.
        light_curve: Light curve data for analysis.
        target: Optional target information.

    Returns:
        List of VetterCheckResult from V05-V07 checks.
    """
    results = []
    for check_class in SECONDARY_CHECKS.values():
        check = check_class()
        result = check.run(candidate, light_curve, target)
        results.append(result)
    return results
