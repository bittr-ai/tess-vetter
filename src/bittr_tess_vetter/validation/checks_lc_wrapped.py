"""Light curve-only vetting check wrappers (V01-V05).

This module provides VettingCheck-compliant wrappers around the existing
LC-only check implementations. These wrappers:
- Implement the VettingCheck protocol
- Convert VetterCheckResult to the new CheckResult schema
- Preserve all existing functionality while normalizing the interface

Novelty: standard (wrapper pattern)
"""

from __future__ import annotations

from typing import Any

from bittr_tess_vetter.validation.registry import (
    CheckConfig,
    CheckInputs,
    CheckRequirements,
    CheckTier,
)
from bittr_tess_vetter.validation.result_schema import (
    CheckResult,
    error_result,
    ok_result,
)


def _convert_legacy_result(
    legacy: Any,
    check_id: str,
    check_name: str,
) -> CheckResult:
    """Convert a legacy VetterCheckResult to the new CheckResult schema.

    Args:
        legacy: VetterCheckResult from existing implementations.
        check_id: Check identifier.
        check_name: Human-readable check name.

    Returns:
        New-schema CheckResult.
    """
    details = dict(legacy.details) if legacy.details else {}

    # Extract warnings from details
    warnings = details.pop("warnings", [])
    notes = list(warnings) if isinstance(warnings, list) else []

    # Extract metrics - only keep JSON-serializable scalar values
    metrics: dict[str, float | int | str | bool | None] = {}
    for key, value in details.items():
        # Skip internal/meta keys
        if key.startswith("_"):
            continue
        # Only include scalar types that are JSON-serializable
        if isinstance(value, (float, int, str, bool)) or value is None:
            metrics[key] = value
        elif isinstance(value, (list, dict)):
            # Skip complex types for metrics, they go in raw
            pass

    # Determine flags from warnings and metrics
    flags: list[str] = []
    if "insufficient_data" in str(warnings).lower():
        flags.append("INSUFFICIENT_DATA")

    return ok_result(
        check_id,
        check_name,
        metrics=metrics,
        confidence=legacy.confidence,
        flags=flags,
        notes=notes,
        raw=details,
    )


class OddEvenDepthCheck:
    """V01: Odd-even transit depth comparison.

    Detects eclipsing binaries masquerading as planets at 2x the true period.
    If odd and even depths differ significantly, likely an EB.

    References:
        - Thompson et al. 2018, ApJS 235, 38 (Kepler Robovetter odd/even test)
        - Pont et al. 2006, MNRAS 373, 231 (correlated noise in transit photometry)
    """

    id = "V01"
    name = "Odd-Even Depth"
    tier = CheckTier.LC_ONLY
    requirements = CheckRequirements()
    citations = [
        "Thompson et al. 2018, ApJS 235, 38",
        "Pont et al. 2006, MNRAS 373, 231",
    ]

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        """Execute the odd-even depth check."""
        try:
            from bittr_tess_vetter.validation.lc_checks import check_odd_even_depth

            result = check_odd_even_depth(
                inputs.lc,
                inputs.candidate.period,
                inputs.candidate.t0,
                inputs.candidate.duration_hours,
            )

            return _convert_legacy_result(result, self.id, self.name)
        except Exception as e:
            return error_result(
                self.id,
                self.name,
                error=type(e).__name__,
                notes=[str(e)],
            )


class SecondaryEclipseCheck:
    """V02: Secondary eclipse detection.

    Searches for secondary eclipse at phase 0.5. Presence of secondary eclipse
    indicates hot planet (thermal emission) or eclipsing binary.

    References:
        - Coughlin & Lopez-Morales 2012, AJ 143, 39
        - Thompson et al. 2018, ApJS 235, 38 (Robovetter significant secondary test)
    """

    id = "V02"
    name = "Secondary Eclipse"
    tier = CheckTier.LC_ONLY
    requirements = CheckRequirements()
    citations = [
        "Coughlin & Lopez-Morales 2012, AJ 143, 39",
        "Thompson et al. 2018, ApJS 235, 38",
    ]

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        """Execute the secondary eclipse check."""
        try:
            from bittr_tess_vetter.validation.lc_checks import check_secondary_eclipse

            result = check_secondary_eclipse(
                inputs.lc,
                inputs.candidate.period,
                inputs.candidate.t0,
            )

            return _convert_legacy_result(result, self.id, self.name)
        except Exception as e:
            return error_result(
                self.id,
                self.name,
                error=type(e).__name__,
                notes=[str(e)],
            )


class DurationConsistencyCheck:
    """V03: Transit duration vs stellar density consistency.

    Transit duration depends on stellar density. Large mismatches between the
    observed duration and expectation can indicate host/parameter mismatch.

    References:
        - Seager & Mallen-Ornelas 2003, ApJ 585, 1038
    """

    id = "V03"
    name = "Duration Consistency"
    tier = CheckTier.LC_ONLY
    requirements = CheckRequirements(needs_stellar=True)
    citations = ["Seager & Mallen-Ornelas 2003, ApJ 585, 1038"]

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        """Execute the duration consistency check."""
        try:
            from bittr_tess_vetter.validation.lc_checks import check_duration_consistency

            result = check_duration_consistency(
                inputs.candidate.period,
                inputs.candidate.duration_hours,
                inputs.stellar,
            )

            return _convert_legacy_result(result, self.id, self.name)
        except Exception as e:
            return error_result(
                self.id,
                self.name,
                error=type(e).__name__,
                notes=[str(e)],
            )


class DepthStabilityCheck:
    """V04: Depth consistency across individual transits.

    Variable depth suggests blended eclipsing binary or systematic issues.
    Real planets have consistent depths.

    References:
        - Thompson et al. 2018, ApJS 235, 38 (depth consistency tests)
        - Wang & Espinoza 2023, arXiv:2311.02154 (per-transit depth fitting)
    """

    id = "V04"
    name = "Depth Stability"
    tier = CheckTier.LC_ONLY
    requirements = CheckRequirements()
    citations = [
        "Thompson et al. 2018, ApJS 235, 38",
        "Wang & Espinoza 2023, arXiv:2311.02154",
    ]

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        """Execute the depth stability check."""
        try:
            from bittr_tess_vetter.validation.lc_checks import check_depth_stability

            result = check_depth_stability(
                inputs.lc,
                inputs.candidate.period,
                inputs.candidate.t0,
                inputs.candidate.duration_hours,
            )

            return _convert_legacy_result(result, self.id, self.name)
        except Exception as e:
            return error_result(
                self.id,
                self.name,
                error=type(e).__name__,
                notes=[str(e)],
            )


class VShapeCheck:
    """V05: Transit shape analysis (U-shape vs V-shape).

    Uses trapezoid model fitting to extract tF/tT ratio. V-shaped transits
    (tF/tT ~ 0) suggest grazing eclipsing binaries.

    References:
        - Seager & Mallen-Ornelas 2003, ApJ 585, 1038
        - Thompson et al. 2018, ApJS 235, 38 (Not Transit-Like metric)
        - Prsa et al. 2011, AJ 141, 83 (EB morphology)
    """

    id = "V05"
    name = "V-Shape"
    tier = CheckTier.LC_ONLY
    requirements = CheckRequirements()
    citations = [
        "Seager & Mallen-Ornelas 2003, ApJ 585, 1038",
        "Thompson et al. 2018, ApJS 235, 38",
        "Prsa et al. 2011, AJ 141, 83",
    ]

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        """Execute the V-shape check."""
        try:
            from bittr_tess_vetter.validation.lc_checks import check_v_shape

            result = check_v_shape(
                inputs.lc,
                inputs.candidate.period,
                inputs.candidate.t0,
                inputs.candidate.duration_hours,
            )

            return _convert_legacy_result(result, self.id, self.name)
        except Exception as e:
            return error_result(
                self.id,
                self.name,
                error=type(e).__name__,
                notes=[str(e)],
            )


def register_lc_checks(registry: Any) -> None:
    """Register LC-only checks V01-V05 with a registry.

    Args:
        registry: A CheckRegistry instance.
    """
    registry.register(OddEvenDepthCheck())
    registry.register(SecondaryEclipseCheck())
    registry.register(DurationConsistencyCheck())
    registry.register(DepthStabilityCheck())
    registry.register(VShapeCheck())


# All check classes for explicit imports
__all__ = [
    "OddEvenDepthCheck",
    "SecondaryEclipseCheck",
    "DurationConsistencyCheck",
    "DepthStabilityCheck",
    "VShapeCheck",
    "register_lc_checks",
]
