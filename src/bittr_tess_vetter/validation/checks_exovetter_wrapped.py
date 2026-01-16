"""Exovetter-based vetting check wrappers implementing VettingCheck protocol.

This module wraps the existing exovetter check implementations (V11-V12)
to conform to the new VettingCheck protocol and CheckResult schema.

Check Summary:
- V11 ModShiftCheck: Detect secondary eclipses at arbitrary phases
- V12 SweetCheck: Detect stellar variability mimicking transits

Novelty: standard (wrapping existing implementations)

References:
    [1] Thompson et al. 2018, ApJS 235, 38 - DR25 Robovetter ModShift/SWEET
    [2] Coughlin et al. 2016, ApJS 224, 12 - DR24 Robovetter
"""

from __future__ import annotations

from typing import Any

from bittr_tess_vetter.domain.detection import TransitCandidate
from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.validation.exovetter_checks import run_modshift, run_sweet
from bittr_tess_vetter.validation.registry import (
    CheckConfig,
    CheckInputs,
    CheckRegistry,
    CheckRequirements,
    CheckTier,
)
from bittr_tess_vetter.validation.result_schema import (
    CheckResult,
    error_result,
    ok_result,
    skipped_result,
)


def _candidate_to_internal(candidate: Any) -> TransitCandidate:
    """Convert API Candidate or internal TransitCandidate to TransitCandidate.

    Args:
        candidate: Either an API Candidate with ephemeris attribute,
            or an internal TransitCandidate with flat fields.

    Returns:
        Internal TransitCandidate for vetting checks.

    Raises:
        ValueError: If depth is not provided (required for exovetter checks).
    """
    # If already a TransitCandidate, use it directly
    if isinstance(candidate, TransitCandidate):
        if candidate.depth is None:
            raise ValueError("Candidate depth is required for exovetter checks")
        return candidate

    # Handle API Candidate with nested ephemeris
    if hasattr(candidate, "ephemeris"):
        depth = candidate.depth
        if depth is None:
            raise ValueError("Candidate depth is required for exovetter checks")
        return TransitCandidate(
            period=candidate.ephemeris.period_days,
            t0=candidate.ephemeris.t0_btjd,
            duration_hours=candidate.ephemeris.duration_hours,
            depth=depth,
            snr=0.0,  # Placeholder - not used by exovetter checks
        )

    # Handle object with flat fields (like TransitCandidate but not instance)
    depth = getattr(candidate, "depth", None)
    if depth is None:
        raise ValueError("Candidate depth is required for exovetter checks")
    return TransitCandidate(
        period=candidate.period,
        t0=candidate.t0,
        duration_hours=candidate.duration_hours,
        depth=depth,
        snr=getattr(candidate, "snr", 0.0),
    )


def _lc_to_internal(lc: Any) -> LightCurveData:
    """Convert API LightCurve to internal LightCurveData.

    Args:
        lc: API LightCurve with to_internal() method.

    Returns:
        Internal LightCurveData for vetting checks.
    """
    # API LightCurve should have to_internal() method
    if hasattr(lc, "to_internal"):
        return lc.to_internal()

    # Fallback for direct LightCurveData
    return lc


class ModShiftCheck:
    """V11: ModShift test for secondary eclipse detection.

    Detects eccentric eclipsing binaries where the secondary eclipse occurs
    at an unexpected phase (not 0.5). This catches EBs that would be missed
    by the standard secondary eclipse search at phase 0.5.

    Key metrics:
    - primary_signal: Main transit/eclipse signal strength
    - secondary_signal: Secondary eclipse signal at any phase
    - secondary_primary_ratio: sec/pri ratio (note: signal metric, not depth)
    - fred: Red noise level affecting reliability

    References:
        [1] Thompson et al. 2018, ApJS 235, 38 - Section 3.2.3: ModShift technique
        [2] Coughlin et al. 2016, ApJS 224, 12 - DR24 ModShift implementation
    """

    _id = "V11"
    _name = "ModShift"
    _tier = CheckTier.EXOVETTER
    _requirements = CheckRequirements(optional_deps=("exovetter",))
    _citations = [
        "Thompson et al. 2018, ApJS 235, 38",
        "Coughlin et al. 2016, ApJS 224, 12",
    ]

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def tier(self) -> CheckTier:
        return self._tier

    @property
    def requirements(self) -> CheckRequirements:
        return self._requirements

    @property
    def citations(self) -> list[str]:
        return self._citations

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        """Execute ModShift check.

        Args:
            inputs: Check inputs containing light curve and candidate.
            config: Check configuration.

        Returns:
            CheckResult with ModShift metrics.
        """
        # Convert inputs
        try:
            internal_lc = _lc_to_internal(inputs.lc)
            internal_candidate = _candidate_to_internal(inputs.candidate)
        except ValueError as e:
            return skipped_result(
                self.id,
                self.name,
                reason_flag="MISSING_DEPTH",
                notes=[str(e)],
            )
        except Exception as e:
            return error_result(
                self.id,
                self.name,
                error=type(e).__name__,
                notes=[str(e)],
            )

        # Run the underlying check
        try:
            result = run_modshift(candidate=internal_candidate, lightcurve=internal_lc)
        except Exception as e:
            return error_result(
                self.id,
                self.name,
                error=type(e).__name__,
                notes=[str(e)],
            )

        details = dict(result.details)
        status = details.get("status")

        # Handle various status cases
        if status == "invalid":
            reason = details.get("reason", "unknown")
            return skipped_result(
                self.id,
                self.name,
                reason_flag=reason.upper(),
                notes=[f"Invalid input: {reason}"],
                raw=details,
            )

        if status == "error":
            error_type = details.get("reason", "unknown_error")
            error_msg = details.get("error", "Unknown error")
            return error_result(
                self.id,
                self.name,
                error=error_type,
                notes=[error_msg],
                raw=details,
            )

        # Build metrics from successful run
        flags: list[str] = []
        warnings = details.get("warnings", [])
        for w in warnings:
            if isinstance(w, str):
                flags.append(w)

        # Extract inputs_summary for provenance
        inputs_summary = details.get("inputs_summary", {})

        metrics: dict[str, float | int | str | bool | None] = {
            "primary_signal": details.get("primary_signal"),
            "secondary_signal": details.get("secondary_signal"),
            "tertiary_signal": details.get("tertiary_signal"),
            "fred": details.get("fred"),
            "false_alarm_threshold": details.get("false_alarm_threshold"),
            "secondary_primary_ratio": details.get("secondary_primary_ratio"),
            "tertiary_primary_ratio": details.get("tertiary_primary_ratio"),
        }

        # Add inputs summary to metrics for analysis
        if inputs_summary:
            metrics["n_points"] = inputs_summary.get("n_points")
            metrics["baseline_days"] = inputs_summary.get("baseline_days")
            metrics["n_transits_expected"] = inputs_summary.get("n_transits_expected")

        return ok_result(
            self.id,
            self.name,
            metrics=metrics,
            confidence=result.confidence,
            flags=flags,
            raw=details,
        )


class SweetCheck:
    """V12: SWEET test for stellar variability.

    SWEET (Sine Wave Evaluation for Ephemeris Transits) checks whether the
    observed signal could be explained by stellar variability (rotation,
    pulsation) rather than a planetary transit.

    Tests sinusoidal fits at:
    - Half the transit period (P/2): even harmonics
    - The transit period (P): direct variability
    - Twice the transit period (2P): subharmonics

    References:
        [1] Thompson et al. 2018, ApJS 235, 38 - Section 3.2.4: SWEET test
        [2] Coughlin et al. 2016, ApJS 224, 12 - Section 4.4: Original SWEET
    """

    _id = "V12"
    _name = "SWEET"
    _tier = CheckTier.EXOVETTER
    _requirements = CheckRequirements(optional_deps=("exovetter",))
    _citations = [
        "Thompson et al. 2018, ApJS 235, 38",
        "Coughlin et al. 2016, ApJS 224, 12",
    ]

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def tier(self) -> CheckTier:
        return self._tier

    @property
    def requirements(self) -> CheckRequirements:
        return self._requirements

    @property
    def citations(self) -> list[str]:
        return self._citations

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        """Execute SWEET check.

        Args:
            inputs: Check inputs containing light curve and candidate.
            config: Check configuration.

        Returns:
            CheckResult with SWEET metrics.
        """
        # Convert inputs
        try:
            internal_lc = _lc_to_internal(inputs.lc)
            internal_candidate = _candidate_to_internal(inputs.candidate)
        except ValueError as e:
            return skipped_result(
                self.id,
                self.name,
                reason_flag="MISSING_DEPTH",
                notes=[str(e)],
            )
        except Exception as e:
            return error_result(
                self.id,
                self.name,
                error=type(e).__name__,
                notes=[str(e)],
            )

        # Run the underlying check
        try:
            result = run_sweet(candidate=internal_candidate, lightcurve=internal_lc)
        except Exception as e:
            return error_result(
                self.id,
                self.name,
                error=type(e).__name__,
                notes=[str(e)],
            )

        details = dict(result.details)
        status = details.get("status")

        # Handle error status
        if status == "error":
            error_type = details.get("reason", "unknown_error")
            error_msg = details.get("error", "Unknown error")
            return error_result(
                self.id,
                self.name,
                error=error_type,
                notes=[error_msg],
                raw=details,
            )

        # Build metrics from successful run
        flags: list[str] = []
        warnings = details.get("warnings", [])
        for w in warnings:
            if isinstance(w, str):
                flags.append(w)

        # Extract inputs_summary
        inputs_summary = details.get("inputs_summary", {})
        raw_metrics = details.get("raw_metrics", {})

        metrics: dict[str, float | int | str | bool | None] = {}

        # Add inputs summary
        if inputs_summary:
            metrics["n_points"] = inputs_summary.get("n_points")
            metrics["baseline_days"] = inputs_summary.get("baseline_days")
            metrics["n_transits_expected"] = inputs_summary.get("n_transits_expected")
            metrics["cadence_median_min"] = inputs_summary.get("cadence_median_min")

        # Extract the explicit SNR fields from details (new approach)
        # These are more useful for ML than the raw msg pass/fail
        if details.get("snr_half_period") is not None:
            metrics["snr_half_period"] = details.get("snr_half_period")
        if details.get("snr_at_period") is not None:
            metrics["snr_at_period"] = details.get("snr_at_period")
        if details.get("snr_double_period") is not None:
            metrics["snr_double_period"] = details.get("snr_double_period")

        # Flatten raw_metrics from exovetter SWEET
        # SWEET returns metrics for half, full, and double period fits
        for key, value in raw_metrics.items():
            if isinstance(value, (int, float, bool, str)):
                metrics[f"sweet_{key}"] = value

        return ok_result(
            self.id,
            self.name,
            metrics=metrics,
            confidence=result.confidence,
            flags=flags,
            raw=details,
        )


def register_exovetter_checks(registry: CheckRegistry) -> None:
    """Register exovetter checks V11-V12 with the registry.

    Args:
        registry: CheckRegistry to register checks with.
    """
    registry.register(ModShiftCheck())
    registry.register(SweetCheck())


__all__ = [
    "ModShiftCheck",
    "SweetCheck",
    "register_exovetter_checks",
]
