"""ModShift uniqueness check wrapper implementing VettingCheck protocol.

This module wraps the independent ModShift implementation (V11b) to conform
to the VettingCheck protocol and CheckResult schema.

Check Summary:
- V11b ModShiftUniquenessCheck: Signal uniqueness and secondary eclipse detection
  with properly-scaled Fred and MS1-MS6 metrics

Novelty: independent implementation from published papers (no GPL code)

References:
    [1] Thompson et al. 2018, ApJS 235, 38 - DR25 Robovetter ModShift
    [2] Coughlin et al. 2016, ApJS 224, 12 - ModShift algorithm
    [3] Kunimoto et al. 2025, AJ 170, 280 - LEO-vetter (methodology reference)
"""

from __future__ import annotations

from typing import Any

from bittr_tess_vetter.domain.detection import TransitCandidate
from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.validation.modshift_uniqueness import run_modshift_uniqueness
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
        ValueError: If depth is not provided (required for modshift checks).
    """
    # If already a TransitCandidate, use it directly
    if isinstance(candidate, TransitCandidate):
        if candidate.depth is None:
            raise ValueError("Candidate depth is required for modshift checks")
        return candidate

    # Handle API Candidate with nested ephemeris
    if hasattr(candidate, "ephemeris"):
        depth = candidate.depth
        if depth is None:
            raise ValueError("Candidate depth is required for modshift checks")
        return TransitCandidate(
            period=candidate.ephemeris.period_days,
            t0=candidate.ephemeris.t0_btjd,
            duration_hours=candidate.ephemeris.duration_hours,
            depth=depth,
            snr=0.0,  # Placeholder - not used by modshift checks
        )

    # Handle object with flat fields (like TransitCandidate but not instance)
    depth = getattr(candidate, "depth", None)
    if depth is None:
        raise ValueError("Candidate depth is required for modshift checks")
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


class ModShiftUniquenessCheck:
    """V11b: ModShift signal uniqueness test.

    Independent implementation of ModShift signal uniqueness metrics for
    secondary eclipse detection and signal quality assessment. Uses properly-
    scaled Fred (red/white noise ratio) that works correctly for TESS data.

    Key metrics:
    - sig_pri/sec/ter/pos: Signal significances at primary, secondary, etc.
    - fred: Red/white noise ratio (properly scaled, ~1-10 for TESS)
    - ms1: Primary uniqueness (sig_pri/fred - FA1)
    - ms2: Primary vs tertiary (sig_pri - sig_ter - FA2)
    - ms3: Primary vs positive (sig_pri - sig_pos - FA2)
    - ms4-ms6: Secondary eclipse detection metrics
    - med_chases: Local event uniqueness (0-1, higher is better)
    - chi: Transit depth consistency

    Thresholds (from published Robovetter calibration):
    - MS1 < 0.2: Weak primary uniqueness (fail)
    - MS2 or MS3 < 0.8: Contaminated by tertiary/positive (fail)
    - MS4 > 0 AND MS5 > -1 AND MS6 > -1: Secondary eclipse detected (EB)
    - CHASES < 0.78: Local systematic contamination
    - CHI < 7.8: Variable-depth transits (EB characteristic)

    References:
        [1] Thompson et al. 2018, ApJS 235, 38 - Section 3.2.3: ModShift
        [2] Coughlin et al. 2016, ApJS 224, 12 - ModShift algorithm
        [3] Kunimoto et al. 2025, AJ 170, 280 - LEO-vetter methodology
    """

    _id = "V11b"
    _name = "ModShiftUniqueness"
    _tier = CheckTier.LC_ONLY  # No external dependencies
    _requirements = CheckRequirements()  # Pure numpy/scipy implementation
    _citations = [
        "Thompson et al. 2018, ApJS 235, 38",
        "Coughlin et al. 2016, ApJS 224, 12",
        "Kunimoto et al. 2025, AJ 170, 280",
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
        """Execute ModShift uniqueness check.

        Args:
            inputs: Check inputs containing light curve and candidate.
            config: Check configuration.

        Returns:
            CheckResult with ModShift uniqueness metrics.
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

        # Extract config parameters
        extra = config.extra_params
        n_tce = extra.get("n_tce", 20000)

        # Run the underlying check
        try:
            result = run_modshift_uniqueness(
                time=internal_lc.time,
                flux=internal_lc.flux,
                flux_err=internal_lc.flux_err,
                period=internal_candidate.period,
                t0=internal_candidate.t0,
                duration_hours=internal_candidate.duration_hours,
                n_tce=n_tce,
            )
        except Exception as e:
            return error_result(
                self.id,
                self.name,
                error=type(e).__name__,
                notes=[str(e)],
            )

        status = result.get("status", "ok")
        warnings = result.get("warnings", [])

        # Handle various status cases
        if status == "invalid":
            reason = warnings[0] if warnings else "unknown"
            return skipped_result(
                self.id,
                self.name,
                reason_flag=reason.upper().replace(" ", "_"),
                notes=warnings,
                raw=result,
            )

        if status == "error":
            return error_result(
                self.id,
                self.name,
                error="modshift_error",
                notes=warnings,
                raw=result,
            )

        # Build metrics from successful run
        flags: list[str] = list(warnings)

        metrics: dict[str, float | int | str | bool | None] = {
            # Signal significances
            "sig_pri": result.get("sig_pri"),
            "sig_sec": result.get("sig_sec"),
            "sig_ter": result.get("sig_ter"),
            "sig_pos": result.get("sig_pos"),
            # Fred and false alarm thresholds
            "fred": result.get("fred"),
            "fa1": result.get("fa1"),
            "fa2": result.get("fa2"),
            # MS metrics (uniqueness)
            "ms1": result.get("ms1"),
            "ms2": result.get("ms2"),
            "ms3": result.get("ms3"),
            "ms4": result.get("ms4"),
            "ms5": result.get("ms5"),
            "ms6": result.get("ms6"),
            # Bonus metrics
            "med_chases": result.get("med_chases"),
            "chi": result.get("chi"),
            # Provenance
            "n_in": result.get("n_in"),
            "n_out": result.get("n_out"),
            "n_transits": result.get("n_transits"),
        }

        # Add backward-compatible ratios for downstream consumers
        sig_pri = result.get("sig_pri", 0.0)
        sig_sec = result.get("sig_sec", 0.0)
        sig_ter = result.get("sig_ter", 0.0)
        sig_pos = result.get("sig_pos", 0.0)

        if sig_pri and sig_pri > 0:
            metrics["secondary_primary_ratio"] = sig_sec / sig_pri
            metrics["tertiary_primary_ratio"] = sig_ter / sig_pri
            metrics["positive_primary_ratio"] = sig_pos / sig_pri

        return ok_result(
            self.id,
            self.name,
            metrics=metrics,
            confidence=0.9 if status == "ok" else 0.5,
            flags=flags,
            raw=result,
        )


def register_modshift_uniqueness_check(registry: CheckRegistry) -> None:
    """Register ModShift uniqueness check V11b with the registry.

    Args:
        registry: CheckRegistry to register check with.
    """
    registry.register(ModShiftUniquenessCheck())


__all__ = [
    "ModShiftUniquenessCheck",
    "register_modshift_uniqueness_check",
]
