"""Exovetter-based vetting checks for the public API (V11-V12).

This module provides thin wrappers around the exovetter library's ModShift and
SWEET tests, converting between facade types and internal types.

Check Summary:
- V11 modshift: Detect secondary eclipses at arbitrary phases (eccentric EBs)
- V12 sweet: Detect stellar variability at transit period mimicking transits

Novelty: standard (both checks implement well-established Kepler techniques)

References:
    [1] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T) - DR25 Robovetter ModShift/SWEET
    [2] Coughlin et al. 2016, ApJS 224, 12 (2016ApJS..224...12C) - DR24 Robovetter
"""

from __future__ import annotations

from typing import Any

from bittr_tess_vetter.api.references import (
    COUGHLIN_2016,
    THOMPSON_2018,
    cite,
    cites,
)
from bittr_tess_vetter.api.types import Candidate, CheckResult, LightCurve
from bittr_tess_vetter.domain.detection import TransitCandidate, VetterCheckResult
from bittr_tess_vetter.validation.exovetter_checks import run_modshift, run_sweet

# Module-level references for programmatic access (generated from central registry)
REFERENCES = [ref.to_dict() for ref in [THOMPSON_2018, COUGHLIN_2016]]


def _convert_result(result: VetterCheckResult) -> CheckResult:
    """Convert internal VetterCheckResult to facade CheckResult.

    Args:
        result: Internal VetterCheckResult (pydantic model)

    Returns:
        Facade CheckResult dataclass
    """
    details = dict(result.details)
    details["_metrics_only"] = True
    return CheckResult(
        id=result.id,
        name=result.name,
        passed=None,
        confidence=result.confidence,
        details=details,
    )


def _candidate_to_internal(candidate: Candidate) -> TransitCandidate:
    """Convert facade Candidate to internal TransitCandidate.

    Args:
        candidate: Facade Candidate with ephemeris and optional depth

    Returns:
        Internal TransitCandidate for vetting checks

    Raises:
        ValueError: If depth is not provided (required for exovetter checks)
    """
    depth = candidate.depth
    if depth is None:
        raise ValueError("Candidate depth is required for exovetter checks")

    # TransitCandidate requires snr but exovetter doesn't use it
    # Use a placeholder value
    return TransitCandidate(
        period=candidate.ephemeris.period_days,
        t0=candidate.ephemeris.t0_btjd,
        duration_hours=candidate.ephemeris.duration_hours,
        depth=depth,
        snr=0.0,  # Placeholder - not used by exovetter checks
    )


def _make_skipped_result(check_id: str, check_name: str, reason: str) -> CheckResult:
    """Create a skipped result when check is disabled.

    Args:
        check_id: Check identifier (V11 or V12)
        check_name: Human-readable check name
        reason: Reason for skipping

    Returns:
        CheckResult with status="skipped"
    """
    return CheckResult(
        id=check_id,
        name=check_name,
        passed=None,
        confidence=0.0,  # No confidence - not run
        details={"status": "skipped", "reason": reason, "_metrics_only": True},
    )


@cites(
    cite(THOMPSON_2018, "§3.2.3 ModShift secondary eclipse detection"),
    cite(COUGHLIN_2016, "DR24 Robovetter ModShift implementation"),
)
def modshift(
    lc: LightCurve,
    candidate: Candidate,
    *,
    enabled: bool = True,
    config: dict[str, Any] | None = None,
) -> CheckResult:
    """V11: ModShift test for secondary eclipse detection at arbitrary phase.

    Detects eccentric eclipsing binaries where the secondary eclipse occurs at
    an unexpected phase (not 0.5). This catches EBs that would be missed by
    the standard secondary eclipse search at phase 0.5.

    Key metrics include primary/secondary signal strength, a secondary/primary
    ratio (of ModShift signal metrics, not a depth ratio), and an estimate of
    red-noise impact on reliability. The secondary/primary ratio can become
    unstable when the primary signal is very small, so corroborate with a
    depth-style secondary eclipse measurement (phase ~0.5) and pixel/localization
    evidence.

    Args:
        lc: Light curve data
        candidate: Transit candidate with ephemeris and depth
        enabled: If False, return skipped result without running check
        config: Optional configuration overrides (reserved)

    Returns:
        CheckResult with ModShift metrics (metrics-only; no pass/fail)

    Novelty: standard

    References:
        [1] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T)
            Section 3.2.3: ModShift technique for detecting secondary eclipses
        [2] Coughlin et al. 2016, ApJS 224, 12 (2016ApJS..224...12C)
            DR24 Robovetter automated ModShift implementation
    """
    if not enabled:
        return _make_skipped_result("V11", "modshift", "Check disabled by caller")

    # Convert facade types to internal types
    internal_lc = lc.to_internal()
    try:
        internal_candidate = _candidate_to_internal(candidate)
    except ValueError as e:
        return CheckResult(
            id="V11",
            name="modshift",
            passed=None,
            confidence=0.20,
            details={"status": "error", "reason": str(e)},
        )

    del config
    result = run_modshift(candidate=internal_candidate, lightcurve=internal_lc)

    return _convert_result(result)


@cites(
    cite(THOMPSON_2018, "§3.2.4 SWEET stellar variability test"),
    cite(COUGHLIN_2016, "§4.4 original SWEET implementation"),
)
def sweet(
    lc: LightCurve,
    candidate: Candidate,
    *,
    enabled: bool = True,
    config: dict[str, Any] | None = None,
) -> CheckResult:
    """V12: SWEET test for stellar variability masquerading as transits.

    SWEET (Sine Wave Evaluation for Ephemeris Transits) checks whether the
    observed signal could be explained by stellar variability (rotation,
    pulsation) rather than a planetary transit.

    Tests sinusoidal fits at:
    - Half the transit period (P/2): even harmonics
    - The transit period (P): direct variability
    - Twice the transit period (2P): subharmonics

    Key metric: amplitude-to-uncertainty ratio
    - If ratio > threshold at P, signal may be stellar variability

    Args:
        lc: Light curve data
        candidate: Transit candidate with ephemeris and depth
        enabled: If False, return skipped result without running check
        config: Optional configuration overrides (reserved)

    Returns:
        CheckResult with SWEET metrics (metrics-only; no pass/fail)

    Novelty: standard

    References:
        [1] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T)
            Section 3.2.4: SWEET test for stellar variability in DR25 Robovetter
        [2] Coughlin et al. 2016, ApJS 224, 12 (2016ApJS..224...12C)
            Section 4.4: Original SWEET implementation in DR24
    """
    if not enabled:
        return _make_skipped_result("V12", "sweet", "Check disabled by caller")

    # Convert facade types to internal types
    internal_lc = lc.to_internal()
    try:
        internal_candidate = _candidate_to_internal(candidate)
    except ValueError as e:
        return CheckResult(
            id="V12",
            name="sweet",
            passed=None,
            confidence=0.20,
            details={"status": "error", "reason": str(e)},
        )

    del config
    result = run_sweet(candidate=internal_candidate, lightcurve=internal_lc)

    return _convert_result(result)


@cites(
    cite(THOMPSON_2018, "DR25 Robovetter ModShift/SWEET"),
    cite(COUGHLIN_2016, "DR24 Robovetter ModShift/SWEET"),
)
def vet_exovetter(
    lc: LightCurve,
    candidate: Candidate,
    *,
    enabled: set[str] | None = None,
    config: dict[str, dict[str, Any]] | None = None,
) -> list[CheckResult]:
    """Run all exovetter checks (V11-V12).

    This is the orchestrator function for exovetter-based vetting. Runs
    ModShift and SWEET in order, optionally filtered by the enabled set.

    Args:
        lc: Light curve data
        candidate: Transit candidate with ephemeris and depth
        enabled: Set of check IDs to run (e.g., {"V11", "V12"}).
            If None, runs all checks.
        config: Per-check configuration override, keyed by check ID.
            (reserved for future use; interpretation thresholds belong in the host application)

    Returns:
        List of CheckResult objects for each enabled check

    Example:
        >>> from bittr_tess_vetter.api import LightCurve, Ephemeris, Candidate
        >>> from bittr_tess_vetter.api.exovetter import vet_exovetter
        >>> lc = LightCurve(time=time, flux=flux)
        >>> eph = Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=2.5)
        >>> cand = Candidate(ephemeris=eph, depth_ppm=1000)
        >>> results = vet_exovetter(lc, cand)
        >>> for r in results:
        ...     print(f\"{r.id} {r.name}: confidence={r.confidence:.2f}\")

    Novelty: standard

    References:
        See individual check functions (V11-V12) for specific citations.
        General methodology follows the Kepler Robovetter approach:
        [1] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T)
    """
    config = config or {}
    results: list[CheckResult] = []

    # V11: ModShift
    v11_enabled = enabled is None or "V11" in enabled
    results.append(
        modshift(
            lc,
            candidate,
            enabled=v11_enabled,
            config=config.get("V11"),
        )
    )

    # V12: SWEET
    v12_enabled = enabled is None or "V12" in enabled
    results.append(
        sweet(
            lc,
            candidate,
            enabled=v12_enabled,
            config=config.get("V12"),
        )
    )

    return results
