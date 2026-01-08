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

from bittr_tess_vetter.api.types import Candidate, CheckResult, LightCurve
from bittr_tess_vetter.domain.detection import TransitCandidate, VetterCheckResult
from bittr_tess_vetter.validation.exovetter_checks import ModshiftCheck, SWEETCheck

# Module-level references for programmatic access
REFERENCES: list[dict[str, str]] = [
    {
        "id": "Thompson2018",
        "type": "ads",
        "bibcode": "2018ApJS..235...38T",
        "title": "Planetary Candidates Observed by Kepler. VIII. A Fully Automated "
        "Catalog With Measured Completeness and Reliability Based on Data Release 25",
        "authors": "Thompson, S.E.; Coughlin, J.L.; Hoffman, K.; et al.",
        "journal": "ApJS 235, 38",
        "year": "2018",
        "note": "DR25 Robovetter: ModShift (Sec 3.2.3) and SWEET (Sec 3.2.4) tests",
    },
    {
        "id": "Coughlin2016",
        "type": "ads",
        "bibcode": "2016ApJS..224...12C",
        "title": "Planetary Candidates Observed by Kepler. VII. The First Fully Uniform "
        "Catalog Based on the Entire 48-month Data Set (Q1-Q17 DR24)",
        "authors": "Coughlin, J.L.; Mullally, F.; Thompson, S.E.; et al.",
        "journal": "ApJS 224, 12",
        "year": "2016",
        "note": "DR24 Robovetter with automated vetting including ModShift and SWEET",
    },
]


def _convert_result(result: VetterCheckResult) -> CheckResult:
    """Convert internal VetterCheckResult to facade CheckResult.

    Args:
        result: Internal VetterCheckResult (pydantic model)

    Returns:
        Facade CheckResult dataclass
    """
    return CheckResult(
        id=result.id,
        name=result.name,
        passed=result.passed,
        confidence=result.confidence,
        details=dict(result.details),
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
        passed=True,  # Non-blocking
        confidence=0.0,  # No confidence - not run
        details={"status": "skipped", "reason": reason},
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

    Key metrics:
    - primary_signal: Main transit/eclipse signal strength
    - secondary_signal: Secondary eclipse signal at any phase
    - secondary_primary_ratio: sec/pri ratio (>0.5 indicates EB)
    - fred: Red noise level affecting reliability

    Args:
        lc: Light curve data
        candidate: Transit candidate with ephemeris and depth
        enabled: If False, return skipped result without running check
        config: Optional configuration override (threshold, fred_warning_threshold)

    Returns:
        CheckResult with ModShift metrics and pass/fail status

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
            passed=True,
            confidence=0.20,
            details={"status": "error", "reason": str(e)},
        )

    # Run the internal check with optional config
    check_config = None
    if config:
        from bittr_tess_vetter.validation.base import CheckConfig

        check_config = CheckConfig(
            enabled=True,
            threshold=config.get("threshold", 0.5),
            additional={k: v for k, v in config.items() if k != "threshold"},
        )
    check = ModshiftCheck(config=check_config)
    result = check.run(internal_candidate, internal_lc)

    return _convert_result(result)


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
        config: Optional configuration override (threshold, half_period_threshold)

    Returns:
        CheckResult with SWEET metrics and pass/fail status

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
            passed=True,
            confidence=0.20,
            details={"status": "error", "reason": str(e)},
        )

    # Run the internal check with optional config
    check_config = None
    if config:
        from bittr_tess_vetter.validation.base import CheckConfig

        check_config = CheckConfig(
            enabled=True,
            threshold=config.get("threshold", 3.0),
            additional={k: v for k, v in config.items() if k != "threshold"},
        )
    check = SWEETCheck(config=check_config)
    result = check.run(internal_candidate, internal_lc)

    return _convert_result(result)


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
            Example: {"V11": {"threshold": 0.3}, "V12": {"threshold": 4.0}}

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
        ...     print(f"{r.id} {r.name}: {'PASS' if r.passed else 'FAIL'}")

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
