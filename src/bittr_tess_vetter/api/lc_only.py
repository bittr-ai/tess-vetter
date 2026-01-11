"""Light-curve-only vetting checks for the public API.

This module provides thin wrappers around the LC-only vetting checks (V01-V05),
converting between facade types and internal types.

Check Summary:
- V01 odd_even_depth: Compare depth of odd vs even transits (detect EBs at 2x period)
- V02 secondary_eclipse: Search for secondary eclipse at phase 0.5
- V03 duration_consistency: Check transit duration vs stellar density expectation
- V04 depth_stability: Check depth consistency across individual transits
- V05 v_shape: Distinguish U-shaped (planet) vs V-shaped (grazing EB) transits

Novelty: standard (all checks implement well-established techniques from literature)

References:
    [1] Coughlin et al. 2016, ApJS 224, 12 (2016ApJS..224...12C) - Kepler DR24 Robovetter
    [2] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T) - Kepler DR25 catalog
    [3] Seager & Mallen-Ornelas 2003, ApJ 585, 1038 (2003ApJ...585.1038S) - Transit duration
    [4] Guerrero et al. 2021, ApJS 254, 39 (2021ApJS..254...39G) - TESS TOI catalog
    [5] Twicken et al. 2018, PASP 130, 064502 (2018PASP..130f4502T) - Kepler Data Validation
"""

from __future__ import annotations

import warnings
from typing import Any

from bittr_tess_vetter.api.references import (
    COUGHLIN_2016,
    COUGHLIN_LOPEZ_MORALES_2012,
    FRESSIN_2013,
    GUERRERO_2021,
    PONT_2006,
    PRSA_2011,
    SEAGER_MALLEN_ORNELAS_2003,
    THOMPSON_2018,
    TWICKEN_2018,
    cite,
    cites,
)
from bittr_tess_vetter.api.types import CheckResult, Ephemeris, LightCurve, StellarParams
from bittr_tess_vetter.validation.lc_checks import (
    DepthStabilityConfig,
    OddEvenConfig,
    SecondaryEclipseConfig,
    VShapeConfig,
    check_depth_stability as _check_depth_stability,
)
from bittr_tess_vetter.validation.lc_checks import (
    check_duration_consistency as _check_duration_consistency,
)
from bittr_tess_vetter.validation.lc_checks import (
    check_odd_even_depth as _check_odd_even_depth,
)
from bittr_tess_vetter.validation.lc_checks import (
    check_secondary_eclipse as _check_secondary_eclipse,
)
from bittr_tess_vetter.validation.lc_checks import (
    check_v_shape as _check_v_shape,
)

# Module-level references for programmatic access (generated from central registry)
REFERENCES = [
    ref.to_dict()
    for ref in [
        SEAGER_MALLEN_ORNELAS_2003,
        PRSA_2011,
        COUGHLIN_LOPEZ_MORALES_2012,
        FRESSIN_2013,
        COUGHLIN_2016,
        THOMPSON_2018,
        TWICKEN_2018,
        GUERRERO_2021,
    ]
]


def _convert_result(result: object) -> CheckResult:
    """Convert internal VetterCheckResult to facade CheckResult.

    Args:
        result: Internal VetterCheckResult (pydantic model)

    Returns:
        Facade CheckResult dataclass
    """
    # VetterCheckResult is a pydantic model with these attributes
    return CheckResult(
        id=result.id,  # type: ignore[attr-defined]
        name=result.name,  # type: ignore[attr-defined]
        passed=result.passed,  # type: ignore[attr-defined]
        confidence=result.confidence,  # type: ignore[attr-defined]
        details=dict(result.details),  # type: ignore[attr-defined]
    )


def _apply_policy_mode(check: CheckResult, *, policy_mode: str) -> CheckResult:
    if policy_mode != "metrics_only":
        warnings.warn(
            "bittr_tess_vetter.api.* `policy_mode` is deprecated and ignored; "
            "bittr-tess-vetter always returns metrics-only results. "
            "Move interpretation/policy decisions to astro-arc-tess validation (tess-validate).",
            category=FutureWarning,
            stacklevel=2,
        )
    if check.passed is None and check.details.get("_metrics_only") is True:
        return check
    details = dict(check.details)
    details["_metrics_only"] = True
    if policy_mode != "metrics_only":
        details["_policy_mode_ignored"] = policy_mode
    return CheckResult(
        id=check.id,
        name=check.name,
        passed=None,
        confidence=check.confidence,
        details=details,
    )


@cites(
    cite(COUGHLIN_2016, "§4.2 odd/even depth test"),
    cite(THOMPSON_2018, "§3.3.1 DR25 odd/even comparison"),
    cite(
        PONT_2006,
        "§2–3 time-binning correlated noise; used here as OOT red-noise inflation heuristic",
    ),
)
def odd_even_depth(
    lc: LightCurve,
    ephemeris: Ephemeris,
    *,
    config: dict[str, Any] | None = None,
    policy_mode: str = "metrics_only",
) -> CheckResult:
    """V01: Compare depth of odd vs even transits.

    Detects eclipsing binaries masquerading as planets at 2x the true period.
    If odd and even depths differ significantly, likely an EB.

    Args:
        lc: Light curve data
        ephemeris: Transit ephemeris (period, t0, duration)

    Returns:
        CheckResult with odd/even depth metrics (metrics-only; no pass/fail)

    Novelty: standard

    References:
        [1] Coughlin et al. 2016, ApJS 224, 12 (2016ApJS..224...12C)
            Section 4.2: Odd/even depth test in Kepler Robovetter
        [2] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T)
            Section 3.3.1: DR25 odd/even transit depth comparison
        [3] Pont et al. 2006, MNRAS 373, 231 (2006MNRAS.373..231P)
            Sections 2-3: Time-correlated (red) noise; binning-based inflation
    """
    internal_lc = lc.to_internal()
    internal_config = OddEvenConfig(**config) if config else None
    result = _check_odd_even_depth(
        lightcurve=internal_lc,
        period=ephemeris.period_days,
        t0=ephemeris.t0_btjd,
        duration_hours=ephemeris.duration_hours,
        config=internal_config,
    )
    return _apply_policy_mode(_convert_result(result), policy_mode=policy_mode)


@cites(
    cite(COUGHLIN_LOPEZ_MORALES_2012, "secondary eclipse search methodology"),
    cite(THOMPSON_2018, "§3.2 Significant Secondary test"),
    cite(FRESSIN_2013, "§3 FP scenarios incl. secondary eclipses"),
)
def secondary_eclipse(
    lc: LightCurve,
    ephemeris: Ephemeris,
    *,
    config: dict[str, Any] | None = None,
    policy_mode: str = "metrics_only",
) -> CheckResult:
    """V02: Search for secondary eclipse at phase 0.5.

    Presence of secondary eclipse indicates hot planet (thermal emission)
    or eclipsing binary. Significant secondary suggests EB.

    Args:
        lc: Light curve data
        ephemeris: Transit ephemeris (period, t0, duration)

    Returns:
        CheckResult with details on secondary eclipse search

    Novelty: standard

    References:
        [1] Coughlin & Lopez-Morales 2012, AJ 143, 39 (2012AJ....143...39C)
            Uniform search for secondary eclipses of hot Jupiters in Kepler
        [2] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T)
            Section 3.2: Significant Secondary test in DR25 Robovetter
        [3] Fressin et al. 2013, ApJ 766, 81 (2013ApJ...766...81F)
            Section 3: False positive scenarios including secondary eclipses
    """
    internal_lc = lc.to_internal()
    internal_config = SecondaryEclipseConfig(**config) if config else None
    result = _check_secondary_eclipse(
        lightcurve=internal_lc,
        period=ephemeris.period_days,
        t0=ephemeris.t0_btjd,
        config=internal_config,
    )
    return _apply_policy_mode(_convert_result(result), policy_mode=policy_mode)


@cites(
    cite(SEAGER_MALLEN_ORNELAS_2003, "Eq.3,9,19 duration-density relation"),
    cite(TWICKEN_2018, "§4.3 duration consistency test"),
    cite(THOMPSON_2018, "§3.4 Planet in Star metric"),
)
def duration_consistency(
    ephemeris: Ephemeris,
    stellar: StellarParams | None,
    *,
    policy_mode: str = "metrics_only",
) -> CheckResult:
    """V03: Check transit duration vs stellar density expectation.

    Transit duration depends on stellar density. Unphysical durations
    (too long or too short) indicate false positive.

    Expected: T_dur ~ P^(1/3) / rho_star^(1/3)

    Args:
        ephemeris: Transit ephemeris (period, t0, duration)
        stellar: Stellar parameters (optional but strongly recommended)

    Returns:
        CheckResult with duration consistency analysis

    Novelty: standard

    References:
        [1] Seager & Mallen-Ornelas 2003, ApJ 585, 1038 (2003ApJ...585.1038S)
            Equations 3, 9, 19: Transit duration and stellar density relationship
        [2] Twicken et al. 2018, PASP 130, 064502 (2018PASP..130f4502T)
            Section 4.3: Transit duration consistency test in Kepler DV
        [3] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T)
            Section 3.4: Planet in Star metric for duration validation
    """
    result = _check_duration_consistency(
        period=ephemeris.period_days,
        duration_hours=ephemeris.duration_hours,
        stellar=stellar,
    )
    return _apply_policy_mode(_convert_result(result), policy_mode=policy_mode)


@cites(
    cite(THOMPSON_2018, "§3.5 individual transit metrics"),
    cite(TWICKEN_2018, "§4.5 transit depth stability"),
    cite(GUERRERO_2021, "§3.2 TESS TOI depth consistency"),
)
def depth_stability(
    lc: LightCurve,
    ephemeris: Ephemeris,
    *,
    config: dict[str, Any] | None = None,
    policy_mode: str = "metrics_only",
) -> CheckResult:
    """V04: Check depth consistency across individual transits.

    Variable depth suggests blended eclipsing binary or systematic issues.
    Real planets have consistent depths.

    Args:
        lc: Light curve data
        ephemeris: Transit ephemeris (period, t0, duration)

    Returns:
        CheckResult with depth stability metrics

    Novelty: standard

    References:
        [1] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T)
            Section 3.5: Individual transit metrics for depth consistency
        [2] Twicken et al. 2018, PASP 130, 064502 (2018PASP..130f4502T)
            Section 4.5: Transit depth stability in Kepler DV pipeline
        [3] Guerrero et al. 2021, ApJS 254, 39 (2021ApJS..254...39G)
            Section 3.2: TESS TOI vetting including depth consistency
    """
    internal_lc = lc.to_internal()
    internal_config = DepthStabilityConfig(**config) if config else None
    result = _check_depth_stability(
        lightcurve=internal_lc,
        period=ephemeris.period_days,
        t0=ephemeris.t0_btjd,
        duration_hours=ephemeris.duration_hours,
        config=internal_config,
    )
    return _apply_policy_mode(_convert_result(result), policy_mode=policy_mode)


@cites(
    cite(SEAGER_MALLEN_ORNELAS_2003, "§3 transit shape parameters tF/tT"),
    cite(PRSA_2011, "§3.2 EB morphology classification"),
    cite(THOMPSON_2018, "§3.1 Not Transit-Like V-shape metric"),
)
def v_shape(
    lc: LightCurve,
    ephemeris: Ephemeris,
    *,
    config: dict[str, Any] | None = None,
    policy_mode: str = "metrics_only",
) -> CheckResult:
    """V05: Distinguish U-shaped (planet) vs V-shaped (grazing EB) transits.

    Planets have flat-bottomed U-shaped transits. Grazing eclipsing binaries
    show V-shaped transits with no flat bottom.

    Args:
        lc: Light curve data
        ephemeris: Transit ephemeris (period, t0, duration)

    Returns:
        CheckResult with shape analysis

    Novelty: standard

    References:
        [1] Seager & Mallen-Ornelas 2003, ApJ 585, 1038 (2003ApJ...585.1038S)
            Section 3: Transit shape parameters tF/tT and impact parameter b
        [2] Prsa et al. 2011, AJ 141, 83 (2011AJ....141...83P)
            Section 3.2: Morphology classification of eclipsing binaries
        [3] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T)
            Section 3.1: Not Transit-Like (V-shape) metric in DR25 Robovetter
    """
    internal_lc = lc.to_internal()
    internal_config = VShapeConfig(**config) if config else None
    result = _check_v_shape(
        lightcurve=internal_lc,
        period=ephemeris.period_days,
        t0=ephemeris.t0_btjd,
        duration_hours=ephemeris.duration_hours,
        config=internal_config,
    )
    return _apply_policy_mode(_convert_result(result), policy_mode=policy_mode)


# Define the default enabled checks and their order
_DEFAULT_CHECKS = ["V01", "V02", "V03", "V04", "V05"]


def vet_lc_only(
    lc: LightCurve,
    ephemeris: Ephemeris,
    *,
    stellar: StellarParams | None = None,
    enabled: set[str] | None = None,
    config: dict[str, dict[str, Any]] | None = None,
    policy_mode: str = "metrics_only",
) -> list[CheckResult]:
    """Run all LC-only vetting checks (V01-V05).

    This is the main orchestrator function for light-curve-only vetting.
    Runs checks in order V01-V05, optionally filtered by the enabled set.

    Args:
        lc: Light curve data
        ephemeris: Transit ephemeris (period, t0, duration)
        stellar: Stellar parameters (optional, improves V03)
        enabled: Set of check IDs to run (e.g., {"V01", "V03"}).
            If None, runs all checks.

    Returns:
        List of CheckResult objects for each enabled check

    Example:
        >>> from bittr_tess_vetter.api import LightCurve, Ephemeris, vet_lc_only
        >>> lc = LightCurve(time=time, flux=flux)
        >>> eph = Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=2.5)
        >>> results = vet_lc_only(lc, eph)
        >>> for r in results:
        ...     print(f\"{r.id} {r.name}: passed={r.passed} confidence={r.confidence:.2f}\")

    Novelty: standard

    References:
        See individual check functions (V01-V05) for specific citations.
        General methodology follows the Kepler Robovetter approach:
        [1] Coughlin et al. 2016, ApJS 224, 12 (2016ApJS..224...12C)
        [2] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T)
    """
    # Determine which checks to run
    checks_to_run = (
        _DEFAULT_CHECKS if enabled is None else [c for c in _DEFAULT_CHECKS if c in enabled]
    )

    results: list[CheckResult] = []
    config = config or {}

    for check_id in checks_to_run:
        if check_id == "V01":
            results.append(
                odd_even_depth(lc, ephemeris, config=config.get("V01"), policy_mode=policy_mode)
            )
        elif check_id == "V02":
            results.append(
                secondary_eclipse(lc, ephemeris, config=config.get("V02"), policy_mode=policy_mode)
            )
        elif check_id == "V03":
            results.append(duration_consistency(ephemeris, stellar, policy_mode=policy_mode))
        elif check_id == "V04":
            results.append(
                depth_stability(lc, ephemeris, config=config.get("V04"), policy_mode=policy_mode)
            )
        elif check_id == "V05":
            results.append(v_shape(lc, ephemeris, config=config.get("V05"), policy_mode=policy_mode))

    return results
