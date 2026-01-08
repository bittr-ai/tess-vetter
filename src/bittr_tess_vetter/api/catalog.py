"""Catalog-based vetting checks for the public API (V06-V07).

This module provides thin wrappers around the catalog vetting checks,
converting between facade types and internal types.

Check Summary:
- V06 nearby_eb_search: Query TESS-EB catalog for known eclipsing binaries near target
- V07 exofop_disposition: Query ExoFOP-TESS for existing TOI dispositions

These checks require network access to query external catalogs. When network=False,
checks return a skipped result with details={"status": "skipped"}.

Novelty: standard (queries external catalogs using well-established methods)

References:
    [1] Prsa et al. 2022, ApJS 258, 16 (2022ApJS..258...16P) - TESS-EB catalog
    [2] Guerrero et al. 2021, ApJS 254, 39 (2021ApJS..254...39G) - TESS TOI catalog
    [3] ExoFOP-TESS: https://exofop.ipac.caltech.edu/tess/
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bittr_tess_vetter.api.types import Candidate, CheckResult
from bittr_tess_vetter.domain.detection import TransitCandidate
from bittr_tess_vetter.validation.checks_catalog import (
    ExoFOPDispositionCheck,
    NearbyEBCheck,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# Module-level references for programmatic access
REFERENCES: list[dict[str, str]] = [
    {
        "id": "Prsa2022",
        "type": "ads",
        "bibcode": "2022ApJS..258...16P",
        "title": "TESS Eclipsing Binary Stars. I. Short-cadence Observations of 4584 "
        "Eclipsing Binaries in Sectors 1-26",
        "authors": "Prsa, A.; Kochoska, A.; Conroy, K.E.; et al.",
        "journal": "ApJS 258, 16",
        "year": "2022",
        "note": "TESS-EB catalog for nearby eclipsing binary search (V06)",
    },
    {
        "id": "Guerrero2021",
        "type": "ads",
        "bibcode": "2021ApJS..254...39G",
        "title": "The TESS Objects of Interest Catalog from the TESS Prime Mission",
        "authors": "Guerrero, N.M.; Seager, S.; Huang, C.X.; et al.",
        "journal": "ApJS 254, 39",
        "year": "2021",
        "note": "TESS TOI catalog and ExoFOP disposition definitions (V07)",
    },
]


def _convert_result(result: object) -> CheckResult:
    """Convert internal VetterCheckResult to facade CheckResult.

    Args:
        result: Internal VetterCheckResult (pydantic model)

    Returns:
        Facade CheckResult dataclass
    """
    return CheckResult(
        id=result.id,  # type: ignore[attr-defined]
        name=result.name,  # type: ignore[attr-defined]
        passed=result.passed,  # type: ignore[attr-defined]
        confidence=result.confidence,  # type: ignore[attr-defined]
        details=dict(result.details),  # type: ignore[attr-defined]
    )


def _make_skipped_result(check_id: str, name: str) -> CheckResult:
    """Create a skipped result when network is disabled.

    Args:
        check_id: Check identifier (e.g., "V06")
        name: Check name (e.g., "nearby_eb_search")

    Returns:
        CheckResult with status="skipped"
    """
    return CheckResult(
        id=check_id,
        name=name,
        passed=True,
        confidence=0.0,
        details={
            "status": "skipped",
            "reason": "network_disabled",
            "note": "Network access disabled; set network=True to query catalog",
        },
    )


def _candidate_to_internal(candidate: Candidate) -> TransitCandidate:
    """Convert facade Candidate to internal TransitCandidate.

    Args:
        candidate: Facade Candidate with ephemeris

    Returns:
        Internal TransitCandidate
    """
    return TransitCandidate(
        period=candidate.ephemeris.period_days,
        t0=candidate.ephemeris.t0_btjd,
        duration_hours=candidate.ephemeris.duration_hours,
        depth=candidate.depth if candidate.depth is not None else 0.001,
        snr=10.0,  # Default SNR for catalog checks (not used)
    )


def nearby_eb_search(
    candidate: Candidate,
    *,
    ra_deg: float,
    dec_deg: float,
    network: bool = False,
    cache: Any | None = None,
    search_radius_arcsec: float = 42.0,
    period_tolerance: float = 0.1,
    http_get: Callable[..., Any] | None = None,
) -> CheckResult:
    """V06: Search for known eclipsing binaries near target.

    Queries the TESS-EB catalog (VizieR J/ApJS/258/16) for known eclipsing
    binaries within the TESS aperture (~21 arcsec pixel scale).

    A known EB within the photometric aperture can contaminate the light curve
    and produce transit-like signals that mimic planetary transits.

    Args:
        candidate: Transit candidate with ephemeris
        ra_deg: Target RA in degrees
        dec_deg: Target Dec in degrees
        network: If False, return skipped result without querying catalog
        cache: Optional cache for dependency injection (reserved for future use)
        search_radius_arcsec: Cone search radius (default 42 = 2 TESS pixels)
        period_tolerance: Fractional tolerance for period match (default 0.1 = 10%)
        http_get: Optional HTTP GET callable for dependency injection

    Returns:
        CheckResult with pass if no matching EB found

    Novelty: standard

    References:
        [1] Prsa et al. 2022, ApJS 258, 16 (2022ApJS..258...16P)
            TESS-EB catalog: 4584 eclipsing binaries from Sectors 1-26
        [2] Guerrero et al. 2021, ApJS 254, 39 (2021ApJS..254...39G)
            Section 3.1: Nearby EB contamination as false positive source
    """
    _ = cache  # Reserved for future use

    if not network:
        return _make_skipped_result("V06", "nearby_eb_search")

    internal_candidate = _candidate_to_internal(candidate)

    check = NearbyEBCheck(
        search_radius_arcsec=search_radius_arcsec,
        period_tolerance=period_tolerance,
        http_get=http_get,
    )
    result = check.run(
        internal_candidate,
        ra=ra_deg,
        dec=dec_deg,
    )
    return _convert_result(result)


def exofop_disposition(
    candidate: Candidate,
    *,
    tic_id: int,
    network: bool = False,
    cache: Any | None = None,
    toi: float | None = None,
    http_get: Callable[..., Any] | None = None,
) -> CheckResult:
    """V07: Check ExoFOP-TESS for existing dispositions.

    Queries ExoFOP-TESS API to check if the target already has a
    TFOPWG disposition flagging it as a false positive.

    False positive dispositions: FP, FA, EB, NEB, BEB, V, IS, O
    Planet dispositions: CP, KP, PC, APC

    Args:
        candidate: Transit candidate with ephemeris
        tic_id: TIC ID to query
        network: If False, return skipped result without querying ExoFOP
        cache: Optional cache for dependency injection (reserved for future use)
        toi: Optional specific TOI number (e.g., 123.01)
        http_get: Optional HTTP GET callable for dependency injection

    Returns:
        CheckResult with pass if no FP disposition found

    Novelty: standard

    References:
        [1] Guerrero et al. 2021, ApJS 254, 39 (2021ApJS..254...39G)
            Section 3.2: TFOPWG dispositions and vetting workflow
        [2] ExoFOP-TESS: https://exofop.ipac.caltech.edu/tess/
            Community follow-up program for TESS candidates
    """
    _ = cache  # Reserved for future use

    if not network:
        return _make_skipped_result("V07", "exofop_disposition")

    internal_candidate = _candidate_to_internal(candidate)

    check = ExoFOPDispositionCheck(http_get=http_get)
    result = check.run(
        internal_candidate,
        tic_id=tic_id,
        toi=toi,
    )
    return _convert_result(result)


# Define the default enabled checks and their order
_DEFAULT_CHECKS = ["V06", "V07"]


def vet_catalog(
    candidate: Candidate,
    *,
    tic_id: int,
    ra_deg: float,
    dec_deg: float,
    network: bool = False,
    cache: Any | None = None,
    toi: float | None = None,
    enabled: set[str] | None = None,
    search_radius_arcsec: float = 42.0,
    period_tolerance: float = 0.1,
    http_get: Callable[..., Any] | None = None,
) -> list[CheckResult]:
    """Run catalog vetting checks (V06-V07).

    This orchestrator runs both catalog pre-filter checks that can quickly
    reject known false positives before expensive analysis.

    Args:
        candidate: Transit candidate with ephemeris
        tic_id: TIC ID for ExoFOP query
        ra_deg: Target RA in degrees for TESS-EB query
        dec_deg: Target Dec in degrees for TESS-EB query
        network: If False, return skipped results for all checks
        cache: Optional cache for dependency injection (reserved for future use)
        toi: Optional specific TOI number for ExoFOP
        enabled: Set of check IDs to run (e.g., {"V06", "V07"}).
            If None, runs all checks.
        search_radius_arcsec: V06 search radius (default 42 arcsec)
        period_tolerance: V06 period match tolerance (default 10%)
        http_get: Optional HTTP GET callable for dependency injection

    Returns:
        List of CheckResult objects for each enabled check

    Example:
        >>> from bittr_tess_vetter.api import Candidate, Ephemeris, vet_catalog
        >>> eph = Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=2.5)
        >>> candidate = Candidate(ephemeris=eph, depth_ppm=1000)
        >>> results = vet_catalog(
        ...     candidate,
        ...     tic_id=123456789,
        ...     ra_deg=150.0,
        ...     dec_deg=-30.0,
        ...     network=True,
        ... )
        >>> for r in results:
        ...     print(f"{r.id} {r.name}: {'PASS' if r.passed else 'FAIL'}")

    Novelty: standard

    References:
        See individual check functions (V06-V07) for specific citations.
        General methodology follows TESS vetting procedures:
        [1] Guerrero et al. 2021, ApJS 254, 39 (2021ApJS..254...39G)
    """
    checks_to_run = (
        _DEFAULT_CHECKS if enabled is None else [c for c in _DEFAULT_CHECKS if c in enabled]
    )

    results: list[CheckResult] = []

    for check_id in checks_to_run:
        if check_id == "V06":
            results.append(
                nearby_eb_search(
                    candidate,
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    network=network,
                    cache=cache,
                    search_radius_arcsec=search_radius_arcsec,
                    period_tolerance=period_tolerance,
                    http_get=http_get,
                )
            )
        elif check_id == "V07":
            results.append(
                exofop_disposition(
                    candidate,
                    tic_id=tic_id,
                    network=network,
                    cache=cache,
                    toi=toi,
                    http_get=http_get,
                )
            )

    return results
