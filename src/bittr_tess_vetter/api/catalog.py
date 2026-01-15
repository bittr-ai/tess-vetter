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

from bittr_tess_vetter.api.references import (
    GUERRERO_2021,
    PRSA_2022,
    cite,
    cites,
)
from bittr_tess_vetter.api.types import (
    Candidate,
    CheckResult,
    ok_result,
    skipped_result,
)
from bittr_tess_vetter.domain.detection import TransitCandidate
from bittr_tess_vetter.validation.checks_catalog import (
    run_exofop_toi_lookup,
    run_nearby_eb_search,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# Module-level references for programmatic access (generated from central registry)
REFERENCES = [ref.to_dict() for ref in [PRSA_2022, GUERRERO_2021]]


def _convert_result(result: object) -> CheckResult:
    """Convert internal VetterCheckResult to canonical CheckResult.

    Args:
        result: Internal VetterCheckResult (pydantic model)

    Returns:
        Canonical CheckResult (Pydantic model from validation.result_schema)
    """
    from typing import Any

    details = dict(result.details)  # type: ignore[attr-defined]

    # Convert details dict to structured metrics (filter to JSON-serializable scalars)
    metrics: dict[str, float | int | str | bool | None] = {}
    raw_data: dict[str, Any] = {}
    for k, v in details.items():
        if isinstance(v, (float, int, str, bool, type(None))):
            metrics[k] = v
        else:
            raw_data[k] = v

    return ok_result(
        id=result.id,  # type: ignore[attr-defined]
        name=result.name,  # type: ignore[attr-defined]
        metrics=metrics,
        confidence=result.confidence,  # type: ignore[attr-defined]
        raw=raw_data if raw_data else None,
    )


def _make_skipped_result(check_id: str, name: str) -> CheckResult:
    """Create a skipped result when network is disabled.

    Args:
        check_id: Check identifier (e.g., "V06")
        name: Check name (e.g., "nearby_eb_search")

    Returns:
        CheckResult with status="skipped"
    """
    return skipped_result(
        id=check_id,
        name=name,
        reason_flag="NETWORK_DISABLED",
        notes=["Network access disabled; set network=True to query catalog"],
    )


def _make_missing_metadata_result(check_id: str, name: str, *, missing: list[str]) -> CheckResult:
    """Create a skipped result when required metadata is missing."""
    return skipped_result(
        id=check_id,
        name=name,
        reason_flag="MISSING_METADATA",
        notes=[f"Missing required metadata: {', '.join(missing)}"],
        raw={"missing": missing, "reason": "missing_metadata"},
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


@cites(
    cite(PRSA_2022, "TESS-EB catalog Sectors 1-26"),
    cite(GUERRERO_2021, "§3.1 nearby EB contamination"),
)
def nearby_eb_search(
    candidate: Candidate,
    *,
    ra_deg: float,
    dec_deg: float,
    network: bool = False,
    cache: Any | None = None,
    search_radius_arcsec: float = 42.0,
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
        http_get: Optional HTTP GET callable for dependency injection

    Returns:
        CheckResult with raw nearby-EB matches (metrics-only)

    Novelty: standard

    References:
        [1] Prsa et al. 2022, ApJS 258, 16 (2022ApJS..258...16P)
            TESS-EB catalog: 4584 eclipsing binaries from Sectors 1-26
        [2] Guerrero et al. 2021, ApJS 254, 39 (2021ApJS..254...39G)
            Section 3.1: Nearby EB contamination as false positive source
    """
    if not network:
        return _make_skipped_result("V06", "nearby_eb_search")

    internal_candidate = _candidate_to_internal(candidate)

    _ = cache  # Reserved for future use
    result = run_nearby_eb_search(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        candidate_period_days=float(internal_candidate.period),
        search_radius_arcsec=search_radius_arcsec,
        http_get=http_get,
    )
    return _convert_result(result)


@cites(
    cite(GUERRERO_2021, "§3.2 TFOPWG dispositions"),
)
def exofop_disposition(
    candidate: Candidate,
    *,
    tic_id: int,
    network: bool = False,
    cache: Any | None = None,
    toi: float | None = None,
    http_get: Callable[..., Any] | None = None,
) -> CheckResult:
    """V07: Look up the ExoFOP-TESS TOI table row for a TIC.

    Returns the raw row (if present). Interpretation of dispositions is
    applied by the host.

    Args:
        candidate: Transit candidate with ephemeris
        tic_id: TIC ID to query
        network: If False, return skipped result without querying ExoFOP
        cache: Optional cache for dependency injection (reserved for future use)
        toi: Optional specific TOI number (e.g., 123.01)
        http_get: Optional HTTP GET callable for dependency injection

    Returns:
        CheckResult with raw ExoFOP row fields (metrics-only)

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

    del candidate
    del cache
    result = run_exofop_toi_lookup(tic_id=tic_id, toi=toi, http_get=http_get)
    return _convert_result(result)


# Define the default enabled checks and their order
_DEFAULT_CHECKS = ["V06", "V07"]


@cites(
    cite(PRSA_2022, "TESS-EB catalog"),
    cite(GUERRERO_2021, "TESS TOI vetting procedures context"),
)
def vet_catalog(
    candidate: Candidate,
    *,
    tic_id: int | None = None,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    network: bool = False,
    cache: Any | None = None,
    toi: float | None = None,
    enabled: set[str] | None = None,
    search_radius_arcsec: float = 42.0,
    http_get: Callable[..., Any] | None = None,
) -> list[CheckResult]:
    """Run catalog vetting checks (V06-V07).

    This orchestrator runs catalog lookups and returns raw results for downstream
    interpretation (metrics-only).

    Args:
        candidate: Transit candidate with ephemeris
        tic_id: TIC ID for ExoFOP query (required for V07)
        ra_deg: Target RA in degrees for TESS-EB query (required for V06)
        dec_deg: Target Dec in degrees for TESS-EB query (required for V06)
        network: If False, return skipped results for all checks
        cache: Optional cache for dependency injection (reserved for future use)
        toi: Optional specific TOI number for ExoFOP
        enabled: Set of check IDs to run (e.g., {"V06", "V07"}).
            If None, runs all checks.
        search_radius_arcsec: V06 search radius (default 42 arcsec)
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
        ...     print(f"{r.id} {r.name}: confidence={r.confidence:.2f}")

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
            missing = []
            if ra_deg is None:
                missing.append("ra_deg")
            if dec_deg is None:
                missing.append("dec_deg")
            if missing:
                results.append(
                    _make_missing_metadata_result("V06", "nearby_eb_search", missing=missing)
                )
                continue
            assert ra_deg is not None
            assert dec_deg is not None
            results.append(
                nearby_eb_search(
                    candidate,
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    network=network,
                    cache=cache,
                    search_radius_arcsec=search_radius_arcsec,
                    http_get=http_get,
                )
            )
        elif check_id == "V07":
            if tic_id is None:
                results.append(
                    _make_missing_metadata_result("V07", "exofop_disposition", missing=["tic_id"])
                )
                continue
            assert tic_id is not None
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
