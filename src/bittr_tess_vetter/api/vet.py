"""Unified vetting orchestrator for the public API.

This module provides `vet_candidate`, the main entry point for running
a complete vetting pipeline on a transit candidate.

Tiers:
- LC-only (V01-V05): Always run, require only light curve
- Catalog (V06-V07): Run if network=True and coordinates/TIC ID provided
- Pixel (V08-V10): Run if TPF data provided
- Exovetter (V11-V12): Run if exovetter available

Novelty: standard (orchestration of established vetting checks)

References:
    [1] Coughlin et al. 2016, ApJS 224, 12 (2016ApJS..224...12C)
        Kepler DR24 Robovetter methodology
    [2] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T)
        Kepler DR25 vetting pipeline architecture
    [3] Guerrero et al. 2021, ApJS 254, 39 (2021ApJS..254...39G)
        TESS TOI catalog vetting procedures
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

from bittr_tess_vetter.api.references import (
    COUGHLIN_2016,
    GUERRERO_2021,
    THOMPSON_2018,
)
from bittr_tess_vetter.api.types import (
    Candidate,
    CheckResult,
    LightCurve,
    StellarParams,
    TPFStamp,
    VettingBundleResult,
)

# Module-level references for programmatic access (generated from central registry)
REFERENCES = [ref.to_dict() for ref in [COUGHLIN_2016, THOMPSON_2018, GUERRERO_2021]]


# Default check sets per tier
_LC_ONLY_CHECKS = {"V01", "V02", "V03", "V04", "V05"}
_CATALOG_CHECKS = {"V06", "V07"}
_PIXEL_CHECKS = {"V08", "V09", "V10"}
_EXOVETTER_CHECKS = {"V11", "V12"}
_ALL_CHECKS = _LC_ONLY_CHECKS | _CATALOG_CHECKS | _PIXEL_CHECKS | _EXOVETTER_CHECKS


def _get_package_version() -> str:
    """Get bittr-tess-vetter package version for provenance."""
    try:
        return importlib.metadata.version("bittr-tess-vetter")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def vet_candidate(
    lc: LightCurve,
    candidate: Candidate,
    *,
    stellar: StellarParams | None = None,
    tpf: TPFStamp | None = None,
    enabled: set[str] | None = None,
    config: dict[str, dict[str, Any]] | None = None,
    network: bool = False,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    tic_id: int | None = None,
    context: dict[str, Any] | None = None,
) -> VettingBundleResult:
    """Run a complete vetting pipeline on a transit candidate.

    This is the main entry point for researcher-facing vetting. It runs
    checks in tiers based on available data:

    Tier 1 (LC-only, V01-V05): Always run
    Tier 2 (Catalog, V06-V07): Run if network=True and metadata available
    Tier 3 (Pixel, V08-V10): Run if tpf provided
    Tier 4 (Exovetter, V11-V12): Run if exovetter package available

    Args:
        lc: Light curve data
        candidate: Transit candidate with ephemeris and optional depth
        stellar: Stellar parameters (optional, improves V03)
        tpf: Target Pixel File data (optional, enables V08-V10)
        enabled: Set of check IDs to run. If None, runs default tiered set.
            Pass specific IDs like {"V01", "V03", "V08"} to filter.
        config: Per-check configuration dict. Keys are check IDs (e.g., "V01"),
            values are config dicts passed to individual checks.
        network: If True and metadata available, run catalog checks (V06-V07)
        ra_deg: Right ascension in degrees (for V06)
        dec_deg: Declination in degrees (for V06)
        tic_id: TIC identifier (for V07)
        context: Additional context dict stored in provenance

    Returns:
        VettingBundleResult with all check results and provenance metadata

    Example:
        >>> from bittr_tess_vetter.api import (
        ...     LightCurve, Ephemeris, Candidate, vet_candidate
        ... )
        >>> lc = LightCurve(time=time, flux=flux)
        >>> eph = Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=2.5)
        >>> candidate = Candidate(ephemeris=eph, depth_ppm=500)
        >>> result = vet_candidate(lc, candidate)
        >>> print(f"Passed: {result.n_passed}/{len(result.results)}")

    Novelty: standard

    References:
        [1] Coughlin et al. 2016, ApJS 224, 12 (2016ApJS..224...12C)
            Kepler Robovetter tiered architecture
        [2] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T)
            Section 3: Check execution ordering and aggregation
    """
    config = config or {}
    results: list[CheckResult] = []
    warnings: list[str] = []

    # Determine which checks to run
    if enabled is None:
        # Default: LC-only always, others based on available data
        checks_to_run = set(_LC_ONLY_CHECKS)

        if network and (ra_deg is not None or tic_id is not None):
            checks_to_run |= _CATALOG_CHECKS

        if tpf is not None:
            checks_to_run |= _PIXEL_CHECKS

        # Always try exovetter checks (they handle missing dependency)
        checks_to_run |= _EXOVETTER_CHECKS
    else:
        checks_to_run = enabled & _ALL_CHECKS
        unknown = enabled - _ALL_CHECKS
        if unknown:
            warnings.append(f"Unknown check IDs ignored: {sorted(unknown)}")

    # Run LC-only checks (V01-V05)
    lc_checks_to_run = checks_to_run & _LC_ONLY_CHECKS
    if lc_checks_to_run:
        from bittr_tess_vetter.api.lc_only import vet_lc_only

        lc_results = vet_lc_only(
            lc,
            candidate.ephemeris,
            stellar=stellar,
            enabled=lc_checks_to_run,
        )
        results.extend(lc_results)

    # Run catalog checks (V06-V07) if enabled
    catalog_checks_to_run = checks_to_run & _CATALOG_CHECKS
    if catalog_checks_to_run:
        from bittr_tess_vetter.api.catalog import vet_catalog

        catalog_results = vet_catalog(
            candidate,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            tic_id=tic_id,
            network=network,
            enabled=catalog_checks_to_run,
            config={k: v for k, v in config.items() if k in _CATALOG_CHECKS},
        )
        results.extend(catalog_results)

    # Run pixel checks (V08-V10) if TPF provided
    pixel_checks_to_run = checks_to_run & _PIXEL_CHECKS
    if pixel_checks_to_run and tpf is not None:
        from bittr_tess_vetter.api.pixel import vet_pixel

        pixel_results = vet_pixel(
            tpf,
            candidate,
            enabled=pixel_checks_to_run,
            config={k: v for k, v in config.items() if k in _PIXEL_CHECKS},
        )
        results.extend(pixel_results)
    elif pixel_checks_to_run and tpf is None:
        warnings.append("Pixel checks requested but no TPF provided; skipping V08-V10")

    # Run exovetter checks (V11-V12)
    exovetter_checks_to_run = checks_to_run & _EXOVETTER_CHECKS
    if exovetter_checks_to_run:
        from bittr_tess_vetter.api.exovetter import vet_exovetter

        exovetter_results = vet_exovetter(
            lc,
            candidate,
            enabled=exovetter_checks_to_run,
            config={k: v for k, v in config.items() if k in _EXOVETTER_CHECKS},
        )
        results.extend(exovetter_results)

    # Build provenance metadata
    provenance: dict[str, Any] = {
        "package_version": _get_package_version(),
        "checks_requested": sorted(checks_to_run),
        "checks_executed": sorted(r.id for r in results),
        "config": config,
        "network_enabled": network,
        "tpf_provided": tpf is not None,
        "stellar_provided": stellar is not None,
    }
    if context:
        provenance["context"] = context

    return VettingBundleResult(
        results=results,
        provenance=provenance,
        warnings=warnings,
    )
