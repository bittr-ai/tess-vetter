"""Unified vetting orchestrator for the public API.

This module provides `vet_candidate`, the main entry point for running
a complete vetting pipeline on a transit candidate.

For batch processing, custom check selection, or pipeline extensibility,
use `VettingPipeline` directly from `bittr_tess_vetter.api.pipeline`.

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

import warnings
from typing import Any

from bittr_tess_vetter.api.references import (
    COUGHLIN_2016,
    GUERRERO_2021,
    THOMPSON_2018,
    cite,
    cites,
)
from bittr_tess_vetter.api.types import (
    Candidate,
    LightCurve,
    StellarParams,
    TPFStamp,
)
from bittr_tess_vetter.validation.result_schema import VettingBundleResult

# Module-level references for programmatic access (generated from central registry)
REFERENCES = [ref.to_dict() for ref in [COUGHLIN_2016, THOMPSON_2018, GUERRERO_2021]]


@cites(
    cite(COUGHLIN_2016, "tiered Robovetter methodology"),
    cite(THOMPSON_2018, "DR25 vetting pipeline architecture"),
    cite(GUERRERO_2021, "TESS TOI vetting procedures context"),
)
def vet_candidate(
    lc: LightCurve,
    candidate: Candidate,
    *,
    stellar: StellarParams | None = None,
    tpf: TPFStamp | None = None,
    network: bool = False,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    tic_id: int | None = None,
    checks: list[str] | None = None,
    context: dict[str, Any] | None = None,
) -> VettingBundleResult:
    """Run vetting checks on a transit candidate.

    This is a convenience wrapper around VettingPipeline for single-candidate
    vetting. For batch processing or custom check selection, use VettingPipeline
    directly.

    Args:
        lc: Light curve data.
        candidate: Transit candidate with ephemeris.
        stellar: Optional stellar parameters.
        tpf: Optional TPF data for pixel-level checks.
        network: Whether to allow network access for catalog checks.
        ra_deg: Right ascension in degrees.
        dec_deg: Declination in degrees.
        tic_id: TIC identifier.
        checks: Optional list of check IDs to run. If None, runs all registered.
        context: Additional context for checks.

    Returns:
        VettingBundleResult with all check results.

    Example:
        >>> from bittr_tess_vetter.api import (
        ...     LightCurve, Ephemeris, Candidate, vet_candidate
        ... )
        >>> import numpy as np
        >>> lc = LightCurve(
        ...     time=np.linspace(0, 27, 1000),
        ...     flux=np.ones(1000),
        ...     flux_err=np.ones(1000) * 0.001,
        ... )
        >>> eph = Ephemeris(period_days=3.5, t0_btjd=1.0, duration_hours=2.0)
        >>> candidate = Candidate(ephemeris=eph)
        >>> result = vet_candidate(lc, candidate, network=False)
        >>> print(f"Ran {len(result.results)} checks")

    Novelty: standard

    References:
        [1] Coughlin et al. 2016, ApJS 224, 12 (2016ApJS..224...12C)
            Kepler Robovetter tiered architecture
        [2] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T)
            Section 3: Check execution ordering and aggregation
    """
    from bittr_tess_vetter.api.pipeline import VettingPipeline
    from bittr_tess_vetter.domain.detection import TransitCandidate
    from bittr_tess_vetter.validation.register_defaults import register_all_defaults
    from bittr_tess_vetter.validation.registry import CheckRegistry

    # Create registry with default checks
    registry = CheckRegistry()
    register_all_defaults(registry)

    # Convert public API types to internal types
    lc_internal = lc.to_internal(tic_id=tic_id or 0)

    # Convert Candidate to TransitCandidate
    # Use depth if provided, otherwise default to a small depth for metrics-only mode
    depth = candidate.depth if candidate.depth is not None else 0.001
    # Use a default SNR of 0.0 - this is metrics-only mode
    snr = 0.0

    candidate_internal = TransitCandidate(
        period=candidate.ephemeris.period_days,
        t0=candidate.ephemeris.t0_btjd,
        duration_hours=candidate.ephemeris.duration_hours,
        depth=depth,
        snr=snr,
    )

    # Create and run pipeline
    pipeline = VettingPipeline(checks=checks, registry=registry)
    return pipeline.run(
        lc_internal,
        candidate_internal,
        stellar=stellar,
        tpf=tpf,
        network=network,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        tic_id=tic_id,
        context=context,
    )


# Legacy wrapper for backward compatibility with 'enabled' parameter
def _vet_candidate_legacy(
    lc: LightCurve,
    candidate: Candidate,
    *,
    stellar: StellarParams | None = None,
    tpf: TPFStamp | None = None,
    enabled: set[str] | None = None,
    config: dict[str, dict[str, Any]] | None = None,
    policy_mode: str = "metrics_only",
    network: bool = False,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    tic_id: int | None = None,
    context: dict[str, Any] | None = None,
) -> VettingBundleResult:
    """Legacy vet_candidate with deprecated parameters."""
    if policy_mode != "metrics_only":
        warnings.warn(
            "`policy_mode` is deprecated and ignored; bittr-tess-vetter always returns metrics-only "
            "results. Move interpretation/policy decisions to astro-arc-tess validation (tess-validate).",
            category=FutureWarning,
            stacklevel=2,
        )
    if config is not None:
        warnings.warn(
            "`config` parameter is deprecated. Use VettingPipeline with PipelineConfig for advanced "
            "configuration.",
            category=FutureWarning,
            stacklevel=2,
        )

    # Convert enabled set to checks list
    checks_list = list(enabled) if enabled is not None else None

    return vet_candidate(
        lc,
        candidate,
        stellar=stellar,
        tpf=tpf,
        network=network,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        tic_id=tic_id,
        checks=checks_list,
        context=context,
    )
