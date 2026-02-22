"""Convenience API for generating LC-only vetting reports.

Composes MAST data fetching, multi-sector stitching, and report assembly
into a single function call.  Designed for MCP server integration and
interactive use.

Public API:
    generate_report       -- Download data, stitch, and build a report
    GenerateReportResult  -- Frozen result dataclass
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypedDict, overload

import numpy as np

from tess_vetter.api.pipeline import PipelineConfig, VettingPipeline
from tess_vetter.api.report_vet_reuse import (
    VetArtifactReuseSummary,
    build_report_with_vet_artifact,
    coerce_vetting_bundle,
)
from tess_vetter.api.stitch import SectorDiagnostics, stitch_lightcurve_data
from tess_vetter.api.types import (
    Candidate,
    Ephemeris,
    LightCurve,
    StellarParams,
    TPFStamp,
)
from tess_vetter.domain.detection import TransitCandidate
from tess_vetter.platform.catalogs.exofop_toi_table import fetch_exofop_toi_table
from tess_vetter.platform.io.mast_client import (
    DownloadProgress,
    LightCurveNotFoundError,
    MASTClient,
)
from tess_vetter.report import (
    EnrichmentBlockData,
    ReportData,
    ReportEnrichmentData,
    build_report,
    render_html,
)
from tess_vetter.validation.base import get_in_transit_mask
from tess_vetter.validation.register_defaults import register_all_defaults
from tess_vetter.validation.registry import CheckRegistry, CheckTier
from tess_vetter.validation.result_schema import VettingBundleResult

logger = logging.getLogger(__name__)

GENERATE_REPORT_RESULT_SCHEMA_VERSION = 1
GENERATE_REPORT_ENRICHMENT_SCHEMA_VERSION = "0.1.0"
GENERATE_REPORT_PLOT_DATA_KEY = "plot_data"


class GenerateReportJSONContract(TypedDict, total=False):
    """Stable report JSON boundary payload (plot_data removed)."""

    schema_version: str
    verdict: str
    verdict_source: str
    summary: dict[str, Any]
    custom_views: dict[str, Any]


class GenerateReportPlotDataContract(TypedDict, total=False):
    """Stable plot-data payload emitted separately from report_json."""

    full_lc: dict[str, Any]


def _to_optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _fallback_stellar_from_exofop(tic_id: int) -> StellarParams | None:
    try:
        rows = fetch_exofop_toi_table().entries_for_tic(int(tic_id))
    except Exception:
        return None
    if not rows:
        return None

    row = dict(rows[0])
    radius = _to_optional_finite_float(
        row.get("stellar_radius_r_sun") or row.get("stellar_radius") or row.get("radius")
    )
    mass = _to_optional_finite_float(
        row.get("stellar_mass_m_sun") or row.get("stellar_mass") or row.get("mass")
    )
    tmag = _to_optional_finite_float(row.get("tess_mag") or row.get("tmag"))
    teff = _to_optional_finite_float(row.get("stellar_eff_temp_k") or row.get("teff"))

    if radius is None and mass is None and tmag is None and teff is None:
        return None
    return StellarParams(radius=radius, mass=mass, tmag=tmag, teff=teff)


@dataclass(frozen=True)
class GenerateReportResult:
    """Result of a generate_report() call.

    Attributes:
        report: Structured report data packet.
        report_json: JSON-serializable summary/metadata payload (no top-level plot_data).
        plot_data_json: JSON-serializable plot payload emitted as a separate artifact.
        html: Rendered HTML string, or None if include_html was False.
        sectors_used: Sorted list of sectors that contributed data.
        stitch_diagnostics: Per-sector stitching diagnostics, or None
            for single-sector (no stitching performed).
    """

    report: ReportData
    report_json: dict[str, Any]
    plot_data_json: dict[str, Any]
    html: str | None
    sectors_used: list[int]
    stitch_diagnostics: list[SectorDiagnostics] | None
    vet_artifact_reuse: VetArtifactReuseSummary | None = None

    @property
    def schema_version(self) -> int:
        """Contract schema version for the GenerateReportResult wrapper."""
        return GENERATE_REPORT_RESULT_SCHEMA_VERSION


@dataclass(frozen=True)
class EnrichmentConfig:
    """Configuration for optional non-LC enrichment scaffolding."""

    include_pixel_diagnostics: bool = True
    include_catalog_context: bool = True
    include_followup_context: bool = True
    fail_open: bool = True
    network: bool = False
    max_network_seconds: float = 60.0
    per_request_timeout_seconds: float = 30.0
    max_concurrent_requests: int = 3
    fetch_tpf: bool = True
    tpf_sector_strategy: str = "best"  # best | all | requested
    sectors_for_tpf: list[int] | None = None
    max_pixel_points: int = 20_000
    max_catalog_rows: int = 200


def generate_report(
    tic_id: int,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    *,
    depth_ppm: float | None = None,
    stellar: StellarParams | None = None,
    toi: str | None = None,
    sectors: list[int] | None = None,
    flux_type: str = "pdcsap",
    mast_client: MASTClient | None = None,
    include_html: bool = False,
    include_v03: bool = False,
    bin_minutes: float = 30.0,
    max_lc_points: int = 50_000,
    max_phase_points: int = 10_000,
    include_additional_plots: bool = True,
    max_transit_windows: int = 24,
    max_points_per_window: int = 300,
    max_timing_points: int = 200,
    include_lc_robustness: bool = True,
    max_lc_robustness_epochs: int = 128,
    check_config: dict[str, dict[str, Any]] | None = None,
    pipeline_config: PipelineConfig | None = None,
    include_enrichment: bool = False,
    enrichment_config: EnrichmentConfig | None = None,
    custom_views: dict[str, Any] | None = None,
    progress_callback: Callable[[DownloadProgress], None] | None = None,
    vet_result: VettingBundleResult | dict[str, Any] | None = None,
) -> GenerateReportResult:
    """Generate a complete LC-only vetting report for a TESS candidate.

    Downloads light curve data from MAST, stitches multiple sectors,
    auto-fetches stellar parameters from TIC, runs vetting checks,
    and assembles a structured report.

    Args:
        tic_id: TESS Input Catalog identifier.
        period_days: Orbital period in days.
        t0_btjd: Reference transit epoch in BTJD.
        duration_hours: Transit duration in hours.
        depth_ppm: Transit depth in parts per million (optional).
        stellar: Stellar parameters (skips TIC lookup if provided).
        toi: TOI designation for display.
        sectors: Specific sectors to download (None for all available).
        flux_type: Flux column to use ("pdcsap" or "sap").
        mast_client: Injected MASTClient (created if None).
        include_html: If True, render standalone HTML.
        include_v03: If True, include V03 duration consistency check.
        bin_minutes: Phase-fold bin width in minutes.
        max_lc_points: Downsample full LC if longer than this.
        max_phase_points: Downsample phase-folded if longer than this.
        include_additional_plots: If True, include per-transit stack,
            odd/even phase, and secondary scan payloads.
        max_transit_windows: Max transit windows in per-transit plots.
        max_points_per_window: Max points per per-transit window.
        max_timing_points: Max per-epoch timing points.
        include_lc_robustness: If True, include LC robustness payloads.
        max_lc_robustness_epochs: Max epochs in LC robustness payload.
        check_config: Per-check config overrides.
        pipeline_config: Optional PipelineConfig forwarded to enrichment-tier
            VettingPipeline runs (catalog/pixel blocks).
        include_enrichment: If True, attach non-LC enrichment scaffold blocks.
        enrichment_config: Optional config for enrichment scaffolding.
        custom_views: Optional authored custom-view contract payload.
        progress_callback: Callback for download progress updates.

    Returns:
        GenerateReportResult with report, JSON, optional HTML,
        sectors used, and stitch diagnostics.

    Raises:
        LightCurveNotFoundError: If no sectors are available.
        ValueError: If flux_type is invalid.
    """
    # 1. Sanitize sectors: dedupe and sort
    if sectors is not None:
        sectors = sorted(set(sectors))

    # 2. Client
    client = mast_client or MASTClient()

    # 3. Download (flux_type forwarded)
    lightcurves = client.download_all_sectors(
        tic_id,
        flux_type=flux_type,
        sectors=sectors,
        progress_callback=progress_callback,
    )
    if not lightcurves:
        raise LightCurveNotFoundError(f"No sectors available for TIC {tic_id}")

    # 4. Stitch (or bypass for single sector)
    if len(lightcurves) == 1:
        stitched_lc_data = lightcurves[0]
        stitch_diag = None
    else:
        stitched_lc_data, stitched_obj = stitch_lightcurve_data(
            lightcurves, tic_id=tic_id
        )
        stitch_diag = stitched_obj.per_sector_diagnostics

    sectors_used = sorted({lc.sector for lc in lightcurves})

    # 5. Stellar (best-effort from TIC)
    if stellar is None:
        try:
            target = client.get_target_info(tic_id)
            stellar = target.stellar
        except Exception:
            logger.warning(
                "TIC stellar query failed for %d; proceeding without", tic_id
            )
            stellar = _fallback_stellar_from_exofop(tic_id)

    # 6. Build report (pure, network-free)
    lc = LightCurve.from_internal(stitched_lc_data)
    ephemeris = Ephemeris(
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
    )
    candidate = Candidate(ephemeris=ephemeris, depth_ppm=depth_ppm)

    vet_reuse: VetArtifactReuseSummary | None = None
    if vet_result is None:
        report = build_report(
            lc,
            candidate,
            stellar=stellar,
            tic_id=tic_id,
            toi=toi,
            include_v03=include_v03,
            bin_minutes=bin_minutes,
            max_lc_points=max_lc_points,
            max_phase_points=max_phase_points,
            include_additional_plots=include_additional_plots,
            max_transit_windows=max_transit_windows,
            max_points_per_window=max_points_per_window,
            max_timing_points=max_timing_points,
            include_lc_robustness=include_lc_robustness,
            max_lc_robustness_epochs=max_lc_robustness_epochs,
            check_config=check_config,
            custom_views=custom_views,
        )
    else:
        report, vet_reuse = build_report_with_vet_artifact(
            lc=lc,
            candidate=candidate,
            vet_bundle=coerce_vetting_bundle(vet_result),
            stellar=stellar,
            tic_id=tic_id,
            toi=toi,
            include_v03=include_v03,
            bin_minutes=bin_minutes,
            check_config=check_config,
            max_lc_points=max_lc_points,
            max_phase_points=max_phase_points,
            include_additional_plots=include_additional_plots,
            max_transit_windows=max_transit_windows,
            max_points_per_window=max_points_per_window,
            max_timing_points=max_timing_points,
            include_lc_robustness=include_lc_robustness,
            max_lc_robustness_epochs=max_lc_robustness_epochs,
            custom_views=custom_views,
        )

    if include_enrichment:
        cfg = enrichment_config or EnrichmentConfig()
        sector_times = {
            int(lc_data.sector): np.asarray(lc_data.time, dtype=np.float64)
            for lc_data in lightcurves
        }
        report.enrichment = _build_enrichment_data(
            lc_api=lc,
            candidate_api=candidate,
            tic_id=tic_id,
            sectors_used=sectors_used,
            mast_client=client,
            stellar=stellar,
            sector_times=sector_times,
            target=target if "target" in locals() else None,
            config=cfg,
            pipeline_config=pipeline_config,
        )

    full_report_json = report.to_json()
    plot_data_json: GenerateReportPlotDataContract = dict(
        full_report_json.get(GENERATE_REPORT_PLOT_DATA_KEY, {})
    )
    report_json: GenerateReportJSONContract = dict(full_report_json)
    report_json.pop(GENERATE_REPORT_PLOT_DATA_KEY, None)

    # 7. Optional HTML
    html = render_html(report) if include_html else None

    return GenerateReportResult(
        report=report,
        report_json=report_json,
        plot_data_json=plot_data_json,
        html=html,
        sectors_used=sectors_used,
        stitch_diagnostics=stitch_diag,
        vet_artifact_reuse=vet_reuse,
    )


def _skipped_enrichment_block(reason_flag: str) -> EnrichmentBlockData:
    """Build a deterministic skipped enrichment block scaffold."""
    return EnrichmentBlockData(
        status="skipped",
        flags=[reason_flag],
        quality={"is_degraded": False},
        checks={},
        provenance={"scaffold": True, "reason": reason_flag},
        payload={},
    )


def _build_enrichment_data(
    *,
    lc_api: LightCurve,
    candidate_api: Candidate,
    tic_id: int,
    sectors_used: list[int],
    mast_client: MASTClient,
    stellar: StellarParams | None,
    sector_times: dict[int, np.ndarray],
    target: Any | None,
    config: EnrichmentConfig,
    pipeline_config: PipelineConfig | None = None,
) -> ReportEnrichmentData:
    """Build enrichment blocks using existing pipeline tiers."""
    started = time.monotonic()
    budget_s = max(float(config.max_network_seconds), 0.0)
    max_workers = max(int(config.max_concurrent_requests), 1)

    def budget_exhausted() -> bool:
        return budget_s > 0 and (time.monotonic() - started) >= budget_s

    blocks: dict[str, EnrichmentBlockData | None] = {
        "catalog_context": None,
        "pixel_diagnostics": None,
        "followup_context": None,
    }

    def _error_block(exc: Exception) -> EnrichmentBlockData:
        is_timeout = isinstance(exc, TimeoutError)
        return EnrichmentBlockData(
            status="error",
            flags=["ENRICHMENT_TIMEOUT" if is_timeout else "ENRICHMENT_BLOCK_ERROR"],
            quality={"is_degraded": True},
            checks={},
            provenance={
                "error": type(exc).__name__,
                "message": str(exc),
                "timeout_seconds": float(config.per_request_timeout_seconds),
            },
            payload={},
        )

    requested_blocks: list[tuple[str, Any, dict[str, Any]]] = []
    if config.include_catalog_context:
        requested_blocks.append(
            (
                "catalog_context",
                _run_catalog_context,
                {
                    "lc_api": lc_api,
                    "candidate_api": candidate_api,
                    "tic_id": tic_id,
                    "stellar": stellar,
                    "target": target,
                    "network": config.network,
                    "max_catalog_rows": int(config.max_catalog_rows),
                    "config": config,
                    "pipeline_config": pipeline_config,
                },
            )
        )
    if config.include_pixel_diagnostics:
        requested_blocks.append(
            (
                "pixel_diagnostics",
                _run_pixel_diagnostics,
                {
                    "lc_api": lc_api,
                    "candidate_api": candidate_api,
                    "tic_id": tic_id,
                    "sectors_used": sectors_used,
                    "sector_times": sector_times,
                    "stellar": stellar,
                    "target": target,
                    "mast_client": mast_client,
                    "config": config,
                    "pipeline_config": pipeline_config,
                },
            )
        )

    if requested_blocks:
        if budget_exhausted():
            for block_name, _, _ in requested_blocks:
                blocks[block_name] = _skipped_enrichment_block("NETWORK_BUDGET_EXHAUSTED")
        else:
            executor = ThreadPoolExecutor(max_workers=max_workers)
            futures: dict[str, Future[EnrichmentBlockData]] = {}
            try:
                for block_name, block_func, kwargs in requested_blocks:
                    futures[block_name] = executor.submit(
                        _invoke_block_with_timeout,
                        timeout_seconds=config.per_request_timeout_seconds,
                        func=block_func,
                        **kwargs,
                    )
                if budget_s > 0:
                    remaining_s = max(0.0, budget_s - (time.monotonic() - started))
                    _, pending = wait(list(futures.values()), timeout=remaining_s)
                else:
                    pending = set()

                pending_names: set[str] = set()
                for name, future in futures.items():
                    if future in pending:
                        future.cancel()
                        pending_names.add(name)

                for name in pending_names:
                    blocks[name] = _skipped_enrichment_block("NETWORK_BUDGET_EXHAUSTED")

                for name, future in futures.items():
                    if name in pending_names:
                        continue
                    try:
                        blocks[name] = future.result()
                    except Exception as exc:
                        if not config.fail_open:
                            raise
                        blocks[name] = _error_block(exc)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

    if config.include_followup_context:
        if budget_exhausted():
            blocks["followup_context"] = _skipped_enrichment_block("NETWORK_BUDGET_EXHAUSTED")
        else:
            blocks["followup_context"] = _build_followup_context(
                tic_id=tic_id,
                sectors_used=sectors_used,
                target=target,
                config=config,
            )

    return ReportEnrichmentData(
        version=GENERATE_REPORT_ENRICHMENT_SCHEMA_VERSION,
        pixel_diagnostics=blocks["pixel_diagnostics"],
        catalog_context=blocks["catalog_context"],
        followup_context=blocks["followup_context"],
    )


def _tier_bundle(
    *,
    tier: CheckTier,
    lc_api: LightCurve,
    candidate_api: Candidate,
    tic_id: int,
    stellar: StellarParams | None,
    target: Any | None,
    network: bool,
    tpf: TPFStamp | None = None,
    check_timeout_seconds: float | None = None,
    check_ids_override: list[str] | None = None,
    pipeline_config: PipelineConfig | None = None,
) -> VettingBundleResult:
    registry = CheckRegistry()
    register_all_defaults(registry)
    check_ids = (
        [str(cid) for cid in check_ids_override]
        if check_ids_override is not None
        else [c.id for c in registry.list_by_tier(tier)]
    )
    effective_pipeline_config = _resolve_effective_pipeline_config(
        pipeline_config=pipeline_config,
        check_timeout_seconds=check_timeout_seconds,
    )
    pipeline = VettingPipeline(
        checks=check_ids,
        registry=registry,
        config=effective_pipeline_config,
    )
    lc_internal = lc_api.to_internal(tic_id=tic_id)
    candidate_internal = TransitCandidate(
        period=candidate_api.ephemeris.period_days,
        t0=candidate_api.ephemeris.t0_btjd,
        duration_hours=candidate_api.ephemeris.duration_hours,
        depth=candidate_api.depth or 0.001,
        snr=0.0,
    )
    ra = getattr(target, "ra", None) if target is not None else None
    dec = getattr(target, "dec", None) if target is not None else None
    bundle = pipeline.run(
        lc_internal,
        candidate_internal,
        stellar=stellar if stellar is not None else getattr(target, "stellar", None),
        tpf=tpf,
        network=network,
        ra_deg=ra,
        dec_deg=dec,
        tic_id=tic_id,
    )
    bundle.provenance["pipeline_config"] = asdict(effective_pipeline_config)
    return bundle


def _resolve_effective_pipeline_config(
    *,
    pipeline_config: PipelineConfig | None,
    check_timeout_seconds: float | None,
) -> PipelineConfig:
    """Merge optional user config with enrichment timeout defaults."""
    base = pipeline_config or PipelineConfig()
    extra_params = dict(base.extra_params)
    if "request_timeout_seconds" not in extra_params:
        extra_params["request_timeout_seconds"] = check_timeout_seconds

    timeout_seconds = (
        base.timeout_seconds
        if base.timeout_seconds is not None
        else check_timeout_seconds
    )
    return PipelineConfig(
        timeout_seconds=timeout_seconds,
        random_seed=base.random_seed,
        emit_warnings=base.emit_warnings,
        fail_fast=base.fail_fast,
        extra_params=extra_params,
    )


def _invoke_block_with_timeout(
    *,
    timeout_seconds: float,
    func: Any,
    **kwargs: Any,
) -> EnrichmentBlockData:
    """Run one enrichment block with timeout semantics."""
    timeout = max(float(timeout_seconds), 0.0)
    if timeout <= 0:
        return func(**kwargs)
    result: EnrichmentBlockData | None = None
    error: BaseException | None = None

    def _runner() -> None:
        nonlocal result, error
        try:
            result = func(**kwargs)
        except BaseException as exc:  # pragma: no cover - re-raised by caller
            error = exc

    # Daemon thread ensures timed-out work won't block caller wall time.
    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join(timeout=timeout)
    if worker.is_alive():
        raise TimeoutError(f"Block timed out after {timeout:.2f}s")
    if error is not None:
        raise error
    if result is None:
        raise RuntimeError("Enrichment block completed without result")
    return result


def _block_from_bundle(
    *,
    bundle: VettingBundleResult,
    block_name: str,
    flags: list[str] | None = None,
    quality_extra: dict[str, float | int | str | bool | None] | None = None,
    payload: dict[str, Any] | None = None,
    provenance_extra: dict[str, Any] | None = None,
) -> EnrichmentBlockData:
    results = list(bundle.results)
    n_ok = sum(1 for r in results if r.status == "ok")
    n_err = sum(1 for r in results if r.status == "error")
    n_skip = sum(1 for r in results if r.status == "skipped")
    if n_err > 0:
        status = "error"
    elif n_ok > 0:
        status = "ok"
    else:
        status = "skipped"
    is_degraded = status == "ok" and (n_skip > 0 or len(bundle.warnings) > 0)
    quality: dict[str, float | int | str | bool | None] = {
        "is_degraded": is_degraded,
        "n_checks": len(results),
        "n_ok": n_ok,
        "n_error": n_err,
        "n_skipped": n_skip,
    }
    if quality_extra:
        quality.update(quality_extra)
    checks = {r.id: r.model_dump() for r in results}
    provenance: dict[str, Any] = {
        "block": block_name,
        "pipeline": dict(bundle.provenance),
        "warnings": list(bundle.warnings),
    }
    if provenance_extra:
        provenance.update(provenance_extra)
    return EnrichmentBlockData(
        status=status,
        flags=list(flags or []),
        quality=quality,
        checks=checks,
        provenance=provenance,
        payload=payload or {},
    )


def _run_catalog_context(
    *,
    lc_api: LightCurve,
    candidate_api: Candidate,
    tic_id: int,
    stellar: StellarParams | None,
    target: Any | None,
    network: bool,
    max_catalog_rows: int,
    config: EnrichmentConfig,
    pipeline_config: PipelineConfig | None = None,
) -> EnrichmentBlockData:
    bundle = _tier_bundle(
        tier=CheckTier.CATALOG,
        lc_api=lc_api,
        candidate_api=candidate_api,
        tic_id=tic_id,
        stellar=stellar,
        target=target,
        network=network,
        check_timeout_seconds=float(config.per_request_timeout_seconds),
        # V06 (Vizier nearby-EB query) is intentionally excluded from inline
        # report enrichment due endpoint latency flakiness. Keep V07 only.
        check_ids_override=["V07"],
        pipeline_config=pipeline_config,
    )
    check_rows = []
    for result in bundle.results:
        check_rows.append(
            {
                "id": result.id,
                "status": result.status,
                "flags": list(result.flags),
                "metrics": dict(result.metrics),
            }
        )
    budget_applied = len(check_rows) > max_catalog_rows
    if budget_applied:
        check_rows = check_rows[:max_catalog_rows]
    payload = {
        "checks_summary": check_rows,
        "network_enabled": network,
    }
    return _block_from_bundle(
        bundle=bundle,
        block_name="catalog_context",
        quality_extra={"is_degraded": any(r.status == "skipped" for r in bundle.results)},
        payload=payload,
        provenance_extra={
            "tic_id": tic_id,
            "budget": {
                "max_catalog_rows": max_catalog_rows,
                "budget_applied": budget_applied,
                "rows_before": len(bundle.results),
                "rows_after": len(check_rows),
            },
            "network_config": {
                "network": config.network,
                "per_request_timeout_seconds": config.per_request_timeout_seconds,
                "max_network_seconds": config.max_network_seconds,
                "max_concurrent_requests": config.max_concurrent_requests,
            },
            "coordinates_available": bool(target is not None and getattr(target, "ra", None) is not None and getattr(target, "dec", None) is not None),
        },
    )


@overload
def _select_tpf_sectors(
    *,
    strategy: str,
    sectors_used: list[int],
    requested: list[int] | None,
    lc_api: LightCurve,
    candidate_api: Candidate,
    sector_times: dict[int, np.ndarray] | None = None,
    return_scores: Literal[True],
) -> tuple[list[int], dict[int, float]]: ...


@overload
def _select_tpf_sectors(
    *,
    strategy: str,
    sectors_used: list[int],
    requested: list[int] | None,
    lc_api: LightCurve,
    candidate_api: Candidate,
    sector_times: dict[int, np.ndarray] | None = None,
    return_scores: Literal[False] = False,
) -> list[int]: ...


def _select_tpf_sectors(
    *,
    strategy: str,
    sectors_used: list[int],
    requested: list[int] | None,
    lc_api: LightCurve,
    candidate_api: Candidate,
    sector_times: dict[int, np.ndarray] | None = None,
    return_scores: bool = False,
) -> list[int] | tuple[list[int], dict[int, float]]:
    if strategy == "requested":
        if requested:
            req = sorted({int(s) for s in requested})
            selected = [s for s in req if s in set(sectors_used)]
            return (selected, {}) if return_scores else selected
        return ([], {}) if return_scores else []
    if strategy == "all":
        selected = sorted(set(sectors_used))
        return (selected, {}) if return_scores else selected
    # best
    if len(sectors_used) == 0:
        return ([], {}) if return_scores else []
    best_sector = int(min(sectors_used))
    best_score = -1.0
    scores: dict[int, float] = {}
    for sector in sorted(set(sectors_used)):
        if sector <= 0:
            continue
        # Score sector by in-transit coverage fraction from that sector's samples.
        # Fall back to stitched LC samples if per-sector arrays are unavailable.
        if sector_times is not None and sector in sector_times:
            time_arr = np.asarray(sector_times[sector], dtype=np.float64)
        else:
            time_arr = np.asarray(lc_api.time, dtype=np.float64)
        valid = np.isfinite(time_arr)
        if np.any(valid):
            in_mask = get_in_transit_mask(
                time_arr[valid],
                candidate_api.ephemeris.period_days,
                candidate_api.ephemeris.t0_btjd,
                candidate_api.ephemeris.duration_hours,
            )
            score = float(np.mean(in_mask)) if len(in_mask) > 0 else 0.0
        else:
            score = 0.0
        scores[int(sector)] = score
        if score > best_score:
            best_score = score
            best_sector = sector
    selected = [best_sector]
    if return_scores:
        return selected, scores
    return selected


def _run_pixel_diagnostics(
    *,
    lc_api: LightCurve,
    candidate_api: Candidate,
    tic_id: int,
    sectors_used: list[int],
    sector_times: dict[int, np.ndarray],
    stellar: StellarParams | None,
    target: Any | None,
    mast_client: MASTClient,
    config: EnrichmentConfig,
    pipeline_config: PipelineConfig | None = None,
) -> EnrichmentBlockData:
    if not config.fetch_tpf:
        return _skipped_enrichment_block("TPF_FETCH_DISABLED")
    select_result = _select_tpf_sectors(
        strategy=config.tpf_sector_strategy,
        sectors_used=sectors_used,
        requested=config.sectors_for_tpf,
        lc_api=lc_api,
        candidate_api=candidate_api,
        sector_times=sector_times,
        return_scores=True,
    )
    sectors_for_tpf, sector_scores = select_result
    if len(sectors_for_tpf) == 0:
        return _skipped_enrichment_block("NO_TPF_SECTOR_SELECTED")

    tpf_obj: TPFStamp | None = None
    selected_sector: int | None = None
    acquisition_mode = "none"
    last_exc: Exception | None = None
    for sector in sectors_for_tpf:
        try:
            time_arr, flux_cube, flux_err, wcs, aperture_mask, quality = mast_client.download_tpf_cached(
                tic_id=tic_id, sector=sector
            )
            acquisition_mode = "cache"
        except Exception as exc_cached:
            last_exc = exc_cached
            if not config.network:
                continue
            try:
                time_arr, flux_cube, flux_err, wcs, aperture_mask, quality = mast_client.download_tpf(
                    tic_id=tic_id, sector=sector
                )
                acquisition_mode = "network"
            except Exception as exc:
                last_exc = exc
                continue

        tpf_obj = TPFStamp(
            time=np.asarray(time_arr, dtype=np.float64),
            flux=np.asarray(flux_cube, dtype=np.float64),
            flux_err=np.asarray(flux_err, dtype=np.float64) if flux_err is not None else None,
            wcs=wcs,
            aperture_mask=np.asarray(aperture_mask) if aperture_mask is not None else None,
            quality=np.asarray(quality, dtype=np.int32) if quality is not None else None,
        )
        selected_sector = int(sector)
        break

    if tpf_obj is None:
        reason = "TPF_UNAVAILABLE_OFFLINE" if not config.network else "TPF_DOWNLOAD_FAILED"
        block = _skipped_enrichment_block(reason)
        block.provenance.update(
            {
                "requested_sectors": sectors_for_tpf,
                "last_error": str(last_exc) if last_exc is not None else None,
            }
        )
        return block

    max_points = int(config.max_pixel_points)
    original_shape = tuple(int(v) for v in tpf_obj.shape)
    n_points = int(np.prod(original_shape))
    downsample_applied = False
    downsample_stride = 1
    if n_points > max_points:
        n_frames = int(tpf_obj.shape[0])
        pixels_per_frame = int(np.prod(tpf_obj.shape[1:])) if len(tpf_obj.shape) >= 2 else 1
        max_frames = max_points // max(pixels_per_frame, 1)
        min_frames_required = 64
        if max_frames >= min_frames_required and n_frames > 0:
            downsample_stride = max(1, (n_frames + max_frames - 1) // max_frames)
            if downsample_stride > 1:
                tpf_obj = TPFStamp(
                    time=tpf_obj.time[::downsample_stride],
                    flux=tpf_obj.flux[::downsample_stride, :, :],
                    flux_err=tpf_obj.flux_err[::downsample_stride, :, :]
                    if tpf_obj.flux_err is not None
                    else None,
                    wcs=tpf_obj.wcs,
                    aperture_mask=tpf_obj.aperture_mask,
                    quality=tpf_obj.quality[::downsample_stride]
                    if tpf_obj.quality is not None
                    else None,
                )
                downsample_applied = True
        n_points = int(np.prod(tpf_obj.shape))
        if n_points > max_points:
            block = _skipped_enrichment_block("PIXEL_POINT_BUDGET_EXCEEDED")
            block.quality["is_degraded"] = True
            block.provenance.update(
                {
                    "block": "pixel_diagnostics",
                    "tic_id": tic_id,
                    "tpf_sector_strategy": config.tpf_sector_strategy,
                    "sector_scores": {str(k): float(v) for k, v in sector_scores.items()},
                    "budget": {
                        "max_pixel_points": max_points,
                        "points_estimate": int(np.prod(original_shape)),
                        "points_after_downsample": n_points,
                        "downsample_applied": downsample_applied,
                        "downsample_stride": int(downsample_stride),
                        "budget_applied": True,
                    },
                }
            )
            block.payload.update(
                {
                    "selected_sector": selected_sector,
                    "selected_from": sectors_for_tpf,
                    "tpf_shape": [int(v) for v in original_shape],
                    "tpf_shape_after_downsample": [int(v) for v in tpf_obj.shape],
                    "acquisition_mode": acquisition_mode,
                    "aperture_pixels": int(np.sum(np.asarray(tpf_obj.aperture_mask) > 0))
                    if tpf_obj.aperture_mask is not None
                    else None,
                }
            )
            return block

    bundle = _tier_bundle(
        tier=CheckTier.PIXEL,
        lc_api=lc_api,
        candidate_api=candidate_api,
        tic_id=tic_id,
        stellar=stellar,
        target=target,
        network=config.network,
        tpf=tpf_obj,
        check_timeout_seconds=float(config.per_request_timeout_seconds),
        pipeline_config=pipeline_config,
    )

    payload = {
        "selected_sector": selected_sector,
        "selected_from": sectors_for_tpf,
        "tpf_shape": [int(v) for v in original_shape],
        "tpf_shape_after_downsample": [int(v) for v in tpf_obj.shape],
        "downsample_applied": downsample_applied,
        "downsample_stride": int(downsample_stride),
        "acquisition_mode": acquisition_mode,
        "aperture_pixels": int(np.sum(np.asarray(tpf_obj.aperture_mask) > 0))
        if tpf_obj.aperture_mask is not None
        else None,
    }
    return _block_from_bundle(
        bundle=bundle,
        block_name="pixel_diagnostics",
        payload=payload,
        provenance_extra={
            "tic_id": tic_id,
            "tpf_sector_strategy": config.tpf_sector_strategy,
            "sector_scores": {str(k): float(v) for k, v in sector_scores.items()},
            "budget": {
                "max_pixel_points": int(config.max_pixel_points),
                "points_estimate": n_points,
                "points_original": int(np.prod(original_shape)),
                "downsample_applied": downsample_applied,
                "downsample_stride": int(downsample_stride),
                "budget_applied": downsample_applied or (n_points > int(config.max_pixel_points)),
            },
        },
    )


def _build_followup_context(
    *,
    tic_id: int,
    sectors_used: list[int],
    target: Any | None,
    config: EnrichmentConfig,
) -> EnrichmentBlockData:
    has_target_meta = target is not None
    payload = {
        "tic_id": tic_id,
        "sectors_used": list(sectors_used),
        "target_metadata_available": has_target_meta,
        "references": [],
    }
    if has_target_meta:
        payload["target"] = {
            "ra": getattr(target, "ra", None),
            "dec": getattr(target, "dec", None),
            "gaia_dr3_id": getattr(target, "gaia_dr3_id", None),
            "toi_id": getattr(target, "toi_id", None),
        }
    status = "ok" if has_target_meta else "skipped"
    flags = [] if has_target_meta else ["NO_TARGET_METADATA"]
    return EnrichmentBlockData(
        status=status,
        flags=flags,
        quality={"is_degraded": not has_target_meta},
        checks={},
        provenance={
            "block": "followup_context",
            "scaffold": False,
            "network_enabled": config.network,
        },
        payload=payload,
    )


__all__ = [
    "GENERATE_REPORT_ENRICHMENT_SCHEMA_VERSION",
    "GENERATE_REPORT_PLOT_DATA_KEY",
    "GENERATE_REPORT_RESULT_SCHEMA_VERSION",
    "EnrichmentConfig",
    "GenerateReportJSONContract",
    "GenerateReportPlotDataContract",
    "GenerateReportResult",
    "generate_report",
]
