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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bittr_tess_vetter.api.stitch import SectorDiagnostics, stitch_lightcurve_data
from bittr_tess_vetter.api.types import (
    Candidate,
    Ephemeris,
    LightCurve,
    StellarParams,
)
from bittr_tess_vetter.platform.io.mast_client import (
    DownloadProgress,
    LightCurveNotFoundError,
    MASTClient,
)
from bittr_tess_vetter.report import ReportData, build_report, render_html

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerateReportResult:
    """Result of a generate_report() call.

    Attributes:
        report: Structured report data packet.
        report_json: JSON-serializable dict (always computed, cheap).
        html: Rendered HTML string, or None if include_html was False.
        sectors_used: Sorted list of sectors that contributed data.
        stitch_diagnostics: Per-sector stitching diagnostics, or None
            for single-sector (no stitching performed).
    """

    report: ReportData
    report_json: dict[str, Any]
    html: str | None
    sectors_used: list[int]
    stitch_diagnostics: list[SectorDiagnostics] | None


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
    check_config: dict[str, dict[str, Any]] | None = None,
    progress_callback: Callable[[DownloadProgress], None] | None = None,
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
        check_config: Per-check config overrides.
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

    # 6. Build report (pure, network-free)
    lc = LightCurve.from_internal(stitched_lc_data)
    ephemeris = Ephemeris(
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
    )
    candidate = Candidate(ephemeris=ephemeris, depth_ppm=depth_ppm)

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
        check_config=check_config,
    )

    # 7. Optional HTML
    html = render_html(report) if include_html else None

    return GenerateReportResult(
        report=report,
        report_json=report.to_json(),
        html=html,
        sectors_used=sectors_used,
        stitch_diagnostics=stitch_diag,
    )


__all__ = ["GenerateReportResult", "generate_report"]
