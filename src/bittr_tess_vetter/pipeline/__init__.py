"""Pipeline module for batch enrichment of TESS transit candidates.

This module provides the core pipeline functions for enriching worklists
of transit candidates with vetting features.

Key Functions
-------------
enrich_candidate : function
    Enrich a single candidate with vetting features.
enrich_worklist : function
    Batch-enrich a worklist of candidates to JSONL output.

Data Classes
------------
EnrichmentSummary : dataclass
    Summary statistics from a batch enrichment run.

Example
-------
>>> from bittr_tess_vetter.pipeline import (
...     enrich_candidate,
...     enrich_worklist,
...     EnrichmentSummary,
... )
>>> from bittr_tess_vetter.features import FeatureConfig
>>> config = FeatureConfig(bulk_mode=True)
>>> summary = enrich_worklist(
...     worklist_iter=iter(candidates),
...     output_path="enriched.jsonl",
...     config=config,
... )
>>> print(f"Processed {summary.processed} candidates")
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bittr_tess_vetter.domain.lightcurve import LightCurveData

from bittr_tess_vetter.features import (
    EnrichedRow,
    FeatureConfig,
    RawEvidencePacket,
    build_features,
)

logger = logging.getLogger(__name__)


def _pipeline_version() -> str:
    try:
        return metadata.version("bittr-tess-vetter")
    except Exception:
        return "unknown"


def _load_lightcurves_from_local(
    tic_id: int,
    local_data_path: str,
    *,
    requested_sectors: list[int] | None = None,
) -> tuple[list[LightCurveData], list[int]]:
    """Load light curves from local CSV files.

    Looks for files matching tic{tic_id}/sector*_pdcsap.csv in the local
    data directory.

    Args:
        tic_id: TIC identifier.
        local_data_path: Path to local data directory.

    Returns:
        Tuple of (list of LightCurveData, list of sector numbers).

    Raises:
        FileNotFoundError: If no light curve files found for the TIC.
    """
    from bittr_tess_vetter.api.datasets import load_local_dataset

    # Build path to TIC directory (support both ticXXX and tic_XXX formats)
    base_path = Path(local_data_path).expanduser().resolve()
    tic_dir = base_path / f"tic{tic_id}"
    if not tic_dir.exists():
        # Try alternate format
        tic_dir = base_path / f"tic_{tic_id}"
    if not tic_dir.exists():
        raise FileNotFoundError(f"No local data directory for TIC {tic_id} at {base_path}")

    # Use the existing load_local_dataset utility
    dataset = load_local_dataset(tic_dir)

    if not dataset.lc_by_sector:
        raise FileNotFoundError(f"No light curve CSV files found in {tic_dir}")

    # Convert API LightCurve objects to internal LightCurveData
    lightcurves: list[LightCurveData] = []
    sectors_loaded: list[int] = []

    sector_items = sorted(dataset.lc_by_sector.items())
    if requested_sectors is not None:
        requested = {int(s) for s in requested_sectors}
        sector_items = [(sector, lc) for (sector, lc) in sector_items if int(sector) in requested]

    for sector, lc_api in sector_items:
        # Infer cadence from time spacing
        if len(lc_api.time) > 1:
            dt = float(lc_api.time[1] - lc_api.time[0])
            cadence_seconds = dt * 86400.0  # days to seconds
        else:
            cadence_seconds = 120.0  # default to 2-min

        lc_data = lc_api.to_internal(
            tic_id=tic_id,
            sector=sector,
            cadence_seconds=cadence_seconds,
        )
        lightcurves.append(lc_data)
        sectors_loaded.append(sector)
        logger.debug(
            "Loaded local sector %d: %d points, cadence=%.0fs",
            sector,
            lc_data.n_points,
            lc_data.cadence_seconds,
        )

    return lightcurves, sectors_loaded


def make_candidate_key(tic_id: int, period_days: float, t0_btjd: float) -> str:
    """Generate a unique candidate key from ephemeris parameters.

    The candidate key is immutable and does not change based on T0 refinement.
    This ensures consistent identification across pipeline stages.

    Args:
        tic_id: TIC identifier for the target.
        period_days: Orbital period in days.
        t0_btjd: Transit epoch in BTJD.

    Returns:
        Candidate key in format "tic_id|period_days|t0_btjd".
    """
    return f"{tic_id}|{period_days}|{t0_btjd}"


@dataclass
class EnrichmentSummary:
    """Summary statistics from a batch enrichment run.

    Attributes:
        total_input: Total number of candidates in the input worklist.
        processed: Number of candidates successfully processed.
        skipped_resume: Number of candidates skipped due to resume mode.
        errors: Number of candidates that failed with errors.
        wall_time_seconds: Total wall-clock time for the enrichment run.
        error_class_counts: Counts of each error class encountered.
    """

    total_input: int
    processed: int
    skipped_resume: int
    errors: int
    wall_time_seconds: float
    error_class_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_input": self.total_input,
            "processed": self.processed,
            "skipped_resume": self.skipped_resume,
            "errors": self.errors,
            "wall_time_seconds": self.wall_time_seconds,
            "error_class_counts": self.error_class_counts,
        }


def enrich_candidate(
    tic_id: int,
    *,
    toi: str | None = None,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    depth_ppm: float | None,
    config: FeatureConfig,
    sectors: list[int] | None = None,
) -> tuple[RawEvidencePacket, EnrichedRow]:
    """Enrich a single transit candidate with vetting features.

    This function runs the full vetting pipeline on a single candidate
    and returns both the raw evidence packet and the extracted features.

    Args:
        tic_id: TIC identifier for the target.
        period_days: Orbital period in days.
        t0_btjd: Transit epoch in BTJD.
        duration_hours: Transit duration in hours.
        depth_ppm: Transit depth in parts per million (optional).
        config: Feature extraction configuration.

    Returns:
        Tuple of (RawEvidencePacket, EnrichedRow) containing the raw
        vetting evidence and extracted ML features.

    Note:
        The candidate_key is generated as f"{tic_id}|{period_days}|{t0_btjd}"
        and is immutable - it does not change based on T0 refinement.
    """
    import numpy as np

    from bittr_tess_vetter.api.aperture_family import compute_aperture_family_depth_curve
    from bittr_tess_vetter.api.ghost_features import compute_ghost_features
    from bittr_tess_vetter.api.io import LightCurveNotFoundError, MASTClient, MASTClientError
    from bittr_tess_vetter.api.localization import TransitParams, compute_localization_diagnostics
    from bittr_tess_vetter.api.evidence_contracts import compute_code_hash
    from bittr_tess_vetter.api.stellar_dilution import (
        HostHypothesis,
        compute_dilution_scenarios,
        compute_flux_fraction_from_mag_list,
        evaluate_physics_flags,
    )
    from bittr_tess_vetter.api.stitch import stitch_lightcurve_data
    from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve, TPFStamp
    from bittr_tess_vetter.api.vet import vet_candidate as run_vetting
    from bittr_tess_vetter.data_sources.sector_selection import select_sectors
    from bittr_tess_vetter.features import FEATURE_SCHEMA_VERSION
    from bittr_tess_vetter.features.evidence import make_skip_block
    from bittr_tess_vetter.pixel.tpf_fits import TPFFitsData, TPFFitsRef
    from bittr_tess_vetter.platform.catalogs.gaia_client import query_gaia_by_position_sync

    start_time_ms = time.perf_counter() * 1000.0
    pipeline_version = _pipeline_version()
    candidate_key = make_candidate_key(tic_id, period_days, t0_btjd)
    code_hash = compute_code_hash()

    def _dependency_versions() -> dict[str, str]:
        from importlib import metadata

        pkgs = [
            "numpy",
            "lightkurve",
            "astropy",
            "astroquery",
            "exovetter",
        ]
        out: dict[str, str] = {}
        for p in pkgs:
            try:
                out[p] = str(metadata.version(p))
            except Exception:
                continue
        return out

    def _make_mast_client() -> MASTClient:
        """Construct MASTClient with optional cache_dir (test-safe).

        Some tests monkeypatch MASTClient with a no-kwargs lambda; keep a
        backwards-compatible fallback when the injected constructor rejects
        keyword args.
        """
        if config.cache_dir is None:
            return MASTClient()
        try:
            return MASTClient(cache_dir=config.cache_dir)
        except TypeError:
            return MASTClient()

    # Helper to create error response
    def _make_error_response(
        error_class: str, error_msg: str, wall_ms: float
    ) -> tuple[RawEvidencePacket, EnrichedRow]:
        raw: RawEvidencePacket = {
            "target": {"tic_id": tic_id, "toi": toi},
            "ephemeris": {
                "period_days": period_days,
                "t0_btjd": t0_btjd,
                "duration_hours": duration_hours,
            },
            "depth_ppm": {"input_depth_ppm": depth_ppm},
            "check_results": [],
            "pixel_host_hypotheses": make_skip_block("pipeline_error"),
            "localization": make_skip_block("pipeline_error"),
            "sector_quality_report": make_skip_block("pipeline_error"),
            "candidate_evidence": make_skip_block("pipeline_error"),
            "provenance": {
                "pipeline_version": pipeline_version,
                "code_hash": code_hash,
                "dependency_versions": _dependency_versions(),
                "error_class": error_class,
                "error": error_msg,
            },
        }
        row: EnrichedRow = {
            "tic_id": tic_id,
            "toi": toi,
            "period_days": period_days,
            "t0_btjd": t0_btjd,
            "duration_hours": duration_hours,
            "depth_ppm": depth_ppm,
            "status": "ERROR",
            "error_class": error_class,
            "error": error_msg,
            "candidate_key": candidate_key,
            "pipeline_version": pipeline_version,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_config": dataclasses.asdict(config),
            "inputs_summary": {},
            "missing_feature_families": ["all"],
            "item_wall_ms": wall_ms,
        }
        return raw, row

    # Step 1 & 2: Load light curves (local or MAST)
    lightcurves: list[LightCurveData] = []
    sectors_loaded: list[int] = []
    selection_summary: dict[str, Any] | None = None
    sector_exptimes: dict[int, set[int]] = {}

    if config.no_download and not config.local_data_path and not config.cache_dir:
        wall_ms = time.perf_counter() * 1000.0 - start_time_ms
        return _make_error_response(
            "NoDownloadError",
            "no_download=True requires local_data_path or cache_dir",
            wall_ms,
        )

    if not config.network_ok and not config.local_data_path and not config.cache_dir:
        wall_ms = time.perf_counter() * 1000.0 - start_time_ms
        return _make_error_response(
            "OfflineNoLocalDataError",
            "network_ok=False requires local_data_path for offline enrichment",
            wall_ms,
        )

    if config.local_data_path:
        # Load from local files instead of MAST
        logger.info("Loading local light curves for TIC %d from %s", tic_id, config.local_data_path)
        try:
            lightcurves, sectors_loaded = _load_lightcurves_from_local(
                tic_id, config.local_data_path, requested_sectors=sectors
            )
            selection_summary = {
                "available_sectors": sectors_loaded,
                "selected_sectors": sectors_loaded,
                "excluded_sectors": {},
            }
            logger.info(
                "Loaded %d sectors from local data for TIC %d: %s",
                len(sectors_loaded),
                tic_id,
                sectors_loaded,
            )
        except FileNotFoundError as e:
            wall_ms = time.perf_counter() * 1000.0 - start_time_ms
            logger.warning("Local data not found for TIC %d: %s", tic_id, e)
            return _make_error_response("LocalDataNotFoundError", str(e), wall_ms)
    else:
        # Standard MAST-based loading (or cache-only when no_download+cache_dir)
        logger.info("Searching for light curves for TIC %d", tic_id)
        try:
            client = _make_mast_client()
            if config.no_download and config.cache_dir:
                search_results = client.search_lightcurve_cached(tic_id)
            else:
                search_results = client.search_lightcurve(tic_id)
        except MASTClientError as e:
            wall_ms = time.perf_counter() * 1000.0 - start_time_ms
            logger.warning("MAST search failed for TIC %d: %s", tic_id, e)
            return _make_error_response("MASTClientError", str(e), wall_ms)

        if not search_results:
            wall_ms = time.perf_counter() * 1000.0 - start_time_ms
            logger.warning("No light curves found for TIC %d", tic_id)
            return _make_error_response(
                "LightCurveNotFoundError",
                f"No light curves found for TIC {tic_id}",
                wall_ms,
            )

        # Determine available sectors and preferred cadence
        available_sectors = sorted({r.sector for r in search_results})
        logger.info(
            "Found %d sectors for TIC %d: %s", len(available_sectors), tic_id, available_sectors
        )

        selection = select_sectors(
            available_sectors=available_sectors,
            requested_sectors=sectors,
            allow_20s=config.allow_20s,
            search_results=search_results,
        )
        selection_summary = {
            "available_sectors": selection.available_sectors,
            "selected_sectors": selection.selected_sectors,
            "excluded_sectors": selection.excluded_sectors,
        }
        if not selection.selected_sectors:
            wall_ms = time.perf_counter() * 1000.0 - start_time_ms
            return _make_error_response(
                "NoSectorsSelectedError",
                "No sectors selected (requested sectors not available)",
                wall_ms,
            )

        # Per-sector cadence choice (for requested downloads)
        for r in search_results:
            try:
                sector_exptimes.setdefault(int(r.sector), set()).add(int(round(float(r.exptime))))
            except Exception:
                continue

        sector_to_exptime: dict[int, float] = {}
        for sector in selection.selected_sectors:
            exps = sector_exptimes.get(int(sector), set())
            if 120 in exps:
                sector_to_exptime[int(sector)] = 120.0
            elif 20 in exps and config.allow_20s:
                sector_to_exptime[int(sector)] = 20.0
            else:
                # Should be excluded already by cadence gating, but keep a guardrail.
                sector_to_exptime[int(sector)] = 120.0

        download_errors: list[str] = []

        for sector in selection.selected_sectors:
            requested_exptime = sector_to_exptime.get(int(sector), 120.0)
            try:
                if config.no_download and config.cache_dir:
                    lc_data = client.download_lightcurve_cached(
                        tic_id=int(tic_id),
                        sector=int(sector),
                        flux_type="pdcsap",
                        exptime=float(requested_exptime),
                    )
                else:
                    lc_data = client.download_lightcurve(
                        tic_id,
                        sector,
                        flux_type="pdcsap",
                        exptime=requested_exptime,
                    )
                lightcurves.append(lc_data)
                sectors_loaded.append(sector)
                logger.debug(
                    "Downloaded sector %d: %d points, cadence=%.0fs",
                    sector,
                    lc_data.n_points,
                    lc_data.cadence_seconds,
                )
            except LightCurveNotFoundError as e:
                # Try fallback cadence if preferred failed
                if requested_exptime == 120.0 and config.allow_20s:
                    try:
                        if config.no_download and config.cache_dir:
                            lc_data = client.download_lightcurve_cached(
                                tic_id=int(tic_id),
                                sector=int(sector),
                                flux_type="pdcsap",
                                exptime=20.0,
                            )
                        else:
                            lc_data = client.download_lightcurve(
                                tic_id, sector, flux_type="pdcsap", exptime=20.0
                            )
                        lightcurves.append(lc_data)
                        sectors_loaded.append(sector)
                        logger.debug(
                            "Downloaded sector %d (20s fallback): %d points",
                            sector,
                            lc_data.n_points,
                        )
                        continue
                    except (LightCurveNotFoundError, MASTClientError):
                        pass
                download_errors.append(f"Sector {sector}: {e}")
                logger.debug("Skipping sector %d: %s", sector, e)
            except MASTClientError as e:
                download_errors.append(f"Sector {sector}: {e}")
                logger.warning("Download error for sector %d: %s", sector, e)

        if not lightcurves:
            wall_ms = time.perf_counter() * 1000.0 - start_time_ms
            logger.warning("No light curves could be downloaded for TIC %d", tic_id)
            return _make_error_response(
                "LightCurveNotFoundError",
                f"No light curves could be downloaded for TIC {tic_id}: {download_errors}",
                wall_ms,
            )

    # Post-download per-sector gating: drop unusable sectors where possible.
    # This prevents a single broken sector from failing the whole candidate.
    excluded = selection_summary.get("excluded_sectors") if isinstance(selection_summary, dict) else None
    if excluded is None:
        excluded = {}
        selection_summary = selection_summary or {}
        selection_summary["excluded_sectors"] = excluded

    def _exclude_sector(sector: int, reason: str) -> None:
        if int(sector) not in excluded:
            excluded[int(sector)] = reason

    kept_lightcurves: list[LightCurveData] = []
    kept_sectors: list[int] = []
    for lc in lightcurves:
        sector = int(lc.sector)
        if lc.n_valid <= 0:
            _exclude_sector(sector, "no_usable_points")
            continue
        t_min = float(lc.time[lc.valid_mask].min())
        t_max = float(lc.time[lc.valid_mask].max())
        if not (t_min <= float(t0_btjd) <= t_max):
            _exclude_sector(sector, "insufficient_time_coverage")
            continue
        kept_lightcurves.append(lc)
        kept_sectors.append(sector)

    lightcurves = kept_lightcurves
    sectors_loaded = kept_sectors

    if isinstance(selection_summary, dict):
        selection_summary["selected_sectors"] = sectors_loaded

    if not lightcurves:
        wall_ms = time.perf_counter() * 1000.0 - start_time_ms
        if any(v == "no_usable_points" for v in excluded.values()):
            error_class = "NoUsablePointsError"
            msg = "All sectors excluded: no usable points"
        elif any(v == "insufficient_time_coverage" for v in excluded.values()):
            error_class = "InsufficientTimeCoverageError"
            msg = "All sectors excluded: insufficient time coverage for t0"
        else:
            error_class = "SectorGatingError"
            msg = "All sectors excluded by gating rules"
        return _make_error_response(error_class, msg, wall_ms)

    # Step 3: Stitch multi-sector light curves if needed
    if len(lightcurves) == 1:
        stitched_lc_data = lightcurves[0]
        logger.info("Single sector loaded: sector %d", sectors_loaded[0])
    else:
        try:
            stitched_lc_data, _stitch_diag = stitch_lightcurve_data(lightcurves, tic_id=tic_id)
            logger.info(
                "Stitched %d sectors: %d total points",
                len(lightcurves),
                stitched_lc_data.n_points,
            )
        except Exception as e:
            wall_ms = time.perf_counter() * 1000.0 - start_time_ms
            logger.warning("Stitching failed for TIC %d: %s", tic_id, e)
            return _make_error_response("StitchError", str(e), wall_ms)

    # Sanity check: stitched series should have usable points if any sector survived gating.
    if stitched_lc_data.n_valid <= 0:
        wall_ms = time.perf_counter() * 1000.0 - start_time_ms
        return _make_error_response("NoUsablePointsError", "No usable points after stitching", wall_ms)

    # Step 4: Get target info for coordinates and stellar parameters
    target = None
    stellar = None
    ra_deg: float | None = None
    dec_deg: float | None = None

    if config.network_ok:
        try:
            # Create client for target info query (may not exist if using local data)
            target_client = _make_mast_client()
            target = target_client.get_target_info(tic_id)
            ra_deg = target.ra
            dec_deg = target.dec
            stellar = target.stellar
            logger.debug(
                "Target info: ra=%.4f, dec=%.4f, Teff=%s",
                ra_deg or 0.0,
                dec_deg or 0.0,
                getattr(stellar, "teff", None) if stellar else None,
            )
        except MASTClientError as e:
            logger.debug("Could not retrieve target info for TIC %d: %s", tic_id, e)

    # Step 4b: Load TPF for pixel-level vetting (V08-V10)
    # Skip if we cannot access products (network_ok or cache_dir required).
    # TPF loading is slow, so we only load one sector (prefer most recent)
    tpf_stamp: TPFStamp | None = None
    tpf_sector_used: int | None = None
    tpf_exptime_used: float | None = None
    tpf_attempts: list[dict[str, Any]] = []
    if sectors_loaded and (config.network_ok or config.cache_dir):
        # Prefer the most recent sector (likely best), but fall back through others.
        for tpf_sector_to_try in sorted({int(s) for s in sectors_loaded}, reverse=True):
            logger.info("Attempting to load TPF for TIC %d sector %d", tic_id, tpf_sector_to_try)
            exps = sector_exptimes.get(int(tpf_sector_to_try), {120})
            exptime_candidates: list[float] = []
            if 120 in exps:
                exptime_candidates.append(120.0)
            if 20 in exps and config.allow_20s:
                exptime_candidates.append(20.0)
            if not exptime_candidates:
                exptime_candidates = [120.0]
            try:
                tpf_client = _make_mast_client()
                for exptime in exptime_candidates:
                    try:
                        if config.no_download and config.cache_dir:
                            time_arr, flux_arr, flux_err_arr, wcs, aperture_mask, quality_arr = (
                                tpf_client.download_tpf_cached(tic_id, tpf_sector_to_try, exptime=exptime)
                            )
                        else:
                            time_arr, flux_arr, flux_err_arr, wcs, aperture_mask, quality_arr = (
                                tpf_client.download_tpf(tic_id, tpf_sector_to_try, exptime=exptime)
                            )
                        tpf_exptime_used = float(exptime)
                        tpf_attempts.append(
                            {"sector": int(tpf_sector_to_try), "exptime": float(exptime), "ok": True}
                        )
                        break
                    except Exception as e:
                        tpf_attempts.append(
                            {
                                "sector": int(tpf_sector_to_try),
                                "exptime": float(exptime),
                                "ok": False,
                                "error_class": type(e).__name__,
                                "error": str(e),
                            }
                        )
                        continue
                else:
                    raise LightCurveNotFoundError("No TPF product matched allowed cadence")

                tpf_stamp = TPFStamp(
                    time=time_arr,
                    flux=flux_arr,
                    flux_err=flux_err_arr,
                    wcs=wcs,
                    aperture_mask=aperture_mask,
                    quality=quality_arr,
                )
                tpf_sector_used = tpf_sector_to_try
                logger.info(
                    "Loaded TPF for TIC %d sector %d: %d cadences, %dx%d pixels",
                    tic_id,
                    tpf_sector_to_try,
                    flux_arr.shape[0],
                    flux_arr.shape[1],
                    flux_arr.shape[2],
                )
                break
            except LightCurveNotFoundError as e:
                logger.debug("No TPF found for TIC %d sector %d: %s", tic_id, tpf_sector_to_try, e)
            except MASTClientError as e:
                logger.warning(
                    "TPF download failed for TIC %d sector %d: %s", tic_id, tpf_sector_to_try, e
                )
            except Exception as e:
                logger.warning("Unexpected error loading TPF for TIC %d: %s", tic_id, e)

    if config.require_tpf and tpf_stamp is None:
        wall_ms = time.perf_counter() * 1000.0 - start_time_ms
        return _make_error_response(
            "TPFRequiredError",
            "require_tpf=True but no TPF could be loaded",
            wall_ms,
        )

    # Step 4c: Compute pixel-level diagnostics into evidence packet fields.
    # Use explicit skip blocks instead of None for auditability.
    localization: dict[str, Any] = make_skip_block("tpf_unavailable")
    pixel_host_hypotheses: dict[str, Any] = make_skip_block("tpf_unavailable")
    sector_quality_report: dict[str, Any] = make_skip_block("tpf_unavailable")

    if tpf_stamp is not None and tpf_sector_used is not None:
        # Localization diagnostics (WCS-optional; uses OOT-derived references).
        try:
            transit_params = TransitParams(
                period=float(period_days),
                t0=float(t0_btjd),
                duration=float(duration_hours / 24.0),
            )
            diag, _images = compute_localization_diagnostics(
                tpf_data=np.asarray(tpf_stamp.flux, dtype=np.float64),
                time=np.asarray(tpf_stamp.time, dtype=np.float64),
                transit_params=transit_params,
            )
            px_scale_arcsec = 21.0
            if getattr(tpf_stamp, "wcs", None) is not None:
                try:
                    from bittr_tess_vetter.api.wcs_utils import compute_pixel_scale

                    px_scale_arcsec = float(compute_pixel_scale(tpf_stamp.wcs))
                except Exception:
                    px_scale_arcsec = 21.0
            dist_px = float(diag.dist_diff_to_ootbright_px)
            dist_arcsec = float(dist_px * px_scale_arcsec)

            if dist_px <= 1.0:
                verdict = "on_target"
            elif dist_px >= 2.0:
                verdict = "off_target"
            else:
                verdict = "ambiguous"

            localization = {
                "verdict": verdict,
                "target_distance_arcsec": dist_arcsec,
                "uncertainty_semimajor_arcsec": None,
                "target_distance_pixels": dist_px,
                "pixel_scale_arcsec": px_scale_arcsec,
                "diagnostics": diag.to_dict(),
            }
        except Exception as e:
            localization = {
                "verdict": "invalid",
                "warnings": [f"localization_failed:{type(e).__name__}:{e}"],
            }

        # Ghost / scattered-light features (aperture sign consistency, etc.)
        ghost_row: dict[str, Any] | None = None
        try:
            ap_mask = np.asarray(tpf_stamp.aperture_mask, dtype=bool)
            gf = compute_ghost_features(
                np.asarray(tpf_stamp.flux, dtype=np.float64),
                np.asarray(tpf_stamp.time, dtype=np.float64),
                ap_mask,
                float(period_days),
                float(t0_btjd),
                float(duration_hours),
                tic_id=int(tic_id),
                sector=int(tpf_sector_used),
            )
            ghost_row = {
                "sector": int(tpf_sector_used),
                "ghost_like_score_adjusted": float(gf.ghost_like_score),
                "scattered_light_risk": float(gf.scattered_light_risk),
                "aperture_contrast": float(gf.aperture_contrast),
                "aperture_sign_consistent": bool(gf.aperture_sign_consistent),
                "spatial_uniformity": float(gf.spatial_uniformity),
                "prf_likeness": float(gf.prf_likeness),
            }
        except Exception as e:
            ghost_row = {
                "sector": int(tpf_sector_used),
                "ghost_like_score_adjusted": None,
                "scattered_light_risk": None,
                "aperture_sign_consistent": None,
                "warnings": [f"ghost_failed:{type(e).__name__}:{e}"],
            }

        # Aperture family depth curve (requires TPFFitsData container; WCS is unused by the metric).
        aperture_family: dict[str, Any] | None = None
        try:
            from astropy.wcs import WCS

            ref = TPFFitsRef(
                tic_id=int(tic_id),
                sector=int(tpf_sector_used),
                author="spoc",
                exptime_seconds=int(tpf_exptime_used) if tpf_exptime_used is not None else None,
            )
            wcs_obj = tpf_stamp.wcs if getattr(tpf_stamp, "wcs", None) is not None else WCS(naxis=2)
            tpf_fits = TPFFitsData(
                ref=ref,
                time=np.asarray(tpf_stamp.time, dtype=np.float64),
                flux=np.asarray(tpf_stamp.flux, dtype=np.float64),
                flux_err=np.asarray(tpf_stamp.flux_err, dtype=np.float64)
                if getattr(tpf_stamp, "flux_err", None) is not None
                else None,
                wcs=wcs_obj,
                aperture_mask=np.asarray(tpf_stamp.aperture_mask, dtype=np.int32),
                quality=np.asarray(tpf_stamp.quality, dtype=np.int32)
                if getattr(tpf_stamp, "quality", None) is not None
                else np.zeros(int(np.asarray(tpf_stamp.time).shape[0]), dtype=np.int32),
                camera=0,
                ccd=0,
                meta={},
            )
            afr = compute_aperture_family_depth_curve(
                tpf_fits=tpf_fits,
                period=float(period_days),
                t0=float(t0_btjd),
                duration_hours=float(duration_hours),
            )
            aperture_family = afr.to_dict()
        except Exception as e:
            aperture_family = {"warnings": [f"aperture_family_failed:{type(e).__name__}:{e}"]}

        pixel_host_hypotheses = {
            "consensus_best_source_id": f"tic:{int(tic_id)}",
            "host_ambiguity": "unknown",
            "disagreement_flag": "stable",
            "flip_rate": 0.0,
            "timeseries_verdict": None,
            "timeseries_delta_chi2": None,
            "ghost_summary_by_sector": [ghost_row] if ghost_row is not None else [],
            "host_plausibility_auto": {"skipped": True, "reason": "network_or_coords_unavailable"},
        }
        sector_quality_report = {
            "tpf_sector_used": int(tpf_sector_used),
            "aperture_family": aperture_family,
        }

        # Host plausibility (Gaia cone search) - network gated.
        if (
            config.network_ok
            and config.enable_host_plausibility
            and ra_deg is not None
            and dec_deg is not None
            and depth_ppm is not None
        ):
            try:
                gaia = query_gaia_by_position_sync(float(ra_deg), float(dec_deg), radius_arcsec=60.0)
                primary_mag = gaia.source.phot_g_mean_mag if gaia.source else None

                # Estimate stellar radius (solar radii) for implied-size checks.
                radius_rsun = None
                if stellar is not None and getattr(stellar, "radius", None) is not None:
                    radius_rsun = float(stellar.radius)
                elif gaia.astrophysical is not None and gaia.astrophysical.radius_gspphot is not None:
                    radius_rsun = float(gaia.astrophysical.radius_gspphot)

                mags: list[float] = []
                if primary_mag is not None:
                    mags.append(float(primary_mag))
                for n in gaia.neighbors:
                    if n.phot_g_mean_mag is not None:
                        mags.append(float(n.phot_g_mean_mag))

                def _flux_fraction(mag: float | None) -> float | None:
                    if mag is None or not mags:
                        return None
                    return float(compute_flux_fraction_from_mag_list(float(mag), mags))

                primary_id = str(gaia.source.source_id) if gaia.source else f"tic:{int(tic_id)}"
                primary_h = HostHypothesis(
                    source_id=int(gaia.source.source_id) if gaia.source else int(tic_id),
                    name="primary",
                    separation_arcsec=0.0,
                    g_mag=float(primary_mag) if primary_mag is not None else None,
                    estimated_flux_fraction=float(_flux_fraction(primary_mag) or 1.0),
                    radius_rsun=radius_rsun,
                )
                companions: list[HostHypothesis] = []
                for n in gaia.neighbors:
                    companions.append(
                        HostHypothesis(
                            source_id=int(n.source_id),
                            name="neighbor",
                            separation_arcsec=float(n.separation_arcsec),
                            g_mag=float(n.phot_g_mean_mag) if n.phot_g_mean_mag is not None else None,
                            estimated_flux_fraction=float(_flux_fraction(n.phot_g_mean_mag) or 0.0),
                            radius_rsun=None,
                        )
                    )

                scenarios = compute_dilution_scenarios(
                    observed_depth_ppm=float(depth_ppm),
                    primary=primary_h,
                    companions=companions,
                )
                host_ambiguous = any(float(c.separation_arcsec) <= 21.0 for c in companions)
                flags = evaluate_physics_flags(scenarios, host_ambiguous=host_ambiguous)

                scenario_dicts: list[dict[str, Any]] = []
                impossible_ids: list[str] = []
                for sc in scenarios:
                    sid = str(sc.host.source_id)
                    phys_impossible = bool(sc.planet_radius_inconsistent or sc.stellar_companion_likely)
                    if phys_impossible:
                        impossible_ids.append(sid)
                    scenario_dicts.append(
                        {
                            "source_id": sid,
                            "flux_fraction": float(sc.host.estimated_flux_fraction),
                            "true_depth_ppm": float(sc.true_depth_ppm),
                            "depth_correction_factor": float(sc.depth_correction_factor),
                            "physically_impossible": phys_impossible,
                        }
                    )

                pixel_host_hypotheses["host_plausibility_auto"] = {
                    "skipped": False,
                    "primary_source_id": primary_id,
                    "cone_radius_arcsec": 60.0,
                    "physics_flags": {
                        "requires_resolved_followup": bool(flags.requires_resolved_followup),
                        "planet_radius_inconsistent": bool(flags.planet_radius_inconsistent),
                        "rationale": str(flags.rationale),
                    },
                    "physically_impossible_source_ids": impossible_ids,
                    "scenarios": scenario_dicts,
                }
            except Exception as e:
                pixel_host_hypotheses["host_plausibility_auto"] = {
                    "skipped": True,
                    "reason": f"gaia_query_failed:{type(e).__name__}",
                    "error": str(e),
                }

    # Step 5: Create API types for vetting
    ephemeris = Ephemeris(
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
    )
    candidate = Candidate(
        ephemeris=ephemeris,
        depth_ppm=depth_ppm,
    )
    lc_api = LightCurve.from_internal(stitched_lc_data)

    # Step 6: Run vetting pipeline
    logger.info("Running vetting pipeline for TIC %d", tic_id)
    try:
        bundle = run_vetting(
            lc_api,
            candidate,
            stellar=stellar,
            tpf=tpf_stamp,
            network=config.network_ok,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            tic_id=tic_id,
        )
    except Exception as e:
        wall_ms = time.perf_counter() * 1000.0 - start_time_ms
        logger.warning("Vetting pipeline failed for TIC %d: %s", tic_id, e)
        return _make_error_response("VettingPipelineError", str(e), wall_ms)

    logger.info(
        "Vetting complete: %d checks run, %d warnings",
        len(bundle.results),
        len(bundle.warnings),
    )

    # Step 7: Build RawEvidencePacket from pipeline results
    check_results_dicts = [r.model_dump() for r in bundle.results]

    raw: RawEvidencePacket = {
        "target": {
            "tic_id": tic_id,
            "toi": toi,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
        },
        "ephemeris": {
            "period_days": period_days,
            "t0_btjd": t0_btjd,
            "duration_hours": duration_hours,
            "sectors": sectors_loaded,
            "cadence_seconds": stitched_lc_data.cadence_seconds,
        },
        "depth_ppm": {
            "input_depth_ppm": depth_ppm,
        },
        "check_results": check_results_dicts,
        "pixel_host_hypotheses": pixel_host_hypotheses,
        "localization": localization,
        "sector_quality_report": sector_quality_report,
        "candidate_evidence": make_skip_block("not_implemented"),
        "provenance": {
            "pipeline_version": pipeline_version,
            "code_hash": code_hash,
            "dependency_versions": _dependency_versions(),
            "sectors_used": sectors_loaded,
            "sector_selection": selection_summary,
            "n_points": stitched_lc_data.n_points,
            "cadence_seconds": stitched_lc_data.cadence_seconds,
            "inputs_summary": bundle.inputs_summary,
            "duration_ms": bundle.provenance.get("duration_ms"),
            "tpf_sector_used": tpf_sector_used,
            "tpf_exptime_used": tpf_exptime_used,
            "tpf_attempts": tpf_attempts,
            "has_tpf": tpf_stamp is not None,
        },
    }

    # Step 8: Build features from raw evidence
    wall_ms = time.perf_counter() * 1000.0 - start_time_ms

    try:
        row = build_features(raw, config)
        # Ensure required fields are set
        row["item_wall_ms"] = wall_ms
        row["candidate_key"] = candidate_key
    except NotImplementedError:
        # build_features is a stub - create minimal EnrichedRow manually
        logger.debug("build_features not implemented, creating minimal EnrichedRow")
        row = EnrichedRow(
            tic_id=tic_id,
            toi=toi,
            period_days=period_days,
            t0_btjd=t0_btjd,
            duration_hours=duration_hours,
            depth_ppm=depth_ppm,
            status="OK",
            error_class=None,
            error=None,
            candidate_key=candidate_key,
            pipeline_version=pipeline_version,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_config=dataclasses.asdict(config),
            inputs_summary={
                "sectors_used": sectors_loaded,
                "n_points": stitched_lc_data.n_points,
                "cadence_seconds": stitched_lc_data.cadence_seconds,
                "has_stellar": stellar is not None,
                "has_coordinates": ra_deg is not None and dec_deg is not None,
                "has_tpf": tpf_stamp is not None,
                "tpf_sector_used": tpf_sector_used,
            },
            missing_feature_families=["features_not_extracted"],
            item_wall_ms=wall_ms,
        )

    return raw, row


def enrich_worklist(
    worklist_iter: Iterator[dict[str, Any]],
    output_path: str | Path,
    config: FeatureConfig,
    *,
    resume: bool = False,
    limit: int | None = None,
    progress_interval: int = 10,
) -> EnrichmentSummary:
    """Batch-enrich a worklist of candidates to JSONL output.

    This function processes a stream of candidate dictionaries, enriching
    each with vetting features and writing results to a JSONL file.

    Args:
        worklist_iter: Iterator yielding candidate dictionaries with keys:
            - tic_id: int
            - period_days: float
            - t0_btjd: float
            - duration_hours: float
            - depth_ppm: float
        output_path: Path to output JSONL file.
        config: Feature extraction configuration.
        resume: If True, skip candidates already in output file.
        limit: If set, process only the first N candidates (for testing).
        progress_interval: Print progress every N candidates (default: 10).

    Returns:
        EnrichmentSummary with statistics from the run.

    Note:
        - Output is written incrementally with file locking for concurrent safety.
        - Resume mode streams the existing output to build a skip set without
          loading the entire file into memory.
        - Candidate keys are generated as f"{tic_id}|{period_days}|{t0_btjd}"
          and are immutable throughout the pipeline.
    """
    from bittr_tess_vetter.api.jsonl import append_jsonl, stream_existing_candidate_keys

    out_path = Path(output_path)
    progress_path = out_path.with_suffix(out_path.suffix + ".progress.json")

    start_time = time.time()
    processed = 0
    skipped_resume = 0
    errors = 0
    total_input = 0
    last_candidate_key: str | None = None
    error_class_counts: dict[str, int] = {}

    skip_keys: set[str] = set()
    if resume:
        skip_keys = stream_existing_candidate_keys(out_path)

    def _write_progress() -> None:
        payload = {
            "output_path": str(out_path),
            "resume": bool(resume),
            "total_input": int(total_input),
            "processed": int(processed),
            "skipped_resume": int(skipped_resume),
            "errors": int(errors),
            "wall_time_seconds": float(time.time() - start_time),
            "error_class_counts": dict(sorted(error_class_counts.items())),
            "last_candidate_key": last_candidate_key,
            "updated_unix": time.time(),
        }
        try:
            progress_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        except Exception:
            return

    for row in worklist_iter:
        total_input += 1
        if limit is not None and total_input > limit:
            break

        try:
            tic_id = int(row["tic_id"])
            period_days = float(row["period_days"])
            t0_btjd = float(row["t0_btjd"])
            duration_hours = float(row["duration_hours"])
        except (KeyError, ValueError, TypeError) as e:
            errors += 1
            cls = type(e).__name__
            error_class_counts[cls] = error_class_counts.get(cls, 0) + 1
            continue

        depth_ppm_raw = row.get("depth_ppm")
        if depth_ppm_raw is None:
            depth_ppm = None
        else:
            try:
                depth_ppm = float(depth_ppm_raw)
            except (ValueError, TypeError) as e:
                errors += 1
                cls = type(e).__name__
                error_class_counts[cls] = error_class_counts.get(cls, 0) + 1
                continue

        candidate_key = make_candidate_key(tic_id, period_days, t0_btjd)
        last_candidate_key = candidate_key

        if resume and candidate_key in skip_keys:
            skipped_resume += 1
        else:
            toi = row.get("toi")
            toi_str = str(toi) if toi is not None else None
            requested_sectors = row.get("sectors")
            sectors_list: list[int] | None = None
            if isinstance(requested_sectors, list):
                sectors_list = [int(s) for s in requested_sectors]
            try:
                _raw, enriched = enrich_candidate(
                    tic_id=tic_id,
                    toi=toi_str,
                    period_days=period_days,
                    t0_btjd=t0_btjd,
                    duration_hours=duration_hours,
                    depth_ppm=depth_ppm,
                    config=config,
                    sectors=sectors_list,
                )
                append_jsonl(out_path, dict(enriched))
                if enriched.get("status") == "OK":
                    processed += 1
                else:
                    errors += 1
                    cls = str(enriched.get("error_class") or "EnrichmentError")
                    error_class_counts[cls] = error_class_counts.get(cls, 0) + 1
            except Exception as e:
                errors += 1
                cls = type(e).__name__
                error_class_counts[cls] = error_class_counts.get(cls, 0) + 1
                append_jsonl(
                    out_path,
                    {
                        "tic_id": tic_id,
                        "toi": toi_str,
                        "period_days": period_days,
                        "t0_btjd": t0_btjd,
                        "duration_hours": duration_hours,
                        "depth_ppm": depth_ppm,
                        "status": "ERROR",
                        "error_class": cls,
                        "error": str(e),
                        "candidate_key": candidate_key,
                        "pipeline_version": _pipeline_version(),
                        "feature_schema_version": "unknown",
                        "feature_config": dataclasses.asdict(config),
                        "inputs_summary": {},
                        "missing_feature_families": ["all"],
                        "item_wall_ms": 0.0,
                    },
                )

        count = processed + skipped_resume + errors
        if progress_interval > 0 and count % progress_interval == 0:
            _write_progress()

    _write_progress()
    return EnrichmentSummary(
        total_input=total_input,
        processed=processed,
        skipped_resume=skipped_resume,
        errors=errors,
        wall_time_seconds=time.time() - start_time,
        error_class_counts=error_class_counts,
    )


__all__ = [
    "EnrichmentSummary",
    "enrich_candidate",
    "enrich_worklist",
    "make_candidate_key",
]
