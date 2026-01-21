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
    depth_ppm: float,
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
        depth_ppm: Transit depth in parts per million.
        config: Feature extraction configuration.

    Returns:
        Tuple of (RawEvidencePacket, EnrichedRow) containing the raw
        vetting evidence and extracted ML features.

    Note:
        The candidate_key is generated as f"{tic_id}|{period_days}|{t0_btjd}"
        and is immutable - it does not change based on T0 refinement.
    """
    from bittr_tess_vetter.api.io import (
        LightCurveNotFoundError,
        MASTClient,
        MASTClientError,
    )
    from bittr_tess_vetter.api.stitch import stitch_lightcurve_data
    from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve, TPFStamp
    from bittr_tess_vetter.api.vet import vet_candidate as run_vetting
    from bittr_tess_vetter.data_sources.sector_selection import select_sectors
    from bittr_tess_vetter.features import FEATURE_SCHEMA_VERSION

    start_time_ms = time.perf_counter() * 1000.0
    pipeline_version = _pipeline_version()
    candidate_key = make_candidate_key(tic_id, period_days, t0_btjd)

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
            "pixel_host_hypotheses": None,
            "localization": None,
            "sector_quality_report": None,
            "candidate_evidence": None,
            "provenance": {
                "pipeline_version": pipeline_version,
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

    if config.no_download and not config.local_data_path:
        wall_ms = time.perf_counter() * 1000.0 - start_time_ms
        return _make_error_response(
            "NoDownloadError",
            "no_download=True requires local_data_path (cache-only MAST fetch is not implemented)",
            wall_ms,
        )

    if not config.network_ok and not config.local_data_path:
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
        # Standard MAST-based loading
        logger.info("Searching for light curves for TIC %d", tic_id)
        try:
            client = MASTClient()
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

        # Prefer 120s cadence by default, allow 20s if configured
        preferred_exptime: float | None = 120.0
        if config.allow_20s:
            # Check if 20s is available
            has_20s = any(abs(r.exptime - 20.0) < 1.0 for r in search_results)
            if has_20s:
                preferred_exptime = 20.0
                logger.info("Using 20s cadence data (allow_20s=True)")

        download_errors: list[str] = []

        for sector in selection.selected_sectors:
            try:
                lc_data = client.download_lightcurve(
                    tic_id,
                    sector,
                    flux_type="pdcsap",
                    exptime=preferred_exptime,
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
                if preferred_exptime == 120.0 and config.allow_20s:
                    try:
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

    # Post-stitch usability gating: ensure at least some valid samples exist.
    # Note: `LightCurveData.valid_mask` already includes quality==0 and finite checks.
    if stitched_lc_data.n_valid <= 0:
        wall_ms = time.perf_counter() * 1000.0 - start_time_ms
        return _make_error_response(
            "NoUsablePointsError",
            "Stitched light curve has no usable points after quality/finite filtering",
            wall_ms,
        )

    # Minimal coverage gating: require the target epoch to fall inside the observed time span.
    # This is intentionally conservative and avoids any model assumptions.
    t_min = float(stitched_lc_data.time[stitched_lc_data.valid_mask].min())
    t_max = float(stitched_lc_data.time[stitched_lc_data.valid_mask].max())
    if not (t_min <= float(t0_btjd) <= t_max):
        wall_ms = time.perf_counter() * 1000.0 - start_time_ms
        return _make_error_response(
            "InsufficientTimeCoverageError",
            f"t0_btjd={t0_btjd} outside observed span [{t_min}, {t_max}]",
            wall_ms,
        )

    # Step 4: Get target info for coordinates and stellar parameters
    target = None
    stellar = None
    ra_deg: float | None = None
    dec_deg: float | None = None

    if config.network_ok:
        try:
            # Create client for target info query (may not exist if using local data)
            target_client = MASTClient()
            target = target_client.get_target_info(tic_id)
            ra_deg = target.ra
            dec_deg = target.dec
            stellar = target.stellar
            logger.debug(
                "Target info: ra=%.4f, dec=%.4f, Teff=%s",
                ra_deg or 0.0,
                dec_deg or 0.0,
                stellar.teff if stellar else None,
            )
        except MASTClientError as e:
            logger.debug("Could not retrieve target info for TIC %d: %s", tic_id, e)

    # Step 4b: Load TPF for pixel-level vetting (V08-V10)
    # Skip if no_download is set or we don't have network access
    # TPF loading is slow, so we only load one sector (prefer most recent)
    tpf_stamp: TPFStamp | None = None
    tpf_sector_used: int | None = None

    if not config.no_download and config.network_ok and sectors_loaded:
        # Prefer the most recent sector (likely best data quality)
        tpf_sector_to_try = max(sectors_loaded)
        logger.info("Attempting to load TPF for TIC %d sector %d", tic_id, tpf_sector_to_try)

        try:
            # Create client for TPF download
            tpf_client = MASTClient()
            time_arr, flux_arr, flux_err_arr, wcs, aperture_mask, quality_arr = (
                tpf_client.download_tpf(tic_id, tpf_sector_to_try, exptime=120.0)
            )

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
        except LightCurveNotFoundError as e:
            logger.debug("No TPF found for TIC %d sector %d: %s", tic_id, tpf_sector_to_try, e)
        except MASTClientError as e:
            logger.warning(
                "TPF download failed for TIC %d sector %d: %s", tic_id, tpf_sector_to_try, e
            )
        except Exception as e:
            logger.warning("Unexpected error loading TPF for TIC %d: %s", tic_id, e)

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
        "pixel_host_hypotheses": None,
        "localization": None,
        "sector_quality_report": None,
        "candidate_evidence": None,
        "provenance": {
            "pipeline_version": pipeline_version,
            "sectors_used": sectors_loaded,
            "sector_selection": selection_summary,
            "n_points": stitched_lc_data.n_points,
            "cadence_seconds": stitched_lc_data.cadence_seconds,
            "inputs_summary": bundle.inputs_summary,
            "duration_ms": bundle.provenance.get("duration_ms"),
            "tpf_sector_used": tpf_sector_used,
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
            depth_ppm = float(row["depth_ppm"])
        except (KeyError, ValueError, TypeError) as e:
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
