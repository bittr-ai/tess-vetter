"""`btv vet` command for single-candidate vetting."""

from __future__ import annotations

import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import click
import numpy as np

from bittr_tess_vetter.api.detrend import bin_median_trend, sigma_clip
from bittr_tess_vetter.api.generate_report import _select_tpf_sectors
from bittr_tess_vetter.api.pipeline import PipelineConfig
from bittr_tess_vetter.api.stitch import stitch_lightcurve_data
from bittr_tess_vetter.api.transit_masks import get_out_of_transit_mask
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve, TPFStamp
from bittr_tess_vetter.api.vet import vet_candidate
from bittr_tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_PROGRESS_ERROR,
    EXIT_REMOTE_TIMEOUT,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    load_json_file,
    parse_extra_params,
    resolve_optional_output_path,
)
from bittr_tess_vetter.cli.progress_metadata import (
    ProgressIOError,
    build_single_candidate_progress,
    decide_resume_for_single_candidate,
    read_progress_metadata,
    write_progress_metadata_atomic,
)
from bittr_tess_vetter.platform.catalogs.toi_resolution import (
    LookupStatus,
    lookup_tic_coordinates,
    resolve_toi_to_tic_ephemeris_depth,
)
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError, MASTClient


def _looks_like_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    return "timeout" in type(exc).__name__.lower()


def _progress_path_from_args(
    out_path: Path | None,
    progress_path_arg: str | None,
    resume: bool,
) -> Path | None:
    if progress_path_arg:
        return Path(progress_path_arg)
    if resume:
        if out_path is None:
            raise BtvCliError(
                "--resume requires --out to be a file path or explicit --progress-path",
                exit_code=EXIT_INPUT_ERROR,
            )
        return out_path.with_suffix(out_path.suffix + ".progress.json")
    return None


def _to_optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_required_finite_float(value: Any, *, label: str) -> float:
    if value is None:
        raise BtvCliError(f"Sector measurements schema error: missing {label}")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise BtvCliError(f"Sector measurements schema error: {label} must be numeric") from exc
    if not np.isfinite(out):
        raise BtvCliError(f"Sector measurements schema error: {label} must be finite")
    return out


def _to_required_int(value: Any, *, label: str) -> int:
    if value is None:
        raise BtvCliError(f"Sector measurements schema error: missing {label}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BtvCliError(f"Sector measurements schema error: {label} must be an integer") from exc


def _to_optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


_HIGH_SALIENCE_FLAGS = {"MODEL_PREFERS_NON_TRANSIT"}
_SUPPORTED_DETREND_METHODS: tuple[str, ...] = ("transit_masked_bin_median",)


def _validate_detrend_args(
    *,
    detrend_bin_hours: float,
    detrend_buffer: float,
    detrend_sigma_clip: float,
) -> None:
    if float(detrend_bin_hours) <= 0.0:
        raise BtvCliError("--detrend-bin-hours must be > 0", exit_code=EXIT_INPUT_ERROR)
    if float(detrend_buffer) <= 0.0:
        raise BtvCliError("--detrend-buffer must be > 0", exit_code=EXIT_INPUT_ERROR)
    if float(detrend_sigma_clip) <= 0.0:
        raise BtvCliError("--detrend-sigma-clip must be > 0", exit_code=EXIT_INPUT_ERROR)


def _normalize_detrend_method(detrend: str | None) -> str | None:
    if detrend is None:
        return None
    method = str(detrend).strip().lower()
    if method == "":
        return None
    if method not in _SUPPORTED_DETREND_METHODS:
        choices = ", ".join(_SUPPORTED_DETREND_METHODS)
        raise BtvCliError(
            f"--detrend must be one of: {choices}",
            exit_code=EXIT_INPUT_ERROR,
        )
    return method


def _detrend_lightcurve_for_vetting(
    *,
    lc: LightCurve,
    candidate: Candidate,
    method: str,
    bin_hours: float,
    buffer_factor: float,
    clip_sigma: float,
) -> tuple[LightCurve, dict[str, Any]]:
    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)
    flux_err = np.asarray(lc.flux_err, dtype=np.float64) if lc.flux_err is not None else np.zeros_like(flux)
    valid_mask = np.asarray(lc.valid_mask, dtype=bool) if lc.valid_mask is not None else np.ones_like(flux, dtype=bool)

    finite_mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    oot_mask = get_out_of_transit_mask(
        time,
        candidate.ephemeris.period_days,
        candidate.ephemeris.t0_btjd,
        candidate.ephemeris.duration_hours,
        buffer_factor=float(buffer_factor),
    )
    trend_fit_mask = valid_mask & finite_mask & oot_mask

    n_sigma_clipped = 0
    if int(np.sum(trend_fit_mask)) >= 3:
        clip_keep = sigma_clip(flux[trend_fit_mask], sigma=float(clip_sigma))
        n_sigma_clipped = int(np.sum(trend_fit_mask)) - int(np.sum(clip_keep))
        trend_indices = np.flatnonzero(trend_fit_mask)
        trend_fit_mask = trend_fit_mask.copy()
        trend_fit_mask[trend_indices[~clip_keep]] = False

    fit_flux = flux.copy()
    fit_flux[~trend_fit_mask] = np.nan
    trend = bin_median_trend(time, fit_flux, bin_hours=float(bin_hours), min_bin_points=1)

    trend_ref = float(np.nanmedian(trend[trend_fit_mask])) if np.any(trend_fit_mask) else float(np.nanmedian(trend))
    if not np.isfinite(trend_ref) or trend_ref == 0.0:
        trend_ref = 1.0
    safe_trend = np.where(np.isfinite(trend) & (trend != 0.0), trend, trend_ref)

    detrended_flux = flux / safe_trend * trend_ref
    detrended_flux_err = flux_err / safe_trend * trend_ref
    detrended_lc = LightCurve(
        time=time,
        flux=detrended_flux,
        flux_err=detrended_flux_err,
        quality=np.asarray(lc.quality, dtype=np.int32) if lc.quality is not None else None,
        valid_mask=valid_mask,
    )
    detrend_provenance: dict[str, Any] = {
        "applied": True,
        "method": str(method),
        "bin_hours": float(bin_hours),
        "buffer_factor": float(buffer_factor),
        "sigma_clip": float(clip_sigma),
        "n_points": int(len(time)),
        "n_trend_fit_points": int(np.sum(trend_fit_mask)),
        "n_sigma_clipped": int(n_sigma_clipped),
    }
    return detrended_lc, detrend_provenance


def _to_optional_toi(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_resolved_toi(payload: dict[str, Any]) -> str | None:
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    for row in results:
        if not isinstance(row, dict):
            continue
        if str(row.get("id")) != "V07":
            continue
        metrics = row.get("metrics")
        if isinstance(metrics, dict):
            resolved = _to_optional_toi(metrics.get("toi"))
            if resolved is not None:
                return resolved
        raw = row.get("raw")
        if isinstance(raw, dict):
            nested = raw.get("row")
            if isinstance(nested, dict):
                resolved = _to_optional_toi(nested.get("toi"))
                if resolved is not None:
                    return resolved
    return None


def _load_sector_measurements(path: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    payload = load_json_file(Path(path), label="sector measurements file")
    rows_raw = payload.get("sector_measurements")
    if not isinstance(rows_raw, list):
        raise BtvCliError(
            "Sector measurements schema error: top-level 'sector_measurements' must be a list",
            exit_code=EXIT_INPUT_ERROR,
        )

    rows: list[dict[str, Any]] = []
    for idx, row_raw in enumerate(rows_raw):
        if not isinstance(row_raw, dict):
            raise BtvCliError(
                f"Sector measurements schema error: row {idx} must be an object",
                exit_code=EXIT_INPUT_ERROR,
            )
        sector = _to_required_int(row_raw.get("sector"), label=f"row {idx}.sector")
        depth_ppm = _to_required_finite_float(row_raw.get("depth_ppm"), label=f"row {idx}.depth_ppm")
        depth_err_ppm = _to_required_finite_float(
            row_raw.get("depth_err_ppm"), label=f"row {idx}.depth_err_ppm"
        )
        if not depth_err_ppm > 0.0:
            raise BtvCliError(
                f"Sector measurements schema error: row {idx}.depth_err_ppm must be > 0",
                exit_code=EXIT_INPUT_ERROR,
            )

        normalized: dict[str, Any] = {
            "sector": int(sector),
            "depth_ppm": float(depth_ppm),
            "depth_err_ppm": float(depth_err_ppm),
        }
        if "duration_hours" in row_raw:
            duration_hours = _to_optional_finite_float(row_raw.get("duration_hours"))
            if duration_hours is None:
                raise BtvCliError(
                    f"Sector measurements schema error: row {idx}.duration_hours must be finite",
                    exit_code=EXIT_INPUT_ERROR,
                )
            normalized["duration_hours"] = float(duration_hours)
        if "duration_err_hours" in row_raw:
            duration_err_hours = _to_optional_finite_float(row_raw.get("duration_err_hours"))
            if duration_err_hours is None:
                raise BtvCliError(
                    f"Sector measurements schema error: row {idx}.duration_err_hours must be finite",
                    exit_code=EXIT_INPUT_ERROR,
                )
            normalized["duration_err_hours"] = float(duration_err_hours)
        if "n_transits" in row_raw:
            normalized["n_transits"] = _to_required_int(row_raw.get("n_transits"), label=f"row {idx}.n_transits")
        if "shape_metric" in row_raw:
            shape_metric = _to_optional_finite_float(row_raw.get("shape_metric"))
            if shape_metric is None:
                raise BtvCliError(
                    f"Sector measurements schema error: row {idx}.shape_metric must be finite",
                    exit_code=EXIT_INPUT_ERROR,
                )
            normalized["shape_metric"] = float(shape_metric)
        if "quality_weight" in row_raw:
            quality_weight = _to_optional_finite_float(row_raw.get("quality_weight"))
            if quality_weight is None:
                raise BtvCliError(
                    f"Sector measurements schema error: row {idx}.quality_weight must be finite",
                    exit_code=EXIT_INPUT_ERROR,
                )
            normalized["quality_weight"] = float(quality_weight)

        rows.append(normalized)

    provenance = payload.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise BtvCliError(
            "Sector measurements schema error: top-level 'provenance' must be an object when present",
            exit_code=EXIT_INPUT_ERROR,
        )
    return rows, provenance


def _build_sector_gating_block(
    *,
    payload: dict[str, Any],
    sector_measurements: list[dict[str, Any]],
    source_path: str,
    source_provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    v21_row: dict[str, Any] | None = None
    results = payload.get("results")
    if isinstance(results, list):
        for row in results:
            if isinstance(row, dict) and str(row.get("id")) == "V21":
                v21_row = row
                break

    positive_weight = 0
    positive_depth_err = 0
    for row in sector_measurements:
        weight = _to_optional_float(row.get("quality_weight"))
        err = _to_optional_float(row.get("depth_err_ppm"))
        if weight is None or weight > 0.0:
            positive_weight += 1
        if err is not None and err > 0.0:
            positive_depth_err += 1

    v21_status = "not_run"
    v21_flags: list[str] = []
    v21_measurements_used = 0
    if v21_row is not None:
        v21_status = str(v21_row.get("status") or "unknown")
        flags = v21_row.get("flags")
        if isinstance(flags, list):
            v21_flags = [str(flag) for flag in flags]
        raw = v21_row.get("raw")
        if isinstance(raw, dict):
            raw_measurements = raw.get("measurements")
            if isinstance(raw_measurements, list):
                v21_measurements_used = len([item for item in raw_measurements if isinstance(item, dict)])

    return {
        "source_path": source_path,
        "source_provenance": source_provenance,
        "n_input_rows": int(len(sector_measurements)),
        "n_positive_weight_rows": int(positive_weight),
        "n_positive_depth_err_rows": int(positive_depth_err),
        "v21_status": v21_status,
        "v21_flags": v21_flags,
        "v21_measurements_used": int(v21_measurements_used),
        "used_by_v21": bool(v21_row is not None),
    }


def _build_root_summary(*, payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results")
    rows = [r for r in results if isinstance(r, dict)] if isinstance(results, list) else []
    n_ok = sum(1 for row in rows if row.get("status") == "ok")
    n_skipped = sum(1 for row in rows if row.get("status") == "skipped")
    n_failed = sum(1 for row in rows if row.get("status") == "error")

    flagged_checks: set[str] = set()
    concerns: set[str] = set()
    for row in rows:
        flags = row.get("flags")
        check_id = str(row.get("id") or "")
        if row.get("status") == "error" and check_id:
            flagged_checks.add(check_id)
        if isinstance(flags, list):
            if any(str(flag) in _HIGH_SALIENCE_FLAGS for flag in flags) and check_id:
                flagged_checks.add(check_id)
            for flag in flags:
                if str(flag) in _HIGH_SALIENCE_FLAGS:
                    concerns.add(str(flag))

    if "MODEL_PREFERS_NON_TRANSIT" in concerns:
        disposition_hint = "needs_model_competition_review"
    elif n_failed > 0:
        disposition_hint = "needs_failed_checks_review"
    elif n_skipped > 0:
        disposition_hint = "needs_additional_data"
    else:
        disposition_hint = "all_clear"
    return {
        "n_ok": int(n_ok),
        "n_failed": int(n_failed),
        "n_skipped": int(n_skipped),
        "flagged_checks": sorted(flagged_checks),
        "concerns": sorted(concerns),
        "disposition_hint": disposition_hint,
    }


def _apply_cli_payload_contract(
    *,
    payload: dict[str, Any],
    toi: str | None,
    input_resolution: dict[str, Any] | None,
    coordinate_resolution: dict[str, Any] | None,
) -> dict[str, Any]:
    inputs_summary_raw = payload.get("inputs_summary")
    if isinstance(inputs_summary_raw, dict):
        inputs_summary = inputs_summary_raw
    else:
        inputs_summary = {}
        payload["inputs_summary"] = inputs_summary

    if input_resolution is not None:
        inputs_summary["input_resolution"] = input_resolution
    if coordinate_resolution is not None:
        inputs_summary["coordinate_resolution"] = coordinate_resolution

    resolved_toi = _extract_resolved_toi(payload)
    effective_toi = toi if toi is not None else resolved_toi
    if input_resolution is not None and effective_toi is not None:
        inputs = inputs_summary.get("input_resolution", {}).get("inputs")
        if isinstance(inputs, dict):
            inputs["toi"] = effective_toi

    payload["summary"] = _build_root_summary(payload=payload)
    return payload


def _resolution_error_to_exit(status: LookupStatus) -> int:
    if status == LookupStatus.TIMEOUT:
        return EXIT_REMOTE_TIMEOUT
    if status == LookupStatus.DATA_UNAVAILABLE:
        return EXIT_DATA_UNAVAILABLE
    return EXIT_RUNTIME_ERROR


def _resolve_candidate_inputs(
    *,
    network_ok: bool,
    toi: str | None,
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
) -> tuple[int, float, float, float, float | None, dict[str, Any]]:
    resolved = {
        "tic_id": None,
        "period_days": None,
        "t0_btjd": None,
        "duration_hours": None,
        "depth_ppm": None,
    }
    overrides: list[str] = []
    errors: list[str] = []
    source = "cli"
    resolved_from = "cli"

    if toi is not None:
        if not network_ok:
            raise BtvCliError(
                "--toi requires --network-ok to resolve ExoFOP inputs",
                exit_code=EXIT_DATA_UNAVAILABLE,
            )
        toi_result = resolve_toi_to_tic_ephemeris_depth(toi)
        source = "toi_catalog"
        resolved_from = "exofop_toi_table"
        if toi_result.status != LookupStatus.OK:
            if tic_id is None and period_days is None and t0_btjd is None and duration_hours is None:
                raise BtvCliError(
                    toi_result.message or f"Failed to resolve TOI {toi}",
                    exit_code=_resolution_error_to_exit(toi_result.status),
                )
            errors.append(toi_result.message or f"TOI resolution degraded for {toi}")
        resolved["tic_id"] = toi_result.tic_id
        resolved["period_days"] = toi_result.period_days
        resolved["t0_btjd"] = toi_result.t0_btjd
        resolved["duration_hours"] = toi_result.duration_hours
        resolved["depth_ppm"] = toi_result.depth_ppm

    manual_inputs = {
        "tic_id": tic_id,
        "period_days": period_days,
        "t0_btjd": t0_btjd,
        "duration_hours": duration_hours,
        "depth_ppm": depth_ppm,
    }
    for key, value in manual_inputs.items():
        if value is not None:
            if resolved.get(key) is not None:
                overrides.append(key)
            resolved[key] = value

    if resolved["tic_id"] is None:
        raise BtvCliError(
            "Missing TIC identifier. Provide --tic-id or --toi.",
            exit_code=EXIT_INPUT_ERROR,
        )
    missing_ephemeris = [
        name
        for name in ("period_days", "t0_btjd", "duration_hours")
        if resolved[name] is None
    ]
    if missing_ephemeris:
        if toi is not None:
            raise BtvCliError(
                f"Resolved TOI is missing required fields: {', '.join(missing_ephemeris)}",
                exit_code=EXIT_DATA_UNAVAILABLE,
            )
        raise BtvCliError(
            f"Missing required inputs: {', '.join(missing_ephemeris)}",
            exit_code=EXIT_INPUT_ERROR,
        )

    input_resolution = {
        "source": source,
        "inputs": {
            "tic_id": resolved["tic_id"],
            "period_days": resolved["period_days"],
            "t0_btjd": resolved["t0_btjd"],
            "duration_hours": resolved["duration_hours"],
            "depth_ppm": resolved["depth_ppm"],
        },
        "resolved_from": resolved_from,
        "overrides": overrides,
        "errors": errors,
    }
    return (
        int(resolved["tic_id"]),
        float(resolved["period_days"]),
        float(resolved["t0_btjd"]),
        float(resolved["duration_hours"]),
        float(resolved["depth_ppm"]) if resolved["depth_ppm"] is not None else None,
        input_resolution,
    )


def _resolve_coordinates(
    *,
    tic_id: int,
    ra_deg: float | None,
    dec_deg: float | None,
    network_ok: bool,
    require_coordinates: bool,
) -> tuple[float | None, float | None, dict[str, Any]]:
    errors: list[str] = []
    if ra_deg is not None and dec_deg is not None:
        return ra_deg, dec_deg, {
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "source": "user",
            "errors": errors,
        }

    if network_ok:
        coord_result = lookup_tic_coordinates(tic_id=tic_id)
        if coord_result.status == LookupStatus.OK:
            return coord_result.ra_deg, coord_result.dec_deg, {
                "ra_deg": coord_result.ra_deg,
                "dec_deg": coord_result.dec_deg,
                "source": "mast",
                "errors": errors,
            }
        errors.append(coord_result.message or "Coordinate lookup failed")
        if require_coordinates:
            raise BtvCliError(
                coord_result.message or "Coordinates unavailable",
                exit_code=_resolution_error_to_exit(coord_result.status),
            )
    elif require_coordinates:
        raise BtvCliError(
            "Coordinates required but --network-ok is disabled and no coordinates were provided.",
            exit_code=EXIT_DATA_UNAVAILABLE,
        )

    return ra_deg, dec_deg, {
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "source": "missing",
        "errors": errors,
    }


def _execute_vet(
    *,
    tic_id: int,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    depth_ppm: float | None,
    toi: str | None,
    ra_deg: float | None,
    dec_deg: float | None,
    preset: str,
    checks: list[str] | None,
    network_ok: bool,
    sectors: list[int] | None,
    flux_type: str,
    fetch_tpf: bool,
    require_tpf: bool,
    tpf_sector_strategy: str,
    tpf_sectors: list[int] | None,
    pipeline_config: PipelineConfig,
    detrend: str | None,
    detrend_bin_hours: float,
    detrend_buffer: float,
    detrend_sigma_clip: float,
    input_resolution: dict[str, Any] | None = None,
    coordinate_resolution: dict[str, Any] | None = None,
    sector_measurements: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    client = MASTClient()
    lightcurves = client.download_all_sectors(tic_id, flux_type=flux_type, sectors=sectors)
    if not lightcurves:
        raise LightCurveNotFoundError(f"No sectors available for TIC {tic_id}")

    if len(lightcurves) == 1:
        stitched_lc = lightcurves[0]
    else:
        stitched_lc, _ = stitch_lightcurve_data(lightcurves, tic_id=tic_id)

    lc = LightCurve.from_internal(stitched_lc)
    candidate = Candidate(
        ephemeris=Ephemeris(
            period_days=period_days,
            t0_btjd=t0_btjd,
            duration_hours=duration_hours,
        ),
        depth_ppm=depth_ppm,
    )
    sectors_used = sorted({int(lc_data.sector) for lc_data in lightcurves if lc_data.sector is not None})
    sector_times = {
        int(lc_data.sector): np.asarray(lc_data.time, dtype=np.float64)
        for lc_data in lightcurves
        if lc_data.sector is not None and lc_data.time is not None
    }

    tpf = _load_tpf_for_vetting(
        client=client,
        tic_id=tic_id,
        lc=lc,
        candidate=candidate,
        sectors_used=sectors_used,
        sector_times=sector_times,
        fetch_tpf=fetch_tpf,
        require_tpf=require_tpf,
        network_ok=network_ok,
        tpf_sector_strategy=tpf_sector_strategy,
        requested_tpf_sectors=tpf_sectors,
    )

    context: dict[str, Any] | None = None
    if toi is not None or sector_measurements is not None:
        context = {}
        if toi is not None:
            context["toi"] = toi
        if sector_measurements is not None:
            context["sector_measurements"] = sector_measurements

    detrend_provenance: dict[str, Any] | None = None
    vet_lc = lc
    if detrend is not None:
        vet_lc, detrend_provenance = _detrend_lightcurve_for_vetting(
            lc=lc,
            candidate=candidate,
            method=detrend,
            bin_hours=float(detrend_bin_hours),
            buffer_factor=float(detrend_buffer),
            clip_sigma=float(detrend_sigma_clip),
        )

    bundle = vet_candidate(
        lc=vet_lc,
        candidate=candidate,
        tpf=tpf,
        network=network_ok,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        tic_id=tic_id,
        preset=preset,
        checks=checks,
        context=context,
        pipeline_config=pipeline_config,
    )

    payload = bundle.model_dump(mode="json")
    if detrend_provenance is not None:
        provenance_raw = payload.get("provenance")
        provenance = provenance_raw if isinstance(provenance_raw, dict) else {}
        provenance["detrend"] = detrend_provenance
        payload["provenance"] = provenance
    return _apply_cli_payload_contract(
        payload=payload,
        toi=toi,
        input_resolution=input_resolution,
        coordinate_resolution=coordinate_resolution,
    )


def _load_tpf_for_vetting(
    *,
    client: MASTClient,
    tic_id: int,
    lc: LightCurve,
    candidate: Candidate,
    sectors_used: list[int],
    sector_times: dict[int, np.ndarray],
    fetch_tpf: bool,
    require_tpf: bool,
    network_ok: bool,
    tpf_sector_strategy: str,
    requested_tpf_sectors: list[int] | None,
) -> TPFStamp | None:
    """Best-effort TPF acquisition for vetting flows."""
    if not fetch_tpf and not require_tpf:
        return None

    selected = _select_tpf_sectors(
        strategy=tpf_sector_strategy,
        sectors_used=sectors_used,
        requested=requested_tpf_sectors,
        lc_api=lc,
        candidate_api=candidate,
        sector_times=sector_times,
    )
    if len(selected) == 0:
        if require_tpf:
            raise LightCurveNotFoundError("No TPF sector selected for this candidate")
        return None

    last_exc: Exception | None = None
    for sector in selected:
        try:
            time_arr, flux_cube, flux_err, wcs, aperture_mask, quality = client.download_tpf_cached(
                tic_id=tic_id,
                sector=sector,
            )
        except Exception as exc_cached:
            last_exc = exc_cached
            if not network_ok:
                continue
            try:
                time_arr, flux_cube, flux_err, wcs, aperture_mask, quality = client.download_tpf(
                    tic_id=tic_id,
                    sector=sector,
                )
            except Exception as exc:
                last_exc = exc
                continue

        return TPFStamp(
            time=np.asarray(time_arr, dtype=np.float64),
            flux=np.asarray(flux_cube, dtype=np.float64),
            flux_err=np.asarray(flux_err, dtype=np.float64) if flux_err is not None else None,
            wcs=wcs,
            aperture_mask=np.asarray(aperture_mask) if aperture_mask is not None else None,
            quality=np.asarray(quality, dtype=np.int32) if quality is not None else None,
        )

    if require_tpf:
        if last_exc is None:
            raise LightCurveNotFoundError(f"No TPF available for TIC {tic_id}")
        raise LightCurveNotFoundError(f"TPF unavailable for TIC {tic_id}: {last_exc}")
    return None


@click.command("vet")
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--period-days", type=float, default=None, help="Orbital period in days.")
@click.option("--t0-btjd", type=float, default=None, help="Reference epoch in BTJD.")
@click.option("--duration-hours", type=float, default=None, help="Transit duration in hours.")
@click.option("--depth-ppm", type=float, default=None, help="Transit depth in ppm.")
@click.option("--toi", type=str, default=None, help="Optional TOI label (overrides resolved value).")
@click.option("--ra-deg", type=float, default=None, help="Right ascension in degrees.")
@click.option("--dec-deg", type=float, default=None, help="Declination in degrees.")
@click.option(
    "--require-coordinates",
    is_flag=True,
    default=False,
    help="Fail if coordinates cannot be resolved or provided.",
)
@click.option(
    "--preset",
    type=click.Choice(["default", "extended"], case_sensitive=False),
    default="default",
    show_default=True,
    help="Vetting preset.",
)
@click.option("--check", "checks", multiple=True, help="Repeat for specific check IDs.")
@click.option(
    "--network-ok/--no-network",
    default=False,
    show_default=True,
    help="Allow network-dependent checks.",
)
@click.option(
    "--fetch-tpf/--no-fetch-tpf",
    default=False,
    show_default=True,
    help="Attempt to fetch TPF for pixel-level checks.",
)
@click.option(
    "--require-tpf",
    is_flag=True,
    default=False,
    help="Fail if TPF cannot be loaded.",
)
@click.option(
    "--tpf-sector-strategy",
    type=click.Choice(["best", "all", "requested"], case_sensitive=False),
    default="best",
    show_default=True,
    help="How to choose sector(s) for TPF fetch.",
)
@click.option("--tpf-sector", "tpf_sectors", multiple=True, type=int, help="Sector(s) when strategy=requested.")
@click.option("--sectors", multiple=True, type=int, help="Optional sector filters.")
@click.option(
    "--flux-type",
    type=click.Choice(["pdcsap", "sap"], case_sensitive=False),
    default="pdcsap",
    show_default=True,
)
@click.option("--timeout-seconds", type=float, default=None)
@click.option("--random-seed", type=int, default=None)
@click.option(
    "--detrend",
    type=str,
    default=None,
    show_default=True,
    help=(
        "Pre-vetting detrend method. Supported: "
        "transit_masked_bin_median."
    ),
)
@click.option("--detrend-bin-hours", type=float, default=6.0, show_default=True)
@click.option("--detrend-buffer", type=float, default=2.0, show_default=True)
@click.option("--detrend-sigma-clip", type=float, default=5.0, show_default=True)
@click.option("--extra-param", "extra_params", multiple=True, help="Repeat KEY=VALUE entries.")
@click.option("--fail-fast/--no-fail-fast", default=False, show_default=True)
@click.option("--emit-warnings/--no-emit-warnings", default=False, show_default=True)
@click.option(
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
@click.option("--progress-path", type=str, default=None, help="Optional progress metadata path.")
@click.option("--resume", is_flag=True, default=False, help="Skip when completed output already exists.")
@click.option(
    "--sector-measurements",
    type=str,
    default=None,
    help="Path to JSON file containing top-level sector_measurements list for V21.",
)
def vet_command(
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
    toi: str | None,
    ra_deg: float | None,
    dec_deg: float | None,
    require_coordinates: bool,
    preset: str,
    checks: tuple[str, ...],
    network_ok: bool,
    fetch_tpf: bool,
    require_tpf: bool,
    tpf_sector_strategy: str,
    tpf_sectors: tuple[int, ...],
    sectors: tuple[int, ...],
    flux_type: str,
    timeout_seconds: float | None,
    random_seed: int | None,
    detrend: str | None,
    detrend_bin_hours: float,
    detrend_buffer: float,
    detrend_sigma_clip: float,
    extra_params: tuple[str, ...],
    fail_fast: bool,
    emit_warnings: bool,
    output_path_arg: str,
    progress_path: str | None,
    resume: bool,
    sector_measurements: str | None,
) -> None:
    """Run candidate vetting and emit `VettingBundleResult` JSON.

    See `docs/quickstart.rst` for agent quickstart and `docs/api.rst` for API recipes.
    """
    out_path = resolve_optional_output_path(output_path_arg)
    progress_file = _progress_path_from_args(out_path, progress_path, resume)

    (
        resolved_tic_id,
        resolved_period_days,
        resolved_t0_btjd,
        resolved_duration_hours,
        resolved_depth_ppm,
        input_resolution,
    ) = _resolve_candidate_inputs(
        network_ok=network_ok,
        toi=toi,
        tic_id=tic_id,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_ppm=depth_ppm,
    )
    resolved_ra, resolved_dec, coordinate_resolution = _resolve_coordinates(
        tic_id=resolved_tic_id,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        network_ok=network_ok,
        require_coordinates=require_coordinates,
    )
    candidate_meta = {
        "tic_id": resolved_tic_id,
        "period_days": resolved_period_days,
        "t0_btjd": resolved_t0_btjd,
        "duration_hours": resolved_duration_hours,
    }
    sector_measurements_rows: list[dict[str, Any]] | None = None
    sector_measurements_provenance: dict[str, Any] | None = None
    if sector_measurements:
        sector_measurements_rows, sector_measurements_provenance = _load_sector_measurements(
            sector_measurements
        )

    if tpf_sectors and str(tpf_sector_strategy).lower() != "requested":
        raise BtvCliError(
            "--tpf-sector requires --tpf-sector-strategy=requested",
            exit_code=EXIT_INPUT_ERROR,
        )
    effective_fetch_tpf = bool(fetch_tpf or require_tpf)
    detrend_method = _normalize_detrend_method(detrend)
    if detrend_method is not None:
        _validate_detrend_args(
            detrend_bin_hours=float(detrend_bin_hours),
            detrend_buffer=float(detrend_buffer),
            detrend_sigma_clip=float(detrend_sigma_clip),
        )

    if resume:
        existing = None
        if progress_file is not None:
            try:
                existing = read_progress_metadata(progress_file)
            except ProgressIOError as exc:
                raise BtvCliError(str(exc), exit_code=EXIT_PROGRESS_ERROR) from exc

        decision = decide_resume_for_single_candidate(
            command="vet",
            candidate=candidate_meta,
            resume=True,
            output_exists=bool(out_path and out_path.exists()),
            progress=existing,
        )
        if decision["resume"]:
            if progress_file is not None:
                skipped = build_single_candidate_progress(
                    command="vet",
                    output_path=str(out_path or "stdout"),
                    candidate=candidate_meta,
                    resume=True,
                    status="skipped_resume",
                )
                try:
                    write_progress_metadata_atomic(progress_file, skipped)
                except ProgressIOError as exc:
                    raise BtvCliError(str(exc), exit_code=EXIT_PROGRESS_ERROR) from exc
            return

    started = time.monotonic()
    if progress_file is not None:
        running = build_single_candidate_progress(
            command="vet",
            output_path=str(out_path or "stdout"),
            candidate=candidate_meta,
            resume=resume,
            status="running",
        )
        try:
            write_progress_metadata_atomic(progress_file, running)
        except ProgressIOError as exc:
            raise BtvCliError(str(exc), exit_code=EXIT_PROGRESS_ERROR) from exc

    config = PipelineConfig(
        timeout_seconds=timeout_seconds,
        random_seed=random_seed,
        emit_warnings=emit_warnings,
        fail_fast=fail_fast,
        extra_params=parse_extra_params(extra_params),
    )

    try:
        payload = _execute_vet(
            tic_id=resolved_tic_id,
            period_days=resolved_period_days,
            t0_btjd=resolved_t0_btjd,
            duration_hours=resolved_duration_hours,
            depth_ppm=resolved_depth_ppm,
            toi=toi,
            ra_deg=resolved_ra,
            dec_deg=resolved_dec,
            preset=str(preset).lower(),
            checks=list(checks) if checks else None,
            network_ok=network_ok,
            fetch_tpf=effective_fetch_tpf,
            require_tpf=require_tpf,
            tpf_sector_strategy=str(tpf_sector_strategy).lower(),
            tpf_sectors=list(tpf_sectors) if tpf_sectors else None,
            sectors=list(sectors) if sectors else None,
            flux_type=str(flux_type).lower(),
            pipeline_config=config,
            detrend=detrend_method,
            detrend_bin_hours=float(detrend_bin_hours),
            detrend_buffer=float(detrend_buffer),
            detrend_sigma_clip=float(detrend_sigma_clip),
            input_resolution=input_resolution,
            coordinate_resolution=coordinate_resolution,
            sector_measurements=sector_measurements_rows,
        )
        payload = _apply_cli_payload_contract(
            payload=payload,
            toi=toi,
            input_resolution=input_resolution,
            coordinate_resolution=coordinate_resolution,
        )
        if sector_measurements_rows is not None and sector_measurements is not None:
            payload["sector_measurements"] = sector_measurements_rows
            payload["sector_gating"] = _build_sector_gating_block(
                payload=payload,
                sector_measurements=sector_measurements_rows,
                source_path=sector_measurements,
                source_provenance=sector_measurements_provenance,
            )
        dump_json_output(payload, out_path)

        if progress_file is not None:
            completed = build_single_candidate_progress(
                command="vet",
                output_path=str(out_path or "stdout"),
                candidate=candidate_meta,
                resume=resume,
                status="completed",
                wall_time_seconds=time.monotonic() - started,
            )
            write_progress_metadata_atomic(progress_file, completed)
    except BtvCliError:
        raise
    except ProgressIOError as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_PROGRESS_ERROR) from exc
    except LightCurveNotFoundError as exc:
        if progress_file is not None:
            errored = build_single_candidate_progress(
                command="vet",
                output_path=str(out_path or "stdout"),
                candidate=candidate_meta,
                resume=resume,
                status="error",
                wall_time_seconds=time.monotonic() - started,
                error_class=type(exc).__name__,
                error_message=str(exc),
            )
            with suppress(ProgressIOError):
                write_progress_metadata_atomic(progress_file, errored)
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except Exception as exc:
        if progress_file is not None:
            errored = build_single_candidate_progress(
                command="vet",
                output_path=str(out_path or "stdout"),
                candidate=candidate_meta,
                resume=resume,
                status="error",
                wall_time_seconds=time.monotonic() - started,
                error_class=type(exc).__name__,
                error_message=str(exc),
            )
            with suppress(ProgressIOError):
                write_progress_metadata_atomic(progress_file, errored)
        mapped = EXIT_REMOTE_TIMEOUT if _looks_like_timeout(exc) else EXIT_RUNTIME_ERROR
        raise BtvCliError(str(exc), exit_code=mapped) from exc


__all__ = ["vet_command"]
