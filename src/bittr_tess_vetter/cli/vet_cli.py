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
from bittr_tess_vetter.api.report_vet_reuse import build_report_with_vet_artifact
from bittr_tess_vetter.api.stitch import stitch_lightcurve_data
from bittr_tess_vetter.api.transit_masks import (
    get_in_transit_mask,
    get_out_of_transit_mask,
    measure_transit_depth,
)
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve, TPFStamp
from bittr_tess_vetter.api.vet import vet_candidate
from bittr_tess_vetter.cli.stellar_inputs import load_auto_stellar_with_fallback, resolve_stellar_inputs
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
from bittr_tess_vetter.domain.target import StellarParameters
from bittr_tess_vetter.platform.catalogs.toi_resolution import (
    LookupStatus,
    lookup_tic_coordinates,
    resolve_toi_to_tic_ephemeris_depth,
)
from bittr_tess_vetter.platform.io import TargetNotFoundError
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError, MASTClient
from bittr_tess_vetter.report import build_vet_lc_summary_blocks
from bittr_tess_vetter.validation.result_schema import VettingBundleResult

CLI_VET_SCHEMA_VERSION = "cli.vet.v2"
CLI_VET_PLOT_DATA_SCHEMA_VERSION = "cli.vet.plot_data.v1"
CONFIDENCE_SEMANTICS_DOC = "docs/verification/confidence_semantics.md"


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


_HIGH_SALIENCE_FLAGS = {
    "MODEL_PREFERS_NON_TRANSIT",
    "V17_REGIME_MARGINAL",
    "V17_REGIME_CONFUSED",
    "DIFFIMG_UNRELIABLE",
    "DIFFIMG_TARGET_DEPTH_NONPOSITIVE",
    "DIFFIMG_MAX_DEPTH_NONPOSITIVE",
}
_NETWORK_ERROR_SKIP_FLAGS = {"SKIPPED:NETWORK_TIMEOUT", "SKIPPED:NETWORK_ERROR"}
_SUPPORTED_DETREND_METHODS: tuple[str, ...] = ("transit_masked_bin_median",)
_V04_CHI2_UNSTABLE_THRESHOLD = 3.0
_V04_SCATTER_RATIO_UNSTABLE_THRESHOLD = 0.5
_V04_DOM_RATIO_THRESHOLD = 1.5


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


def _load_auto_stellar_inputs(
    tic_id: int,
    toi: str | None = None,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    return load_auto_stellar_with_fallback(tic_id=int(tic_id), toi=toi)


def _build_stellar_block(
    *,
    resolved_stellar: dict[str, float | None],
    stellar_resolution: dict[str, Any],
) -> dict[str, Any]:
    sources_raw = stellar_resolution.get("sources")
    sources = sources_raw if isinstance(sources_raw, dict) else {}

    mapped_sources = {"explicit": "user", "file": "file", "auto": "tic_catalog"}
    present_field_sources: set[str] = set()
    for field in ("radius", "mass", "tmag"):
        if resolved_stellar.get(field) is None:
            continue
        source = str(sources.get(field, "missing"))
        present_field_sources.add(mapped_sources.get(source, source))

    if len(present_field_sources) == 1:
        source_label = next(iter(present_field_sources))
    elif len(present_field_sources) > 1:
        source_label = "mixed"
    else:
        source_label = "missing"

    radius_source = str(sources.get("radius", "missing"))
    mass_source = str(sources.get("mass", "missing"))
    quality = "catalog_estimate" if ("auto" in {radius_source, mass_source}) else "explicit"

    missing: list[str] = []
    if resolved_stellar.get("radius") is None:
        missing.append("radius_rsun")
    if resolved_stellar.get("mass") is None:
        missing.append("mass_msun")
    if resolved_stellar.get("tmag") is None:
        missing.append("tmag")
    missing.extend(["teff_k", "logg_cgs"])

    return {
        "teff_k": None,
        "logg_cgs": None,
        "radius_rsun": resolved_stellar.get("radius"),
        "mass_msun": resolved_stellar.get("mass"),
        "tmag": resolved_stellar.get("tmag"),
        "source": source_label,
        "quality": quality,
        "missing": missing,
    }


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
    n_network_errors = 0

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
            if any(str(flag) in _NETWORK_ERROR_SKIP_FLAGS for flag in flags):
                n_network_errors += 1

        # U19: surface extreme V04 depth-instability concerns in root summary.
        if check_id == "V04":
            metrics = row.get("metrics")
            if isinstance(metrics, dict):
                mean_depth = _to_optional_float(metrics.get("mean_depth_ppm"))
                scatter = _to_optional_float(metrics.get("depth_scatter_ppm"))
                chi2_reduced = _to_optional_float(metrics.get("chi2_reduced"))
                dom_ratio = _to_optional_float(metrics.get("dom_ratio"))

                unstable = False
                if chi2_reduced is not None and chi2_reduced > _V04_CHI2_UNSTABLE_THRESHOLD:
                    unstable = True
                if (
                    mean_depth is not None
                    and scatter is not None
                    and abs(mean_depth) > 0.0
                    and (abs(scatter) / abs(mean_depth)) > _V04_SCATTER_RATIO_UNSTABLE_THRESHOLD
                ):
                    unstable = True

                if unstable:
                    concerns.add("DEPTH_HIGHLY_UNSTABLE")
                    if check_id:
                        flagged_checks.add(check_id)

                if dom_ratio is not None and dom_ratio > _V04_DOM_RATIO_THRESHOLD:
                    concerns.add("DEPTH_DOMINATED_BY_SINGLE_EPOCH")
                    if check_id:
                        flagged_checks.add(check_id)

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
        "n_network_errors": int(n_network_errors),
        "n_flagged": int(len(flagged_checks)),
        "flagged_checks": sorted(flagged_checks),
        "concerns": sorted(concerns),
        "disposition_hint": disposition_hint,
    }


def _derive_vet_verdict(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    summary = payload.get("summary")
    if isinstance(summary, dict):
        disposition_hint = summary.get("disposition_hint")
        if disposition_hint is not None:
            return str(disposition_hint), "$.summary.disposition_hint"

        n_failed = _to_optional_float(summary.get("n_failed"))
        n_skipped = _to_optional_float(summary.get("n_skipped"))
        if n_failed is not None and n_failed > 0.0:
            return "needs_failed_checks_review", "$.summary.n_failed"
        if n_skipped is not None and n_skipped > 0.0:
            return "needs_additional_data", "$.summary.n_skipped"
        if n_failed is not None and n_skipped is not None:
            return "all_clear", "$.summary"
    return None, None


def _apply_cli_payload_contract(
    *,
    payload: dict[str, Any],
    toi: str | None,
    input_resolution: dict[str, Any] | None,
    coordinate_resolution: dict[str, Any] | None,
) -> dict[str, Any]:
    payload["schema_version"] = CLI_VET_SCHEMA_VERSION

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
    inputs_summary["confidence_semantics_ref"] = CONFIDENCE_SEMANTICS_DOC

    provenance_raw = payload.get("provenance")
    provenance = provenance_raw if isinstance(provenance_raw, dict) else {}
    provenance["confidence_semantics_ref"] = CONFIDENCE_SEMANTICS_DOC
    payload["provenance"] = provenance

    resolved_toi = _extract_resolved_toi(payload)
    effective_toi = toi if toi is not None else resolved_toi
    if input_resolution is not None and effective_toi is not None:
        inputs = inputs_summary.get("input_resolution", {}).get("inputs")
        if isinstance(inputs, dict):
            inputs["toi"] = effective_toi

    payload["summary"] = _build_root_summary(payload=payload)
    verdict, verdict_source = _derive_vet_verdict(payload)
    payload["verdict"] = verdict
    payload["verdict_source"] = verdict_source

    result_raw = payload.get("result")
    if isinstance(result_raw, dict):
        result_payload = result_raw
    else:
        result_payload = {}
        payload["result"] = result_payload
    result_payload["verdict"] = verdict
    result_payload["verdict_source"] = verdict_source
    return payload


def _split_vet_plot_data_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    core_payload = dict(payload)
    results_raw = core_payload.get("results")
    if not isinstance(results_raw, list):
        return core_payload, {"schema_version": CLI_VET_PLOT_DATA_SCHEMA_VERSION, "checks": []}

    split_checks: list[dict[str, Any]] = []
    updated_results: list[dict[str, Any]] = []
    for row in results_raw:
        if not isinstance(row, dict):
            updated_results.append(row)
            continue
        row_copy = dict(row)
        if "plot_data" in row_copy:
            check_id = row_copy.get("id")
            split_checks.append(
                {
                    "id": str(check_id) if check_id is not None else None,
                    "plot_data": row_copy.pop("plot_data"),
                }
            )
            row_copy["plot_data_ref"] = {
                "check_id": str(check_id) if check_id is not None else None,
            }
        updated_results.append(row_copy)
    core_payload["results"] = updated_results
    return core_payload, {"schema_version": CLI_VET_PLOT_DATA_SCHEMA_VERSION, "checks": split_checks}


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


def _lc_summary_inputs_sufficient(*, lc: LightCurve, candidate: Candidate) -> bool:
    time = np.asarray(lc.time, dtype=np.float64) if lc.time is not None else np.asarray([], dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64) if lc.flux is not None else np.asarray([], dtype=np.float64)
    if time.size == 0 or flux.size == 0 or time.size != flux.size:
        return False
    valid = np.isfinite(time) & np.isfinite(flux)
    if not np.any(valid):
        return False
    eph = candidate.ephemeris
    if not (
        np.isfinite(float(eph.period_days))
        and np.isfinite(float(eph.t0_btjd))
        and np.isfinite(float(eph.duration_hours))
    ):
        return False
    if float(eph.period_days) <= 0.0 or float(eph.duration_hours) <= 0.0:
        return False
    return True


def _attach_lc_summary_payload(
    *,
    payload: dict[str, Any],
    lc: LightCurve,
    candidate: Candidate,
    bundle: VettingBundleResult,
    stellar_params: StellarParameters | None,
    tic_id: int,
    toi: str | None,
    include_lc_summary: bool,
) -> None:
    meta: dict[str, Any] = {
        "enabled": True,
        "computed": False,
        "reason": "compute_failed",
        "reason_unavailable": "compute_failed",
        "source": "report.vet_lc_summary",
        "schema_version": "1",
    }
    payload["lc_summary"] = None

    if not include_lc_summary:
        meta["enabled"] = False
        meta["reason"] = "disabled_by_flag"
        meta["reason_unavailable"] = "disabled_by_flag"
        payload["lc_summary_meta"] = meta
        return

    if not _lc_summary_inputs_sufficient(lc=lc, candidate=candidate):
        meta["reason"] = "insufficient_inputs"
        meta["reason_unavailable"] = "insufficient_inputs"
        payload["lc_summary_meta"] = meta
        return

    try:
        report, _ = build_report_with_vet_artifact(
            lc=lc,
            candidate=candidate,
            vet_bundle=bundle,
            stellar=stellar_params,
            tic_id=tic_id,
            toi=toi,
            include_additional_plots=True,
            include_lc_robustness=True,
        )
        summary_blocks = build_vet_lc_summary_blocks(report)
        if isinstance(summary_blocks, dict):
            payload["lc_summary"] = dict(summary_blocks)
            payload["lc_summary_meta"] = {
                "enabled": True,
                "computed": True,
                "reason": None,
                "reason_unavailable": None,
                "source": "report.vet_lc_summary",
                "schema_version": "1",
            }
            return
    except Exception as exc:
        meta["reason"] = "component_timeout" if _looks_like_timeout(exc) else "compute_failed"
        meta["reason_unavailable"] = str(meta["reason"])
        payload["lc_summary_meta"] = meta
        return

    payload["lc_summary_meta"] = meta


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
    cache_dir: Path | None,
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
    include_lc_summary: bool,
    input_resolution: dict[str, Any] | None = None,
    coordinate_resolution: dict[str, Any] | None = None,
    sector_measurements: list[dict[str, Any]] | None = None,
    stellar_params: StellarParameters | None = None,
    stellar_block: dict[str, Any] | None = None,
    stellar_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    client = MASTClient(cache_dir=str(cache_dir)) if cache_dir is not None else MASTClient()
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
    if toi is not None or sector_measurements is not None or stellar_block is not None:
        context = {}
        if toi is not None:
            context["toi"] = toi
        if sector_measurements is not None:
            context["sector_measurements"] = sector_measurements
        if stellar_block is not None:
            context["stellar"] = stellar_block

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
        depth_source = "transit_masked_in_out_median"
        depth_availability = "unavailable"
        depth_note = "insufficient_in_or_out_of_transit_points"
        depth_estimate_ppm: float | None = None
        depth_err_estimate_ppm: float | None = None

        time_arr = np.asarray(vet_lc.time, dtype=np.float64)
        flux_arr = np.asarray(vet_lc.flux, dtype=np.float64)
        valid_mask = (
            np.asarray(vet_lc.valid_mask, dtype=bool)
            if vet_lc.valid_mask is not None
            else np.ones_like(flux_arr, dtype=bool)
        )
        finite_mask = np.isfinite(time_arr) & np.isfinite(flux_arr) & valid_mask

        in_mask = get_in_transit_mask(
            time_arr,
            float(period_days),
            float(t0_btjd),
            float(duration_hours),
        ) & finite_mask
        out_mask = get_out_of_transit_mask(
            time_arr,
            float(period_days),
            float(t0_btjd),
            float(duration_hours),
            buffer_factor=float(detrend_buffer),
        ) & finite_mask

        n_in = int(np.sum(in_mask))
        n_out = int(np.sum(out_mask))
        if n_in > 0 and n_out > 0:
            depth_frac, depth_err_frac = measure_transit_depth(flux_arr, in_mask, out_mask)
            measured_depth_ppm = float(depth_frac * 1_000_000.0)
            measured_depth_err_ppm = float(depth_err_frac * 1_000_000.0)
            if np.isfinite(measured_depth_ppm) and np.isfinite(measured_depth_err_ppm):
                if measured_depth_ppm > 0.0:
                    depth_estimate_ppm = measured_depth_ppm
                    depth_err_estimate_ppm = measured_depth_err_ppm
                    depth_availability = "available"
                    depth_note = ""
                else:
                    depth_note = "non_positive_depth_from_transit_mask_measurement"
            else:
                depth_note = "non_finite_depth_from_transit_mask_measurement"

        detrend_provenance["depth_source"] = depth_source
        detrend_provenance["depth_availability"] = depth_availability
        detrend_provenance["depth_ppm"] = depth_estimate_ppm
        detrend_provenance["depth_err_ppm"] = depth_err_estimate_ppm
        if depth_note:
            detrend_provenance["depth_note"] = depth_note

    bundle = vet_candidate(
        lc=vet_lc,
        candidate=candidate,
        stellar=stellar_params,
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
    provenance_raw = payload.get("provenance")
    provenance = provenance_raw if isinstance(provenance_raw, dict) else {}
    provenance["sectors_used"] = [int(s) for s in sectors_used]
    provenance["discovered_sectors"] = [int(s) for s in sectors_used]
    provenance["sectors_requested"] = [int(s) for s in sectors] if sectors is not None else None
    payload["provenance"] = provenance
    if detrend_provenance is not None:
        provenance["detrend"] = detrend_provenance
        payload["provenance"] = provenance
    if stellar_block is not None:
        payload["stellar"] = stellar_block
    if stellar_resolution is not None:
        provenance_raw = payload.get("provenance")
        provenance = provenance_raw if isinstance(provenance_raw, dict) else {}
        provenance["stellar"] = stellar_resolution
        payload["provenance"] = provenance
    if stellar_block is not None and str(stellar_block.get("source")) == "missing" and bool(network_ok):
        warnings_raw = payload.get("warnings")
        warnings = warnings_raw if isinstance(warnings_raw, list) else []
        warnings.append(
            "Stellar parameters unavailable from TIC/MAST or ExoFOP. Retry recommended, "
            "or provide --stellar-file / --stellar-radius / --stellar-mass."
        )
        payload["warnings"] = warnings
    _attach_lc_summary_payload(
        payload=payload,
        lc=vet_lc,
        candidate=candidate,
        bundle=bundle,
        stellar_params=stellar_params,
        tic_id=tic_id,
        toi=toi,
        include_lc_summary=bool(include_lc_summary),
    )
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
@click.argument("toi_arg", required=False)
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--period-days", type=float, default=None, help="Orbital period in days.")
@click.option("--t0-btjd", type=float, default=None, help="Reference epoch in BTJD.")
@click.option("--duration-hours", type=float, default=None, help="Transit duration in hours.")
@click.option("--depth-ppm", type=float, default=None, help="Transit depth in ppm.")
@click.option("--toi", type=str, default=None, help="Optional TOI label (overrides resolved value).")
@click.option("--report-file", type=str, default=None, help="Optional report JSON path for candidate inputs.")
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
    "--cache-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Optional cache directory for MAST/lightkurve products.",
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
@click.option("--stellar-radius", type=float, default=None, help="Stellar radius (Rsun).")
@click.option("--stellar-mass", type=float, default=None, help="Stellar mass (Msun).")
@click.option("--stellar-tmag", type=float, default=None, help="TESS magnitude.")
@click.option("--stellar-file", type=str, default=None, help="JSON file with stellar inputs.")
@click.option(
    "--use-stellar-auto/--no-use-stellar-auto",
    default=False,
    show_default=True,
    help="Resolve stellar inputs from TIC when missing from explicit/file inputs.",
)
@click.option(
    "--require-stellar/--no-require-stellar",
    default=False,
    show_default=True,
    help="Fail unless stellar radius and mass resolve.",
)
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
@click.option(
    "--include-lc-summary/--no-lc-summary",
    default=True,
    show_default=True,
    help="Attach top-level lc_summary computed through the report seam.",
)
@click.option("--extra-param", "extra_params", multiple=True, help="Repeat KEY=VALUE entries.")
@click.option("--fail-fast/--no-fail-fast", default=False, show_default=True)
@click.option("--emit-warnings/--no-emit-warnings", default=False, show_default=True)
@click.option(
    "-o",
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
@click.option(
    "--split-plot-data/--no-split-plot-data",
    default=True,
    show_default=True,
    help="Write per-check plot_data into <out>.plot_data.json sidecar for file outputs.",
)
@click.option("--progress-path", type=str, default=None, help="Optional progress metadata path.")
@click.option("--resume", is_flag=True, default=False, help="Skip when completed output already exists.")
@click.option(
    "--sector-measurements",
    type=str,
    default=None,
    help="Path to JSON file containing top-level sector_measurements list for V21.",
)
@click.option(
    "--auto-measure-sectors/--no-auto-measure-sectors",
    default=False,
    show_default=True,
    help="Inline-run measure-sectors when --sector-measurements is not provided.",
)
def vet_command(
    toi_arg: str | None,
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
    toi: str | None,
    report_file: str | None,
    ra_deg: float | None,
    dec_deg: float | None,
    require_coordinates: bool,
    preset: str,
    checks: tuple[str, ...],
    network_ok: bool,
    cache_dir: Path | None,
    fetch_tpf: bool,
    require_tpf: bool,
    tpf_sector_strategy: str,
    tpf_sectors: tuple[int, ...],
    sectors: tuple[int, ...],
    flux_type: str,
    timeout_seconds: float | None,
    random_seed: int | None,
    stellar_radius: float | None,
    stellar_mass: float | None,
    stellar_tmag: float | None,
    stellar_file: str | None,
    use_stellar_auto: bool,
    require_stellar: bool,
    detrend: str | None,
    detrend_bin_hours: float,
    detrend_buffer: float,
    detrend_sigma_clip: float,
    include_lc_summary: bool,
    extra_params: tuple[str, ...],
    fail_fast: bool,
    emit_warnings: bool,
    output_path_arg: str,
    split_plot_data: bool,
    progress_path: str | None,
    resume: bool,
    sector_measurements: str | None,
    auto_measure_sectors: bool,
) -> None:
    """Run candidate vetting and emit `VettingBundleResult` JSON.

    See `docs/quickstart.rst` for agent quickstart and `docs/api.rst` for API recipes.
    """
    out_path = resolve_optional_output_path(output_path_arg)
    progress_file = _progress_path_from_args(out_path, progress_path, resume)
    if (
        report_file is None
        and toi_arg is not None
        and toi is not None
        and str(toi_arg).strip() != str(toi).strip()
    ):
        raise BtvCliError(
            "Positional TOI argument and --toi must match when both are provided.",
            exit_code=EXIT_INPUT_ERROR,
        )
    resolved_toi_arg = toi if toi is not None else toi_arg

    from bittr_tess_vetter.cli.diagnostics_report_inputs import (
        choose_effective_sectors,
        resolve_inputs_from_report_file,
    )

    report_file_path: str | None = None
    report_sectors_used: list[int] | None = None
    if report_file is not None:
        if resolved_toi_arg is not None:
            click.echo(
                "Warning: --report-file provided; ignoring --toi and using report-file candidate inputs.",
                err=True,
            )
        resolved_from_report = resolve_inputs_from_report_file(str(report_file))
        resolved_tic_id = int(resolved_from_report.tic_id)
        resolved_period_days = float(resolved_from_report.period_days)
        resolved_t0_btjd = float(resolved_from_report.t0_btjd)
        resolved_duration_hours = float(resolved_from_report.duration_hours)
        resolved_depth_ppm = (
            float(resolved_from_report.depth_ppm)
            if resolved_from_report.depth_ppm is not None
            else None
        )
        input_resolution = dict(resolved_from_report.input_resolution)
        report_file_path = str(resolved_from_report.report_file_path)
        report_sectors_used = (
            [int(s) for s in resolved_from_report.sectors_used]
            if resolved_from_report.sectors_used is not None
            else None
        )
    else:
        (
            resolved_tic_id,
            resolved_period_days,
            resolved_t0_btjd,
            resolved_duration_hours,
            resolved_depth_ppm,
            input_resolution,
        ) = _resolve_candidate_inputs(
            network_ok=network_ok,
            toi=resolved_toi_arg,
            tic_id=tic_id,
            period_days=period_days,
            t0_btjd=t0_btjd,
            duration_hours=duration_hours,
            depth_ppm=depth_ppm,
        )

    effective_sectors, _sectors_explicit, sector_selection_source = choose_effective_sectors(
        sectors_arg=sectors,
        report_sectors_used=report_sectors_used,
    )
    effective_toi = None if report_file is not None else resolved_toi_arg
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

    sector_measurements_rows: list[dict[str, Any]] | None = None
    sector_measurements_provenance: dict[str, Any] | None = None
    sector_measurements_source: str | None = None
    auto_measure_warning: str | None = None
    if sector_measurements:
        sector_measurements_rows, sector_measurements_provenance = _load_sector_measurements(
            sector_measurements
        )
        sector_measurements_source = str(sector_measurements)
    elif auto_measure_sectors:
        from bittr_tess_vetter.cli.measure_sectors_cli import _execute_measure_sectors

        try:
            measured = _execute_measure_sectors(
                tic_id=int(resolved_tic_id),
                period_days=float(resolved_period_days),
                t0_btjd=float(resolved_t0_btjd),
                duration_hours=float(resolved_duration_hours),
                sectors=[int(s) for s in effective_sectors] if effective_sectors else None,
                sectors_explicit=bool(effective_sectors is not None),
                sector_selection_source=str(sector_selection_source),
                flux_type=str(flux_type).lower(),
                detrend=detrend_method,
                detrend_bin_hours=float(detrend_bin_hours),
                detrend_buffer=float(detrend_buffer),
                detrend_sigma_clip=float(detrend_sigma_clip),
                cache_dir=cache_dir,
                input_resolution=input_resolution,
            )
            measured_rows = measured.get("sector_measurements")
            if not isinstance(measured_rows, list):
                raise BtvCliError(
                    "Auto sector measurements schema error: expected top-level sector_measurements list",
                    exit_code=EXIT_RUNTIME_ERROR,
                )
            sector_measurements_rows = [row for row in measured_rows if isinstance(row, dict)]
            measured_provenance = measured.get("provenance")
            if isinstance(measured_provenance, dict):
                sector_measurements_provenance = measured_provenance
            sector_measurements_source = "inline:auto_measure_sectors"
        except BtvCliError as exc:
            if fail_fast:
                raise
            auto_measure_warning = (
                "Warning: --auto-measure-sectors failed; continuing without sector measurements: "
                f"{exc}"
            )
            click.echo(auto_measure_warning, err=True)
        except (TargetNotFoundError, LightCurveNotFoundError) as exc:
            if fail_fast:
                raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
            auto_measure_warning = (
                "Warning: --auto-measure-sectors failed; continuing without sector measurements: "
                f"{exc}"
            )
            click.echo(auto_measure_warning, err=True)
        except Exception as exc:
            if fail_fast:
                mapped = EXIT_REMOTE_TIMEOUT if _looks_like_timeout(exc) else EXIT_RUNTIME_ERROR
                raise BtvCliError(str(exc), exit_code=mapped) from exc
            auto_measure_warning = (
                "Warning: --auto-measure-sectors failed; continuing without sector measurements: "
                f"{exc}"
            )
            click.echo(auto_measure_warning, err=True)
    if use_stellar_auto and not network_ok:
        raise BtvCliError("--use-stellar-auto requires --network-ok", exit_code=EXIT_DATA_UNAVAILABLE)

    try:
        resolved_stellar, stellar_resolution = resolve_stellar_inputs(
            tic_id=resolved_tic_id,
            stellar_radius=stellar_radius,
            stellar_mass=stellar_mass,
            stellar_tmag=stellar_tmag,
            stellar_file=stellar_file,
            use_stellar_auto=use_stellar_auto,
            require_stellar=require_stellar,
            auto_loader=(
                (lambda _tic_id: _load_auto_stellar_inputs(_tic_id, toi=effective_toi))
                if effective_toi is not None
                else (lambda _tic_id: _load_auto_stellar_inputs(_tic_id))
            )
            if use_stellar_auto
            else None,
        )
    except (TargetNotFoundError, LightCurveNotFoundError) as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except Exception as exc:
        if isinstance(exc, BtvCliError):
            raise
        mapped = EXIT_REMOTE_TIMEOUT if _looks_like_timeout(exc) else EXIT_RUNTIME_ERROR
        raise BtvCliError(str(exc), exit_code=mapped) from exc

    stellar_params: StellarParameters | None = None
    if any(resolved_stellar.get(key) is not None for key in ("radius", "mass", "tmag")):
        stellar_params = StellarParameters(
            radius=resolved_stellar.get("radius"),
            mass=resolved_stellar.get("mass"),
            tmag=resolved_stellar.get("tmag"),
        )
    stellar_block = _build_stellar_block(
        resolved_stellar=resolved_stellar,
        stellar_resolution=stellar_resolution,
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
            toi=effective_toi,
            ra_deg=resolved_ra,
            dec_deg=resolved_dec,
            preset=str(preset).lower(),
            checks=list(checks) if checks else None,
            network_ok=network_ok,
            cache_dir=cache_dir,
            fetch_tpf=effective_fetch_tpf,
            require_tpf=require_tpf,
            tpf_sector_strategy=str(tpf_sector_strategy).lower(),
            tpf_sectors=list(tpf_sectors) if tpf_sectors else None,
            sectors=[int(s) for s in effective_sectors] if effective_sectors else None,
            flux_type=str(flux_type).lower(),
            pipeline_config=config,
            detrend=detrend_method,
            detrend_bin_hours=float(detrend_bin_hours),
            detrend_buffer=float(detrend_buffer),
            detrend_sigma_clip=float(detrend_sigma_clip),
            include_lc_summary=bool(include_lc_summary),
            input_resolution=input_resolution,
            coordinate_resolution=coordinate_resolution,
            sector_measurements=sector_measurements_rows,
            stellar_params=stellar_params,
            stellar_block=stellar_block,
            stellar_resolution=stellar_resolution,
        )
        payload = _apply_cli_payload_contract(
            payload=payload,
            toi=effective_toi,
            input_resolution=input_resolution,
            coordinate_resolution=coordinate_resolution,
        )
        if auto_measure_warning is not None:
            warnings_raw = payload.get("warnings")
            warnings = warnings_raw if isinstance(warnings_raw, list) else []
            warnings.append(auto_measure_warning)
            payload["warnings"] = warnings
        provenance_raw = payload.get("provenance")
        provenance = provenance_raw if isinstance(provenance_raw, dict) else {}
        provenance["inputs_source"] = (
            "report_file" if report_file_path is not None else str(input_resolution.get("source"))
        )
        provenance["report_file"] = report_file_path
        provenance["sector_selection_source"] = sector_selection_source
        payload["provenance"] = provenance
        if sector_measurements_rows is not None and sector_measurements_source is not None:
            payload["sector_measurements"] = sector_measurements_rows
            payload["sector_gating"] = _build_sector_gating_block(
                payload=payload,
                sector_measurements=sector_measurements_rows,
                source_path=sector_measurements_source,
                source_provenance=sector_measurements_provenance,
            )
        payload_to_write = payload
        plot_data_sidecar_path: Path | None = None
        if bool(split_plot_data) and out_path is not None:
            payload_to_write, plot_data_payload = _split_vet_plot_data_payload(payload)
            plot_data_sidecar_path = out_path.with_suffix(out_path.suffix + ".plot_data.json")
            dump_json_output(plot_data_payload, plot_data_sidecar_path)

            provenance_raw = payload_to_write.get("provenance")
            provenance = provenance_raw if isinstance(provenance_raw, dict) else {}
            provenance["plot_data_split"] = True
            provenance["plot_data_path"] = str(plot_data_sidecar_path)
            payload_to_write["provenance"] = provenance
        else:
            provenance_raw = payload_to_write.get("provenance")
            provenance = provenance_raw if isinstance(provenance_raw, dict) else {}
            provenance["plot_data_split"] = False
            provenance["plot_data_path"] = None
            payload_to_write["provenance"] = provenance

        dump_json_output(payload_to_write, out_path)

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
