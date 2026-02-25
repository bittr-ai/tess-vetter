"""Shared helpers for seeding report inputs from prior report artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_REMOTE_TIMEOUT,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    load_json_file,
)
from tess_vetter.platform.catalogs.time_conventions import normalize_epoch_to_btjd
from tess_vetter.platform.catalogs.toi_resolution import (
    LookupStatus,
    resolve_toi_to_tic_ephemeris_depth,
)
from tess_vetter.platform.io.mast_client import LightCurveNotFoundError, MASTClient


@dataclass(frozen=True)
class ReportSeed:
    tic_id: int | None
    period_days: float | None
    t0_btjd: float | None
    duration_hours: float | None
    depth_ppm: float | None
    sectors_used: list[int] | None
    toi: str | None
    source_path: str


def _to_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_optional_t0_btjd(value: Any) -> float | None:
    raw = _to_optional_float(value)
    if raw is None:
        return None
    return normalize_epoch_to_btjd(raw)


def _to_optional_toi(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_optional_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    out: list[int] = []
    for item in value:
        parsed = _to_optional_int(item)
        if parsed is None:
            continue
        out.append(parsed)
    if not out:
        return []
    return sorted(set(out))


def _extract_sectors_used(*, cli_payload: dict[str, Any], report_payload: dict[str, Any]) -> list[int] | None:
    cli_prov = cli_payload.get("provenance")
    if isinstance(cli_prov, dict):
        cli_sectors = _to_optional_int_list(cli_prov.get("sectors_used"))
        if cli_sectors is not None:
            return cli_sectors

    report_prov = report_payload.get("provenance")
    if isinstance(report_prov, dict):
        report_sectors = _to_optional_int_list(report_prov.get("sectors_used"))
        if report_sectors is not None:
            return report_sectors
        pipeline = report_prov.get("pipeline")
        if isinstance(pipeline, dict):
            pipeline_sectors = _to_optional_int_list(pipeline.get("sectors_used"))
            if pipeline_sectors is not None:
                return pipeline_sectors

    summary = report_payload.get("summary")
    if isinstance(summary, dict):
        summary_sectors = _to_optional_int_list(summary.get("sectors_used"))
        if summary_sectors is not None:
            return summary_sectors
    return None


def load_report_seed(path: str | Path) -> ReportSeed:
    payload = load_json_file(Path(path), label="report file")
    report_payload = payload.get("report") if isinstance(payload.get("report"), dict) else payload
    if not isinstance(report_payload, dict):
        raise BtvCliError(
            "Report file schema error: expected object payload",
            exit_code=EXIT_INPUT_ERROR,
        )

    summary = report_payload.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    ephemeris = summary.get("ephemeris")
    if not isinstance(ephemeris, dict):
        ephemeris = {}

    return ReportSeed(
        tic_id=_to_optional_int(summary.get("tic_id")),
        period_days=_to_optional_float(
            ephemeris.get("period_days") if ephemeris.get("period_days") is not None else ephemeris.get("period")
        ),
        t0_btjd=_to_optional_t0_btjd(
            ephemeris.get("t0_btjd") if ephemeris.get("t0_btjd") is not None else ephemeris.get("t0")
        ),
        duration_hours=_to_optional_float(ephemeris.get("duration_hours")),
        depth_ppm=_to_optional_float(summary.get("input_depth_ppm")),
        sectors_used=_extract_sectors_used(
            cli_payload=payload if isinstance(payload, dict) else {},
            report_payload=report_payload,
        ),
        toi=_to_optional_toi(summary.get("toi")),
        source_path=str(path),
    )


def resolve_candidate_inputs_with_report_seed(
    *,
    network_ok: bool,
    toi: str | None,
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
    report_seed: ReportSeed | None,
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
    has_complete_manual_candidate = (
        tic_id is not None
        and period_days is not None
        and t0_btjd is not None
        and duration_hours is not None
    )
    has_complete_report_candidate = report_seed is not None and all(
        value is not None
        for value in (
            report_seed.tic_id,
            report_seed.period_days,
            report_seed.t0_btjd,
            report_seed.duration_hours,
        )
    )

    if toi is not None and network_ok and not (has_complete_manual_candidate or has_complete_report_candidate):
        toi_result = resolve_toi_to_tic_ephemeris_depth(toi)
        source = "toi_catalog"
        resolved_from = "exofop_toi_table"
        if toi_result.status != LookupStatus.OK:
            has_report_ephemeris = report_seed is not None and all(
                value is not None
                for value in (
                    report_seed.tic_id,
                    report_seed.period_days,
                    report_seed.t0_btjd,
                    report_seed.duration_hours,
                )
            )
            has_manual_ephemeris = all(
                value is not None for value in (tic_id, period_days, t0_btjd, duration_hours)
            )
            if not has_report_ephemeris and not has_manual_ephemeris:
                if toi_result.status == LookupStatus.TIMEOUT:
                    exit_code = EXIT_REMOTE_TIMEOUT
                elif toi_result.status == LookupStatus.DATA_UNAVAILABLE:
                    exit_code = EXIT_DATA_UNAVAILABLE
                else:
                    exit_code = EXIT_RUNTIME_ERROR
                raise BtvCliError(
                    toi_result.message or f"Failed to resolve TOI {toi}",
                    exit_code=exit_code,
                )
            errors.append(toi_result.message or f"TOI resolution degraded for {toi}")
        resolved["tic_id"] = toi_result.tic_id
        resolved["period_days"] = toi_result.period_days
        resolved["t0_btjd"] = toi_result.t0_btjd
        resolved["duration_hours"] = toi_result.duration_hours
        resolved["depth_ppm"] = toi_result.depth_ppm

    if report_seed is not None:
        report_inputs = {
            "tic_id": report_seed.tic_id,
            "period_days": report_seed.period_days,
            "t0_btjd": report_seed.t0_btjd,
            "duration_hours": report_seed.duration_hours,
            "depth_ppm": report_seed.depth_ppm,
        }
        report_applied = False
        for key, value in report_inputs.items():
            if value is None:
                continue
            if resolved.get(key) is not None:
                overrides.append(f"report_file:{key}")
            resolved[key] = value
            report_applied = True
        if report_applied:
            source = "report_file"
            resolved_from = "report_file"

    manual_inputs = {
        "tic_id": tic_id,
        "period_days": period_days,
        "t0_btjd": normalize_epoch_to_btjd(t0_btjd) if t0_btjd is not None else None,
        "duration_hours": duration_hours,
        "depth_ppm": depth_ppm,
    }
    for key, value in manual_inputs.items():
        if value is None:
            continue
        if resolved.get(key) is not None:
            overrides.append(key)
        resolved[key] = value

    if resolved["tic_id"] is None:
        if toi is not None and not network_ok and report_seed is None and tic_id is None:
            raise BtvCliError(
                "--toi requires --network-ok to resolve TIC/ephemeris when --tic-id or --report-file is not provided",
                exit_code=EXIT_DATA_UNAVAILABLE,
            )
        raise BtvCliError(
            "Missing TIC identifier. Provide --tic-id, --report-file, or --toi.",
            exit_code=EXIT_INPUT_ERROR,
        )

    missing_ephemeris = [
        name
        for name in ("period_days", "t0_btjd", "duration_hours")
        if resolved[name] is None
    ]
    if missing_ephemeris:
        if toi is not None and not network_ok:
            raise BtvCliError(
                "--toi requires --network-ok to resolve missing inputs",
                exit_code=EXIT_DATA_UNAVAILABLE,
            )
        if toi is not None and network_ok:
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
            "toi": toi if toi is not None else (report_seed.toi if report_seed is not None else None),
            "report_file": report_seed.source_path if report_seed is not None else None,
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


def detect_lightkurve_cache_dir() -> Path:
    cache = os.getenv("LIGHTKURVE_CACHE_DIR")
    if cache:
        return Path(cache).expanduser()
    return Path("~/.lightkurve/cache").expanduser()


def load_cache_only_lightcurves_for_sectors(
    *,
    tic_id: int,
    sectors: list[int],
    flux_type: str,
) -> list[Any]:
    if not sectors:
        raise BtvCliError("Cache-only loading requires explicit sectors", exit_code=EXIT_INPUT_ERROR)
    cache_dir = detect_lightkurve_cache_dir()
    client = MASTClient(cache_dir=str(cache_dir))
    loaded = []
    for sector in sorted({int(s) for s in sectors}):
        loaded.append(
            client.download_lightcurve_cached(
                tic_id=int(tic_id),
                sector=int(sector),
                flux_type=str(flux_type),
            )
        )
    if not loaded:
        raise LightCurveNotFoundError(
            f"No cached light curves found for TIC {int(tic_id)} in sectors {sorted({int(s) for s in sectors})}"
        )
    return loaded


__all__ = [
    "ReportSeed",
    "detect_lightkurve_cache_dir",
    "load_cache_only_lightcurves_for_sectors",
    "load_report_seed",
    "resolve_candidate_inputs_with_report_seed",
]
