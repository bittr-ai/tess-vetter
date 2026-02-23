"""`btv model-compete` command for model competition diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import numpy as np

from tess_vetter.api import model_competition as model_competition_api
from tess_vetter.api.detrend import median_detrend
from tess_vetter.api.stitch import stitch_lightcurve_data
from tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    emit_progress,
    resolve_optional_output_path,
)
from tess_vetter.cli.diagnostics_report_inputs import (
    choose_effective_sectors,
    load_lightcurves_with_sector_policy,
    resolve_inputs_from_report_file,
)
from tess_vetter.cli.vet_cli import (
    _detrend_lightcurve_for_vetting,
    _resolve_candidate_inputs,
    _validate_detrend_args,
)
from tess_vetter.platform.io.mast_client import LightCurveNotFoundError, TargetNotFoundError

_SUPPORTED_DETREND_METHODS: tuple[str, ...] = (
    "running_median_0p5d",
    "transit_masked_bin_median",
)


def _to_jsonable_result(result: Any) -> Any:
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json", exclude_none=True)
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return result


def _derive_model_compete_verdict(result_payload: dict[str, Any]) -> tuple[str | None, str | None]:
    interpretation_label = result_payload.get("interpretation_label")
    if interpretation_label is not None:
        return str(interpretation_label), "$.result.interpretation_label"
    model_competition = result_payload.get("model_competition")
    if isinstance(model_competition, dict):
        nested_label = model_competition.get("interpretation_label")
        if nested_label is not None:
            return str(nested_label), "$.result.model_competition.interpretation_label"
    return None, None


def _normalize_detrend_method(detrend: str | None) -> str | None:
    if detrend is None:
        return None
    method = str(detrend).strip().lower()
    if method == "":
        return None
    if method == "running_median_0.5d":
        return "running_median_0p5d"
    if method not in _SUPPORTED_DETREND_METHODS:
        choices = ", ".join(_SUPPORTED_DETREND_METHODS)
        raise BtvCliError(
            f"--detrend must be one of: {choices}",
            exit_code=EXIT_INPUT_ERROR,
        )
    return method


def _running_median_window_cadences(time: np.ndarray, *, window_days: float) -> int:
    if len(time) < 3:
        return 3
    cadence_days = float(np.nanmedian(np.diff(time)))
    if not np.isfinite(cadence_days) or cadence_days <= 0.0:
        return 3
    window_cadences = max(3, int(float(window_days) / cadence_days))
    if window_cadences % 2 == 0:
        window_cadences += 1
    return int(window_cadences)


def _maybe_detrend_lightcurve(
    *,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    detrend_method: str | None,
    detrend_bin_hours: float,
    detrend_buffer: float,
    detrend_sigma_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    if detrend_method is None:
        return time, flux, flux_err, None

    if detrend_method == "running_median_0p5d":
        window_days = 0.5
        window_cadences = _running_median_window_cadences(time, window_days=window_days)
        detrended_flux = np.asarray(median_detrend(flux, window=window_cadences), dtype=np.float64)
        return (
            np.asarray(time, dtype=np.float64),
            detrended_flux,
            np.asarray(flux_err, dtype=np.float64),
            {
                "applied": True,
                "method": "running_median_0p5d",
                "window_days": float(window_days),
                "window_cadences": int(window_cadences),
                "bin_hours": float(detrend_bin_hours),
                "buffer_factor": float(detrend_buffer),
                "sigma_clip": float(detrend_sigma_clip),
            },
        )

    lc = LightCurve(
        time=np.asarray(time, dtype=np.float64),
        flux=np.asarray(flux, dtype=np.float64),
        flux_err=np.asarray(flux_err, dtype=np.float64),
        quality=None,
        valid_mask=np.ones_like(flux, dtype=bool),
    )
    candidate = Candidate(
        ephemeris=Ephemeris(
            period_days=float(period_days),
            t0_btjd=float(t0_btjd),
            duration_hours=float(duration_hours),
        ),
        depth_ppm=100.0,
    )
    detrended_lc, detrend_provenance = _detrend_lightcurve_for_vetting(
        lc=lc,
        candidate=candidate,
        method=str(detrend_method),
        bin_hours=float(detrend_bin_hours),
        buffer_factor=float(detrend_buffer),
        clip_sigma=float(detrend_sigma_clip),
    )
    return (
        np.asarray(detrended_lc.time, dtype=np.float64),
        np.asarray(detrended_lc.flux, dtype=np.float64),
        np.asarray(detrended_lc.flux_err, dtype=np.float64),
        detrend_provenance,
    )


def _download_and_prepare_arrays(
    *,
    tic_id: int,
    sectors: list[int] | None,
    sectors_explicit: bool,
    flux_type: str,
    network_ok: bool,
    cache_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], str]:
    lightcurves, sector_load_path = load_lightcurves_with_sector_policy(
        tic_id=int(tic_id),
        sectors=sectors,
        flux_type=str(flux_type).lower(),
        explicit_sectors=bool(sectors_explicit),
        network_ok=bool(network_ok),
        cache_dir=cache_dir,
    )

    if len(lightcurves) == 1:
        lc = lightcurves[0]
    else:
        lc, _stitched = stitch_lightcurve_data(lightcurves, tic_id=int(tic_id))

    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)
    if lc.flux_err is None:
        fallback_err = float(np.nanstd(flux))
        if not np.isfinite(fallback_err) or fallback_err <= 0.0:
            fallback_err = 1e-3
        flux_err = np.full_like(flux, fallback_err, dtype=np.float64)
    else:
        flux_err = np.asarray(lc.flux_err, dtype=np.float64)
    quality = (
        np.asarray(lc.quality, dtype=np.int32)
        if lc.quality is not None
        else np.zeros_like(flux, dtype=np.int32)
    )
    valid_mask = (
        np.asarray(lc.valid_mask, dtype=bool)
        if lc.valid_mask is not None
        else np.ones_like(flux, dtype=bool)
    )

    valid = (
        valid_mask
        & (quality == 0)
        & np.isfinite(time)
        & np.isfinite(flux)
        & np.isfinite(flux_err)
    )
    if not np.any(valid):
        raise LightCurveNotFoundError(f"No valid finite cadences available for TIC {tic_id}")

    sectors_used = sorted({int(item.sector) for item in lightcurves if getattr(item, "sector", None) is not None})
    return time[valid], flux[valid], flux_err[valid], sectors_used, sector_load_path


@click.command("model-compete")
@click.argument("toi_arg", required=False)
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--period-days", type=float, default=None, help="Orbital period in days.")
@click.option("--t0-btjd", type=float, default=None, help="Reference epoch in BTJD.")
@click.option("--duration-hours", type=float, default=None, help="Transit duration in hours.")
@click.option("--depth-ppm", type=float, default=None, help="Transit depth in ppm.")
@click.option("--toi", type=str, default=None, help="Optional TOI label to resolve candidate inputs.")
@click.option("--report-file", type=str, default=None, help="Optional report JSON path for candidate inputs.")
@click.option(
    "--network-ok/--no-network",
    default=False,
    show_default=True,
    help="Allow network-dependent TOI resolution.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Optional cache directory for MAST/lightkurve products.",
)
@click.option("--sectors", multiple=True, type=int, help="Optional sector filters.")
@click.option(
    "--flux-type",
    type=click.Choice(["pdcsap", "sap"], case_sensitive=False),
    default="pdcsap",
    show_default=True,
)
@click.option("--bic-threshold", type=float, default=10.0, show_default=True)
@click.option("--n-harmonics", type=int, default=2, show_default=True)
@click.option("--alias-tolerance", type=float, default=0.01, show_default=True)
@click.option(
    "--detrend",
    type=str,
    default=None,
    help=(
        "Optional pre-model-competition detrend method. "
        "Supported: running_median_0p5d, transit_masked_bin_median."
    ),
)
@click.option("--detrend-bin-hours", type=float, default=6.0, show_default=True)
@click.option("--detrend-buffer", type=float, default=2.0, show_default=True)
@click.option("--detrend-sigma-clip", type=float, default=5.0, show_default=True)
@click.option(
    "-o",
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
def model_compete_command(
    toi_arg: str | None,
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
    toi: str | None,
    report_file: str | None,
    network_ok: bool,
    cache_dir: Path | None,
    sectors: tuple[int, ...],
    flux_type: str,
    bic_threshold: float,
    n_harmonics: int,
    alias_tolerance: float,
    detrend: str | None,
    detrend_bin_hours: float,
    detrend_buffer: float,
    detrend_sigma_clip: float,
    output_path_arg: str,
) -> None:
    """Run model competition + artifact prior and emit schema-stable JSON."""
    out_path = resolve_optional_output_path(output_path_arg)
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
            _resolved_depth_ppm,
            input_resolution,
        ) = _resolve_candidate_inputs(
            network_ok=bool(network_ok),
            toi=resolved_toi_arg,
            tic_id=tic_id,
            period_days=period_days,
            t0_btjd=t0_btjd,
            duration_hours=duration_hours,
            depth_ppm=depth_ppm,
        )

    effective_sectors, sectors_explicit, sector_selection_source = choose_effective_sectors(
        sectors_arg=sectors,
        report_sectors_used=report_sectors_used,
    )
    detrend_method = _normalize_detrend_method(detrend)
    if detrend_method is not None:
        _validate_detrend_args(
            detrend_bin_hours=float(detrend_bin_hours),
            detrend_buffer=float(detrend_buffer),
            detrend_sigma_clip=float(detrend_sigma_clip),
        )
    detrend_provenance: dict[str, Any] | None = None

    emit_progress("model-compete", "start")
    try:
        time, flux, flux_err, sectors_used, sector_load_path = _download_and_prepare_arrays(
            tic_id=int(resolved_tic_id),
            sectors=effective_sectors,
            sectors_explicit=bool(sectors_explicit),
            flux_type=str(flux_type).lower(),
            network_ok=bool(network_ok),
            cache_dir=cache_dir,
        )
        time, flux, flux_err, detrend_provenance = _maybe_detrend_lightcurve(
            time=time,
            flux=flux,
            flux_err=flux_err,
            period_days=float(resolved_period_days),
            t0_btjd=float(resolved_t0_btjd),
            duration_hours=float(resolved_duration_hours),
            detrend_method=detrend_method,
            detrend_bin_hours=float(detrend_bin_hours),
            detrend_buffer=float(detrend_buffer),
            detrend_sigma_clip=float(detrend_sigma_clip),
        )

        model_competition = model_competition_api.run_model_competition(
            time=time,
            flux=flux,
            flux_err=flux_err,
            period=float(resolved_period_days),
            t0=float(resolved_t0_btjd),
            duration_hours=float(resolved_duration_hours),
            bic_threshold=float(bic_threshold),
            n_harmonics=int(n_harmonics),
        )
        artifact_prior = model_competition_api.compute_artifact_prior(
            period=float(resolved_period_days),
            alias_tolerance=float(alias_tolerance),
        )
    except (LightCurveNotFoundError, TargetNotFoundError) as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except BtvCliError:
        raise
    except Exception as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_RUNTIME_ERROR) from exc

    options = {
        "network_ok": bool(network_ok),
        "sectors": [int(s) for s in effective_sectors] if effective_sectors else None,
        "flux_type": str(flux_type).lower(),
        "bic_threshold": float(bic_threshold),
        "n_harmonics": int(n_harmonics),
        "alias_tolerance": float(alias_tolerance),
        "detrend": detrend_method,
        "detrend_bin_hours": float(detrend_bin_hours),
        "detrend_buffer": float(detrend_buffer),
        "detrend_sigma_clip": float(detrend_sigma_clip),
    }
    model_competition_dict = _to_jsonable_result(model_competition)
    artifact_prior_dict = _to_jsonable_result(artifact_prior)
    result_payload = {
        "model_competition": model_competition_dict,
        "artifact_prior": artifact_prior_dict,
    }
    if isinstance(model_competition_dict, dict):
        for key in (
            "winner",
            "winner_margin",
            "model_competition_label",
            "interpretation_label",
            "interpretation_metrics",
            "artifact_risk",
            "warnings",
        ):
            if key in model_competition_dict:
                result_payload[key] = model_competition_dict[key]
    verdict, verdict_source = _derive_model_compete_verdict(result_payload)
    result_payload["verdict"] = verdict
    result_payload["verdict_source"] = verdict_source

    payload = {
        "schema_version": "cli.model_compete.v1",
        "result": result_payload,
        "verdict": verdict,
        "verdict_source": verdict_source,
        "inputs_summary": {
            "input_resolution": input_resolution,
        },
        "provenance": {
            "sectors_used": sectors_used,
            "inputs_source": "report_file" if report_file_path is not None else str(input_resolution.get("source")),
            "report_file": report_file_path,
            "sector_selection_source": sector_selection_source,
            "sector_load_path": sector_load_path,
            "detrend": detrend_provenance,
            "options": options,
        },
    }
    dump_json_output(payload, out_path)
    emit_progress("model-compete", "completed")


__all__ = ["model_compete_command"]
