"""`btv activity` command for stellar activity characterization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import numpy as np

from bittr_tess_vetter.api.activity import characterize_activity
from bittr_tess_vetter.api.stitch import stitch_lightcurve_data
from bittr_tess_vetter.api.types import LightCurve
from bittr_tess_vetter.activity.rotation_context import build_rotation_context
from bittr_tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    resolve_optional_output_path,
)
from bittr_tess_vetter.cli.diagnostics_report_inputs import (
    choose_effective_sectors,
    load_lightcurves_with_sector_policy,
    resolve_inputs_from_report_file,
)
from bittr_tess_vetter.cli.vet_cli import _resolve_candidate_inputs
from bittr_tess_vetter.platform.io import LightCurveNotFoundError, TargetNotFoundError


def _derive_activity_verdict(activity_payload: Any) -> tuple[str | None, str | None]:
    if not isinstance(activity_payload, dict):
        return None, None
    interpretation_label = activity_payload.get("interpretation_label")
    if interpretation_label is not None:
        return str(interpretation_label), "$.activity.interpretation_label"
    recommendation = activity_payload.get("recommendation")
    if recommendation is not None:
        return str(recommendation), "$.activity.recommendation"
    variability_class = activity_payload.get("variability_class")
    if variability_class is not None:
        return str(variability_class), "$.activity.variability_class"
    activity_regime = activity_payload.get("activity_regime")
    if activity_regime is not None:
        return str(activity_regime), "$.activity.activity_regime"
    return None, None


def _resolve_tic_and_inputs(
    *,
    tic_id: int | None,
    toi: str | None,
    network_ok: bool,
) -> tuple[int, dict[str, Any]]:
    if toi is not None:
        (
            resolved_tic_id,
            _period_days,
            _t0_btjd,
            _duration_hours,
            _depth_ppm,
            input_resolution,
        ) = _resolve_candidate_inputs(
            network_ok=bool(network_ok),
            toi=toi,
            tic_id=tic_id,
            period_days=None,
            t0_btjd=None,
            duration_hours=None,
            depth_ppm=None,
        )
        return int(resolved_tic_id), input_resolution

    if tic_id is None:
        raise BtvCliError(
            "Missing TIC identifier. Provide --tic-id or --toi.",
            exit_code=EXIT_INPUT_ERROR,
        )
    return int(tic_id), {"source": "cli", "resolved_from": "cli", "inputs": {"tic_id": int(tic_id)}}


def _download_and_stitch_lightcurve(
    *,
    tic_id: int,
    sectors: list[int] | None,
    sectors_explicit: bool,
    flux_type: str,
    network_ok: bool,
    cache_dir: Path | None,
) -> tuple[LightCurve, list[int], str]:
    lightcurves, sector_load_path = load_lightcurves_with_sector_policy(
        tic_id=int(tic_id),
        sectors=sectors,
        flux_type=str(flux_type).lower(),
        explicit_sectors=bool(sectors_explicit),
        network_ok=bool(network_ok),
        cache_dir=cache_dir,
    )

    if len(lightcurves) == 1:
        stitched_lc = lightcurves[0]
    else:
        stitched_lc, _ = stitch_lightcurve_data(lightcurves, tic_id=int(tic_id))

    lc = LightCurve(
        time=np.asarray(stitched_lc.time, dtype=np.float64),
        flux=np.asarray(stitched_lc.flux, dtype=np.float64),
        flux_err=(
            np.asarray(stitched_lc.flux_err, dtype=np.float64)
            if stitched_lc.flux_err is not None
            else None
        ),
        quality=(
            np.asarray(stitched_lc.quality, dtype=np.int32)
            if getattr(stitched_lc, "quality", None) is not None
            else None
        ),
        valid_mask=(
            np.asarray(stitched_lc.valid_mask, dtype=bool)
            if getattr(stitched_lc, "valid_mask", None) is not None
            else None
        ),
    )
    sectors_used = sorted({int(item.sector) for item in lightcurves if getattr(item, "sector", None) is not None})
    return lc, sectors_used, sector_load_path


@click.command("activity")
@click.argument("toi_arg", required=False)
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--toi", type=str, default=None, help="Optional TOI label.")
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
@click.option("--detect-flares/--no-detect-flares", default=True, show_default=True)
@click.option("--flare-sigma", type=float, default=5.0, show_default=True)
@click.option("--rotation-min-period", type=float, default=0.5, show_default=True)
@click.option("--rotation-max-period", type=float, default=30.0, show_default=True)
@click.option(
    "-o",
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
def activity_command(
    toi_arg: str | None,
    tic_id: int | None,
    toi: str | None,
    report_file: str | None,
    network_ok: bool,
    cache_dir: Path | None,
    sectors: tuple[int, ...],
    flux_type: str,
    detect_flares: bool,
    flare_sigma: float,
    rotation_min_period: float,
    rotation_max_period: float,
    output_path_arg: str,
) -> None:
    """Characterize stellar activity and emit schema-stable JSON."""
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
    report_stellar_radius_rsun: float | None = None
    if report_file is not None:
        if resolved_toi_arg is not None:
            click.echo(
                "Warning: --report-file provided; ignoring --toi and using report-file candidate inputs.",
                err=True,
            )
        resolved_from_report = resolve_inputs_from_report_file(str(report_file))
        resolved_tic_id = int(resolved_from_report.tic_id)
        input_resolution = dict(resolved_from_report.input_resolution)
        report_file_path = str(resolved_from_report.report_file_path)
        report_sectors_used = (
            [int(s) for s in resolved_from_report.sectors_used]
            if resolved_from_report.sectors_used is not None
            else None
        )
        report_stellar_radius_rsun = resolved_from_report.stellar_radius_rsun
    else:
        resolved_tic_id = 0
        input_resolution = {}

    effective_sectors, sectors_explicit, sector_selection_source = choose_effective_sectors(
        sectors_arg=sectors,
        report_sectors_used=report_sectors_used,
    )

    try:
        if report_file is None:
            resolved_tic_id, input_resolution = _resolve_tic_and_inputs(
                tic_id=tic_id,
                toi=resolved_toi_arg,
                network_ok=bool(network_ok),
            )
        lc, sectors_used, sector_load_path = _download_and_stitch_lightcurve(
            tic_id=int(resolved_tic_id),
            sectors=effective_sectors,
            sectors_explicit=bool(sectors_explicit),
            flux_type=str(flux_type).lower(),
            network_ok=bool(network_ok),
            cache_dir=cache_dir,
        )
        activity = characterize_activity(
            lc=lc,
            detect_flares=bool(detect_flares),
            flare_sigma=float(flare_sigma),
            rotation_min_period=float(rotation_min_period),
            rotation_max_period=float(rotation_max_period),
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
        "detect_flares": bool(detect_flares),
        "flare_sigma": float(flare_sigma),
        "rotation_min_period": float(rotation_min_period),
        "rotation_max_period": float(rotation_max_period),
    }
    activity_payload = activity.to_dict()
    rotation_period_days = None
    try:
        rotation_raw = activity_payload.get("rotation_period")
        rotation_period_days = float(rotation_raw) if rotation_raw is not None else None
    except (TypeError, ValueError):
        rotation_period_days = None
    rotation_context = build_rotation_context(
        rotation_period_days=rotation_period_days,
        stellar_radius_rsun=report_stellar_radius_rsun,
        rotation_period_source="activity.rotation_period",
        stellar_radius_source=(
            "report_file.summary.stellar.radius"
            if report_stellar_radius_rsun is not None
            else None
        ),
    )
    activity_payload["rotation_context"] = rotation_context
    verdict, verdict_source = _derive_activity_verdict(activity_payload)
    payload = {
        "schema_version": "cli.activity.v1",
        "result": {
            "activity": activity_payload,
            "verdict": verdict,
            "verdict_source": verdict_source,
        },
        "activity": activity_payload,
        "verdict": verdict,
        "verdict_source": verdict_source,
        "inputs_summary": {
            "tic_id": int(resolved_tic_id),
            "toi": toi,
            "input_resolution": input_resolution,
        },
        "provenance": {
            "sectors_used": sectors_used,
            "inputs_source": "report_file" if report_file_path is not None else str(input_resolution.get("source")),
            "report_file": report_file_path,
            "sector_selection_source": sector_selection_source,
            "sector_load_path": sector_load_path,
            "options": options,
        },
    }
    dump_json_output(payload, out_path)


__all__ = ["activity_command"]
