"""`btv rv-feasibility` command for RV follow-up triage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import numpy as np

from tess_vetter.api.activity import characterize_activity
from tess_vetter.api.stitch import stitch_lightcurve_data
from tess_vetter.api.types import LightCurve
from tess_vetter.activity.rotation_context import build_rotation_context
from tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    resolve_optional_output_path,
)
from tess_vetter.cli.diagnostics_report_inputs import (
    choose_effective_sectors,
    load_lightcurves_with_sector_policy,
    resolve_inputs_from_report_file,
)
from tess_vetter.cli.stellar_inputs import load_auto_stellar_with_fallback
from tess_vetter.cli.vet_cli import _resolve_candidate_inputs
from tess_vetter.platform.io import LightCurveNotFoundError, TargetNotFoundError


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


def _brightness_score(tmag: float | None) -> float:
    if tmag is None:
        return 0.6
    if tmag <= 8.0:
        return 1.0
    if tmag <= 10.0:
        return 0.85
    if tmag <= 12.0:
        return 0.65
    if tmag <= 13.5:
        return 0.45
    return 0.2


def _variability_score(variability_ppm: float | None) -> float:
    if variability_ppm is None:
        return 0.6
    if variability_ppm <= 1000.0:
        return 1.0
    if variability_ppm <= 3000.0:
        return 0.8
    if variability_ppm <= 10000.0:
        return 0.5
    if variability_ppm <= 30000.0:
        return 0.2
    return 0.05


def _rotation_score(rotation_period_days: float | None) -> float:
    if rotation_period_days is None or rotation_period_days <= 0.0:
        return 0.6
    if rotation_period_days < 1.5:
        return 0.3
    if rotation_period_days < 3.0:
        return 0.5
    if rotation_period_days < 8.0:
        return 0.8
    return 1.0


def _classify_feasibility(score: float) -> tuple[str, str]:
    if score >= 0.75:
        return "HIGH_RV_FEASIBILITY", "PRIORITIZE_RV_FOLLOWUP"
    if score >= 0.5:
        return "MODERATE_RV_FEASIBILITY", "RV_POSSIBLE_WITH_ACTIVITY_MODELING"
    return "LOW_RV_FEASIBILITY", "DEPRIORITIZE_RV_USE_PHOTOMETRIC_FOLLOWUP"


def _line_broadening_bin(v_eq_est_kms: float | None) -> str:
    if v_eq_est_kms is None:
        return "UNKNOWN"
    if v_eq_est_kms < 10.0:
        return "GOOD"
    if v_eq_est_kms <= 15.0:
        return "OK"
    return "HARD"


def _apply_line_broadening_adjustment(*, base_verdict: str, broadening_bin: str) -> tuple[str, str]:
    if broadening_bin == "HARD":
        return "LOW_RV_FEASIBILITY", "RV_CHALLENGING_FAST_ROTATOR"
    if broadening_bin == "OK" and base_verdict == "HIGH_RV_FEASIBILITY":
        return "MODERATE_RV_FEASIBILITY", "RV_POSSIBLE_WITH_ACTIVITY_MODELING"
    return base_verdict, (
        "PRIORITIZE_RV_FOLLOWUP"
        if base_verdict == "HIGH_RV_FEASIBILITY"
        else (
            "RV_POSSIBLE_WITH_ACTIVITY_MODELING"
            if base_verdict == "MODERATE_RV_FEASIBILITY"
            else "DEPRIORITIZE_RV_USE_PHOTOMETRIC_FOLLOWUP"
        )
    )


def _compute_rv_feasibility(
    *,
    activity_payload: dict[str, Any],
    tmag: float | None,
    stellar_radius_rsun: float | None,
    stellar_radius_source_path: str | None,
    stellar_radius_source_authority: str | None,
) -> dict[str, Any]:
    rotation_period = activity_payload.get("rotation_period")
    variability_ppm = activity_payload.get("variability_amplitude_ppm")
    try:
        rotation_days = float(rotation_period) if rotation_period is not None else None
    except (TypeError, ValueError):
        rotation_days = None
    try:
        variability = float(variability_ppm) if variability_ppm is not None else None
    except (TypeError, ValueError):
        variability = None

    s_brightness = _brightness_score(tmag)
    s_variability = _variability_score(variability)
    s_rotation = _rotation_score(rotation_days)
    score = (0.50 * s_brightness) + (0.35 * s_variability) + (0.15 * s_rotation)
    base_verdict, _base_action_hint = _classify_feasibility(score)
    rotation_context = build_rotation_context(
        stellar_radius_rsun=stellar_radius_rsun,
        rotation_period_days=rotation_days,
        rotation_period_source_path="activity.rotation_period",
        stellar_radius_source_path=stellar_radius_source_path,
        rotation_period_source_authority="activity_lomb_scargle",
        stellar_radius_source_authority=stellar_radius_source_authority,
    )
    v_eq_est_kms = rotation_context.get("v_eq_est_kms")
    broadening_bin = _line_broadening_bin(v_eq_est_kms)
    verdict, action_hint = _apply_line_broadening_adjustment(
        base_verdict=base_verdict,
        broadening_bin=broadening_bin,
    )
    return {
        "score": round(float(score), 3),
        "verdict": verdict,
        "action_hint": action_hint,
        "inputs": {
            "tmag": float(tmag) if tmag is not None else None,
            "stellar_radius_rsun": float(stellar_radius_rsun) if stellar_radius_rsun is not None else None,
            "rotation_period_days": rotation_days,
            "variability_amplitude_ppm": variability,
        },
        "rotation_broadening": {
            "v_eq_est_kms": round(float(v_eq_est_kms), 2) if v_eq_est_kms is not None else None,
            "line_broadening_bin": broadening_bin,
            "velocity_source": "v_eq_estimated",
        },
        "rotation_context": rotation_context,
        "components": {
            "brightness_score": round(float(s_brightness), 3),
            "variability_score": round(float(s_variability), 3),
            "rotation_score": round(float(s_rotation), 3),
        },
        "weights": {"brightness": 0.5, "variability": 0.35, "rotation": 0.15},
    }


@click.command("rv-feasibility")
@click.argument("toi_arg", required=False)
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--toi", type=str, default=None, help="Optional TOI label.")
@click.option("--report-file", type=str, default=None, help="Optional report JSON path for candidate inputs.")
@click.option(
    "--network-ok/--no-network",
    default=False,
    show_default=True,
    help="Allow network-dependent TOI resolution and stellar lookup.",
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
@click.option("--rotation-min-period", type=float, default=0.5, show_default=True)
@click.option("--rotation-max-period", type=float, default=30.0, show_default=True)
@click.option("--stellar-tmag", type=float, default=None, help="Optional manual stellar Tmag override.")
@click.option("--stellar-radius", type=float, default=None, help="Optional manual stellar radius (Rsun) override.")
@click.option(
    "--use-stellar-auto/--no-use-stellar-auto",
    default=True,
    show_default=True,
    help="Attempt TIC/ExoFOP stellar lookup for Tmag/radius when missing from explicit inputs.",
)
@click.option(
    "-o",
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
def rv_feasibility_command(
    toi_arg: str | None,
    tic_id: int | None,
    toi: str | None,
    report_file: str | None,
    network_ok: bool,
    cache_dir: Path | None,
    sectors: tuple[int, ...],
    flux_type: str,
    rotation_min_period: float,
    rotation_max_period: float,
    stellar_tmag: float | None,
    stellar_radius: float | None,
    use_stellar_auto: bool,
    output_path_arg: str,
) -> None:
    """Estimate RV follow-up feasibility and rotation diagnostics."""
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
        input_resolution = dict(resolved_from_report.input_resolution)
        report_file_path = str(resolved_from_report.report_file_path)
        report_sectors_used = (
            [int(s) for s in resolved_from_report.sectors_used]
            if resolved_from_report.sectors_used is not None
            else None
        )
    else:
        resolved_tic_id = 0
        input_resolution = {}

    effective_sectors, sectors_explicit, sector_selection_source = choose_effective_sectors(
        sectors_arg=sectors,
        report_sectors_used=report_sectors_used,
    )

    stellar_provenance: dict[str, Any] = {"source": "none", "lookup": None}
    resolved_tmag: float | None = float(stellar_tmag) if stellar_tmag is not None else None
    resolved_radius_rsun: float | None = float(stellar_radius) if stellar_radius is not None else None

    try:
        if report_file is None:
            resolved_tic_id, input_resolution = _resolve_tic_and_inputs(
                tic_id=tic_id,
                toi=resolved_toi_arg,
                network_ok=bool(network_ok),
            )

        if bool(use_stellar_auto) and bool(network_ok) and (resolved_tmag is None or resolved_radius_rsun is None):
            stellar_values, lookup = load_auto_stellar_with_fallback(
                tic_id=int(resolved_tic_id),
                toi=resolved_toi_arg,
            )
            if resolved_tmag is None:
                auto_tmag = stellar_values.get("tmag")
                resolved_tmag = float(auto_tmag) if auto_tmag is not None else None
            if resolved_radius_rsun is None:
                auto_radius = stellar_values.get("radius")
                resolved_radius_rsun = float(auto_radius) if auto_radius is not None else None
            stellar_provenance = {"source": "auto", "lookup": lookup}
        elif bool(use_stellar_auto) and not bool(network_ok) and (resolved_tmag is None or resolved_radius_rsun is None):
            stellar_provenance = {
                "source": "none",
                "lookup": {"status": "skipped", "reason": "network_disabled"},
            }
        elif resolved_tmag is not None or resolved_radius_rsun is not None:
            stellar_provenance = {"source": "explicit", "lookup": None}

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
            detect_flares=False,
            flare_sigma=5.0,
            rotation_min_period=float(rotation_min_period),
            rotation_max_period=float(rotation_max_period),
        )
    except (LightCurveNotFoundError, TargetNotFoundError) as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except BtvCliError:
        raise
    except Exception as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_RUNTIME_ERROR) from exc

    activity_payload = activity.to_dict()
    rv_feasibility = _compute_rv_feasibility(
        activity_payload=activity_payload,
        tmag=resolved_tmag,
        stellar_radius_rsun=resolved_radius_rsun,
        stellar_radius_source_path=(
            "stellar_auto.radius" if str(stellar_provenance.get("source")) == "auto" else "cli.stellar_radius"
        ),
        stellar_radius_source_authority=(
            str(stellar_provenance.get("lookup", {}).get("selected_source"))
            if isinstance(stellar_provenance.get("lookup"), dict)
            and stellar_provenance.get("lookup", {}).get("selected_source") is not None
            else ("explicit" if str(stellar_provenance.get("source")) == "explicit" else None)
        ),
    )
    verdict = str(rv_feasibility.get("verdict"))
    verdict_source = "$.result.rv_feasibility.verdict"

    options = {
        "network_ok": bool(network_ok),
        "sectors": [int(s) for s in effective_sectors] if effective_sectors else None,
        "flux_type": str(flux_type).lower(),
        "rotation_min_period": float(rotation_min_period),
        "rotation_max_period": float(rotation_max_period),
        "stellar_tmag": float(stellar_tmag) if stellar_tmag is not None else None,
        "stellar_radius": float(stellar_radius) if stellar_radius is not None else None,
        "use_stellar_auto": bool(use_stellar_auto),
    }

    payload = {
        "schema_version": "cli.rv_feasibility.v1",
        "result": {
            "activity": activity_payload,
            "rv_feasibility": rv_feasibility,
            "verdict": verdict,
            "verdict_source": verdict_source,
        },
        "activity": activity_payload,
        "rv_feasibility": rv_feasibility,
        "verdict": verdict,
        "verdict_source": verdict_source,
        "inputs_summary": {
            "tic_id": int(resolved_tic_id),
            "toi": resolved_toi_arg,
            "flux_type": str(flux_type).lower(),
            "sectors_requested": [int(s) for s in sectors] if sectors else None,
            "sectors_used": sectors_used,
            "input_resolution": input_resolution,
        },
        "provenance": {
            "input_resolution": input_resolution,
            "inputs_source": "report_file" if report_file_path is not None else "cli",
            "report_file": report_file_path,
            "sectors_used": sectors_used,
            "sector_load_path": sector_load_path,
            "sector_selection_source": sector_selection_source,
            "stellar": stellar_provenance,
            "options": options,
        },
    }
    dump_json_output(payload, out_path)


__all__ = ["rv_feasibility_command"]
