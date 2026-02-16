"""`btv model-compete` command for model competition diagnostics."""

from __future__ import annotations

from typing import Any

import click
import numpy as np

from bittr_tess_vetter.api import model_competition as model_competition_api
from bittr_tess_vetter.api.stitch import stitch_lightcurve_data
from bittr_tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    resolve_optional_output_path,
)
from bittr_tess_vetter.cli.vet_cli import _resolve_candidate_inputs
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError, MASTClient, TargetNotFoundError


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


def _download_and_prepare_arrays(
    *,
    tic_id: int,
    sectors: list[int] | None,
    flux_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    client = MASTClient()
    lightcurves = client.download_all_sectors(
        tic_id=int(tic_id),
        flux_type=str(flux_type).lower(),
        sectors=sectors,
    )
    if not lightcurves:
        raise LightCurveNotFoundError(f"No sectors available for TIC {tic_id}")

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
    return time[valid], flux[valid], flux_err[valid], sectors_used


@click.command("model-compete")
@click.argument("toi_arg", required=False)
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--period-days", type=float, default=None, help="Orbital period in days.")
@click.option("--t0-btjd", type=float, default=None, help="Reference epoch in BTJD.")
@click.option("--duration-hours", type=float, default=None, help="Transit duration in hours.")
@click.option("--depth-ppm", type=float, default=None, help="Transit depth in ppm.")
@click.option("--toi", type=str, default=None, help="Optional TOI label to resolve candidate inputs.")
@click.option(
    "--network-ok/--no-network",
    default=False,
    show_default=True,
    help="Allow network-dependent TOI resolution.",
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
    network_ok: bool,
    sectors: tuple[int, ...],
    flux_type: str,
    bic_threshold: float,
    n_harmonics: int,
    alias_tolerance: float,
    output_path_arg: str,
) -> None:
    """Run model competition + artifact prior and emit schema-stable JSON."""
    out_path = resolve_optional_output_path(output_path_arg)
    if toi_arg is not None and toi is not None and str(toi_arg).strip() != str(toi).strip():
        raise BtvCliError(
            "Positional TOI argument and --toi must match when both are provided.",
            exit_code=EXIT_INPUT_ERROR,
        )
    resolved_toi_arg = toi if toi is not None else toi_arg

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

    try:
        time, flux, flux_err, sectors_used = _download_and_prepare_arrays(
            tic_id=int(resolved_tic_id),
            sectors=[int(s) for s in sectors] if sectors else None,
            flux_type=str(flux_type).lower(),
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
        "sectors": [int(s) for s in sectors] if sectors else None,
        "flux_type": str(flux_type).lower(),
        "bic_threshold": float(bic_threshold),
        "n_harmonics": int(n_harmonics),
        "alias_tolerance": float(alias_tolerance),
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
            "options": options,
        },
    }
    dump_json_output(payload, out_path)


__all__ = ["model_compete_command"]
