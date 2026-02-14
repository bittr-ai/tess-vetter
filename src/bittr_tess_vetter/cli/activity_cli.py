"""`btv activity` command for stellar activity characterization."""

from __future__ import annotations

from typing import Any

import click
import numpy as np

from bittr_tess_vetter.api.activity import characterize_activity
from bittr_tess_vetter.api.stitch import stitch_lightcurve_data
from bittr_tess_vetter.api.types import LightCurve
from bittr_tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    resolve_optional_output_path,
)
from bittr_tess_vetter.cli.vet_cli import _resolve_candidate_inputs
from bittr_tess_vetter.platform.io import LightCurveNotFoundError, MASTClient, TargetNotFoundError


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
    flux_type: str,
) -> tuple[LightCurve, list[int]]:
    client = MASTClient()
    lightcurves = client.download_all_sectors(
        tic_id=int(tic_id),
        flux_type=str(flux_type).lower(),
        sectors=sectors,
    )
    if not lightcurves:
        raise LightCurveNotFoundError(f"No sectors available for TIC {tic_id}")

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
    return lc, sectors_used


@click.command("activity")
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--toi", type=str, default=None, help="Optional TOI label.")
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
@click.option("--detect-flares/--no-detect-flares", default=True, show_default=True)
@click.option("--flare-sigma", type=float, default=5.0, show_default=True)
@click.option("--rotation-min-period", type=float, default=0.5, show_default=True)
@click.option("--rotation-max-period", type=float, default=30.0, show_default=True)
@click.option(
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
def activity_command(
    tic_id: int | None,
    toi: str | None,
    network_ok: bool,
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

    try:
        resolved_tic_id, input_resolution = _resolve_tic_and_inputs(
            tic_id=tic_id,
            toi=toi,
            network_ok=bool(network_ok),
        )
        lc, sectors_used = _download_and_stitch_lightcurve(
            tic_id=int(resolved_tic_id),
            sectors=list(sectors) if sectors else None,
            flux_type=str(flux_type).lower(),
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
        "sectors": [int(s) for s in sectors] if sectors else None,
        "flux_type": str(flux_type).lower(),
        "detect_flares": bool(detect_flares),
        "flare_sigma": float(flare_sigma),
        "rotation_min_period": float(rotation_min_period),
        "rotation_max_period": float(rotation_max_period),
    }
    payload = {
        "schema_version": "cli.activity.v1",
        "activity": activity.to_dict(),
        "inputs_summary": {
            "tic_id": int(resolved_tic_id),
            "toi": toi,
            "input_resolution": input_resolution,
        },
        "provenance": {
            "sectors_used": sectors_used,
            "options": options,
        },
    }
    dump_json_output(payload, out_path)


__all__ = ["activity_command"]
