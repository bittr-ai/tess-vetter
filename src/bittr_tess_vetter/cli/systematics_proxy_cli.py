"""`btv systematics-proxy` command for LC-only systematics proxy diagnostics."""

from __future__ import annotations

from typing import Any

import click
import numpy as np

from bittr_tess_vetter.api import systematics as systematics_api
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
    if result is None:
        return None
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json", exclude_none=True)
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return result


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
    quality = (
        np.asarray(lc.quality, dtype=np.int32)
        if getattr(lc, "quality", None) is not None
        else np.zeros_like(flux, dtype=np.int32)
    )
    lc_valid_mask = (
        np.asarray(lc.valid_mask, dtype=bool)
        if getattr(lc, "valid_mask", None) is not None
        else np.ones_like(flux, dtype=bool)
    )
    valid_mask = lc_valid_mask & (quality == 0) & np.isfinite(time) & np.isfinite(flux)
    if not np.any(valid_mask):
        raise LightCurveNotFoundError(f"No valid finite cadences available for TIC {tic_id}")

    sectors_used = sorted({int(item.sector) for item in lightcurves if getattr(item, "sector", None) is not None})
    return time, flux, valid_mask, sectors_used


@click.command("systematics-proxy")
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
@click.option(
    "-o",
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
def systematics_proxy_command(
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
    output_path_arg: str,
) -> None:
    """Compute LC-only systematics proxy diagnostics and emit JSON."""
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
        time, flux, valid_mask, sectors_used = _download_and_prepare_arrays(
            tic_id=int(resolved_tic_id),
            sectors=[int(s) for s in sectors] if sectors else None,
            flux_type=str(flux_type).lower(),
        )
        systematics_proxy = systematics_api.compute_systematics_proxy(
            time=time,
            flux=flux,
            valid_mask=valid_mask,
            period_days=float(resolved_period_days),
            t0_btjd=float(resolved_t0_btjd),
            duration_hours=float(resolved_duration_hours),
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
    }
    payload = {
        "schema_version": "cli.systematics_proxy.v1",
        "systematics_proxy": _to_jsonable_result(systematics_proxy),
        "inputs_summary": {
            "input_resolution": input_resolution,
        },
        "provenance": {
            "sectors_used": sectors_used,
            "options": options,
        },
    }
    dump_json_output(payload, out_path)


__all__ = ["systematics_proxy_command"]
