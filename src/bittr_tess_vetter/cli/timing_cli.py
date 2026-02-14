"""`btv timing` command for transit timing diagnostics."""

from __future__ import annotations

from typing import Any

import click

from bittr_tess_vetter import api
from bittr_tess_vetter.api.stitch import stitch_lightcurve_data
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
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


def _transit_times_to_dicts(transit_times: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for transit in transit_times:
        rows.append(
            {
                "epoch": int(transit.epoch),
                "tc": float(transit.tc),
                "tc_err": float(transit.tc_err),
                "depth_ppm": float(transit.depth_ppm),
                "duration_hours": float(transit.duration_hours),
                "snr": float(transit.snr),
                "is_outlier": bool(transit.is_outlier),
                "outlier_reason": transit.outlier_reason,
            }
        )
    return rows


@click.command("timing")
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
@click.option("--min-snr", type=float, default=2.0, show_default=True)
@click.option(
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
def timing_command(
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
    toi: str | None,
    network_ok: bool,
    sectors: tuple[int, ...],
    flux_type: str,
    min_snr: float,
    output_path_arg: str,
) -> None:
    """Measure transit times and compute TTV diagnostics."""
    out_path = resolve_optional_output_path(output_path_arg)

    (
        resolved_tic_id,
        resolved_period_days,
        resolved_t0_btjd,
        resolved_duration_hours,
        resolved_depth_ppm,
        input_resolution,
    ) = _resolve_candidate_inputs(
        network_ok=bool(network_ok),
        toi=toi,
        tic_id=tic_id,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_ppm=depth_ppm,
    )

    try:
        lightcurves = MASTClient().download_all_sectors(
            tic_id=int(resolved_tic_id),
            flux_type=str(flux_type).lower(),
            sectors=[int(s) for s in sectors] if sectors else None,
        )
        if not lightcurves:
            raise LightCurveNotFoundError(f"No sectors available for TIC {resolved_tic_id}")

        if len(lightcurves) == 1:
            stitched_lc = lightcurves[0]
        else:
            stitched_lc, _ = stitch_lightcurve_data(lightcurves, tic_id=int(resolved_tic_id))
        lc = LightCurve.from_internal(stitched_lc)

        candidate = Candidate(
            ephemeris=Ephemeris(
                period_days=float(resolved_period_days),
                t0_btjd=float(resolved_t0_btjd),
                duration_hours=float(resolved_duration_hours),
            ),
            depth_ppm=float(resolved_depth_ppm) if resolved_depth_ppm is not None else None,
        )

        transit_times = api.timing.measure_transit_times(
            lc=lc,
            candidate=candidate,
            min_snr=float(min_snr),
        )
        ttv = api.timing.analyze_ttvs(
            transit_times=transit_times,
            period_days=float(resolved_period_days),
            t0_btjd=float(resolved_t0_btjd),
        )
        series = api.timing.timing_series(
            lc=lc,
            candidate=candidate,
            min_snr=float(min_snr),
        )
    except (LightCurveNotFoundError, TargetNotFoundError) as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except BtvCliError:
        raise
    except Exception as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_RUNTIME_ERROR) from exc

    sectors_used = sorted(
        {int(item.sector) for item in lightcurves if getattr(item, "sector", None) is not None}
    )

    options = {
        "network_ok": bool(network_ok),
        "sectors": [int(s) for s in sectors] if sectors else None,
        "flux_type": str(flux_type).lower(),
        "min_snr": float(min_snr),
    }
    payload = {
        "schema_version": "cli.timing.v1",
        "transit_times": _transit_times_to_dicts(transit_times),
        "ttv": _to_jsonable_result(ttv),
        "timing_series": _to_jsonable_result(series),
        "inputs_summary": {
            "input_resolution": input_resolution,
        },
        "provenance": {
            "sectors_used": sectors_used,
            "options": options,
        },
    }
    dump_json_output(payload, out_path)


__all__ = ["timing_command"]
