"""`btv periodogram` command for period search and refinement."""

from __future__ import annotations

from typing import Any

import click
import numpy as np

from bittr_tess_vetter.api.periodogram import refine_period, run_periodogram
from bittr_tess_vetter.api.stitch import stitch_lightcurve_data
from bittr_tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    resolve_optional_output_path,
)
from bittr_tess_vetter.cli.stellar_inputs import load_auto_stellar_with_fallback, resolve_stellar_inputs
from bittr_tess_vetter.cli.vet_cli import _resolve_candidate_inputs
from bittr_tess_vetter.platform.io import LightCurveNotFoundError, MASTClient, TargetNotFoundError


def _load_auto_stellar_inputs(
    tic_id: int,
    toi: str | None = None,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    return load_auto_stellar_with_fallback(tic_id=int(tic_id), toi=toi)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[int]]:
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
    flux_err = np.asarray(lc.flux_err, dtype=np.float64) if lc.flux_err is not None else None
    sectors_used = sorted({int(item.sector) for item in lightcurves if getattr(item, "sector", None) is not None})
    return time, flux, flux_err, sectors_used


def _to_jsonable_result(result: Any) -> Any:
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json", exclude_none=True)
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return result


@click.command("periodogram")
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--toi", type=str, default=None, help="Optional TOI label.")
@click.option(
    "--network-ok/--no-network",
    default=False,
    show_default=True,
    help="Allow network-dependent TOI or stellar auto resolution.",
)
@click.option("--sectors", multiple=True, type=int, help="Optional sector filters.")
@click.option(
    "--flux-type",
    type=click.Choice(["pdcsap", "sap"], case_sensitive=False),
    default="pdcsap",
    show_default=True,
)
@click.option("--min-period", type=float, default=0.5, show_default=True)
@click.option("--max-period", type=float, default=None)
@click.option(
    "--preset",
    type=click.Choice(["fast", "thorough", "deep"], case_sensitive=False),
    default="fast",
    show_default=True,
)
@click.option(
    "--method",
    type=click.Choice(["auto", "tls", "ls"], case_sensitive=False),
    default="auto",
    show_default=True,
)
@click.option("--max-planets", type=int, default=1, show_default=True)
@click.option("--per-sector/--no-per-sector", default=True, show_default=True)
@click.option("--downsample-factor", type=int, default=1, show_default=True)
@click.option("--refine", is_flag=True, default=False, help="Run period refinement mode.")
@click.option("--initial-period", type=float, default=None, help="Initial period for refinement (days).")
@click.option("--initial-duration", type=float, default=None, help="Initial duration for refinement (hours).")
@click.option("--refine-factor", type=float, default=0.1, show_default=True)
@click.option("--n-refine", type=int, default=100, show_default=True)
@click.option("--stellar-radius", type=float, default=None, help="Stellar radius (Rsun).")
@click.option("--stellar-mass", type=float, default=None, help="Stellar mass (Msun).")
@click.option("--stellar-file", type=str, default=None, help="JSON file with stellar inputs.")
@click.option(
    "--use-stellar-auto/--no-use-stellar-auto",
    default=False,
    show_default=True,
    help="Resolve stellar inputs from TIC when missing from explicit/file inputs.",
)
@click.option(
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
def periodogram_command(
    tic_id: int | None,
    toi: str | None,
    network_ok: bool,
    sectors: tuple[int, ...],
    flux_type: str,
    min_period: float,
    max_period: float | None,
    preset: str,
    method: str,
    max_planets: int,
    per_sector: bool,
    downsample_factor: int,
    refine: bool,
    initial_period: float | None,
    initial_duration: float | None,
    refine_factor: float,
    n_refine: int,
    stellar_radius: float | None,
    stellar_mass: float | None,
    stellar_file: str | None,
    use_stellar_auto: bool,
    output_path_arg: str,
) -> None:
    """Run periodogram search or refinement and emit schema-stable JSON."""
    out_path = resolve_optional_output_path(output_path_arg)
    mode = "refine" if bool(refine) else "search"

    if use_stellar_auto and not network_ok:
        raise BtvCliError("--use-stellar-auto requires --network-ok", exit_code=EXIT_DATA_UNAVAILABLE)

    if mode == "refine" and (initial_period is None or initial_duration is None):
        raise BtvCliError(
            "--refine requires --initial-period and --initial-duration",
            exit_code=EXIT_INPUT_ERROR,
        )

    try:
        resolved_tic_id, input_resolution = _resolve_tic_and_inputs(
            tic_id=tic_id,
            toi=toi,
            network_ok=bool(network_ok),
        )

        resolved_stellar, stellar_resolution = resolve_stellar_inputs(
            tic_id=resolved_tic_id,
            stellar_radius=stellar_radius,
            stellar_mass=stellar_mass,
            stellar_tmag=None,
            stellar_file=stellar_file,
            use_stellar_auto=bool(use_stellar_auto),
            require_stellar=False,
            auto_loader=(
                (lambda _tic_id: _load_auto_stellar_inputs(_tic_id, toi=toi))
                if toi is not None
                else (lambda _tic_id: _load_auto_stellar_inputs(_tic_id))
            )
            if use_stellar_auto
            else None,
        )

        time, flux, flux_err, sectors_used = _download_and_stitch_lightcurve(
            tic_id=resolved_tic_id,
            sectors=list(sectors) if sectors else None,
            flux_type=str(flux_type).lower(),
        )

        if mode == "search":
            raw_result = run_periodogram(
                time=time,
                flux=flux,
                flux_err=flux_err,
                min_period=float(min_period),
                max_period=float(max_period) if max_period is not None else None,
                preset=str(preset).lower(),
                method=str(method).lower(),  # type: ignore[arg-type]
                max_planets=int(max_planets),
                data_ref=f"tic{int(resolved_tic_id)}",
                tic_id=int(resolved_tic_id),
                stellar_radius_rsun=resolved_stellar.get("radius"),
                stellar_mass_msun=resolved_stellar.get("mass"),
                per_sector=bool(per_sector),
                downsample_factor=int(downsample_factor),
            )
            result = _to_jsonable_result(raw_result)
        else:
            refined_period, refined_t0, refined_power = refine_period(
                time=time,
                flux=flux,
                flux_err=flux_err,
                initial_period=float(initial_period),
                initial_duration=float(initial_duration),
                refine_factor=float(refine_factor),
                n_refine=int(n_refine),
                tic_id=int(resolved_tic_id),
                stellar_radius_rsun=resolved_stellar.get("radius"),
                stellar_mass_msun=resolved_stellar.get("mass"),
            )
            result = {
                "refined_period_days": float(refined_period),
                "refined_t0_btjd": float(refined_t0),
                "refined_power": float(refined_power),
            }
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
        "min_period": float(min_period),
        "max_period": float(max_period) if max_period is not None else None,
        "preset": str(preset).lower(),
        "method": str(method).lower(),
        "max_planets": int(max_planets),
        "per_sector": bool(per_sector),
        "downsample_factor": int(downsample_factor),
        "refine": bool(refine),
        "initial_period": float(initial_period) if initial_period is not None else None,
        "initial_duration": float(initial_duration) if initial_duration is not None else None,
        "refine_factor": float(refine_factor),
        "n_refine": int(n_refine),
        "stellar_radius": stellar_radius,
        "stellar_mass": stellar_mass,
        "stellar_file": stellar_file,
        "use_stellar_auto": bool(use_stellar_auto),
    }
    payload = {
        "schema_version": "cli.periodogram.v1",
        "mode": mode,
        "result": result,
        "inputs_summary": {
            "tic_id": int(resolved_tic_id),
            "toi": toi,
            "flux_type": str(flux_type).lower(),
            "sectors_requested": [int(s) for s in sectors] if sectors else None,
            "sectors_used": sectors_used,
        },
        "provenance": {
            "input_resolution": input_resolution,
            "stellar": stellar_resolution,
            "sectors_used": sectors_used,
            "options": options,
        },
    }
    dump_json_output(payload, out_path)


__all__ = ["periodogram_command"]
