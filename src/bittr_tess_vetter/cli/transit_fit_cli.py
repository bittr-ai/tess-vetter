"""`btv fit` command for physical transit model fitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from bittr_tess_vetter import api
from bittr_tess_vetter.api.stitch import stitch_lightcurve_data
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve, StellarParams
from bittr_tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    resolve_optional_output_path,
)
from bittr_tess_vetter.cli.stellar_inputs import load_auto_stellar_with_fallback, resolve_stellar_inputs
from bittr_tess_vetter.cli.vet_cli import _resolve_candidate_inputs
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError, MASTClient


def _load_auto_stellar_inputs(
    tic_id: int,
    toi: str | None = None,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    return load_auto_stellar_with_fallback(tic_id=int(tic_id), toi=toi)


@click.command("fit")
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
    help="Allow network-dependent TOI and stellar auto resolution.",
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
    "--method",
    type=click.Choice(["optimize", "mcmc"], case_sensitive=False),
    default="optimize",
    show_default=True,
)
@click.option(
    "--fit-limb-darkening/--no-fit-limb-darkening",
    default=False,
    show_default=True,
)
@click.option("--mcmc-samples", type=int, default=2000, show_default=True)
@click.option("--mcmc-burn", type=int, default=500, show_default=True)
@click.option(
    "-o",
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
def fit_command(
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
    toi: str | None,
    network_ok: bool,
    cache_dir: Path | None,
    sectors: tuple[int, ...],
    flux_type: str,
    stellar_radius: float | None,
    stellar_mass: float | None,
    stellar_tmag: float | None,
    stellar_file: str | None,
    use_stellar_auto: bool,
    require_stellar: bool,
    method: str,
    fit_limb_darkening: bool,
    mcmc_samples: int,
    mcmc_burn: int,
    output_path_arg: str,
) -> None:
    """Fit a physical transit model and emit schema-stable JSON."""
    out_path = resolve_optional_output_path(output_path_arg)

    (
        resolved_tic_id,
        resolved_period_days,
        resolved_t0_btjd,
        resolved_duration_hours,
        resolved_depth_ppm,
        input_resolution,
    ) = _resolve_candidate_inputs(
        network_ok=network_ok,
        toi=toi,
        tic_id=tic_id,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_ppm=depth_ppm,
    )

    if use_stellar_auto and not network_ok:
        raise BtvCliError("--use-stellar-auto requires --network-ok", exit_code=EXIT_DATA_UNAVAILABLE)

    resolved_stellar, stellar_resolution = resolve_stellar_inputs(
        tic_id=resolved_tic_id,
        stellar_radius=stellar_radius,
        stellar_mass=stellar_mass,
        stellar_tmag=stellar_tmag,
        stellar_file=stellar_file,
        use_stellar_auto=use_stellar_auto,
        require_stellar=require_stellar,
        auto_loader=(
            (lambda _tic_id: _load_auto_stellar_inputs(_tic_id, toi=toi))
            if toi is not None
            else (lambda _tic_id: _load_auto_stellar_inputs(_tic_id))
        )
        if use_stellar_auto
        else None,
    )

    try:
        client = MASTClient(cache_dir=str(cache_dir)) if cache_dir is not None else MASTClient()
        lightcurves = client.download_all_sectors(
            tic_id=int(resolved_tic_id),
            flux_type=str(flux_type).lower(),
            sectors=list(sectors) if sectors else None,
        )
        if not lightcurves:
            raise LightCurveNotFoundError(f"No sectors available for TIC {resolved_tic_id}")

        if len(lightcurves) == 1:
            lc = LightCurve.from_internal(lightcurves[0])
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
        stellar = StellarParams(
            radius=resolved_stellar.get("radius"),
            mass=resolved_stellar.get("mass"),
            tmag=resolved_stellar.get("tmag"),
        )

        result = api.transit_fit.fit_transit(
            lc=lc,
            candidate=candidate,
            stellar=stellar,
            method=str(method).lower(),
            fit_limb_darkening=bool(fit_limb_darkening),
            mcmc_samples=int(mcmc_samples),
            mcmc_burn=int(mcmc_burn),
        )
    except LightCurveNotFoundError as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except BtvCliError:
        raise
    except Exception as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_RUNTIME_ERROR) from exc

    sectors_used = sorted(
        {int(item.sector) for item in lightcurves if getattr(item, "sector", None) is not None}
    )

    payload = {
        "schema_version": "cli.fit.v1",
        "fit": result.to_dict(),
        "inputs_summary": {
            "input_resolution": input_resolution,
            "stellar_resolution": stellar_resolution,
        },
        "provenance": {
            "tic_id": int(resolved_tic_id),
            "flux_type": str(flux_type).lower(),
            "requested_sectors": [int(v) for v in sectors] if sectors else None,
            "sectors_used": sectors_used,
            "method": str(method).lower(),
            "fit_limb_darkening": bool(fit_limb_darkening),
            "mcmc_samples": int(mcmc_samples),
            "mcmc_burn": int(mcmc_burn),
        },
    }
    dump_json_output(payload, out_path)


__all__ = ["fit_command"]
