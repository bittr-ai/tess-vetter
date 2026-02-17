"""`btv localize` command for WCS-aware transit source localization."""

from __future__ import annotations

from typing import Any

import click
import numpy as np

from bittr_tess_vetter.api.generate_report import _select_tpf_sectors
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.api.wcs_localization import localize_transit_source
from bittr_tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    resolve_optional_output_path,
)
from bittr_tess_vetter.cli.vet_cli import _resolve_candidate_inputs
from bittr_tess_vetter.platform.catalogs.toi_resolution import LookupStatus, lookup_tic_coordinates
from bittr_tess_vetter.pixel.tpf_fits import TPFFitsData, TPFFitsRef
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError, MASTClient

CLI_LOCALIZE_SCHEMA_VERSION = "cli.localize.v1"


def _coerce_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _extract_tpf_meta(*, wcs: Any) -> dict[str, Any]:
    if wcs is None:
        return {}
    try:
        header = wcs.to_header(relax=True)
    except Exception:
        return {}
    try:
        return dict(header)
    except Exception:
        return {}


def _build_reference_sources(
    *,
    tic_id: int,
    tpf_meta: dict[str, Any],
    ra_deg: float | None,
    dec_deg: float | None,
    network_ok: bool,
) -> tuple[list[dict[str, Any]], str]:
    ra_from_meta = _coerce_finite_float(tpf_meta.get("RA_OBJ"))
    dec_from_meta = _coerce_finite_float(tpf_meta.get("DEC_OBJ"))
    if ra_from_meta is not None and dec_from_meta is not None:
        return (
            [
                {
                    "name": f"Target TIC {int(tic_id)}",
                    "ra": float(ra_from_meta),
                    "dec": float(dec_from_meta),
                    "meta": {"source": "tpf_meta", "ra_key": "RA_OBJ", "dec_key": "DEC_OBJ"},
                }
            ],
            "tpf_meta",
        )

    if ra_deg is not None and dec_deg is not None:
        return (
            [
                {
                    "name": f"Target TIC {int(tic_id)}",
                    "ra": float(ra_deg),
                    "dec": float(dec_deg),
                    "meta": {"source": "cli"},
                }
            ],
            "cli",
        )

    if network_ok:
        coord_result = lookup_tic_coordinates(tic_id=int(tic_id))
        if coord_result.status == LookupStatus.OK and coord_result.ra_deg is not None and coord_result.dec_deg is not None:
            return (
                [
                    {
                        "name": f"Target TIC {int(tic_id)}",
                        "ra": float(coord_result.ra_deg),
                        "dec": float(coord_result.dec_deg),
                        "meta": {"source": "tic_lookup"},
                    }
                ],
                "tic_lookup",
            )
        if coord_result.status == LookupStatus.TIMEOUT:
            raise BtvCliError(
                coord_result.message or f"TIC coordinate lookup timed out for TIC {tic_id}",
                exit_code=EXIT_DATA_UNAVAILABLE,
            )
        if coord_result.status == LookupStatus.DATA_UNAVAILABLE:
            raise BtvCliError(
                coord_result.message or f"TIC coordinate lookup returned no RA/Dec for TIC {tic_id}",
                exit_code=EXIT_DATA_UNAVAILABLE,
            )
        raise BtvCliError(
            coord_result.message or f"TIC coordinate lookup failed for TIC {tic_id}",
            exit_code=EXIT_RUNTIME_ERROR,
        )

    raise BtvCliError(
        "Target coordinates unavailable. Provide --ra-deg and --dec-deg when TPF RA_OBJ/DEC_OBJ are missing.",
        exit_code=EXIT_INPUT_ERROR,
    )


def _execute_localize(
    *,
    tic_id: int,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    ra_deg: float | None,
    dec_deg: float | None,
    network_ok: bool,
    sectors: list[int] | None,
    tpf_sector_strategy: str,
    tpf_sectors: list[int] | None,
    bootstrap_draws: int,
    bootstrap_seed: int,
    oot_margin_mult: float,
    oot_window_mult: float,
    input_resolution: dict[str, Any] | None,
) -> dict[str, Any]:
    client = MASTClient()
    lightcurves = client.download_all_sectors(int(tic_id), flux_type="pdcsap", sectors=sectors)
    if not lightcurves:
        raise LightCurveNotFoundError(f"No sectors available for TIC {tic_id}")

    if len(lightcurves) == 1:
        stitched_lc = lightcurves[0]
    else:
        from bittr_tess_vetter.api.stitch import stitch_lightcurve_data

        stitched_lc, _ = stitch_lightcurve_data(lightcurves, tic_id=int(tic_id))

    lc = LightCurve.from_internal(stitched_lc)
    candidate = Candidate(
        ephemeris=Ephemeris(
            period_days=float(period_days),
            t0_btjd=float(t0_btjd),
            duration_hours=float(duration_hours),
        )
    )
    sectors_used = sorted({int(lc_data.sector) for lc_data in lightcurves if lc_data.sector is not None})
    sector_times = {
        int(lc_data.sector): np.asarray(lc_data.time, dtype=np.float64)
        for lc_data in lightcurves
        if lc_data.sector is not None and lc_data.time is not None
    }

    selected = _select_tpf_sectors(
        strategy=str(tpf_sector_strategy).lower(),
        sectors_used=sectors_used,
        requested=tpf_sectors,
        lc_api=lc,
        candidate_api=candidate,
        sector_times=sector_times,
    )
    if len(selected) == 0:
        raise LightCurveNotFoundError("No TPF sector selected for this candidate")

    selected_sector: int | None = None
    tpf_tuple: tuple[np.ndarray, np.ndarray, np.ndarray | None, Any | None, np.ndarray | None, np.ndarray | None] | None = None
    for sector in selected:
        try:
            tpf_tuple = client.download_tpf_cached(tic_id=int(tic_id), sector=int(sector))
            selected_sector = int(sector)
            break
        except Exception:
            if not network_ok:
                continue
            try:
                tpf_tuple = client.download_tpf(tic_id=int(tic_id), sector=int(sector))
                selected_sector = int(sector)
                break
            except Exception:
                continue

    if tpf_tuple is None or selected_sector is None:
        raise LightCurveNotFoundError(f"TPF unavailable for TIC {tic_id}")

    time_arr, flux_cube, flux_err, wcs, aperture_mask, quality = tpf_tuple
    flux_arr = np.asarray(flux_cube, dtype=np.float64)
    n_cadences = int(np.asarray(time_arr).shape[0])
    tpf_meta = _extract_tpf_meta(wcs=wcs)

    tpf_fits = TPFFitsData(
        ref=TPFFitsRef(tic_id=int(tic_id), sector=int(selected_sector), author="spoc"),
        time=np.asarray(time_arr, dtype=np.float64),
        flux=flux_arr,
        flux_err=np.asarray(flux_err, dtype=np.float64) if flux_err is not None else None,
        wcs=wcs,
        aperture_mask=(
            np.asarray(aperture_mask, dtype=np.int32)
            if aperture_mask is not None
            else np.zeros(flux_arr.shape[1:], dtype=np.int32)
        ),
        quality=(
            np.asarray(quality, dtype=np.int32)
            if quality is not None
            else np.zeros(n_cadences, dtype=np.int32)
        ),
        camera=0,
        ccd=0,
        meta=tpf_meta,
    )
    reference_sources, coordinate_source = _build_reference_sources(
        tic_id=int(tic_id),
        tpf_meta=tpf_meta,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        network_ok=bool(network_ok),
    )

    localization_result = localize_transit_source(
        tpf_fits=tpf_fits,
        period=float(period_days),
        t0=float(t0_btjd),
        duration_hours=float(duration_hours),
        reference_sources=reference_sources,
        bootstrap_draws=int(bootstrap_draws),
        bootstrap_seed=int(bootstrap_seed),
        oot_margin_mult=float(oot_margin_mult),
        oot_window_mult=float(oot_window_mult),
    )

    return {
        "schema_version": CLI_LOCALIZE_SCHEMA_VERSION,
        "result": localization_result.to_dict(),
        "inputs_summary": {
            "input_resolution": input_resolution,
        },
        "provenance": {
            "selected_sector": int(selected_sector),
            "sectors_requested": [int(s) for s in sectors] if sectors else None,
            "sectors_used": sectors_used,
            "requested_sectors": [int(s) for s in tpf_sectors] if tpf_sectors else None,
            "tpf_sector_strategy": str(tpf_sector_strategy).lower(),
            "network_ok": bool(network_ok),
            "coordinate_source": str(coordinate_source),
        },
    }


@click.command("localize")
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--period-days", type=float, default=None, help="Orbital period in days.")
@click.option("--t0-btjd", type=float, default=None, help="Reference epoch in BTJD.")
@click.option("--duration-hours", type=float, default=None, help="Transit duration in hours.")
@click.option("--depth-ppm", type=float, default=None, help="Transit depth in ppm.")
@click.option("--toi", type=str, default=None, help="Optional TOI label to resolve candidate inputs.")
@click.option("--ra-deg", type=float, default=None, help="Fallback target right ascension in degrees.")
@click.option("--dec-deg", type=float, default=None, help="Fallback target declination in degrees.")
@click.option(
    "--network-ok/--no-network",
    default=False,
    show_default=True,
    help="Allow network-dependent TOI resolution and TPF download fallback.",
)
@click.option("--sectors", multiple=True, type=int, help="Optional sector filters.")
@click.option(
    "--tpf-sector-strategy",
    type=click.Choice(["best", "all", "requested"], case_sensitive=False),
    default="best",
    show_default=True,
    help="How to choose sector(s) for TPF selection.",
)
@click.option("--tpf-sector", "tpf_sectors", multiple=True, type=int, help="Sector(s) when strategy=requested.")
@click.option("--bootstrap-draws", type=int, default=500, show_default=True)
@click.option("--bootstrap-seed", type=int, default=42, show_default=True)
@click.option("--oot-margin-mult", type=float, default=1.5, show_default=True)
@click.option("--oot-window-mult", type=float, default=10.0, show_default=True)
@click.option(
    "-o",
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
def localize_command(
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
    toi: str | None,
    ra_deg: float | None,
    dec_deg: float | None,
    network_ok: bool,
    sectors: tuple[int, ...],
    tpf_sector_strategy: str,
    tpf_sectors: tuple[int, ...],
    bootstrap_draws: int,
    bootstrap_seed: int,
    oot_margin_mult: float,
    oot_window_mult: float,
    output_path_arg: str,
) -> None:
    """Run WCS-aware source localization for a single transit candidate."""
    out_path = resolve_optional_output_path(output_path_arg)
    strategy = str(tpf_sector_strategy).lower()

    if tpf_sectors and strategy != "requested":
        raise BtvCliError(
            "--tpf-sector requires --tpf-sector-strategy=requested",
            exit_code=EXIT_INPUT_ERROR,
        )
    if strategy == "requested" and len(tpf_sectors) == 0:
        raise BtvCliError(
            "--tpf-sector-strategy=requested requires at least one --tpf-sector",
            exit_code=EXIT_INPUT_ERROR,
        )
    if (ra_deg is None) != (dec_deg is None):
        raise BtvCliError("Provide both --ra-deg and --dec-deg together.", exit_code=EXIT_INPUT_ERROR)

    (
        resolved_tic_id,
        resolved_period_days,
        resolved_t0_btjd,
        resolved_duration_hours,
        _resolved_depth_ppm,
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

    try:
        payload = _execute_localize(
            tic_id=resolved_tic_id,
            period_days=resolved_period_days,
            t0_btjd=resolved_t0_btjd,
            duration_hours=resolved_duration_hours,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            network_ok=bool(network_ok),
            sectors=[int(s) for s in sectors] if sectors else None,
            tpf_sector_strategy=strategy,
            tpf_sectors=[int(s) for s in tpf_sectors] if tpf_sectors else None,
            bootstrap_draws=int(bootstrap_draws),
            bootstrap_seed=int(bootstrap_seed),
            oot_margin_mult=float(oot_margin_mult),
            oot_window_mult=float(oot_window_mult),
            input_resolution=input_resolution,
        )
    except LightCurveNotFoundError as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except BtvCliError:
        raise
    except Exception as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_RUNTIME_ERROR) from exc

    dump_json_output(payload, out_path)


__all__ = ["localize_command"]
