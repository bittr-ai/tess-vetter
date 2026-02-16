"""`btv localize-host` command for multi-sector transit host localization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import numpy as np

from bittr_tess_vetter.api.generate_report import _select_tpf_sectors
from bittr_tess_vetter.api.pixel_localize import localize_transit_host_multi_sector
from bittr_tess_vetter.api.stitch import stitch_lightcurve_data
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    resolve_optional_output_path,
)
from bittr_tess_vetter.cli.localize_cli import _build_reference_sources, _extract_tpf_meta
from bittr_tess_vetter.cli.reference_sources import load_reference_sources_file
from bittr_tess_vetter.cli.vet_cli import _resolve_candidate_inputs
from bittr_tess_vetter.pixel.tpf_fits import TPFFitsData, TPFFitsRef
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError, MASTClient

CLI_LOCALIZE_HOST_SCHEMA_VERSION = "cli.localize_host.v1"


def _download_tpf_fits(
    *,
    client: MASTClient,
    tic_id: int,
    sector: int,
    network_ok: bool,
) -> TPFFitsData | None:
    try:
        time_arr, flux_cube, flux_err, wcs, aperture_mask, quality = client.download_tpf_cached(
            tic_id=int(tic_id),
            sector=int(sector),
        )
    except Exception:
        if not network_ok:
            return None
        try:
            time_arr, flux_cube, flux_err, wcs, aperture_mask, quality = client.download_tpf(
                tic_id=int(tic_id),
                sector=int(sector),
            )
        except Exception:
            return None

    flux_arr = np.asarray(flux_cube, dtype=np.float64)
    n_cadences = int(np.asarray(time_arr).shape[0])
    tpf_meta = _extract_tpf_meta(wcs=wcs)
    return TPFFitsData(
        ref=TPFFitsRef(tic_id=int(tic_id), sector=int(sector), author="spoc"),
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


def _execute_localize_host(
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
    oot_margin_mult: float,
    oot_window_mult: float | None,
    centroid_method: str,
    prf_backend: str,
    baseline_shift_threshold: float,
    random_seed: int,
    input_resolution: dict[str, Any] | None,
    reference_sources_override: list[dict[str, Any]] | None = None,
    brightness_prior_enabled: bool = True,
    brightness_prior_weight: float = 40.0,
    brightness_prior_softening_mag: float = 2.5,
) -> dict[str, Any]:
    client = MASTClient()
    if not network_ok:
        if not sectors:
            raise BtvCliError(
                "--no-network requires explicit --sectors for cache-only light curve loading.",
                exit_code=EXIT_INPUT_ERROR,
            )
        lightcurves = []
        for sector in sectors:
            try:
                lightcurves.append(
                    client.download_lightcurve_cached(
                        tic_id=int(tic_id),
                        sector=int(sector),
                        flux_type="pdcsap",
                    )
                )
            except Exception:
                continue
    else:
        lightcurves = client.download_all_sectors(int(tic_id), flux_type="pdcsap", sectors=sectors)
    if not lightcurves:
        raise LightCurveNotFoundError(f"No sectors available for TIC {tic_id}")

    if len(lightcurves) == 1:
        stitched_lc = lightcurves[0]
    else:
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

    tpf_fits_list: list[TPFFitsData] = []
    for sector in selected:
        tpf_fits = _download_tpf_fits(
            client=client,
            tic_id=int(tic_id),
            sector=int(sector),
            network_ok=bool(network_ok),
        )
        if tpf_fits is not None:
            tpf_fits_list.append(tpf_fits)

    if not tpf_fits_list:
        raise LightCurveNotFoundError(f"TPF unavailable for TIC {tic_id}")

    if reference_sources_override is not None:
        reference_sources = [dict(src) for src in reference_sources_override]
        coordinate_source = "reference_sources_file"
    else:
        reference_sources, coordinate_source = _build_reference_sources(
            tic_id=int(tic_id),
            tpf_meta=dict(tpf_fits_list[0].meta or {}),
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            network_ok=bool(network_ok),
        )

    result = localize_transit_host_multi_sector(
        tpf_fits_list=tpf_fits_list,
        period_days=float(period_days),
        t0_btjd=float(t0_btjd),
        duration_hours=float(duration_hours),
        reference_sources=reference_sources,
        oot_margin_mult=float(oot_margin_mult),
        oot_window_mult=float(oot_window_mult) if oot_window_mult is not None else None,
        centroid_method=str(centroid_method).lower(),  # type: ignore[arg-type]
        prf_backend=str(prf_backend).lower(),  # type: ignore[arg-type]
        random_seed=int(random_seed),
        centroid_shift_threshold_pixels=float(baseline_shift_threshold),
        brightness_prior_enabled=bool(brightness_prior_enabled),
        brightness_prior_weight=float(brightness_prior_weight),
        brightness_prior_softening_mag=float(brightness_prior_softening_mag),
    )

    selected_sectors = [int(getattr(tpf.ref, "sector", -1)) for tpf in tpf_fits_list]
    return {
        "schema_version": CLI_LOCALIZE_HOST_SCHEMA_VERSION,
        "result": dict(result),
        "inputs_summary": {
            "input_resolution": input_resolution,
        },
        "provenance": {
            "selected_sectors": selected_sectors,
            "sectors_requested": [int(s) for s in sectors] if sectors else None,
            "sectors_used": sectors_used,
            "requested_sectors": [int(s) for s in tpf_sectors] if tpf_sectors else None,
            "tpf_sector_strategy": str(tpf_sector_strategy).lower(),
            "network_ok": bool(network_ok),
            "coordinate_source": str(coordinate_source),
            "centroid_method": str(centroid_method).lower(),
            "prf_backend": str(prf_backend).lower(),
            "baseline_shift_threshold": float(baseline_shift_threshold),
            "random_seed": int(random_seed),
            "brightness_prior_enabled": bool(brightness_prior_enabled),
            "brightness_prior_weight": float(brightness_prior_weight),
            "brightness_prior_softening_mag": float(brightness_prior_softening_mag),
        },
    }


@click.command("localize-host")
@click.argument("toi_arg", required=False)
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--period-days", type=float, default=None, help="Orbital period in days.")
@click.option("--t0-btjd", type=float, default=None, help="Reference epoch in BTJD.")
@click.option("--duration-hours", type=float, default=None, help="Transit duration in hours.")
@click.option("--depth-ppm", type=float, default=None, help="Transit depth in ppm.")
@click.option("--toi", type=str, default=None, help="Optional TOI label to resolve candidate inputs.")
@click.option("--ra-deg", type=float, default=None, help="Fallback target right ascension in degrees.")
@click.option("--dec-deg", type=float, default=None, help="Fallback target declination in degrees.")
@click.option(
    "--reference-sources-file",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Optional standardized reference sources JSON file (schema_version=reference_sources.v1).",
)
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
@click.option("--oot-margin-mult", type=float, default=1.5, show_default=True)
@click.option("--oot-window-mult", type=float, default=10.0, show_default=True)
@click.option(
    "--centroid-method",
    type=click.Choice(["centroid", "gaussian_fit"], case_sensitive=False),
    default="centroid",
    show_default=True,
)
@click.option(
    "--prf-backend",
    type=click.Choice(["prf_lite", "parametric", "instrument"], case_sensitive=False),
    default="prf_lite",
    show_default=True,
)
@click.option("--baseline-shift-threshold", type=float, default=0.5, show_default=True)
@click.option("--random-seed", type=int, default=42, show_default=True)
@click.option(
    "--brightness-prior/--no-brightness-prior",
    default=True,
    show_default=True,
    help="Apply a soft brightness prior that penalizes implausibly faint host hypotheses.",
)
@click.option(
    "--brightness-prior-weight",
    type=float,
    default=40.0,
    show_default=True,
    help="Penalty scale for brightness prior (higher => stronger down-weighting of faint neighbors).",
)
@click.option(
    "--brightness-prior-softening-mag",
    type=float,
    default=2.5,
    show_default=True,
    help="Delta-mag below which no brightness prior penalty is applied.",
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
def localize_host_command(
    toi_arg: str | None,
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
    toi: str | None,
    ra_deg: float | None,
    dec_deg: float | None,
    reference_sources_file: str | None,
    network_ok: bool,
    sectors: tuple[int, ...],
    tpf_sector_strategy: str,
    tpf_sectors: tuple[int, ...],
    oot_margin_mult: float,
    oot_window_mult: float,
    centroid_method: str,
    prf_backend: str,
    baseline_shift_threshold: float,
    random_seed: int,
    brightness_prior: bool,
    brightness_prior_weight: float,
    brightness_prior_softening_mag: float,
    output_path_arg: str,
) -> None:
    """Run multi-sector host localization for a single transit candidate."""
    out_path = resolve_optional_output_path(output_path_arg)
    strategy = str(tpf_sector_strategy).lower()
    if toi_arg is not None and toi is not None and str(toi_arg).strip() != str(toi).strip():
        raise BtvCliError(
            "Positional TOI argument and --toi must match when both are provided.",
            exit_code=EXIT_INPUT_ERROR,
        )
    resolved_toi_arg = toi if toi is not None else toi_arg

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
    reference_sources_override = None
    if reference_sources_file is not None:
        reference_sources_override = load_reference_sources_file(Path(reference_sources_file))

    (
        resolved_tic_id,
        resolved_period_days,
        resolved_t0_btjd,
        resolved_duration_hours,
        _resolved_depth_ppm,
        input_resolution,
    ) = _resolve_candidate_inputs(
        network_ok=network_ok,
        toi=resolved_toi_arg,
        tic_id=tic_id,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_ppm=depth_ppm,
    )

    try:
        payload = _execute_localize_host(
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
            oot_margin_mult=float(oot_margin_mult),
            oot_window_mult=float(oot_window_mult),
            centroid_method=str(centroid_method).lower(),
            prf_backend=str(prf_backend).lower(),
            baseline_shift_threshold=float(baseline_shift_threshold),
            random_seed=int(random_seed),
            input_resolution=input_resolution,
            reference_sources_override=reference_sources_override,
            brightness_prior_enabled=bool(brightness_prior),
            brightness_prior_weight=float(brightness_prior_weight),
            brightness_prior_softening_mag=float(brightness_prior_softening_mag),
        )
    except LightCurveNotFoundError as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except BtvCliError:
        raise
    except Exception as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_RUNTIME_ERROR) from exc

    dump_json_output(payload, out_path)


__all__ = ["localize_host_command"]
