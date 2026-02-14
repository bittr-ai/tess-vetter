"""`btv measure-sectors` command for per-sector transit depth measurements."""

from __future__ import annotations

from typing import Any

import click
import numpy as np

from bittr_tess_vetter.api.sector_metrics import (
    SectorEphemerisMetrics,
    compute_sector_ephemeris_metrics,
)
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
from bittr_tess_vetter.cli.vet_cli import (
    _detrend_lightcurve_for_vetting,
    _normalize_detrend_method,
    _resolve_candidate_inputs,
    _validate_detrend_args,
)
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError, MASTClient


def _metric_to_measurement(metric: SectorEphemerisMetrics, *, duration_hours: float) -> dict[str, Any]:
    depth_ppm = float(metric.depth_hat_ppm)
    depth_err_ppm = float(metric.depth_sigma_ppm)
    valid_depth = np.isfinite(depth_ppm)
    valid_err = np.isfinite(depth_err_ppm) and depth_err_ppm > 0.0
    quality_weight = 1.0 if valid_depth and valid_err and int(metric.n_transits) > 0 else 0.0
    return {
        "sector": int(metric.sector),
        "depth_ppm": depth_ppm,
        "depth_err_ppm": depth_err_ppm,
        "duration_hours": float(duration_hours),
        "duration_err_hours": 0.0,
        "n_transits": int(metric.n_transits),
        "shape_metric": float(metric.score),
        "quality_weight": float(quality_weight),
    }


def _execute_measure_sectors(
    *,
    tic_id: int,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    sectors: list[int] | None,
    flux_type: str,
    detrend: str | None,
    detrend_bin_hours: float,
    detrend_buffer: float,
    detrend_sigma_clip: float,
    input_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        time = np.asarray(lc.time, dtype=np.float64)
        flux = np.asarray(lc.flux, dtype=np.float64)
        flux_err = (
            np.asarray(lc.flux_err, dtype=np.float64)
            if lc.flux_err is not None
            else np.zeros_like(flux, dtype=np.float64)
        )
        quality = (
            np.asarray(lc.quality, dtype=np.int32)
            if getattr(lc, "quality", None) is not None
            else np.zeros(len(time), dtype=np.int32)
        )
        sector = np.full(len(time), int(lc.sector), dtype=np.int32)
    else:
        stitched_lc, stitched = stitch_lightcurve_data(lightcurves, tic_id=int(tic_id))
        time = np.asarray(stitched.time, dtype=np.float64)
        flux = np.asarray(stitched.flux, dtype=np.float64)
        flux_err = np.asarray(stitched.flux_err, dtype=np.float64)
        quality = np.asarray(stitched_lc.quality, dtype=np.int32)
        sector = np.asarray(stitched.sector, dtype=np.int32)

    detrend_provenance: dict[str, Any] | None = None
    if detrend is not None:
        candidate = Candidate(
            ephemeris=Ephemeris(
                period_days=float(period_days),
                t0_btjd=float(t0_btjd),
                duration_hours=float(duration_hours),
            )
        )
        lc_for_detrend = LightCurve(
            time=time,
            flux=flux,
            flux_err=flux_err,
            quality=quality,
            valid_mask=(quality == 0),
        )
        detrended_lc, detrend_provenance = _detrend_lightcurve_for_vetting(
            lc=lc_for_detrend,
            candidate=candidate,
            method=str(detrend),
            bin_hours=float(detrend_bin_hours),
            buffer_factor=float(detrend_buffer),
            clip_sigma=float(detrend_sigma_clip),
        )
        flux = np.asarray(detrended_lc.flux, dtype=np.float64)
        flux_err = np.asarray(detrended_lc.flux_err, dtype=np.float64)

    metrics = compute_sector_ephemeris_metrics(
        time=time,
        flux=flux,
        flux_err=flux_err,
        sector=sector,
        period_days=float(period_days),
        t0_btjd=float(t0_btjd),
        duration_hours=float(duration_hours),
    )

    sector_measurements = [
        _metric_to_measurement(metric, duration_hours=float(duration_hours)) for metric in metrics
    ]
    sectors_loaded = sorted(
        {int(lc.sector) for lc in lightcurves if getattr(lc, "sector", None) is not None}
    )
    return {
        "schema_version": 1,
        "sector_measurements": sector_measurements,
        "provenance": {
            "command": "measure-sectors",
            "tic_id": int(tic_id),
            "period_days": float(period_days),
            "t0_btjd": float(t0_btjd),
            "duration_hours": float(duration_hours),
            "flux_type": str(flux_type).lower(),
            "requested_sectors": [int(s) for s in sectors] if sectors else None,
            "loaded_sectors": sectors_loaded,
            "sectors_requested": [int(s) for s in sectors] if sectors else None,
            "sectors_used": sectors_loaded,
            "detrend": detrend_provenance
            if detrend_provenance is not None
            else {"applied": False, "method": None},
            "input_resolution": input_resolution,
        },
    }


@click.command("measure-sectors")
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
    "--detrend",
    type=click.Choice(["transit_masked_bin_median"], case_sensitive=False),
    default=None,
    help="Optional pre-measurement detrend method (matches `btv vet --detrend`).",
)
@click.option("--detrend-bin-hours", type=float, default=6.0, show_default=True)
@click.option("--detrend-buffer", type=float, default=2.0, show_default=True)
@click.option("--detrend-sigma-clip", type=float, default=5.0, show_default=True)
@click.option(
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
def measure_sectors_command(
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
    toi: str | None,
    network_ok: bool,
    sectors: tuple[int, ...],
    flux_type: str,
    detrend: str | None,
    detrend_bin_hours: float,
    detrend_buffer: float,
    detrend_sigma_clip: float,
    output_path_arg: str,
) -> None:
    """Measure per-sector transit depths for V21 sector consistency checks."""
    out_path = resolve_optional_output_path(output_path_arg)

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
    detrend_method = _normalize_detrend_method(detrend)
    if detrend_method is not None:
        try:
            _validate_detrend_args(
                detrend_bin_hours=float(detrend_bin_hours),
                detrend_buffer=float(detrend_buffer),
                detrend_sigma_clip=float(detrend_sigma_clip),
            )
        except BtvCliError:
            raise
        except Exception as exc:
            raise BtvCliError(str(exc), exit_code=EXIT_INPUT_ERROR) from exc

    try:
        payload = _execute_measure_sectors(
            tic_id=resolved_tic_id,
            period_days=resolved_period_days,
            t0_btjd=resolved_t0_btjd,
            duration_hours=resolved_duration_hours,
            sectors=list(sectors) if sectors else None,
            flux_type=str(flux_type).lower(),
            detrend=detrend_method,
            detrend_bin_hours=float(detrend_bin_hours),
            detrend_buffer=float(detrend_buffer),
            detrend_sigma_clip=float(detrend_sigma_clip),
            input_resolution=input_resolution,
        )
    except LightCurveNotFoundError as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except BtvCliError:
        raise
    except Exception as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_RUNTIME_ERROR) from exc

    dump_json_output(payload, out_path)


__all__ = ["measure_sectors_command"]
