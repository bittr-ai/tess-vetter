"""`btv measure-sectors` command for per-sector transit depth measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import numpy as np

from bittr_tess_vetter.api.sector_consistency import SectorMeasurement, compute_sector_consistency
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
    load_json_file,
    resolve_optional_output_path,
)
from bittr_tess_vetter.cli.diagnostics_report_inputs import (
    choose_effective_sectors,
    load_lightcurves_with_sector_policy,
    resolve_inputs_from_report_file,
)
from bittr_tess_vetter.cli.vet_cli import (
    _detrend_lightcurve_for_vetting,
    _normalize_detrend_method,
    _resolve_candidate_inputs,
    _validate_detrend_args,
)
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError


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
    sectors_explicit: bool,
    sector_selection_source: str,
    flux_type: str,
    network_ok: bool,
    detrend: str | None,
    detrend_bin_hours: float,
    detrend_buffer: float,
    detrend_sigma_clip: float,
    input_resolution: dict[str, Any] | None = None,
    report_file_path: str | None = None,
) -> dict[str, Any]:
    lightcurves, sector_load_path = load_lightcurves_with_sector_policy(
        tic_id=int(tic_id),
        flux_type=str(flux_type).lower(),
        sectors=sectors,
        explicit_sectors=bool(sectors_explicit),
        network_ok=bool(network_ok),
    )

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
    consistency = _build_consistency_enrichment(sector_measurements)
    recommended_sectors = _recommended_sectors(sector_measurements, consistency=consistency)
    routing = _build_gating_routing(
        sector_measurements=sector_measurements,
        recommended_sectors=recommended_sectors,
        consistency=consistency,
    )
    consistency.update(routing)
    sectors_loaded = sorted(
        {int(lc.sector) for lc in lightcurves if getattr(lc, "sector", None) is not None}
    )
    return {
        "schema_version": 1,
        "sector_measurements": sector_measurements,
        "consistency": consistency,
        "recommended_sectors": recommended_sectors,
        "recommended_sector_criterion": (
            "quality_weight > 0, n_transits > 0, finite depth/error, and not flagged as outlier "
            "(|depth - weighted_mean| / depth_err_ppm <= 3.0); fallback: all measured sectors."
        ),
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
            "sector_selection_source": str(sector_selection_source),
            "sector_load_path": str(sector_load_path),
            "detrend": detrend_provenance
            if detrend_provenance is not None
            else {"applied": False, "method": None},
            "input_resolution": input_resolution,
            "report_file": report_file_path,
        },
    }


@click.command("measure-sectors")
@click.argument("toi_arg", required=False)
@click.option("--tic-id", type=int, default=None, help="TIC identifier.")
@click.option("--period-days", type=float, default=None, help="Orbital period in days.")
@click.option("--t0-btjd", type=float, default=None, help="Reference epoch in BTJD.")
@click.option("--duration-hours", type=float, default=None, help="Transit duration in hours.")
@click.option("--depth-ppm", type=float, default=None, help="Transit depth in ppm.")
@click.option("--toi", type=str, default=None, help="Optional TOI label to resolve candidate inputs.")
@click.option("--report-file", type=str, default=None, help="Optional report JSON path for candidate inputs.")
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
@click.option("--resume", is_flag=True, default=False, help="Skip when completed output already exists.")
@click.option(
    "-o",
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
def measure_sectors_command(
    toi_arg: str | None,
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
    toi: str | None,
    report_file: str | None,
    network_ok: bool,
    sectors: tuple[int, ...],
    flux_type: str,
    detrend: str | None,
    detrend_bin_hours: float,
    detrend_buffer: float,
    detrend_sigma_clip: float,
    resume: bool,
    output_path_arg: str,
) -> None:
    """Measure per-sector transit depths for V21 sector consistency checks."""
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
    if resume and out_path is not None and _is_completed_output(out_path):
        click.echo(f"Skipped resume: existing completed output at {out_path}")
        return

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
        resolved_period_days = float(resolved_from_report.period_days)
        resolved_t0_btjd = float(resolved_from_report.t0_btjd)
        resolved_duration_hours = float(resolved_from_report.duration_hours)
        input_resolution = dict(resolved_from_report.input_resolution)
        report_file_path = str(resolved_from_report.report_file_path)
        report_sectors_used = (
            [int(s) for s in resolved_from_report.sectors_used]
            if resolved_from_report.sectors_used is not None
            else None
        )
    else:
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

    effective_sectors, sectors_explicit, sector_selection_source = choose_effective_sectors(
        sectors_arg=sectors,
        report_sectors_used=report_sectors_used,
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
            sectors=effective_sectors,
            sectors_explicit=bool(sectors_explicit),
            sector_selection_source=sector_selection_source,
            flux_type=str(flux_type).lower(),
            network_ok=bool(network_ok),
            detrend=detrend_method,
            detrend_bin_hours=float(detrend_bin_hours),
            detrend_buffer=float(detrend_buffer),
            detrend_sigma_clip=float(detrend_sigma_clip),
            input_resolution=input_resolution,
            report_file_path=report_file_path,
        )
    except LightCurveNotFoundError as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except BtvCliError:
        raise
    except Exception as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_RUNTIME_ERROR) from exc

    dump_json_output(payload, out_path)


def _is_completed_output(path: Path) -> bool:
    try:
        payload = load_json_file(path, label="measure-sectors output")
    except BtvCliError:
        return False

    rows = payload.get("sector_measurements")
    if not isinstance(rows, list):
        return False
    if payload.get("schema_version") != 1:
        return False
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return False
    return str(provenance.get("command") or "") == "measure-sectors"


def _build_consistency_enrichment(sector_measurements: list[dict[str, Any]]) -> dict[str, Any]:
    measurements: list[SectorMeasurement] = []
    for row in sector_measurements:
        try:
            measurements.append(
                SectorMeasurement(
                    sector=int(row.get("sector")),
                    depth_ppm=float(row.get("depth_ppm")),
                    depth_err_ppm=float(row.get("depth_err_ppm")),
                    duration_hours=float(row.get("duration_hours", 0.0) or 0.0),
                    duration_err_hours=float(row.get("duration_err_hours", 0.0) or 0.0),
                    n_transits=int(row.get("n_transits", 0) or 0),
                    shape_metric=float(row.get("shape_metric", 0.0) or 0.0),
                    quality_weight=float(row.get("quality_weight", 0.0) or 0.0),
                )
            )
        except (TypeError, ValueError):
            continue

    valid = [
        m
        for m in measurements
        if float(m.quality_weight) > 0.0
        and np.isfinite(float(m.depth_ppm))
        and np.isfinite(float(m.depth_err_ppm))
        and float(m.depth_err_ppm) > 0.0
    ]
    if valid:
        depths = np.asarray([float(m.depth_ppm) for m in valid], dtype=np.float64)
        depth_median_ppm = float(np.median(depths))
        depth_std_ppm = float(np.std(depths, ddof=0))
    else:
        depth_median_ppm = None
        depth_std_ppm = None

    chi2: float | None = None
    chi2_dof: int | None = None
    if len(valid) >= 2:
        depths = np.asarray([float(m.depth_ppm) for m in valid], dtype=np.float64)
        errors = np.asarray([float(m.depth_err_ppm) for m in valid], dtype=np.float64)
        inv_var = 1.0 / (errors**2)
        weighted_mean = float(np.sum(depths * inv_var) / np.sum(inv_var))
        chi2 = float(np.sum(((depths - weighted_mean) / errors) ** 2))
        chi2_dof = int(len(valid) - 1)

    verdict, outlier_sectors, chi2_pvalue = compute_sector_consistency(measurements)
    return {
        "chi2": chi2,
        "chi2_dof": chi2_dof,
        "chi2_pvalue": float(chi2_pvalue),
        "verdict": str(verdict),
        "depth_median_ppm": depth_median_ppm,
        "depth_std_ppm": depth_std_ppm,
        "outlier_sectors": [int(s) for s in outlier_sectors],
        "outlier_criterion": "|depth - weighted_mean| / depth_err_ppm > 3.0",
    }


def _recommended_sectors(
    sector_measurements: list[dict[str, Any]],
    *,
    consistency: dict[str, Any] | None = None,
) -> list[int]:
    consistency = consistency if consistency is not None else _build_consistency_enrichment(sector_measurements)
    outlier_set = {int(s) for s in consistency.get("outlier_sectors", [])}
    recommended: list[int] = []
    all_sectors: list[int] = []
    for row in sector_measurements:
        try:
            sector = int(row.get("sector"))
            all_sectors.append(sector)
            quality_weight = float(row.get("quality_weight", 0.0) or 0.0)
            n_transits = int(row.get("n_transits", 0) or 0)
            depth = float(row.get("depth_ppm"))
            depth_err = float(row.get("depth_err_ppm"))
        except (TypeError, ValueError):
            continue
        if (
            quality_weight > 0.0
            and n_transits > 0
            and np.isfinite(depth)
            and np.isfinite(depth_err)
            and depth_err > 0.0
            and sector not in outlier_set
        ):
            recommended.append(sector)
    chosen = recommended if recommended else all_sectors
    return sorted({int(s) for s in chosen})


def _build_gating_routing(
    *,
    sector_measurements: list[dict[str, Any]],
    recommended_sectors: list[int],
    consistency: dict[str, Any],
) -> dict[str, Any]:
    n_total = len(sector_measurements)
    n_recommended = len(recommended_sectors)
    verdict = str(consistency.get("verdict") or "")

    if n_recommended < 2:
        return {
            "gating_actionable": False,
            "action_hint": "DETREND_RECOMMENDED",
            "reason": "recommended_sector_count_lt_2",
            "n_sectors_total": int(n_total),
            "n_sectors_recommended": int(n_recommended),
        }

    if verdict == "CONSISTENT":
        return {
            "gating_actionable": True,
            "action_hint": "USE_ALL_SECTORS",
            "reason": "sector_depths_consistent",
            "n_sectors_total": int(n_total),
            "n_sectors_recommended": int(n_recommended),
        }

    return {
        "gating_actionable": True,
        "action_hint": "SECTOR_GATING_RECOMMENDED",
        "reason": "sector_depths_inconsistent",
        "n_sectors_total": int(n_total),
        "n_sectors_recommended": int(n_recommended),
    }


__all__ = ["measure_sectors_command"]
