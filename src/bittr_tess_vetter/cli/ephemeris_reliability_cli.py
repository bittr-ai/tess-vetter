"""`btv ephemeris-reliability` command for reliability-regime diagnostics."""

from __future__ import annotations

from typing import Any

import click
import numpy as np

from bittr_tess_vetter.api import ephemeris_reliability as ephemeris_reliability_api
from bittr_tess_vetter.api.ephemeris_specificity import SmoothTemplateConfig
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
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError, MASTClient


def _derive_ephemeris_reliability_verdict(result_payload: Any) -> tuple[str | None, str | None]:
    if not isinstance(result_payload, dict):
        return None, None
    interpretation_label = result_payload.get("interpretation_label")
    if interpretation_label is not None:
        return str(interpretation_label), "$.result.interpretation_label"
    label = result_payload.get("label")
    if label is not None:
        return str(label), "$.result.label"
    reliability_label = result_payload.get("reliability_label")
    if reliability_label is not None:
        return str(reliability_label), "$.result.reliability_label"
    return None, None


@click.command("ephemeris-reliability")
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
@click.option("--ingress-egress-fraction", type=float, default=0.2, show_default=True)
@click.option("--sharpness", type=float, default=30.0, show_default=True)
@click.option("--n-phase-shifts", type=int, default=200, show_default=True)
@click.option(
    "--phase-shift-strategy",
    type=click.Choice(["grid", "random"], case_sensitive=False),
    default="grid",
    show_default=True,
)
@click.option("--random-seed", type=int, default=0, show_default=True)
@click.option("--period-jitter-frac", type=float, default=0.002, show_default=True)
@click.option("--period-jitter-n", type=int, default=21, show_default=True)
@click.option("--t0-scan-n", type=int, default=81, show_default=True)
@click.option("--t0-scan-half-span-minutes", type=float, default=None)
@click.option(
    "-o",
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
def ephemeris_reliability_command(
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
    ingress_egress_fraction: float,
    sharpness: float,
    n_phase_shifts: int,
    phase_shift_strategy: str,
    random_seed: int,
    period_jitter_frac: float,
    period_jitter_n: int,
    t0_scan_n: int,
    t0_scan_half_span_minutes: float | None,
    output_path_arg: str,
) -> None:
    """Compute ephemeris reliability regime diagnostics and emit JSON."""
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
        lightcurves = MASTClient().download_all_sectors(
            tic_id=int(resolved_tic_id),
            flux_type=str(flux_type).lower(),
            sectors=list(sectors) if sectors else None,
        )
        if not lightcurves:
            raise LightCurveNotFoundError(f"No sectors available for TIC {resolved_tic_id}")

        if len(lightcurves) == 1:
            lc = lightcurves[0]
        else:
            lc, _ = stitch_lightcurve_data(lightcurves, tic_id=int(resolved_tic_id))

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
            if getattr(lc, "quality", None) is not None
            else np.zeros_like(flux, dtype=np.int32)
        )
        valid_mask = (
            np.asarray(lc.valid_mask, dtype=bool)
            if getattr(lc, "valid_mask", None) is not None
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
            raise LightCurveNotFoundError(
                f"No valid finite cadences available for TIC {resolved_tic_id}"
            )
        time = time[valid]
        flux = flux[valid]
        flux_err = flux_err[valid]

        config = SmoothTemplateConfig(
            ingress_egress_fraction=float(ingress_egress_fraction),
            sharpness=float(sharpness),
        )
        result = ephemeris_reliability_api.compute_reliability_regime_numpy(
            time=time,
            flux=flux,
            flux_err=flux_err,
            period_days=float(resolved_period_days),
            t0_btjd=float(resolved_t0_btjd),
            duration_hours=float(resolved_duration_hours),
            config=config,
            n_phase_shifts=int(n_phase_shifts),
            phase_shift_strategy=str(phase_shift_strategy).lower(),  # type: ignore[arg-type]
            random_seed=int(random_seed),
            period_jitter_frac=float(period_jitter_frac),
            period_jitter_n=int(period_jitter_n),
            t0_scan_n=int(t0_scan_n),
            t0_scan_half_span_minutes=(
                float(t0_scan_half_span_minutes)
                if t0_scan_half_span_minutes is not None
                else None
            ),
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
    options = {
        "network_ok": bool(network_ok),
        "sectors": [int(v) for v in sectors] if sectors else None,
        "flux_type": str(flux_type).lower(),
        "ingress_egress_fraction": float(ingress_egress_fraction),
        "sharpness": float(sharpness),
        "n_phase_shifts": int(n_phase_shifts),
        "phase_shift_strategy": str(phase_shift_strategy).lower(),
        "random_seed": int(random_seed),
        "period_jitter_frac": float(period_jitter_frac),
        "period_jitter_n": int(period_jitter_n),
        "t0_scan_n": int(t0_scan_n),
        "t0_scan_half_span_minutes": (
            float(t0_scan_half_span_minutes)
            if t0_scan_half_span_minutes is not None
            else None
        ),
    }
    result_payload = result.to_dict()
    verdict, verdict_source = _derive_ephemeris_reliability_verdict(result_payload)
    result_payload["verdict"] = verdict
    result_payload["verdict_source"] = verdict_source
    payload = {
        "schema_version": "cli.ephemeris_reliability.v1",
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


__all__ = ["ephemeris_reliability_command"]
