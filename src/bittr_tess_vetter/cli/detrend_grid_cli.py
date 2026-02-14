"""`btv detrend-grid` command for deterministic detrending variant sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import numpy as np

from bittr_tess_vetter.api.ephemeris_sensitivity_sweep import compute_sensitivity_sweep_numpy
from bittr_tess_vetter.api.ephemeris_specificity import SmoothTemplateConfig
from bittr_tess_vetter.api.stitch import stitch_lightcurve_data
from bittr_tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_RUNTIME_ERROR,
    BtvCliError,
    dump_json_output,
    load_json_file,
    resolve_optional_output_path,
)
from bittr_tess_vetter.cli.vet_cli import _resolve_candidate_inputs
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError, MASTClient
from bittr_tess_vetter.validation.detrend_grid_defaults import (
    DEFAULT_TRANSIT_MASKED_BIN_HOURS,
    DEFAULT_TRANSIT_MASKED_BUFFER_FACTORS,
    DEFAULT_TRANSIT_MASKED_SIGMA_CLIPS,
    expanded_detrender_count,
    resolve_detrend_grid_axes,
)


def _to_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _row_score_abs(row: dict[str, Any]) -> float | None:
    score = row.get("score")
    if score is None:
        return None
    try:
        return abs(float(score))
    except (TypeError, ValueError):
        return None


def _rank_sweep_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = [row for row in rows if str(row.get("status")) == "ok" and row.get("score") is not None]
    failed_rows = [row for row in rows if row not in ok_rows]

    ok_rows_sorted = sorted(
        ok_rows,
        key=lambda row: (-(_row_score_abs(row) or 0.0), str(row.get("variant_id") or "")),
    )
    failed_rows_sorted = sorted(failed_rows, key=lambda row: str(row.get("variant_id") or ""))

    ranked: list[dict[str, Any]] = []
    for idx, row in enumerate(ok_rows_sorted, start=1):
        score_abs = _row_score_abs(row)
        ranked.append({**row, "rank": int(idx), "score_abs": float(score_abs or 0.0)})
    for row in failed_rows_sorted:
        ranked.append({**row, "rank": None, "score_abs": None})
    return ranked


def _annotate_depth_semantics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    note = (
        "Depth is from template least-squares fitting, optimized for variant ranking. "
        "For downstream use (e.g. FPP), measure depth with btv vet --detrend using this variant's config."
    )
    annotated: list[dict[str, Any]] = []
    for row in rows:
        annotated.append(
            {
                **row,
                "depth_method": "template_ls",
                "depth_note": note,
            }
        )
    return annotated


def _build_recommended_next_step(best_variant: dict[str, Any] | None) -> str | None:
    if not isinstance(best_variant, dict):
        return None
    config = best_variant.get("config")
    if not isinstance(config, dict):
        return None
    detrender = config.get("detrender")
    if str(detrender) != "transit_masked_bin_median":
        return None
    bin_hours = config.get("detrender_bin_hours", 6.0)
    buffer_factor = config.get("detrender_buffer_factor", 2.0)
    sigma_clip = config.get("detrender_sigma_clip", 5.0)
    return (
        "btv vet --detrend transit_masked_bin_median "
        f"--detrend-bin-hours {float(bin_hours):g} "
        f"--detrend-buffer {float(buffer_factor):g} "
        f"--detrend-sigma-clip {float(sigma_clip):g}"
    )


def _extract_vet_summary(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    summary_raw = payload.get("summary")
    if isinstance(summary_raw, dict):
        return summary_raw, "payload.summary"
    return payload, "payload_root"


def _build_check_resolution_note(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return None
    concerns_raw = summary.get("concerns")
    concerns = {str(item) for item in concerns_raw if item is not None} if isinstance(concerns_raw, list) else set()
    disposition_hint = str(summary.get("disposition_hint") or "")

    has_v16_model_competition_concern = "MODEL_PREFERS_NON_TRANSIT" in concerns
    disposition_requests_model_review = disposition_hint == "needs_model_competition_review"
    if not has_v16_model_competition_concern and not disposition_requests_model_review:
        return None

    return {
        "check_id": "V16",
        "reason": "model_competition_concern",
        "triggers": {
            "concerns": sorted(concerns),
            "disposition_hint": disposition_hint,
        },
    }


def _build_best_variant(
    *,
    best_variant_id: str | None,
    ranked_rows: list[dict[str, Any]],
    metric_variance: float | None,
    stability_threshold: float,
    stable: bool,
    notes: list[str],
) -> dict[str, Any] | None:
    if not ranked_rows:
        return None

    best_row: dict[str, Any] | None = None
    if best_variant_id is not None:
        for row in ranked_rows:
            if str(row.get("variant_id")) == str(best_variant_id):
                best_row = row
                break
    if best_row is None:
        for row in ranked_rows:
            if row.get("rank") == 1:
                best_row = row
                break
    if best_row is None:
        return None

    rationale: list[str] = ["Selected highest absolute sensitivity-sweep score among successful variants."]
    if metric_variance is not None:
        relation = "<=" if stable else ">"
        rationale.append(
            f"Stability metric {metric_variance:.6g} {relation} "
            f"threshold {float(stability_threshold):.6g}."
        )
    if notes:
        rationale.extend([str(note) for note in notes])

    return {
        "variant_id": best_row.get("variant_id"),
        "rank": best_row.get("rank"),
        "score": best_row.get("score"),
        "score_abs": best_row.get("score_abs"),
        "status": best_row.get("status"),
        "config": best_row.get("variant_config"),
        "rationale": rationale,
        "row": best_row,
    }


def _execute_detrend_grid(
    *,
    tic_id: int,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    sectors: list[int] | None,
    flux_type: str,
    random_seed: int,
    downsample_levels: list[int] | None,
    outlier_policies: list[str] | None,
    detrenders: list[str] | None,
    include_celerite2_sho: bool,
    stability_threshold: float,
    gp_max_iterations: int,
    gp_timeout_seconds: float,
    checks: list[str] | None,
    toi: str | None,
    input_resolution: dict[str, Any] | None = None,
    check_resolution_note: dict[str, Any] | None = None,
    vet_summary_provenance: dict[str, Any] | None = None,
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
        stitched_lc = lightcurves[0]
    else:
        stitched_lc, _ = stitch_lightcurve_data(lightcurves, tic_id=int(tic_id))

    time = _to_float_array(stitched_lc.time)
    flux = _to_float_array(stitched_lc.flux)
    if stitched_lc.flux_err is None:
        fallback_err = float(np.nanstd(flux))
        if not np.isfinite(fallback_err) or fallback_err <= 0:
            fallback_err = 1e-3
        flux_err = np.full_like(flux, fallback_err, dtype=np.float64)
        flux_err_source = "estimated_from_flux_std"
    else:
        flux_err = _to_float_array(stitched_lc.flux_err)
        flux_err_source = "lightcurve_flux_err"

    effective_downsample_levels, effective_outlier_policies, effective_detrenders = resolve_detrend_grid_axes(
        downsample_levels=downsample_levels,
        outlier_policies=outlier_policies,
        detrenders=detrenders,
    )
    expanded_count = expanded_detrender_count(effective_detrenders)
    cross_product_variant_count = len(effective_downsample_levels) * len(effective_outlier_policies) * int(
        expanded_count
    )

    sweep = compute_sensitivity_sweep_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=float(period_days),
        t0_btjd=float(t0_btjd),
        duration_hours=float(duration_hours),
        config=SmoothTemplateConfig(),
        downsample_levels=effective_downsample_levels,
        outlier_policies=effective_outlier_policies,
        detrenders=effective_detrenders,
        include_celerite2_sho=bool(include_celerite2_sho),
        stability_threshold=float(stability_threshold),
        random_seed=int(random_seed),
        gp_max_iterations=int(gp_max_iterations),
        gp_timeout_seconds=float(gp_timeout_seconds),
    )
    payload = sweep.to_dict()
    raw_rows = payload.get("sweep_table")
    rows = [row for row in raw_rows if isinstance(row, dict)] if isinstance(raw_rows, list) else []
    rows_annotated = _annotate_depth_semantics(rows)
    ranked_rows = _rank_sweep_rows(rows_annotated)

    best_variant = _build_best_variant(
        best_variant_id=payload.get("best_variant_id"),
        ranked_rows=ranked_rows,
        metric_variance=(
            float(payload["metric_variance"])
            if payload.get("metric_variance") is not None
            else None
        ),
        stability_threshold=float(payload.get("stability_threshold") or stability_threshold),
        stable=bool(payload.get("stable", False)),
        notes=[str(note) for note in (payload.get("notes") or []) if note is not None],
    )

    sectors_loaded = sorted(
        {int(lc.sector) for lc in lightcurves if getattr(lc, "sector", None) is not None}
    )
    recommended_next_step = _build_recommended_next_step(best_variant)
    out_payload: dict[str, Any] = {
        "schema_version": 1,
        **{**payload, "sweep_table": rows_annotated},
        "ranked_sweep_table": ranked_rows,
        "best_variant": best_variant,
        "recommended_next_step": recommended_next_step,
        "variant_axes": {
            "downsample_levels": effective_downsample_levels,
            "outlier_policies": effective_outlier_policies,
            "detrenders": effective_detrenders,
            "transit_masked_bin_median": {
                "bin_hours": list(DEFAULT_TRANSIT_MASKED_BIN_HOURS),
                "buffer_factor": list(DEFAULT_TRANSIT_MASKED_BUFFER_FACTORS),
                "sigma_clip": list(DEFAULT_TRANSIT_MASKED_SIGMA_CLIPS),
            },
            "include_celerite2_sho": bool(include_celerite2_sho),
        },
        "provenance": {
            "command": "detrend-grid",
            "tic_id": int(tic_id),
            "toi": toi,
            "period_days": float(period_days),
            "t0_btjd": float(t0_btjd),
            "duration_hours": float(duration_hours),
            "random_seed": int(random_seed),
            "flux_type": str(flux_type).lower(),
            "requested_sectors": [int(s) for s in sectors] if sectors else None,
            "loaded_sectors": sectors_loaded,
            "requested_checks": [str(item) for item in checks] if checks else None,
            "input_resolution": input_resolution,
            "flux_err_source": flux_err_source,
            "vet_summary": vet_summary_provenance,
            "grid_config": {
                "downsample_levels": list(downsample_levels) if downsample_levels is not None else None,
                "outlier_policies": list(outlier_policies) if outlier_policies is not None else None,
                "detrenders": list(detrenders) if detrenders is not None else None,
                "include_celerite2_sho": bool(include_celerite2_sho),
                "stability_threshold": float(stability_threshold),
            },
            "effective_grid_config": {
                "downsample_levels": effective_downsample_levels,
                "outlier_policies": effective_outlier_policies,
                "detrenders": effective_detrenders,
                "transit_masked_bin_median": {
                    "bin_hours": list(DEFAULT_TRANSIT_MASKED_BIN_HOURS),
                    "buffer_factor": list(DEFAULT_TRANSIT_MASKED_BUFFER_FACTORS),
                    "sigma_clip": list(DEFAULT_TRANSIT_MASKED_SIGMA_CLIPS),
                },
                "include_celerite2_sho": bool(include_celerite2_sho),
                "stability_threshold": float(stability_threshold),
            },
            "variant_counts": {
                "cross_product": int(cross_product_variant_count),
                "with_optional_gp": int(
                    cross_product_variant_count + (1 if bool(include_celerite2_sho) else 0)
                ),
            },
            "runtime_caps": {
                "gp_max_iterations": int(gp_max_iterations),
                "gp_timeout_seconds": float(gp_timeout_seconds),
            },
        },
    }
    if check_resolution_note is not None:
        out_payload["check_resolution_note"] = check_resolution_note
        provenance = out_payload.get("provenance")
        if isinstance(provenance, dict):
            provenance["check_resolution_note"] = check_resolution_note
    return out_payload


@click.command("detrend-grid")
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
    "--random-seed",
    type=int,
    default=0,
    show_default=True,
    help="Deterministic seed used for downsampling and optional GP components.",
)
@click.option(
    "--downsample-level",
    "downsample_levels",
    multiple=True,
    type=int,
    help="Repeatable downsampling factors (default: 1,2,5).",
)
@click.option(
    "--outlier-policy",
    "outlier_policies",
    multiple=True,
    type=str,
    help="Repeatable outlier policies (default: none,sigma_clip_4).",
)
@click.option(
    "--detrender",
    "detrenders",
    multiple=True,
    type=str,
    help="Repeatable detrenders (default: none,running_median_0.5d,transit_masked_bin_median).",
)
@click.option(
    "--include-celerite2-sho/--no-include-celerite2-sho",
    default=False,
    show_default=True,
    help="Include optional celerite2 SHO variant in the sweep.",
)
@click.option(
    "--stability-threshold",
    type=float,
    default=0.20,
    show_default=True,
    help="Threshold for sweep stability metric.",
)
@click.option(
    "--gp-max-iterations",
    type=int,
    default=100,
    show_default=True,
    help="Maximum optimizer iterations for optional celerite2 variant.",
)
@click.option(
    "--gp-timeout-seconds",
    type=float,
    default=30.0,
    show_default=True,
    help="Per-variant timeout for optional celerite2 optimization.",
)
@click.option("--check", "checks", multiple=True, help="Optional check IDs to annotate provenance.")
@click.option(
    "--vet-summary-path",
    type=str,
    default=None,
    help="Optional vet JSON file path (full payload or summary block) to contextualize V16 concerns.",
)
@click.option(
    "--out",
    "output_path_arg",
    type=str,
    default="-",
    show_default=True,
    help="JSON output path; '-' writes to stdout.",
)
def detrend_grid_command(
    tic_id: int | None,
    period_days: float | None,
    t0_btjd: float | None,
    duration_hours: float | None,
    depth_ppm: float | None,
    toi: str | None,
    network_ok: bool,
    sectors: tuple[int, ...],
    flux_type: str,
    random_seed: int,
    downsample_levels: tuple[int, ...],
    outlier_policies: tuple[str, ...],
    detrenders: tuple[str, ...],
    include_celerite2_sho: bool,
    stability_threshold: float,
    gp_max_iterations: int,
    gp_timeout_seconds: float,
    checks: tuple[str, ...],
    vet_summary_path: str | None,
    output_path_arg: str,
) -> None:
    """Sweep detrending variants and emit ranked machine-readable diagnostics."""
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

    check_resolution_note: dict[str, Any] | None = None
    vet_summary_provenance: dict[str, Any] | None = None
    if vet_summary_path:
        summary_payload = load_json_file(Path(vet_summary_path), label="vet summary file")
        summary, summary_source = _extract_vet_summary(summary_payload)
        check_resolution_note = _build_check_resolution_note(summary)
        vet_summary_provenance = {
            "source_path": str(vet_summary_path),
            "summary_source": summary_source,
        }

    try:
        payload = _execute_detrend_grid(
            tic_id=resolved_tic_id,
            period_days=resolved_period_days,
            t0_btjd=resolved_t0_btjd,
            duration_hours=resolved_duration_hours,
            sectors=list(sectors) if sectors else None,
            flux_type=str(flux_type).lower(),
            random_seed=int(random_seed),
            downsample_levels=list(downsample_levels) if downsample_levels else None,
            outlier_policies=list(outlier_policies) if outlier_policies else None,
            detrenders=list(detrenders) if detrenders else None,
            include_celerite2_sho=bool(include_celerite2_sho),
            stability_threshold=float(stability_threshold),
            gp_max_iterations=int(gp_max_iterations),
            gp_timeout_seconds=float(gp_timeout_seconds),
            checks=list(checks) if checks else None,
            toi=toi,
            input_resolution=input_resolution,
            check_resolution_note=check_resolution_note,
            vet_summary_provenance=vet_summary_provenance,
        )
    except LightCurveNotFoundError as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_DATA_UNAVAILABLE) from exc
    except BtvCliError:
        raise
    except Exception as exc:
        raise BtvCliError(str(exc), exit_code=EXIT_RUNTIME_ERROR) from exc

    dump_json_output(payload, out_path)


__all__ = ["detrend_grid_command"]
