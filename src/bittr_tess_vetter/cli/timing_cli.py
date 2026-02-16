"""`btv timing` command for transit timing diagnostics."""

from __future__ import annotations

from typing import Any

import click
import numpy as np

from bittr_tess_vetter import api
from bittr_tess_vetter.api import ephemeris_refinement as ephemeris_refinement_api
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
from bittr_tess_vetter.cli.vet_cli import _resolve_candidate_inputs
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError, MASTClient, TargetNotFoundError


def _to_jsonable_result(result: Any) -> Any:
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json", exclude_none=True)
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return result


def _derive_timing_verdict(next_actions: Any) -> tuple[str | None, str | None]:
    if isinstance(next_actions, dict):
        primary_recommendation = next_actions.get("primary_recommendation")
        if primary_recommendation is not None:
            return str(primary_recommendation), "$.next_actions.primary_recommendation"
    if isinstance(next_actions, list) and next_actions:
        first_action = next_actions[0]
        if isinstance(first_action, dict):
            code = first_action.get("code")
            if code is not None:
                return str(code), "$.next_actions[0].code"
            action = first_action.get("action")
            if action is not None:
                return str(action), "$.next_actions[0].action"
    return None, None


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


def _build_alignment_metadata(
    *,
    prealign_requested: bool,
    prealign_applied: bool,
    candidate_initial: Candidate,
    candidate_final: Candidate,
    n_transits_pre: int,
    n_transits_post: int,
    prealign_score_z: float | None,
    prealign_error: str | None,
) -> dict[str, Any]:
    delta_t0_minutes = (
        float(candidate_final.ephemeris.t0_btjd - candidate_initial.ephemeris.t0_btjd) * 24.0 * 60.0
    )
    delta_period_ppm = (
        (float(candidate_final.ephemeris.period_days - candidate_initial.ephemeris.period_days) * 1e6)
        / max(float(candidate_initial.ephemeris.period_days), 1e-12)
    )

    if prealign_error:
        alignment_quality = "prealign_error_fallback"
    elif not prealign_requested:
        alignment_quality = "disabled"
    elif prealign_applied and n_transits_post > n_transits_pre:
        alignment_quality = "improved_transit_recovery"
    elif prealign_applied and n_transits_post == n_transits_pre:
        alignment_quality = "unchanged"
    elif prealign_applied and n_transits_post < n_transits_pre:
        alignment_quality = "degraded_rejected"
    else:
        alignment_quality = "not_applied"

    return {
        "prealign_requested": bool(prealign_requested),
        "prealign_applied": bool(prealign_applied),
        "alignment_quality": alignment_quality,
        "delta_t0_minutes": float(delta_t0_minutes),
        "delta_period_ppm": float(delta_period_ppm),
        "n_transits_pre": int(n_transits_pre),
        "n_transits_post": int(n_transits_post),
        "prealign_score_z": float(prealign_score_z) if prealign_score_z is not None else None,
        "prealign_error": prealign_error,
    }


def _build_next_actions(
    *,
    alignment_metadata: dict[str, Any],
    measurement_diagnostics: dict[str, Any],
    min_snr: float | None = None,
) -> list[dict[str, str]]:
    selected = measurement_diagnostics.get("selected") or {}
    reject_counts = selected.get("reject_counts") or {}

    attempted_epochs = int(selected.get("attempted_epochs") or 0)
    accepted_epochs = int(selected.get("accepted_epochs") or 0)
    snr_rejects = int(reject_counts.get("snr_below_threshold") or 0)
    rejected_epochs = max(attempted_epochs - accepted_epochs, 0)

    if accepted_epochs >= 2:
        primary = {
            "code": "TIMING_MEASURABLE",
            "summary": "Sufficient accepted transits for timing analysis.",
            "guidance": "Proceed with TTV interpretation and extend the baseline with new sectors when available.",
        }
    elif rejected_epochs > 0 and snr_rejects >= rejected_epochs:
        suggested_min_snr = (
            max(1.0, min(float(min_snr), 1.5))
            if min_snr is not None and np.isfinite(float(min_snr))
            else 1.0
        )
        primary = {
            "code": "NOISE_LIMITED",
            "summary": "Most rejected epochs are below the SNR threshold.",
            "guidance": (
                "Lower photometric noise or rerun with a lower acceptance threshold "
                f"(for example --min-snr {suggested_min_snr:.1f}) to inspect tentative per-epoch fits."
            ),
        }
    else:
        primary = {
            "code": "DATA_LIMITED",
            "summary": "Insufficient accepted transits for stable timing constraints.",
            "guidance": "Acquire additional cadence coverage or sectors and rerun timing diagnostics.",
        }

    next_actions: list[dict[str, str]] = [primary]
    alignment_quality = str(alignment_metadata.get("alignment_quality") or "")
    if alignment_quality == "improved_transit_recovery":
        next_actions.append(
            {
                "code": "PREALIGNMENT_HELPFUL",
                "summary": "Pre-alignment increased recovered transits.",
                "guidance": "Keep pre-alignment enabled for this target.",
            }
        )
    elif alignment_quality in {"degraded_rejected", "prealign_error_fallback"}:
        next_actions.append(
            {
                "code": "ALIGNMENT_REVIEW",
                "summary": "Pre-alignment did not improve usable transit fits.",
                "guidance": "Review ephemeris inputs and pre-alignment settings before rerunning.",
            }
        )

    return next_actions


def _prealign_candidate(
    *,
    lc: LightCurve,
    candidate: Candidate,
    min_snr: float,
    prealign_enabled: bool,
    prealign_steps: int,
    prealign_lr: float,
    prealign_window_phase: float,
) -> tuple[Candidate, list[Any], dict[str, Any], dict[str, Any]]:
    pre_transit_times, pre_diag = api.timing.measure_transit_times_with_diagnostics(
        lc=lc,
        candidate=candidate,
        min_snr=float(min_snr),
    )
    n_pre = len(pre_transit_times)

    if not prealign_enabled:
        metadata = _build_alignment_metadata(
            prealign_requested=False,
            prealign_applied=False,
            candidate_initial=candidate,
            candidate_final=candidate,
            n_transits_pre=n_pre,
            n_transits_post=n_pre,
            prealign_score_z=None,
            prealign_error=None,
        )
        return candidate, pre_transit_times, metadata, {
            "pre": pre_diag,
            "post": pre_diag,
            "selected": pre_diag,
        }

    try:
        internal_lc = lc.to_internal()
        valid = np.asarray(internal_lc.valid_mask, dtype=bool)
        time = np.asarray(internal_lc.time, dtype=np.float64)[valid]
        flux = np.asarray(internal_lc.flux, dtype=np.float64)[valid]
        flux_err = np.asarray(internal_lc.flux_err, dtype=np.float64)[valid]

        refined = ephemeris_refinement_api.refine_one_candidate_numpy(
            time=time,
            flux=flux,
            flux_err=flux_err,
            candidate=ephemeris_refinement_api.EphemerisRefinementCandidate(
                period_days=float(candidate.ephemeris.period_days),
                t0_btjd=float(candidate.ephemeris.t0_btjd),
                duration_hours=float(candidate.ephemeris.duration_hours),
            ),
            config=ephemeris_refinement_api.EphemerisRefinementConfig(
                steps=int(prealign_steps),
                lr=float(prealign_lr),
                t0_window_phase=float(prealign_window_phase),
            ),
        )

        refined_candidate = Candidate(
            ephemeris=Ephemeris(
                period_days=float(candidate.ephemeris.period_days),
                t0_btjd=float(refined.t0_refined_btjd),
                duration_hours=float(refined.duration_refined_hours),
            ),
            depth_ppm=candidate.depth_ppm,
            depth_fraction=candidate.depth_fraction,
        )
        post_transit_times, post_diag = api.timing.measure_transit_times_with_diagnostics(
            lc=lc,
            candidate=refined_candidate,
            min_snr=float(min_snr),
        )
        n_post = len(post_transit_times)

        if n_post >= n_pre:
            metadata = _build_alignment_metadata(
                prealign_requested=True,
                prealign_applied=True,
                candidate_initial=candidate,
                candidate_final=refined_candidate,
                n_transits_pre=n_pre,
                n_transits_post=n_post,
                prealign_score_z=float(refined.score_z),
                prealign_error=None,
            )
            return refined_candidate, post_transit_times, metadata, {
                "pre": pre_diag,
                "post": post_diag,
                "selected": post_diag,
            }

        metadata = _build_alignment_metadata(
            prealign_requested=True,
            prealign_applied=True,
            candidate_initial=candidate,
            candidate_final=refined_candidate,
            n_transits_pre=n_pre,
            n_transits_post=n_post,
            prealign_score_z=float(refined.score_z),
            prealign_error=None,
        )
        return candidate, pre_transit_times, metadata, {
            "pre": pre_diag,
            "post": post_diag,
            "selected": pre_diag,
        }
    except Exception as exc:
        metadata = _build_alignment_metadata(
            prealign_requested=True,
            prealign_applied=False,
            candidate_initial=candidate,
            candidate_final=candidate,
            n_transits_pre=n_pre,
            n_transits_post=n_pre,
            prealign_score_z=None,
            prealign_error=f"{type(exc).__name__}: {exc}",
        )
        return candidate, pre_transit_times, metadata, {
            "pre": pre_diag,
            "post": None,
            "selected": pre_diag,
        }


@click.command("timing")
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
@click.option("--min-snr", type=float, default=2.0, show_default=True)
@click.option(
    "--prealign/--no-prealign",
    default=True,
    show_default=True,
    help="Apply bounded T0 pre-alignment before strict per-transit fitting.",
)
@click.option(
    "--prealign-steps",
    type=int,
    default=25,
    show_default=True,
    help="Optimizer steps for ephemeris pre-alignment.",
)
@click.option(
    "--prealign-lr",
    type=float,
    default=0.05,
    show_default=True,
    help="Learning rate for ephemeris pre-alignment optimizer.",
)
@click.option(
    "--prealign-window-phase",
    type=float,
    default=0.02,
    show_default=True,
    help="Max fractional phase shift allowed during T0 pre-alignment.",
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
def timing_command(
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
    min_snr: float,
    prealign: bool,
    prealign_steps: int,
    prealign_lr: float,
    prealign_window_phase: float,
    output_path_arg: str,
) -> None:
    """Measure transit times and compute TTV diagnostics."""
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
        resolved_depth_ppm,
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

        (
            candidate_for_timing,
            transit_times,
            alignment_metadata,
            measurement_diagnostics,
        ) = _prealign_candidate(
            lc=lc,
            candidate=candidate,
            min_snr=float(min_snr),
            prealign_enabled=bool(prealign),
            prealign_steps=int(prealign_steps),
            prealign_lr=float(prealign_lr),
            prealign_window_phase=float(prealign_window_phase),
        )
        ttv = api.timing.analyze_ttvs(
            transit_times=transit_times,
            period_days=float(candidate_for_timing.ephemeris.period_days),
            t0_btjd=float(candidate_for_timing.ephemeris.t0_btjd),
        )
        series = api.timing.timing_series(
            lc=lc,
            candidate=candidate_for_timing,
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
        "prealign": bool(prealign),
        "prealign_steps": int(prealign_steps),
        "prealign_lr": float(prealign_lr),
        "prealign_window_phase": float(prealign_window_phase),
    }
    result_payload = {
        "transit_times": _transit_times_to_dicts(transit_times),
        "ttv": _to_jsonable_result(ttv),
        "timing_series": _to_jsonable_result(series),
        "alignment": alignment_metadata,
        "diagnostics": measurement_diagnostics,
        "next_actions": _build_next_actions(
            alignment_metadata=alignment_metadata,
            measurement_diagnostics=measurement_diagnostics,
            min_snr=float(min_snr),
        ),
    }
    verdict, verdict_source = _derive_timing_verdict(result_payload["next_actions"])
    result_payload["verdict"] = verdict
    result_payload["verdict_source"] = verdict_source
    payload = {
        "schema_version": "cli.timing.v1",
        "result": result_payload,
        "transit_times": result_payload["transit_times"],
        "ttv": result_payload["ttv"],
        "timing_series": result_payload["timing_series"],
        "alignment": result_payload["alignment"],
        "diagnostics": result_payload["diagnostics"],
        "next_actions": result_payload["next_actions"],
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


__all__ = ["timing_command"]
