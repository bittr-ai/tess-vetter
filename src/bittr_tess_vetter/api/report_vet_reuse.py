"""Helpers for report generation reuse from prior vet artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from bittr_tess_vetter.report._build_core import _compute_lc_summary, _validate_build_inputs
from bittr_tess_vetter.report._build_panels import (
    _build_alias_harmonic_summary_data,
    _build_full_lc_plot_data,
    _build_lc_robustness_data,
    _build_local_detrend_diagnostic_plot_data,
    _build_odd_even_phase_plot_data,
    _build_oot_context_plot_data,
    _build_per_transit_stack_plot_data,
    _build_phase_folded_plot_data,
    _build_secondary_scan_plot_data,
    _build_timing_series_plot_data,
)
from bittr_tess_vetter.report._build_utils import _to_internal_lightcurve
from bittr_tess_vetter.report._data import CheckExecutionState, ReportData
from bittr_tess_vetter.validation.report_bridge import run_lc_checks
from bittr_tess_vetter.validation.result_schema import CheckResult, VettingBundleResult

_REPORT_BASE_CHECK_ORDER: tuple[str, ...] = ("V01", "V02", "V04", "V05", "V13", "V15")
CLI_REPORT_V3_SCHEMA = "cli.report.v3"


@dataclass(frozen=True)
class VetArtifactReuseSummary:
    """Computed reuse details for report generation."""

    missing_fields: list[str]
    incremental_compute: list[str]
    reused: bool

    def to_json(self, *, provided: bool, path: str | None) -> dict[str, Any]:
        return {
            "provided": bool(provided),
            "path": path,
            "reused": bool(self.reused),
            "missing_fields": list(self.missing_fields),
            "incremental_compute": list(self.incremental_compute),
        }


def required_report_check_ids(*, include_v03: bool, has_stellar: bool) -> list[str]:
    check_ids = list(_REPORT_BASE_CHECK_ORDER)
    if include_v03 and has_stellar:
        check_ids.append("V03")
    return check_ids


def coerce_vetting_bundle(payload: VettingBundleResult | dict[str, Any]) -> VettingBundleResult:
    if isinstance(payload, VettingBundleResult):
        return payload
    # The CLI vet v2 output wraps VettingBundleResult with extra top-level fields
    # (verdict, summary, stellar, lc_summary, schema_version, etc.).  Strip these
    # known envelope keys so extra="forbid" on VettingBundleResult doesn't reject
    # valid vet output files.  Unknown keys are kept so truly malformed payloads
    # still fail validation.
    _CLI_VET_ENVELOPE_KEYS = {
        "schema_version", "verdict", "verdict_source", "summary",
        "result", "stellar", "lc_summary", "lc_summary_meta",
        "sector_gating", "sector_measurements",
    }
    filtered = {k: v for k, v in payload.items() if k not in _CLI_VET_ENVELOPE_KEYS}
    return VettingBundleResult.model_validate(filtered)


def validate_vet_artifact_candidate_match(
    *,
    vet_bundle: VettingBundleResult,
    tic_id: int,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
) -> None:
    """Fail fast when a vet artifact encodes a different candidate."""
    inputs_summary = vet_bundle.inputs_summary
    input_resolution = inputs_summary.get("input_resolution")
    if not isinstance(input_resolution, dict):
        return

    candidate_meta: dict[str, Any] | None = None
    resolved = input_resolution.get("resolved")
    if isinstance(resolved, dict):
        candidate_meta = resolved
    else:
        raw_inputs = input_resolution.get("inputs")
        if isinstance(raw_inputs, dict):
            candidate_meta = raw_inputs
    if candidate_meta is None:
        return

    mismatches: list[str] = []

    artifact_tic = candidate_meta.get("tic_id")
    if artifact_tic is not None and int(artifact_tic) != int(tic_id):
        mismatches.append(f"tic_id artifact={artifact_tic} requested={tic_id}")

    float_fields = (
        ("period_days", float(period_days)),
        ("t0_btjd", float(t0_btjd)),
        ("duration_hours", float(duration_hours)),
    )
    for field, expected in float_fields:
        value = candidate_meta.get(field)
        if value is None:
            continue
        artifact_value = float(value)
        if not math.isclose(artifact_value, expected, rel_tol=0.0, abs_tol=1e-9):
            mismatches.append(f"{field} artifact={artifact_value} requested={expected}")

    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(f"vet result candidate mismatch: {joined}")


def build_report_with_vet_artifact(
    *,
    lc: Any,
    candidate: Any,
    vet_bundle: VettingBundleResult,
    stellar: Any | None = None,
    tic_id: int | None = None,
    toi: str | None = None,
    include_v03: bool = False,
    bin_minutes: float = 30.0,
    check_config: dict[str, dict[str, Any]] | None = None,
    max_lc_points: int = 50_000,
    max_phase_points: int = 10_000,
    include_additional_plots: bool = True,
    max_transit_windows: int = 24,
    max_points_per_window: int = 300,
    max_timing_points: int = 200,
    include_lc_robustness: bool = True,
    max_lc_robustness_epochs: int = 128,
    custom_views: dict[str, Any] | None = None,
) -> tuple[ReportData, VetArtifactReuseSummary]:
    """Build a report while reusing checks from a vet artifact when possible."""
    ephemeris = candidate.ephemeris
    _validate_build_inputs(
        ephemeris,
        bin_minutes,
        max_lc_points,
        max_phase_points,
        max_transit_windows,
        max_points_per_window,
        max_timing_points,
        max_lc_robustness_epochs,
    )

    v03_disabled_reason: str | None = None
    if include_v03 and stellar is None:
        v03_disabled_reason = "stellar is required to enable V03"

    desired_ids = required_report_check_ids(
        include_v03=include_v03,
        has_stellar=stellar is not None,
    )
    by_id = {result.id: result for result in vet_bundle.results}
    missing_ids = [check_id for check_id in desired_ids if check_id not in by_id]

    incremental_results: dict[str, CheckResult] = {}
    if missing_ids:
        internal_lc = _to_internal_lightcurve(lc)
        computed = run_lc_checks(
            internal_lc,
            period_days=ephemeris.period_days,
            t0_btjd=ephemeris.t0_btjd,
            duration_hours=ephemeris.duration_hours,
            stellar=stellar,
            enabled=set(missing_ids),
            config=check_config,
        )
        incremental_results = {result.id: result for result in computed}

    ordered_results: list[CheckResult] = []
    for check_id in desired_ids:
        if check_id in by_id:
            ordered_results.append(by_id[check_id])
            continue
        if check_id in incremental_results:
            ordered_results.append(incremental_results[check_id])

    lc_summary = _compute_lc_summary(lc, ephemeris)
    full_lc = _build_full_lc_plot_data(lc, ephemeris, max_lc_points)
    phase_folded = _build_phase_folded_plot_data(
        lc, ephemeris, bin_minutes, candidate.depth_ppm, max_phase_points
    )

    per_transit_stack = None
    local_detrend = None
    oot_context = None
    timing_diag = None
    timing_summary_source = None
    alias_diag = None
    lc_robustness_data = None
    odd_even_phase = None
    secondary_scan = None
    if include_additional_plots:
        per_transit_stack = _build_per_transit_stack_plot_data(
            lc,
            ephemeris,
            max_windows=max_transit_windows,
            max_points_per_window=max_points_per_window,
        )
        odd_even_phase = _build_odd_even_phase_plot_data(
            lc,
            ephemeris,
            bin_minutes=bin_minutes,
            max_points=max_phase_points,
        )
        local_detrend = _build_local_detrend_diagnostic_plot_data(
            lc,
            ephemeris,
            max_windows=max_transit_windows,
            max_points_per_window=max_points_per_window,
        )
        oot_context = _build_oot_context_plot_data(
            lc,
            ephemeris,
            max_points=max_phase_points,
        )
        timing_diag, timing_summary_source = _build_timing_series_plot_data(
            lc,
            candidate,
            max_points=max_timing_points,
        )
        alias_diag = _build_alias_harmonic_summary_data(
            lc,
            candidate,
        )
        secondary_scan = _build_secondary_scan_plot_data(
            lc,
            ephemeris,
            bin_minutes=bin_minutes,
            max_points=max_phase_points,
        )

    check_map = {result.id: result for result in ordered_results}
    if include_lc_robustness:
        lc_robustness_data = _build_lc_robustness_data(
            lc,
            candidate,
            checks=check_map,
            max_epochs=max_lc_robustness_epochs,
        )

    bundle = VettingBundleResult(
        results=ordered_results,
        warnings=list(vet_bundle.warnings),
        provenance=dict(vet_bundle.provenance),
        inputs_summary=dict(vet_bundle.inputs_summary),
    )
    report = ReportData(
        tic_id=tic_id,
        toi=toi,
        candidate=candidate,
        stellar=stellar,
        lc_summary=lc_summary,
        checks=check_map,
        bundle=bundle,
        full_lc=full_lc,
        phase_folded=phase_folded,
        per_transit_stack=per_transit_stack,
        local_detrend=local_detrend,
        oot_context=oot_context,
        timing_series=timing_diag,
        timing_summary_series=timing_summary_source,
        alias_summary=alias_diag,
        lc_robustness=lc_robustness_data,
        odd_even_phase=odd_even_phase,
        secondary_scan=secondary_scan,
        check_execution=CheckExecutionState(
            v03_requested=include_v03,
            v03_enabled=include_v03 and stellar is not None,
            v03_disabled_reason=v03_disabled_reason,
        ),
        custom_views=custom_views,
        checks_run=[result.id for result in ordered_results],
    )
    reuse = VetArtifactReuseSummary(
        missing_fields=[f"checks.{check_id}" for check_id in missing_ids],
        incremental_compute=[f"compute_check.{check_id}" for check_id in missing_ids],
        reused=len(ordered_results) > 0,
    )
    return report, reuse


def build_cli_report_payload(
    *,
    report_json: dict[str, Any],
    vet_artifact: dict[str, Any],
    sectors_used: list[int] | None = None,
    resolved_inputs: dict[str, Any] | None = None,
    diagnostic_artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    report_payload = dict(report_json)
    summary_payload = report_payload.get("summary")
    if not isinstance(summary_payload, dict):
        summary_payload = {}
        report_payload["summary"] = summary_payload

    if diagnostic_artifacts is not None:
        summary_payload["diagnostic_artifacts"] = [dict(item) for item in diagnostic_artifacts]

    verdict = summary_payload.get("verdict")
    verdict_source = summary_payload.get("verdict_source")
    provenance_payload: dict[str, Any] = {"vet_artifact": dict(vet_artifact)}
    if sectors_used is not None:
        provenance_payload["sectors_used"] = [int(sector) for sector in sectors_used]
    if resolved_inputs is not None:
        provenance_payload["resolved_inputs"] = {
            "tic_id": int(resolved_inputs["tic_id"]) if resolved_inputs.get("tic_id") is not None else None,
            "period_days": (
                float(resolved_inputs["period_days"]) if resolved_inputs.get("period_days") is not None else None
            ),
            "t0_btjd": float(resolved_inputs["t0_btjd"]) if resolved_inputs.get("t0_btjd") is not None else None,
            "duration_hours": (
                float(resolved_inputs["duration_hours"])
                if resolved_inputs.get("duration_hours") is not None
                else None
            ),
            "depth_ppm": float(resolved_inputs["depth_ppm"]) if resolved_inputs.get("depth_ppm") is not None else None,
        }
    return {
        "schema_version": CLI_REPORT_V3_SCHEMA,
        "provenance": provenance_payload,
        "verdict": verdict,
        "verdict_source": verdict_source,
        "report": report_payload,
    }
