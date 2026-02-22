"""Trace/evidence parity bridge for Code Mode runtime outputs.

This module intentionally mirrors the evidence extraction semantics in
``tess_vetter.pipeline_composition.executor`` while keeping implementation local
for Code Mode runtime integrations.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_DETREND_INVARIANCE_POLICY_VERSION = "v1"
_DETREND_INVARIANCE_FPP_DELTA_ABS_THRESHOLD = 0.01

EVIDENCE_SCHEMA_VERSION = "pipeline.evidence_table.v5"
RUNTIME_TRACE_METADATA_SCHEMA_VERSION = "code_mode.trace.meta.v1"

EVIDENCE_FIELDNAMES: tuple[str, ...] = (
    "toi",
    "model_compete_verdict",
    "model_compete_raw_verdict",
    "model_compete_detrended_verdict",
    "systematics_verdict",
    "ephemeris_verdict",
    "timing_verdict",
    "localize_host_action_hint",
    "localize_host_reliability_status",
    "localize_host_reliability_action_hint",
    "dilution_n_plausible_scenarios",
    "dilution_reliability_status",
    "dilution_reliability_action_hint",
    "multiplicity_risk_status",
    "multiplicity_risk_reasons",
    "fpp",
    "fpp_raw",
    "fpp_detrended",
    "fpp_delta_detrended_minus_raw",
    "robustness_recommended_variant",
    "detrend_invariance_policy_version",
    "detrend_invariance_policy_verdict",
    "detrend_invariance_policy_reason_code",
    "detrend_invariance_policy_fpp_delta_abs_threshold",
    "detrend_invariance_policy_observed_fpp_delta_abs",
    "detrend_invariance_policy_observed_model_verdict_changed",
    "contrast_curve_availability",
    "contrast_curve_n_observations",
    "contrast_curve_filter",
    "contrast_curve_quality",
    "contrast_curve_depth0p5",
    "contrast_curve_depth1p0",
    "contrast_curve_selected_id",
    "contrast_curve_selected_source",
    "contrast_curve_selected_filter",
    "contrast_curve_selected_quality",
    "contrast_curve_selected_depth0p5",
    "contrast_curve_selected_depth1p0",
    "contrast_curve_selected_metadata",
    "known_planet_status",
    "known_planet_name",
    "known_planet_period",
    "stellar_contamination_risk_scalar",
    "concern_flags",
)

REQUIRED_EVIDENCE_KEYS = frozenset(EVIDENCE_FIELDNAMES)

RUNTIME_TRACE_METADATA_KEYS: tuple[str, ...] = (
    "schema_version",
    "trace_id",
    "policy_profile",
    "network_ok",
    "catalog_version_hash",
    "timestamp",
    "detrend_invariance",
    "evidence",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_concern_flags(step_payload: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    summary = step_payload.get("summary")
    if isinstance(summary, dict):
        concerns = summary.get("concerns")
        if isinstance(concerns, list):
            flags.extend(str(x) for x in concerns if x is not None)
    result = step_payload.get("result")
    if isinstance(result, dict):
        consensus = result.get("consensus")
        if isinstance(consensus, dict):
            rflags = consensus.get("reliability_flags")
            if isinstance(rflags, list):
                flags.extend(str(x) for x in rflags if x is not None)
        warnings = result.get("warnings")
        if isinstance(warnings, list):
            flags.extend(str(x) for x in warnings if x is not None)
    warnings_top = step_payload.get("warnings")
    if isinstance(warnings_top, list):
        flags.extend(str(x) for x in warnings_top if x is not None)
    return sorted(set(flags))


def _is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in {float("inf"), float("-inf")}


def _to_finite_float(value: Any) -> float | None:
    if not _is_finite_number(value):
        return None
    return float(value)


def _is_transit_like_verdict(verdict: Any) -> bool:
    if verdict is None:
        return False
    label = str(verdict).strip().upper()
    if label == "":
        return False
    return "TRANSIT" in label and "NON_TRANSIT" not in label


def _extract_verdict(payload: dict[str, Any] | None) -> Any:
    if not isinstance(payload, dict):
        return None
    verdict = payload.get("verdict")
    if verdict is not None:
        return verdict
    result = payload.get("result")
    if isinstance(result, dict):
        return result.get("verdict")
    return None


def _extract_reliability_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    summary = payload.get("reliability_summary")
    if isinstance(summary, dict):
        return summary
    result = payload.get("result")
    if isinstance(result, dict):
        nested = result.get("reliability_summary")
        if isinstance(nested, dict):
            return nested
    return None


def _extract_value_from_payload(payload: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(payload, dict):
        return None
    direct = payload.get(key)
    if direct is not None:
        return direct
    result = payload.get("result")
    if isinstance(result, dict):
        nested = result.get(key)
        if nested is not None:
            return nested
        summary = result.get("summary")
        if isinstance(summary, dict):
            nested_summary = summary.get(key)
            if nested_summary is not None:
                return nested_summary
    summary = payload.get("summary")
    if isinstance(summary, dict):
        nested = summary.get(key)
        if nested is not None:
            return nested
    return None


def _maybe_load_step(
    op: str,
    *,
    payloads_by_op: dict[str, dict[str, Any]],
    payloads_by_step_id: dict[str, dict[str, Any]],
    payloads_by_step_id_and_op: dict[tuple[str, str], dict[str, Any]],
    step_id: str | None = None,
    allow_op_fallback: bool = True,
) -> dict[str, Any] | None:
    if step_id is not None:
        payload = payloads_by_step_id_and_op.get((step_id, op))
        if payload is not None:
            return payload
        payload = payloads_by_step_id.get(step_id)
        if payload is not None and allow_op_fallback:
            return payload
        if not allow_op_fallback:
            return None
    return payloads_by_op.get(op)


def normalize_concern_flags(values: list[Any] | None) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({str(x) for x in values if x is not None})


def normalize_multiplicity_risk(payload: dict[str, Any] | None) -> tuple[Any, list[str] | None]:
    if not isinstance(payload, dict):
        return None, None
    status = payload.get("status")
    reasons = payload.get("reasons")
    if isinstance(reasons, list):
        return status, [str(x) for x in reasons if x is not None]
    return status, None


def to_semicolon_list(values: list[Any] | None, *, sort_values: bool, dedupe: bool) -> str:
    if not isinstance(values, list):
        return ""
    normalized = [str(x) for x in values if x is not None]
    if dedupe:
        normalized = list(dict.fromkeys(normalized))
    if sort_values:
        normalized = sorted(normalized)
    return ";".join(normalized)


def _load_step_payloads(
    toi_result: Mapping[str, Any],
    *,
    step_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    payloads_by_step_id: dict[str, dict[str, Any]] = {}
    payloads_by_op: dict[str, dict[str, Any]] = {}
    payloads_by_step_id_and_op: dict[tuple[str, str], dict[str, Any]] = {}

    for row_any in toi_result.get("steps", []):
        if not isinstance(row_any, dict):
            continue
        if row_any.get("status") != "ok":
            continue

        step_id = str(row_any.get("step_id") or "")
        op = str(row_any.get("op") or "")

        payload: dict[str, Any] | None = None
        if isinstance(row_any.get("payload"), dict):
            payload = dict(row_any.get("payload") or {})
        elif step_payloads is not None and step_id:
            candidate = step_payloads.get(step_id)
            if isinstance(candidate, Mapping):
                payload = dict(candidate)
        else:
            path = row_any.get("step_output_path")
            if path:
                p = Path(path)
                if p.exists():
                    loaded = _load_json(p)
                    if isinstance(loaded, dict):
                        payload = loaded

        if not isinstance(payload, dict):
            continue

        if step_id and step_id not in payloads_by_step_id:
            payloads_by_step_id[step_id] = payload
        if op and op not in payloads_by_op:
            payloads_by_op[op] = payload
        if step_id and op and (step_id, op) not in payloads_by_step_id_and_op:
            payloads_by_step_id_and_op[(step_id, op)] = payload

    return payloads_by_step_id, payloads_by_op, payloads_by_step_id_and_op


def build_evidence_compatible_row(
    toi_result: Mapping[str, Any],
    *,
    step_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one evidence-compatible row with executor-parity extraction rules."""
    toi = str(toi_result.get("toi"))
    payloads_by_step_id, payloads_by_op, payloads_by_step_id_and_op = _load_step_payloads(
        toi_result,
        step_payloads=step_payloads,
    )

    model_compete = _maybe_load_step(
        "model_compete",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    systematics = _maybe_load_step(
        "systematics_proxy",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    ephemeris = _maybe_load_step(
        "ephemeris_reliability",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    timing = _maybe_load_step(
        "timing",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    localize_host = _maybe_load_step(
        "localize_host",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    vet_payload = _maybe_load_step(
        "vet",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    report_payload = _maybe_load_step(
        "report",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    dilution = _maybe_load_step(
        "dilution",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    fpp = _maybe_load_step(
        "fpp_run",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    if fpp is None:
        fpp = _maybe_load_step(
            "fpp",
            payloads_by_op=payloads_by_op,
            payloads_by_step_id=payloads_by_step_id,
            payloads_by_step_id_and_op=payloads_by_step_id_and_op,
        )
    resolve_neighbors = _maybe_load_step(
        "resolve_neighbors",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    model_compete_raw = _maybe_load_step(
        "model_compete",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
        step_id="model_compete_raw",
    )
    model_compete_detrended = _maybe_load_step(
        "model_compete",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
        step_id="model_compete_detrended",
    )
    fpp_raw_payload = _maybe_load_step(
        "fpp_run",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
        step_id="fpp_raw",
        allow_op_fallback=False,
    )
    contrast_curve_summary = _maybe_load_step(
        "contrast_curve_summary",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    contrast_curves = _maybe_load_step(
        "contrast_curves",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
    )
    contrast_payload = contrast_curve_summary if contrast_curve_summary is not None else contrast_curves
    fpp_detrended_payload = _maybe_load_step(
        "fpp_run",
        payloads_by_op=payloads_by_op,
        payloads_by_step_id=payloads_by_step_id,
        payloads_by_step_id_and_op=payloads_by_step_id_and_op,
        step_id="fpp_detrended",
        allow_op_fallback=False,
    )

    localize_action_hint = None
    if isinstance(localize_host, dict):
        result = localize_host.get("result")
        if isinstance(result, dict):
            consensus = result.get("consensus")
            if isinstance(consensus, dict):
                localize_action_hint = consensus.get("action_hint")
    localize_reliability_summary = _extract_reliability_summary(localize_host)
    localize_reliability_status = (
        localize_reliability_summary.get("status") if isinstance(localize_reliability_summary, dict) else None
    )
    localize_reliability_action_hint = (
        localize_reliability_summary.get("action_hint") if isinstance(localize_reliability_summary, dict) else None
    )

    dilution_n_plausible = None
    if isinstance(dilution, dict):
        dilution_n_plausible = dilution.get("n_plausible_scenarios")
        if dilution_n_plausible is None:
            result = dilution.get("result")
            if isinstance(result, dict):
                dilution_n_plausible = result.get("n_plausible_scenarios")
    dilution_reliability_summary = _extract_reliability_summary(dilution)
    dilution_reliability_status = (
        dilution_reliability_summary.get("status") if isinstance(dilution_reliability_summary, dict) else None
    )
    dilution_reliability_action_hint = (
        dilution_reliability_summary.get("action_hint") if isinstance(dilution_reliability_summary, dict) else None
    )

    multiplicity_risk_payload: dict[str, Any] | None = None
    if isinstance(resolve_neighbors, dict):
        direct = resolve_neighbors.get("multiplicity_risk")
        if isinstance(direct, dict):
            multiplicity_risk_payload = direct
        else:
            prov = resolve_neighbors.get("provenance")
            if isinstance(prov, dict):
                nested = prov.get("multiplicity_risk")
                if isinstance(nested, dict):
                    multiplicity_risk_payload = nested
    multiplicity_risk_status, multiplicity_risk_reasons = normalize_multiplicity_risk(multiplicity_risk_payload)

    fpp_value = None
    if isinstance(fpp, dict):
        fpp_value = fpp.get("fpp")
        if fpp_value is None:
            result = fpp.get("result")
            if isinstance(result, dict):
                fpp_value = result.get("fpp")

    fpp_raw = None
    if isinstance(fpp_raw_payload, dict):
        fpp_raw = fpp_raw_payload.get("fpp")
        if fpp_raw is None:
            result = fpp_raw_payload.get("result")
            if isinstance(result, dict):
                fpp_raw = result.get("fpp")

    fpp_detrended = None
    if isinstance(fpp_detrended_payload, dict):
        fpp_detrended = fpp_detrended_payload.get("fpp")
        if fpp_detrended is None:
            result = fpp_detrended_payload.get("result")
            if isinstance(result, dict):
                fpp_detrended = result.get("fpp")

    fpp_raw_value = _to_finite_float(fpp_raw)
    fpp_detrended_value = _to_finite_float(fpp_detrended)
    fpp_delta_detrended_minus_raw: float | None = None
    if fpp_raw_value is not None and fpp_detrended_value is not None:
        fpp_delta_detrended_minus_raw = fpp_detrended_value - fpp_raw_value

    model_compete_raw_verdict = _extract_verdict(model_compete_raw)
    model_compete_detrended_verdict = _extract_verdict(model_compete_detrended)
    robustness_recommended_variant = None
    robustness_present = any(
        payload is not None
        for payload in (model_compete_raw, model_compete_detrended, fpp_raw_payload, fpp_detrended_payload)
    )
    if robustness_present:
        robustness_recommended_variant = "raw"
        if (
            fpp_raw_value is not None
            and fpp_detrended_value is not None
            and fpp_detrended_value < fpp_raw_value
            and _is_transit_like_verdict(model_compete_detrended_verdict)
        ):
            robustness_recommended_variant = "detrended"

    detrend_invariance_policy_verdict = "NOT_EVALUATED"
    detrend_invariance_policy_reason_code = "ROBUSTNESS_INPUTS_ABSENT"
    detrend_invariance_policy_observed_fpp_delta_abs = None
    detrend_invariance_policy_observed_model_verdict_changed = None
    if robustness_present:
        detrend_invariance_policy_verdict = "INSUFFICIENT_DATA"
        detrend_invariance_policy_reason_code = "MISSING_MODEL_VERDICTS_OR_FPP_VALUES"
        raw_verdict_norm = str(model_compete_raw_verdict).strip().upper() if model_compete_raw_verdict is not None else None
        detrended_verdict_norm = (
            str(model_compete_detrended_verdict).strip().upper() if model_compete_detrended_verdict is not None else None
        )
        if fpp_delta_detrended_minus_raw is not None:
            detrend_invariance_policy_observed_fpp_delta_abs = abs(fpp_delta_detrended_minus_raw)
        if raw_verdict_norm is not None and detrended_verdict_norm is not None:
            detrend_invariance_policy_observed_model_verdict_changed = raw_verdict_norm != detrended_verdict_norm
        if (
            detrend_invariance_policy_observed_model_verdict_changed is not None
            and detrend_invariance_policy_observed_fpp_delta_abs is not None
        ):
            if detrend_invariance_policy_observed_model_verdict_changed:
                detrend_invariance_policy_verdict = "NON_INVARIANT"
                detrend_invariance_policy_reason_code = "MODEL_VERDICT_CHANGED"
            elif detrend_invariance_policy_observed_fpp_delta_abs > _DETREND_INVARIANCE_FPP_DELTA_ABS_THRESHOLD:
                detrend_invariance_policy_verdict = "NON_INVARIANT"
                detrend_invariance_policy_reason_code = "FPP_DELTA_ABOVE_THRESHOLD"
            else:
                detrend_invariance_policy_verdict = "INVARIANT"
                detrend_invariance_policy_reason_code = "PASS"

    contrast_curve_availability = _extract_value_from_payload(contrast_payload, "availability")
    contrast_curve_n_observations = _extract_value_from_payload(contrast_payload, "n_observations")
    contrast_curve_filter = _extract_value_from_payload(contrast_payload, "filter")
    contrast_curve_quality = _extract_value_from_payload(contrast_payload, "quality")
    contrast_curve_depth0p5 = _extract_value_from_payload(contrast_payload, "depth0p5")
    contrast_curve_depth1p0 = _extract_value_from_payload(contrast_payload, "depth1p0")

    selected_curve = _extract_value_from_payload(contrast_payload, "selected_curve")
    if not isinstance(selected_curve, dict):
        selected_curve = {}
    contrast_curve_selected_id = selected_curve.get("id")
    contrast_curve_selected_source = selected_curve.get("source")
    contrast_curve_selected_filter = selected_curve.get("filter")
    contrast_curve_selected_quality = selected_curve.get("quality")
    contrast_curve_selected_depth0p5 = selected_curve.get("depth0p5")
    contrast_curve_selected_depth1p0 = selected_curve.get("depth1p0")
    contrast_curve_selected_metadata = selected_curve if selected_curve else None

    known_planet_status = _extract_value_from_payload(vet_payload, "known_planet_match_status")
    known_planet_payload = _extract_value_from_payload(vet_payload, "known_planet_match")
    if not isinstance(known_planet_payload, dict):
        known_planet_payload = {}
    matched_planet = known_planet_payload.get("matched_planet")
    if not isinstance(matched_planet, dict):
        matched_planet = {}
    known_planet_name = matched_planet.get("name")
    known_planet_period = matched_planet.get("period")

    stellar_contamination_risk_scalar = _extract_value_from_payload(report_payload, "stellar_contamination_risk_scalar")
    if stellar_contamination_risk_scalar is None:
        stellar_contamination_risk_scalar = _extract_value_from_payload(vet_payload, "stellar_contamination_risk_scalar")
    if stellar_contamination_risk_scalar is None:
        contamination_summary = _extract_value_from_payload(report_payload, "stellar_contamination_summary")
        if not isinstance(contamination_summary, dict):
            contamination_summary = _extract_value_from_payload(vet_payload, "stellar_contamination_summary")
        if isinstance(contamination_summary, dict):
            stellar_contamination_risk_scalar = contamination_summary.get("risk_scalar")

    concern_flags = {str(x) for x in (toi_result.get("concern_flags") or []) if x is not None}
    for payload in payloads_by_step_id.values():
        concern_flags.update(_extract_concern_flags(payload))
    for payload in payloads_by_op.values():
        concern_flags.update(_extract_concern_flags(payload))

    row: dict[str, Any] = {
        "toi": toi,
        "model_compete_verdict": _extract_verdict(model_compete),
        "model_compete_raw_verdict": model_compete_raw_verdict,
        "model_compete_detrended_verdict": model_compete_detrended_verdict,
        "systematics_verdict": _extract_verdict(systematics),
        "ephemeris_verdict": _extract_verdict(ephemeris),
        "timing_verdict": _extract_verdict(timing),
        "localize_host_action_hint": localize_action_hint,
        "localize_host_reliability_status": localize_reliability_status,
        "localize_host_reliability_action_hint": localize_reliability_action_hint,
        "dilution_n_plausible_scenarios": dilution_n_plausible,
        "dilution_reliability_status": dilution_reliability_status,
        "dilution_reliability_action_hint": dilution_reliability_action_hint,
        "multiplicity_risk_status": multiplicity_risk_status,
        "multiplicity_risk_reasons": multiplicity_risk_reasons,
        "fpp": fpp_value,
        "fpp_raw": fpp_raw,
        "fpp_detrended": fpp_detrended,
        "fpp_delta_detrended_minus_raw": fpp_delta_detrended_minus_raw,
        "robustness_recommended_variant": robustness_recommended_variant,
        "detrend_invariance_policy_version": _DETREND_INVARIANCE_POLICY_VERSION,
        "detrend_invariance_policy_verdict": detrend_invariance_policy_verdict,
        "detrend_invariance_policy_reason_code": detrend_invariance_policy_reason_code,
        "detrend_invariance_policy_fpp_delta_abs_threshold": _DETREND_INVARIANCE_FPP_DELTA_ABS_THRESHOLD,
        "detrend_invariance_policy_observed_fpp_delta_abs": detrend_invariance_policy_observed_fpp_delta_abs,
        "detrend_invariance_policy_observed_model_verdict_changed": detrend_invariance_policy_observed_model_verdict_changed,
        "contrast_curve_availability": contrast_curve_availability,
        "contrast_curve_n_observations": contrast_curve_n_observations,
        "contrast_curve_filter": contrast_curve_filter,
        "contrast_curve_quality": contrast_curve_quality,
        "contrast_curve_depth0p5": contrast_curve_depth0p5,
        "contrast_curve_depth1p0": contrast_curve_depth1p0,
        "contrast_curve_selected_id": contrast_curve_selected_id,
        "contrast_curve_selected_source": contrast_curve_selected_source,
        "contrast_curve_selected_filter": contrast_curve_selected_filter,
        "contrast_curve_selected_quality": contrast_curve_selected_quality,
        "contrast_curve_selected_depth0p5": contrast_curve_selected_depth0p5,
        "contrast_curve_selected_depth1p0": contrast_curve_selected_depth1p0,
        "contrast_curve_selected_metadata": contrast_curve_selected_metadata,
        "known_planet_status": known_planet_status,
        "known_planet_name": known_planet_name,
        "known_planet_period": known_planet_period,
        "stellar_contamination_risk_scalar": stellar_contamination_risk_scalar,
        "concern_flags": sorted(concern_flags),
    }
    return with_required_evidence_keys(row)


def with_required_evidence_keys(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a row containing all required evidence keys with deterministic defaults."""
    normalized = {key: row.get(key) for key in EVIDENCE_FIELDNAMES}
    if normalized["concern_flags"] is None:
        normalized["concern_flags"] = []
    return normalized


def to_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize one evidence row using executor CSV normalization semantics."""
    normalized = with_required_evidence_keys(row)
    normalized["concern_flags"] = to_semicolon_list(
        normalized.get("concern_flags"),
        sort_values=True,
        dedupe=False,
    )
    normalized["multiplicity_risk_reasons"] = to_semicolon_list(
        normalized.get("multiplicity_risk_reasons"),
        sort_values=False,
        dedupe=False,
    )
    return normalized


def build_runtime_trace_metadata(
    *,
    trace_id: str,
    policy_profile: str,
    network_ok: bool,
    catalog_version_hash: str | None = None,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """Build deterministic runtime trace metadata for downstream emitters.

    The function intentionally avoids implicit wall-clock calls. Pass ``timestamp``
    explicitly only when the caller requires wall-clock provenance.
    """
    return {
        "schema_version": RUNTIME_TRACE_METADATA_SCHEMA_VERSION,
        "trace_id": str(trace_id),
        "policy_profile": str(policy_profile),
        "network_ok": bool(network_ok),
        "catalog_version_hash": catalog_version_hash,
        "timestamp": timestamp,
        "detrend_invariance": {
            "policy_version": _DETREND_INVARIANCE_POLICY_VERSION,
            "fpp_delta_abs_threshold": _DETREND_INVARIANCE_FPP_DELTA_ABS_THRESHOLD,
        },
        "evidence": {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "fieldnames": list(EVIDENCE_FIELDNAMES),
        },
    }


__all__ = [
    "EVIDENCE_FIELDNAMES",
    "EVIDENCE_SCHEMA_VERSION",
    "REQUIRED_EVIDENCE_KEYS",
    "RUNTIME_TRACE_METADATA_KEYS",
    "RUNTIME_TRACE_METADATA_SCHEMA_VERSION",
    "build_evidence_compatible_row",
    "build_runtime_trace_metadata",
    "normalize_concern_flags",
    "normalize_multiplicity_risk",
    "to_csv_row",
    "to_semicolon_list",
    "with_required_evidence_keys",
]
