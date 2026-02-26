"""Deterministic compute-guidance helpers for FPP CLI commands."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from tess_vetter.api.fpp import get_fpp_preset_metadata

_ALLOWED_PRESETS = frozenset({"fast", "standard", "tutorial"})


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _extract_prepare_guidance_recommendation(prepared: Mapping[str, Any]) -> dict[str, Any] | None:
    compute_insights = prepared.get("compute_insights")
    if not isinstance(compute_insights, Mapping):
        return None
    recommendation = compute_insights.get("recommendation")
    if not isinstance(recommendation, Mapping):
        return None
    return dict(recommendation)


def build_prepare_compute_guidance(
    *,
    sectors_loaded: list[int],
    runtime_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Build non-binding guidance from observed cache/runtime state."""
    preset_metadata = get_fpp_preset_metadata()
    runtime_ready = bool(runtime_artifacts.get("target_cached")) and bool(
        runtime_artifacts.get("trilegal_cached")
    )
    recommended_preset = "standard" if runtime_ready else "fast"
    recommendation_defaults = preset_metadata[recommended_preset]["guidance_defaults"]

    return serialize_compute_guidance_payload(
        {
            "model_version": "fpp_compute_guidance.v2",
            "non_binding": True,
            "reason_codes": ["preset_tradeoff_known", "cache_state_observed"],
            "recommendation": {
                "preset": recommended_preset,
                "replicates": recommendation_defaults.get("replicates"),
                "timeout_seconds": recommendation_defaults.get("timeout_seconds"),
            },
            "evidence": {
                "sectors_loaded": int(len(sectors_loaded)),
                "runtime_artifacts_ready": runtime_ready,
                "staged_with_network": bool(runtime_artifacts.get("staged_with_network", False)),
            },
            "notes": [
                "Guidance is advisory; explicit CLI overrides win.",
                preset_metadata[recommended_preset]["intent"],
            ],
        }
    )


def apply_compute_guidance_to_runtime_knobs(
    *,
    prepared: Mapping[str, Any],
    apply_compute_guidance: bool,
    preset: str,
    replicates: int | None,
    timeout_seconds: float | None,
    preset_was_explicit: bool,
    replicates_was_explicit: bool,
    timeout_was_explicit: bool,
) -> tuple[str, int | None, float | None, bool, bool, str | None]:
    """Apply prepare-manifest guidance to unset runtime knobs only."""
    effective_preset = str(preset).lower()
    effective_replicates = replicates
    effective_timeout_seconds = timeout_seconds
    recommendation = _extract_prepare_guidance_recommendation(prepared)

    guidance_available = recommendation is not None
    guidance_applied = False
    guidance_source: str | None = None

    if apply_compute_guidance and recommendation is not None:
        guidance_source = "prepare_manifest"

        recommended_preset = recommendation.get("preset")
        if (not preset_was_explicit) and isinstance(recommended_preset, str) and recommended_preset.strip():
            selected = str(recommended_preset).lower()
            if selected in _ALLOWED_PRESETS:
                effective_preset = selected
                guidance_applied = True

        recommended_replicates = _coerce_positive_int(recommendation.get("replicates"))
        if (not replicates_was_explicit) and effective_replicates is None and recommended_replicates is not None:
            effective_replicates = int(recommended_replicates)
            guidance_applied = True

        raw_timeout = recommendation.get("timeout_seconds")
        if (not timeout_was_explicit) and effective_timeout_seconds is None and raw_timeout is not None:
            try:
                parsed_timeout = float(raw_timeout)
            except (TypeError, ValueError):
                parsed_timeout = None
            if parsed_timeout is not None and np.isfinite(parsed_timeout) and parsed_timeout > 0:
                effective_timeout_seconds = parsed_timeout
                guidance_applied = True

    return (
        effective_preset,
        effective_replicates,
        effective_timeout_seconds,
        guidance_available,
        guidance_applied,
        guidance_source,
    )


def serialize_compute_guidance_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize compute guidance into a deterministic JSON-ready payload."""
    recommendation = payload.get("recommendation")
    if isinstance(recommendation, Mapping):
        recommendation_payload: dict[str, Any] = {}

        raw_preset = recommendation.get("preset")
        if isinstance(raw_preset, str) and raw_preset.strip().lower() in _ALLOWED_PRESETS:
            recommendation_payload["preset"] = raw_preset.strip().lower()

        parsed_replicates = _coerce_positive_int(recommendation.get("replicates"))
        recommendation_payload["replicates"] = parsed_replicates

        timeout_raw = recommendation.get("timeout_seconds")
        timeout_value: float | None = None
        if timeout_raw is not None:
            try:
                parsed_timeout = float(timeout_raw)
            except (TypeError, ValueError):
                parsed_timeout = None
            if parsed_timeout is not None and np.isfinite(parsed_timeout) and parsed_timeout > 0:
                timeout_value = float(parsed_timeout)
        recommendation_payload["timeout_seconds"] = timeout_value
    else:
        recommendation_payload = {"preset": None, "replicates": None, "timeout_seconds": None}

    evidence = payload.get("evidence")
    evidence_payload = {
        "sectors_loaded": 0,
        "runtime_artifacts_ready": False,
        "staged_with_network": False,
    }
    if isinstance(evidence, Mapping):
        sectors_loaded = _coerce_positive_int(evidence.get("sectors_loaded"))
        evidence_payload["sectors_loaded"] = int(sectors_loaded or 0)
        evidence_payload["runtime_artifacts_ready"] = bool(evidence.get("runtime_artifacts_ready", False))
        evidence_payload["staged_with_network"] = bool(evidence.get("staged_with_network", False))

    reason_codes_raw = payload.get("reason_codes")
    reason_codes = (
        [str(item) for item in reason_codes_raw if isinstance(item, str)]
        if isinstance(reason_codes_raw, list)
        else []
    )

    notes_raw = payload.get("notes")
    notes = [str(item) for item in notes_raw if isinstance(item, str)] if isinstance(notes_raw, list) else []

    model_version = payload.get("model_version")
    model_version_value = str(model_version) if isinstance(model_version, str) else "fpp_compute_guidance.v2"

    return {
        "model_version": model_version_value,
        "non_binding": True,
        "reason_codes": reason_codes,
        "recommendation": recommendation_payload,
        "evidence": evidence_payload,
        "notes": notes,
    }


__all__ = [
    "apply_compute_guidance_to_runtime_knobs",
    "build_prepare_compute_guidance",
    "serialize_compute_guidance_payload",
]
