"""Legacy-to-PRD error code mapping helpers.

This adapter is additive-only: legacy error fields remain unchanged and callers
may attach mapped PRD codes as metadata (for example, ``provenance.error_code``).
"""

from __future__ import annotations

from typing import Any

DEFAULT_PRD_ERROR_CODE = "SCHEMA_VIOLATION_OUTPUT"

_LEGACY_ERROR_CLASS_TO_PRD_CODE: dict[str, str] = {
    # Policy enforcement/gating failures.
    "NoDownloadError": "POLICY_DENIED",
    "OfflineNoLocalDataError": "POLICY_DENIED",
    "POLICY_DENIED": "POLICY_DENIED",
    # Optional dependency/extras failures.
    "MissingOptionalDependencyError": "DEPENDENCY_MISSING",
    "DEPENDENCY_MISSING": "DEPENDENCY_MISSING",
    # Input schema failures where legacy class is explicit.
    "KeyError": "SCHEMA_VIOLATION_INPUT",
    "TypeError": "SCHEMA_VIOLATION_INPUT",
    "ValueError": "SCHEMA_VIOLATION_INPUT",
    # Retry/time-budget style runtime failures.
    "TimeoutError": "TIMEOUT_EXCEEDED",
    "NetworkTimeoutError": "TIMEOUT_EXCEEDED",
    "TRANSIENT_EXHAUSTION": "TRANSIENT_EXHAUSTION",
    # Legacy pipeline output/data fulfillment failures.
    "LocalDataNotFoundError": "SCHEMA_VIOLATION_OUTPUT",
    "LightCurveNotFoundError": "SCHEMA_VIOLATION_OUTPUT",
    "NoSectorsSelectedError": "SCHEMA_VIOLATION_OUTPUT",
    "NoUsablePointsError": "SCHEMA_VIOLATION_OUTPUT",
    "InsufficientTimeCoverageError": "SCHEMA_VIOLATION_OUTPUT",
    "SectorGatingError": "SCHEMA_VIOLATION_OUTPUT",
    "StitchError": "SCHEMA_VIOLATION_OUTPUT",
    "NoInTransitCadencesError": "SCHEMA_VIOLATION_OUTPUT",
    "TPFRequiredError": "SCHEMA_VIOLATION_OUTPUT",
    "VettingPipelineError": "SCHEMA_VIOLATION_OUTPUT",
    "SCHEMA_VIOLATION_OUTPUT": "SCHEMA_VIOLATION_OUTPUT",
}

_POLICY_REASON_FLAGS: frozenset[str] = frozenset({"NETWORK_DISABLED", "NETWORK_ERROR"})


def _normalized_reason_flag(reason_flag: str | None) -> str | None:
    if not isinstance(reason_flag, str):
        return None
    normalized_reason = reason_flag.strip().upper()
    return normalized_reason or None


def map_legacy_denial_details(
    error_class: str | None,
    *,
    reason_flag: str | None = None,
) -> dict[str, Any]:
    """Map legacy denial/error context into standardized blocker detail fields."""
    normalized_reason = _normalized_reason_flag(reason_flag)
    if normalized_reason and normalized_reason.startswith("EXTRA_MISSING:"):
        extra = normalized_reason.split(":", 1)[1].strip().lower()
        return {
            "reason_flag": normalized_reason,
            "dependency": extra,
            "dependency_blockers": [
                {
                    "type": "optional_dependency_missing",
                    "summary": f"Missing optional dependency extra '{extra}'.",
                    "action": f"Install optional dependency with: pip install 'tess-vetter[{extra}]'",
                    "dependency": extra,
                    "install_hint": f"pip install 'tess-vetter[{extra}]'",
                }
            ],
            "policy_blockers": [],
            "constructor_blockers": [],
        }

    if normalized_reason in _POLICY_REASON_FLAGS or error_class in {
        "NoDownloadError",
        "OfflineNoLocalDataError",
        "POLICY_DENIED",
    }:
        reason = normalized_reason or "POLICY_DENIED"
        return {
            "reason_flag": reason,
            "policy_blockers": [
                {
                    "type": "network_policy_denied",
                    "summary": "Network access is denied by policy.",
                    "action": "Re-run in a network-enabled policy profile or provide local artifacts.",
                    "reason": reason,
                }
            ],
            "dependency_blockers": [],
            "constructor_blockers": [],
        }

    return {}


def map_legacy_error_to_prd_code(
    error_class: str | None,
    *,
    reason_flag: str | None = None,
) -> str:
    """Map legacy error classification to a PRD error code.

    Args:
        error_class: Legacy exception/error class string.
        reason_flag: Optional legacy skip reason (for example ``EXTRA_MISSING:tls``).

    Returns:
        PRD taxonomy code.
    """
    normalized_reason = _normalized_reason_flag(reason_flag)
    if normalized_reason:
        if normalized_reason.startswith("EXTRA_MISSING:"):
            return "DEPENDENCY_MISSING"
        if normalized_reason in _POLICY_REASON_FLAGS:
            return "POLICY_DENIED"
        if normalized_reason == "NETWORK_TIMEOUT":
            return "TIMEOUT_EXCEEDED"
        if normalized_reason in {
            "NO_TPF",
            "NO_COORDINATES",
            "NO_TIC_ID",
            "INSUFFICIENT_DATA",
            "NO_APERTURE_MASK",
            "NO_SECTOR_MEASUREMENTS",
            "INSUFFICIENT_SECTORS",
        }:
            return "SCHEMA_VIOLATION_INPUT"

    if isinstance(error_class, str):
        mapped = _LEGACY_ERROR_CLASS_TO_PRD_CODE.get(error_class)
        if mapped:
            return mapped

    return DEFAULT_PRD_ERROR_CODE


__all__ = [
    "DEFAULT_PRD_ERROR_CODE",
    "map_legacy_denial_details",
    "map_legacy_error_to_prd_code",
]
