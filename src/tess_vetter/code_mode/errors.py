"""Legacy-to-PRD error code mapping helpers.

This adapter is additive-only: legacy error fields remain unchanged and callers
may attach mapped PRD codes as metadata (for example, ``provenance.error_code``).
"""

from __future__ import annotations

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
    if isinstance(reason_flag, str):
        if reason_flag.startswith("EXTRA_MISSING:"):
            return "DEPENDENCY_MISSING"
        if reason_flag == "NETWORK_DISABLED":
            return "POLICY_DENIED"
        if reason_flag in {"NO_TPF", "NO_COORDINATES", "NO_TIC_ID"}:
            return "SCHEMA_VIOLATION_INPUT"

    if isinstance(error_class, str):
        mapped = _LEGACY_ERROR_CLASS_TO_PRD_CODE.get(error_class)
        if mapped:
            return mapped

    return DEFAULT_PRD_ERROR_CODE


__all__ = [
    "DEFAULT_PRD_ERROR_CODE",
    "map_legacy_error_to_prd_code",
]
