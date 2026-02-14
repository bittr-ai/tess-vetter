"""Catalog-based vetting check wrappers (V06-V07).

This module provides VettingCheck-compliant wrappers around the existing
catalog check implementations. These wrappers:
- Implement the VettingCheck protocol
- Convert VetterCheckResult to the new CheckResult schema
- Preserve all existing functionality while normalizing the interface

Novelty: standard (wrapper pattern)
"""

from __future__ import annotations

from typing import Any

from bittr_tess_vetter.validation.registry import (
    CheckConfig,
    CheckInputs,
    CheckRequirements,
    CheckTier,
)
from bittr_tess_vetter.validation.result_schema import (
    CheckResult,
    error_result,
    ok_result,
    skipped_result,
)


_TIMEOUT_ERROR_TYPES = {
    "TIMEOUT",
    "TIMEOUTERROR",
    "READTIMEOUT",
    "CONNECTTIMEOUT",
}
_NETWORK_ERROR_TYPES = {
    "CONNECTIONERROR",
    "CONNECTION_ERROR",
    "REQUESTERROR",
    "REQUEST_ERROR",
    "REQUESTEXCEPTION",
}


def _is_timeout_like(error_type: str, error_text: str) -> bool:
    error_type_norm = error_type.strip().upper()
    if error_type_norm in _TIMEOUT_ERROR_TYPES:
        return True
    error_text_norm = error_text.lower()
    return any(
        token in error_text_norm
        for token in ("timeout", "timed out", "read timed out", "connect timeout")
    )


def _is_network_error_like(error_type: str, error_text: str) -> bool:
    error_type_norm = error_type.strip().upper()
    if error_type_norm in _NETWORK_ERROR_TYPES:
        return True
    error_text_norm = error_text.lower()
    return any(
        token in error_text_norm
        for token in (
            "connection error",
            "connection aborted",
            "name resolution",
            "dns",
            "failed to establish a new connection",
            "max retries exceeded",
            "request failed",
            "request exception",
        )
    )


def _convert_catalog_result(
    legacy: Any,
    check_id: str,
    check_name: str,
) -> CheckResult:
    """Convert a legacy VetterCheckResult to the new CheckResult schema.

    Args:
        legacy: VetterCheckResult from catalog check implementations.
        check_id: Check identifier.
        check_name: Human-readable check name.

    Returns:
        New-schema CheckResult.
    """
    details = dict(legacy.details) if legacy.details else {}

    # Check for error status
    status = details.get("status")
    if status == "error":
        error_type = str(details.get("error_type", ""))
        error_msg = details.get("error", "Unknown error")
        note = details.get("note")

        if check_id == "V06":
            error_text = " ".join(
                str(v) for v in (error_type, error_msg, note) if v not in (None, "")
            )
            if _is_timeout_like(error_type, error_text):
                return skipped_result(
                    check_id,
                    check_name,
                    reason_flag="NETWORK_TIMEOUT",
                    notes=[str(note)] if note is not None else [],
                    raw=details,
                )
            if _is_network_error_like(error_type, error_text):
                return skipped_result(
                    check_id,
                    check_name,
                    reason_flag="NETWORK_ERROR",
                    notes=[str(note)] if note is not None else [],
                    raw=details,
                )

        return error_result(
            check_id,
            check_name,
            error=error_msg,
            notes=[str(note)] if note is not None else [],
            raw=details,
        )

    # Extract metrics - only keep JSON-serializable scalar values
    metrics: dict[str, float | int | str | bool | None] = {}
    for key, value in details.items():
        # Skip internal/meta keys
        if key.startswith("_"):
            continue
        # Skip complex types for metrics
        if isinstance(value, (float, int, str, bool)) or value is None:
            metrics[key] = value

    # Determine flags based on content
    flags: list[str] = []
    notes: list[str] = []

    # For V06 (nearby EB search)
    if check_id == "V06":
        n_ebs = details.get("n_ebs_found", 0)
        if n_ebs > 0:
            flags.append("NEARBY_EBS_FOUND")
            notes.append(f"Found {n_ebs} nearby eclipsing binary(ies)")
        min_delta = details.get("min_period_ratio_delta_any")
        if min_delta is not None and min_delta < 0.05:
            flags.append("PERIOD_HARMONIC_MATCH")
            notes.append("EB period is harmonic of candidate period")

    # For V07 (ExoFOP TOI lookup)
    if check_id == "V07":
        found = details.get("found", False)
        if found:
            flags.append("EXOFOP_MATCH")
            notes.append("Candidate found in ExoFOP TOI table")
            row = details.get("row")
            if isinstance(row, dict):
                toi_val = row.get("toi")
                if toi_val not in (None, ""):
                    metrics["toi"] = str(toi_val)
        if details.get("cache_stale"):
            flags.append("STALE_CACHE")
            notes.append("Using cached data that may be outdated")

    return ok_result(
        check_id,
        check_name,
        metrics=metrics,
        confidence=legacy.confidence,
        flags=flags,
        notes=notes,
        raw=details,
    )


class NearbyEBSearchCheck:
    """V06: Nearby eclipsing binary search.

    Queries the TESS-EB catalog (Prsa et al. 2022) via VizieR to find
    nearby eclipsing binaries that could be the source of the transit signal.

    References:
        - Prsa et al. 2022, ApJS 258, 16 (TESS-EB catalog)
    """

    id = "V06"
    name = "Nearby EB Search"
    tier = CheckTier.CATALOG
    requirements = CheckRequirements(
        needs_network=True,
        needs_ra_dec=True,
    )
    citations = ["Prsa et al. 2022, ApJS 258, 16"]

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        """Execute the nearby EB search check."""
        # Validate inputs first
        if inputs.ra_deg is None or inputs.dec_deg is None:
            return skipped_result(
                self.id,
                self.name,
                reason_flag="NO_COORDINATES",
                notes=["RA/Dec coordinates not provided"],
            )

        if not inputs.network:
            return skipped_result(
                self.id,
                self.name,
                reason_flag="NETWORK_DISABLED",
                notes=["Network access disabled"],
            )

        try:
            from bittr_tess_vetter.validation.checks_catalog import run_nearby_eb_search

            # Extract candidate period if available
            candidate_period = None
            if inputs.candidate is not None:
                candidate_period = inputs.candidate.period
            timeout_s = config.extra_params.get("request_timeout_seconds", config.timeout_seconds)

            result = run_nearby_eb_search(
                ra_deg=inputs.ra_deg,
                dec_deg=inputs.dec_deg,
                candidate_period_days=candidate_period,
                request_timeout_s=timeout_s if timeout_s is not None else 10.0,
            )

            return _convert_catalog_result(result, self.id, self.name)
        except Exception as e:
            return error_result(
                self.id,
                self.name,
                error=type(e).__name__,
                notes=[str(e)],
            )


class ExoFOPTOILookupCheck:
    """V07: ExoFOP TOI table lookup.

    Queries the ExoFOP TOI table to check if this target is a known TOI
    and retrieve any disposition/metadata from the community vetting.

    References:
        - Guerrero et al. 2021, ApJS 254, 39 (TESS TOI catalog)
    """

    id = "V07"
    name = "ExoFOP TOI Lookup"
    tier = CheckTier.CATALOG
    requirements = CheckRequirements(
        needs_network=True,
        needs_tic_id=True,
    )
    citations = ["Guerrero et al. 2021, ApJS 254, 39"]

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        """Execute the ExoFOP TOI lookup check."""
        # Validate inputs first
        if inputs.tic_id is None:
            return skipped_result(
                self.id,
                self.name,
                reason_flag="NO_TIC_ID",
                notes=["TIC ID not provided"],
            )

        if not inputs.network:
            return skipped_result(
                self.id,
                self.name,
                reason_flag="NETWORK_DISABLED",
                notes=["Network access disabled"],
            )

        try:
            from bittr_tess_vetter.validation.checks_catalog import run_exofop_toi_lookup

            result = run_exofop_toi_lookup(
                tic_id=inputs.tic_id,
            )

            return _convert_catalog_result(result, self.id, self.name)
        except Exception as e:
            return error_result(
                self.id,
                self.name,
                error=type(e).__name__,
                notes=[str(e)],
            )


def register_catalog_checks(registry: Any) -> None:
    """Register catalog checks V06-V07 with a registry.

    Args:
        registry: A CheckRegistry instance.
    """
    registry.register(NearbyEBSearchCheck())
    registry.register(ExoFOPTOILookupCheck())


# All check classes for explicit imports
__all__ = [
    "NearbyEBSearchCheck",
    "ExoFOPTOILookupCheck",
    "register_catalog_checks",
]
