"""Structured result types for vetting checks.

This module defines the canonical result schemas for all vetting checks.
All checks must return CheckResult instances created via the helper functions.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CheckStatus = Literal["ok", "skipped", "error"]


class CheckResult(BaseModel):
    """Structured result from a single vetting check.

    Attributes:
        id: Check identifier (e.g., "V01", "V02").
        name: Human-readable check name.
        status: One of "ok", "skipped", or "error".
        confidence: Optional confidence score (0-1) when status is "ok".
        metrics: Machine-readable metrics (JSON-serializable scalars only).
        flags: Machine-readable flag identifiers (e.g., "ODD_EVEN_MISMATCH").
        notes: Human-readable notes (may change between versions).
        provenance: Minimal provenance info (versions, parameters used).
        raw: Optional unstructured data for backwards compatibility.
    """

    id: str
    name: str
    status: CheckStatus
    confidence: float | None = None
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    provenance: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    raw: dict[str, Any] | None = None

    model_config = {"extra": "forbid"}

    # Backward-compatible properties for legacy api/types.CheckResult interface
    @property
    def passed(self) -> bool | None:
        """Backward-compatible property: True if status='ok', False if 'error', None if 'skipped'."""
        if self.status == "ok":
            return True
        if self.status == "error":
            return False
        return None  # skipped

    @property
    def details(self) -> dict[str, Any]:
        """Backward-compatible property: combined metrics/flags/notes/raw as details dict."""
        result: dict[str, Any] = dict(self.metrics)
        result["status"] = self.status
        if self.flags:
            result["flags"] = self.flags
        if self.notes:
            result["notes"] = self.notes
        if self.raw:
            result.update(self.raw)
        # Mark as metrics-only for legacy code that checks this
        result["_metrics_only"] = True
        return result


class VettingBundleResult(BaseModel):
    """Aggregated result from running multiple vetting checks.

    Attributes:
        results: List of individual check results.
        warnings: Human-readable warning messages.
        provenance: Pipeline-level provenance (versions, timing, config).
        inputs_summary: Summary of what inputs were provided.
    """

    results: list[CheckResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    inputs_summary: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    @property
    def n_passed(self) -> int:
        """Count of checks with status 'ok'."""
        return sum(1 for r in self.results if r.status == "ok")

    @property
    def n_failed(self) -> int:
        """Count of checks with status 'error'."""
        return sum(1 for r in self.results if r.status == "error")

    @property
    def n_unknown(self) -> int:
        """Count of checks with status 'skipped'."""
        return sum(1 for r in self.results if r.status == "skipped")

    @property
    def all_passed(self) -> bool:
        """True if all checks have status 'ok'."""
        return all(r.status == "ok" for r in self.results)

    @property
    def failed_check_ids(self) -> list[str]:
        """List of IDs for checks with status 'error'."""
        return [r.id for r in self.results if r.status == "error"]

    @property
    def unknown_check_ids(self) -> list[str]:
        """List of IDs for checks with status 'skipped'."""
        return [r.id for r in self.results if r.status == "skipped"]

    def get_result(self, check_id: str) -> CheckResult | None:
        """Get result for a specific check by ID.

        Args:
            check_id: Check identifier (e.g., "V01").

        Returns:
            CheckResult if found, None otherwise.
        """
        for r in self.results:
            if r.id == check_id:
                return r
        return None


def ok_result(
    id: str,
    name: str,
    *,
    metrics: dict[str, float | int | str | bool | None],
    confidence: float | None = None,
    flags: list[str] | None = None,
    notes: list[str] | None = None,
    provenance: dict[str, float | int | str | bool | None] | None = None,
    raw: dict[str, Any] | None = None,
) -> CheckResult:
    """Create a successful check result.

    Args:
        id: Check identifier (e.g., "V01").
        name: Human-readable check name.
        metrics: Required metrics dict with JSON-serializable values.
        confidence: Optional confidence score (0-1).
        flags: Optional list of flag identifiers.
        notes: Optional human-readable notes.
        provenance: Optional provenance information.
        raw: Optional unstructured data.

    Returns:
        CheckResult with status="ok".
    """
    return CheckResult(
        id=id,
        name=name,
        status="ok",
        confidence=confidence,
        metrics=metrics,
        flags=flags or [],
        notes=notes or [],
        provenance=provenance or {},
        raw=raw,
    )


def skipped_result(
    id: str,
    name: str,
    *,
    reason_flag: str,
    notes: list[str] | None = None,
    provenance: dict[str, float | int | str | bool | None] | None = None,
    raw: dict[str, Any] | None = None,
) -> CheckResult:
    """Create a skipped check result.

    Args:
        id: Check identifier (e.g., "V01").
        name: Human-readable check name.
        reason_flag: Machine-readable reason for skipping (e.g., "NO_TPF", "NETWORK_DISABLED").
        notes: Optional human-readable notes explaining the skip.
        provenance: Optional provenance information.
        raw: Optional unstructured data.

    Returns:
        CheckResult with status="skipped" and reason in flags.
    """
    return CheckResult(
        id=id,
        name=name,
        status="skipped",
        confidence=None,
        metrics={},
        flags=[f"SKIPPED:{reason_flag}"],
        notes=notes or [],
        provenance=provenance or {},
        raw=raw,
    )


def error_result(
    id: str,
    name: str,
    *,
    error: str,
    flags: list[str] | None = None,
    notes: list[str] | None = None,
    provenance: dict[str, float | int | str | bool | None] | None = None,
    raw: dict[str, Any] | None = None,
) -> CheckResult:
    """Create an error check result.

    Args:
        id: Check identifier (e.g., "V01").
        name: Human-readable check name.
        error: Error message or type.
        flags: Optional additional flags.
        notes: Optional human-readable notes.
        provenance: Optional provenance information.
        raw: Optional unstructured data.

    Returns:
        CheckResult with status="error" and error in flags.
    """
    all_flags = [f"ERROR:{error}"]
    if flags:
        all_flags.extend(flags)
    return CheckResult(
        id=id,
        name=name,
        status="error",
        confidence=None,
        metrics={},
        flags=all_flags,
        notes=notes or [],
        provenance=provenance or {},
        raw=raw,
    )
