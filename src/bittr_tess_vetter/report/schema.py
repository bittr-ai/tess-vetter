"""Typed JSON contract for report payloads.

This module defines the external report JSON shape for clients:
- schema_version
- summary
- plot_data
- payload_meta
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CheckSummaryModel(BaseModel):
    """Projected check summary (array-safe, client-friendly)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    status: str
    confidence: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    method_refs: list[str] = Field(default_factory=list)


class ReferenceEntryModel(BaseModel):
    """Typed bibliographic reference entry for summary payloads."""

    model_config = ConfigDict(extra="forbid")

    key: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    citation: str | None = None
    notes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class OddEvenSummaryModel(BaseModel):
    """Deterministic odd/even depth comparison summary."""

    model_config = ConfigDict(extra="forbid")

    odd_depth_ppm: float | None = None
    even_depth_ppm: float | None = None
    depth_diff_ppm: float | None = None
    depth_diff_sigma: float | None = None
    is_significant: bool | None = None
    flags: list[str] = Field(default_factory=list)


class NoiseSummaryModel(BaseModel):
    """Deterministic noise diagnostics with semantics-ready extensions."""

    model_config = ConfigDict(extra="forbid")

    white_noise_ppm: float | None = None
    red_noise_beta_30m: float | None = None
    red_noise_beta_60m: float | None = None
    red_noise_beta_duration: float | None = None
    trend_stat: float | None = None
    trend_stat_unit: str | None = None
    flags: list[str] = Field(default_factory=list)
    semantics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


class VariabilitySummaryModel(BaseModel):
    """Deterministic stellar/activity variability summary."""

    model_config = ConfigDict(extra="forbid")

    variability_index: float | None = None
    periodicity_score: float | None = None
    flare_rate_per_day: float | None = None
    classification: str | None = None
    flags: list[str] = Field(default_factory=list)
    semantics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


class BundleSummaryModel(BaseModel):
    """Aggregate check counts."""

    model_config = ConfigDict(extra="forbid")

    n_checks: int
    n_ok: int
    n_failed: int
    n_skipped: int
    failed_ids: list[str] = Field(default_factory=list)


class ReportSummaryModel(BaseModel):
    """Non-plot summary payload."""

    model_config = ConfigDict(extra="allow")

    tic_id: int | None = None
    toi: str | None = None
    checks_run: list[str] = Field(default_factory=list)
    ephemeris: dict[str, Any] | None = None
    input_depth_ppm: float | None = None
    stellar: dict[str, Any] | None = None
    lc_summary: dict[str, Any] | None = None
    checks: dict[str, CheckSummaryModel] = Field(default_factory=dict)
    bundle_summary: BundleSummaryModel | None = None
    odd_even_summary: OddEvenSummaryModel | None = None
    noise_summary: NoiseSummaryModel | None = None
    variability_summary: VariabilitySummaryModel | None = None
    references: list[ReferenceEntryModel] = Field(default_factory=list)
    enrichment: dict[str, Any] | None = None
    lc_robustness_summary: dict[str, Any] | None = None


class ReportPlotDataModel(BaseModel):
    """Plot payload domain (plot-ready arrays and hints)."""

    model_config = ConfigDict(extra="allow")

    check_overlays: dict[str, Any] | None = None
    full_lc: dict[str, Any] | None = None
    phase_folded: dict[str, Any] | None = None
    per_transit_stack: dict[str, Any] | None = None
    local_detrend: dict[str, Any] | None = None
    odd_even_phase: dict[str, Any] | None = None
    secondary_scan: dict[str, Any] | None = None
    oot_context: dict[str, Any] | None = None
    timing_series: dict[str, Any] | None = None
    alias_summary: dict[str, Any] | None = None
    lc_robustness: dict[str, Any] | None = None


class ReportPayloadMetaModel(BaseModel):
    """Deterministic payload metadata for caching and versioning."""

    model_config = ConfigDict(extra="forbid")

    summary_version: str
    plot_data_version: str
    summary_hash: str
    plot_data_hash: str


class ReportPayloadModel(BaseModel):
    """Top-level typed report payload contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    summary: ReportSummaryModel
    plot_data: ReportPlotDataModel
    payload_meta: ReportPayloadMetaModel


def report_payload_json_schema() -> dict[str, Any]:
    """Return machine-readable JSON schema for external clients."""
    return ReportPayloadModel.model_json_schema()
