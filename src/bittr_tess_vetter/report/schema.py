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

from bittr_tess_vetter.report.custom_views_schema import CustomViewsModel


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
    rotation_context: dict[str, Any] | None = None
    semantics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


class AliasScalarSummaryModel(BaseModel):
    """Scalar alias diagnostics rollup for summary consumers."""

    model_config = ConfigDict(extra="forbid")

    best_harmonic: str | None = None
    best_ratio_over_p: float | None = None
    score_p: float | None = None
    score_p_over_2: float | None = None
    score_2p: float | None = None
    depth_ppm_peak: float | None = None
    classification: str | None = None
    phase_shift_event_count: int | None = None
    phase_shift_peak_sigma: float | None = None
    secondary_significance: float | None = None
    alias_interpretation: str | None = None


class TimingSummaryModel(BaseModel):
    """Scalar timing rollup derived from per-epoch timing series."""

    model_config = ConfigDict(extra="forbid")

    n_epochs_measured: int = 0
    rms_seconds: float | None = None
    periodicity_score: float | None = None
    linear_trend_sec_per_epoch: float | None = None
    max_abs_oc_seconds: float | None = None
    max_snr: float | None = None
    snr_median: float | None = None
    oc_median: float | None = None
    outlier_count: int = 0
    outlier_fraction: float | None = None
    deepest_epoch: int | None = None
    n_transits_measured: int | None = None
    depth_scatter_ppm: float | None = None
    chi2_reduced: float | None = None


class SecondaryScanSummaryModel(BaseModel):
    """Scalar secondary-scan coverage and strongest-dip rollup."""

    model_config = ConfigDict(extra="forbid")

    n_raw_points: int | None = None
    n_bins: int | None = None
    phase_coverage_fraction: float | None = None
    largest_phase_gap: float | None = None
    n_bins_with_error: int | None = None
    strongest_dip_phase: float | None = None
    strongest_dip_depth_ppm: float | None = None
    is_degraded: bool | None = None
    quality_flag_count: int = 0


class DataGapSummaryModel(BaseModel):
    """Scalar V13 data-gap diagnostics for in-coverage epochs."""

    model_config = ConfigDict(extra="forbid")

    missing_frac_max_in_coverage: float | None = None
    missing_frac_median_in_coverage: float | None = None
    n_epochs_missing_ge_0p25_in_coverage: int | None = None
    n_epochs_excluded_no_coverage: int | None = None
    n_epochs_evaluated_in_coverage: int | None = None


class BundleSummaryModel(BaseModel):
    """Aggregate check counts."""

    model_config = ConfigDict(extra="forbid")

    n_checks: int
    n_ok: int
    n_failed: int
    n_skipped: int
    failed_ids: list[str] = Field(default_factory=list)


class CheckExecutionSummaryModel(BaseModel):
    """Execution-state block for explicit check enablement decisions."""

    model_config = ConfigDict(extra="forbid")

    v03_requested: bool
    v03_enabled: bool
    v03_disabled_reason: str | None = None


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
    verdict: str | None = None
    verdict_source: str | None = None
    caveats: list[str] = Field(default_factory=list)
    check_execution: CheckExecutionSummaryModel | None = None
    bundle_summary: BundleSummaryModel | None = None
    odd_even_summary: OddEvenSummaryModel | None = None
    noise_summary: NoiseSummaryModel | None = None
    variability_summary: VariabilitySummaryModel | None = None
    references: list[ReferenceEntryModel] = Field(default_factory=list)
    enrichment: dict[str, Any] | None = None
    lc_robustness_summary: dict[str, Any] | None = None
    alias_scalar_summary: AliasScalarSummaryModel | None = None
    timing_summary: TimingSummaryModel | None = None
    secondary_scan_summary: SecondaryScanSummaryModel | None = None
    data_gap_summary: DataGapSummaryModel | None = None


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
    custom_views_version: str
    summary_hash: str
    plot_data_hash: str
    custom_views_hash: str
    custom_view_hashes_by_id: dict[str, str]
    custom_views_includes_ad_hoc: bool
    contract_version: str | None = None
    required_metrics_by_check: dict[str, list[str]] = Field(default_factory=dict)
    missing_required_metrics_by_check: dict[str, list[str]] = Field(default_factory=dict)
    metric_keys_by_check: dict[str, list[str]] = Field(default_factory=dict)
    has_missing_required_metrics: bool = False


class ReportPayloadModel(BaseModel):
    """Top-level typed report payload contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    summary: ReportSummaryModel
    plot_data: ReportPlotDataModel
    custom_views: CustomViewsModel
    payload_meta: ReportPayloadMetaModel


def report_payload_json_schema() -> dict[str, Any]:
    """Return machine-readable JSON schema for external clients."""
    return ReportPayloadModel.model_json_schema()
