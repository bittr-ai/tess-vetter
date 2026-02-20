"""Report data structures for LC-only vetting reports.

Defines the structured output of build_report(). Consumers
(plotting module, frontend, AI agent) read from these objects.
The report module never renders -- it only assembles data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from tess_vetter.report._assembly import (
    ReportAssemblyContext,
    assemble_plot_data,
    assemble_summary,
)
from tess_vetter.report._custom_view_hash import custom_view_hashes_by_id, custom_views_hash
from tess_vetter.report._custom_view_validate import validate_custom_views_payload
from tess_vetter.report._serialization_utils import _canonical_sha256, _scrub_non_finite
from tess_vetter.report._summary_builders import _build_check_metric_contract_meta
from tess_vetter.report.schema import ReportPayloadModel
from tess_vetter.validation.result_schema import CheckResult, VettingBundleResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LCSummary:
    """Light curve vital signs for triage context."""

    n_points: int  # total data points
    n_valid: int  # after quality/NaN masking
    n_transits: int  # transits with >= 3 in-transit points
    n_in_transit_total: int  # total in-transit data points across all transits
    duration_days: float  # time baseline
    cadence_seconds: float  # median cadence
    flux_std_ppm: float  # OOT flux scatter in ppm
    flux_mad_ppm: float  # OOT MAD-based scatter in ppm (robust)
    gap_fraction: float  # fraction of invalid points (0-1)
    snr: float  # box-model transit SNR
    depth_ppm: float  # measured depth in ppm (from box model)
    depth_err_ppm: float | None  # depth uncertainty in ppm (None if unmeasurable)


@dataclass(frozen=True)
class FullLCPlotData:
    """Plot-ready arrays for the full light curve panel."""

    time: list[float]  # BTJD timestamps
    flux: list[float]  # normalized flux
    transit_mask: list[bool]  # True for in-transit points


@dataclass(frozen=True)
class PhaseFoldedPlotData:
    """Plot-ready arrays for the phase-folded transit panel.

    Raw points use a two-zone strategy: full-resolution near the transit
    (within ``phase_range``) and heavily downsampled out-of-transit baseline.
    Bins cover only the ``phase_range`` window — the transit-centric region
    that matters for triage.

    The ``phase_range`` window is determined by the transit duration:
    ±3 transit durations in phase units, giving enough baseline context
    around ingress/egress while avoiding the vast irrelevant orbital baseline.
    """

    phase: list[float]  # phase values (-0.5 to 0.5)
    flux: list[float]  # normalized flux at each phase
    bin_centers: list[float]  # binned phase centers (within phase_range)
    bin_flux: list[float]  # binned flux means
    bin_err: list[float | None]  # binned flux standard error of mean (None if single-point bin)
    bin_minutes: float  # bin width used
    transit_duration_phase: float  # transit duration expressed in phase units (duration_hours / period_days / 24)
    phase_range: tuple[float, float]  # display window in phase units, e.g. (-0.03, 0.03)
    y_range_suggested: tuple[float, float] | None = None  # display-only y-axis suggestion (robust percentile clip)
    depth_reference_flux: float | None = None  # display-only horizontal reference line in normalized flux


@dataclass(frozen=True)
class TransitWindowData:
    """One observed transit window for per-transit stack visualization."""

    epoch: int
    t_mid_btjd: float
    dt_hours: list[float]  # relative time from t_mid in hours
    flux: list[float]
    in_transit_mask: list[bool]


@dataclass(frozen=True)
class PerTransitStackPlotData:
    """Plot-ready data for per-transit small-multiples panel."""

    windows: list[TransitWindowData]
    window_half_hours: float
    max_windows: int


@dataclass(frozen=True)
class OddEvenPhasePlotData:
    """Plot-ready arrays for odd/even phase-fold comparison panel."""

    phase_range: tuple[float, float]
    odd_phase: list[float]
    odd_flux: list[float]
    even_phase: list[float]
    even_flux: list[float]
    odd_bin_centers: list[float]
    even_bin_centers: list[float]
    odd_bin_flux: list[float]
    even_bin_flux: list[float]
    bin_minutes: float


@dataclass(frozen=True)
class SecondaryScanPlotData:
    """Plot-ready arrays for full-orbit secondary-eclipse/phase scan."""

    phase: list[float]
    flux: list[float]
    bin_centers: list[float]
    bin_flux: list[float]
    bin_err: list[float | None]
    bin_minutes: float
    primary_phase: float
    secondary_phase: float
    strongest_dip_phase: float | None
    strongest_dip_flux: float | None
    quality: SecondaryScanQuality | None = None
    render_hints: SecondaryScanRenderHints | None = None


@dataclass(frozen=True)
class SecondaryScanQuality:
    """Deterministic quality metrics for robust batch rendering decisions."""

    n_raw_points: int
    n_bins: int
    n_bins_with_error: int
    phase_coverage_fraction: float  # occupied-bin fraction over full [-0.5, 0.5]
    largest_phase_gap: float
    is_degraded: bool
    flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SecondaryScanRenderHints:
    """Display policy hints derived from SecondaryScanQuality."""

    style_mode: str  # "normal" | "degraded"
    connect_bins: bool
    max_connect_phase_gap: float
    show_error_bars: bool
    error_bar_stride: int
    raw_marker_opacity: float
    binned_marker_size: float
    binned_line_width: float


@dataclass(frozen=True)
class LocalDetrendWindowData:
    """One transit-centered window with local baseline fit."""

    epoch: int
    t_mid_btjd: float
    dt_hours: list[float]
    flux: list[float]
    baseline_flux: list[float]
    in_transit_mask: list[bool]


@dataclass(frozen=True)
class LocalDetrendDiagnosticPlotData:
    """Plot-ready data for local baseline diagnostics around transits."""

    windows: list[LocalDetrendWindowData]
    window_half_hours: float
    max_windows: int
    baseline_method: str


@dataclass(frozen=True)
class OOTContextPlotData:
    """Out-of-transit flux distribution + scatter context payload."""

    flux_sample: list[float]
    flux_residual_ppm_sample: list[float]
    sample_indices: list[int]
    hist_centers: list[float]
    hist_counts: list[int]
    median_flux: float | None
    std_ppm: float | None
    mad_ppm: float | None
    robust_sigma_ppm: float | None
    n_oot_points: int


@dataclass(frozen=True)
class TransitTimingPlotData:
    """Per-epoch timing diagnostics for LC-only triage."""

    epochs: list[int]
    oc_seconds: list[float]
    snr: list[float]
    rms_seconds: float | None
    periodicity_score: float | None
    linear_trend_sec_per_epoch: float | None


@dataclass(frozen=True)
class AliasHarmonicSummaryData:
    """Compact harmonic score summary for P, P/2, and 2P."""

    harmonic_labels: list[str]
    periods: list[float]
    scores: list[float]
    harmonic_depth_ppm: list[float]
    best_harmonic: str
    best_ratio_over_p: float
    classification: str | None = None
    phase_shift_event_count: int | None = None
    phase_shift_peak_sigma: float | None = None
    secondary_significance: float | None = None


@dataclass(frozen=True)
class LCRobustnessEpochMetrics:
    """Per-epoch LC diagnostics for robustness analysis."""

    epoch_index: int
    t_mid_expected_btjd: float
    t_mid_measured_btjd: float | None
    time_coverage_fraction: float
    n_points_total: int
    n_in_transit: int
    n_oot: int
    depth_ppm: float | None
    depth_err_ppm: float | None
    baseline_level: float | None
    baseline_slope_per_day: float | None
    oot_scatter_ppm: float | None
    oot_mad_ppm: float | None
    in_transit_outlier_count: int
    oot_outlier_count: int
    quality_in_transit_nonzero: int | None
    quality_oot_nonzero: int | None


@dataclass(frozen=True)
class LCRobustnessMetrics:
    """LOTO robustness summary derived from per-epoch depth measurements."""

    n_epochs_measured: int
    loto_snr_min: float | None
    loto_snr_max: float | None
    loto_snr_mean: float | None
    loto_depth_ppm_min: float | None
    loto_depth_ppm_max: float | None
    loto_depth_shift_ppm_max: float | None
    dominance_index: float | None


@dataclass(frozen=True)
class LCRobustnessRedNoiseMetrics:
    """Red-noise proxy at standard timescales."""

    beta_30m: float | None
    beta_60m: float | None
    beta_duration: float | None


@dataclass(frozen=True)
class LCFPSignals:
    """Compact LC-only false-positive signal summary."""

    odd_even_depth_diff_sigma: float | None
    secondary_depth_sigma: float | None
    phase_0p5_bin_depth_ppm: float | None
    v_shape_metric: float | None
    asymmetry_sigma: float | None


@dataclass(frozen=True)
class LCRobustnessData:
    """Additive LC robustness data block."""

    version: str
    baseline_window_mult: float
    per_epoch: list[LCRobustnessEpochMetrics]
    robustness: LCRobustnessMetrics
    red_noise: LCRobustnessRedNoiseMetrics
    fp_signals: LCFPSignals


@dataclass(frozen=True)
class EnrichmentBlockData:
    """Normalized non-LC enrichment block payload."""

    status: str  # ok|skipped|error
    flags: list[str]
    quality: dict[str, float | int | str | bool | None]
    checks: dict[str, dict[str, Any]]
    provenance: dict[str, Any]
    payload: dict[str, Any]


@dataclass(frozen=True)
class ReportEnrichmentData:
    """Optional non-LC enrichment payload attached to report output."""

    version: str
    pixel_diagnostics: EnrichmentBlockData | None
    catalog_context: EnrichmentBlockData | None
    followup_context: EnrichmentBlockData | None


@dataclass(frozen=True)
class CheckExecutionState:
    """Execution-state metadata for check enablement decisions."""

    v03_requested: bool
    v03_enabled: bool
    v03_disabled_reason: str | None = None


@dataclass
class ReportData:
    """LC-only report data packet.

    This is the structured output of build_report(). Consumers
    (plotting module, frontend, AI agent) read from this object.
    The report module never renders -- it only assembles data.
    """

    # --- Identity ---
    tic_id: int | None = None
    toi: str | None = None

    # --- Inputs (for provenance / re-rendering) ---
    candidate: Any | None = None
    stellar: Any | None = None

    # --- LC Summary (vital signs) ---
    lc_summary: LCSummary | None = None

    # --- Check Results ---
    # Individual results keyed by check ID for direct access.
    # Values are the original CheckResult objects with plot_data intact.
    checks: dict[str, CheckResult] = field(default_factory=dict)

    # Bundle for aggregate stats (n_passed, failed_check_ids, etc.)
    bundle: VettingBundleResult | None = None

    # --- Plot-Ready Arrays (not from checks) ---
    full_lc: FullLCPlotData | None = None
    phase_folded: PhaseFoldedPlotData | None = None
    per_transit_stack: PerTransitStackPlotData | None = None
    local_detrend: LocalDetrendDiagnosticPlotData | None = None
    oot_context: OOTContextPlotData | None = None
    timing_series: TransitTimingPlotData | None = None
    # Internal full-series timing source for scalar summaries.
    timing_summary_series: TransitTimingPlotData | None = None
    alias_summary: AliasHarmonicSummaryData | None = None
    lc_robustness: LCRobustnessData | None = None
    enrichment: ReportEnrichmentData | None = None
    odd_even_phase: OddEvenPhasePlotData | None = None
    secondary_scan: SecondaryScanPlotData | None = None
    check_execution: CheckExecutionState | None = None
    custom_views: dict[str, Any] | None = None

    # --- Metadata ---
    version: str = "2.0.0"
    checks_run: list[str] = field(default_factory=list)  # ordered IDs

    def to_json(self) -> dict[str, Any]:
        """Serialize to modular JSON contract: summary + plot_data."""
        context = ReportAssemblyContext(
            tic_id=self.tic_id,
            toi=self.toi,
            candidate=self.candidate,
            stellar=self.stellar,
            lc_summary=self.lc_summary,
            check_execution=self.check_execution,
            checks=self.checks,
            bundle=self.bundle,
            enrichment=self.enrichment,
            lc_robustness=self.lc_robustness,
            full_lc=self.full_lc,
            phase_folded=self.phase_folded,
            per_transit_stack=self.per_transit_stack,
            local_detrend=self.local_detrend,
            oot_context=self.oot_context,
            timing_series=self.timing_series,
            timing_summary_series=self.timing_summary_series,
            alias_summary=self.alias_summary,
            odd_even_phase=self.odd_even_phase,
            secondary_scan=self.secondary_scan,
            checks_run=self.checks_run,
        )
        summary, check_overlays = assemble_summary(context)
        plot_data = assemble_plot_data(context, check_overlays=check_overlays)

        summary = _scrub_non_finite(summary)
        plot_data = _scrub_non_finite(plot_data)
        custom_views_model = validate_custom_views_payload(
            self.custom_views if self.custom_views is not None else {"version": "1", "views": []},
            summary=summary,
            plot_data=plot_data,
        )
        custom_views_payload = custom_views_model.model_dump(
            mode="json",
            exclude_none=True,
        )
        custom_view_hash_map = custom_view_hashes_by_id(custom_views_payload)
        metric_contract_meta = _build_check_metric_contract_meta(self.checks)
        for check_id in sorted(metric_contract_meta["missing_required_metrics_by_check"]):
            missing_keys = metric_contract_meta["missing_required_metrics_by_check"][check_id]
            logger.warning(
                "Missing required check metrics for %s: %s",
                check_id,
                ",".join(missing_keys),
            )
        result: dict[str, Any] = {
            "schema_version": self.version,
            "summary": summary,
            "plot_data": plot_data,
            "custom_views": custom_views_payload,
            "payload_meta": {
                "summary_version": "1",
                "plot_data_version": "1",
                "custom_views_version": custom_views_payload["version"],
                "summary_hash": _canonical_sha256(summary),
                "plot_data_hash": _canonical_sha256(plot_data),
                "custom_views_hash": custom_views_hash(custom_views_payload),
                "custom_view_hashes_by_id": custom_view_hash_map,
                "custom_views_includes_ad_hoc": any(
                    view.get("mode") == "ad_hoc" for view in custom_views_payload["views"]
                ),
                **metric_contract_meta,
            },
        }
        result = _scrub_non_finite(result)
        # Enforce typed payload contract at the producer boundary.
        return ReportPayloadModel.model_validate(result).model_dump(exclude_none=True)
