"""Report data structures for LC-only vetting reports.

Defines the structured output of build_report(). Consumers
(plotting module, frontend, AI agent) read from these objects.
The report module never renders -- it only assembles data.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from bittr_tess_vetter.api.types import (
    Candidate,
    CheckResult,
    StellarParams,
    VettingBundleResult,
)
from bittr_tess_vetter.report.schema import ReportPayloadModel


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
    best_harmonic: str
    best_ratio_over_p: float


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


def _scrub_non_finite(obj: Any) -> Any:
    """Replace NaN/Inf float values with None for JSON safety (RFC 8259)."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _scrub_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_non_finite(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_scrub_non_finite(v) for v in obj)
    return obj


def _normalize_for_hash(obj: Any) -> Any:
    """Normalize payload for deterministic hashing."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _normalize_for_hash(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_for_hash(v) for v in obj]
    if isinstance(obj, tuple):
        return [_normalize_for_hash(v) for v in obj]
    return obj


def _canonical_sha256(payload: dict[str, Any]) -> str:
    normalized = _normalize_for_hash(payload)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    candidate: Candidate | None = None
    stellar: StellarParams | None = None

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
    alias_summary: AliasHarmonicSummaryData | None = None
    lc_robustness: LCRobustnessData | None = None
    enrichment: ReportEnrichmentData | None = None
    odd_even_phase: OddEvenPhasePlotData | None = None
    secondary_scan: SecondaryScanPlotData | None = None

    # --- Metadata ---
    version: str = "1.0.0"
    checks_run: list[str] = field(default_factory=list)  # ordered IDs

    def to_json(self) -> dict[str, Any]:
        """Serialize to modular JSON contract: summary + plot_data."""
        summary: dict[str, Any] = {
            "tic_id": self.tic_id,
            "toi": self.toi,
            "checks_run": list(self.checks_run),
        }
        plot_data: dict[str, Any] = {}

        if self.candidate is not None:
            eph = self.candidate.ephemeris
            summary["ephemeris"] = {
                "period_days": eph.period_days,
                "t0_btjd": eph.t0_btjd,
                "duration_hours": eph.duration_hours,
            }
            summary["input_depth_ppm"] = self.candidate.depth_ppm

        if self.stellar is not None:
            summary["stellar"] = self.stellar.model_dump()

        if self.lc_summary is not None:
            summary["lc_summary"] = asdict(self.lc_summary)

        checks_summary: dict[str, Any] = {}
        check_overlays: dict[str, Any] = {}
        for check_id, cr in self.checks.items():
            checks_summary[check_id] = {
                "id": cr.id,
                "name": cr.name,
                "status": cr.status,
                "confidence": cr.confidence,
                "metrics": cr.metrics,
                "flags": cr.flags,
                "notes": cr.notes,
                "provenance": cr.provenance,
            }
            if isinstance(cr.raw, dict) and "plot_data" in cr.raw:
                check_overlays[check_id] = cr.raw["plot_data"]
        summary["checks"] = checks_summary
        if check_overlays:
            plot_data["check_overlays"] = check_overlays

        if self.bundle is not None:
            summary["bundle_summary"] = {
                "n_checks": len(self.bundle.results),
                "n_ok": self.bundle.n_passed,
                "n_failed": self.bundle.n_failed,
                "n_skipped": self.bundle.n_unknown,
                "failed_ids": self.bundle.failed_check_ids,
            }

        if self.enrichment is not None:
            summary["enrichment"] = asdict(self.enrichment)

        if self.lc_robustness is not None:
            rb = self.lc_robustness.robustness
            rn = self.lc_robustness.red_noise
            fp = self.lc_robustness.fp_signals
            summary["lc_robustness_summary"] = {
                "version": self.lc_robustness.version,
                "n_epochs_stored": len(self.lc_robustness.per_epoch),
                "n_epochs_measured": rb.n_epochs_measured,
                "dominance_index": rb.dominance_index,
                "loto_snr_min": rb.loto_snr_min,
                "loto_snr_mean": rb.loto_snr_mean,
                "loto_snr_max": rb.loto_snr_max,
                "loto_depth_shift_ppm_max": rb.loto_depth_shift_ppm_max,
                "beta_30m": rn.beta_30m,
                "beta_60m": rn.beta_60m,
                "beta_duration": rn.beta_duration,
                "odd_even_depth_diff_sigma": fp.odd_even_depth_diff_sigma,
                "secondary_depth_sigma": fp.secondary_depth_sigma,
                "phase_0p5_bin_depth_ppm": fp.phase_0p5_bin_depth_ppm,
            }
            plot_data["lc_robustness"] = asdict(self.lc_robustness)

        if self.full_lc is not None:
            plot_data["full_lc"] = asdict(self.full_lc)
        if self.phase_folded is not None:
            plot_data["phase_folded"] = asdict(self.phase_folded)
        if self.per_transit_stack is not None:
            plot_data["per_transit_stack"] = asdict(self.per_transit_stack)
        if self.local_detrend is not None:
            plot_data["local_detrend"] = asdict(self.local_detrend)
        if self.oot_context is not None:
            plot_data["oot_context"] = asdict(self.oot_context)
        if self.timing_series is not None:
            plot_data["timing_series"] = asdict(self.timing_series)
        if self.alias_summary is not None:
            plot_data["alias_summary"] = asdict(self.alias_summary)
        if self.odd_even_phase is not None:
            plot_data["odd_even_phase"] = asdict(self.odd_even_phase)
        if self.secondary_scan is not None:
            plot_data["secondary_scan"] = asdict(self.secondary_scan)

        summary = _scrub_non_finite(summary)
        plot_data = _scrub_non_finite(plot_data)
        result: dict[str, Any] = {
            "schema_version": self.version,
            "summary": summary,
            "plot_data": plot_data,
            "payload_meta": {
                "summary_version": "1",
                "plot_data_version": "1",
                "summary_hash": _canonical_sha256(summary),
                "plot_data_hash": _canonical_sha256(plot_data),
            },
        }
        result = _scrub_non_finite(result)
        # Enforce typed payload contract at the producer boundary.
        return ReportPayloadModel.model_validate(result).model_dump(exclude_none=True)
