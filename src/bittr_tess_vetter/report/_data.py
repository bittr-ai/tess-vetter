"""Report data structures for LC-only vetting reports.

Defines the structured output of build_report(). Consumers
(plotting module, frontend, AI agent) read from these objects.
The report module never renders -- it only assembles data.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from bittr_tess_vetter.api.types import (
    Candidate,
    CheckResult,
    StellarParams,
    VettingBundleResult,
)


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


@dataclass
class ReportData:
    """Phase 1 LC-only report data packet.

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
    odd_even_phase: OddEvenPhasePlotData | None = None
    secondary_scan: SecondaryScanPlotData | None = None

    # --- Metadata ---
    version: str = "1.0.0"
    checks_run: list[str] = field(default_factory=list)  # ordered IDs

    def to_json(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict.

        - CheckResult objects serialize via .model_dump()
        - Numpy arrays are already converted to lists in plot data
        - LCSummary and plot data are dataclasses -> asdict()
        - NaN/Inf values are scrubbed to None for RFC 8259 compliance
        """
        result: dict[str, Any] = {
            "version": self.version,
            "tic_id": self.tic_id,
            "toi": self.toi,
            "checks_run": self.checks_run,
        }

        # Candidate ephemeris
        if self.candidate is not None:
            eph = self.candidate.ephemeris
            result["ephemeris"] = {
                "period_days": eph.period_days,
                "t0_btjd": eph.t0_btjd,
                "duration_hours": eph.duration_hours,
            }
            result["input_depth_ppm"] = self.candidate.depth_ppm

        # Stellar params (provenance)
        if self.stellar is not None:
            result["stellar"] = self.stellar.model_dump()

        # LC Summary
        if self.lc_summary is not None:
            result["lc_summary"] = asdict(self.lc_summary)

        # Check results -- full serialization including plot_data
        result["checks"] = {}
        for check_id, cr in self.checks.items():
            result["checks"][check_id] = cr.model_dump()

        # Bundle summary
        if self.bundle is not None:
            result["bundle_summary"] = {
                "n_checks": len(self.bundle.results),
                "n_ok": self.bundle.n_passed,
                "n_failed": self.bundle.n_failed,
                "n_skipped": self.bundle.n_unknown,
                "failed_ids": self.bundle.failed_check_ids,
            }

        # Plot-ready arrays (full LC + phase folded)
        if self.full_lc is not None:
            result["full_lc"] = asdict(self.full_lc)
        if self.phase_folded is not None:
            result["phase_folded"] = asdict(self.phase_folded)
        if self.per_transit_stack is not None:
            result["per_transit_stack"] = asdict(self.per_transit_stack)
        if self.local_detrend is not None:
            result["local_detrend"] = asdict(self.local_detrend)
        if self.oot_context is not None:
            result["oot_context"] = asdict(self.oot_context)
        if self.odd_even_phase is not None:
            result["odd_even_phase"] = asdict(self.odd_even_phase)
        if self.secondary_scan is not None:
            result["secondary_scan"] = asdict(self.secondary_scan)

        return _scrub_non_finite(result)
