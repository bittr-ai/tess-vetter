"""TypedDict contracts for aggregate feature inputs and outputs.

All TypedDicts use total=False for optional fields, making them flexible
for partial data scenarios common in astronomical pipelines.
"""

from typing import Literal, TypedDict

# -----------------------------------------------------------------------------
# Verdict type
# -----------------------------------------------------------------------------

Verdict = Literal["ON_TARGET", "OFF_TARGET", "AMBIGUOUS", "INVALID", "NO_EVIDENCE"]


# -----------------------------------------------------------------------------
# Input contracts (minimal, tool-agnostic)
# -----------------------------------------------------------------------------


class GhostSectorInput(TypedDict, total=False):
    """Per-sector ghost diagnostic input."""

    sector: int
    ghost_like_score_adjusted: float | None
    scattered_light_risk: float | None
    aperture_sign_consistent: bool | None


class LocalizationInput(TypedDict, total=False):
    """Localization result input."""

    # Accept raw strings; caller may pass already-normalized verdicts.
    verdict: str | None
    target_distance_arcsec: float | None
    uncertainty_semimajor_arcsec: float | None
    host_ambiguous_within_1pix: bool | None
    warnings: list[str] | None


class V09Metrics(TypedDict, total=False):
    """V09 difference-image localization metrics input."""

    # V09 in btv is pixel-space; 1 pixel ~ 21 arcsec.
    distance_to_target_pixels: float | None
    localization_reliable: bool | None
    warnings: list[str] | None


class HostScenario(TypedDict, total=False):
    """Individual host scenario from plausibility analysis."""

    source_id: str | None
    flux_fraction: float | None
    true_depth_ppm: float | None
    depth_correction_factor: float | None
    physically_impossible: bool | None


class HostPlausibilityInput(TypedDict, total=False):
    """Host plausibility analysis input."""

    requires_resolved_followup: bool | None
    rationale: str | None
    physically_impossible_source_ids: list[str] | None
    scenarios: list[HostScenario] | None


class PixelTimeseriesInput(TypedDict, total=False):
    """Pixel timeseries analysis input."""

    verdict: Verdict | None
    delta_chi2: float | None
    best_source_id: str | None
    n_windows: int | None
    agrees_with_consensus: bool | None


class PixelHostInput(TypedDict, total=False):
    """Pixel-level host identification input."""

    ambiguity: str | None
    # Typically "stable" | "mixed" | "flipping"
    disagreement_flag: str | None
    flip_rate: float | None
    timeseries: PixelTimeseriesInput | None
    ghost_by_sector: list[GhostSectorInput] | None
    host_plausibility: HostPlausibilityInput | None


class CheckPresenceFlags(TypedDict, total=False):
    """Flags indicating presence of various data products."""

    has_tpf: bool
    has_localization: bool
    has_diff_image: bool
    has_aperture_family: bool
    has_pixel_timeseries: bool
    has_ghost_summary: bool
    has_host_plausibility: bool


# -----------------------------------------------------------------------------
# Output contracts
# -----------------------------------------------------------------------------


class GhostSummary(TypedDict, total=False):
    """Aggregated ghost diagnostic summary."""

    ghost_like_score_adjusted_median: float | None
    ghost_like_score_adjusted_max: float | None
    scattered_light_risk_median: float | None
    scattered_light_risk_max: float | None
    aperture_sign_consistent_all: bool | None
    aperture_sign_consistent_any_false: bool | None


class LocalizationSummary(TypedDict, total=False):
    """Aggregated localization summary."""

    localization_verdict: Verdict | None
    localization_target_distance_arcsec: float | None
    localization_uncertainty_semimajor_arcsec: float | None
    localization_low_confidence: bool | None
    host_ambiguous_within_1pix: bool | None
    v09_localization_reliable: bool | None


class HostPlausibilitySummary(TypedDict, total=False):
    """Aggregated host plausibility summary."""

    host_requires_resolved_followup: bool | None
    host_physically_impossible_count: int | None
    host_physically_impossible_source_ids: list[str] | None
    host_feasible_best_source_id: str | None
    host_feasible_best_flux_fraction: float | None
    host_feasible_best_true_depth_ppm: float | None
    host_plausibility_rationale: str | None


class PixelHostSummary(TypedDict, total=False):
    """Aggregated pixel-level host summary."""

    pixel_host_ambiguity: str | None
    pixel_disagreement_flag: str | None
    pixel_flip_rate: float | None
    pixel_timeseries_verdict: Verdict | None
    pixel_timeseries_delta_chi2: float | None
    pixel_timeseries_best_source_id: str | None
    pixel_timeseries_n_windows: int | None
    pixel_timeseries_agrees_with_consensus: bool | None


class CoverageSummary(TypedDict, total=False):
    """Data coverage summary."""

    tpf_coverage_ok: bool
    missing_feature_families: list[str]


class Aggregates(TypedDict, total=False):
    """Top-level container for all aggregated summaries."""

    ghost: GhostSummary
    localization: LocalizationSummary
    host_plausibility: HostPlausibilitySummary
    pixel_host: PixelHostSummary
    coverage: CoverageSummary
