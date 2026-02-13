"""Typed field metadata registry for report JSON paths."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FieldKey(StrEnum):
    """Stable field-key enum for client/UI metadata lookup."""

    SUMMARY_SNR = "summary.lc_summary.snr"
    SUMMARY_DEPTH_PPM = "summary.lc_summary.depth_ppm"
    SUMMARY_DEPTH_ERR_PPM = "summary.lc_summary.depth_err_ppm"
    SUMMARY_TRANSITS = "summary.lc_summary.n_transits"
    SUMMARY_IN_TRANSIT_POINTS = "summary.lc_summary.n_in_transit_total"
    SUMMARY_GAP_FRACTION = "summary.lc_summary.gap_fraction"
    SUMMARY_CADENCE_SECONDS = "summary.lc_summary.cadence_seconds"
    SUMMARY_CHECK_METHOD_REFS = "summary.checks[*].method_refs"
    SUMMARY_REFERENCES = "summary.references"
    SUMMARY_REFERENCE_KEY = "summary.references[*].key"
    SUMMARY_REFERENCE_TITLE = "summary.references[*].title"
    SUMMARY_REFERENCE_AUTHORS = "summary.references[*].authors"
    SUMMARY_REFERENCE_YEAR = "summary.references[*].year"
    SUMMARY_REFERENCE_VENUE = "summary.references[*].venue"
    SUMMARY_REFERENCE_DOI = "summary.references[*].doi"
    SUMMARY_REFERENCE_URL = "summary.references[*].url"
    SUMMARY_REFERENCE_CITATION = "summary.references[*].citation"
    SUMMARY_REFERENCE_NOTES = "summary.references[*].notes"
    SUMMARY_REFERENCE_TAGS = "summary.references[*].tags"
    SUMMARY_ODD_EVEN = "summary.odd_even_summary"
    SUMMARY_ODD_EVEN_ODD_DEPTH_PPM = "summary.odd_even_summary.odd_depth_ppm"
    SUMMARY_ODD_EVEN_EVEN_DEPTH_PPM = "summary.odd_even_summary.even_depth_ppm"
    SUMMARY_ODD_EVEN_DEPTH_DIFF_PPM = "summary.odd_even_summary.depth_diff_ppm"
    SUMMARY_ODD_EVEN_DEPTH_DIFF_SIGMA = "summary.odd_even_summary.depth_diff_sigma"
    SUMMARY_ODD_EVEN_IS_SIGNIFICANT = "summary.odd_even_summary.is_significant"
    SUMMARY_ODD_EVEN_FLAGS = "summary.odd_even_summary.flags"
    SUMMARY_NOISE = "summary.noise_summary"
    SUMMARY_NOISE_WHITE_NOISE_PPM = "summary.noise_summary.white_noise_ppm"
    SUMMARY_NOISE_RED_NOISE_BETA_30M = "summary.noise_summary.red_noise_beta_30m"
    SUMMARY_NOISE_RED_NOISE_BETA_60M = "summary.noise_summary.red_noise_beta_60m"
    SUMMARY_NOISE_RED_NOISE_BETA_DURATION = "summary.noise_summary.red_noise_beta_duration"
    SUMMARY_NOISE_TREND_STAT = "summary.noise_summary.trend_stat"
    SUMMARY_NOISE_TREND_STAT_UNIT = "summary.noise_summary.trend_stat_unit"
    SUMMARY_NOISE_FLAGS = "summary.noise_summary.flags"
    SUMMARY_NOISE_SEMANTICS = "summary.noise_summary.semantics"
    SUMMARY_VARIABILITY = "summary.variability_summary"
    SUMMARY_VARIABILITY_INDEX = "summary.variability_summary.variability_index"
    SUMMARY_VARIABILITY_PERIODICITY_SCORE = "summary.variability_summary.periodicity_score"
    SUMMARY_VARIABILITY_FLARE_RATE_PER_DAY = "summary.variability_summary.flare_rate_per_day"
    SUMMARY_VARIABILITY_CLASSIFICATION = "summary.variability_summary.classification"
    SUMMARY_VARIABILITY_FLAGS = "summary.variability_summary.flags"
    SUMMARY_VARIABILITY_SEMANTICS = "summary.variability_summary.semantics"
    SUMMARY_ALIAS_SCALAR = "summary.alias_scalar_summary"
    SUMMARY_ALIAS_SCALAR_BEST_HARMONIC = "summary.alias_scalar_summary.best_harmonic"
    SUMMARY_ALIAS_SCALAR_BEST_RATIO_OVER_P = "summary.alias_scalar_summary.best_ratio_over_p"
    SUMMARY_ALIAS_SCALAR_SCORE_P = "summary.alias_scalar_summary.score_p"
    SUMMARY_ALIAS_SCALAR_SCORE_P_OVER_2 = "summary.alias_scalar_summary.score_p_over_2"
    SUMMARY_ALIAS_SCALAR_SCORE_2P = "summary.alias_scalar_summary.score_2p"
    SUMMARY_ALIAS_SCALAR_DEPTH_PPM_PEAK = "summary.alias_scalar_summary.depth_ppm_peak"
    SUMMARY_ALIAS_SCALAR_CLASSIFICATION = "summary.alias_scalar_summary.classification"
    SUMMARY_ALIAS_SCALAR_PHASE_SHIFT_EVENT_COUNT = (
        "summary.alias_scalar_summary.phase_shift_event_count"
    )
    SUMMARY_ALIAS_SCALAR_PHASE_SHIFT_PEAK_SIGMA = (
        "summary.alias_scalar_summary.phase_shift_peak_sigma"
    )
    SUMMARY_ALIAS_SCALAR_SECONDARY_SIGNIFICANCE = (
        "summary.alias_scalar_summary.secondary_significance"
    )
    SUMMARY_LC_ROBUSTNESS_V_SHAPE_METRIC = "summary.lc_robustness_summary.v_shape_metric"
    SUMMARY_LC_ROBUSTNESS_ASYMMETRY_SIGMA = "summary.lc_robustness_summary.asymmetry_sigma"
    SUMMARY_TIMING = "summary.timing_summary"
    SUMMARY_TIMING_N_EPOCHS_MEASURED = "summary.timing_summary.n_epochs_measured"
    SUMMARY_TIMING_RMS_SECONDS = "summary.timing_summary.rms_seconds"
    SUMMARY_TIMING_PERIODICITY_SCORE = "summary.timing_summary.periodicity_score"
    SUMMARY_TIMING_LINEAR_TREND_SEC_PER_EPOCH = "summary.timing_summary.linear_trend_sec_per_epoch"
    SUMMARY_TIMING_MAX_ABS_OC_SECONDS = "summary.timing_summary.max_abs_oc_seconds"
    SUMMARY_TIMING_MAX_SNR = "summary.timing_summary.max_snr"
    SUMMARY_TIMING_OUTLIER_COUNT = "summary.timing_summary.outlier_count"
    SUMMARY_TIMING_OUTLIER_FRACTION = "summary.timing_summary.outlier_fraction"
    SUMMARY_TIMING_DEEPEST_EPOCH = "summary.timing_summary.deepest_epoch"
    SUMMARY_SECONDARY_SCAN = "summary.secondary_scan_summary"
    SUMMARY_SECONDARY_SCAN_PHASE_COVERAGE_FRACTION = (
        "summary.secondary_scan_summary.phase_coverage_fraction"
    )
    SUMMARY_SECONDARY_SCAN_LARGEST_PHASE_GAP = "summary.secondary_scan_summary.largest_phase_gap"
    SUMMARY_SECONDARY_SCAN_N_BINS_WITH_ERROR = "summary.secondary_scan_summary.n_bins_with_error"
    SUMMARY_SECONDARY_SCAN_STRONGEST_DIP_PHASE = "summary.secondary_scan_summary.strongest_dip_phase"
    SUMMARY_SECONDARY_SCAN_STRONGEST_DIP_DEPTH_PPM = (
        "summary.secondary_scan_summary.strongest_dip_depth_ppm"
    )
    SUMMARY_SECONDARY_SCAN_IS_DEGRADED = "summary.secondary_scan_summary.is_degraded"
    SUMMARY_SECONDARY_SCAN_QUALITY_FLAG_COUNT = "summary.secondary_scan_summary.quality_flag_count"
    SUMMARY_DATA_GAP = "summary.data_gap_summary"
    SUMMARY_DATA_GAP_MISSING_FRAC_MAX_IN_COVERAGE = (
        "summary.data_gap_summary.missing_frac_max_in_coverage"
    )
    SUMMARY_DATA_GAP_MISSING_FRAC_MEDIAN_IN_COVERAGE = (
        "summary.data_gap_summary.missing_frac_median_in_coverage"
    )
    SUMMARY_DATA_GAP_N_EPOCHS_MISSING_GE_0P25_IN_COVERAGE = (
        "summary.data_gap_summary.n_epochs_missing_ge_0p25_in_coverage"
    )
    SUMMARY_DATA_GAP_N_EPOCHS_EXCLUDED_NO_COVERAGE = (
        "summary.data_gap_summary.n_epochs_excluded_no_coverage"
    )
    SUMMARY_DATA_GAP_N_EPOCHS_EVALUATED_IN_COVERAGE = (
        "summary.data_gap_summary.n_epochs_evaluated_in_coverage"
    )
    SUMMARY_ENRICHMENT = "summary.enrichment"
    PLOT_PHASE_FOLDED = "plot_data.phase_folded"
    PLOT_FULL_LC = "plot_data.full_lc"


@dataclass(frozen=True)
class FieldSpec:
    """Display metadata for a report field."""

    path: str
    display_name: str
    description: str
    units: str | None = None
    category: str = "summary"


FIELD_CATALOG: dict[FieldKey, FieldSpec] = {
    FieldKey.SUMMARY_SNR: FieldSpec(
        path=FieldKey.SUMMARY_SNR.value,
        display_name="SNR",
        description="Box-model transit signal-to-noise ratio.",
        units="sigma",
    ),
    FieldKey.SUMMARY_DEPTH_PPM: FieldSpec(
        path=FieldKey.SUMMARY_DEPTH_PPM.value,
        display_name="Depth",
        description="Measured transit depth from LC-only analysis.",
        units="ppm",
    ),
    FieldKey.SUMMARY_DEPTH_ERR_PPM: FieldSpec(
        path=FieldKey.SUMMARY_DEPTH_ERR_PPM.value,
        display_name="Depth Error",
        description="Estimated uncertainty on measured depth.",
        units="ppm",
    ),
    FieldKey.SUMMARY_TRANSITS: FieldSpec(
        path=FieldKey.SUMMARY_TRANSITS.value,
        display_name="Observed Transits",
        description="Number of transit epochs with sufficient in-transit points.",
        units=None,
    ),
    FieldKey.SUMMARY_IN_TRANSIT_POINTS: FieldSpec(
        path=FieldKey.SUMMARY_IN_TRANSIT_POINTS.value,
        display_name="In-Transit Points",
        description="Total in-transit samples across observed epochs.",
        units=None,
    ),
    FieldKey.SUMMARY_GAP_FRACTION: FieldSpec(
        path=FieldKey.SUMMARY_GAP_FRACTION.value,
        display_name="Gap Fraction",
        description="Fraction of invalid/missing cadences.",
        units="fraction",
    ),
    FieldKey.SUMMARY_CADENCE_SECONDS: FieldSpec(
        path=FieldKey.SUMMARY_CADENCE_SECONDS.value,
        display_name="Cadence",
        description="Median time spacing between valid samples.",
        units="seconds",
    ),
    FieldKey.SUMMARY_CHECK_METHOD_REFS: FieldSpec(
        path=FieldKey.SUMMARY_CHECK_METHOD_REFS.value,
        display_name="Check Method References",
        description="Method/function reference IDs attached to each check summary.",
        category="reference",
    ),
    FieldKey.SUMMARY_REFERENCES: FieldSpec(
        path=FieldKey.SUMMARY_REFERENCES.value,
        display_name="References",
        description="Bibliographic references used by report methods and summaries.",
        category="reference",
    ),
    FieldKey.SUMMARY_REFERENCE_KEY: FieldSpec(
        path=FieldKey.SUMMARY_REFERENCE_KEY.value,
        display_name="Reference Key",
        description="Stable reference identifier.",
        category="reference",
    ),
    FieldKey.SUMMARY_REFERENCE_TITLE: FieldSpec(
        path=FieldKey.SUMMARY_REFERENCE_TITLE.value,
        display_name="Reference Title",
        description="Human-readable reference title.",
        category="reference",
    ),
    FieldKey.SUMMARY_REFERENCE_AUTHORS: FieldSpec(
        path=FieldKey.SUMMARY_REFERENCE_AUTHORS.value,
        display_name="Reference Authors",
        description="Ordered author list for the reference.",
        category="reference",
    ),
    FieldKey.SUMMARY_REFERENCE_YEAR: FieldSpec(
        path=FieldKey.SUMMARY_REFERENCE_YEAR.value,
        display_name="Reference Year",
        description="Publication year.",
        units="year",
        category="reference",
    ),
    FieldKey.SUMMARY_REFERENCE_VENUE: FieldSpec(
        path=FieldKey.SUMMARY_REFERENCE_VENUE.value,
        display_name="Reference Venue",
        description="Journal/conference/source venue.",
        category="reference",
    ),
    FieldKey.SUMMARY_REFERENCE_DOI: FieldSpec(
        path=FieldKey.SUMMARY_REFERENCE_DOI.value,
        display_name="Reference DOI",
        description="Digital Object Identifier (DOI).",
        category="reference",
    ),
    FieldKey.SUMMARY_REFERENCE_URL: FieldSpec(
        path=FieldKey.SUMMARY_REFERENCE_URL.value,
        display_name="Reference URL",
        description="Web URL for the reference landing page.",
        category="reference",
    ),
    FieldKey.SUMMARY_REFERENCE_CITATION: FieldSpec(
        path=FieldKey.SUMMARY_REFERENCE_CITATION.value,
        display_name="Reference Citation",
        description="Preformatted citation string for direct display.",
        category="reference",
    ),
    FieldKey.SUMMARY_REFERENCE_NOTES: FieldSpec(
        path=FieldKey.SUMMARY_REFERENCE_NOTES.value,
        display_name="Reference Notes",
        description="Optional free-form annotation notes.",
        category="reference",
    ),
    FieldKey.SUMMARY_REFERENCE_TAGS: FieldSpec(
        path=FieldKey.SUMMARY_REFERENCE_TAGS.value,
        display_name="Reference Tags",
        description="Reference classification tags for filtering/grouping.",
        category="reference",
    ),
    FieldKey.SUMMARY_ODD_EVEN: FieldSpec(
        path=FieldKey.SUMMARY_ODD_EVEN.value,
        display_name="Odd/Even Summary",
        description="Deterministic odd-even depth comparison summary block.",
    ),
    FieldKey.SUMMARY_ODD_EVEN_ODD_DEPTH_PPM: FieldSpec(
        path=FieldKey.SUMMARY_ODD_EVEN_ODD_DEPTH_PPM.value,
        display_name="Odd Depth",
        description="Estimated odd-transit depth.",
        units="ppm",
    ),
    FieldKey.SUMMARY_ODD_EVEN_EVEN_DEPTH_PPM: FieldSpec(
        path=FieldKey.SUMMARY_ODD_EVEN_EVEN_DEPTH_PPM.value,
        display_name="Even Depth",
        description="Estimated even-transit depth.",
        units="ppm",
    ),
    FieldKey.SUMMARY_ODD_EVEN_DEPTH_DIFF_PPM: FieldSpec(
        path=FieldKey.SUMMARY_ODD_EVEN_DEPTH_DIFF_PPM.value,
        display_name="Odd-Even Depth Difference",
        description="Odd minus even depth difference.",
        units="ppm",
    ),
    FieldKey.SUMMARY_ODD_EVEN_DEPTH_DIFF_SIGMA: FieldSpec(
        path=FieldKey.SUMMARY_ODD_EVEN_DEPTH_DIFF_SIGMA.value,
        display_name="Odd-Even Difference Significance",
        description="Significance of odd-even depth mismatch.",
        units="sigma",
    ),
    FieldKey.SUMMARY_ODD_EVEN_IS_SIGNIFICANT: FieldSpec(
        path=FieldKey.SUMMARY_ODD_EVEN_IS_SIGNIFICANT.value,
        display_name="Odd-Even Significant",
        description="True when odd-even mismatch exceeds configured significance threshold.",
    ),
    FieldKey.SUMMARY_ODD_EVEN_FLAGS: FieldSpec(
        path=FieldKey.SUMMARY_ODD_EVEN_FLAGS.value,
        display_name="Odd-Even Flags",
        description="Machine-readable odd-even diagnostic flags.",
    ),
    FieldKey.SUMMARY_NOISE: FieldSpec(
        path=FieldKey.SUMMARY_NOISE.value,
        display_name="Noise Summary",
        description="Deterministic noise diagnostic summary block.",
    ),
    FieldKey.SUMMARY_NOISE_WHITE_NOISE_PPM: FieldSpec(
        path=FieldKey.SUMMARY_NOISE_WHITE_NOISE_PPM.value,
        display_name="White Noise",
        description="Estimated white-noise floor.",
        units="ppm",
    ),
    FieldKey.SUMMARY_NOISE_RED_NOISE_BETA_30M: FieldSpec(
        path=FieldKey.SUMMARY_NOISE_RED_NOISE_BETA_30M.value,
        display_name="Red Noise Beta (30m)",
        description="Red-noise beta factor at 30-minute bins.",
        units="ratio",
    ),
    FieldKey.SUMMARY_NOISE_RED_NOISE_BETA_60M: FieldSpec(
        path=FieldKey.SUMMARY_NOISE_RED_NOISE_BETA_60M.value,
        display_name="Red Noise Beta (60m)",
        description="Red-noise beta factor at 60-minute bins.",
        units="ratio",
    ),
    FieldKey.SUMMARY_NOISE_RED_NOISE_BETA_DURATION: FieldSpec(
        path=FieldKey.SUMMARY_NOISE_RED_NOISE_BETA_DURATION.value,
        display_name="Red Noise Beta (Duration)",
        description="Red-noise beta factor at transit-duration bins.",
        units="ratio",
    ),
    FieldKey.SUMMARY_NOISE_TREND_STAT: FieldSpec(
        path=FieldKey.SUMMARY_NOISE_TREND_STAT.value,
        display_name="Trend Statistic",
        description="Absolute linear slope magnitude of detrended residual baseline vs time.",
        units="relative_flux_per_day",
    ),
    FieldKey.SUMMARY_NOISE_TREND_STAT_UNIT: FieldSpec(
        path=FieldKey.SUMMARY_NOISE_TREND_STAT_UNIT.value,
        display_name="Trend Statistic Unit",
        description="Physical/statistical unit for trend statistic value.",
    ),
    FieldKey.SUMMARY_NOISE_FLAGS: FieldSpec(
        path=FieldKey.SUMMARY_NOISE_FLAGS.value,
        display_name="Noise Flags",
        description="Machine-readable noise diagnostics flags.",
    ),
    FieldKey.SUMMARY_NOISE_SEMANTICS: FieldSpec(
        path=FieldKey.SUMMARY_NOISE_SEMANTICS.value,
        display_name="Noise Semantics",
        description="Semantics-ready map for downstream interpretation of noise metrics.",
    ),
    FieldKey.SUMMARY_VARIABILITY: FieldSpec(
        path=FieldKey.SUMMARY_VARIABILITY.value,
        display_name="Variability Summary",
        description="Deterministic variability summary block.",
    ),
    FieldKey.SUMMARY_VARIABILITY_INDEX: FieldSpec(
        path=FieldKey.SUMMARY_VARIABILITY_INDEX.value,
        display_name="Variability Index",
        description="Aggregate variability metric for out-of-transit behavior.",
    ),
    FieldKey.SUMMARY_VARIABILITY_PERIODICITY_SCORE: FieldSpec(
        path=FieldKey.SUMMARY_VARIABILITY_PERIODICITY_SCORE.value,
        display_name="Periodicity Score",
        description="Strength of periodic variability signatures.",
    ),
    FieldKey.SUMMARY_VARIABILITY_FLARE_RATE_PER_DAY: FieldSpec(
        path=FieldKey.SUMMARY_VARIABILITY_FLARE_RATE_PER_DAY.value,
        display_name="Flare Rate",
        description="Estimated flare occurrence rate.",
        units="1/day",
    ),
    FieldKey.SUMMARY_VARIABILITY_CLASSIFICATION: FieldSpec(
        path=FieldKey.SUMMARY_VARIABILITY_CLASSIFICATION.value,
        display_name="Variability Classification",
        description="Deterministic variability class label.",
    ),
    FieldKey.SUMMARY_VARIABILITY_FLAGS: FieldSpec(
        path=FieldKey.SUMMARY_VARIABILITY_FLAGS.value,
        display_name="Variability Flags",
        description="Machine-readable variability diagnostics flags.",
    ),
    FieldKey.SUMMARY_VARIABILITY_SEMANTICS: FieldSpec(
        path=FieldKey.SUMMARY_VARIABILITY_SEMANTICS.value,
        display_name="Variability Semantics",
        description="Semantics-ready map for downstream interpretation of variability metrics.",
    ),
    FieldKey.SUMMARY_ALIAS_SCALAR: FieldSpec(
        path=FieldKey.SUMMARY_ALIAS_SCALAR.value,
        display_name="Alias Scalar Summary",
        description="Scalar rollup of alias diagnostics from the alias summary payload.",
    ),
    FieldKey.SUMMARY_ALIAS_SCALAR_BEST_HARMONIC: FieldSpec(
        path=FieldKey.SUMMARY_ALIAS_SCALAR_BEST_HARMONIC.value,
        display_name="Best Harmonic",
        description="Best-matching harmonic label relative to the candidate period.",
    ),
    FieldKey.SUMMARY_ALIAS_SCALAR_BEST_RATIO_OVER_P: FieldSpec(
        path=FieldKey.SUMMARY_ALIAS_SCALAR_BEST_RATIO_OVER_P.value,
        display_name="Best Ratio Over P",
        description="Ratio of the strongest alias period to the candidate period.",
        units="ratio",
    ),
    FieldKey.SUMMARY_ALIAS_SCALAR_SCORE_P: FieldSpec(
        path=FieldKey.SUMMARY_ALIAS_SCALAR_SCORE_P.value,
        display_name="Alias Score (P)",
        description="Alias score evaluated at the candidate period.",
    ),
    FieldKey.SUMMARY_ALIAS_SCALAR_SCORE_P_OVER_2: FieldSpec(
        path=FieldKey.SUMMARY_ALIAS_SCALAR_SCORE_P_OVER_2.value,
        display_name="Alias Score (P/2)",
        description="Alias score evaluated at half the candidate period.",
    ),
    FieldKey.SUMMARY_ALIAS_SCALAR_SCORE_2P: FieldSpec(
        path=FieldKey.SUMMARY_ALIAS_SCALAR_SCORE_2P.value,
        display_name="Alias Score (2P)",
        description="Alias score evaluated at twice the candidate period.",
    ),
    FieldKey.SUMMARY_ALIAS_SCALAR_DEPTH_PPM_PEAK: FieldSpec(
        path=FieldKey.SUMMARY_ALIAS_SCALAR_DEPTH_PPM_PEAK.value,
        display_name="Peak Alias Depth",
        description="Peak alias depth among tested harmonics.",
        units="ppm",
    ),
    FieldKey.SUMMARY_ALIAS_SCALAR_CLASSIFICATION: FieldSpec(
        path=FieldKey.SUMMARY_ALIAS_SCALAR_CLASSIFICATION.value,
        display_name="Alias Classification",
        description=(
            "Alias class from strongest non-P harmonic score ratio "
            "(ALIAS_STRONG >= 1.5, ALIAS_WEAK >= 1.1, else NONE)."
        ),
    ),
    FieldKey.SUMMARY_ALIAS_SCALAR_PHASE_SHIFT_EVENT_COUNT: FieldSpec(
        path=FieldKey.SUMMARY_ALIAS_SCALAR_PHASE_SHIFT_EVENT_COUNT.value,
        display_name="Phase-Shift Event Count",
        description="Count of non-primary phase bins with event significance >= 3.0 sigma.",
    ),
    FieldKey.SUMMARY_ALIAS_SCALAR_PHASE_SHIFT_PEAK_SIGMA: FieldSpec(
        path=FieldKey.SUMMARY_ALIAS_SCALAR_PHASE_SHIFT_PEAK_SIGMA.value,
        display_name="Peak Phase-Shift Significance",
        description="Maximum sigma among detected non-primary phase-shift events.",
        units="sigma",
    ),
    FieldKey.SUMMARY_ALIAS_SCALAR_SECONDARY_SIGNIFICANCE: FieldSpec(
        path=FieldKey.SUMMARY_ALIAS_SCALAR_SECONDARY_SIGNIFICANCE.value,
        display_name="Secondary Significance",
        description="Secondary eclipse depth significance at phase 0.5.",
        units="sigma",
    ),
    FieldKey.SUMMARY_LC_ROBUSTNESS_V_SHAPE_METRIC: FieldSpec(
        path=FieldKey.SUMMARY_LC_ROBUSTNESS_V_SHAPE_METRIC.value,
        display_name="V-Shape Metric",
        description="Transit shape ratio t_flat/t_total from LC robustness FP signals.",
        units="ratio",
    ),
    FieldKey.SUMMARY_LC_ROBUSTNESS_ASYMMETRY_SIGMA: FieldSpec(
        path=FieldKey.SUMMARY_LC_ROBUSTNESS_ASYMMETRY_SIGMA.value,
        display_name="Asymmetry Significance",
        description="Ingress/egress asymmetry significance from LC robustness FP signals.",
        units="sigma",
    ),
    FieldKey.SUMMARY_TIMING: FieldSpec(
        path=FieldKey.SUMMARY_TIMING.value,
        display_name="Timing Summary",
        description="Scalar rollup of transit timing variation diagnostics.",
    ),
    FieldKey.SUMMARY_TIMING_N_EPOCHS_MEASURED: FieldSpec(
        path=FieldKey.SUMMARY_TIMING_N_EPOCHS_MEASURED.value,
        display_name="Measured Epochs",
        description="Number of epochs with measured O-C timing values.",
    ),
    FieldKey.SUMMARY_TIMING_RMS_SECONDS: FieldSpec(
        path=FieldKey.SUMMARY_TIMING_RMS_SECONDS.value,
        display_name="Timing RMS",
        description="Root-mean-square timing residual.",
        units="seconds",
    ),
    FieldKey.SUMMARY_TIMING_PERIODICITY_SCORE: FieldSpec(
        path=FieldKey.SUMMARY_TIMING_PERIODICITY_SCORE.value,
        display_name="Timing Periodicity Score",
        description="Strength of periodic structure in timing residuals.",
    ),
    FieldKey.SUMMARY_TIMING_LINEAR_TREND_SEC_PER_EPOCH: FieldSpec(
        path=FieldKey.SUMMARY_TIMING_LINEAR_TREND_SEC_PER_EPOCH.value,
        display_name="Linear Trend",
        description="Best-fit linear timing trend per epoch.",
        units="sec/epoch",
    ),
    FieldKey.SUMMARY_TIMING_MAX_ABS_OC_SECONDS: FieldSpec(
        path=FieldKey.SUMMARY_TIMING_MAX_ABS_OC_SECONDS.value,
        display_name="Max |O-C|",
        description="Maximum absolute observed-minus-computed timing residual.",
        units="seconds",
    ),
    FieldKey.SUMMARY_TIMING_MAX_SNR: FieldSpec(
        path=FieldKey.SUMMARY_TIMING_MAX_SNR.value,
        display_name="Max Timing SNR",
        description="Largest per-epoch timing-fit signal-to-noise ratio.",
        units="sigma",
    ),
    FieldKey.SUMMARY_TIMING_OUTLIER_COUNT: FieldSpec(
        path=FieldKey.SUMMARY_TIMING_OUTLIER_COUNT.value,
        display_name="Timing Outlier Count",
        description="Count of epochs flagged as timing outliers.",
    ),
    FieldKey.SUMMARY_TIMING_OUTLIER_FRACTION: FieldSpec(
        path=FieldKey.SUMMARY_TIMING_OUTLIER_FRACTION.value,
        display_name="Timing Outlier Fraction",
        description="Outlier count divided by measured epoch count.",
        units="fraction",
    ),
    FieldKey.SUMMARY_TIMING_DEEPEST_EPOCH: FieldSpec(
        path=FieldKey.SUMMARY_TIMING_DEEPEST_EPOCH.value,
        display_name="Deepest Epoch",
        description="Epoch index of the deepest transit event (smallest index on ties).",
        units="epoch_index",
    ),
    FieldKey.SUMMARY_SECONDARY_SCAN: FieldSpec(
        path=FieldKey.SUMMARY_SECONDARY_SCAN.value,
        display_name="Secondary Scan Summary",
        description="Scalar rollup of secondary eclipse scan coverage and quality.",
    ),
    FieldKey.SUMMARY_SECONDARY_SCAN_PHASE_COVERAGE_FRACTION: FieldSpec(
        path=FieldKey.SUMMARY_SECONDARY_SCAN_PHASE_COVERAGE_FRACTION.value,
        display_name="Phase Coverage",
        description="Fraction of phase bins with valid secondary-scan coverage.",
        units="fraction",
    ),
    FieldKey.SUMMARY_SECONDARY_SCAN_LARGEST_PHASE_GAP: FieldSpec(
        path=FieldKey.SUMMARY_SECONDARY_SCAN_LARGEST_PHASE_GAP.value,
        display_name="Largest Phase Gap",
        description="Largest contiguous uncovered phase interval.",
        units="phase",
    ),
    FieldKey.SUMMARY_SECONDARY_SCAN_N_BINS_WITH_ERROR: FieldSpec(
        path=FieldKey.SUMMARY_SECONDARY_SCAN_N_BINS_WITH_ERROR.value,
        display_name="Bins With Error",
        description="Number of secondary-scan bins with finite error estimates.",
    ),
    FieldKey.SUMMARY_SECONDARY_SCAN_STRONGEST_DIP_PHASE: FieldSpec(
        path=FieldKey.SUMMARY_SECONDARY_SCAN_STRONGEST_DIP_PHASE.value,
        display_name="Strongest Dip Phase",
        description="Phase location of the strongest dip in the secondary scan.",
        units="phase",
    ),
    FieldKey.SUMMARY_SECONDARY_SCAN_STRONGEST_DIP_DEPTH_PPM: FieldSpec(
        path=FieldKey.SUMMARY_SECONDARY_SCAN_STRONGEST_DIP_DEPTH_PPM.value,
        display_name="Strongest Dip Depth",
        description="Depth of the strongest secondary-scan dip.",
        units="ppm",
    ),
    FieldKey.SUMMARY_SECONDARY_SCAN_IS_DEGRADED: FieldSpec(
        path=FieldKey.SUMMARY_SECONDARY_SCAN_IS_DEGRADED.value,
        display_name="Secondary Scan Degraded",
        description="True when scan quality is degraded by coverage/quality failures.",
    ),
    FieldKey.SUMMARY_SECONDARY_SCAN_QUALITY_FLAG_COUNT: FieldSpec(
        path=FieldKey.SUMMARY_SECONDARY_SCAN_QUALITY_FLAG_COUNT.value,
        display_name="Secondary Quality Flag Count",
        description="Number of quality flags raised for the secondary scan.",
    ),
    FieldKey.SUMMARY_DATA_GAP: FieldSpec(
        path=FieldKey.SUMMARY_DATA_GAP.value,
        display_name="Data Gap Summary",
        description="Scalar rollup of V13 in-coverage missing-cadence diagnostics.",
    ),
    FieldKey.SUMMARY_DATA_GAP_MISSING_FRAC_MAX_IN_COVERAGE: FieldSpec(
        path=FieldKey.SUMMARY_DATA_GAP_MISSING_FRAC_MAX_IN_COVERAGE.value,
        display_name="Max Missing Fraction In Coverage",
        description="Maximum missing cadence fraction among epochs with coverage.",
        units="fraction",
    ),
    FieldKey.SUMMARY_DATA_GAP_MISSING_FRAC_MEDIAN_IN_COVERAGE: FieldSpec(
        path=FieldKey.SUMMARY_DATA_GAP_MISSING_FRAC_MEDIAN_IN_COVERAGE.value,
        display_name="Median Missing Fraction In Coverage",
        description="Median missing cadence fraction among epochs with coverage.",
        units="fraction",
    ),
    FieldKey.SUMMARY_DATA_GAP_N_EPOCHS_MISSING_GE_0P25_IN_COVERAGE: FieldSpec(
        path=FieldKey.SUMMARY_DATA_GAP_N_EPOCHS_MISSING_GE_0P25_IN_COVERAGE.value,
        display_name="Epochs Missing >= 0.25 In Coverage",
        description="Count of in-coverage epochs with missing fraction >= 0.25.",
        units="epochs",
    ),
    FieldKey.SUMMARY_DATA_GAP_N_EPOCHS_EXCLUDED_NO_COVERAGE: FieldSpec(
        path=FieldKey.SUMMARY_DATA_GAP_N_EPOCHS_EXCLUDED_NO_COVERAGE.value,
        display_name="Epochs Excluded No Coverage",
        description="Count of candidate epochs excluded because no cadence coverage exists.",
        units="epochs",
    ),
    FieldKey.SUMMARY_DATA_GAP_N_EPOCHS_EVALUATED_IN_COVERAGE: FieldSpec(
        path=FieldKey.SUMMARY_DATA_GAP_N_EPOCHS_EVALUATED_IN_COVERAGE.value,
        display_name="Epochs Evaluated In Coverage",
        description="Count of candidate epochs evaluated with cadence coverage.",
        units="epochs",
    ),
    FieldKey.SUMMARY_ENRICHMENT: FieldSpec(
        path=FieldKey.SUMMARY_ENRICHMENT.value,
        display_name="Enrichment",
        description="Operational status/provenance for non-LC context blocks.",
        category="enrichment",
    ),
    FieldKey.PLOT_PHASE_FOLDED: FieldSpec(
        path=FieldKey.PLOT_PHASE_FOLDED.value,
        display_name="Phase-Folded Transit Plot",
        description="Plot payload for transit-phase visualization.",
        category="plot",
    ),
    FieldKey.PLOT_FULL_LC: FieldSpec(
        path=FieldKey.PLOT_FULL_LC.value,
        display_name="Full Light Curve Plot",
        description="Plot payload for full time-series view.",
        category="plot",
    ),
}
