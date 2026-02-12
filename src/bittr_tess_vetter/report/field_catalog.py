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
