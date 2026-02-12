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

