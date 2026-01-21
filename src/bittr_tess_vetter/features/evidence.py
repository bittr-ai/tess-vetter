"""Raw evidence packet types for feature extraction."""

from typing import Any, TypedDict


class RawEvidencePacket(TypedDict, total=False):
    """
    Raw outputs from vetting checks before feature extraction.

    This serves as the intermediate representation between the vetting
    pipeline outputs and the final ML features. All fields are optional
    (total=False) since different pipeline configurations may produce
    different subsets of evidence.

    The feature builder consumes this packet and extracts deterministic
    features based on the current FeatureConfig.

    Attributes
    ----------
    target : dict[str, Any]
        Target identification: tic_id, toi, ra_deg, dec_deg, etc.
    ephemeris : dict[str, Any]
        Transit ephemeris: period_days, t0_btjd, duration_hours, etc.
    depth_ppm : dict[str, Any]
        Depth measurements from various sources (BLS, transit fit, etc.).
    check_results : list[dict[str, Any]]
        List of CheckResult dicts from validation checks.
    pixel_host_hypotheses : dict[str, Any] | None
        Pixel-level host hypothesis testing results.
    localization : dict[str, Any] | None
        Centroid and difference image localization results.
    sector_quality_report : dict[str, Any] | None
        Per-sector data quality metrics.
    candidate_evidence : dict[str, Any] | None
        Aggregated candidate-level evidence summary.
    provenance : dict[str, Any]
        Pipeline provenance: versions, timestamps, input files.
    """

    target: dict[str, Any]
    ephemeris: dict[str, Any]
    depth_ppm: dict[str, Any]
    check_results: list[dict[str, Any]]
    pixel_host_hypotheses: dict[str, Any] | None
    localization: dict[str, Any] | None
    sector_quality_report: dict[str, Any] | None
    candidate_evidence: dict[str, Any] | None
    provenance: dict[str, Any]
