"""Raw evidence packet types for feature extraction.

These are btv-internal contracts used between the pipeline and feature builder.
They are intentionally JSON-serializable and streaming-friendly.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class SkipBlock(TypedDict, total=False):
    """Explicit skip block for optional subsystems.

    Avoids ambiguous `null`/missing values in enriched JSONL rows.
    """

    skipped: Literal[True]
    reason: str
    details: dict[str, Any] | None
    error_class: str | None
    error: str | None


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
    pixel_host_hypotheses : dict[str, Any] | SkipBlock
        Pixel-level host hypothesis testing results.
    localization : dict[str, Any] | SkipBlock
        Centroid and difference image localization results.
    sector_quality_report : dict[str, Any] | SkipBlock
        Per-sector data quality metrics.
    candidate_evidence : dict[str, Any] | SkipBlock
        Aggregated candidate-level evidence summary.
    ephemeris_specificity : dict[str, Any] | SkipBlock
        Light-curve-only ephemeris-specificity diagnostics (smooth score, phase-shift null, etc.).
    alias_diagnostics : dict[str, Any] | SkipBlock
        Light-curve-only alias/harmonic diagnostics (classification + supporting scores).
    systematics_proxy : dict[str, Any] | SkipBlock
        Light-curve-only systematics proxy block (cheap false-alarm heuristics).
    lc_stats : dict[str, Any]
        Light curve summary stats (cadence, n_valid, downsampling, etc.).
    provenance : dict[str, Any]
        Pipeline provenance: versions, timestamps, input files.
    """

    target: dict[str, Any]
    ephemeris: dict[str, Any]
    depth_ppm: dict[str, Any]
    check_results: list[dict[str, Any]]
    pixel_host_hypotheses: dict[str, Any] | SkipBlock
    localization: dict[str, Any] | SkipBlock
    sector_quality_report: dict[str, Any] | SkipBlock
    candidate_evidence: dict[str, Any] | SkipBlock
    ephemeris_specificity: dict[str, Any] | SkipBlock
    alias_diagnostics: dict[str, Any] | SkipBlock
    systematics_proxy: dict[str, Any] | SkipBlock
    lc_stats: dict[str, Any]
    provenance: dict[str, Any]


def make_skip_block(
    reason: str,
    *,
    details: dict[str, Any] | None = None,
    error_class: str | None = None,
    error: str | None = None,
) -> SkipBlock:
    return {
        "skipped": True,
        "reason": str(reason),
        "details": details,
        "error_class": error_class,
        "error": error,
    }


def is_skip_block(obj: object) -> bool:
    return isinstance(obj, dict) and obj.get("skipped") is True
