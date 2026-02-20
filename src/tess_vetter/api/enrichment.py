"""Public API for candidate enrichment (checks + aggregates + provenance).

This module is a stable import surface for downstream consumers that want to:
- enrich a single candidate, or
- batch-enrich a worklist to JSONL.

Downstream packages should prefer importing from `tess_vetter.api.enrichment`
instead of deep-importing `tess_vetter.pipeline` or CLI internals.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict

from tess_vetter.features import EnrichedRow, FeatureConfig, RawEvidencePacket
from tess_vetter.pipeline import (
    EnrichmentSummary,
    enrich_candidate,
    enrich_worklist,
    make_candidate_key,
)


class WorklistRow(TypedDict, total=False):
    """Minimal worklist row contract for enrichment."""

    tic_id: int
    toi: str | None
    period_days: float
    t0_btjd: float
    duration_hours: float
    depth_ppm: float


def candidate_key_from_row(row: Mapping[str, Any]) -> str:
    """Generate a stable candidate key from a worklist/enriched row."""
    return make_candidate_key(
        int(row["tic_id"]),
        float(row["period_days"]),
        float(row["t0_btjd"]),
    )


def validate_enriched_row(row: Mapping[str, Any], *, strict: bool = True) -> None:
    """Validate an enriched row has the required shape.

    This is intentionally lightweight (no pydantic dependency). It is meant to
    catch obvious schema breakages early in bulk runs.
    """
    required = [
        "tic_id",
        "period_days",
        "t0_btjd",
        "duration_hours",
        "candidate_key",
        "status",
        "pipeline_version",
        "feature_schema_version",
        "feature_config",
        "inputs_summary",
        "missing_feature_families",
        "item_wall_ms",
    ]
    missing = [k for k in required if k not in row]
    if missing:
        raise ValueError(f"Missing required enriched-row fields: {missing}")

    if strict:
        if row["candidate_key"] != candidate_key_from_row(row):
            raise ValueError("candidate_key does not match (tic_id, period_days, t0_btjd)")
        if row["status"] not in ("OK", "ERROR"):
            raise ValueError(f"Invalid status: {row['status']!r}")


def normalize_feature_config(config: FeatureConfig | Mapping[str, Any]) -> FeatureConfig:
    """Accept either a FeatureConfig or a dict-like config and return FeatureConfig."""
    if isinstance(config, FeatureConfig):
        return config
    return FeatureConfig(**dict(config))


__all__ = [
    # Types
    "WorklistRow",
    "EnrichedRow",
    "RawEvidencePacket",
    "FeatureConfig",
    "EnrichmentSummary",
    # Core enrichment
    "enrich_candidate",
    "enrich_worklist",
    # Keys + validation
    "make_candidate_key",
    "candidate_key_from_row",
    "validate_enriched_row",
    # Helpers
    "normalize_feature_config",
]

