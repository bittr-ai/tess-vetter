"""Public API for aggregate feature builders.

This module provides a stable import surface for higher-level feature
aggregations (ghost reliability, localization summaries, host plausibility,
and coverage/missing-family tracking).

Downstream packages (e.g., astro-arc-tess) should prefer importing from here
instead of deep-importing `bittr_tess_vetter.features.aggregates` directly.
"""

from __future__ import annotations

from bittr_tess_vetter.features.aggregates import (
    FAMILY_GHOST_RELIABILITY,
    FAMILY_HOST_PLAUSIBILITY,
    FAMILY_MODSHIFT,
    FAMILY_PIXEL_TIMESERIES,
    FAMILY_TPF_LOCALIZATION,
    Aggregates,
    CheckPresenceFlags,
    CoverageSummary,
    GhostSectorInput,
    GhostSummary,
    HostPlausibilityInput,
    HostPlausibilitySummary,
    HostScenario,
    LocalizationInput,
    LocalizationSummary,
    PixelHostInput,
    PixelHostSummary,
    PixelTimeseriesInput,
    V09Metrics,
    Verdict,
    build_aggregates,
    build_ghost_summary,
    build_host_plausibility_summary,
    build_localization_summary,
    build_pixel_host_summary,
    compute_missing_families,
    normalize_verdict,
)

__all__ = [
    # Builders
    "build_aggregates",
    "build_ghost_summary",
    "build_localization_summary",
    "build_host_plausibility_summary",
    "build_pixel_host_summary",
    "compute_missing_families",
    # Verdict normalization
    "normalize_verdict",
    "Verdict",
    # Family constants
    "FAMILY_MODSHIFT",
    "FAMILY_TPF_LOCALIZATION",
    "FAMILY_PIXEL_TIMESERIES",
    "FAMILY_GHOST_RELIABILITY",
    "FAMILY_HOST_PLAUSIBILITY",
    # Contracts
    "Aggregates",
    "CheckPresenceFlags",
    "CoverageSummary",
    "GhostSectorInput",
    "GhostSummary",
    "HostPlausibilityInput",
    "HostPlausibilitySummary",
    "HostScenario",
    "LocalizationInput",
    "LocalizationSummary",
    "PixelHostInput",
    "PixelHostSummary",
    "PixelTimeseriesInput",
    "V09Metrics",
]

