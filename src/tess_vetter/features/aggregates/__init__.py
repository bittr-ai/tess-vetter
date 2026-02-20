"""
Aggregation functions for computing higher-level features from check results.

This subpackage provides pure functions that operate on check outputs and return
aggregated features. Both tess-vetter and astro-arc-tess call these functions
to ensure consistent feature computation.
"""

from .contracts import (
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
)
from .coverage import (
    FAMILY_GHOST_RELIABILITY,
    FAMILY_HOST_PLAUSIBILITY,
    FAMILY_MODSHIFT,
    FAMILY_PIXEL_TIMESERIES,
    FAMILY_TPF_LOCALIZATION,
    compute_missing_families,
)
from .ghost import build_ghost_summary
from .host import build_host_plausibility_summary
from .localization import build_localization_summary
from .verdicts import normalize_verdict


def build_pixel_host_summary(pixel: PixelHostInput | None) -> PixelHostSummary:
    """Build pixel host summary from combined input."""
    if not pixel:
        return PixelHostSummary()

    result: PixelHostSummary = {}

    if (ambiguity := pixel.get("ambiguity")) is not None:
        result["pixel_host_ambiguity"] = ambiguity
    if (disagreement := pixel.get("disagreement_flag")) is not None:
        result["pixel_disagreement_flag"] = str(disagreement)
    if (flip_rate := pixel.get("flip_rate")) is not None:
        result["pixel_flip_rate"] = flip_rate

    ts = pixel.get("timeseries")
    if ts:
        if (verdict := ts.get("verdict")) is not None:
            result["pixel_timeseries_verdict"] = normalize_verdict(verdict)
        if (delta_chi2 := ts.get("delta_chi2")) is not None:
            result["pixel_timeseries_delta_chi2"] = delta_chi2
        if (best_source := ts.get("best_source_id")) is not None:
            result["pixel_timeseries_best_source_id"] = best_source
        if (n_windows := ts.get("n_windows")) is not None:
            result["pixel_timeseries_n_windows"] = n_windows
        if (agrees := ts.get("agrees_with_consensus")) is not None:
            result["pixel_timeseries_agrees_with_consensus"] = agrees

    return result


def build_aggregates(
    ghost_sectors: list[GhostSectorInput] | None = None,
    localization: LocalizationInput | None = None,
    v09: V09Metrics | None = None,
    pixel_host: PixelHostInput | None = None,
    host_plausibility: HostPlausibilityInput | None = None,
    presence_flags: CheckPresenceFlags | None = None,
) -> Aggregates:
    """
    Build complete aggregations from all available inputs.

    This is the main entry point for aggregation. Callers extract the minimal
    contract inputs from their internal structures and pass them here.
    """
    return Aggregates(
        ghost=build_ghost_summary(ghost_sectors),
        localization=build_localization_summary(localization, v09),
        host_plausibility=build_host_plausibility_summary(host_plausibility),
        pixel_host=build_pixel_host_summary(pixel_host),
        coverage=compute_missing_families(presence_flags or CheckPresenceFlags()),
    )


__all__ = [
    # Main entry point
    "build_aggregates",
    # Individual builders
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
