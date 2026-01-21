"""Ghost/scattered-light aggregation across sectors."""

from math import isfinite
from statistics import median

from .contracts import GhostSectorInput, GhostSummary


def build_ghost_summary(sectors: list[GhostSectorInput] | None) -> GhostSummary:
    """
    Aggregate ghost/scattered-light features across sectors.

    IMPORTANT: Filters out non-finite floats (NaN/inf) before aggregation
    to prevent poisoning medians/maxes.
    """
    if not sectors:
        return GhostSummary()

    ghost_scores: list[float] = []
    scatter_risks: list[float] = []
    sign_flags: list[bool] = []

    for row in sectors:
        gs = row.get("ghost_like_score_adjusted")
        if gs is not None and isfinite(gs):
            ghost_scores.append(gs)
        sr = row.get("scattered_light_risk")
        if sr is not None and isfinite(sr):
            scatter_risks.append(sr)
        sc = row.get("aperture_sign_consistent")
        if isinstance(sc, bool):
            sign_flags.append(sc)

    result: GhostSummary = {}

    if ghost_scores:
        result["ghost_like_score_adjusted_median"] = median(ghost_scores)
        result["ghost_like_score_adjusted_max"] = max(ghost_scores)

    if scatter_risks:
        result["scattered_light_risk_median"] = median(scatter_risks)
        result["scattered_light_risk_max"] = max(scatter_risks)

    if sign_flags:
        result["aperture_sign_consistent_all"] = all(sign_flags)
        result["aperture_sign_consistent_any_false"] = any(not f for f in sign_flags)

    return result
