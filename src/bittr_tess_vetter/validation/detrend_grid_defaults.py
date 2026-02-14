"""Shared default variant axes for ``btv detrend-grid`` workflows."""

from __future__ import annotations

DEFAULT_DOWNSAMPLE_LEVELS: tuple[int, ...] = (1, 2, 5)
DEFAULT_OUTLIER_POLICIES: tuple[str, ...] = ("none", "sigma_clip_4")
DEFAULT_DETRENDERS: tuple[str, ...] = (
    "none",
    "running_median_0.5d",
    "transit_masked_bin_median",
)
DEFAULT_TRANSIT_MASKED_BIN_HOURS: tuple[float, ...] = (4.0, 6.0, 8.0)
DEFAULT_TRANSIT_MASKED_BUFFER_FACTORS: tuple[float, ...] = (1.5, 2.0, 3.0)
DEFAULT_TRANSIT_MASKED_SIGMA_CLIPS: tuple[float, ...] = (3.0, 5.0)


def resolve_detrend_grid_axes(
    *,
    downsample_levels: list[int] | None,
    outlier_policies: list[str] | None,
    detrenders: list[str] | None,
) -> tuple[list[int], list[str], list[str]]:
    """Resolve requested grid axes to concrete defaults when omitted."""
    effective_downsample = list(downsample_levels) if downsample_levels is not None else list(DEFAULT_DOWNSAMPLE_LEVELS)
    effective_outlier = list(outlier_policies) if outlier_policies is not None else list(DEFAULT_OUTLIER_POLICIES)
    effective_detrenders = list(detrenders) if detrenders is not None else list(DEFAULT_DETRENDERS)
    return effective_downsample, effective_outlier, effective_detrenders


def expanded_detrender_count(detrenders: list[str]) -> int:
    """Return expanded detrender count accounting for transit-masked sub-variants."""
    tm_count = (
        len(DEFAULT_TRANSIT_MASKED_BIN_HOURS)
        * len(DEFAULT_TRANSIT_MASKED_BUFFER_FACTORS)
        * len(DEFAULT_TRANSIT_MASKED_SIGMA_CLIPS)
    )
    total = 0
    for detrender in detrenders:
        if str(detrender) == "transit_masked_bin_median":
            total += tm_count
        else:
            total += 1
    return int(total)


__all__ = [
    "DEFAULT_DOWNSAMPLE_LEVELS",
    "DEFAULT_OUTLIER_POLICIES",
    "DEFAULT_DETRENDERS",
    "DEFAULT_TRANSIT_MASKED_BIN_HOURS",
    "DEFAULT_TRANSIT_MASKED_BUFFER_FACTORS",
    "DEFAULT_TRANSIT_MASKED_SIGMA_CLIPS",
    "expanded_detrender_count",
    "resolve_detrend_grid_axes",
]
