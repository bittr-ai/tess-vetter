"""Shared default variant axes for ``btv detrend-grid`` workflows."""

from __future__ import annotations

DEFAULT_DOWNSAMPLE_LEVELS: tuple[int, ...] = (1, 2, 5)
DEFAULT_OUTLIER_POLICIES: tuple[str, ...] = ("none", "sigma_clip_4")
DEFAULT_DETRENDERS: tuple[str, ...] = (
    "none",
    "running_median_0.5d",
    "transit_masked_bin_median",
)


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


__all__ = [
    "DEFAULT_DOWNSAMPLE_LEVELS",
    "DEFAULT_OUTLIER_POLICIES",
    "DEFAULT_DETRENDERS",
    "resolve_detrend_grid_axes",
]
