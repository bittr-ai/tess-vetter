"""Periodogram API surface (host-facing).

This module provides a stable facade for periodogram operations so host
applications don't need to import from internal `compute.*` modules.
"""

from __future__ import annotations

from bittr_tess_vetter.compute.periodogram import (  # noqa: F401
    PerformancePreset,
    auto_periodogram,
    compute_bls_model,
    detect_sector_gaps,
    ls_periodogram,
    merge_candidates,
    refine_period,
    search_planets,
    split_by_sectors,
    tls_search,
    tls_search_per_sector,
)
from bittr_tess_vetter.domain.detection import PeriodogramPeak, PeriodogramResult  # noqa: F401

__all__ = [
    "PerformancePreset",
    "PeriodogramPeak",
    "PeriodogramResult",
    "auto_periodogram",
    "compute_bls_model",
    "detect_sector_gaps",
    "ls_periodogram",
    "merge_candidates",
    "refine_period",
    "search_planets",
    "split_by_sectors",
    "tls_search",
    "tls_search_per_sector",
]

