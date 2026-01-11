"""Curated facade API for `bittr-tess-vetter`.

This module is the recommended import surface for new clients. It is intentionally
small and stable.

Notes:
- This package remains metrics-only: no interpretation/policy/disposition is produced here.
- For broad compatibility exports, see `bittr_tess_vetter.api` (top-level aggregator).
"""

from __future__ import annotations

from typing import Any


__all__ = [
    # Core types
    "LightCurve",
    "Ephemeris",
    "Candidate",
    "TPFStamp",
    "VettingBundleResult",
    # Primary orchestration
    "vet",
    "vet_candidate",
    # Common analysis entrypoints
    "periodogram",
    "run_periodogram",
    "auto_periodogram",
    "fit_transit",
    "measure_transit_times",
    "analyze_ttvs",
    # Pixel-level host identification (WCS-aware)
    "localize",
    "localize_transit_source",
    "aperture_family_depth_curve",
    "compute_aperture_family_depth_curve",
]


_ALIASES: dict[str, str] = {
    "vet": "vet_candidate",
    "periodogram": "run_periodogram",
    "localize": "localize_transit_source",
    "aperture_family_depth_curve": "compute_aperture_family_depth_curve",
}


def __getattr__(name: str) -> Any:
    from bittr_tess_vetter import api as _api

    target = _ALIASES.get(name, name)
    return getattr(_api, target)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_ALIASES.keys()))

