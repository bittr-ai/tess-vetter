"""Sector consistency diagnostics for the public API.

Re-exports metrics-only computations from
`tess_vetter.validation.sector_consistency`.
"""

from __future__ import annotations

from tess_vetter.validation.sector_consistency import (  # noqa: F401
    ConsistencyClass,
    SectorMeasurement,
    compute_sector_consistency,
)

__all__ = [
    "ConsistencyClass",
    "SectorMeasurement",
    "compute_sector_consistency",
]
