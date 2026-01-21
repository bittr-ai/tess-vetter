"""Domain contracts for candidate data acquisition.

These contracts describe what enrichment needs from data sources without
committing to a specific backend (MAST, local folders, future cache layers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bittr_tess_vetter.domain.lightcurve import LightCurveData


@dataclass(frozen=True)
class SectorLightCurve:
    sector: int
    cadence_seconds: float
    flux_type: Literal["pdcsap", "sap"]
    author: str | None
    lc: LightCurveData


@dataclass(frozen=True)
class SectorSelection:
    """Result of selecting sectors for a candidate.

    This is intended to be persisted into enriched provenance so downstream
    consumers can understand which data were used and why.
    """

    available_sectors: list[int]
    selected_sectors: list[int]
    excluded_sectors: dict[int, str]
    cadence_seconds: float | None


__all__ = ["SectorLightCurve", "SectorSelection"]

