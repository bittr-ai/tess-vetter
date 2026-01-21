"""Sector selection and gating (domain logic).

This module chooses which sectors to use for enrichment. The goal is to be:
- deterministic,
- explainable (records reasons for exclusions),
- conservative by default.

At this stage we implement a minimal scaffold that can be expanded to match
the richer gating used by downstream orchestration layers.
"""

from __future__ import annotations

from bittr_tess_vetter.data_sources.contracts import SectorSelection


def select_sectors(
    *,
    available_sectors: list[int],
    requested_sectors: list[int] | None = None,
) -> SectorSelection:
    """Select sectors from a candidate's available sectors.

    Current behavior (minimal scaffold):
    - If requested_sectors provided: intersect with available_sectors and
      mark missing ones as excluded.
    - Else: select all available_sectors.
    """
    available_sorted = sorted({int(s) for s in available_sectors})
    excluded: dict[int, str] = {}

    if requested_sectors is None:
        return SectorSelection(
            available_sectors=available_sorted,
            selected_sectors=available_sorted,
            excluded_sectors={},
            cadence_seconds=None,
        )

    req_sorted = sorted({int(s) for s in requested_sectors})
    selected = [s for s in req_sorted if s in available_sorted]
    for s in req_sorted:
        if s not in available_sorted:
            excluded[s] = "not_available"

    return SectorSelection(
        available_sectors=available_sorted,
        selected_sectors=selected,
        excluded_sectors=excluded,
        cadence_seconds=None,
    )


__all__ = ["select_sectors"]

