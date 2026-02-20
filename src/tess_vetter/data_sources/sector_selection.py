"""Sector selection and gating (domain logic).

This module chooses which sectors to use for enrichment. The goal is to be:
- deterministic,
- explainable (records reasons for exclusions),
- conservative by default.

At this stage we implement a minimal scaffold that can be expanded to match
the richer gating used by downstream orchestration layers.
"""

from __future__ import annotations

from tess_vetter.data_sources.contracts import SectorSelection


def select_sectors(
    *,
    available_sectors: list[int],
    requested_sectors: list[int] | None = None,
    allow_20s: bool = False,
    search_results: list[object] | None = None,
) -> SectorSelection:
    """Select sectors from a candidate's available sectors.

    Current behavior (minimal scaffold):
    - If requested_sectors provided: intersect with available_sectors and
      mark missing ones as excluded.
    - Else: select all available_sectors.

    Cadence gating (when `search_results` provided):
    - Prefer 120s cadence by default.
    - If 120s not present in a sector, allow 20s only if allow_20s=True.
    - Exclude sectors with no allowed cadence product.
    """
    available_sorted = sorted({int(s) for s in available_sectors})
    excluded: dict[int, str] = {}

    # Cadence availability map (sector -> set of exptimes)
    sector_exptimes: dict[int, set[int]] = {}
    if search_results is not None:
        for r in search_results:
            try:
                sector = int(r.sector)  # type: ignore[attr-defined]
                exptime = int(round(float(r.exptime)))  # type: ignore[attr-defined]
            except Exception:
                continue
            sector_exptimes.setdefault(sector, set()).add(exptime)

        # Exclude sectors with no allowed cadence.
        for sector in list(available_sorted):
            exps = sector_exptimes.get(sector)
            if not exps:
                excluded[sector] = "cadence_unavailable"
                continue
            if 120 in exps:
                continue
            if 20 in exps and allow_20s:
                continue
            excluded[sector] = "cadence_not_allowed" if 20 in exps else "cadence_unavailable"

    if requested_sectors is None:
        selected = [s for s in available_sorted if s not in excluded]
        return SectorSelection(
            available_sectors=available_sorted,
            selected_sectors=selected,
            excluded_sectors=excluded,
            cadence_seconds=None,
        )

    req_sorted = sorted({int(s) for s in requested_sectors})
    selected = [s for s in req_sorted if s in available_sorted]
    for s in req_sorted:
        if s not in available_sorted:
            excluded[s] = "not_available"

    # Apply cadence gating exclusions to requested selection.
    selected = [s for s in selected if s not in excluded]

    return SectorSelection(
        available_sectors=available_sorted,
        selected_sectors=selected,
        excluded_sectors=excluded,
        cadence_seconds=None,
    )


__all__ = ["select_sectors"]
