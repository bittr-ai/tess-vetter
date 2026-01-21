from __future__ import annotations

from bittr_tess_vetter.data_sources.sector_selection import select_sectors


def test_select_sectors_default_selects_all() -> None:
    sel = select_sectors(available_sectors=[3, 1, 2])
    assert sel.selected_sectors == [1, 2, 3]
    assert sel.excluded_sectors == {}


def test_select_sectors_requested_intersects_and_explains_exclusions() -> None:
    sel = select_sectors(available_sectors=[1, 2], requested_sectors=[2, 3])
    assert sel.selected_sectors == [2]
    assert sel.excluded_sectors == {3: "not_available"}

