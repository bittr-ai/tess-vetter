from __future__ import annotations

from tess_vetter.data_sources.sector_selection import select_sectors


def test_select_sectors_default_selects_all() -> None:
    sel = select_sectors(available_sectors=[3, 1, 2])
    assert sel.selected_sectors == [1, 2, 3]
    assert sel.excluded_sectors == {}


def test_select_sectors_requested_intersects_and_explains_exclusions() -> None:
    sel = select_sectors(available_sectors=[1, 2], requested_sectors=[2, 3])
    assert sel.selected_sectors == [2]
    assert sel.excluded_sectors == {3: "not_available"}


def test_select_sectors_cadence_gates_without_allow_20s() -> None:
    class R:
        def __init__(self, sector: int, exptime: float):
            self.sector = sector
            self.exptime = exptime

    sel = select_sectors(
        available_sectors=[1, 2],
        requested_sectors=None,
        allow_20s=False,
        search_results=[R(1, 120.0), R(2, 20.0)],
    )
    assert sel.selected_sectors == [1]
    assert sel.excluded_sectors[2] == "cadence_not_allowed"


def test_select_sectors_cadence_allows_20s_when_enabled() -> None:
    class R:
        def __init__(self, sector: int, exptime: float):
            self.sector = sector
            self.exptime = exptime

    sel = select_sectors(
        available_sectors=[1, 2],
        requested_sectors=None,
        allow_20s=True,
        search_results=[R(1, 120.0), R(2, 20.0)],
    )
    assert sel.selected_sectors == [1, 2]
