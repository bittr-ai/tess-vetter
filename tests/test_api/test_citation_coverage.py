from __future__ import annotations

from collections.abc import Callable

import pytest

from tess_vetter.api.catalog import vet_catalog
from tess_vetter.api.exovetter import vet_exovetter
from tess_vetter.api.lc_only import vet_lc_only
from tess_vetter.api.pixel import vet_pixel
from tess_vetter.api.pixel_localize import (
    localize_transit_host_multi_sector,
    localize_transit_host_single_sector,
    localize_transit_host_single_sector_with_baseline_check,
)
from tess_vetter.api.recovery import prepare_recovery_inputs, stack_transits
from tess_vetter.api.references import get_function_references
from tess_vetter.api.ttv_track_search import (
    run_ttv_track_search,
    run_ttv_track_search_for_candidate,
)
from tess_vetter.api.vet import vet_many


def _ref_ids(refs: list[object]) -> set[str]:
    ids: set[str] = set()
    for r in refs:
        if hasattr(r, "ref") and hasattr(r.ref, "id"):  # type: ignore[attr-defined]
            ids.add(str(r.ref.id))  # type: ignore[attr-defined]
            continue
        if hasattr(r, "id"):
            ids.add(str(r.id))
            continue
        if hasattr(r, "to_dict"):
            d = r.to_dict()
            if isinstance(d, dict) and d.get("id"):
                ids.add(str(d["id"]))
            continue
        if isinstance(r, dict) and r.get("id"):
            ids.add(str(r["id"]))
    return ids


@pytest.mark.parametrize(
    ("fn", "expected"),
    [
        (vet_many, {"coughlin_2016", "thompson_2018", "guerrero_2021"}),
        (vet_lc_only, {"coughlin_2016", "thompson_2018"}),
        (vet_pixel, {"bryson_2013", "twicken_2018"}),
        (vet_catalog, {"prsa_2022", "guerrero_2021"}),
        (vet_exovetter, {"thompson_2018", "coughlin_2016"}),
        (
            localize_transit_host_single_sector,
            {
                "bryson_2013",
                "twicken_2018",
                "bryson_2010",
                "greisen_calabretta_2002",
                "calabretta_greisen_2002",
            },
        ),
        (localize_transit_host_single_sector_with_baseline_check, {"bryson_2013", "twicken_2018"}),
        (localize_transit_host_multi_sector, {"bryson_2013", "twicken_2018", "bryson_2010"}),
        (prepare_recovery_inputs, {"hippke_heller_2019_tls", "kovacs_2002"}),
        (stack_transits, {"hippke_heller_2019_tls", "kovacs_2002"}),
        (run_ttv_track_search, {"holman_murray_2005", "agol_2005", "steffen_agol_2006"}),
        (
            run_ttv_track_search_for_candidate,
            {"holman_murray_2005", "agol_2005", "steffen_agol_2006"},
        ),
    ],
)
def test_orchestrators_have_citations(fn: Callable[..., object], expected: set[str]) -> None:
    refs = get_function_references(fn)
    ids = _ref_ids(refs)
    assert ids, f"{fn.__module__}.{fn.__name__} should expose citations"
    assert expected.issubset(ids), (
        f"missing refs for {fn.__module__}.{fn.__name__}: {expected - ids}"
    )
