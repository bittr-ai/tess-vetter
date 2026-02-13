from __future__ import annotations

from bittr_tess_vetter.api.primitives import PRIMITIVES_CATALOG, list_primitives


def test_list_primitives_filters_unimplemented_by_default() -> None:
    prims = list_primitives()
    assert isinstance(prims, dict)
    assert "astro.fold_transit" in prims
    assert "astro.detect_transit" in prims
    assert "astro.measure_depth" in prims
    assert "astro.transit_mask" in prims
    assert "astro.normalize" not in prims
    assert all(info.implemented for info in prims.values())
    assert prims["astro.fold_transit"].status == "available"


def test_list_primitives_can_include_unimplemented() -> None:
    prims = list_primitives(include_unimplemented=True)
    assert "astro.fold_transit" in prims
    assert "astro.normalize" in prims
    assert prims["astro.fold_transit"].implemented is True
    assert prims["astro.fold_transit"].status == "available"
    assert prims["astro.normalize"].implemented is False
    assert prims["astro.normalize"].status == "planned"
    assert len(prims) == len(PRIMITIVES_CATALOG)
