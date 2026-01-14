from __future__ import annotations

from bittr_tess_vetter.api.primitives import list_primitives


def test_list_primitives_filters_unimplemented_by_default() -> None:
    prims = list_primitives()
    assert isinstance(prims, dict)
    assert "astro.fold_transit" not in prims


def test_list_primitives_can_include_unimplemented() -> None:
    prims = list_primitives(include_unimplemented=True)
    assert "astro.fold_transit" in prims
