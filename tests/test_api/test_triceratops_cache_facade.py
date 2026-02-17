from __future__ import annotations


def test_triceratops_cache_facade_exports() -> None:
    from bittr_tess_vetter.api import triceratops_cache

    assert callable(triceratops_cache.prefetch_trilegal_csv)
    assert callable(triceratops_cache.load_cached_triceratops_target)
    assert callable(triceratops_cache.save_cached_triceratops_target)
    assert callable(triceratops_cache.stage_triceratops_runtime_artifacts)
    assert callable(triceratops_cache.estimate_transit_duration)
