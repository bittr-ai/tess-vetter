from __future__ import annotations

from bittr_tess_vetter.utils.caps import cap_neighbors, cap_plots, cap_top_k


def test_caps_do_not_truncate_when_under_limit(caplog) -> None:
    items = [1, 2, 3]
    out = cap_top_k(items, max_items=10, context="test")
    assert out is items
    assert caplog.records == []


def test_caps_truncate_and_log_warning(caplog) -> None:
    items = list(range(100))
    with caplog.at_level("WARNING"):
        out = cap_neighbors(items, max_items=5, context="TIC 1")
    assert out == list(range(5))
    assert any("Truncated" in r.message for r in caplog.records)


def test_cap_plots_uses_first_n() -> None:
    items = [{"i": i} for i in range(10)]
    out = cap_plots(items, max_items=3)
    assert out == [{"i": 0}, {"i": 1}, {"i": 2}]
