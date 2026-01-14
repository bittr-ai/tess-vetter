from __future__ import annotations

from bittr_tess_vetter.api.reliability_curves import (
    compute_conditional_rates,
    compute_reliability_curves,
    recommend_thresholds,
)


def test_compute_reliability_curves_bins_and_rates() -> None:
    results = [
        {"score": 6.5, "detected": False},
        {"score": 6.6, "detected": False},
        {"score": 7.2, "detected": True},
        {"score": 7.3, "detected": True},
    ]
    curves = compute_reliability_curves(results, n_bins=2, score_range=(6.0, 8.0))

    # bins: 6.0-7.0 (2 false), 7.0-8.0 (2 true)
    assert curves["6.0-7.0"] == 0.0
    assert curves["7.0-8.0"] == 1.0


def test_compute_conditional_rates_groups() -> None:
    results = [
        {"score": 6.5, "detected": False, "group": "A"},
        {"score": 7.5, "detected": True, "group": "A"},
        {"score": 6.5, "detected": False, "group": "B"},
        {"score": 6.6, "detected": False, "group": "B"},
    ]
    grouped = compute_conditional_rates(results, "group", n_bins=2)
    assert set(grouped.keys()) == {"A", "B"}


def test_recommend_thresholds_picks_high_edge_when_far_met() -> None:
    far = {"7.0-8.0": 0.2, "8.0-9.0": 0.01, "9.0-10.0": 0.0}
    rec = recommend_thresholds(far, target_far=0.01)
    assert rec["score_threshold"] == 10.0
    assert rec["achieved_far"] == 0.0
