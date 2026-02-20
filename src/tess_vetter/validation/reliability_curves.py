"""Reliability curve computations (metrics-only).

These utilities summarize negative-control (or labeled) runs into binned
false-alarm-rate curves and simple threshold recommendations.

Design notes:
- Open-safe: no curated baselines or mission-specific operating points.
- Host apps can serialize curves/baselines however they choose.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_reliability_curves(
    results: list[dict[str, Any]],
    score_key: str = "score",
    *,
    n_bins: int = 20,
    score_range: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Compute binned false-alarm-rate curves from results.

    Each result must include:
    - `score_key` (default "score")
    - "detected" (bool)
    """
    if not results:
        return {}

    scores: list[float] = []
    detected: list[bool] = []
    for r in results:
        if score_key in r and "detected" in r:
            scores.append(float(r[score_key]))
            detected.append(bool(r["detected"]))

    if not scores:
        return {}

    scores_arr = np.asarray(scores, dtype=np.float64)
    detected_arr = np.asarray(detected, dtype=bool)

    if score_range is None:
        min_score = float(np.floor(np.min(scores_arr)))
        max_score = float(np.ceil(np.max(scores_arr))) + 0.01
    else:
        min_score, max_score = score_range

    bin_edges = np.linspace(min_score, max_score, int(n_bins) + 1)

    false_alarm_rates: dict[str, float] = {}
    for i in range(int(n_bins)):
        low = float(bin_edges[i])
        high = float(bin_edges[i + 1])
        bin_key = f"{low:.1f}-{high:.1f}"

        mask = (scores_arr >= low) & (scores_arr < high)
        n_in_bin = int(np.sum(mask))
        rate = float(np.sum(detected_arr[mask]) / n_in_bin) if n_in_bin > 0 else 0.0
        false_alarm_rates[bin_key] = float(rate)

    return false_alarm_rates


def compute_conditional_rates(
    results: list[dict[str, Any]],
    condition_key: str,
    score_key: str = "score",
    *,
    n_bins: int = 10,
) -> dict[str, dict[str, float]]:
    """Compute reliability curves conditioned on a categorical variable."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        if condition_key in r:
            cond_value = str(r[condition_key])
            groups.setdefault(cond_value, []).append(r)

    return {
        cond_value: compute_reliability_curves(group_results, score_key, n_bins=n_bins)
        for cond_value, group_results in groups.items()
    }


def recommend_thresholds(
    false_alarm_rates: dict[str, float],
    target_far: float = 0.01,
) -> dict[str, float]:
    """Recommend a score threshold achieving `target_far` (best-effort)."""
    thresholds: dict[str, float] = {}

    for bin_str, rate in sorted(false_alarm_rates.items(), reverse=True):
        try:
            high_str = bin_str.split("-")[1]
            high = float(high_str)
        except Exception:
            continue

        if float(rate) <= float(target_far):
            thresholds["score_threshold"] = float(high)
            thresholds["target_far"] = float(target_far)
            thresholds["achieved_far"] = float(rate)
            break

    return thresholds


__all__ = [
    "compute_conditional_rates",
    "compute_reliability_curves",
    "recommend_thresholds",
]
