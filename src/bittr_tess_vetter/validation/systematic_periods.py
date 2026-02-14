"""Systematic-period proximity diagnostics for candidate periods."""

from __future__ import annotations

from typing import Sequence

import numpy as np

_KNOWN_SYSTEMATIC_PERIODS: tuple[tuple[str, float], ...] = (
    ("tess_momentum_dump", 3.0),
    ("tess_orbital", 13.7),
    ("tess_orbital_harmonic_2x", 27.4),
)
_KNOWN_SYSTEMATIC_PERIODS_DAYS: tuple[float, ...] = (3.0, 13.7, 27.4)
_DEFAULT_THRESHOLD_FRACTION = 0.05


def compute_systematic_period_proximity(
    *,
    period_days: float,
    threshold_fraction: float = _DEFAULT_THRESHOLD_FRACTION,
    systematic_periods_days: Sequence[float] = _KNOWN_SYSTEMATIC_PERIODS_DAYS,
    systematic_period_names: Sequence[str] | None = None,
) -> dict[str, float | bool | str]:
    """Return proximity of a candidate period to known TESS systematic periods."""
    period = float(period_days)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period_days must be a positive finite float")

    threshold = float(threshold_fraction)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold_fraction must be a positive finite float")

    periods = np.asarray([float(x) for x in systematic_periods_days], dtype=np.float64)
    if periods.size == 0:
        raise ValueError("systematic_periods_days must contain at least one period")
    if not np.all(np.isfinite(periods)) or np.any(periods <= 0.0):
        raise ValueError("systematic_periods_days must contain only positive finite periods")
    if systematic_period_names is None:
        known_days = np.asarray([p for _, p in _KNOWN_SYSTEMATIC_PERIODS], dtype=np.float64)
        known_names = [n for n, _ in _KNOWN_SYSTEMATIC_PERIODS]
        names = [
            known_names[int(np.argmin(np.abs(known_days - p)))] if np.any(np.isclose(known_days, p, rtol=0.0, atol=1e-9)) else f"systematic_{i + 1}"
            for i, p in enumerate(periods)
        ]
    else:
        names = [str(x) for x in systematic_period_names]
        if len(names) != periods.size:
            raise ValueError("systematic_period_names must have same length as systematic_periods_days")
        if any(not name for name in names):
            raise ValueError("systematic_period_names must contain non-empty names")

    abs_dist = np.abs(periods - period)
    idx = int(np.argmin(abs_dist))
    nearest = float(periods[idx])
    nearest_name = str(names[idx])
    frac_dist = float(abs_dist[idx] / nearest)

    return {
        "nearest_systematic_period": nearest,
        "systematic_period_name": nearest_name,
        "nearest_systematic_days": nearest,
        "fractional_distance": frac_dist,
        "within_threshold": bool(frac_dist <= threshold),
        "threshold_fraction": threshold,
    }


__all__ = ["compute_systematic_period_proximity"]
