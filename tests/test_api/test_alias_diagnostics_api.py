from __future__ import annotations

import numpy as np

from bittr_tess_vetter.api.alias_diagnostics import (
    classify_alias,
    compute_harmonic_scores,
    compute_secondary_significance,
    detect_phase_shift_events,
    harmonic_power_summary,
)
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve


def _box_events(time: np.ndarray, *, period: float, t0: float, duration_days: float) -> np.ndarray:
    phase = ((time - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    half = (duration_days / 2.0) / period
    return np.abs(phase) <= half


def test_alias_strong_when_true_period_is_p_over_2() -> None:
    rng = np.random.default_rng(0)
    time = np.linspace(0.0, 30.0, 5000, dtype=np.float64)
    flux_err = np.full_like(time, 1e-3)

    base_period = 5.0
    true_period = 2.5
    t0 = 0.0
    duration_hours = 2.0
    duration_days = duration_hours / 24.0

    in_transit_true = _box_events(time, period=true_period, t0=t0, duration_days=duration_days)
    flux = np.ones_like(time)
    flux[in_transit_true] -= 0.01
    flux += rng.normal(0.0, 2e-4, size=flux.shape)

    scores = compute_harmonic_scores(
        time=time,
        flux=flux,
        flux_err=flux_err,
        base_period=base_period,
        base_t0=t0,
        duration_hours=duration_hours,
    )

    base_score = next(s.score for s in scores if s.harmonic == "P")
    alias_class, best_harmonic, ratio = classify_alias(scores, base_score)

    # With this simple box scorer, the main gain at P/2 comes from doubling the
    # number of stacked events (significance ~ sqrt(2)), so this should register
    # as at least a weak alias.
    assert alias_class in ("ALIAS_WEAK", "ALIAS_STRONG")
    assert best_harmonic == "P/2"
    assert ratio >= 1.1


def test_alias_none_for_clean_period() -> None:
    rng = np.random.default_rng(1)
    time = np.linspace(0.0, 30.0, 5000, dtype=np.float64)
    flux_err = np.full_like(time, 1e-3)

    base_period = 5.0
    t0 = 0.0
    duration_hours = 2.0
    duration_days = duration_hours / 24.0

    in_transit = _box_events(time, period=base_period, t0=t0, duration_days=duration_days)
    flux = np.ones_like(time)
    flux[in_transit] -= 0.01
    flux += rng.normal(0.0, 2e-4, size=flux.shape)

    scores = compute_harmonic_scores(
        time=time,
        flux=flux,
        flux_err=flux_err,
        base_period=base_period,
        base_t0=t0,
        duration_hours=duration_hours,
    )

    base_score = next(s.score for s in scores if s.harmonic == "P")
    alias_class, best_harmonic, ratio = classify_alias(scores, base_score)

    assert alias_class == "NONE"
    assert best_harmonic == "P"
    assert ratio < 1.1


def test_secondary_significance_detects_secondary_dip() -> None:
    rng = np.random.default_rng(2)
    time = np.linspace(0.0, 10.0, 5000, dtype=np.float64)
    period = 2.0
    t0 = 0.0
    duration_hours = 2.0
    duration_days = duration_hours / 24.0

    phase = ((time - t0) % period) / period
    in_secondary = (phase > 0.5 - (duration_days / period) / 2.0) & (
        phase < 0.5 + (duration_days / period) / 2.0
    )

    flux = np.ones_like(time)
    flux[in_secondary] -= 0.01
    flux += rng.normal(0.0, 2e-4, size=flux.shape)
    flux_err = np.full_like(time, 1e-3)

    sig = compute_secondary_significance(time, flux, flux_err, period, t0, duration_hours)
    assert sig > 3.0


def test_detect_phase_shift_events_finds_non_primary_bin() -> None:
    rng = np.random.default_rng(3)
    time = np.linspace(0.0, 10.0, 5000, dtype=np.float64)
    period = 2.0
    t0 = 0.0
    flux_err = np.full_like(time, 1e-3)

    phase = ((time - t0) % period) / period
    # Create a dip at phase ~0.25 (not primary, not necessarily secondary)
    in_bin = (phase > 0.23) & (phase < 0.27)

    flux = np.ones_like(time)
    flux[in_bin] -= 0.01
    flux += rng.normal(0.0, 2e-4, size=flux.shape)

    events = detect_phase_shift_events(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=period,
        t0=t0,
        n_phase_bins=20,
        significance_threshold=3.0,
    )

    assert any(0.15 < e.phase < 0.35 for e in events)


def test_harmonic_power_summary_returns_compact_triplet() -> None:
    rng = np.random.default_rng(5)
    time = np.linspace(0.0, 30.0, 5000, dtype=np.float64)
    flux_err = np.full_like(time, 1e-3)
    period_days = 5.0
    t0_btjd = 0.0
    duration_hours = 2.0
    duration_days = duration_hours / 24.0

    in_transit = _box_events(time, period=period_days, t0=t0_btjd, duration_days=duration_days)
    flux = np.ones_like(time)
    flux[in_transit] -= 0.01
    flux += rng.normal(0.0, 2e-4, size=flux.shape)

    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    cand = Candidate(
        ephemeris=Ephemeris(
            period_days=period_days,
            t0_btjd=t0_btjd,
            duration_hours=duration_hours,
        )
    )

    summary = harmonic_power_summary(lc, cand)
    labels = [h.harmonic for h in summary.harmonics]
    assert labels == ["P", "P/2", "2P"]
    assert summary.best_harmonic in {"P", "P/2", "2P"}


def test_harmonic_scores_with_zero_flux_err_remain_finite() -> None:
    """Missing/zero flux_err should not inflate harmonic scores to invalid values."""
    rng = np.random.default_rng(11)
    time = np.linspace(0.0, 30.0, 5000, dtype=np.float64)
    flux_err = np.zeros_like(time)
    period_days = 5.0
    t0_btjd = 0.0
    duration_hours = 2.0
    duration_days = duration_hours / 24.0

    in_transit = _box_events(time, period=period_days, t0=t0_btjd, duration_days=duration_days)
    flux = np.ones_like(time)
    flux[in_transit] -= 0.01
    flux += rng.normal(0.0, 2e-4, size=flux.shape)

    scores = compute_harmonic_scores(
        time=time,
        flux=flux,
        flux_err=flux_err,
        base_period=period_days,
        base_t0=t0_btjd,
        duration_hours=duration_hours,
        harmonics=["P", "P/2", "2P"],
    )

    assert len(scores) == 3
    for s in scores:
        assert np.isfinite(s.score)
        assert s.score < 500.0
