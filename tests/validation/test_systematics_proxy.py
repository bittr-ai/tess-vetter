"""Tests for `bittr_tess_vetter.validation.systematics_proxy`."""

from __future__ import annotations

import numpy as np

from bittr_tess_vetter.validation.systematics_proxy import compute_systematics_proxy


def _inject_box_transit(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    depth_frac: float,
) -> np.ndarray:
    flux = np.array(flux, copy=True)
    duration_days = duration_hours / 24.0
    phase = ((time - t0_btjd) % period_days) / period_days
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    in_transit = np.abs(phase) <= (duration_days / 2.0) / period_days
    flux[in_transit] *= 1.0 - depth_frac
    return flux


def test_clean_transit_low_systematics_score() -> None:
    n = 3000
    time = np.linspace(0.0, 30.0, n, dtype=np.float64)
    rng = np.random.default_rng(123)
    flux = np.ones(n, dtype=np.float64) + rng.normal(0.0, 2e-4, n)

    period_days = 3.0
    t0_btjd = 1.5
    duration_hours = 2.0
    flux = _inject_box_transit(
        time,
        flux,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_frac=0.002,  # 2000 ppm
    )

    res = compute_systematics_proxy(
        time=time,
        flux=flux,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
    )
    assert res is not None
    assert 0.0 <= res.score <= 1.0
    # Transit edges should not dominate the step metric (computed on OOT).
    assert res.max_step_sigma is None or res.max_step_sigma < 8.0
    assert res.score < 0.5


def test_step_systematic_increases_score() -> None:
    n = 3000
    time = np.linspace(0.0, 30.0, n, dtype=np.float64)
    rng = np.random.default_rng(456)
    flux = np.ones(n, dtype=np.float64) + rng.normal(0.0, 2e-4, n)

    # Inject a step discontinuity unrelated to transits.
    step_idx = n // 2
    flux[step_idx:] += 0.01  # 1% step

    res = compute_systematics_proxy(
        time=time,
        flux=flux,
        period_days=3.0,
        t0_btjd=1.5,
        duration_hours=2.0,
    )
    assert res is not None
    assert res.max_step_sigma is not None
    assert res.max_step_sigma > 8.0
    assert res.score >= 0.3


def test_valid_mask_drops_nan_points() -> None:
    n = 2500
    time = np.linspace(0.0, 30.0, n, dtype=np.float64)
    rng = np.random.default_rng(789)
    flux = np.ones(n, dtype=np.float64) + rng.normal(0.0, 2e-4, n)

    period_days = 3.0
    t0_btjd = 1.5
    duration_hours = 2.0
    flux = _inject_box_transit(
        time,
        flux,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_frac=0.002,
    )

    base = compute_systematics_proxy(
        time=time,
        flux=flux,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
    )
    assert base is not None

    # Poison some points and use valid_mask to drop them.
    bad_idx = np.arange(0, 50)
    flux_bad = np.array(flux, copy=True)
    flux_bad[bad_idx] = np.nan
    valid_mask = np.ones(n, dtype=bool)
    valid_mask[bad_idx] = False

    masked = compute_systematics_proxy(
        time=time,
        flux=flux_bad,
        valid_mask=valid_mask,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
    )
    assert masked is not None
    assert np.isfinite(masked.score)
    assert abs(masked.score - base.score) < 0.25

