from __future__ import annotations

import numpy as np

from tess_vetter.domain.lightcurve import LightCurveData
from tess_vetter.validation.lc_checks import (
    OddEvenConfig,
    SecondaryEclipseConfig,
    check_odd_even_depth,
    check_secondary_eclipse,
)


def _make_lc_for_parity(*, odd_depth: float, even_depth: float) -> LightCurveData:
    period = 5.0
    t0 = 1.0
    duration_days = 2.0 / 24.0

    time = np.arange(0.0, 60.0, 0.02, dtype=np.float64)
    rng = np.random.default_rng(123)
    flux_err = np.ones_like(time, dtype=np.float64) * 2e-4
    flux = np.ones_like(time, dtype=np.float64) + rng.normal(0.0, flux_err, size=time.shape)
    quality = np.zeros_like(time, dtype=np.int32)
    valid_mask = np.ones_like(time, dtype=bool)

    # Alternate depth by epoch parity.
    for k in range(0, 20):
        center = t0 + k * period
        in_tr = np.abs(time - center) <= (duration_days / 2.0)
        depth = odd_depth if (k % 2 == 1) else even_depth
        flux[in_tr] -= depth

    lc = LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality,
        valid_mask=valid_mask,
        tic_id=1,
        sector=1,
        cadence_seconds=1800.0,
    )
    return lc


def test_odd_even_detects_large_parity_difference() -> None:
    period = 5.0
    t0 = 1.0
    duration_hours = 2.0
    lc = _make_lc_for_parity(odd_depth=0.0012, even_depth=0.0008)
    cfg = OddEvenConfig(
        min_transits_per_parity=3,
        min_points_in_transit_per_parity=20,
        baseline_window_mult=6.0,
        min_points_in_transit_per_epoch=3,
    )
    r = check_odd_even_depth(
        lightcurve=lc,
        period=period,
        t0=t0,
        duration_hours=duration_hours,
        config=cfg,
    )
    assert r.id == "V01"
    assert r.details.get("delta_sigma") is not None
    assert float(r.details["delta_sigma"]) > 3.0
    assert float(r.details.get("rel_diff", 0.0)) > 0.1


def test_secondary_eclipse_significance_increases_with_injected_secondary() -> None:
    period = 5.0
    t0 = 1.0
    duration_days = 2.0 / 24.0

    time = np.arange(0.0, 60.0, 0.02, dtype=np.float64)
    flux_err = np.ones_like(time, dtype=np.float64) * 2e-4
    rng = np.random.default_rng(456)
    flux0 = np.ones_like(time, dtype=np.float64) + rng.normal(0.0, flux_err, size=time.shape)
    flux1 = np.ones_like(time, dtype=np.float64) + rng.normal(0.0, flux_err, size=time.shape)
    quality = np.zeros_like(time, dtype=np.int32)
    valid_mask = np.ones_like(time, dtype=bool)

    # Primary transits (same in both)
    for k in range(0, 20):
        center = t0 + k * period
        in_tr = np.abs(time - center) <= (duration_days / 2.0)
        flux0[in_tr] -= 0.001
        flux1[in_tr] -= 0.001

    # Add secondary at phase 0.5 to flux1 only
    for k in range(0, 20):
        center = t0 + (k + 0.5) * period
        in_sec = np.abs(time - center) <= (duration_days / 2.0)
        flux1[in_sec] -= 0.001

    lc0 = LightCurveData(
        time=time,
        flux=flux0,
        flux_err=flux_err,
        quality=quality,
        valid_mask=valid_mask,
        tic_id=1,
        sector=1,
        cadence_seconds=1800.0,
    )
    lc1 = LightCurveData(
        time=time,
        flux=flux1,
        flux_err=flux_err,
        quality=quality,
        valid_mask=valid_mask,
        tic_id=1,
        sector=1,
        cadence_seconds=1800.0,
    )

    cfg = SecondaryEclipseConfig(
        secondary_center=0.5,
        secondary_half_width=0.05,
        baseline_half_width=0.10,
        min_secondary_points=10,
        min_baseline_points=10,
        min_secondary_events=3,
        n_coverage_bins=20,
        min_phase_coverage=0.3,
    )

    r0 = check_secondary_eclipse(lightcurve=lc0, period=period, t0=t0, config=cfg)
    r1 = check_secondary_eclipse(lightcurve=lc1, period=period, t0=t0, config=cfg)

    s0 = float(r0.details.get("secondary_depth_sigma") or 0.0)
    s1 = float(r1.details.get("secondary_depth_sigma") or 0.0)
    assert s1 > s0
    assert s1 > 3.0
