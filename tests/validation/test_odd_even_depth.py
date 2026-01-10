from __future__ import annotations

import numpy as np

from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.domain.detection import TransitCandidate
from bittr_tess_vetter.validation.lc_checks import check_odd_even_depth


def _make_lightcurve_with_box_transits(
    *,
    period_days: float = 3.0,
    t0_btjd: float = 1.0,
    duration_hours: float = 2.0,
    baseline_days: float = 27.0,
    cadence_minutes: float = 30.0,
    depth_frac: float = 0.001,
    depth_frac_even: float | None = None,
    noise_ppm: float = 50.0,
    seed: int = 123,
) -> LightCurveData:
    rng = np.random.default_rng(seed)
    dt_days = cadence_minutes / (24.0 * 60.0)
    time = np.arange(0.0, baseline_days, dt_days, dtype=np.float64)
    flux = np.ones_like(time)
    flux += rng.normal(0.0, noise_ppm * 1e-6, size=time.size)
    flux_err = np.full_like(time, noise_ppm * 1e-6)
    quality = np.zeros(time.size, dtype=np.int32)
    valid_mask = np.ones(time.size, dtype=bool)

    dur_days = duration_hours / 24.0
    half = dur_days / 2.0
    epoch = np.floor((time - t0_btjd + period_days / 2.0) / period_days).astype(int)
    phase = ((time - t0_btjd) / period_days) % 1.0
    phase_dist = np.minimum(phase, 1.0 - phase)
    in_transit = phase_dist < (half / period_days)

    if depth_frac_even is None:
        depth_by_epoch = {int(e): depth_frac for e in np.unique(epoch)}
    else:
        depth_by_epoch = {int(e): (depth_frac_even if (e % 2 == 0) else depth_frac) for e in np.unique(epoch)}

    for e, dep in depth_by_epoch.items():
        m = in_transit & (epoch == e)
        flux[m] *= (1.0 - float(dep))

    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality,
        valid_mask=valid_mask,
        tic_id=1,
        sector=1,
        cadence_seconds=int(cadence_minutes * 60),
    )


def test_odd_even_depth_metrics_only_clean_injection() -> None:
    lc = _make_lightcurve_with_box_transits(
        depth_frac=0.001, depth_frac_even=None, noise_ppm=30.0, cadence_minutes=2.0
    )
    cand = TransitCandidate(period=3.0, t0=1.0, duration_hours=2.0, depth=0.001, snr=10.0)
    r = check_odd_even_depth(lc, cand.period, cand.t0, cand.duration_hours)

    assert r.passed is None
    assert r.details.get("_metrics_only") is True
    assert r.details["n_odd_transits"] >= 2
    assert r.details["n_even_transits"] >= 2
    assert abs(float(r.details["rel_diff"])) < 0.1
    assert float(r.details["delta_sigma"]) < 2.5


def test_odd_even_depth_metrics_only_alternating_depths() -> None:
    lc = _make_lightcurve_with_box_transits(
        depth_frac=0.0015, depth_frac_even=0.0007, noise_ppm=30.0, cadence_minutes=2.0
    )
    cand = TransitCandidate(period=3.0, t0=1.0, duration_hours=2.0, depth=0.001, snr=10.0)
    r = check_odd_even_depth(lc, cand.period, cand.t0, cand.duration_hours)

    assert r.passed is None
    assert r.details.get("_metrics_only") is True
    assert float(r.details["rel_diff"]) > 0.2
    assert float(r.details["delta_sigma"]) > 2.0
