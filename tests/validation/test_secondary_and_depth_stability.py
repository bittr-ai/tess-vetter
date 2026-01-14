from __future__ import annotations

import numpy as np

from bittr_tess_vetter.domain.detection import TransitCandidate
from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.validation.lc_checks import check_depth_stability, check_secondary_eclipse


def _make_lc_with_primary_and_optional_secondary(
    *,
    period_days: float = 5.0,
    t0_btjd: float = 1.0,
    duration_hours: float = 3.0,
    depth_primary: float = 0.001,
    depth_secondary: float = 0.0,
    baseline_days: float = 27.0,
    cadence_minutes: float = 10.0,
    noise_ppm: float = 80.0,
    seed: int = 0,
) -> LightCurveData:
    rng = np.random.default_rng(seed)
    dt_days = cadence_minutes / (24.0 * 60.0)
    time = np.arange(0.0, baseline_days, dt_days, dtype=np.float64)
    flux = np.ones_like(time)
    flux += rng.normal(0.0, noise_ppm * 1e-6, size=time.size)
    flux_err = np.full_like(time, noise_ppm * 1e-6)
    quality = np.zeros(time.size, dtype=np.int32)
    valid_mask = np.ones(time.size, dtype=bool)

    duration_days = duration_hours / 24.0
    half_phase = (duration_days / period_days) / 2.0
    phase = ((time - t0_btjd) / period_days) % 1.0
    d_to_primary = np.minimum(phase, 1.0 - phase)
    in_primary = d_to_primary < half_phase
    flux[in_primary] *= (1.0 - depth_primary)

    if depth_secondary > 0:
        d_to_secondary = np.abs(phase - 0.5)
        in_secondary = d_to_secondary < half_phase
        flux[in_secondary] *= (1.0 - depth_secondary)

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


def test_secondary_eclipse_metrics_no_secondary_low_sigma() -> None:
    lc = _make_lc_with_primary_and_optional_secondary(depth_secondary=0.0, noise_ppm=50.0)
    r = check_secondary_eclipse(lc, period=5.0, t0=1.0)
    assert r.passed is None
    assert r.details.get("_metrics_only") is True
    assert float(r.details.get("secondary_depth_sigma", 0.0)) < 3.0


def test_secondary_eclipse_metrics_injected_secondary_higher_sigma() -> None:
    lc = _make_lc_with_primary_and_optional_secondary(
        depth_secondary=0.002, noise_ppm=20.0, cadence_minutes=2.0, seed=1
    )
    r = check_secondary_eclipse(lc, period=5.0, t0=1.0)
    assert r.passed is None
    assert float(r.details.get("secondary_depth_sigma", 0.0)) > 2.0


def test_depth_stability_metrics_smoke() -> None:
    lc = _make_lc_with_primary_and_optional_secondary(depth_secondary=0.0, noise_ppm=50.0, seed=2)
    cand = TransitCandidate(period=5.0, t0=1.0, duration_hours=3.0, depth=0.001, snr=10.0)
    r = check_depth_stability(lc, cand.period, cand.t0, cand.duration_hours)
    assert r.passed is None
    assert r.details.get("_metrics_only") is True
    assert r.details["n_transits_measured"] >= 3
    assert np.isfinite(float(r.details["chi2_reduced"]))
