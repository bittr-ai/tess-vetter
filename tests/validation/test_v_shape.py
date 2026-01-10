from __future__ import annotations

import numpy as np

from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.validation.lc_checks import check_v_shape


def _make_trapezoid_transit_lc(
    *,
    period_days: float = 5.0,
    t0_btjd: float = 1.0,
    duration_hours: float = 3.0,
    tflat_ttotal_ratio: float = 0.6,
    depth_frac: float = 0.001,
    baseline_days: float = 27.0,
    cadence_minutes: float = 2.0,
    noise_ppm: float = 50.0,
    seed: int = 42,
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
    half = duration_days / 2.0
    flat_half = half * float(tflat_ttotal_ratio)

    # Phase centered at 0 for each transit
    phase = ((time - t0_btjd) / period_days + 0.5) % 1.0 - 0.5
    dt = np.abs(phase) * period_days  # days from center

    # Trapezoid profile in flux (fractional depth)
    dep = np.zeros_like(flux)
    in_total = dt <= half
    in_flat = dt <= flat_half
    dep[in_flat] = depth_frac
    # linear ramps
    ramp = in_total & ~in_flat
    if np.any(ramp):
        dep[ramp] = depth_frac * (half - dt[ramp]) / max(1e-12, (half - flat_half))

    flux[in_total] *= (1.0 - dep[in_total])

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


def test_v_shape_metrics_only_u_like_vs_v_like_ratio() -> None:
    # U-like (clear flat bottom)
    lc_u = _make_trapezoid_transit_lc(tflat_ttotal_ratio=0.7, noise_ppm=30.0)
    r_u = check_v_shape(lc_u, period=5.0, t0=1.0, duration_hours=3.0)
    assert r_u.passed is None
    assert r_u.details.get("_metrics_only") is True

    # V-like (no flat bottom)
    lc_v = _make_trapezoid_transit_lc(tflat_ttotal_ratio=0.0, noise_ppm=30.0, seed=43)
    r_v = check_v_shape(lc_v, period=5.0, t0=1.0, duration_hours=3.0)
    assert r_v.passed is None
    assert r_v.details.get("_metrics_only") is True

    ratio_u = float(r_u.details["tflat_ttotal_ratio"])
    ratio_v = float(r_v.details["tflat_ttotal_ratio"])
    assert ratio_u > ratio_v

