from __future__ import annotations

import numpy as np

from tess_vetter.domain.lightcurve import LightCurveData
from tess_vetter.validation.lc_checks import VShapeConfig, check_v_shape


def _trapezoid_flux(
    phase: np.ndarray, *, tflat_ttotal_ratio: float, t_total_phase: float, depth: float
) -> np.ndarray:
    half_total = t_total_phase / 2
    half_flat = (tflat_ttotal_ratio * t_total_phase) / 2

    flux = np.ones_like(phase, dtype=np.float64)

    if half_flat <= 0:
        in_transit = np.abs(phase) < half_total
        if np.any(in_transit):
            flux[in_transit] = 1 - depth * (1 - np.abs(phase[in_transit]) / half_total)
        return flux

    flat_mask = np.abs(phase) < half_flat
    flux[flat_mask] = 1 - depth

    if half_total > half_flat:
        slope_width = half_total - half_flat
        ingress_mask = (phase < -half_flat) & (phase > -half_total)
        egress_mask = (phase > half_flat) & (phase < half_total)

        if np.any(ingress_mask):
            frac = (-phase[ingress_mask] - half_flat) / slope_width
            flux[ingress_mask] = (1 - depth) + depth * frac
        if np.any(egress_mask):
            frac = (phase[egress_mask] - half_flat) / slope_width
            flux[egress_mask] = (1 - depth) + depth * frac

    return flux


def _make_lc(*, tflat_ttotal_ratio: float) -> LightCurveData:
    rng = np.random.default_rng(20260122)
    period = 5.0
    t0 = 1.0
    duration_hours = 4.0

    time = np.arange(0.0, 50.0, 0.01, dtype=np.float64)
    phase = ((time - t0) / period + 0.5) % 1 - 0.5
    t_total_phase = (duration_hours / 24.0) / period

    flux_err = np.full_like(time, 2e-4, dtype=np.float64)
    flux = _trapezoid_flux(
        phase, tflat_ttotal_ratio=tflat_ttotal_ratio, t_total_phase=t_total_phase, depth=0.001
    )
    flux = flux + rng.normal(0.0, flux_err, size=time.shape)

    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=np.zeros_like(time, dtype=np.int32),
        valid_mask=np.ones_like(time, dtype=bool),
        tic_id=1,
        sector=1,
        cadence_seconds=1800.0,
    )


def test_v_shape_ratio_higher_for_u_shaped_than_v_shaped() -> None:
    period = 5.0
    t0 = 1.0
    duration_hours = 4.0

    lc_u = _make_lc(tflat_ttotal_ratio=0.8)
    lc_v = _make_lc(tflat_ttotal_ratio=0.0)

    cfg = VShapeConfig(min_points_in_transit=10, min_transit_coverage=0.6, n_bootstrap=20)
    ru = check_v_shape(lc_u, period=period, t0=t0, duration_hours=duration_hours, config=cfg)
    rv = check_v_shape(lc_v, period=period, t0=t0, duration_hours=duration_hours, config=cfg)

    assert ru.id == "V05"
    assert rv.id == "V05"

    ratio_u = float(ru.details["tflat_ttotal_ratio"])
    ratio_v = float(rv.details["tflat_ttotal_ratio"])

    # Grid-search resolution is coarse; assert clear separation rather than exact recovery.
    assert ratio_u > 0.6
    assert ratio_v < 0.2
    assert ratio_u - ratio_v > 0.4

