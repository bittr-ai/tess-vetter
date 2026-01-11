from __future__ import annotations

import numpy as np

from bittr_tess_vetter.validation.ephemeris_specificity import (
    SmoothTemplateConfig,
    compute_concentration_metrics,
    compute_local_t0_sensitivity_numpy,
    compute_phase_shift_null,
    score_fixed_period_numpy,
    smooth_box_template_numpy,
)


def _make_box_transit_flux(
    *,
    time: np.ndarray,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    depth_fractional: float,
) -> np.ndarray:
    duration_days = float(duration_hours) / 24.0
    phase = ((time - t0_btjd) / float(period_days) + 0.5) % 1.0 - 0.5
    in_transit = np.abs(phase) < (duration_days / (2.0 * float(period_days)))
    flux = np.ones_like(time, dtype=np.float64)
    flux[in_transit] *= 1.0 - float(depth_fractional)
    return flux


def test_smooth_box_template_numpy_bounded_and_periodic() -> None:
    time = np.linspace(1000.0, 1027.0, 20000, dtype=np.float64)
    period = 3.5
    t0 = 1001.0
    duration_hours = 2.5
    cfg = SmoothTemplateConfig()

    template = smooth_box_template_numpy(
        time=time,
        period_days=period,
        t0_btjd=t0,
        duration_hours=duration_hours,
        config=cfg,
    )
    assert template.shape == time.shape
    assert np.all(template >= 0.0)
    assert np.all(template <= 1.0)

    i0 = int(np.argmin(np.abs(time - t0)))
    i1 = int(np.argmin(np.abs(time - (t0 + period))))
    assert abs(float(template[i0]) - float(template[i1])) < 0.1
    assert float(template[i0]) > 0.9


def test_score_fixed_period_numpy_detects_injected_transit() -> None:
    rng = np.random.default_rng(0)
    time = np.linspace(0.0, 27.0, 20000, dtype=np.float64)
    period = 5.0
    t0 = 2.0
    duration_hours = 2.4
    depth = 0.0015

    flux = _make_box_transit_flux(
        time=time,
        period_days=period,
        t0_btjd=t0,
        duration_hours=duration_hours,
        depth_fractional=depth,
    )
    flux += rng.normal(0.0, 2e-4, size=time.shape)
    flux = flux.astype(np.float64)
    flux_err = np.full_like(time, 2e-4, dtype=np.float64)

    cfg = SmoothTemplateConfig()
    result = score_fixed_period_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=period,
        t0_btjd=t0,
        duration_hours=duration_hours,
        config=cfg,
    )
    assert result.depth_hat > 0
    assert result.depth_sigma > 0
    assert result.score > 3.0


def test_compute_phase_shift_null_returns_valid_p_value() -> None:
    rng = np.random.default_rng(1)
    time = np.linspace(0.0, 27.0, 10000, dtype=np.float64)
    period = 6.0
    t0 = 1.0
    duration_hours = 3.0
    depth = 0.001

    flux = _make_box_transit_flux(
        time=time,
        period_days=period,
        t0_btjd=t0,
        duration_hours=duration_hours,
        depth_fractional=depth,
    )
    flux += rng.normal(0.0, 3e-4, size=time.shape)
    flux = flux.astype(np.float64)
    flux_err = np.full_like(time, 3e-4, dtype=np.float64)

    cfg = SmoothTemplateConfig()
    observed = score_fixed_period_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=period,
        t0_btjd=t0,
        duration_hours=duration_hours,
        config=cfg,
    ).score

    null = compute_phase_shift_null(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=period,
        t0_btjd=t0,
        duration_hours=duration_hours,
        observed_score=float(observed),
        n_trials=51,
        strategy="grid",
        random_seed=123,
        config=cfg,
    )

    assert null.n_trials == 51
    assert 0.0 < null.p_value_one_sided <= 1.0


def test_compute_concentration_metrics_outputs_finite_when_signal_present() -> None:
    rng = np.random.default_rng(2)
    time = np.linspace(0.0, 27.0, 8000, dtype=np.float64)
    period = 4.0
    t0 = 0.5
    duration_hours = 2.0
    depth = 0.002

    flux = _make_box_transit_flux(
        time=time,
        period_days=period,
        t0_btjd=t0,
        duration_hours=duration_hours,
        depth_fractional=depth,
    )
    flux += rng.normal(0.0, 2e-4, size=time.shape)
    flux = flux.astype(np.float64)
    flux_err = np.full_like(time, 2e-4, dtype=np.float64)

    cfg = SmoothTemplateConfig()
    template = smooth_box_template_numpy(
        time=time, period_days=period, t0_btjd=t0, duration_hours=duration_hours, config=cfg
    )
    metrics = compute_concentration_metrics(
        time=time,
        flux=flux,
        flux_err=flux_err,
        template=template,
        period_days=period,
        t0_btjd=t0,
        duration_hours=duration_hours,
    )

    assert metrics.n_in_transit > 0
    assert np.isfinite(metrics.in_transit_contribution_abs)
    assert np.isfinite(metrics.max_point_fraction_abs)
    assert np.isfinite(metrics.top_5_fraction_abs)
    assert np.isfinite(metrics.effective_n_points)


def test_compute_local_t0_sensitivity_prefers_near_true_t0() -> None:
    rng = np.random.default_rng(3)
    time = np.linspace(0.0, 27.0, 12000, dtype=np.float64)
    period = 5.0
    t0_true = 1.2
    duration_hours = 2.4
    depth = 0.001

    flux = _make_box_transit_flux(
        time=time,
        period_days=period,
        t0_btjd=t0_true,
        duration_hours=duration_hours,
        depth_fractional=depth,
    )
    flux += rng.normal(0.0, 2e-4, size=time.shape)
    flux = flux.astype(np.float64)
    flux_err = np.full_like(time, 2e-4, dtype=np.float64)

    cfg = SmoothTemplateConfig()
    res = compute_local_t0_sensitivity_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=period,
        t0_btjd=t0_true,
        duration_hours=duration_hours,
        config=cfg,
        n_grid=81,
        half_span_minutes=120.0,
    )
    assert abs(res.t0_best_btjd - t0_true) < (120.0 / (24.0 * 60.0))
    assert res.score_best >= res.score_at_input

