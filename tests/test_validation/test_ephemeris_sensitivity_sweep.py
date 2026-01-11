from __future__ import annotations

import numpy as np

from bittr_tess_vetter.api.ephemeris_sensitivity_sweep import compute_sensitivity_sweep_numpy
from bittr_tess_vetter.api.ephemeris_specificity import SmoothTemplateConfig


def _inject_box_transit(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    depth_frac: float,
) -> np.ndarray:
    duration_days = float(duration_hours) / 24.0
    half = duration_days / 2.0
    phase = (time - float(t0_btjd)) / float(period_days)
    phase = phase - np.floor(phase + 0.5)
    dt_days = np.abs(phase * float(period_days))
    in_transit = dt_days <= half
    out = flux.copy()
    out[in_transit] -= float(depth_frac)
    return out


def test_compute_sensitivity_sweep_numpy_deterministic() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    time = np.linspace(0.0, 27.4, n, dtype=np.float64)
    flux = np.ones(n, dtype=np.float64) + rng.normal(0.0, 2e-4, size=n).astype(np.float64)
    flux_err = np.full(n, 2e-4, dtype=np.float64)

    period_days = 4.5
    t0_btjd = 1.0
    duration_hours = 2.8
    flux = _inject_box_transit(
        time,
        flux,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_frac=9e-4,
    )

    cfg = SmoothTemplateConfig(ingress_egress_fraction=0.2, sharpness=30.0)
    r1 = compute_sensitivity_sweep_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        config=cfg,
        downsample_levels=[1, 2],
        outlier_policies=["none", "sigma_clip_4"],
        detrenders=["none", "running_median_0.5d"],
        include_celerite2_sho=False,
        random_seed=123,
    )
    r2 = compute_sensitivity_sweep_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        config=cfg,
        downsample_levels=[1, 2],
        outlier_policies=["none", "sigma_clip_4"],
        detrenders=["none", "running_median_0.5d"],
        include_celerite2_sho=False,
        random_seed=123,
    )

    assert r1.n_variants_total == r2.n_variants_total
    assert r1.n_variants_ok == r2.n_variants_ok
    assert r1.metric_variance == r2.metric_variance
    assert r1.best_variant_id == r2.best_variant_id
    assert r1.worst_variant_id == r2.worst_variant_id

