from __future__ import annotations

import numpy as np

from tess_vetter.api.ephemeris_reliability import compute_reliability_regime_numpy
from tess_vetter.api.ephemeris_specificity import SmoothTemplateConfig


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


def test_compute_reliability_regime_numpy_smoke() -> None:
    rng = np.random.default_rng(123)
    n = 4000
    time = np.linspace(0.0, 27.4, n, dtype=np.float64)
    flux = np.ones(n, dtype=np.float64) + rng.normal(0.0, 2e-4, size=n).astype(np.float64)
    flux_err = np.full(n, 2e-4, dtype=np.float64)

    period_days = 3.2
    t0_btjd = 1.1
    duration_hours = 2.4
    flux = _inject_box_transit(
        time,
        flux,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_frac=8e-4,
    )

    res = compute_reliability_regime_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        config=SmoothTemplateConfig(ingress_egress_fraction=0.2, sharpness=30.0),
        n_phase_shifts=50,
        period_jitter_n=21,
        include_harmonics=True,
    )

    assert np.isfinite(res.base.score)
    assert res.period_neighborhood.period_grid_days.size == 21
    assert 0.0 <= res.phase_shift_null.p_value_one_sided <= 1.0
    assert isinstance(res.warnings, list)
    assert isinstance(res.label, str)
