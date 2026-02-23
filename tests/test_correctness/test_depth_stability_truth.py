from __future__ import annotations

import numpy as np

from tess_vetter.domain.lightcurve import LightCurveData
from tess_vetter.validation.lc_checks import DepthStabilityConfig, check_depth_stability


def _make_lc(*, depths_by_epoch: dict[int, float]) -> LightCurveData:
    rng = np.random.default_rng(20260122)

    period = 5.0
    t0 = 1.0
    duration_hours = 2.0
    duration_days = duration_hours / 24.0

    time = np.arange(0.0, 80.0, 0.01, dtype=np.float64)
    flux_err = np.full_like(time, 2e-4, dtype=np.float64)
    flux = np.ones_like(time, dtype=np.float64) + rng.normal(0.0, flux_err, size=time.shape)

    for ep, depth in depths_by_epoch.items():
        center = t0 + ep * period
        in_tr = np.abs(time - center) <= (duration_days / 2.0)
        flux[in_tr] -= depth

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


def test_depth_stability_metrics_increase_with_epoch_depth_scatter() -> None:
    period = 5.0
    t0 = 1.0
    duration_hours = 2.0

    # Use the same set of epochs for both LCs to isolate depth variability.
    epochs = list(range(0, 12))
    stable = dict.fromkeys(epochs, 0.001)
    variable = {ep: (0.0016 if (ep % 2 == 0) else 0.0004) for ep in epochs}

    lc_stable = _make_lc(depths_by_epoch=stable)
    lc_var = _make_lc(depths_by_epoch=variable)

    cfg = DepthStabilityConfig(
        min_transits_for_confidence=3,
        min_points_per_epoch=3,
        baseline_window_mult=6.0,
        outlier_sigma=4.0,
        use_red_noise_inflation=False,
    )

    r_stable = check_depth_stability(
        lightcurve=lc_stable, period=period, t0=t0, duration_hours=duration_hours, config=cfg
    )
    r_var = check_depth_stability(
        lightcurve=lc_var, period=period, t0=t0, duration_hours=duration_hours, config=cfg
    )

    assert r_stable.id == "V04"
    assert r_var.id == "V04"

    scatter_stable = float(r_stable.details["depth_scatter_ppm"])
    scatter_var = float(r_var.details["depth_scatter_ppm"])
    chi2_stable = float(r_stable.details["chi2_reduced"])
    chi2_var = float(r_var.details["chi2_reduced"])

    assert scatter_var > scatter_stable
    assert chi2_var > chi2_stable
    assert scatter_var - scatter_stable > 200.0

