from __future__ import annotations

import numpy as np

from tess_vetter.api.ephemeris_specificity import (
    SmoothTemplateConfig,
    compute_depth_threshold_numpy,
)


def test_compute_depth_threshold_numpy_structure() -> None:
    n = 1000
    time = np.linspace(1500.0, 1527.0, n, dtype=np.float64)
    flux = np.ones(n, dtype=np.float64)
    flux_err = np.ones(n, dtype=np.float64) * 2e-4

    res = compute_depth_threshold_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=5.0,
        t0_btjd=1502.0,
        duration_hours=3.0,
        target_score=7.0,
        config=SmoothTemplateConfig(ingress_egress_fraction=0.2, sharpness=30.0),
    )

    assert res.backend == "numpy"
    assert np.isfinite(res.score_current)
    assert res.target_score == 7.0
    assert np.isfinite(res.depth_hat_ppm)
    assert np.isfinite(res.depth_sigma_ppm)
    assert res.depth_needed_ppm >= 0.0
