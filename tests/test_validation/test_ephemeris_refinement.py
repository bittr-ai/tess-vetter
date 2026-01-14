from __future__ import annotations

import numpy as np

from bittr_tess_vetter.api.ephemeris_refinement import (
    EphemerisRefinementCandidate,
    EphemerisRefinementConfig,
    refine_candidates_numpy,
)


def test_refine_candidates_numpy_structure() -> None:
    n = 2000
    time = np.linspace(1500.0, 1527.0, n, dtype=np.float64)
    period = 5.0
    t0 = 1502.0
    duration_hours = 3.0

    flux = np.ones(n, dtype=np.float64)
    phase = ((time - t0) / period) % 1.0
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    in_transit = np.abs(phase) < (duration_hours / 24.0 / period) / 2.0
    flux[in_transit] -= 0.001  # 1000 ppm
    flux += np.random.default_rng(42).normal(0.0, 2e-4, size=n)

    flux_err = np.ones(n, dtype=np.float64) * 2e-4

    res = refine_candidates_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        candidates=[
            EphemerisRefinementCandidate(
                period_days=period, t0_btjd=t0, duration_hours=duration_hours
            )
        ],
        config=EphemerisRefinementConfig(steps=5),
    )

    assert res.n_points_used == n
    assert len(res.refined) == 1
    row = res.refined[0]
    assert row.period_days == period
    assert np.isfinite(row.t0_refined_btjd)
    assert np.isfinite(row.duration_refined_hours)
    assert np.isfinite(row.depth_hat_ppm)
    assert np.isfinite(row.score_z)
    assert row.score_z > 0.0
