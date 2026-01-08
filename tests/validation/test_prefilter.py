from __future__ import annotations

import numpy as np

from bittr_tess_vetter.validation.prefilter import (
    compute_depth_over_depth_err_snr,
    compute_phase_coverage,
)


def test_compute_phase_coverage_full_coverage() -> None:
    time = np.linspace(0.0, 30.0, 5000, dtype=np.float64)
    res = compute_phase_coverage(time=time, period_days=3.0, t0_btjd=0.1, n_bins=20)
    assert 0.95 <= res.coverage_fraction <= 1.0
    assert 0.0 <= res.transit_phase_coverage <= 1.0
    assert res.total_bins == 20
    assert 1 <= res.bins_with_data <= 20


def test_compute_depth_over_depth_err_snr_reasonable() -> None:
    rng = np.random.default_rng(0)
    time = np.linspace(0.0, 10.0, 5000, dtype=np.float64)
    flux = (1.0 + rng.normal(0.0, 1e-4, size=time.shape)).astype(np.float64)
    depth = 0.001
    period = 2.5
    t0 = 1.0
    duration_hours = 2.0

    phase = ((time - t0) / period + 0.5) % 1.0 - 0.5
    in_transit = np.abs(phase) < ((duration_hours / 24.0) / (2.0 * period))
    flux[in_transit] -= depth

    snr, depth_err = compute_depth_over_depth_err_snr(
        time=time,
        flux=flux,
        period_days=period,
        t0_btjd=t0,
        duration_hours=duration_hours,
        depth_fractional=depth,
    )
    assert depth_err > 0
    assert snr > 0
