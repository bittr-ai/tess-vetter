import numpy as np

from bittr_tess_vetter.api.periodogram import compute_transit_model, run_periodogram


def test_run_periodogram_ls_returns_finite_power() -> None:
    time = np.linspace(1500.0, 1527.0, 1000, dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 1e-4)

    result = run_periodogram(
        time=time,
        flux=flux,
        flux_err=flux_err,
        method="ls",
        min_period=1.0,
        max_period=10.0,
        data_ref="lc:test:1:pdcsap",
    )

    assert result.method == "ls"
    assert len(result.peaks) == 1
    assert float(result.peaks[0].power) == float(result.peaks[0].power)  # not NaN


def test_compute_transit_model_returns_metrics() -> None:
    time = np.linspace(1500.0, 1527.0, 1000, dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 1e-4)

    metrics = compute_transit_model(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=2.0,
        t0=1500.0,
        duration_hours=2.0,
        depth_ppm=500.0,
    )

    assert metrics["period"] == 2.0
    assert metrics["t0"] == 1500.0
    assert metrics["duration_hours"] == 2.0
    assert metrics["depth_ppm"] == 500.0
    assert metrics["n_in_transit"] > 0
