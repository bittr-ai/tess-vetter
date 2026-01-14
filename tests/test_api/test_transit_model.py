import numpy as np
import pytest

from bittr_tess_vetter.api.transit_model import compute_transit_model
from bittr_tess_vetter.compute.periodogram import compute_bls_model


def test_compute_transit_model_perfect_box_has_zero_residuals() -> None:
    time = np.linspace(1500.0, 1527.0, 5000, dtype=np.float64)
    flux_err = np.full_like(time, 1e-4)

    period = 3.0
    t0 = 1500.2
    duration_hours = 3.0
    depth_ppm = 1000.0
    depth_fraction = depth_ppm / 1e6

    model = compute_bls_model(
        time=time,
        period=period,
        t0=t0,
        duration_hours=duration_hours,
        depth=depth_fraction,
    )

    metrics = compute_transit_model(
        time=time,
        flux=model,
        flux_err=flux_err,
        period=period,
        t0=t0,
        duration_hours=duration_hours,
        depth_ppm=depth_ppm,
    )

    assert metrics["rms_residual"] == pytest.approx(0.0, abs=1e-15)
    assert metrics["chi2"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["reduced_chi2"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["n_in_transit"] > 0


def test_compute_transit_model_ignores_nans_in_diagnostics() -> None:
    time = np.linspace(1500.0, 1527.0, 2000, dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 1e-4)

    flux[10] = np.nan
    time[20] = np.nan
    flux_err[30] = np.nan

    metrics = compute_transit_model(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=2.0,
        t0=1500.0,
        duration_hours=2.0,
        depth_ppm=500.0,
    )

    assert metrics["chi2"] == float(metrics["chi2"])
    assert metrics["reduced_chi2"] == float(metrics["reduced_chi2"])
    assert metrics["n_in_transit"] >= 0


def test_compute_transit_model_raises_on_too_few_finite_points() -> None:
    time = np.array([1.0, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.ones_like(time) * 1e-4

    with pytest.raises(ValueError, match="Insufficient finite points"):
        compute_transit_model(
            time=time,
            flux=flux,
            flux_err=flux_err,
            period=2.0,
            t0=1.0,
            duration_hours=2.0,
            depth_ppm=1000.0,
        )
