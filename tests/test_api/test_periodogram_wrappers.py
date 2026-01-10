import importlib.util
import numpy as np
import pytest

from bittr_tess_vetter.api.periodogram import refine_period, run_periodogram
from bittr_tess_vetter.api.transit_model import compute_transit_model

TLS_AVAILABLE = importlib.util.find_spec("transitleastsquares") is not None


def _inject_transit(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period: float,
    t0: float,
    duration_days: float,
    depth: float,
) -> np.ndarray:
    flux_out = flux.copy()
    phase = ((time - t0) / period + 0.5) % 1.0 - 0.5
    half_dur = duration_days / (2.0 * period)
    in_transit = np.abs(phase) < half_dur
    flux_out[in_transit] *= 1.0 - depth
    return flux_out


def test_run_periodogram_ls_returns_finite_power() -> None:
    time = np.linspace(1500.0, 1527.0, 1000, dtype=np.float64)
    # Constant flux can yield NaNs/div0 in some normalized LS implementations.
    flux = 1.0 + 1e-6 * np.sin(2 * np.pi * time / 2.5)
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


def test_run_periodogram_ls_recovers_sinusoid_period() -> None:
    time = np.linspace(1500.0, 1527.0, 4000, dtype=np.float64)
    true_period = 5.0
    amp = 2e-4
    phase = 0.7
    flux = 1.0 + amp * np.sin(2.0 * np.pi * time / true_period + phase)
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
    assert np.isfinite(result.best_period)
    assert abs(result.best_period - true_period) / true_period < 0.05
    assert result.best_t0 >= time.min()
    assert result.best_t0 < time.min() + result.best_period


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


@pytest.mark.skipif(not TLS_AVAILABLE, reason="transitleastsquares not available")
def test_refine_period_improves_precision() -> None:
    time = np.linspace(1500.0, 1527.0, 2500, dtype=np.float64)
    flux = np.ones_like(time)

    true_period = 3.567
    t0 = 1501.5
    duration_days = 0.1
    depth = 0.01

    flux = _inject_transit(
        time,
        flux,
        period=true_period,
        t0=t0,
        duration_days=duration_days,
        depth=depth,
    )

    initial_period = 3.5
    refined_period, refined_t0, refined_power = refine_period(
        time=time,
        flux=flux,
        flux_err=None,
        initial_period=initial_period,
        initial_duration=duration_days * 24,
        refine_factor=0.1,
        n_refine=100,
    )

    assert abs(refined_period - true_period) < abs(initial_period - true_period)
    assert float(refined_t0) == float(refined_t0)
    assert float(refined_power) == float(refined_power)
