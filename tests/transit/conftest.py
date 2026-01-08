"""Test fixtures for transit timing and vetting modules."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def multi_transit_lc() -> dict[str, NDArray[np.float64] | float | int]:
    """Generate light curve with multiple clean transits.

    Returns:
        Dictionary with time, flux, flux_err arrays and transit parameters.
    """
    np.random.seed(42)

    time = np.linspace(0, 100, 10000, dtype=np.float64)
    period = 8.46
    t0 = 1.0
    transit_depth = 0.004
    noise = 0.001
    transit_duration_hours = 3.5
    transit_duration_days = transit_duration_hours / 24.0

    # Create light curve with multiple transits
    flux = 1.0 + np.random.normal(0, noise, len(time))

    for i in range(-5, 15):
        center = t0 + i * period
        in_transit = np.abs(time - center) < transit_duration_days / 2
        flux[in_transit] -= transit_depth

    flux = flux.astype(np.float64)
    flux_err = np.ones_like(flux, dtype=np.float64) * noise

    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "period": period,
        "t0": t0,
        "transit_depth": transit_depth,
        "transit_duration_hours": transit_duration_hours,
    }


@pytest.fixture
def ttv_lc() -> dict[str, NDArray[np.float64] | float | list[float]]:
    """Generate light curve with injected TTVs (sinusoidal timing variations).

    Returns:
        Dictionary with time, flux, flux_err arrays and TTV parameters.
    """
    np.random.seed(42)

    time = np.linspace(0, 150, 15000, dtype=np.float64)
    period = 5.0
    t0 = 2.0
    transit_depth = 0.005
    noise = 0.0008
    transit_duration_hours = 2.5
    transit_duration_days = transit_duration_hours / 24.0

    # TTV parameters
    ttv_amplitude_days = 0.02  # ~30 minutes
    ttv_period_epochs = 5  # Super-period in epochs

    # Create light curve with TTV-modulated transits
    flux = 1.0 + np.random.normal(0, noise, len(time))

    true_transit_times: list[float] = []
    n_transits = int((time[-1] - t0) / period) + 2

    for epoch in range(-2, n_transits):
        # Calculate TTV offset
        ttv_offset = ttv_amplitude_days * np.sin(2 * np.pi * epoch / ttv_period_epochs)
        center = t0 + epoch * period + ttv_offset
        true_transit_times.append(center)

        in_transit = np.abs(time - center) < transit_duration_days / 2
        flux[in_transit] -= transit_depth

    flux = flux.astype(np.float64)
    flux_err = np.ones_like(flux, dtype=np.float64) * noise

    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "period": period,
        "t0": t0,
        "transit_depth": transit_depth,
        "transit_duration_hours": transit_duration_hours,
        "ttv_amplitude_days": ttv_amplitude_days,
        "ttv_period_epochs": ttv_period_epochs,
        "true_transit_times": true_transit_times,
    }


@pytest.fixture
def odd_even_diff_lc() -> dict[str, NDArray[np.float64] | float]:
    """Generate light curve with different depths for odd vs even transits (EB-like).

    Returns:
        Dictionary with depth difference between odd and even epochs.
    """
    np.random.seed(42)

    time = np.linspace(0, 100, 10000, dtype=np.float64)
    period = 5.0
    t0 = 1.0
    depth_odd = 0.006  # Primary eclipse
    depth_even = 0.002  # Secondary eclipse
    noise = 0.0005
    transit_duration_hours = 2.0
    transit_duration_days = transit_duration_hours / 24.0

    flux = 1.0 + np.random.normal(0, noise, len(time))

    for epoch in range(-2, 25):
        center = t0 + epoch * period
        in_transit = np.abs(time - center) < transit_duration_days / 2

        if epoch % 2 == 0:
            flux[in_transit] -= depth_even
        else:
            flux[in_transit] -= depth_odd

    flux = flux.astype(np.float64)
    flux_err = np.ones_like(flux, dtype=np.float64) * noise

    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "period": period,
        "t0": t0,
        "depth_odd": depth_odd,
        "depth_even": depth_even,
        "transit_duration_hours": transit_duration_hours,
    }


@pytest.fixture
def equal_depth_lc() -> dict[str, NDArray[np.float64] | float]:
    """Generate light curve with equal depths for odd and even transits (planet-like).

    Returns:
        Dictionary with equal depth transits.
    """
    np.random.seed(42)

    time = np.linspace(0, 100, 10000, dtype=np.float64)
    period = 5.0
    t0 = 1.0
    transit_depth = 0.004
    noise = 0.0008
    transit_duration_hours = 2.0
    transit_duration_days = transit_duration_hours / 24.0

    flux = 1.0 + np.random.normal(0, noise, len(time))

    for epoch in range(-2, 25):
        center = t0 + epoch * period
        in_transit = np.abs(time - center) < transit_duration_days / 2
        flux[in_transit] -= transit_depth

    flux = flux.astype(np.float64)
    flux_err = np.ones_like(flux, dtype=np.float64) * noise

    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "period": period,
        "t0": t0,
        "transit_depth": transit_depth,
        "transit_duration_hours": transit_duration_hours,
    }
