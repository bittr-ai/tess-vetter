"""Test fixtures for transit recovery module."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def synthetic_active_star_lc() -> dict[str, NDArray[np.float64] | float | int]:
    """Generate synthetic light curve with stellar activity + transit.

    Creates a realistic light curve with:
    - Stellar rotation signal (4.86 day period, 5% amplitude)
    - First harmonic of rotation
    - Transit signal (8.46 day period, 0.35% depth)
    - Gaussian noise (0.2% level)

    Returns:
        Dictionary with time, flux, flux_err arrays and true parameters.
    """
    np.random.seed(42)

    # 50 days of 2-minute cadence
    time = np.linspace(0, 50, 50 * 720, dtype=np.float64)

    # Stellar rotation (4.86 day period, 5% amplitude)
    rotation_period = 4.86
    flux = 1.0 + 0.05 * np.sin(2 * np.pi * time / rotation_period)
    flux += 0.02 * np.sin(4 * np.pi * time / rotation_period)  # Harmonic

    # Add transit (8.46 day period, 0.35% depth)
    transit_period = 8.46
    transit_depth = 0.0035
    t0 = 1.0
    transit_duration_hours = 3.5
    transit_duration_days = transit_duration_hours / 24.0

    for i in range(-2, 10):
        center = t0 + i * transit_period
        in_transit = np.abs(time - center) < transit_duration_days / 2
        flux[in_transit] -= transit_depth

    # Add noise
    noise = 0.002
    flux += np.random.normal(0, noise, len(time))
    flux_err = np.ones_like(flux, dtype=np.float64) * noise

    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "rotation_period": rotation_period,
        "transit_period": transit_period,
        "transit_depth": transit_depth,
        "t0": t0,
        "transit_duration_hours": transit_duration_hours,
    }


@pytest.fixture
def simple_rotation_lc() -> dict[str, NDArray[np.float64] | float]:
    """Generate simple sinusoidal rotation signal.

    Returns:
        Dictionary with time, flux arrays and rotation period.
    """
    time = np.linspace(0, 100, 10000, dtype=np.float64)
    rotation_period = 4.86
    flux = 1.0 + 0.05 * np.sin(2 * np.pi * time / rotation_period)
    flux = flux.astype(np.float64)

    return {
        "time": time,
        "flux": flux,
        "rotation_period": rotation_period,
    }


@pytest.fixture
def flat_lc() -> dict[str, NDArray[np.float64]]:
    """Generate flat light curve with only noise.

    Returns:
        Dictionary with time, flux arrays.
    """
    np.random.seed(123)
    time = np.linspace(0, 100, 10000, dtype=np.float64)
    flux = 1.0 + np.random.normal(0, 0.001, len(time))
    flux = flux.astype(np.float64)

    return {
        "time": time,
        "flux": flux,
    }


@pytest.fixture
def multi_transit_lc() -> dict[str, NDArray[np.float64] | float | int]:
    """Generate light curve with multiple transits but no stellar variability.

    Returns:
        Dictionary with time, flux, flux_err arrays and transit parameters.
    """
    np.random.seed(42)

    time = np.linspace(0, 100, 10000, dtype=np.float64)
    period = 8.46
    t0 = 1.0
    transit_depth = 0.004
    noise = 0.005
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
