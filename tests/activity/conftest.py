"""Test fixtures for stellar activity characterization module."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def quiet_star_lc() -> dict[str, NDArray[np.float64] | float]:
    """Generate quiet star light curve with minimal variability.

    Creates a realistic light curve with:
    - Very low variability (100 ppm level)
    - Gaussian noise (0.05% level)
    - No flares

    Returns:
        Dictionary with time, flux, flux_err arrays.
    """
    np.random.seed(42)

    # 27 days of 2-minute cadence (one TESS sector)
    time = np.linspace(0, 27, 27 * 720, dtype=np.float64)

    # Very quiet star (100 ppm RMS variability)
    flux = 1.0 + np.random.normal(0, 0.0001, len(time))
    flux = flux.astype(np.float64)
    flux_err = np.ones_like(flux, dtype=np.float64) * 0.0001

    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "variability_ppm": 100.0,
    }


@pytest.fixture
def spotted_rotator_lc() -> dict[str, NDArray[np.float64] | float]:
    """Generate spotted rotator light curve (like AU Mic).

    Creates a realistic light curve with:
    - Strong rotation signal (5% amplitude, 4.86 day period)
    - First harmonic of rotation
    - Gaussian noise (0.2% level)

    Returns:
        Dictionary with time, flux, flux_err arrays and rotation period.
    """
    np.random.seed(42)

    # 50 days of 2-minute cadence
    time = np.linspace(0, 50, 50 * 720, dtype=np.float64)

    # Stellar rotation (4.86 day period, 5% amplitude)
    rotation_period = 4.86
    flux = 1.0 + 0.05 * np.sin(2 * np.pi * time / rotation_period)
    flux += 0.02 * np.sin(4 * np.pi * time / rotation_period)  # Harmonic

    # Add noise
    noise = 0.002
    flux += np.random.normal(0, noise, len(time))
    flux = flux.astype(np.float64)
    flux_err = np.ones_like(flux, dtype=np.float64) * noise

    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "rotation_period": rotation_period,
        "variability_ppm": 50000.0,
    }


@pytest.fixture
def flare_star_lc() -> dict[str, NDArray[np.float64] | float | int]:
    """Generate flare star light curve (like Proxima Centauri).

    Creates a realistic light curve with:
    - Multiple flares (various amplitudes)
    - Low-level variability
    - Gaussian noise

    Returns:
        Dictionary with time, flux, flux_err arrays and flare info.
    """
    np.random.seed(42)

    # 30 days of 2-minute cadence
    time = np.linspace(0, 30, 30 * 720, dtype=np.float64)
    n_points = len(time)

    # Base flux with some low-level variability
    flux = 1.0 + 0.001 * np.sin(2 * np.pi * time / 5.0)
    noise = 0.001
    flux += np.random.normal(0, noise, n_points)

    # Inject flares at specific times
    flare_times = [2.5, 5.3, 8.1, 12.7, 15.2, 18.9, 22.4, 25.8, 28.3]
    flare_amplitudes = [0.02, 0.05, 0.03, 0.10, 0.04, 0.08, 0.06, 0.03, 0.15]
    flare_durations_minutes = [5, 15, 8, 30, 10, 20, 12, 6, 45]

    for t_flare, amp, dur in zip(
        flare_times, flare_amplitudes, flare_durations_minutes, strict=True
    ):
        dur_days = dur / (24 * 60)
        # Simple exponential decay flare profile
        decay = 3.0 * dur_days  # e-folding time
        flare_mask = (time >= t_flare) & (time < t_flare + 5 * decay)
        dt = time[flare_mask] - t_flare
        # Quick rise, exponential decay
        flare_profile = amp * np.exp(-dt / decay) * (1 - np.exp(-dt / (0.1 * decay)))
        flux[flare_mask] += flare_profile

    flux = flux.astype(np.float64)
    flux_err = np.ones_like(flux, dtype=np.float64) * noise

    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "n_flares_injected": len(flare_times),
        "flare_times": flare_times,
    }


@pytest.fixture
def single_flare_lc() -> dict[str, NDArray[np.float64] | float]:
    """Generate light curve with a single prominent flare.

    Returns:
        Dictionary with time, flux, flux_err arrays and flare parameters.
    """
    np.random.seed(123)

    # 5 days of 2-minute cadence
    time = np.linspace(0, 5, 5 * 720, dtype=np.float64)

    # Flat baseline with noise
    noise = 0.0005
    flux = 1.0 + np.random.normal(0, noise, len(time))

    # Inject one clear flare at t=2.5 days
    flare_time = 2.5
    flare_amplitude = 0.05
    flare_duration_minutes = 20.0
    dur_days = flare_duration_minutes / (24 * 60)
    decay = 3.0 * dur_days

    flare_mask = (time >= flare_time) & (time < flare_time + 5 * decay)
    dt = time[flare_mask] - flare_time
    flare_profile = flare_amplitude * np.exp(-dt / decay) * (1 - np.exp(-dt / (0.1 * decay)))
    flux[flare_mask] += flare_profile

    flux = flux.astype(np.float64)
    flux_err = np.ones_like(flux, dtype=np.float64) * noise

    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "flare_time": flare_time,
        "flare_amplitude": flare_amplitude,
        "flare_duration_minutes": flare_duration_minutes,
    }
