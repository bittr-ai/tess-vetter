"""Pytest fixtures for plotting tests.

All tests in this directory require matplotlib, so we skip the entire
module if matplotlib is not available.
"""

from __future__ import annotations

import pytest

# Skip all tests in this directory if matplotlib is not installed
pytest.importorskip("matplotlib")

# Use non-interactive backend for tests
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from bittr_tess_vetter.validation.result_schema import CheckResult, ok_result


@pytest.fixture
def mock_v01_result() -> CheckResult:
    """Create a mock V01 (odd/even) CheckResult with plot_data.

    Returns a CheckResult that mimics what the odd_even_depth check would
    return, including the plot_data structure defined in the spec.
    """
    return ok_result(
        id="V01",
        name="Odd/Even Depth",
        metrics={
            "odd_mean_depth_ppm": 500.0,
            "even_mean_depth_ppm": 510.0,
            "sigma_diff": 0.5,
        },
        confidence=0.95,
        raw={
            "plot_data": {
                "version": 1,
                "odd_epochs": [1, 3, 5, 7, 9],
                "odd_depths_ppm": [490.0, 510.0, 495.0, 505.0, 500.0],
                "odd_errs_ppm": [20.0, 22.0, 18.0, 21.0, 19.0],
                "even_epochs": [2, 4, 6, 8, 10],
                "even_depths_ppm": [505.0, 515.0, 508.0, 512.0, 510.0],
                "even_errs_ppm": [21.0, 20.0, 19.0, 22.0, 18.0],
                "mean_odd_ppm": 500.0,
                "mean_even_ppm": 510.0,
            }
        },
    )


@pytest.fixture
def mock_result_no_plot_data() -> CheckResult:
    """Create a mock CheckResult with no plot_data.

    Useful for testing error handling when plot_data is missing.
    """
    return ok_result(
        id="V01",
        name="Odd/Even Depth",
        metrics={"sigma_diff": 0.5},
        raw={"some_other_data": "value"},
    )


@pytest.fixture
def mock_result_no_raw() -> CheckResult:
    """Create a mock CheckResult with raw=None.

    Useful for testing error handling when raw is None.
    """
    return ok_result(
        id="V01",
        name="Odd/Even Depth",
        metrics={"sigma_diff": 0.5},
        raw=None,
    )


@pytest.fixture
def mock_v02_result() -> CheckResult:
    """Create a mock V02 (secondary eclipse) CheckResult with plot_data."""
    import numpy as np

    # Generate mock phase-folded data
    phase = np.linspace(0.0, 1.0, 100).tolist()
    flux = (np.ones(100) + np.random.normal(0, 0.001, 100)).tolist()

    return ok_result(
        id="V02",
        name="Secondary Eclipse",
        metrics={
            "secondary_depth_ppm": 150.0,
            "secondary_depth_sigma": 2.5,
        },
        confidence=0.75,
        raw={
            "plot_data": {
                "version": 1,
                "phase": phase,
                "flux": flux,
                "flux_err": [0.001] * 100,
                "secondary_window": [0.35, 0.65],
                "primary_window": [-0.05, 0.05],
                "secondary_depth_ppm": 150.0,
            }
        },
    )


@pytest.fixture
def mock_v03_result() -> CheckResult:
    """Create a mock V03 (duration consistency) CheckResult with plot_data."""
    return ok_result(
        id="V03",
        name="Duration Consistency",
        metrics={
            "duration_ratio": 1.15,
        },
        confidence=0.85,
        raw={
            "plot_data": {
                "version": 1,
                "observed_hours": 3.5,
                "expected_hours": 3.0,
                "expected_hours_err": 0.6,
                "duration_ratio": 1.15,
            }
        },
    )


@pytest.fixture
def mock_v04_result() -> CheckResult:
    """Create a mock V04 (depth stability) CheckResult with plot_data."""
    return ok_result(
        id="V04",
        name="Depth Stability",
        metrics={
            "chi2_reduced": 1.2,
            "n_transits_measured": 10,
        },
        confidence=0.80,
        raw={
            "plot_data": {
                "version": 1,
                "epoch_times_btjd": [2458600.0 + i * 5.0 for i in range(10)],
                "depths_ppm": [500.0 + i * 5 for i in range(10)],
                "depth_errs_ppm": [25.0] * 10,
                "mean_depth_ppm": 522.5,
                "expected_scatter_ppm": 30.0,
                "dominating_epoch_idx": 3,
            }
        },
    )


@pytest.fixture
def mock_v05_result() -> CheckResult:
    """Create a mock V05 (V-shape) CheckResult with plot_data."""
    import numpy as np

    # Generate mock binned transit data
    binned_phase = np.linspace(-0.02, 0.02, 20).tolist()
    binned_flux = (1.0 - 0.001 * (1 - (np.abs(binned_phase) / 0.02))).tolist()
    binned_flux_err = [0.0002] * 20

    # Generate trapezoid model
    model_phase = np.linspace(-0.025, 0.025, 100).tolist()
    model_flux = np.ones(100).tolist()
    for i, p in enumerate(model_phase):
        if abs(p) < 0.02:
            if abs(p) < 0.01:
                model_flux[i] = 0.999  # Flat bottom
            else:
                model_flux[i] = 1.0 - 0.001 * (1 - (abs(p) - 0.01) / 0.01)

    return ok_result(
        id="V05",
        name="V-Shape",
        metrics={
            "tflat_ttotal_ratio": 0.5,
        },
        confidence=0.70,
        raw={
            "plot_data": {
                "version": 1,
                "binned_phase": binned_phase,
                "binned_flux": binned_flux,
                "binned_flux_err": binned_flux_err,
                "trapezoid_phase": model_phase,
                "trapezoid_flux": model_flux,
                "t_flat_hours": 1.5,
                "t_total_hours": 3.0,
            }
        },
    )


@pytest.fixture(autouse=True)
def close_figures():
    """Automatically close all figures after each test.

    This prevents memory leaks from open figures during test runs.
    """
    yield
    plt.close("all")
