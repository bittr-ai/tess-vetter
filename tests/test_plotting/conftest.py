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


@pytest.fixture(autouse=True)
def close_figures():
    """Automatically close all figures after each test.

    This prevents memory leaks from open figures during test runs.
    """
    yield
    plt.close("all")
