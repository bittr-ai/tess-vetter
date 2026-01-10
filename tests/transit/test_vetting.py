"""Tests for transit vetting primitives (odd/even comparison).

Tests for:
- split_odd_even: Separating transits by epoch parity
- compare_odd_even_depths: Depth difference measurement
- compute_odd_even_result: Full odd/even analysis
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bittr_tess_vetter.transit import (
    OddEvenResult,
    compare_odd_even_depths,
    compute_odd_even_result,
    split_odd_even,
)


class TestSplitOddEven:
    """Tests for splitting transits by parity."""

    def test_splits_correctly(self, equal_depth_lc: dict[str, NDArray[np.float64] | float]) -> None:
        """Correctly splits transits into odd and even groups."""
        time = equal_depth_lc["time"]
        flux = equal_depth_lc["flux"]
        flux_err = equal_depth_lc["flux_err"]
        period = equal_depth_lc["period"]
        t0 = equal_depth_lc["t0"]
        duration_hours = equal_depth_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)

        (
            (odd_time, odd_flux, odd_flux_err),
            (
                even_time,
                even_flux,
                even_flux_err,
            ),
            n_odd,
            n_even,
        ) = split_odd_even(time, flux, flux_err, period, t0, duration_hours)

        # Should have both odd and even transits
        assert n_odd > 0
        assert n_even > 0
        assert len(odd_flux) > 0
        assert len(even_flux) > 0

    def test_counts_unique_transits(
        self, equal_depth_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Counts unique transit epochs correctly."""
        time = equal_depth_lc["time"]
        flux = equal_depth_lc["flux"]
        flux_err = equal_depth_lc["flux_err"]
        period = equal_depth_lc["period"]
        t0 = equal_depth_lc["t0"]
        duration_hours = equal_depth_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)

        _, _, n_odd, n_even = split_odd_even(time, flux, flux_err, period, t0, duration_hours)

        # Total should be reasonable for 100-day baseline with 5-day period
        total_expected = int(100 / period)
        assert n_odd + n_even <= total_expected + 5  # Allow some margin


class TestCompareOddEvenDepths:
    """Tests for comparing depths between odd and even transits."""

    def test_detects_depth_difference(
        self, odd_even_diff_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Detects significant depth difference between odd and even."""
        time = odd_even_diff_lc["time"]
        flux = odd_even_diff_lc["flux"]
        flux_err = odd_even_diff_lc["flux_err"]
        period = odd_even_diff_lc["period"]
        t0 = odd_even_diff_lc["t0"]
        duration_hours = odd_even_diff_lc["transit_duration_hours"]
        depth_odd = odd_even_diff_lc["depth_odd"]
        depth_even = odd_even_diff_lc["depth_even"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)
        assert isinstance(depth_odd, float)
        assert isinstance(depth_even, float)

        (
            (odd_time, odd_flux, odd_flux_err),
            (
                even_time,
                even_flux,
                even_flux_err,
            ),
            n_odd,
            n_even,
        ) = split_odd_even(time, flux, flux_err, period, t0, duration_hours)

        (
            depth_odd_ppm,
            depth_even_ppm,
            depth_diff_ppm,
            diff_err_ppm,
            significance,
        ) = compare_odd_even_depths(odd_flux, odd_flux_err, even_flux, even_flux_err)

        # Should detect significant difference
        # True difference is 4000 ppm (6000 - 2000)
        assert depth_diff_ppm > 1000  # At least 1000 ppm difference
        assert significance > 3.0  # Should be significant

    def test_equal_depths_low_significance(
        self, equal_depth_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Equal depths produce low significance."""
        time = equal_depth_lc["time"]
        flux = equal_depth_lc["flux"]
        flux_err = equal_depth_lc["flux_err"]
        period = equal_depth_lc["period"]
        t0 = equal_depth_lc["t0"]
        duration_hours = equal_depth_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)

        (
            (odd_time, odd_flux, odd_flux_err),
            (
                even_time,
                even_flux,
                even_flux_err,
            ),
            n_odd,
            n_even,
        ) = split_odd_even(time, flux, flux_err, period, t0, duration_hours)

        (
            depth_odd_ppm,
            depth_even_ppm,
            depth_diff_ppm,
            diff_err_ppm,
            significance,
        ) = compare_odd_even_depths(odd_flux, odd_flux_err, even_flux, even_flux_err)

        # Should have low significance
        assert significance < 3.0

    def test_handles_insufficient_data(self) -> None:
        """Returns zero significance for insufficient data."""
        odd_flux = np.array([0.99, 0.98], dtype=np.float64)
        odd_flux_err = np.array([0.001, 0.001], dtype=np.float64)
        even_flux = np.array([0.995], dtype=np.float64)
        even_flux_err = np.array([0.001], dtype=np.float64)

        (
            depth_odd_ppm,
            depth_even_ppm,
            depth_diff_ppm,
            diff_err_ppm,
            significance,
        ) = compare_odd_even_depths(odd_flux, odd_flux_err, even_flux, even_flux_err)

        # Should return zero or low significance
        assert significance == 0.0 or diff_err_ppm == float("inf")


class TestComputeOddEvenResult:
    """Tests for full odd/even analysis."""

    def test_detects_eb_like_signal(
        self, odd_even_diff_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Flags EB-like signal as suspicious based on relative depth difference.

        The odd_even_diff_lc fixture has depths of 6000 ppm vs 2000 ppm,
        which is a 100% relative difference ((6000-2000)/4000 = 100%).
        Real EBs show 50-100% depth differences.
        """
        time = odd_even_diff_lc["time"]
        flux = odd_even_diff_lc["flux"]
        flux_err = odd_even_diff_lc["flux_err"]
        period = odd_even_diff_lc["period"]
        t0 = odd_even_diff_lc["t0"]
        duration_hours = odd_even_diff_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)

        result = compute_odd_even_result(time, flux, flux_err, period, t0, duration_hours)

        assert isinstance(result, OddEvenResult)
        # Relative difference should be ~100% for this EB-like signal
        assert result.relative_depth_diff_percent > 50.0  # Well above 10% threshold

    def test_planet_like_not_suspicious(
        self, equal_depth_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Does not flag planet-like signal as suspicious.

        Confirmed planets like HD 209458 b and WASP-18 b have equal odd/even
        depths but may show high sigma values due to precise measurements.
        The key is that the RELATIVE depth difference is small (<10%).
        """
        time = equal_depth_lc["time"]
        flux = equal_depth_lc["flux"]
        flux_err = equal_depth_lc["flux_err"]
        period = equal_depth_lc["period"]
        t0 = equal_depth_lc["t0"]
        duration_hours = equal_depth_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)

        result = compute_odd_even_result(time, flux, flux_err, period, t0, duration_hours)

        # Relative difference should be small for planet-like signals
        assert result.relative_depth_diff_percent < 10.0

    def test_to_dict_method(self, equal_depth_lc: dict[str, NDArray[np.float64] | float]) -> None:
        """OddEvenResult.to_dict() returns valid dictionary."""
        time = equal_depth_lc["time"]
        flux = equal_depth_lc["flux"]
        flux_err = equal_depth_lc["flux_err"]
        period = equal_depth_lc["period"]
        t0 = equal_depth_lc["t0"]
        duration_hours = equal_depth_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)

        result = compute_odd_even_result(time, flux, flux_err, period, t0, duration_hours)

        result_dict = result.to_dict()

        assert "depth_odd_ppm" in result_dict
        assert "depth_even_ppm" in result_dict
        assert "depth_diff_ppm" in result_dict
        assert "relative_depth_diff_percent" in result_dict
        assert "significance_sigma" in result_dict
        assert "n_odd" in result_dict
        assert "n_even" in result_dict

    def test_counts_transits(self, equal_depth_lc: dict[str, NDArray[np.float64] | float]) -> None:
        """Correctly counts odd and even transits."""
        time = equal_depth_lc["time"]
        flux = equal_depth_lc["flux"]
        flux_err = equal_depth_lc["flux_err"]
        period = equal_depth_lc["period"]
        t0 = equal_depth_lc["t0"]
        duration_hours = equal_depth_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)

        result = compute_odd_even_result(time, flux, flux_err, period, t0, duration_hours)

        assert result.n_odd > 0
        assert result.n_even > 0

    def test_handles_insufficient_transits(self) -> None:
        """Returns appropriate result for insufficient transits."""
        # Very short baseline with long period
        time = np.linspace(0, 3, 1000, dtype=np.float64)
        flux = np.ones_like(time, dtype=np.float64)
        flux_err = np.ones_like(flux) * 0.001

        result = compute_odd_even_result(
            time, flux, flux_err, period=10.0, t0=5.0, duration_hours=2.0
        )

        # Should have low counts
        assert result.n_odd + result.n_even <= 1


class TestIntegration:
    """Integration tests for vetting workflow."""

    def test_full_vetting_eb_detection(
        self, odd_even_diff_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Full vetting correctly identifies EB-like signal."""
        time = odd_even_diff_lc["time"]
        flux = odd_even_diff_lc["flux"]
        flux_err = odd_even_diff_lc["flux_err"]
        period = odd_even_diff_lc["period"]
        t0 = odd_even_diff_lc["t0"]
        duration_hours = odd_even_diff_lc["transit_duration_hours"]
        depth_odd = odd_even_diff_lc["depth_odd"]
        depth_even = odd_even_diff_lc["depth_even"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)
        assert isinstance(depth_odd, float)
        assert isinstance(depth_even, float)

        result = compute_odd_even_result(time, flux, flux_err, period, t0, duration_hours)

        # Measured depths should roughly match injected values
        true_diff_ppm = abs(depth_odd - depth_even) * 1e6
        assert abs(result.depth_diff_ppm - true_diff_ppm) < true_diff_ppm * 0.5

    def test_full_vetting_planet_confirmation(
        self, equal_depth_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Full vetting correctly confirms planet-like signal."""
        time = equal_depth_lc["time"]
        flux = equal_depth_lc["flux"]
        flux_err = equal_depth_lc["flux_err"]
        period = equal_depth_lc["period"]
        t0 = equal_depth_lc["t0"]
        duration_hours = equal_depth_lc["transit_duration_hours"]
        transit_depth = equal_depth_lc["transit_depth"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)
        assert isinstance(transit_depth, float)

        result = compute_odd_even_result(time, flux, flux_err, period, t0, duration_hours)

        # Both depths should be similar to injected value
        true_depth_ppm = transit_depth * 1e6
        assert abs(result.depth_odd_ppm - true_depth_ppm) < true_depth_ppm * 0.5
        assert abs(result.depth_even_ppm - true_depth_ppm) < true_depth_ppm * 0.5

    def test_high_sigma_but_low_relative_diff_not_suspicious(self) -> None:
        """High statistical sigma but low relative difference should NOT be suspicious.

        This tests the fix for the bug where confirmed planets like HD 209458 b
        (21.3 sigma) and WASP-18 b (6.1 sigma) were incorrectly flagged as suspicious.
        These planets have ~2-3% relative depth differences, not 50-100% like EBs.
        """
        np.random.seed(42)

        # Simulate a confirmed planet with very precise measurements
        # Small relative difference (2%) but high precision = high sigma
        time = np.linspace(0, 100, 10000, dtype=np.float64)
        period = 3.5
        t0 = 1.0
        transit_depth = 0.015  # 15000 ppm (deep transit like hot Jupiter)
        depth_variation = 0.0003  # 300 ppm difference = 2% relative
        noise = 0.00005  # Very low noise (precise measurements)
        transit_duration_hours = 3.0
        transit_duration_days = transit_duration_hours / 24.0

        flux = 1.0 + np.random.normal(0, noise, len(time))

        for epoch in range(-2, 30):
            center = t0 + epoch * period
            in_transit = np.abs(time - center) < transit_duration_days / 2

            # Small systematic difference between odd and even
            if epoch % 2 == 0:
                flux[in_transit] -= transit_depth
            else:
                flux[in_transit] -= transit_depth + depth_variation

        flux = flux.astype(np.float64)
        flux_err = np.ones_like(flux, dtype=np.float64) * noise

        result = compute_odd_even_result(time, flux, flux_err, period, t0, transit_duration_hours)

        # Should have high sigma due to precise measurements
        # But should NOT be suspicious because relative difference is small
        assert result.significance_sigma > 3.0  # High sigma
        assert result.relative_depth_diff_percent < 10.0  # Low relative diff
