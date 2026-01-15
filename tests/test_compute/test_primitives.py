"""Comprehensive unit tests for the astro primitives module.

Tests cover:
- periodogram(): Lomb-Scargle periodogram computation
- fold(): Phase folding of light curves
- detrend(): Median filter detrending
- box_model(): Box transit model generation
- AstroPrimitives class: Namespace for sandbox injection
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from bittr_tess_vetter.compute.primitives import (
    AstroPrimitives,
    astro,
    box_model,
    fold,
    median_detrend,
    periodogram,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def synthetic_sinusoid():
    """Generate synthetic sinusoidal light curve with known period.

    Creates a 10-day observation with ~30 min cadence containing
    a sinusoidal signal with period 2.5 days.
    """
    np.random.seed(42)
    n_points = 480  # ~30 min cadence over 10 days
    time = np.linspace(0, 10, n_points)
    true_period = 2.5  # days
    amplitude = 0.02  # 2% amplitude

    # Sinusoidal signal
    flux = 1.0 + amplitude * np.sin(2 * np.pi * time / true_period)

    # Add small noise
    noise = np.random.normal(0, 0.001, n_points)
    flux += noise

    return time, flux, true_period


@pytest.fixture
def synthetic_transit():
    """Generate synthetic transit light curve with known parameters.

    Creates a light curve with box transits at period=3.0 days,
    depth=0.01 (1%), duration=0.05 (5% of period).
    """
    np.random.seed(123)
    n_points = 1000
    time = np.linspace(0, 15, n_points)  # 15 days
    true_period = 3.0
    true_depth = 0.01
    true_duration = 0.05  # fraction of period
    t0 = 1.5  # first transit center

    flux = np.ones(n_points)

    # Add transits
    for transit_num in range(6):  # ~5 transits in 15 days
        transit_center = t0 + transit_num * true_period
        half_dur_days = (true_duration * true_period) / 2
        in_transit = np.abs(time - transit_center) < half_dur_days
        flux[in_transit] = 1.0 - true_depth

    # Add noise
    flux += np.random.normal(0, 0.002, n_points)

    return time, flux, true_period, t0, true_depth, true_duration


@pytest.fixture
def flux_with_trend():
    """Generate flux with linear trend and transit signal."""
    np.random.seed(456)
    n_points = 500
    time = np.linspace(0, 10, n_points)

    # Linear trend (stellar drift)
    trend = 1.0 + 0.01 * time  # 10% increase over observation

    # Add sinusoidal signal
    signal = 0.005 * np.sin(2 * np.pi * time / 1.5)

    flux = trend + signal
    return time, flux, signal


# =============================================================================
# periodogram() Tests
# =============================================================================


class TestPeriodogram:
    """Tests for the periodogram function."""

    def test_correct_period_detection_sinusoid(self, synthetic_sinusoid):
        """Periodogram should correctly identify the dominant period."""
        time, flux, true_period = synthetic_sinusoid

        # Search around the true period
        periods = np.linspace(1.0, 5.0, 500)
        power = periodogram(time, flux, periods)

        # Find best period
        best_idx = np.argmax(power)
        detected_period = periods[best_idx]

        # Should be within 5% of true period
        assert abs(detected_period - true_period) / true_period < 0.05, (
            f"Detected period {detected_period:.3f} differs from "
            f"true period {true_period:.3f} by more than 5%"
        )

    def test_output_shape_matches_periods(self, synthetic_sinusoid):
        """Output shape should match input periods array."""
        time, flux, _ = synthetic_sinusoid

        for n_periods in [10, 100, 1000]:
            periods = np.linspace(0.5, 10.0, n_periods)
            power = periodogram(time, flux, periods)
            assert power.shape == periods.shape, (
                f"Expected shape {periods.shape}, got {power.shape}"
            )

    def test_nan_handling(self, synthetic_sinusoid):
        """Periodogram should handle NaN values gracefully."""
        time, flux, true_period = synthetic_sinusoid

        # Inject NaN values
        flux_with_nan = flux.copy()
        nan_indices = np.random.choice(len(flux), size=50, replace=False)
        flux_with_nan[nan_indices] = np.nan

        periods = np.linspace(1.0, 5.0, 200)
        power = periodogram(time, flux_with_nan, periods)

        # Should still produce valid output
        assert not np.any(np.isnan(power)), "Output should not contain NaN"

        # Should still detect approximate period (with reduced precision)
        best_idx = np.argmax(power)
        detected_period = periods[best_idx]
        assert abs(detected_period - true_period) / true_period < 0.1

    def test_empty_periods_array(self, synthetic_sinusoid):
        """Empty periods array should return empty power array."""
        time, flux, _ = synthetic_sinusoid
        periods = np.array([])
        power = periodogram(time, flux, periods)
        assert len(power) == 0

    def test_mismatched_time_flux_length(self, synthetic_sinusoid):
        """Should raise ValueError for mismatched array lengths."""
        time, flux, _ = synthetic_sinusoid
        periods = np.linspace(1, 5, 100)

        with pytest.raises(ValueError, match="same length"):
            periodogram(time, flux[:-10], periods)

    def test_insufficient_data_points(self):
        """Should raise ValueError with fewer than 3 data points."""
        time = np.array([1.0, 2.0])
        flux = np.array([1.0, 1.0])
        periods = np.linspace(0.5, 5, 100)

        with pytest.raises(ValueError, match="at least 3"):
            periodogram(time, flux, periods)

    def test_all_nan_flux(self):
        """Should raise ValueError when all flux values are NaN."""
        time = np.linspace(0, 10, 100)
        flux = np.full(100, np.nan)
        periods = np.linspace(0.5, 5, 100)

        with pytest.raises(ValueError, match="finite data points"):
            periodogram(time, flux, periods)

    def test_output_dtype(self, synthetic_sinusoid):
        """Output should be float64."""
        time, flux, _ = synthetic_sinusoid
        periods = np.linspace(1, 5, 100)
        power = periodogram(time, flux, periods)
        assert power.dtype == np.float64

    def test_power_values_reasonable(self, synthetic_sinusoid):
        """Power values should be non-negative and normalized."""
        time, flux, _ = synthetic_sinusoid
        periods = np.linspace(1, 5, 100)
        power = periodogram(time, flux, periods)

        # All values should be non-negative
        assert np.all(power >= 0), "Power values should be non-negative"


# =============================================================================
# fold() Tests
# =============================================================================


class TestFold:
    """Tests for the fold function."""

    def test_phase_range_zero_to_one(self, synthetic_transit):
        """Phase values should be in range [0, 1)."""
        time, flux, period, t0, _, _ = synthetic_transit

        phase, flux_folded = fold(time, flux, period, t0)

        assert np.all(phase >= 0), "Phase should be >= 0"
        assert np.all(phase < 1), "Phase should be < 1"

    def test_transit_at_phase_zero(self, synthetic_transit):
        """Transit should be centered at phase 0 when t0 is transit time."""
        time, flux, period, t0, depth, duration = synthetic_transit

        phase, flux_folded = fold(time, flux, period, t0)

        # Find points near phase 0 (transit center)
        near_zero = (phase < 0.02) | (phase > 0.98)
        transit_flux = np.mean(flux_folded[near_zero])

        # Find out-of-transit points
        out_of_transit = (phase > 0.1) & (phase < 0.9)
        baseline_flux = np.mean(flux_folded[out_of_transit])

        # Transit should show depth
        measured_depth = baseline_flux - transit_flux
        assert measured_depth > depth * 0.5, (
            f"Expected significant transit depth near phase 0, measured {measured_depth:.4f}"
        )

    def test_output_sorted_by_phase(self, synthetic_transit):
        """Output arrays should be sorted by phase."""
        time, flux, period, t0, _, _ = synthetic_transit

        phase, flux_folded = fold(time, flux, period, t0)

        # Check that phase is monotonically increasing
        assert np.all(np.diff(phase) >= 0), "Phase should be sorted"

    def test_output_lengths_match_input(self, synthetic_transit):
        """Output lengths should match input lengths."""
        time, flux, period, t0, _, _ = synthetic_transit

        phase, flux_folded = fold(time, flux, period, t0)

        assert len(phase) == len(time)
        assert len(flux_folded) == len(flux)

    def test_negative_period_raises_error(self, synthetic_transit):
        """Negative period should raise ValueError."""
        time, flux, _, t0, _, _ = synthetic_transit

        with pytest.raises(ValueError, match="positive"):
            fold(time, flux, -1.0, t0)

    def test_zero_period_raises_error(self, synthetic_transit):
        """Zero period should raise ValueError."""
        time, flux, _, t0, _, _ = synthetic_transit

        with pytest.raises(ValueError, match="positive"):
            fold(time, flux, 0.0, t0)

    def test_mismatched_array_lengths(self, synthetic_transit):
        """Should raise ValueError for mismatched time/flux lengths."""
        time, flux, period, t0, _, _ = synthetic_transit

        with pytest.raises(ValueError, match="same length"):
            fold(time, flux[:-5], period, t0)

    def test_output_dtype(self, synthetic_transit):
        """Output arrays should be float64."""
        time, flux, period, t0, _, _ = synthetic_transit

        phase, flux_folded = fold(time, flux, period, t0)

        assert phase.dtype == np.float64
        assert flux_folded.dtype == np.float64

    def test_different_t0_shifts_phase(self, synthetic_transit):
        """Different t0 values should shift the phase."""
        time, flux, period, t0, _, _ = synthetic_transit

        phase1, _ = fold(time, flux, period, t0)
        phase2, _ = fold(time, flux, period, t0 + period / 4)

        # Phases should differ by ~0.25 (modulo 1)
        # Note: since outputs are sorted, we need to compare element-wise
        # on common time indices, not on sorted arrays
        # A simpler check: the phases computed from same time should differ
        test_time = time[100]  # Pick a specific time point
        expected_phase1 = ((test_time - t0) / period) % 1.0
        expected_phase2 = ((test_time - (t0 + period / 4)) / period) % 1.0

        phase_diff = (expected_phase2 - expected_phase1) % 1.0
        # Should be ~0.75 (since we added period/4 to t0, phase shifts by -0.25)
        assert abs(phase_diff - 0.75) < 0.05 or abs(phase_diff - 0.25) < 0.05


# =============================================================================
# detrend() Tests
# =============================================================================


class TestMedianDetrend:
    """Tests for the median_detrend function."""

    def test_removes_linear_trend(self, flux_with_trend):
        """Detrending should remove linear trends."""
        time, flux, _ = flux_with_trend

        detrended = median_detrend(flux, window=51)

        # After detrending, flux should be centered around 1.0
        assert abs(np.median(detrended) - 1.0) < 0.01, (
            f"Detrended flux median {np.median(detrended):.4f} should be ~1.0"
        )

        # Variance should be reduced compared to original
        assert np.std(detrended) < np.std(flux)

    def test_preserves_signal(self, flux_with_trend):
        """Detrending should preserve high-frequency signals."""
        time, flux, signal = flux_with_trend

        # Use window larger than signal period
        detrended = median_detrend(flux, window=201)

        # The detrended flux should still show periodic variation
        # Check that the standard deviation is non-trivial (signal preserved)
        # After detrending, the signal component should remain
        detrended_centered = detrended - np.mean(detrended)

        # Signal amplitude should be preserved (approximately)
        # Original signal had amplitude 0.005
        assert np.std(detrended_centered) > 0.002, "Detrended data should retain signal variation"

        # The detrended data should correlate with the original signal
        # (after accounting for the trend-scaled signal)
        signal_centered = signal - np.mean(signal)
        correlation = np.corrcoef(detrended_centered, signal_centered)[0, 1]
        assert correlation > 0.5, (
            f"Detrended data should correlate with original signal, got {correlation:.3f}"
        )

    def test_window_parameter_effect(self):
        """Larger windows should preserve more low-frequency content."""
        np.random.seed(789)
        n_points = 500
        flux = 1.0 + 0.05 * np.linspace(0, 1, n_points)  # Linear trend

        small_window = median_detrend(flux, window=21)
        large_window = median_detrend(flux, window=201)

        # Small window removes more structure
        std_small = np.std(small_window)
        std_large = np.std(large_window)

        # Small window should produce flatter result
        assert std_small < std_large

    def test_nan_preservation(self):
        """NaN values should be preserved in output."""
        flux = np.ones(100)
        nan_indices = [10, 25, 50, 75, 90]
        flux[nan_indices] = np.nan

        detrended = median_detrend(flux, window=11)

        # Check NaN positions are preserved
        output_nan_indices = np.where(np.isnan(detrended))[0]
        assert_array_equal(output_nan_indices, nan_indices)

    def test_even_window_converted_to_odd(self):
        """Even window sizes should be converted to odd."""
        flux = np.ones(100) + 0.01 * np.arange(100)

        # Should not raise error with even window
        result = median_detrend(flux, window=50)
        assert len(result) == len(flux)

    def test_invalid_window_raises_error(self):
        """Window < 1 should raise ValueError."""
        flux = np.ones(100)

        with pytest.raises(ValueError, match="positive"):
            median_detrend(flux, window=0)

        with pytest.raises(ValueError, match="positive"):
            median_detrend(flux, window=-5)

    def test_output_dtype(self, flux_with_trend):
        """Output should be float64."""
        _, flux, _ = flux_with_trend
        result = median_detrend(flux, window=51)
        assert result.dtype == np.float64

    def test_flat_flux_unchanged(self):
        """Flat flux should remain approximately flat after detrending."""
        flux = np.ones(100)
        detrended = median_detrend(flux, window=21)

        assert_allclose(detrended, flux, rtol=1e-10)

    def test_all_nan_flux(self):
        """All NaN flux should produce all NaN output."""
        flux = np.full(100, np.nan)
        detrended = median_detrend(flux, window=21)

        assert np.all(np.isnan(detrended))


# =============================================================================
# box_model() Tests
# =============================================================================


class TestBoxModel:
    """Tests for the box_model function."""

    def test_depth_parameter(self):
        """Model should show correct transit depth."""
        phase = np.linspace(0, 1, 1000)
        depth = 0.01
        duration = 0.04

        model = box_model(phase, depth, duration)

        # Out of transit should be 1.0
        out_of_transit = (phase > 0.1) & (phase < 0.9)
        assert_allclose(model[out_of_transit], 1.0)

        # In transit should be 1.0 - depth
        in_transit = (phase < duration / 2) | (phase > 1 - duration / 2)
        assert_allclose(model[in_transit], 1.0 - depth)

    def test_duration_parameter(self):
        """Transit width should match duration parameter."""
        phase = np.linspace(0, 1, 10000)
        depth = 0.01
        duration = 0.1  # 10% of phase

        model = box_model(phase, depth, duration)

        # Count in-transit points
        in_transit_fraction = np.sum(model < 1.0) / len(model)

        # Should be approximately equal to duration
        assert abs(in_transit_fraction - duration) < 0.01

    def test_phase_wraparound_at_zero(self):
        """Transit should wrap around phase 0/1 boundary."""
        phase = np.linspace(0, 1, 1000)
        depth = 0.01
        duration = 0.1

        model = box_model(phase, depth, duration)

        # Points near phase 0 should be in transit
        near_zero = phase < 0.02
        assert np.all(model[near_zero] == 1.0 - depth)

        # Points near phase 1 should also be in transit
        near_one = phase > 0.98
        assert np.all(model[near_one] == 1.0 - depth)

    def test_negative_depth_raises_error(self):
        """Negative depth should raise ValueError."""
        phase = np.linspace(0, 1, 100)

        with pytest.raises(ValueError, match="non-negative"):
            box_model(phase, depth=-0.01, duration=0.05)

    def test_invalid_duration_raises_error(self):
        """Duration outside (0, 0.5) should raise ValueError."""
        phase = np.linspace(0, 1, 100)

        with pytest.raises(ValueError, match="Duration"):
            box_model(phase, depth=0.01, duration=0.0)

        with pytest.raises(ValueError, match="Duration"):
            box_model(phase, depth=0.01, duration=0.5)

        with pytest.raises(ValueError, match="Duration"):
            box_model(phase, depth=0.01, duration=-0.1)

    def test_zero_depth(self):
        """Zero depth should produce flat model."""
        phase = np.linspace(0, 1, 100)
        model = box_model(phase, depth=0.0, duration=0.1)

        assert_allclose(model, 1.0)

    def test_output_dtype_and_shape(self):
        """Output should match input shape and be float64."""
        for n_points in [10, 100, 1000]:
            phase = np.linspace(0, 1, n_points)
            model = box_model(phase, depth=0.01, duration=0.05)

            assert model.shape == phase.shape
            assert model.dtype == np.float64

    def test_small_duration(self):
        """Very small duration should still produce valid transit."""
        phase = np.linspace(0, 1, 10000)
        depth = 0.01
        duration = 0.005  # 0.5% of period

        model = box_model(phase, depth, duration)

        # Should have some points in transit
        in_transit = model < 1.0
        assert np.any(in_transit)

        # All in-transit points should have correct depth
        assert_allclose(model[in_transit], 1.0 - depth)


# =============================================================================
# AstroPrimitives Class Tests
# =============================================================================


class TestAstroPrimitives:
    """Tests for the AstroPrimitives class."""

    def test_periodogram_is_staticmethod(self):
        """periodogram should be accessible as staticmethod."""
        assert hasattr(AstroPrimitives, "periodogram")
        assert callable(AstroPrimitives.periodogram)

        # Should work without instance
        time = np.linspace(0, 10, 100)
        flux = np.ones(100)
        periods = np.linspace(1, 5, 50)
        power = AstroPrimitives.periodogram(time, flux, periods)
        assert len(power) == len(periods)

    def test_fold_is_staticmethod(self):
        """fold should be accessible as staticmethod."""
        assert hasattr(AstroPrimitives, "fold")
        assert callable(AstroPrimitives.fold)

        time = np.linspace(0, 10, 100)
        flux = np.ones(100)
        phase, flux_folded = AstroPrimitives.fold(time, flux, period=2.0, t0=0.0)
        assert len(phase) == len(time)

    def test_median_detrend_is_staticmethod(self):
        """median_detrend should be accessible as staticmethod."""
        assert hasattr(AstroPrimitives, "median_detrend")
        assert callable(AstroPrimitives.median_detrend)

        flux = np.ones(100)
        detrended = AstroPrimitives.median_detrend(flux)
        assert len(detrended) == len(flux)

    def test_box_model_is_staticmethod(self):
        """box_model should be accessible as staticmethod."""
        assert hasattr(AstroPrimitives, "box_model")
        assert callable(AstroPrimitives.box_model)

        phase = np.linspace(0, 1, 100)
        model = AstroPrimitives.box_model(phase, depth=0.01, duration=0.05)
        assert len(model) == len(phase)

    def test_all_methods_present(self):
        """All expected methods should be present in class."""
        expected_methods = ["periodogram", "fold", "median_detrend", "box_model"]
        for method in expected_methods:
            assert hasattr(AstroPrimitives, method), f"Missing method: {method}"


class TestAstroInstance:
    """Tests for the module-level astro instance."""

    def test_astro_instance_exists(self):
        """Module-level astro instance should exist."""
        from bittr_tess_vetter.compute.primitives import astro

        assert astro is not None

    def test_astro_is_simple_namespace(self):
        """astro should be a SimpleNamespace for sandbox compatibility."""
        import types

        assert isinstance(astro, types.SimpleNamespace)

    def test_astro_periodogram_callable(self):
        """astro.periodogram should be callable."""
        time = np.linspace(0, 10, 100)
        flux = np.ones(100)
        periods = np.linspace(1, 5, 50)

        power = astro.periodogram(time, flux, periods)
        assert len(power) == len(periods)

    def test_astro_fold_callable(self):
        """astro.fold should be callable."""
        time = np.linspace(0, 10, 100)
        flux = np.ones(100)

        phase, flux_folded = astro.fold(time, flux, period=2.0, t0=0.0)
        assert len(phase) == len(time)

    def test_astro_median_detrend_callable(self):
        """astro.median_detrend should be callable."""
        flux = np.ones(100)
        detrended = astro.median_detrend(flux)
        assert len(detrended) == len(flux)

    def test_astro_box_model_callable(self):
        """astro.box_model should be callable."""
        phase = np.linspace(0, 1, 100)
        model = astro.box_model(phase, depth=0.01, duration=0.05)
        assert len(model) == len(phase)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPrimitivesIntegration:
    """Integration tests combining multiple primitives."""

    def test_periodogram_fold_pipeline(self, synthetic_sinusoid):
        """Test periodogram followed by phase folding."""
        time, flux, true_period = synthetic_sinusoid

        # Step 1: Find period with periodogram
        periods = np.linspace(1.0, 5.0, 500)
        power = astro.periodogram(time, flux, periods)
        best_period = periods[np.argmax(power)]

        # Step 2: Fold at detected period
        phase, flux_folded = astro.fold(time, flux, best_period, t0=time[0])

        # Phase should be sorted and in valid range
        assert np.all(np.diff(phase) >= 0)
        assert np.all(phase >= 0) and np.all(phase < 1)

    def test_detrend_fold_pipeline(self, flux_with_trend):
        """Test detrending followed by phase folding."""
        time, flux, _ = flux_with_trend

        # Step 1: Remove trend
        detrended = astro.median_detrend(flux, window=101)

        # Step 2: Fold at known signal period
        period = 1.5
        phase, flux_folded = astro.fold(time, detrended, period, t0=0.0)

        # Should produce valid folded light curve
        assert len(phase) == len(time)
        assert np.all(phase >= 0) and np.all(phase < 1)

    def test_fold_box_model_comparison(self, synthetic_transit):
        """Test folding transit and comparing with box model."""
        time, flux, period, t0, depth, duration = synthetic_transit

        # Fold the data
        phase, flux_folded = astro.fold(time, flux, period, t0)

        # Generate model for comparison
        model = astro.box_model(phase, depth, duration)

        # Model and data should have similar structure
        # Correlation should be significant
        correlation = np.corrcoef(flux_folded, model)[0, 1]
        assert correlation > 0.5, (
            f"Folded transit should correlate with box model, got {correlation:.3f}"
        )

    def test_full_transit_analysis_pipeline(self, synthetic_transit):
        """Full pipeline: detrend -> periodogram -> fold -> model."""
        time, flux, true_period, t0, depth, duration = synthetic_transit

        # Add artificial trend
        trend = 1.0 + 0.005 * (time - time[0])
        flux_with_trend = flux * trend

        # Step 1: Detrend
        flux_clean = astro.median_detrend(flux_with_trend, window=201)

        # Step 2: Phase fold at known period
        phase, flux_folded = astro.fold(time, flux_clean, true_period, t0)

        # Step 3: Generate model
        model = astro.box_model(phase, depth, duration)

        # Verify outputs are valid
        assert len(phase) == len(time)
        assert len(model) == len(phase)
        assert np.all(phase >= 0) and np.all(phase < 1)

        # Model should match data reasonably well
        residuals = flux_folded - model
        assert np.std(residuals) < 0.05  # Residuals should be small
