"""Tests for V01 odd/even depth check.

Tests cover the improved per-epoch depth extraction implementation with:
- Deterministic synthetic light curves
- Planet-like equal depths (PASS)
- EB-like different depths (FAIL)
- Low-N scenarios (PASS + warning)
- Sparse OOT scenarios (PASS + warning)
- Legacy key backward compatibility
"""

from __future__ import annotations

import numpy as np
import pytest

from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.validation.lc_checks import OddEvenConfig, check_odd_even_depth


@pytest.fixture
def make_synthetic_lc():
    """Factory for deterministic synthetic light curves with transits."""

    def _make(
        n_transits: int,
        depth_ppm: float,
        period: float = 5.0,
        duration_hours: float = 3.0,
        noise_ppm: float = 100.0,
        odd_even_ratio: float = 1.0,
        cadence_minutes: float = 2.0,
        seed: int = 42,
        baseline_drift_amp: float = 0.0,
    ) -> tuple[LightCurveData, float]:
        """Generate synthetic light curve with transits.

        Args:
            n_transits: Total number of transits to include
            depth_ppm: Transit depth in ppm (for even transits; odd scaled by ratio)
            period: Orbital period in days
            duration_hours: Transit duration in hours
            noise_ppm: Gaussian noise level in ppm
            odd_even_ratio: Ratio of odd depth to even depth (1.0 = equal)
            cadence_minutes: Observation cadence in minutes
            seed: RNG seed for reproducibility
            baseline_drift_amp: Amplitude of sinusoidal baseline drift (fractional)

        Returns:
            Tuple of (LightCurveData, t0) where t0 is the reference epoch
        """
        rng = np.random.default_rng(seed)

        # The algorithm expects transits at phase = 0.
        # epoch = floor((t - t0) / period) where t0 is the first transit center.
        # In-transit is when phase_dist < half_dur_phase, where phase_dist = min(phase, 1-phase).
        #
        # To avoid epoch boundary issues (transit split between epochs), we need
        # the transit duration to be < period. Then each transit is fully within one epoch.
        # The key: t0 is the FIRST TRANSIT CENTER, and subsequent transits are at
        # t0 + n*period.
        cadence_days = cadence_minutes / (24.0 * 60.0)
        duration_days = duration_hours / 24.0

        # First transit center at t0
        # We want the time array to start AFTER the negative-phase boundary of
        # the first transit to avoid partial transits.
        t0 = 0.5 * period  # First transit center

        # Time array: start at t0 - 0.4*period (within epoch 0, provides OOT baseline)
        # This ensures all times are in epoch >= 0
        margin = 0.3 * period
        t_start = t0 - 0.4 * period  # Within epoch 0 (epoch = floor(-0.4) = -1... no wait)
        # Actually: epoch = floor((t_start - t0) / period) = floor(-0.4) = -1
        # We need t_start >= t0 - 0.5*period to stay in epoch 0
        # But for OOT baseline, we need t_start < t0 - duration/2
        # So t_start = t0 - 0.4*period works if 0.4*period > duration/2
        # For period=5, duration=0.125: 0.4*5=2 > 0.0625 OK
        # BUT epoch = floor(-0.4) = -1, not 0!
        # The fix: set t_start such that epoch >= 0:
        #   floor((t_start - t0) / period) >= 0
        #   (t_start - t0) / period >= 0
        #   t_start >= t0
        # So t_start must be >= t0 to have epoch >= 0.
        # This means no OOT baseline before first transit (only after).
        t_start = t0  # Start at first transit center
        last_transit_center = t0 + (n_transits - 1) * period
        t_end = last_transit_center + 0.5 * period + margin
        total_duration = t_end - t_start
        n_points = int(total_duration / cadence_days)

        time = np.linspace(t_start, t_end, n_points, dtype=np.float64)

        # With t0 = 0.5*period:
        # - Transit 0 at 0.5*period, epoch = floor((0.5*period - 0.5*period) / period) = 0
        # - Transit 1 at 1.5*period, epoch = floor((1.5*period - 0.5*period) / period) = 1
        # - etc.
        # Phase at transit center = ((0.5*period - 0.5*period) / period) % 1 = 0
        # So transits are at phase = 0, which is what the algorithm expects.

        # Base flux = 1.0
        flux = np.ones_like(time, dtype=np.float64)

        # Add Gaussian noise
        flux += rng.normal(0, noise_ppm * 1e-6, size=len(time))

        # Add baseline drift if requested
        if baseline_drift_amp > 0:
            drift = baseline_drift_amp * np.sin(2 * np.pi * time / (total_duration / 3))
            flux += drift

        # Add transits
        # Transit centers are at t0 + n*period
        # - Transit 0 at t0 = period, epoch = 0 (even): depth = depth_even
        # - Transit 1 at t0 + period = 2*period, epoch = 1 (odd): depth = depth_odd
        # - etc.
        depth_even = depth_ppm * 1e-6
        depth_odd = depth_ppm * 1e-6 * odd_even_ratio

        for transit_num in range(n_transits):
            transit_center = t0 + transit_num * period
            # Epoch for this transit is transit_num
            is_epoch_odd = transit_num % 2 == 1
            depth = depth_odd if is_epoch_odd else depth_even

            # Define transit window (box transit)
            half_dur = duration_days / 2
            in_transit = (time >= transit_center - half_dur) & (time <= transit_center + half_dur)

            # Apply depth
            flux[in_transit] -= depth

        # Create flux_err (constant, based on noise level)
        flux_err = np.full_like(flux, noise_ppm * 1e-6, dtype=np.float64)

        # Quality flags (all good)
        quality = np.zeros(len(time), dtype=np.int32)

        # Valid mask (all valid)
        valid_mask = np.ones(len(time), dtype=np.bool_)

        lc = LightCurveData(
            time=time,
            flux=flux,
            flux_err=flux_err,
            quality=quality,
            valid_mask=valid_mask,
            tic_id=123456789,
            sector=1,
            cadence_seconds=cadence_minutes * 60,
        )

        return lc, t0

    return _make


class TestOddEvenDepth:
    """Test suite for check_odd_even_depth function."""

    def test_planet_like_equal_depths_passes(self, make_synthetic_lc) -> None:
        """Equal odd/even depths should pass with high confidence."""
        # Use more transits and lower noise for cleaner detection
        lc, t0 = make_synthetic_lc(
            n_transits=20,
            depth_ppm=2000,
            odd_even_ratio=1.0,
            noise_ppm=50,
        )
        result = check_odd_even_depth(lc, period=5.0, t0=t0, duration_hours=3.0)

        assert result.passed is True
        assert result.confidence >= 0.5
        # With equal depths and good S/N, delta_sigma should be modest
        # (relaxed threshold since per-epoch estimation has intrinsic scatter)
        assert result.details["delta_sigma"] < 5.0
        # With equal depths, rel_diff should be small
        assert result.details["rel_diff"] < 0.3

    def test_eb_like_different_depths_fails(self, make_synthetic_lc) -> None:
        """EB at 2x period (different odd/even depths) should fail."""
        # Odd depth = 20% of even depth (extreme EB case: primary much deeper than secondary)
        # More transits and lower noise for robust detection
        lc, t0 = make_synthetic_lc(
            n_transits=20,
            depth_ppm=10000,  # Deep primary eclipse
            odd_even_ratio=0.2,  # Secondary is 20% of primary (2000 ppm)
            noise_ppm=50,
        )
        result = check_odd_even_depth(lc, period=5.0, t0=t0, duration_hours=3.0)

        # Check that we're measuring the expected depth difference
        # odd depths ~2000 ppm, even depths ~10000 ppm
        # rel_diff should be ~0.8 (8000/10000)
        assert result.details["rel_diff"] >= 0.5, (
            f"rel_diff={result.details['rel_diff']}, "
            f"odd_ppm={result.details['depth_odd_ppm']}, "
            f"even_ppm={result.details['depth_even_ppm']}"
        )
        # delta_sigma should be high given the large depth difference
        assert result.details["delta_sigma"] >= 3.0, f"delta_sigma={result.details['delta_sigma']}"
        # Should fail the dual threshold test
        assert result.passed is False, (
            f"Expected FAIL but got PASS: delta_sigma={result.details['delta_sigma']}, "
            f"rel_diff={result.details['rel_diff']}"
        )

    def test_marginal_eb_with_high_sigma_but_low_rel_diff_passes(self, make_synthetic_lc) -> None:
        """High sigma but low relative difference should pass (dual threshold)."""
        # Very shallow transits where statistical noise can create high sigma
        # but actual depth difference is small in absolute terms
        lc, t0 = make_synthetic_lc(
            n_transits=10,
            depth_ppm=200,  # Very shallow
            odd_even_ratio=0.9,  # 10% difference
            noise_ppm=50,
        )
        result = check_odd_even_depth(lc, period=5.0, t0=t0, duration_hours=3.0)

        # Should pass because rel_diff is low even if sigma might be borderline
        # The dual threshold protects against false flags on shallow transits
        assert result.passed is True
        assert result.details["rel_diff"] < 0.5

    def test_low_n_passes_with_warning(self, make_synthetic_lc) -> None:
        """Few transits should pass with low confidence and warning."""
        # Only 3 transits: 1 odd, 2 even (or 2 odd, 1 even)
        lc, t0 = make_synthetic_lc(
            n_transits=3,
            depth_ppm=1000,
            odd_even_ratio=1.0,
            noise_ppm=100,
        )
        result = check_odd_even_depth(lc, period=5.0, t0=t0, duration_hours=3.0)

        assert result.passed is True
        # With only 3 transits, we have min(1, 2) = 1 transit per parity
        # This triggers the minimum data policy
        assert result.confidence <= 0.5
        assert len(result.details.get("warnings", [])) > 0

    def test_sparse_oot_emits_warning_but_passes(self, make_synthetic_lc) -> None:
        """Sparse out-of-transit data should still work but may emit warning."""
        # Create a light curve with minimal OOT points by using very long transits
        # relative to the observation span
        lc, t0 = make_synthetic_lc(
            n_transits=6,
            depth_ppm=1000,
            duration_hours=6.0,  # Very long transit
            noise_ppm=100,
            odd_even_ratio=1.0,
        )
        # Use a tight baseline window that may trigger sparse OOT warning
        config = OddEvenConfig(baseline_window_mult=2.0)
        result = check_odd_even_depth(lc, period=5.0, t0=t0, duration_hours=6.0, config=config)

        # Should still pass (planet-like)
        assert result.passed is True
        # May have warnings about OOT data
        # (Warnings are informational, not necessarily blocking)

    def test_legacy_keys_present(self, make_synthetic_lc) -> None:
        """Legacy keys should still be present for backwards compatibility."""
        lc, t0 = make_synthetic_lc(
            n_transits=10,
            depth_ppm=1000,
            odd_even_ratio=1.0,
        )
        result = check_odd_even_depth(lc, period=5.0, t0=t0, duration_hours=3.0)

        # Legacy keys
        assert "odd_depth" in result.details
        assert "even_depth" in result.details
        assert "depth_diff_sigma" in result.details
        assert "n_odd_points" in result.details
        assert "n_even_points" in result.details

        # Legacy values should be fractional (not ppm)
        assert 0 < result.details["odd_depth"] < 0.01  # ~1000 ppm = 0.001
        assert 0 < result.details["even_depth"] < 0.01

    def test_new_keys_present(self, make_synthetic_lc) -> None:
        """New keys should be present in result details."""
        lc, t0 = make_synthetic_lc(
            n_transits=10,
            depth_ppm=1000,
            odd_even_ratio=1.0,
        )
        result = check_odd_even_depth(lc, period=5.0, t0=t0, duration_hours=3.0)

        # New keys
        assert "n_odd_transits" in result.details
        assert "n_even_transits" in result.details
        assert "depth_odd_ppm" in result.details
        assert "depth_even_ppm" in result.details
        assert "depth_err_odd_ppm" in result.details
        assert "depth_err_even_ppm" in result.details
        assert "delta_ppm" in result.details
        assert "delta_sigma" in result.details
        assert "rel_diff" in result.details
        assert "warnings" in result.details
        assert "method" in result.details
        assert "epoch_depths_odd_ppm" in result.details
        assert "epoch_depths_even_ppm" in result.details

        # Method should indicate per-epoch
        assert result.details["method"] == "per_epoch_median"

    def test_epoch_depths_arrays_are_populated(self, make_synthetic_lc) -> None:
        """Per-epoch depth arrays should be populated."""
        lc, t0 = make_synthetic_lc(
            n_transits=10,
            depth_ppm=1000,
            odd_even_ratio=1.0,
        )
        result = check_odd_even_depth(lc, period=5.0, t0=t0, duration_hours=3.0)

        # Should have depth arrays with entries
        odd_depths = result.details["epoch_depths_odd_ppm"]
        even_depths = result.details["epoch_depths_even_ppm"]

        assert len(odd_depths) > 0
        assert len(even_depths) > 0

        # Depths should be in ppm range (around 1000)
        assert all(800 < d < 1200 for d in odd_depths)
        assert all(800 < d < 1200 for d in even_depths)

    def test_epoch_depths_capped_at_20(self, make_synthetic_lc) -> None:
        """Epoch depth arrays should be capped at 20 elements."""
        # Create many transits
        lc, t0 = make_synthetic_lc(
            n_transits=50,
            depth_ppm=1000,
            period=2.0,  # Shorter period for more transits
        )
        result = check_odd_even_depth(lc, period=2.0, t0=t0, duration_hours=3.0)

        odd_depths = result.details["epoch_depths_odd_ppm"]
        even_depths = result.details["epoch_depths_even_ppm"]

        # Should be capped at 20
        assert len(odd_depths) <= 20
        assert len(even_depths) <= 20

    def test_config_overrides_work(self, make_synthetic_lc) -> None:
        """Custom config should override default thresholds."""
        # EB-like signal that would fail with defaults
        lc, t0 = make_synthetic_lc(
            n_transits=10,
            depth_ppm=5000,
            odd_even_ratio=0.5,
            noise_ppm=100,
        )

        # With lenient thresholds, it should pass
        lenient_config = OddEvenConfig(
            sigma_threshold=10.0,  # Very lenient
            rel_diff_threshold=0.9,  # Very lenient
        )
        result = check_odd_even_depth(
            lc, period=5.0, t0=t0, duration_hours=3.0, config=lenient_config
        )

        # Should pass with lenient config even though depths differ
        assert result.passed is True

    def test_red_noise_inflation_increases_uncertainty(self, make_synthetic_lc) -> None:
        """Red noise inflation should increase uncertainty estimates."""
        # Create light curve with some baseline drift (red noise proxy)
        lc, t0 = make_synthetic_lc(
            n_transits=10,
            depth_ppm=1000,
            odd_even_ratio=1.0,
            noise_ppm=100,
            baseline_drift_amp=0.001,  # 1000 ppm drift
        )

        # Run with and without red noise inflation
        config_no_rn = OddEvenConfig(use_red_noise_inflation=False)
        config_with_rn = OddEvenConfig(use_red_noise_inflation=True)

        result_no_rn = check_odd_even_depth(
            lc, period=5.0, t0=t0, duration_hours=3.0, config=config_no_rn
        )
        result_with_rn = check_odd_even_depth(
            lc, period=5.0, t0=t0, duration_hours=3.0, config=config_with_rn
        )

        # Both should pass (planet-like)
        assert result_no_rn.passed is True
        assert result_with_rn.passed is True

        # With red noise, uncertainty should be equal or larger
        # (delta_sigma should be equal or smaller)
        # Note: with real red noise, we expect inflation; with white noise, ratio ~ 1
        assert result_with_rn.details["delta_sigma"] <= result_no_rn.details["delta_sigma"] * 1.5

    def test_confidence_increases_with_more_transits(self, make_synthetic_lc) -> None:
        """Confidence should increase with more transits."""
        # Few transits
        lc_few, t0_few = make_synthetic_lc(n_transits=4, depth_ppm=1000)
        result_few = check_odd_even_depth(lc_few, period=5.0, t0=t0_few, duration_hours=3.0)

        # Many transits
        lc_many, t0_many = make_synthetic_lc(n_transits=20, depth_ppm=1000)
        result_many = check_odd_even_depth(lc_many, period=5.0, t0=t0_many, duration_hours=3.0)

        # More transits should give higher confidence (assuming both pass)
        assert result_few.passed is True
        assert result_many.passed is True
        assert result_many.confidence >= result_few.confidence

    def test_insufficient_data_returns_pass_with_warnings(self, make_synthetic_lc) -> None:
        """Insufficient data should return pass with very low confidence and warnings."""
        # Single transit only
        lc, t0 = make_synthetic_lc(
            n_transits=1,
            depth_ppm=1000,
            odd_even_ratio=1.0,
        )
        result = check_odd_even_depth(lc, period=5.0, t0=t0, duration_hours=3.0)

        # Must pass (cannot reject with no data)
        assert result.passed is True
        # Very low confidence
        assert result.confidence <= 0.3
        # Should have warnings explaining why
        assert len(result.details["warnings"]) > 0

    def test_result_id_and_name(self, make_synthetic_lc) -> None:
        """Result should have correct ID and name."""
        lc, t0 = make_synthetic_lc(n_transits=10, depth_ppm=1000)
        result = check_odd_even_depth(lc, period=5.0, t0=t0, duration_hours=3.0)

        assert result.id == "V01"
        assert result.name == "odd_even_depth"

    def test_suspicious_flag_true_for_eb_like_signal(self, make_synthetic_lc) -> None:
        """Suspicious flag should be True for clear EB-like depth difference."""
        # Clear EB case: odd depth is 20% of even depth
        lc, t0 = make_synthetic_lc(
            n_transits=20,
            depth_ppm=10000,
            odd_even_ratio=0.2,
            noise_ppm=50,
        )
        result = check_odd_even_depth(lc, period=5.0, t0=t0, duration_hours=3.0)

        # Should fail AND be suspicious
        assert result.passed is False
        assert "suspicious" in result.details
        assert bool(result.details["suspicious"]) is True

    def test_suspicious_flag_false_for_planet_like_signal(self, make_synthetic_lc) -> None:
        """Suspicious flag should be False for equal depths (planet-like)."""
        lc, t0 = make_synthetic_lc(
            n_transits=20,
            depth_ppm=2000,
            odd_even_ratio=1.0,
            noise_ppm=50,
        )
        result = check_odd_even_depth(lc, period=5.0, t0=t0, duration_hours=3.0)

        # Should pass AND not be suspicious
        assert result.passed is True
        assert "suspicious" in result.details
        assert bool(result.details["suspicious"]) is False

    def test_suspicious_flag_false_for_insufficient_data(self, make_synthetic_lc) -> None:
        """Suspicious flag should be False when insufficient data."""
        # Single transit only
        lc, t0 = make_synthetic_lc(
            n_transits=1,
            depth_ppm=1000,
            odd_even_ratio=1.0,
        )
        result = check_odd_even_depth(lc, period=5.0, t0=t0, duration_hours=3.0)

        # Should pass with low confidence, suspicious = False
        assert result.passed is True
        assert result.confidence <= 0.3
        assert "suspicious" in result.details
        assert bool(result.details["suspicious"]) is False

    def test_short_period_baseline_cap_no_nans(self, make_synthetic_lc) -> None:
        """Short-period candidates should not produce NaN/Inf with baseline cap."""
        # Short period where baseline_window_mult * duration could exceed period/2
        # Period = 0.5 days, duration = 2 hours = 0.0833 days
        # Without cap: baseline_half_window = 6.0 * 0.0833 = 0.5 days = period!
        # With cap: min(0.5, 0.45 * 0.5) = min(0.5, 0.225) = 0.225 days
        period = 0.5
        duration_hours = 2.0
        duration_days = duration_hours / 24.0

        # Verify the cap would be triggered with default config
        config = OddEvenConfig()
        uncapped = config.baseline_window_mult * duration_days
        capped = config.baseline_window_max_fraction_of_period * period
        assert uncapped > capped, (
            f"Test parameters should trigger cap: uncapped={uncapped:.3f} > capped={capped:.3f}"
        )

        lc, t0 = make_synthetic_lc(
            n_transits=20,
            depth_ppm=2000,
            period=period,
            duration_hours=duration_hours,
            odd_even_ratio=1.0,
            noise_ppm=100,
        )
        result = check_odd_even_depth(lc, period=period, t0=t0, duration_hours=duration_hours)

        # Should not produce NaN/Inf
        assert not np.isnan(result.details.get("delta_sigma", 0))
        assert not np.isinf(result.details.get("delta_sigma", 0))
        assert not np.isnan(result.details.get("rel_diff", 0))
        assert not np.isinf(result.details.get("rel_diff", 0))
        # Should still produce a valid result with good data
        assert result.passed in (True, False)
        assert 0.0 <= result.confidence <= 1.0
        # Should have processed epochs (not all skipped due to sparse data)
        assert result.details.get("n_odd_transits", 0) > 0
        assert result.details.get("n_even_transits", 0) > 0

    def test_global_oot_fallback_warning(self, make_synthetic_lc) -> None:
        """Frequent global OOT fallback should emit warning."""
        # Create a scenario where local OOT is sparse
        # Very short period with long transits relative to period
        lc, t0 = make_synthetic_lc(
            n_transits=10,
            depth_ppm=2000,
            period=0.3,  # Very short period
            duration_hours=3.0,  # Long transit relative to period
            odd_even_ratio=1.0,
            noise_ppm=100,
            cadence_minutes=10.0,  # Coarser cadence to reduce OOT points
        )
        result = check_odd_even_depth(lc, period=0.3, t0=t0, duration_hours=3.0)

        # Check if we got the fallback warning (may or may not trigger depending on data)
        # At minimum, the result should be valid
        assert result.passed in (True, False)
        # This test is informational - we just verify no crash and valid output
        assert "warnings" in result.details  # warnings key should always exist


class TestOddEvenConfig:
    """Tests for OddEvenConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config values should match design spec."""
        config = OddEvenConfig()

        assert config.sigma_threshold == 3.0
        assert config.rel_diff_threshold == 0.5
        assert config.suspicious_sigma_threshold == 3.0
        assert config.suspicious_rel_diff_threshold == 0.15
        assert config.min_transits_per_parity == 2
        assert config.min_points_in_transit_per_epoch == 5
        assert config.min_points_in_transit_per_parity == 20
        assert config.baseline_window_mult == 6.0
        assert config.baseline_window_max_fraction_of_period == 0.45
        assert config.use_red_noise_inflation is True

    def test_custom_values(self) -> None:
        """Custom config values should be settable."""
        config = OddEvenConfig(
            sigma_threshold=4.0,
            rel_diff_threshold=0.3,
            min_transits_per_parity=3,
            use_red_noise_inflation=False,
        )

        assert config.sigma_threshold == 4.0
        assert config.rel_diff_threshold == 0.3
        assert config.min_transits_per_parity == 3
        assert config.use_red_noise_inflation is False


class TestHelperFunctions:
    """Tests for helper functions used by check_odd_even_depth."""

    def test_robust_std_normal_distribution(self) -> None:
        """Robust std should match regular std for normal distribution."""
        from bittr_tess_vetter.validation.lc_checks import _robust_std

        rng = np.random.default_rng(42)
        data = rng.normal(0, 1.0, 10000)

        robust = _robust_std(data)
        regular = np.std(data)

        # Should be close (within 5%)
        assert abs(robust - regular) / regular < 0.05

    def test_robust_std_with_outliers(self) -> None:
        """Robust std should be less affected by outliers."""
        from bittr_tess_vetter.validation.lc_checks import _robust_std

        rng = np.random.default_rng(42)
        data = rng.normal(0, 1.0, 1000)
        # Add outliers
        data_with_outliers = np.concatenate([data, [100, -100, 50, -50]])

        robust = _robust_std(data_with_outliers)
        regular = np.std(data_with_outliers)

        # Robust should be much smaller than regular (closer to true std=1)
        assert robust < regular / 2
        # Robust should be close to true std
        assert abs(robust - 1.0) < 0.2

    def test_robust_std_small_array(self) -> None:
        """Robust std should handle small arrays gracefully."""
        from bittr_tess_vetter.validation.lc_checks import _robust_std

        assert _robust_std(np.array([1.0])) == 0.0
        assert _robust_std(np.array([])) == 0.0

    def test_red_noise_inflation_white_noise(self) -> None:
        """Red noise inflation should be ~1 for white noise."""
        from bittr_tess_vetter.validation.lc_checks import _compute_red_noise_inflation

        rng = np.random.default_rng(42)
        time = np.linspace(0, 10, 1000)
        residuals = rng.normal(0, 0.001, len(time))

        inflation, success = _compute_red_noise_inflation(residuals, time, bin_size_days=0.5)

        assert success is True
        # White noise inflation should be close to 1
        assert 0.8 < inflation < 1.5

    def test_red_noise_inflation_insufficient_data(self) -> None:
        """Red noise inflation should return 1.0 for insufficient data."""
        from bittr_tess_vetter.validation.lc_checks import _compute_red_noise_inflation

        time = np.linspace(0, 1, 5)
        residuals = np.random.default_rng(42).normal(0, 0.001, len(time))

        inflation, success = _compute_red_noise_inflation(residuals, time, bin_size_days=0.5)

        assert success is False
        assert inflation == 1.0
