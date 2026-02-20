"""Unit tests for transit recovery primitives.

Tests for:
- estimate_rotation_period: Rotation period detection
- remove_stellar_variability: Harmonic subtraction
- stack_transits: Phase-folding and binning
- fit_trapezoid: Trapezoid model fitting

Note: Transit masking now uses TLS transit_mask() - see transitleastsquares.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from numpy.typing import NDArray

from tess_vetter.recovery.primitives import (
    count_transits,
    detrend_for_recovery,
    estimate_rotation_period,
    fit_trapezoid,
    remove_stellar_variability,
    stack_transits,
)

TLS_AVAILABLE = importlib.util.find_spec("transitleastsquares") is not None
WOTAN_AVAILABLE = importlib.util.find_spec("wotan") is not None


class TestEstimateRotationPeriod:
    """Tests for rotation period estimation."""

    def test_detects_known_period(
        self, simple_rotation_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Correctly identifies injected rotation signal."""
        time = simple_rotation_lc["time"]
        flux = simple_rotation_lc["flux"]
        true_period = simple_rotation_lc["rotation_period"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(true_period, float)

        period, snr = estimate_rotation_period(time, flux)

        assert abs(period - true_period) < 0.1  # Within 0.1 days
        assert snr > 10  # Strong detection

    def test_handles_multi_peaked(self) -> None:
        """Handles multi-peaked rotation (spots at different longitudes)."""
        time = np.linspace(0, 100, 10000, dtype=np.float64)
        flux = 1.0 + 0.03 * np.sin(2 * np.pi * time / 5.0)
        flux += 0.02 * np.sin(4 * np.pi * time / 5.0)  # First harmonic
        flux = flux.astype(np.float64)

        period, snr = estimate_rotation_period(time, flux)

        # Should detect fundamental or its harmonic
        assert abs(period - 5.0) < 0.2 or abs(period - 2.5) < 0.1

    def test_returns_lower_snr_for_flat(self, flat_lc: dict[str, NDArray[np.float64]]) -> None:
        """Returns lower SNR for flat light curve than for periodic signal."""
        time = flat_lc["time"]
        flux = flat_lc["flux"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)

        _, snr_flat = estimate_rotation_period(time, flux)

        # Compare with strong periodic signal
        flux_periodic = 1.0 + 0.05 * np.sin(2 * np.pi * time / 5.0)
        _, snr_periodic = estimate_rotation_period(time, flux_periodic.astype(np.float64))

        # Flat should have lower SNR than periodic
        assert snr_flat < snr_periodic

    def test_uses_known_period_hint(self) -> None:
        """Uses known_period hint when detected period is harmonic."""
        time = np.linspace(0, 100, 10000, dtype=np.float64)
        # Signal at 2.5 days (half of true period)
        flux = 1.0 + 0.05 * np.sin(2 * np.pi * time / 2.5)
        flux = flux.astype(np.float64)

        # Without hint, should find 2.5 days
        period_no_hint, _ = estimate_rotation_period(time, flux)
        assert abs(period_no_hint - 2.5) < 0.2

        # With hint at 5.0 days, should recognize 2.5 as harmonic
        period_with_hint, _ = estimate_rotation_period(time, flux, known_period=5.0)
        # Should use known period since 2.5 is half of 5.0
        assert abs(period_with_hint - 5.0) < 0.5


class TestRemoveStellarVariability:
    """Tests for stellar variability removal."""

    def test_reduces_amplitude(self) -> None:
        """Harmonic subtraction reduces variability amplitude."""
        time = np.linspace(0, 50, 5000, dtype=np.float64)
        rotation_period = 4.86
        flux = 1.0 + 0.05 * np.sin(2 * np.pi * time / rotation_period)
        flux = flux.astype(np.float64)
        transit_mask = np.zeros(len(time), dtype=np.bool_)

        detrended = remove_stellar_variability(
            time, flux, rotation_period, transit_mask, n_harmonics=3
        )

        # Amplitude should be reduced by >90%
        original_amp = float(np.max(flux) - np.min(flux))
        detrended_amp = float(np.max(detrended) - np.min(detrended))

        assert detrended_amp < 0.1 * original_amp

    def test_preserves_transit(self) -> None:
        """Transit signal is preserved after detrending."""
        time = np.linspace(0, 50, 5000, dtype=np.float64)
        rotation_period = 4.86
        transit_period = 8.46
        transit_depth = 0.004

        # Create light curve with rotation + transit
        flux = 1.0 + 0.05 * np.sin(2 * np.pi * time / rotation_period)

        # Add transit
        phase = ((time - 1.0) / transit_period) % 1.0
        in_transit = phase < 0.015
        flux[in_transit] -= transit_depth
        flux = flux.astype(np.float64)

        transit_mask = in_transit

        detrended = remove_stellar_variability(
            time, flux, rotation_period, transit_mask, n_harmonics=3
        )

        # Transit should still be visible
        transit_flux = float(np.median(detrended[in_transit]))
        baseline_flux = float(np.median(detrended[~in_transit]))

        measured_depth = baseline_flux - transit_flux
        assert abs(measured_depth - transit_depth) < 0.001  # Within 0.1%

    def test_handles_insufficient_data(self) -> None:
        """Returns original flux when insufficient out-of-transit data."""
        time = np.linspace(0, 10, 100, dtype=np.float64)
        flux = np.ones_like(time, dtype=np.float64)
        transit_mask = np.ones(len(time), dtype=np.bool_)  # All masked

        detrended = remove_stellar_variability(time, flux, 5.0, transit_mask, n_harmonics=3)

        # Should return unchanged flux
        np.testing.assert_array_almost_equal(detrended, flux)


class TestDetrendForRecovery:
    """Tests for the unified detrend_for_recovery function."""

    def test_harmonic_method_delegates_correctly(self) -> None:
        """Harmonic method delegates to remove_stellar_variability."""
        time = np.linspace(0, 50, 5000, dtype=np.float64)
        rotation_period = 4.86
        flux = 1.0 + 0.05 * np.sin(2 * np.pi * time / rotation_period)
        flux = flux.astype(np.float64)
        transit_mask_arr = np.zeros(len(time), dtype=np.bool_)

        # Use detrend_for_recovery with harmonic method
        detrended = detrend_for_recovery(
            time, flux, transit_mask_arr, method="harmonic", rotation_period=rotation_period
        )

        # Should reduce amplitude significantly
        original_amp = float(np.max(flux) - np.min(flux))
        detrended_amp = float(np.max(detrended) - np.min(detrended))
        assert detrended_amp < 0.1 * original_amp

    def test_harmonic_requires_rotation_period(self) -> None:
        """Harmonic method raises ValueError when rotation_period is None."""
        time = np.linspace(0, 50, 5000, dtype=np.float64)
        flux = np.ones_like(time, dtype=np.float64)
        transit_mask_arr = np.zeros(len(time), dtype=np.bool_)

        with pytest.raises(ValueError, match="rotation_period is required"):
            detrend_for_recovery(time, flux, transit_mask_arr, method="harmonic")

    def test_wotan_biweight_reduces_variability(self) -> None:
        """Wotan biweight method reduces stellar variability."""
        if not WOTAN_AVAILABLE:
            pytest.skip("wotan not available")
        time = np.linspace(0, 50, 5000, dtype=np.float64)
        rotation_period = 4.86
        flux = 1.0 + 0.05 * np.sin(2 * np.pi * time / rotation_period)
        flux = flux.astype(np.float64)
        transit_mask_arr = np.zeros(len(time), dtype=np.bool_)

        # Use wotan biweight detrending
        detrended = detrend_for_recovery(
            time, flux, transit_mask_arr, method="wotan_biweight", window_length=0.5
        )

        # Should reduce variability amplitude
        original_amp = float(np.max(flux) - np.min(flux))
        detrended_amp = float(np.max(detrended) - np.min(detrended))
        assert detrended_amp < 0.5 * original_amp  # At least 50% reduction

    def test_unknown_method_raises_error(self) -> None:
        """Unknown detrend method raises ValueError."""
        time = np.linspace(0, 50, 100, dtype=np.float64)
        flux = np.ones_like(time, dtype=np.float64)
        transit_mask_arr = np.zeros(len(time), dtype=np.bool_)

        with pytest.raises(ValueError, match="Unknown detrend method"):
            detrend_for_recovery(time, flux, transit_mask_arr, method="invalid_method")

    def test_wotan_preserves_transit(self) -> None:
        """Wotan biweight preserves transit signal when using transit mask."""
        if not WOTAN_AVAILABLE:
            pytest.skip("wotan not available")
        time = np.linspace(0, 50, 5000, dtype=np.float64)
        rotation_period = 4.86
        transit_period = 8.46
        transit_depth = 0.004

        # Create light curve with rotation + transit
        flux = 1.0 + 0.05 * np.sin(2 * np.pi * time / rotation_period)

        # Add transit
        phase = ((time - 1.0) / transit_period) % 1.0
        in_transit = phase < 0.015
        flux[in_transit] -= transit_depth
        flux = flux.astype(np.float64)

        transit_mask_arr = in_transit.astype(np.bool_)

        detrended = detrend_for_recovery(
            time, flux, transit_mask_arr, method="wotan_biweight", window_length=0.5
        )

        # Transit should still be visible
        transit_flux = float(np.median(detrended[in_transit]))
        baseline_flux = float(np.median(detrended[~in_transit]))

        measured_depth = baseline_flux - transit_flux
        # Transit depth should be preserved within 50% (wotan is approximate)
        assert measured_depth > transit_depth * 0.5


class TestStackTransits:
    """Tests for transit stacking."""

    def test_snr_improvement(
        self, multi_transit_lc: dict[str, NDArray[np.float64] | float | int]
    ) -> None:
        """Stacking improves SNR by sqrt(N)."""
        time = multi_transit_lc["time"]
        flux = multi_transit_lc["flux"]
        flux_err = multi_transit_lc["flux_err"]
        period = multi_transit_lc["period"]
        t0 = multi_transit_lc["t0"]
        transit_depth = multi_transit_lc["transit_depth"]
        duration_hours = multi_transit_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(transit_depth, float)
        assert isinstance(duration_hours, float)

        stacked = stack_transits(time, flux, flux_err, period, t0, duration_hours)

        # Check that we have data in transit region
        transit_mask = np.abs(stacked.phase - 0.5) < 0.02
        baseline_mask = np.abs(stacked.phase - 0.5) > 0.04

        # There should be a dip in the transit region
        if np.any(transit_mask) and np.any(baseline_mask):
            baseline = float(np.mean(stacked.flux[baseline_mask]))
            transit_level = float(np.mean(stacked.flux[transit_mask]))
            measured_depth = baseline - transit_level

            # Should show a dip (positive depth)
            assert measured_depth > 0, "Transit dip should be visible"

        # Check n_transits is reasonable
        assert stacked.n_transits >= 5

        # Error bars should be smaller than raw noise due to stacking
        # Raw noise is 0.005, stacked should be better
        mean_err = float(np.mean(stacked.flux_err[stacked.flux_err < 0.99]))
        assert mean_err < 0.005, "Stacking should reduce errors"

    def test_returns_valid_structure(
        self, multi_transit_lc: dict[str, NDArray[np.float64] | float | int]
    ) -> None:
        """Returns StackedTransit with valid structure."""
        time = multi_transit_lc["time"]
        flux = multi_transit_lc["flux"]
        flux_err = multi_transit_lc["flux_err"]
        period = multi_transit_lc["period"]
        t0 = multi_transit_lc["t0"]
        duration_hours = multi_transit_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)

        stacked = stack_transits(time, flux, flux_err, period, t0, duration_hours)

        # Check structure
        assert len(stacked.phase) == len(stacked.flux)
        assert len(stacked.phase) == len(stacked.flux_err)
        assert len(stacked.phase) == len(stacked.n_points_per_bin)
        assert stacked.n_transits > 0


class TestFitTrapezoid:
    """Tests for trapezoid model fitting."""

    def test_recovers_depth(self) -> None:
        """Correctly recovers injected transit depth."""
        from tess_vetter.recovery.primitives import _trapezoid_model

        phase = np.linspace(0.4, 0.6, 100, dtype=np.float64)
        true_depth = 0.004
        true_duration = 0.04
        true_ingress = 0.2

        # Create trapezoid transit using the same model function
        flux = _trapezoid_model(phase, true_depth, true_duration, true_ingress)
        flux = flux.astype(np.float64)
        flux_err = np.ones_like(flux, dtype=np.float64) * 0.0005

        fit = fit_trapezoid(phase, flux, flux_err)

        assert fit.converged
        # Should recover depth within 20%
        assert abs(fit.depth - true_depth) < 0.2 * true_depth

    def test_returns_unconverged_for_flat(self) -> None:
        """Returns reasonable result for flat data (no transit)."""
        phase = np.linspace(0.4, 0.6, 100, dtype=np.float64)
        flux = np.ones_like(phase, dtype=np.float64) + np.random.normal(0, 0.001, 100)
        flux = flux.astype(np.float64)
        flux_err = np.ones_like(flux, dtype=np.float64) * 0.001

        fit = fit_trapezoid(phase, flux, flux_err)

        # Should converge but with very small depth
        assert fit.depth < 0.01  # Less than 1%

    def test_handles_insufficient_data(self) -> None:
        """Handles case with insufficient valid data."""
        phase = np.linspace(0.4, 0.6, 10, dtype=np.float64)
        flux = np.ones_like(phase, dtype=np.float64)
        flux_err = np.ones_like(flux, dtype=np.float64)  # All marked as empty

        fit = fit_trapezoid(phase, flux, flux_err)

        # Should not converge
        assert not fit.converged

    def test_recovers_duration_within_factor_of_two(self) -> None:
        """Correctly recovers injected transit duration within 2x.

        This test validates the fix for a bug where durations were
        measured as ~5x the true value due to unconstrained fitting
        bounds. The fit_trapezoid function now constrains duration
        to be within a factor of 2 of the initial estimate.
        """
        from tess_vetter.recovery.primitives import _trapezoid_model

        phase = np.linspace(0.4, 0.6, 100, dtype=np.float64)
        true_depth = 0.004  # 4000 ppm
        true_duration = 0.02  # 2% of orbital phase
        true_ingress = 0.2

        # Create trapezoid transit using the same model function
        flux = _trapezoid_model(phase, true_depth, true_duration, true_ingress)
        flux = flux.astype(np.float64)
        flux_err = np.ones_like(flux, dtype=np.float64) * 0.0005

        # Use true duration as initial guess (as would be done in recover_transit)
        fit = fit_trapezoid(
            phase,
            flux,
            flux_err,
            initial_depth=0.003,  # Initial guess close to true
            initial_duration_phase=true_duration,  # Use known duration
        )

        assert fit.converged
        # Duration should be recovered within 50% (more lenient than 2x constraint)
        assert abs(fit.duration_phase - true_duration) < 0.5 * true_duration, (
            f"Duration mismatch: measured {fit.duration_phase:.4f}, expected {true_duration:.4f}"
        )

    def test_duration_constraint_prevents_runaway(self) -> None:
        """Duration constraint prevents fitting unrealistically wide transits.

        This test simulates the AU Mic scenario where stellar variability
        residuals could cause the fitter to find a wide, shallow transit.
        The duration constraint should prevent this.
        """
        from tess_vetter.recovery.primitives import _trapezoid_model

        phase = np.linspace(0.4, 0.6, 100, dtype=np.float64)
        true_depth = 0.003  # 3000 ppm
        true_duration = 0.017  # ~1.7% of phase (like AU Mic b)
        true_ingress = 0.2

        # Create the true transit
        flux = _trapezoid_model(phase, true_depth, true_duration, true_ingress)

        # Add a slight baseline slope to simulate imperfect detrending
        # This could cause the fitter to prefer a wider transit
        baseline_slope = 0.001 * (phase - 0.5)
        flux = flux + baseline_slope

        flux = flux.astype(np.float64)
        flux_err = np.ones_like(flux, dtype=np.float64) * 0.0005

        fit = fit_trapezoid(
            phase,
            flux,
            flux_err,
            initial_depth=0.003,
            initial_duration_phase=true_duration,
            duration_constraint_factor=2.0,  # Constrain to 0.5x to 2x
        )

        assert fit.converged
        # With constraint, duration should stay within 2x of initial
        assert fit.duration_phase <= true_duration * 2.0, (
            f"Duration {fit.duration_phase:.4f} exceeded 2x constraint "
            f"(max {true_duration * 2.0:.4f})"
        )
        assert fit.duration_phase >= true_duration / 2.0, (
            f"Duration {fit.duration_phase:.4f} below 0.5x constraint "
            f"(min {true_duration / 2.0:.4f})"
        )


class TestCountTransits:
    """Tests for transit counting."""

    def test_counts_transits_correctly(self) -> None:
        """Counts correct number of transit epochs."""
        time = np.linspace(0, 100, 1000, dtype=np.float64)
        period = 10.0
        t0 = 5.0

        n_transits = count_transits(time, period, t0)

        # The count includes all unique transit epochs covered by the data
        # floor((t - t0) / period) gives transit number:
        # t=0: floor(-5/10) = -1 (transit at t=-5 is transit -1)
        # t=5: floor(0/10) = 0 (transit at t=5)
        # t=95: floor(90/10) = 9 (transit at t=95)
        # t=100: floor(95/10) = 9 (same transit)
        # Unique transit numbers: -1, 0, 1, ..., 9 = 11 transits
        assert n_transits == 11

    def test_handles_partial_coverage(self) -> None:
        """Handles partial transit coverage."""
        time = np.linspace(0, 25, 250, dtype=np.float64)
        period = 10.0
        t0 = 5.0

        n_transits = count_transits(time, period, t0)

        # floor((t - t0) / period):
        # t=0: floor(-5/10) = -1
        # t=5: floor(0/10) = 0
        # t=15: floor(10/10) = 1
        # t=25: floor(20/10) = 2
        # Unique: -1, 0, 1, 2 = 4 transits
        assert n_transits == 4


class TestIntegration:
    """Integration tests combining multiple primitives."""

    def test_full_pipeline_synthetic_data(
        self, synthetic_active_star_lc: dict[str, NDArray[np.float64] | float | int]
    ) -> None:
        """Full pipeline recovers transit from synthetic active star."""
        if not TLS_AVAILABLE:
            pytest.skip("transitleastsquares not available")

        from transitleastsquares import transit_mask

        time = synthetic_active_star_lc["time"]
        flux = synthetic_active_star_lc["flux"]
        flux_err = synthetic_active_star_lc["flux_err"]
        rotation_period = synthetic_active_star_lc["rotation_period"]
        transit_period = synthetic_active_star_lc["transit_period"]
        transit_depth = synthetic_active_star_lc["transit_depth"]
        t0 = synthetic_active_star_lc["t0"]
        duration_hours = synthetic_active_star_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(rotation_period, float)
        assert isinstance(transit_period, float)
        assert isinstance(transit_depth, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)

        # Step 1: Estimate rotation period
        est_rotation, rot_snr = estimate_rotation_period(time, flux)
        assert rot_snr > 5  # Should detect rotation

        # Step 2: Build transit mask using TLS transit_mask
        # TLS transit_mask args: (time, period, duration_in_days, t0)
        duration_days_wide = (duration_hours * 1.5) / 24.0
        tr_mask = transit_mask(time, transit_period, duration_days_wide, t0)

        # Step 3: Remove stellar variability
        detrended = remove_stellar_variability(time, flux, rotation_period, tr_mask, n_harmonics=3)

        # Variability should be reduced
        original_std = float(np.std(flux))
        detrended_std = float(np.std(detrended))
        assert detrended_std < original_std * 0.5

        # Step 4: Stack transits
        stacked = stack_transits(time, detrended, flux_err, transit_period, t0, duration_hours)
        assert stacked.n_transits >= 3

        # Step 5: Fit trapezoid
        fit = fit_trapezoid(stacked.phase, stacked.flux, stacked.flux_err)
        assert fit.converged

        # Should recover depth within 50% (synthetic data with noise)
        assert abs(fit.depth - transit_depth) < 0.5 * transit_depth

        # Detection SNR should be significant
        snr = fit.depth / fit.depth_err if fit.depth_err > 0 else 0
        assert snr > 3.0
