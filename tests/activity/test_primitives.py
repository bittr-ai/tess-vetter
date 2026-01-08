"""Unit tests for stellar activity characterization primitives.

Tests for:
- detect_flares: Flare detection using sigma-clipping
- measure_rotation_period: Rotation period with uncertainty
- classify_variability: Variability classification
- compute_activity_index: Photometric activity proxy
- mask_flares: Flare masking/interpolation
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bittr_tess_vetter.activity.primitives import (
    classify_variability,
    compute_activity_index,
    compute_phase_amplitude,
    detect_flares,
    generate_recommendation,
    mask_flares,
    measure_rotation_period,
)


class TestDetectFlares:
    """Tests for flare detection."""

    def test_detects_injected_flares(
        self, flare_star_lc: dict[str, NDArray[np.float64] | float | int]
    ) -> None:
        """Detects majority of injected flares."""
        time = flare_star_lc["time"]
        flux = flare_star_lc["flux"]
        flux_err = flare_star_lc["flux_err"]
        n_injected = flare_star_lc["n_flares_injected"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(n_injected, int)

        flares = detect_flares(time, flux, flux_err, sigma_threshold=5.0)

        # Should detect at least 50% of injected flares
        assert len(flares) >= n_injected // 2

    def test_detects_single_flare(
        self, single_flare_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Detects a single prominent flare."""
        time = single_flare_lc["time"]
        flux = single_flare_lc["flux"]
        flux_err = single_flare_lc["flux_err"]
        flare_time = single_flare_lc["flare_time"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(flare_time, float)

        flares = detect_flares(time, flux, flux_err, sigma_threshold=5.0)

        assert len(flares) >= 1

        # Check that detected flare is near the injected time
        detected_times = [f.peak_time for f in flares]
        closest = min(abs(t - flare_time) for t in detected_times)
        assert closest < 0.1  # Within ~2.4 hours

    def test_no_false_positives_on_quiet_star(
        self, quiet_star_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """No flare detections on quiet star with only noise."""
        time = quiet_star_lc["time"]
        flux = quiet_star_lc["flux"]
        flux_err = quiet_star_lc["flux_err"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)

        flares = detect_flares(time, flux, flux_err, sigma_threshold=5.0)

        # Should detect zero or very few false positives
        assert len(flares) <= 1

    def test_flare_properties_reasonable(
        self, single_flare_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Detected flare has reasonable properties."""
        time = single_flare_lc["time"]
        flux = single_flare_lc["flux"]
        flux_err = single_flare_lc["flux_err"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)

        flares = detect_flares(time, flux, flux_err, sigma_threshold=5.0)

        assert len(flares) >= 1
        flare = flares[0]

        # Check properties
        assert flare.start_time < flare.peak_time <= flare.end_time
        assert flare.amplitude > 0
        assert flare.duration_minutes > 0
        assert flare.energy_estimate > 0

    def test_returns_empty_for_short_data(self) -> None:
        """Returns empty list for insufficient data."""
        time = np.linspace(0, 1, 50, dtype=np.float64)
        flux = np.ones_like(time, dtype=np.float64)
        flux_err = np.ones_like(flux, dtype=np.float64) * 0.001

        flares = detect_flares(time, flux, flux_err)

        assert len(flares) == 0


class TestMeasureRotationPeriod:
    """Tests for rotation period measurement."""

    def test_detects_known_period(
        self, spotted_rotator_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Correctly identifies injected rotation period."""
        time = spotted_rotator_lc["time"]
        flux = spotted_rotator_lc["flux"]
        true_period = spotted_rotator_lc["rotation_period"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(true_period, float)

        period, period_err, snr = measure_rotation_period(time, flux)

        # Should find period within 5%
        assert abs(period - true_period) < 0.05 * true_period
        # Should have high SNR
        assert snr > 10
        # Error should be reasonable
        assert period_err < 0.5

    def test_returns_uncertainty(
        self, spotted_rotator_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Returns uncertainty estimate."""
        time = spotted_rotator_lc["time"]
        flux = spotted_rotator_lc["flux"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)

        _, period_err, _ = measure_rotation_period(time, flux)

        # Uncertainty should be positive and reasonable
        assert period_err > 0
        assert period_err < 1.0  # Less than 1 day error for strong signal

    def test_low_snr_for_quiet_star(
        self, quiet_star_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Returns low SNR for quiet star."""
        time = quiet_star_lc["time"]
        flux = quiet_star_lc["flux"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)

        _, _, snr = measure_rotation_period(time, flux)

        # SNR should be lower than for spotted rotator
        # (may still have some peaks from noise)
        assert snr < 15

    def test_handles_short_data(self) -> None:
        """Handles short data gracefully."""
        time = np.linspace(0, 1, 50, dtype=np.float64)
        flux = np.ones_like(time, dtype=np.float64)

        period, period_err, snr = measure_rotation_period(time, flux)

        # Should return defaults
        assert period == 1.0
        assert period_err == 1.0
        assert snr == 0.0


class TestClassifyVariability:
    """Tests for variability classification."""

    def test_classifies_spotted_rotator(self) -> None:
        """Classifies strong periodic signal as spotted_rotator."""
        result = classify_variability(
            periodogram_power=15.0,  # Strong LS peak
            phase_amplitude=0.05,  # 5% amplitude
            flare_count=2,
            baseline_days=30.0,
        )

        assert result == "spotted_rotator"

    def test_classifies_flare_star(self) -> None:
        """Classifies high flare rate as flare_star."""
        result = classify_variability(
            periodogram_power=3.0,
            phase_amplitude=0.01,
            flare_count=20,  # High flare count
            baseline_days=30.0,  # Results in >0.5/day rate
        )

        assert result == "flare_star"

    def test_classifies_quiet_star(self) -> None:
        """Classifies low variability as quiet."""
        result = classify_variability(
            periodogram_power=2.0,  # Low SNR
            phase_amplitude=0.0003,  # <500 ppm
            flare_count=0,
            baseline_days=30.0,
        )

        assert result == "quiet"

    def test_priority_of_classifications(self) -> None:
        """Flare star takes priority over spotted rotator."""
        # Even with strong rotation, high flare rate should classify as flare_star
        result = classify_variability(
            periodogram_power=15.0,  # Strong rotation
            phase_amplitude=0.05,
            flare_count=30,  # Very high flare count
            baseline_days=30.0,  # 1/day rate
        )

        assert result == "flare_star"


class TestComputeActivityIndex:
    """Tests for activity index computation."""

    def test_quiet_star_low_index(self) -> None:
        """Quiet star has low activity index."""
        index = compute_activity_index(
            variability_ppm=200.0,  # Low variability
            rotation_period=25.0,  # Slow rotation
            flare_rate=0.01,  # Low flare rate
        )

        assert index < 0.3

    def test_active_star_high_index(self) -> None:
        """Active star has high activity index."""
        index = compute_activity_index(
            variability_ppm=50000.0,  # High variability
            rotation_period=2.0,  # Fast rotation
            flare_rate=1.0,  # High flare rate
        )

        assert index > 0.7

    def test_index_bounds(self) -> None:
        """Activity index is between 0 and 1."""
        # Extreme quiet
        index_quiet = compute_activity_index(
            variability_ppm=10.0,
            rotation_period=50.0,
            flare_rate=0.001,
        )

        # Extreme active
        index_active = compute_activity_index(
            variability_ppm=100000.0,
            rotation_period=0.5,
            flare_rate=10.0,
        )

        assert 0.0 <= index_quiet <= 1.0
        assert 0.0 <= index_active <= 1.0


class TestMaskFlares:
    """Tests for flare masking."""

    def test_masks_flare_regions(
        self, single_flare_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Replaces flare regions with interpolated baseline."""
        time = single_flare_lc["time"]
        flux = single_flare_lc["flux"]
        flux_err = single_flare_lc["flux_err"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)

        # Detect flares first
        flares = detect_flares(time, flux, flux_err, sigma_threshold=5.0)
        assert len(flares) >= 1

        # Mask the flares
        masked_flux = mask_flares(time, flux, flares)

        # Masked flux should have lower max than original
        assert np.max(masked_flux) < np.max(flux)

        # Masked flux should be smoother (lower std in flare region)
        flare = flares[0]
        flare_mask = (time >= flare.start_time) & (time <= flare.end_time)
        assert np.std(masked_flux[flare_mask]) < np.std(flux[flare_mask])

    def test_preserves_baseline(
        self, single_flare_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Preserves flux outside flare regions."""
        time = single_flare_lc["time"]
        flux = single_flare_lc["flux"]
        flux_err = single_flare_lc["flux_err"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)

        # Detect flares
        flares = detect_flares(time, flux, flux_err, sigma_threshold=5.0)
        assert len(flares) >= 1

        # Mask flares
        masked_flux = mask_flares(time, flux, flares, buffer_minutes=5.0)

        # Create mask for regions far from flares
        far_from_flares = np.ones(len(time), dtype=bool)
        for flare in flares:
            buffer_days = 10.0 / (24.0 * 60.0)  # 10 minutes buffer
            near_flare = (time >= flare.start_time - buffer_days) & (
                time <= flare.end_time + buffer_days
            )
            far_from_flares &= ~near_flare

        # Baseline should be unchanged
        np.testing.assert_array_almost_equal(
            masked_flux[far_from_flares], flux[far_from_flares], decimal=10
        )

    def test_empty_flare_list(self) -> None:
        """Returns original flux when no flares."""
        time = np.linspace(0, 10, 1000, dtype=np.float64)
        flux = np.ones_like(time, dtype=np.float64)

        masked_flux = mask_flares(time, flux, [])

        np.testing.assert_array_equal(masked_flux, flux)


class TestComputePhaseAmplitude:
    """Tests for phase amplitude computation."""

    def test_measures_amplitude(
        self, spotted_rotator_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Correctly measures phase curve amplitude."""
        time = spotted_rotator_lc["time"]
        flux = spotted_rotator_lc["flux"]
        rotation_period = spotted_rotator_lc["rotation_period"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(rotation_period, float)

        amplitude = compute_phase_amplitude(time, flux, rotation_period)

        # Should be close to 2 * 0.05 + 2 * 0.02 = 0.14 (max - min)
        # But with noise and binning, expect ~0.10-0.18
        assert 0.08 < amplitude < 0.20

    def test_low_amplitude_for_quiet(
        self, quiet_star_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Returns low amplitude for quiet star."""
        time = quiet_star_lc["time"]
        flux = quiet_star_lc["flux"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)

        amplitude = compute_phase_amplitude(time, flux, period=5.0)

        # Should be small (just noise)
        assert amplitude < 0.001


class TestGenerateRecommendation:
    """Tests for recommendation generation."""

    def test_quiet_star_recommendation(self) -> None:
        """Generates appropriate recommendation for quiet star."""
        rec, params = generate_recommendation(
            variability_class="quiet",
            variability_ppm=200.0,
            rotation_period=25.0,
            flare_rate=0.01,
            activity_index=0.1,
        )

        assert "Standard transit search" in rec or "Low stellar activity" in rec
        assert "min_detectable_depth_ppm" in params

    def test_active_star_recommendation(self) -> None:
        """Generates appropriate recommendation for active star."""
        rec, params = generate_recommendation(
            variability_class="spotted_rotator",
            variability_ppm=50000.0,
            rotation_period=4.86,
            flare_rate=0.1,
            activity_index=0.8,
        )

        assert "recover_transit" in rec or "rotation_period" in rec
        assert "rotation_period" in params
        assert "n_harmonics" in params
        assert "min_detectable_depth_ppm" in params
        assert params["rotation_period"] == 4.86

    def test_flare_star_recommendation(self) -> None:
        """Generates appropriate recommendation for flare star."""
        rec, params = generate_recommendation(
            variability_class="flare_star",
            variability_ppm=2000.0,
            rotation_period=10.0,
            flare_rate=2.0,
            activity_index=0.7,
        )

        assert "flare" in rec.lower()
        assert "use_flare_masking" in params
        assert params["use_flare_masking"] is True

    def test_suggested_params_structure(self) -> None:
        """Verify suggested_params contains expected fields."""
        rec, params = generate_recommendation(
            variability_class="spotted_rotator",
            variability_ppm=30000.0,
            rotation_period=5.0,
            flare_rate=0.2,
            activity_index=0.7,
            n_expected_transits=15,
        )

        # Check all expected fields are present
        assert isinstance(params, dict)
        assert "rotation_period" in params
        assert "n_harmonics" in params
        assert "min_detectable_depth_ppm" in params

        # Check types
        assert isinstance(params["rotation_period"], float)
        assert isinstance(params["n_harmonics"], int)
        assert isinstance(params["min_detectable_depth_ppm"], int)

    def test_min_depth_scales_with_transits(self) -> None:
        """Verify min_detectable_depth decreases with more transits."""
        _, params_few = generate_recommendation(
            variability_class="spotted_rotator",
            variability_ppm=10000.0,
            rotation_period=5.0,
            flare_rate=0.1,
            activity_index=0.6,
            n_expected_transits=4,
        )

        _, params_many = generate_recommendation(
            variability_class="spotted_rotator",
            variability_ppm=10000.0,
            rotation_period=5.0,
            flare_rate=0.1,
            activity_index=0.6,
            n_expected_transits=25,
        )

        # More transits should result in lower min detectable depth
        assert params_many["min_detectable_depth_ppm"] < params_few["min_detectable_depth_ppm"]


class TestIntegration:
    """Integration tests combining multiple primitives."""

    def test_full_characterization_active_star(
        self, spotted_rotator_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Full characterization pipeline for active star."""
        time = spotted_rotator_lc["time"]
        flux = spotted_rotator_lc["flux"]
        flux_err = spotted_rotator_lc["flux_err"]
        true_period = spotted_rotator_lc["rotation_period"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(true_period, float)

        # Measure rotation
        period, period_err, snr = measure_rotation_period(time, flux)
        assert abs(period - true_period) < 0.1 * true_period

        # Detect flares (should be few/none in this synthetic data)
        flares = detect_flares(time, flux, flux_err)
        flare_rate = len(flares) / 50.0  # 50 days baseline

        # Compute amplitude
        amplitude = compute_phase_amplitude(time, flux, period)

        # Classify
        variability_class = classify_variability(snr, amplitude, len(flares), 50.0)
        assert variability_class == "spotted_rotator"

        # Compute index
        variability_ppm = float(np.std(flux) * 1e6)
        index = compute_activity_index(variability_ppm, period, flare_rate)
        assert index > 0.4  # Active star

        # Generate recommendation
        rec, params = generate_recommendation(
            variability_class, variability_ppm, period, flare_rate, index
        )
        assert len(rec) > 0
        assert isinstance(params, dict)

    def test_full_characterization_quiet_star(
        self, quiet_star_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Full characterization pipeline for quiet star."""
        time = quiet_star_lc["time"]
        flux = quiet_star_lc["flux"]
        flux_err = quiet_star_lc["flux_err"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)

        # Measure rotation (no strong signal expected)
        period, _, snr = measure_rotation_period(time, flux)

        # Detect flares
        flares = detect_flares(time, flux, flux_err)
        flare_rate = len(flares) / 27.0

        # Compute amplitude
        amplitude = compute_phase_amplitude(time, flux, period)

        # Classify
        variability_class = classify_variability(snr, amplitude, len(flares), 27.0)
        assert variability_class == "quiet"

        # Compute index
        variability_ppm = float(np.std(flux) * 1e6)
        index = compute_activity_index(variability_ppm, period, flare_rate)
        assert index < 0.4  # Quiet star
