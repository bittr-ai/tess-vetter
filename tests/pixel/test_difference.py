"""Tests for tess_vetter.pixel.difference module.

Comprehensive tests for difference imaging including:
- TransitParams validation
- DifferenceImageResult creation
- compute_difference_image algorithm
- Localization score computation
- Edge cases and error handling
"""

from __future__ import annotations

import numpy as np
import pytest

from tess_vetter.pixel.difference import (
    DifferenceImageResult,
    TransitParams,
    compute_difference_image,
)

# =============================================================================
# Test Fixtures - Helper functions for creating test data
# =============================================================================


def make_simple_tpf(
    n_times: int = 100,
    n_rows: int = 11,
    n_cols: int = 11,
    target_flux: float = 1000.0,
    background_flux: float = 100.0,
    noise_level: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create a simple TPF with target star at center.

    Args:
        n_times: Number of time points
        n_rows: Number of rows in TPF
        n_cols: Number of columns in TPF
        target_flux: Flux of target star at center pixel
        background_flux: Flux of background pixels
        noise_level: Standard deviation of Gaussian noise (0 = no noise)
        rng: Random number generator for reproducibility

    Returns:
        3D array of shape (n_times, n_rows, n_cols)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Start with uniform background
    tpf = np.full((n_times, n_rows, n_cols), background_flux, dtype=np.float64)

    # Add target star at center
    center_row = n_rows // 2
    center_col = n_cols // 2
    tpf[:, center_row, center_col] = target_flux

    # Add noise if requested
    if noise_level > 0:
        tpf += rng.normal(0, noise_level, tpf.shape)

    return tpf


def inject_transit(
    tpf: np.ndarray,
    time: np.ndarray,
    transit_params: TransitParams,
    transit_depth: float,
    source_row: int | None = None,
    source_col: int | None = None,
) -> np.ndarray:
    """Inject a transit signal into TPF data.

    Args:
        tpf: 3D TPF array (time, rows, cols)
        time: 1D time array
        transit_params: Transit parameters
        transit_depth: Fractional depth of transit (0 to 1)
        source_row: Row of transiting source (default: center)
        source_col: Column of transiting source (default: center)

    Returns:
        TPF with transit injected
    """
    tpf = tpf.copy()
    n_times, n_rows, n_cols = tpf.shape

    if source_row is None:
        source_row = n_rows // 2
    if source_col is None:
        source_col = n_cols // 2

    # Compute phase
    phase = ((time - transit_params.t0) % transit_params.period) / transit_params.period
    phase = np.where(phase > 0.5, phase - 1.0, phase)

    # Transit mask
    half_duration_phase = (transit_params.duration / 2) / transit_params.period
    in_transit = np.abs(phase) <= half_duration_phase

    # Apply transit (reduce flux during transit)
    tpf[in_transit, source_row, source_col] *= 1 - transit_depth

    return tpf


# =============================================================================
# TransitParams Tests
# =============================================================================


class TestTransitParams:
    """Tests for TransitParams dataclass."""

    def test_valid_transit_params(self) -> None:
        """Valid parameters create TransitParams successfully."""
        params = TransitParams(period=3.0, t0=100.0, duration=0.2)
        assert params.period == 3.0
        assert params.t0 == 100.0
        assert params.duration == 0.2

    def test_transit_params_immutable(self) -> None:
        """TransitParams is frozen/immutable."""
        params = TransitParams(period=3.0, t0=100.0, duration=0.2)
        with pytest.raises(AttributeError):
            params.period = 5.0

    @pytest.mark.parametrize("invalid_period", [0.0, -1.0, -0.001])
    def test_period_must_be_positive(self, invalid_period: float) -> None:
        """Period must be strictly positive."""
        with pytest.raises(ValueError, match="period must be positive"):
            TransitParams(period=invalid_period, t0=100.0, duration=0.2)

    @pytest.mark.parametrize("invalid_duration", [0.0, -1.0, -0.001])
    def test_duration_must_be_positive(self, invalid_duration: float) -> None:
        """Duration must be strictly positive."""
        with pytest.raises(ValueError, match="duration must be positive"):
            TransitParams(period=3.0, t0=100.0, duration=invalid_duration)

    def test_duration_must_be_less_than_period(self) -> None:
        """Duration must be less than period."""
        with pytest.raises(ValueError, match="duration.*must be less than period"):
            TransitParams(period=1.0, t0=100.0, duration=1.0)

        with pytest.raises(ValueError, match="duration.*must be less than period"):
            TransitParams(period=1.0, t0=100.0, duration=1.5)

    def test_t0_can_be_any_float(self) -> None:
        """t0 can be negative, zero, or positive."""
        # Negative t0
        params1 = TransitParams(period=3.0, t0=-100.0, duration=0.2)
        assert params1.t0 == -100.0

        # Zero t0
        params2 = TransitParams(period=3.0, t0=0.0, duration=0.2)
        assert params2.t0 == 0.0

        # Positive t0
        params3 = TransitParams(period=3.0, t0=100.0, duration=0.2)
        assert params3.t0 == 100.0


# =============================================================================
# DifferenceImageResult Tests
# =============================================================================


class TestDifferenceImageResult:
    """Tests for DifferenceImageResult dataclass."""

    def test_valid_result_creation(self) -> None:
        """Valid DifferenceImageResult creation succeeds."""
        diff_image = np.zeros((5, 5))
        result = DifferenceImageResult(
            difference_image=diff_image,
            localization_score=0.95,
            brightest_pixel_coords=(2, 2),
            target_coords=(2, 2),
            distance_to_target=0.0,
        )
        assert result.localization_score == 0.95
        assert result.brightest_pixel_coords == (2, 2)
        assert result.target_coords == (2, 2)
        assert result.distance_to_target == 0.0
        assert result.difference_image.shape == (5, 5)

    def test_result_immutable(self) -> None:
        """DifferenceImageResult is frozen/immutable."""
        diff_image = np.zeros((5, 5))
        result = DifferenceImageResult(
            difference_image=diff_image,
            localization_score=0.95,
            brightest_pixel_coords=(2, 2),
            target_coords=(2, 2),
            distance_to_target=0.0,
        )
        with pytest.raises(AttributeError):
            result.localization_score = 0.5


# =============================================================================
# compute_difference_image Basic Tests
# =============================================================================


class TestComputeDifferenceImageBasic:
    """Basic tests for compute_difference_image function."""

    def test_nan_cadences_are_ignored(self) -> None:
        """All-NaN cadences should be dropped and not corrupt localization."""
        n_times, n_rows, n_cols = 200, 11, 11
        time = np.linspace(0, 30, n_times)
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        tpf = make_simple_tpf(n_times=n_times, n_rows=n_rows, n_cols=n_cols)
        tpf = inject_transit(tpf, time, transit_params, transit_depth=0.02)

        baseline = compute_difference_image(tpf, time, transit_params)

        # Poison a small number of in- and out-of-transit cadences.
        phase = ((time - transit_params.t0) % transit_params.period) / transit_params.period
        phase = np.where(phase > 0.5, phase - 1.0, phase)
        in_transit = np.abs(phase) <= (transit_params.duration / 2) / transit_params.period

        in_idx = np.where(in_transit)[0][:2]
        out_idx = np.where(~in_transit)[0][:5]
        bad_idx = np.unique(np.concatenate([in_idx, out_idx]))

        tpf_bad = np.array(tpf, copy=True)
        tpf_bad[bad_idx] = np.nan

        bad = compute_difference_image(tpf_bad, time, transit_params)

        assert bad.brightest_pixel_coords == baseline.brightest_pixel_coords
        assert abs(bad.localization_score - baseline.localization_score) < 0.05

    def test_transit_at_target_gives_high_localization(self) -> None:
        """Transit at target star produces high localization score."""
        # Create TPF with target at center
        n_times, n_rows, n_cols = 100, 11, 11
        time = np.linspace(0, 30, n_times)  # 30 days of data
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        tpf = make_simple_tpf(n_times=n_times, n_rows=n_rows, n_cols=n_cols)
        tpf = inject_transit(tpf, time, transit_params, transit_depth=0.01)

        result = compute_difference_image(tpf, time, transit_params)

        # Target is at center (5, 5)
        assert result.target_coords == (5, 5)
        # Brightest pixel should be at or very close to target
        assert result.distance_to_target < 1.0
        # High localization score
        assert result.localization_score > 0.9

    def test_transit_off_target_gives_low_localization(self) -> None:
        """Transit on background star produces low localization score."""
        n_times, n_rows, n_cols = 100, 11, 11
        time = np.linspace(0, 30, n_times)
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        # Create TPF with additional bright star in corner
        tpf = make_simple_tpf(n_times=n_times, n_rows=n_rows, n_cols=n_cols)
        # Add bright contaminating star at corner
        tpf[:, 1, 1] = 800.0

        # Inject transit on the contaminating star, not target
        tpf = inject_transit(
            tpf, time, transit_params, transit_depth=0.02, source_row=1, source_col=1
        )

        result = compute_difference_image(tpf, time, transit_params)

        # Brightest pixel should be at the contaminating star
        assert result.brightest_pixel_coords == (1, 1)
        # Distance to target (center at 5,5) should be significant
        expected_distance = np.sqrt((5 - 1) ** 2 + (5 - 1) ** 2)
        assert np.isclose(result.distance_to_target, expected_distance, atol=0.01)
        # Lower localization score due to off-target transit
        assert result.localization_score < 0.5

    def test_difference_image_shape_matches_tpf_spatial(self) -> None:
        """Difference image has same spatial shape as TPF."""
        n_times, n_rows, n_cols = 50, 7, 9
        time = np.linspace(0, 15, n_times)
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        tpf = make_simple_tpf(n_times=n_times, n_rows=n_rows, n_cols=n_cols)
        tpf = inject_transit(tpf, time, transit_params, transit_depth=0.01)

        result = compute_difference_image(tpf, time, transit_params)

        assert result.difference_image.shape == (n_rows, n_cols)

    def test_no_transit_gives_flat_difference(self) -> None:
        """TPF with no transit signal gives near-zero difference image."""
        n_times, n_rows, n_cols = 100, 11, 11
        time = np.linspace(0, 30, n_times)
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        # TPF with no transit
        tpf = make_simple_tpf(n_times=n_times, n_rows=n_rows, n_cols=n_cols)

        result = compute_difference_image(tpf, time, transit_params)

        # Difference image should be essentially zero everywhere
        assert np.allclose(result.difference_image, 0.0, atol=1e-10)


# =============================================================================
# Localization Score Tests
# =============================================================================


class TestLocalizationScore:
    """Tests for localization score computation."""

    def test_perfect_localization_score_at_target(self) -> None:
        """Brightest pixel exactly at target gives localization near 1.0."""
        n_times, n_rows, n_cols = 100, 11, 11
        time = np.linspace(0, 30, n_times)
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        tpf = make_simple_tpf(n_times=n_times, n_rows=n_rows, n_cols=n_cols)
        tpf = inject_transit(tpf, time, transit_params, transit_depth=0.01)

        result = compute_difference_image(tpf, time, transit_params)

        # Brightest should be at center
        assert result.brightest_pixel_coords == (5, 5)
        assert result.localization_score == 1.0

    def test_localization_score_bounded_0_to_1(self) -> None:
        """Localization score is always between 0 and 1."""
        n_times, n_rows, n_cols = 100, 11, 11
        time = np.linspace(0, 30, n_times)
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        # Test with transit at various locations
        for row, col in [(5, 5), (0, 0), (10, 10), (0, 10), (3, 7)]:
            tpf = make_simple_tpf(n_times=n_times, n_rows=n_rows, n_cols=n_cols)
            tpf[:, row, col] = 900.0  # Add source at this location
            tpf = inject_transit(
                tpf, time, transit_params, transit_depth=0.02, source_row=row, source_col=col
            )

            result = compute_difference_image(tpf, time, transit_params)

            assert 0.0 <= result.localization_score <= 1.0

    def test_localization_score_decreases_with_distance(self) -> None:
        """Localization score decreases as brightest pixel moves from target."""
        n_times, n_rows, n_cols = 100, 11, 11
        time = np.linspace(0, 30, n_times)
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        scores = []
        positions = [(5, 5), (5, 6), (5, 8), (5, 10)]  # Increasing distance from center

        for row, col in positions:
            tpf = make_simple_tpf(n_times=n_times, n_rows=n_rows, n_cols=n_cols)
            # Make this position the brightest by making the star there brighter
            tpf[:, row, col] = 2000.0
            tpf = inject_transit(
                tpf, time, transit_params, transit_depth=0.02, source_row=row, source_col=col
            )

            result = compute_difference_image(tpf, time, transit_params)
            scores.append(result.localization_score)

        # Scores should decrease monotonically (or stay same if distance didn't change)
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation in compute_difference_image."""

    def test_tpf_must_be_3d(self) -> None:
        """TPF data must be 3-dimensional."""
        time = np.linspace(0, 30, 100)
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        # 2D array
        with pytest.raises(ValueError, match="tpf_data must be 3D"):
            compute_difference_image(np.zeros((10, 10)), time, transit_params)

        # 1D array
        with pytest.raises(ValueError, match="tpf_data must be 3D"):
            compute_difference_image(np.zeros(100), time, transit_params)

        # 4D array
        with pytest.raises(ValueError, match="tpf_data must be 3D"):
            compute_difference_image(np.zeros((10, 5, 5, 3)), time, transit_params)

    def test_time_must_be_1d(self) -> None:
        """Time array must be 1-dimensional."""
        tpf = np.zeros((100, 11, 11))
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        # 2D time array
        with pytest.raises(ValueError, match="time must be 1D"):
            compute_difference_image(tpf, np.zeros((100, 2)), transit_params)

    def test_time_length_must_match_tpf(self) -> None:
        """Time array length must match TPF first dimension."""
        tpf = np.zeros((100, 11, 11))
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        # Wrong length
        with pytest.raises(ValueError, match="time length.*must match"):
            compute_difference_image(tpf, np.zeros(50), transit_params)

    def test_empty_spatial_dimensions_rejected(self) -> None:
        """TPF with zero spatial dimensions is rejected."""
        time = np.linspace(0, 30, 100)
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        with pytest.raises(ValueError, match="spatial dimensions must be non-zero"):
            compute_difference_image(np.zeros((100, 0, 11)), time, transit_params)

        with pytest.raises(ValueError, match="spatial dimensions must be non-zero"):
            compute_difference_image(np.zeros((100, 11, 0)), time, transit_params)

    def test_no_in_transit_data_raises_error(self) -> None:
        """Error when no data points fall in transit."""
        n_times = 10
        # Very short time span that doesn't cover a full transit window
        time = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        # Transit at t0=5.0 with period=10.0, so transit window is at phase 0.5
        # Our time points are at phase 0.0-0.09, far from transit
        transit_params = TransitParams(period=10.0, t0=5.0, duration=0.05)

        tpf = make_simple_tpf(n_times=n_times)

        with pytest.raises(ValueError, match="No in-transit data"):
            compute_difference_image(tpf, time, transit_params)

    def test_no_out_of_transit_data_raises_error(self) -> None:
        """Error when all data points are in transit (duration too long)."""
        n_times = 10
        # All points clustered around transit center
        time = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
        # Duration that covers more than all our time points
        transit_params = TransitParams(period=10.0, t0=0.05, duration=0.2)

        tpf = make_simple_tpf(n_times=n_times)

        with pytest.raises(ValueError, match="No out-of-transit data"):
            compute_difference_image(tpf, time, transit_params)


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and corner cases."""

    def test_single_transit_in_data(self) -> None:
        """Works correctly with just one transit in the data."""
        n_times = 50
        time = np.linspace(0, 2, n_times)  # Only covers ~0.67 periods
        transit_params = TransitParams(period=3.0, t0=1.0, duration=0.2)

        tpf = make_simple_tpf(n_times=n_times)
        tpf = inject_transit(tpf, time, transit_params, transit_depth=0.01)

        result = compute_difference_image(tpf, time, transit_params)

        # Should still work with single transit
        assert result.difference_image.shape == (11, 11)
        assert 0.0 <= result.localization_score <= 1.0

    def test_small_tpf(self) -> None:
        """Works with small TPF (e.g., 3x3)."""
        n_times, n_rows, n_cols = 100, 3, 3
        time = np.linspace(0, 30, n_times)
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        tpf = make_simple_tpf(n_times=n_times, n_rows=n_rows, n_cols=n_cols)
        tpf = inject_transit(tpf, time, transit_params, transit_depth=0.01)

        result = compute_difference_image(tpf, time, transit_params)

        assert result.difference_image.shape == (3, 3)
        assert result.target_coords == (1, 1)

    def test_single_pixel_tpf(self) -> None:
        """Works with degenerate 1x1 TPF."""
        n_times = 100
        time = np.linspace(0, 30, n_times)
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        # 1x1 TPF
        tpf = np.full((n_times, 1, 1), 1000.0)
        # Inject transit manually
        phase = ((time - transit_params.t0) % transit_params.period) / transit_params.period
        phase = np.where(phase > 0.5, phase - 1.0, phase)
        in_transit = np.abs(phase) <= (transit_params.duration / 2) / transit_params.period
        tpf[in_transit, 0, 0] *= 0.99

        result = compute_difference_image(tpf, time, transit_params)

        assert result.difference_image.shape == (1, 1)
        assert result.brightest_pixel_coords == (0, 0)
        assert result.target_coords == (0, 0)
        assert result.distance_to_target == 0.0
        assert result.localization_score == 1.0

    def test_non_square_tpf(self) -> None:
        """Works with non-square TPF."""
        n_times, n_rows, n_cols = 100, 7, 13
        time = np.linspace(0, 30, n_times)
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        tpf = make_simple_tpf(n_times=n_times, n_rows=n_rows, n_cols=n_cols)
        tpf = inject_transit(tpf, time, transit_params, transit_depth=0.01)

        result = compute_difference_image(tpf, time, transit_params)

        assert result.difference_image.shape == (7, 13)
        assert result.target_coords == (3, 6)  # Center of 7x13

    def test_negative_epoch(self) -> None:
        """Works with negative epoch (t0)."""
        n_times = 100
        time = np.linspace(-50, -20, n_times)
        transit_params = TransitParams(period=3.0, t0=-35.0, duration=0.2)

        tpf = make_simple_tpf(n_times=n_times)
        tpf = inject_transit(tpf, time, transit_params, transit_depth=0.01)

        result = compute_difference_image(tpf, time, transit_params)

        assert result.brightest_pixel_coords == (5, 5)
        assert result.localization_score > 0.9

    def test_with_noise(self) -> None:
        """Works correctly with noisy data."""
        rng = np.random.default_rng(42)
        n_times = 200
        time = np.linspace(0, 60, n_times)
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        # Add significant noise
        tpf = make_simple_tpf(
            n_times=n_times,
            noise_level=10.0,  # 1% of target flux
            rng=rng,
        )
        tpf = inject_transit(tpf, time, transit_params, transit_depth=0.01)

        result = compute_difference_image(tpf, time, transit_params)

        # Should still localize reasonably well despite noise
        assert result.distance_to_target < 2.0
        assert result.localization_score > 0.7


# =============================================================================
# Algorithm Correctness Tests
# =============================================================================


class TestAlgorithmCorrectness:
    """Tests verifying the algorithm computes expected values."""

    def test_difference_is_out_minus_in(self) -> None:
        """Difference image is out_of_transit - in_transit."""
        n_times = 100
        time = np.linspace(0, 30, n_times)
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        tpf = make_simple_tpf(n_times=n_times)
        # Deep transit to make difference clear
        tpf = inject_transit(tpf, time, transit_params, transit_depth=0.10)

        result = compute_difference_image(tpf, time, transit_params)

        # At target (center), difference should be positive because:
        # out_of_transit flux > in_transit flux (transit causes dimming)
        center_row, center_col = 5, 5
        assert result.difference_image[center_row, center_col] > 0

        # Expected difference at center: 10% of 1000 = 100
        expected_diff = 0.10 * 1000.0
        assert np.isclose(result.difference_image[center_row, center_col], expected_diff, rtol=0.01)

    def test_uses_median_not_mean(self) -> None:
        """Algorithm uses median, which is robust to outliers."""
        n_times = 100
        time = np.linspace(0, 30, n_times)
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        tpf = make_simple_tpf(n_times=n_times)
        tpf = inject_transit(tpf, time, transit_params, transit_depth=0.01)

        # Add extreme outlier in one frame
        tpf[50, :, :] = 1e6

        result = compute_difference_image(tpf, time, transit_params)

        # Result should not be dominated by outlier
        assert np.all(np.abs(result.difference_image) < 1e5)

    def test_distance_calculation(self) -> None:
        """Distance to target is computed correctly."""
        n_times, n_rows, n_cols = 100, 11, 11
        time = np.linspace(0, 30, n_times)
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        # Place transit source at known offset
        tpf = make_simple_tpf(n_times=n_times, n_rows=n_rows, n_cols=n_cols)
        source_row, source_col = 8, 5  # 3 pixels away from center (5, 5)
        tpf[:, source_row, source_col] = 1500.0
        tpf = inject_transit(
            tpf,
            time,
            transit_params,
            transit_depth=0.02,
            source_row=source_row,
            source_col=source_col,
        )

        result = compute_difference_image(tpf, time, transit_params)

        # Expected distance from (8,5) to (5,5)
        expected_distance = np.sqrt((8 - 5) ** 2 + (5 - 5) ** 2)
        assert np.isclose(result.distance_to_target, expected_distance)


# =============================================================================
# Determinism Tests
# =============================================================================


class TestDeterminism:
    """Tests verifying deterministic behavior."""

    def test_same_input_gives_same_output(self) -> None:
        """Same inputs always produce same outputs."""
        n_times = 100
        time = np.linspace(0, 30, n_times)
        transit_params = TransitParams(period=3.0, t0=1.5, duration=0.2)

        tpf = make_simple_tpf(n_times=n_times)
        tpf = inject_transit(tpf, time, transit_params, transit_depth=0.01)

        result1 = compute_difference_image(tpf, time, transit_params)
        result2 = compute_difference_image(tpf, time, transit_params)

        np.testing.assert_array_equal(result1.difference_image, result2.difference_image)
        assert result1.localization_score == result2.localization_score
        assert result1.brightest_pixel_coords == result2.brightest_pixel_coords
        assert result1.target_coords == result2.target_coords
        assert result1.distance_to_target == result2.distance_to_target
