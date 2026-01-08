"""Unit tests for pixel PRF-lite model construction and weight evaluation.

Tests the PRF-lite module which provides simplified Point Response Function
modeling for TESS pixel-level host identification using 2D Gaussian approximations.

All tests use synthetic data only - no network access or FITS files required.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from bittr_tess_vetter.compute.pixel_prf_lite import (
    build_prf_model,
    build_prf_model_at_pixels,
    evaluate_prf_weights,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def standard_shape() -> tuple[int, int]:
    """Standard 15x15 pixel stamp for tests."""
    return (15, 15)


@pytest.fixture
def center_coords() -> tuple[float, float]:
    """Center coordinates for 15x15 stamp."""
    return (7.0, 7.0)


# =============================================================================
# build_prf_model Tests
# =============================================================================


class TestBuildPrfModel:
    """Tests for the build_prf_model function."""

    def test_build_prf_model_normalized(
        self, standard_shape: tuple[int, int], center_coords: tuple[float, float]
    ) -> None:
        """PRF model should be normalized so sum equals 1.0."""
        center_row, center_col = center_coords
        model = build_prf_model(center_row, center_col, standard_shape, sigma=1.5)

        # Model sum should be approximately 1.0
        assert_allclose(model.sum(), 1.0, rtol=1e-10)

    def test_build_prf_model_centered(
        self, standard_shape: tuple[int, int], center_coords: tuple[float, float]
    ) -> None:
        """PRF model peak should be at the specified center."""
        center_row, center_col = center_coords
        model = build_prf_model(center_row, center_col, standard_shape, sigma=1.5)

        # Find peak location
        peak_idx = np.unravel_index(np.argmax(model), model.shape)

        # Peak should be at (7, 7) for centered model
        assert peak_idx[0] == int(center_row)
        assert peak_idx[1] == int(center_col)

    def test_build_prf_model_off_center(self, standard_shape: tuple[int, int]) -> None:
        """PRF model peak should follow off-center coordinates."""
        # Off-center position
        center_row, center_col = 3.0, 10.0
        model = build_prf_model(center_row, center_col, standard_shape, sigma=1.5)

        # Find peak location
        peak_idx = np.unravel_index(np.argmax(model), model.shape)

        # Peak should be at (3, 10)
        assert peak_idx[0] == 3
        assert peak_idx[1] == 10

    def test_build_prf_model_fractional_center(self, standard_shape: tuple[int, int]) -> None:
        """PRF model should handle fractional center coordinates."""
        # Fractional position between pixels
        center_row, center_col = 7.5, 7.5
        model = build_prf_model(center_row, center_col, standard_shape, sigma=1.5)

        # Model should still be normalized
        assert_allclose(model.sum(), 1.0, rtol=1e-10)

        # Peak should be at one of the adjacent pixels
        peak_idx = np.unravel_index(np.argmax(model), model.shape)
        assert peak_idx[0] in (7, 8)
        assert peak_idx[1] in (7, 8)

    def test_build_prf_model_shape(self, standard_shape: tuple[int, int]) -> None:
        """PRF model should have the requested shape."""
        model = build_prf_model(7.0, 7.0, standard_shape, sigma=1.5)
        assert model.shape == standard_shape

        # Test different shapes
        for shape in [(11, 11), (15, 15), (21, 21), (10, 15)]:
            model = build_prf_model(5.0, 5.0, shape, sigma=1.5)
            assert model.shape == shape

    def test_build_prf_model_dtype(
        self, standard_shape: tuple[int, int], center_coords: tuple[float, float]
    ) -> None:
        """PRF model should be float64."""
        center_row, center_col = center_coords
        model = build_prf_model(center_row, center_col, standard_shape, sigma=1.5)
        assert model.dtype == np.float64

    def test_build_prf_model_sigma_effect(self, standard_shape: tuple[int, int]) -> None:
        """Larger sigma should produce wider PSF (more flux spread)."""
        center_row, center_col = 7.0, 7.0

        model_narrow = build_prf_model(center_row, center_col, standard_shape, sigma=0.5)
        model_wide = build_prf_model(center_row, center_col, standard_shape, sigma=3.0)

        # Both should be normalized
        assert_allclose(model_narrow.sum(), 1.0, rtol=1e-10)
        assert_allclose(model_wide.sum(), 1.0, rtol=1e-10)

        # Narrow PSF should have higher peak value
        assert model_narrow.max() > model_wide.max()

        # Check flux in central 3x3 region - narrow should capture more
        central_narrow = model_narrow[6:9, 6:9].sum()
        central_wide = model_wide[6:9, 6:9].sum()
        assert central_narrow > central_wide

    def test_build_prf_model_invalid_sigma(self, standard_shape: tuple[int, int]) -> None:
        """Non-positive sigma should raise ValueError."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            build_prf_model(7.0, 7.0, standard_shape, sigma=0.0)

        with pytest.raises(ValueError, match="sigma must be positive"):
            build_prf_model(7.0, 7.0, standard_shape, sigma=-1.0)

    def test_build_prf_model_invalid_shape(self) -> None:
        """Non-positive shape dimensions should raise ValueError."""
        with pytest.raises(ValueError, match="dimensions must be positive"):
            build_prf_model(7.0, 7.0, (0, 15), sigma=1.5)

        with pytest.raises(ValueError, match="dimensions must be positive"):
            build_prf_model(7.0, 7.0, (15, -1), sigma=1.5)


# =============================================================================
# build_prf_model_at_pixels Tests
# =============================================================================


class TestBuildPrfModelAtPixels:
    """Tests for the build_prf_model_at_pixels function."""

    def test_at_pixels_basic(self) -> None:
        """Model at specific pixels returns correct values."""
        center_row, center_col = 5.0, 5.0
        pixel_rows = np.array([5, 5, 6], dtype=np.intp)
        pixel_cols = np.array([5, 6, 5], dtype=np.intp)

        values = build_prf_model_at_pixels(
            center_row, center_col, pixel_rows, pixel_cols, sigma=1.5
        )

        # Center pixel (5, 5) should have highest value
        assert values[0] > values[1]  # Center > (5, 6)
        assert values[0] > values[2]  # Center > (6, 5)

        # (5, 6) and (6, 5) should be equal (same distance from center)
        assert_allclose(values[1], values[2], rtol=1e-10)

    def test_at_pixels_empty(self) -> None:
        """Empty pixel arrays return empty result."""
        pixel_rows = np.array([], dtype=np.intp)
        pixel_cols = np.array([], dtype=np.intp)

        values = build_prf_model_at_pixels(5.0, 5.0, pixel_rows, pixel_cols, sigma=1.5)

        assert len(values) == 0
        assert values.dtype == np.float64

    def test_at_pixels_mismatched_lengths(self) -> None:
        """Mismatched array lengths raise ValueError."""
        pixel_rows = np.array([5, 6, 7], dtype=np.intp)
        pixel_cols = np.array([5, 6], dtype=np.intp)  # One fewer

        with pytest.raises(ValueError, match="same length"):
            build_prf_model_at_pixels(5.0, 5.0, pixel_rows, pixel_cols, sigma=1.5)

    def test_at_pixels_invalid_sigma(self) -> None:
        """Non-positive sigma should raise ValueError."""
        pixel_rows = np.array([5], dtype=np.intp)
        pixel_cols = np.array([5], dtype=np.intp)

        with pytest.raises(ValueError, match="sigma must be positive"):
            build_prf_model_at_pixels(5.0, 5.0, pixel_rows, pixel_cols, sigma=0.0)

    def test_at_pixels_consistency_with_full_model(self) -> None:
        """Values should preserve relative ordering consistent with full model."""
        center_row, center_col = 5.0, 5.0
        sigma = 1.5

        # Sample some pixels at varying distances from center
        pixel_rows = np.array([3, 5, 7, 2, 8], dtype=np.intp)
        pixel_cols = np.array([4, 5, 6, 2, 8], dtype=np.intp)

        values = build_prf_model_at_pixels(
            center_row, center_col, pixel_rows, pixel_cols, sigma=sigma
        )

        # Verify relative ordering is preserved (center should have highest value)
        # Note: build_prf_model_at_pixels is unnormalized, so we check relative order
        # instead of direct value comparison with full model
        center_idx = np.where((pixel_rows == 5) & (pixel_cols == 5))[0][0]
        assert values[center_idx] == values.max()


# =============================================================================
# evaluate_prf_weights Tests
# =============================================================================


class TestEvaluatePrfWeights:
    """Tests for the evaluate_prf_weights function."""

    def test_evaluate_prf_weights_full_aperture(
        self, standard_shape: tuple[int, int], center_coords: tuple[float, float]
    ) -> None:
        """Full aperture should capture approximately all flux (weight ~1.0)."""
        center_row, center_col = center_coords
        model = build_prf_model(center_row, center_col, standard_shape, sigma=1.5)

        # Full aperture mask (all pixels)
        full_mask = np.ones(standard_shape, dtype=bool)

        weight = evaluate_prf_weights(model, full_mask)

        # Full aperture should capture 100% of flux
        assert_allclose(weight, 1.0, rtol=1e-10)

    def test_evaluate_prf_weights_partial(
        self, standard_shape: tuple[int, int], center_coords: tuple[float, float]
    ) -> None:
        """Partial aperture should capture less than 100% of flux."""
        center_row, center_col = center_coords
        model = build_prf_model(center_row, center_col, standard_shape, sigma=1.5)

        # Small central aperture (3x3)
        partial_mask = np.zeros(standard_shape, dtype=bool)
        partial_mask[6:9, 6:9] = True

        weight = evaluate_prf_weights(model, partial_mask)

        # Should capture significant flux but not all
        assert 0.0 < weight < 1.0
        # Central 3x3 should capture substantial flux for sigma=1.5
        # (For Gaussian with sigma=1.5, 3x3 centered aperture captures ~48%)
        assert weight > 0.4

    def test_evaluate_prf_weights_empty_aperture(
        self, standard_shape: tuple[int, int], center_coords: tuple[float, float]
    ) -> None:
        """Empty aperture should capture zero flux."""
        center_row, center_col = center_coords
        model = build_prf_model(center_row, center_col, standard_shape, sigma=1.5)

        # Empty mask
        empty_mask = np.zeros(standard_shape, dtype=bool)

        weight = evaluate_prf_weights(model, empty_mask)

        assert weight == 0.0

    def test_evaluate_prf_weights_off_center_aperture(
        self, standard_shape: tuple[int, int]
    ) -> None:
        """Off-center aperture captures less flux than centered aperture."""
        # PSF centered at (7, 7)
        model = build_prf_model(7.0, 7.0, standard_shape, sigma=1.5)

        # Centered 3x3 aperture
        centered_mask = np.zeros(standard_shape, dtype=bool)
        centered_mask[6:9, 6:9] = True

        # Off-center 3x3 aperture (shifted by 3 pixels)
        off_center_mask = np.zeros(standard_shape, dtype=bool)
        off_center_mask[9:12, 9:12] = True

        weight_centered = evaluate_prf_weights(model, centered_mask)
        weight_off_center = evaluate_prf_weights(model, off_center_mask)

        # Centered aperture should capture more flux
        assert weight_centered > weight_off_center

    def test_evaluate_prf_weights_shape_mismatch(
        self, standard_shape: tuple[int, int], center_coords: tuple[float, float]
    ) -> None:
        """Mismatched shapes should raise ValueError."""
        center_row, center_col = center_coords
        model = build_prf_model(center_row, center_col, standard_shape, sigma=1.5)

        # Wrong shape mask
        wrong_mask = np.ones((10, 10), dtype=bool)

        with pytest.raises(ValueError, match="same shape"):
            evaluate_prf_weights(model, wrong_mask)

    def test_evaluate_prf_weights_zero_flux_model(self, standard_shape: tuple[int, int]) -> None:
        """Zero flux model should return 0.0 weight."""
        model = np.zeros(standard_shape, dtype=np.float64)
        mask = np.ones(standard_shape, dtype=bool)

        weight = evaluate_prf_weights(model, mask)

        assert weight == 0.0

    def test_evaluate_prf_weights_aperture_family(
        self, standard_shape: tuple[int, int], center_coords: tuple[float, float]
    ) -> None:
        """Weights should increase with aperture size (aperture family test)."""
        center_row, center_col = center_coords
        model = build_prf_model(center_row, center_col, standard_shape, sigma=1.5)

        # Create circular aperture masks of increasing radius
        row_grid, col_grid = np.mgrid[0 : standard_shape[0], 0 : standard_shape[1]]
        dist_sq = (row_grid - center_row) ** 2 + (col_grid - center_col) ** 2

        weights = []
        for radius in [1.0, 2.0, 3.0, 4.0, 5.0]:
            mask = dist_sq <= radius**2
            weight = evaluate_prf_weights(model, mask)
            weights.append(weight)

        # Weights should be monotonically increasing
        for i in range(len(weights) - 1):
            assert weights[i] < weights[i + 1], (
                f"Weight at radius {i + 1} ({weights[i]:.4f}) should be less than "
                f"weight at radius {i + 2} ({weights[i + 1]:.4f})"
            )


# =============================================================================
# Determinism Tests
# =============================================================================


class TestPrfDeterminism:
    """Tests for deterministic behavior of PRF functions."""

    def test_build_prf_model_deterministic(
        self, standard_shape: tuple[int, int], center_coords: tuple[float, float]
    ) -> None:
        """PRF model should be identical for same inputs."""
        center_row, center_col = center_coords

        model1 = build_prf_model(center_row, center_col, standard_shape, sigma=1.5)
        model2 = build_prf_model(center_row, center_col, standard_shape, sigma=1.5)

        assert_allclose(model1, model2, rtol=0, atol=0)

    def test_evaluate_prf_weights_deterministic(
        self, standard_shape: tuple[int, int], center_coords: tuple[float, float]
    ) -> None:
        """PRF weights should be identical for same inputs."""
        center_row, center_col = center_coords
        model = build_prf_model(center_row, center_col, standard_shape, sigma=1.5)
        mask = np.zeros(standard_shape, dtype=bool)
        mask[5:10, 5:10] = True

        weight1 = evaluate_prf_weights(model, mask)
        weight2 = evaluate_prf_weights(model, mask)

        assert weight1 == weight2
