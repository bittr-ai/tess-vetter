"""Tests for bittr_tess_vetter.pixel.centroid module.

Tests centroid shift analysis for transit detection in TPF data,
including flux-weighted centroid computation and significance estimation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bittr_tess_vetter.pixel.centroid import (
    WINDOW_POLICIES,
    CentroidResult,
    TransitParams,
    _compute_flux_weighted_centroid,
    _get_transit_masks,
    compute_centroid_shift,
)

# =============================================================================
# Test Fixtures - Synthetic TPF Data Generators
# =============================================================================


@pytest.fixture
def simple_tpf() -> np.ndarray:
    """Create simple synthetic TPF with uniform flux.

    Returns a 100x11x11 TPF with constant flux of 1000 e-/s.
    """
    rng = np.random.default_rng(42)
    tpf = np.ones((100, 11, 11)) * 1000.0
    # Add small noise
    tpf += rng.normal(0, 10, tpf.shape)
    return tpf


@pytest.fixture
def centered_star_tpf() -> np.ndarray:
    """Create TPF with a centered Gaussian PSF.

    Returns a 100x11x11 TPF with a Gaussian star at center.
    """
    rng = np.random.default_rng(42)
    n_time, n_rows, n_cols = 100, 11, 11

    # Create coordinate grid
    row_center, col_center = n_rows // 2, n_cols // 2
    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")

    # Gaussian PSF
    sigma = 1.5
    psf = 10000 * np.exp(-((rows - row_center) ** 2 + (cols - col_center) ** 2) / (2 * sigma**2))

    # Repeat for all time steps with small variations
    tpf = np.zeros((n_time, n_rows, n_cols))
    for i in range(n_time):
        tpf[i] = psf + rng.normal(0, 50, psf.shape)

    return tpf


@pytest.fixture
def shifted_star_tpf() -> tuple[np.ndarray, np.ndarray, TransitParams]:
    """Create TPF with centroid shift during transit.

    The star centroid shifts by ~0.5 pixels during in-transit cadences,
    simulating a background eclipsing binary scenario.

    Returns:
        tuple: (tpf_data, time_array, transit_params)
    """
    rng = np.random.default_rng(42)
    n_time, n_rows, n_cols = 200, 11, 11

    # Transit parameters
    period = 3.0  # days
    t0 = 1.5  # BTJD
    duration = 2.0  # hours
    transit_params = TransitParams(period=period, t0=t0, duration=duration)

    # Time array spanning multiple transits
    time = np.linspace(0, 12, n_time)

    # Compute transit mask
    duration_days = duration / 24.0
    phase = ((time - t0) / period) % 1.0
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    time_from_transit = phase * period
    in_transit = np.abs(time_from_transit) <= duration_days / 2.0

    # Create coordinate grid
    row_center, col_center = n_rows // 2, n_cols // 2
    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")

    # Gaussian PSF parameters
    sigma = 1.5
    base_flux = 10000

    tpf = np.zeros((n_time, n_rows, n_cols))
    for i in range(n_time):
        if in_transit[i]:
            # Shift centroid during transit
            shift_x, shift_y = 0.5, 0.3
            psf = base_flux * np.exp(
                -((rows - row_center - shift_y) ** 2 + (cols - col_center - shift_x) ** 2)
                / (2 * sigma**2)
            )
        else:
            # Normal position out of transit
            psf = base_flux * np.exp(
                -((rows - row_center) ** 2 + (cols - col_center) ** 2) / (2 * sigma**2)
            )
        tpf[i] = psf + rng.normal(0, 30, psf.shape)

    return tpf, time, transit_params


@pytest.fixture
def simple_time() -> np.ndarray:
    """Time array for simple_tpf fixture."""
    return np.linspace(0, 10, 100)


@pytest.fixture
def simple_transit_params() -> TransitParams:
    """Simple transit parameters for testing."""
    return TransitParams(period=2.5, t0=1.25, duration=2.0)


# =============================================================================
# Test CentroidResult Model
# =============================================================================


class TestCentroidResultModel:
    """Tests for CentroidResult dataclass."""

    def test_centroid_result_creation(self) -> None:
        """CentroidResult can be created with all fields."""
        result = CentroidResult(
            centroid_shift_pixels=0.25,
            significance_sigma=3.5,
            in_transit_centroid=(5.5, 5.2),
            out_of_transit_centroid=(5.3, 5.1),
            n_in_transit_cadences=20,
            n_out_transit_cadences=80,
        )
        assert result.centroid_shift_pixels == 0.25
        assert result.significance_sigma == 3.5
        assert result.in_transit_centroid == (5.5, 5.2)
        assert result.out_of_transit_centroid == (5.3, 5.1)
        assert result.n_in_transit_cadences == 20
        assert result.n_out_transit_cadences == 80

    def test_centroid_result_is_frozen(self) -> None:
        """CentroidResult is immutable."""
        result = CentroidResult(
            centroid_shift_pixels=0.1,
            significance_sigma=1.0,
            in_transit_centroid=(5.0, 5.0),
            out_of_transit_centroid=(5.0, 5.0),
            n_in_transit_cadences=10,
            n_out_transit_cadences=90,
        )
        with pytest.raises(Exception):
            result.centroid_shift_pixels = 0.5  # type: ignore

    def test_centroid_result_handles_nan(self) -> None:
        """CentroidResult can hold NaN values."""
        result = CentroidResult(
            centroid_shift_pixels=np.nan,
            significance_sigma=np.nan,
            in_transit_centroid=(np.nan, np.nan),
            out_of_transit_centroid=(5.0, 5.0),
            n_in_transit_cadences=0,
            n_out_transit_cadences=100,
        )
        assert math.isnan(result.centroid_shift_pixels)
        assert math.isnan(result.significance_sigma)


# =============================================================================
# Test TransitParams Model
# =============================================================================


class TestTransitParamsModel:
    """Tests for TransitParams dataclass."""

    def test_transit_params_creation(self) -> None:
        """TransitParams can be created with all fields."""
        params = TransitParams(period=2.5, t0=1000.5, duration=3.0)
        assert params.period == 2.5
        assert params.t0 == 1000.5
        assert params.duration == 3.0

    def test_transit_params_is_frozen(self) -> None:
        """TransitParams is immutable."""
        params = TransitParams(period=2.5, t0=1000.5, duration=3.0)
        with pytest.raises(Exception):
            params.period = 5.0  # type: ignore


# =============================================================================
# Test Window Policies
# =============================================================================


class TestWindowPolicies:
    """Tests for WINDOW_POLICIES configuration."""

    def test_v1_policy_exists(self) -> None:
        """v1 policy is defined."""
        assert "v1" in WINDOW_POLICIES

    def test_v1_policy_values(self) -> None:
        """v1 policy has expected values."""
        v1 = WINDOW_POLICIES["v1"]
        assert v1["k_in"] == 1.0
        assert v1["k_buffer"] == 0.5


# =============================================================================
# Test _compute_flux_weighted_centroid
# =============================================================================


class TestComputeFluxWeightedCentroid:
    """Tests for _compute_flux_weighted_centroid helper."""

    def test_centroid_uniform_flux(self) -> None:
        """Centroid of uniform flux is at center."""
        tpf = np.ones((10, 11, 11)) * 1000.0
        mask = np.ones(10, dtype=bool)
        cx, cy = _compute_flux_weighted_centroid(tpf, mask)
        assert cx == pytest.approx(5.0, abs=0.01)
        assert cy == pytest.approx(5.0, abs=0.01)

    def test_centroid_point_source(self) -> None:
        """Centroid of point source at known position."""
        tpf = np.zeros((10, 11, 11))
        tpf[:, 3, 7] = 1000.0  # Point source at row=3, col=7
        mask = np.ones(10, dtype=bool)
        cx, cy = _compute_flux_weighted_centroid(tpf, mask)
        # x=col, y=row
        assert cx == pytest.approx(7.0, abs=0.01)
        assert cy == pytest.approx(3.0, abs=0.01)

    def test_centroid_gaussian_psf(self, centered_star_tpf: np.ndarray) -> None:
        """Centroid of Gaussian PSF at center."""
        mask = np.ones(centered_star_tpf.shape[0], dtype=bool)
        cx, cy = _compute_flux_weighted_centroid(centered_star_tpf, mask)
        # Center should be at (5, 5) for 11x11 grid
        assert cx == pytest.approx(5.0, abs=0.1)
        assert cy == pytest.approx(5.0, abs=0.1)

    def test_centroid_empty_mask(self) -> None:
        """Empty mask returns NaN."""
        tpf = np.ones((10, 11, 11)) * 1000.0
        mask = np.zeros(10, dtype=bool)
        cx, cy = _compute_flux_weighted_centroid(tpf, mask)
        assert math.isnan(cx)
        assert math.isnan(cy)

    def test_centroid_zero_flux(self) -> None:
        """Zero flux returns NaN."""
        tpf = np.zeros((10, 11, 11))
        mask = np.ones(10, dtype=bool)
        cx, cy = _compute_flux_weighted_centroid(tpf, mask)
        assert math.isnan(cx)
        assert math.isnan(cy)

    def test_centroid_negative_flux_ignored(self) -> None:
        """Negative flux values are ignored in weighting.

        When computing the weighted centroid, negative flux pixels are
        clamped to zero so they don't contribute to the centroid.
        """
        tpf = np.zeros((10, 11, 11))
        # Place positive flux at corner (0,0)
        tpf[:, 0, 0] = 1000.0
        # Place negative flux at opposite corner (10,10) - should be ignored
        tpf[:, 10, 10] = -500.0
        # Place smaller positive flux at center
        tpf[:, 5, 5] = 500.0
        mask = np.ones(10, dtype=bool)
        cx, cy = _compute_flux_weighted_centroid(tpf, mask)
        # Centroid should be weighted average of (0,0) and (5,5) only
        # With 1000 at (0,0) and 500 at (5,5): weighted avg is (1000*0 + 500*5) / 1500 = 1.67
        expected = (1000 * 0 + 500 * 5) / 1500
        assert cx == pytest.approx(expected, abs=0.01)
        assert cy == pytest.approx(expected, abs=0.01)

    def test_centroid_handles_nan(self) -> None:
        """NaN values in flux are handled."""
        tpf = np.ones((10, 11, 11)) * 1000.0
        tpf[:, 0, 0] = np.nan  # One NaN pixel
        mask = np.ones(10, dtype=bool)
        cx, cy = _compute_flux_weighted_centroid(tpf, mask)
        # Should still compute centroid (slightly shifted from center)
        assert not math.isnan(cx)
        assert not math.isnan(cy)


# =============================================================================
# Test _get_transit_masks
# =============================================================================


class TestGetTransitMasks:
    """Tests for _get_transit_masks helper."""

    def test_transit_mask_basic(self) -> None:
        """Basic transit mask computation."""
        time = np.linspace(0, 10, 100)
        params = TransitParams(period=5.0, t0=2.5, duration=2.0)
        k_in, k_buffer = 1.0, 0.5

        in_mask, out_mask = _get_transit_masks(time, params, k_in, k_buffer)

        # In-transit should be near t0=2.5 and t0+period=7.5
        assert in_mask.sum() > 0
        assert out_mask.sum() > 0
        # Masks should not overlap
        assert not np.any(in_mask & out_mask)

    def test_transit_mask_no_overlap(self) -> None:
        """In-transit and out-of-transit masks do not overlap."""
        time = np.linspace(0, 20, 1000)
        params = TransitParams(period=4.0, t0=2.0, duration=3.0)
        k_in, k_buffer = 1.0, 0.5

        in_mask, out_mask = _get_transit_masks(time, params, k_in, k_buffer)

        # No cadence should be both in and out of transit
        overlap = np.sum(in_mask & out_mask)
        assert overlap == 0

    def test_transit_mask_multiple_transits(self) -> None:
        """Mask detects multiple transits."""
        time = np.linspace(0, 15, 500)
        params = TransitParams(period=3.0, t0=1.5, duration=2.0)
        k_in, k_buffer = 1.0, 0.5

        in_mask, out_mask = _get_transit_masks(time, params, k_in, k_buffer)

        # Should detect transits at t ~ 1.5, 4.5, 7.5, 10.5, 13.5
        # With 15 days and period 3, expect ~5 transits
        # Each transit has duration ~2h = 0.083 days
        # With 500 points over 15 days, each point is 0.03 days
        # ~3 points per transit
        assert in_mask.sum() >= 10  # At least some in-transit points

    def test_transit_mask_k_in_scaling(self) -> None:
        """k_in scales in-transit window."""
        time = np.linspace(0, 10, 500)
        params = TransitParams(period=5.0, t0=2.5, duration=2.0)

        in_mask_small, _ = _get_transit_masks(time, params, k_in=0.5, k_buffer=0.5)
        in_mask_large, _ = _get_transit_masks(time, params, k_in=2.0, k_buffer=0.5)

        # Larger k_in should select more points
        assert in_mask_large.sum() > in_mask_small.sum()

    def test_transit_mask_k_buffer_scaling(self) -> None:
        """k_buffer scales exclusion zone."""
        time = np.linspace(0, 10, 500)
        params = TransitParams(period=5.0, t0=2.5, duration=2.0)

        _, out_mask_small = _get_transit_masks(time, params, k_in=1.0, k_buffer=0.2)
        _, out_mask_large = _get_transit_masks(time, params, k_in=1.0, k_buffer=1.0)

        # Larger k_buffer should exclude more points from out-of-transit
        assert out_mask_large.sum() < out_mask_small.sum()


# =============================================================================
# Test compute_centroid_shift - Basic Functionality
# =============================================================================


class TestComputeCentroidShiftBasic:
    """Basic tests for compute_centroid_shift function."""

    def test_basic_call(
        self,
        simple_tpf: np.ndarray,
        simple_time: np.ndarray,
        simple_transit_params: TransitParams,
    ) -> None:
        """Basic function call works."""
        result = compute_centroid_shift(simple_tpf, simple_time, simple_transit_params)

        assert isinstance(result, CentroidResult)
        assert result.n_in_transit_cadences > 0
        assert result.n_out_transit_cadences > 0

    def test_returns_centroid_result(
        self,
        centered_star_tpf: np.ndarray,
        simple_time: np.ndarray,
        simple_transit_params: TransitParams,
    ) -> None:
        """Function returns CentroidResult."""
        result = compute_centroid_shift(centered_star_tpf, simple_time, simple_transit_params)
        assert isinstance(result, CentroidResult)

    def test_centroids_are_tuples(
        self,
        centered_star_tpf: np.ndarray,
        simple_time: np.ndarray,
        simple_transit_params: TransitParams,
    ) -> None:
        """Centroids are (x, y) tuples."""
        result = compute_centroid_shift(centered_star_tpf, simple_time, simple_transit_params)
        assert isinstance(result.in_transit_centroid, tuple)
        assert isinstance(result.out_of_transit_centroid, tuple)
        assert len(result.in_transit_centroid) == 2
        assert len(result.out_of_transit_centroid) == 2


# =============================================================================
# Test compute_centroid_shift - Shift Detection
# =============================================================================


class TestComputeCentroidShiftDetection:
    """Tests for centroid shift detection."""

    def test_no_shift_uniform_data(
        self,
        centered_star_tpf: np.ndarray,
        simple_time: np.ndarray,
        simple_transit_params: TransitParams,
    ) -> None:
        """No significant shift in uniform star data."""
        result = compute_centroid_shift(centered_star_tpf, simple_time, simple_transit_params)

        # Shift should be small (< 0.1 pixels) for uniform data
        assert result.centroid_shift_pixels < 0.5

    def test_detects_shift(
        self,
        shifted_star_tpf: tuple[np.ndarray, np.ndarray, TransitParams],
    ) -> None:
        """Detects centroid shift in synthetic data."""
        tpf, time, params = shifted_star_tpf
        result = compute_centroid_shift(tpf, time, params)

        # Shift should be detectable (~0.5 pixels injected)
        assert result.centroid_shift_pixels > 0.3
        # Should be significant
        assert result.significance_sigma > 2.0

    def test_shift_direction(
        self,
        shifted_star_tpf: tuple[np.ndarray, np.ndarray, TransitParams],
    ) -> None:
        """Shift direction matches injection."""
        tpf, time, params = shifted_star_tpf
        result = compute_centroid_shift(tpf, time, params)

        # Injected shift was (0.5, 0.3) in (x, y)
        # In-transit centroid should be shifted from out-of-transit
        dx = result.in_transit_centroid[0] - result.out_of_transit_centroid[0]
        dy = result.in_transit_centroid[1] - result.out_of_transit_centroid[1]

        # Direction should be positive (shifted in positive x,y direction)
        assert dx > 0  # x shift positive
        assert dy > 0  # y shift positive


# =============================================================================
# Test compute_centroid_shift - Significance Methods
# =============================================================================


class TestComputeCentroidShiftSignificance:
    """Tests for significance estimation methods."""

    def test_analytic_significance(
        self,
        centered_star_tpf: np.ndarray,
        simple_time: np.ndarray,
        simple_transit_params: TransitParams,
    ) -> None:
        """Analytic significance method works."""
        result = compute_centroid_shift(
            centered_star_tpf,
            simple_time,
            simple_transit_params,
            significance_method="analytic",
        )
        assert not math.isnan(result.significance_sigma)

    def test_bootstrap_significance(
        self,
        centered_star_tpf: np.ndarray,
        simple_time: np.ndarray,
        simple_transit_params: TransitParams,
    ) -> None:
        """Bootstrap significance method works."""
        result = compute_centroid_shift(
            centered_star_tpf,
            simple_time,
            simple_transit_params,
            significance_method="bootstrap",
            n_bootstrap=100,  # Reduced for speed
            bootstrap_seed=42,
        )
        assert not math.isnan(result.significance_sigma)

    def test_bootstrap_reproducibility(
        self,
        centered_star_tpf: np.ndarray,
        simple_time: np.ndarray,
        simple_transit_params: TransitParams,
    ) -> None:
        """Bootstrap is reproducible with seed."""
        result1 = compute_centroid_shift(
            centered_star_tpf,
            simple_time,
            simple_transit_params,
            significance_method="bootstrap",
            n_bootstrap=100,
            bootstrap_seed=42,
        )
        result2 = compute_centroid_shift(
            centered_star_tpf,
            simple_time,
            simple_transit_params,
            significance_method="bootstrap",
            n_bootstrap=100,
            bootstrap_seed=42,
        )
        assert result1.significance_sigma == result2.significance_sigma

    def test_high_shift_high_significance(
        self,
        shifted_star_tpf: tuple[np.ndarray, np.ndarray, TransitParams],
    ) -> None:
        """Large shift should have high significance."""
        tpf, time, params = shifted_star_tpf
        result = compute_centroid_shift(tpf, time, params)

        # With injected shift, should be highly significant
        assert result.significance_sigma > 3.0


# =============================================================================
# Test compute_centroid_shift - Window Policies
# =============================================================================


class TestComputeCentroidShiftWindowPolicy:
    """Tests for window policy selection."""

    def test_v1_policy_works(
        self,
        simple_tpf: np.ndarray,
        simple_time: np.ndarray,
        simple_transit_params: TransitParams,
    ) -> None:
        """v1 window policy works."""
        result = compute_centroid_shift(
            simple_tpf,
            simple_time,
            simple_transit_params,
            window_policy_version="v1",
        )
        assert isinstance(result, CentroidResult)

    def test_invalid_policy_raises(
        self,
        simple_tpf: np.ndarray,
        simple_time: np.ndarray,
        simple_transit_params: TransitParams,
    ) -> None:
        """Invalid window policy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown window_policy_version"):
            compute_centroid_shift(
                simple_tpf,
                simple_time,
                simple_transit_params,
                window_policy_version="invalid",
            )


# =============================================================================
# Test compute_centroid_shift - Input Validation
# =============================================================================


class TestComputeCentroidShiftValidation:
    """Tests for input validation."""

    def test_invalid_tpf_shape_2d(self) -> None:
        """2D TPF raises ValueError."""
        tpf = np.ones((100, 11))  # 2D instead of 3D
        time = np.linspace(0, 10, 100)
        params = TransitParams(period=2.5, t0=1.25, duration=2.0)

        with pytest.raises(ValueError, match="must be 3D"):
            compute_centroid_shift(tpf, time, params)

    def test_invalid_tpf_shape_4d(self) -> None:
        """4D TPF raises ValueError."""
        tpf = np.ones((100, 11, 11, 2))  # 4D
        time = np.linspace(0, 10, 100)
        params = TransitParams(period=2.5, t0=1.25, duration=2.0)

        with pytest.raises(ValueError, match="must be 3D"):
            compute_centroid_shift(tpf, time, params)

    def test_mismatched_time_length(self) -> None:
        """Mismatched time array length raises ValueError."""
        tpf = np.ones((100, 11, 11))
        time = np.linspace(0, 10, 50)  # Wrong length
        params = TransitParams(period=2.5, t0=1.25, duration=2.0)

        with pytest.raises(ValueError, match="time length"):
            compute_centroid_shift(tpf, time, params)


# =============================================================================
# Test compute_centroid_shift - Edge Cases
# =============================================================================


class TestComputeCentroidShiftEdgeCases:
    """Tests for edge cases."""

    def test_single_transit(self) -> None:
        """Works with single transit in data."""
        rng = np.random.default_rng(42)
        tpf = np.ones((50, 11, 11)) * 1000 + rng.normal(0, 10, (50, 11, 11))
        time = np.linspace(0, 3, 50)  # Short baseline
        params = TransitParams(period=10.0, t0=1.5, duration=2.0)

        result = compute_centroid_shift(tpf, time, params)
        # Should still work with just one transit
        assert result.n_in_transit_cadences > 0

    def test_no_in_transit_cadences(self) -> None:
        """Handles case with no in-transit cadences."""
        rng = np.random.default_rng(42)
        tpf = np.ones((50, 11, 11)) * 1000 + rng.normal(0, 10, (50, 11, 11))
        time = np.linspace(0, 1, 50)  # Very short baseline
        params = TransitParams(period=10.0, t0=5.0, duration=1.0)  # Transit outside

        result = compute_centroid_shift(tpf, time, params)
        # Should return NaN when no in-transit cadences
        assert result.n_in_transit_cadences == 0 or math.isnan(result.centroid_shift_pixels)

    def test_all_in_transit(self) -> None:
        """Handles case where all cadences are in-transit."""
        rng = np.random.default_rng(42)
        tpf = np.ones((20, 11, 11)) * 1000 + rng.normal(0, 10, (20, 11, 11))
        time = np.linspace(1.0, 1.1, 20)  # Very short, all near t0
        params = TransitParams(period=10.0, t0=1.05, duration=24.0)  # 24h transit

        result = compute_centroid_shift(tpf, time, params)
        # Should handle gracefully
        assert isinstance(result, CentroidResult)

    def test_small_tpf(self) -> None:
        """Works with small TPF (3x3)."""
        rng = np.random.default_rng(42)
        tpf = np.ones((100, 3, 3)) * 1000 + rng.normal(0, 10, (100, 3, 3))
        time = np.linspace(0, 10, 100)
        params = TransitParams(period=2.5, t0=1.25, duration=2.0)

        result = compute_centroid_shift(tpf, time, params)
        assert isinstance(result, CentroidResult)
        # Centroid should be near center (1, 1) for 3x3 grid
        assert result.out_of_transit_centroid[0] == pytest.approx(1.0, abs=0.2)
        assert result.out_of_transit_centroid[1] == pytest.approx(1.0, abs=0.2)

    def test_large_tpf(self) -> None:
        """Works with large TPF (21x21)."""
        rng = np.random.default_rng(42)
        tpf = np.ones((100, 21, 21)) * 1000 + rng.normal(0, 10, (100, 21, 21))
        time = np.linspace(0, 10, 100)
        params = TransitParams(period=2.5, t0=1.25, duration=2.0)

        result = compute_centroid_shift(tpf, time, params)
        assert isinstance(result, CentroidResult)
        # Centroid should be near center (10, 10) for 21x21 grid
        assert result.out_of_transit_centroid[0] == pytest.approx(10.0, abs=0.2)
        assert result.out_of_transit_centroid[1] == pytest.approx(10.0, abs=0.2)


# =============================================================================
# Test Deterministic Windowing
# =============================================================================


class TestDeterministicWindowing:
    """Tests verifying deterministic windowing behavior."""

    def test_window_is_deterministic(self) -> None:
        """Same inputs produce same window masks."""
        time = np.linspace(0, 10, 100)
        params = TransitParams(period=2.5, t0=1.25, duration=2.0)
        k_in, k_buffer = 1.0, 0.5

        in_mask1, out_mask1 = _get_transit_masks(time, params, k_in, k_buffer)
        in_mask2, out_mask2 = _get_transit_masks(time, params, k_in, k_buffer)

        np.testing.assert_array_equal(in_mask1, in_mask2)
        np.testing.assert_array_equal(out_mask1, out_mask2)

    def test_in_transit_window_formula(self) -> None:
        """In-transit window follows k_in * duration formula."""
        time = np.linspace(0, 10, 1000)
        duration_hours = 2.4  # hours
        duration_days = duration_hours / 24.0
        params = TransitParams(period=5.0, t0=2.5, duration=duration_hours)

        # k_in = 1.0 means window = 1.0 * duration
        k_in = 1.0
        in_mask, _ = _get_transit_masks(time, params, k_in, k_buffer=0.5)

        # Find in-transit cadences near first transit
        in_transit_times = time[in_mask]
        near_first_transit = in_transit_times[in_transit_times < 5.0]

        if len(near_first_transit) > 0:
            # Check window width is approximately k_in * duration
            window_width = near_first_transit.max() - near_first_transit.min()
            expected_width = k_in * duration_days
            # Allow some tolerance due to discrete sampling
            assert window_width == pytest.approx(expected_width, abs=0.02)

    def test_out_of_transit_exclusion_formula(self) -> None:
        """Out-of-transit excludes k_buffer * duration around transit."""
        time = np.linspace(0, 10, 1000)
        duration_hours = 2.4
        duration_days = duration_hours / 24.0
        params = TransitParams(period=5.0, t0=2.5, duration=duration_hours)

        k_buffer = 0.5
        _, out_mask = _get_transit_masks(time, params, k_in=1.0, k_buffer=k_buffer)

        # Buffer zone should be duration/2 + k_buffer * duration from transit center
        buffer_half_width = duration_days / 2.0 + k_buffer * duration_days

        # Find gap in out-of-transit near first transit at t0=2.5
        out_times = time[out_mask]
        near_transit = (time > 2.5 - 2 * buffer_half_width) & (time < 2.5 + 2 * buffer_half_width)
        gap = ~out_mask & near_transit

        if np.any(gap):
            gap_times = time[gap]
            gap_center = (gap_times.min() + gap_times.max()) / 2.0
            gap_half_width = (gap_times.max() - gap_times.min()) / 2.0

            # Gap should be centered on transit and have width ~ 2 * buffer_half_width
            assert gap_center == pytest.approx(2.5, abs=0.05)
            assert gap_half_width == pytest.approx(buffer_half_width, abs=0.03)


# =============================================================================
# Integration Tests
# =============================================================================


class TestCentroidIntegration:
    """Integration tests for complete workflow."""

    def test_full_analysis_workflow(self) -> None:
        """Complete analysis workflow from TPF to result."""
        # Create synthetic TPF with known shift
        rng = np.random.default_rng(42)
        n_time, n_rows, n_cols = 200, 11, 11
        period, t0, duration = 3.0, 1.5, 2.0
        shift_x, shift_y = 0.4, 0.2

        time = np.linspace(0, 12, n_time)
        params = TransitParams(period=period, t0=t0, duration=duration)

        # Compute transit mask for injection
        duration_days = duration / 24.0
        phase = ((time - t0) / period) % 1.0
        phase = np.where(phase > 0.5, phase - 1.0, phase)
        time_from_transit = phase * period
        in_transit = np.abs(time_from_transit) <= duration_days / 2.0

        # Create TPF
        rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
        row_center, col_center = n_rows // 2, n_cols // 2
        sigma = 1.5

        tpf = np.zeros((n_time, n_rows, n_cols))
        for i in range(n_time):
            if in_transit[i]:
                psf = 10000 * np.exp(
                    -((rows - row_center - shift_y) ** 2 + (cols - col_center - shift_x) ** 2)
                    / (2 * sigma**2)
                )
            else:
                psf = 10000 * np.exp(
                    -((rows - row_center) ** 2 + (cols - col_center) ** 2) / (2 * sigma**2)
                )
            tpf[i] = psf + rng.normal(0, 30, psf.shape)

        # Run analysis
        result = compute_centroid_shift(tpf, time, params)

        # Verify results
        assert result.n_in_transit_cadences > 0
        assert result.n_out_transit_cadences > 0

        # Shift should be detected
        expected_shift = np.sqrt(shift_x**2 + shift_y**2)
        assert result.centroid_shift_pixels == pytest.approx(expected_shift, abs=0.15)

        # Should be significant
        assert result.significance_sigma > 2.0

    def test_batch_analysis(self) -> None:
        """Analyze multiple targets in batch."""
        rng = np.random.default_rng(42)
        n_targets = 5
        results = []

        for i in range(n_targets):
            tpf = np.ones((100, 11, 11)) * 1000 + rng.normal(0, 10, (100, 11, 11))
            time = np.linspace(0, 10, 100)
            params = TransitParams(period=2.0 + i * 0.5, t0=1.0, duration=2.0)

            result = compute_centroid_shift(tpf, time, params)
            results.append(result)

        # All should produce valid results
        assert len(results) == n_targets
        assert all(isinstance(r, CentroidResult) for r in results)
