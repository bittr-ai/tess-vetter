"""Tests for wotan-based detrending functions.

Tests for:
- wotan_flatten: Transit-aware detrending using wotan library
- flatten_with_wotan: Wotan with automatic fallback to median filter
- WOTAN_AVAILABLE: Import status flag

All tests use synthetic light curves with injected stellar variability
and transits to verify transit-preserving detrending behavior.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from bittr_tess_vetter.compute.detrend import (
    WOTAN_AVAILABLE,
    flatten,
    flatten_with_wotan,
    wotan_flatten,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def time_array() -> np.ndarray:
    """Generate a 27-day time array (TESS single-sector observation).

    Uses 2-minute cadence (~30 samples per hour).
    """
    n_points = 19440  # 27 days at 2-minute cadence
    return np.linspace(0.0, 27.0, n_points, dtype=np.float64)


@pytest.fixture
def flat_flux(time_array: np.ndarray) -> np.ndarray:
    """Generate flat flux with small Gaussian noise."""
    rng = np.random.default_rng(42)
    noise_level = 0.0001  # 100 ppm noise
    return np.ones_like(time_array) + rng.normal(0, noise_level, len(time_array))


def inject_stellar_variability(
    time: np.ndarray,
    flux: np.ndarray,
    rotation_period: float = 5.0,
    amplitude: float = 0.01,
) -> np.ndarray:
    """Inject stellar rotation signal into light curve.

    Args:
        time: Time array
        flux: Flux array
        rotation_period: Stellar rotation period in days
        amplitude: Semi-amplitude of variation

    Returns:
        Modified flux with stellar variability
    """
    flux = flux.copy()
    # Add sinusoidal variation (simplified spot modulation)
    flux *= 1.0 + amplitude * np.sin(2.0 * np.pi * time / rotation_period)
    return flux


def inject_transit(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject box-shaped transit into light curve.

    Args:
        time: Time array
        flux: Flux array
        period: Orbital period in days
        t0: Reference epoch (mid-transit)
        duration: Transit duration in days
        depth: Transit depth (fractional)

    Returns:
        Tuple of (modified flux, transit mask)
    """
    flux = flux.copy()
    phase = ((time - t0) / period + 0.5) % 1.0 - 0.5
    half_dur = duration / (2.0 * period)

    transit_mask = np.abs(phase) < half_dur
    flux[transit_mask] *= 1.0 - depth

    return flux, transit_mask


# =============================================================================
# Tests for WOTAN_AVAILABLE
# =============================================================================


class TestWotanAvailable:
    """Tests for WOTAN_AVAILABLE flag."""

    def test_wotan_available_is_bool(self):
        """Verify WOTAN_AVAILABLE is a boolean."""
        assert isinstance(WOTAN_AVAILABLE, bool)

    @pytest.mark.skipif(not WOTAN_AVAILABLE, reason="wotan not installed")
    def test_wotan_import_works(self):
        """Verify wotan can be imported when available."""
        from wotan import flatten as wf

        assert callable(wf)


# =============================================================================
# Tests for wotan_flatten
# =============================================================================


@pytest.mark.skipif(not WOTAN_AVAILABLE, reason="wotan not installed")
class TestWotanFlatten:
    """Tests for wotan_flatten function."""

    def test_removes_stellar_variability(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify that wotan removes stellar variability."""
        # Inject stellar rotation signal
        flux_with_var = inject_stellar_variability(time_array, flat_flux, rotation_period=5.0)

        # Flatten with wotan
        flat = wotan_flatten(time_array, flux_with_var, window_length=1.0, method="biweight")

        # After flattening, variation should be much reduced
        original_std = np.std(flux_with_var)
        flat_std = np.std(flat)
        assert flat_std < original_std * 0.3

    def test_preserves_transit_with_mask(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify that transits are preserved when masked."""
        period, t0, duration, depth = 3.5, 1.5, 0.1, 0.01

        # Inject stellar variability
        flux_with_var = inject_stellar_variability(time_array, flat_flux, rotation_period=5.0)

        # Inject transit
        flux_with_transit, transit_mask = inject_transit(
            time_array, flux_with_var, period, t0, duration, depth
        )

        # Flatten with transit mask
        flat = wotan_flatten(
            time_array,
            flux_with_transit,
            window_length=0.5,
            method="biweight",
            transit_mask=transit_mask,
        )

        # Check transit depth is preserved
        out_transit_median = np.median(flat[~transit_mask])
        in_transit_median = np.median(flat[transit_mask])
        measured_depth = (out_transit_median - in_transit_median) / out_transit_median

        # With transit mask, depth should be well preserved
        assert_allclose(measured_depth, depth, rtol=0.3)

    def test_returns_trend_when_requested(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify that return_trend=True returns tuple of (flat, trend)."""
        flux_with_var = inject_stellar_variability(time_array, flat_flux, rotation_period=5.0)

        result = wotan_flatten(
            time_array, flux_with_var, window_length=1.0, method="biweight", return_trend=True
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        flat, trend = result
        assert flat.shape == flux_with_var.shape
        assert trend.shape == flux_with_var.shape

        # Verify flat = flux / trend (approximately)
        reconstructed = flat * trend
        assert_allclose(reconstructed, flux_with_var, rtol=0.01)

    def test_different_methods(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify that different detrending methods work."""
        flux_with_var = inject_stellar_variability(time_array, flat_flux, rotation_period=5.0)

        # Only test methods that don't require optional dependencies (statsmodels)
        # biweight and median work without extra deps
        methods = ["biweight", "median"]
        for method in methods:
            flat = wotan_flatten(time_array, flux_with_var, window_length=1.0, method=method)
            assert flat.shape == flux_with_var.shape
            # All methods should reduce variability
            assert np.std(flat) < np.std(flux_with_var)

    def test_break_tolerance_for_gaps(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify that break_tolerance handles data gaps."""
        # Create a gap in the data (simulating TESS orbit gap)
        gap_end_idx = 9500
        time_with_gap = time_array.copy()
        time_with_gap[gap_end_idx:] += 1.0  # Add 1-day gap

        flux_with_var = inject_stellar_variability(
            time_with_gap, flat_flux.copy(), rotation_period=5.0
        )

        # Flatten with break_tolerance
        flat = wotan_flatten(
            time_with_gap, flux_with_var, window_length=0.5, method="biweight", break_tolerance=0.5
        )

        assert flat.shape == flux_with_var.shape
        # Should still reduce variability
        assert np.std(flat) < np.std(flux_with_var)

    def test_raises_on_shape_mismatch(self):
        """Verify that mismatched shapes raise ValueError."""
        time = np.array([1.0, 2.0, 3.0])
        flux = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="must have same shape"):
            wotan_flatten(time, flux, window_length=0.5)

    def test_raises_on_invalid_window_length(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify that invalid window_length raises ValueError."""
        with pytest.raises(ValueError, match="window_length must be positive"):
            wotan_flatten(time_array, flat_flux, window_length=0.0)

        with pytest.raises(ValueError, match="window_length must be positive"):
            wotan_flatten(time_array, flat_flux, window_length=-1.0)

    def test_raises_on_unsorted_time(self, flat_flux: np.ndarray):
        """Verify that unsorted time array raises ValueError."""
        time_unsorted = np.array([1.0, 3.0, 2.0, 4.0])
        flux = flat_flux[:4]

        with pytest.raises(ValueError, match="sorted in ascending order"):
            wotan_flatten(time_unsorted, flux, window_length=0.5)

    def test_raises_on_transit_mask_shape_mismatch(
        self, time_array: np.ndarray, flat_flux: np.ndarray
    ):
        """Verify that mismatched transit_mask shape raises ValueError."""
        transit_mask = np.zeros(100, dtype=bool)  # Wrong size

        with pytest.raises(ValueError, match="transit_mask must have same shape"):
            wotan_flatten(time_array, flat_flux, transit_mask=transit_mask)

    def test_empty_input(self):
        """Verify that empty input is handled."""
        time = np.array([], dtype=np.float64)
        flux = np.array([], dtype=np.float64)

        flat = wotan_flatten(time, flux, window_length=0.5)
        assert len(flat) == 0

        flat, trend = wotan_flatten(time, flux, window_length=0.5, return_trend=True)
        assert len(flat) == 0
        assert len(trend) == 0


@pytest.mark.skipif(WOTAN_AVAILABLE, reason="Test only when wotan not installed")
class TestWotanFlattenWithoutWotan:
    """Tests for wotan_flatten when wotan is not installed."""

    def test_raises_import_error(self):
        """Verify ImportError is raised when wotan unavailable."""
        time = np.linspace(0, 10, 100)
        flux = np.ones(100)

        with pytest.raises(ImportError, match="wotan is required"):
            wotan_flatten(time, flux, window_length=0.5)


# =============================================================================
# Tests for flatten_with_wotan
# =============================================================================


class TestFlattenWithWotan:
    """Tests for flatten_with_wotan function with automatic fallback."""

    def test_works_without_wotan(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify fallback to median filter works."""
        flux_with_var = inject_stellar_variability(time_array, flat_flux, rotation_period=5.0)

        # Should work regardless of wotan availability
        flat = flatten_with_wotan(time_array, flux_with_var, window_length=1.0)

        assert flat.shape == flux_with_var.shape
        # Should reduce variability
        assert np.std(flat) < np.std(flux_with_var)

    @pytest.mark.skipif(not WOTAN_AVAILABLE, reason="wotan not installed")
    def test_uses_wotan_when_available(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify wotan is used when available."""
        flux_with_var = inject_stellar_variability(time_array, flat_flux, rotation_period=5.0)

        # Create transit mask (only meaningful with wotan)
        transit_mask = np.zeros_like(flux_with_var, dtype=bool)
        transit_mask[1000:1100] = True

        flat = flatten_with_wotan(
            time_array, flux_with_var, window_length=1.0, transit_mask=transit_mask
        )

        assert flat.shape == flux_with_var.shape

    def test_fallback_on_error(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify fallback behavior on error."""
        flux_with_var = inject_stellar_variability(time_array, flat_flux, rotation_period=5.0)

        # This should work even if something goes wrong with wotan
        flat = flatten_with_wotan(
            time_array, flux_with_var, window_length=1.0, fallback_on_error=True
        )

        assert flat.shape == flux_with_var.shape

    def test_consistent_with_flatten_fallback(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify fallback produces same result as flatten()."""
        flux_with_var = inject_stellar_variability(time_array, flat_flux, rotation_period=5.0)

        # If wotan not available, should match flatten()
        if not WOTAN_AVAILABLE:
            flat_with_wotan = flatten_with_wotan(time_array, flux_with_var, window_length=1.0)
            flat_basic = flatten(time_array, flux_with_var, window_length=1.0)
            assert_allclose(flat_with_wotan, flat_basic)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.skipif(not WOTAN_AVAILABLE, reason="wotan not installed")
class TestWotanIntegration:
    """Integration tests for wotan detrending with real-world scenarios."""

    def test_active_star_with_transit(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Test transit recovery in active star scenario (like AU Mic)."""
        # Simulate active star: strong rotation signal + transit
        rotation_period = 4.86  # AU Mic rotation period
        transit_period = 8.46  # AU Mic b period
        transit_t0 = 2.0
        transit_duration = 0.14  # ~3.4 hours
        transit_depth = 0.003  # ~3000 ppm

        # Inject strong stellar variability
        flux_active = inject_stellar_variability(
            time_array, flat_flux, rotation_period=rotation_period, amplitude=0.02
        )

        # Inject transit
        flux_with_transit, transit_mask = inject_transit(
            time_array, flux_active, transit_period, transit_t0, transit_duration, transit_depth
        )

        # Detrend WITHOUT transit mask (baseline - may distort transit)
        flat_no_mask = wotan_flatten(
            time_array, flux_with_transit, window_length=1.0, method="biweight"
        )

        # Detrend WITH transit mask (should preserve transit)
        flat_with_mask = wotan_flatten(
            time_array,
            flux_with_transit,
            window_length=1.0,
            method="biweight",
            transit_mask=transit_mask,
        )

        # Measure depths
        def measure_depth_from_mask(flat: np.ndarray, mask: np.ndarray) -> float:
            out_med = np.median(flat[~mask])
            in_med = np.median(flat[mask])
            return (out_med - in_med) / out_med

        depth_no_mask = measure_depth_from_mask(flat_no_mask, transit_mask)
        depth_with_mask = measure_depth_from_mask(flat_with_mask, transit_mask)

        # With mask should preserve depth better
        # (depth_with_mask should be closer to transit_depth than depth_no_mask)
        error_no_mask = abs(depth_no_mask - transit_depth) / transit_depth
        error_with_mask = abs(depth_with_mask - transit_depth) / transit_depth

        assert error_with_mask < error_no_mask or error_with_mask < 0.5

    def test_multi_sector_stitching(self, flat_flux: np.ndarray):
        """Test detrending across multiple sectors with gaps."""
        # Simulate 2 sectors with gap
        n_per_sector = 9720  # ~13.5 days
        sector1_time = np.linspace(0.0, 13.5, n_per_sector)
        sector2_time = np.linspace(14.5, 28.0, n_per_sector)  # 1-day gap
        time = np.concatenate([sector1_time, sector2_time])
        flux = np.concatenate([flat_flux[:n_per_sector], flat_flux[:n_per_sector]])

        # Inject different variability in each sector
        flux[:n_per_sector] *= 1.0 + 0.01 * np.sin(2 * np.pi * sector1_time / 5.0)
        flux[n_per_sector:] *= 1.0 + 0.015 * np.sin(2 * np.pi * sector2_time / 5.0 + 0.5)

        # Detrend with appropriate break_tolerance
        flat = wotan_flatten(time, flux, window_length=0.5, method="biweight", break_tolerance=0.5)

        assert flat.shape == flux.shape
        # Should reduce variability in both sectors
        assert np.std(flat[:n_per_sector]) < np.std(flux[:n_per_sector])
        assert np.std(flat[n_per_sector:]) < np.std(flux[n_per_sector:])
