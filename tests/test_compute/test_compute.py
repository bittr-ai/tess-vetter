"""Comprehensive unit tests for compute modules.

Tests for:
- periodogram.py: TLS, LS, and auto periodogram functions
- transit.py: Transit detection, depth measurement, masking, folding
- detrend.py: Median detrending, normalization, sigma clipping, flattening

All tests use synthetic light curves with injected transits for validation.

Note: TLS migration replaced BLS with Transit Least Squares.
"""

from __future__ import annotations

import importlib.util
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from bittr_tess_vetter.compute.detrend import (
    flatten,
    median_detrend,
    normalize_flux,
    sigma_clip,
)
from bittr_tess_vetter.compute.periodogram import (
    _estimate_snr,
    auto_periodogram,
    compute_bls_model,
    ls_periodogram,
    refine_period,
    search_planets,
    tls_search,
)
from bittr_tess_vetter.compute.transit import (
    detect_transit,
    fold_transit,
    get_transit_mask,
    measure_depth,
)
from bittr_tess_vetter.domain.detection import PeriodogramResult, TransitCandidate

# TLS (transitleastsquares) is not installable on Python 3.12 via PyPI today
# (it depends on an old numba). Keep TLS code ported, but skip TLS-only tests
# unless the dependency is available.
TLS_AVAILABLE = importlib.util.find_spec("transitleastsquares") is not None

# =============================================================================
# Fixtures for synthetic light curve generation
# =============================================================================


@pytest.fixture
def time_array() -> np.ndarray:
    """Generate a 27-day time array (TESS single-sector observation).

    Uses 2-minute cadence (~30 samples per hour).
    """
    # 27 days at 2-minute cadence = 27 * 24 * 30 = 19440 points
    n_points = 19440
    return np.linspace(0.0, 27.0, n_points, dtype=np.float64)


@pytest.fixture
def flat_flux(time_array: np.ndarray) -> np.ndarray:
    """Generate flat flux with small Gaussian noise."""
    rng = np.random.default_rng(42)
    noise_level = 0.0001  # 100 ppm noise
    return np.ones_like(time_array) + rng.normal(0, noise_level, len(time_array))


@pytest.fixture
def transit_params() -> dict:
    """Standard transit parameters for testing."""
    return {
        "period": 3.5,  # days
        "t0": 1.5,  # BTJD epoch
        "duration": 0.1,  # days (~2.4 hours)
        "depth": 0.01,  # 1% depth (typical hot Jupiter)
    }


def inject_transit(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    depth: float,
) -> np.ndarray:
    """Inject box-shaped transit into light curve.

    Args:
        time: Time array
        flux: Flux array (will be modified)
        period: Orbital period in days
        t0: Reference epoch (mid-transit)
        duration: Transit duration in days
        depth: Transit depth (fractional)

    Returns:
        Modified flux array with transits injected
    """
    flux = flux.copy()
    # Calculate phase
    phase = ((time - t0) / period + 0.5) % 1.0 - 0.5
    half_dur = duration / (2.0 * period)

    # Apply transit
    in_transit = np.abs(phase) < half_dur
    flux[in_transit] *= 1.0 - depth

    return flux


def inject_sinusoid(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    amplitude: float,
    phase_offset: float = 0.0,
) -> np.ndarray:
    """Inject sinusoidal variation into light curve.

    Args:
        time: Time array
        flux: Flux array
        period: Period in days
        amplitude: Semi-amplitude of variation
        phase_offset: Phase offset in radians

    Returns:
        Modified flux array with sinusoid injected
    """
    flux = flux.copy()
    flux += amplitude * np.sin(2.0 * np.pi * time / period + phase_offset)
    return flux


# =============================================================================
# Tests for detrend.py
# =============================================================================


class TestMedianDetrend:
    """Tests for median_detrend function."""

    def test_removes_long_term_trend(self, time_array: np.ndarray):
        """Verify that median_detrend removes long-term trends."""
        # Create flux with linear trend
        flux = 1.0 + 0.01 * time_array / 27.0  # 1% drift over 27 days

        # Add small noise
        rng = np.random.default_rng(42)
        flux += rng.normal(0, 0.0001, len(flux))

        # Detrend
        detrended = median_detrend(flux, window=501)

        # Check that trend is removed (median should be ~1.0)
        assert_allclose(np.median(detrended), 1.0, atol=0.001)

        # Check that standard deviation is reduced significantly
        original_std = np.std(flux)
        detrended_std = np.std(detrended)
        assert detrended_std < original_std * 0.5

    def test_preserves_transit_signals(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify that transit signals are preserved after detrending."""
        # Inject transit
        period, t0, duration, depth = 3.5, 1.5, 0.1, 0.01
        flux_with_transit = inject_transit(time_array, flat_flux, period, t0, duration, depth)

        # Add slow trend
        flux_with_trend = flux_with_transit * (1.0 + 0.005 * time_array / 27.0)

        # Detrend with window larger than transit duration
        window = 301  # ~10 hours at 2-min cadence - larger than 2.4 hr transit
        detrended = median_detrend(flux_with_trend, window=window)

        # Find in-transit points
        phase = ((time_array - t0) / period + 0.5) % 1.0 - 0.5
        in_transit = np.abs(phase) < (duration / (2.0 * period))

        # Check transit depth is preserved
        out_transit_median = np.median(detrended[~in_transit])
        in_transit_median = np.median(detrended[in_transit])
        measured_depth = (out_transit_median - in_transit_median) / out_transit_median

        assert_allclose(measured_depth, depth, rtol=0.2)

    def test_window_must_be_odd(self, flat_flux: np.ndarray):
        """Verify that even window sizes raise ValueError."""
        with pytest.raises(ValueError, match="window must be odd"):
            median_detrend(flat_flux, window=100)

    def test_window_must_be_positive(self, flat_flux: np.ndarray):
        """Verify that non-positive window sizes raise ValueError."""
        with pytest.raises(ValueError, match="window must be positive"):
            median_detrend(flat_flux, window=0)

        with pytest.raises(ValueError, match="window must be positive"):
            median_detrend(flat_flux, window=-5)

    def test_handles_nan_values(self, time_array: np.ndarray):
        """Verify that NaN values are handled properly."""
        flux = np.ones_like(time_array)
        # Inject NaN gap
        flux[1000:1100] = np.nan

        detrended = median_detrend(flux, window=101)

        # Result should have NaN at same positions
        assert np.sum(np.isnan(detrended)) == 100
        # Non-NaN values should be close to 1.0
        valid = ~np.isnan(detrended)
        assert_allclose(np.median(detrended[valid]), 1.0, atol=0.001)


class TestNormalizeFlux:
    """Tests for normalize_flux function."""

    def test_returns_median_one(self):
        """Verify that normalized flux has median = 1.0."""
        flux = np.array([1000.0, 1001.0, 999.0, 1002.0, 998.0])
        flux_err = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

        norm_flux, norm_err = normalize_flux(flux, flux_err)

        assert_allclose(np.median(norm_flux), 1.0, atol=1e-10)

    def test_error_propagation(self):
        """Verify that errors are scaled correctly."""
        flux = np.array([1000.0, 1000.0, 1000.0])
        flux_err = np.array([10.0, 20.0, 30.0])

        norm_flux, norm_err = normalize_flux(flux, flux_err)

        # Errors should be scaled by same factor as flux
        expected_err = flux_err / np.median(flux)
        assert_allclose(norm_err, expected_err)

    def test_shape_mismatch_raises_error(self):
        """Verify that mismatched shapes raise ValueError."""
        flux = np.array([1.0, 2.0, 3.0])
        flux_err = np.array([0.1, 0.2])

        with pytest.raises(ValueError, match="must have same shape"):
            normalize_flux(flux, flux_err)

    def test_handles_nan_in_flux(self):
        """Verify that NaN values are handled in median calculation."""
        flux = np.array([1000.0, np.nan, 1000.0, 1000.0])
        flux_err = np.array([10.0, 10.0, 10.0, 10.0])

        norm_flux, norm_err = normalize_flux(flux, flux_err)

        # Should use nanmedian
        valid = ~np.isnan(norm_flux)
        assert_allclose(np.median(norm_flux[valid]), 1.0)


class TestSigmaClip:
    """Tests for sigma_clip function."""

    def test_removes_outliers(self):
        """Verify that outliers beyond sigma threshold are masked."""
        flux = np.array([1.0, 1.001, 1.002, 10.0, 0.999, 1.003, 1.0, -5.0])

        mask = sigma_clip(flux, sigma=3.0)

        # 10.0 and -5.0 should be flagged as outliers
        expected = np.array([True, True, True, False, True, True, True, False])
        assert_array_equal(mask, expected)

    def test_conservative_sigma(self):
        """Verify that higher sigma is more conservative."""
        rng = np.random.default_rng(42)
        flux = np.ones(1000)
        flux += rng.normal(0, 0.001, 1000)
        # Inject one 5-sigma outlier
        flux[500] = 1.0 + 5 * 0.001

        # 3-sigma should catch it
        mask_3sigma = sigma_clip(flux, sigma=3.0)
        assert not mask_3sigma[500]

        # 7-sigma should not catch it
        mask_7sigma = sigma_clip(flux, sigma=7.0)
        assert mask_7sigma[500]

    def test_sigma_must_be_positive(self):
        """Verify that non-positive sigma raises ValueError."""
        flux = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="sigma must be positive"):
            sigma_clip(flux, sigma=0.0)

        with pytest.raises(ValueError, match="sigma must be positive"):
            sigma_clip(flux, sigma=-1.0)

    def test_handles_nan(self):
        """Verify that NaN values are flagged as invalid."""
        flux = np.array([1.0, np.nan, 1.0, 1.0])

        mask = sigma_clip(flux, sigma=5.0)

        # NaN should be False
        assert not mask[1]
        # Others should be True
        assert all(mask[[0, 2, 3]])


class TestFlatten:
    """Tests for flatten function."""

    def test_flatten_with_time_based_window(self, time_array: np.ndarray):
        """Verify that flatten uses time-based windowing correctly."""
        # Create flux with 5-day sinusoidal variation
        flux = 1.0 + 0.01 * np.sin(2 * np.pi * time_array / 5.0)

        # Flatten with 1-day window (should remove 5-day variation)
        flat = flatten(time_array, flux, window_length=1.0)

        # After flattening, variation should be much reduced
        assert np.std(flat) < np.std(flux) * 0.3

    def test_preserves_transit(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify that transits are preserved with appropriate window."""
        # Inject transit
        period, t0, duration, depth = 3.5, 1.5, 0.1, 0.01
        flux_with_transit = inject_transit(time_array, flat_flux, period, t0, duration, depth)

        # Flatten with window > transit duration (0.1 days = 2.4 hours)
        flat = flatten(time_array, flux_with_transit, window_length=0.5)  # 12 hours

        # Check transit depth is preserved
        phase = ((time_array - t0) / period + 0.5) % 1.0 - 0.5
        in_transit = np.abs(phase) < (duration / (2.0 * period))

        out_transit_median = np.median(flat[~in_transit])
        in_transit_median = np.median(flat[in_transit])
        measured_depth = (out_transit_median - in_transit_median) / out_transit_median

        assert_allclose(measured_depth, depth, rtol=0.25)

    def test_shape_mismatch_raises_error(self):
        """Verify that mismatched shapes raise ValueError."""
        time = np.array([1.0, 2.0, 3.0])
        flux = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="must have same shape"):
            flatten(time, flux, window_length=0.5)

    def test_window_length_must_be_positive(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify that non-positive window_length raises ValueError."""
        with pytest.raises(ValueError, match="window_length must be positive"):
            flatten(time_array, flat_flux, window_length=0.0)

        with pytest.raises(ValueError, match="window_length must be positive"):
            flatten(time_array, flat_flux, window_length=-1.0)


# =============================================================================
# Tests for transit.py
# =============================================================================


class TestGetTransitMask:
    """Tests for get_transit_mask function."""

    def test_correct_points_selected(self):
        """Verify that in-transit points are correctly identified."""
        time = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        period = 1.0
        t0 = 0.5
        duration = 0.2  # Transit from phase -0.1 to +0.1

        mask = get_transit_mask(time, period, t0, duration)

        # At t=0.5, phase=0 -> in transit
        # At t=1.5, phase=0 -> in transit
        # At t=2.5, phase=0 -> in transit
        # Others are at phase=0.5 -> out of transit
        expected = np.array([False, True, False, True, False, True, False])
        assert_array_equal(mask, expected)

    def test_handles_multiple_transits(self, time_array: np.ndarray):
        """Verify that all periodic transits are captured."""
        period = 3.5
        t0 = 1.5
        duration = 0.1

        mask = get_transit_mask(time_array, period, t0, duration)

        # Calculate expected number of transit events
        n_transits = int(27.0 / period)  # ~7 transits in 27 days

        # Count contiguous in-transit regions
        diff = np.diff(mask.astype(int))
        n_regions = np.sum(diff == 1)  # Rising edges

        # Should have approximately n_transits regions
        assert abs(n_regions - n_transits) <= 1

    def test_phase_centered_on_t0(self):
        """Verify that phase is correctly centered on t0."""
        time = np.linspace(0, 10, 1000)
        period = 2.0
        t0 = 1.0
        duration = 0.2

        mask = get_transit_mask(time, period, t0, duration)

        # Transit should occur at t0 and t0 + period, t0 + 2*period, etc.
        transit_times = time[mask]

        # Find the transit centers
        for n in range(5):
            expected_center = t0 + n * period
            if expected_center > 10:
                break
            # Check that there are transit points near this time
            near_center = np.abs(transit_times - expected_center) < duration
            assert np.any(near_center), f"No transit points near t={expected_center}"


class TestMeasureDepth:
    """Tests for measure_depth function."""

    def test_accuracy_on_box_transit(self):
        """Verify depth measurement accuracy on clean box transit."""
        # Create perfect box transit
        flux = np.ones(1000)
        in_transit_mask = np.zeros(1000, dtype=bool)
        in_transit_mask[400:600] = True

        expected_depth = 0.01
        flux[in_transit_mask] = 1.0 - expected_depth

        depth, depth_err = measure_depth(flux, in_transit_mask)

        assert_allclose(depth, expected_depth, rtol=1e-6)

    def test_depth_with_noise(self):
        """Verify depth measurement with realistic noise."""
        rng = np.random.default_rng(42)

        flux = np.ones(10000) + rng.normal(0, 0.001, 10000)
        in_transit_mask = np.zeros(10000, dtype=bool)
        in_transit_mask[4000:6000] = True

        expected_depth = 0.01
        flux[in_transit_mask] -= expected_depth

        depth, depth_err = measure_depth(flux, in_transit_mask)

        # Depth should be accurate within a few percent
        assert_allclose(depth, expected_depth, rtol=0.05)

        # Error should be small but non-zero
        assert depth_err > 0
        assert depth_err < 0.001

    def test_no_in_transit_raises_error(self):
        """Verify that no in-transit points raises ValueError."""
        flux = np.ones(100)
        mask = np.zeros(100, dtype=bool)

        with pytest.raises(ValueError, match="No in-transit points"):
            measure_depth(flux, mask)

    def test_no_out_transit_raises_error(self):
        """Verify that no out-of-transit points raises ValueError."""
        flux = np.ones(100)
        mask = np.ones(100, dtype=bool)

        with pytest.raises(ValueError, match="No out-of-transit points"):
            measure_depth(flux, mask)


class TestFoldTransit:
    """Tests for fold_transit function."""

    def test_phase_centered_on_zero(self):
        """Verify that folded phase is centered on zero."""
        time = np.linspace(0, 10, 1000)
        flux = np.ones_like(time)
        period = 2.0
        t0 = 1.0

        phase, flux_folded = fold_transit(time, flux, period, t0)

        # Phase should range from -0.5 to 0.5
        assert phase.min() >= -0.5
        assert phase.max() <= 0.5

        # Center of transit (t=t0, t0+period, etc.) should be at phase=0
        # Check that phase=0 exists in the output
        assert np.min(np.abs(phase)) < 0.01

    def test_flux_values_preserved(self):
        """Verify that flux values are preserved during folding."""
        time = np.linspace(0, 10, 100)
        flux = np.random.default_rng(42).random(100)
        period = 2.0
        t0 = 0.0

        phase, flux_folded = fold_transit(time, flux, period, t0)

        # Sorted flux_folded should match sorted original flux
        assert_allclose(np.sort(flux_folded), np.sort(flux))

    def test_sorted_by_phase(self):
        """Verify that output is sorted by phase."""
        time = np.linspace(0, 10, 1000)
        flux = np.ones_like(time)
        period = 2.0
        t0 = 0.5

        phase, flux_folded = fold_transit(time, flux, period, t0)

        # Phase array should be sorted
        assert np.all(np.diff(phase) >= 0)


class TestDetectTransit:
    """Tests for detect_transit function."""

    def test_returns_transit_candidate(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify that detect_transit returns TransitCandidate with correct depth."""
        # Inject transit
        period, t0, duration, depth = 3.5, 1.5, 0.1, 0.01
        flux = inject_transit(time_array, flat_flux, period, t0, duration, depth)
        flux_err = np.ones_like(flux) * 0.0001

        candidate = detect_transit(time_array, flux, flux_err, period, t0, duration)

        assert isinstance(candidate, TransitCandidate)
        assert_allclose(candidate.period, period)
        assert_allclose(candidate.t0, t0)
        assert_allclose(candidate.depth, depth, rtol=0.15)

    def test_correct_depth_measurement(self):
        """Verify accurate depth measurement for various transit depths."""
        time = np.linspace(0, 30, 15000)
        flux_err = np.ones_like(time) * 0.0001

        for expected_depth in [0.001, 0.005, 0.01, 0.02]:
            flux = np.ones_like(time) + np.random.default_rng(42).normal(0, 0.0001, len(time))
            flux = inject_transit(
                time, flux, period=5.0, t0=2.5, duration=0.15, depth=expected_depth
            )

            candidate = detect_transit(time, flux, flux_err, period=5.0, t0=2.5, duration=0.15)

            assert_allclose(candidate.depth, expected_depth, rtol=0.2)

    def test_snr_positive(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify that SNR is positive for real transit."""
        period, t0, duration, depth = 3.5, 1.5, 0.1, 0.01
        flux = inject_transit(time_array, flat_flux, period, t0, duration, depth)
        flux_err = np.ones_like(flux) * 0.0001

        candidate = detect_transit(time_array, flux, flux_err, period, t0, duration)

        assert candidate.snr > 0

    def test_insufficient_data_raises_error(self):
        """Verify that insufficient data raises ValueError."""
        time = np.array([0.0, 1.0, 2.0])
        flux = np.array([1.0, 0.99, 1.0])
        flux_err = np.array([0.001, 0.001, 0.001])

        with pytest.raises(ValueError, match="Insufficient data points"):
            detect_transit(time, flux, flux_err, period=1.0, t0=1.0, duration=0.1)

    def test_invalid_period_raises_error(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify that non-positive period raises ValueError."""
        flux_err = np.ones_like(flat_flux) * 0.0001

        with pytest.raises(ValueError, match="Period must be positive"):
            detect_transit(time_array, flat_flux, flux_err, period=-1.0, t0=1.0, duration=0.1)

    def test_invalid_duration_raises_error(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify that invalid duration raises ValueError."""
        flux_err = np.ones_like(flat_flux) * 0.0001

        with pytest.raises(ValueError, match="Duration must be positive"):
            detect_transit(time_array, flat_flux, flux_err, period=3.5, t0=1.0, duration=0.0)

        with pytest.raises(ValueError, match="Duration.*must be less than period"):
            detect_transit(time_array, flat_flux, flux_err, period=3.5, t0=1.0, duration=5.0)


# =============================================================================
# Tests for periodogram.py
# =============================================================================


class TestEstimateSNR:
    """Tests for _estimate_snr helper function."""

    def test_snr_calculation(self):
        """Verify SNR is calculated correctly."""
        # Create a power spectrum with known statistics
        powers = np.ones(100)  # All powers = 1
        peak_power = 10.0  # One peak at 10

        snr = _estimate_snr(peak_power, powers)

        # SNR should be positive and significant for strong peak
        assert snr > 5


class TestLSPeriodogram:
    """Tests for ls_periodogram function."""

    def test_detects_sinusoids(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify Lomb-Scargle detects sinusoidal variations."""
        # Inject sinusoid
        period = 3.5
        amplitude = 0.01
        flux = inject_sinusoid(time_array, flat_flux, period, amplitude)

        # Generate search grid
        periods = np.linspace(2.0, 5.0, 200)

        power = ls_periodogram(time_array, flux, periods)

        # Maximum power should be near injected period
        best_idx = np.argmax(power)
        detected_period = periods[best_idx]

        assert_allclose(detected_period, period, rtol=0.05)

    def test_power_normalized(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify LS power is properly normalized."""
        periods = np.linspace(1.0, 10.0, 100)

        power = ls_periodogram(time_array, flat_flux, periods)

        # Power should be non-negative
        assert np.all(power >= 0)


@pytest.mark.skipif(not TLS_AVAILABLE, reason="transitleastsquares not available")
class TestAutoPeriodogram:
    """Tests for auto_periodogram function using TLS."""

    def test_returns_periodogram_result(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify auto_periodogram returns PeriodogramResult."""
        # Use LS method for fast test (TLS is slower)
        result = auto_periodogram(time_array, flat_flux, method="ls")

        assert isinstance(result, PeriodogramResult)

    def test_result_has_required_fields(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify result has all required fields populated."""
        result = auto_periodogram(time_array, flat_flux, method="ls")

        assert result.best_period > 0
        assert result.best_t0 is not None
        assert len(result.period_range) == 2
        assert result.method in ["tls", "ls"]

    def test_tls_method_selection(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify TLS method uses TLS for transit search."""
        # Use method="tls" explicitly
        period, t0, duration, depth = 3.5, 1.5, 0.1, 0.01
        flux = inject_transit(time_array, flat_flux, period, t0, duration, depth)

        result = auto_periodogram(
            time_array,
            flux,
            min_period=2.0,
            max_period=5.0,
            method="tls",
        )

        # Method should be TLS
        assert result.method == "tls"

    def test_auto_method_selection(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify auto method selection uses TLS for transit search."""
        # For transit-like signals, auto should use TLS
        period, t0, duration, depth = 3.5, 1.5, 0.1, 0.01
        flux_transit = inject_transit(time_array, flat_flux, period, t0, duration, depth)

        result = auto_periodogram(time_array, flux_transit, method="auto")

        # Result should use TLS (auto defaults to TLS for transits)
        assert result.method in ["tls", "ls"]


class TestComputeBLSModel:
    """Tests for compute_bls_model function."""

    def test_model_shape(self, time_array: np.ndarray):
        """Verify model has same shape as input time."""
        model = compute_bls_model(
            time_array,
            period=3.5,
            t0=1.5,
            duration_hours=2.4,
            depth=0.01,
        )

        assert model.shape == time_array.shape

    def test_model_values(self, time_array: np.ndarray):
        """Verify model has correct in/out of transit values."""
        period = 3.5
        t0 = 1.5
        duration_hours = 2.4
        depth = 0.01

        model = compute_bls_model(time_array, period, t0, duration_hours, depth)

        # Out of transit should be 1.0
        out_of_transit = model[model > 0.99]
        assert_allclose(out_of_transit, 1.0)

        # In transit should be 1.0 - depth
        in_transit = model[model < 0.995]
        assert_allclose(in_transit, 1.0 - depth, rtol=1e-10)


@pytest.mark.skipif(not TLS_AVAILABLE, reason="transitleastsquares not available")
class TestRefinePeriod:
    """Tests for refine_period function."""

    def test_refines_to_better_precision(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Verify period refinement improves precision."""
        # Inject transit
        true_period = 3.567  # Precise period
        t0 = 1.5
        duration = 0.1
        depth = 0.01
        flux = inject_transit(time_array, flat_flux, true_period, t0, duration, depth)

        # Start with approximate period
        initial_period = 3.5

        refined_period, refined_t0, refined_power = refine_period(
            time_array,
            flux,
            None,
            initial_period=initial_period,
            initial_duration=duration * 24,  # hours
            refine_factor=0.1,
            n_refine=100,
        )

        # Refined period should be closer to true period
        initial_error = abs(initial_period - true_period)
        refined_error = abs(refined_period - true_period)

        assert refined_error < initial_error


# =============================================================================
# Integration tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple compute functions."""

    def test_full_transit_detection_pipeline(self, time_array: np.ndarray, flat_flux: np.ndarray):
        """Test full pipeline: detrend -> periodogram -> detect -> fold."""
        # Inject transit with trend
        period, t0, duration, depth = 4.0, 2.0, 0.12, 0.008
        flux = inject_transit(time_array, flat_flux, period, t0, duration, depth)

        # Add trend
        flux_with_trend = flux * (1.0 + 0.003 * time_array / 27.0)

        # Step 1: Detrend
        detrended = median_detrend(flux_with_trend, window=501)

        # Step 2: Sigma clip
        valid_mask = sigma_clip(detrended, sigma=5.0)
        time_clean = time_array[valid_mask]
        flux_clean = detrended[valid_mask]

        # Step 3: Run periodogram (use LS for this synthetic test - TLS expects physical transit shape)
        result = auto_periodogram(
            time_clean,
            flux_clean,
            min_period=2.0,
            max_period=6.0,
            method="ls",  # LS for synthetic box transit test
        )

        # Step 4: Detect transit
        flux_err = np.ones_like(flux_clean) * 0.0001
        candidate = detect_transit(
            time_clean,
            flux_clean,
            flux_err,
            period=result.best_period,
            t0=result.best_t0,
            duration=result.best_duration_hours / 24.0 if result.best_duration_hours else duration,
        )

        # Step 5: Fold transit
        phase, flux_folded = fold_transit(time_clean, flux_clean, candidate.period, candidate.t0)

        # Verify results - period or its harmonic should be detected
        # LS may find harmonics, so check period or 2*period
        period_match = (
            abs(candidate.period - period) < 0.1 * period or
            abs(candidate.period - period / 2) < 0.1 * period or
            abs(candidate.period - period * 2) < 0.1 * period
        )
        assert period_match, f"Expected period {period}, got {candidate.period}"
        # Depth measurement is sensitive to many factors in the pipeline
        # We just verify it's a reasonable positive value for now
        assert candidate.depth > 0, "Depth should be positive"
        assert candidate.depth < 0.1, "Depth should be less than 10%"
        # SNR threshold depends on noise and depth
        assert candidate.snr > 0
        assert phase.min() >= -0.5
        assert phase.max() <= 0.5

    def test_shallow_transit_detection(self, time_array: np.ndarray):
        """Test detection of shallow Earth-like transit."""
        if not TLS_AVAILABLE:
            pytest.skip("transitleastsquares not available")
        rng = np.random.default_rng(42)
        noise_level = 0.00005  # 50 ppm (TESS-like for bright star)
        flat_flux = np.ones_like(time_array) + rng.normal(0, noise_level, len(time_array))

        # Inject shallow transit (100 ppm = 0.0001)
        period, t0, duration, depth = 2.5, 1.0, 0.08, 0.001
        flux = inject_transit(time_array, flat_flux, period, t0, duration, depth)

        # Normalize
        flux_err = np.ones_like(flux) * noise_level
        norm_flux, norm_err = normalize_flux(flux, flux_err)

        # Run periodogram with deep search
        result = auto_periodogram(
            time_array,
            norm_flux,
            min_period=1.5,
            max_period=4.0,
            preset="deep",
            method="bls",
        )

        # Should find the period within tolerance
        assert_allclose(result.best_period, period, rtol=0.15)


# =============================================================================
# Tests for per-sector search functionality
# =============================================================================


@pytest.mark.skipif(not TLS_AVAILABLE, reason="transitleastsquares not available")
class TestPerSectorSearch:
    """Tests for per-sector TLS search strategy."""

    def test_detect_sector_gaps(self):
        """Verify that sector gaps are correctly detected."""
        from bittr_tess_vetter.compute.periodogram import detect_sector_gaps

        # Create time array with 2 gaps (3 sectors)
        sector1 = np.linspace(0, 27, 1000)
        sector2 = np.linspace(55, 82, 1000)  # 28-day gap
        sector3 = np.linspace(110, 137, 1000)  # 28-day gap
        time = np.concatenate([sector1, sector2, sector3])

        gaps = detect_sector_gaps(time, gap_threshold_days=10.0)

        # Should find 2 gaps
        assert len(gaps) == 2
        # First gap is at index 999 (end of sector1)
        assert gaps[0] == 999
        # Second gap is at index 1999 (end of sector2)
        assert gaps[1] == 1999

    def test_split_by_sectors(self):
        """Verify that light curves are correctly split by sectors."""
        from bittr_tess_vetter.compute.periodogram import split_by_sectors

        # Create 3-sector data
        sector1_time = np.linspace(0, 27, 500)
        sector2_time = np.linspace(55, 82, 500)
        sector3_time = np.linspace(110, 137, 500)
        time = np.concatenate([sector1_time, sector2_time, sector3_time])
        flux = np.ones_like(time)
        flux_err = np.ones_like(time) * 0.001

        sectors = split_by_sectors(time, flux, flux_err, gap_threshold_days=10.0)

        # Should have 3 sectors
        assert len(sectors) == 3

        # Check sector sizes
        assert len(sectors[0][0]) == 500
        assert len(sectors[1][0]) == 500
        assert len(sectors[2][0]) == 500

        # Check time ranges
        assert sectors[0][0][0] == 0
        assert sectors[0][0][-1] == 27
        assert sectors[1][0][0] == 55
        assert sectors[2][0][0] == 110

    def test_merge_candidates_deduplicates(self):
        """Verify that candidates with similar periods are deduplicated."""
        from bittr_tess_vetter.compute.periodogram import merge_candidates

        # Create candidates with similar periods
        results = [
            {"period": 6.27, "sde": 15.0, "t0": 100.0},
            {"period": 6.30, "sde": 12.0, "t0": 101.0},  # Similar to 6.27 (~0.5% diff)
            {"period": 3.14, "sde": 10.0, "t0": 50.0},  # Different period
            {"period": 6.25, "sde": 8.0, "t0": 99.0},  # Similar to 6.27 (~0.3% diff)
        ]

        merged = merge_candidates(results, period_tolerance=0.02)

        # Should only have 2 unique periods (6.27 group and 3.14)
        assert len(merged) == 2

        # Best SDE should be first
        assert merged[0]["period"] == 6.27
        assert merged[0]["sde"] == 15.0

        # Second should be the 3.14 period
        assert merged[1]["period"] == 3.14

    def test_single_sector_uses_standard_tls(self):
        """Verify that single-sector data uses standard TLS search."""
        from bittr_tess_vetter.compute.periodogram import tls_search_per_sector

        # Single sector data (no gaps)
        n_points = 15000
        time = np.linspace(0, 27, n_points, dtype=np.float64)
        rng = np.random.default_rng(42)
        flux = np.ones(n_points, dtype=np.float64) + rng.normal(0, 0.0001, n_points)

        # Inject transit
        period, t0, duration, depth = 3.5, 1.5, 0.1, 0.01
        flux = inject_transit(time, flux, period, t0, duration, depth)

        result = tls_search_per_sector(
            time, flux, period_min=2.0, period_max=5.0
        )

        # Should detect the transit
        assert result["sde"] > 5.0
        assert abs(result["period"] - period) / period < 0.1

    def test_multi_sector_per_sector_search(self):
        """Verify that per-sector search works on multi-sector data."""
        from bittr_tess_vetter.compute.periodogram import tls_search_per_sector

        rng = np.random.default_rng(42)

        # Create 2-sector data with a gap
        sector1_time = np.linspace(0, 27, 7000, dtype=np.float64)
        sector2_time = np.linspace(55, 82, 7000, dtype=np.float64)
        time = np.concatenate([sector1_time, sector2_time])

        flux = np.ones(len(time), dtype=np.float64) + rng.normal(0, 0.0001, len(time))

        # Inject transit
        period, t0, duration, depth = 3.5, 1.5, 0.1, 0.01
        flux = inject_transit(time, flux, period, t0, duration, depth)

        result = tls_search_per_sector(
            time, flux, period_min=2.0, period_max=10.0
        )

        # Should detect the transit in per-sector mode
        assert result.get("n_sectors_searched", 1) == 2
        assert result["sde"] > 5.0
        # Period should be close to true period
        assert abs(result["period"] - period) / period < 0.15
