"""Unit tests for BLS-like periodogram search implementation.

Tests the bls_like_search module which provides:
- NumPy-based BLS-like transit search
- Rolling mean computation for phase binning
- Period grid exploration
- Top-K candidate selection

This is a fallback/comparison implementation - TLS is the primary method.

All tests are deterministic and require no network or file I/O.
"""

from __future__ import annotations

import numpy as np
import pytest

from bittr_tess_vetter.compute.bls_like_search import (
    BlsLikeCandidate,
    BlsLikeSearchResult,
    _bls_score_from_binned_flux,
    _phase_bin_means,
    _rolling_mean_circular,
    bls_like_search_numpy,
    bls_like_search_numpy_top_k,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def time_array() -> np.ndarray:
    """Synthetic time array spanning 27 days with cadence ~2 minutes."""
    return np.linspace(1000.0, 1027.0, 20000)


@pytest.fixture
def flux_err(time_array: np.ndarray) -> np.ndarray:
    """Flux uncertainty array (constant 100 ppm)."""
    return np.full_like(time_array, 100e-6)


def make_box_transit(
    time: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
    depth: float,
) -> np.ndarray:
    """Create synthetic box transit light curve."""
    duration_days = duration_hours / 24.0
    half_dur = duration_days / 2.0
    phase = (time - t0) / period
    phase = phase - np.floor(phase + 0.5)
    in_transit = np.abs(phase * period) < half_dur
    flux = np.ones_like(time)
    flux[in_transit] = 1.0 - depth
    return flux


# =============================================================================
# Test _rolling_mean_circular
# =============================================================================


class TestRollingMeanCircular:
    """Tests for circular rolling mean helper function."""

    def test_window_one_returns_copy(self):
        """Test that window=1 returns a copy of input."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _rolling_mean_circular(x, window=1)
        np.testing.assert_array_equal(result, x)
        # Should be a copy, not same object
        assert result is not x

    def test_window_size_two(self):
        """Test rolling mean with window=2."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = _rolling_mean_circular(x, window=2)
        # Expected: [avg(1,2), avg(2,3), avg(3,4), avg(4,1)]
        expected = np.array([1.5, 2.5, 3.5, 2.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_window_full_length(self):
        """Test rolling mean with window equal to array length."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = _rolling_mean_circular(x, window=4)
        # All values should be the mean of all elements
        expected = np.full(4, 2.5)
        np.testing.assert_array_almost_equal(result, expected)

    def test_circularity(self):
        """Test that wrapping is circular."""
        x = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
        result = _rolling_mean_circular(x, window=3)
        # First element: avg(10, 0, 0) = 3.33
        # Last element: avg(0, 0, 10) = 3.33 (wraps)
        assert result[0] == pytest.approx(10.0 / 3.0)
        assert result[-1] == pytest.approx(10.0 / 3.0)

    def test_invalid_window_zero(self):
        """Test that window=0 raises error."""
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="window must be >= 1"):
            _rolling_mean_circular(x, window=0)

    def test_invalid_window_negative(self):
        """Test that negative window raises error."""
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="window must be >= 1"):
            _rolling_mean_circular(x, window=-1)

    def test_preserves_length(self):
        """Test that output has same length as input."""
        x = np.random.randn(100)
        for window in [1, 5, 10, 50]:
            result = _rolling_mean_circular(x, window=window)
            assert len(result) == len(x)


# =============================================================================
# Test _phase_bin_means
# =============================================================================


class TestPhaseBinMeans:
    """Tests for phase binning helper function."""

    def test_output_shapes(self):
        """Test that outputs have correct shapes."""
        time = np.linspace(0, 10, 1000)
        flux = np.ones(1000)
        nbins = 50
        means, counts = _phase_bin_means(time, flux, period=2.5, nbins=nbins)
        assert len(means) == nbins
        assert len(counts) == nbins

    def test_uniform_flux_gives_uniform_means(self):
        """Test that uniform flux gives uniform bin means."""
        time = np.linspace(0, 100, 10000)
        flux = np.ones(10000) * 0.5
        nbins = 20
        means, counts = _phase_bin_means(time, flux, period=5.0, nbins=nbins)
        # All means should be 0.5
        valid = counts > 0
        np.testing.assert_array_almost_equal(means[valid], 0.5)

    def test_transit_creates_dip(self):
        """Test that transit signal creates dip in binned flux."""
        time = np.linspace(0, 100, 10000)
        period = 5.0
        flux = make_box_transit(time, period, t0=0.0, duration_hours=1.0, depth=0.01)
        nbins = 100
        means, counts = _phase_bin_means(time, flux, period=period, nbins=nbins)
        valid = counts > 0
        # Minimum should be below 1.0 due to transit (diluted by bin averaging)
        assert np.nanmin(means[valid]) < 0.998

    def test_empty_bins_are_nan(self):
        """Test that empty bins have NaN values."""
        # Very sparse time array
        time = np.array([0.0, 0.1, 0.2])  # Only first few bins populated
        flux = np.array([1.0, 1.0, 1.0])
        nbins = 100
        means, counts = _phase_bin_means(time, flux, period=10.0, nbins=nbins)
        # Most bins should be empty (count=0) with NaN means
        empty_bins = counts == 0
        assert np.sum(empty_bins) > 90
        assert np.all(np.isnan(means[empty_bins]))

    def test_counts_sum_to_total_points(self):
        """Test that bin counts sum to total number of points."""
        time = np.linspace(0, 50, 5000)
        flux = np.ones(5000)
        nbins = 50
        means, counts = _phase_bin_means(time, flux, period=2.5, nbins=nbins)
        assert np.sum(counts) == 5000


# =============================================================================
# Test _bls_score_from_binned_flux
# =============================================================================


class TestBlsScoreFromBinnedFlux:
    """Tests for BLS score computation from binned flux."""

    def test_returns_score_and_index(self):
        """Test that function returns score and index."""
        binned_flux = np.ones(100) - 0.01 * (np.arange(100) == 50).astype(float)
        binned_counts = np.ones(100, dtype=np.int64) * 10
        score, min_idx = _bls_score_from_binned_flux(
            binned_flux, binned_counts, duration_bins=5
        )
        assert isinstance(score, float)
        assert isinstance(min_idx, int)

    def test_finds_transit_location(self):
        """Test that minimum index corresponds to transit location."""
        nbins = 100
        binned_flux = np.ones(nbins)
        # Create dip at bin 30
        transit_bin = 30
        binned_flux[transit_bin - 2 : transit_bin + 3] = 0.99
        binned_counts = np.ones(nbins, dtype=np.int64) * 10

        score, min_idx = _bls_score_from_binned_flux(
            binned_flux, binned_counts, duration_bins=5
        )

        # Min index should be near transit location
        assert abs(min_idx - transit_bin) < 5

    def test_score_positive_for_transit(self):
        """Test that score is positive for transit-like signal."""
        nbins = 100
        binned_flux = np.ones(nbins)
        binned_flux[45:55] = 0.99  # Transit dip
        binned_counts = np.ones(nbins, dtype=np.int64) * 10

        score, min_idx = _bls_score_from_binned_flux(
            binned_flux, binned_counts, duration_bins=10
        )

        assert score > 0

    def test_returns_negative_inf_for_sparse_data(self):
        """Test that sparse data returns -inf score."""
        nbins = 100
        binned_flux = np.ones(nbins)
        binned_counts = np.zeros(nbins, dtype=np.int64)
        binned_counts[:3] = 1  # Only 3 valid bins

        score, min_idx = _bls_score_from_binned_flux(
            binned_flux, binned_counts, duration_bins=5
        )

        assert score == float("-inf")

    def test_deeper_transit_higher_score(self):
        """Test that deeper transits give higher scores."""
        nbins = 100
        binned_counts = np.ones(nbins, dtype=np.int64) * 10

        # Shallow transit
        flux_shallow = np.ones(nbins)
        flux_shallow[45:55] = 0.999
        score_shallow, _ = _bls_score_from_binned_flux(
            flux_shallow, binned_counts, duration_bins=10
        )

        # Deep transit
        flux_deep = np.ones(nbins)
        flux_deep[45:55] = 0.99
        score_deep, _ = _bls_score_from_binned_flux(
            flux_deep, binned_counts, duration_bins=10
        )

        assert score_deep > score_shallow


# =============================================================================
# Test BlsLikeSearchResult Dataclass
# =============================================================================


class TestBlsLikeSearchResult:
    """Tests for the BlsLikeSearchResult dataclass."""

    def test_attributes(self):
        """Test that dataclass has expected attributes."""
        result = BlsLikeSearchResult(
            method="numpy_bls_like",
            best_period_days=3.5,
            best_t0_btjd=1001.0,
            best_duration_hours=2.5,
            score=15.0,
            runtime_seconds=1.5,
            notes={"nbins": 200},
        )
        assert result.method == "numpy_bls_like"
        assert result.best_period_days == 3.5
        assert result.best_t0_btjd == 1001.0
        assert result.best_duration_hours == 2.5
        assert result.score == 15.0
        assert result.runtime_seconds == 1.5
        assert result.notes["nbins"] == 200

    def test_frozen(self):
        """Test that dataclass is immutable."""
        result = BlsLikeSearchResult(
            method="numpy_bls_like",
            best_period_days=3.5,
            best_t0_btjd=1001.0,
            best_duration_hours=2.5,
            score=15.0,
            runtime_seconds=1.5,
            notes={},
        )
        with pytest.raises(AttributeError):
            result.score = 20.0  # type: ignore


# =============================================================================
# Test BlsLikeCandidate Dataclass
# =============================================================================


class TestBlsLikeCandidate:
    """Tests for the BlsLikeCandidate dataclass."""

    def test_attributes(self):
        """Test that dataclass has expected attributes."""
        candidate = BlsLikeCandidate(
            period_days=3.5,
            t0_btjd=1001.0,
            duration_hours=2.5,
            score=15.0,
        )
        assert candidate.period_days == 3.5
        assert candidate.t0_btjd == 1001.0
        assert candidate.duration_hours == 2.5
        assert candidate.score == 15.0

    def test_frozen(self):
        """Test that dataclass is immutable."""
        candidate = BlsLikeCandidate(
            period_days=3.5,
            t0_btjd=1001.0,
            duration_hours=2.5,
            score=15.0,
        )
        with pytest.raises(AttributeError):
            candidate.score = 20.0  # type: ignore


# =============================================================================
# Test bls_like_search_numpy
# =============================================================================


class TestBlsLikeSearchNumpy:
    """Tests for the main BLS-like search function."""

    def test_returns_result_structure(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that search returns correct result structure."""
        flux = np.ones_like(time_array) + rng.normal(0, 50e-6, len(time_array))
        period_grid = np.linspace(2.0, 5.0, 10)
        duration_hours_grid = [2.0, 3.0]

        result = bls_like_search_numpy(
            time_btjd=time_array,
            flux=flux,
            flux_err=flux_err,
            period_grid=period_grid,
            duration_hours_grid=duration_hours_grid,
        )

        assert isinstance(result, BlsLikeSearchResult)
        assert result.method == "numpy_bls_like"
        assert result.runtime_seconds > 0
        assert "nbins" in result.notes
        assert "n_periods" in result.notes
        assert result.notes["n_periods"] == 10
        assert result.notes["n_durations"] == 2

    def test_finds_injected_transit(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that search finds injected transit signal."""
        true_period = 3.5
        true_t0 = 1001.0
        duration_hours = 2.5
        depth = 0.002

        flux = make_box_transit(time_array, true_period, true_t0, duration_hours, depth)
        flux += rng.normal(0, 50e-6, len(flux))

        period_grid = np.linspace(3.0, 4.0, 20)
        duration_hours_grid = [2.0, 2.5, 3.0]

        result = bls_like_search_numpy(
            time_btjd=time_array,
            flux=flux,
            flux_err=flux_err,
            period_grid=period_grid,
            duration_hours_grid=duration_hours_grid,
        )

        # Best period should be close to true period
        assert abs(result.best_period_days - true_period) < 0.2
        # Score should be positive
        assert result.score > 0

    def test_best_duration_from_grid(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that best duration is from the provided grid."""
        flux = make_box_transit(time_array, 3.5, 1001.0, 2.5, 0.001)
        flux += rng.normal(0, 50e-6, len(flux))

        period_grid = np.array([3.5])
        duration_hours_grid = [1.5, 2.0, 2.5, 3.0, 3.5]

        result = bls_like_search_numpy(
            time_btjd=time_array,
            flux=flux,
            flux_err=flux_err,
            period_grid=period_grid,
            duration_hours_grid=duration_hours_grid,
        )

        assert result.best_duration_hours in duration_hours_grid

    def test_handles_no_flux_err(
        self,
        time_array: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that search works without flux errors."""
        flux = make_box_transit(time_array, 3.5, 1001.0, 2.5, 0.001)
        flux += rng.normal(0, 50e-6, len(flux))

        period_grid = np.array([3.5])
        duration_hours_grid = [2.5]

        result = bls_like_search_numpy(
            time_btjd=time_array,
            flux=flux,
            flux_err=None,
            period_grid=period_grid,
            duration_hours_grid=duration_hours_grid,
        )

        assert np.isfinite(result.score)

    def test_local_refinement(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that local refinement improves T0 estimate."""
        true_period = 3.5
        true_t0 = 1001.0
        duration_hours = 2.5
        depth = 0.002

        flux = make_box_transit(time_array, true_period, true_t0, duration_hours, depth)
        flux += rng.normal(0, 50e-6, len(flux))

        period_grid = np.array([true_period])
        duration_hours_grid = [duration_hours]

        # With refinement
        result_refined = bls_like_search_numpy(
            time_btjd=time_array,
            flux=flux,
            flux_err=flux_err,
            period_grid=period_grid,
            duration_hours_grid=duration_hours_grid,
            local_refine_steps=21,
        )

        # Without refinement
        result_no_refine = bls_like_search_numpy(
            time_btjd=time_array,
            flux=flux,
            flux_err=flux_err,
            period_grid=period_grid,
            duration_hours_grid=duration_hours_grid,
            local_refine_steps=1,
        )

        # Refined should have better or equal score
        assert result_refined.score >= result_no_refine.score - 0.01

    def test_custom_nbins(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test custom number of phase bins."""
        flux = make_box_transit(time_array, 3.5, 1001.0, 2.5, 0.001)
        flux += rng.normal(0, 50e-6, len(flux))

        period_grid = np.array([3.5])
        duration_hours_grid = [2.5]

        result = bls_like_search_numpy(
            time_btjd=time_array,
            flux=flux,
            flux_err=flux_err,
            period_grid=period_grid,
            duration_hours_grid=duration_hours_grid,
            nbins=100,
        )

        assert result.notes["nbins"] == 100


# =============================================================================
# Test bls_like_search_numpy_top_k
# =============================================================================


class TestBlsLikeSearchNumpyTopK:
    """Tests for the top-K BLS-like search function."""

    def test_returns_result_and_candidates(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that search returns both result and candidates."""
        flux = make_box_transit(time_array, 3.5, 1001.0, 2.5, 0.001)
        flux += rng.normal(0, 50e-6, len(flux))

        period_grid = np.linspace(2.0, 5.0, 20)
        duration_hours_grid = [2.0, 2.5, 3.0]

        result, candidates = bls_like_search_numpy_top_k(
            time_btjd=time_array,
            flux=flux,
            flux_err=flux_err,
            period_grid=period_grid,
            duration_hours_grid=duration_hours_grid,
            top_k=5,
        )

        assert isinstance(result, BlsLikeSearchResult)
        assert isinstance(candidates, list)
        assert len(candidates) <= 5
        assert all(isinstance(c, BlsLikeCandidate) for c in candidates)

    def test_candidates_sorted_by_score(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that candidates are sorted by score descending."""
        flux = make_box_transit(time_array, 3.5, 1001.0, 2.5, 0.001)
        flux += rng.normal(0, 50e-6, len(flux))

        period_grid = np.linspace(2.0, 5.0, 30)
        duration_hours_grid = [2.0, 2.5, 3.0]

        _, candidates = bls_like_search_numpy_top_k(
            time_btjd=time_array,
            flux=flux,
            flux_err=flux_err,
            period_grid=period_grid,
            duration_hours_grid=duration_hours_grid,
            top_k=10,
        )

        # Check descending order
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_best_result_matches_top_candidate(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that best result matches top candidate."""
        flux = make_box_transit(time_array, 3.5, 1001.0, 2.5, 0.001)
        flux += rng.normal(0, 50e-6, len(flux))

        period_grid = np.linspace(3.0, 4.0, 20)
        duration_hours_grid = [2.5]

        result, candidates = bls_like_search_numpy_top_k(
            time_btjd=time_array,
            flux=flux,
            flux_err=flux_err,
            period_grid=period_grid,
            duration_hours_grid=duration_hours_grid,
            top_k=5,
        )

        if candidates:
            top_candidate = candidates[0]
            assert result.best_period_days == top_candidate.period_days
            assert result.best_t0_btjd == top_candidate.t0_btjd
            assert result.best_duration_hours == top_candidate.duration_hours
            assert result.score == top_candidate.score

    def test_top_k_limits_candidates(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that top_k parameter limits number of candidates."""
        flux = make_box_transit(time_array, 3.5, 1001.0, 2.5, 0.001)
        flux += rng.normal(0, 50e-6, len(flux))

        period_grid = np.linspace(2.0, 6.0, 50)
        duration_hours_grid = [2.0, 2.5, 3.0]

        _, candidates_5 = bls_like_search_numpy_top_k(
            time_btjd=time_array,
            flux=flux,
            flux_err=flux_err,
            period_grid=period_grid,
            duration_hours_grid=duration_hours_grid,
            top_k=5,
        )

        _, candidates_10 = bls_like_search_numpy_top_k(
            time_btjd=time_array,
            flux=flux,
            flux_err=flux_err,
            period_grid=period_grid,
            duration_hours_grid=duration_hours_grid,
            top_k=10,
        )

        assert len(candidates_5) <= 5
        assert len(candidates_10) <= 10

    def test_top_k_validation(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
    ):
        """Test that invalid top_k raises error."""
        flux = np.ones_like(time_array)
        period_grid = np.array([3.5])
        duration_hours_grid = [2.5]

        with pytest.raises(ValueError, match="top_k must be >= 1"):
            bls_like_search_numpy_top_k(
                time_btjd=time_array,
                flux=flux,
                flux_err=flux_err,
                period_grid=period_grid,
                duration_hours_grid=duration_hours_grid,
                top_k=0,
            )

    def test_notes_include_top_k(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that notes include top_k parameter."""
        flux = make_box_transit(time_array, 3.5, 1001.0, 2.5, 0.001)
        flux += rng.normal(0, 50e-6, len(flux))

        period_grid = np.array([3.5])
        duration_hours_grid = [2.5]

        result, _ = bls_like_search_numpy_top_k(
            time_btjd=time_array,
            flux=flux,
            flux_err=flux_err,
            period_grid=period_grid,
            duration_hours_grid=duration_hours_grid,
            top_k=7,
        )

        assert result.notes["top_k"] == 7


# =============================================================================
# Test Integration
# =============================================================================


class TestBlsLikeSearchIntegration:
    """Integration tests for BLS-like search."""

    def test_recovery_of_known_planet(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test recovery of a planet with known parameters."""
        true_period = 4.2
        true_t0 = 1002.5
        true_duration = 3.0
        true_depth = 0.0015

        flux = make_box_transit(
            time_array, true_period, true_t0, true_duration, true_depth
        )
        flux += rng.normal(0, 40e-6, len(flux))

        # Search with fine period grid
        period_grid = np.linspace(3.5, 5.0, 50)
        duration_hours_grid = [2.0, 2.5, 3.0, 3.5, 4.0]

        result = bls_like_search_numpy(
            time_btjd=time_array,
            flux=flux,
            flux_err=flux_err,
            period_grid=period_grid,
            duration_hours_grid=duration_hours_grid,
            nbins=200,
            local_refine_steps=21,
        )

        # Check recovery
        period_error = abs(result.best_period_days - true_period)
        assert period_error < 0.1, f"Period error {period_error} too large"
        assert result.score > 10, f"Score {result.score} too low"

    def test_multiple_candidates_different_periods(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that different periods give different candidates."""
        flux = make_box_transit(time_array, 3.5, 1001.0, 2.5, 0.0015)
        flux += rng.normal(0, 50e-6, len(flux))

        period_grid = np.linspace(2.5, 5.0, 50)
        duration_hours_grid = [2.5]

        _, candidates = bls_like_search_numpy_top_k(
            time_btjd=time_array,
            flux=flux,
            flux_err=flux_err,
            period_grid=period_grid,
            duration_hours_grid=duration_hours_grid,
            top_k=10,
        )

        # Should have multiple candidates at different periods
        periods = [c.period_days for c in candidates]
        # Check that we have some diversity (not all same period)
        assert len({round(p, 1) for p in periods}) > 1 or len(candidates) == 1

    def test_determinism(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that search is deterministic."""
        flux = make_box_transit(time_array, 3.5, 1001.0, 2.5, 0.001)
        flux += rng.normal(0, 50e-6, len(flux))

        period_grid = np.linspace(3.0, 4.0, 20)
        duration_hours_grid = [2.5]

        result1 = bls_like_search_numpy(
            time_btjd=time_array,
            flux=flux,
            flux_err=flux_err,
            period_grid=period_grid,
            duration_hours_grid=duration_hours_grid,
        )

        result2 = bls_like_search_numpy(
            time_btjd=time_array,
            flux=flux,
            flux_err=flux_err,
            period_grid=period_grid,
            duration_hours_grid=duration_hours_grid,
        )

        assert result1.best_period_days == result2.best_period_days
        assert result1.best_t0_btjd == result2.best_t0_btjd
        assert result1.score == result2.score

    def test_flat_light_curve_gives_low_score(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that flat light curve gives low score."""
        flux = np.ones_like(time_array) + rng.normal(0, 50e-6, len(time_array))

        period_grid = np.linspace(2.0, 5.0, 20)
        duration_hours_grid = [2.0, 3.0]

        result = bls_like_search_numpy(
            time_btjd=time_array,
            flux=flux,
            flux_err=flux_err,
            period_grid=period_grid,
            duration_hours_grid=duration_hours_grid,
        )

        # Score should be relatively low for no transit
        assert result.score < 5.0
