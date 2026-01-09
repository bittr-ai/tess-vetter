"""Unit tests for MLX-based transit detection primitives.

Tests the mlx_detection module which provides GPU-accelerated (Apple Silicon):
- Smooth box transit templates
- Differentiable matched-filter scoring
- Top-K period scoring
- Integrated gradients attribution
- T0 refinement

MLX is an optional dependency - tests are skipped if not installed.

All tests are deterministic and require no network or file I/O.
"""

from __future__ import annotations

import numpy as np
import pytest

# Check if MLX is available
try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

# Skip all tests in this module if MLX is not available
pytestmark = pytest.mark.skipif(
    not MLX_AVAILABLE,
    reason="MLX not installed (requires Apple Silicon)",
)


# Conditionally import module functions only if MLX is available
if MLX_AVAILABLE:
    from bittr_tess_vetter.compute.mlx_detection import (
        MlxT0RefinementResult,
        MlxTopKScoreResult,
        integrated_gradients,
        score_fixed_period,
        score_fixed_period_refine_t0,
        score_top_k_periods,
        smooth_box_template,
    )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def time_array_np() -> np.ndarray:
    """Synthetic time array spanning 27 days with cadence ~2 minutes."""
    return np.linspace(1000.0, 1027.0, 20000)


@pytest.fixture
def time_array_mx(time_array_np: np.ndarray):
    """MLX version of time array."""
    return mx.array(time_array_np)


@pytest.fixture
def flux_err_np(time_array_np: np.ndarray) -> np.ndarray:
    """Flux uncertainty array (constant 100 ppm)."""
    return np.full_like(time_array_np, 100e-6)


@pytest.fixture
def flux_err_mx(flux_err_np: np.ndarray):
    """MLX version of flux error array."""
    return mx.array(flux_err_np)


def make_box_transit_np(
    time: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
    depth: float,
) -> np.ndarray:
    """Create synthetic box transit light curve (NumPy)."""
    duration_days = duration_hours / 24.0
    half_dur = duration_days / 2.0
    phase = (time - t0) / period
    phase = phase - np.floor(phase + 0.5)
    in_transit = np.abs(phase * period) < half_dur
    flux = np.ones_like(time)
    flux[in_transit] = 1.0 - depth
    return flux


# =============================================================================
# Test smooth_box_template
# =============================================================================


class TestSmoothBoxTemplate:
    """Tests for the smooth transit template function."""

    def test_template_shape(self, time_array_mx):
        """Test that template has correct shape."""
        template = smooth_box_template(
            time=time_array_mx,
            period_days=3.5,
            t0_btjd=1001.0,
            duration_hours=2.5,
        )
        mx.eval(template)
        assert template.shape == time_array_mx.shape

    def test_template_bounded_zero_one(self, time_array_mx):
        """Test that template values are in [0, 1]."""
        template = smooth_box_template(
            time=time_array_mx,
            period_days=3.5,
            t0_btjd=1001.0,
            duration_hours=2.5,
        )
        mx.eval(template)
        template_np = np.array(template)
        assert np.all(template_np >= 0.0)
        assert np.all(template_np <= 1.0)

    def test_template_high_at_transit_center(self, time_array_mx):
        """Test that template is high at transit center."""
        t0 = 1001.0
        template = smooth_box_template(
            time=time_array_mx,
            period_days=3.5,
            t0_btjd=t0,
            duration_hours=2.5,
        )
        mx.eval(template)
        template_np = np.array(template)
        time_np = np.array(time_array_mx)

        # Find index closest to t0
        center_idx = np.argmin(np.abs(time_np - t0))
        assert template_np[center_idx] > 0.9

    def test_template_low_out_of_transit(self, time_array_mx):
        """Test that template is low far from transit."""
        t0 = 1001.0
        period = 3.5
        template = smooth_box_template(
            time=time_array_mx,
            period_days=period,
            t0_btjd=t0,
            duration_hours=2.5,
        )
        mx.eval(template)
        template_np = np.array(template)
        time_np = np.array(time_array_mx)

        # Find index at phase 0.25 (quarter period away from transit)
        target_time = t0 + period * 0.25
        far_idx = np.argmin(np.abs(time_np - target_time))
        assert template_np[far_idx] < 0.1

    def test_template_periodicity(self, time_array_mx):
        """Test that template is periodic."""
        t0 = 1001.0
        period = 3.5
        template = smooth_box_template(
            time=time_array_mx,
            period_days=period,
            t0_btjd=t0,
            duration_hours=2.5,
        )
        mx.eval(template)
        template_np = np.array(template)
        time_np = np.array(time_array_mx)

        # Values at t0 and t0 + period should be similar
        idx1 = np.argmin(np.abs(time_np - t0))
        idx2 = np.argmin(np.abs(time_np - (t0 + period)))
        assert abs(template_np[idx1] - template_np[idx2]) < 0.1

    def test_sharpness_parameter(self, time_array_mx):
        """Test that sharpness parameter affects transition width."""
        t0 = 1001.0
        period = 3.5
        duration = 2.5

        template_sharp = smooth_box_template(
            time=time_array_mx,
            period_days=period,
            t0_btjd=t0,
            duration_hours=duration,
            sharpness=100.0,
        )
        template_soft = smooth_box_template(
            time=time_array_mx,
            period_days=period,
            t0_btjd=t0,
            duration_hours=duration,
            sharpness=10.0,
        )
        mx.eval(template_sharp, template_soft)

        # Sharp template should have steeper transitions (more variance)
        sharp_np = np.array(template_sharp)
        soft_np = np.array(template_soft)

        # Count points in transition zone (0.1 < value < 0.9)
        sharp_transition = np.sum((sharp_np > 0.1) & (sharp_np < 0.9))
        soft_transition = np.sum((soft_np > 0.1) & (soft_np < 0.9))

        # Softer template has more transition points
        assert soft_transition > sharp_transition

    def test_ingress_egress_fraction(self, time_array_mx):
        """Test ingress/egress fraction parameter."""
        t0 = 1001.0
        template_narrow = smooth_box_template(
            time=time_array_mx,
            period_days=3.5,
            t0_btjd=t0,
            duration_hours=2.5,
            ingress_egress_fraction=0.1,
        )
        template_wide = smooth_box_template(
            time=time_array_mx,
            period_days=3.5,
            t0_btjd=t0,
            duration_hours=2.5,
            ingress_egress_fraction=0.4,
        )
        mx.eval(template_narrow, template_wide)

        narrow_np = np.array(template_narrow)
        wide_np = np.array(template_wide)

        # Both should still have values near 1 at center
        time_np = np.array(time_array_mx)
        center_idx = np.argmin(np.abs(time_np - t0))
        assert narrow_np[center_idx] > 0.8
        assert wide_np[center_idx] > 0.8


# =============================================================================
# Test score_fixed_period
# =============================================================================


class TestScoreFixedPeriod:
    """Tests for the fixed-period scoring function."""

    def test_score_returns_scalar(
        self,
        time_array_np: np.ndarray,
        flux_err_np: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that score returns a scalar value."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        depth = 0.001

        flux_np = make_box_transit_np(time_array_np, period, t0, duration_hours, depth)
        flux_np += rng.normal(0, 50e-6, len(flux_np))

        time_mx = mx.array(time_array_np)
        flux_mx = mx.array(flux_np)
        flux_err_mx = mx.array(flux_err_np)

        score = score_fixed_period(
            time=time_mx,
            flux=flux_mx,
            flux_err=flux_err_mx,
            period_days=period,
            t0_btjd=t0,
            duration_hours=duration_hours,
        )
        mx.eval(score)
        score_val = float(score)

        assert isinstance(score_val, float)
        assert np.isfinite(score_val)

    def test_higher_score_for_deeper_transit(
        self,
        time_array_np: np.ndarray,
        flux_err_np: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that deeper transits get higher scores."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5

        time_mx = mx.array(time_array_np)
        flux_err_mx = mx.array(flux_err_np)

        # Shallow transit
        flux_shallow = make_box_transit_np(
            time_array_np, period, t0, duration_hours, 0.0005
        )
        flux_shallow += rng.normal(0, 50e-6, len(flux_shallow))
        score_shallow = score_fixed_period(
            time=time_mx,
            flux=mx.array(flux_shallow),
            flux_err=flux_err_mx,
            period_days=period,
            t0_btjd=t0,
            duration_hours=duration_hours,
        )

        # Deep transit
        flux_deep = make_box_transit_np(
            time_array_np, period, t0, duration_hours, 0.002
        )
        flux_deep += rng.normal(0, 50e-6, len(flux_deep))
        score_deep = score_fixed_period(
            time=time_mx,
            flux=mx.array(flux_deep),
            flux_err=flux_err_mx,
            period_days=period,
            t0_btjd=t0,
            duration_hours=duration_hours,
        )

        mx.eval(score_shallow, score_deep)
        assert float(score_deep) > float(score_shallow)

    def test_low_score_for_no_transit(
        self,
        time_array_np: np.ndarray,
        flux_err_np: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that flat light curve gets low score."""
        flux_np = np.ones_like(time_array_np) + rng.normal(0, 50e-6, len(time_array_np))

        time_mx = mx.array(time_array_np)
        flux_mx = mx.array(flux_np)
        flux_err_mx = mx.array(flux_err_np)

        score = score_fixed_period(
            time=time_mx,
            flux=flux_mx,
            flux_err=flux_err_mx,
            period_days=3.5,
            t0_btjd=1001.0,
            duration_hours=2.5,
        )
        mx.eval(score)

        # Score should be near zero for no transit
        assert abs(float(score)) < 5.0

    def test_score_without_flux_err(
        self,
        time_array_np: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test scoring without flux errors (uniform weights)."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        depth = 0.001

        flux_np = make_box_transit_np(time_array_np, period, t0, duration_hours, depth)
        flux_np += rng.normal(0, 50e-6, len(flux_np))

        time_mx = mx.array(time_array_np)
        flux_mx = mx.array(flux_np)

        score = score_fixed_period(
            time=time_mx,
            flux=flux_mx,
            flux_err=None,
            period_days=period,
            t0_btjd=t0,
            duration_hours=duration_hours,
        )
        mx.eval(score)

        assert np.isfinite(float(score))
        assert float(score) > 0  # Should detect the transit


# =============================================================================
# Test score_top_k_periods
# =============================================================================


class TestScoreTopKPeriods:
    """Tests for top-K period scoring function."""

    def test_returns_correct_structure(
        self,
        time_array_np: np.ndarray,
        flux_err_np: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that result has correct structure."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        depth = 0.001

        flux_np = make_box_transit_np(time_array_np, period, t0, duration_hours, depth)
        flux_np += rng.normal(0, 50e-6, len(flux_np))

        time_mx = mx.array(time_array_np)
        flux_mx = mx.array(flux_np)
        flux_err_mx = mx.array(flux_err_np)

        periods_top_k = mx.array([3.0, 3.5, 4.0, 4.5, 5.0])

        result = score_top_k_periods(
            time=time_mx,
            flux=flux_mx,
            flux_err=flux_err_mx,
            periods_days_top_k=periods_top_k,
            t0_btjd=t0,
            duration_hours=duration_hours,
        )

        assert isinstance(result, MlxTopKScoreResult)
        mx.eval(result.scores, result.weights)

        assert result.scores.shape == (5,)
        assert result.weights.shape == (5,)

    def test_highest_score_at_true_period(
        self,
        time_array_np: np.ndarray,
        flux_err_np: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that true period gets highest score."""
        true_period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        depth = 0.002

        flux_np = make_box_transit_np(
            time_array_np, true_period, t0, duration_hours, depth
        )
        flux_np += rng.normal(0, 50e-6, len(flux_np))

        time_mx = mx.array(time_array_np)
        flux_mx = mx.array(flux_np)
        flux_err_mx = mx.array(flux_err_np)

        periods_top_k = mx.array([2.5, 3.0, 3.5, 4.0, 4.5])

        result = score_top_k_periods(
            time=time_mx,
            flux=flux_mx,
            flux_err=flux_err_mx,
            periods_days_top_k=periods_top_k,
            t0_btjd=t0,
            duration_hours=duration_hours,
        )
        mx.eval(result.scores)

        scores_np = np.array(result.scores)
        best_idx = np.argmax(scores_np)
        periods_np = np.array(periods_top_k)

        # Best period should be close to true period
        assert abs(periods_np[best_idx] - true_period) < 0.5

    def test_weights_sum_to_one(
        self,
        time_array_np: np.ndarray,
        flux_err_np: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that softmax weights sum to 1."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        depth = 0.001

        flux_np = make_box_transit_np(time_array_np, period, t0, duration_hours, depth)
        flux_np += rng.normal(0, 50e-6, len(flux_np))

        time_mx = mx.array(time_array_np)
        flux_mx = mx.array(flux_np)
        flux_err_mx = mx.array(flux_err_np)

        periods_top_k = mx.array([3.0, 3.5, 4.0])

        result = score_top_k_periods(
            time=time_mx,
            flux=flux_mx,
            flux_err=flux_err_mx,
            periods_days_top_k=periods_top_k,
            t0_btjd=t0,
            duration_hours=duration_hours,
        )
        mx.eval(result.weights)

        weights_np = np.array(result.weights)
        assert abs(weights_np.sum() - 1.0) < 1e-5

    def test_temperature_affects_weight_distribution(
        self,
        time_array_np: np.ndarray,
        flux_err_np: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that temperature parameter affects weight sharpness."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        depth = 0.002

        flux_np = make_box_transit_np(time_array_np, period, t0, duration_hours, depth)
        flux_np += rng.normal(0, 50e-6, len(flux_np))

        time_mx = mx.array(time_array_np)
        flux_mx = mx.array(flux_np)
        flux_err_mx = mx.array(flux_err_np)

        periods_top_k = mx.array([3.0, 3.5, 4.0])

        # Low temperature = sharper weights
        result_cold = score_top_k_periods(
            time=time_mx,
            flux=flux_mx,
            flux_err=flux_err_mx,
            periods_days_top_k=periods_top_k,
            t0_btjd=t0,
            duration_hours=duration_hours,
            temperature=0.1,
        )

        # High temperature = more uniform weights
        result_hot = score_top_k_periods(
            time=time_mx,
            flux=flux_mx,
            flux_err=flux_err_mx,
            periods_days_top_k=periods_top_k,
            t0_btjd=t0,
            duration_hours=duration_hours,
            temperature=2.0,
        )

        mx.eval(result_cold.weights, result_hot.weights)

        weights_cold = np.array(result_cold.weights)
        weights_hot = np.array(result_hot.weights)

        # Cold weights should have higher max (more peaked)
        assert np.max(weights_cold) > np.max(weights_hot)


# =============================================================================
# Test integrated_gradients
# =============================================================================


class TestIntegratedGradients:
    """Tests for integrated gradients attribution."""

    def test_returns_correct_shape(
        self,
        time_array_np: np.ndarray,
        flux_err_np: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that attribution has same shape as flux."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        depth = 0.001

        flux_np = make_box_transit_np(time_array_np, period, t0, duration_hours, depth)
        flux_np += rng.normal(0, 50e-6, len(flux_np))

        time_mx = mx.array(time_array_np)
        flux_mx = mx.array(flux_np)
        flux_err_mx = mx.array(flux_err_np)

        baseline = mx.ones_like(flux_mx)

        def score_fn(f):
            return score_fixed_period(
                time=time_mx,
                flux=f,
                flux_err=flux_err_mx,
                period_days=period,
                t0_btjd=t0,
                duration_hours=duration_hours,
            )

        attribution = integrated_gradients(
            score_fn=score_fn,
            flux=flux_mx,
            baseline=baseline,
            steps=20,
        )
        mx.eval(attribution)

        assert attribution.shape == flux_mx.shape

    def test_attribution_peaks_at_transit(
        self,
        time_array_np: np.ndarray,
        flux_err_np: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that attribution is higher at transit location."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        depth = 0.002

        flux_np = make_box_transit_np(time_array_np, period, t0, duration_hours, depth)
        flux_np += rng.normal(0, 30e-6, len(flux_np))

        time_mx = mx.array(time_array_np)
        flux_mx = mx.array(flux_np)
        flux_err_mx = mx.array(flux_err_np)

        baseline = mx.ones_like(flux_mx)

        def score_fn(f):
            return score_fixed_period(
                time=time_mx,
                flux=f,
                flux_err=flux_err_mx,
                period_days=period,
                t0_btjd=t0,
                duration_hours=duration_hours,
            )

        attribution = integrated_gradients(
            score_fn=score_fn,
            flux=flux_mx,
            baseline=baseline,
            steps=30,
        )
        mx.eval(attribution)

        attr_np = np.abs(np.array(attribution))

        # Find transit indices
        duration_days = duration_hours / 24.0
        phase = (time_array_np - t0) / period
        phase = phase - np.floor(phase + 0.5)
        in_transit = np.abs(phase * period) < (duration_days / 2.0)
        out_transit = ~in_transit

        # Attribution should be higher at transit points
        mean_in = np.mean(attr_np[in_transit])
        mean_out = np.mean(attr_np[out_transit])

        assert mean_in > mean_out


# =============================================================================
# Test score_fixed_period_refine_t0
# =============================================================================


class TestScoreFixedPeriodRefineT0:
    """Tests for T0 refinement function."""

    def test_returns_correct_structure(
        self,
        time_array_np: np.ndarray,
        flux_err_np: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that result has correct structure."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        depth = 0.001

        flux_np = make_box_transit_np(time_array_np, period, t0, duration_hours, depth)
        flux_np += rng.normal(0, 50e-6, len(flux_np))

        time_mx = mx.array(time_array_np)
        flux_mx = mx.array(flux_np)
        flux_err_mx = mx.array(flux_err_np)

        result = score_fixed_period_refine_t0(
            time=time_mx,
            flux=flux_mx,
            flux_err=flux_err_mx,
            period_days=period,
            t0_btjd=t0,
            duration_hours=duration_hours,
        )

        assert isinstance(result, MlxT0RefinementResult)
        assert isinstance(result.t0_best_btjd, float)
        assert isinstance(result.score_best, float)
        assert isinstance(result.score_at_input, float)
        assert isinstance(result.delta_score, float)
        assert isinstance(result.t0_grid_btjd, np.ndarray)
        assert isinstance(result.scores, np.ndarray)

    def test_finds_correct_t0(
        self,
        time_array_np: np.ndarray,
        flux_err_np: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that refinement finds the true T0."""
        period = 3.5
        true_t0 = 1001.0
        duration_hours = 2.5
        depth = 0.002

        flux_np = make_box_transit_np(
            time_array_np, period, true_t0, duration_hours, depth
        )
        flux_np += rng.normal(0, 30e-6, len(flux_np))

        time_mx = mx.array(time_array_np)
        flux_mx = mx.array(flux_np)
        flux_err_mx = mx.array(flux_err_np)

        # Start with slightly offset T0
        initial_t0 = true_t0 + 0.02  # 30 minutes off

        result = score_fixed_period_refine_t0(
            time=time_mx,
            flux=flux_mx,
            flux_err=flux_err_mx,
            period_days=period,
            t0_btjd=initial_t0,
            duration_hours=duration_hours,
            t0_scan_n=81,
        )

        # Refined T0 should be closer to true T0
        error_before = abs(initial_t0 - true_t0)
        error_after = abs(result.t0_best_btjd - true_t0)

        assert error_after < error_before

    def test_best_score_at_best_t0(
        self,
        time_array_np: np.ndarray,
        flux_err_np: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that best score corresponds to best T0 in grid."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        depth = 0.001

        flux_np = make_box_transit_np(time_array_np, period, t0, duration_hours, depth)
        flux_np += rng.normal(0, 50e-6, len(flux_np))

        time_mx = mx.array(time_array_np)
        flux_mx = mx.array(flux_np)
        flux_err_mx = mx.array(flux_err_np)

        result = score_fixed_period_refine_t0(
            time=time_mx,
            flux=flux_mx,
            flux_err=flux_err_mx,
            period_days=period,
            t0_btjd=t0,
            duration_hours=duration_hours,
        )

        # Best T0 should be at argmax of scores
        best_idx = np.argmax(result.scores)
        assert abs(result.t0_grid_btjd[best_idx] - result.t0_best_btjd) < 1e-10
        assert abs(result.scores[best_idx] - result.score_best) < 1e-10

    def test_delta_score_computed_correctly(
        self,
        time_array_np: np.ndarray,
        flux_err_np: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that delta_score is correctly computed."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        depth = 0.001

        flux_np = make_box_transit_np(time_array_np, period, t0, duration_hours, depth)
        flux_np += rng.normal(0, 50e-6, len(flux_np))

        time_mx = mx.array(time_array_np)
        flux_mx = mx.array(flux_np)
        flux_err_mx = mx.array(flux_err_np)

        result = score_fixed_period_refine_t0(
            time=time_mx,
            flux=flux_mx,
            flux_err=flux_err_mx,
            period_days=period,
            t0_btjd=t0,
            duration_hours=duration_hours,
        )

        expected_delta = result.score_best - result.score_at_input
        assert abs(result.delta_score - expected_delta) < 1e-10

    def test_minimum_scan_n(
        self,
        time_array_np: np.ndarray,
        flux_err_np: np.ndarray,
    ):
        """Test that minimum t0_scan_n is enforced."""
        time_mx = mx.array(time_array_np)
        flux_mx = mx.array(np.ones_like(time_array_np))
        flux_err_mx = mx.array(flux_err_np)

        with pytest.raises(ValueError, match="t0_scan_n must be >= 21"):
            score_fixed_period_refine_t0(
                time=time_mx,
                flux=flux_mx,
                flux_err=flux_err_mx,
                period_days=3.5,
                t0_btjd=1001.0,
                duration_hours=2.5,
                t0_scan_n=10,  # Too small
            )

    def test_custom_scan_half_span(
        self,
        time_array_np: np.ndarray,
        flux_err_np: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test custom scan half span."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        depth = 0.001

        flux_np = make_box_transit_np(time_array_np, period, t0, duration_hours, depth)
        flux_np += rng.normal(0, 50e-6, len(flux_np))

        time_mx = mx.array(time_array_np)
        flux_mx = mx.array(flux_np)
        flux_err_mx = mx.array(flux_err_np)

        result_narrow = score_fixed_period_refine_t0(
            time=time_mx,
            flux=flux_mx,
            flux_err=flux_err_mx,
            period_days=period,
            t0_btjd=t0,
            duration_hours=duration_hours,
            t0_scan_half_span_minutes=30.0,
        )

        result_wide = score_fixed_period_refine_t0(
            time=time_mx,
            flux=flux_mx,
            flux_err=flux_err_mx,
            period_days=period,
            t0_btjd=t0,
            duration_hours=duration_hours,
            t0_scan_half_span_minutes=120.0,
        )

        # Wide scan should cover more range
        narrow_range = result_narrow.t0_grid_btjd[-1] - result_narrow.t0_grid_btjd[0]
        wide_range = result_wide.t0_grid_btjd[-1] - result_wide.t0_grid_btjd[0]

        assert wide_range > narrow_range


# =============================================================================
# Test Integration
# =============================================================================


class TestMlxDetectionIntegration:
    """Integration tests for MLX detection module."""

    def test_full_pipeline_synthetic_transit(
        self,
        time_array_np: np.ndarray,
        flux_err_np: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test full detection pipeline on synthetic transit."""
        true_period = 3.5
        true_t0 = 1001.0
        duration_hours = 2.5
        depth = 0.002

        flux_np = make_box_transit_np(
            time_array_np, true_period, true_t0, duration_hours, depth
        )
        flux_np += rng.normal(0, 40e-6, len(flux_np))

        time_mx = mx.array(time_array_np)
        flux_mx = mx.array(flux_np)
        flux_err_mx = mx.array(flux_err_np)

        # 1. Score at true period
        score = score_fixed_period(
            time=time_mx,
            flux=flux_mx,
            flux_err=flux_err_mx,
            period_days=true_period,
            t0_btjd=true_t0,
            duration_hours=duration_hours,
        )
        mx.eval(score)
        assert float(score) > 10.0  # Should have strong detection

        # 2. Find best period from candidates
        periods_top_k = mx.array([3.0, 3.25, 3.5, 3.75, 4.0])
        top_k_result = score_top_k_periods(
            time=time_mx,
            flux=flux_mx,
            flux_err=flux_err_mx,
            periods_days_top_k=periods_top_k,
            t0_btjd=true_t0,
            duration_hours=duration_hours,
        )
        mx.eval(top_k_result.scores)

        scores_np = np.array(top_k_result.scores)
        best_period = float(np.array(periods_top_k)[np.argmax(scores_np)])
        assert abs(best_period - true_period) < 0.25

        # 3. Refine T0
        refined = score_fixed_period_refine_t0(
            time=time_mx,
            flux=flux_mx,
            flux_err=flux_err_mx,
            period_days=true_period,
            t0_btjd=true_t0 + 0.02,  # Start offset
            duration_hours=duration_hours,
        )
        assert abs(refined.t0_best_btjd - true_t0) < 0.01

    def test_determinism(
        self,
        time_array_np: np.ndarray,
        flux_err_np: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that results are deterministic."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        depth = 0.001

        flux_np = make_box_transit_np(time_array_np, period, t0, duration_hours, depth)
        flux_np += rng.normal(0, 50e-6, len(flux_np))

        time_mx = mx.array(time_array_np)
        flux_mx = mx.array(flux_np)
        flux_err_mx = mx.array(flux_err_np)

        # Run twice
        score1 = score_fixed_period(
            time=time_mx,
            flux=flux_mx,
            flux_err=flux_err_mx,
            period_days=period,
            t0_btjd=t0,
            duration_hours=duration_hours,
        )
        score2 = score_fixed_period(
            time=time_mx,
            flux=flux_mx,
            flux_err=flux_err_mx,
            period_days=period,
            t0_btjd=t0,
            duration_hours=duration_hours,
        )
        mx.eval(score1, score2)

        assert float(score1) == float(score2)
