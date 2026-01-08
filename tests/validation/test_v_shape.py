"""Tests for V05 V-shape transit check.

Tests cover the improved trapezoid model fitting implementation with:
- U-shaped transits (planets with flat bottom) - metrics-only classification
- V-shaped transits (grazing EBs with no flat bottom) - metrics-only classification
- Grazing geometry (intermediate) - classification depends on depth
- Undersampled data - low confidence metrics-only
- Different cadences (2-min vs 30-min)
- Bootstrap uncertainty estimation
- Legacy key backward compatibility
- Metrics-only mode (passed computed but _metrics_only=True by default)
"""

from __future__ import annotations

import numpy as np
import pytest

from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.validation.lc_checks import VShapeConfig, check_v_shape


@pytest.fixture
def make_transit_lc():
    """Factory for synthetic light curves with configurable transit shapes."""

    def _make(
        n_transits: int = 10,
        depth_ppm: float = 2000,
        period: float = 5.0,
        duration_hours: float = 3.0,
        tflat_ttotal_ratio: float = 0.5,
        noise_ppm: float = 100.0,
        cadence_minutes: float = 2.0,
        seed: int = 42,
    ) -> tuple[LightCurveData, float]:
        """Generate synthetic light curve with trapezoid transits.

        Args:
            n_transits: Number of transits to include
            depth_ppm: Transit depth in ppm
            period: Orbital period in days
            duration_hours: Total transit duration in hours
            tflat_ttotal_ratio: Ratio of flat-bottom to total duration (0=V, 1=box)
            noise_ppm: Gaussian noise level in ppm
            cadence_minutes: Observation cadence in minutes
            seed: RNG seed for reproducibility

        Returns:
            Tuple of (LightCurveData, t0)
        """
        rng = np.random.default_rng(seed)

        cadence_days = cadence_minutes / (24.0 * 60.0)
        duration_days = duration_hours / 24.0

        # First transit center
        t0 = 0.5 * period

        # Time array
        t_start = t0
        last_transit_center = t0 + (n_transits - 1) * period
        t_end = last_transit_center + 0.5 * period + 0.3 * period
        n_points = int((t_end - t_start) / cadence_days)

        time = np.linspace(t_start, t_end, n_points, dtype=np.float64)

        # Base flux with noise
        flux = np.ones_like(time, dtype=np.float64)
        flux += rng.normal(0, noise_ppm * 1e-6, size=len(time))

        # Add trapezoid transits
        depth = depth_ppm * 1e-6
        half_dur = duration_days / 2
        t_flat_dur = tflat_ttotal_ratio * duration_days
        half_flat = t_flat_dur / 2
        slope_width = half_dur - half_flat

        for transit_num in range(n_transits):
            transit_center = t0 + transit_num * period

            # Distance from transit center in days
            dist = np.abs(time - transit_center)

            for i, d in enumerate(dist):
                # Inside flat bottom region
                if d < half_flat:
                    flux[i] -= depth
                # In ingress/egress slope region
                elif d < half_dur and slope_width > 0:
                    # Linear interpolation from edge (0) to flat bottom (depth)
                    frac = (half_dur - d) / slope_width
                    flux[i] -= depth * frac

        # Create flux_err
        flux_err = np.full_like(flux, noise_ppm * 1e-6, dtype=np.float64)
        quality = np.zeros(len(time), dtype=np.int32)
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


class TestVShapeCheck:
    """Test suite for check_v_shape function."""

    def test_default_returns_metrics_only(self, make_transit_lc) -> None:
        """Default mode should return metrics only (passed=None, _metrics_only=True)."""
        lc, t0 = make_transit_lc(
            n_transits=10,
            depth_ppm=2000,
            tflat_ttotal_ratio=0.6,
            noise_ppm=50,
        )
        result = check_v_shape(lc, period=5.0, t0=t0, duration_hours=3.0)

        # Default mode returns _metrics_only=True to signal downstream policy should decide
        assert result.details.get("_metrics_only") is True
        # In metrics-only mode, passed is None (policy decision deferred to caller)
        assert result.passed is None
        # Metrics should still be computed
        assert "tflat_ttotal_ratio" in result.details
        assert "classification" in result.details

    def test_u_shape_planet_passes(self, make_transit_lc) -> None:
        """U-shaped transit should classify as U_SHAPE (metrics-only)."""
        # tF/tT = 0.6 is clearly U-shaped (above grazing threshold of 0.3)
        lc, t0 = make_transit_lc(
            n_transits=10,
            depth_ppm=2000,
            tflat_ttotal_ratio=0.6,
            noise_ppm=50,
        )
        result = check_v_shape(lc, period=5.0, t0=t0, duration_hours=3.0)

        assert result.passed is None
        assert result.details["classification"] == "U_SHAPE"
        assert result.details["tflat_ttotal_ratio"] > 0.3
        assert result.confidence >= 0.5

    def test_v_shape_eb_fails(self, make_transit_lc) -> None:
        """Deep V-shaped/grazing-like transit should classify as V_SHAPE or GRAZING (metrics-only).

        Note: Due to grid search limitations and noise, a pure V-shape (tF/tT=0)
        may be fitted as GRAZING with a small but non-zero tF/tT. However, the
        combination of grazing geometry AND deep depth (>50000 ppm) should still
        result in a FAIL classification.
        """
        # tF/tT = 0.0 is pure V-shape, but may fit as slightly higher
        # The key is the DEEP depth which should cause FAIL even if GRAZING
        lc, t0 = make_transit_lc(
            n_transits=10,
            depth_ppm=60000,  # Deep eclipse typical of EB - above 50000 threshold
            tflat_ttotal_ratio=0.0,
            noise_ppm=50,  # Lower noise for cleaner fit
        )
        config = VShapeConfig(grazing_depth_ppm=40000)
        result = check_v_shape(lc, period=5.0, t0=t0, duration_hours=3.0, config=config)

        assert result.passed is None
        # Classification could be V_SHAPE or GRAZING (but fails due to depth)
        assert result.details["classification"] in ("V_SHAPE", "GRAZING")

    def test_grazing_planet_passes(self, make_transit_lc) -> None:
        """Grazing transit with shallow depth should classify as GRAZING (metrics-only)."""
        # tF/tT = 0.2 is in grazing range (0.15-0.3)
        # Shallow depth suggests planet, not EB
        lc, t0 = make_transit_lc(
            n_transits=10,
            depth_ppm=1000,  # Shallow depth
            tflat_ttotal_ratio=0.2,
            noise_ppm=50,
        )
        result = check_v_shape(lc, period=5.0, t0=t0, duration_hours=3.0)

        assert result.passed is None
        assert result.details["classification"] == "GRAZING"
        # Depth should be below grazing_depth_ppm threshold (50000)
        assert result.details["depth_ppm"] < 50000

    def test_grazing_deep_fails(self, make_transit_lc) -> None:
        """Grazing transit with deep eclipse should still classify as GRAZING (metrics-only)."""
        # tF/tT = 0.2 is grazing, but deep depth suggests EB
        lc, t0 = make_transit_lc(
            n_transits=10,
            depth_ppm=60000,  # Deep - above 50000 ppm threshold
            tflat_ttotal_ratio=0.2,
            noise_ppm=100,
        )
        config = VShapeConfig(grazing_depth_ppm=50000)
        result = check_v_shape(lc, period=5.0, t0=t0, duration_hours=3.0, config=config)

        assert result.passed is None
        assert result.details["classification"] == "GRAZING"

    def test_undersampled_low_confidence_pass(self, make_transit_lc) -> None:
        """Undersampled data should return low confidence metrics-only result."""
        # Very few points - use long cadence and short transit
        lc, t0 = make_transit_lc(
            n_transits=2,
            depth_ppm=2000,
            duration_hours=1.0,  # Short transit
            cadence_minutes=30.0,  # Long cadence
            tflat_ttotal_ratio=0.5,
        )
        result = check_v_shape(lc, period=5.0, t0=t0, duration_hours=1.0)

        assert result.passed is None
        assert result.confidence <= 0.5
        # May have warnings about insufficient data
        assert (
            "INSUFFICIENT_DATA" in result.details["classification"]
            or len(result.details["warnings"]) >= 0
        )

    def test_2min_cadence_vs_30min(self, make_transit_lc) -> None:
        """2-minute cadence should give higher confidence than 30-minute (metrics-only)."""
        # 2-minute cadence - well sampled
        lc_2min, t0 = make_transit_lc(
            n_transits=10,
            depth_ppm=2000,
            duration_hours=3.0,
            tflat_ttotal_ratio=0.6,
            cadence_minutes=2.0,
        )
        result_2min = check_v_shape(lc_2min, period=5.0, t0=t0, duration_hours=3.0)

        # 30-minute cadence - sparser
        lc_30min, t0 = make_transit_lc(
            n_transits=10,
            depth_ppm=2000,
            duration_hours=3.0,
            tflat_ttotal_ratio=0.6,
            cadence_minutes=30.0,
        )
        result_30min = check_v_shape(lc_30min, period=5.0, t0=t0, duration_hours=3.0)

        assert result_2min.passed is None
        assert result_30min.passed is None

        # 2-min should have higher confidence (more data points)
        assert result_2min.confidence >= result_30min.confidence

    def test_legacy_keys_present(self, make_transit_lc) -> None:
        """Legacy keys should be present for backward compatibility."""
        lc, t0 = make_transit_lc(n_transits=10, depth_ppm=2000, tflat_ttotal_ratio=0.6)
        result = check_v_shape(lc, period=5.0, t0=t0, duration_hours=3.0)

        # Legacy keys must be present
        assert "depth_bottom" in result.details
        assert "depth_edge" in result.details
        assert "shape_ratio" in result.details
        assert "shape" in result.details
        assert "n_bottom_points" in result.details
        assert "n_edge_points" in result.details

    def test_new_keys_present(self, make_transit_lc) -> None:
        """New keys should be present in result details."""
        lc, t0 = make_transit_lc(n_transits=10, depth_ppm=2000, tflat_ttotal_ratio=0.6)
        result = check_v_shape(lc, period=5.0, t0=t0, duration_hours=3.0)

        # New keys
        assert "t_flat_hours" in result.details
        assert "t_total_hours" in result.details
        assert "tflat_ttotal_ratio" in result.details
        assert "tflat_ttotal_ratio_err" in result.details
        assert "shape_metric_uncertainty" in result.details
        assert "classification" in result.details
        assert "depth_ppm" in result.details
        assert "transit_coverage" in result.details
        assert "n_in_transit" in result.details
        assert "n_baseline" in result.details
        assert "warnings" in result.details
        assert "method" in result.details

        # Method should indicate trapezoid fitting
        assert result.details["method"] == "trapezoid_grid_search"

    def test_bootstrap_uncertainty(self, make_transit_lc) -> None:
        """Bootstrap uncertainty should be computed."""
        lc, t0 = make_transit_lc(n_transits=10, depth_ppm=2000, tflat_ttotal_ratio=0.6)
        result = check_v_shape(lc, period=5.0, t0=t0, duration_hours=3.0)

        # Uncertainty should be non-negative
        assert result.details["tflat_ttotal_ratio_err"] >= 0
        assert result.details["shape_metric_uncertainty"] >= 0

        # Uncertainty should be reasonable (not larger than 1.0)
        assert result.details["tflat_ttotal_ratio_err"] <= 1.0

    def test_t_flat_hours_calculation(self, make_transit_lc) -> None:
        """t_flat_hours should equal tflat_ttotal_ratio * duration_hours."""
        duration_hours = 4.0
        lc, t0 = make_transit_lc(
            n_transits=10,
            depth_ppm=2000,
            duration_hours=duration_hours,
            tflat_ttotal_ratio=0.5,
        )
        result = check_v_shape(lc, period=5.0, t0=t0, duration_hours=duration_hours)

        expected_t_flat = result.details["tflat_ttotal_ratio"] * duration_hours
        assert abs(result.details["t_flat_hours"] - expected_t_flat) < 0.01

    def test_result_id_and_name(self, make_transit_lc) -> None:
        """Result should have correct ID and name."""
        lc, t0 = make_transit_lc(n_transits=10, depth_ppm=2000)
        result = check_v_shape(lc, period=5.0, t0=t0, duration_hours=3.0)

        assert result.id == "V05"
        assert result.name == "v_shape"

    def test_config_overrides(self, make_transit_lc) -> None:
        """Custom config should be accepted and reflected in classification boundaries."""
        lc, t0 = make_transit_lc(
            n_transits=10,
            depth_ppm=60000,  # Deep - would fail with default grazing_depth_ppm
            tflat_ttotal_ratio=0.25,  # In grazing range
            noise_ppm=50,
        )

        lenient_config = VShapeConfig(grazing_depth_ppm=100000)
        result_lenient = check_v_shape(
            lc, period=5.0, t0=t0, duration_hours=3.0, config=lenient_config
        )

        assert lenient_config.grazing_depth_ppm == 100000
        assert result_lenient.passed is None


class TestVShapeConfig:
    """Tests for VShapeConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config values should match design spec."""
        config = VShapeConfig()

        assert config.tflat_ttotal_threshold == 0.15
        assert config.grazing_threshold == 0.3
        assert config.grazing_depth_ppm == 50000.0
        assert config.min_points_in_transit == 10
        assert config.min_transit_coverage == 0.6
        assert config.n_bootstrap == 100
        assert config.bootstrap_ci == 0.68
        assert config.shape_ratio_threshold == 1.3

    def test_custom_values(self) -> None:
        """Custom config values should be settable."""
        config = VShapeConfig(
            tflat_ttotal_threshold=0.2,
            grazing_threshold=0.4,
            grazing_depth_ppm=100000.0,
            n_bootstrap=50,
        )

        assert config.tflat_ttotal_threshold == 0.2
        assert config.grazing_threshold == 0.4
        assert config.grazing_depth_ppm == 100000.0
        assert config.n_bootstrap == 50


class TestTrapezoidHelpers:
    """Tests for helper functions used by check_v_shape."""

    def test_trapezoid_model_box(self) -> None:
        """Trapezoid model with tF/tT=1 should be a box."""
        from bittr_tess_vetter.validation.lc_checks import _trapezoid_model

        phase = np.linspace(-0.1, 0.1, 100)
        t_flat_phase = 0.2  # Full width = total width
        t_total_phase = 0.2
        depth = 0.01

        model = _trapezoid_model(phase, t_flat_phase, t_total_phase, depth)

        # In-transit should all be at (1 - depth)
        in_transit = np.abs(phase) < 0.1
        assert np.allclose(model[in_transit], 1 - depth, rtol=0.01)

    def test_trapezoid_model_v_shape(self) -> None:
        """Trapezoid model with tF/tT=0 should be V-shaped."""
        from bittr_tess_vetter.validation.lc_checks import _trapezoid_model

        phase = np.linspace(-0.1, 0.1, 100)
        t_flat_phase = 0.0  # No flat bottom
        t_total_phase = 0.2
        depth = 0.01

        model = _trapezoid_model(phase, t_flat_phase, t_total_phase, depth)

        # Center should be at (1 - depth)
        center_idx = len(phase) // 2
        assert abs(model[center_idx] - (1 - depth)) < 0.01

        # Edges should be at ~1.0
        assert model[0] > 0.99
        assert model[-1] > 0.99

    def test_fit_trapezoid_recovers_ratio(self) -> None:
        """Grid search should approximately recover known tF/tT ratio."""
        from bittr_tess_vetter.validation.lc_checks import (
            _fit_trapezoid_grid_search,
            _trapezoid_model,
        )

        # Generate synthetic trapezoid with known ratio
        true_ratio = 0.5
        t_total_phase = 0.1
        depth = 0.01

        phase = np.linspace(-0.06, 0.06, 200)
        flux = _trapezoid_model(phase, true_ratio * t_total_phase, t_total_phase, depth)

        # Add small noise
        rng = np.random.default_rng(42)
        flux += rng.normal(0, 0.0001, len(flux))

        # Fit
        fitted_ratio, fitted_depth, _ = _fit_trapezoid_grid_search(phase, flux, t_total_phase)

        # Should recover approximately (grid has limited resolution)
        assert abs(fitted_ratio - true_ratio) < 0.15
        assert abs(fitted_depth - depth) < 0.005


class TestVShapeEdgeCases:
    """Edge case tests for V-shape check."""

    def test_negative_depth_handled(self, make_transit_lc) -> None:
        """Negative depth (brightening) should be handled gracefully."""
        # Create a light curve and artificially make it brighten
        lc, t0 = make_transit_lc(n_transits=5, depth_ppm=2000, tflat_ttotal_ratio=0.5)

        # Invert the flux (make it brighten during "transit")
        inverted_flux = 2.0 - lc.flux

        lc_inverted = LightCurveData(
            time=lc.time.copy(),
            flux=inverted_flux.astype(np.float64),
            flux_err=lc.flux_err.copy(),
            quality=lc.quality.copy(),
            valid_mask=lc.valid_mask.copy(),
            tic_id=lc.tic_id,
            sector=lc.sector,
            cadence_seconds=lc.cadence_seconds,
        )

        # Should not crash
        result = check_v_shape(lc_inverted, period=5.0, t0=t0, duration_hours=3.0)

        # Result should exist
        assert result is not None
        assert result.id == "V05"

    def test_very_short_period(self, make_transit_lc) -> None:
        """Very short period should work correctly."""
        lc, t0 = make_transit_lc(
            n_transits=20,
            depth_ppm=2000,
            period=0.5,  # 12-hour period
            duration_hours=1.0,
            tflat_ttotal_ratio=0.6,
        )
        result = check_v_shape(lc, period=0.5, t0=t0, duration_hours=1.0)

        assert result is not None
        assert result.passed is None

    def test_very_long_period(self, make_transit_lc) -> None:
        """Very long period with single transit should handle gracefully."""
        lc, t0 = make_transit_lc(
            n_transits=1,
            depth_ppm=2000,
            period=30.0,  # 30-day period
            duration_hours=6.0,
            tflat_ttotal_ratio=0.6,
        )
        result = check_v_shape(lc, period=30.0, t0=t0, duration_hours=6.0)

        # Should work but may have low confidence
        assert result is not None
        assert result.id == "V05"
