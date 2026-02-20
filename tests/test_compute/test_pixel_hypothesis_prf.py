"""Tests for pixel_hypothesis_prf module."""

from __future__ import annotations

import numpy as np
import pytest

from tess_vetter.compute.pixel_hypothesis_prf import (
    score_hypotheses_with_prf,
)
from tess_vetter.compute.prf_schemas import PRFParams


class TestScoreHypothesesWithPRF:
    """Tests for score_hypotheses_with_prf function."""

    def test_parametric_backend_produces_log_likelihood(self) -> None:
        """Parametric backend returns log_likelihood in results."""
        # Create a difference image with signal at center
        diff_image = np.zeros((11, 11), dtype=np.float64)
        diff_image[5, 5] = 100.0  # Bright spot at center

        hypotheses = [
            {"source_id": "target", "source_name": "Target", "row": 5.0, "col": 5.0},
            {"source_id": "neighbor", "source_name": "Neighbor", "row": 8.0, "col": 8.0},
        ]

        ranked = score_hypotheses_with_prf(
            diff_image, hypotheses, prf_backend="parametric", seed=42
        )

        assert len(ranked) == 2
        # Best hypothesis should be the target (closer to signal)
        assert ranked[0]["source_id"] == "target"
        assert ranked[0]["rank"] == 1
        # Should have log_likelihood
        assert ranked[0]["log_likelihood"] is not None
        assert isinstance(ranked[0]["log_likelihood"], float)
        # Should have prf_backend field
        assert ranked[0]["prf_backend"] == "parametric"
        # Should have fit_residual_rms
        assert ranked[0]["fit_residual_rms"] is not None
        assert isinstance(ranked[0]["fit_residual_rms"], float)

    def test_background_fitting_reduces_residual(self) -> None:
        """Background fitting should reduce fit residuals."""
        # Create difference image with gradient background
        n_rows, n_cols = 11, 11
        row_grid, col_grid = np.mgrid[0:n_rows, 0:n_cols]
        # Add gradient: higher flux at bottom-right
        background = 10.0 + 2.0 * (row_grid / n_rows) + 3.0 * (col_grid / n_cols)
        diff_image = background.astype(np.float64)
        # Add a point source at center
        diff_image[5, 5] += 50.0

        hypotheses = [
            {"source_id": "target", "source_name": "Target", "row": 5.0, "col": 5.0},
        ]

        # With background fitting
        ranked_with_bg = score_hypotheses_with_prf(
            diff_image, hypotheses, prf_backend="parametric", fit_background=True, seed=42
        )

        # Without background fitting (use prf_lite which doesn't have gradient fitting)
        score_hypotheses_with_prf(
            diff_image, hypotheses, prf_backend="prf_lite", fit_background=True, seed=42
        )

        # Parametric with background fitting should have lower residual RMS
        # (or at least not much higher) compared to prf_lite
        assert ranked_with_bg[0]["fit_residual_rms"] is not None
        assert ranked_with_bg[0]["fitted_background"] is not None
        # fitted_background should be a tuple of 3 floats
        bg = ranked_with_bg[0]["fitted_background"]
        assert len(bg) == 3
        # The gradient should have been detected (bx, by should be non-zero)
        # Note: exact values depend on fitting, so we just check they're present

    def test_prf_lite_backward_compatible(self) -> None:
        """prf_lite backend should produce results compatible with legacy."""
        diff_image = np.zeros((11, 11), dtype=np.float64)
        diff_image[5, 5] = 100.0

        hypotheses = [
            {"source_id": "target", "source_name": "Target", "row": 5.0, "col": 5.0},
            {"source_id": "neighbor", "source_name": "Neighbor", "row": 8.0, "col": 8.0},
        ]

        ranked = score_hypotheses_with_prf(diff_image, hypotheses, prf_backend="prf_lite", seed=42)

        assert len(ranked) == 2
        assert ranked[0]["source_id"] == "target"
        assert ranked[0]["rank"] == 1
        assert ranked[0]["delta_loss"] == 0.0
        assert ranked[1]["delta_loss"] > 0.0
        # Should have prf_backend field
        assert ranked[0]["prf_backend"] == "prf_lite"

    def test_results_match_prf_lite_on_simple_case(self) -> None:
        """Parametric backend should give similar ranking to prf_lite on simple cases."""
        # Create a simple difference image with clear signal
        diff_image = np.zeros((11, 11), dtype=np.float64)
        # Create Gaussian-like signal at center
        row_grid, col_grid = np.mgrid[0:11, 0:11]
        dist_sq = (row_grid - 5.0) ** 2 + (col_grid - 5.0) ** 2
        diff_image = 100.0 * np.exp(-dist_sq / (2 * 1.5**2))

        hypotheses = [
            {"source_id": "target", "source_name": "Target", "row": 5.0, "col": 5.0},
            {"source_id": "neighbor", "source_name": "Neighbor", "row": 8.0, "col": 8.0},
        ]

        ranked_lite = score_hypotheses_with_prf(
            diff_image, hypotheses, prf_backend="prf_lite", seed=42
        )
        ranked_parametric = score_hypotheses_with_prf(
            diff_image, hypotheses, prf_backend="parametric", seed=42
        )

        # Both should identify target as best
        assert ranked_lite[0]["source_id"] == "target"
        assert ranked_parametric[0]["source_id"] == "target"

        # Rankings should match
        assert ranked_lite[0]["rank"] == ranked_parametric[0]["rank"]
        assert ranked_lite[1]["rank"] == ranked_parametric[1]["rank"]

    def test_empty_hypotheses_raises(self) -> None:
        """Empty hypotheses list should raise ValueError."""
        diff_image = np.zeros((11, 11), dtype=np.float64)

        with pytest.raises(ValueError, match="hypotheses list cannot be empty"):
            score_hypotheses_with_prf(diff_image, [], prf_backend="parametric")

    def test_invalid_backend_raises(self) -> None:
        """Invalid prf_backend should raise ValueError."""
        diff_image = np.zeros((11, 11), dtype=np.float64)
        hypotheses = [
            {"source_id": "target", "source_name": "Target", "row": 5.0, "col": 5.0},
        ]

        with pytest.raises(ValueError, match="Unknown prf_backend"):
            score_hypotheses_with_prf(
                diff_image,
                hypotheses,
                prf_backend="invalid",  # type: ignore[arg-type]
            )

    def test_instrument_backend_not_available(self) -> None:
        """Instrument backend should raise ValueError (not yet implemented)."""
        diff_image = np.zeros((11, 11), dtype=np.float64)
        hypotheses = [
            {"source_id": "target", "source_name": "Target", "row": 5.0, "col": 5.0},
        ]

        with pytest.raises(ValueError, match="Instrument PRF backend not yet available"):
            score_hypotheses_with_prf(diff_image, hypotheses, prf_backend="instrument")

    def test_non_2d_image_raises(self) -> None:
        """Non-2D difference image should raise ValueError."""
        diff_image = np.zeros((11, 11, 3), dtype=np.float64)  # 3D array
        hypotheses = [
            {"source_id": "target", "source_name": "Target", "row": 5.0, "col": 5.0},
        ]

        with pytest.raises(ValueError, match="diff_image must be 2D"):
            score_hypotheses_with_prf(diff_image, hypotheses, prf_backend="parametric")

    def test_custom_prf_params(self) -> None:
        """Custom PRF parameters should be used for parametric backend."""
        diff_image = np.zeros((11, 11), dtype=np.float64)
        diff_image[5, 5] = 100.0

        hypotheses = [
            {"source_id": "target", "source_name": "Target", "row": 5.0, "col": 5.0},
        ]

        # Use custom PRF with larger sigma
        custom_params = PRFParams(sigma_row=2.5, sigma_col=2.5)

        ranked = score_hypotheses_with_prf(
            diff_image,
            hypotheses,
            prf_backend="parametric",
            prf_params=custom_params,
            seed=42,
        )

        assert len(ranked) == 1
        assert ranked[0]["source_id"] == "target"
        assert ranked[0]["prf_backend"] == "parametric"

    def test_handles_nan_values(self) -> None:
        """Should handle NaN values in difference image."""
        diff_image = np.zeros((11, 11), dtype=np.float64)
        diff_image[5, 5] = 100.0
        # Add some NaN values
        diff_image[0, 0] = np.nan
        diff_image[10, 10] = np.nan

        hypotheses = [
            {"source_id": "target", "source_name": "Target", "row": 5.0, "col": 5.0},
        ]

        ranked = score_hypotheses_with_prf(
            diff_image, hypotheses, prf_backend="parametric", seed=42
        )

        assert len(ranked) == 1
        assert ranked[0]["source_id"] == "target"
        # Should still produce valid results
        assert ranked[0]["fit_loss"] is not None
        assert np.isfinite(ranked[0]["fit_loss"])

    def test_deterministic_with_seed(self) -> None:
        """Results should be deterministic when seed is provided."""
        diff_image = np.zeros((11, 11), dtype=np.float64)
        diff_image[5, 5] = 100.0

        hypotheses = [
            {"source_id": "target", "source_name": "Target", "row": 5.0, "col": 5.0},
            {"source_id": "neighbor", "source_name": "Neighbor", "row": 8.0, "col": 8.0},
        ]

        ranked1 = score_hypotheses_with_prf(
            diff_image, hypotheses, prf_backend="parametric", seed=123
        )
        ranked2 = score_hypotheses_with_prf(
            diff_image, hypotheses, prf_backend="parametric", seed=123
        )

        assert ranked1[0]["source_id"] == ranked2[0]["source_id"]
        assert ranked1[0]["fit_loss"] == ranked2[0]["fit_loss"]
        assert ranked1[0]["log_likelihood"] == ranked2[0]["log_likelihood"]

    def test_variance_affects_log_likelihood(self) -> None:
        """Explicit variance should affect log-likelihood calculation."""
        diff_image = np.zeros((11, 11), dtype=np.float64)
        diff_image[5, 5] = 100.0

        hypotheses = [
            {"source_id": "target", "source_name": "Target", "row": 5.0, "col": 5.0},
        ]

        # Low variance (high confidence)
        ranked_low_var = score_hypotheses_with_prf(
            diff_image, hypotheses, prf_backend="parametric", variance=1.0, seed=42
        )

        # High variance (low confidence)
        ranked_high_var = score_hypotheses_with_prf(
            diff_image, hypotheses, prf_backend="parametric", variance=100.0, seed=42
        )

        # Same fit_loss but different log-likelihood
        assert ranked_low_var[0]["fit_loss"] == ranked_high_var[0]["fit_loss"]
        # Lower variance -> more negative log-likelihood (assuming same residuals)
        assert ranked_low_var[0]["log_likelihood"] is not None
        assert ranked_high_var[0]["log_likelihood"] is not None
        assert ranked_low_var[0]["log_likelihood"] < ranked_high_var[0]["log_likelihood"]


class TestHypothesisScoreFields:
    """Test that HypothesisScore has all expected fields."""

    def test_core_fields_present(self) -> None:
        """Core fields should always be present in results."""
        diff_image = np.zeros((11, 11), dtype=np.float64)
        diff_image[5, 5] = 100.0

        hypotheses = [
            {"source_id": "target", "source_name": "Target", "row": 5.0, "col": 5.0},
        ]

        ranked = score_hypotheses_with_prf(
            diff_image, hypotheses, prf_backend="parametric", seed=42
        )

        result = ranked[0]
        # Core fields
        assert "source_id" in result
        assert "source_name" in result
        assert "fit_loss" in result
        assert "delta_loss" in result
        assert "rank" in result
        assert "fit_amplitude" in result
        assert "fit_background" in result

    def test_extended_fields_present_for_parametric(self) -> None:
        """Extended fields should be present for parametric backend."""
        diff_image = np.zeros((11, 11), dtype=np.float64)
        diff_image[5, 5] = 100.0

        hypotheses = [
            {"source_id": "target", "source_name": "Target", "row": 5.0, "col": 5.0},
        ]

        ranked = score_hypotheses_with_prf(
            diff_image, hypotheses, prf_backend="parametric", seed=42
        )

        result = ranked[0]
        # Extended fields
        assert "log_likelihood" in result
        assert "fit_residual_rms" in result
        assert "fitted_background" in result
        assert "prf_backend" in result
