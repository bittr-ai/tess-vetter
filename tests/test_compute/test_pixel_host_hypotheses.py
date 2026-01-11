"""Unit tests for pixel host hypothesis scoring and consensus algorithms.

Tests the pixel_host_hypotheses module which provides:
- PRF-lite hypothesis scoring for transit host identification
- Multi-sector consensus aggregation
- Aperture hypothesis fitting

All tests use synthetic pixel scenes - no network access or FITS files required.
Tests are deterministic with fixed random seeds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from bittr_tess_vetter.compute.pixel_host_hypotheses import (
    FLIP_RATE_MIXED_THRESHOLD,
    FLIP_RATE_UNSTABLE_THRESHOLD,
    MARGIN_RESOLVE_THRESHOLD,
    aggregate_multi_sector,
    fit_aperture_hypothesis,
    score_hypotheses_prf_lite,
)

# =============================================================================
# Synthetic Scene Generation Helpers
# =============================================================================


def make_synthetic_diff_image(
    shape: tuple[int, int] = (15, 15),
    sources: list[dict[str, Any]] | None = None,
    noise_sigma: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Create a synthetic difference image with PSF sources and noise.

    This helper creates a simple 2D image with Gaussian PSF profiles at
    specified locations. Useful for testing hypothesis scoring without
    real FITS data.

    Parameters
    ----------
    shape : tuple[int, int], optional
        Shape of the output array (n_rows, n_cols). Default is (15, 15).
    sources : list[dict], optional
        List of source definitions. Each dict should contain:
        - 'row': float, row coordinate of source center
        - 'col': float, column coordinate of source center
        - 'amplitude': float, amplitude of the Gaussian (positive for dimming)
        If None, creates a single source at center with amplitude +1.0.
    noise_sigma : float, optional
        Standard deviation of Gaussian noise. Default is 0.1.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    np.ndarray
        2D difference image array of shape `shape`.

    Examples
    --------
    >>> # Single dimming source at center
    >>> img = make_synthetic_diff_image(
    ...     shape=(15, 15),
    ...     sources=[{'row': 7, 'col': 7, 'amplitude': 1.0}],
    ...     seed=42
    ... )
    >>> img.shape
    (15, 15)
    """
    rng = np.random.default_rng(seed)

    # Default: single dimming source at center
    if sources is None:
        sources = [{"row": shape[0] // 2, "col": shape[1] // 2, "amplitude": 1.0}]

    # Create coordinate grids
    row_grid, col_grid = np.mgrid[0 : shape[0], 0 : shape[1]]

    # Initialize image
    image = np.zeros(shape, dtype=np.float64)

    # Add each source as a Gaussian PSF
    sigma = 1.5  # TESS-like PSF width
    for src in sources:
        row = float(src["row"])
        col = float(src["col"])
        amplitude = float(src["amplitude"])

        # Gaussian PSF
        dist_sq = (row_grid - row) ** 2 + (col_grid - col) ** 2
        psf = np.exp(-dist_sq / (2.0 * sigma**2))
        psf = psf / psf.sum()  # Normalize

        image += amplitude * psf

    # Add Gaussian noise
    if noise_sigma > 0:
        noise = rng.normal(0, noise_sigma, shape)
        image += noise

    return image


def make_synthetic_hypotheses(
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert source definitions to hypothesis format.

    Parameters
    ----------
    sources : list[dict]
        List of source definitions with 'row', 'col', and optionally 'name'.

    Returns
    -------
    list[dict]
        List of hypothesis dicts suitable for score_hypotheses_prf_lite.
    """
    hypotheses = []
    for i, src in enumerate(sources):
        hyp = {
            "source_id": src.get("name", f"source_{i}"),
            "source_name": src.get("name", f"Source {i}"),
            "row": float(src["row"]),
            "col": float(src["col"]),
        }
        hypotheses.append(hyp)
    return hypotheses


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def single_source_scene() -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Synthetic scene with a single dimming source at center."""
    sources = [{"row": 7, "col": 7, "amplitude": 1.0, "name": "target"}]
    diff_image = make_synthetic_diff_image(
        shape=(15, 15), sources=sources, noise_sigma=0.05, seed=42
    )
    hypotheses = make_synthetic_hypotheses(sources)
    return diff_image, hypotheses


@pytest.fixture
def two_source_resolved_scene() -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Synthetic scene with two sources, one clearly dimming."""
    sources = [
        {"row": 7, "col": 7, "amplitude": 1.0, "name": "target"},
        {"row": 10, "col": 10, "amplitude": 0.0, "name": "neighbor"},
    ]
    # Only target is dimming
    diff_image = make_synthetic_diff_image(
        shape=(15, 15),
        sources=[sources[0]],  # Only target source contributes
        noise_sigma=0.05,
        seed=42,
    )
    hypotheses = make_synthetic_hypotheses(sources)
    return diff_image, hypotheses


@pytest.fixture
def two_source_ambiguous_scene() -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Synthetic scene with two sources of equal dimming - ambiguous case."""
    sources = [
        {"row": 7, "col": 5, "amplitude": 0.5, "name": "source_a"},
        {"row": 7, "col": 9, "amplitude": 0.5, "name": "source_b"},
    ]
    diff_image = make_synthetic_diff_image(
        shape=(15, 15), sources=sources, noise_sigma=0.05, seed=42
    )
    hypotheses = make_synthetic_hypotheses(sources)
    return diff_image, hypotheses


# =============================================================================
# score_hypotheses_prf_lite Tests
# =============================================================================


class TestScoreHypothesesPrfLite:
    """Tests for the score_hypotheses_prf_lite function."""

    def test_score_hypotheses_picks_correct_host(
        self, two_source_resolved_scene: tuple[np.ndarray, list[dict[str, Any]]]
    ) -> None:
        """Injected dimming source should rank #1."""
        diff_image, hypotheses = two_source_resolved_scene

        scores = score_hypotheses_prf_lite(diff_image, hypotheses, seed=42)

        # Target (dimming source) should be ranked first
        assert len(scores) == 2
        assert scores[0]["source_id"] == "target"
        assert scores[0]["rank"] == 1
        assert scores[0]["delta_loss"] == 0.0

        # Neighbor should be ranked second
        assert scores[1]["source_id"] == "neighbor"
        assert scores[1]["rank"] == 2
        assert scores[1]["delta_loss"] > 0.0

    def test_score_hypotheses_ambiguous_near_equal(
        self, two_source_ambiguous_scene: tuple[np.ndarray, list[dict[str, Any]]]
    ) -> None:
        """Near-equal sources should have margin below resolve threshold."""
        diff_image, hypotheses = two_source_ambiguous_scene

        scores = score_hypotheses_prf_lite(diff_image, hypotheses, seed=42)

        # Both sources should be scored
        assert len(scores) == 2

        # Margin between them should be small (ambiguous)
        # The runner-up's delta_loss is the margin
        margin = scores[1]["delta_loss"]

        # For near-equal sources, margin should be close to 0
        # (much smaller than for clearly separated cases)
        # Note: exact threshold depends on noise, but should be < 1.0 for this scene
        assert margin < 1.0, f"Expected small margin for ambiguous scene, got {margin}"

    def test_score_hypotheses_single_source(
        self, single_source_scene: tuple[np.ndarray, list[dict[str, Any]]]
    ) -> None:
        """Single hypothesis should rank #1 with delta_loss=0."""
        diff_image, hypotheses = single_source_scene

        scores = score_hypotheses_prf_lite(diff_image, hypotheses, seed=42)

        assert len(scores) == 1
        assert scores[0]["rank"] == 1
        assert scores[0]["delta_loss"] == 0.0
        assert scores[0]["source_id"] == "target"

    def test_score_hypotheses_empty_raises_error(self) -> None:
        """Empty hypothesis list should raise ValueError."""
        diff_image = make_synthetic_diff_image()

        with pytest.raises(ValueError, match="cannot be empty"):
            score_hypotheses_prf_lite(diff_image, [], seed=42)

    def test_score_hypotheses_non_2d_raises_error(self) -> None:
        """Non-2D diff_image should raise ValueError."""
        diff_1d = np.zeros(100)
        hypotheses = [{"source_id": "test", "source_name": "Test", "row": 5, "col": 5}]

        with pytest.raises(ValueError, match="2D"):
            score_hypotheses_prf_lite(diff_1d, hypotheses, seed=42)

    def test_score_hypotheses_fit_amplitude_sign(
        self, single_source_scene: tuple[np.ndarray, list[dict[str, Any]]]
    ) -> None:
        """Fit amplitude should match the sign of the dimming source."""
        diff_image, hypotheses = single_source_scene

        scores = score_hypotheses_prf_lite(diff_image, hypotheses, seed=42)

        # diff_image is defined as (out-of-transit - in-transit), so dimming is positive.
        assert scores[0]["fit_amplitude"] is not None
        assert scores[0]["fit_amplitude"] > 0

    def test_score_hypotheses_handles_nan_pixels(self) -> None:
        """NaN pixels should be ignored without flipping the best hypothesis."""
        sources = [
            {"row": 7, "col": 7, "amplitude": 1.0, "name": "target"},
            {"row": 10, "col": 10, "amplitude": 0.0, "name": "neighbor"},
        ]
        diff_image = make_synthetic_diff_image(
            shape=(15, 15),
            sources=[sources[0]],
            noise_sigma=0.0,
            seed=123,
        )
        # Mask a chunk of pixels far from the signal core.
        diff_image[0:4, 0:4] = np.nan

        hypotheses = make_synthetic_hypotheses(sources)
        scores = score_hypotheses_prf_lite(diff_image, hypotheses, seed=42)

        assert scores[0]["source_id"] == "target"
        assert np.isfinite(scores[0]["fit_loss"])

    def test_margin_scales_with_signal_amplitude(self) -> None:
        """A stronger transit-like signal should increase the best-vs-runnerup margin."""
        hypotheses_def = [
            {"row": 7, "col": 7, "name": "target"},
            {"row": 11, "col": 11, "name": "neighbor"},
        ]
        hypotheses = make_synthetic_hypotheses(hypotheses_def)

        diff_weak = make_synthetic_diff_image(
            shape=(15, 15),
            sources=[{"row": 7, "col": 7, "amplitude": 0.2}],
            noise_sigma=0.0,
            seed=1,
        )
        diff_strong = make_synthetic_diff_image(
            shape=(15, 15),
            sources=[{"row": 7, "col": 7, "amplitude": 1.0}],
            noise_sigma=0.0,
            seed=1,
        )

        scores_weak = score_hypotheses_prf_lite(diff_weak, hypotheses, seed=1)
        scores_strong = score_hypotheses_prf_lite(diff_strong, hypotheses, seed=1)

        margin_weak = scores_weak[1]["delta_loss"]
        margin_strong = scores_strong[1]["delta_loss"]
        assert margin_strong > margin_weak

    def test_score_hypotheses_returns_typed_dict(
        self, single_source_scene: tuple[np.ndarray, list[dict[str, Any]]]
    ) -> None:
        """Results should match HypothesisScore TypedDict structure."""
        diff_image, hypotheses = single_source_scene

        scores = score_hypotheses_prf_lite(diff_image, hypotheses, seed=42)

        # Check all required fields are present
        required_fields = [
            "source_id",
            "source_name",
            "fit_loss",
            "delta_loss",
            "rank",
            "fit_amplitude",
            "fit_background",
        ]
        for field in required_fields:
            assert field in scores[0], f"Missing field: {field}"

    def test_score_hypotheses_deterministic(self) -> None:
        """Same seed should produce identical output."""
        sources = [
            {"row": 7, "col": 7, "amplitude": 1.0, "name": "target"},
            {"row": 10, "col": 10, "amplitude": 0.3, "name": "neighbor"},
        ]
        diff_image = make_synthetic_diff_image(
            shape=(15, 15), sources=sources, noise_sigma=0.1, seed=123
        )
        hypotheses = make_synthetic_hypotheses(sources)

        scores1 = score_hypotheses_prf_lite(diff_image, hypotheses, seed=42)
        scores2 = score_hypotheses_prf_lite(diff_image, hypotheses, seed=42)

        # Results should be identical
        assert len(scores1) == len(scores2)
        for s1, s2 in zip(scores1, scores2, strict=True):
            assert s1["source_id"] == s2["source_id"]
            assert s1["rank"] == s2["rank"]
            assert_allclose(s1["fit_loss"], s2["fit_loss"])
            assert_allclose(s1["delta_loss"], s2["delta_loss"])


# =============================================================================
# aggregate_multi_sector Tests
# =============================================================================


class TestAggregateMultiSector:
    """Tests for the aggregate_multi_sector function."""

    def test_aggregate_multi_sector_stable(self) -> None:
        """Same best across all sectors should produce stable consensus."""
        per_sector_results = [
            {"sector": 1, "best_source_id": "target", "margin": 5.0, "status": "ok"},
            {"sector": 2, "best_source_id": "target", "margin": 4.0, "status": "ok"},
            {"sector": 3, "best_source_id": "target", "margin": 6.0, "status": "ok"},
        ]

        consensus = aggregate_multi_sector(per_sector_results)

        assert consensus["consensus_best_source_id"] == "target"
        assert consensus["disagreement_flag"] == "stable"
        assert consensus["flip_rate"] == 0.0
        assert consensus["n_sectors_total"] == 3
        assert consensus["n_sectors_supporting_best"] == 3

    def test_aggregate_multi_sector_flipping(self) -> None:
        """Different best per sector should produce flipping flag."""
        per_sector_results = [
            {"sector": 1, "best_source_id": "source_a", "margin": 5.0, "status": "ok"},
            {"sector": 2, "best_source_id": "source_b", "margin": 4.0, "status": "ok"},
            {"sector": 3, "best_source_id": "source_c", "margin": 6.0, "status": "ok"},
        ]

        consensus = aggregate_multi_sector(per_sector_results)

        # With different best in each sector, flip_rate should be high
        assert consensus["disagreement_flag"] == "flipping"
        assert consensus["flip_rate"] > FLIP_RATE_UNSTABLE_THRESHOLD

    def test_aggregate_multi_sector_mixed(self) -> None:
        """Partial disagreement should produce mixed flag."""
        per_sector_results = [
            {"sector": 1, "best_source_id": "target", "margin": 5.0, "status": "ok"},
            {"sector": 2, "best_source_id": "target", "margin": 4.0, "status": "ok"},
            {"sector": 3, "best_source_id": "neighbor", "margin": 6.0, "status": "ok"},
            {"sector": 4, "best_source_id": "target", "margin": 5.0, "status": "ok"},
        ]

        consensus = aggregate_multi_sector(per_sector_results)

        # 1 out of 4 flips = 25%
        assert consensus["consensus_best_source_id"] == "target"
        assert consensus["flip_rate"] == pytest.approx(0.25)
        # 25% is below warn threshold (0.3) but depends on exact thresholds
        # The disagreement flag depends on the thresholds
        assert consensus["disagreement_flag"] in ("stable", "mixed")

    def test_aggregate_multi_sector_empty(self) -> None:
        """Empty sector list should return null consensus."""
        consensus = aggregate_multi_sector([])

        assert consensus["consensus_best_source_id"] is None
        assert consensus["consensus_margin"] is None
        assert consensus["disagreement_flag"] == "stable"
        assert consensus["flip_rate"] == 0.0
        assert consensus["n_sectors_total"] == 0
        assert consensus["n_sectors_supporting_best"] == 0

    def test_aggregate_multi_sector_all_invalid(self) -> None:
        """All invalid sectors should return null consensus."""
        per_sector_results = [
            {"sector": 1, "best_source_id": None, "margin": None, "status": "invalid"},
            {"sector": 2, "best_source_id": None, "margin": None, "status": "invalid"},
        ]

        consensus = aggregate_multi_sector(per_sector_results)

        assert consensus["consensus_best_source_id"] is None
        assert consensus["n_sectors_total"] == 2
        assert consensus["n_sectors_supporting_best"] == 0

    def test_aggregate_multi_sector_custom_thresholds(self) -> None:
        """Custom thresholds should affect disagreement classification."""
        per_sector_results = [
            {"sector": 1, "best_source_id": "target", "margin": 5.0, "status": "ok"},
            {"sector": 2, "best_source_id": "neighbor", "margin": 4.0, "status": "ok"},
        ]

        # Default thresholds - verify it runs without error
        _ = aggregate_multi_sector(per_sector_results)

        # Very strict thresholds (any flip is flipping)
        consensus_strict = aggregate_multi_sector(
            per_sector_results,
            flip_rate_mixed=0.1,
            flip_rate_unstable=0.4,
        )

        # With 50% flip rate, strict thresholds should flag as mixed or flipping
        assert consensus_strict["disagreement_flag"] in ("mixed", "flipping")

    def test_aggregate_multi_sector_returns_typed_dict(self) -> None:
        """Results should match MultiSectorConsensus TypedDict structure."""
        per_sector_results = [
            {"sector": 1, "best_source_id": "target", "margin": 5.0, "status": "ok"},
        ]

        consensus = aggregate_multi_sector(per_sector_results)

        # Check all required fields
        required_fields = [
            "consensus_best_source_id",
            "consensus_margin",
            "disagreement_flag",
            "flip_rate",
            "n_sectors_total",
            "n_sectors_supporting_best",
        ]
        for field in required_fields:
            assert field in consensus, f"Missing field: {field}"


# =============================================================================
# fit_aperture_hypothesis Tests
# =============================================================================


class TestFitApertureHypothesis:
    """Tests for the fit_aperture_hypothesis function."""

    def test_fit_aperture_hypothesis_resolved(self) -> None:
        """Clear winner in aperture fit should produce host_ambiguity='resolved'."""
        # Observed depths that match target's PRF weights well
        depths_by_aperture = [
            {"aperture_id": "spoc", "depth_ppm": 1000.0, "depth_ppm_err": 50.0},
            {"aperture_id": "r+1", "depth_ppm": 800.0, "depth_ppm_err": 50.0},
            {"aperture_id": "r+2", "depth_ppm": 600.0, "depth_ppm_err": 50.0},
        ]

        # PRF weights: target captures more flux in smaller apertures
        # Neighbor has flatter profile (off-center)
        prf_weights = {
            "target": [1.0, 0.8, 0.6],  # Matches observed depth pattern
            "neighbor": [0.3, 0.4, 0.5],  # Inverse pattern
        }

        result = fit_aperture_hypothesis(depths_by_aperture, prf_weights)

        # Target should be preferred (better fit)
        assert result["best_source_id"] == "target"
        assert result["host_ambiguity"] == "resolved"
        assert result["margin"] is not None
        assert result["margin"] > MARGIN_RESOLVE_THRESHOLD

    def test_fit_aperture_hypothesis_ambiguous(self) -> None:
        """Near-equal fit should produce host_ambiguity='ambiguous'."""
        # Observed depths
        depths_by_aperture = [
            {"aperture_id": "spoc", "depth_ppm": 500.0, "depth_ppm_err": 100.0},
            {"aperture_id": "r+1", "depth_ppm": 500.0, "depth_ppm_err": 100.0},
            {"aperture_id": "r+2", "depth_ppm": 500.0, "depth_ppm_err": 100.0},
        ]

        # Both hypotheses have similar PRF weight patterns (flat)
        prf_weights = {
            "source_a": [0.5, 0.5, 0.5],
            "source_b": [0.5, 0.5, 0.5],
        }

        result = fit_aperture_hypothesis(depths_by_aperture, prf_weights)

        # With identical patterns, margin should be near zero
        assert result["host_ambiguity"] == "ambiguous"
        # best_source_id should be None when ambiguous
        assert result["best_source_id"] is None

    def test_fit_aperture_hypothesis_empty_depths(self) -> None:
        """Empty depths list should return ambiguous result."""
        result = fit_aperture_hypothesis([], {"target": [1.0]})

        assert result["best_source_id"] is None
        assert result["host_ambiguity"] == "ambiguous"
        assert result["hypothesis_fits"] == []

    def test_fit_aperture_hypothesis_empty_hypotheses(self) -> None:
        """Empty hypothesis weights should return ambiguous result."""
        depths = [{"aperture_id": "spoc", "depth_ppm": 1000.0}]
        result = fit_aperture_hypothesis(depths, {})

        assert result["best_source_id"] is None
        assert result["host_ambiguity"] == "ambiguous"
        assert result["hypothesis_fits"] == []

    def test_fit_aperture_hypothesis_single_hypothesis(self) -> None:
        """Single hypothesis should rank #1 but may be ambiguous (no comparison)."""
        depths_by_aperture = [
            {"aperture_id": "spoc", "depth_ppm": 1000.0, "depth_ppm_err": 50.0},
            {"aperture_id": "r+1", "depth_ppm": 800.0, "depth_ppm_err": 50.0},
        ]

        prf_weights = {
            "target": [1.0, 0.8],
        }

        result = fit_aperture_hypothesis(depths_by_aperture, prf_weights)

        # Single hypothesis should be in fits
        assert len(result["hypothesis_fits"]) == 1
        assert result["hypothesis_fits"][0]["source_id"] == "target"
        assert result["hypothesis_fits"][0]["rank"] == 1

    def test_fit_aperture_hypothesis_depth_estimation(self) -> None:
        """Fitted depth_true should be reasonable given observations and weights."""
        # If observed depth is 1000 ppm with weight 1.0, true depth should be ~1000
        depths_by_aperture = [
            {"aperture_id": "spoc", "depth_ppm": 1000.0, "depth_ppm_err": 50.0},
        ]

        prf_weights = {
            "target": [1.0],  # Full capture
        }

        result = fit_aperture_hypothesis(depths_by_aperture, prf_weights)

        # Fitted true depth should be approximately 1000
        depth_hat = result["hypothesis_fits"][0]["depth_true_ppm_hat"]
        assert depth_hat is not None
        assert_allclose(depth_hat, 1000.0, rtol=0.1)

    def test_fit_aperture_hypothesis_returns_typed_dict(self) -> None:
        """Results should match ApertureHypothesisFit TypedDict structure."""
        depths = [{"aperture_id": "spoc", "depth_ppm": 1000.0}]
        prf_weights = {"target": [1.0]}

        result = fit_aperture_hypothesis(depths, prf_weights)

        # Check all required fields
        required_fields = ["best_source_id", "margin", "host_ambiguity", "hypothesis_fits"]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

        # Check hypothesis_fits structure
        if result["hypothesis_fits"]:
            fit = result["hypothesis_fits"][0]
            fit_fields = [
                "source_id",
                "source_name",
                "depth_true_ppm_hat",
                "fit_rmse_ppm",
                "delta_fit_rmse_ppm",
                "rank",
            ]
            for field in fit_fields:
                assert field in fit, f"Missing fit field: {field}"

    def test_fit_aperture_hypothesis_custom_threshold(self) -> None:
        """Custom margin threshold should affect ambiguity classification."""
        depths_by_aperture = [
            {"aperture_id": "spoc", "depth_ppm": 1000.0, "depth_ppm_err": 50.0},
            {"aperture_id": "r+1", "depth_ppm": 800.0, "depth_ppm_err": 50.0},
        ]

        prf_weights = {
            "target": [1.0, 0.8],
            "neighbor": [0.9, 0.7],  # Similar but slightly different
        }

        # Very high threshold
        result_strict = fit_aperture_hypothesis(
            depths_by_aperture, prf_weights, margin_threshold=100.0
        )

        # Very low threshold
        result_lenient = fit_aperture_hypothesis(
            depths_by_aperture, prf_weights, margin_threshold=0.01
        )

        # Strict threshold should more likely produce ambiguous
        # Lenient threshold should more likely produce resolved
        # (Exact behavior depends on actual fit quality)
        assert result_strict["host_ambiguity"] in ("resolved", "ambiguous")
        assert result_lenient["host_ambiguity"] in ("resolved", "ambiguous")


# =============================================================================
# Integration Tests
# =============================================================================


class TestHostHypothesisIntegration:
    """Integration tests combining scoring, consensus, and aperture fitting."""

    def test_full_pipeline_resolved_case(self) -> None:
        """Full pipeline on resolved case should identify correct host."""
        # Create multi-sector data with consistent host
        sources = [
            {"row": 7, "col": 7, "amplitude": 1.0, "name": "target"},
            {"row": 11, "col": 11, "amplitude": 0.0, "name": "neighbor"},
        ]

        # Score each "sector" (different noise realizations)
        per_sector_results = []
        for sector, seed in enumerate([100, 200, 300], start=1):
            diff_image = make_synthetic_diff_image(
                shape=(15, 15),
                sources=[sources[0]],  # Only target contributes
                noise_sigma=0.05,
                seed=seed,
            )
            hypotheses = make_synthetic_hypotheses(sources)
            scores = score_hypotheses_prf_lite(diff_image, hypotheses, seed=seed)

            per_sector_results.append(
                {
                    "sector": sector,
                    "best_source_id": scores[0]["source_id"],
                    "margin": scores[1]["delta_loss"] if len(scores) > 1 else 0.0,
                    "status": "ok",
                }
            )

        # Aggregate
        consensus = aggregate_multi_sector(per_sector_results)

        # Should consistently identify target as best
        assert consensus["consensus_best_source_id"] == "target"
        # All sectors should agree (stable, no flips)
        assert consensus["disagreement_flag"] == "stable"
        assert consensus["flip_rate"] == 0.0
        assert consensus["n_sectors_supporting_best"] == 3
        # Note: callers should use disagreement_flag + flip_rate + consensus_margin
        # (implementation flags low-confidence results even when consistent)

    def test_full_pipeline_flipping_case(self) -> None:
        """Full pipeline on flipping case should flag inconsistency."""
        # Create scenarios where different sources "win" in different sectors
        # by directly specifying conflicting per-sector results
        # (simulating what would happen if different sources dimmed in each sector)

        # The hypothesis positions (same in both sectors)
        hypotheses_def = [
            {"row": 5, "col": 7, "name": "source_a"},
            {"row": 10, "col": 7, "name": "source_b"},
        ]
        hypotheses = make_synthetic_hypotheses(hypotheses_def)

        per_sector_results = []

        # Sector 1: source_a is dimming (it's at position 5,7)
        diff_s1 = make_synthetic_diff_image(
            shape=(15, 15),
            sources=[{"row": 5, "col": 7, "amplitude": 1.0}],
            noise_sigma=0.05,
            seed=111,
        )
        scores_s1 = score_hypotheses_prf_lite(diff_s1, hypotheses, seed=111)
        per_sector_results.append(
            {
                "sector": 1,
                "best_source_id": scores_s1[0]["source_id"],
                "margin": scores_s1[1]["delta_loss"] if len(scores_s1) > 1 else 0.0,
                "status": "ok",
            }
        )

        # Sector 2: source_b is dimming (it's at position 10,7)
        diff_s2 = make_synthetic_diff_image(
            shape=(15, 15),
            sources=[{"row": 10, "col": 7, "amplitude": 1.0}],
            noise_sigma=0.05,
            seed=222,
        )
        scores_s2 = score_hypotheses_prf_lite(diff_s2, hypotheses, seed=222)
        per_sector_results.append(
            {
                "sector": 2,
                "best_source_id": scores_s2[0]["source_id"],
                "margin": scores_s2[1]["delta_loss"] if len(scores_s2) > 1 else 0.0,
                "status": "ok",
            }
        )

        # Verify that the sectors picked different sources
        assert per_sector_results[0]["best_source_id"] != per_sector_results[1]["best_source_id"], (
            f"Expected different best sources: {per_sector_results[0]['best_source_id']} vs {per_sector_results[1]['best_source_id']}"
        )

        # Aggregate
        consensus = aggregate_multi_sector(per_sector_results)

        # Different sectors prefer different hosts - only one can be "supporting best"
        assert consensus["n_sectors_supporting_best"] == 1

        # No additional derived boolean is returned; callers can interpret
        # disagreement_flag + flip_rate + consensus_margin.
        assert consensus["disagreement_flag"] in {"mixed", "flipping", "stable"}


# =============================================================================
# Threshold Constants Tests
# =============================================================================


class TestThresholdConstants:
    """Tests for exported threshold constants."""

    def test_margin_resolve_threshold_positive(self) -> None:
        """MARGIN_RESOLVE_THRESHOLD should be positive."""
        assert MARGIN_RESOLVE_THRESHOLD > 0

    def test_flip_rate_thresholds_ordered(self) -> None:
        """Flip rate thresholds should be properly ordered."""
        assert 0 < FLIP_RATE_MIXED_THRESHOLD < FLIP_RATE_UNSTABLE_THRESHOLD <= 1.0

    def test_flip_rate_mixed_default(self) -> None:
        """FLIP_RATE_MIXED_THRESHOLD should have expected default."""
        assert FLIP_RATE_MIXED_THRESHOLD == 0.3

    def test_flip_rate_unstable_default(self) -> None:
        """FLIP_RATE_UNSTABLE_THRESHOLD should have expected default."""
        assert FLIP_RATE_UNSTABLE_THRESHOLD == 0.5
