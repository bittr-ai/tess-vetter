"""Tests for PRF-based aperture depth prediction and conflict detection.

Tests verify:
- Centered source produces correct host fractions
- Dilution from neighbors reduces observed depth
- Uncertainty propagation works correctly
- Conflict detection triggers on mismatched best sources
- Chi-squared computation
"""

from __future__ import annotations

import numpy as np
import pytest

from bittr_tess_vetter.compute.aperture_prediction import (
    ApertureConflict,
    AperturePrediction,
    compute_aperture_chi2,
    detect_aperture_conflict,
    predict_all_hypotheses,
    predict_depth_vs_aperture,
    propagate_aperture_uncertainty,
)
from bittr_tess_vetter.compute.prf_psf import get_prf_model

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def prf_model():
    """Get a standard PRF model for testing."""
    return get_prf_model(backend="parametric")


@pytest.fixture
def stamp_shape() -> tuple[int, int]:
    """Standard TESS stamp size (11x11)."""
    return (11, 11)


@pytest.fixture
def aperture_radii() -> list[float]:
    """Standard aperture radii for testing."""
    return [1.5, 2.0, 2.5, 3.0, 3.5]


# =============================================================================
# Test predict_depth_vs_aperture
# =============================================================================


class TestPredictDepthVsAperture:
    """Tests for predict_depth_vs_aperture function."""

    def test_centered_source_has_high_host_fraction(
        self,
        prf_model,
        stamp_shape,
        aperture_radii,
    ):
        """Centered source should have host fraction close to 1."""
        # Source at stamp center
        center_row = (stamp_shape[0] - 1) / 2.0
        center_col = (stamp_shape[1] - 1) / 2.0

        prediction = predict_depth_vs_aperture(
            hypothesis_row=center_row,
            hypothesis_col=center_col,
            depth_true=1000.0,
            aperture_radii=aperture_radii,
            prf_model=prf_model,
            stamp_shape=stamp_shape,
        )

        # All host fractions should be close to 1 (no dilution)
        for hf in prediction.host_fractions:
            assert hf > 0.9, f"Host fraction {hf} should be close to 1 for centered source"

        # Predicted depths should be close to true depth
        for depth in prediction.predicted_depths:
            assert 900.0 < depth <= 1000.0, f"Predicted depth {depth} should be close to 1000"

    def test_host_fraction_varies_with_geometry(
        self,
        prf_model,
        stamp_shape,
        aperture_radii,
    ):
        """Host fraction should vary based on source and aperture geometry."""
        center_row = (stamp_shape[0] - 1) / 2.0
        center_col = (stamp_shape[1] - 1) / 2.0

        # Off-center hypothesis with centered neighbor
        prediction = predict_depth_vs_aperture(
            hypothesis_row=center_row + 2.0,  # Off-center hypothesis
            hypothesis_col=center_col,
            depth_true=1000.0,
            aperture_radii=aperture_radii,
            prf_model=prf_model,
            stamp_shape=stamp_shape,
            aperture_center=(center_row, center_col),  # Aperture centered at neighbor
            other_sources=[(center_row, center_col, 1.0)],  # Neighbor at center
        )

        # With dilution, host fractions should be less than 1.0
        for hf in prediction.host_fractions:
            assert hf < 1.0, "Host fraction should be diluted by neighbor"
            assert hf > 0.0, "Host fraction should be positive"

        # Host fraction should vary with aperture size (not constant)
        # The exact direction depends on geometry, but variation should exist
        assert prediction.host_fractions[0] != prediction.host_fractions[-1]

    def test_dilution_from_neighbor(
        self,
        prf_model,
        stamp_shape,
        aperture_radii,
    ):
        """Adding a neighbor should reduce observed depth due to dilution."""
        center_row = (stamp_shape[0] - 1) / 2.0
        center_col = (stamp_shape[1] - 1) / 2.0

        # Prediction without neighbor
        pred_solo = predict_depth_vs_aperture(
            hypothesis_row=center_row,
            hypothesis_col=center_col,
            depth_true=1000.0,
            aperture_radii=aperture_radii,
            prf_model=prf_model,
            stamp_shape=stamp_shape,
        )

        # Prediction with neighbor (equal brightness, 2 pixels away)
        pred_with_neighbor = predict_depth_vs_aperture(
            hypothesis_row=center_row,
            hypothesis_col=center_col,
            depth_true=1000.0,
            aperture_radii=aperture_radii,
            prf_model=prf_model,
            stamp_shape=stamp_shape,
            other_sources=[(center_row, center_col + 2.0, 1.0)],
        )

        # With neighbor, host fractions should be lower
        for i, (hf_solo, hf_neighbor) in enumerate(
            zip(pred_solo.host_fractions, pred_with_neighbor.host_fractions, strict=True)
        ):
            assert hf_neighbor < hf_solo, (
                f"At radius {aperture_radii[i]}: host fraction with neighbor "
                f"({hf_neighbor:.3f}) should be less than solo ({hf_solo:.3f})"
            )

    def test_returns_correct_structure(
        self,
        prf_model,
        stamp_shape,
        aperture_radii,
    ):
        """Prediction should have correct structure."""
        center_row = (stamp_shape[0] - 1) / 2.0
        center_col = (stamp_shape[1] - 1) / 2.0

        prediction = predict_depth_vs_aperture(
            hypothesis_row=center_row,
            hypothesis_col=center_col,
            depth_true=1000.0,
            aperture_radii=aperture_radii,
            prf_model=prf_model,
            stamp_shape=stamp_shape,
        )

        assert isinstance(prediction, AperturePrediction)
        assert len(prediction.radii_px) == len(aperture_radii)
        assert len(prediction.predicted_depths) == len(aperture_radii)
        assert len(prediction.host_fractions) == len(aperture_radii)
        assert prediction.uncertainties is None  # Not propagated yet


# =============================================================================
# Test predict_all_hypotheses
# =============================================================================


class TestPredictAllHypotheses:
    """Tests for predict_all_hypotheses function."""

    def test_returns_prediction_for_each_hypothesis(
        self,
        prf_model,
        stamp_shape,
        aperture_radii,
    ):
        """Should return a prediction for each hypothesis."""
        center_row = (stamp_shape[0] - 1) / 2.0
        center_col = (stamp_shape[1] - 1) / 2.0

        hypotheses = [
            {"source_id": "target", "row": center_row, "col": center_col, "flux_ratio": 1.0},
            {
                "source_id": "neighbor",
                "row": center_row + 2.0,
                "col": center_col,
                "flux_ratio": 0.5,
            },
        ]

        predictions = predict_all_hypotheses(
            hypotheses=hypotheses,
            depth_estimate=1000.0,
            aperture_radii=aperture_radii,
            prf_model=prf_model,
            stamp_shape=stamp_shape,
        )

        assert "target" in predictions
        assert "neighbor" in predictions
        assert predictions["target"].source_id == "target"
        assert predictions["neighbor"].source_id == "neighbor"

    def test_empty_hypotheses_returns_empty_dict(
        self,
        prf_model,
        stamp_shape,
        aperture_radii,
    ):
        """Empty hypothesis list should return empty dict."""
        predictions = predict_all_hypotheses(
            hypotheses=[],
            depth_estimate=1000.0,
            aperture_radii=aperture_radii,
            prf_model=prf_model,
            stamp_shape=stamp_shape,
        )

        assert predictions == {}


# =============================================================================
# Test propagate_aperture_uncertainty
# =============================================================================


class TestPropagateApertureUncertainty:
    """Tests for propagate_aperture_uncertainty function."""

    def test_adds_uncertainties_to_prediction(
        self,
        prf_model,
        stamp_shape,
        aperture_radii,
    ):
        """Should add uncertainties to the prediction."""
        center_row = (stamp_shape[0] - 1) / 2.0
        center_col = (stamp_shape[1] - 1) / 2.0

        pred = predict_depth_vs_aperture(
            hypothesis_row=center_row,
            hypothesis_col=center_col,
            depth_true=1000.0,
            aperture_radii=aperture_radii,
            prf_model=prf_model,
            stamp_shape=stamp_shape,
        )

        assert pred.uncertainties is None

        pred_with_unc = propagate_aperture_uncertainty(
            pred,
            depth_uncertainty=100.0,
            prf_position_uncertainty=0.1,
        )

        assert pred_with_unc.uncertainties is not None
        assert len(pred_with_unc.uncertainties) == len(aperture_radii)
        for unc in pred_with_unc.uncertainties:
            assert unc > 0, "Uncertainties should be positive"

    def test_larger_depth_uncertainty_increases_overall(
        self,
        prf_model,
        stamp_shape,
        aperture_radii,
    ):
        """Larger depth uncertainty should increase overall uncertainty."""
        center_row = (stamp_shape[0] - 1) / 2.0
        center_col = (stamp_shape[1] - 1) / 2.0

        pred = predict_depth_vs_aperture(
            hypothesis_row=center_row,
            hypothesis_col=center_col,
            depth_true=1000.0,
            aperture_radii=aperture_radii,
            prf_model=prf_model,
            stamp_shape=stamp_shape,
        )

        pred_small = propagate_aperture_uncertainty(pred, depth_uncertainty=50.0)
        pred_large = propagate_aperture_uncertainty(pred, depth_uncertainty=200.0)

        for i in range(len(aperture_radii)):
            assert pred_large.uncertainties is not None
            assert pred_small.uncertainties is not None
            assert pred_large.uncertainties[i] > pred_small.uncertainties[i]


# =============================================================================
# Test compute_aperture_chi2
# =============================================================================


class TestComputeApertureChi2:
    """Tests for compute_aperture_chi2 function."""

    def test_perfect_fit_has_low_chi2(self):
        """Perfect fit should have chi2 close to 0."""
        observed = [100.0, 200.0, 300.0]
        predicted = [100.0, 200.0, 300.0]
        uncertainties = [10.0, 10.0, 10.0]

        chi2, pvalue = compute_aperture_chi2(observed, predicted, uncertainties)

        assert chi2 == pytest.approx(0.0)
        assert pvalue > 0.99  # Very high p-value for perfect fit

    def test_poor_fit_has_high_chi2(self):
        """Poor fit should have high chi2."""
        observed = [100.0, 200.0, 300.0]
        predicted = [200.0, 300.0, 400.0]  # All off by 100
        uncertainties = [10.0, 10.0, 10.0]

        chi2, pvalue = compute_aperture_chi2(observed, predicted, uncertainties)

        assert chi2 > 100.0  # Should be very high
        assert pvalue < 0.01  # Low p-value

    def test_handles_empty_inputs(self):
        """Empty inputs should return nan."""
        chi2, pvalue = compute_aperture_chi2([], [], [])

        assert np.isnan(chi2)
        assert np.isnan(pvalue)

    def test_handles_nan_values(self):
        """NaN values should be filtered out."""
        observed = [100.0, float("nan"), 300.0]
        predicted = [100.0, 200.0, 300.0]
        uncertainties = [10.0, 10.0, 10.0]

        chi2, pvalue = compute_aperture_chi2(observed, predicted, uncertainties)

        # Should not be nan (filtered out)
        assert np.isfinite(chi2)
        assert np.isfinite(pvalue)


# =============================================================================
# Test detect_aperture_conflict
# =============================================================================


class TestDetectApertureConflict:
    """Tests for detect_aperture_conflict function."""

    @pytest.fixture
    def predictions_target_better(self):
        """Predictions where target is clearly better fit."""
        return {
            "target": AperturePrediction(
                source_id="target",
                radii_px=[1.5, 2.0, 2.5],
                predicted_depths=[1000.0, 990.0, 980.0],
                host_fractions=[0.98, 0.95, 0.92],
                uncertainties=[50.0, 50.0, 50.0],
            ),
            "neighbor": AperturePrediction(
                source_id="neighbor",
                radii_px=[1.5, 2.0, 2.5],
                predicted_depths=[600.0, 700.0, 800.0],  # Very different
                host_fractions=[0.6, 0.7, 0.8],
                uncertainties=[50.0, 50.0, 50.0],
            ),
        }

    @pytest.fixture
    def predictions_neighbor_better(self):
        """Predictions where neighbor is clearly better fit."""
        return {
            "target": AperturePrediction(
                source_id="target",
                radii_px=[1.5, 2.0, 2.5],
                predicted_depths=[600.0, 700.0, 800.0],  # Very different from observed
                host_fractions=[0.6, 0.7, 0.8],
                uncertainties=[50.0, 50.0, 50.0],
            ),
            "neighbor": AperturePrediction(
                source_id="neighbor",
                radii_px=[1.5, 2.0, 2.5],
                predicted_depths=[1000.0, 990.0, 980.0],  # Close to observed
                host_fractions=[0.98, 0.95, 0.92],
                uncertainties=[50.0, 50.0, 50.0],
            ),
        }

    def test_no_conflict_when_same_best(self, predictions_target_better):
        """No conflict when localization and aperture agree."""
        localization_result = {"consensus_best_source_id": "target"}
        observed_depths = [1000.0, 990.0, 980.0]

        conflict = detect_aperture_conflict(
            localization_result=localization_result,
            observed_depths=observed_depths,
            predictions=predictions_target_better,
        )

        assert conflict is None

    def test_conflict_when_different_best(self, predictions_neighbor_better):
        """Conflict detected when localization prefers target but aperture prefers neighbor."""
        localization_result = {"consensus_best_source_id": "target"}
        observed_depths = [1000.0, 990.0, 980.0]

        conflict = detect_aperture_conflict(
            localization_result=localization_result,
            observed_depths=observed_depths,
            predictions=predictions_neighbor_better,
            margin_threshold=2.0,
        )

        assert conflict is not None
        assert isinstance(conflict, ApertureConflict)
        assert conflict.localization_best == "target"
        assert conflict.aperture_best == "neighbor"
        assert conflict.conflict_type == "CONFLICT_APERTURE_LOCALIZATION"

    def test_returns_none_for_empty_predictions(self):
        """Empty predictions should return None."""
        localization_result = {"consensus_best_source_id": "target"}
        observed_depths = [1000.0, 990.0, 980.0]

        conflict = detect_aperture_conflict(
            localization_result=localization_result,
            observed_depths=observed_depths,
            predictions={},
        )

        assert conflict is None

    def test_returns_none_for_empty_localization(self, predictions_target_better):
        """Missing localization best should return None."""
        localization_result = {}
        observed_depths = [1000.0, 990.0, 980.0]

        conflict = detect_aperture_conflict(
            localization_result=localization_result,
            observed_depths=observed_depths,
            predictions=predictions_target_better,
        )

        assert conflict is None

    def test_conflict_has_to_dict(self, predictions_neighbor_better):
        """Conflict should be serializable via to_dict."""
        localization_result = {"consensus_best_source_id": "target"}
        observed_depths = [1000.0, 990.0, 980.0]

        conflict = detect_aperture_conflict(
            localization_result=localization_result,
            observed_depths=observed_depths,
            predictions=predictions_neighbor_better,
        )

        assert conflict is not None
        d = conflict.to_dict()
        assert isinstance(d, dict)
        assert d["localization_best"] == "target"
        assert d["aperture_best"] == "neighbor"
        assert d["conflict_type"] == "CONFLICT_APERTURE_LOCALIZATION"


# =============================================================================
# Test AperturePrediction
# =============================================================================


class TestAperturePrediction:
    """Tests for AperturePrediction dataclass."""

    def test_to_dict(self):
        """to_dict should return serializable dict."""
        pred = AperturePrediction(
            source_id="test",
            radii_px=[1.5, 2.0],
            predicted_depths=[1000.0, 950.0],
            host_fractions=[0.98, 0.95],
            uncertainties=[50.0, 45.0],
        )

        d = pred.to_dict()

        assert isinstance(d, dict)
        assert d["source_id"] == "test"
        assert d["radii_px"] == [1.5, 2.0]
        assert d["predicted_depths"] == [1000.0, 950.0]
        assert d["host_fractions"] == [0.98, 0.95]
        assert d["uncertainties"] == [50.0, 45.0]

    def test_frozen(self):
        """AperturePrediction should be immutable."""
        pred = AperturePrediction(
            source_id="test",
            radii_px=[1.5, 2.0],
            predicted_depths=[1000.0, 950.0],
            host_fractions=[0.98, 0.95],
            uncertainties=None,
        )

        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            pred.source_id = "modified"  # type: ignore
