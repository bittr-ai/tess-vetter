"""Unit tests for joint multi-sector likelihood computation.

Tests the joint_likelihood module which provides:
- Sector quality assessment and weighting
- Joint log-likelihood computation across sectors
- Hypothesis selection with ambiguity detection

All tests are deterministic and require no network or file I/O.
"""

from __future__ import annotations

import pytest

from bittr_tess_vetter.compute.joint_inference_schemas import SectorEvidence
from bittr_tess_vetter.compute.joint_likelihood import (
    assess_sector_quality,
    compute_all_hypotheses_joint,
    compute_joint_log_likelihood,
    compute_sector_weights,
    select_best_hypothesis_joint,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_hypothesis_a() -> dict:
    """Sample hypothesis A (target)."""
    return {
        "source_id": "tic:target:123",
        "source_name": "target",
        "fit_loss": 100.0,
        "delta_loss": 0.0,
        "rank": 1,
        "fit_amplitude": -0.001,
        "fit_background": 0.0001,
    }


@pytest.fixture
def sample_hypothesis_b() -> dict:
    """Sample hypothesis B (neighbor)."""
    return {
        "source_id": "gaia_dr3:987654321",
        "source_name": "Gaia DR3 987654321",
        "fit_loss": 150.0,
        "delta_loss": 50.0,
        "rank": 2,
        "fit_amplitude": -0.0008,
        "fit_background": 0.0002,
    }


@pytest.fixture
def sector_evidence_high_quality(
    sample_hypothesis_a: dict,
    sample_hypothesis_b: dict,
) -> SectorEvidence:
    """High quality sector evidence with two hypotheses."""
    return SectorEvidence(
        sector=15,
        tpf_fits_ref="tpf_fits:123456789:15:spoc",
        hypotheses=[sample_hypothesis_a, sample_hypothesis_b],
        residual_rms=0.00010,
        quality_weight=1.0,
        downweight_reason=None,
        nuisance_params={},
    )


@pytest.fixture
def sector_evidence_low_quality(
    sample_hypothesis_a: dict,
    sample_hypothesis_b: dict,
) -> SectorEvidence:
    """Low quality sector evidence (high residual)."""
    return SectorEvidence(
        sector=42,
        tpf_fits_ref="tpf_fits:123456789:42:spoc",
        hypotheses=[sample_hypothesis_a, sample_hypothesis_b],
        residual_rms=0.0005,
        quality_weight=0.5,
        downweight_reason="high_residual",
        nuisance_params={},
    )


@pytest.fixture
def sector_evidence_b_wins(sample_hypothesis_b: dict) -> SectorEvidence:
    """Sector evidence where hypothesis B wins."""
    hyp_b_wins = sample_hypothesis_b.copy()
    hyp_b_wins["fit_loss"] = 80.0
    hyp_b_wins["delta_loss"] = 0.0
    hyp_b_wins["rank"] = 1

    hyp_a_loses = {
        "source_id": "tic:target:123",
        "source_name": "target",
        "fit_loss": 180.0,
        "delta_loss": 100.0,
        "rank": 2,
        "fit_amplitude": -0.0005,
        "fit_background": 0.0001,
    }

    return SectorEvidence(
        sector=67,
        tpf_fits_ref="tpf_fits:123456789:67:spoc",
        hypotheses=[hyp_b_wins, hyp_a_loses],
        residual_rms=0.00015,
        quality_weight=1.0,
        downweight_reason=None,
        nuisance_params={},
    )


# =============================================================================
# Sector Quality Assessment Tests
# =============================================================================


class TestAssessSectorQuality:
    """Tests for assess_sector_quality function."""

    def test_full_weight_for_good_sector(
        self, sector_evidence_high_quality: SectorEvidence
    ) -> None:
        """High quality sector gets full weight."""
        weight, reason = assess_sector_quality(sector_evidence_high_quality)

        assert weight == 1.0
        assert reason is None

    def test_reduced_weight_for_low_quality_sector(
        self, sector_evidence_low_quality: SectorEvidence
    ) -> None:
        """Low quality sector with pre-set weight returns that weight."""
        weight, reason = assess_sector_quality(sector_evidence_low_quality)

        # Pre-set quality_weight is returned as-is
        assert weight == 0.5
        assert reason == "high_residual"

    def test_weight_in_valid_range(self, sector_evidence_high_quality: SectorEvidence) -> None:
        """Weight is always in [0, 1]."""
        weight, _ = assess_sector_quality(sector_evidence_high_quality)

        assert 0.0 <= weight <= 1.0

    def test_empty_hypotheses_returns_full_weight(self) -> None:
        """Sector with empty hypotheses returns full weight (no SNR check)."""
        se = SectorEvidence(
            sector=1,
            tpf_fits_ref="tpf_fits:111:1:spoc",
            hypotheses=[],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )

        weight, reason = assess_sector_quality(se)

        assert weight == 1.0
        assert reason is None


class TestComputeSectorWeights:
    """Tests for compute_sector_weights function."""

    def test_computes_weights_for_all_sectors(
        self,
        sector_evidence_high_quality: SectorEvidence,
        sector_evidence_low_quality: SectorEvidence,
    ) -> None:
        """Returns weights for all sectors."""
        weights = compute_sector_weights(
            [sector_evidence_high_quality, sector_evidence_low_quality]
        )

        assert 15 in weights
        assert 42 in weights
        assert len(weights) == 2

    def test_high_quality_gets_higher_weight(
        self,
        sector_evidence_high_quality: SectorEvidence,
        sector_evidence_low_quality: SectorEvidence,
    ) -> None:
        """High quality sector gets higher weight."""
        weights = compute_sector_weights(
            [sector_evidence_high_quality, sector_evidence_low_quality]
        )

        assert weights[15] >= weights[42]


# =============================================================================
# Joint Log-Likelihood Computation Tests
# =============================================================================


class TestComputeJointLogLikelihood:
    """Tests for compute_joint_log_likelihood function."""

    def test_sums_across_sectors(
        self,
        sector_evidence_high_quality: SectorEvidence,
        sector_evidence_low_quality: SectorEvidence,
        sample_hypothesis_a: dict,
    ) -> None:
        """Joint log-likelihood is sum of weighted sector contributions."""
        sectors = [sector_evidence_high_quality, sector_evidence_low_quality]
        hyp_id = sample_hypothesis_a["source_id"]

        joint_ll, contributions = compute_joint_log_likelihood(
            sectors, hyp_id, sector_weights={15: 1.0, 42: 0.5}
        )

        # Contributions should sum to joint
        assert joint_ll == pytest.approx(sum(contributions.values()))

    def test_returns_per_sector_contributions(
        self,
        sector_evidence_high_quality: SectorEvidence,
        sample_hypothesis_a: dict,
    ) -> None:
        """Returns contribution from each sector."""
        sectors = [sector_evidence_high_quality]
        hyp_id = sample_hypothesis_a["source_id"]

        _, contributions = compute_joint_log_likelihood(sectors, hyp_id)

        assert 15 in contributions
        assert isinstance(contributions[15], float)

    def test_missing_hypothesis_contributes_zero(
        self, sector_evidence_high_quality: SectorEvidence
    ) -> None:
        """Hypothesis not in sector contributes zero."""
        sectors = [sector_evidence_high_quality]

        _, contributions = compute_joint_log_likelihood(sectors, "nonexistent_source_id")

        assert contributions[15] == 0.0

    def test_downweighting_reduces_contribution(
        self, sector_evidence_high_quality: SectorEvidence, sample_hypothesis_a: dict
    ) -> None:
        """Lower sector weight reduces contribution."""
        sectors = [sector_evidence_high_quality]
        hyp_id = sample_hypothesis_a["source_id"]

        _, contributions_full = compute_joint_log_likelihood(
            sectors, hyp_id, sector_weights={15: 1.0}
        )
        _, contributions_half = compute_joint_log_likelihood(
            sectors, hyp_id, sector_weights={15: 0.5}
        )

        assert contributions_half[15] == pytest.approx(contributions_full[15] * 0.5)


class TestComputeAllHypothesesJoint:
    """Tests for compute_all_hypotheses_joint function."""

    def test_returns_all_hypotheses(self, sector_evidence_high_quality: SectorEvidence) -> None:
        """Returns joint log-likelihood for all hypotheses."""
        sectors = [sector_evidence_high_quality]
        hypotheses = ["tic:target:123", "gaia_dr3:987654321"]

        joint_lls = compute_all_hypotheses_joint(sectors, hypotheses)

        assert "tic:target:123" in joint_lls
        assert "gaia_dr3:987654321" in joint_lls

    def test_best_hypothesis_has_highest_likelihood(
        self, sector_evidence_high_quality: SectorEvidence
    ) -> None:
        """Best hypothesis (lower fit_loss) has higher log-likelihood."""
        sectors = [sector_evidence_high_quality]
        hypotheses = ["tic:target:123", "gaia_dr3:987654321"]

        joint_lls = compute_all_hypotheses_joint(sectors, hypotheses)

        # Target has fit_loss=100, neighbor has fit_loss=150
        # Lower fit_loss -> higher log-likelihood
        assert joint_lls["tic:target:123"] > joint_lls["gaia_dr3:987654321"]


# =============================================================================
# Best Hypothesis Selection Tests
# =============================================================================


class TestSelectBestHypothesisJoint:
    """Tests for select_best_hypothesis_joint function."""

    def test_selects_highest_likelihood(self) -> None:
        """Selects hypothesis with highest log-likelihood."""
        joint_lls = {
            "tic:target:123": -100.0,
            "gaia_dr3:987654321": -150.0,
        }

        best_id, verdict, delta = select_best_hypothesis_joint(joint_lls)

        assert best_id == "tic:target:123"
        assert delta == 50.0

    def test_on_target_verdict_for_target_winner(self) -> None:
        """Returns ON_TARGET when target wins with margin."""
        joint_lls = {
            "tic:target:123": -100.0,
            "gaia_dr3:987654321": -105.0,  # margin = 5 > 2
        }

        _, verdict, _ = select_best_hypothesis_joint(joint_lls)

        assert verdict == "ON_TARGET"

    def test_off_target_verdict_for_neighbor_winner(self) -> None:
        """Returns OFF_TARGET when non-target wins with margin."""
        joint_lls = {
            "tic:123456789": -150.0,
            "gaia_dr3:best": -100.0,  # margin = 50 > 2
        }

        best_id, verdict, _ = select_best_hypothesis_joint(joint_lls)

        assert best_id == "gaia_dr3:best"
        assert verdict == "OFF_TARGET"

    def test_ambiguous_verdict_for_close_likelihoods(self) -> None:
        """Returns AMBIGUOUS when likelihoods are too close."""
        joint_lls = {
            "tic:target:123": -100.0,
            "gaia_dr3:987654321": -101.0,  # margin = 1 < 2
        }

        _, verdict, delta = select_best_hypothesis_joint(joint_lls)

        assert verdict == "AMBIGUOUS"
        assert delta == pytest.approx(1.0)

    def test_custom_margin_threshold(self) -> None:
        """Respects custom margin threshold."""
        joint_lls = {
            "tic:target:123": -100.0,
            "gaia_dr3:987654321": -103.0,  # margin = 3
        }

        # With threshold 5, this is ambiguous
        _, verdict, _ = select_best_hypothesis_joint(joint_lls, margin_threshold=5.0)
        assert verdict == "AMBIGUOUS"

        # With threshold 2, this is resolved
        _, verdict, _ = select_best_hypothesis_joint(joint_lls, margin_threshold=2.0)
        assert verdict == "ON_TARGET"

    def test_single_hypothesis(self) -> None:
        """Single hypothesis returns that hypothesis."""
        joint_lls = {"tic:target:123": -100.0}

        best_id, verdict, delta = select_best_hypothesis_joint(joint_lls)

        assert best_id == "tic:target:123"
        assert verdict == "ON_TARGET"
        assert delta == 0.0

    def test_empty_hypotheses(self) -> None:
        """Empty hypotheses returns empty result."""
        joint_lls: dict[str, float] = {}

        best_id, verdict, delta = select_best_hypothesis_joint(joint_lls)

        assert best_id == ""
        assert verdict == "AMBIGUOUS"
        assert delta == 0.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestJointLikelihoodIntegration:
    """Integration tests for joint likelihood workflow."""

    def test_end_to_end_stable_result(
        self,
        sector_evidence_high_quality: SectorEvidence,
        sector_evidence_low_quality: SectorEvidence,
    ) -> None:
        """Full workflow produces expected result for stable sectors."""
        sectors = [sector_evidence_high_quality, sector_evidence_low_quality]
        hypotheses = ["tic:target:123", "gaia_dr3:987654321"]

        # Compute weights
        weights = compute_sector_weights(sectors)

        # Compute all joint likelihoods
        joint_lls = compute_all_hypotheses_joint(sectors, hypotheses, sector_weights=weights)

        # Select best
        best_id, verdict, delta = select_best_hypothesis_joint(joint_lls)

        # Target should win (lower fit_loss)
        assert "target" in best_id.lower()
        assert verdict == "ON_TARGET"
        assert delta > 0

    def test_mixed_sector_preferences(
        self,
        sector_evidence_high_quality: SectorEvidence,
        sector_evidence_b_wins: SectorEvidence,
    ) -> None:
        """Handles sectors with different preferences."""
        # sector_evidence_high_quality: A wins
        # sector_evidence_b_wins: B wins
        sectors = [sector_evidence_high_quality, sector_evidence_b_wins]
        hypotheses = ["tic:target:123", "gaia_dr3:987654321"]

        joint_lls = compute_all_hypotheses_joint(sectors, hypotheses)

        # Both hypotheses should have non-zero likelihoods
        assert joint_lls["tic:target:123"] != 0.0 or joint_lls["gaia_dr3:987654321"] != 0.0
