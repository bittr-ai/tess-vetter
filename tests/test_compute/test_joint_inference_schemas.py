"""Unit tests for joint multi-sector inference output schemas.

Tests the joint_inference_schemas module which provides:
- SectorEvidence and JointInferenceResult dataclasses
- Serialization helpers (to_dict, from_dict)
- Evidence block generation for vetting pipeline
- Placeholder factory for creating joint results

All tests are deterministic and require no network or file I/O.
"""

from __future__ import annotations

import json

import pytest

from tess_vetter.compute.joint_inference_schemas import (
    JointInferenceResult,
    SectorEvidence,
    create_joint_result_from_sectors,
    joint_result_from_dict,
    joint_result_to_dict,
    sector_evidence_from_dict,
    sector_evidence_to_dict,
    to_evidence_block,
)
from tess_vetter.compute.pixel_host_hypotheses import HypothesisScore

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_hypothesis_target() -> HypothesisScore:
    """Sample hypothesis for target source."""
    return HypothesisScore(
        source_id="tic:123456789",
        source_name="target",
        fit_loss=100.0,
        delta_loss=0.0,
        rank=1,
        fit_amplitude=-0.001,
        fit_background=0.0001,
    )


@pytest.fixture
def sample_hypothesis_neighbor() -> HypothesisScore:
    """Sample hypothesis for neighbor source."""
    return HypothesisScore(
        source_id="gaia_dr3:987654321",
        source_name="Gaia DR3 987654321",
        fit_loss=150.0,
        delta_loss=50.0,
        rank=2,
        fit_amplitude=-0.0008,
        fit_background=0.0002,
    )


@pytest.fixture
def sample_sector_evidence(
    sample_hypothesis_target: HypothesisScore,
    sample_hypothesis_neighbor: HypothesisScore,
) -> SectorEvidence:
    """Sample sector evidence with two hypotheses."""
    return SectorEvidence(
        sector=15,
        tpf_fits_ref="tpf_fits:123456789:15:spoc",
        hypotheses=[sample_hypothesis_target, sample_hypothesis_neighbor],
        residual_rms=0.00015,
        quality_weight=1.0,
        downweight_reason=None,
        nuisance_params={"background": 0.0001, "jitter": 0.3},
    )


@pytest.fixture
def sample_sector_evidence_downweighted(
    sample_hypothesis_target: HypothesisScore,
) -> SectorEvidence:
    """Sample sector evidence with downweighted quality."""
    return SectorEvidence(
        sector=42,
        tpf_fits_ref="tpf_fits:123456789:42:spoc",
        hypotheses=[sample_hypothesis_target],
        residual_rms=0.0005,
        quality_weight=0.5,
        downweight_reason="high_residual",
        nuisance_params={"background": 0.0002},
    )


@pytest.fixture
def sample_joint_result(
    sample_sector_evidence: SectorEvidence,
    sample_sector_evidence_downweighted: SectorEvidence,
) -> JointInferenceResult:
    """Sample joint inference result with two sectors."""
    return JointInferenceResult(
        joint_best_source_id="tic:123456789",
        verdict="ON_TARGET",
        resolved_probability=None,
        calibration_version=None,
        joint_log_likelihood=-200.0,
        delta_log_likelihood=5.0,
        posterior_odds=148.41,
        sector_evidence=[sample_sector_evidence, sample_sector_evidence_downweighted],
        sector_weights={15: 1.0, 42: 0.5},
        flip_rate=0.0,
        consistency_verdict="stable",
        hypotheses_considered=["tic:123456789", "gaia_dr3:987654321"],
        computation_time_seconds=0.123,
        warnings=[],
        blobs={"per_sector_table": "blob:abc123"},
    )


# =============================================================================
# SectorEvidence Tests
# =============================================================================


class TestSectorEvidence:
    """Tests for SectorEvidence dataclass."""

    def test_sector_evidence_instantiation(self, sample_hypothesis_target: HypothesisScore) -> None:
        """SectorEvidence can be instantiated with valid data."""
        se = SectorEvidence(
            sector=15,
            tpf_fits_ref="tpf_fits:123456789:15:spoc",
            hypotheses=[sample_hypothesis_target],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
            nuisance_params={},
        )

        assert se.sector == 15
        assert se.tpf_fits_ref == "tpf_fits:123456789:15:spoc"
        assert len(se.hypotheses) == 1
        assert se.residual_rms == 0.0001
        assert se.quality_weight == 1.0
        assert se.downweight_reason is None

    def test_sector_evidence_with_downweight(self) -> None:
        """SectorEvidence can capture downweight reasons."""
        se = SectorEvidence(
            sector=42,
            tpf_fits_ref="tpf_fits:123456789:42:spoc",
            hypotheses=[],
            residual_rms=0.001,
            quality_weight=0.3,
            downweight_reason="systematic_pattern",
            nuisance_params={"jitter": 0.8},
        )

        assert se.quality_weight == 0.3
        assert se.downweight_reason == "systematic_pattern"
        assert se.nuisance_params["jitter"] == 0.8

    def test_sector_evidence_default_nuisance_params(
        self, sample_hypothesis_target: HypothesisScore
    ) -> None:
        """SectorEvidence has empty nuisance_params by default."""
        se = SectorEvidence(
            sector=1,
            tpf_fits_ref="tpf_fits:111:1:spoc",
            hypotheses=[sample_hypothesis_target],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )

        assert se.nuisance_params == {}


# =============================================================================
# JointInferenceResult Tests
# =============================================================================


class TestJointInferenceResult:
    """Tests for JointInferenceResult dataclass."""

    def test_joint_result_instantiation(self, sample_sector_evidence: SectorEvidence) -> None:
        """JointInferenceResult can be instantiated with valid data."""
        result = JointInferenceResult(
            joint_best_source_id="tic:123456789",
            verdict="ON_TARGET",
            resolved_probability=None,
            calibration_version=None,
            joint_log_likelihood=-100.0,
            delta_log_likelihood=3.0,
            posterior_odds=20.08,
            sector_evidence=[sample_sector_evidence],
            sector_weights={15: 1.0},
            flip_rate=0.0,
            consistency_verdict="stable",
            hypotheses_considered=["tic:123456789"],
            computation_time_seconds=0.05,
            warnings=[],
            blobs={},
        )

        assert result.joint_best_source_id == "tic:123456789"
        assert result.verdict == "ON_TARGET"
        assert result.resolved_probability is None
        assert result.calibration_version is None

    def test_joint_result_all_verdicts(self, sample_sector_evidence: SectorEvidence) -> None:
        """JointInferenceResult accepts all valid verdict values."""
        for verdict in ["ON_TARGET", "OFF_TARGET", "AMBIGUOUS", "INVALID"]:
            result = JointInferenceResult(
                joint_best_source_id="test",
                verdict=verdict,
                resolved_probability=None,
                calibration_version=None,
                joint_log_likelihood=0.0,
                delta_log_likelihood=0.0,
                posterior_odds=None,
                sector_evidence=[sample_sector_evidence],
                sector_weights={15: 1.0},
                flip_rate=0.0,
                consistency_verdict="stable",
                hypotheses_considered=["test"],
                computation_time_seconds=0.0,
            )
            assert result.verdict == verdict

    def test_joint_result_all_consistency_verdicts(
        self, sample_sector_evidence: SectorEvidence
    ) -> None:
        """JointInferenceResult accepts all valid consistency_verdict values."""
        for cv in ["stable", "mixed", "flipping"]:
            result = JointInferenceResult(
                joint_best_source_id="test",
                verdict="AMBIGUOUS",
                resolved_probability=None,
                calibration_version=None,
                joint_log_likelihood=0.0,
                delta_log_likelihood=0.0,
                posterior_odds=None,
                sector_evidence=[sample_sector_evidence],
                sector_weights={15: 1.0},
                flip_rate=0.5,
                consistency_verdict=cv,
                hypotheses_considered=["test"],
                computation_time_seconds=0.0,
            )
            assert result.consistency_verdict == cv

    def test_joint_result_with_calibration(self, sample_sector_evidence: SectorEvidence) -> None:
        """JointInferenceResult can store calibration info (Phase 3.5)."""
        result = JointInferenceResult(
            joint_best_source_id="tic:123456789",
            verdict="ON_TARGET",
            resolved_probability=0.95,
            calibration_version="v3.5.0-discovery",
            joint_log_likelihood=-100.0,
            delta_log_likelihood=5.0,
            posterior_odds=148.41,
            sector_evidence=[sample_sector_evidence],
            sector_weights={15: 1.0},
            flip_rate=0.0,
            consistency_verdict="stable",
            hypotheses_considered=["tic:123456789"],
            computation_time_seconds=0.05,
        )

        assert result.resolved_probability == 0.95
        assert result.calibration_version == "v3.5.0-discovery"


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSectorEvidenceSerialization:
    """Tests for SectorEvidence serialization helpers."""

    def test_sector_evidence_to_dict(self, sample_sector_evidence: SectorEvidence) -> None:
        """sector_evidence_to_dict produces JSON-serializable output."""
        d = sector_evidence_to_dict(sample_sector_evidence)

        # Check JSON serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

        # Check key fields
        assert d["sector"] == 15
        assert d["tpf_fits_ref"] == "tpf_fits:123456789:15:spoc"
        assert len(d["hypotheses"]) == 2
        assert d["quality_weight"] == 1.0

    def test_sector_evidence_roundtrip(self, sample_sector_evidence: SectorEvidence) -> None:
        """sector_evidence round-trips through to_dict/from_dict."""
        d = sector_evidence_to_dict(sample_sector_evidence)
        recovered = sector_evidence_from_dict(d)

        assert recovered.sector == sample_sector_evidence.sector
        assert recovered.tpf_fits_ref == sample_sector_evidence.tpf_fits_ref
        assert len(recovered.hypotheses) == len(sample_sector_evidence.hypotheses)
        assert recovered.quality_weight == sample_sector_evidence.quality_weight
        assert recovered.downweight_reason == sample_sector_evidence.downweight_reason

    def test_sector_evidence_roundtrip_through_json(
        self, sample_sector_evidence: SectorEvidence
    ) -> None:
        """sector_evidence survives JSON serialization."""
        d = sector_evidence_to_dict(sample_sector_evidence)
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        recovered = sector_evidence_from_dict(d2)

        assert recovered.sector == sample_sector_evidence.sector
        assert recovered.tpf_fits_ref == sample_sector_evidence.tpf_fits_ref


class TestJointResultSerialization:
    """Tests for JointInferenceResult serialization helpers."""

    def test_joint_result_to_dict(self, sample_joint_result: JointInferenceResult) -> None:
        """joint_result_to_dict produces JSON-serializable output."""
        d = joint_result_to_dict(sample_joint_result)

        # Check JSON serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

        # Check key fields
        assert d["joint_best_source_id"] == "tic:123456789"
        assert d["verdict"] == "ON_TARGET"
        assert d["resolved_probability"] is None
        assert d["delta_log_likelihood"] == 5.0
        assert len(d["sector_evidence"]) == 2
        assert d["sector_weights"][15] == 1.0

    def test_joint_result_roundtrip(self, sample_joint_result: JointInferenceResult) -> None:
        """joint_result round-trips through to_dict/from_dict."""
        d = joint_result_to_dict(sample_joint_result)
        recovered = joint_result_from_dict(d)

        assert recovered.joint_best_source_id == sample_joint_result.joint_best_source_id
        assert recovered.verdict == sample_joint_result.verdict
        assert recovered.resolved_probability == sample_joint_result.resolved_probability
        assert recovered.delta_log_likelihood == sample_joint_result.delta_log_likelihood
        assert len(recovered.sector_evidence) == len(sample_joint_result.sector_evidence)
        assert recovered.flip_rate == sample_joint_result.flip_rate

    def test_joint_result_roundtrip_through_json(
        self, sample_joint_result: JointInferenceResult
    ) -> None:
        """joint_result survives JSON serialization."""
        d = joint_result_to_dict(sample_joint_result)
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        recovered = joint_result_from_dict(d2)

        assert recovered.joint_best_source_id == sample_joint_result.joint_best_source_id
        assert recovered.verdict == sample_joint_result.verdict


# =============================================================================
# Evidence Block Tests
# =============================================================================


class TestEvidenceBlock:
    """Tests for to_evidence_block function."""

    def test_evidence_block_structure(self, sample_joint_result: JointInferenceResult) -> None:
        """to_evidence_block produces expected structure."""
        block = to_evidence_block(sample_joint_result)

        assert block["source"] == "joint_multi_sector_localization"
        assert block["version"] == "3.2.0"
        assert block["verdict"] == "ON_TARGET"
        assert "key_metrics" in block
        assert "warnings" in block
        assert "details_ref" in block

    def test_evidence_block_high_confidence(self, sample_sector_evidence: SectorEvidence) -> None:
        """Evidence block does not add derived policy fields."""
        result = JointInferenceResult(
            joint_best_source_id="tic:123",
            verdict="ON_TARGET",
            resolved_probability=None,
            calibration_version=None,
            joint_log_likelihood=-100.0,
            delta_log_likelihood=10.0,  # >= 5.0
            posterior_odds=None,
            sector_evidence=[sample_sector_evidence],
            sector_weights={15: 1.0},
            flip_rate=0.0,
            consistency_verdict="stable",
            hypotheses_considered=["tic:123"],
            computation_time_seconds=0.0,
        )

        block = to_evidence_block(result)
        assert "confidence_level" not in block
        assert "flags" not in block

    def test_evidence_block_low_confidence(self, sample_sector_evidence: SectorEvidence) -> None:
        """Evidence block preserves raw verdict and metrics regardless of thresholds."""
        result = JointInferenceResult(
            joint_best_source_id="tic:123",
            verdict="AMBIGUOUS",
            resolved_probability=None,
            calibration_version=None,
            joint_log_likelihood=-100.0,
            delta_log_likelihood=1.0,  # < 2.0
            posterior_odds=None,
            sector_evidence=[sample_sector_evidence],
            sector_weights={15: 1.0},
            flip_rate=0.0,
            consistency_verdict="stable",
            hypotheses_considered=["tic:123"],
            computation_time_seconds=0.0,
        )

        block = to_evidence_block(result)
        assert block["verdict"] == "AMBIGUOUS"
        assert block["key_metrics"]["delta_log_likelihood"] == 1.0

    def test_evidence_block_flags_flipping(self, sample_sector_evidence: SectorEvidence) -> None:
        """Evidence block does not add derived flags."""
        result = JointInferenceResult(
            joint_best_source_id="tic:123",
            verdict="AMBIGUOUS",
            resolved_probability=None,
            calibration_version=None,
            joint_log_likelihood=-100.0,
            delta_log_likelihood=1.0,
            posterior_odds=None,
            sector_evidence=[sample_sector_evidence],
            sector_weights={15: 1.0},
            flip_rate=0.6,
            consistency_verdict="flipping",
            hypotheses_considered=["tic:123"],
            computation_time_seconds=0.0,
        )

        block = to_evidence_block(result)
        assert "flags" not in block
        assert block["verdict"] == "AMBIGUOUS"

    def test_evidence_block_flags_off_target(self, sample_sector_evidence: SectorEvidence) -> None:
        """OFF_TARGET verdict is preserved without adding derived flags."""
        result = JointInferenceResult(
            joint_best_source_id="gaia_dr3:999",
            verdict="OFF_TARGET",
            resolved_probability=None,
            calibration_version=None,
            joint_log_likelihood=-100.0,
            delta_log_likelihood=5.0,
            posterior_odds=None,
            sector_evidence=[sample_sector_evidence],
            sector_weights={15: 1.0},
            flip_rate=0.0,
            consistency_verdict="stable",
            hypotheses_considered=["tic:123", "gaia_dr3:999"],
            computation_time_seconds=0.0,
        )

        block = to_evidence_block(result)
        assert "flags" not in block
        assert block["verdict"] == "OFF_TARGET"

    def test_evidence_block_key_metrics(self, sample_joint_result: JointInferenceResult) -> None:
        """Evidence block includes expected key metrics."""
        block = to_evidence_block(sample_joint_result)

        km = block["key_metrics"]
        assert km["joint_best_source_id"] == "tic:123456789"
        assert km["delta_log_likelihood"] == 5.0
        assert km["flip_rate"] == 0.0
        assert km["n_sectors"] == 2


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateJointResultFromSectors:
    """Tests for create_joint_result_from_sectors factory."""

    def test_factory_empty_sectors(self) -> None:
        """Empty sector list produces INVALID result."""
        result = create_joint_result_from_sectors([], ["tic:123"])

        assert result.verdict == "INVALID"
        assert result.joint_best_source_id == ""
        assert len(result.sector_evidence) == 0
        assert "No sector evidence provided" in result.warnings

    def test_factory_single_sector(self, sample_sector_evidence: SectorEvidence) -> None:
        """Single sector produces valid result."""
        result = create_joint_result_from_sectors(
            [sample_sector_evidence],
            ["tic:123456789", "gaia_dr3:987654321"],
        )

        assert result.joint_best_source_id == "tic:123456789"
        assert result.flip_rate == 0.0
        assert result.consistency_verdict == "stable"
        assert len(result.sector_evidence) == 1

    def test_factory_single_sector_no_hypotheses(self) -> None:
        """Single sector with no hypotheses produces INVALID."""
        se = SectorEvidence(
            sector=1,
            tpf_fits_ref="tpf_fits:111:1:spoc",
            hypotheses=[],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )

        result = create_joint_result_from_sectors([se], [])

        assert result.verdict == "INVALID"
        assert "Single sector with no hypotheses" in result.warnings

    def test_factory_multi_sector_stable(self, sample_hypothesis_target: HypothesisScore) -> None:
        """Multiple agreeing sectors produce stable result."""
        se1 = SectorEvidence(
            sector=15,
            tpf_fits_ref="tpf_fits:123:15:spoc",
            hypotheses=[sample_hypothesis_target],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )
        se2 = SectorEvidence(
            sector=42,
            tpf_fits_ref="tpf_fits:123:42:spoc",
            hypotheses=[sample_hypothesis_target],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )

        result = create_joint_result_from_sectors([se1, se2], ["tic:123456789"])

        assert result.joint_best_source_id == "tic:123456789"
        assert result.flip_rate == 0.0
        assert result.consistency_verdict == "stable"

    def test_factory_multi_sector_flipping(self) -> None:
        """Disagreeing sectors produce flipping result."""
        hyp_a = HypothesisScore(
            source_id="source_a",
            source_name="Source A",
            fit_loss=100.0,
            delta_loss=0.0,
            rank=1,
            fit_amplitude=-0.001,
            fit_background=0.0,
        )
        hyp_b = HypothesisScore(
            source_id="source_b",
            source_name="Source B",
            fit_loss=100.0,
            delta_loss=0.0,
            rank=1,
            fit_amplitude=-0.001,
            fit_background=0.0,
        )

        se1 = SectorEvidence(
            sector=15,
            tpf_fits_ref="tpf_fits:123:15:spoc",
            hypotheses=[hyp_a],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )
        se2 = SectorEvidence(
            sector=42,
            tpf_fits_ref="tpf_fits:123:42:spoc",
            hypotheses=[hyp_b],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )

        result = create_joint_result_from_sectors([se1, se2], ["source_a", "source_b"])

        assert result.flip_rate == 0.5
        assert result.consistency_verdict == "flipping"
        assert result.verdict == "AMBIGUOUS"

    def test_factory_quality_weighted_voting(self) -> None:
        """Higher quality_weight sectors have more influence."""
        hyp_a = HypothesisScore(
            source_id="source_a",
            source_name="Source A",
            fit_loss=100.0,
            delta_loss=0.0,
            rank=1,
            fit_amplitude=-0.001,
            fit_background=0.0,
        )
        hyp_b = HypothesisScore(
            source_id="source_b",
            source_name="Source B",
            fit_loss=100.0,
            delta_loss=0.0,
            rank=1,
            fit_amplitude=-0.001,
            fit_background=0.0,
        )

        # Sector 15 has low weight, prefers source_a
        se1 = SectorEvidence(
            sector=15,
            tpf_fits_ref="tpf_fits:123:15:spoc",
            hypotheses=[hyp_a],
            residual_rms=0.001,
            quality_weight=0.2,
            downweight_reason="high_residual",
        )
        # Sector 42 has high weight, prefers source_b
        se2 = SectorEvidence(
            sector=42,
            tpf_fits_ref="tpf_fits:123:42:spoc",
            hypotheses=[hyp_b],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )

        result = create_joint_result_from_sectors([se1, se2], ["source_a", "source_b"])

        # source_b should win due to higher weight
        assert result.joint_best_source_id == "source_b"

    def test_factory_sector_weights_populated(
        self,
        sample_sector_evidence: SectorEvidence,
        sample_sector_evidence_downweighted: SectorEvidence,
    ) -> None:
        """Factory populates sector_weights correctly."""
        result = create_joint_result_from_sectors(
            [sample_sector_evidence, sample_sector_evidence_downweighted],
            ["tic:123456789"],
        )

        assert result.sector_weights[15] == 1.0
        assert result.sector_weights[42] == 0.5

    def test_factory_hypotheses_considered(self, sample_sector_evidence: SectorEvidence) -> None:
        """Factory records hypotheses_considered."""
        hypotheses = ["tic:123", "gaia_dr3:456", "gaia_dr3:789"]
        result = create_joint_result_from_sectors([sample_sector_evidence], hypotheses)

        assert result.hypotheses_considered == hypotheses

    def test_factory_computation_time(self, sample_sector_evidence: SectorEvidence) -> None:
        """Factory records computation time."""
        result = create_joint_result_from_sectors(
            [sample_sector_evidence],
            ["tic:123456789"],
        )

        assert result.computation_time_seconds >= 0.0
        assert result.computation_time_seconds < 1.0  # Should be fast

    def test_factory_verdict_on_target(self) -> None:
        """Factory produces ON_TARGET for resolved target hypothesis."""
        hyp_target = HypothesisScore(
            source_id="tic:target:123",
            source_name="target",
            fit_loss=100.0,
            delta_loss=0.0,
            rank=1,
            fit_amplitude=-0.001,
            fit_background=0.0,
        )
        hyp_neighbor = HypothesisScore(
            source_id="gaia:neighbor",
            source_name="neighbor",
            fit_loss=200.0,
            delta_loss=100.0,
            rank=2,
            fit_amplitude=-0.0005,
            fit_background=0.0,
        )

        se = SectorEvidence(
            sector=15,
            tpf_fits_ref="tpf_fits:123:15:spoc",
            hypotheses=[hyp_target, hyp_neighbor],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )

        result = create_joint_result_from_sectors([se], ["tic:target:123", "gaia:neighbor"])

        assert result.verdict == "ON_TARGET"
        assert "target" in result.joint_best_source_id.lower()

    def test_factory_verdict_off_target(self) -> None:
        """Factory produces OFF_TARGET for resolved non-target hypothesis."""
        hyp_neighbor = HypothesisScore(
            source_id="gaia:neighbor",
            source_name="neighbor",
            fit_loss=100.0,
            delta_loss=0.0,
            rank=1,
            fit_amplitude=-0.001,
            fit_background=0.0,
        )
        hyp_target = HypothesisScore(
            source_id="tic:target:123",
            source_name="target",
            fit_loss=200.0,
            delta_loss=100.0,
            rank=2,
            fit_amplitude=-0.0005,
            fit_background=0.0,
        )

        se = SectorEvidence(
            sector=15,
            tpf_fits_ref="tpf_fits:123:15:spoc",
            hypotheses=[hyp_neighbor, hyp_target],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )

        result = create_joint_result_from_sectors([se], ["gaia:neighbor", "tic:target:123"])

        assert result.verdict == "OFF_TARGET"
        assert result.joint_best_source_id == "gaia:neighbor"

    def test_factory_low_weight_warnings(self) -> None:
        """Factory warns about low quality_weight sectors."""
        hyp = HypothesisScore(
            source_id="tic:123",
            source_name="target",
            fit_loss=100.0,
            delta_loss=0.0,
            rank=1,
            fit_amplitude=-0.001,
            fit_background=0.0,
        )

        # Low weight warning only triggers for multi-sector case
        se1 = SectorEvidence(
            sector=15,
            tpf_fits_ref="tpf_fits:123:15:spoc",
            hypotheses=[hyp],
            residual_rms=0.001,
            quality_weight=0.3,
            downweight_reason="high_residual",
        )
        se2 = SectorEvidence(
            sector=42,
            tpf_fits_ref="tpf_fits:123:42:spoc",
            hypotheses=[hyp],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )

        result = create_joint_result_from_sectors([se1, se2], ["tic:123"])

        assert any("Low quality weight" in w for w in result.warnings)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_sector_evidence_empty_hypotheses(self) -> None:
        """SectorEvidence accepts empty hypotheses list."""
        se = SectorEvidence(
            sector=1,
            tpf_fits_ref="tpf_fits:111:1:spoc",
            hypotheses=[],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )

        assert len(se.hypotheses) == 0

        d = sector_evidence_to_dict(se)
        recovered = sector_evidence_from_dict(d)
        assert len(recovered.hypotheses) == 0

    def test_joint_result_empty_sector_weights(
        self, sample_sector_evidence: SectorEvidence
    ) -> None:
        """JointInferenceResult can have empty sector_weights (edge case)."""
        result = JointInferenceResult(
            joint_best_source_id="tic:123",
            verdict="INVALID",
            resolved_probability=None,
            calibration_version=None,
            joint_log_likelihood=0.0,
            delta_log_likelihood=0.0,
            posterior_odds=None,
            sector_evidence=[],
            sector_weights={},
            flip_rate=0.0,
            consistency_verdict="stable",
            hypotheses_considered=[],
            computation_time_seconds=0.0,
        )

        d = joint_result_to_dict(result)
        recovered = joint_result_from_dict(d)
        assert recovered.sector_weights == {}

    def test_hypothesis_with_none_values(self) -> None:
        """HypothesisScore with None optional fields serializes correctly."""
        hyp = HypothesisScore(
            source_id="test",
            source_name="Test",
            fit_loss=float("inf"),
            delta_loss=0.0,
            rank=1,
            fit_amplitude=None,
            fit_background=None,
        )

        se = SectorEvidence(
            sector=1,
            tpf_fits_ref="tpf_fits:111:1:spoc",
            hypotheses=[hyp],
            residual_rms=0.0,
            quality_weight=1.0,
            downweight_reason=None,
        )

        d = sector_evidence_to_dict(se)
        recovered = sector_evidence_from_dict(d)

        assert recovered.hypotheses[0]["fit_amplitude"] is None
        assert recovered.hypotheses[0]["fit_background"] is None

    def test_factory_all_sectors_disagree(self) -> None:
        """Factory handles case where all sectors pick different sources."""
        hypotheses_list = []
        sector_evs = []

        for i, sector in enumerate([15, 42, 67]):
            hyp = HypothesisScore(
                source_id=f"source_{i}",
                source_name=f"Source {i}",
                fit_loss=100.0,
                delta_loss=0.0,
                rank=1,
                fit_amplitude=-0.001,
                fit_background=0.0,
            )
            se = SectorEvidence(
                sector=sector,
                tpf_fits_ref=f"tpf_fits:123:{sector}:spoc",
                hypotheses=[hyp],
                residual_rms=0.0001,
                quality_weight=1.0,
                downweight_reason=None,
            )
            hypotheses_list.append(f"source_{i}")
            sector_evs.append(se)

        result = create_joint_result_from_sectors(sector_evs, hypotheses_list)

        # One source will win by vote count (all equal, so first in sorted order)
        # But flip rate should be high
        assert result.flip_rate > 0.5
        assert result.consistency_verdict == "flipping"
        assert result.verdict == "AMBIGUOUS"


# =============================================================================
# Joint Inference Mode Tests (Phase 3.2)
# =============================================================================


class TestJointInferenceMode:
    """Tests for joint inference mode (Phase 3.2)."""

    def test_joint_mode_returns_valid_result(self, sample_sector_evidence: SectorEvidence) -> None:
        """Joint mode returns a valid JointInferenceResult."""
        se2 = SectorEvidence(
            sector=42,
            tpf_fits_ref="tpf_fits:123456789:42:spoc",
            hypotheses=[sample_sector_evidence.hypotheses[0]],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )

        result = create_joint_result_from_sectors(
            [sample_sector_evidence, se2],
            ["tic:123456789", "gaia_dr3:987654321"],
            inference_mode="joint",
        )

        assert result.joint_best_source_id != ""
        assert result.verdict in ["ON_TARGET", "OFF_TARGET", "AMBIGUOUS", "INVALID"]
        assert result.joint_log_likelihood != 0.0

    def test_joint_mode_uses_log_likelihood_not_votes(
        self, sample_hypothesis_target: HypothesisScore
    ) -> None:
        """Joint mode uses log-likelihood, not weighted votes."""
        # Create two sectors with the same hypothesis
        se1 = SectorEvidence(
            sector=15,
            tpf_fits_ref="tpf_fits:123:15:spoc",
            hypotheses=[sample_hypothesis_target],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )
        se2 = SectorEvidence(
            sector=42,
            tpf_fits_ref="tpf_fits:123:42:spoc",
            hypotheses=[sample_hypothesis_target],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )

        result = create_joint_result_from_sectors(
            [se1, se2],
            ["tic:123456789"],
            inference_mode="joint",
        )

        # Joint log-likelihood should be sum of sector likelihoods
        # (each sector contributes -fit_loss)
        assert result.joint_log_likelihood < 0

    def test_joint_mode_downweights_bad_sectors(self) -> None:
        """Joint mode can downweight sectors with high residuals."""
        hyp = HypothesisScore(
            source_id="tic:123",
            source_name="target",
            fit_loss=100.0,
            delta_loss=0.0,
            rank=1,
            fit_amplitude=-0.001,
            fit_background=0.0,
        )

        # High quality sector
        se1 = SectorEvidence(
            sector=15,
            tpf_fits_ref="tpf_fits:123:15:spoc",
            hypotheses=[hyp],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )
        # Low quality sector (pre-downweighted)
        se2 = SectorEvidence(
            sector=42,
            tpf_fits_ref="tpf_fits:123:42:spoc",
            hypotheses=[hyp],
            residual_rms=0.001,
            quality_weight=0.3,
            downweight_reason="high_residual",
        )

        result = create_joint_result_from_sectors(
            [se1, se2],
            ["tic:123"],
            inference_mode="joint",
            downweight_high_residual=True,
        )

        # Check that sector weights are populated
        assert result.sector_weights[15] >= result.sector_weights[42]

    def test_joint_mode_backward_compatible_with_vote(
        self, sample_sector_evidence: SectorEvidence
    ) -> None:
        """Vote mode (default) still works for backward compatibility."""
        se2 = SectorEvidence(
            sector=42,
            tpf_fits_ref="tpf_fits:123456789:42:spoc",
            hypotheses=[sample_sector_evidence.hypotheses[0]],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )

        # Default is vote mode
        result_vote = create_joint_result_from_sectors(
            [sample_sector_evidence, se2],
            ["tic:123456789", "gaia_dr3:987654321"],
        )

        # Explicit vote mode
        result_explicit = create_joint_result_from_sectors(
            [sample_sector_evidence, se2],
            ["tic:123456789", "gaia_dr3:987654321"],
            inference_mode="vote",
        )

        # Both should produce similar results
        assert result_vote.joint_best_source_id == result_explicit.joint_best_source_id
        assert result_vote.verdict == result_explicit.verdict

    def test_joint_mode_computes_delta_log_likelihood(self) -> None:
        """Joint mode computes delta_log_likelihood correctly."""
        hyp_a = HypothesisScore(
            source_id="tic:target:123",
            source_name="target",
            fit_loss=100.0,
            delta_loss=0.0,
            rank=1,
            fit_amplitude=-0.001,
            fit_background=0.0,
        )
        hyp_b = HypothesisScore(
            source_id="gaia:neighbor",
            source_name="neighbor",
            fit_loss=150.0,
            delta_loss=50.0,
            rank=2,
            fit_amplitude=-0.0008,
            fit_background=0.0,
        )

        se1 = SectorEvidence(
            sector=15,
            tpf_fits_ref="tpf_fits:123:15:spoc",
            hypotheses=[hyp_a, hyp_b],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )
        se2 = SectorEvidence(
            sector=42,
            tpf_fits_ref="tpf_fits:123:42:spoc",
            hypotheses=[hyp_a, hyp_b],
            residual_rms=0.0001,
            quality_weight=1.0,
            downweight_reason=None,
        )

        result = create_joint_result_from_sectors(
            [se1, se2],
            ["tic:target:123", "gaia:neighbor"],
            inference_mode="joint",
        )

        # Delta should be positive since target has lower fit_loss
        assert result.delta_log_likelihood > 0

    def test_single_sector_uses_same_logic_for_both_modes(
        self, sample_sector_evidence: SectorEvidence
    ) -> None:
        """Single sector case uses same logic regardless of mode."""
        result_vote = create_joint_result_from_sectors(
            [sample_sector_evidence],
            ["tic:123456789", "gaia_dr3:987654321"],
            inference_mode="vote",
        )

        result_joint = create_joint_result_from_sectors(
            [sample_sector_evidence],
            ["tic:123456789", "gaia_dr3:987654321"],
            inference_mode="joint",
        )

        # Single sector should have identical results
        assert result_vote.joint_best_source_id == result_joint.joint_best_source_id
        assert result_vote.verdict == result_joint.verdict
