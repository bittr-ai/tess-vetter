"""End-to-end tests for the vetting pipeline.

This module tests the full vetting pipeline with:
- VettingPipeline class and configuration
- CheckRegistry and default check registration
- vet_candidate convenience function
- Contract tests for API stability
- JSON serialization requirements

Novelty: standard (testing existing pipeline infrastructure)
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from bittr_tess_vetter.api import (
    Candidate,
    Ephemeris,
    LightCurve,
    VettingPipeline,
    list_checks,
    vet_candidate,
)
from bittr_tess_vetter.validation.register_defaults import register_all_defaults
from bittr_tess_vetter.validation.registry import CheckRegistry
from bittr_tess_vetter.validation.result_schema import CheckResult, VettingBundleResult


@pytest.fixture
def synthetic_lc() -> LightCurve:
    """Create a synthetic light curve with a transit-like signal."""
    np.random.seed(42)
    time = np.linspace(0, 27, 2000)
    flux = np.ones_like(time)

    # Add simple transit signal
    period = 3.5
    epoch = 1.0
    duration = 0.1  # days
    depth = 0.01  # 1%

    phase = ((time - epoch) % period) / period
    in_transit = (phase < duration / period) | (phase > 1 - duration / period)
    flux[in_transit] -= depth

    # Add noise
    flux += np.random.normal(0, 0.001, len(flux))
    flux_err = np.ones_like(flux) * 0.001

    return LightCurve(time=time, flux=flux, flux_err=flux_err)


@pytest.fixture
def synthetic_candidate() -> Candidate:
    """Create a synthetic candidate matching the light curve."""
    return Candidate(
        ephemeris=Ephemeris(
            period_days=3.5,
            t0_btjd=1.0,
            duration_hours=2.4,  # ~0.1 days
        ),
    )


class TestVettingPipelineE2E:
    """End-to-end tests for VettingPipeline."""

    def test_pipeline_with_default_registry(
        self, synthetic_lc: LightCurve, synthetic_candidate: Candidate
    ) -> None:
        """Test pipeline runs with default check registry."""
        registry = CheckRegistry()
        register_all_defaults(registry)

        pipeline = VettingPipeline(registry=registry)
        result = pipeline.run(
            synthetic_lc.to_internal(),
            _candidate_to_internal(synthetic_candidate),
            network=False,
        )

        assert isinstance(result, VettingBundleResult)
        assert len(result.results) > 0
        assert all(isinstance(r, CheckResult) for r in result.results)

    def test_all_results_have_valid_status(
        self, synthetic_lc: LightCurve, synthetic_candidate: Candidate
    ) -> None:
        """Every check result must have a valid status."""
        registry = CheckRegistry()
        register_all_defaults(registry)

        pipeline = VettingPipeline(registry=registry)
        result = pipeline.run(
            synthetic_lc.to_internal(),
            _candidate_to_internal(synthetic_candidate),
            network=False,
        )

        valid_statuses = {"ok", "skipped", "error"}
        for check_result in result.results:
            assert check_result.status in valid_statuses, (
                f"Check {check_result.id} has invalid status: {check_result.status}"
            )

    def test_all_metrics_json_serializable(
        self, synthetic_lc: LightCurve, synthetic_candidate: Candidate
    ) -> None:
        """All metrics must be JSON-serializable."""
        registry = CheckRegistry()
        register_all_defaults(registry)

        pipeline = VettingPipeline(registry=registry)
        result = pipeline.run(
            synthetic_lc.to_internal(),
            _candidate_to_internal(synthetic_candidate),
            network=False,
        )

        for check_result in result.results:
            # Should not raise
            try:
                json.dumps(check_result.metrics)
            except (TypeError, ValueError) as e:
                pytest.fail(
                    f"Check {check_result.id} has non-JSON-serializable metrics: {e}"
                )

    def test_bundle_result_json_serializable(
        self, synthetic_lc: LightCurve, synthetic_candidate: Candidate
    ) -> None:
        """Entire bundle result must be JSON-serializable."""
        registry = CheckRegistry()
        register_all_defaults(registry)

        pipeline = VettingPipeline(registry=registry)
        result = pipeline.run(
            synthetic_lc.to_internal(),
            _candidate_to_internal(synthetic_candidate),
            network=False,
        )

        # Should not raise
        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        assert "results" in parsed
        assert "provenance" in parsed

    def test_no_deferred_status(
        self, synthetic_lc: LightCurve, synthetic_candidate: Candidate
    ) -> None:
        """No check should return 'deferred' - use 'skipped' with reason."""
        registry = CheckRegistry()
        register_all_defaults(registry)

        pipeline = VettingPipeline(registry=registry)
        result = pipeline.run(
            synthetic_lc.to_internal(),
            _candidate_to_internal(synthetic_candidate),
            network=False,
        )

        for check_result in result.results:
            # Status should never be "deferred"
            assert check_result.status != "deferred", (
                f"Check {check_result.id} returned deprecated 'deferred' status"
            )
            # Flags should not contain "deferred" (case-insensitive)
            for flag in check_result.flags:
                assert "deferred" not in flag.lower(), (
                    f"Check {check_result.id} has 'deferred' in flags: {flag}"
                )


class TestVetCandidateE2E:
    """End-to-end tests for vet_candidate convenience function."""

    def test_vet_candidate_returns_bundle(
        self, synthetic_lc: LightCurve, synthetic_candidate: Candidate
    ) -> None:
        """vet_candidate should return VettingBundleResult."""
        result = vet_candidate(synthetic_lc, synthetic_candidate, network=False)

        assert isinstance(result, VettingBundleResult)
        assert len(result.results) > 0

    def test_vet_candidate_with_specific_checks(
        self, synthetic_lc: LightCurve, synthetic_candidate: Candidate
    ) -> None:
        """vet_candidate should run only specified checks."""
        result = vet_candidate(
            synthetic_lc,
            synthetic_candidate,
            checks=["V01", "V02"],
            network=False,
        )

        assert len(result.results) == 2
        check_ids = {r.id for r in result.results}
        assert check_ids == {"V01", "V02"}

    def test_inputs_summary_populated(
        self, synthetic_lc: LightCurve, synthetic_candidate: Candidate
    ) -> None:
        """inputs_summary should reflect what was provided."""
        result = vet_candidate(
            synthetic_lc,
            synthetic_candidate,
            network=True,
            tpf=None,
        )

        assert result.inputs_summary["network"] is True
        assert result.inputs_summary["has_tpf"] is False


class TestListChecks:
    """Tests for list_checks introspection."""

    def test_list_checks_returns_all_defaults(self) -> None:
        """list_checks should return all registered V01-V07 checks."""
        registry = CheckRegistry()
        register_all_defaults(registry)

        checks = list_checks(registry)

        check_ids = {c["id"] for c in checks}
        # V01-V05 LC-only, V06-V07 catalog (7 checks total)
        expected_ids = {f"V{i:02d}" for i in range(1, 8)}
        assert check_ids == expected_ids

    def test_list_checks_has_required_fields(self) -> None:
        """Each check info should have required fields."""
        registry = CheckRegistry()
        register_all_defaults(registry)

        checks = list_checks(registry)

        required_fields = {"id", "name", "tier", "requirements", "citations"}
        for check in checks:
            assert required_fields <= set(check.keys()), (
                f"Check {check.get('id')} missing fields"
            )


class TestContractTests:
    """Contract tests ensuring API stability."""

    def test_golden_path_imports(self) -> None:
        """All golden path imports should work."""
        from bittr_tess_vetter.api import (  # noqa: F401
            Candidate,
            # Registry
            CheckRegistry,
            CheckRequirements,
            CheckTier,
            Ephemeris,
            # Types
            LightCurve,
            PipelineConfig,
            VettingCheck,
            VettingPipeline,
            describe_checks,
            # Introspection
            list_checks,
            # Entry points
            vet_candidate,
        )

        # All should be importable
        assert LightCurve is not None
        assert VettingPipeline is not None
        assert list_checks is not None

    def test_primitives_imports(self) -> None:
        """Primitives module imports should work."""
        from bittr_tess_vetter.api.primitives import (  # noqa: F401
            detrend,
            fold,
        )

        assert fold is not None
        assert detrend is not None

    def test_check_result_schema_stable(self) -> None:
        """CheckResult should have stable fields."""
        result = CheckResult(
            id="V01",
            name="Test",
            status="ok",
            metrics={"value": 1.0},
        )

        # These fields must exist
        assert hasattr(result, "id")
        assert hasattr(result, "name")
        assert hasattr(result, "status")
        assert hasattr(result, "confidence")
        assert hasattr(result, "metrics")
        assert hasattr(result, "flags")
        assert hasattr(result, "notes")
        assert hasattr(result, "provenance")

    def test_vetting_bundle_result_schema_stable(self) -> None:
        """VettingBundleResult should have stable fields."""
        result = VettingBundleResult(
            results=[],
            warnings=[],
            provenance={"pipeline_version": "0.1.0"},
            inputs_summary={"network": False},
        )

        # These fields must exist
        assert hasattr(result, "results")
        assert hasattr(result, "warnings")
        assert hasattr(result, "provenance")
        assert hasattr(result, "inputs_summary")

        # Methods must exist
        assert callable(getattr(result, "get_result", None))
        assert callable(getattr(result, "model_dump_json", None))


class TestPipelineDescribe:
    """Tests for pipeline describe functionality."""

    def test_describe_shows_will_run_and_will_skip(self) -> None:
        """Pipeline.describe() should categorize checks by what will run."""
        registry = CheckRegistry()
        register_all_defaults(registry)

        pipeline = VettingPipeline(registry=registry)
        description = pipeline.describe(network=False)

        assert "will_run" in description
        assert "will_skip" in description
        assert "total_checks" in description

        # LC-only checks should run
        will_run_ids = {c["id"] for c in description["will_run"]}
        assert {"V01", "V02", "V03", "V04", "V05"} <= will_run_ids

        # Catalog checks should be skipped without network
        will_skip_ids = {c["id"] for c in description["will_skip"]}
        assert {"V06", "V07"} <= will_skip_ids

    def test_describe_with_network_enabled(self) -> None:
        """With network enabled, catalog checks should still need metadata."""
        registry = CheckRegistry()
        register_all_defaults(registry)

        pipeline = VettingPipeline(registry=registry)
        description = pipeline.describe(network=True)

        # Without RA/Dec and TIC ID, catalog checks are still skipped
        will_skip_ids = {c["id"] for c in description["will_skip"]}
        # V06 needs RA/Dec, V07 needs TIC ID
        assert {"V06", "V07"} <= will_skip_ids


def _candidate_to_internal(candidate: Candidate) -> Any:
    """Convert public Candidate to internal TransitCandidate."""
    from bittr_tess_vetter.domain.detection import TransitCandidate

    depth = candidate.depth if candidate.depth is not None else 0.001
    return TransitCandidate(
        period=candidate.ephemeris.period_days,
        t0=candidate.ephemeris.t0_btjd,
        duration_hours=candidate.ephemeris.duration_hours,
        depth=depth,
        snr=0.0,
    )
