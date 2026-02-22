"""Tests for vetting pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from tess_vetter.api.pipeline import (
    DESCRIBE_CHECKS_INPUT_SCHEMA,
    DESCRIBE_CHECKS_OUTPUT_SCHEMA,
    LIST_CHECKS_INPUT_SCHEMA,
    LIST_CHECKS_OUTPUT_SCHEMA,
    PIPELINE_DESCRIBE_INPUT_SCHEMA,
    PIPELINE_DESCRIBE_OUTPUT_SCHEMA,
    VettingPipeline,
    describe_checks,
    list_checks,
)
from tess_vetter.validation.registry import (
    CheckConfig,
    CheckInputs,
    CheckRegistry,
    CheckRequirements,
    CheckTier,
)
from tess_vetter.validation.result_schema import CheckResult, ok_result


class MockCheck:
    """Mock check for testing pipeline."""

    def __init__(
        self,
        id: str,
        name: str,
        tier: CheckTier = CheckTier.LC_ONLY,
        requirements: CheckRequirements | None = None,
        should_raise: bool = False,
    ) -> None:
        self._id = id
        self._name = name
        self._tier = tier
        self._requirements = requirements or CheckRequirements()
        self._should_raise = should_raise

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def tier(self) -> CheckTier:
        return self._tier

    @property
    def requirements(self) -> CheckRequirements:
        return self._requirements

    @property
    def citations(self) -> list[str]:
        return ["Test Citation"]

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        if self._should_raise:
            raise ValueError("Mock error")
        return ok_result(self.id, self.name, metrics={"ran": True})


@pytest.fixture
def mock_lc() -> Any:
    """Create a mock light curve."""
    lc = MagicMock()
    lc.time = np.linspace(0, 27, 100)
    lc.flux = np.ones(100)
    return lc


@pytest.fixture
def mock_candidate() -> Any:
    """Create a mock candidate."""
    candidate = MagicMock()
    candidate.ephemeris.epoch = 1.0
    candidate.ephemeris.period = 3.5
    return candidate


class TestVettingPipeline:
    """Tests for VettingPipeline."""

    def test_run_empty_registry(self, mock_lc: Any, mock_candidate: Any) -> None:
        registry = CheckRegistry()
        pipeline = VettingPipeline(registry=registry)

        result = pipeline.run(mock_lc, mock_candidate)

        assert len(result.results) == 0
        assert result.inputs_summary["network"] is False

    def test_run_single_check(self, mock_lc: Any, mock_candidate: Any) -> None:
        registry = CheckRegistry()
        registry.register(MockCheck("V01", "Test Check"))

        pipeline = VettingPipeline(registry=registry)
        result = pipeline.run(mock_lc, mock_candidate)

        assert len(result.results) == 1
        assert result.results[0].id == "V01"
        assert result.results[0].status == "ok"

    def test_run_skips_when_requirements_not_met(self, mock_lc: Any, mock_candidate: Any) -> None:
        registry = CheckRegistry()
        registry.register(
            MockCheck(
                "V08",
                "Pixel Check",
                tier=CheckTier.PIXEL,
                requirements=CheckRequirements(needs_tpf=True),
            )
        )

        pipeline = VettingPipeline(registry=registry)
        result = pipeline.run(mock_lc, mock_candidate, tpf=None)

        assert len(result.results) == 1
        assert result.results[0].status == "skipped"
        assert "SKIPPED:NO_TPF" in result.results[0].flags

    def test_run_skips_when_optional_dep_missing(self, mock_lc: Any, mock_candidate: Any) -> None:
        registry = CheckRegistry()
        registry.register(
            MockCheck(
                "V11",
                "OptionalDepCheck",
                tier=CheckTier.AUX,
                requirements=CheckRequirements(optional_deps=("definitely_not_installed_foo",)),
            )
        )

        pipeline = VettingPipeline(registry=registry)
        result = pipeline.run(mock_lc, mock_candidate)

        assert len(result.results) == 1
        assert result.results[0].status == "skipped"
        assert "SKIPPED:EXTRA_MISSING:definitely_not_installed_foo" in result.results[0].flags
        assert result.results[0].notes

    def test_run_with_tpf(self, mock_lc: Any, mock_candidate: Any) -> None:
        registry = CheckRegistry()
        registry.register(
            MockCheck(
                "V08",
                "Pixel Check",
                requirements=CheckRequirements(needs_tpf=True),
            )
        )

        pipeline = VettingPipeline(registry=registry)
        result = pipeline.run(mock_lc, mock_candidate, tpf=MagicMock())

        assert result.results[0].status == "ok"

    def test_run_handles_check_error(self, mock_lc: Any, mock_candidate: Any) -> None:
        registry = CheckRegistry()
        registry.register(MockCheck("V01", "Failing Check", should_raise=True))

        pipeline = VettingPipeline(registry=registry)
        result = pipeline.run(mock_lc, mock_candidate)

        assert len(result.results) == 1
        assert result.results[0].status == "error"
        assert "ERROR:ValueError" in result.results[0].flags

    def test_run_specific_checks(self, mock_lc: Any, mock_candidate: Any) -> None:
        registry = CheckRegistry()
        registry.register(MockCheck("V01", "Check 1"))
        registry.register(MockCheck("V02", "Check 2"))
        registry.register(MockCheck("V03", "Check 3"))

        pipeline = VettingPipeline(checks=["V01", "V03"], registry=registry)
        result = pipeline.run(mock_lc, mock_candidate)

        assert len(result.results) == 2
        assert result.results[0].id == "V01"
        assert result.results[1].id == "V03"

    def test_provenance(self, mock_lc: Any, mock_candidate: Any) -> None:
        registry = CheckRegistry()
        registry.register(MockCheck("V01", "Check"))

        pipeline = VettingPipeline(registry=registry)
        result = pipeline.run(mock_lc, mock_candidate)

        assert "duration_ms" in result.provenance
        assert result.provenance["checks_run"] == 1

    def test_inputs_summary(self, mock_lc: Any, mock_candidate: Any) -> None:
        registry = CheckRegistry()
        pipeline = VettingPipeline(registry=registry)

        result = pipeline.run(
            mock_lc,
            mock_candidate,
            network=True,
            tpf=MagicMock(),
            ra_deg=180.0,
            dec_deg=-45.0,
        )

        assert result.inputs_summary["network"] is True
        assert result.inputs_summary["has_tpf"] is True
        assert result.inputs_summary["has_coordinates"] is True


class TestDescribe:
    """Tests for pipeline describe functionality."""

    def test_describe(self) -> None:
        registry = CheckRegistry()
        registry.register(MockCheck("V01", "LC Check", CheckTier.LC_ONLY))
        registry.register(
            MockCheck(
                "V08",
                "Pixel Check",
                CheckTier.PIXEL,
                CheckRequirements(needs_tpf=True),
            )
        )

        pipeline = VettingPipeline(registry=registry)
        desc = pipeline.describe(tpf=None, network=False)

        assert len(desc["will_run"]) == 1
        assert len(desc["will_skip"]) == 1
        assert desc["will_skip"][0]["reason"] == "NO_TPF"

    def test_describe_payload_is_deterministic(self) -> None:
        registry = CheckRegistry()
        registry.register(
            MockCheck(
                "V08",
                "Pixel Check",
                CheckTier.PIXEL,
                CheckRequirements(needs_tpf=True),
            )
        )
        registry.register(MockCheck("V01", "LC Check", CheckTier.LC_ONLY))

        pipeline = VettingPipeline(registry=registry)
        first = pipeline.describe(tpf=None, network=False)
        second = pipeline.describe(tpf=None, network=False)

        assert first == second
        assert [item["id"] for item in first["will_run"]] == ["V01"]
        assert [item["id"] for item in first["will_skip"]] == ["V08"]


class TestRunMany:
    def test_run_many_preserves_order_and_returns_summary(self, mock_lc: Any) -> None:
        registry = CheckRegistry()
        registry.register(MockCheck("V01", "Check 1"))

        pipeline = VettingPipeline(registry=registry)

        def make_candidate(period: float, t0: float) -> Any:
            c = MagicMock()
            c.period = period
            c.t0 = t0
            c.duration_hours = 2.0
            c.depth = 0.001
            return c

        candidates = [make_candidate(3.0, 1.0), make_candidate(5.0, 2.0)]
        bundles, summary = pipeline.run_many(mock_lc, candidates)

        assert len(bundles) == 2
        assert len(summary) == 2
        assert summary[0]["candidate_index"] == 0
        assert summary[1]["candidate_index"] == 1
        assert summary[0]["period_days"] == 3.0
        assert summary[1]["period_days"] == 5.0


class TestListChecks:
    """Tests for list_checks and describe_checks functions."""

    def test_list_checks_empty(self) -> None:
        registry = CheckRegistry()
        checks = list_checks(registry)
        assert checks == []

    def test_list_checks(self) -> None:
        registry = CheckRegistry()
        registry.register(MockCheck("V01", "Test Check"))

        checks = list_checks(registry)
        assert len(checks) == 1
        assert checks[0]["id"] == "V01"
        assert checks[0]["citations"] == ["Test Citation"]

    def test_list_checks_is_deterministic_and_returns_copy(self) -> None:
        registry = CheckRegistry()
        registry.register(MockCheck("V02", "Second Check"))
        registry.register(MockCheck("V01", "First Check"))

        first = list_checks(registry)
        second = list_checks(registry)

        assert first == second
        assert [item["id"] for item in first] == ["V01", "V02"]

        first[0]["citations"].append("Mutated")
        first[0]["requirements"]["optional_deps"].append("fake_extra")

        third = list_checks(registry)
        assert third[0]["citations"] == ["Test Citation"]
        assert third[0]["requirements"]["optional_deps"] == []

    def test_describe_checks_empty(self) -> None:
        registry = CheckRegistry()
        desc = describe_checks(registry)
        assert "No checks" in desc

    def test_describe_checks(self) -> None:
        registry = CheckRegistry()
        registry.register(MockCheck("V01", "Test Check"))

        desc = describe_checks(registry)
        assert "V01" in desc
        assert "Test Check" in desc


def test_pipeline_contract_schemas_are_stable() -> None:
    assert LIST_CHECKS_INPUT_SCHEMA == {
        "type": "object",
        "properties": {"registry": {"description": "Optional check registry override."}},
        "additionalProperties": False,
    }
    assert LIST_CHECKS_OUTPUT_SCHEMA == {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "citations": {"type": "array", "items": {"type": "string"}},
                "id": {"type": "string"},
                "name": {"type": "string"},
                "requirements": {
                    "type": "object",
                    "properties": {
                        "needs_network": {"type": "boolean"},
                        "needs_ra_dec": {"type": "boolean"},
                        "needs_stellar": {"type": "boolean"},
                        "needs_tic_id": {"type": "boolean"},
                        "needs_tpf": {"type": "boolean"},
                        "optional_deps": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "needs_network",
                        "needs_ra_dec",
                        "needs_stellar",
                        "needs_tic_id",
                        "needs_tpf",
                        "optional_deps",
                    ],
                    "additionalProperties": False,
                },
                "tier": {"type": "string"},
            },
            "required": ["citations", "id", "name", "requirements", "tier"],
            "additionalProperties": False,
        },
    }
    assert DESCRIBE_CHECKS_INPUT_SCHEMA == {
        "type": "object",
        "properties": {"registry": {"description": "Optional check registry override."}},
        "additionalProperties": False,
    }
    assert DESCRIBE_CHECKS_OUTPUT_SCHEMA == {"type": "string"}
    assert PIPELINE_DESCRIBE_INPUT_SCHEMA == {
        "type": "object",
        "properties": {
            "dec_deg": {"type": ["number", "null"]},
            "network": {"type": "boolean"},
            "ra_deg": {"type": ["number", "null"]},
            "tic_id": {"type": ["integer", "null"]},
            "tpf": {
                "description": "Opaque TPF object; null means unavailable.",
                "type": ["object", "null"],
            },
        },
        "additionalProperties": False,
    }
    assert PIPELINE_DESCRIBE_OUTPUT_SCHEMA == {
        "type": "object",
        "properties": {
            "total_checks": {"type": "integer"},
            "will_run": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "tier": {"type": "string"},
                    },
                    "required": ["id", "name", "tier"],
                    "additionalProperties": False,
                },
            },
            "will_skip": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "reason": {"type": ["string", "null"]},
                        "tier": {"type": "string"},
                    },
                    "required": ["id", "name", "reason", "tier"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["total_checks", "will_run", "will_skip"],
        "additionalProperties": False,
    }
