"""Tests for vetting pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from bittr_tess_vetter.api.pipeline import (
    VettingPipeline,
    describe_checks,
    list_checks,
)
from bittr_tess_vetter.validation.registry import (
    CheckConfig,
    CheckInputs,
    CheckRegistry,
    CheckRequirements,
    CheckTier,
)
from bittr_tess_vetter.validation.result_schema import CheckResult, ok_result


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

    def test_run_skips_when_requirements_not_met(
        self, mock_lc: Any, mock_candidate: Any
    ) -> None:
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

    def test_run_handles_check_error(
        self, mock_lc: Any, mock_candidate: Any
    ) -> None:
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
