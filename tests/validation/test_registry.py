"""Tests for check registry."""

from __future__ import annotations

import pytest

from bittr_tess_vetter.validation.registry import (
    CheckConfig,
    CheckInputs,
    CheckRegistry,
    CheckRequirements,
    CheckTier,
    VettingCheck,
)
from bittr_tess_vetter.validation.result_schema import CheckResult, ok_result


class MockCheck:
    """Mock check for testing."""

    def __init__(
        self,
        id: str,
        name: str,
        tier: CheckTier = CheckTier.LC_ONLY,
        requirements: CheckRequirements | None = None,
    ) -> None:
        self._id = id
        self._name = name
        self._tier = tier
        self._requirements = requirements or CheckRequirements()

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
        return []

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        return ok_result(self.id, self.name, metrics={"mock": True})


class TestCheckTier:
    """Tests for CheckTier enum."""

    def test_values(self) -> None:
        assert CheckTier.LC_ONLY.value == "lc_only"
        assert CheckTier.CATALOG.value == "catalog"
        assert CheckTier.PIXEL.value == "pixel"


class TestCheckRequirements:
    """Tests for CheckRequirements."""

    def test_defaults(self) -> None:
        req = CheckRequirements()
        assert req.needs_tpf is False
        assert req.needs_network is False
        assert req.optional_deps == ()

    def test_custom(self) -> None:
        req = CheckRequirements(
            needs_tpf=True,
            needs_network=True,
            optional_deps=("tls",),
        )
        assert req.needs_tpf is True
        assert req.optional_deps == ("tls",)


class TestCheckRegistry:
    """Tests for CheckRegistry."""

    def test_register_and_get(self) -> None:
        registry = CheckRegistry()
        check = MockCheck("V01", "Test Check")
        registry.register(check)

        retrieved = registry.get("V01")
        assert retrieved.id == "V01"
        assert retrieved.name == "Test Check"

    def test_register_duplicate_raises(self) -> None:
        registry = CheckRegistry()
        registry.register(MockCheck("V01", "Check 1"))

        with pytest.raises(ValueError, match="already registered"):
            registry.register(MockCheck("V01", "Check 2"))

    def test_get_unknown_raises(self) -> None:
        registry = CheckRegistry()
        with pytest.raises(KeyError, match="No check registered"):
            registry.get("V99")

    def test_list(self) -> None:
        registry = CheckRegistry()
        registry.register(MockCheck("V02", "Check 2"))
        registry.register(MockCheck("V01", "Check 1"))

        checks = registry.list()
        assert len(checks) == 2
        assert checks[0].id == "V01"  # Sorted
        assert checks[1].id == "V02"

    def test_list_by_tier(self) -> None:
        registry = CheckRegistry()
        registry.register(MockCheck("V01", "LC Check", CheckTier.LC_ONLY))
        registry.register(MockCheck("V02", "Pixel Check", CheckTier.PIXEL))
        registry.register(MockCheck("V03", "Another LC", CheckTier.LC_ONLY))

        lc_checks = registry.list_by_tier(CheckTier.LC_ONLY)
        assert len(lc_checks) == 2
        assert all(c.tier == CheckTier.LC_ONLY for c in lc_checks)

    def test_list_ids(self) -> None:
        registry = CheckRegistry()
        registry.register(MockCheck("V02", "Check 2"))
        registry.register(MockCheck("V01", "Check 1"))

        ids = registry.list_ids()
        assert ids == ["V01", "V02"]

    def test_contains(self) -> None:
        registry = CheckRegistry()
        registry.register(MockCheck("V01", "Check"))

        assert "V01" in registry
        assert "V99" not in registry

    def test_len(self) -> None:
        registry = CheckRegistry()
        assert len(registry) == 0

        registry.register(MockCheck("V01", "Check"))
        assert len(registry) == 1


class TestVettingCheckProtocol:
    """Tests for VettingCheck protocol."""

    def test_mock_implements_protocol(self) -> None:
        check = MockCheck("V01", "Test")
        assert isinstance(check, VettingCheck)
