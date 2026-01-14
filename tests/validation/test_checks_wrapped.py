"""Tests for check wrapper classes (V01-V07).

These tests verify that:
- Wrapper classes implement the VettingCheck protocol
- Results are proper CheckResult instances
- Metrics are JSON-serializable
- Status is one of "ok", "skipped", "error"
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bittr_tess_vetter.domain.detection import TransitCandidate
from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.validation.checks_catalog_wrapped import (
    ExoFOPTOILookupCheck,
    NearbyEBSearchCheck,
    register_catalog_checks,
)
from bittr_tess_vetter.validation.checks_exovetter_wrapped import (
    ModShiftCheck,
    SweetCheck,
    register_exovetter_checks,
)
from bittr_tess_vetter.validation.checks_lc_wrapped import (
    DepthStabilityCheck,
    DurationConsistencyCheck,
    OddEvenDepthCheck,
    SecondaryEclipseCheck,
    VShapeCheck,
    register_lc_checks,
)
from bittr_tess_vetter.validation.checks_pixel_wrapped import (
    ApertureDependenceCheck,
    CentroidShiftCheck,
    DifferenceImageCheck,
    register_pixel_checks,
)
from bittr_tess_vetter.validation.register_defaults import register_all_defaults
from bittr_tess_vetter.validation.registry import (
    CheckConfig,
    CheckInputs,
    CheckRegistry,
    CheckRequirements,
    CheckTier,
    VettingCheck,
)
from bittr_tess_vetter.validation.result_schema import CheckResult


def _make_test_lightcurve(
    *,
    period_days: float = 3.0,
    t0_btjd: float = 1.0,
    duration_hours: float = 2.0,
    baseline_days: float = 27.0,
    cadence_minutes: float = 30.0,
    depth_frac: float = 0.001,
    noise_ppm: float = 50.0,
    seed: int = 42,
) -> LightCurveData:
    """Create a test light curve with injected transits."""
    rng = np.random.default_rng(seed)
    dt_days = cadence_minutes / (24.0 * 60.0)
    time = np.arange(0.0, baseline_days, dt_days, dtype=np.float64)
    flux = np.ones_like(time)
    flux += rng.normal(0.0, noise_ppm * 1e-6, size=time.size)
    flux_err = np.full_like(time, noise_ppm * 1e-6)
    quality = np.zeros(time.size, dtype=np.int32)
    valid_mask = np.ones(time.size, dtype=bool)

    # Inject box transits
    dur_days = duration_hours / 24.0
    half = dur_days / 2.0
    phase = ((time - t0_btjd) / period_days) % 1.0
    phase_dist = np.minimum(phase, 1.0 - phase)
    in_transit = phase_dist < (half / period_days)
    flux[in_transit] *= 1.0 - depth_frac

    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality,
        valid_mask=valid_mask,
        tic_id=123456789,
        sector=1,
        cadence_seconds=int(cadence_minutes * 60),
    )


def _make_test_candidate(
    period: float = 3.0,
    t0: float = 1.0,
    duration_hours: float = 2.0,
    depth: float = 0.001,
    snr: float = 10.0,
) -> TransitCandidate:
    """Create a test transit candidate."""
    return TransitCandidate(
        period=period,
        t0=t0,
        duration_hours=duration_hours,
        depth=depth,
        snr=snr,
    )


def _make_test_inputs(
    lc: LightCurveData | None = None,
    candidate: TransitCandidate | None = None,
    **kwargs,
) -> CheckInputs:
    """Create test CheckInputs."""
    if lc is None:
        lc = _make_test_lightcurve()
    if candidate is None:
        candidate = _make_test_candidate()
    return CheckInputs(lc=lc, candidate=candidate, **kwargs)


class TestOddEvenDepthCheck:
    """Tests for V01 OddEvenDepthCheck wrapper."""

    def test_implements_protocol(self) -> None:
        check = OddEvenDepthCheck()
        assert isinstance(check, VettingCheck)

    def test_check_attributes(self) -> None:
        check = OddEvenDepthCheck()
        assert check.id == "V01"
        assert check.name == "Odd-Even Depth"
        assert check.tier == CheckTier.LC_ONLY
        assert isinstance(check.requirements, CheckRequirements)
        assert len(check.citations) > 0

    def test_run_returns_check_result(self) -> None:
        check = OddEvenDepthCheck()
        inputs = _make_test_inputs()
        config = CheckConfig()

        result = check.run(inputs, config)

        assert isinstance(result, CheckResult)
        assert result.id == "V01"
        assert result.status in ("ok", "skipped", "error")

    def test_result_is_json_serializable(self) -> None:
        check = OddEvenDepthCheck()
        inputs = _make_test_inputs()
        config = CheckConfig()

        result = check.run(inputs, config)

        # Should not raise
        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "V01"


class TestSecondaryEclipseCheck:
    """Tests for V02 SecondaryEclipseCheck wrapper."""

    def test_implements_protocol(self) -> None:
        check = SecondaryEclipseCheck()
        assert isinstance(check, VettingCheck)

    def test_check_attributes(self) -> None:
        check = SecondaryEclipseCheck()
        assert check.id == "V02"
        assert check.tier == CheckTier.LC_ONLY

    def test_run_returns_check_result(self) -> None:
        check = SecondaryEclipseCheck()
        inputs = _make_test_inputs()
        config = CheckConfig()

        result = check.run(inputs, config)

        assert isinstance(result, CheckResult)
        assert result.id == "V02"
        assert result.status in ("ok", "skipped", "error")


class TestDurationConsistencyCheck:
    """Tests for V03 DurationConsistencyCheck wrapper."""

    def test_implements_protocol(self) -> None:
        check = DurationConsistencyCheck()
        assert isinstance(check, VettingCheck)

    def test_check_attributes(self) -> None:
        check = DurationConsistencyCheck()
        assert check.id == "V03"
        assert check.requirements.needs_stellar is True

    def test_run_returns_check_result(self) -> None:
        check = DurationConsistencyCheck()
        inputs = _make_test_inputs()
        config = CheckConfig()

        result = check.run(inputs, config)

        assert isinstance(result, CheckResult)
        assert result.id == "V03"
        assert result.status in ("ok", "skipped", "error")


class TestDepthStabilityCheck:
    """Tests for V04 DepthStabilityCheck wrapper."""

    def test_implements_protocol(self) -> None:
        check = DepthStabilityCheck()
        assert isinstance(check, VettingCheck)

    def test_check_attributes(self) -> None:
        check = DepthStabilityCheck()
        assert check.id == "V04"
        assert check.tier == CheckTier.LC_ONLY

    def test_run_returns_check_result(self) -> None:
        check = DepthStabilityCheck()
        inputs = _make_test_inputs()
        config = CheckConfig()

        result = check.run(inputs, config)

        assert isinstance(result, CheckResult)
        assert result.id == "V04"
        assert result.status in ("ok", "skipped", "error")


class TestVShapeCheck:
    """Tests for V05 VShapeCheck wrapper."""

    def test_implements_protocol(self) -> None:
        check = VShapeCheck()
        assert isinstance(check, VettingCheck)

    def test_check_attributes(self) -> None:
        check = VShapeCheck()
        assert check.id == "V05"
        assert check.tier == CheckTier.LC_ONLY

    def test_run_returns_check_result(self) -> None:
        check = VShapeCheck()
        inputs = _make_test_inputs()
        config = CheckConfig()

        result = check.run(inputs, config)

        assert isinstance(result, CheckResult)
        assert result.id == "V05"
        assert result.status in ("ok", "skipped", "error")


class TestNearbyEBSearchCheck:
    """Tests for V06 NearbyEBSearchCheck wrapper."""

    def test_implements_protocol(self) -> None:
        check = NearbyEBSearchCheck()
        assert isinstance(check, VettingCheck)

    def test_check_attributes(self) -> None:
        check = NearbyEBSearchCheck()
        assert check.id == "V06"
        assert check.tier == CheckTier.CATALOG
        assert check.requirements.needs_network is True
        assert check.requirements.needs_ra_dec is True

    def test_skips_without_coordinates(self) -> None:
        check = NearbyEBSearchCheck()
        inputs = _make_test_inputs(network=True, ra_deg=None, dec_deg=None)
        config = CheckConfig()

        result = check.run(inputs, config)

        assert result.status == "skipped"
        assert "SKIPPED:NO_COORDINATES" in result.flags

    def test_skips_without_network(self) -> None:
        check = NearbyEBSearchCheck()
        inputs = _make_test_inputs(network=False, ra_deg=10.0, dec_deg=20.0)
        config = CheckConfig()

        result = check.run(inputs, config)

        assert result.status == "skipped"
        assert "SKIPPED:NETWORK_DISABLED" in result.flags


class TestExoFOPTOILookupCheck:
    """Tests for V07 ExoFOPTOILookupCheck wrapper."""

    def test_implements_protocol(self) -> None:
        check = ExoFOPTOILookupCheck()
        assert isinstance(check, VettingCheck)

    def test_check_attributes(self) -> None:
        check = ExoFOPTOILookupCheck()
        assert check.id == "V07"
        assert check.tier == CheckTier.CATALOG
        assert check.requirements.needs_network is True
        assert check.requirements.needs_tic_id is True

    def test_skips_without_tic_id(self) -> None:
        check = ExoFOPTOILookupCheck()
        inputs = _make_test_inputs(network=True, tic_id=None)
        config = CheckConfig()

        result = check.run(inputs, config)

        assert result.status == "skipped"
        assert "SKIPPED:NO_TIC_ID" in result.flags

    def test_skips_without_network(self) -> None:
        check = ExoFOPTOILookupCheck()
        inputs = _make_test_inputs(network=False, tic_id=123456789)
        config = CheckConfig()

        result = check.run(inputs, config)

        assert result.status == "skipped"
        assert "SKIPPED:NETWORK_DISABLED" in result.flags


class TestRegisterFunctions:
    """Tests for check registration functions."""

    def test_register_lc_checks(self) -> None:
        registry = CheckRegistry()
        register_lc_checks(registry)

        assert len(registry) == 5
        assert "V01" in registry
        assert "V02" in registry
        assert "V03" in registry
        assert "V04" in registry
        assert "V05" in registry

    def test_register_catalog_checks(self) -> None:
        registry = CheckRegistry()
        register_catalog_checks(registry)

        assert len(registry) == 2
        assert "V06" in registry
        assert "V07" in registry

    def test_register_all_defaults(self) -> None:
        registry = CheckRegistry()
        register_all_defaults(registry)

        assert len(registry) == 7
        assert registry.list_ids() == ["V01", "V02", "V03", "V04", "V05", "V06", "V07"]

    def test_registered_checks_are_valid(self) -> None:
        registry = CheckRegistry()
        register_all_defaults(registry)

        for check in registry.list():
            assert isinstance(check, VettingCheck)
            assert check.id.startswith("V0")
            assert check.name
            assert isinstance(check.tier, CheckTier)


class TestCheckResultMetrics:
    """Tests verifying metrics are properly JSON-serializable."""

    def test_all_lc_checks_produce_serializable_results(self) -> None:
        """All LC checks should produce JSON-serializable results."""
        lc = _make_test_lightcurve(cadence_minutes=2.0)  # Higher cadence for better results
        candidate = _make_test_candidate()
        inputs = _make_test_inputs(lc=lc, candidate=candidate)
        config = CheckConfig()

        checks = [
            OddEvenDepthCheck(),
            SecondaryEclipseCheck(),
            DurationConsistencyCheck(),
            DepthStabilityCheck(),
            VShapeCheck(),
        ]

        for check in checks:
            result = check.run(inputs, config)

            # Verify result is a proper CheckResult
            assert isinstance(result, CheckResult), f"{check.id} did not return CheckResult"

            # Verify JSON serialization works
            try:
                json_str = result.model_dump_json()
                parsed = json.loads(json_str)
                assert parsed["id"] == check.id
            except Exception as e:
                pytest.fail(f"{check.id} result not JSON-serializable: {e}")

            # Verify metrics contain only scalar types
            for key, value in result.metrics.items():
                assert isinstance(
                    value, (float, int, str, bool, type(None))
                ), f"{check.id} metric '{key}' has non-scalar type: {type(value)}"


# =============================================================================
# Pixel check tests (V08-V10)
# =============================================================================


class MockTPF:
    """Mock TPF for testing pixel checks."""

    def __init__(self, time: np.ndarray, flux: np.ndarray) -> None:
        self.time = time
        self.flux = flux


class MockEphemeris:
    """Mock ephemeris for testing."""

    def __init__(
        self,
        period_days: float = 5.0,
        t0_btjd: float = 1000.0,
        duration_hours: float = 3.0,
    ) -> None:
        self.period_days = period_days
        self.t0_btjd = t0_btjd
        self.duration_hours = duration_hours


class MockCandidate:
    """Mock candidate for testing."""

    def __init__(
        self,
        ephemeris: MockEphemeris | None = None,
        depth: float | None = 0.001,
    ) -> None:
        self.ephemeris = ephemeris or MockEphemeris()
        self.depth = depth


class MockLightCurve:
    """Mock light curve for testing exovetter checks."""

    def __init__(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray | None = None,
    ) -> None:
        self.time = time
        self.flux = flux
        self.flux_err = flux_err

    def to_internal(self) -> LightCurveData:
        """Return internal representation."""
        valid_mask = np.ones(len(self.time), dtype=bool)
        quality = np.zeros(len(self.time), dtype=np.int32)
        flux_err = self.flux_err if self.flux_err is not None else np.ones_like(self.flux) * 0.001

        return LightCurveData(
            time=self.time,
            flux=self.flux,
            flux_err=flux_err,
            quality=quality,
            valid_mask=valid_mask,
            tic_id=123,
            sector=1,
            cadence_seconds=120,
        )


def _make_test_tpf(n_cadences: int = 100, n_pixels: int = 5) -> MockTPF:
    """Create a test TPF with synthetic data."""
    time = np.linspace(0, 50, n_cadences)
    flux = np.full((n_cadences, n_pixels, n_pixels), 1000.0, dtype=np.float64)

    # Add a small transit signal at center pixel
    in_transit = (time % 5.0) < 0.125  # ~3 hour transit for 5-day period
    flux[in_transit, n_pixels // 2, n_pixels // 2] *= 0.99  # 1% depth

    return MockTPF(time=time, flux=flux)


def _make_test_mock_lc(n_points: int = 1000, baseline_days: float = 50.0) -> MockLightCurve:
    """Create a test light curve with synthetic data."""
    time = np.linspace(0, baseline_days, n_points)
    flux = np.ones(n_points, dtype=np.float64)
    flux_err = np.ones(n_points, dtype=np.float64) * 0.001

    # Add transit signal
    period = 5.0
    for i in range(int(baseline_days / period)):
        t_mid = i * period + 2.5
        in_transit = np.abs(time - t_mid) < 0.0625  # ~3 hours
        flux[in_transit] *= 0.999  # Small dip

    return MockLightCurve(time=time, flux=flux, flux_err=flux_err)


def _make_pixel_inputs(
    tpf: MockTPF | None = None,
    lc: MockLightCurve | None = None,
    depth: float | None = 0.001,
) -> CheckInputs:
    """Create CheckInputs for pixel/exovetter tests."""
    return CheckInputs(
        lc=lc or _make_test_mock_lc(),
        candidate=MockCandidate(MockEphemeris(), depth=depth),
        tpf=tpf,
    )


class TestCentroidShiftCheck:
    """Tests for V08 CentroidShiftCheck wrapper."""

    def test_implements_protocol(self) -> None:
        check = CentroidShiftCheck()
        assert isinstance(check, VettingCheck)

    def test_check_attributes(self) -> None:
        check = CentroidShiftCheck()
        assert check.id == "V08"
        assert check.name == "Centroid Shift"
        assert check.tier == CheckTier.PIXEL
        assert check.requirements.needs_tpf is True
        assert len(check.citations) > 0

    def test_skips_without_tpf(self) -> None:
        check = CentroidShiftCheck()
        inputs = _make_pixel_inputs(tpf=None)
        config = CheckConfig()

        result = check.run(inputs, config)

        assert result.status == "skipped"
        assert "SKIPPED:NO_TPF" in result.flags
        assert result.confidence is None

    def test_returns_check_result_with_tpf(self) -> None:
        check = CentroidShiftCheck()
        inputs = _make_pixel_inputs(tpf=_make_test_tpf())
        config = CheckConfig()

        result = check.run(inputs, config)

        assert isinstance(result, CheckResult)
        assert result.id == "V08"
        assert result.status in ("ok", "skipped", "error")

    def test_result_is_json_serializable(self) -> None:
        check = CentroidShiftCheck()
        inputs = _make_pixel_inputs(tpf=_make_test_tpf())
        config = CheckConfig()

        result = check.run(inputs, config)

        # Should not raise
        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "V08"


class TestDifferenceImageCheck:
    """Tests for V09 DifferenceImageCheck wrapper."""

    def test_implements_protocol(self) -> None:
        check = DifferenceImageCheck()
        assert isinstance(check, VettingCheck)

    def test_check_attributes(self) -> None:
        check = DifferenceImageCheck()
        assert check.id == "V09"
        assert check.name == "Difference Image"
        assert check.tier == CheckTier.PIXEL
        assert check.requirements.needs_tpf is True

    def test_skips_without_tpf(self) -> None:
        check = DifferenceImageCheck()
        inputs = _make_pixel_inputs(tpf=None)
        config = CheckConfig()

        result = check.run(inputs, config)

        assert result.status == "skipped"
        assert "SKIPPED:NO_TPF" in result.flags

    def test_returns_check_result_with_tpf(self) -> None:
        check = DifferenceImageCheck()
        inputs = _make_pixel_inputs(tpf=_make_test_tpf())
        config = CheckConfig()

        result = check.run(inputs, config)

        assert isinstance(result, CheckResult)
        assert result.id == "V09"
        assert result.status in ("ok", "skipped", "error")


class TestApertureDependenceCheck:
    """Tests for V10 ApertureDependenceCheck wrapper."""

    def test_implements_protocol(self) -> None:
        check = ApertureDependenceCheck()
        assert isinstance(check, VettingCheck)

    def test_check_attributes(self) -> None:
        check = ApertureDependenceCheck()
        assert check.id == "V10"
        assert check.name == "Aperture Dependence"
        assert check.tier == CheckTier.PIXEL
        assert check.requirements.needs_tpf is True

    def test_skips_without_tpf(self) -> None:
        check = ApertureDependenceCheck()
        inputs = _make_pixel_inputs(tpf=None)
        config = CheckConfig()

        result = check.run(inputs, config)

        assert result.status == "skipped"
        assert "SKIPPED:NO_TPF" in result.flags

    def test_returns_check_result_with_tpf(self) -> None:
        check = ApertureDependenceCheck()
        inputs = _make_pixel_inputs(tpf=_make_test_tpf())
        config = CheckConfig()

        result = check.run(inputs, config)

        assert isinstance(result, CheckResult)
        assert result.id == "V10"
        assert result.status in ("ok", "skipped", "error")


# =============================================================================
# Exovetter check tests (V11-V12)
# =============================================================================


class TestModShiftCheck:
    """Tests for V11 ModShiftCheck wrapper."""

    def test_implements_protocol(self) -> None:
        check = ModShiftCheck()
        assert isinstance(check, VettingCheck)

    def test_check_attributes(self) -> None:
        check = ModShiftCheck()
        assert check.id == "V11"
        assert check.name == "ModShift"
        assert check.tier == CheckTier.EXOVETTER
        assert "exovetter" in check.requirements.optional_deps

    def test_skips_without_depth(self) -> None:
        check = ModShiftCheck()
        inputs = _make_pixel_inputs(lc=_make_test_mock_lc(), depth=None)
        config = CheckConfig()

        result = check.run(inputs, config)

        assert result.status == "skipped"
        assert "SKIPPED:MISSING_DEPTH" in result.flags

    def test_returns_check_result(self) -> None:
        check = ModShiftCheck()
        inputs = _make_pixel_inputs(lc=_make_test_mock_lc())
        config = CheckConfig()

        result = check.run(inputs, config)

        assert isinstance(result, CheckResult)
        assert result.id == "V11"
        assert result.status in ("ok", "skipped", "error")

    def test_result_is_json_serializable(self) -> None:
        check = ModShiftCheck()
        inputs = _make_pixel_inputs(lc=_make_test_mock_lc())
        config = CheckConfig()

        result = check.run(inputs, config)

        # Should not raise
        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "V11"


class TestSweetCheck:
    """Tests for V12 SweetCheck wrapper."""

    def test_implements_protocol(self) -> None:
        check = SweetCheck()
        assert isinstance(check, VettingCheck)

    def test_check_attributes(self) -> None:
        check = SweetCheck()
        assert check.id == "V12"
        assert check.name == "SWEET"
        assert check.tier == CheckTier.EXOVETTER
        assert "exovetter" in check.requirements.optional_deps

    def test_skips_without_depth(self) -> None:
        check = SweetCheck()
        inputs = _make_pixel_inputs(lc=_make_test_mock_lc(), depth=None)
        config = CheckConfig()

        result = check.run(inputs, config)

        assert result.status == "skipped"
        assert "SKIPPED:MISSING_DEPTH" in result.flags

    def test_returns_check_result(self) -> None:
        check = SweetCheck()
        inputs = _make_pixel_inputs(lc=_make_test_mock_lc())
        config = CheckConfig()

        result = check.run(inputs, config)

        assert isinstance(result, CheckResult)
        assert result.id == "V12"
        assert result.status in ("ok", "skipped", "error")


# =============================================================================
# Registration tests for pixel and exovetter checks
# =============================================================================


class TestPixelAndExovetterRegistration:
    """Tests for pixel and exovetter check registration functions."""

    def test_register_pixel_checks(self) -> None:
        registry = CheckRegistry()
        register_pixel_checks(registry)

        assert len(registry) == 3
        assert "V08" in registry
        assert "V09" in registry
        assert "V10" in registry

    def test_register_exovetter_checks(self) -> None:
        registry = CheckRegistry()
        register_exovetter_checks(registry)

        assert len(registry) == 2
        assert "V11" in registry
        assert "V12" in registry

    def test_registered_pixel_checks_have_correct_tier(self) -> None:
        registry = CheckRegistry()
        register_pixel_checks(registry)

        pixel_checks = registry.list_by_tier(CheckTier.PIXEL)
        assert len(pixel_checks) == 3
        assert all(c.tier == CheckTier.PIXEL for c in pixel_checks)

    def test_registered_exovetter_checks_have_correct_tier(self) -> None:
        registry = CheckRegistry()
        register_exovetter_checks(registry)

        exovetter_checks = registry.list_by_tier(CheckTier.EXOVETTER)
        assert len(exovetter_checks) == 2
        assert all(c.tier == CheckTier.EXOVETTER for c in exovetter_checks)

    def test_duplicate_pixel_registration_raises(self) -> None:
        registry = CheckRegistry()
        register_pixel_checks(registry)

        with pytest.raises(ValueError, match="already registered"):
            register_pixel_checks(registry)

    def test_duplicate_exovetter_registration_raises(self) -> None:
        registry = CheckRegistry()
        register_exovetter_checks(registry)

        with pytest.raises(ValueError, match="already registered"):
            register_exovetter_checks(registry)


# =============================================================================
# Full registry with all checks V01-V12
# =============================================================================


class TestFullRegistryWithAllChecks:
    """Tests for combining all checks V01-V12 in a single registry."""

    def test_register_all_checks(self) -> None:
        registry = CheckRegistry()
        register_all_defaults(registry)  # V01-V07
        register_pixel_checks(registry)  # V08-V10
        register_exovetter_checks(registry)  # V11-V12

        assert len(registry) == 12
        expected_ids = [f"V{i:02d}" for i in range(1, 13)]
        assert registry.list_ids() == expected_ids

    def test_all_checks_implement_protocol(self) -> None:
        registry = CheckRegistry()
        register_all_defaults(registry)
        register_pixel_checks(registry)
        register_exovetter_checks(registry)

        for check in registry.list():
            assert isinstance(check, VettingCheck)

    def test_tier_distribution(self) -> None:
        registry = CheckRegistry()
        register_all_defaults(registry)
        register_pixel_checks(registry)
        register_exovetter_checks(registry)

        lc_checks = registry.list_by_tier(CheckTier.LC_ONLY)
        catalog_checks = registry.list_by_tier(CheckTier.CATALOG)
        pixel_checks = registry.list_by_tier(CheckTier.PIXEL)
        exovetter_checks = registry.list_by_tier(CheckTier.EXOVETTER)

        assert len(lc_checks) == 5  # V01-V05
        assert len(catalog_checks) == 2  # V06-V07
        assert len(pixel_checks) == 3  # V08-V10
        assert len(exovetter_checks) == 2  # V11-V12


# =============================================================================
# Pixel check error handling tests
# =============================================================================


class TestPixelCheckErrorHandling:
    """Tests for error handling in pixel checks."""

    def test_centroid_handles_invalid_tpf_shape(self) -> None:
        """CentroidShiftCheck handles invalid TPF gracefully."""
        check = CentroidShiftCheck()

        # Create invalid TPF (2D instead of 3D)
        invalid_tpf = MockTPF(
            time=np.linspace(0, 10, 100),
            flux=np.ones((100, 5), dtype=np.float64),  # Missing dimension
        )
        inputs = _make_pixel_inputs(tpf=invalid_tpf)
        config = CheckConfig()

        result = check.run(inputs, config)

        # Should return error, not raise
        assert result.status == "error"
        assert any("ERROR:" in f for f in result.flags)

    def test_difference_image_handles_invalid_tpf_shape(self) -> None:
        """DifferenceImageCheck handles invalid TPF gracefully."""
        check = DifferenceImageCheck()

        invalid_tpf = MockTPF(
            time=np.linspace(0, 10, 100),
            flux=np.ones((100, 5), dtype=np.float64),
        )
        inputs = _make_pixel_inputs(tpf=invalid_tpf)
        config = CheckConfig()

        result = check.run(inputs, config)

        # Should return error or skipped, not raise
        assert result.status in ("error", "skipped")
        # For insufficient data, check returns skipped with INSUFFICIENT_DATA flag
        # For actual errors, returns error with ERROR: flag
        has_error_or_skip = any("ERROR:" in f or "SKIPPED:" in f for f in result.flags)
        assert has_error_or_skip

    def test_aperture_handles_invalid_tpf_shape(self) -> None:
        """ApertureDependenceCheck handles invalid TPF gracefully."""
        check = ApertureDependenceCheck()

        invalid_tpf = MockTPF(
            time=np.linspace(0, 10, 100),
            flux=np.ones((100, 5), dtype=np.float64),
        )
        inputs = _make_pixel_inputs(tpf=invalid_tpf)
        config = CheckConfig()

        result = check.run(inputs, config)

        # Should return error or skipped, not raise
        assert result.status in ("error", "skipped")
        has_error_or_skip = any("ERROR:" in f or "SKIPPED:" in f for f in result.flags)
        assert has_error_or_skip
