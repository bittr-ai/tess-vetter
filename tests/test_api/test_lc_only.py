"""Tests for api/lc_only.py and api/transit_primitives.py."""

import numpy as np

from bittr_tess_vetter.api import (
    CheckResult,
    Ephemeris,
    LightCurve,
    OddEvenResult,
    StellarParams,
    depth_stability,
    duration_consistency,
    odd_even_depth,
    odd_even_result,
    secondary_eclipse,
    v_shape,
    vet_lc_only,
)


def _make_synthetic_transit_lc(
    n_points: int = 1000,
    period_days: float = 3.5,
    t0_btjd: float = 0.5,
    duration_hours: float = 2.5,
    depth: float = 0.01,
    noise_level: float = 0.001,
    n_periods: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic light curve with transits.

    Args:
        n_points: Number of data points
        period_days: Orbital period in days
        t0_btjd: Reference epoch
        duration_hours: Transit duration in hours
        depth: Transit depth (fractional)
        noise_level: Gaussian noise level
        n_periods: Number of orbital periods to cover

    Returns:
        Tuple of (time, flux, flux_err) arrays
    """
    # Time array covering multiple periods
    total_time = n_periods * period_days
    time = np.linspace(0, total_time, n_points)

    # Start with flat flux
    flux = np.ones(n_points)

    # Add transits
    duration_days = duration_hours / 24.0
    phase = ((time - t0_btjd) / period_days) % 1.0
    # Transit at phase 0 (and near 1)
    in_transit = (phase < duration_days / period_days / 2) | (
        phase > 1.0 - duration_days / period_days / 2
    )
    flux[in_transit] = 1.0 - depth

    # Add noise
    rng = np.random.default_rng(42)
    flux += rng.normal(0, noise_level, n_points)

    # Flux errors
    flux_err = np.ones(n_points) * noise_level

    return time, flux, flux_err


class TestOddEvenResult:
    """Tests for odd_even_result transit primitive."""

    def test_odd_even_result_returns_odd_even_result(self) -> None:
        """Test that odd_even_result returns OddEvenResult."""
        time, flux, flux_err = _make_synthetic_transit_lc()
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = odd_even_result(lc, eph)

        assert isinstance(result, OddEvenResult)
        assert hasattr(result, "depth_odd_ppm")
        assert hasattr(result, "depth_even_ppm")
        assert hasattr(result, "is_suspicious")

    def test_odd_even_result_consistent_depths(self) -> None:
        """Test that consistent odd/even depths are not suspicious."""
        time, flux, flux_err = _make_synthetic_transit_lc(depth=0.01)
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = odd_even_result(lc, eph)

        # Synthetic data should have consistent odd/even depths
        assert not result.is_suspicious


class TestOddEvenDepth:
    """Tests for V01 odd_even_depth check."""

    def test_odd_even_depth_returns_check_result(self) -> None:
        """Test that odd_even_depth returns CheckResult."""
        time, flux, flux_err = _make_synthetic_transit_lc()
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = odd_even_depth(lc, eph)

        assert isinstance(result, CheckResult)
        assert result.id == "V01"
        assert result.name == "odd_even_depth"
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.confidence <= 1.0

    def test_odd_even_depth_passes_for_planet(self) -> None:
        """Test that odd_even_depth passes for planet-like transit."""
        time, flux, flux_err = _make_synthetic_transit_lc(depth=0.01)
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = odd_even_depth(lc, eph)

        # Consistent depths should pass
        assert result.passed


class TestSecondaryEclipse:
    """Tests for V02 secondary_eclipse check."""

    def test_secondary_eclipse_returns_check_result(self) -> None:
        """Test that secondary_eclipse returns CheckResult."""
        time, flux, flux_err = _make_synthetic_transit_lc()
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = secondary_eclipse(lc, eph)

        assert isinstance(result, CheckResult)
        assert result.id == "V02"
        assert result.name == "secondary_eclipse"
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.confidence <= 1.0

    def test_secondary_eclipse_passes_no_secondary(self) -> None:
        """Test that secondary_eclipse passes when no secondary present."""
        time, flux, flux_err = _make_synthetic_transit_lc(depth=0.01)
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = secondary_eclipse(lc, eph)

        # No secondary eclipse -> should pass
        assert result.passed


class TestDurationConsistency:
    """Tests for V03 duration_consistency check."""

    def test_duration_consistency_returns_check_result(self) -> None:
        """Test that duration_consistency returns CheckResult."""
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = duration_consistency(eph, stellar=None)

        assert isinstance(result, CheckResult)
        assert result.id == "V03"
        assert result.name == "duration_consistency"
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.confidence <= 1.0

    def test_duration_consistency_with_stellar_params(self) -> None:
        """Test duration_consistency with stellar parameters."""
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
        stellar = StellarParams(radius=1.0, mass=1.0)

        result = duration_consistency(eph, stellar=stellar)

        assert isinstance(result, CheckResult)
        assert result.id == "V03"
        # With stellar params, confidence should be higher
        assert "density_corrected" in result.details

    def test_duration_consistency_reasonable_duration(self) -> None:
        """Test that reasonable duration passes."""
        # For P=3.5 days around a Sun-like star, ~2.5 hours is reasonable
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
        stellar = StellarParams(radius=1.0, mass=1.0)

        result = duration_consistency(eph, stellar=stellar)

        assert result.passed


class TestDepthStability:
    """Tests for V04 depth_stability check."""

    def test_depth_stability_returns_check_result(self) -> None:
        """Test that depth_stability returns CheckResult."""
        time, flux, flux_err = _make_synthetic_transit_lc()
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = depth_stability(lc, eph)

        assert isinstance(result, CheckResult)
        assert result.id == "V04"
        assert result.name == "depth_stability"
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.confidence <= 1.0

    def test_depth_stability_stable_transits(self) -> None:
        """Test that stable transits pass depth_stability."""
        time, flux, flux_err = _make_synthetic_transit_lc(depth=0.01, n_periods=10)
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = depth_stability(lc, eph)

        # Consistent depth should pass
        assert result.passed


class TestVShape:
    """Tests for V05 v_shape check."""

    def test_v_shape_returns_check_result(self) -> None:
        """Test that v_shape returns CheckResult."""
        time, flux, flux_err = _make_synthetic_transit_lc()
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = v_shape(lc, eph)

        assert isinstance(result, CheckResult)
        assert result.id == "V05"
        assert result.name == "v_shape"
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.confidence <= 1.0


class TestVetLcOnly:
    """Tests for vet_lc_only orchestrator."""

    def test_vet_lc_only_returns_all_checks(self) -> None:
        """Test that vet_lc_only returns results for all 5 checks."""
        time, flux, flux_err = _make_synthetic_transit_lc()
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        results = vet_lc_only(lc, eph)

        assert len(results) == 5
        check_ids = {r.id for r in results}
        assert check_ids == {"V01", "V02", "V03", "V04", "V05"}

    def test_vet_lc_only_all_return_check_result(self) -> None:
        """Test that all results are CheckResult instances."""
        time, flux, flux_err = _make_synthetic_transit_lc()
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        results = vet_lc_only(lc, eph)

        for result in results:
            assert isinstance(result, CheckResult)

    def test_vet_lc_only_with_stellar_params(self) -> None:
        """Test vet_lc_only with stellar parameters."""
        time, flux, flux_err = _make_synthetic_transit_lc()
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
        stellar = StellarParams(radius=1.0, mass=1.0)

        results = vet_lc_only(lc, eph, stellar=stellar)

        assert len(results) == 5
        # V03 should have density_corrected=True
        v03 = next(r for r in results if r.id == "V03")
        assert v03.details.get("density_corrected") is True

    def test_vet_lc_only_enabled_subset(self) -> None:
        """Test vet_lc_only with subset of enabled checks."""
        time, flux, flux_err = _make_synthetic_transit_lc()
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        results = vet_lc_only(lc, eph, enabled={"V01", "V03"})

        assert len(results) == 2
        check_ids = {r.id for r in results}
        assert check_ids == {"V01", "V03"}

    def test_vet_lc_only_empty_enabled(self) -> None:
        """Test vet_lc_only with empty enabled set."""
        time, flux, flux_err = _make_synthetic_transit_lc()
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        results = vet_lc_only(lc, eph, enabled=set())

        assert len(results) == 0

    def test_vet_lc_only_preserves_order(self) -> None:
        """Test that vet_lc_only returns results in order V01-V05."""
        time, flux, flux_err = _make_synthetic_transit_lc()
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        results = vet_lc_only(lc, eph)

        expected_order = ["V01", "V02", "V03", "V04", "V05"]
        actual_order = [r.id for r in results]
        assert actual_order == expected_order
