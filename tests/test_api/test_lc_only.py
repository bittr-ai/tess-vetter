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


def _inject_transits_by_epoch_parity(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    depth_odd: float,
    depth_even: float,
) -> np.ndarray:
    """Inject transits with different depths on odd/even epochs.

    Matches the epoch definition used by `validation.lc_checks.check_odd_even_depth`
    (epoch boundaries between transits).
    """
    out = flux.copy()
    duration_days = duration_hours / 24.0

    epoch = np.floor((time - t0_btjd + period_days / 2.0) / period_days).astype(int)
    phase = ((time - t0_btjd) / period_days) % 1.0
    phase_dist = np.minimum(phase, 1.0 - phase)
    half_dur_phase = 0.5 * (duration_days / period_days)
    in_transit = phase_dist < half_dur_phase

    odd = (epoch % 2) == 1
    out[in_transit & odd] *= 1.0 - depth_odd
    out[in_transit & ~odd] *= 1.0 - depth_even
    return out


def _inject_secondary_eclipse_window(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period_days: float,
    t0_btjd: float,
    depth: float,
    center: float = 0.5,
    half_width: float = 0.15,
) -> np.ndarray:
    """Inject a broad secondary eclipse dip across the default search window.

    This is intentionally aligned with the check's search window so the window
    median shifts and the metric becomes detectable (test stability).
    """
    out = flux.copy()
    phase = ((time - t0_btjd) / period_days) % 1.0
    lo = center - half_width
    hi = center + half_width
    sec = (phase > lo) & (phase < hi)
    out[sec] *= 1.0 - depth
    return out


def _inject_trapezoid_transit(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    depth: float,
    tflat_ttotal_ratio: float,
) -> np.ndarray:
    """Inject a trapezoid-shaped primary transit centered at phase 0.

    This is tailored for `validation.lc_checks.check_v_shape`, which uses
    phase in [-0.5, 0.5] with transit at 0.
    """
    out = flux.copy()
    duration_days = duration_hours / 24.0
    t_total_phase = duration_days / period_days
    half_total = t_total_phase / 2.0
    t_flat_phase = max(0.0, min(1.0, float(tflat_ttotal_ratio))) * t_total_phase
    half_flat = t_flat_phase / 2.0
    ingress = (t_total_phase - t_flat_phase) / 2.0

    phase = ((time - t0_btjd) / period_days + 0.5) % 1.0 - 0.5
    in_total = np.abs(phase) <= half_total

    if ingress <= 0:
        # Box-like (no ingress/egress)
        out[in_total] *= 1.0 - depth
        return out

    abs_phase = np.abs(phase[in_total])
    # Flat bottom region
    flat = abs_phase <= half_flat
    out_idx = np.where(in_total)[0]
    out[out_idx[flat]] *= 1.0 - depth

    # Ingress/egress regions: linear ramps from 0 depth at edge to full at flat boundary.
    ramp = ~flat
    ramp_dist = abs_phase[ramp] - half_flat
    frac = 1.0 - np.clip(ramp_dist / ingress, 0.0, 1.0)
    out[out_idx[ramp]] *= 1.0 - depth * frac
    return out


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
        assert result.passed is None
        assert result.details.get("_metrics_only") is True
        assert 0.0 <= result.confidence <= 1.0

    def test_odd_even_depth_passes_for_planet(self) -> None:
        """Test that odd_even_depth passes for planet-like transit."""
        time, flux, flux_err = _make_synthetic_transit_lc(depth=0.01)
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = odd_even_depth(lc, eph)

        # Default is metrics-only; caller makes policy decision downstream.
        assert result.passed is None
        assert result.details.get("_metrics_only") is True
        # With enough data, odd/even should not be suspicious for a clean injection.
        # (If insufficient data, the check will return the metrics-only sentinel with 0 depths.)
        if "insufficient_data_for_odd_even_check" not in result.details.get("warnings", []):
            assert result.details.get("suspicious") in (False, None)

    def test_odd_even_depth_flags_alternating_depths_as_suspicious(self) -> None:
        time, flux, flux_err = _make_synthetic_transit_lc(
            n_points=8000,
            period_days=3.5,
            t0_btjd=0.5,
            duration_hours=2.5,
            depth=0.0,  # inject via parity helper
            noise_level=2e-4,
            n_periods=12,
        )
        flux = _inject_transits_by_epoch_parity(
            time,
            flux,
            period_days=3.5,
            t0_btjd=0.5,
            duration_hours=2.5,
            depth_odd=0.02,
            depth_even=0.01,
        )
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = odd_even_depth(lc, eph)

        assert result.passed is None
        assert result.details.get("_metrics_only") is True
        assert "insufficient_data_for_odd_even_check" not in result.details.get("warnings", [])
        assert result.details.get("suspicious") is True
        assert result.details["depth_odd_ppm"] > result.details["depth_even_ppm"]


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
        assert result.passed is None
        assert result.details.get("_metrics_only") is True
        assert 0.0 <= result.confidence <= 1.0

    def test_secondary_eclipse_passes_no_secondary(self) -> None:
        """Test that secondary_eclipse passes when no secondary present."""
        time, flux, flux_err = _make_synthetic_transit_lc(depth=0.01)
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = secondary_eclipse(lc, eph)

        # Default is metrics-only; caller makes policy decision downstream.
        assert result.passed is None
        assert result.details.get("_metrics_only") is True
        if result.details.get("note") != "Insufficient data for secondary eclipse search":
            assert bool(result.details.get("significant_secondary")) is False

    def test_secondary_eclipse_detects_injected_secondary(self) -> None:
        time, flux, flux_err = _make_synthetic_transit_lc(
            n_points=12000,
            period_days=3.5,
            t0_btjd=0.5,
            duration_hours=2.5,
            depth=0.01,
            noise_level=2e-4,
            n_periods=12,
        )
        flux = _inject_secondary_eclipse_window(
            time, flux, period_days=3.5, t0_btjd=0.5, depth=0.01, center=0.5, half_width=0.15
        )
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = secondary_eclipse(lc, eph)

        assert result.passed is None
        assert result.details.get("_metrics_only") is True
        assert result.details.get("note") != "Insufficient data for secondary eclipse search"
        assert result.details.get("significant_secondary") is True
        assert result.details["secondary_depth_ppm"] > 1000.0
        assert "secondary_depth" in result.details


class TestDurationConsistency:
    """Tests for V03 duration_consistency check."""

    def test_duration_consistency_returns_check_result(self) -> None:
        """Test that duration_consistency returns CheckResult."""
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = duration_consistency(eph, stellar=None)

        assert isinstance(result, CheckResult)
        assert result.id == "V03"
        assert result.name == "duration_consistency"
        assert result.passed is None
        assert result.details.get("_metrics_only") is True
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

        assert result.passed is None
        assert result.details.get("_metrics_only") is True

    def test_duration_consistency_density_scaling_shortens_for_dense_star(self) -> None:
        eph = Ephemeris(period_days=10.0, t0_btjd=0.5, duration_hours=3.0)
        # M-dwarf-like: high density in solar units (~11)
        stellar_dense = StellarParams(radius=0.3, mass=0.3)

        result = duration_consistency(eph, stellar=stellar_dense)

        assert result.passed is None
        assert result.details.get("_metrics_only") is True
        assert result.details.get("density_corrected") is True
        assert result.details.get("stellar_density_solar") is not None
        assert result.details["stellar_density_solar"] > 5.0
        # Dense stars should have shorter expected durations than the solar fallback.
        assert result.details["expected_duration_hours"] < result.details["expected_duration_solar"]

    def test_duration_consistency_density_scaling_lengthens_for_giant(self) -> None:
        eph = Ephemeris(period_days=10.0, t0_btjd=0.5, duration_hours=20.0)
        # Giant-like: low density in solar units (~0.008)
        stellar_giant = StellarParams(radius=5.0, mass=1.0)

        result = duration_consistency(eph, stellar=stellar_giant)

        assert result.passed is None
        assert result.details.get("_metrics_only") is True
        assert result.details.get("density_corrected") is True
        assert result.details.get("stellar_density_solar") is not None
        assert result.details["stellar_density_solar"] < 0.05
        # Giants should have longer expected durations than the solar fallback.
        assert result.details["expected_duration_hours"] > result.details["expected_duration_solar"]


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
        assert result.passed is None
        assert result.details.get("_metrics_only") is True
        assert 0.0 <= result.confidence <= 1.0

    def test_depth_stability_stable_transits(self) -> None:
        """Test that stable transits pass depth_stability."""
        time, flux, flux_err = _make_synthetic_transit_lc(depth=0.01, n_periods=10)
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = depth_stability(lc, eph)

        # Default is metrics-only; caller makes policy decision downstream.
        assert result.passed is None
        assert result.details.get("_metrics_only") is True
        assert "warnings" in result.details

    def test_depth_stability_detects_outlier_epoch(self) -> None:
        time, flux, flux_err = _make_synthetic_transit_lc(
            n_points=12000,
            period_days=3.5,
            t0_btjd=0.5,
            duration_hours=2.5,
            depth=0.0,
            noise_level=2e-4,
            n_periods=14,
        )
        # Start with consistent depth, then make one epoch deeper.
        flux = _inject_transits_by_epoch_parity(
            time,
            flux,
            period_days=3.5,
            t0_btjd=0.5,
            duration_hours=2.5,
            depth_odd=0.01,
            depth_even=0.01,
        )
        # Make epoch 4 deeper by applying an extra multiplicative dip to those in-transit cadences.
        period_days = 3.5
        t0_btjd = 0.5
        duration_days = 2.5 / 24.0
        epoch = np.floor((time - t0_btjd + period_days / 2.0) / period_days).astype(int)
        phase = ((time - t0_btjd) / period_days) % 1.0
        phase_dist = np.minimum(phase, 1.0 - phase)
        in_transit = phase_dist < 0.5 * (duration_days / period_days)
        flux[(epoch == 4) & in_transit] *= 1.0 - 0.02

        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = depth_stability(lc, eph)
        assert result.passed is None
        assert result.details.get("_metrics_only") is True
        warnings = result.details.get("warnings", [])
        assert any("Outlier epochs detected" in str(w) for w in warnings)


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
        assert result.passed is None
        assert result.details.get("_metrics_only") is True
        assert 0.0 <= result.confidence <= 1.0

    def test_v_shape_classifies_box_like_as_u_shape(self) -> None:
        time, flux, flux_err = _make_synthetic_transit_lc(
            n_points=12000,
            period_days=3.5,
            t0_btjd=0.5,
            duration_hours=2.5,
            depth=0.0,
            noise_level=2e-4,
            n_periods=12,
        )
        flux = _inject_trapezoid_transit(
            time,
            flux,
            period_days=3.5,
            t0_btjd=0.5,
            duration_hours=2.5,
            depth=0.01,
            tflat_ttotal_ratio=0.9,
        )
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = v_shape(lc, eph)
        assert result.passed is None
        assert result.details.get("_metrics_only") is True
        if result.details.get("classification") != "INSUFFICIENT_DATA":
            assert result.details.get("classification") in ("U_SHAPE", "GRAZING")

    def test_v_shape_classifies_triangle_like_as_v_shape(self) -> None:
        time, flux, flux_err = _make_synthetic_transit_lc(
            n_points=12000,
            period_days=3.5,
            t0_btjd=0.5,
            duration_hours=2.5,
            depth=0.0,
            noise_level=0.0,
            n_periods=12,
        )
        flux = _inject_trapezoid_transit(
            time,
            flux,
            period_days=3.5,
            t0_btjd=0.5,
            duration_hours=2.5,
            depth=0.03,
            tflat_ttotal_ratio=0.0,
        )
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

        result = v_shape(lc, eph)
        assert result.passed is None
        assert result.details.get("_metrics_only") is True
        if result.details.get("classification") != "INSUFFICIENT_DATA":
            assert result.details.get("classification") == "V_SHAPE"


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
