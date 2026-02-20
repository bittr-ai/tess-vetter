"""Tests for batman physical transit model fitting.

Tests for:
- quick_estimate: Analytic initial guesses
- detect_exposure_time: Cadence detection from time array
- compute_batman_model: Batman light curve generation
- compute_derived_parameters: Derived parameter calculations
- fit_transit_model: Full fitting workflow (optimize and mcmc)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

# Check if batman is available
try:
    import batman  # noqa: F401

    HAS_BATMAN = True
except ImportError:
    HAS_BATMAN = False

requires_batman = pytest.mark.skipif(not HAS_BATMAN, reason="batman-package not installed")


class TestQuickEstimate:
    """Tests for analytic initial parameter estimates."""

    def test_estimates_rp_rs_from_depth(self) -> None:
        """Correctly estimates Rp/Rs from transit depth."""
        from tess_vetter.transit import quick_estimate

        # 1% depth = 10000 ppm -> Rp/Rs = 0.1
        result = quick_estimate(
            depth_ppm=10000,
            duration_hours=3.0,
            period_days=5.0,
        )

        assert "rp_rs" in result
        assert abs(result["rp_rs"] - 0.1) < 0.01

    def test_estimates_a_rs_from_period(self) -> None:
        """Estimates a/Rs from period and stellar density."""
        from tess_vetter.transit import quick_estimate

        result = quick_estimate(
            depth_ppm=1000,
            duration_hours=3.0,
            period_days=10.0,
            stellar_density_gcc=1.41,  # Solar
        )

        assert "a_rs" in result
        # For P=10d, solar density, a/Rs should be ~15-25
        assert 5 < result["a_rs"] < 50

    def test_returns_required_keys(self) -> None:
        """Returns all required initial guess keys."""
        from tess_vetter.transit import quick_estimate

        result = quick_estimate(
            depth_ppm=5000,
            duration_hours=2.5,
            period_days=3.0,
        )

        assert "rp_rs" in result
        assert "a_rs" in result
        assert "inc" in result
        assert "t0_offset" in result

    def test_handles_small_depth(self) -> None:
        """Handles small transit depths (Earth-like)."""
        from tess_vetter.transit import quick_estimate

        # 100 ppm depth (Earth-like)
        result = quick_estimate(
            depth_ppm=100,
            duration_hours=4.0,
            period_days=365.0,
        )

        assert result["rp_rs"] > 0
        assert result["rp_rs"] < 0.05  # Small planet

    def test_initial_inclination_is_transiting(self) -> None:
        """Initial guess avoids a non-transiting (flat-model) geometry."""
        from tess_vetter.transit import quick_estimate

        result = quick_estimate(
            depth_ppm=1000,
            duration_hours=3.75,
            period_days=13.94,
        )

        rp_rs = result["rp_rs"]
        a_rs = result["a_rs"]
        inc = result["inc"]
        impact_parameter = a_rs * np.cos(np.deg2rad(inc))
        assert impact_parameter < 1.0 + rp_rs


class TestDetectExposureTime:
    """Tests for exposure time detection from cadence."""

    def test_detects_2min_cadence(self) -> None:
        """Correctly detects 2-minute cadence."""
        from tess_vetter.transit.batman_model import detect_exposure_time

        # 2-minute cadence
        time = np.arange(0, 1, 2.0 / 60 / 24)  # 2 min in days
        exp_time = detect_exposure_time(time)

        # Should be ~2 minutes in days
        exp_time_minutes = exp_time * 24 * 60
        assert 1.5 < exp_time_minutes < 2.5

    def test_detects_20sec_cadence(self) -> None:
        """Correctly detects 20-second cadence."""
        from tess_vetter.transit.batman_model import detect_exposure_time

        # 20-second cadence
        time = np.arange(0, 0.5, 20.0 / 60 / 60 / 24)
        exp_time = detect_exposure_time(time)

        exp_time_seconds = exp_time * 24 * 60 * 60
        assert 15 < exp_time_seconds < 25

    def test_handles_short_array(self) -> None:
        """Returns default for very short arrays."""
        from tess_vetter.transit.batman_model import detect_exposure_time

        time = np.array([0.0])
        exp_time = detect_exposure_time(time)

        # Should return default (2 min in days)
        assert exp_time > 0


@requires_batman
class TestComputeBatmanModel:
    """Tests for batman light curve computation."""

    def test_generates_normalized_model(self) -> None:
        """Model is normalized to 1.0 out of transit."""
        from tess_vetter.transit import compute_batman_model

        time = np.linspace(-0.1, 0.1, 1000) + 100.0  # Around t0=100
        model = compute_batman_model(
            time,
            period=5.0,
            t0=100.0,
            rp_rs=0.1,
            a_rs=15.0,
            inc=88.0,
            u=(0.3, 0.2),
        )

        # Out-of-transit should be ~1.0
        oot_mask = np.abs(time - 100.0) > 0.08
        assert np.allclose(model[oot_mask], 1.0, atol=1e-6)

    def test_transit_depth_correct(self) -> None:
        """Transit depth approximately matches (Rp/Rs)^2."""
        from tess_vetter.transit import compute_batman_model

        rp_rs = 0.1  # 1% depth expected
        time = np.linspace(-0.05, 0.05, 500) + 100.0
        model = compute_batman_model(
            time,
            period=5.0,
            t0=100.0,
            rp_rs=rp_rs,
            a_rs=15.0,
            inc=90.0,  # Central transit
            u=(0.0, 0.0),  # No limb darkening for simple test
        )

        depth = 1.0 - np.min(model)
        expected_depth = rp_rs**2

        # Should be close (within 10% due to exposure time integration)
        assert abs(depth - expected_depth) < 0.1 * expected_depth

    def test_limb_darkening_affects_shape(self) -> None:
        """Limb darkening changes transit shape."""
        from tess_vetter.transit import compute_batman_model

        time = np.linspace(-0.05, 0.05, 500) + 100.0

        # Without limb darkening
        model_no_ld = compute_batman_model(
            time,
            period=5.0,
            t0=100.0,
            rp_rs=0.1,
            a_rs=15.0,
            inc=90.0,
            u=(0.0, 0.0),
        )

        # With limb darkening
        model_with_ld = compute_batman_model(
            time,
            period=5.0,
            t0=100.0,
            rp_rs=0.1,
            a_rs=15.0,
            inc=90.0,
            u=(0.4, 0.2),
        )

        # Shapes should differ
        assert not np.allclose(model_no_ld, model_with_ld)

        # Limb-darkened transit is typically deeper at bottom
        assert np.min(model_with_ld) < np.min(model_no_ld)


class TestComputeDerivedParameters:
    """Tests for derived parameter calculations."""

    def test_transit_depth_from_rp_rs(self) -> None:
        """Correctly computes transit depth in ppm."""
        from tess_vetter.transit import compute_derived_parameters

        result = compute_derived_parameters(
            rp_rs=0.1,  # 1% depth
            a_rs=15.0,
            inc=88.0,
            period=5.0,
        )

        assert "transit_depth_ppm" in result
        assert abs(result["transit_depth_ppm"] - 10000) < 100

    def test_impact_parameter_calculation(self) -> None:
        """Correctly computes impact parameter."""
        from tess_vetter.transit import compute_derived_parameters

        # inc=90 -> b=0 (central transit)
        result_central = compute_derived_parameters(
            rp_rs=0.1,
            a_rs=15.0,
            inc=90.0,
            period=5.0,
        )
        assert abs(result_central["impact_parameter"]) < 0.01

        # inc=87 -> b ~ a/Rs * cos(87 deg) ~ 0.78
        result_grazing = compute_derived_parameters(
            rp_rs=0.1,
            a_rs=15.0,
            inc=87.0,
            period=5.0,
        )
        expected_b = 15.0 * np.cos(np.radians(87.0))
        assert abs(result_grazing["impact_parameter"] - expected_b) < 0.1

    def test_stellar_density_calculation(self) -> None:
        """Correctly computes stellar density."""
        from tess_vetter.transit import compute_derived_parameters

        # For solar-like star: a/Rs ~ 14.7 for P=5d
        result = compute_derived_parameters(
            rp_rs=0.1,
            a_rs=14.7,
            inc=88.0,
            period=5.0,
        )

        assert "stellar_density_gcc" in result
        # Should be close to solar density (1.41 g/cm^3)
        assert 0.5 < result["stellar_density_gcc"] < 3.0


@requires_batman
class TestFitTransitModelOptimize:
    """Tests for optimize fitting method."""

    @pytest.fixture
    def synthetic_transit_lc(self) -> dict[str, NDArray[np.float64] | float]:
        """Generate synthetic light curve with known transit parameters."""
        np.random.seed(42)

        # True parameters
        period = 5.0
        t0 = 100.0
        rp_rs = 0.1
        a_rs = 15.0
        inc = 88.0
        u = (0.3, 0.2)

        # Generate time array
        time = np.linspace(95, 105, 5000, dtype=np.float64)

        # Generate transit model
        try:
            from tess_vetter.transit import compute_batman_model

            flux = compute_batman_model(time, period, t0, rp_rs, a_rs, inc, u)
        except ImportError:
            # Skip if batman not available
            pytest.skip("batman-package not installed")

        # Add noise
        noise_level = 0.0005
        flux = flux + np.random.normal(0, noise_level, len(flux))
        flux_err = np.ones_like(flux) * noise_level

        return {
            "time": time,
            "flux": flux.astype(np.float64),
            "flux_err": flux_err.astype(np.float64),
            "period": period,
            "t0": t0,
            "rp_rs": rp_rs,
            "a_rs": a_rs,
            "inc": inc,
        }

    def test_recovers_rp_rs(
        self, synthetic_transit_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Recovers Rp/Rs to within 10%."""
        try:
            from tess_vetter.transit import fit_transit_model
        except ImportError:
            pytest.skip("batman-package not installed")

        time = synthetic_transit_lc["time"]
        flux = synthetic_transit_lc["flux"]
        flux_err = synthetic_transit_lc["flux_err"]
        period = synthetic_transit_lc["period"]
        t0 = synthetic_transit_lc["t0"]
        true_rp_rs = synthetic_transit_lc["rp_rs"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(true_rp_rs, float)

        result = fit_transit_model(
            time=time,
            flux=flux,
            flux_err=flux_err,
            period=period,
            t0=t0,
            stellar_params={"teff": 5800, "logg": 4.44, "feh": 0.0},
            method="optimize",
        )

        assert result.converged
        # Rp/Rs should be within 10% of true value
        relative_error = abs(result.rp_rs.value - true_rp_rs) / true_rp_rs
        assert relative_error < 0.10

    def test_returns_valid_result(
        self, synthetic_transit_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """Returns complete TransitFitResult."""
        try:
            from tess_vetter.transit import fit_transit_model
        except ImportError:
            pytest.skip("batman-package not installed")

        time = synthetic_transit_lc["time"]
        flux = synthetic_transit_lc["flux"]
        flux_err = synthetic_transit_lc["flux_err"]
        period = synthetic_transit_lc["period"]
        t0 = synthetic_transit_lc["t0"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)

        result = fit_transit_model(
            time=time,
            flux=flux,
            flux_err=flux_err,
            period=period,
            t0=t0,
            stellar_params={"teff": 5800, "logg": 4.44, "feh": 0.0},
            method="optimize",
        )

        # Check all required fields
        assert result.fit_method == "optimize"
        assert result.rp_rs.value > 0
        assert result.a_rs.value > 0
        assert 70 < result.inc.value <= 90
        assert result.chi_squared > 0
        assert result.rms_ppm > 0
        assert len(result.phase) > 0
        assert len(result.flux_model) > 0


@requires_batman
class TestFitTransitModelMCMC:
    """Tests for MCMC fitting method (marked slow)."""

    @pytest.fixture
    def synthetic_transit_lc(self) -> dict[str, NDArray[np.float64] | float]:
        """Generate synthetic light curve with known transit parameters."""
        np.random.seed(42)

        # True parameters
        period = 5.0
        t0 = 100.0
        rp_rs = 0.1
        a_rs = 15.0
        inc = 88.0
        u = (0.3, 0.2)

        # Generate shorter time array for faster MCMC tests
        time = np.linspace(99.5, 100.5, 1000, dtype=np.float64)

        try:
            from tess_vetter.transit import compute_batman_model

            flux = compute_batman_model(time, period, t0, rp_rs, a_rs, inc, u)
        except ImportError:
            pytest.skip("batman-package not installed")

        noise_level = 0.0003
        flux = flux + np.random.normal(0, noise_level, len(flux))
        flux_err = np.ones_like(flux) * noise_level

        return {
            "time": time,
            "flux": flux.astype(np.float64),
            "flux_err": flux_err.astype(np.float64),
            "period": period,
            "t0": t0,
            "rp_rs": rp_rs,
            "a_rs": a_rs,
            "inc": inc,
        }

    @pytest.mark.slow
    def test_mcmc_provides_uncertainties(
        self, synthetic_transit_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """MCMC provides meaningful uncertainties."""
        try:
            from tess_vetter.transit import fit_transit_model
        except ImportError:
            pytest.skip("batman-package and emcee not installed")
        try:
            import arviz  # noqa: F401
        except ImportError:
            pytest.skip("arviz not installed")

        time = synthetic_transit_lc["time"]
        flux = synthetic_transit_lc["flux"]
        flux_err = synthetic_transit_lc["flux_err"]
        period = synthetic_transit_lc["period"]
        t0 = synthetic_transit_lc["t0"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)

        result = fit_transit_model(
            time=time,
            flux=flux,
            flux_err=flux_err,
            period=period,
            t0=t0,
            stellar_params={"teff": 5800, "logg": 4.44, "feh": 0.0},
            method="mcmc",
            mcmc_samples=500,  # Small for testing
            mcmc_burn=100,
        )

        # MCMC should provide credible intervals
        assert result.rp_rs.credible_interval_68 is not None
        assert result.a_rs.credible_interval_68 is not None

        # Uncertainties should be positive
        assert result.rp_rs.uncertainty > 0
        assert result.a_rs.uncertainty > 0
        assert result.inc.uncertainty > 0

    @pytest.mark.slow
    def test_mcmc_diagnostics(
        self, synthetic_transit_lc: dict[str, NDArray[np.float64] | float]
    ) -> None:
        """MCMC provides convergence diagnostics."""
        try:
            from tess_vetter.transit import fit_transit_model
        except ImportError:
            pytest.skip("batman-package, emcee, and arviz not installed")
        try:
            import arviz  # noqa: F401
        except ImportError:
            pytest.skip("arviz not installed")

        time = synthetic_transit_lc["time"]
        flux = synthetic_transit_lc["flux"]
        flux_err = synthetic_transit_lc["flux_err"]
        period = synthetic_transit_lc["period"]
        t0 = synthetic_transit_lc["t0"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)

        result = fit_transit_model(
            time=time,
            flux=flux,
            flux_err=flux_err,
            period=period,
            t0=t0,
            stellar_params={"teff": 5800, "logg": 4.44, "feh": 0.0},
            method="mcmc",
            mcmc_samples=500,
            mcmc_burn=100,
        )

        # Should have diagnostics
        assert result.mcmc_diagnostics is not None
        assert "gelman_rubin" in result.mcmc_diagnostics
        assert "acceptance_rate" in result.mcmc_diagnostics

        # Gelman-Rubin should be close to 1.0 for converged chains
        for param, rhat in result.mcmc_diagnostics["gelman_rubin"].items():
            # May not fully converge with few samples, but should be < 1.5
            assert rhat < 1.5, f"R-hat for {param} too high: {rhat}"


class TestGetLDCoefficients:
    """Tests for limb darkening coefficient retrieval."""

    def test_returns_coefficients_and_errors(self) -> None:
        """Returns both coefficients and uncertainties."""
        from tess_vetter.transit import get_ld_coefficients

        (u1, u2), (u1_err, u2_err) = get_ld_coefficients(teff=5800, logg=4.44, feh=0.0)

        # Should return values in expected range
        assert 0 < u1 < 1
        assert -0.5 < u2 < 1
        assert u1_err > 0
        assert u2_err > 0

    def test_solar_values_reasonable(self) -> None:
        """Solar-like star has reasonable LD coefficients."""
        from tess_vetter.transit import get_ld_coefficients

        (u1, u2), _ = get_ld_coefficients(teff=5800, logg=4.44, feh=0.0)

        # For TESS band, solar-like star should have u1 ~ 0.3-0.4
        assert 0.2 < u1 < 0.5
        assert 0.1 < u2 < 0.4


class TestTransitFitResultSerialization:
    """Tests for result serialization."""

    def test_to_dict_returns_valid_structure(self) -> None:
        """to_dict returns proper dictionary structure."""
        from tess_vetter.transit import ParameterEstimate, TransitFitResult

        result = TransitFitResult(
            fit_method="optimize",
            stellar_params={"teff": 5800, "logg": 4.44, "feh": 0.0},
            rp_rs=ParameterEstimate(0.1, 0.005),
            a_rs=ParameterEstimate(15.0, 1.0),
            inc=ParameterEstimate(88.0, 0.5),
            t0=ParameterEstimate(100.0, 0.001),
            u1=ParameterEstimate(0.3, 0.05),
            u2=ParameterEstimate(0.2, 0.05),
            transit_depth_ppm=10000,
            duration_hours=3.5,
            impact_parameter=0.5,
            stellar_density_gcc=1.41,
            chi_squared=1.02,
            bic=1234.5,
            rms_ppm=312,
            phase=[0.0, 0.1, 0.2],
            flux_model=[1.0, 0.99, 1.0],
            flux_data=[1.0, 0.989, 1.001],
            flux_err=[0.001, 0.001, 0.001],
            mcmc_diagnostics=None,
            converged=True,
        )

        d = result.to_dict()

        assert d["status"] == "success"
        assert d["fit_method"] == "optimize"
        assert "parameters" in d
        assert "rp_rs" in d["parameters"]
        assert "derived" in d
        assert "goodness_of_fit" in d
        assert "model_lc" in d
