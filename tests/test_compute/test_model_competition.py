"""Unit tests for model competition and artifact detection.

Tests the model_competition module which provides:
- Model fitting (transit-only, transit+sinusoid, EB-like)
- Model competition via BIC comparison
- Artifact prior computation based on known TESS systematics
- Period alias detection

Phase 3.6 deliverable - reduces false positives by detecting non-transit signals.

All tests are deterministic and require no network or file I/O.
"""

from __future__ import annotations

import numpy as np
import pytest

from bittr_tess_vetter.compute.model_competition import (
    KNOWN_ARTIFACT_PERIODS,
    ArtifactPrior,
    ModelCompetitionResult,
    ModelFit,
    check_period_alias,
    compute_artifact_prior,
    fit_eb_like,
    fit_transit_only,
    fit_transit_sinusoid,
    run_model_competition,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def time_array() -> np.ndarray:
    """Synthetic time array spanning 27 days with cadence ~2 minutes."""
    return np.linspace(1000.0, 1027.0, 20000)


@pytest.fixture
def flux_err(time_array: np.ndarray) -> np.ndarray:
    """Flux uncertainty array (constant 100 ppm)."""
    return np.full_like(time_array, 100e-6)


def make_box_transit(
    time: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
    depth: float,
) -> np.ndarray:
    """Create synthetic box transit light curve."""
    duration_days = duration_hours / 24.0
    half_dur = duration_days / 2.0
    phase = (time - t0) / period
    phase = phase - np.floor(phase + 0.5)
    in_transit = np.abs(phase * period) < half_dur
    flux = np.ones_like(time)
    flux[in_transit] = 1.0 - depth
    return flux


def make_sinusoid(
    time: np.ndarray,
    period: float,
    amplitude: float,
    phase_rad: float = 0.0,
) -> np.ndarray:
    """Create sinusoidal variability."""
    return amplitude * np.sin(2 * np.pi * time / period + phase_rad)


def make_eb_odd_even(
    time: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
    depth_odd: float,
    depth_even: float,
    depth_secondary: float = 0.0,
) -> np.ndarray:
    """Create EB-like light curve with odd/even depth difference and optional secondary."""
    duration_days = duration_hours / 24.0
    half_dur = duration_days / 2.0
    flux = np.ones_like(time)

    # Phase calculation
    phase = (time - t0) / period
    phase_centered = phase - np.floor(phase + 0.5)
    orbit_number = np.floor(phase + 0.5)
    is_odd = (orbit_number.astype(int) % 2) == 1

    # Primary transits
    in_primary = np.abs(phase_centered * period) < half_dur
    flux[in_primary & is_odd] = 1.0 - depth_odd
    flux[in_primary & ~is_odd] = 1.0 - depth_even

    # Secondary eclipse at phase 0.5
    phase_secondary = phase_centered - 0.5
    phase_secondary = phase_secondary - np.round(phase_secondary)
    in_secondary = np.abs(phase_secondary * period) < half_dur
    flux[in_secondary] = 1.0 - depth_secondary

    return flux


# =============================================================================
# Test ModelFit Dataclass
# =============================================================================


class TestModelFit:
    """Tests for the ModelFit dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        fit = ModelFit(
            model_type="transit_only",
            n_params=1,
            log_likelihood=-100.0,
            aic=202.0,
            bic=205.0,
            residual_rms=0.0001,
            fitted_params={"depth": 0.001},
        )
        d = fit.to_dict()
        assert d["model_type"] == "transit_only"
        assert d["n_params"] == 1
        assert d["log_likelihood"] == -100.0
        assert d["aic"] == 202.0
        assert d["bic"] == 205.0
        assert d["residual_rms"] == 0.0001
        assert d["fitted_params"]["depth"] == 0.001

    def test_model_types(self):
        """Test valid model types."""
        for model_type in ["transit_only", "transit_sinusoid", "eb_like"]:
            fit = ModelFit(
                model_type=model_type,
                n_params=1,
                log_likelihood=-100.0,
                aic=202.0,
                bic=205.0,
                residual_rms=0.0001,
                fitted_params={},
            )
            assert fit.model_type == model_type


# =============================================================================
# Test ModelCompetitionResult Dataclass
# =============================================================================


class TestModelCompetitionResult:
    """Tests for the ModelCompetitionResult dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        transit_fit = ModelFit(
            model_type="transit_only",
            n_params=1,
            log_likelihood=-100.0,
            aic=202.0,
            bic=205.0,
            residual_rms=0.0001,
            fitted_params={"depth": 0.001},
        )
        result = ModelCompetitionResult(
            fits={"transit_only": transit_fit},
            winner="transit_only",
            winner_margin=15.0,
            model_competition_label="TRANSIT",
            artifact_risk=0.0,
            warnings=[],
        )
        d = result.to_dict()
        assert d["winner"] == "transit_only"
        assert d["winner_margin"] == 15.0
        assert d["model_competition_label"] == "TRANSIT"
        assert d["artifact_risk"] == 0.0
        assert "transit_only" in d["fits"]

    def test_with_warnings(self):
        """Test result with warnings included."""
        transit_fit = ModelFit(
            model_type="transit_only",
            n_params=1,
            log_likelihood=-100.0,
            aic=202.0,
            bic=205.0,
            residual_rms=0.0001,
            fitted_params={},
        )
        result = ModelCompetitionResult(
            fits={"transit_only": transit_fit},
            winner="transit_only",
            winner_margin=5.0,
            model_competition_label="AMBIGUOUS",
            artifact_risk=0.5,
            warnings=["Model competition inconclusive"],
        )
        d = result.to_dict()
        assert len(d["warnings"]) == 1
        assert "inconclusive" in d["warnings"][0]


# =============================================================================
# Test ArtifactPrior Dataclass
# =============================================================================


class TestArtifactPrior:
    """Tests for the ArtifactPrior dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        prior = ArtifactPrior(
            period_alias_risk=0.8,
            sector_quality_risk=0.2,
            scattered_light_risk=0.0,
            combined_risk=0.5,
        )
        d = prior.to_dict()
        assert d["period_alias_risk"] == 0.8
        assert d["sector_quality_risk"] == 0.2
        assert d["scattered_light_risk"] == 0.0
        assert d["combined_risk"] == 0.5


# =============================================================================
# Test fit_transit_only
# =============================================================================


class TestFitTransitOnly:
    """Tests for the transit-only model fitting function."""

    def test_fits_clean_transit(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test fitting a clean synthetic transit."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        true_depth = 0.001

        flux = make_box_transit(time_array, period, t0, duration_hours, true_depth)
        flux += rng.normal(0, 50e-6, len(flux))

        result = fit_transit_only(
            time_array, flux, flux_err, period, t0, duration_hours
        )

        assert result.model_type == "transit_only"
        assert result.n_params == 1
        # Fitted depth should be close to true depth
        assert abs(result.fitted_params["depth"] - true_depth) < 0.0002
        assert result.fitted_params["depth_ppm"] == pytest.approx(
            result.fitted_params["depth"] * 1e6, rel=1e-10
        )
        assert result.residual_rms > 0
        assert np.isfinite(result.log_likelihood)
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)

    def test_no_transit(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test fitting when there's no transit (flat light curve)."""
        flux = np.ones_like(time_array) + rng.normal(0, 50e-6, len(time_array))

        result = fit_transit_only(
            time_array, flux, flux_err, period=3.5, t0=1001.0, duration_hours=2.5
        )

        # Depth should be near zero
        assert abs(result.fitted_params["depth"]) < 0.0002

    def test_handles_no_in_transit_points(self):
        """Test graceful handling when no points fall in transit."""
        # Very short time array that doesn't span a full period
        time = np.linspace(1000.0, 1000.5, 100)
        flux = np.ones(100)
        flux_err = np.full(100, 100e-6)

        result = fit_transit_only(
            time, flux, flux_err, period=10.0, t0=990.0, duration_hours=1.0
        )

        # Should return zero depth without error
        assert result.fitted_params["depth"] == 0.0


# =============================================================================
# Test fit_transit_sinusoid
# =============================================================================


class TestFitTransitSinusoid:
    """Tests for the transit + sinusoid model fitting function."""

    def test_fits_transit_with_sinusoid(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test fitting transit with sinusoidal variability."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        true_depth = 0.001
        sin_amplitude = 0.0005

        flux = make_box_transit(time_array, period, t0, duration_hours, true_depth)
        flux += make_sinusoid(time_array, period, sin_amplitude)
        flux += rng.normal(0, 50e-6, len(flux))

        result = fit_transit_sinusoid(
            time_array, flux, flux_err, period, t0, duration_hours, n_harmonics=2
        )

        assert result.model_type == "transit_sinusoid"
        assert result.n_params == 5  # depth + 2 harmonics * 2 coefficients
        # Depth fit may differ due to sinusoid
        assert np.isfinite(result.fitted_params["depth"])
        assert "amplitude_k1" in result.fitted_params
        assert "phase_k1_rad" in result.fitted_params
        assert "amplitude_k2" in result.fitted_params
        assert result.fitted_params["n_harmonics"] == 2

    def test_pure_sinusoid(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test fitting pure sinusoidal variability (no transit)."""
        period = 3.5
        sin_amplitude = 0.001

        flux = 1.0 + make_sinusoid(time_array, period, sin_amplitude)
        flux += rng.normal(0, 50e-6, len(flux))

        result = fit_transit_sinusoid(
            time_array, flux, flux_err, period, t0=1001.0, duration_hours=2.5
        )

        # Should detect the sinusoidal amplitude
        assert result.fitted_params["amplitude_k1"] > 0.0005

    def test_residual_rms_improves_over_transit_only(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that sinusoid model has lower residual RMS when sinusoid present."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        true_depth = 0.001
        sin_amplitude = 0.001

        flux = make_box_transit(time_array, period, t0, duration_hours, true_depth)
        flux += make_sinusoid(time_array, period, sin_amplitude)
        flux += rng.normal(0, 50e-6, len(flux))

        result_transit = fit_transit_only(
            time_array, flux, flux_err, period, t0, duration_hours
        )
        result_sinusoid = fit_transit_sinusoid(
            time_array, flux, flux_err, period, t0, duration_hours
        )

        # Sinusoid model should have lower residual RMS
        assert result_sinusoid.residual_rms < result_transit.residual_rms


# =============================================================================
# Test fit_eb_like
# =============================================================================


class TestFitEbLike:
    """Tests for the EB-like model fitting function."""

    def test_fits_eb_with_odd_even_difference(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test fitting EB with odd/even depth difference."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        depth_odd = 0.002
        depth_even = 0.001

        flux = make_eb_odd_even(
            time_array, period, t0, duration_hours, depth_odd, depth_even
        )
        flux += rng.normal(0, 50e-6, len(flux))

        result = fit_eb_like(
            time_array, flux, flux_err, period, t0, duration_hours
        )

        assert result.model_type == "eb_like"
        assert result.n_params == 3  # depth_odd, depth_even, depth_secondary
        # Should detect odd/even difference
        assert result.fitted_params["depth_odd"] > result.fitted_params["depth_even"]
        assert result.fitted_params["odd_even_diff_frac"] > 0.1

    def test_fits_eb_with_secondary_eclipse(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test fitting EB with secondary eclipse."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        depth_primary = 0.002
        depth_secondary = 0.0008

        flux = make_eb_odd_even(
            time_array, period, t0, duration_hours,
            depth_primary, depth_primary, depth_secondary
        )
        flux += rng.normal(0, 50e-6, len(flux))

        result = fit_eb_like(
            time_array, flux, flux_err, period, t0, duration_hours
        )

        # Should detect secondary eclipse
        assert result.fitted_params["depth_secondary"] > 0.0005
        assert result.fitted_params["depth_secondary_ppm"] > 500

    def test_symmetric_transit_gives_equal_odd_even(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that symmetric transit gives similar odd/even depths."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        true_depth = 0.001

        flux = make_box_transit(time_array, period, t0, duration_hours, true_depth)
        flux += rng.normal(0, 50e-6, len(flux))

        result = fit_eb_like(
            time_array, flux, flux_err, period, t0, duration_hours
        )

        # Odd and even depths should be similar
        depth_diff = abs(
            result.fitted_params["depth_odd"] - result.fitted_params["depth_even"]
        )
        assert depth_diff < 0.0003


# =============================================================================
# Test run_model_competition
# =============================================================================


class TestRunModelCompetition:
    """Tests for the model competition function."""

    def test_transit_wins_for_clean_transit(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that transit-only model wins for clean transit signal."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        true_depth = 0.001

        flux = make_box_transit(time_array, period, t0, duration_hours, true_depth)
        flux += rng.normal(0, 50e-6, len(flux))

        result = run_model_competition(
            time_array, flux, flux_err, period, t0, duration_hours
        )

        assert result.winner == "transit_only"
        assert result.model_competition_label == "TRANSIT"
        assert result.artifact_risk == 0.0
        assert len(result.fits) == 3

    def test_sinusoid_wins_for_stellar_variability(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that sinusoid model wins for stellar variability."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        sin_amplitude = 0.002

        # Strong sinusoidal signal with weak transit
        flux = 1.0 + make_sinusoid(time_array, period, sin_amplitude)
        flux += make_box_transit(time_array, period, t0, duration_hours, 0.0001) - 1.0
        flux += rng.normal(0, 50e-6, len(flux))

        result = run_model_competition(
            time_array, flux, flux_err, period, t0, duration_hours
        )

        assert result.winner == "transit_sinusoid"
        assert result.model_competition_label == "SINUSOID"
        assert result.artifact_risk > 0.5
        assert any("Sinusoidal" in w for w in result.warnings)

    def test_eb_wins_for_odd_even_difference(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that EB model wins for odd/even depth difference."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        depth_odd = 0.003
        depth_even = 0.001

        flux = make_eb_odd_even(
            time_array, period, t0, duration_hours, depth_odd, depth_even
        )
        flux += rng.normal(0, 50e-6, len(flux))

        result = run_model_competition(
            time_array, flux, flux_err, period, t0, duration_hours
        )

        assert result.winner == "eb_like"
        assert result.model_competition_label == "EB_LIKE"
        assert result.artifact_risk > 0.8
        assert any("odd/even" in w.lower() for w in result.warnings)

    def test_ambiguous_result(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test ambiguous result when models are similar."""
        # Very weak signal leads to ambiguous result
        flux = np.ones_like(time_array) + rng.normal(0, 100e-6, len(time_array))

        result = run_model_competition(
            time_array, flux, flux_err, period=3.5, t0=1001.0, duration_hours=2.5,
            bic_threshold=10.0
        )

        # With weak signal, margin likely small
        if result.winner_margin < 10.0:
            assert result.model_competition_label == "AMBIGUOUS"
            assert result.artifact_risk == 0.5

    def test_custom_bic_threshold(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test custom BIC threshold changes ambiguity classification."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        true_depth = 0.001

        flux = make_box_transit(time_array, period, t0, duration_hours, true_depth)
        flux += rng.normal(0, 50e-6, len(flux))

        # Very high threshold should make result ambiguous
        result_high = run_model_competition(
            time_array, flux, flux_err, period, t0, duration_hours,
            bic_threshold=1000.0
        )

        # Low threshold should give clear winner
        result_low = run_model_competition(
            time_array, flux, flux_err, period, t0, duration_hours,
            bic_threshold=0.1
        )

        assert result_high.model_competition_label == "AMBIGUOUS"
        assert result_low.model_competition_label != "AMBIGUOUS"

    def test_custom_n_harmonics(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test custom number of harmonics for sinusoid model."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        true_depth = 0.001

        flux = make_box_transit(time_array, period, t0, duration_hours, true_depth)
        flux += rng.normal(0, 50e-6, len(flux))

        result = run_model_competition(
            time_array, flux, flux_err, period, t0, duration_hours,
            n_harmonics=3
        )

        # Check sinusoid model has more parameters
        assert result.fits["transit_sinusoid"].n_params == 7  # 1 + 3*2

    def test_result_serialization(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test that result can be serialized to dict."""
        period = 3.5
        t0 = 1001.0
        duration_hours = 2.5
        true_depth = 0.001

        flux = make_box_transit(time_array, period, t0, duration_hours, true_depth)
        flux += rng.normal(0, 50e-6, len(flux))

        result = run_model_competition(
            time_array, flux, flux_err, period, t0, duration_hours
        )

        d = result.to_dict()
        assert "fits" in d
        assert "winner" in d
        assert "winner_margin" in d
        assert "model_competition_label" in d
        assert "artifact_risk" in d
        assert "warnings" in d


# =============================================================================
# Test check_period_alias
# =============================================================================


class TestCheckPeriodAlias:
    """Tests for period alias detection."""

    def test_detects_known_artifact_periods(self):
        """Test detection of known TESS systematic periods."""
        for known_period in KNOWN_ARTIFACT_PERIODS:
            is_alias, closest, frac_diff = check_period_alias(known_period)
            assert is_alias is True
            assert closest == pytest.approx(known_period)
            assert frac_diff < 0.001

    def test_detects_near_artifact_periods(self):
        """Test detection of periods near known artifacts."""
        # 1% off spacecraft orbital period
        test_period = 13.7 * 1.005
        is_alias, closest, frac_diff = check_period_alias(test_period, tolerance=0.01)
        assert is_alias is True
        assert closest == 13.7
        assert frac_diff < 0.01

    def test_rejects_distant_periods(self):
        """Test that distant periods are not flagged."""
        test_period = 5.0  # Not near any known artifact period
        is_alias, closest, frac_diff = check_period_alias(test_period)
        assert is_alias is False
        assert closest is not None  # Still reports closest
        assert frac_diff > 0.01

    def test_custom_known_periods(self):
        """Test with custom list of known periods."""
        custom_periods = [1.234, 5.678]
        is_alias, closest, frac_diff = check_period_alias(
            1.234, known_periods=custom_periods
        )
        assert is_alias is True
        assert closest == 1.234

    def test_custom_tolerance(self):
        """Test with custom fractional tolerance."""
        test_period = 13.7 * 1.05  # 5% off
        # Default tolerance (1%) should reject
        is_alias_default, _, _ = check_period_alias(test_period, tolerance=0.01)
        # Larger tolerance (10%) should accept
        is_alias_large, _, _ = check_period_alias(test_period, tolerance=0.10)
        assert is_alias_default is False
        assert is_alias_large is True

    def test_invalid_period(self):
        """Test handling of invalid (non-positive) period."""
        is_alias, closest, frac_diff = check_period_alias(0.0)
        assert is_alias is False
        assert closest is None
        assert frac_diff == float("inf")

        is_alias, closest, frac_diff = check_period_alias(-1.0)
        assert is_alias is False

    def test_empty_known_periods(self):
        """Test with empty list of known periods."""
        is_alias, closest, frac_diff = check_period_alias(13.7, known_periods=[])
        assert is_alias is False
        assert closest is None


# =============================================================================
# Test compute_artifact_prior
# =============================================================================


class TestComputeArtifactPrior:
    """Tests for artifact prior computation."""

    def test_high_risk_for_artifact_period(self):
        """Test high risk score for known artifact period."""
        prior = compute_artifact_prior(13.7)  # Spacecraft orbital period
        assert prior.period_alias_risk > 0.9
        assert prior.combined_risk > 0.4

    def test_low_risk_for_clean_period(self):
        """Test low risk score for period far from artifacts."""
        prior = compute_artifact_prior(5.0)
        assert prior.period_alias_risk < 0.5
        assert prior.combined_risk < 0.5

    def test_quality_flags_increase_risk(self):
        """Test that quality flags increase risk."""
        prior_clean = compute_artifact_prior(5.0, quality_flags=None)
        prior_scattered = compute_artifact_prior(
            5.0, quality_flags={"scattered_light": True}
        )
        prior_background = compute_artifact_prior(
            5.0, quality_flags={"high_background": True}
        )
        prior_momentum = compute_artifact_prior(
            5.0, quality_flags={"momentum_dump": True}
        )

        assert prior_scattered.sector_quality_risk == 0.5
        assert prior_background.sector_quality_risk == 0.3
        assert prior_momentum.sector_quality_risk == 0.2
        assert prior_scattered.combined_risk > prior_clean.combined_risk

    def test_combined_quality_flags(self):
        """Test combined quality flags take maximum."""
        prior = compute_artifact_prior(
            5.0,
            quality_flags={
                "scattered_light": True,
                "high_background": True,
            }
        )
        # scattered_light (0.5) > high_background (0.3), so max is 0.5
        assert prior.sector_quality_risk == 0.5

    def test_custom_alias_tolerance(self):
        """Test custom alias tolerance affects risk."""
        # Period 5% off artifact
        test_period = 13.7 * 1.05
        prior_strict = compute_artifact_prior(test_period, alias_tolerance=0.01)
        prior_loose = compute_artifact_prior(test_period, alias_tolerance=0.10)

        # Stricter tolerance should give lower alias risk (not a match)
        assert prior_loose.period_alias_risk > prior_strict.period_alias_risk

    def test_serialization(self):
        """Test prior can be serialized to dict."""
        prior = compute_artifact_prior(13.7)
        d = prior.to_dict()
        assert "period_alias_risk" in d
        assert "sector_quality_risk" in d
        assert "scattered_light_risk" in d
        assert "combined_risk" in d


# =============================================================================
# Test Known Artifact Periods
# =============================================================================


class TestKnownArtifactPeriods:
    """Tests for the KNOWN_ARTIFACT_PERIODS constant."""

    def test_contains_spacecraft_orbital_period(self):
        """Test that spacecraft orbital period (~13.7 days) is included."""
        assert any(abs(p - 13.7) < 0.1 for p in KNOWN_ARTIFACT_PERIODS)

    def test_contains_daily_systematics(self):
        """Test that daily systematics period is included."""
        assert any(abs(p - 1.0) < 0.01 for p in KNOWN_ARTIFACT_PERIODS)

    def test_contains_half_day(self):
        """Test that half-day period is included."""
        assert any(abs(p - 0.5) < 0.01 for p in KNOWN_ARTIFACT_PERIODS)

    def test_all_positive(self):
        """Test that all known periods are positive."""
        assert all(p > 0 for p in KNOWN_ARTIFACT_PERIODS)


# =============================================================================
# Test Integration
# =============================================================================


class TestModelCompetitionIntegration:
    """Integration tests for the complete model competition workflow."""

    def test_full_workflow_clean_transit(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test full workflow for a clean planetary transit."""
        period = 5.0  # Not near artifact periods
        t0 = 1002.0
        duration_hours = 3.0
        true_depth = 0.0015

        flux = make_box_transit(time_array, period, t0, duration_hours, true_depth)
        flux += rng.normal(0, 50e-6, len(flux))

        # Run model competition
        result = run_model_competition(
            time_array, flux, flux_err, period, t0, duration_hours
        )

        # Check artifact prior
        prior = compute_artifact_prior(period)

        # Verify clean transit classification
        assert result.winner == "transit_only"
        assert result.artifact_risk < 0.1
        assert prior.period_alias_risk < 0.5

    def test_full_workflow_artifact_detection(
        self,
        time_array: np.ndarray,
        flux_err: np.ndarray,
        rng: np.random.Generator,
    ):
        """Test full workflow for artifact detection at known period."""
        period = 13.7  # Spacecraft orbital period
        t0 = 1002.0
        duration_hours = 3.0
        true_depth = 0.001

        flux = make_box_transit(time_array, period, t0, duration_hours, true_depth)
        flux += make_sinusoid(time_array, period, 0.0008)
        flux += rng.normal(0, 50e-6, len(flux))

        # Run model competition
        result = run_model_competition(
            time_array, flux, flux_err, period, t0, duration_hours
        )

        # Check artifact prior
        prior = compute_artifact_prior(period)

        # Should flag high artifact risk due to period
        assert prior.period_alias_risk > 0.9
        assert prior.combined_risk > 0.4
