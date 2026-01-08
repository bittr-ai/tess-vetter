"""Tests for V11 (ModShift) and V12 (SWEET) exovetter-based checks.

These tests cover the improved implementations with:
- Fred-gated reliability for ModShift
- Harmonic aliasing detection for SWEET
- Structured warnings and inputs_summary
- Confidence scaling based on data quality
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bittr_tess_vetter.domain.detection import TransitCandidate, VetterCheckResult
from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.validation.base import CheckConfig
from bittr_tess_vetter.validation.exovetter_checks import (
    ModshiftCheck,
    SWEETCheck,
    _classify_fred_regime,
    _compute_inputs_summary,
    _is_likely_folded,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def make_lightcurve():
    """Factory for creating synthetic light curves."""

    def _make(
        n_points: int = 1000,
        baseline_days: float = 27.0,
        cadence_minutes: float = 2.0,
        noise_ppm: float = 100.0,
        seed: int = 42,
    ) -> LightCurveData:
        """Create a synthetic light curve.

        Args:
            n_points: Number of data points
            baseline_days: Total time baseline
            cadence_minutes: Cadence in minutes
            noise_ppm: Gaussian noise level in ppm
            seed: RNG seed

        Returns:
            LightCurveData instance
        """
        rng = np.random.default_rng(seed)

        time = np.linspace(0, baseline_days, n_points)
        flux = np.ones(n_points) + rng.normal(0, noise_ppm * 1e-6, n_points)
        flux_err = np.full(n_points, noise_ppm * 1e-6)
        quality = np.zeros(n_points, dtype=np.int32)
        valid_mask = np.ones(n_points, dtype=np.bool_)

        return LightCurveData(
            time=time,
            flux=flux,
            flux_err=flux_err,
            quality=quality,
            valid_mask=valid_mask,
            tic_id=123456789,
            sector=1,
            cadence_seconds=cadence_minutes * 60,
        )

    return _make


@pytest.fixture
def make_candidate():
    """Factory for creating transit candidates."""

    def _make(
        period: float = 5.0,
        t0: float = 1000.0,
        duration_hours: float = 3.0,
        depth: float = 0.001,
        snr: float = 10.0,
    ) -> TransitCandidate:
        """Create a transit candidate.

        Args:
            period: Orbital period in days
            t0: Transit epoch in BTJD
            duration_hours: Transit duration in hours
            depth: Transit depth (fractional)
            snr: Signal-to-noise ratio

        Returns:
            TransitCandidate instance
        """
        return TransitCandidate(
            period=period,
            t0=t0,
            duration_hours=duration_hours,
            depth=depth,
            snr=snr,
        )

    return _make


@pytest.fixture
def mock_exovetter():
    """Create mock exovetter modules for tests where exovetter is not installed."""
    # Create mock modules
    mock_tce = MagicMock()
    mock_vetters = MagicMock()
    mock_const = MagicMock()
    mock_const.btjd = 2457000.0
    mock_const.ppm = 1e-6

    # Set up mock module hierarchy
    mock_exovetter_module = MagicMock()
    mock_exovetter_module.tce = mock_tce
    mock_exovetter_module.vetters = mock_vetters
    mock_exovetter_module.const = mock_const

    return {
        "exovetter": mock_exovetter_module,
        "exovetter.tce": mock_tce,
        "exovetter.vetters": mock_vetters,
        "exovetter.const": mock_const,
    }


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestClassifyFredRegime:
    """Tests for _classify_fred_regime helper function."""

    def test_low_regime(self) -> None:
        """Fred < 1.5 should be 'low' regime."""
        assert _classify_fred_regime(0.5) == "low"
        assert _classify_fred_regime(1.0) == "low"
        assert _classify_fred_regime(1.49) == "low"

    def test_standard_regime(self) -> None:
        """Fred 1.5-2.5 should be 'standard' regime."""
        assert _classify_fred_regime(1.5) == "standard"
        assert _classify_fred_regime(2.0) == "standard"
        assert _classify_fred_regime(2.49) == "standard"

    def test_high_regime(self) -> None:
        """Fred 2.5-3.5 should be 'high' regime."""
        assert _classify_fred_regime(2.5) == "high"
        assert _classify_fred_regime(3.0) == "high"
        assert _classify_fred_regime(3.49) == "high"

    def test_critical_regime(self) -> None:
        """Fred >= 3.5 should be 'critical' regime."""
        assert _classify_fred_regime(3.5) == "critical"
        assert _classify_fred_regime(5.0) == "critical"
        assert _classify_fred_regime(10.0) == "critical"


class TestIsLikelyFolded:
    """Tests for _is_likely_folded helper function."""

    def test_unfolded_long_baseline(self) -> None:
        """Long baseline relative to period should not be flagged as folded."""
        time = np.linspace(0, 100, 1000)
        assert _is_likely_folded(time, period=5.0) is False

    def test_folded_short_baseline(self) -> None:
        """Short baseline relative to period should be flagged as folded."""
        time = np.linspace(0, 5, 1000)  # Only 1 period
        assert _is_likely_folded(time, period=5.0) is True

    def test_folded_normalized(self) -> None:
        """Time normalized to [0, period] should be flagged as folded."""
        time = np.linspace(0, 4.9, 1000)  # Just under 1 period
        assert _is_likely_folded(time, period=5.0) is True

    def test_empty_array(self) -> None:
        """Empty array should return False."""
        time = np.array([])
        assert _is_likely_folded(time, period=5.0) is False

    def test_single_point(self) -> None:
        """Single point should return False."""
        time = np.array([0.0])
        assert _is_likely_folded(time, period=5.0) is False


class TestComputeInputsSummary:
    """Tests for _compute_inputs_summary helper function."""

    def test_basic_summary(self, make_lightcurve, make_candidate) -> None:
        """Test basic input summary computation."""
        lc = make_lightcurve(n_points=1000, baseline_days=27.0)
        candidate = make_candidate(period=5.0, snr=15.0)

        summary = _compute_inputs_summary(lc, candidate, is_folded=False)

        assert summary["n_points"] == 1000
        assert summary["n_transits_expected"] == 5  # 27 / 5 = 5
        assert summary["baseline_days"] == 27.0
        assert summary["snr"] == 15.0
        assert summary["is_folded"] is False

    def test_folded_flag(self, make_lightcurve, make_candidate) -> None:
        """Test folded flag is passed through."""
        lc = make_lightcurve()
        candidate = make_candidate()

        summary = _compute_inputs_summary(lc, candidate, is_folded=True)
        assert summary["is_folded"] is True

    def test_minimal_data(self, make_candidate) -> None:
        """Test with minimal data (< 2 points)."""
        time = np.array([0.0])
        flux = np.array([1.0])
        flux_err = np.array([0.001])
        quality = np.array([0], dtype=np.int32)
        valid_mask = np.array([True])

        lc = LightCurveData(
            time=time,
            flux=flux,
            flux_err=flux_err,
            quality=quality,
            valid_mask=valid_mask,
            tic_id=123,
            sector=1,
            cadence_seconds=120,
        )
        candidate = make_candidate()

        summary = _compute_inputs_summary(lc, candidate, is_folded=False)

        assert summary["n_points"] == 1
        assert summary["n_transits_expected"] == 0
        assert summary["baseline_days"] == 0.0


# =============================================================================
# V11 ModShift Tests
# =============================================================================


class TestModshiftCheck:
    """Tests for ModshiftCheck (V11)."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        check = ModshiftCheck()
        assert check.id == "V11"
        assert check.name == "modshift"
        assert check.config.threshold == 0.5
        assert check.config.additional["fred_warning_threshold"] == 2.0
        assert check.config.additional["fred_critical_threshold"] == 3.5

    def test_no_lightcurve_returns_pass(self, make_candidate) -> None:
        """Test that missing lightcurve returns pass with low confidence.

        Note: In metrics-only mode (default), passed=None. In legacy_mode, passed=True.
        For skip cases, we still return passed=True for backward compatibility.
        """
        check = ModshiftCheck()
        candidate = make_candidate()

        result = check.run(candidate, lightcurve=None)

        # Skip cases return passed=True even in metrics-only mode
        assert result.passed is True
        assert result.confidence == 0.30
        assert result.details["status"] == "skipped"
        assert "NO_LIGHTCURVE_DATA" in result.details["warnings"]
        assert result.details["passed_meaning"] == "no_strong_eb_evidence"

    def test_folded_input_returns_invalid(self, make_candidate) -> None:
        """Test that folded input is detected and rejected."""
        check = ModshiftCheck()
        candidate = make_candidate(period=5.0)

        # Create a folded-looking light curve (baseline < 1.5 * period)
        time = np.linspace(0, 5, 500)  # Only 1 period
        flux = np.ones(500)
        flux_err = np.full(500, 0.001)
        quality = np.zeros(500, dtype=np.int32)
        valid_mask = np.ones(500, dtype=np.bool_)

        lc = LightCurveData(
            time=time,
            flux=flux,
            flux_err=flux_err,
            quality=quality,
            valid_mask=valid_mask,
            tic_id=123,
            sector=1,
            cadence_seconds=120,
        )

        result = check.run(candidate, lightcurve=lc)

        assert result.passed is True
        assert result.confidence == 0.10
        assert result.details["status"] == "invalid"
        assert "FOLDED_INPUT_DETECTED" in result.details["warnings"]
        assert result.details["fred_regime"] == "unknown"

    @patch("bittr_tess_vetter.validation.exovetter_checks._create_lightkurve_like")
    def test_successful_metrics_only(
        self, mock_create_lk, make_lightcurve, make_candidate, mock_exovetter
    ) -> None:
        """Test ModShift in metrics-only mode (default) - passed is None."""
        mock_lk = MagicMock()
        mock_create_lk.return_value = mock_lk

        # Configure mock vetter
        mock_vetter = MagicMock()
        mock_vetter.run.return_value = {
            "pri": 100.0,
            "sec": 10.0,  # 10% of primary - would pass threshold in legacy mode
            "ter": 5.0,
            "pos": 3.0,
            "Fred": 1.5,  # Standard regime
            "false_alarm_threshold": 20.0,
        }
        mock_exovetter["exovetter.vetters"].ModShift.return_value = mock_vetter

        with patch.dict(sys.modules, mock_exovetter):
            check = ModshiftCheck()
            lc = make_lightcurve(baseline_days=50.0)
            candidate = make_candidate(period=5.0)

            result = check.run(candidate, lightcurve=lc)

            # Metrics-only mode: passed=None
            assert result.passed is None
            assert result.details["_metrics_only"] is True
            assert result.confidence > 0.5
            assert result.details["secondary_primary_ratio"] == 0.1
            assert result.details["fred_regime"] == "standard"
            assert result.details["passed_meaning"] == "no_strong_eb_evidence"
            assert "warnings" in result.details
            assert "inputs_summary" in result.details

    @patch("bittr_tess_vetter.validation.exovetter_checks._create_lightkurve_like")
    def test_successful_pass_legacy_mode(
        self, mock_create_lk, make_lightcurve, make_candidate, mock_exovetter
    ) -> None:
        """Test successful ModShift pass in legacy_mode (no significant secondary)."""
        mock_lk = MagicMock()
        mock_create_lk.return_value = mock_lk

        # Configure mock vetter
        mock_vetter = MagicMock()
        mock_vetter.run.return_value = {
            "pri": 100.0,
            "sec": 10.0,  # 10% of primary - passes threshold
            "ter": 5.0,
            "pos": 3.0,
            "Fred": 1.5,  # Standard regime
            "false_alarm_threshold": 20.0,
        }
        mock_exovetter["exovetter.vetters"].ModShift.return_value = mock_vetter

        with patch.dict(sys.modules, mock_exovetter):
            config = CheckConfig(
                enabled=True,
                threshold=0.5,
                additional={"legacy_mode": True},
            )
            check = ModshiftCheck(config=config)
            lc = make_lightcurve(baseline_days=50.0)
            candidate = make_candidate(period=5.0)

            result = check.run(candidate, lightcurve=lc)

            assert result.passed is True
            assert result.details["_metrics_only"] is False
            assert result.confidence > 0.5
            assert result.details["secondary_primary_ratio"] == 0.1
            assert result.details["fred_regime"] == "standard"
            assert result.details["passed_meaning"] == "no_strong_eb_evidence"
            assert "warnings" in result.details
            assert "inputs_summary" in result.details

    @patch("bittr_tess_vetter.validation.exovetter_checks._create_lightkurve_like")
    def test_successful_fail_eb_detected_legacy_mode(
        self, mock_create_lk, make_lightcurve, make_candidate, mock_exovetter
    ) -> None:
        """Test ModShift fail in legacy_mode (significant secondary eclipse detected)."""
        mock_lk = MagicMock()
        mock_create_lk.return_value = mock_lk

        # Configure mock vetter
        mock_vetter = MagicMock()
        mock_vetter.run.return_value = {
            "pri": 100.0,
            "sec": 60.0,  # 60% of primary - exceeds threshold
            "ter": 5.0,
            "pos": 3.0,
            "Fred": 1.0,
            "false_alarm_threshold": 20.0,
        }
        mock_exovetter["exovetter.vetters"].ModShift.return_value = mock_vetter

        with patch.dict(sys.modules, mock_exovetter):
            config = CheckConfig(
                enabled=True,
                threshold=0.5,
                additional={"legacy_mode": True},
            )
            check = ModshiftCheck(config=config)
            lc = make_lightcurve(baseline_days=50.0)
            candidate = make_candidate(period=5.0)

            result = check.run(candidate, lightcurve=lc)

            assert result.passed is False
            assert result.details["_metrics_only"] is False
            assert result.confidence >= 0.85
            assert result.details["secondary_primary_ratio"] == 0.6
            assert result.details["significant_secondary"] is True

    @patch("bittr_tess_vetter.validation.exovetter_checks._create_lightkurve_like")
    def test_fred_critical_defaults_to_pass_legacy_mode(
        self, mock_create_lk, make_lightcurve, make_candidate, mock_exovetter
    ) -> None:
        """Test that Fred > critical threshold defaults to pass with low confidence in legacy_mode."""
        mock_lk = MagicMock()
        mock_create_lk.return_value = mock_lk

        mock_vetter = MagicMock()
        mock_vetter.run.return_value = {
            "pri": 100.0,
            "sec": 60.0,  # Would fail normally
            "ter": 5.0,
            "pos": 3.0,
            "Fred": 4.0,  # Critical regime
            "false_alarm_threshold": 20.0,
        }
        mock_exovetter["exovetter.vetters"].ModShift.return_value = mock_vetter

        with patch.dict(sys.modules, mock_exovetter):
            config = CheckConfig(
                enabled=True,
                threshold=0.5,
                additional={"legacy_mode": True, "fred_critical_threshold": 3.5},
            )
            check = ModshiftCheck(config=config)
            lc = make_lightcurve(baseline_days=50.0)
            candidate = make_candidate(period=5.0)

            result = check.run(candidate, lightcurve=lc)

            # Should pass because Fred is unreliable in legacy_mode
            assert result.passed is True
            assert result.details["_metrics_only"] is False
            assert result.confidence == 0.35
            assert result.details["fred_regime"] == "critical"
            assert "FRED_UNRELIABLE" in result.details["warnings"]

    @patch("bittr_tess_vetter.validation.exovetter_checks._create_lightkurve_like")
    def test_warnings_list_populated(
        self, mock_create_lk, make_lightcurve, make_candidate, mock_exovetter
    ) -> None:
        """Test that warnings are properly populated."""
        mock_lk = MagicMock()
        mock_create_lk.return_value = mock_lk

        mock_vetter = MagicMock()
        mock_vetter.run.return_value = {
            "pri": 100.0,
            "sec": 35.0,  # Marginal secondary (> 0.3, < 0.5)
            "ter": 40.0,  # High tertiary
            "pos": 150.0,  # Positive > primary
            "Fred": 2.5,  # High red noise
            "false_alarm_threshold": 30.0,
        }
        mock_exovetter["exovetter.vetters"].ModShift.return_value = mock_vetter

        with patch.dict(sys.modules, mock_exovetter):
            check = ModshiftCheck()
            # Short baseline -> low transit count
            lc = make_lightcurve(baseline_days=15.0)
            candidate = make_candidate(period=5.0)

            result = check.run(candidate, lightcurve=lc)

            warnings = result.details["warnings"]
            assert "HIGH_RED_NOISE" in warnings
            assert "TERTIARY_SIGNAL" in warnings
            assert "POSITIVE_SIGNAL_HIGH" in warnings
            assert "LOW_TRANSIT_COUNT" in warnings
            assert "MARGINAL_SECONDARY" in warnings

    def test_exovetter_import_error(self, make_lightcurve, make_candidate) -> None:
        """Test graceful handling of exovetter import error."""
        check = ModshiftCheck()
        lc = make_lightcurve()
        candidate = make_candidate()

        with patch(
            "bittr_tess_vetter.validation.exovetter_checks._create_lightkurve_like",
            side_effect=ImportError("No module named exovetter"),
        ):
            result = check.run(candidate, lightcurve=lc)

        assert result.passed is True
        assert result.confidence == 0.20
        assert result.details["status"] == "error"
        assert "EXOVETTER_IMPORT_ERROR" in result.details["warnings"]


# =============================================================================
# V12 SWEET Tests
# =============================================================================


class TestSWEETCheck:
    """Tests for SWEETCheck (V12)."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        check = SWEETCheck()
        assert check.id == "V12"
        assert check.name == "sweet"
        assert check.config.threshold == 3.5
        assert check.config.additional["half_period_threshold"] == 3.5
        assert check.config.additional["double_period_threshold"] == 4.0
        assert check.config.additional["include_harmonic_analysis"] is True

    def test_no_lightcurve_returns_pass(self, make_candidate) -> None:
        """Test that missing lightcurve returns pass with low confidence.

        Note: Skip cases return passed=True even in metrics-only mode for backward compatibility.
        """
        check = SWEETCheck()
        candidate = make_candidate()

        result = check.run(candidate, lightcurve=None)

        # Skip cases return passed=True even in metrics-only mode
        assert result.passed is True
        assert result.confidence == 0.30
        assert result.details["status"] == "skipped"
        assert "NO_LIGHTCURVE_DATA" in result.details["warnings"]

    @patch("bittr_tess_vetter.validation.exovetter_checks._create_lightkurve_like")
    def test_successful_metrics_only_no_variability(
        self, mock_create_lk, make_lightcurve, make_candidate, mock_exovetter
    ) -> None:
        """Test SWEET in metrics-only mode (default) - passed is None."""
        mock_lk = MagicMock()
        mock_create_lk.return_value = mock_lk

        mock_vetter = MagicMock()
        mock_vetter.run.return_value = {
            "amp": {
                "half_period": (1e-6, 5e-7, 2.0),  # ratio = 2.0 < 3.5
                "period": (5e-7, 5e-7, 1.0),  # ratio = 1.0 < 3.5
                "double_period": (8e-7, 5e-7, 1.6),  # ratio = 1.6 < 4.0
            },
            "msg": "",
        }
        mock_exovetter["exovetter.vetters"].Sweet.return_value = mock_vetter

        with patch.dict(sys.modules, mock_exovetter):
            check = SWEETCheck()
            lc = make_lightcurve(baseline_days=50.0)
            candidate = make_candidate(period=5.0)

            result = check.run(candidate, lightcurve=lc)

            # Metrics-only mode: passed=None
            assert result.passed is None
            assert result.details["_metrics_only"] is True
            assert result.confidence > 0.5
            assert result.details["period_amplitude_ratio"] == 1.0
            assert "warnings" in result.details
            assert "inputs_summary" in result.details
            assert "harmonic_analysis" in result.details
            assert "aliasing_flags" in result.details

    @patch("bittr_tess_vetter.validation.exovetter_checks._create_lightkurve_like")
    def test_successful_pass_no_variability_legacy_mode(
        self, mock_create_lk, make_lightcurve, make_candidate, mock_exovetter
    ) -> None:
        """Test SWEET pass in legacy_mode (no significant variability)."""
        mock_lk = MagicMock()
        mock_create_lk.return_value = mock_lk

        mock_vetter = MagicMock()
        mock_vetter.run.return_value = {
            "amp": {
                "half_period": (1e-6, 5e-7, 2.0),  # ratio = 2.0 < 3.5
                "period": (5e-7, 5e-7, 1.0),  # ratio = 1.0 < 3.5
                "double_period": (8e-7, 5e-7, 1.6),  # ratio = 1.6 < 4.0
            },
            "msg": "",
        }
        mock_exovetter["exovetter.vetters"].Sweet.return_value = mock_vetter

        with patch.dict(sys.modules, mock_exovetter):
            config = CheckConfig(
                enabled=True,
                threshold=3.5,
                additional={"legacy_mode": True},
            )
            check = SWEETCheck(config=config)
            lc = make_lightcurve(baseline_days=50.0)
            candidate = make_candidate(period=5.0)

            result = check.run(candidate, lightcurve=lc)

            assert result.passed is True
            assert result.details["_metrics_only"] is False
            assert result.confidence > 0.5
            assert result.details["period_amplitude_ratio"] == 1.0
            assert "warnings" in result.details
            assert "inputs_summary" in result.details
            assert "harmonic_analysis" in result.details
            assert "aliasing_flags" in result.details

    @patch("bittr_tess_vetter.validation.exovetter_checks._create_lightkurve_like")
    def test_fail_variability_at_period_legacy_mode(
        self, mock_create_lk, make_lightcurve, make_candidate, mock_exovetter
    ) -> None:
        """Test SWEET fail in legacy_mode (significant variability at transit period)."""
        mock_lk = MagicMock()
        mock_create_lk.return_value = mock_lk

        mock_vetter = MagicMock()
        mock_vetter.run.return_value = {
            "amp": {
                "half_period": (1e-6, 5e-7, 2.0),
                "period": (5e-6, 1e-6, 5.0),  # ratio = 5.0 > 3.5
                "double_period": (8e-7, 5e-7, 1.6),
            },
            "msg": "",
        }
        mock_exovetter["exovetter.vetters"].Sweet.return_value = mock_vetter

        with patch.dict(sys.modules, mock_exovetter):
            config = CheckConfig(
                enabled=True,
                threshold=3.5,
                additional={"legacy_mode": True},
            )
            check = SWEETCheck(config=config)
            lc = make_lightcurve(baseline_days=50.0)
            candidate = make_candidate(period=5.0, depth=0.001)

            result = check.run(candidate, lightcurve=lc)

            assert result.passed is False
            assert result.details["_metrics_only"] is False
            assert result.details["fails_at_period"] is True
            assert result.details["period_amplitude_ratio"] == 5.0

    @patch("bittr_tess_vetter.validation.exovetter_checks._create_lightkurve_like")
    def test_harmonic_aliasing_at_half_period_legacy_mode(
        self, mock_create_lk, make_lightcurve, make_candidate, mock_exovetter
    ) -> None:
        """Test harmonic aliasing detection at P/2 in legacy_mode."""
        mock_lk = MagicMock()
        mock_create_lk.return_value = mock_lk

        # High amplitude at P/2 that can explain significant depth
        mock_vetter = MagicMock()
        mock_vetter.run.return_value = {
            "amp": {
                # Very high ratio, amplitude ~ 500 ppm
                "half_period": (5e-4, 1e-5, 50.0),
                "period": (1e-6, 1e-6, 1.0),  # Low at period
                "double_period": (1e-6, 1e-6, 1.0),
            },
            "msg": "",
        }
        mock_exovetter["exovetter.vetters"].Sweet.return_value = mock_vetter

        with patch.dict(sys.modules, mock_exovetter):
            config = CheckConfig(
                enabled=True,
                threshold=3.5,
                additional={"legacy_mode": True, "include_harmonic_analysis": True},
            )
            check = SWEETCheck(config=config)
            lc = make_lightcurve(baseline_days=50.0)
            # Transit depth of 1000 ppm; P/2 amplitude of 500 ppm creates 1000 ppm depth
            candidate = make_candidate(period=5.0, depth=0.001)

            result = check.run(candidate, lightcurve=lc)

            # With include_harmonic_analysis=True, this should fail in legacy_mode
            assert result.passed is False
            assert result.details["_metrics_only"] is False
            assert result.details["fails_at_half_period"] is True
            assert result.details["harmonic_analysis"]["dominant_variability_period"] == "P/2"

    @patch("bittr_tess_vetter.validation.exovetter_checks._create_lightkurve_like")
    def test_inputs_summary_computed(
        self, mock_create_lk, make_lightcurve, make_candidate, mock_exovetter
    ) -> None:
        """Test that inputs_summary is properly computed."""
        mock_lk = MagicMock()
        mock_create_lk.return_value = mock_lk

        mock_vetter = MagicMock()
        mock_vetter.run.return_value = {
            "amp": {
                "half_period": (1e-6, 5e-7, 2.0),
                "period": (5e-7, 5e-7, 1.0),
                "double_period": (8e-7, 5e-7, 1.6),
            },
            "msg": "",
        }
        mock_exovetter["exovetter.vetters"].Sweet.return_value = mock_vetter

        with patch.dict(sys.modules, mock_exovetter):
            check = SWEETCheck()
            lc = make_lightcurve(n_points=1000, baseline_days=27.0)
            candidate = make_candidate(period=5.0, snr=15.0)

            result = check.run(candidate, lightcurve=lc)

            summary = result.details["inputs_summary"]
            assert summary["n_points"] == 1000
            assert summary["n_cycles_observed"] == 5.4  # 27 / 5
            assert summary["n_transits"] == 5
            assert summary["snr"] == 15.0
            assert summary["can_detect_2p"] is True

    @patch("bittr_tess_vetter.validation.exovetter_checks._create_lightkurve_like")
    def test_low_baseline_warning(
        self, mock_create_lk, make_lightcurve, make_candidate, mock_exovetter
    ) -> None:
        """Test warning for low baseline relative to period."""
        mock_lk = MagicMock()
        mock_create_lk.return_value = mock_lk

        mock_vetter = MagicMock()
        mock_vetter.run.return_value = {
            "amp": {
                "half_period": (1e-6, 5e-7, 2.0),
                "period": (5e-7, 5e-7, 1.0),
                "double_period": (8e-7, 5e-7, 1.6),
            },
            "msg": "",
        }
        mock_exovetter["exovetter.vetters"].Sweet.return_value = mock_vetter

        with patch.dict(sys.modules, mock_exovetter):
            check = SWEETCheck()
            # Only 5 days with 5-day period = 1 cycle
            lc = make_lightcurve(n_points=200, baseline_days=5.0)
            candidate = make_candidate(period=5.0)

            result = check.run(candidate, lightcurve=lc)

            warnings = result.details["warnings"]
            assert "LOW_BASELINE_CYCLES" in warnings
            assert "CANNOT_DETECT_2P" in warnings

    @patch("bittr_tess_vetter.validation.exovetter_checks._create_lightkurve_like")
    def test_harmonic_analysis_output(
        self, mock_create_lk, make_lightcurve, make_candidate, mock_exovetter
    ) -> None:
        """Test harmonic_analysis output structure."""
        mock_lk = MagicMock()
        mock_create_lk.return_value = mock_lk

        mock_vetter = MagicMock()
        mock_vetter.run.return_value = {
            "amp": {
                "half_period": (1e-4, 5e-5, 2.0),  # 100 ppm amplitude
                "period": (5e-5, 5e-5, 1.0),  # 50 ppm amplitude
                "double_period": (8e-5, 5e-5, 1.6),
            },
            "msg": "",
        }
        mock_exovetter["exovetter.vetters"].Sweet.return_value = mock_vetter

        with patch.dict(sys.modules, mock_exovetter):
            check = SWEETCheck()
            lc = make_lightcurve(baseline_days=50.0)
            candidate = make_candidate(period=5.0, depth=0.001)  # 1000 ppm depth

            result = check.run(candidate, lightcurve=lc)

            harmonic = result.details["harmonic_analysis"]
            assert "variability_induced_depth_at_P_ppm" in harmonic
            assert "variability_induced_depth_at_half_P_ppm" in harmonic
            assert "variability_explains_depth_fraction" in harmonic
            assert "dominant_variability_period" in harmonic

    @patch("bittr_tess_vetter.validation.exovetter_checks._create_lightkurve_like")
    def test_aliasing_flags_output(
        self, mock_create_lk, make_lightcurve, make_candidate, mock_exovetter
    ) -> None:
        """Test aliasing_flags output structure."""
        mock_lk = MagicMock()
        mock_create_lk.return_value = mock_lk

        mock_vetter = MagicMock()
        mock_vetter.run.return_value = {
            "amp": {
                "half_period": (1e-6, 5e-7, 2.0),
                "period": (5e-7, 5e-7, 1.0),
                "double_period": (8e-7, 5e-7, 1.6),
            },
            "msg": "",
        }
        mock_exovetter["exovetter.vetters"].Sweet.return_value = mock_vetter

        with patch.dict(sys.modules, mock_exovetter):
            check = SWEETCheck()
            lc = make_lightcurve(baseline_days=50.0)
            candidate = make_candidate(period=5.0)

            result = check.run(candidate, lightcurve=lc)

            aliasing = result.details["aliasing_flags"]
            assert "half_period_alias_risk" in aliasing
            assert "double_period_alias_risk" in aliasing
            assert "dominant_alias" in aliasing

    @patch("bittr_tess_vetter.validation.exovetter_checks._create_lightkurve_like")
    def test_confidence_degraded_with_few_cycles(
        self, mock_create_lk, make_lightcurve, make_candidate, mock_exovetter
    ) -> None:
        """Test that confidence is degraded with few cycles."""
        mock_lk = MagicMock()
        mock_create_lk.return_value = mock_lk

        mock_vetter = MagicMock()
        mock_vetter.run.return_value = {
            "amp": {
                "half_period": (1e-6, 5e-7, 2.0),
                "period": (5e-7, 5e-7, 1.0),  # Would give high conf
                "double_period": (8e-7, 5e-7, 1.6),
            },
            "msg": "",
        }
        mock_exovetter["exovetter.vetters"].Sweet.return_value = mock_vetter

        with patch.dict(sys.modules, mock_exovetter):
            check = SWEETCheck()

            # Long baseline - high confidence
            lc_long = make_lightcurve(baseline_days=50.0)
            candidate = make_candidate(period=5.0)
            result_long = check.run(candidate, lightcurve=lc_long)

            # Short baseline - lower confidence
            lc_short = make_lightcurve(baseline_days=8.0)  # ~1.6 cycles
            result_short = check.run(candidate, lightcurve=lc_short)

            assert result_short.confidence < result_long.confidence

    def test_exovetter_import_error(self, make_lightcurve, make_candidate) -> None:
        """Test graceful handling of exovetter import error."""
        check = SWEETCheck()
        lc = make_lightcurve()
        candidate = make_candidate()

        with patch(
            "bittr_tess_vetter.validation.exovetter_checks._create_lightkurve_like",
            side_effect=ImportError("No module named exovetter"),
        ):
            result = check.run(candidate, lightcurve=lc)

        assert result.passed is True
        assert result.confidence == 0.20
        assert result.details["status"] == "error"
        assert "EXOVETTER_IMPORT_ERROR" in result.details["warnings"]


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for V11 and V12 checks."""

    def test_modshift_custom_config(self) -> None:
        """Test ModshiftCheck with custom configuration."""
        config = CheckConfig(
            enabled=True,
            threshold=0.3,
            additional={
                "fred_warning_threshold": 1.5,
                "fred_critical_threshold": 3.0,
            },
        )
        check = ModshiftCheck(config=config)

        assert check.config.threshold == 0.3
        assert check.config.additional["fred_warning_threshold"] == 1.5

    def test_sweet_custom_config(self) -> None:
        """Test SWEETCheck with custom configuration."""
        config = CheckConfig(
            enabled=True,
            threshold=4.0,
            additional={
                "half_period_threshold": 4.0,
                "include_harmonic_analysis": False,
            },
        )
        check = SWEETCheck(config=config)

        assert check.config.threshold == 4.0
        assert check.config.additional["include_harmonic_analysis"] is False

    def test_result_structure_completeness(self, make_candidate) -> None:
        """Test that result structure is complete even for skip cases."""
        modshift = ModshiftCheck()
        sweet = SWEETCheck()
        candidate = make_candidate()

        modshift_result = modshift.run(candidate, lightcurve=None)
        sweet_result = sweet.run(candidate, lightcurve=None)

        # Both should have required fields
        for result in [modshift_result, sweet_result]:
            assert isinstance(result, VetterCheckResult)
            assert result.id in ["V11", "V12"]
            assert result.name in ["modshift", "sweet"]
            assert isinstance(result.passed, bool)
            assert isinstance(result.confidence, float)
            assert isinstance(result.details, dict)
            assert "warnings" in result.details
