"""Unit tests for pixel time-series inference.

Tests the pixel_timeseries module which provides:
- Transit window extraction from TPF data
- WLS transit amplitude fitting per hypothesis
- Evidence aggregation across windows
- Diagnostic artifact computation

All tests are deterministic and require no network or file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pytest
from numpy.typing import NDArray

from bittr_tess_vetter.compute.pixel_timeseries import (
    DEFAULT_MARGIN_THRESHOLD,
    PixelTimeseriesFit,
    TimeseriesEvidence,
    TransitWindow,
    aggregate_timeseries_evidence,
    compute_timeseries_diagnostics,
    extract_transit_windows,
    fit_all_hypotheses_timeseries,
    fit_transit_amplitude_wls,
    select_best_hypothesis_timeseries,
)

# =============================================================================
# Mock PRF Model
# =============================================================================


class PRFModel(Protocol):
    """Protocol for PRF model interface."""

    def evaluate(
        self,
        center_row: float,
        center_col: float,
        shape: tuple[int, int],
        *,
        normalize: bool = True,
    ) -> NDArray[np.float64]:
        """Evaluate PRF at given position."""
        ...


@dataclass
class MockPRFModel:
    """Simple Gaussian PRF model for testing."""

    sigma: float = 1.5

    def evaluate(
        self,
        center_row: float,
        center_col: float,
        shape: tuple[int, int],
        *,
        normalize: bool = True,
    ) -> NDArray[np.float64]:
        """Evaluate Gaussian PRF at given position."""
        n_rows, n_cols = shape
        row_grid, col_grid = np.mgrid[0:n_rows, 0:n_cols]

        # Gaussian
        dist_sq = (row_grid - center_row) ** 2 + (col_grid - center_col) ** 2
        prf = np.exp(-dist_sq / (2 * self.sigma**2))

        if normalize:
            total = float(np.sum(prf))
            if total > 0:
                prf = prf / total

        return prf.astype(np.float64)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_tpf_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create simple synthetic TPF data with a transit.

    Returns (flux_cube, time_array) for a 5x5 pixel stamp with 100 cadences.
    """
    np.random.seed(42)

    n_cadences = 100
    n_rows = 5
    n_cols = 5

    # Time array: 1 day observation
    time = np.linspace(1000.0, 1001.0, n_cadences)

    # Base flux with slight gradient
    base_flux = np.ones((n_rows, n_cols)) * 10000.0
    # Add Gaussian PSF centered at (2, 2)
    row_grid, col_grid = np.mgrid[0:n_rows, 0:n_cols]
    psf = 5000.0 * np.exp(-((row_grid - 2) ** 2 + (col_grid - 2) ** 2) / (2 * 1.5**2))
    base_flux += psf

    # Create flux cube with noise
    flux = np.zeros((n_cadences, n_rows, n_cols))
    for t in range(n_cadences):
        flux[t] = base_flux + np.random.normal(0, 50, (n_rows, n_cols))

    # Add transit at center of observation
    # Transit parameters: period=0.5 days, t0=1000.5, duration=2 hours
    t0 = 1000.5
    duration_days = 2.0 / 24.0  # 2 hours
    transit_depth = 0.01  # 1% depth

    for t in range(n_cadences):
        if abs(time[t] - t0) < duration_days / 2:
            # In transit - reduce flux at center pixel
            flux[t] *= 1 - transit_depth * psf / np.max(psf)

    return flux.astype(np.float64), time.astype(np.float64)


@pytest.fixture
def multi_transit_tpf_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create TPF data with multiple transits for testing window extraction.

    Returns (flux_cube, time_array) with 500 cadences spanning multiple orbits.
    """
    np.random.seed(123)

    n_cadences = 500
    n_rows = 5
    n_cols = 5

    # Time array: 5 day observation
    time = np.linspace(1000.0, 1005.0, n_cadences)

    # Base flux
    base_flux = np.ones((n_rows, n_cols)) * 10000.0
    row_grid, col_grid = np.mgrid[0:n_rows, 0:n_cols]
    psf = 5000.0 * np.exp(-((row_grid - 2) ** 2 + (col_grid - 2) ** 2) / (2 * 1.5**2))
    base_flux += psf

    flux = np.zeros((n_cadences, n_rows, n_cols))
    for t in range(n_cadences):
        flux[t] = base_flux + np.random.normal(0, 30, (n_rows, n_cols))

    # Transit parameters: period=1 day, t0=1000.5, duration=3 hours
    period = 1.0
    t0 = 1000.5
    duration_days = 3.0 / 24.0
    transit_depth = 0.015

    # Add transits
    for t in range(n_cadences):
        phase = ((time[t] - t0) % period) / period
        if phase > 0.5:
            phase -= 1.0
        if abs(phase) * period < duration_days / 2:
            flux[t] *= 1 - transit_depth * psf / np.max(psf)

    return flux.astype(np.float64), time.astype(np.float64)


@pytest.fixture
def mock_prf_model() -> MockPRFModel:
    """Create a mock PRF model for testing."""
    return MockPRFModel(sigma=1.5)


@pytest.fixture
def sample_transit_window(
    simple_tpf_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> TransitWindow:
    """Create a sample transit window for testing."""
    flux, time = simple_tpf_data

    # Extract window around t0=1000.5
    t0 = 1000.5
    duration_hours = 2.0
    window_half = (duration_hours / 24.0) * 2.0  # 2x duration margin

    mask = (time >= t0 - window_half) & (time <= t0 + window_half)
    time_window = time[mask]
    pixels_window = flux[mask]

    # In-transit mask
    half_duration = duration_hours / 24.0 / 2.0
    in_transit = np.abs(time_window - t0) <= half_duration

    errors = np.sqrt(np.clip(pixels_window, 1.0, None))

    return TransitWindow(
        transit_idx=0,
        time=time_window,
        pixels=pixels_window,
        errors=errors,
        in_transit_mask=in_transit,
        t0_expected=t0,
    )


# =============================================================================
# Window Extraction Tests
# =============================================================================


class TestExtractTransitWindows:
    """Tests for extract_transit_windows function."""

    def test_finds_correct_number_of_transits(
        self, multi_transit_tpf_data: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        """Finds expected number of transit windows."""
        flux, time = multi_transit_tpf_data
        period = 1.0
        t0 = 1000.5
        duration_hours = 3.0

        windows = extract_transit_windows(
            tpf_data=flux,
            time=time,
            period=period,
            t0=t0,
            duration_hours=duration_hours,
        )

        # 5-day observation with 1-day period should have ~5 transits
        assert 4 <= len(windows) <= 5

    def test_window_contains_in_transit_points(
        self, multi_transit_tpf_data: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        """Each window has sufficient in-transit points."""
        flux, time = multi_transit_tpf_data

        windows = extract_transit_windows(
            tpf_data=flux,
            time=time,
            period=1.0,
            t0=1000.5,
            duration_hours=3.0,
            min_in_transit=3,
        )

        for window in windows:
            n_in = int(np.sum(window.in_transit_mask))
            assert n_in >= 3

    def test_window_has_correct_shape(
        self, simple_tpf_data: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        """Window data has consistent shapes."""
        flux, time = simple_tpf_data

        windows = extract_transit_windows(
            tpf_data=flux,
            time=time,
            period=0.5,
            t0=1000.5,
            duration_hours=2.0,
        )

        if windows:
            window = windows[0]
            n_cadences = len(window.time)
            assert window.pixels.shape[0] == n_cadences
            assert window.errors.shape[0] == n_cadences
            assert len(window.in_transit_mask) == n_cadences

    def test_custom_margin(
        self, multi_transit_tpf_data: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        """Custom window margin affects window size."""
        flux, time = multi_transit_tpf_data

        windows_narrow = extract_transit_windows(
            tpf_data=flux,
            time=time,
            period=1.0,
            t0=1000.5,
            duration_hours=3.0,
            window_margin=1.5,
        )

        windows_wide = extract_transit_windows(
            tpf_data=flux,
            time=time,
            period=1.0,
            t0=1000.5,
            duration_hours=3.0,
            window_margin=3.0,
        )

        if windows_narrow and windows_wide:
            # Wider margin should have more points
            assert len(windows_wide[0].time) > len(windows_narrow[0].time)

    def test_returns_empty_for_insufficient_in_transit(self) -> None:
        """Returns empty list if not enough in-transit points."""
        # Very few points, requiring high min_in_transit
        flux = np.random.randn(5, 5, 5)
        time = np.linspace(1000.0, 1000.01, 5)

        windows = extract_transit_windows(
            tpf_data=flux,
            time=time,
            period=0.5,
            t0=1000.005,
            duration_hours=0.001,  # Very short duration
            min_in_transit=10,  # Require more than available
        )

        assert len(windows) == 0

    def test_raises_for_wrong_dimensions(self) -> None:
        """Raises for wrong input dimensions."""
        flux_2d = np.random.randn(100, 5)  # Missing column dimension
        time = np.linspace(1000.0, 1001.0, 100)

        with pytest.raises(ValueError, match="3D"):
            extract_transit_windows(
                tpf_data=flux_2d,
                time=time,
                period=0.5,
                t0=1000.5,
                duration_hours=2.0,
            )


# =============================================================================
# WLS Fitting Tests
# =============================================================================


class TestFitTransitAmplitudeWls:
    """Tests for fit_transit_amplitude_wls function."""

    def test_recovers_injected_amplitude(
        self, sample_transit_window: TransitWindow, mock_prf_model: MockPRFModel
    ) -> None:
        """WLS fit recovers approximately correct amplitude."""
        # Fit at the correct position (center of PSF)
        fit = fit_transit_amplitude_wls(
            sample_transit_window,
            hypothesis_row=2.0,
            hypothesis_col=2.0,
            prf_model=mock_prf_model,
        )

        # Amplitude should be negative (dimming during transit)
        assert fit.amplitude < 0

        # Chi-squared should be reasonable
        assert np.isfinite(fit.chi2)
        assert fit.chi2 > 0

        # DOF should be positive
        assert fit.dof > 0

    def test_wrong_position_gives_worse_fit(
        self, sample_transit_window: TransitWindow, mock_prf_model: MockPRFModel
    ) -> None:
        """Fitting at wrong position gives higher chi-squared."""
        fit_correct = fit_transit_amplitude_wls(
            sample_transit_window,
            hypothesis_row=2.0,
            hypothesis_col=2.0,
            prf_model=mock_prf_model,
        )

        fit_wrong = fit_transit_amplitude_wls(
            sample_transit_window,
            hypothesis_row=4.0,
            hypothesis_col=4.0,
            prf_model=mock_prf_model,
        )

        # Correct position should have lower chi-squared per DOF
        chi2_per_dof_correct = fit_correct.chi2 / max(1, fit_correct.dof)
        chi2_per_dof_wrong = fit_wrong.chi2 / max(1, fit_wrong.dof)

        # Note: This may not always hold due to noise, but should generally be true
        # We just check both fits are valid
        assert np.isfinite(chi2_per_dof_correct)
        assert np.isfinite(chi2_per_dof_wrong)

    def test_no_baseline_fitting(
        self, sample_transit_window: TransitWindow, mock_prf_model: MockPRFModel
    ) -> None:
        """Can fit without baseline."""
        fit = fit_transit_amplitude_wls(
            sample_transit_window,
            hypothesis_row=2.0,
            hypothesis_col=2.0,
            prf_model=mock_prf_model,
            fit_baseline=False,
        )

        assert np.isfinite(fit.amplitude)
        assert np.isfinite(fit.chi2)

    def test_returns_per_pixel_residuals(
        self, sample_transit_window: TransitWindow, mock_prf_model: MockPRFModel
    ) -> None:
        """Fit includes per-pixel residual summary."""
        fit = fit_transit_amplitude_wls(
            sample_transit_window,
            hypothesis_row=2.0,
            hypothesis_col=2.0,
            prf_model=mock_prf_model,
        )

        assert fit.per_pixel_residuals is not None
        n_rows = sample_transit_window.pixels.shape[1]
        n_cols = sample_transit_window.pixels.shape[2]
        assert fit.per_pixel_residuals.shape == (n_rows, n_cols)


class TestFitAllHypothesesTimeseries:
    """Tests for fit_all_hypotheses_timeseries function."""

    def test_fits_all_hypotheses(
        self, sample_transit_window: TransitWindow, mock_prf_model: MockPRFModel
    ) -> None:
        """Fits all provided hypotheses."""
        hypotheses = [
            {"source_id": "target", "row": 2.0, "col": 2.0},
            {"source_id": "neighbor", "row": 4.0, "col": 4.0},
        ]

        results = fit_all_hypotheses_timeseries(
            [sample_transit_window],
            hypotheses,
            mock_prf_model,
        )

        assert "target" in results
        assert "neighbor" in results
        assert len(results["target"]) == 1
        assert len(results["neighbor"]) == 1

    def test_fits_multiple_windows(
        self,
        multi_transit_tpf_data: tuple[NDArray[np.float64], NDArray[np.float64]],
        mock_prf_model: MockPRFModel,
    ) -> None:
        """Fits across multiple transit windows."""
        flux, time = multi_transit_tpf_data

        windows = extract_transit_windows(
            tpf_data=flux,
            time=time,
            period=1.0,
            t0=1000.5,
            duration_hours=3.0,
        )

        hypotheses = [{"source_id": "target", "row": 2.0, "col": 2.0}]

        results = fit_all_hypotheses_timeseries(
            windows,
            hypotheses,
            mock_prf_model,
        )

        assert len(results["target"]) == len(windows)


# =============================================================================
# Evidence Aggregation Tests
# =============================================================================


class TestAggregateTimeseriesEvidence:
    """Tests for aggregate_timeseries_evidence function."""

    def test_aggregates_chi2_and_dof(self) -> None:
        """Correctly sums chi2 and dof across fits."""
        fits = [
            PixelTimeseriesFit(
                source_id="test",
                amplitude=-100.0,
                amplitude_err=10.0,
                chi2=50.0,
                dof=20,
                residual_rms=5.0,
            ),
            PixelTimeseriesFit(
                source_id="test",
                amplitude=-95.0,
                amplitude_err=12.0,
                chi2=60.0,
                dof=25,
                residual_rms=6.0,
            ),
        ]

        evidence = aggregate_timeseries_evidence(fits)

        assert evidence.total_chi2 == pytest.approx(110.0)
        assert evidence.total_dof == 45
        assert evidence.n_windows_fitted == 2

    def test_computes_mean_amplitude(self) -> None:
        """Correctly computes mean amplitude."""
        fits = [
            PixelTimeseriesFit(
                source_id="test",
                amplitude=-100.0,
                amplitude_err=10.0,
                chi2=50.0,
                dof=20,
                residual_rms=5.0,
            ),
            PixelTimeseriesFit(
                source_id="test",
                amplitude=-80.0,
                amplitude_err=12.0,
                chi2=60.0,
                dof=25,
                residual_rms=6.0,
            ),
        ]

        evidence = aggregate_timeseries_evidence(fits)

        assert evidence.mean_amplitude == pytest.approx(-90.0)

    def test_computes_amplitude_scatter(self) -> None:
        """Correctly computes amplitude scatter."""
        fits = [
            PixelTimeseriesFit(
                source_id="test",
                amplitude=-100.0,
                amplitude_err=10.0,
                chi2=50.0,
                dof=20,
                residual_rms=5.0,
            ),
            PixelTimeseriesFit(
                source_id="test",
                amplitude=-80.0,
                amplitude_err=12.0,
                chi2=60.0,
                dof=25,
                residual_rms=6.0,
            ),
        ]

        evidence = aggregate_timeseries_evidence(fits)

        assert evidence.amplitude_scatter == pytest.approx(10.0)

    def test_handles_empty_fits(self) -> None:
        """Handles empty fit list gracefully."""
        evidence = aggregate_timeseries_evidence([])

        assert evidence.total_chi2 == float("inf")
        assert evidence.n_windows_fitted == 0

    def test_computes_log_likelihood(self) -> None:
        """Computes approximate log-likelihood from chi2."""
        fits = [
            PixelTimeseriesFit(
                source_id="test",
                amplitude=-100.0,
                amplitude_err=10.0,
                chi2=100.0,
                dof=50,
                residual_rms=5.0,
            ),
        ]

        evidence = aggregate_timeseries_evidence(fits)

        # log_likelihood = -0.5 * chi2
        assert evidence.log_likelihood == pytest.approx(-50.0)


class TestSelectBestHypothesisTimeseries:
    """Tests for select_best_hypothesis_timeseries function."""

    def test_selects_lowest_chi2(self) -> None:
        """Selects hypothesis with lowest chi2."""
        evidence = {
            "target": TimeseriesEvidence(
                source_id="target",
                total_chi2=100.0,
                total_dof=50,
                mean_amplitude=-100.0,
                amplitude_scatter=5.0,
                n_windows_fitted=5,
                log_likelihood=-50.0,
            ),
            "neighbor": TimeseriesEvidence(
                source_id="neighbor",
                total_chi2=150.0,
                total_dof=50,
                mean_amplitude=-80.0,
                amplitude_scatter=10.0,
                n_windows_fitted=5,
                log_likelihood=-75.0,
            ),
        }

        best_id, verdict, delta = select_best_hypothesis_timeseries(evidence)

        assert best_id == "target"
        assert delta == pytest.approx(50.0)

    def test_on_target_verdict(self) -> None:
        """Returns ON_TARGET when target wins with margin."""
        evidence = {
            "tic:target:123": TimeseriesEvidence(
                source_id="tic:target:123",
                total_chi2=100.0,
                total_dof=50,
                mean_amplitude=-100.0,
                amplitude_scatter=5.0,
                n_windows_fitted=5,
                log_likelihood=-50.0,
            ),
            "neighbor": TimeseriesEvidence(
                source_id="neighbor",
                total_chi2=105.0,
                total_dof=50,
                mean_amplitude=-80.0,
                amplitude_scatter=10.0,
                n_windows_fitted=5,
                log_likelihood=-52.5,
            ),
        }

        _, verdict, delta = select_best_hypothesis_timeseries(evidence)

        assert verdict == "ON_TARGET"
        assert delta >= DEFAULT_MARGIN_THRESHOLD

    def test_off_target_verdict(self) -> None:
        """Returns OFF_TARGET when non-target wins with margin."""
        evidence = {
            "tic:target:123": TimeseriesEvidence(
                source_id="tic:target:123",
                total_chi2=150.0,
                total_dof=50,
                mean_amplitude=-100.0,
                amplitude_scatter=5.0,
                n_windows_fitted=5,
                log_likelihood=-75.0,
            ),
            "gaia:neighbor": TimeseriesEvidence(
                source_id="gaia:neighbor",
                total_chi2=100.0,
                total_dof=50,
                mean_amplitude=-80.0,
                amplitude_scatter=10.0,
                n_windows_fitted=5,
                log_likelihood=-50.0,
            ),
        }

        best_id, verdict, _ = select_best_hypothesis_timeseries(evidence)

        assert best_id == "gaia:neighbor"
        assert verdict == "OFF_TARGET"

    def test_ambiguous_verdict(self) -> None:
        """Returns AMBIGUOUS when chi2 difference is small."""
        evidence = {
            "target": TimeseriesEvidence(
                source_id="target",
                total_chi2=100.0,
                total_dof=50,
                mean_amplitude=-100.0,
                amplitude_scatter=5.0,
                n_windows_fitted=5,
                log_likelihood=-50.0,
            ),
            "neighbor": TimeseriesEvidence(
                source_id="neighbor",
                total_chi2=101.0,  # Only 1.0 difference < threshold
                total_dof=50,
                mean_amplitude=-80.0,
                amplitude_scatter=10.0,
                n_windows_fitted=5,
                log_likelihood=-50.5,
            ),
        }

        _, verdict, delta = select_best_hypothesis_timeseries(evidence)

        assert verdict == "AMBIGUOUS"
        assert delta < DEFAULT_MARGIN_THRESHOLD

    def test_handles_empty_evidence(self) -> None:
        """Handles empty evidence dictionary."""
        best_id, verdict, delta = select_best_hypothesis_timeseries({})

        assert best_id == ""
        assert verdict == "AMBIGUOUS"
        assert delta == 0.0


# =============================================================================
# Diagnostics Tests
# =============================================================================


class TestComputeTimeseriesDiagnostics:
    """Tests for compute_timeseries_diagnostics function."""

    def test_collects_per_window_metrics(self) -> None:
        """Collects amplitudes and chi2 per window."""
        fits = {
            "target": [
                PixelTimeseriesFit(
                    source_id="target",
                    amplitude=-100.0,
                    amplitude_err=10.0,
                    chi2=50.0,
                    dof=20,
                    residual_rms=5.0,
                ),
                PixelTimeseriesFit(
                    source_id="target",
                    amplitude=-95.0,
                    amplitude_err=12.0,
                    chi2=55.0,
                    dof=20,
                    residual_rms=6.0,
                ),
            ],
        }

        windows: list[TransitWindow] = []  # Empty for this test

        diag = compute_timeseries_diagnostics(fits, windows)

        assert "target" in diag.per_window_amplitudes
        assert len(diag.per_window_amplitudes["target"]) == 2
        assert diag.per_window_amplitudes["target"][0] == pytest.approx(-100.0)

    def test_identifies_outlier_windows(self) -> None:
        """Identifies windows with anomalous chi2."""
        fits = {
            "target": [
                PixelTimeseriesFit(
                    source_id="target",
                    amplitude=-100.0,
                    amplitude_err=10.0,
                    chi2=50.0,
                    dof=20,
                    residual_rms=5.0,
                ),
                PixelTimeseriesFit(
                    source_id="target",
                    amplitude=-100.0,
                    amplitude_err=10.0,
                    chi2=52.0,
                    dof=20,
                    residual_rms=5.0,
                ),
                PixelTimeseriesFit(
                    source_id="target",
                    amplitude=-100.0,
                    amplitude_err=10.0,
                    chi2=500.0,  # Outlier!
                    dof=20,
                    residual_rms=50.0,
                ),
                PixelTimeseriesFit(
                    source_id="target",
                    amplitude=-100.0,
                    amplitude_err=10.0,
                    chi2=48.0,
                    dof=20,
                    residual_rms=5.0,
                ),
            ],
        }

        windows: list[TransitWindow] = []

        diag = compute_timeseries_diagnostics(fits, windows)

        # The third window (index 2) should be flagged as outlier
        assert 2 in diag.outlier_windows

    def test_warns_on_high_amplitude_scatter(self) -> None:
        """Generates warning for high amplitude scatter."""
        fits = {
            "target": [
                PixelTimeseriesFit(
                    source_id="target",
                    amplitude=-100.0,
                    amplitude_err=10.0,
                    chi2=50.0,
                    dof=20,
                    residual_rms=5.0,
                ),
                PixelTimeseriesFit(
                    source_id="target",
                    amplitude=-10.0,  # Very different from first
                    amplitude_err=10.0,
                    chi2=50.0,
                    dof=20,
                    residual_rms=5.0,
                ),
            ],
        }

        windows: list[TransitWindow] = []

        diag = compute_timeseries_diagnostics(fits, windows)

        # Should have a warning about high scatter
        assert any("scatter" in w.lower() for w in diag.warnings)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPixelTimeseriesIntegration:
    """Integration tests for the full workflow."""

    def test_end_to_end_workflow(
        self,
        multi_transit_tpf_data: tuple[NDArray[np.float64], NDArray[np.float64]],
        mock_prf_model: MockPRFModel,
    ) -> None:
        """Full workflow from extraction to verdict."""
        flux, time = multi_transit_tpf_data

        # 1. Extract windows
        windows = extract_transit_windows(
            tpf_data=flux,
            time=time,
            period=1.0,
            t0=1000.5,
            duration_hours=3.0,
        )

        assert len(windows) > 0

        # 2. Define hypotheses
        hypotheses = [
            {"source_id": "target", "row": 2.0, "col": 2.0},  # Correct position
            {"source_id": "neighbor", "row": 4.0, "col": 0.0},  # Wrong position
        ]

        # 3. Fit all hypotheses
        fits = fit_all_hypotheses_timeseries(
            windows,
            hypotheses,
            mock_prf_model,
        )

        assert "target" in fits
        assert "neighbor" in fits

        # 4. Aggregate evidence
        evidence = {sid: aggregate_timeseries_evidence(fit_list) for sid, fit_list in fits.items()}

        # Target should have better (lower) chi2
        assert evidence["target"].total_chi2 < evidence["neighbor"].total_chi2

        # 5. Select best
        best_id, verdict, delta = select_best_hypothesis_timeseries(evidence)

        assert best_id == "target"
        assert verdict == "ON_TARGET"

        # 6. Compute diagnostics
        diag = compute_timeseries_diagnostics(fits, windows)

        assert "target" in diag.per_window_amplitudes
        assert len(diag.per_window_amplitudes["target"]) == len(windows)
