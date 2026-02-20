"""Tests for DVR-style vetting summary report functions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from tess_vetter.plotting.report import (
    _render_metrics_table,
    plot_vetting_summary,
    save_vetting_report,
)
from tess_vetter.validation.result_schema import (
    VettingBundleResult,
    ok_result,
)


@pytest.fixture
def mock_lightcurve() -> LightCurve:
    """Create a mock light curve for testing."""
    np.random.seed(42)
    n_points = 1000
    time = np.linspace(2458600.0, 2458650.0, n_points)
    flux = np.ones(n_points) + np.random.normal(0, 0.001, n_points)
    flux_err = np.full(n_points, 0.001)
    return LightCurve(time=time, flux=flux, flux_err=flux_err)


@pytest.fixture
def mock_candidate() -> Candidate:
    """Create a mock candidate for testing."""
    ephemeris = Ephemeris(
        period_days=5.0,
        t0_btjd=2458605.0,
        duration_hours=3.0,
    )
    return Candidate(ephemeris=ephemeris, depth_ppm=500.0)


@pytest.fixture
def mock_v01_result():
    """Create a mock V01 (odd/even) CheckResult with plot_data."""
    return ok_result(
        id="V01",
        name="Odd/Even Depth",
        metrics={
            "odd_mean_depth_ppm": 500.0,
            "even_mean_depth_ppm": 510.0,
            "sigma_diff": 0.5,
        },
        confidence=0.95,
        raw={
            "plot_data": {
                "version": 1,
                "odd_epochs": [1, 3, 5, 7, 9],
                "odd_depths_ppm": [490.0, 510.0, 495.0, 505.0, 500.0],
                "odd_errs_ppm": [20.0, 22.0, 18.0, 21.0, 19.0],
                "even_epochs": [2, 4, 6, 8, 10],
                "even_depths_ppm": [505.0, 515.0, 508.0, 512.0, 510.0],
                "even_errs_ppm": [21.0, 20.0, 19.0, 22.0, 18.0],
                "mean_odd_ppm": 500.0,
                "mean_even_ppm": 510.0,
            }
        },
    )


@pytest.fixture
def mock_v02_result():
    """Create a mock V02 (secondary eclipse) CheckResult with plot_data."""
    phase = np.linspace(0.0, 1.0, 100).tolist()
    flux = (np.ones(100) + np.random.normal(0, 0.001, 100)).tolist()

    return ok_result(
        id="V02",
        name="Secondary Eclipse",
        metrics={
            "secondary_depth_ppm": 150.0,
            "secondary_depth_sigma": 2.5,
        },
        confidence=0.75,
        raw={
            "plot_data": {
                "version": 1,
                "phase": phase,
                "flux": flux,
                "flux_err": [0.001] * 100,
                "secondary_window": [0.35, 0.65],
                "primary_window": [-0.05, 0.05],
                "secondary_depth_ppm": 150.0,
            }
        },
    )


@pytest.fixture
def mock_v04_result():
    """Create a mock V04 (depth stability) CheckResult with plot_data."""
    return ok_result(
        id="V04",
        name="Depth Stability",
        metrics={
            "chi2_reduced": 1.2,
            "n_transits_measured": 10,
        },
        confidence=0.80,
        raw={
            "plot_data": {
                "version": 1,
                "epoch_times_btjd": [2458600.0 + i * 5.0 for i in range(10)],
                "depths_ppm": [500.0 + i * 5 for i in range(10)],
                "depth_errs_ppm": [25.0] * 10,
                "mean_depth_ppm": 522.5,
                "expected_scatter_ppm": 30.0,
                "dominating_epoch_idx": 3,
            }
        },
    )


@pytest.fixture
def mock_v05_result():
    """Create a mock V05 (V-shape) CheckResult with plot_data."""
    binned_phase = np.linspace(-0.02, 0.02, 20).tolist()
    binned_flux = (1.0 - 0.001 * (1 - (np.abs(np.array(binned_phase)) / 0.02))).tolist()
    binned_flux_err = [0.0002] * 20

    model_phase = np.linspace(-0.025, 0.025, 100).tolist()
    model_flux = [1.0] * 100

    return ok_result(
        id="V05",
        name="V-Shape",
        metrics={
            "tflat_ttotal_ratio": 0.5,
        },
        confidence=0.70,
        raw={
            "plot_data": {
                "version": 1,
                "binned_phase": binned_phase,
                "binned_flux": binned_flux,
                "binned_flux_err": binned_flux_err,
                "trapezoid_phase": model_phase,
                "trapezoid_flux": model_flux,
                "t_flat_hours": 1.5,
                "t_total_hours": 3.0,
            }
        },
    )


@pytest.fixture
def mock_bundle_with_results(
    mock_v01_result, mock_v02_result, mock_v04_result, mock_v05_result
) -> VettingBundleResult:
    """Create a VettingBundleResult with V01, V02, V04, V05 results."""
    return VettingBundleResult(
        results=[mock_v01_result, mock_v02_result, mock_v04_result, mock_v05_result],
        warnings=[],
        provenance={"version": "1.0.0"},
        inputs_summary={"tic_id": 123456789},
    )


@pytest.fixture
def mock_empty_bundle() -> VettingBundleResult:
    """Create an empty VettingBundleResult."""
    return VettingBundleResult(
        results=[],
        warnings=[],
        provenance={},
        inputs_summary={},
    )


@pytest.fixture(autouse=True)
def close_figures():
    """Automatically close all figures after each test."""
    yield
    plt.close("all")


class TestPlotVettingSummary:
    """Tests for plot_vetting_summary function."""

    def test_creates_figure(
        self, mock_bundle_with_results, mock_lightcurve, mock_candidate
    ):
        """plot_vetting_summary creates a matplotlib Figure."""
        fig = plot_vetting_summary(
            mock_bundle_with_results, mock_lightcurve, mock_candidate
        )

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_default_figsize(
        self, mock_bundle_with_results, mock_lightcurve, mock_candidate
    ):
        """plot_vetting_summary uses default figsize (11, 8.5)."""
        fig = plot_vetting_summary(
            mock_bundle_with_results, mock_lightcurve, mock_candidate
        )

        assert fig.get_figwidth() == 11
        assert fig.get_figheight() == 8.5

    def test_custom_figsize(
        self, mock_bundle_with_results, mock_lightcurve, mock_candidate
    ):
        """plot_vetting_summary respects custom figsize."""
        fig = plot_vetting_summary(
            mock_bundle_with_results,
            mock_lightcurve,
            mock_candidate,
            figsize=(8, 6),
        )

        assert fig.get_figwidth() == 8
        assert fig.get_figheight() == 6

    def test_default_title(
        self, mock_bundle_with_results, mock_lightcurve, mock_candidate
    ):
        """plot_vetting_summary uses default title 'Vetting Summary'."""
        fig = plot_vetting_summary(
            mock_bundle_with_results, mock_lightcurve, mock_candidate
        )

        # Check suptitle
        suptitle = fig._suptitle
        assert suptitle is not None
        assert suptitle.get_text() == "Vetting Summary"

    def test_custom_title(
        self, mock_bundle_with_results, mock_lightcurve, mock_candidate
    ):
        """plot_vetting_summary respects custom title."""
        fig = plot_vetting_summary(
            mock_bundle_with_results,
            mock_lightcurve,
            mock_candidate,
            title="TIC 123456789",
        )

        suptitle = fig._suptitle
        assert suptitle is not None
        assert suptitle.get_text() == "TIC 123456789"

    def test_creates_all_panels_by_default(
        self, mock_bundle_with_results, mock_lightcurve, mock_candidate
    ):
        """plot_vetting_summary creates all 8 panels by default."""
        fig = plot_vetting_summary(
            mock_bundle_with_results, mock_lightcurve, mock_candidate
        )

        # Should have 8 axes (8 panels)
        assert len(fig.axes) == 8

    def test_panel_selection(
        self, mock_bundle_with_results, mock_lightcurve, mock_candidate
    ):
        """plot_vetting_summary respects include_panels parameter."""
        fig = plot_vetting_summary(
            mock_bundle_with_results,
            mock_lightcurve,
            mock_candidate,
            include_panels=["A", "D", "H"],
        )

        # Should have 3 axes (3 panels)
        assert len(fig.axes) == 3

    def test_invalid_panel_raises(
        self, mock_bundle_with_results, mock_lightcurve, mock_candidate
    ):
        """plot_vetting_summary raises ValueError for invalid panel IDs."""
        with pytest.raises(ValueError, match="Invalid panel IDs"):
            plot_vetting_summary(
                mock_bundle_with_results,
                mock_lightcurve,
                mock_candidate,
                include_panels=["A", "Z", "X"],
            )

    def test_handles_missing_checks_gracefully(
        self, mock_empty_bundle, mock_lightcurve, mock_candidate
    ):
        """plot_vetting_summary handles missing checks without error."""
        # Should not raise even with empty bundle
        fig = plot_vetting_summary(mock_empty_bundle, mock_lightcurve, mock_candidate)

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_handles_partial_results(
        self, mock_v01_result, mock_lightcurve, mock_candidate
    ):
        """plot_vetting_summary handles bundle with only some results."""
        bundle = VettingBundleResult(
            results=[mock_v01_result],
            warnings=[],
            provenance={},
            inputs_summary={},
        )

        fig = plot_vetting_summary(bundle, mock_lightcurve, mock_candidate)

        assert fig is not None
        assert len(fig.axes) == 8  # All panels created, some with "not available"

    def test_style_parameter_accepted(
        self, mock_bundle_with_results, mock_lightcurve, mock_candidate
    ):
        """plot_vetting_summary accepts style parameter."""
        for style in ["default", "paper", "presentation"]:
            fig = plot_vetting_summary(
                mock_bundle_with_results,
                mock_lightcurve,
                mock_candidate,
                style=style,
            )
            assert fig is not None
            plt.close(fig)

    def test_invalid_style_raises(
        self, mock_bundle_with_results, mock_lightcurve, mock_candidate
    ):
        """plot_vetting_summary raises ValueError for invalid style."""
        with pytest.raises(ValueError, match="Unknown style"):
            plot_vetting_summary(
                mock_bundle_with_results,
                mock_lightcurve,
                mock_candidate,
                style="invalid_style",
            )


class TestRenderMetricsTable:
    """Tests for _render_metrics_table function."""

    def test_renders_without_error(
        self, mock_bundle_with_results, mock_candidate
    ):
        """_render_metrics_table renders without errors."""
        fig, ax = plt.subplots()
        _render_metrics_table(ax, mock_bundle_with_results, mock_candidate)

        # Should have title
        assert ax.get_title() == "Metrics Summary"

    def test_renders_ephemeris_info(self, mock_bundle_with_results, mock_candidate):
        """_render_metrics_table includes ephemeris information."""
        fig, ax = plt.subplots()
        _render_metrics_table(ax, mock_bundle_with_results, mock_candidate)

        # Check that text was added (ax.texts contains Text objects)
        assert len(ax.texts) > 0
        text_content = ax.texts[0].get_text()
        assert "Period" in text_content
        assert "T0" in text_content
        assert "Duration" in text_content

    def test_renders_check_summary(self, mock_bundle_with_results, mock_candidate):
        """_render_metrics_table includes check summary."""
        fig, ax = plt.subplots()
        _render_metrics_table(ax, mock_bundle_with_results, mock_candidate)

        text_content = ax.texts[0].get_text()
        assert "CHECK SUMMARY" in text_content
        assert "Passed" in text_content

    def test_handles_empty_bundle(self, mock_empty_bundle, mock_candidate):
        """_render_metrics_table handles empty bundle."""
        fig, ax = plt.subplots()
        _render_metrics_table(ax, mock_empty_bundle, mock_candidate)

        # Should not raise, should render empty metrics
        assert len(ax.texts) > 0


class TestSaveVettingReport:
    """Tests for save_vetting_report function."""

    def test_saves_pdf(
        self, mock_bundle_with_results, mock_lightcurve, mock_candidate
    ):
        """save_vetting_report saves PDF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.pdf"

            result = save_vetting_report(
                mock_bundle_with_results,
                mock_lightcurve,
                mock_candidate,
                output_path,
                format="pdf",
            )

            assert result == output_path
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_saves_png(
        self, mock_bundle_with_results, mock_lightcurve, mock_candidate
    ):
        """save_vetting_report saves PNG file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.png"

            result = save_vetting_report(
                mock_bundle_with_results,
                mock_lightcurve,
                mock_candidate,
                output_path,
                format="png",
                dpi=100,  # Lower DPI for faster tests
            )

            assert result == output_path
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_saves_svg(
        self, mock_bundle_with_results, mock_lightcurve, mock_candidate
    ):
        """save_vetting_report saves SVG file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.svg"

            result = save_vetting_report(
                mock_bundle_with_results,
                mock_lightcurve,
                mock_candidate,
                output_path,
                format="svg",
            )

            assert result == output_path
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_accepts_string_path(
        self, mock_bundle_with_results, mock_lightcurve, mock_candidate
    ):
        """save_vetting_report accepts string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/report.pdf"

            result = save_vetting_report(
                mock_bundle_with_results,
                mock_lightcurve,
                mock_candidate,
                output_path,
            )

            assert result == Path(output_path)
            assert Path(output_path).exists()

    def test_passes_kwargs_to_summary(
        self, mock_bundle_with_results, mock_lightcurve, mock_candidate
    ):
        """save_vetting_report passes kwargs to plot_vetting_summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.pdf"

            # This should not raise - kwargs are passed through
            result = save_vetting_report(
                mock_bundle_with_results,
                mock_lightcurve,
                mock_candidate,
                output_path,
                title="Custom Title",
                include_panels=["A", "H"],
            )

            assert result == output_path
            assert output_path.exists()

    def test_respects_dpi(
        self, mock_bundle_with_results, mock_lightcurve, mock_candidate
    ):
        """save_vetting_report respects dpi parameter for PNG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            low_dpi_path = Path(tmpdir) / "low_dpi.png"
            high_dpi_path = Path(tmpdir) / "high_dpi.png"

            save_vetting_report(
                mock_bundle_with_results,
                mock_lightcurve,
                mock_candidate,
                low_dpi_path,
                format="png",
                dpi=50,
            )

            save_vetting_report(
                mock_bundle_with_results,
                mock_lightcurve,
                mock_candidate,
                high_dpi_path,
                format="png",
                dpi=150,
            )

            # Higher DPI should result in larger file
            assert high_dpi_path.stat().st_size > low_dpi_path.stat().st_size
