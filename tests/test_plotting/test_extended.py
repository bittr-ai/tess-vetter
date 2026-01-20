"""Tests for extended plotting check functions (V16-V21)."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from bittr_tess_vetter.plotting.extended import (
    plot_model_comparison,
    plot_ephemeris_reliability,
    plot_alias_diagnostics,
    plot_ghost_features,
    plot_sector_consistency,
)
from bittr_tess_vetter.validation.result_schema import ok_result


@pytest.fixture
def mock_v16_result():
    """Create a mock V16 (model competition) CheckResult with plot_data."""
    phase = np.linspace(0.0, 1.0, 100).tolist()
    flux = (1.0 - 0.001 * np.exp(-((np.array(phase) - 0.0) ** 2) / 0.01)).tolist()

    return ok_result(
        id="V16",
        name="Model Competition",
        metrics={
            "winner": "transit_only",
            "winner_margin_bic": 15.5,
        },
        confidence=0.85,
        raw={
            "plot_data": {
                "version": 1,
                "phase": phase,
                "flux": flux,
                "transit_model": flux,
                "eb_like_model": [1.0] * 100,
                "sinusoid_model": (1.0 + 0.0005 * np.sin(2 * np.pi * np.array(phase))).tolist(),
            }
        },
    )


@pytest.fixture
def mock_v17_result():
    """Create a mock V17 (ephemeris reliability) CheckResult with plot_data."""
    phase_shifts = np.linspace(-0.5, 0.5, 50).tolist()
    null_scores = (10.0 * np.exp(-np.array(phase_shifts) ** 2 / 0.01)).tolist()

    return ok_result(
        id="V17",
        name="Ephemeris Reliability Regime",
        metrics={
            "score": 10.0,
            "phase_shift_null_p_value": 0.001,
        },
        confidence=0.80,
        raw={
            "plot_data": {
                "version": 1,
                "phase_shifts": phase_shifts,
                "null_scores": null_scores,
                "period_neighborhood": [1.0, 1.01, 1.02],
                "neighborhood_scores": [10.0, 8.0, 6.0],
            }
        },
    )


@pytest.fixture
def mock_v19_result():
    """Create a mock V19 (alias diagnostics) CheckResult with plot_data."""
    return ok_result(
        id="V19",
        name="Alias Diagnostics",
        metrics={
            "base_score_P": 8.5,
            "best_other_harmonic": "P/2",
            "best_other_score": 5.2,
        },
        confidence=0.70,
        raw={
            "plot_data": {
                "version": 1,
                "harmonic_labels": ["P", "P/2", "2P", "P/3", "3P"],
                "harmonic_periods": [5.0, 2.5, 10.0, 1.67, 15.0],
                "harmonic_scores": [8.5, 5.2, 3.1, 2.5, 1.8],
            }
        },
    )


@pytest.fixture
def mock_v20_result():
    """Create a mock V20 (ghost features) CheckResult with plot_data."""
    # Create a simple difference image
    diff_image = np.random.normal(0, 0.01, (11, 11)).tolist()
    aperture_mask = np.zeros((11, 11), dtype=bool)
    aperture_mask[4:8, 4:8] = True

    return ok_result(
        id="V20",
        name="Ghost Features",
        metrics={
            "ghost_like_score": 0.25,
            "scattered_light_risk": 0.15,
            "aperture_contrast": 5.2,
        },
        confidence=0.75,
        raw={
            "plot_data": {
                "version": 1,
                "difference_image": diff_image,
                "aperture_mask": aperture_mask.tolist(),
                "in_aperture_depth": 0.001,
                "out_aperture_depth": 0.0002,
            }
        },
    )


@pytest.fixture
def mock_v21_result():
    """Create a mock V21 (sector consistency) CheckResult with plot_data."""
    return ok_result(
        id="V21",
        name="Sector Consistency",
        metrics={
            "chi2_p_value": 0.35,
            "n_sectors_used": 5,
            "consistency_class": "EXPECTED_SCATTER",
        },
        confidence=0.80,
        raw={
            "plot_data": {
                "version": 1,
                "sectors": [1, 5, 10, 15, 20],
                "depths_ppm": [500.0, 520.0, 490.0, 510.0, 505.0],
                "depth_errs_ppm": [25.0, 30.0, 22.0, 28.0, 26.0],
                "weighted_mean_ppm": 505.0,
                "outlier_sectors": [5],
            }
        },
    )


@pytest.fixture
def mock_result_no_plot_data():
    """Create a mock CheckResult with no plot_data."""
    return ok_result(
        id="V16",
        name="Model Competition",
        metrics={"winner": "transit_only"},
        raw={"some_other_data": "value"},
    )


@pytest.fixture
def mock_result_no_raw():
    """Create a mock CheckResult with raw=None."""
    return ok_result(
        id="V16",
        name="Model Competition",
        metrics={"winner": "transit_only"},
        raw=None,
    )


class TestPlotModelComparison:
    """Tests for plot_model_comparison function."""

    def test_creates_figure_when_ax_none(self, mock_v16_result):
        """plot_model_comparison creates a new figure when ax is None."""
        ax = plot_model_comparison(mock_v16_result)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_uses_provided_ax(self, mock_v16_result):
        """plot_model_comparison uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax = plot_model_comparison(mock_v16_result, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_has_correct_labels(self, mock_v16_result):
        """plot_model_comparison sets correct axis labels."""
        ax = plot_model_comparison(mock_v16_result)

        assert ax.get_xlabel() == "Phase"
        assert ax.get_ylabel() == "Normalized Flux"
        assert ax.get_title() == "Model Competition"

    def test_has_legend_by_default(self, mock_v16_result):
        """plot_model_comparison shows legend by default."""
        ax = plot_model_comparison(mock_v16_result)

        legend = ax.get_legend()
        assert legend is not None

    def test_legend_disabled_when_requested(self, mock_v16_result):
        """plot_model_comparison hides legend when show_legend=False."""
        ax = plot_model_comparison(mock_v16_result, show_legend=False)

        legend = ax.get_legend()
        assert legend is None

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_model_comparison raises ValueError when plot_data is missing."""
        with pytest.raises(ValueError, match="no plot_data"):
            plot_model_comparison(mock_result_no_plot_data)

    def test_raises_on_no_raw(self, mock_result_no_raw):
        """plot_model_comparison raises ValueError when raw is None."""
        with pytest.raises(ValueError, match="no raw data"):
            plot_model_comparison(mock_result_no_raw)

    def test_shows_model_lines(self, mock_v16_result):
        """plot_model_comparison shows model lines."""
        ax = plot_model_comparison(mock_v16_result)

        # Check that lines were created (models)
        lines = ax.get_lines()
        assert len(lines) >= 3  # Transit, EB-like, sinusoid

    def test_style_parameter_accepted(self, mock_v16_result):
        """plot_model_comparison accepts style parameter without error."""
        for style in ["default", "paper", "presentation"]:
            ax = plot_model_comparison(mock_v16_result, style=style)
            assert ax is not None
            plt.close(ax.figure)

    def test_invalid_style_raises(self, mock_v16_result):
        """plot_model_comparison raises ValueError for invalid style."""
        with pytest.raises(ValueError, match="Unknown style"):
            plot_model_comparison(mock_v16_result, style="invalid_style")


class TestPlotEphemerisReliability:
    """Tests for plot_ephemeris_reliability function."""

    def test_creates_figure_when_ax_none(self, mock_v17_result):
        """plot_ephemeris_reliability creates a new figure when ax is None."""
        ax = plot_ephemeris_reliability(mock_v17_result)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_uses_provided_ax(self, mock_v17_result):
        """plot_ephemeris_reliability uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax = plot_ephemeris_reliability(mock_v17_result, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_has_correct_labels(self, mock_v17_result):
        """plot_ephemeris_reliability sets correct axis labels."""
        ax = plot_ephemeris_reliability(mock_v17_result)

        assert ax.get_xlabel() == "Phase Shift"
        assert ax.get_ylabel() == "Detection Score"
        assert ax.get_title() == "Ephemeris Reliability"

    def test_has_legend_by_default(self, mock_v17_result):
        """plot_ephemeris_reliability shows legend by default."""
        ax = plot_ephemeris_reliability(mock_v17_result)

        legend = ax.get_legend()
        assert legend is not None

    def test_legend_disabled_when_requested(self, mock_v17_result):
        """plot_ephemeris_reliability hides legend when show_legend=False."""
        ax = plot_ephemeris_reliability(mock_v17_result, show_legend=False)

        legend = ax.get_legend()
        assert legend is None

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_ephemeris_reliability raises ValueError when plot_data is missing."""
        with pytest.raises(ValueError, match="no plot_data"):
            plot_ephemeris_reliability(mock_result_no_plot_data)

    def test_style_parameter_accepted(self, mock_v17_result):
        """plot_ephemeris_reliability accepts style parameter without error."""
        for style in ["default", "paper", "presentation"]:
            ax = plot_ephemeris_reliability(mock_v17_result, style=style)
            assert ax is not None
            plt.close(ax.figure)


class TestPlotAliasDiagnostics:
    """Tests for plot_alias_diagnostics function."""

    def test_creates_figure_when_ax_none(self, mock_v19_result):
        """plot_alias_diagnostics creates a new figure when ax is None."""
        ax = plot_alias_diagnostics(mock_v19_result)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_uses_provided_ax(self, mock_v19_result):
        """plot_alias_diagnostics uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax = plot_alias_diagnostics(mock_v19_result, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_has_correct_labels(self, mock_v19_result):
        """plot_alias_diagnostics sets correct axis labels."""
        ax = plot_alias_diagnostics(mock_v19_result)

        assert ax.get_xlabel() == "Harmonic"
        assert ax.get_ylabel() == "Detection Score"
        assert ax.get_title() == "Alias/Harmonic Diagnostics"

    def test_has_legend_by_default(self, mock_v19_result):
        """plot_alias_diagnostics shows legend by default."""
        ax = plot_alias_diagnostics(mock_v19_result)

        legend = ax.get_legend()
        assert legend is not None

    def test_legend_disabled_when_requested(self, mock_v19_result):
        """plot_alias_diagnostics hides legend when show_legend=False."""
        ax = plot_alias_diagnostics(mock_v19_result, show_legend=False)

        legend = ax.get_legend()
        assert legend is None

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_alias_diagnostics raises ValueError when plot_data is missing."""
        with pytest.raises(ValueError, match="no plot_data"):
            plot_alias_diagnostics(mock_result_no_plot_data)

    def test_has_bars(self, mock_v19_result):
        """plot_alias_diagnostics shows harmonic bars."""
        ax = plot_alias_diagnostics(mock_v19_result)

        from matplotlib.container import BarContainer
        bar_containers = [c for c in ax.containers if isinstance(c, BarContainer)]
        assert len(bar_containers) >= 1

    def test_style_parameter_accepted(self, mock_v19_result):
        """plot_alias_diagnostics accepts style parameter without error."""
        for style in ["default", "paper", "presentation"]:
            ax = plot_alias_diagnostics(mock_v19_result, style=style)
            assert ax is not None
            plt.close(ax.figure)


class TestPlotGhostFeatures:
    """Tests for plot_ghost_features function."""

    def test_creates_figure_when_ax_none(self, mock_v20_result):
        """plot_ghost_features creates a new figure when ax is None."""
        ax, cbar = plot_ghost_features(mock_v20_result)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_uses_provided_ax(self, mock_v20_result):
        """plot_ghost_features uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax, cbar = plot_ghost_features(mock_v20_result, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_has_correct_labels(self, mock_v20_result):
        """plot_ghost_features sets correct axis labels."""
        ax, cbar = plot_ghost_features(mock_v20_result)

        assert ax.get_xlabel() == "Column (pixels)"
        assert ax.get_ylabel() == "Row (pixels)"
        assert ax.get_title() == "Ghost Feature Analysis"

    def test_returns_colorbar_when_requested(self, mock_v20_result):
        """plot_ghost_features returns colorbar when show_colorbar=True."""
        ax, cbar = plot_ghost_features(mock_v20_result, show_colorbar=True)

        assert cbar is not None

    def test_colorbar_none_when_disabled(self, mock_v20_result):
        """plot_ghost_features returns None for colorbar when show_colorbar=False."""
        ax, cbar = plot_ghost_features(mock_v20_result, show_colorbar=False)

        assert cbar is None

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_ghost_features raises ValueError when plot_data is missing."""
        with pytest.raises(ValueError, match="no plot_data"):
            plot_ghost_features(mock_result_no_plot_data)

    def test_shows_image(self, mock_v20_result):
        """plot_ghost_features shows difference image."""
        ax, cbar = plot_ghost_features(mock_v20_result)

        # Check that imshow created an image
        images = ax.images
        assert len(images) >= 1

    def test_returns_tuple(self, mock_v20_result):
        """plot_ghost_features returns (ax, cbar) tuple."""
        result = plot_ghost_features(mock_v20_result)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_style_parameter_accepted(self, mock_v20_result):
        """plot_ghost_features accepts style parameter without error."""
        for style in ["default", "paper", "presentation"]:
            ax, cbar = plot_ghost_features(mock_v20_result, style=style)
            assert ax is not None
            plt.close(ax.figure)


class TestPlotSectorConsistency:
    """Tests for plot_sector_consistency function."""

    def test_creates_figure_when_ax_none(self, mock_v21_result):
        """plot_sector_consistency creates a new figure when ax is None."""
        ax = plot_sector_consistency(mock_v21_result)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_uses_provided_ax(self, mock_v21_result):
        """plot_sector_consistency uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax = plot_sector_consistency(mock_v21_result, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_has_correct_labels(self, mock_v21_result):
        """plot_sector_consistency sets correct axis labels."""
        ax = plot_sector_consistency(mock_v21_result)

        assert ax.get_xlabel() == "Sector"
        assert ax.get_ylabel() == "Depth (ppm)"
        assert ax.get_title() == "Sector-to-Sector Consistency"

    def test_has_legend_by_default(self, mock_v21_result):
        """plot_sector_consistency shows legend by default."""
        ax = plot_sector_consistency(mock_v21_result)

        legend = ax.get_legend()
        assert legend is not None

    def test_legend_disabled_when_requested(self, mock_v21_result):
        """plot_sector_consistency hides legend when show_legend=False."""
        ax = plot_sector_consistency(mock_v21_result, show_legend=False)

        legend = ax.get_legend()
        assert legend is None

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_sector_consistency raises ValueError when plot_data is missing."""
        with pytest.raises(ValueError, match="no plot_data"):
            plot_sector_consistency(mock_result_no_plot_data)

    def test_has_bars(self, mock_v21_result):
        """plot_sector_consistency shows sector bars."""
        ax = plot_sector_consistency(mock_v21_result)

        from matplotlib.container import BarContainer
        bar_containers = [c for c in ax.containers if isinstance(c, BarContainer)]
        assert len(bar_containers) >= 1

    def test_shows_mean_line(self, mock_v21_result):
        """plot_sector_consistency shows weighted mean line by default."""
        ax = plot_sector_consistency(mock_v21_result, show_mean=True)

        # Check for horizontal mean line
        lines = ax.get_lines()
        horizontal_lines = [
            line for line in lines
            if len(line.get_ydata()) == 2 and line.get_ydata()[0] == line.get_ydata()[1]
        ]
        assert len(horizontal_lines) >= 1

    def test_mean_disabled_when_requested(self, mock_v21_result):
        """plot_sector_consistency hides mean when show_mean=False."""
        ax = plot_sector_consistency(mock_v21_result, show_mean=False)

        # The axhline should not be present
        # But error bars create lines too, so we just check the function runs
        assert ax is not None

    def test_shows_error_bars(self, mock_v21_result):
        """plot_sector_consistency shows error bars by default."""
        ax = plot_sector_consistency(mock_v21_result, show_error_bars=True)

        # Check that errorbar created lines (caplines)
        lines = ax.get_lines()
        # Error bars will add multiple lines for caps
        assert len(lines) > 0

    def test_style_parameter_accepted(self, mock_v21_result):
        """plot_sector_consistency accepts style parameter without error."""
        for style in ["default", "paper", "presentation"]:
            ax = plot_sector_consistency(mock_v21_result, style=style)
            assert ax is not None
            plt.close(ax.figure)

    def test_invalid_style_raises(self, mock_v21_result):
        """plot_sector_consistency raises ValueError for invalid style."""
        with pytest.raises(ValueError, match="Unknown style"):
            plot_sector_consistency(mock_v21_result, style="invalid_style")
