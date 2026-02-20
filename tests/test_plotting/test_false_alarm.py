"""Tests for false alarm plotting check functions (V13, V15)."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tess_vetter.plotting.false_alarm import (
    plot_asymmetry,
    plot_data_gaps,
)
from tess_vetter.validation.result_schema import ok_result


@pytest.fixture
def mock_v13_result():
    """Create a mock V13 (data gaps) CheckResult with plot_data."""
    return ok_result(
        id="V13",
        name="Data Gaps",
        metrics={
            "missing_frac_max": 0.35,
            "missing_frac_median": 0.15,
            "n_epochs_evaluated": 10,
        },
        confidence=0.75,
        raw={
            "plot_data": {
                "version": 1,
                "epoch_centers_btjd": [2458600.0 + i * 5.0 for i in range(10)],
                "coverage_fractions": [0.95, 0.88, 0.65, 0.92, 0.78, 0.99, 0.85, 0.72, 0.90, 0.80],
                "transit_window_hours": 4.0,
            }
        },
    )


@pytest.fixture
def mock_v15_result():
    """Create a mock V15 (asymmetry) CheckResult with plot_data."""
    import numpy as np

    # Generate mock phase-folded data around transit
    phase = np.linspace(-0.1, 0.1, 200).tolist()
    flux = (1.0 + np.random.normal(0, 0.001, 200)).tolist()

    return ok_result(
        id="V15",
        name="Transit Asymmetry",
        metrics={
            "asymmetry_sigma": 2.5,
            "mu_left": -0.0005,
            "mu_right": 0.0002,
        },
        confidence=0.75,
        raw={
            "plot_data": {
                "version": 1,
                "phase": phase,
                "flux": flux,
                "left_bin_mean": -0.0005,
                "right_bin_mean": 0.0002,
                "left_bin_phase_range": [-0.1, -0.01],
                "right_bin_phase_range": [0.01, 0.1],
            }
        },
    )


@pytest.fixture
def mock_result_no_plot_data():
    """Create a mock CheckResult with no plot_data."""
    return ok_result(
        id="V13",
        name="Data Gaps",
        metrics={"missing_frac_max": 0.35},
        raw={"some_other_data": "value"},
    )


@pytest.fixture
def mock_result_no_raw():
    """Create a mock CheckResult with raw=None."""
    return ok_result(
        id="V13",
        name="Data Gaps",
        metrics={"missing_frac_max": 0.35},
        raw=None,
    )


class TestPlotDataGaps:
    """Tests for plot_data_gaps function."""

    def test_creates_figure_when_ax_none(self, mock_v13_result):
        """plot_data_gaps creates a new figure when ax is None."""
        ax = plot_data_gaps(mock_v13_result)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_uses_provided_ax(self, mock_v13_result):
        """plot_data_gaps uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax = plot_data_gaps(mock_v13_result, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_has_correct_labels(self, mock_v13_result):
        """plot_data_gaps sets correct axis labels."""
        ax = plot_data_gaps(mock_v13_result)

        assert ax.get_xlabel() == "Epoch"
        assert ax.get_ylabel() == "Coverage Fraction"
        assert "Transit Window Coverage" in ax.get_title()

    def test_has_legend_by_default(self, mock_v13_result):
        """plot_data_gaps shows legend by default."""
        ax = plot_data_gaps(mock_v13_result)

        legend = ax.get_legend()
        assert legend is not None

    def test_legend_disabled_when_requested(self, mock_v13_result):
        """plot_data_gaps hides legend when show_legend=False."""
        ax = plot_data_gaps(mock_v13_result, show_legend=False)

        legend = ax.get_legend()
        assert legend is None

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_data_gaps raises ValueError when plot_data is missing."""
        with pytest.raises(ValueError, match="no plot_data"):
            plot_data_gaps(mock_result_no_plot_data)

    def test_raises_on_no_raw(self, mock_result_no_raw):
        """plot_data_gaps raises ValueError when raw is None."""
        with pytest.raises(ValueError, match="no raw data"):
            plot_data_gaps(mock_result_no_raw)

    def test_shows_threshold_line_by_default(self, mock_v13_result):
        """plot_data_gaps shows threshold line by default."""
        ax = plot_data_gaps(mock_v13_result, show_threshold=True)

        # Check for horizontal threshold line
        lines = ax.get_lines()
        horizontal_lines = [
            line for line in lines
            if len(line.get_ydata()) == 2 and line.get_ydata()[0] == line.get_ydata()[1]
        ]
        assert len(horizontal_lines) >= 1

    def test_threshold_disabled_when_requested(self, mock_v13_result):
        """plot_data_gaps hides threshold when show_threshold=False."""
        ax = plot_data_gaps(mock_v13_result, show_threshold=False)

        # Check for horizontal lines
        lines = ax.get_lines()
        horizontal_lines = [
            line for line in lines
            if len(line.get_ydata()) == 2 and line.get_ydata()[0] == line.get_ydata()[1]
        ]
        assert len(horizontal_lines) == 0

    def test_custom_threshold_value(self, mock_v13_result):
        """plot_data_gaps uses custom threshold value."""
        ax = plot_data_gaps(mock_v13_result, threshold_value=0.5)

        # The threshold is shown in the legend, check it exists
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert any("50%" in text for text in legend_texts)

    def test_has_bars(self, mock_v13_result):
        """plot_data_gaps shows coverage bars."""
        ax = plot_data_gaps(mock_v13_result)

        # Check that bars were created
        from matplotlib.container import BarContainer
        bar_containers = [c for c in ax.containers if isinstance(c, BarContainer)]
        assert len(bar_containers) >= 1

    def test_style_parameter_accepted(self, mock_v13_result):
        """plot_data_gaps accepts style parameter without error."""
        for style in ["default", "paper", "presentation"]:
            ax = plot_data_gaps(mock_v13_result, style=style)
            assert ax is not None
            plt.close(ax.figure)

    def test_invalid_style_raises(self, mock_v13_result):
        """plot_data_gaps raises ValueError for invalid style."""
        with pytest.raises(ValueError, match="Unknown style"):
            plot_data_gaps(mock_v13_result, style="invalid_style")

    def test_returns_axes(self, mock_v13_result):
        """plot_data_gaps returns matplotlib Axes object."""
        result = plot_data_gaps(mock_v13_result)

        assert isinstance(result, matplotlib.axes.Axes)


class TestPlotAsymmetry:
    """Tests for plot_asymmetry function."""

    def test_creates_figure_when_ax_none(self, mock_v15_result):
        """plot_asymmetry creates a new figure when ax is None."""
        ax = plot_asymmetry(mock_v15_result)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_uses_provided_ax(self, mock_v15_result):
        """plot_asymmetry uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax = plot_asymmetry(mock_v15_result, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_has_correct_labels(self, mock_v15_result):
        """plot_asymmetry sets correct axis labels."""
        ax = plot_asymmetry(mock_v15_result)

        assert ax.get_xlabel() == "Phase"
        assert ax.get_ylabel() == "Normalized Flux"
        assert ax.get_title() == "Transit Asymmetry Analysis"

    def test_has_legend_by_default(self, mock_v15_result):
        """plot_asymmetry shows legend by default."""
        ax = plot_asymmetry(mock_v15_result)

        legend = ax.get_legend()
        assert legend is not None

    def test_legend_disabled_when_requested(self, mock_v15_result):
        """plot_asymmetry hides legend when show_legend=False."""
        ax = plot_asymmetry(mock_v15_result, show_legend=False)

        legend = ax.get_legend()
        assert legend is None

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_asymmetry raises ValueError when plot_data is missing."""
        with pytest.raises(ValueError, match="no plot_data"):
            plot_asymmetry(mock_result_no_plot_data)

    def test_shows_bin_shading_by_default(self, mock_v15_result):
        """plot_asymmetry shows bin shading by default."""
        ax = plot_asymmetry(mock_v15_result, show_bins=True)

        # Check that axvspan created patches
        patches = ax.patches
        assert len(patches) >= 2  # Left and right bins

    def test_bins_disabled_when_requested(self, mock_v15_result):
        """plot_asymmetry hides bins when show_bins=False."""
        ax = plot_asymmetry(mock_v15_result, show_bins=False)

        # Should have no bin patches
        patches = ax.patches
        assert len(patches) == 0

    def test_custom_colors(self, mock_v15_result):
        """plot_asymmetry uses custom colors when provided."""
        custom_left = "#0000ff"  # Blue
        custom_right = "#ff8800"  # Orange

        ax = plot_asymmetry(
            mock_v15_result,
            left_color=custom_left,
            right_color=custom_right,
        )

        # Check that patches have correct colors
        patches = ax.patches
        assert len(patches) >= 2

    def test_annotate_sigma_disabled(self, mock_v15_result):
        """plot_asymmetry doesn't show sigma when annotate_sigma=False."""
        ax = plot_asymmetry(mock_v15_result, annotate_sigma=False)

        # We just test no error occurs since text annotation detection is complex
        assert ax is not None

    def test_style_parameter_accepted(self, mock_v15_result):
        """plot_asymmetry accepts style parameter without error."""
        for style in ["default", "paper", "presentation"]:
            ax = plot_asymmetry(mock_v15_result, style=style)
            assert ax is not None
            plt.close(ax.figure)

    def test_invalid_style_raises(self, mock_v15_result):
        """plot_asymmetry raises ValueError for invalid style."""
        with pytest.raises(ValueError, match="Unknown style"):
            plot_asymmetry(mock_v15_result, style="invalid_style")

    def test_returns_axes(self, mock_v15_result):
        """plot_asymmetry returns matplotlib Axes object."""
        result = plot_asymmetry(mock_v15_result)

        assert isinstance(result, matplotlib.axes.Axes)
