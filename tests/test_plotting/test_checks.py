"""Tests for plotting check functions (V01-V05)."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from bittr_tess_vetter.plotting.checks import (
    plot_odd_even,
    plot_secondary_eclipse,
    plot_duration_consistency,
    plot_depth_stability,
    plot_v_shape,
)


class TestPlotOddEven:
    """Tests for plot_odd_even function."""

    def test_creates_figure_when_ax_none(self, mock_v01_result):
        """plot_odd_even creates a new figure when ax is None."""
        ax = plot_odd_even(mock_v01_result)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_uses_provided_ax(self, mock_v01_result):
        """plot_odd_even uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax = plot_odd_even(mock_v01_result, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_has_correct_labels(self, mock_v01_result):
        """plot_odd_even sets correct axis labels."""
        ax = plot_odd_even(mock_v01_result)

        assert ax.get_xlabel() == "Epoch"
        assert ax.get_ylabel() == "Depth (ppm)"

    def test_has_legend_by_default(self, mock_v01_result):
        """plot_odd_even shows legend by default."""
        ax = plot_odd_even(mock_v01_result)

        legend = ax.get_legend()
        assert legend is not None

        # Check legend has both odd and even labels
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "Odd" in legend_texts
        assert "Even" in legend_texts

    def test_legend_disabled_when_requested(self, mock_v01_result):
        """plot_odd_even hides legend when show_legend=False."""
        ax = plot_odd_even(mock_v01_result, show_legend=False)

        legend = ax.get_legend()
        assert legend is None

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_odd_even raises ValueError when plot_data is missing."""
        with pytest.raises(ValueError, match="no plot_data"):
            plot_odd_even(mock_result_no_plot_data)

    def test_raises_on_no_raw(self, mock_result_no_raw):
        """plot_odd_even raises ValueError when raw is None."""
        with pytest.raises(ValueError, match="no raw data"):
            plot_odd_even(mock_result_no_raw)

    def test_custom_colors(self, mock_v01_result):
        """plot_odd_even uses custom colors when provided."""
        custom_odd = "#0000ff"  # Blue
        custom_even = "#ff8800"  # Orange

        ax = plot_odd_even(
            mock_v01_result,
            odd_color=custom_odd,
            even_color=custom_even,
        )

        # Get the plotted lines (containers from errorbar)
        containers = ax.containers
        assert len(containers) >= 2

        # Check that colors were applied (errorbar returns ErrorbarContainer)
        # The first container is odd, second is even
        odd_container = containers[0]
        even_container = containers[1]

        # Get color from the data line (first element of container)
        odd_line_color = odd_container[0].get_color()
        even_line_color = even_container[0].get_color()

        assert odd_line_color == custom_odd
        assert even_line_color == custom_even

    def test_show_means_disabled(self, mock_v01_result):
        """plot_odd_even doesn't show mean lines when show_means=False."""
        ax = plot_odd_even(mock_v01_result, show_means=False)

        # Count horizontal lines (axhline creates Line2D objects)
        # With show_means=False, we should have no axhlines
        # Errorbar creates lines too, so we check for lines that span full x-axis
        lines = ax.get_lines()

        # axhline creates lines with xdata [0, 1] in axes coordinates
        # when transformed. We look for lines with constant y
        horizontal_lines = []
        for line in lines:
            ydata = line.get_ydata()
            if len(ydata) == 2 and ydata[0] == ydata[1]:
                # This is a horizontal line (likely from axhline)
                horizontal_lines.append(line)

        # Should be no horizontal mean lines
        assert len(horizontal_lines) == 0

    def test_show_means_enabled_by_default(self, mock_v01_result):
        """plot_odd_even shows mean lines by default."""
        ax = plot_odd_even(mock_v01_result, show_means=True)

        # With show_means=True, we should have 2 axhlines
        lines = ax.get_lines()

        horizontal_lines = []
        for line in lines:
            ydata = line.get_ydata()
            if len(ydata) == 2 and ydata[0] == ydata[1]:
                horizontal_lines.append(line)

        # Should have 2 horizontal mean lines (odd and even)
        assert len(horizontal_lines) == 2

    def test_annotate_sigma_disabled(self, mock_v01_result):
        """plot_odd_even doesn't show sigma in title when annotate_sigma=False."""
        ax = plot_odd_even(mock_v01_result, annotate_sigma=False)

        title = ax.get_title()
        assert "sigma" not in title.lower()
        assert "$\\sigma$" not in title

    def test_annotate_sigma_enabled_by_default(self, mock_v01_result):
        """plot_odd_even shows sigma in title by default."""
        ax = plot_odd_even(mock_v01_result, annotate_sigma=True)

        title = ax.get_title()
        # Title should contain sigma value
        assert "$\\sigma$" in title or "sigma" in title.lower()

    def test_returns_axes(self, mock_v01_result):
        """plot_odd_even returns matplotlib Axes object."""
        result = plot_odd_even(mock_v01_result)

        assert isinstance(result, matplotlib.axes.Axes)

    def test_style_parameter_accepted(self, mock_v01_result):
        """plot_odd_even accepts style parameter without error."""
        # Test each style preset
        for style in ["default", "paper", "presentation"]:
            ax = plot_odd_even(mock_v01_result, style=style)
            assert ax is not None
            plt.close(ax.figure)

    def test_invalid_style_raises(self, mock_v01_result):
        """plot_odd_even raises ValueError for invalid style."""
        with pytest.raises(ValueError, match="Unknown style"):
            plot_odd_even(mock_v01_result, style="invalid_style")


class TestPlotSecondaryEclipse:
    """Tests for plot_secondary_eclipse function."""

    def test_creates_figure_when_ax_none(self, mock_v02_result):
        """plot_secondary_eclipse creates a new figure when ax is None."""
        ax = plot_secondary_eclipse(mock_v02_result)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_uses_provided_ax(self, mock_v02_result):
        """plot_secondary_eclipse uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax = plot_secondary_eclipse(mock_v02_result, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_has_correct_labels(self, mock_v02_result):
        """plot_secondary_eclipse sets correct axis labels."""
        ax = plot_secondary_eclipse(mock_v02_result)

        assert ax.get_xlabel() == "Orbital Phase"
        assert ax.get_ylabel() == "Normalized Flux"
        assert ax.get_title() == "Secondary Eclipse Search"

    def test_has_legend_by_default(self, mock_v02_result):
        """plot_secondary_eclipse shows legend by default."""
        ax = plot_secondary_eclipse(mock_v02_result)

        legend = ax.get_legend()
        assert legend is not None

    def test_legend_disabled_when_requested(self, mock_v02_result):
        """plot_secondary_eclipse hides legend when show_legend=False."""
        ax = plot_secondary_eclipse(mock_v02_result, show_legend=False)

        legend = ax.get_legend()
        assert legend is None

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_secondary_eclipse raises ValueError when plot_data is missing."""
        with pytest.raises(ValueError, match="no plot_data"):
            plot_secondary_eclipse(mock_result_no_plot_data)

    def test_shows_window_shading(self, mock_v02_result):
        """plot_secondary_eclipse shows window shading by default."""
        ax = plot_secondary_eclipse(mock_v02_result, show_windows=True)

        # Check that axvspan created patches
        patches = ax.patches
        assert len(patches) >= 2  # Secondary and primary windows

    def test_windows_disabled_when_requested(self, mock_v02_result):
        """plot_secondary_eclipse hides windows when show_windows=False."""
        ax = plot_secondary_eclipse(mock_v02_result, show_windows=False)

        # Should have no window patches (only data scatter)
        patches = ax.patches
        assert len(patches) == 0

    def test_style_parameter_accepted(self, mock_v02_result):
        """plot_secondary_eclipse accepts style parameter without error."""
        for style in ["default", "paper", "presentation"]:
            ax = plot_secondary_eclipse(mock_v02_result, style=style)
            assert ax is not None
            plt.close(ax.figure)


class TestPlotDurationConsistency:
    """Tests for plot_duration_consistency function."""

    def test_creates_figure_when_ax_none(self, mock_v03_result):
        """plot_duration_consistency creates a new figure when ax is None."""
        ax = plot_duration_consistency(mock_v03_result)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_uses_provided_ax(self, mock_v03_result):
        """plot_duration_consistency uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax = plot_duration_consistency(mock_v03_result, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_has_correct_labels(self, mock_v03_result):
        """plot_duration_consistency sets correct axis labels."""
        ax = plot_duration_consistency(mock_v03_result)

        assert ax.get_ylabel() == "Duration (hours)"
        assert ax.get_title() == "Duration Consistency"

    def test_has_legend_by_default(self, mock_v03_result):
        """plot_duration_consistency shows legend by default."""
        ax = plot_duration_consistency(mock_v03_result)

        legend = ax.get_legend()
        assert legend is not None

    def test_legend_disabled_when_requested(self, mock_v03_result):
        """plot_duration_consistency hides legend when show_legend=False."""
        ax = plot_duration_consistency(mock_v03_result, show_legend=False)

        legend = ax.get_legend()
        assert legend is None

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_duration_consistency raises ValueError when plot_data is missing."""
        with pytest.raises(ValueError, match="no plot_data"):
            plot_duration_consistency(mock_result_no_plot_data)

    def test_has_two_bars(self, mock_v03_result):
        """plot_duration_consistency shows two bars (observed and expected)."""
        ax = plot_duration_consistency(mock_v03_result)

        # Count bar containers (filter out ErrorbarContainer)
        from matplotlib.container import BarContainer
        bar_containers = [c for c in ax.containers if isinstance(c, BarContainer)]
        assert len(bar_containers) == 2  # Observed and Expected bars

    def test_error_bars_shown_by_default(self, mock_v03_result):
        """plot_duration_consistency shows error bars by default."""
        ax = plot_duration_consistency(mock_v03_result, show_error=True)

        # Check that errorbar created lines (caplines)
        lines = ax.get_lines()
        assert len(lines) > 0  # Error bar creates lines

    def test_style_parameter_accepted(self, mock_v03_result):
        """plot_duration_consistency accepts style parameter without error."""
        for style in ["default", "paper", "presentation"]:
            ax = plot_duration_consistency(mock_v03_result, style=style)
            assert ax is not None
            plt.close(ax.figure)


class TestPlotDepthStability:
    """Tests for plot_depth_stability function."""

    def test_creates_figure_when_ax_none(self, mock_v04_result):
        """plot_depth_stability creates a new figure when ax is None."""
        ax = plot_depth_stability(mock_v04_result)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_uses_provided_ax(self, mock_v04_result):
        """plot_depth_stability uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax = plot_depth_stability(mock_v04_result, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_has_correct_labels(self, mock_v04_result):
        """plot_depth_stability sets correct axis labels."""
        ax = plot_depth_stability(mock_v04_result)

        assert ax.get_xlabel() == "Time (BTJD)"
        assert ax.get_ylabel() == "Depth (ppm)"
        assert ax.get_title() == "Per-Epoch Depth Stability"

    def test_has_legend_by_default(self, mock_v04_result):
        """plot_depth_stability shows legend by default."""
        ax = plot_depth_stability(mock_v04_result)

        legend = ax.get_legend()
        assert legend is not None

    def test_legend_disabled_when_requested(self, mock_v04_result):
        """plot_depth_stability hides legend when show_legend=False."""
        ax = plot_depth_stability(mock_v04_result, show_legend=False)

        legend = ax.get_legend()
        assert legend is None

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_depth_stability raises ValueError when plot_data is missing."""
        with pytest.raises(ValueError, match="no plot_data"):
            plot_depth_stability(mock_result_no_plot_data)

    def test_shows_mean_line(self, mock_v04_result):
        """plot_depth_stability shows mean depth line by default."""
        ax = plot_depth_stability(mock_v04_result, show_mean=True)

        # Check that axhline created horizontal lines
        lines = ax.get_lines()
        horizontal_lines = [
            line for line in lines
            if len(line.get_ydata()) == 2 and line.get_ydata()[0] == line.get_ydata()[1]
        ]
        assert len(horizontal_lines) >= 1  # Mean line

    def test_shows_scatter_band(self, mock_v04_result):
        """plot_depth_stability shows scatter band by default."""
        ax = plot_depth_stability(mock_v04_result, show_scatter_band=True)

        # Check that axhspan created patches
        patches = ax.patches
        assert len(patches) >= 1  # Scatter band

    def test_style_parameter_accepted(self, mock_v04_result):
        """plot_depth_stability accepts style parameter without error."""
        for style in ["default", "paper", "presentation"]:
            ax = plot_depth_stability(mock_v04_result, style=style)
            assert ax is not None
            plt.close(ax.figure)


class TestPlotVShape:
    """Tests for plot_v_shape function."""

    def test_creates_figure_when_ax_none(self, mock_v05_result):
        """plot_v_shape creates a new figure when ax is None."""
        ax = plot_v_shape(mock_v05_result)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_uses_provided_ax(self, mock_v05_result):
        """plot_v_shape uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax = plot_v_shape(mock_v05_result, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_has_correct_labels(self, mock_v05_result):
        """plot_v_shape sets correct axis labels."""
        ax = plot_v_shape(mock_v05_result)

        assert ax.get_xlabel() == "Phase"
        assert ax.get_ylabel() == "Normalized Flux"
        assert ax.get_title() == "Transit Shape Analysis"

    def test_has_legend_by_default(self, mock_v05_result):
        """plot_v_shape shows legend by default."""
        ax = plot_v_shape(mock_v05_result)

        legend = ax.get_legend()
        assert legend is not None

    def test_legend_disabled_when_requested(self, mock_v05_result):
        """plot_v_shape hides legend when show_legend=False."""
        ax = plot_v_shape(mock_v05_result, show_legend=False)

        legend = ax.get_legend()
        assert legend is None

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_v_shape raises ValueError when plot_data is missing."""
        with pytest.raises(ValueError, match="no plot_data"):
            plot_v_shape(mock_result_no_plot_data)

    def test_shows_trapezoid_model(self, mock_v05_result):
        """plot_v_shape shows trapezoid model by default."""
        ax = plot_v_shape(mock_v05_result, show_model=True)

        # Check that there are lines plotted (model)
        lines = ax.get_lines()
        assert len(lines) >= 1  # At least model line

    def test_model_disabled_when_requested(self, mock_v05_result):
        """plot_v_shape hides model when show_model=False."""
        ax = plot_v_shape(mock_v05_result, show_model=False)

        # Should only have errorbar lines, no model line
        # The model line would have 100 points (from model_phase array)
        lines = ax.get_lines()
        # Check that no line has 100 points (model has exactly 100 points)
        for line in lines:
            assert len(line.get_xdata()) != 100

    def test_style_parameter_accepted(self, mock_v05_result):
        """plot_v_shape accepts style parameter without error."""
        for style in ["default", "paper", "presentation"]:
            ax = plot_v_shape(mock_v05_result, style=style)
            assert ax is not None
            plt.close(ax.figure)
