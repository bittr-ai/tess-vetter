"""Tests for plotting check functions (V01-V05)."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from bittr_tess_vetter.plotting.checks import plot_odd_even


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
