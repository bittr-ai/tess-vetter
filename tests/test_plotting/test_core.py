"""Tests for bittr_tess_vetter.plotting._core module."""

from __future__ import annotations

import pytest

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bittr_tess_vetter.plotting._core import (
    compute_subplot_grid,
    ensure_ax,
    extract_plot_data,
    get_sector_color,
    style_context,
)
from bittr_tess_vetter.plotting._styles import STYLES


class TestEnsureAx:
    """Tests for ensure_ax function."""

    def test_creates_figure_when_none(self):
        """ensure_ax creates new figure and axes when ax=None."""
        fig, ax = ensure_ax()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert ax.figure is fig

    def test_uses_provided_ax(self):
        """ensure_ax uses provided axes and returns its figure."""
        existing_fig, existing_ax = plt.subplots()

        fig, ax = ensure_ax(existing_ax)

        assert fig is existing_fig
        assert ax is existing_ax

    def test_returns_correct_figure_for_subplot(self):
        """ensure_ax returns correct figure for subplot axes."""
        fig, axes = plt.subplots(2, 2)

        returned_fig, returned_ax = ensure_ax(axes[0, 1])

        assert returned_fig is fig
        assert returned_ax is axes[0, 1]


class TestStyleContext:
    """Tests for style_context context manager."""

    def test_applies_default_style(self):
        """style_context applies default style within context."""
        original_figsize = plt.rcParams["figure.figsize"]

        with style_context("default"):
            in_context_figsize = plt.rcParams["figure.figsize"]
            assert list(in_context_figsize) == list(STYLES["default"]["figure.figsize"])

        # Verify reverted after context
        assert list(plt.rcParams["figure.figsize"]) == list(original_figsize)

    def test_applies_paper_style(self):
        """style_context applies paper style within context."""
        with style_context("paper"):
            assert list(plt.rcParams["figure.figsize"]) == list(
                STYLES["paper"]["figure.figsize"]
            )
            assert plt.rcParams["figure.dpi"] == STYLES["paper"]["figure.dpi"]

    def test_applies_presentation_style(self):
        """style_context applies presentation style within context."""
        with style_context("presentation"):
            assert list(plt.rcParams["figure.figsize"]) == list(
                STYLES["presentation"]["figure.figsize"]
            )
            assert plt.rcParams["axes.grid"] is True

    def test_reverts_on_exit(self):
        """style_context reverts rcParams on normal exit."""
        original_fontsize = plt.rcParams["font.size"]

        with style_context("presentation"):
            # Font size changed
            assert plt.rcParams["font.size"] != original_fontsize

        # Font size reverted
        assert plt.rcParams["font.size"] == original_fontsize

    def test_reverts_on_exception(self):
        """style_context reverts rcParams even on exception."""
        original_fontsize = plt.rcParams["font.size"]

        with pytest.raises(RuntimeError):
            with style_context("presentation"):
                raise RuntimeError("test error")

        # Font size still reverted
        assert plt.rcParams["font.size"] == original_fontsize

    def test_raises_on_unknown_style(self):
        """style_context raises ValueError for unknown style."""
        with pytest.raises(ValueError, match="Unknown style"):
            with style_context("nonexistent"):
                pass


class TestExtractPlotData:
    """Tests for extract_plot_data function."""

    def test_extracts_plot_data(self, mock_v01_result):
        """extract_plot_data returns plot_data dict."""
        data = extract_plot_data(mock_v01_result, ["version", "odd_epochs"])

        assert data["version"] == 1
        assert data["odd_epochs"] == [1, 3, 5, 7, 9]

    def test_returns_full_dict(self, mock_v01_result):
        """extract_plot_data returns the complete plot_data dict."""
        data = extract_plot_data(mock_v01_result, ["version"])

        # Should have all the keys from plot_data
        assert "odd_depths_ppm" in data
        assert "even_depths_ppm" in data
        assert "mean_odd_ppm" in data

    def test_raises_on_missing_raw(self, mock_result_no_raw):
        """extract_plot_data raises ValueError when raw is None."""
        with pytest.raises(ValueError, match="has no raw data"):
            extract_plot_data(mock_result_no_raw, ["version"])

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """extract_plot_data raises ValueError when plot_data is missing."""
        with pytest.raises(ValueError, match="has no plot_data"):
            extract_plot_data(mock_result_no_plot_data, ["version"])

    def test_raises_on_missing_required_keys(self, mock_v01_result):
        """extract_plot_data raises ValueError when required key is missing."""
        with pytest.raises(ValueError, match="missing required keys"):
            extract_plot_data(mock_v01_result, ["nonexistent_key"])

    def test_raises_lists_all_missing_keys(self, mock_v01_result):
        """extract_plot_data lists all missing keys in error message."""
        with pytest.raises(ValueError, match=r"\['key1', 'key2'\]"):
            extract_plot_data(mock_v01_result, ["version", "key1", "key2"])


class TestComputeSubplotGrid:
    """Tests for compute_subplot_grid function."""

    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, (1, 1)),  # Edge case: zero
            (1, (1, 1)),  # Single subplot
            (2, (1, 2)),  # Two subplots - horizontal
            (3, (1, 3)),  # Three subplots - horizontal
            (4, (2, 2)),  # Four subplots - square
            (5, (2, 3)),  # Five subplots - fits in 2x3
            (6, (2, 3)),  # Six subplots - exactly 2x3
            (7, (3, 3)),  # Seven subplots - needs 3x3
            (8, (3, 3)),  # Eight subplots - 3x3
            (9, (3, 3)),  # Nine subplots - exactly 3x3
            (10, (3, 4)),  # Ten subplots - 3x4 (wider layout preferred)
        ],
    )
    def test_grid_sizes(self, n, expected):
        """compute_subplot_grid returns correct grid for various n."""
        result = compute_subplot_grid(n)
        nrows, ncols = result

        # Verify it fits all subplots
        assert nrows * ncols >= n

        # Verify the specific expected result
        assert result == expected

    def test_grid_fits_all_subplots(self):
        """compute_subplot_grid always returns grid that fits all subplots."""
        for n in range(1, 20):
            nrows, ncols = compute_subplot_grid(n)
            assert nrows * ncols >= n

    def test_prefers_wider_layouts(self):
        """compute_subplot_grid prefers more columns than rows."""
        for n in range(5, 15):
            nrows, ncols = compute_subplot_grid(n)
            # Should have at least as many columns as rows
            assert ncols >= nrows or nrows * ncols == n


class TestGetSectorColor:
    """Tests for get_sector_color function."""

    def test_returns_hex_color(self):
        """get_sector_color returns a hex color string."""
        color = get_sector_color(1, [1, 2, 3])

        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7  # #RRGGBB format

    def test_consistent_for_same_sector(self):
        """get_sector_color returns same color for same sector."""
        sectors = [1, 5, 10, 15]

        color1 = get_sector_color(5, sectors)
        color2 = get_sector_color(5, sectors)

        assert color1 == color2

    def test_different_for_different_sectors(self):
        """get_sector_color returns different colors for different sectors."""
        sectors = [1, 5, 10, 15]

        color_1 = get_sector_color(1, sectors)
        color_5 = get_sector_color(5, sectors)
        color_10 = get_sector_color(10, sectors)

        # At least some should be different
        colors = {color_1, color_5, color_10}
        assert len(colors) >= 2

    def test_uses_sorted_order(self):
        """get_sector_color uses sorted sector order for indexing."""
        # Order of sectors in list shouldn't matter
        color_a = get_sector_color(5, [1, 5, 10])
        color_b = get_sector_color(5, [10, 5, 1])

        assert color_a == color_b

    def test_cycles_through_colors(self):
        """get_sector_color cycles through tab10 colors."""
        # Create more than 10 sectors to test cycling
        sectors = list(range(1, 15))

        # Sector 1 and sector 11 should have same color (both index 0 mod 10)
        color_1 = get_sector_color(1, sectors)
        color_11 = get_sector_color(11, sectors)

        assert color_1 == color_11

    def test_handles_missing_sector(self):
        """get_sector_color handles sector not in list gracefully."""
        # Should not raise, returns first color
        color = get_sector_color(99, [1, 2, 3])
        assert isinstance(color, str)
        assert color.startswith("#")
