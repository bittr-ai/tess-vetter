"""Tests for pixel-level plotting functions (V08-V10)."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tess_vetter.plotting.pixel import (
    plot_aperture_curve,
    plot_centroid_shift,
    plot_difference_image,
)
from tess_vetter.validation.result_schema import ok_result

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_v08_result():
    """Create a mock V08 (centroid shift) CheckResult with plot_data."""
    # Create a simple 11x11 reference image with a bright center
    ref_image = np.zeros((11, 11), dtype=np.float32)
    ref_image[5, 5] = 1000.0
    ref_image[4:7, 4:7] = 500.0

    return ok_result(
        id="V08",
        name="centroid_shift",
        metrics={
            "centroid_shift_pixels": 0.5,
            "centroid_shift_arcsec": 10.5,
            "significance_sigma": 2.5,
        },
        confidence=0.8,
        raw={
            "plot_data": {
                "version": 1,
                "reference_image": ref_image.tolist(),
                "in_centroid_col": 5.2,
                "in_centroid_row": 5.3,
                "out_centroid_col": 4.8,
                "out_centroid_row": 4.9,
                "target_col": 5,
                "target_row": 5,
            }
        },
    )


@pytest.fixture
def mock_v09_result():
    """Create a mock V09 (pixel depth map) CheckResult with plot_data."""
    # Create a simple 11x11 depth map with positive depths at center
    depth_map = np.zeros((11, 11), dtype=np.float32)
    depth_map[5, 5] = 1000.0  # Max depth at center
    depth_map[4:7, 4:7] = 500.0
    depth_map[0, 0] = -200.0  # Some negative depth for testing diverging colormap

    return ok_result(
        id="V09",
        name="pixel_level_lc",
        metrics={
            "max_depth_ppm": 1000.0,
            "target_depth_ppm": 1000.0,
            "concentration_ratio": 1.0,
        },
        confidence=0.7,
        raw={
            "plot_data": {
                "version": 1,
                "difference_image": depth_map.tolist(),
                "depth_map_ppm": depth_map.tolist(),
                "target_pixel": [5, 5],  # [row, col]
                "max_depth_pixel": [5, 5],  # [row, col]
            }
        },
    )


@pytest.fixture
def mock_v10_result():
    """Create a mock V10 (aperture dependence) CheckResult with plot_data."""
    return ok_result(
        id="V10",
        name="aperture_dependence",
        metrics={
            "stability_metric": 0.9,
            "depth_variance_ppm2": 100.0,
        },
        confidence=0.85,
        raw={
            "plot_data": {
                "version": 1,
                "aperture_radii_px": [1.0, 1.5, 2.0, 2.5, 3.0],
                "depths_ppm": [450.0, 480.0, 500.0, 510.0, 505.0],
                "depth_errs_ppm": [20.0, 18.0, 15.0, 14.0, 16.0],
            }
        },
    )


@pytest.fixture
def mock_result_no_plot_data():
    """Create a mock CheckResult with no plot_data."""
    return ok_result(
        id="V08",
        name="centroid_shift",
        metrics={"centroid_shift_pixels": 0.5},
        raw={"some_other_data": "value"},
    )


@pytest.fixture
def mock_result_no_raw():
    """Create a mock CheckResult with raw=None."""
    return ok_result(
        id="V08",
        name="centroid_shift",
        metrics={"centroid_shift_pixels": 0.5},
        raw=None,
    )


# =============================================================================
# Tests for plot_centroid_shift (V08)
# =============================================================================


class TestPlotCentroidShift:
    """Tests for plot_centroid_shift function."""

    def test_creates_figure_when_ax_none(self, mock_v08_result):
        """plot_centroid_shift creates a new figure when ax is None."""
        ax, cbar = plot_centroid_shift(mock_v08_result)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_returns_tuple_with_colorbar(self, mock_v08_result):
        """plot_centroid_shift returns (ax, cbar) tuple."""
        result = plot_centroid_shift(mock_v08_result)

        assert isinstance(result, tuple)
        assert len(result) == 2
        ax, cbar = result
        assert isinstance(ax, matplotlib.axes.Axes)
        assert cbar is not None

    def test_colorbar_disabled(self, mock_v08_result):
        """plot_centroid_shift returns None colorbar when disabled."""
        ax, cbar = plot_centroid_shift(mock_v08_result, show_colorbar=False)

        assert ax is not None
        assert cbar is None

    def test_uses_provided_ax(self, mock_v08_result):
        """plot_centroid_shift uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax, _ = plot_centroid_shift(mock_v08_result, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_has_correct_labels(self, mock_v08_result):
        """plot_centroid_shift sets correct axis labels."""
        ax, _ = plot_centroid_shift(mock_v08_result)

        assert ax.get_xlabel() == "Column (pixels)"
        assert ax.get_ylabel() == "Row (pixels)"

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_centroid_shift raises ValueError when plot_data is missing."""
        with pytest.raises(ValueError, match="no plot_data"):
            plot_centroid_shift(mock_result_no_plot_data)

    def test_raises_on_no_raw(self, mock_result_no_raw):
        """plot_centroid_shift raises ValueError when raw is None."""
        with pytest.raises(ValueError, match="no raw data"):
            plot_centroid_shift(mock_result_no_raw)

    def test_uses_origin_lower(self, mock_v08_result):
        """plot_centroid_shift uses origin='lower' for imshow."""
        ax, _ = plot_centroid_shift(mock_v08_result)

        # Get the imshow image from axes
        images = ax.get_images()
        assert len(images) == 1
        assert images[0].origin == "lower"

    def test_custom_colors(self, mock_v08_result):
        """plot_centroid_shift accepts custom centroid colors."""
        ax, _ = plot_centroid_shift(
            mock_v08_result,
            in_color="#ff0000",
            out_color="#00ff00",
        )

        # Just check it doesn't raise
        assert ax is not None

    def test_style_parameter_accepted(self, mock_v08_result):
        """plot_centroid_shift accepts style parameter without error."""
        for style in ["default", "paper", "presentation"]:
            ax, _ = plot_centroid_shift(mock_v08_result, style=style)
            assert ax is not None
            plt.close(ax.figure)

    def test_invalid_style_raises(self, mock_v08_result):
        """plot_centroid_shift raises ValueError for invalid style."""
        with pytest.raises(ValueError, match="Unknown style"):
            plot_centroid_shift(mock_v08_result, style="invalid_style")

    def test_show_vector_disabled(self, mock_v08_result):
        """plot_centroid_shift works with show_vector=False."""
        ax, _ = plot_centroid_shift(mock_v08_result, show_vector=False)
        assert ax is not None

    def test_show_target_disabled(self, mock_v08_result):
        """plot_centroid_shift works with show_target=False."""
        ax, _ = plot_centroid_shift(mock_v08_result, show_target=False)
        assert ax is not None


# =============================================================================
# Tests for plot_difference_image (V09)
# =============================================================================


class TestPlotDifferenceImage:
    """Tests for plot_difference_image function."""

    def test_creates_figure_when_ax_none(self, mock_v09_result):
        """plot_difference_image creates a new figure when ax is None."""
        ax, cbar = plot_difference_image(mock_v09_result)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_returns_tuple_with_colorbar(self, mock_v09_result):
        """plot_difference_image returns (ax, cbar) tuple."""
        result = plot_difference_image(mock_v09_result)

        assert isinstance(result, tuple)
        assert len(result) == 2
        ax, cbar = result
        assert isinstance(ax, matplotlib.axes.Axes)
        assert cbar is not None

    def test_colorbar_disabled(self, mock_v09_result):
        """plot_difference_image returns None colorbar when disabled."""
        ax, cbar = plot_difference_image(mock_v09_result, show_colorbar=False)

        assert ax is not None
        assert cbar is None

    def test_uses_provided_ax(self, mock_v09_result):
        """plot_difference_image uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax, _ = plot_difference_image(mock_v09_result, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_has_correct_labels(self, mock_v09_result):
        """plot_difference_image sets correct axis labels."""
        ax, _ = plot_difference_image(mock_v09_result)

        assert ax.get_xlabel() == "Column (pixels)"
        assert ax.get_ylabel() == "Row (pixels)"

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_difference_image raises ValueError when plot_data is missing."""
        with pytest.raises(ValueError, match="no plot_data"):
            plot_difference_image(mock_result_no_plot_data)

    def test_raises_on_no_raw(self, mock_result_no_raw):
        """plot_difference_image raises ValueError when raw is None."""
        with pytest.raises(ValueError, match="no raw data"):
            plot_difference_image(mock_result_no_raw)

    def test_uses_origin_lower(self, mock_v09_result):
        """plot_difference_image uses origin='lower' for imshow."""
        ax, _ = plot_difference_image(mock_v09_result)

        # Get the imshow image from axes
        images = ax.get_images()
        assert len(images) == 1
        assert images[0].origin == "lower"

    def test_colormap_centered_at_zero(self, mock_v09_result):
        """plot_difference_image centers colormap at 0."""
        ax, _ = plot_difference_image(mock_v09_result)

        images = ax.get_images()
        assert len(images) == 1
        im = images[0]

        # Check that vmin and vmax are symmetric around 0
        vmin, vmax = im.get_clim()
        assert vmin == -vmax or abs(vmin + vmax) < 1e-10

    def test_custom_colormap(self, mock_v09_result):
        """plot_difference_image accepts custom colormap."""
        ax, _ = plot_difference_image(mock_v09_result, cmap="coolwarm")

        images = ax.get_images()
        assert len(images) == 1
        assert images[0].get_cmap().name == "coolwarm"

    def test_style_parameter_accepted(self, mock_v09_result):
        """plot_difference_image accepts style parameter without error."""
        for style in ["default", "paper", "presentation"]:
            ax, _ = plot_difference_image(mock_v09_result, style=style)
            assert ax is not None
            plt.close(ax.figure)

    def test_invalid_style_raises(self, mock_v09_result):
        """plot_difference_image raises ValueError for invalid style."""
        with pytest.raises(ValueError, match="Unknown style"):
            plot_difference_image(mock_v09_result, style="invalid_style")

    def test_show_target_disabled(self, mock_v09_result):
        """plot_difference_image works with show_target=False."""
        ax, _ = plot_difference_image(mock_v09_result, show_target=False)
        assert ax is not None

    def test_show_max_depth_disabled(self, mock_v09_result):
        """plot_difference_image works with show_max_depth=False."""
        ax, _ = plot_difference_image(mock_v09_result, show_max_depth=False)
        assert ax is not None


# =============================================================================
# Tests for plot_aperture_curve (V10)
# =============================================================================


class TestPlotApertureCurve:
    """Tests for plot_aperture_curve function."""

    def test_creates_figure_when_ax_none(self, mock_v10_result):
        """plot_aperture_curve creates a new figure when ax is None."""
        ax = plot_aperture_curve(mock_v10_result)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_returns_axes_not_tuple(self, mock_v10_result):
        """plot_aperture_curve returns Axes, not tuple."""
        result = plot_aperture_curve(mock_v10_result)

        # Should not be a tuple, just Axes
        assert isinstance(result, matplotlib.axes.Axes)
        assert not isinstance(result, tuple)

    def test_uses_provided_ax(self, mock_v10_result):
        """plot_aperture_curve uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax = plot_aperture_curve(mock_v10_result, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_has_correct_labels(self, mock_v10_result):
        """plot_aperture_curve sets correct axis labels."""
        ax = plot_aperture_curve(mock_v10_result)

        assert ax.get_xlabel() == "Aperture Radius (pixels)"
        assert ax.get_ylabel() == "Transit Depth (ppm)"

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_aperture_curve raises ValueError when plot_data is missing."""
        with pytest.raises(ValueError, match="no plot_data"):
            plot_aperture_curve(mock_result_no_plot_data)

    def test_raises_on_no_raw(self, mock_result_no_raw):
        """plot_aperture_curve raises ValueError when raw is None."""
        with pytest.raises(ValueError, match="no raw data"):
            plot_aperture_curve(mock_result_no_raw)

    def test_errorbars_shown_by_default(self, mock_v10_result):
        """plot_aperture_curve shows error bars by default."""
        ax = plot_aperture_curve(mock_v10_result)

        # Check that there are error bar containers
        containers = ax.containers
        assert len(containers) >= 1

    def test_errorbars_disabled(self, mock_v10_result):
        """plot_aperture_curve hides error bars when show_errorbars=False."""
        ax = plot_aperture_curve(mock_v10_result, show_errorbars=False)

        # Should have line but no error bar containers
        # (just check it runs without error)
        assert ax is not None

    def test_custom_color(self, mock_v10_result):
        """plot_aperture_curve accepts custom color."""
        custom_color = "#ff0000"
        ax = plot_aperture_curve(mock_v10_result, color=custom_color)

        # Get the line from the plot
        lines = ax.get_lines()
        # Find a line that's not the reference line (y=0)
        data_lines = [line for line in lines if line.get_linestyle() != "--"]
        if data_lines:
            # The color might be in RGB tuple format
            line_color = data_lines[0].get_color()
            # Just verify the plot worked
            assert line_color is not None

    def test_style_parameter_accepted(self, mock_v10_result):
        """plot_aperture_curve accepts style parameter without error."""
        for style in ["default", "paper", "presentation"]:
            ax = plot_aperture_curve(mock_v10_result, style=style)
            assert ax is not None
            plt.close(ax.figure)

    def test_invalid_style_raises(self, mock_v10_result):
        """plot_aperture_curve raises ValueError for invalid style."""
        with pytest.raises(ValueError, match="Unknown style"):
            plot_aperture_curve(mock_v10_result, style="invalid_style")

    def test_has_reference_line_at_zero(self, mock_v10_result):
        """plot_aperture_curve shows horizontal reference line at y=0."""
        ax = plot_aperture_curve(mock_v10_result)

        # Find horizontal dashed lines
        horizontal_lines = []
        for line in ax.get_lines():
            ydata = line.get_ydata()
            if len(ydata) == 2 and ydata[0] == ydata[1] == 0:
                horizontal_lines.append(line)

        assert len(horizontal_lines) >= 1
