"""Tests for catalog plotting functions (V06-V07)."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from bittr_tess_vetter.plotting.catalog import plot_nearby_ebs, plot_exofop_card
from bittr_tess_vetter.validation.result_schema import ok_result, CheckResult


@pytest.fixture
def mock_v06_result() -> CheckResult:
    """Create a mock V06 (nearby EB search) CheckResult with plot_data."""
    return ok_result(
        id="V06",
        name="Nearby EB Search",
        metrics={
            "n_ebs_found": 3,
            "min_period_ratio_delta_any": 0.05,
        },
        confidence=0.80,
        raw={
            "plot_data": {
                "version": 1,
                "target_ra": 120.5,
                "target_dec": -45.2,
                "search_radius_arcsec": 42.0,
                "matches": [
                    {
                        "ra": 120.502,
                        "dec": -45.198,
                        "sep_arcsec": 10.5,
                        "period_days": 2.35,
                    },
                    {
                        "ra": 120.508,
                        "dec": -45.205,
                        "sep_arcsec": 25.2,
                        "period_days": 4.71,
                    },
                    {
                        "ra": 120.495,
                        "dec": -45.210,
                        "sep_arcsec": 38.8,
                        "period_days": 1.18,
                    },
                ],
            }
        },
    )


@pytest.fixture
def mock_v06_result_no_matches() -> CheckResult:
    """Create a mock V06 CheckResult with no nearby EBs found."""
    return ok_result(
        id="V06",
        name="Nearby EB Search",
        metrics={
            "n_ebs_found": 0,
        },
        confidence=0.60,
        raw={
            "plot_data": {
                "version": 1,
                "target_ra": 120.5,
                "target_dec": -45.2,
                "search_radius_arcsec": 42.0,
                "matches": [],
            }
        },
    )


@pytest.fixture
def mock_v07_result_found() -> CheckResult:
    """Create a mock V07 (ExoFOP lookup) CheckResult with a TOI found."""
    return ok_result(
        id="V07",
        name="ExoFOP TOI Lookup",
        metrics={
            "found": True,
        },
        confidence=0.80,
        raw={
            "plot_data": {
                "version": 1,
                "tic_id": 12345678,
                "found": True,
                "toi": 1234.01,
                "tfopwg_disposition": "CP",
                "planet_disposition": "Confirmed Planet",
                "comments": "High-quality detection",
            }
        },
    )


@pytest.fixture
def mock_v07_result_not_found() -> CheckResult:
    """Create a mock V07 CheckResult with no TOI found."""
    return ok_result(
        id="V07",
        name="ExoFOP TOI Lookup",
        metrics={
            "found": False,
        },
        confidence=0.70,
        raw={
            "plot_data": {
                "version": 1,
                "tic_id": 87654321,
                "found": False,
            }
        },
    )


@pytest.fixture
def mock_v07_result_fp() -> CheckResult:
    """Create a mock V07 CheckResult with a false positive disposition."""
    return ok_result(
        id="V07",
        name="ExoFOP TOI Lookup",
        metrics={
            "found": True,
        },
        confidence=0.80,
        raw={
            "plot_data": {
                "version": 1,
                "tic_id": 11111111,
                "found": True,
                "toi": 5678.01,
                "tfopwg_disposition": "FP",
                "planet_disposition": "False Positive",
                "comments": "Identified as eclipsing binary",
            }
        },
    )


@pytest.fixture
def mock_result_no_plot_data() -> CheckResult:
    """Create a mock CheckResult with no plot_data."""
    return ok_result(
        id="V06",
        name="Nearby EB Search",
        metrics={"n_ebs_found": 0},
        raw={"some_other_data": "value"},
    )


@pytest.fixture
def mock_result_no_raw() -> CheckResult:
    """Create a mock CheckResult with raw=None."""
    return ok_result(
        id="V06",
        name="Nearby EB Search",
        metrics={"n_ebs_found": 0},
        raw=None,
    )


class TestPlotNearbyEbs:
    """Tests for plot_nearby_ebs function."""

    def test_creates_figure_when_ax_none(self, mock_v06_result):
        """plot_nearby_ebs creates a new figure when ax is None."""
        ax = plot_nearby_ebs(mock_v06_result)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_uses_provided_ax(self, mock_v06_result):
        """plot_nearby_ebs uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax = plot_nearby_ebs(mock_v06_result, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_has_correct_labels(self, mock_v06_result):
        """plot_nearby_ebs sets correct axis labels."""
        ax = plot_nearby_ebs(mock_v06_result)

        assert ax.get_xlabel() == 'RA offset (arcsec)'
        assert ax.get_ylabel() == 'Dec offset (arcsec)'
        assert ax.get_title() == "Nearby Eclipsing Binaries"

    def test_has_legend_by_default(self, mock_v06_result):
        """plot_nearby_ebs shows legend by default."""
        ax = plot_nearby_ebs(mock_v06_result)

        legend = ax.get_legend()
        assert legend is not None

        # Check legend has target and EB labels
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert any("Target" in t for t in legend_texts)

    def test_legend_disabled_when_requested(self, mock_v06_result):
        """plot_nearby_ebs hides legend when show_legend=False."""
        ax = plot_nearby_ebs(mock_v06_result, show_legend=False)

        legend = ax.get_legend()
        assert legend is None

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_nearby_ebs raises ValueError when plot_data is missing."""
        with pytest.raises(ValueError, match="no plot_data"):
            plot_nearby_ebs(mock_result_no_plot_data)

    def test_raises_on_no_raw(self, mock_result_no_raw):
        """plot_nearby_ebs raises ValueError when raw is None."""
        with pytest.raises(ValueError, match="no raw data"):
            plot_nearby_ebs(mock_result_no_raw)

    def test_plots_target_marker(self, mock_v06_result):
        """plot_nearby_ebs plots target marker at center."""
        ax = plot_nearby_ebs(mock_v06_result)

        # Find the target scatter (star marker at 0,0)
        collections = ax.collections
        assert len(collections) >= 1  # At least target

        # Check that first collection (target) is at origin
        target_offsets = collections[0].get_offsets()
        assert len(target_offsets) >= 1

    def test_plots_eb_markers(self, mock_v06_result):
        """plot_nearby_ebs plots markers for nearby EBs."""
        ax = plot_nearby_ebs(mock_v06_result)

        # Should have multiple collections (target + EBs)
        collections = ax.collections
        assert len(collections) >= 2  # Target + EBs

    def test_handles_no_matches(self, mock_v06_result_no_matches):
        """plot_nearby_ebs handles case with no nearby EBs gracefully."""
        ax = plot_nearby_ebs(mock_v06_result_no_matches)

        assert ax is not None
        # Should still have target marker
        collections = ax.collections
        assert len(collections) >= 1

    def test_draws_search_radius_circle(self, mock_v06_result):
        """plot_nearby_ebs draws search radius circle."""
        ax = plot_nearby_ebs(mock_v06_result)

        # Check for circle line
        lines = ax.get_lines()
        assert len(lines) >= 1  # At least the search radius circle

    def test_custom_colors(self, mock_v06_result):
        """plot_nearby_ebs uses custom colors when provided."""
        ax = plot_nearby_ebs(
            mock_v06_result,
            target_color="#00ff00",
            match_color="#ff00ff",
        )

        assert ax is not None

    def test_annotate_separations_disabled(self, mock_v06_result):
        """plot_nearby_ebs can disable separation annotations."""
        ax = plot_nearby_ebs(mock_v06_result, annotate_separations=False)

        # Should have fewer text elements
        assert ax is not None

    def test_marker_scale(self, mock_v06_result):
        """plot_nearby_ebs accepts marker_scale parameter."""
        ax = plot_nearby_ebs(mock_v06_result, marker_scale=2.0)

        assert ax is not None

    def test_style_parameter_accepted(self, mock_v06_result):
        """plot_nearby_ebs accepts style parameter without error."""
        for style in ["default", "paper", "presentation"]:
            ax = plot_nearby_ebs(mock_v06_result, style=style)
            assert ax is not None
            plt.close(ax.figure)

    def test_invalid_style_raises(self, mock_v06_result):
        """plot_nearby_ebs raises ValueError for invalid style."""
        with pytest.raises(ValueError, match="Unknown style"):
            plot_nearby_ebs(mock_v06_result, style="invalid_style")

    def test_returns_axes(self, mock_v06_result):
        """plot_nearby_ebs returns matplotlib Axes object."""
        result = plot_nearby_ebs(mock_v06_result)

        assert isinstance(result, matplotlib.axes.Axes)


class TestPlotExofopCard:
    """Tests for plot_exofop_card function."""

    def test_creates_figure_when_ax_none(self, mock_v07_result_found):
        """plot_exofop_card creates a new figure when ax is None."""
        ax = plot_exofop_card(mock_v07_result_found)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_uses_provided_ax(self, mock_v07_result_found):
        """plot_exofop_card uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax = plot_exofop_card(mock_v07_result_found, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_shows_title_by_default(self, mock_v07_result_found):
        """plot_exofop_card shows title by default."""
        ax = plot_exofop_card(mock_v07_result_found, show_title=True)

        title = ax.get_title()
        assert "ExoFOP" in title

    def test_title_disabled_when_requested(self, mock_v07_result_found):
        """plot_exofop_card hides title when show_title=False."""
        ax = plot_exofop_card(mock_v07_result_found, show_title=False)

        title = ax.get_title()
        assert title == ""

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_exofop_card raises ValueError when plot_data is missing."""
        # Create a V07-specific no-plot-data result
        v07_no_plot = ok_result(
            id="V07",
            name="ExoFOP TOI Lookup",
            metrics={"found": False},
            raw={"some_other_data": "value"},
        )
        with pytest.raises(ValueError, match="no plot_data"):
            plot_exofop_card(v07_no_plot)

    def test_raises_on_no_raw(self, mock_result_no_raw):
        """plot_exofop_card raises ValueError when raw is None."""
        v07_no_raw = ok_result(
            id="V07",
            name="ExoFOP TOI Lookup",
            metrics={"found": False},
            raw=None,
        )
        with pytest.raises(ValueError, match="no raw data"):
            plot_exofop_card(v07_no_raw)

    def test_displays_toi_when_found(self, mock_v07_result_found):
        """plot_exofop_card displays TOI info when found."""
        ax = plot_exofop_card(mock_v07_result_found)

        # Check that text was added (axis is turned off for card display)
        assert ax is not None
        # Card should have text children
        texts = ax.texts
        assert len(texts) >= 1

    def test_displays_not_found_message(self, mock_v07_result_not_found):
        """plot_exofop_card displays not found message appropriately."""
        ax = plot_exofop_card(mock_v07_result_not_found)

        assert ax is not None
        # Should still have text
        texts = ax.texts
        assert len(texts) >= 1

    def test_displays_false_positive(self, mock_v07_result_fp):
        """plot_exofop_card handles false positive disposition."""
        ax = plot_exofop_card(mock_v07_result_fp)

        assert ax is not None

    def test_style_parameter_accepted(self, mock_v07_result_found):
        """plot_exofop_card accepts style parameter without error."""
        for style in ["default", "paper", "presentation"]:
            ax = plot_exofop_card(mock_v07_result_found, style=style)
            assert ax is not None
            plt.close(ax.figure)

    def test_invalid_style_raises(self, mock_v07_result_found):
        """plot_exofop_card raises ValueError for invalid style."""
        with pytest.raises(ValueError, match="Unknown style"):
            plot_exofop_card(mock_v07_result_found, style="invalid_style")

    def test_returns_axes(self, mock_v07_result_found):
        """plot_exofop_card returns matplotlib Axes object."""
        result = plot_exofop_card(mock_v07_result_found)

        assert isinstance(result, matplotlib.axes.Axes)

    def test_has_background_patch(self, mock_v07_result_found):
        """plot_exofop_card adds a background rectangle."""
        ax = plot_exofop_card(mock_v07_result_found)

        # Should have at least one patch (the background rectangle)
        patches = ax.patches
        assert len(patches) >= 1
