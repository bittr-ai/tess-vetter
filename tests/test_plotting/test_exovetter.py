"""Tests for exovetter plotting functions (V11-V12)."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from bittr_tess_vetter.plotting.exovetter import plot_modshift, plot_sweet
from bittr_tess_vetter.validation.result_schema import ok_result, CheckResult


@pytest.fixture
def mock_v11_result() -> CheckResult:
    """Create a mock V11 (ModShift) CheckResult with plot_data."""
    # Generate mock phase-binned periodogram
    n_bins = 200
    phase_bins = np.linspace(0, 1, n_bins).tolist()

    # Create a periodogram with primary peak at 0 and secondary at 0.5
    periodogram = []
    for p in phase_bins:
        # Primary peak at phase 0
        primary = 0.8 * np.exp(-((p - 0.0) ** 2) / (2 * 0.02**2))
        primary += 0.8 * np.exp(-((p - 1.0) ** 2) / (2 * 0.02**2))
        # Secondary peak at phase 0.5
        secondary = 0.3 * np.exp(-((p - 0.5) ** 2) / (2 * 0.02**2))
        # Noise
        noise = np.random.normal(0, 0.05)
        periodogram.append(float(primary + secondary + noise))

    return ok_result(
        id="V11",
        name="ModShift",
        metrics={
            "primary_signal": 0.8,
            "secondary_signal": 0.3,
            "secondary_primary_ratio": 0.375,
        },
        confidence=0.85,
        raw={
            "plot_data": {
                "version": 1,
                "phase_bins": phase_bins,
                "periodogram": periodogram,
                "primary_phase": 0.0,
                "secondary_phase": 0.5,
                "primary_signal": 0.8,
                "secondary_signal": 0.3,
            }
        },
    )


@pytest.fixture
def mock_v11_result_no_secondary() -> CheckResult:
    """Create a mock V11 CheckResult with no secondary peak."""
    n_bins = 200
    phase_bins = np.linspace(0, 1, n_bins).tolist()

    # Create a periodogram with only primary peak
    periodogram = []
    for p in phase_bins:
        primary = 0.8 * np.exp(-((p - 0.0) ** 2) / (2 * 0.02**2))
        primary += 0.8 * np.exp(-((p - 1.0) ** 2) / (2 * 0.02**2))
        noise = np.random.normal(0, 0.05)
        periodogram.append(float(primary + noise))

    return ok_result(
        id="V11",
        name="ModShift",
        metrics={
            "primary_signal": 0.8,
            "secondary_signal": None,
        },
        confidence=0.85,
        raw={
            "plot_data": {
                "version": 1,
                "phase_bins": phase_bins,
                "periodogram": periodogram,
                "primary_phase": 0.0,
                "primary_signal": 0.8,
                "secondary_phase": None,
                "secondary_signal": None,
            }
        },
    )


@pytest.fixture
def mock_v12_result() -> CheckResult:
    """Create a mock V12 (SWEET) CheckResult with plot_data."""
    # Generate mock phase-folded out-of-transit data
    n_points = 500
    phase = np.random.uniform(0, 1, n_points)
    # Add sinusoidal variability at P
    variability = 0.002 * np.sin(2 * np.pi * phase)
    noise = np.random.normal(0, 0.001, n_points)
    flux = (1.0 + variability + noise).tolist()
    phase = phase.tolist()

    # Generate fit curves
    phase_sorted = np.sort(np.array(phase))
    half_period_fit = (1.0 + 0.0005 * np.sin(4 * np.pi * phase_sorted)).tolist()
    at_period_fit = (1.0 + 0.002 * np.sin(2 * np.pi * phase_sorted)).tolist()
    double_period_fit = (1.0 + 0.0002 * np.sin(np.pi * phase_sorted)).tolist()

    # Unsort to match original phase order
    inv_sort = np.argsort(np.argsort(phase))
    half_period_fit = np.array(half_period_fit)[inv_sort].tolist()
    at_period_fit = np.array(at_period_fit)[inv_sort].tolist()
    double_period_fit = np.array(double_period_fit)[inv_sort].tolist()

    return ok_result(
        id="V12",
        name="SWEET",
        metrics={
            "snr_half_period": 1.2,
            "snr_at_period": 5.8,
            "snr_double_period": 0.5,
        },
        confidence=0.80,
        raw={
            "plot_data": {
                "version": 1,
                "phase": phase,
                "flux": flux,
                "half_period_fit": half_period_fit,
                "at_period_fit": at_period_fit,
                "double_period_fit": double_period_fit,
                "snr_half_period": 1.2,
                "snr_at_period": 5.8,
                "snr_double_period": 0.5,
            }
        },
    )


@pytest.fixture
def mock_v12_result_no_fits() -> CheckResult:
    """Create a mock V12 CheckResult with no sinusoid fits."""
    n_points = 500
    phase = np.random.uniform(0, 1, n_points).tolist()
    flux = (np.ones(n_points) + np.random.normal(0, 0.001, n_points)).tolist()

    return ok_result(
        id="V12",
        name="SWEET",
        metrics={
            "snr_half_period": None,
            "snr_at_period": None,
            "snr_double_period": None,
        },
        confidence=0.50,
        raw={
            "plot_data": {
                "version": 1,
                "phase": phase,
                "flux": flux,
                "half_period_fit": None,
                "at_period_fit": None,
                "double_period_fit": None,
            }
        },
    )


@pytest.fixture
def mock_result_no_plot_data() -> CheckResult:
    """Create a mock CheckResult with no plot_data."""
    return ok_result(
        id="V11",
        name="ModShift",
        metrics={"primary_signal": 0.8},
        raw={"some_other_data": "value"},
    )


@pytest.fixture
def mock_result_no_raw() -> CheckResult:
    """Create a mock CheckResult with raw=None."""
    return ok_result(
        id="V11",
        name="ModShift",
        metrics={"primary_signal": 0.8},
        raw=None,
    )


class TestPlotModshift:
    """Tests for plot_modshift function."""

    def test_creates_figure_when_ax_none(self, mock_v11_result):
        """plot_modshift creates a new figure when ax is None."""
        ax = plot_modshift(mock_v11_result)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_uses_provided_ax(self, mock_v11_result):
        """plot_modshift uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax = plot_modshift(mock_v11_result, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_has_correct_labels(self, mock_v11_result):
        """plot_modshift sets correct axis labels."""
        ax = plot_modshift(mock_v11_result)

        assert ax.get_xlabel() == "Phase"
        assert ax.get_ylabel() == "ModShift Signal"
        assert ax.get_title() == "ModShift Analysis"

    def test_has_legend_by_default(self, mock_v11_result):
        """plot_modshift shows legend by default."""
        ax = plot_modshift(mock_v11_result)

        legend = ax.get_legend()
        assert legend is not None

    def test_legend_disabled_when_requested(self, mock_v11_result):
        """plot_modshift hides legend when show_legend=False."""
        ax = plot_modshift(mock_v11_result, show_legend=False)

        legend = ax.get_legend()
        assert legend is None

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_modshift raises ValueError when plot_data is missing."""
        with pytest.raises(ValueError, match="no plot_data"):
            plot_modshift(mock_result_no_plot_data)

    def test_raises_on_no_raw(self, mock_result_no_raw):
        """plot_modshift raises ValueError when raw is None."""
        with pytest.raises(ValueError, match="no raw data"):
            plot_modshift(mock_result_no_raw)

    def test_plots_periodogram_line(self, mock_v11_result):
        """plot_modshift plots the periodogram line."""
        ax = plot_modshift(mock_v11_result)

        # Should have lines plotted
        lines = ax.get_lines()
        assert len(lines) >= 1

    def test_shows_primary_peak(self, mock_v11_result):
        """plot_modshift shows primary peak marker."""
        ax = plot_modshift(mock_v11_result, show_peaks=True)

        # Should have vertical lines for peaks
        lines = ax.get_lines()
        # Periodogram line + zero line + primary line + secondary line
        assert len(lines) >= 3

    def test_shows_secondary_peak(self, mock_v11_result):
        """plot_modshift shows secondary peak when present."""
        ax = plot_modshift(mock_v11_result, show_peaks=True)

        # Check legend for secondary label
        legend = ax.get_legend()
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert any("Secondary" in t for t in legend_texts)

    def test_handles_no_secondary_peak(self, mock_v11_result_no_secondary):
        """plot_modshift handles case with no secondary peak."""
        ax = plot_modshift(mock_v11_result_no_secondary, show_peaks=True)

        # Should still work without secondary
        assert ax is not None

    def test_peaks_disabled_when_requested(self, mock_v11_result):
        """plot_modshift hides peaks when show_peaks=False."""
        ax = plot_modshift(mock_v11_result, show_peaks=False)

        # Should have fewer lines (no peak markers)
        lines = ax.get_lines()
        # Should only have periodogram + zero reference line
        assert len(lines) == 2

    def test_annotate_values_disabled(self, mock_v11_result):
        """plot_modshift can disable value annotations."""
        ax = plot_modshift(mock_v11_result, annotate_values=False)

        # Should have no scatter points for annotations
        collections = ax.collections
        # Without annotations, no scatter markers
        assert len(collections) == 0

    def test_custom_colors(self, mock_v11_result):
        """plot_modshift uses custom colors when provided."""
        ax = plot_modshift(
            mock_v11_result,
            data_color="#00ff00",
            primary_color="#ff0000",
            secondary_color="#0000ff",
        )

        assert ax is not None

    def test_style_parameter_accepted(self, mock_v11_result):
        """plot_modshift accepts style parameter without error."""
        for style in ["default", "paper", "presentation"]:
            ax = plot_modshift(mock_v11_result, style=style)
            assert ax is not None
            plt.close(ax.figure)

    def test_invalid_style_raises(self, mock_v11_result):
        """plot_modshift raises ValueError for invalid style."""
        with pytest.raises(ValueError, match="Unknown style"):
            plot_modshift(mock_v11_result, style="invalid_style")

    def test_returns_axes(self, mock_v11_result):
        """plot_modshift returns matplotlib Axes object."""
        result = plot_modshift(mock_v11_result)

        assert isinstance(result, matplotlib.axes.Axes)

    def test_x_axis_limits(self, mock_v11_result):
        """plot_modshift sets x-axis limits to 0-1."""
        ax = plot_modshift(mock_v11_result)

        xlim = ax.get_xlim()
        assert xlim[0] == 0
        assert xlim[1] == 1


class TestPlotSweet:
    """Tests for plot_sweet function."""

    def test_creates_figure_when_ax_none(self, mock_v12_result):
        """plot_sweet creates a new figure when ax is None."""
        ax = plot_sweet(mock_v12_result)

        assert ax is not None
        assert ax.figure is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_uses_provided_ax(self, mock_v12_result):
        """plot_sweet uses provided axes instead of creating new ones."""
        fig, provided_ax = plt.subplots()

        result_ax = plot_sweet(mock_v12_result, ax=provided_ax)

        assert result_ax is provided_ax
        assert result_ax.figure is fig

    def test_has_correct_labels(self, mock_v12_result):
        """plot_sweet sets correct axis labels."""
        ax = plot_sweet(mock_v12_result)

        assert ax.get_xlabel() == "Phase"
        assert ax.get_ylabel() == "Normalized Flux"
        assert ax.get_title() == "SWEET Analysis"

    def test_has_legend_by_default(self, mock_v12_result):
        """plot_sweet shows legend by default."""
        ax = plot_sweet(mock_v12_result)

        legend = ax.get_legend()
        assert legend is not None

    def test_legend_disabled_when_requested(self, mock_v12_result):
        """plot_sweet hides legend when show_legend=False."""
        ax = plot_sweet(mock_v12_result, show_legend=False)

        legend = ax.get_legend()
        assert legend is None

    def test_raises_on_missing_plot_data(self, mock_result_no_plot_data):
        """plot_sweet raises ValueError when plot_data is missing."""
        v12_no_plot = ok_result(
            id="V12",
            name="SWEET",
            metrics={"snr_at_period": 5.0},
            raw={"some_other_data": "value"},
        )
        with pytest.raises(ValueError, match="no plot_data"):
            plot_sweet(v12_no_plot)

    def test_raises_on_no_raw(self, mock_result_no_raw):
        """plot_sweet raises ValueError when raw is None."""
        v12_no_raw = ok_result(
            id="V12",
            name="SWEET",
            metrics={"snr_at_period": 5.0},
            raw=None,
        )
        with pytest.raises(ValueError, match="no raw data"):
            plot_sweet(v12_no_raw)

    def test_plots_data_points(self, mock_v12_result):
        """plot_sweet plots the data scatter points."""
        ax = plot_sweet(mock_v12_result)

        # Should have scatter collection
        collections = ax.collections
        assert len(collections) >= 1

    def test_shows_fit_lines(self, mock_v12_result):
        """plot_sweet shows sinusoidal fit lines by default."""
        ax = plot_sweet(mock_v12_result, show_fits=True)

        # Should have lines for fits + reference line
        lines = ax.get_lines()
        # Reference line + 3 fit lines (P/2, P, 2P)
        assert len(lines) >= 4

    def test_shows_snr_in_legend(self, mock_v12_result):
        """plot_sweet shows SNR values in legend when available."""
        ax = plot_sweet(mock_v12_result, show_fits=True)

        legend = ax.get_legend()
        legend_texts = [t.get_text() for t in legend.get_texts()]
        # Should have SNR annotations
        assert any("SNR" in t for t in legend_texts)

    def test_handles_no_fits(self, mock_v12_result_no_fits):
        """plot_sweet handles case with no sinusoid fits."""
        ax = plot_sweet(mock_v12_result_no_fits, show_fits=True)

        # Should still work without fits
        assert ax is not None
        # Should only have reference line (no fit lines)
        lines = ax.get_lines()
        assert len(lines) == 1  # Just reference line

    def test_fits_disabled_when_requested(self, mock_v12_result):
        """plot_sweet hides fits when show_fits=False."""
        ax = plot_sweet(mock_v12_result, show_fits=False)

        # Should only have reference line
        lines = ax.get_lines()
        assert len(lines) == 1

    def test_custom_colors(self, mock_v12_result):
        """plot_sweet uses custom colors when provided."""
        ax = plot_sweet(
            mock_v12_result,
            data_color="#cccccc",
            half_period_color="#ff0000",
            at_period_color="#00ff00",
            double_period_color="#0000ff",
        )

        assert ax is not None

    def test_style_parameter_accepted(self, mock_v12_result):
        """plot_sweet accepts style parameter without error."""
        for style in ["default", "paper", "presentation"]:
            ax = plot_sweet(mock_v12_result, style=style)
            assert ax is not None
            plt.close(ax.figure)

    def test_invalid_style_raises(self, mock_v12_result):
        """plot_sweet raises ValueError for invalid style."""
        with pytest.raises(ValueError, match="Unknown style"):
            plot_sweet(mock_v12_result, style="invalid_style")

    def test_returns_axes(self, mock_v12_result):
        """plot_sweet returns matplotlib Axes object."""
        result = plot_sweet(mock_v12_result)

        assert isinstance(result, matplotlib.axes.Axes)

    def test_x_axis_limits(self, mock_v12_result):
        """plot_sweet sets x-axis limits to 0-1."""
        ax = plot_sweet(mock_v12_result)

        xlim = ax.get_xlim()
        assert xlim[0] == 0
        assert xlim[1] == 1

    def test_reference_line_at_one(self, mock_v12_result):
        """plot_sweet draws reference line at flux=1.0."""
        ax = plot_sweet(mock_v12_result, show_fits=False)

        # Check for horizontal reference line at y=1
        lines = ax.get_lines()
        assert len(lines) >= 1
        ref_line = lines[0]
        ydata = ref_line.get_ydata()
        assert ydata[0] == 1.0 and ydata[1] == 1.0
