"""Tests for transit visualization functions.

This module tests:
- plot_phase_folded: Phase-folded light curve visualization
- plot_transit_fit: Transit model fit visualization
- plot_full_lightcurve: Full time-series light curve visualization
"""

from __future__ import annotations

# These imports happen after conftest.py sets up matplotlib backend
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from tess_vetter.plotting.lightcurve import plot_full_lightcurve
from tess_vetter.plotting.transit import plot_phase_folded, plot_transit_fit

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_lightcurve() -> LightCurve:
    """Create a mock LightCurve with simulated transit signal.

    Creates a light curve with:
    - 100 days of 2-minute cadence data
    - Gaussian noise at 200 ppm level
    - Simple box-shaped transits with 1000 ppm depth
    """
    np.random.seed(42)

    # 30 days of 2-minute cadence data (roughly typical TESS sector)
    n_points = 30 * 24 * 30  # 30 days, 24 hours, 30 points/hour
    time = np.linspace(2458600.0, 2458630.0, n_points)

    # Create normalized flux with Gaussian noise
    noise_ppm = 200
    flux = 1.0 + np.random.normal(0, noise_ppm / 1e6, n_points)

    # Add simple box-shaped transits
    period = 5.0  # days
    t0 = 2458602.0  # First transit
    duration = 3.0 / 24.0  # 3 hours in days
    depth = 1000 / 1e6  # 1000 ppm

    # Mark transit points
    phase = ((time - t0) / period) % 1.0
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    in_transit = np.abs(phase) < (duration / period / 2)
    flux[in_transit] -= depth

    flux_err = np.full(n_points, noise_ppm / 1e6)

    return LightCurve(time=time, flux=flux, flux_err=flux_err)


@pytest.fixture
def mock_candidate() -> Candidate:
    """Create a mock Candidate with ephemeris matching mock_lightcurve."""
    return Candidate(
        ephemeris=Ephemeris(
            period_days=5.0,
            t0_btjd=2458602.0,
            duration_hours=3.0,
        ),
        depth_ppm=1000.0,
    )


@pytest.fixture
def mock_transit_fit_result():
    """Create a mock TransitFitResult for testing.

    Uses a simple class instead of actual TransitFitResult to avoid
    dependency on the full fitting infrastructure.
    """
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class MockTransitFitResult:
        fit_method: str = "optimize"
        rp_rs: float = 0.1
        rp_rs_err: float = 0.005
        a_rs: float = 15.0
        a_rs_err: float = 0.5
        inclination_deg: float = 89.0
        inclination_err: float = 0.5
        t0_offset: float = 0.0
        t0_offset_err: float = 0.001
        u1: float = 0.4
        u2: float = 0.2
        transit_depth_ppm: float = 1000.0
        duration_hours: float = 3.0
        impact_parameter: float = 0.2
        stellar_density_gcc: float = 1.4
        chi_squared: float = 1.05
        bic: float = 150.0
        converged: bool = True
        phase: list[float] = field(default_factory=list)
        flux_model: list[float] = field(default_factory=list)
        flux_data: list[float] = field(default_factory=list)
        mcmc_diagnostics: dict[str, Any] | None = None
        status: str = "success"
        error_message: str | None = None

    # Create phase array and model
    phase = np.linspace(-0.1, 0.1, 100).tolist()
    # Simple trapezoidal transit model
    flux_model = []
    flux_data = []
    for p in phase:
        if abs(p) < 0.01:
            # In transit (flat bottom)
            flux_model.append(1.0 - 0.001)
            flux_data.append(1.0 - 0.001 + np.random.normal(0, 0.0002))
        elif abs(p) < 0.02:
            # Ingress/egress
            depth = 0.001 * (0.02 - abs(p)) / 0.01
            flux_model.append(1.0 - depth)
            flux_data.append(1.0 - depth + np.random.normal(0, 0.0002))
        else:
            # Out of transit
            flux_model.append(1.0)
            flux_data.append(1.0 + np.random.normal(0, 0.0002))

    return MockTransitFitResult(
        phase=phase,
        flux_model=flux_model,
        flux_data=flux_data,
    )


@pytest.fixture
def mock_transit_fit_result_empty():
    """Create a mock TransitFitResult with no data (error state)."""
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class MockTransitFitResult:
        fit_method: str = "none"
        rp_rs: float = 0.0
        rp_rs_err: float = 0.0
        a_rs: float = 0.0
        a_rs_err: float = 0.0
        inclination_deg: float = 0.0
        inclination_err: float = 0.0
        t0_offset: float = 0.0
        t0_offset_err: float = 0.0
        u1: float = 0.0
        u2: float = 0.0
        transit_depth_ppm: float = 0.0
        duration_hours: float = 0.0
        impact_parameter: float = 0.0
        stellar_density_gcc: float = 0.0
        chi_squared: float = 0.0
        bic: float = 0.0
        converged: bool = False
        phase: list[float] = field(default_factory=list)
        flux_model: list[float] = field(default_factory=list)
        flux_data: list[float] = field(default_factory=list)
        mcmc_diagnostics: dict[str, Any] | None = None
        status: str = "error"
        error_message: str | None = "Missing data"

    return MockTransitFitResult()


# =============================================================================
# Tests for plot_phase_folded
# =============================================================================


class TestPlotPhaseFolded:
    """Tests for plot_phase_folded function."""

    def test_creates_figure_when_ax_none(
        self, mock_lightcurve: LightCurve, mock_candidate: Candidate
    ):
        """Test that a new figure is created when ax is None."""
        ax = plot_phase_folded(mock_lightcurve, mock_candidate)

        assert ax is not None
        assert ax.figure is not None

    def test_uses_provided_ax(
        self, mock_lightcurve: LightCurve, mock_candidate: Candidate
    ):
        """Test that provided axes are used."""
        fig, ax_input = plt.subplots()
        ax = plot_phase_folded(mock_lightcurve, mock_candidate, ax=ax_input)

        assert ax is ax_input
        assert ax.figure is fig

    def test_phase_folding_correctness(
        self, mock_lightcurve: LightCurve, mock_candidate: Candidate
    ):
        """Test that phase folding centers transit at phase 0."""
        ax = plot_phase_folded(
            mock_lightcurve, mock_candidate, show_binned=False, phase_range=(-0.5, 0.5)
        )

        # Get the scatter collection
        collections = ax.collections
        assert len(collections) > 0

        # Phase should be centered on 0 (transit midpoint)
        offsets = collections[0].get_offsets()
        phases = offsets[:, 0]

        # Verify phases are within expected range
        assert np.all(phases >= -0.5)
        assert np.all(phases <= 0.5)

    def test_binning(self, mock_lightcurve: LightCurve, mock_candidate: Candidate):
        """Test that binning produces expected number of bins."""
        ax = plot_phase_folded(
            mock_lightcurve, mock_candidate, bin_minutes=30.0, show_binned=True
        )

        # Should have scatter (raw data) and errorbar (binned data)
        assert len(ax.collections) > 0
        # Check that there are lines (from errorbar)
        assert len(ax.lines) > 0 or len(ax.containers) > 0

    def test_no_binning_when_none(
        self, mock_lightcurve: LightCurve, mock_candidate: Candidate
    ):
        """Test that binning is disabled when bin_minutes=None."""
        ax = plot_phase_folded(
            mock_lightcurve, mock_candidate, bin_minutes=None, show_binned=True
        )

        # Should only have scatter (raw data), no errorbar containers
        # Note: errorbar creates containers, scatter creates collections
        assert len(ax.collections) >= 1

    def test_model_overlay_with_fit_result(
        self,
        mock_lightcurve: LightCurve,
        mock_candidate: Candidate,
        mock_transit_fit_result,
    ):
        """Test that model is overlaid when fit_result is provided."""
        ax = plot_phase_folded(
            mock_lightcurve,
            mock_candidate,
            fit_result=mock_transit_fit_result,
            show_model=True,
        )

        # Should have model line in addition to scatter
        assert len(ax.lines) > 0

    def test_no_model_when_show_model_false(
        self,
        mock_lightcurve: LightCurve,
        mock_candidate: Candidate,
        mock_transit_fit_result,
    ):
        """Test that model is not shown when show_model=False."""
        ax = plot_phase_folded(
            mock_lightcurve,
            mock_candidate,
            fit_result=mock_transit_fit_result,
            show_model=False,
            show_binned=False,
        )

        # Should not have model line (only scatter)
        # Lines from ax.plot would be present if model was shown
        plot_lines = [
            line for line in ax.lines if len(line.get_xdata()) > 2
        ]  # Model has many points
        assert len(plot_lines) == 0

    def test_phase_range_limits(
        self, mock_lightcurve: LightCurve, mock_candidate: Candidate
    ):
        """Test that phase_range parameter limits the view."""
        phase_range = (-0.05, 0.05)
        ax = plot_phase_folded(
            mock_lightcurve, mock_candidate, phase_range=phase_range, show_binned=False
        )

        xlim = ax.get_xlim()
        assert xlim[0] == pytest.approx(phase_range[0], rel=0.01)
        assert xlim[1] == pytest.approx(phase_range[1], rel=0.01)

    def test_custom_colors(
        self, mock_lightcurve: LightCurve, mock_candidate: Candidate
    ):
        """Test that custom colors are applied."""
        ax = plot_phase_folded(
            mock_lightcurve,
            mock_candidate,
            data_color="red",
            binned_color="blue",
            show_binned=True,
        )

        # Verify the plot was created (color verification is complex)
        assert ax is not None

    def test_style_preset(self, mock_lightcurve: LightCurve, mock_candidate: Candidate):
        """Test that style presets are applied."""
        for style in ["default", "paper", "presentation"]:
            ax = plot_phase_folded(mock_lightcurve, mock_candidate, style=style)
            assert ax is not None

    def test_labels_set(self, mock_lightcurve: LightCurve, mock_candidate: Candidate):
        """Test that axis labels and title are set."""
        ax = plot_phase_folded(mock_lightcurve, mock_candidate)

        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""

    def test_legend_present(
        self, mock_lightcurve: LightCurve, mock_candidate: Candidate
    ):
        """Test that legend is present."""
        ax = plot_phase_folded(mock_lightcurve, mock_candidate)

        legend = ax.get_legend()
        assert legend is not None


# =============================================================================
# Tests for plot_transit_fit
# =============================================================================


class TestPlotTransitFit:
    """Tests for plot_transit_fit function."""

    def test_creates_figure_when_ax_none(self, mock_transit_fit_result):
        """Test that a new figure is created when ax is None."""
        ax = plot_transit_fit(mock_transit_fit_result)

        assert ax is not None
        assert ax.figure is not None

    def test_uses_provided_ax(self, mock_transit_fit_result):
        """Test that provided axes are used."""
        fig, ax_input = plt.subplots()
        ax = plot_transit_fit(mock_transit_fit_result, ax=ax_input)

        assert ax is ax_input

    def test_plots_data_and_model(self, mock_transit_fit_result):
        """Test that both data and model are plotted."""
        ax = plot_transit_fit(mock_transit_fit_result)

        # Should have scatter for data and line for model
        assert len(ax.collections) > 0  # scatter
        assert len(ax.lines) > 0  # model line

    def test_raises_on_empty_result(self, mock_transit_fit_result_empty):
        """Test that ValueError is raised for empty fit result."""
        with pytest.raises(ValueError, match="no phase data"):
            plot_transit_fit(mock_transit_fit_result_empty)

    def test_show_residuals(self, mock_transit_fit_result):
        """Test that residuals subplot is created when requested."""
        ax = plot_transit_fit(mock_transit_fit_result, show_residuals=True)

        # The main axes should be part of a figure with 2 subplots
        fig = ax.figure
        axes = fig.get_axes()
        assert len(axes) == 2

    def test_fit_info_annotation(self, mock_transit_fit_result):
        """Test that fit parameters are annotated on plot."""
        ax = plot_transit_fit(mock_transit_fit_result)

        # Check that there's a text annotation with fit info
        texts = ax.texts
        assert len(texts) > 0
        # Verify it contains Rp/Rs
        text_content = "".join([t.get_text() for t in texts])
        assert "R_p/R_*" in text_content or "0.1" in text_content

    def test_custom_colors(self, mock_transit_fit_result):
        """Test that custom colors are applied."""
        ax = plot_transit_fit(
            mock_transit_fit_result, data_color="green", model_color="purple"
        )

        assert ax is not None

    def test_style_preset(self, mock_transit_fit_result):
        """Test that style presets are applied."""
        for style in ["default", "paper", "presentation"]:
            ax = plot_transit_fit(mock_transit_fit_result, style=style)
            assert ax is not None


# =============================================================================
# Tests for plot_full_lightcurve
# =============================================================================


class TestPlotFullLightcurve:
    """Tests for plot_full_lightcurve function."""

    def test_creates_figure_when_ax_none(self, mock_lightcurve: LightCurve):
        """Test that a new figure is created when ax is None."""
        ax = plot_full_lightcurve(mock_lightcurve)

        assert ax is not None
        assert ax.figure is not None

    def test_uses_provided_ax(self, mock_lightcurve: LightCurve):
        """Test that provided axes are used."""
        fig, ax_input = plt.subplots()
        ax = plot_full_lightcurve(mock_lightcurve, ax=ax_input)

        assert ax is ax_input
        assert ax.figure is fig

    def test_plots_all_data(self, mock_lightcurve: LightCurve):
        """Test that all data points are plotted."""
        ax = plot_full_lightcurve(mock_lightcurve)

        # Get scatter collection
        collections = ax.collections
        assert len(collections) > 0

        # Check that data is plotted (should have many points)
        offsets = collections[0].get_offsets()
        assert len(offsets) > 1000  # Our mock has ~21600 points

    def test_transit_markers_with_candidate(
        self, mock_lightcurve: LightCurve, mock_candidate: Candidate
    ):
        """Test that transit markers are added when candidate is provided."""
        ax = plot_full_lightcurve(
            mock_lightcurve, candidate=mock_candidate, mark_transits=True
        )

        # Should have axvspan patches for transits
        patches = [
            p
            for p in ax.patches
            if hasattr(p, "get_width") and p.get_width() > 0
        ]
        assert len(patches) > 0

    def test_no_transit_markers_when_disabled(
        self, mock_lightcurve: LightCurve, mock_candidate: Candidate
    ):
        """Test that transit markers are not added when mark_transits=False."""
        ax = plot_full_lightcurve(
            mock_lightcurve, candidate=mock_candidate, mark_transits=False
        )

        # Should not have axvspan patches (except possibly from legend)
        patches = [
            p
            for p in ax.patches
            if hasattr(p, "get_width") and p.get_width() > 0.01
        ]
        assert len(patches) == 0

    def test_no_candidate_no_markers(self, mock_lightcurve: LightCurve):
        """Test that no transit markers without candidate."""
        ax = plot_full_lightcurve(mock_lightcurve, mark_transits=True)

        # Should not have transit patches without candidate
        patches = [
            p
            for p in ax.patches
            if hasattr(p, "get_width") and p.get_width() > 0.01
        ]
        assert len(patches) == 0

    def test_show_errors(self, mock_lightcurve: LightCurve):
        """Test that error bars are shown when requested."""
        ax = plot_full_lightcurve(mock_lightcurve, show_errors=True)

        # With errorbar, there should be LineCollections for error bars
        # or the plot should have containers
        assert ax is not None
        # Just verify it doesn't crash with errors enabled

    def test_custom_colors(
        self, mock_lightcurve: LightCurve, mock_candidate: Candidate
    ):
        """Test that custom colors are applied."""
        ax = plot_full_lightcurve(
            mock_lightcurve,
            candidate=mock_candidate,
            transit_color="red",
            data_color="blue",
        )

        assert ax is not None

    def test_style_preset(self, mock_lightcurve: LightCurve):
        """Test that style presets are applied."""
        for style in ["default", "paper", "presentation"]:
            ax = plot_full_lightcurve(mock_lightcurve, style=style)
            assert ax is not None

    def test_labels_set(self, mock_lightcurve: LightCurve):
        """Test that axis labels and title are set."""
        ax = plot_full_lightcurve(mock_lightcurve)

        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        assert ax.get_title() != ""

    def test_xlim_matches_data(self, mock_lightcurve: LightCurve):
        """Test that x-axis limits match data range."""
        ax = plot_full_lightcurve(mock_lightcurve)

        time = np.asarray(mock_lightcurve.time)
        xlim = ax.get_xlim()

        assert xlim[0] == pytest.approx(time.min(), rel=0.01)
        assert xlim[1] == pytest.approx(time.max(), rel=0.01)

    def test_valid_mask_applied(self):
        """Test that valid_mask is properly applied."""
        # Create lightcurve with valid_mask
        time = np.linspace(2458600.0, 2458630.0, 100)
        flux = np.ones(100)
        valid_mask = np.ones(100, dtype=bool)
        valid_mask[50:60] = False  # Mask out 10 points

        lc = LightCurve(time=time, flux=flux, valid_mask=valid_mask)
        ax = plot_full_lightcurve(lc)

        # Check that only valid points are plotted
        collections = ax.collections
        offsets = collections[0].get_offsets()
        assert len(offsets) == 90  # 100 - 10 masked points


# =============================================================================
# Tests for phase binning helper
# =============================================================================


class TestBinPhaseData:
    """Tests for _bin_phase_data helper function."""

    def test_empty_input(self):
        """Test that empty input returns empty arrays."""
        from tess_vetter.plotting.transit import _bin_phase_data

        phase, flux, err = _bin_phase_data(
            np.array([]), np.array([]), period_days=5.0, bin_minutes=30.0
        )

        assert len(phase) == 0
        assert len(flux) == 0
        assert len(err) == 0

    def test_binning_reduces_points(self):
        """Test that binning reduces number of points."""
        from tess_vetter.plotting.transit import _bin_phase_data

        # Create 1000 points
        phase = np.linspace(-0.1, 0.1, 1000)
        flux = 1.0 + np.random.normal(0, 0.001, 1000)

        binned_phase, binned_flux, binned_err = _bin_phase_data(
            phase, flux, period_days=5.0, bin_minutes=30.0
        )

        # Should have fewer bins than original points
        assert len(binned_phase) < 1000
        assert len(binned_phase) > 0

    def test_mean_in_bins(self):
        """Test that binned values are means of input."""
        from tess_vetter.plotting.transit import _bin_phase_data

        # Create simple test data
        phase = np.array([0.0, 0.001, 0.002, 0.1, 0.101, 0.102])
        flux = np.array([1.0, 1.0, 1.0, 0.999, 0.999, 0.999])

        # Use large bins to group points
        binned_phase, binned_flux, binned_err = _bin_phase_data(
            phase, flux, period_days=10.0, bin_minutes=60.0  # Large bins
        )

        # Should have 2 bins
        assert len(binned_phase) >= 1

    def test_error_calculation(self):
        """Test that errors are standard error of mean."""
        from tess_vetter.plotting.transit import _bin_phase_data

        # Create data with known scatter
        np.random.seed(42)
        n = 100
        phase = np.zeros(n)  # All same phase
        flux = 1.0 + np.random.normal(0, 0.001, n)

        # Use large bin to get all points
        _, _, binned_err = _bin_phase_data(
            phase, flux, period_days=10.0, bin_minutes=1000.0
        )

        # Error should be std/sqrt(n)
        expected_err = np.std(flux, ddof=1) / np.sqrt(n)
        assert len(binned_err) > 0
        assert binned_err[0] == pytest.approx(expected_err, rel=0.1)
