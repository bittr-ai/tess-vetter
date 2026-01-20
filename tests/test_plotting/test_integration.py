"""Integration tests for plotting with API and real vetting results.

These tests verify:
1. Plotting functions are accessible from the api module
2. End-to-end workflow from running checks to plotting results
3. No warnings are generated during normal plotting operations
"""

from __future__ import annotations

import warnings

import pytest

pytest.importorskip("matplotlib")

# Use non-interactive backend for tests
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


class TestAPIImports:
    """Test that plotting functions are accessible from api module."""

    def test_matplotlib_available_flag(self):
        """MATPLOTLIB_AVAILABLE is exported and True when matplotlib is installed."""
        from bittr_tess_vetter.api import MATPLOTLIB_AVAILABLE

        assert MATPLOTLIB_AVAILABLE is True

    def test_import_plot_odd_even_from_api(self):
        """plot_odd_even can be imported from api module."""
        from bittr_tess_vetter.api import plot_odd_even

        assert callable(plot_odd_even)

    def test_import_plot_secondary_eclipse_from_api(self):
        """plot_secondary_eclipse can be imported from api module."""
        from bittr_tess_vetter.api import plot_secondary_eclipse

        assert callable(plot_secondary_eclipse)

    def test_import_plot_duration_consistency_from_api(self):
        """plot_duration_consistency can be imported from api module."""
        from bittr_tess_vetter.api import plot_duration_consistency

        assert callable(plot_duration_consistency)

    def test_import_plot_depth_stability_from_api(self):
        """plot_depth_stability can be imported from api module."""
        from bittr_tess_vetter.api import plot_depth_stability

        assert callable(plot_depth_stability)

    def test_import_plot_v_shape_from_api(self):
        """plot_v_shape can be imported from api module."""
        from bittr_tess_vetter.api import plot_v_shape

        assert callable(plot_v_shape)

    def test_import_plot_nearby_ebs_from_api(self):
        """plot_nearby_ebs can be imported from api module."""
        from bittr_tess_vetter.api import plot_nearby_ebs

        assert callable(plot_nearby_ebs)

    def test_import_plot_exofop_card_from_api(self):
        """plot_exofop_card can be imported from api module."""
        from bittr_tess_vetter.api import plot_exofop_card

        assert callable(plot_exofop_card)

    def test_import_plot_centroid_shift_from_api(self):
        """plot_centroid_shift can be imported from api module."""
        from bittr_tess_vetter.api import plot_centroid_shift

        assert callable(plot_centroid_shift)

    def test_import_plot_difference_image_from_api(self):
        """plot_difference_image can be imported from api module."""
        from bittr_tess_vetter.api import plot_difference_image

        assert callable(plot_difference_image)

    def test_import_plot_aperture_curve_from_api(self):
        """plot_aperture_curve can be imported from api module."""
        from bittr_tess_vetter.api import plot_aperture_curve

        assert callable(plot_aperture_curve)

    def test_import_plot_modshift_from_api(self):
        """plot_modshift can be imported from api module."""
        from bittr_tess_vetter.api import plot_modshift

        assert callable(plot_modshift)

    def test_import_plot_sweet_from_api(self):
        """plot_sweet can be imported from api module."""
        from bittr_tess_vetter.api import plot_sweet

        assert callable(plot_sweet)

    def test_import_plot_data_gaps_from_api(self):
        """plot_data_gaps can be imported from api module."""
        from bittr_tess_vetter.api import plot_data_gaps

        assert callable(plot_data_gaps)

    def test_import_plot_asymmetry_from_api(self):
        """plot_asymmetry can be imported from api module."""
        from bittr_tess_vetter.api import plot_asymmetry

        assert callable(plot_asymmetry)

    def test_import_plot_model_comparison_from_api(self):
        """plot_model_comparison can be imported from api module."""
        from bittr_tess_vetter.api import plot_model_comparison

        assert callable(plot_model_comparison)

    def test_import_plot_ephemeris_reliability_from_api(self):
        """plot_ephemeris_reliability can be imported from api module."""
        from bittr_tess_vetter.api import plot_ephemeris_reliability

        assert callable(plot_ephemeris_reliability)

    def test_import_plot_alias_diagnostics_from_api(self):
        """plot_alias_diagnostics can be imported from api module."""
        from bittr_tess_vetter.api import plot_alias_diagnostics

        assert callable(plot_alias_diagnostics)

    def test_import_plot_ghost_features_from_api(self):
        """plot_ghost_features can be imported from api module."""
        from bittr_tess_vetter.api import plot_ghost_features

        assert callable(plot_ghost_features)

    def test_import_plot_sector_consistency_from_api(self):
        """plot_sector_consistency can be imported from api module."""
        from bittr_tess_vetter.api import plot_sector_consistency

        assert callable(plot_sector_consistency)

    def test_import_plot_vetting_summary_from_api(self):
        """plot_vetting_summary can be imported from api module."""
        from bittr_tess_vetter.api import plot_vetting_summary

        assert callable(plot_vetting_summary)

    def test_import_save_vetting_report_from_api(self):
        """save_vetting_report can be imported from api module."""
        from bittr_tess_vetter.api import save_vetting_report

        assert callable(save_vetting_report)

    def test_import_plot_phase_folded_from_api(self):
        """plot_phase_folded can be imported from api module."""
        from bittr_tess_vetter.api import plot_phase_folded

        assert callable(plot_phase_folded)

    def test_import_plot_transit_fit_from_api(self):
        """plot_transit_fit can be imported from api module."""
        from bittr_tess_vetter.api import plot_transit_fit

        assert callable(plot_transit_fit)

    def test_import_plot_full_lightcurve_from_api(self):
        """plot_full_lightcurve can be imported from api module."""
        from bittr_tess_vetter.api import plot_full_lightcurve

        assert callable(plot_full_lightcurve)

    def test_all_plotting_functions_in_dir(self):
        """All plotting functions appear in dir(api)."""
        import bittr_tess_vetter.api as api

        api_dir = dir(api)

        expected_functions = [
            "plot_odd_even",
            "plot_secondary_eclipse",
            "plot_duration_consistency",
            "plot_depth_stability",
            "plot_v_shape",
            "plot_nearby_ebs",
            "plot_exofop_card",
            "plot_centroid_shift",
            "plot_difference_image",
            "plot_aperture_curve",
            "plot_modshift",
            "plot_sweet",
            "plot_data_gaps",
            "plot_asymmetry",
            "plot_model_comparison",
            "plot_ephemeris_reliability",
            "plot_alias_diagnostics",
            "plot_ghost_features",
            "plot_sector_consistency",
            "plot_vetting_summary",
            "save_vetting_report",
            "plot_phase_folded",
            "plot_transit_fit",
            "plot_full_lightcurve",
        ]

        for func_name in expected_functions:
            assert func_name in api_dir, f"{func_name} not in dir(api)"


class TestEndToEnd:
    """Test plotting from actual vetting results."""

    def test_plot_from_mock_v01_result(self, mock_v01_result):
        """Run V01 plotting on a mock result using api import."""
        from bittr_tess_vetter.api import plot_odd_even

        ax = plot_odd_even(mock_v01_result)
        assert ax is not None
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(ax.figure)

    def test_plot_from_mock_v02_result(self, mock_v02_result):
        """Run V02 plotting on a mock result using api import."""
        from bittr_tess_vetter.api import plot_secondary_eclipse

        ax = plot_secondary_eclipse(mock_v02_result)
        assert ax is not None
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(ax.figure)

    def test_plot_from_mock_v03_result(self, mock_v03_result):
        """Run V03 plotting on a mock result using api import."""
        from bittr_tess_vetter.api import plot_duration_consistency

        ax = plot_duration_consistency(mock_v03_result)
        assert ax is not None
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(ax.figure)

    def test_plot_from_mock_v04_result(self, mock_v04_result):
        """Run V04 plotting on a mock result using api import."""
        from bittr_tess_vetter.api import plot_depth_stability

        ax = plot_depth_stability(mock_v04_result)
        assert ax is not None
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(ax.figure)

    def test_plot_from_mock_v05_result(self, mock_v05_result):
        """Run V05 plotting on a mock result using api import."""
        from bittr_tess_vetter.api import plot_v_shape

        ax = plot_v_shape(mock_v05_result)
        assert ax is not None
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(ax.figure)


class TestNoMatplotlibWarnings:
    """Verify no matplotlib warnings during plotting."""

    def test_no_warnings_plot_odd_even(self, mock_v01_result):
        """plot_odd_even should not generate warnings."""
        from bittr_tess_vetter.api import plot_odd_even

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = plot_odd_even(mock_v01_result)
            plt.close(ax.figure)

            # Filter for matplotlib-related warnings
            mpl_warnings = [
                warning for warning in w if "matplotlib" in str(warning.category).lower()
            ]
            assert len(mpl_warnings) == 0, f"Got matplotlib warnings: {mpl_warnings}"

    def test_no_warnings_plot_secondary_eclipse(self, mock_v02_result):
        """plot_secondary_eclipse should not generate warnings."""
        from bittr_tess_vetter.api import plot_secondary_eclipse

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = plot_secondary_eclipse(mock_v02_result)
            plt.close(ax.figure)

            mpl_warnings = [
                warning for warning in w if "matplotlib" in str(warning.category).lower()
            ]
            assert len(mpl_warnings) == 0, f"Got matplotlib warnings: {mpl_warnings}"

    def test_no_deprecation_warnings(self, mock_v01_result):
        """Plotting should not generate deprecation warnings."""
        from bittr_tess_vetter.api import plot_odd_even

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = plot_odd_even(mock_v01_result)
            plt.close(ax.figure)

            # Filter for deprecation warnings
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, (DeprecationWarning, PendingDeprecationWarning))
            ]
            # Note: We filter out third-party deprecations that we can't control
            own_deprecations = [
                warning
                for warning in deprecation_warnings
                if "bittr_tess_vetter" in str(warning.filename)
            ]
            assert len(own_deprecations) == 0, f"Got deprecation warnings: {own_deprecations}"


class TestAPIModuleConsistency:
    """Test that api module exports match plotting module exports."""

    def test_api_exports_match_plotting_module(self):
        """All plotting module exports should be accessible from api."""
        from bittr_tess_vetter import api, plotting

        plotting_all = set(plotting.__all__)
        for name in plotting_all:
            # Should be able to get attribute from api
            assert hasattr(api, name), f"api module missing {name} from plotting"
            # Should be callable
            func = getattr(api, name)
            assert callable(func), f"{name} from api is not callable"

    def test_api_plotting_imports_are_same_objects(self):
        """Plotting functions from api should be same objects as from plotting."""
        from bittr_tess_vetter import api, plotting

        # Check a few key functions
        for name in ["plot_odd_even", "plot_vetting_summary", "plot_centroid_shift"]:
            plotting_func = getattr(plotting, name)
            api_func = getattr(api, name)
            assert plotting_func is api_func, f"{name} is not the same object"
