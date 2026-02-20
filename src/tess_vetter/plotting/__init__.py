"""Plotting utilities for tess-vetter.

This module provides visualization functions for vetting check results.
All plotting functions require matplotlib to be installed.

To install matplotlib, run:
    pip install 'tess-vetter[plotting]'

The module uses lazy loading to avoid importing matplotlib until it is
actually needed, allowing the rest of the library to be used without
matplotlib installed.

Example:
    >>> from tess_vetter.plotting import plot_odd_even
    >>> ax = plot_odd_even(result)  # Requires matplotlib

Style System:
    The plotting module supports three style presets:
    - "default": Balanced for interactive exploration (8x5 inches, 100 dpi)
    - "paper": Publication-ready (3.5x2.5 inches, 300 dpi)
    - "presentation": Large fonts for slides (10x6 inches, 150 dpi)

    Apply styles using the style parameter on any plot function:
    >>> ax = plot_odd_even(result, style="paper")
"""

from __future__ import annotations

import importlib.util

# Check for matplotlib availability without importing it
MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None

__all__: list[str]

# Export plot functions if matplotlib is available
if MATPLOTLIB_AVAILABLE:
    from .catalog import plot_exofop_card, plot_nearby_ebs
    from .checks import (
        plot_depth_stability,
        plot_duration_consistency,
        plot_odd_even,
        plot_secondary_eclipse,
        plot_v_shape,
    )
    from .exovetter import plot_modshift, plot_sweet
    from .extended import (
        plot_alias_diagnostics,
        plot_ephemeris_reliability,
        plot_ghost_features,
        plot_model_comparison,
        plot_sector_consistency,
        plot_sensitivity_sweep,
    )
    from .false_alarm import plot_asymmetry, plot_data_gaps
    from .lightcurve import plot_full_lightcurve
    from .pixel import plot_aperture_curve, plot_centroid_shift, plot_difference_image
    from .report import plot_vetting_summary, save_vetting_report
    from .transit import plot_phase_folded, plot_transit_fit

    __all__ = [
        # V01-V05: Light curve checks
        "plot_odd_even",
        "plot_secondary_eclipse",
        "plot_duration_consistency",
        "plot_depth_stability",
        "plot_v_shape",
        # V06-V07: Catalog checks
        "plot_nearby_ebs",
        "plot_exofop_card",
        # V08-V10: Pixel checks
        "plot_centroid_shift",
        "plot_difference_image",
        "plot_aperture_curve",
        # V11-V12: Exovetter checks
        "plot_modshift",
        "plot_sweet",
        # V13, V15: False alarm checks
        "plot_data_gaps",
        "plot_asymmetry",
        # V16-V21: Extended checks
        "plot_model_comparison",
        "plot_ephemeris_reliability",
        "plot_sensitivity_sweep",
        "plot_alias_diagnostics",
        "plot_ghost_features",
        "plot_sector_consistency",
        # DVR summary report
        "plot_vetting_summary",
        "save_vetting_report",
        # Transit and lightcurve visualization
        "plot_phase_folded",
        "plot_transit_fit",
        "plot_full_lightcurve",
    ]
else:
    __all__ = []


def __getattr__(name: str) -> object:
    """Lazy attribute access that raises ImportError if matplotlib is missing.

    This allows the module to be imported even without matplotlib installed,
    but raises a helpful error when actually trying to use plotting functions.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Plotting requires matplotlib. Install with: "
            "pip install 'tess-vetter[plotting]'"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return module attributes for tab completion."""
    return sorted(set(globals().keys()) | set(__all__))
