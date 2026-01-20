"""Plotting utilities for bittr-tess-vetter.

This module provides visualization functions for vetting check results.
All plotting functions require matplotlib to be installed.

To install matplotlib, run:
    pip install 'bittr-tess-vetter[plotting]'

The module uses lazy loading to avoid importing matplotlib until it is
actually needed, allowing the rest of the library to be used without
matplotlib installed.

Example:
    >>> from bittr_tess_vetter.plotting import plot_odd_even
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

# Export plot functions if matplotlib is available
if MATPLOTLIB_AVAILABLE:
    from .checks import plot_odd_even

    __all__: list[str] = [
        "plot_odd_even",
    ]
else:
    __all__: list[str] = []


def __getattr__(name: str) -> object:
    """Lazy attribute access that raises ImportError if matplotlib is missing.

    This allows the module to be imported even without matplotlib installed,
    but raises a helpful error when actually trying to use plotting functions.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Plotting requires matplotlib. Install with: "
            "pip install 'bittr-tess-vetter[plotting]'"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return module attributes for tab completion."""
    return sorted(set(globals().keys()) | set(__all__))
