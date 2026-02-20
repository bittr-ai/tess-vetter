"""Core plotting utilities for tess-vetter.

This module provides low-level utilities used by all plot functions:
- ensure_ax: Create or validate matplotlib axes
- add_colorbar: Add colorbar with astronomy defaults
- style_context: Context manager for style presets
- extract_plot_data: Extract and validate plot_data from CheckResult
- compute_subplot_grid: Optimal grid layout for multiple subplots
- get_sector_color: Consistent sector coloring

All matplotlib imports are lazy (inside functions) to allow importing
this module even without matplotlib installed.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure, SubFigure

    from tess_vetter.validation.result_schema import CheckResult


def ensure_ax(ax: Axes | None = None) -> tuple[Figure | SubFigure, Axes]:
    """Return (figure, axes), creating new ones if ax is None.

    This is the standard way to handle the optional ax parameter in plot
    functions. If ax is provided, returns its parent figure. If ax is None,
    creates a new figure and axes.

    Args:
        ax: Optional matplotlib Axes. If None, creates new figure and axes.

    Returns:
        Tuple of (Figure, Axes).

    Example:
        >>> fig, ax = ensure_ax()  # Creates new figure
        >>> fig, ax = ensure_ax(existing_ax)  # Uses existing axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax
    return ax.figure, ax


def add_colorbar(
    mappable: ScalarMappable,
    ax: Axes,
    *,
    label: str = "",
    **kwargs: Any,
) -> Colorbar:
    """Add colorbar with astronomy defaults.

    Creates a colorbar for an image plot with sensible defaults for
    astronomical data visualization: no minor ticks, positioned to the
    right of the axes.

    Args:
        mappable: The matplotlib ScalarMappable (e.g., return from imshow).
        ax: The axes to attach the colorbar to.
        label: Colorbar label text.
        **kwargs: Additional arguments passed to figure.colorbar().

    Returns:
        The created Colorbar instance.

    Example:
        >>> im = ax.imshow(data, origin="lower")
        >>> cbar = add_colorbar(im, ax, label="Flux (e-/s)")
    """
    fig = ax.figure
    cbar = fig.colorbar(mappable, ax=ax, **kwargs)
    if label:
        cbar.set_label(label)
    # Disable minor ticks for cleaner appearance
    cbar.ax.minorticks_off()
    return cbar


@contextmanager
def style_context(style: str = "default") -> Iterator[None]:
    """Context manager for applying a style preset temporarily.

    Applies matplotlib rcParams from the specified style preset, then
    reverts to the previous settings on exit. This ensures styles don't
    leak between plot functions.

    Args:
        style: Style preset name. One of "default", "paper", "presentation".

    Yields:
        None. Use in a with statement.

    Raises:
        ValueError: If style is not recognized.

    Example:
        >>> with style_context("paper"):
        ...     fig, ax = ensure_ax()
        ...     ax.plot(x, y)
        >>> # Original rcParams restored here
    """
    import matplotlib.pyplot as plt

    from ._styles import STYLES

    if style not in STYLES:
        valid_styles = ", ".join(sorted(STYLES.keys()))
        raise ValueError(f"Unknown style {style!r}. Valid styles: {valid_styles}")

    rc_params = STYLES[style]
    with plt.rc_context(rc_params):
        yield


def extract_plot_data(
    result: CheckResult,
    required_keys: list[str],
) -> dict[str, Any]:
    """Extract and validate plot_data from a CheckResult.

    Retrieves the plot_data dict from result.raw and validates that all
    required keys are present. This is the standard way to access plotting
    data in plot functions.

    Args:
        result: A CheckResult instance with raw["plot_data"] populated.
        required_keys: List of keys that must be present in plot_data.

    Returns:
        The plot_data dict.

    Raises:
        ValueError: If result.raw is None, plot_data is missing, or any
            required key is missing from plot_data.

    Example:
        >>> data = extract_plot_data(result, ["odd_depths_ppm", "even_depths_ppm"])
        >>> odd = data["odd_depths_ppm"]
    """
    if result.raw is None:
        raise ValueError(
            f"CheckResult {result.id} has no raw data. "
            f"Ensure the check was run with plot_data support."
        )

    plot_data = result.raw.get("plot_data")
    if plot_data is None:
        raise ValueError(
            f"CheckResult {result.id} has no plot_data in raw. "
            f"Ensure the check was run with plot_data support."
        )

    missing_keys = [key for key in required_keys if key not in plot_data]
    if missing_keys:
        raise ValueError(
            f"CheckResult {result.id} plot_data missing required keys: {missing_keys}"
        )

    return plot_data


def compute_subplot_grid(n: int) -> tuple[int, int]:
    """Compute optimal (nrows, ncols) for n subplots.

    Returns a grid layout that minimizes wasted space while preferring
    wider-than-tall layouts (more columns than rows).

    Args:
        n: Number of subplots needed.

    Returns:
        Tuple of (nrows, ncols) such that nrows * ncols >= n.

    Example:
        >>> compute_subplot_grid(1)
        (1, 1)
        >>> compute_subplot_grid(4)
        (2, 2)
        >>> compute_subplot_grid(5)
        (2, 3)
    """
    if n <= 0:
        return (1, 1)
    if n == 1:
        return (1, 1)
    if n == 2:
        return (1, 2)
    if n == 3:
        return (1, 3)
    if n == 4:
        return (2, 2)

    # For larger n, find the smallest grid that fits
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    # Prefer wider layouts (more columns)
    while nrows > ncols and (nrows - 1) * ncols >= n:
        nrows -= 1

    return (nrows, ncols)


def get_sector_color(sector: int, all_sectors: list[int]) -> str:
    """Get a consistent hex color for a sector.

    Uses the matplotlib tab10 colormap to assign colors based on the
    sector's position in the sorted list of all sectors. This ensures
    consistent coloring across multiple plots.

    Args:
        sector: The sector number to get a color for.
        all_sectors: List of all sector numbers for consistent indexing.

    Returns:
        Hex color string (e.g., "#1f77b4").

    Example:
        >>> sectors = [1, 5, 10]
        >>> get_sector_color(5, sectors)
        '#ff7f0e'  # Second color in tab10
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    # Get tab10 colormap colors
    cmap = plt.get_cmap("tab10")

    # Sort sectors for consistent ordering
    sorted_sectors = sorted(all_sectors)

    # Find index of this sector
    try:
        idx = sorted_sectors.index(sector)
    except ValueError:
        idx = 0

    # Cycle through tab10 colors
    color_idx = idx % 10
    rgba = cmap(color_idx)

    return mcolors.to_hex(rgba)
