"""Light curve visualization functions.

This module provides visualization functions for light curve data that work
directly with LightCurve and Candidate objects. These are useful for
inspecting raw data and identifying transits in the time series.

Functions:
    plot_full_lightcurve: Full time-series light curve with optional transit markers
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes

    from tess_vetter.api.types import Candidate, LightCurve


def plot_full_lightcurve(
    lc: LightCurve,
    *,
    ax: matplotlib.axes.Axes | None = None,
    candidate: Candidate | None = None,
    mark_transits: bool = True,
    transit_color: str | None = None,
    data_color: str | None = None,
    show_errors: bool = False,
    style: str = "default",
    **scatter_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot full time-series light curve.

    Creates a scatter plot showing the complete light curve with optional
    vertical spans marking predicted transit times. This visualization is
    useful for inspecting the overall data quality and identifying transits.

    Parameters
    ----------
    lc : LightCurve
        Light curve data with time, flux, and optional flux_err arrays.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    candidate : Candidate, optional
        Transit candidate with ephemeris. If provided and mark_transits=True,
        adds vertical spans at predicted transit times.
    mark_transits : bool, default=True
        Whether to mark predicted transit times (requires candidate).
    transit_color : str, optional
        Color for transit markers. Defaults to COLORS["transit"].
    data_color : str, optional
        Color for data points. Defaults to COLORS["out_of_transit"].
    show_errors : bool, default=False
        Whether to show error bars on data points.
    style : str, default="default"
        Style preset: "default", "paper", or "presentation".
    **scatter_kwargs : Any
        Additional keyword arguments passed to ax.scatter().

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Examples
    --------
    >>> from tess_vetter.plotting import plot_full_lightcurve
    >>> ax = plot_full_lightcurve(lc)  # Basic plot

    >>> # With transit markers
    >>> ax = plot_full_lightcurve(lc, candidate=candidate)

    >>> # Without transit markers
    >>> ax = plot_full_lightcurve(lc, candidate=candidate, mark_transits=False)

    >>> # Show error bars
    >>> ax = plot_full_lightcurve(lc, show_errors=True)
    """
    from ._core import ensure_ax, style_context
    from ._styles import COLORS, LABELS

    # Get colors from styles if not provided
    if transit_color is None:
        transit_color = COLORS["transit"]
    if data_color is None:
        data_color = COLORS["out_of_transit"]

    # Get time and flux arrays
    time = np.asarray(lc.time)
    flux = np.asarray(lc.flux)
    flux_err = np.asarray(lc.flux_err) if lc.flux_err is not None else None

    # Apply valid_mask if present
    if lc.valid_mask is not None:
        valid = np.asarray(lc.valid_mask, dtype=bool)
        time = time[valid]
        flux = flux[valid]
        if flux_err is not None:
            flux_err = flux_err[valid]

    with style_context(style):
        fig, ax = ensure_ax(ax)

        # Set default scatter kwargs
        scatter_defaults: dict[str, Any] = {
            "s": 2,
            "alpha": 0.7,
        }
        scatter_defaults.update(scatter_kwargs)

        # Plot data
        if show_errors and flux_err is not None:
            ax.errorbar(
                time,
                flux,
                yerr=flux_err,
                fmt="o",
                color=data_color,
                markersize=scatter_defaults.get("s", 2),
                alpha=scatter_defaults.get("alpha", 0.7),
                capsize=0,
                elinewidth=0.5,
                label="Data",
            )
        else:
            ax.scatter(
                time,
                flux,
                c=data_color,
                label="Data",
                **scatter_defaults,
            )

        # Mark transits if candidate provided
        if mark_transits and candidate is not None:
            _add_transit_markers(ax, time, candidate, transit_color)

        # Set labels
        ax.set_xlabel(LABELS["time_btjd"])
        ax.set_ylabel(LABELS["flux_normalized"])
        ax.set_title("Light Curve")

        # Set x-axis limits to data range
        if len(time) > 0:
            ax.set_xlim(time.min(), time.max())

    return ax


def _add_transit_markers(
    ax: matplotlib.axes.Axes,
    time: np.ndarray,
    candidate: Candidate,
    color: str,
) -> None:
    """Add vertical spans marking transit times.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add markers to.
    time : np.ndarray
        Time array to determine transit epochs.
    candidate : Candidate
        Transit candidate with ephemeris.
    color : str
        Color for the transit markers.
    """
    if len(time) == 0:
        return

    period = candidate.ephemeris.period_days
    t0 = candidate.ephemeris.t0_btjd
    duration_hours = candidate.ephemeris.duration_hours
    duration_days = duration_hours / 24.0

    # Calculate which transits fall within the data range
    t_min = time.min()
    t_max = time.max()

    # Find the epoch range that covers the data
    epoch_min = int(np.floor((t_min - t0) / period)) - 1
    epoch_max = int(np.ceil((t_max - t0) / period)) + 1

    # Add vertical spans for each transit, limiting to avoid excessive markers
    max_transits = 50  # Cap to avoid performance issues
    n_transits = 0

    for epoch in range(epoch_min, epoch_max + 1):
        t_mid = t0 + epoch * period
        t_start = t_mid - duration_days / 2
        t_end = t_mid + duration_days / 2

        # Check if transit overlaps with data range
        if t_end >= t_min and t_start <= t_max:
            # Clip to data range
            t_start_clip = max(t_start, t_min)
            t_end_clip = min(t_end, t_max)

            ax.axvspan(
                t_start_clip,
                t_end_clip,
                alpha=0.2,
                color=color,
                label="Transit" if n_transits == 0 else None,
            )
            n_transits += 1

            if n_transits >= max_transits:
                break

    # Add legend entry for transit markers if any were added
    if n_transits > 0:
        ax.legend(loc="upper right")
