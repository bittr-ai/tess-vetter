"""Plot functions for false alarm vetting checks (V13, V15)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.colorbar

    from bittr_tess_vetter.validation.result_schema import CheckResult


def plot_data_gaps(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    bar_color: str | None = None,
    threshold_color: str | None = None,
    show_legend: bool = True,
    show_threshold: bool = True,
    threshold_value: float = 0.25,
    annotate_max: bool = True,
    style: str = "default",
    **bar_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot per-epoch data coverage near transit windows.

    Creates a bar chart showing the coverage fraction (1 - missing_frac) for
    each epoch's transit window. Low coverage may indicate gap-edge artifacts.

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V13 (data gaps) check. Must contain plot_data
        with epoch_centers_btjd and coverage_fractions arrays.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    bar_color : str, optional
        Color for coverage bars. Defaults to COLORS["transit"] (blue).
    threshold_color : str, optional
        Color for threshold line. Defaults to COLORS["threshold"] (pink).
    show_legend : bool, default=True
        Whether to show the legend.
    show_threshold : bool, default=True
        Whether to show a horizontal threshold line.
    threshold_value : float, default=0.25
        The coverage threshold below which epochs are flagged.
    annotate_max : bool, default=True
        Whether to annotate the maximum missing fraction.
    style : str, default="default"
        Style preset: "default", "paper", or "presentation".
    **bar_kwargs : Any
        Additional keyword arguments passed to ax.bar().

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Raises
    ------
    ValueError
        If result has no plot_data or is missing required keys.

    Notes
    -----
    Expected plot_data schema (version 1):

    .. code-block:: python

        plot_data = {
            "version": 1,
            "epoch_centers_btjd": [...],      # Transit center times
            "coverage_fractions": [...],       # 0-1 per epoch (1 = fully covered)
            "transit_window_hours": float,     # Window size used
        }

    Examples
    --------
    >>> from bittr_tess_vetter.plotting import plot_data_gaps
    >>> ax = plot_data_gaps(result)  # Basic plot

    >>> # Publication-ready with custom threshold
    >>> ax = plot_data_gaps(result, style="paper", threshold_value=0.5)
    """
    from ._core import ensure_ax, extract_plot_data, style_context
    from ._styles import COLORS

    # Required keys for V13 data gaps plot
    required_keys = [
        "epoch_centers_btjd",
        "coverage_fractions",
        "transit_window_hours",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    # Get colors from styles if not provided
    if bar_color is None:
        bar_color = COLORS["transit"]
    if threshold_color is None:
        threshold_color = COLORS["threshold"]

    with style_context(style):
        fig, ax = ensure_ax(ax)

        epoch_centers = data["epoch_centers_btjd"]
        coverage = data["coverage_fractions"]
        window_hours = data["transit_window_hours"]

        # Set default bar kwargs
        bar_defaults = {
            "width": 0.8,
            "edgecolor": "black",
            "linewidth": 0.5,
            "alpha": 0.8,
        }
        bar_defaults.update(bar_kwargs)

        n_epochs = len(epoch_centers)
        x_positions = list(range(n_epochs))

        # Plot coverage bars
        if n_epochs > 0:
            ax.bar(
                x_positions,
                coverage,
                color=bar_color,
                label="Coverage",
                **bar_defaults,
            )

        # Add threshold line if requested
        if show_threshold:
            ax.axhline(
                threshold_value,
                color=threshold_color,
                linestyle="--",
                linewidth=1.5,
                label=f"Threshold ({threshold_value:.0%})",
            )

        # Annotate maximum missing fraction if requested
        if annotate_max and len(coverage) > 0:
            max_missing = 1.0 - min(coverage)
            ax.text(
                0.95,
                0.95,
                f"Max missing: {max_missing:.1%}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

        # Set labels
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Coverage Fraction")
        ax.set_title(f"Transit Window Coverage ({window_hours:.1f}h window)")

        # Set y-axis limits
        ax.set_ylim(0, 1.05)

        # Simplify x-axis for many epochs
        if n_epochs > 20:
            # Show only every Nth tick
            step = max(1, n_epochs // 10)
            ax.set_xticks(x_positions[::step])
        else:
            ax.set_xticks(x_positions)

        # Add legend if requested
        if show_legend:
            ax.legend(loc="lower right")

    return ax


def plot_asymmetry(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    data_color: str | None = None,
    left_color: str | None = None,
    right_color: str | None = None,
    show_legend: bool = True,
    show_bins: bool = True,
    annotate_sigma: bool = True,
    style: str = "default",
    **scatter_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot phase-folded transit with left/right asymmetry bins highlighted.

    Creates a phase-folded scatter plot showing the transit region with
    colored bins on the left (pre-transit) and right (post-transit) sides.
    Significant asymmetry may indicate ramp or step artifacts.

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V15 (asymmetry) check. Must contain plot_data
        with phase, flux, and bin statistics.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    data_color : str, optional
        Color for data points. Defaults to COLORS["out_of_transit"] (gray).
    left_color : str, optional
        Color for left (pre-transit) bin. Defaults to COLORS["odd"] (red).
    right_color : str, optional
        Color for right (post-transit) bin. Defaults to COLORS["even"] (green).
    show_legend : bool, default=True
        Whether to show the legend.
    show_bins : bool, default=True
        Whether to show shaded regions for left/right bins.
    annotate_sigma : bool, default=True
        Whether to annotate the asymmetry significance.
    style : str, default="default"
        Style preset: "default", "paper", or "presentation".
    **scatter_kwargs : Any
        Additional keyword arguments passed to ax.scatter().

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Raises
    ------
    ValueError
        If result has no plot_data or is missing required keys.

    Notes
    -----
    Expected plot_data schema (version 1):

    .. code-block:: python

        plot_data = {
            "version": 1,
            "phase": [...],
            "flux": [...],
            "left_bin_mean": float,
            "right_bin_mean": float,
            "left_bin_phase_range": [float, float],
            "right_bin_phase_range": [float, float],
        }

    Examples
    --------
    >>> from bittr_tess_vetter.plotting import plot_asymmetry
    >>> ax = plot_asymmetry(result)  # Basic plot

    >>> # Customize bin colors
    >>> ax = plot_asymmetry(result, left_color="blue", right_color="orange")
    """
    from ._core import ensure_ax, extract_plot_data, style_context
    from ._styles import COLORS

    # Required keys for V15 asymmetry plot
    required_keys = [
        "phase",
        "flux",
        "left_bin_mean",
        "right_bin_mean",
        "left_bin_phase_range",
        "right_bin_phase_range",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    # Get colors from styles if not provided
    if data_color is None:
        data_color = COLORS["out_of_transit"]
    if left_color is None:
        left_color = COLORS["odd"]
    if right_color is None:
        right_color = COLORS["even"]

    with style_context(style):
        fig, ax = ensure_ax(ax)

        phase = data["phase"]
        flux = data["flux"]
        left_range = data["left_bin_phase_range"]
        right_range = data["right_bin_phase_range"]
        left_mean = data["left_bin_mean"]
        right_mean = data["right_bin_mean"]

        # Set default scatter kwargs
        scatter_defaults = {
            "s": 4,
            "alpha": 0.5,
        }
        scatter_defaults.update(scatter_kwargs)

        # Plot data points
        if len(phase) > 0:
            ax.scatter(
                phase,
                flux,
                c=data_color,
                label="Data",
                **scatter_defaults,
            )

        # Add bin shading if requested
        if show_bins:
            # Left bin (pre-transit)
            ax.axvspan(
                left_range[0],
                left_range[1],
                alpha=0.2,
                color=left_color,
                label=f"Left bin (mean: {left_mean:.4f})",
            )
            # Right bin (post-transit)
            ax.axvspan(
                right_range[0],
                right_range[1],
                alpha=0.2,
                color=right_color,
                label=f"Right bin (mean: {right_mean:.4f})",
            )

            # Add horizontal lines at bin means
            ax.axhline(
                left_mean,
                xmin=0.1,
                xmax=0.45,
                color=left_color,
                linestyle="--",
                linewidth=1.5,
            )
            ax.axhline(
                right_mean,
                xmin=0.55,
                xmax=0.9,
                color=right_color,
                linestyle="--",
                linewidth=1.5,
            )

        # Annotate asymmetry sigma if available and requested
        if annotate_sigma and result.metrics:
            sigma = result.metrics.get("asymmetry_sigma")
            if sigma is not None:
                ax.text(
                    0.95,
                    0.95,
                    f"Asymmetry: {sigma:.2f}$\\sigma$",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                )

        # Set labels
        ax.set_xlabel("Phase")
        ax.set_ylabel("Normalized Flux")
        ax.set_title("Transit Asymmetry Analysis")

        # Add legend if requested
        if show_legend:
            ax.legend(loc="lower right", fontsize=7)

    return ax


__all__ = [
    "plot_data_gaps",
    "plot_asymmetry",
]
