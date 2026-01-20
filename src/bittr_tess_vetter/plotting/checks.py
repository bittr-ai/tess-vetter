"""Plot functions for light curve vetting checks (V01-V05)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.axes

    from bittr_tess_vetter.validation.result_schema import CheckResult


def plot_odd_even(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    odd_color: str | None = None,
    even_color: str | None = None,
    show_legend: bool = True,
    show_means: bool = True,
    annotate_sigma: bool = True,
    style: str = "default",
    **errorbar_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot odd vs even transit depth comparison.

    Creates a scatter plot comparing transit depths from odd and even epochs
    to check for eclipsing binary contamination. Consistent depths between
    odd and even transits suggest a true planetary signal, while differing
    depths may indicate a blended eclipsing binary.

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V01 (odd/even depth) check. Must contain
        plot_data with odd/even epoch depths and errors.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    odd_color : str, optional
        Color for odd epoch points. Defaults to COLORS["odd"] (red).
    even_color : str, optional
        Color for even epoch points. Defaults to COLORS["even"] (green).
    show_legend : bool, default=True
        Whether to show the legend.
    show_means : bool, default=True
        Whether to show horizontal lines for odd/even mean depths.
    annotate_sigma : bool, default=True
        Whether to show sigma difference in the title.
    style : str, default="default"
        Style preset: "default", "paper", or "presentation".
    **errorbar_kwargs : Any
        Additional keyword arguments passed to ax.errorbar().

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Raises
    ------
    ValueError
        If result has no plot_data or is missing required keys.

    Examples
    --------
    >>> from bittr_tess_vetter.plotting import plot_odd_even
    >>> ax = plot_odd_even(result)  # Basic plot

    >>> # Publication-ready with custom colors
    >>> ax = plot_odd_even(
    ...     result,
    ...     style="paper",
    ...     odd_color="blue",
    ...     even_color="orange",
    ... )

    >>> # Add to existing subplot
    >>> fig, ax = plt.subplots()
    >>> plot_odd_even(result, ax=ax, show_legend=False)
    """
    from ._core import ensure_ax, extract_plot_data, style_context
    from ._styles import COLORS

    # Required keys for V01 odd/even plot
    required_keys = [
        "odd_epochs",
        "odd_depths_ppm",
        "odd_errs_ppm",
        "even_epochs",
        "even_depths_ppm",
        "even_errs_ppm",
        "mean_odd_ppm",
        "mean_even_ppm",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    # Get colors from styles if not provided
    if odd_color is None:
        odd_color = COLORS["odd"]
    if even_color is None:
        even_color = COLORS["even"]

    with style_context(style):
        fig, ax = ensure_ax(ax)

        # Set default errorbar kwargs
        errorbar_defaults = {
            "fmt": "o",
            "capsize": 3,
            "capthick": 1,
            "markersize": 6,
        }
        errorbar_defaults.update(errorbar_kwargs)

        # Plot odd epochs with circles
        ax.errorbar(
            data["odd_epochs"],
            data["odd_depths_ppm"],
            yerr=data["odd_errs_ppm"],
            color=odd_color,
            label="Odd",
            marker="o",
            **{k: v for k, v in errorbar_defaults.items() if k != "fmt"},
        )

        # Plot even epochs with squares
        ax.errorbar(
            data["even_epochs"],
            data["even_depths_ppm"],
            yerr=data["even_errs_ppm"],
            color=even_color,
            label="Even",
            marker="s",
            **{k: v for k, v in errorbar_defaults.items() if k != "fmt"},
        )

        # Add horizontal lines for means if requested
        if show_means:
            ax.axhline(
                data["mean_odd_ppm"],
                color=odd_color,
                linestyle="--",
                alpha=0.7,
                linewidth=1,
            )
            ax.axhline(
                data["mean_even_ppm"],
                color=even_color,
                linestyle="--",
                alpha=0.7,
                linewidth=1,
            )

        # Set labels
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Depth (ppm)")

        # Set title with optional sigma annotation
        if annotate_sigma and result.metrics:
            sigma_diff = result.metrics.get("sigma_diff")
            if sigma_diff is not None:
                ax.set_title(f"Odd/Even Depth Comparison ({sigma_diff:.2f}$\\sigma$)")
            else:
                ax.set_title("Odd/Even Depth Comparison")
        else:
            ax.set_title("Odd/Even Depth Comparison")

        # Add legend if requested
        if show_legend:
            ax.legend()

    return ax
