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

        eb_kw: Any = {k: v for k, v in errorbar_defaults.items() if k != "fmt"}

        # Plot odd epochs with circles
        ax.errorbar(
            data["odd_epochs"],
            data["odd_depths_ppm"],
            yerr=data["odd_errs_ppm"],
            color=odd_color,
            label="Odd",
            marker="o",
            **eb_kw,
        )

        # Plot even epochs with squares
        ax.errorbar(
            data["even_epochs"],
            data["even_depths_ppm"],
            yerr=data["even_errs_ppm"],
            color=even_color,
            label="Even",
            marker="s",
            **eb_kw,
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


def plot_secondary_eclipse(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    data_color: str | None = None,
    secondary_color: str | None = None,
    primary_color: str | None = None,
    show_legend: bool = True,
    show_windows: bool = True,
    annotate_depth: bool = True,
    style: str = "default",
    **scatter_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot phase-folded light curve with secondary eclipse window.

    Creates a phase-folded scatter plot highlighting the secondary eclipse
    search window and primary transit exclusion zone. Shaded regions indicate
    where the secondary eclipse is expected (around phase 0.5) and where the
    primary transit occurs (around phase 0).

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V02 (secondary eclipse) check. Must contain
        plot_data with phase, flux, and window boundaries.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    data_color : str, optional
        Color for data points. Defaults to COLORS["out_of_transit"] (gray).
    secondary_color : str, optional
        Color for secondary window shading. Defaults to COLORS["secondary"].
    primary_color : str, optional
        Color for primary window shading. Defaults to COLORS["transit"].
    show_legend : bool, default=True
        Whether to show the legend.
    show_windows : bool, default=True
        Whether to show shaded window regions.
    annotate_depth : bool, default=True
        Whether to annotate the secondary depth on the plot.
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

    Examples
    --------
    >>> from bittr_tess_vetter.plotting import plot_secondary_eclipse
    >>> ax = plot_secondary_eclipse(result)  # Basic plot

    >>> # Customize window colors
    >>> ax = plot_secondary_eclipse(
    ...     result,
    ...     secondary_color="orange",
    ...     show_legend=False,
    ... )
    """
    from ._core import ensure_ax, extract_plot_data, style_context
    from ._styles import COLORS

    # Required keys for V02 secondary eclipse plot
    required_keys = [
        "phase",
        "flux",
        "secondary_window",
        "primary_window",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    # Get colors from styles if not provided
    if data_color is None:
        data_color = COLORS["out_of_transit"]
    if secondary_color is None:
        secondary_color = COLORS["secondary"]
    if primary_color is None:
        primary_color = COLORS["transit"]

    with style_context(style):
        fig, ax = ensure_ax(ax)

        phase = data["phase"]
        flux = data["flux"]

        # Set default scatter kwargs
        scatter_defaults: dict[str, Any] = {
            "s": 4,
            "alpha": 0.5,
        }
        scatter_defaults.update(scatter_kwargs)
        scatter_kw: Any = scatter_defaults

        # Plot data points
        if len(phase) > 0:
            ax.scatter(
                phase,
                flux,
                c=data_color,
                label="Data",
                **scatter_kw,
            )

        # Add window shading if requested
        if show_windows:
            sec_lo, sec_hi = data["secondary_window"]
            pri_lo, pri_hi = data["primary_window"]

            # Secondary window shading
            ax.axvspan(
                sec_lo,
                sec_hi,
                alpha=0.2,
                color=secondary_color,
                label="Secondary window",
            )

            # Primary window shading (transit exclusion)
            ax.axvspan(
                pri_lo,
                pri_hi,
                alpha=0.2,
                color=primary_color,
                label="Primary transit",
            )

        # Annotate secondary depth if available and requested
        if annotate_depth and data.get("secondary_depth_ppm") is not None:
            depth_ppm = data["secondary_depth_ppm"]
            ax.text(
                0.5,
                0.95,
                f"Secondary: {depth_ppm:.0f} ppm",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )

        # Set labels
        ax.set_xlabel("Orbital Phase")
        ax.set_ylabel("Normalized Flux")
        ax.set_title("Secondary Eclipse Search")

        # Set x-axis limits to show full phase
        ax.set_xlim(0, 1)

        # Add legend if requested
        if show_legend:
            ax.legend(loc="lower right")

    return ax


def plot_duration_consistency(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    observed_color: str | None = None,
    expected_color: str | None = None,
    show_legend: bool = True,
    show_error: bool = True,
    annotate_ratio: bool = True,
    style: str = "default",
    **bar_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot bar chart comparing observed vs expected transit duration.

    Creates a simple bar chart showing the observed transit duration alongside
    the expected duration based on stellar parameters and orbital period.
    Error bars indicate the uncertainty in the expected value.

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V03 (duration consistency) check. Must contain
        plot_data with observed_hours, expected_hours, and expected_hours_err.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    observed_color : str, optional
        Color for observed duration bar. Defaults to COLORS["transit"] (blue).
    expected_color : str, optional
        Color for expected duration bar. Defaults to COLORS["model"] (orange).
    show_legend : bool, default=True
        Whether to show the legend.
    show_error : bool, default=True
        Whether to show error bars on expected duration.
    annotate_ratio : bool, default=True
        Whether to annotate the duration ratio on the plot.
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

    Examples
    --------
    >>> from bittr_tess_vetter.plotting import plot_duration_consistency
    >>> ax = plot_duration_consistency(result)  # Basic plot

    >>> # Without error bars
    >>> ax = plot_duration_consistency(result, show_error=False)
    """
    from ._core import ensure_ax, extract_plot_data, style_context
    from ._styles import COLORS

    # Required keys for V03 duration consistency plot
    required_keys = [
        "observed_hours",
        "expected_hours",
        "expected_hours_err",
        "duration_ratio",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    # Get colors from styles if not provided
    if observed_color is None:
        observed_color = COLORS["transit"]
    if expected_color is None:
        expected_color = COLORS["model"]

    with style_context(style):
        fig, ax = ensure_ax(ax)

        observed = data["observed_hours"]
        expected = data["expected_hours"]
        expected_err = data["expected_hours_err"]
        ratio = data["duration_ratio"]

        # Set default bar kwargs
        bar_defaults: dict[str, Any] = {
            "width": 0.6,
            "edgecolor": "black",
            "linewidth": 1,
        }
        bar_defaults.update(bar_kwargs)
        bar_kw: Any = bar_defaults

        # Plot bars
        x_positions = [0, 1]
        bar_values = [observed, expected]
        bar_colors = [observed_color, expected_color]
        bar_labels = ["Observed", "Expected"]

        bars = []
        for x, val, color, label in zip(
            x_positions, bar_values, bar_colors, bar_labels, strict=True
        ):
            bar = ax.bar(
                x,
                val,
                color=color,
                label=label,
                **bar_kw,
            )
            bars.append(bar)

        # Add error bar for expected duration if requested
        if show_error and expected_err > 0:
            ax.errorbar(
                1,
                expected,
                yerr=expected_err,
                fmt="none",
                color="black",
                capsize=5,
                capthick=1.5,
            )

        # Annotate ratio if requested
        if annotate_ratio:
            ax.text(
                0.5,
                0.95,
                f"Ratio: {ratio:.2f}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
            )

        # Set labels
        ax.set_ylabel("Duration (hours)")
        ax.set_title("Duration Consistency")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(bar_labels)

        # Add legend if requested
        if show_legend:
            ax.legend()

    return ax


def plot_depth_stability(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    data_color: str | None = None,
    mean_color: str | None = None,
    outlier_color: str | None = None,
    show_legend: bool = True,
    show_mean: bool = True,
    show_scatter_band: bool = True,
    highlight_dominating: bool = True,
    style: str = "default",
    **errorbar_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot per-epoch transit depths with error bars.

    Creates a scatter plot showing transit depth measurements for individual
    epochs with error bars. Horizontal lines indicate the mean depth and
    expected scatter band.

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V04 (depth stability) check. Must contain
        plot_data with epoch_times_btjd, depths_ppm, and depth_errs_ppm.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    data_color : str, optional
        Color for data points. Defaults to COLORS["transit"] (blue).
    mean_color : str, optional
        Color for mean depth line. Defaults to COLORS["model"] (orange).
    outlier_color : str, optional
        Color for dominating epoch marker. Defaults to COLORS["outlier"] (red).
    show_legend : bool, default=True
        Whether to show the legend.
    show_mean : bool, default=True
        Whether to show the mean depth horizontal line.
    show_scatter_band : bool, default=True
        Whether to show the expected scatter band around the mean.
    highlight_dominating : bool, default=True
        Whether to highlight the dominating epoch with a different marker.
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
    >>> from bittr_tess_vetter.plotting import plot_depth_stability
    >>> ax = plot_depth_stability(result)  # Basic plot

    >>> # Without scatter band
    >>> ax = plot_depth_stability(result, show_scatter_band=False)
    """
    from ._core import ensure_ax, extract_plot_data, style_context
    from ._styles import COLORS

    # Required keys for V04 depth stability plot
    required_keys = [
        "epoch_times_btjd",
        "depths_ppm",
        "depth_errs_ppm",
        "mean_depth_ppm",
        "expected_scatter_ppm",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    # Get colors from styles if not provided
    if data_color is None:
        data_color = COLORS["transit"]
    if mean_color is None:
        mean_color = COLORS["model"]
    if outlier_color is None:
        outlier_color = COLORS["outlier"]

    with style_context(style):
        fig, ax = ensure_ax(ax)

        times = data["epoch_times_btjd"]
        depths = data["depths_ppm"]
        depth_errs = data["depth_errs_ppm"]
        mean_depth = data["mean_depth_ppm"]
        expected_scatter = data["expected_scatter_ppm"]
        dominating_idx = data.get("dominating_epoch_idx")

        # Set default errorbar kwargs
        errorbar_defaults: dict[str, Any] = {
            "fmt": "o",
            "capsize": 3,
            "capthick": 1,
            "markersize": 6,
        }
        errorbar_defaults.update(errorbar_kwargs)
        eb_kw: Any = errorbar_defaults

        # Plot per-epoch depths
        if len(times) > 0:
            ax.errorbar(
                times,
                depths,
                yerr=depth_errs,
                color=data_color,
                label="Per-epoch depth",
                **eb_kw,
            )

            # Highlight dominating epoch if requested
            if highlight_dominating and dominating_idx is not None and dominating_idx < len(times):
                ax.scatter(
                    [times[dominating_idx]],
                    [depths[dominating_idx]],
                    s=100,
                    marker="*",
                    c=outlier_color,
                    zorder=5,
                    label="Dominating epoch",
                )

        # Add mean line if requested
        if show_mean and mean_depth > 0:
            ax.axhline(
                mean_depth,
                color=mean_color,
                linestyle="--",
                linewidth=1.5,
                label=f"Mean: {mean_depth:.0f} ppm",
            )

            # Add expected scatter band if requested
            if show_scatter_band and expected_scatter > 0:
                ax.axhspan(
                    mean_depth - expected_scatter,
                    mean_depth + expected_scatter,
                    alpha=0.2,
                    color=mean_color,
                    label="Expected scatter",
                )

        # Set labels
        ax.set_xlabel("Time (BTJD)")
        ax.set_ylabel("Depth (ppm)")
        ax.set_title("Per-Epoch Depth Stability")

        # Add legend if requested
        if show_legend:
            ax.legend(loc="best")

    return ax


def plot_v_shape(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    data_color: str | None = None,
    model_color: str | None = None,
    show_legend: bool = True,
    show_model: bool = True,
    annotate_ratio: bool = True,
    style: str = "default",
    **errorbar_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot binned transit data with trapezoid model overlay.

    Creates a phase-folded plot of binned in-transit data with the best-fit
    trapezoid model overlaid. This visualization helps distinguish U-shaped
    (planetary) transits from V-shaped (grazing EB) transits.

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V05 (V-shape) check. Must contain plot_data
        with binned phase/flux and trapezoid model arrays.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    data_color : str, optional
        Color for binned data points. Defaults to COLORS["transit"] (blue).
    model_color : str, optional
        Color for trapezoid model. Defaults to COLORS["model"] (orange).
    show_legend : bool, default=True
        Whether to show the legend.
    show_model : bool, default=True
        Whether to show the trapezoid model overlay.
    annotate_ratio : bool, default=True
        Whether to annotate the tF/tT ratio on the plot.
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
    >>> from bittr_tess_vetter.plotting import plot_v_shape
    >>> ax = plot_v_shape(result)  # Basic plot

    >>> # Without model overlay
    >>> ax = plot_v_shape(result, show_model=False)
    """
    from ._core import ensure_ax, extract_plot_data, style_context
    from ._styles import COLORS

    # Required keys for V05 V-shape plot
    required_keys = [
        "binned_phase",
        "binned_flux",
        "binned_flux_err",
        "trapezoid_phase",
        "trapezoid_flux",
        "t_flat_hours",
        "t_total_hours",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    # Get colors from styles if not provided
    if data_color is None:
        data_color = COLORS["transit"]
    if model_color is None:
        model_color = COLORS["model"]

    with style_context(style):
        fig, ax = ensure_ax(ax)

        binned_phase = data["binned_phase"]
        binned_flux = data["binned_flux"]
        binned_flux_err = data["binned_flux_err"]
        model_phase = data["trapezoid_phase"]
        model_flux = data["trapezoid_flux"]
        t_flat = data["t_flat_hours"]
        t_total = data["t_total_hours"]

        # Set default errorbar kwargs
        errorbar_defaults: dict[str, Any] = {
            "fmt": "o",
            "capsize": 3,
            "capthick": 1,
            "markersize": 5,
        }
        errorbar_defaults.update(errorbar_kwargs)
        eb_kw: Any = errorbar_defaults

        # Plot binned data
        if len(binned_phase) > 0:
            ax.errorbar(
                binned_phase,
                binned_flux,
                yerr=binned_flux_err,
                color=data_color,
                label="Binned data",
                **eb_kw,
            )

        # Plot trapezoid model if requested
        if show_model and len(model_phase) > 0:
            ax.plot(
                model_phase,
                model_flux,
                color=model_color,
                linewidth=2,
                label="Trapezoid model",
            )

        # Annotate tF/tT ratio if requested
        if annotate_ratio and t_total > 0:
            ratio = t_flat / t_total
            ax.text(
                0.95,
                0.05,
                f"$t_F/t_T$ = {ratio:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

        # Set labels
        ax.set_xlabel("Phase")
        ax.set_ylabel("Normalized Flux")
        ax.set_title("Transit Shape Analysis")

        # Add legend if requested
        if show_legend:
            ax.legend(loc="upper right")

    return ax
