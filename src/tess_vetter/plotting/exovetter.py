"""Plot functions for exovetter-based checks (V11-V12)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.axes

    from tess_vetter.validation.result_schema import CheckResult


def plot_modshift(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    data_color: str = "#1f77b4",  # Blue
    primary_color: str = "#d62728",  # Red
    secondary_color: str = "#ff7f0e",  # Orange
    show_legend: bool = True,
    show_peaks: bool = True,
    annotate_values: bool = True,
    style: str = "default",
    **plot_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot ModShift periodogram with primary and secondary peaks marked.

    Creates a phase-binned periodogram showing the ModShift signal with
    markers indicating the primary and secondary eclipse locations.

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V11 (ModShift) check. Must contain
        plot_data with phase_bins, periodogram, and peak locations.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    data_color : str, default="#1f77b4"
        Color for the periodogram line.
    primary_color : str, default="#d62728"
        Color for primary peak marker.
    secondary_color : str, default="#ff7f0e"
        Color for secondary peak marker.
    show_legend : bool, default=True
        Whether to show the legend.
    show_peaks : bool, default=True
        Whether to show peak markers.
    annotate_values : bool, default=True
        Whether to annotate peak values on the plot.
    style : str, default="default"
        Style preset: "default", "paper", or "presentation".
    **plot_kwargs : Any
        Additional keyword arguments passed to ax.plot().

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
    >>> from tess_vetter.plotting import plot_modshift
    >>> ax = plot_modshift(result)  # Basic plot

    >>> # Custom colors
    >>> ax = plot_modshift(
    ...     result,
    ...     primary_color="green",
    ...     secondary_color="purple",
    ... )

    Notes
    -----
    Expected plot_data structure:
    ```python
    plot_data = {
        "version": 1,
        "phase_bins": [...],  # ~200 phase bins
        "periodogram": [...],  # ModShift signal values
        "primary_phase": float,
        "secondary_phase": float | None,
        "primary_signal": float,
        "secondary_signal": float | None,
    }
    ```
    """
    import numpy as np

    from ._core import ensure_ax, extract_plot_data, style_context

    # Required keys for V11 ModShift plot
    required_keys = [
        "phase_bins",
        "periodogram",
        "primary_phase",
        "primary_signal",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    phase_bins = np.array(data["phase_bins"], dtype=np.float64)
    periodogram = np.array(data["periodogram"], dtype=np.float64)
    primary_phase = data["primary_phase"]
    primary_signal = data["primary_signal"]
    secondary_phase = data.get("secondary_phase")
    secondary_signal = data.get("secondary_signal")

    with style_context(style):
        fig, ax = ensure_ax(ax)

        # Set default plot kwargs
        plot_defaults: dict[str, Any] = {
            "linewidth": 1.5,
        }
        plot_defaults.update(plot_kwargs)

        # Plot periodogram
        if len(phase_bins) > 0 and len(periodogram) > 0:
            ax.plot(
                phase_bins,
                periodogram,
                color=data_color,
                label="ModShift signal",
                **plot_defaults,
            )

        # Add zero reference line
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        # Mark peaks if requested
        if show_peaks:
            # Primary peak
            ax.axvline(
                primary_phase,
                color=primary_color,
                linestyle="-",
                alpha=0.7,
                linewidth=2,
                label=f"Primary (phase={primary_phase:.3f})",
            )

            # Annotate primary value
            if annotate_values:
                ax.scatter(
                    [primary_phase],
                    [primary_signal],
                    c=primary_color,
                    s=100,
                    marker="v",
                    zorder=5,
                    edgecolors="black",
                    linewidths=0.5,
                )
                ax.annotate(
                    f"{primary_signal:.3f}",
                    (primary_phase, primary_signal),
                    textcoords="offset points",
                    xytext=(10, 10),
                    fontsize=8,
                    color=primary_color,
                )

            # Secondary peak (if exists)
            if secondary_phase is not None:
                ax.axvline(
                    secondary_phase,
                    color=secondary_color,
                    linestyle="-",
                    alpha=0.7,
                    linewidth=2,
                    label=f"Secondary (phase={secondary_phase:.3f})",
                )

                if annotate_values and secondary_signal is not None:
                    ax.scatter(
                        [secondary_phase],
                        [secondary_signal],
                        c=secondary_color,
                        s=100,
                        marker="v",
                        zorder=5,
                        edgecolors="black",
                        linewidths=0.5,
                    )
                    ax.annotate(
                        f"{secondary_signal:.3f}",
                        (secondary_phase, secondary_signal),
                        textcoords="offset points",
                        xytext=(10, 10),
                        fontsize=8,
                        color=secondary_color,
                    )

        # Set labels
        ax.set_xlabel("Phase")
        ax.set_ylabel("ModShift Signal")
        ax.set_title("ModShift Analysis")

        # Set x-axis limits
        ax.set_xlim(0, 1)

        # Add legend if requested
        if show_legend:
            ax.legend(loc="upper right", fontsize="small")

    return ax


def plot_sweet(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    data_color: str = "#7f7f7f",  # Gray
    half_period_color: str = "#d62728",  # Red
    at_period_color: str = "#2ca02c",  # Green
    double_period_color: str = "#1f77b4",  # Blue
    show_legend: bool = True,
    show_fits: bool = True,
    style: str = "default",
    **scatter_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot SWEET analysis with sinusoidal fits overlay.

    Creates a phase-folded light curve showing out-of-transit data with
    optional sinusoidal fits at P/2, P, and 2P overlaid.

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V12 (SWEET) check. Must contain
        plot_data with phase, flux, and optional sinusoid fits.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    data_color : str, default="#7f7f7f"
        Color for data points.
    half_period_color : str, default="#d62728"
        Color for P/2 fit line.
    at_period_color : str, default="#2ca02c"
        Color for P fit line.
    double_period_color : str, default="#1f77b4"
        Color for 2P fit line.
    show_legend : bool, default=True
        Whether to show the legend.
    show_fits : bool, default=True
        Whether to show the sinusoidal fit lines.
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
    >>> from tess_vetter.plotting import plot_sweet
    >>> ax = plot_sweet(result)  # Basic plot

    >>> # Without fit lines
    >>> ax = plot_sweet(result, show_fits=False)

    Notes
    -----
    Expected plot_data structure:
    ```python
    plot_data = {
        "version": 1,
        "phase": [...],
        "flux": [...],
        "half_period_fit": [...] | None,  # Fit at P/2
        "at_period_fit": [...] | None,    # Fit at P
        "double_period_fit": [...] | None, # Fit at 2P
    }
    ```

    SWEET (Sine Wave Evaluation for Ephemeris Transits) fits sinusoids at
    P/2, P, and 2P to detect stellar variability that might be confused
    with transit signals.
    """
    import numpy as np

    from ._core import ensure_ax, extract_plot_data, style_context

    # Required keys for V12 SWEET plot
    required_keys = [
        "phase",
        "flux",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    phase = np.array(data["phase"], dtype=np.float64)
    flux = np.array(data["flux"], dtype=np.float64)

    # Optional fit arrays
    half_period_fit = data.get("half_period_fit")
    at_period_fit = data.get("at_period_fit")
    double_period_fit = data.get("double_period_fit")

    # Optional SNR values for legend
    snr_half = data.get("snr_half_period")
    snr_at = data.get("snr_at_period")
    snr_double = data.get("snr_double_period")

    with style_context(style):
        fig, ax = ensure_ax(ax)

        # Set default scatter kwargs
        scatter_defaults: dict[str, Any] = {
            "s": 4,
            "alpha": 0.4,
        }
        scatter_defaults.update(scatter_kwargs)

        # Plot data points
        if len(phase) > 0 and len(flux) > 0:
            ax.scatter(
                phase,
                flux,
                c=data_color,
                label="Data",
                **scatter_defaults,
            )

        # Plot sinusoidal fits if requested and available
        if show_fits:
            # Sort phase for clean line plots
            sort_idx = np.argsort(phase)
            phase_sorted = phase[sort_idx]

            if half_period_fit is not None:
                fit_arr = np.array(half_period_fit, dtype=np.float64)
                if len(fit_arr) == len(phase):
                    label = "P/2 fit"
                    if snr_half is not None:
                        label += f" (SNR={snr_half:.1f})"
                    ax.plot(
                        phase_sorted,
                        fit_arr[sort_idx],
                        color=half_period_color,
                        linewidth=1.5,
                        label=label,
                    )

            if at_period_fit is not None:
                fit_arr = np.array(at_period_fit, dtype=np.float64)
                if len(fit_arr) == len(phase):
                    label = "P fit"
                    if snr_at is not None:
                        label += f" (SNR={snr_at:.1f})"
                    ax.plot(
                        phase_sorted,
                        fit_arr[sort_idx],
                        color=at_period_color,
                        linewidth=1.5,
                        label=label,
                    )

            if double_period_fit is not None:
                fit_arr = np.array(double_period_fit, dtype=np.float64)
                if len(fit_arr) == len(phase):
                    label = "2P fit"
                    if snr_double is not None:
                        label += f" (SNR={snr_double:.1f})"
                    ax.plot(
                        phase_sorted,
                        fit_arr[sort_idx],
                        color=double_period_color,
                        linewidth=1.5,
                        label=label,
                    )

        # Add horizontal reference line at 1.0
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        # Set labels
        ax.set_xlabel("Phase")
        ax.set_ylabel("Normalized Flux")
        ax.set_title("SWEET Analysis")

        # Set x-axis limits
        ax.set_xlim(0, 1)

        # Add legend if requested
        if show_legend:
            ax.legend(loc="upper right", fontsize="small")

    return ax
