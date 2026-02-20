"""Transit visualization functions.

This module provides visualization functions for transit data that work
directly with LightCurve and Candidate objects, rather than CheckResult.
These are useful for general transit analysis and diagnostics.

Functions:
    plot_phase_folded: Phase-folded light curve centered on transit
    plot_transit_fit: Transit model fit with data overlay
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes

    from tess_vetter.api.transit_fit import TransitFitResult
    from tess_vetter.api.types import Candidate, LightCurve


def plot_phase_folded(
    lc: LightCurve,
    candidate: Candidate,
    *,
    ax: matplotlib.axes.Axes | None = None,
    fit_result: TransitFitResult | None = None,
    bin_minutes: float | None = 30.0,
    data_color: str | None = None,
    binned_color: str | None = None,
    model_color: str | None = None,
    show_model: bool = True,
    show_binned: bool = True,
    phase_range: tuple[float, float] = (-0.15, 0.15),
    style: str = "default",
    **scatter_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot phase-folded light curve centered on transit.

    Creates a phase-folded scatter plot of the light curve, centered on the
    transit midpoint. This is the most fundamental transit visualization,
    useful for assessing transit depth, duration, and shape.

    Parameters
    ----------
    lc : LightCurve
        Light curve data with time, flux, and optional flux_err arrays.
    candidate : Candidate
        Transit candidate with ephemeris (period, t0, duration).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    fit_result : TransitFitResult, optional
        Transit fit result containing model curve. If provided and show_model
        is True, overlays the model on the data.
    bin_minutes : float, optional
        Bin size in minutes for binned data points. If None, no binning.
        Default is 30 minutes.
    data_color : str, optional
        Color for raw data points. Defaults to COLORS["out_of_transit"].
    binned_color : str, optional
        Color for binned data points. Defaults to COLORS["transit"].
    model_color : str, optional
        Color for model curve. Defaults to COLORS["model"].
    show_model : bool, default=True
        Whether to show the model curve (requires fit_result).
    show_binned : bool, default=True
        Whether to show binned data points.
    phase_range : tuple[float, float], default=(-0.15, 0.15)
        Phase range to display, centered on transit (phase 0).
    style : str, default="default"
        Style preset: "default", "paper", or "presentation".
    **scatter_kwargs : Any
        Additional keyword arguments passed to ax.scatter() for raw data.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Examples
    --------
    >>> from tess_vetter.plotting import plot_phase_folded
    >>> ax = plot_phase_folded(lc, candidate)  # Basic plot

    >>> # With model overlay from fit result
    >>> ax = plot_phase_folded(lc, candidate, fit_result=fit)

    >>> # Finer binning for high-cadence data
    >>> ax = plot_phase_folded(lc, candidate, bin_minutes=10.0)

    >>> # Wider phase range for long-period planets
    >>> ax = plot_phase_folded(lc, candidate, phase_range=(-0.3, 0.3))
    """
    from ._core import ensure_ax, style_context
    from ._styles import COLORS, LABELS

    # Get colors from styles if not provided
    if data_color is None:
        data_color = COLORS["out_of_transit"]
    if binned_color is None:
        binned_color = COLORS["transit"]
    if model_color is None:
        model_color = COLORS["model"]

    # Extract ephemeris
    period = candidate.ephemeris.period_days
    t0 = candidate.ephemeris.t0_btjd

    # Get time and flux arrays
    time = np.asarray(lc.time)
    flux = np.asarray(lc.flux)

    # Apply valid_mask if present
    if lc.valid_mask is not None:
        valid = np.asarray(lc.valid_mask, dtype=bool)
        time = time[valid]
        flux = flux[valid]

    # Compute phase: ((time - t0) / period) % 1, shifted to center on 0
    phase = ((time - t0) / period) % 1.0
    # Shift phase to center on 0 (transit midpoint)
    phase = np.where(phase > 0.5, phase - 1.0, phase)

    # Filter to phase range
    in_range = (phase >= phase_range[0]) & (phase <= phase_range[1])
    phase_plot = phase[in_range]
    flux_plot = flux[in_range]

    with style_context(style):
        fig, ax = ensure_ax(ax)

        # Set default scatter kwargs for raw data
        scatter_defaults: dict[str, Any] = {
            "s": 4,
            "alpha": 0.3,
        }
        scatter_defaults.update(scatter_kwargs)

        # Plot raw data
        if len(phase_plot) > 0:
            ax.scatter(
                phase_plot,
                flux_plot,
                c=data_color,
                label="Data",
                **scatter_defaults,
            )

        # Bin data if requested
        if show_binned and bin_minutes is not None and len(phase_plot) > 0:
            binned_phase, binned_flux, binned_err = _bin_phase_data(
                phase_plot, flux_plot, period, bin_minutes
            )
            if len(binned_phase) > 0:
                ax.errorbar(
                    binned_phase,
                    binned_flux,
                    yerr=binned_err,
                    fmt="o",
                    color=binned_color,
                    markersize=6,
                    capsize=2,
                    label=f"Binned ({bin_minutes:.0f} min)",
                    zorder=5,
                )

        # Plot model if requested and available
        if show_model and fit_result is not None and len(fit_result.phase) > 0:
            model_phase = np.array(fit_result.phase)
            model_flux = np.array(fit_result.flux_model)

            # Filter model to phase range
            model_in_range = (model_phase >= phase_range[0]) & (
                model_phase <= phase_range[1]
            )
            if np.any(model_in_range):
                # Sort by phase for clean line plot
                sort_idx = np.argsort(model_phase[model_in_range])
                ax.plot(
                    model_phase[model_in_range][sort_idx],
                    model_flux[model_in_range][sort_idx],
                    color=model_color,
                    linewidth=2,
                    label="Model",
                    zorder=10,
                )

        # Set labels
        ax.set_xlabel(LABELS["phase"])
        ax.set_ylabel(LABELS["flux_normalized"])
        ax.set_title("Phase-Folded Transit")

        # Set x-axis limits
        ax.set_xlim(phase_range)

        # Add legend
        ax.legend(loc="lower right")

    return ax


def _bin_phase_data(
    phase: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    bin_minutes: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin phase-folded data by phase.

    Parameters
    ----------
    phase : np.ndarray
        Phase values (centered on 0).
    flux : np.ndarray
        Flux values.
    period_days : float
        Orbital period in days.
    bin_minutes : float
        Bin size in minutes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Binned phase, binned flux, and standard error of mean for each bin.
    """
    if len(phase) == 0:
        return np.array([]), np.array([]), np.array([])

    # Convert bin size from minutes to phase units
    bin_phase = (bin_minutes / 60.0 / 24.0) / period_days

    # Determine bin edges
    phase_min = np.min(phase)
    phase_max = np.max(phase)
    n_bins = max(1, int(np.ceil((phase_max - phase_min) / bin_phase)))
    bin_edges = np.linspace(phase_min, phase_max, n_bins + 1)

    # Bin the data
    bin_centers = []
    bin_fluxes = []
    bin_errors = []

    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if i == n_bins - 1:
            # Include right edge in last bin
            mask = (phase >= bin_edges[i]) & (phase <= bin_edges[i + 1])

        n_in_bin = np.sum(mask)
        if n_in_bin > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_fluxes.append(np.mean(flux[mask]))
            # Standard error of the mean
            if n_in_bin > 1:
                bin_errors.append(np.std(flux[mask], ddof=1) / np.sqrt(n_in_bin))
            else:
                bin_errors.append(0.0)

    return np.array(bin_centers), np.array(bin_fluxes), np.array(bin_errors)


def plot_transit_fit(
    fit_result: TransitFitResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    data_color: str | None = None,
    model_color: str | None = None,
    show_residuals: bool = False,
    style: str = "default",
    **scatter_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot transit model fit with data overlay.

    Creates a plot showing the fitted transit model alongside the observed
    data. Uses the phase, flux_data, and flux_model arrays from the
    TransitFitResult.

    Parameters
    ----------
    fit_result : TransitFitResult
        Result from fit_transit() containing phase, flux_data, and flux_model.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    data_color : str, optional
        Color for data points. Defaults to COLORS["transit"].
    model_color : str, optional
        Color for model curve. Defaults to COLORS["model"].
    show_residuals : bool, default=False
        Whether to show residuals below the main plot.
    style : str, default="default"
        Style preset: "default", "paper", or "presentation".
    **scatter_kwargs : Any
        Additional keyword arguments passed to ax.scatter().

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot (main axes if show_residuals=True).

    Raises
    ------
    ValueError
        If fit_result has no phase/flux data (e.g., status="error").

    Examples
    --------
    >>> from tess_vetter.plotting import plot_transit_fit
    >>> ax = plot_transit_fit(fit_result)  # Basic plot

    >>> # Show residuals
    >>> ax = plot_transit_fit(fit_result, show_residuals=True)
    """
    import matplotlib.pyplot as plt

    from ._core import ensure_ax, style_context
    from ._styles import COLORS, LABELS

    # Validate fit_result has data
    if len(fit_result.phase) == 0:
        raise ValueError(
            f"TransitFitResult has no phase data. "
            f"Status: {fit_result.status}, error: {fit_result.error_message}"
        )

    # Get colors from styles if not provided
    if data_color is None:
        data_color = COLORS["transit"]
    if model_color is None:
        model_color = COLORS["model"]

    phase = np.array(fit_result.phase)
    flux_data = np.array(fit_result.flux_data)
    flux_model = np.array(fit_result.flux_model)

    with style_context(style):
        if show_residuals:
            # Create figure with two subplots
            fig, (ax_main, ax_resid) = plt.subplots(
                2, 1, figsize=(8, 6), height_ratios=[3, 1], sharex=True
            )
            ax = ax_main
        else:
            _fig, ax = ensure_ax(ax)

        # Set default scatter kwargs
        scatter_defaults: dict[str, Any] = {
            "s": 10,
            "alpha": 0.6,
        }
        scatter_defaults.update(scatter_kwargs)

        # Sort by phase for proper line plot
        sort_idx = np.argsort(phase)
        phase_sorted = phase[sort_idx]
        flux_data_sorted = flux_data[sort_idx]
        flux_model_sorted = flux_model[sort_idx]

        # Plot data
        ax.scatter(
            phase_sorted,
            flux_data_sorted,
            c=data_color,
            label="Data",
            **scatter_defaults,
        )

        # Plot model
        ax.plot(
            phase_sorted,
            flux_model_sorted,
            color=model_color,
            linewidth=2,
            label="Model",
            zorder=10,
        )

        # Add fit info annotation
        ax.text(
            0.02,
            0.02,
            f"$R_p/R_*$ = {fit_result.rp_rs:.4f}\n"
            f"$a/R_*$ = {fit_result.a_rs:.2f}\n"
            f"$\\chi^2_\\nu$ = {fit_result.chi_squared:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        # Set labels
        ax.set_ylabel(LABELS["flux_normalized"])
        ax.set_title(f"Transit Fit ({fit_result.fit_method})")
        ax.legend(loc="lower right")

        if show_residuals:
            # Plot residuals
            residuals = flux_data_sorted - flux_model_sorted
            ax_resid.scatter(
                phase_sorted,
                residuals * 1e6,  # Convert to ppm
                c=data_color,
                **scatter_defaults,
            )
            ax_resid.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax_resid.set_xlabel(LABELS["phase"])
            ax_resid.set_ylabel("Residuals (ppm)")

            fig.tight_layout()
        else:
            ax.set_xlabel(LABELS["phase"])

    return ax
