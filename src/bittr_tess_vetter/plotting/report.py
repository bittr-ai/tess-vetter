"""DVR-style vetting summary report.

This module provides functions to generate and save complete vetting summary
reports in the style of Kepler DVR (Data Validation Report) summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from bittr_tess_vetter.api.types import Candidate, LightCurve
    from bittr_tess_vetter.validation.result_schema import VettingBundleResult


def plot_vetting_summary(
    bundle: VettingBundleResult,
    lc: LightCurve,
    candidate: Candidate,
    *,
    figsize: tuple[float, float] = (11, 8.5),
    include_panels: list[str] | None = None,
    title: str | None = None,
    style: str = "default",
) -> matplotlib.figure.Figure:
    """Generate DVR-style one-page vetting summary.

    Creates an 8-panel figure following Kepler DVS format:

    Layout (3x3 grid with H spanning bottom right):
    +-------+-------+-------+
    |   A   |   B   |   C   |
    +-------+-------+-------+
    |   D   |   E   |   F   |
    +-------+-------+-------+
    |   G   |       H       |
    +-------+-------+-------+

    Panels:
    - A: Full light curve (plot_full_lightcurve)
    - B: Phase-folded transit (plot_phase_folded)
    - C: Secondary eclipse (plot_secondary_eclipse from V02)
    - D: Odd-even comparison (plot_odd_even from V01)
    - E: V-shape transit (plot_v_shape from V05)
    - F: Centroid shift (plot_centroid_shift from V08, if available)
    - G: Depth stability (plot_depth_stability from V04)
    - H: Metrics summary table

    Parameters
    ----------
    bundle : VettingBundleResult
        Complete vetting results from vet_candidate()
    lc : LightCurve
        Light curve data
    candidate : Candidate
        Transit candidate
    figsize : tuple
        Figure size in inches (default: letter landscape)
    include_panels : list[str], optional
        Panel IDs to include ["A", "B", "C", "D", "E", "F", "G", "H"]
        Default: all panels
    title : str, optional
        Custom title. Default: "Vetting Summary"
    style : str
        Style preset: "default", "paper", or "presentation"

    Returns
    -------
    Figure
        Matplotlib figure with all panels

    Raises
    ------
    ValueError
        If style is not recognized.

    Examples
    --------
    >>> from bittr_tess_vetter.plotting import plot_vetting_summary
    >>> fig = plot_vetting_summary(bundle, lc, candidate)

    >>> # Include only specific panels
    >>> fig = plot_vetting_summary(
    ...     bundle, lc, candidate,
    ...     include_panels=["A", "D", "H"],
    ... )

    >>> # Publication-ready figure
    >>> fig = plot_vetting_summary(
    ...     bundle, lc, candidate,
    ...     style="paper",
    ...     title="TIC 123456789",
    ... )
    """
    import matplotlib.pyplot as plt

    from ._core import style_context
    from .checks import (
        plot_depth_stability,
        plot_odd_even,
        plot_secondary_eclipse,
        plot_v_shape,
    )
    from .pixel import plot_centroid_shift

    # Default to all panels
    if include_panels is None:
        include_panels = ["A", "B", "C", "D", "E", "F", "G", "H"]

    # Validate include_panels
    valid_panels = {"A", "B", "C", "D", "E", "F", "G", "H"}
    invalid = set(include_panels) - valid_panels
    if invalid:
        raise ValueError(f"Invalid panel IDs: {invalid}. Valid: {sorted(valid_panels)}")

    with style_context(style):
        # Create figure with GridSpec (3 rows, 3 cols)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # Panel mapping to GridSpec positions
        panel_positions = {
            "A": gs[0, 0],
            "B": gs[0, 1],
            "C": gs[0, 2],
            "D": gs[1, 0],
            "E": gs[1, 1],
            "F": gs[1, 2],
            "G": gs[2, 0],
            "H": gs[2, 1:3],  # H spans columns 1-2
        }

        # Panel A: Full light curve
        if "A" in include_panels:
            ax_a = fig.add_subplot(panel_positions["A"])
            _plot_full_lightcurve_panel(ax_a, lc, candidate)

        # Panel B: Phase-folded transit
        if "B" in include_panels:
            ax_b = fig.add_subplot(panel_positions["B"])
            _plot_phase_folded_panel(ax_b, lc, candidate)

        # Panel C: Secondary eclipse (V02)
        if "C" in include_panels:
            ax_c = fig.add_subplot(panel_positions["C"])
            v02_result = bundle.get_result("V02")
            if v02_result is not None and v02_result.raw and v02_result.raw.get("plot_data"):
                plot_secondary_eclipse(
                    v02_result,
                    ax=ax_c,
                    show_legend=False,
                    style=style,
                )
            else:
                _render_empty_panel(ax_c, "Secondary Eclipse\n(V02 not available)")

        # Panel D: Odd-even comparison (V01)
        if "D" in include_panels:
            ax_d = fig.add_subplot(panel_positions["D"])
            v01_result = bundle.get_result("V01")
            if v01_result is not None and v01_result.raw and v01_result.raw.get("plot_data"):
                plot_odd_even(
                    v01_result,
                    ax=ax_d,
                    show_legend=False,
                    style=style,
                )
            else:
                _render_empty_panel(ax_d, "Odd/Even Comparison\n(V01 not available)")

        # Panel E: V-shape transit (V05)
        if "E" in include_panels:
            ax_e = fig.add_subplot(panel_positions["E"])
            v05_result = bundle.get_result("V05")
            if v05_result is not None and v05_result.raw and v05_result.raw.get("plot_data"):
                plot_v_shape(
                    v05_result,
                    ax=ax_e,
                    show_legend=False,
                    style=style,
                )
            else:
                _render_empty_panel(ax_e, "Transit Shape\n(V05 not available)")

        # Panel F: Centroid shift (V08)
        if "F" in include_panels:
            ax_f = fig.add_subplot(panel_positions["F"])
            v08_result = bundle.get_result("V08")
            if v08_result is not None and v08_result.raw and v08_result.raw.get("plot_data"):
                plot_centroid_shift(
                    v08_result,
                    ax=ax_f,
                    show_colorbar=False,
                    style=style,
                )
            else:
                _render_empty_panel(ax_f, "Centroid Shift\n(V08 not available)")

        # Panel G: Depth stability (V04)
        if "G" in include_panels:
            ax_g = fig.add_subplot(panel_positions["G"])
            v04_result = bundle.get_result("V04")
            if v04_result is not None and v04_result.raw and v04_result.raw.get("plot_data"):
                plot_depth_stability(
                    v04_result,
                    ax=ax_g,
                    show_legend=False,
                    style=style,
                )
            else:
                _render_empty_panel(ax_g, "Depth Stability\n(V04 not available)")

        # Panel H: Metrics summary table
        if "H" in include_panels:
            ax_h = fig.add_subplot(panel_positions["H"])
            _render_metrics_table(ax_h, bundle, candidate)

        # Add main title
        report_title = title if title else "Vetting Summary"
        fig.suptitle(report_title, fontsize=14, fontweight="bold", y=0.98)

    return fig


def _plot_full_lightcurve_panel(
    ax: matplotlib.axes.Axes,
    lc: LightCurve,
    candidate: Candidate,
) -> None:
    """Plot full light curve in panel A.

    Shows the complete light curve with transit markers.
    """
    import numpy as np

    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)

    # Apply valid mask if available
    if lc.valid_mask is not None:
        mask = np.asarray(lc.valid_mask, dtype=bool)
        time = time[mask]
        flux = flux[mask]

    # Plot light curve
    ax.scatter(time, flux, s=1, c="gray", alpha=0.5, rasterized=True)

    # Mark transit times
    period = candidate.ephemeris.period_days
    t0 = candidate.ephemeris.t0_btjd

    if len(time) > 0:
        n_transits_before = int(np.floor((time.min() - t0) / period))
        n_transits_after = int(np.ceil((time.max() - t0) / period))

        for n in range(n_transits_before, n_transits_after + 1):
            t_transit = t0 + n * period
            if time.min() <= t_transit <= time.max():
                ax.axvline(t_transit, color="red", alpha=0.3, linewidth=0.5)

    ax.set_xlabel("Time (BTJD)")
    ax.set_ylabel("Flux")
    ax.set_title("Light Curve")


def _plot_phase_folded_panel(
    ax: matplotlib.axes.Axes,
    lc: LightCurve,
    candidate: Candidate,
) -> None:
    """Plot phase-folded transit in panel B.

    Shows the phase-folded light curve centered on transit.
    """
    import numpy as np

    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)

    # Apply valid mask if available
    if lc.valid_mask is not None:
        mask = np.asarray(lc.valid_mask, dtype=bool)
        time = time[mask]
        flux = flux[mask]

    # Compute phase
    period = candidate.ephemeris.period_days
    t0 = candidate.ephemeris.t0_btjd
    phase = ((time - t0) % period) / period

    # Shift to center transit at phase 0.5
    phase = np.where(phase > 0.5, phase - 1.0, phase)

    # Plot
    ax.scatter(phase, flux, s=1, c="gray", alpha=0.5, rasterized=True)

    # Mark transit window
    duration_phase = (candidate.ephemeris.duration_hours / 24.0) / period
    ax.axvspan(-duration_phase / 2, duration_phase / 2, alpha=0.2, color="blue")

    ax.set_xlabel("Phase")
    ax.set_ylabel("Flux")
    ax.set_title("Phase-Folded Transit")
    ax.set_xlim(-0.1, 0.1)


def _render_empty_panel(ax: matplotlib.axes.Axes, message: str) -> None:
    """Render an empty panel with a message.

    Used when a check result is not available.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=10,
        color="gray",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def _render_metrics_table(
    ax: matplotlib.axes.Axes,
    bundle: VettingBundleResult,
    candidate: Candidate,
) -> None:
    """Render summary metrics table in axes.

    Shows key metrics from vetting results including ephemeris,
    check summary, and key flags.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to render the table in.
    bundle : VettingBundleResult
        Complete vetting results.
    candidate : Candidate
        Transit candidate with ephemeris.
    """
    # Turn off axes
    ax.axis("off")

    # Build metrics text
    lines: list[str] = []

    # Ephemeris section
    lines.append("EPHEMERIS")
    lines.append(f"  Period: {candidate.ephemeris.period_days:.6f} days")
    lines.append(f"  T0: {candidate.ephemeris.t0_btjd:.4f} BTJD")
    lines.append(f"  Duration: {candidate.ephemeris.duration_hours:.2f} hours")

    if candidate.depth_ppm is not None:
        lines.append(f"  Depth: {candidate.depth_ppm:.0f} ppm")
    elif candidate.depth_fraction is not None:
        lines.append(f"  Depth: {candidate.depth_fraction * 1e6:.0f} ppm")

    lines.append("")

    # Check summary section
    lines.append("CHECK SUMMARY")
    lines.append(f"  Passed: {bundle.n_passed}")
    lines.append(f"  Failed: {bundle.n_failed}")
    lines.append(f"  Skipped: {bundle.n_unknown}")

    if bundle.failed_check_ids:
        lines.append(f"  Failed IDs: {', '.join(bundle.failed_check_ids)}")

    lines.append("")

    # Key metrics from individual checks
    lines.append("KEY METRICS")

    # V01: Odd/even sigma
    v01 = bundle.get_result("V01")
    if v01 is not None and v01.metrics.get("sigma_diff") is not None:
        lines.append(f"  Odd/Even sigma: {v01.metrics['sigma_diff']:.2f}")

    # V02: Secondary depth
    v02 = bundle.get_result("V02")
    if v02 is not None and v02.metrics.get("secondary_depth_ppm") is not None:
        lines.append(f"  Secondary: {v02.metrics['secondary_depth_ppm']:.0f} ppm")

    # V04: Chi2
    v04 = bundle.get_result("V04")
    if v04 is not None and v04.metrics.get("chi2_reduced") is not None:
        lines.append(f"  Depth chi2_red: {v04.metrics['chi2_reduced']:.2f}")

    # V05: V-shape ratio
    v05 = bundle.get_result("V05")
    if v05 is not None and v05.metrics.get("tflat_ttotal_ratio") is not None:
        lines.append(f"  tF/tT ratio: {v05.metrics['tflat_ttotal_ratio']:.2f}")

    # Render text
    text_content = "\n".join(lines)
    ax.text(
        0.05,
        0.95,
        text_content,
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        verticalalignment="top",
        horizontalalignment="left",
    )

    ax.set_title("Metrics Summary")


def save_vetting_report(
    bundle: VettingBundleResult,
    lc: LightCurve,
    candidate: Candidate,
    path: str | Path,
    *,
    format: str = "pdf",
    dpi: int = 300,
    **summary_kwargs: Any,
) -> Path:
    """Save vetting summary to file.

    Parameters
    ----------
    bundle : VettingBundleResult
        Complete vetting results from vet_candidate()
    lc : LightCurve
        Light curve data
    candidate : Candidate
        Transit candidate
    path : str or Path
        Output file path
    format : str
        Output format: "pdf", "png", "svg"
    dpi : int
        Resolution for raster formats
    **summary_kwargs
        Passed to plot_vetting_summary()

    Returns
    -------
    Path
        Path to saved file

    Examples
    --------
    >>> from bittr_tess_vetter.plotting import save_vetting_report
    >>> path = save_vetting_report(bundle, lc, candidate, "report.pdf")

    >>> # Save as PNG with high DPI
    >>> path = save_vetting_report(
    ...     bundle, lc, candidate, "report.png",
    ...     format="png", dpi=300,
    ... )
    """
    import matplotlib.pyplot as plt

    # Generate the figure
    fig = plot_vetting_summary(bundle, lc, candidate, **summary_kwargs)

    # Ensure path is a Path object
    output_path = Path(path)

    # Save with tight bounding box
    fig.savefig(output_path, format=format, dpi=dpi, bbox_inches="tight")

    # Close the figure to free memory
    plt.close(fig)

    return output_path
