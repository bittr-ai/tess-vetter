"""Plot functions for catalog-based vetting checks (V06-V07)."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.axes

    from bittr_tess_vetter.validation.result_schema import CheckResult


def plot_nearby_ebs(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    target_color: str = "#d62728",  # Red
    match_color: str = "#1f77b4",  # Blue
    show_legend: bool = True,
    annotate_separations: bool = True,
    marker_scale: float = 1.0,
    style: str = "default",
    **scatter_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot sky map of target and nearby eclipsing binaries from V06 check.

    Creates a scatter plot showing the target position at center with nearby
    eclipsing binaries from the TESS-EB catalog marked around it. The search
    radius is shown as a dashed circle.

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V06 (nearby EB search) check. Must contain
        plot_data with target_ra, target_dec, matches, and search_radius_arcsec.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    target_color : str, default="#d62728"
        Color for the target marker.
    match_color : str, default="#1f77b4"
        Color for nearby EB markers.
    show_legend : bool, default=True
        Whether to show the legend.
    annotate_separations : bool, default=True
        Whether to annotate EB markers with their separation in arcsec.
    marker_scale : float, default=1.0
        Scale factor for marker sizes.
    style : str, default="default"
        Style preset: "default", "paper", or "presentation".
    **scatter_kwargs : Any
        Additional keyword arguments passed to ax.scatter() for EB markers.

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
    >>> from bittr_tess_vetter.plotting import plot_nearby_ebs
    >>> ax = plot_nearby_ebs(result)  # Basic plot

    >>> # Custom marker colors
    >>> ax = plot_nearby_ebs(
    ...     result,
    ...     target_color="green",
    ...     match_color="orange",
    ... )

    Notes
    -----
    Expected plot_data structure:
    ```python
    plot_data = {
        "version": 1,
        "target_ra": float,
        "target_dec": float,
        "matches": [
            {"ra": float, "dec": float, "sep_arcsec": float, "period_days": float},
            ...
        ],
        "search_radius_arcsec": float,
    }
    ```
    """
    import numpy as np

    from ._core import ensure_ax, extract_plot_data, style_context

    # Required keys for V06 nearby EB plot
    required_keys = [
        "target_ra",
        "target_dec",
        "matches",
        "search_radius_arcsec",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    target_ra = data["target_ra"]
    target_dec = data["target_dec"]
    matches = data["matches"]
    search_radius = data["search_radius_arcsec"]

    with style_context(style):
        fig, ax = ensure_ax(ax)

        # Set default scatter kwargs
        scatter_defaults: dict[str, Any] = {
            "s": 80 * marker_scale,
            "alpha": 0.8,
            "edgecolors": "black",
            "linewidths": 0.5,
        }
        scatter_defaults.update(scatter_kwargs)

        # Plot target at center (use offset coordinates in arcsec)
        ax.scatter(
            [0],
            [0],
            c=target_color,
            s=150 * marker_scale,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            zorder=10,
            label="Target",
        )

        # Plot nearby EBs using offset from target in arcsec
        if matches:
            # Calculate offsets in arcsec
            eb_offsets_ra = []
            eb_offsets_dec = []
            eb_separations = []
            eb_periods = []

            for match in matches:
                ra = match.get("ra") or match.get("ra_deg")
                dec = match.get("dec") or match.get("dec_deg")
                sep = match.get("sep_arcsec")
                period = match.get("period_days")

                if ra is not None and dec is not None:
                    # Convert RA/Dec offset to arcsec (approximate for small offsets)
                    delta_ra = (ra - target_ra) * np.cos(np.radians(target_dec)) * 3600.0
                    delta_dec = (dec - target_dec) * 3600.0
                    eb_offsets_ra.append(delta_ra)
                    eb_offsets_dec.append(delta_dec)
                    eb_separations.append(sep if sep is not None else 0.0)
                    eb_periods.append(period if period is not None else float("nan"))

            if eb_offsets_ra:
                ax.scatter(
                    eb_offsets_ra,
                    eb_offsets_dec,
                    c=match_color,
                    marker="o",
                    label=f"Nearby EBs ({len(eb_offsets_ra)})",
                    **scatter_defaults,
                )

                # Annotate with separations if requested
                if annotate_separations:
                    for ra_off, dec_off, sep in zip(
                        eb_offsets_ra, eb_offsets_dec, eb_separations, strict=True
                    ):
                        if sep > 0:
                            ax.annotate(
                                f'{sep:.1f}"',
                                (ra_off, dec_off),
                                textcoords="offset points",
                                xytext=(5, 5),
                                fontsize=7,
                                alpha=0.8,
                            )

        # Draw search radius circle
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = search_radius * np.cos(theta)
        circle_y = search_radius * np.sin(theta)
        ax.plot(
            circle_x,
            circle_y,
            "--",
            color="gray",
            alpha=0.6,
            linewidth=1,
            label=f'Search radius ({search_radius:.0f}")',
        )

        # Set labels
        ax.set_xlabel('RA offset (arcsec)')
        ax.set_ylabel('Dec offset (arcsec)')
        ax.set_title("Nearby Eclipsing Binaries")

        # Set equal aspect ratio
        ax.set_aspect("equal")

        # Ensure search radius is visible
        margin = search_radius * 0.2
        ax.set_xlim(-search_radius - margin, search_radius + margin)
        ax.set_ylim(-search_radius - margin, search_radius + margin)

        # Add legend if requested
        if show_legend:
            ax.legend(loc="upper right", fontsize="small")

    return ax


def plot_exofop_card(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    show_title: bool = True,
    style: str = "default",
    **text_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot ExoFOP TOI disposition information as a text card.

    Creates a simple text-based visualization showing the ExoFOP TOI
    disposition information including TOI number, TFOPWG disposition,
    and any notes.

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V07 (ExoFOP TOI lookup) check. Must contain
        plot_data with tic_id, found status, and optional row data.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    show_title : bool, default=True
        Whether to show the plot title.
    style : str, default="default"
        Style preset: "default", "paper", or "presentation".
    **text_kwargs : Any
        Additional keyword arguments passed to ax.text().

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
    >>> from bittr_tess_vetter.plotting import plot_exofop_card
    >>> ax = plot_exofop_card(result)  # Basic card

    Notes
    -----
    Expected plot_data structure:
    ```python
    plot_data = {
        "version": 1,
        "tic_id": int,
        "found": bool,
        "toi": float | None,
        "tfopwg_disposition": str | None,
        "planet_disposition": str | None,
        "comments": str | None,
    }
    ```
    """
    from ._core import ensure_ax, extract_plot_data, style_context

    # Required keys for V07 ExoFOP card
    required_keys = [
        "tic_id",
        "found",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    tic_id = data["tic_id"]
    found = data["found"]

    def _wrap_notes(notes: str, *, style: str, max_lines: int) -> list[str]:
        # These widths are tuned for the default verification figsize (5x4) and a
        # monospace font; they’re conservative to avoid overflowing the card box.
        wrap_width_by_style = {
            "paper": 30,
            "default": 38,
            "presentation": 28,
        }
        # Account for the "Notes: " prefix and continuation indentation so we
        # don't overflow horizontally on typical card sizes.
        wrap_width = max(10, wrap_width_by_style.get(style, 38) - 8)

        wrapped = textwrap.wrap(
            notes,
            width=wrap_width,
            break_long_words=True,
            break_on_hyphens=True,
        )
        if not wrapped:
            return []

        if len(wrapped) > max_lines:
            wrapped = wrapped[:max_lines]
            if not wrapped[-1].endswith("..."):
                wrapped[-1] = wrapped[-1].rstrip(".") + "..."
        return wrapped

    with style_context(style):
        fig, ax = ensure_ax(ax)

        # Hide axes for text card
        ax.axis("off")

        # Build text content
        lines = []
        lines.append(f"TIC {tic_id}")
        lines.append("-" * 30)

        if not found:
            lines.append("Not found in ExoFOP TOI table")
            bg_color = "#f0f0f0"  # Light gray for not found
            status_color = "#666666"
        else:
            # Extract optional fields
            toi = data.get("toi")
            tfopwg = data.get("tfopwg_disposition")
            planet_disp = data.get("planet_disposition")
            comments = data.get("comments")

            if toi is not None:
                lines.append(f"TOI: {toi}")

            if tfopwg:
                lines.append(f"TFOPWG: {tfopwg}")
                # Color code based on disposition
                if "CP" in tfopwg or "KP" in tfopwg:
                    bg_color = "#d4edda"  # Light green for confirmed/known
                    status_color = "#155724"
                elif "FP" in tfopwg:
                    bg_color = "#f8d7da"  # Light red for false positive
                    status_color = "#721c24"
                elif "PC" in tfopwg:
                    bg_color = "#fff3cd"  # Light yellow for candidate
                    status_color = "#856404"
                else:
                    bg_color = "#d1ecf1"  # Light blue for other
                    status_color = "#0c5460"
            else:
                bg_color = "#d1ecf1"
                status_color = "#0c5460"

            if planet_disp:
                lines.append(f"Disposition: {planet_disp}")

            if comments:
                # Wrap notes across multiple lines to avoid overflowing the card.
                # Keep the overall card compact by limiting note lines.
                note_lines = _wrap_notes(str(comments), style=style, max_lines=3)
                if note_lines:
                    lines.append(f"Notes: {note_lines[0]}")
                    for extra in note_lines[1:]:
                        lines.append(f"  {extra}")

        # Set default text kwargs
        text_defaults: dict[str, Any] = {
            "fontsize": 10,
            "fontfamily": "monospace",
            "verticalalignment": "top",
            "horizontalalignment": "left",
        }
        text_defaults.update(text_kwargs)

        # Import Rectangle for background
        from matplotlib.patches import Rectangle

        # Draw background box
        rect = Rectangle(
            (0.05, 0.05),
            0.9,
            0.9,
            transform=ax.transAxes,
            facecolor=bg_color,
            edgecolor=status_color,
            linewidth=2,
            zorder=0,
        )
        ax.add_patch(rect)

        # Draw text
        text_content = "\n".join(lines)
        ax.text(
            0.1,
            0.85,
            text_content,
            transform=ax.transAxes,
            color=status_color,
            clip_on=True,
            **text_defaults,
        )

        # Set title if requested
        if show_title:
            ax.set_title("ExoFOP TOI Lookup", pad=10)

    return ax
