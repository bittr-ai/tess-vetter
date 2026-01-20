"""Plot functions for pixel-level vetting checks (V08-V10)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.colorbar

    from bittr_tess_vetter.validation.result_schema import CheckResult


def plot_centroid_shift(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    show_colorbar: bool = True,
    cbar_label: str = "Flux (e-/s)",
    in_color: str = "#e377c2",  # Pink
    out_color: str = "#17becf",  # Cyan
    show_vector: bool = True,
    show_target: bool = True,
    style: str = "default",
    **imshow_kwargs: Any,
) -> tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar | None]:
    """Plot centroid shift visualization for V08 check.

    Creates an image showing the out-of-transit reference image with
    overlaid markers for in-transit and out-of-transit centroids, plus
    an optional vector showing the shift direction.

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V08 (centroid shift) check. Must contain
        plot_data with reference_image and centroid coordinates.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    show_colorbar : bool, default=True
        Whether to show the colorbar.
    cbar_label : str, default="Flux (e-/s)"
        Label for the colorbar.
    in_color : str, default="#e377c2"
        Color for in-transit centroid marker.
    out_color : str, default="#17becf"
        Color for out-of-transit centroid marker.
    show_vector : bool, default=True
        Whether to show the shift vector between centroids.
    show_target : bool, default=True
        Whether to show the target pixel marker.
    style : str, default="default"
        Style preset: "default", "paper", or "presentation".
    **imshow_kwargs : Any
        Additional keyword arguments passed to ax.imshow().

    Returns
    -------
    tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar | None]
        The axes containing the plot and the colorbar (or None if disabled).

    Raises
    ------
    ValueError
        If result has no plot_data or is missing required keys.

    Examples
    --------
    >>> from bittr_tess_vetter.plotting import plot_centroid_shift
    >>> ax, cbar = plot_centroid_shift(result)

    >>> # Without colorbar
    >>> ax, _ = plot_centroid_shift(result, show_colorbar=False)
    """
    from ._core import add_colorbar, ensure_ax, extract_plot_data, style_context

    # Required keys for V08 centroid shift plot
    required_keys = [
        "reference_image",
        "in_centroid_col",
        "in_centroid_row",
        "out_centroid_col",
        "out_centroid_row",
        "target_col",
        "target_row",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    # Convert reference image to numpy array
    reference_image = np.array(data["reference_image"], dtype=np.float64)

    with style_context(style):
        fig, ax = ensure_ax(ax)

        # Set default imshow kwargs - CRITICAL: use origin="lower"
        imshow_defaults: dict[str, Any] = {
            "origin": "lower",
            "cmap": "viridis",
            "aspect": "equal",
        }
        imshow_defaults.update(imshow_kwargs)

        # Plot the reference image
        im = ax.imshow(reference_image, **imshow_defaults)

        # Add colorbar if requested
        cbar = None
        if show_colorbar:
            cbar = add_colorbar(im, ax, label=cbar_label)

        # Get centroid coordinates
        in_col = data["in_centroid_col"]
        in_row = data["in_centroid_row"]
        out_col = data["out_centroid_col"]
        out_row = data["out_centroid_row"]

        # Plot centroids: x = col, y = row (for imshow with origin="lower")
        ax.plot(
            out_col,
            out_row,
            marker="o",
            markersize=10,
            markerfacecolor="none",
            markeredgecolor=out_color,
            markeredgewidth=2,
            label="Out-of-transit",
        )
        ax.plot(
            in_col,
            in_row,
            marker="o",
            markersize=10,
            markerfacecolor="none",
            markeredgecolor=in_color,
            markeredgewidth=2,
            label="In-transit",
        )

        # Draw shift vector if requested
        if show_vector:
            ax.annotate(
                "",
                xy=(in_col, in_row),
                xytext=(out_col, out_row),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#ff7f0e",  # Orange
                    "lw": 2,
                },
            )

        # Show target pixel marker if requested
        if show_target:
            target_col = data["target_col"]
            target_row = data["target_row"]
            ax.plot(
                target_col,
                target_row,
                marker="+",
                markersize=12,
                markeredgecolor="#d62728",  # Red
                markeredgewidth=2,
                label="Target",
            )

        # Set labels
        ax.set_xlabel("Column (pixels)")
        ax.set_ylabel("Row (pixels)")
        ax.set_title("Centroid Shift")

        # Add legend
        ax.legend(loc="upper right", fontsize="small")

    return ax, cbar


def plot_difference_image(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    show_colorbar: bool = True,
    cbar_label: str = "Depth (ppm)",
    cmap: str = "RdBu_r",
    show_target: bool = True,
    show_max_depth: bool = True,
    style: str = "default",
    **imshow_kwargs: Any,
) -> tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar | None]:
    """Plot difference image / pixel depth map for V09 check.

    Creates an image showing per-pixel transit depths with markers for
    the target pixel and the pixel with maximum depth.

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V09 (pixel-level depth) check. Must contain
        plot_data with depth_map_ppm and pixel coordinates.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    show_colorbar : bool, default=True
        Whether to show the colorbar.
    cbar_label : str, default="Depth (ppm)"
        Label for the colorbar.
    cmap : str, default="RdBu_r"
        Colormap for the depth image. RdBu_r centers at 0 (white).
    show_target : bool, default=True
        Whether to show the target pixel marker.
    show_max_depth : bool, default=True
        Whether to show the max depth pixel marker.
    style : str, default="default"
        Style preset: "default", "paper", or "presentation".
    **imshow_kwargs : Any
        Additional keyword arguments passed to ax.imshow().

    Returns
    -------
    tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar | None]
        The axes containing the plot and the colorbar (or None if disabled).

    Raises
    ------
    ValueError
        If result has no plot_data or is missing required keys.

    Examples
    --------
    >>> from bittr_tess_vetter.plotting import plot_difference_image
    >>> ax, cbar = plot_difference_image(result)

    >>> # Custom colormap
    >>> ax, cbar = plot_difference_image(result, cmap="coolwarm")
    """
    from ._core import add_colorbar, ensure_ax, extract_plot_data, style_context

    # Required keys for V09 difference image plot
    required_keys = [
        "depth_map_ppm",
        "target_pixel",
        "max_depth_pixel",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    # Convert depth map to numpy array
    depth_map = np.array(data["depth_map_ppm"], dtype=np.float64)

    with style_context(style):
        fig, ax = ensure_ax(ax)

        # Compute symmetric colormap limits centered at 0
        vmax = np.nanmax(np.abs(depth_map))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        vmin = -vmax

        # Set default imshow kwargs - CRITICAL: use origin="lower"
        imshow_defaults: dict[str, Any] = {
            "origin": "lower",
            "cmap": cmap,
            "aspect": "equal",
            "vmin": vmin,
            "vmax": vmax,
        }
        imshow_defaults.update(imshow_kwargs)

        # Plot the depth map
        im = ax.imshow(depth_map, **imshow_defaults)

        # Add colorbar if requested
        cbar = None
        if show_colorbar:
            cbar = add_colorbar(im, ax, label=cbar_label)

        # Get pixel coordinates: [row, col]
        target_pixel = data["target_pixel"]
        max_depth_pixel = data["max_depth_pixel"]

        # Plot target pixel: x = col, y = row
        if show_target:
            ax.plot(
                target_pixel[1],  # col
                target_pixel[0],  # row
                marker="+",
                markersize=12,
                markeredgecolor="#d62728",  # Red
                markeredgewidth=2,
                label="Target",
            )

        # Plot max depth pixel
        if show_max_depth:
            ax.plot(
                max_depth_pixel[1],  # col
                max_depth_pixel[0],  # row
                marker="x",
                markersize=10,
                markeredgecolor="#2ca02c",  # Green
                markeredgewidth=2,
                label="Max Depth",
            )

        # Set labels
        ax.set_xlabel("Column (pixels)")
        ax.set_ylabel("Row (pixels)")
        ax.set_title("Pixel Depth Map")

        # Add legend
        if show_target or show_max_depth:
            ax.legend(loc="upper right", fontsize="small")

    return ax, cbar


def plot_aperture_curve(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    color: str = "#1f77b4",
    show_errorbars: bool = True,
    style: str = "default",
    **plot_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot aperture dependence curve for V10 check.

    Creates a line plot showing transit depth as a function of aperture
    radius, with optional error bars.

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V10 (aperture dependence) check. Must contain
        plot_data with aperture_radii_px, depths_ppm, and depth_errs_ppm.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    color : str, default="#1f77b4"
        Color for the line and markers.
    show_errorbars : bool, default=True
        Whether to show error bars.
    style : str, default="default"
        Style preset: "default", "paper", or "presentation".
    **plot_kwargs : Any
        Additional keyword arguments passed to ax.plot() or ax.errorbar().

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
    >>> from bittr_tess_vetter.plotting import plot_aperture_curve
    >>> ax = plot_aperture_curve(result)

    >>> # Without error bars
    >>> ax = plot_aperture_curve(result, show_errorbars=False)
    """
    from ._core import ensure_ax, extract_plot_data, style_context

    # Required keys for V10 aperture curve plot
    required_keys = [
        "aperture_radii_px",
        "depths_ppm",
        "depth_errs_ppm",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    # Get data arrays
    radii = np.array(data["aperture_radii_px"], dtype=np.float64)
    depths = np.array(data["depths_ppm"], dtype=np.float64)
    errs = np.array(data["depth_errs_ppm"], dtype=np.float64)

    with style_context(style):
        fig, ax = ensure_ax(ax)

        # Set default plot kwargs
        plot_defaults: dict[str, Any] = {
            "marker": "o",
            "markersize": 8,
            "linewidth": 1.5,
        }
        plot_defaults.update(plot_kwargs)

        if show_errorbars and np.any(errs > 0):
            ax.errorbar(
                radii,
                depths,
                yerr=errs,
                color=color,
                capsize=4,
                capthick=1.5,
                **plot_defaults,
            )
        else:
            ax.plot(
                radii,
                depths,
                color=color,
                **plot_defaults,
            )

        # Add horizontal line at y=0 for reference
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        # Set labels
        ax.set_xlabel("Aperture Radius (pixels)")
        ax.set_ylabel("Transit Depth (ppm)")
        ax.set_title("Aperture Dependence")

    return ax
