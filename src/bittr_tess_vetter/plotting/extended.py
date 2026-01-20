"""Plot functions for extended vetting checks (V16-V21)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.colorbar

    from bittr_tess_vetter.validation.result_schema import CheckResult


def plot_model_comparison(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    data_color: str | None = None,
    transit_color: str | None = None,
    eb_color: str | None = None,
    sinusoid_color: str | None = None,
    show_legend: bool = True,
    show_data: bool = True,
    annotate_winner: bool = True,
    style: str = "default",
    **plot_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot phase-folded data with competing model overlays.

    Creates a phase-folded scatter plot showing the data with overlaid
    best-fit models from the model competition analysis (transit-only,
    EB-like, transit+sinusoid).

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V16 (model competition) check. Must contain
        plot_data with phase, flux, and model arrays.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    data_color : str, optional
        Color for data points. Defaults to COLORS["out_of_transit"] (gray).
    transit_color : str, optional
        Color for transit model. Defaults to COLORS["transit"] (blue).
    eb_color : str, optional
        Color for EB-like model. Defaults to COLORS["secondary"] (brown).
    sinusoid_color : str, optional
        Color for sinusoid model. Defaults to COLORS["model"] (orange).
    show_legend : bool, default=True
        Whether to show the legend.
    show_data : bool, default=True
        Whether to show the data points.
    annotate_winner : bool, default=True
        Whether to annotate the winning model.
    style : str, default="default"
        Style preset: "default", "paper", or "presentation".
    **plot_kwargs : Any
        Additional keyword arguments passed to ax.plot() for models.

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
            "transit_model": [...],
            "eb_like_model": [...],
            "sinusoid_model": [...],
        }

    Examples
    --------
    >>> from bittr_tess_vetter.plotting import plot_model_comparison
    >>> ax = plot_model_comparison(result)  # Basic plot
    """
    from ._core import ensure_ax, extract_plot_data, style_context
    from ._styles import COLORS

    # Required keys for V16 model comparison plot
    required_keys = [
        "phase",
        "flux",
        "transit_model",
        "eb_like_model",
        "sinusoid_model",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    # Get colors from styles if not provided
    if data_color is None:
        data_color = COLORS["out_of_transit"]
    if transit_color is None:
        transit_color = COLORS["transit"]
    if eb_color is None:
        eb_color = COLORS["secondary"]
    if sinusoid_color is None:
        sinusoid_color = COLORS["model"]

    with style_context(style):
        fig, ax = ensure_ax(ax)

        phase = data["phase"]
        flux = data["flux"]
        transit_model = data["transit_model"]
        eb_model = data["eb_like_model"]
        sinusoid_model = data["sinusoid_model"]

        # Set default plot kwargs
        plot_defaults: dict[str, Any] = {
            "linewidth": 1.5,
        }
        plot_defaults.update(plot_kwargs)
        plot_kw: Any = plot_defaults

        # Plot data points
        if show_data and len(phase) > 0:
            ax.scatter(
                phase,
                flux,
                c=data_color,
                s=4,
                alpha=0.4,
                label="Data",
            )

        # Plot model overlays
        if len(transit_model) > 0:
            ax.plot(
                phase,
                transit_model,
                color=transit_color,
                label="Transit",
                **plot_kw,
            )
        if len(eb_model) > 0:
            ax.plot(
                phase,
                eb_model,
                color=eb_color,
                label="EB-like",
                linestyle="--",
                **plot_kw,
            )
        if len(sinusoid_model) > 0:
            ax.plot(
                phase,
                sinusoid_model,
                color=sinusoid_color,
                label="Sinusoid",
                linestyle=":",
                **plot_kw,
            )

        # Annotate winner if available and requested
        if annotate_winner and result.metrics:
            winner = result.metrics.get("winner")
            if winner is not None:
                ax.text(
                    0.95,
                    0.95,
                    f"Winner: {winner}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                )

        # Set labels
        ax.set_xlabel("Phase")
        ax.set_ylabel("Normalized Flux")
        ax.set_title("Model Competition")

        # Add legend if requested
        if show_legend:
            ax.legend(loc="lower right")

    return ax


def plot_ephemeris_reliability(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    score_color: str | None = None,
    null_color: str | None = None,
    show_legend: bool = True,
    show_null_distribution: bool = True,
    annotate_p_value: bool = True,
    style: str = "default",
    **plot_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot ephemeris reliability diagnostics.

    Creates a plot showing the detection score vs phase shift, comparing
    the on-ephemeris score to the null distribution from shifted phases.

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V17 (ephemeris reliability) check. Must contain
        plot_data with phase_shifts and null_scores arrays.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    score_color : str, optional
        Color for the score curve. Defaults to COLORS["transit"] (blue).
    null_color : str, optional
        Color for null distribution. Defaults to COLORS["out_of_transit"] (gray).
    show_legend : bool, default=True
        Whether to show the legend.
    show_null_distribution : bool, default=True
        Whether to show the null score distribution.
    annotate_p_value : bool, default=True
        Whether to annotate the p-value.
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

    Notes
    -----
    Expected plot_data schema (version 1):

    .. code-block:: python

        plot_data = {
            "version": 1,
            "phase_shifts": [...],
            "null_scores": [...],
            "period_neighborhood": [...],
            "neighborhood_scores": [...],
        }

    Examples
    --------
    >>> from bittr_tess_vetter.plotting import plot_ephemeris_reliability
    >>> ax = plot_ephemeris_reliability(result)  # Basic plot
    """
    from ._core import ensure_ax, extract_plot_data, style_context
    from ._styles import COLORS

    # Required keys for V17 ephemeris reliability plot
    required_keys = [
        "phase_shifts",
        "null_scores",
        "period_neighborhood",
        "neighborhood_scores",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    # Get colors from styles if not provided
    if score_color is None:
        score_color = COLORS["transit"]
    if null_color is None:
        null_color = COLORS["out_of_transit"]

    with style_context(style):
        fig, ax = ensure_ax(ax)

        phase_shifts = data["phase_shifts"]
        null_scores = data["null_scores"]

        # Set default plot kwargs
        plot_defaults: dict[str, Any] = {
            "linewidth": 1.5,
        }
        plot_defaults.update(plot_kwargs)
        plot_kw: Any = plot_defaults

        # Plot null score distribution
        if show_null_distribution and len(phase_shifts) > 0 and len(null_scores) > 0:
            ax.plot(
                phase_shifts,
                null_scores,
                color=null_color,
                alpha=0.7,
                label="Null scores",
                **plot_kw,
            )

            # Mark the on-ephemeris point (phase shift = 0)
            on_eph_idx = int(np.argmin(np.abs(np.array(phase_shifts))))
            if on_eph_idx < len(null_scores):
                ax.scatter(
                    [phase_shifts[on_eph_idx]],
                    [null_scores[on_eph_idx]],
                    color=score_color,
                    s=100,
                    marker="*",
                    zorder=5,
                    label="On-ephemeris",
                )

        # Annotate p-value if available and requested
        if annotate_p_value and result.metrics:
            p_value = result.metrics.get("phase_shift_null_p_value")
            if p_value is not None:
                ax.text(
                    0.95,
                    0.95,
                    f"p-value: {p_value:.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                )

        # Set labels
        ax.set_xlabel("Phase Shift")
        ax.set_ylabel("Detection Score")
        ax.set_title("Ephemeris Reliability")

        # Add legend if requested
        if show_legend:
            ax.legend(loc="lower right")

    return ax


def plot_sensitivity_sweep(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    show_failed: bool = True,
    show_colorbar: bool = True,
    colorbar_label: str = "Depth hat (ppm)",
    ok_color: str = "#1f77b4",
    failed_color: str = "#7f7f7f",
    style: str = "default",
    **scatter_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot ephemeris sensitivity sweep results (V18).

    Visualizes the robustness of the detection score across preprocessing variants.
    Each variant is represented as a point at its score; points can be colored by
    depth estimate to highlight variant-driven depth shifts.

    Parameters
    ----------
    result : CheckResult
        A CheckResult from V18. Must contain `raw["plot_data"]["sweep_table"]`.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    show_failed : bool, default=True
        Whether to include failed variants (status != "ok") in the plot.
    show_colorbar : bool, default=True
        Whether to show a colorbar when depth estimates are available.
    colorbar_label : str, default="Depth hat (ppm)"
        Colorbar label when `show_colorbar=True`.
    ok_color : str, default="#1f77b4"
        Color for ok points when depth coloring is not used.
    failed_color : str, default="#7f7f7f"
        Color for failed points.
    style : str, default="default"
        Style preset: "default", "paper", or "presentation".
    **scatter_kwargs : Any
        Additional keyword arguments passed to `ax.scatter()`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    from ._core import add_colorbar, ensure_ax, extract_plot_data, style_context

    data = extract_plot_data(result, required_keys=["sweep_table"])
    sweep_table = data.get("sweep_table") or []

    rows: list[dict[str, Any]] = [r for r in sweep_table if isinstance(r, dict)]
    if not rows:
        raise ValueError("V18 plot requires non-empty plot_data['sweep_table']")

    labels: list[str] = []
    scores: list[float] = []
    depths_ppm: list[float | None] = []
    statuses: list[str] = []

    def _format_variant_label(
        *,
        downsample_factor: object,
        outlier_policy: object,
        detrender: object,
        max_len: int = 28,
    ) -> str:
        ds = "?" if downsample_factor is None else str(downsample_factor)
        op = "?" if outlier_policy is None else str(outlier_policy)
        det = "?" if detrender is None else str(detrender)

        def _abbr_outlier(policy: str) -> str:
            if policy in {"none", "null", "None"}:
                return "none"
            if policy.startswith("sigma_clip_"):
                return "sc" + policy.removeprefix("sigma_clip_")
            if policy.startswith("winsorize_"):
                return "win" + policy.removeprefix("winsorize_")
            return policy

        def _abbr_detrender(name: str) -> str:
            if name in {"none", "null", "None"}:
                return "none"
            if name.startswith("running_median_"):
                return "rm" + name.removeprefix("running_median_")
            if name.startswith("savgol_"):
                return "sg" + name.removeprefix("savgol_")
            if name.startswith("gp_"):
                return "gp" + name.removeprefix("gp_")
            return name

        op_short = _abbr_outlier(op)
        det_short = _abbr_detrender(det)

        label = f"ds={ds} {op_short} {det_short}".strip()
        label = " ".join(label.split())

        if len(label) <= max_len:
            return label

        label = f"ds={ds} {op_short}".strip()
        if len(label) <= max_len:
            return label

        return f"ds={ds}"

    for r in rows:
        status = str(r.get("status", "unknown"))
        if status != "ok" and not show_failed:
            continue

        label = _format_variant_label(
            downsample_factor=r.get("downsample_factor"),
            outlier_policy=r.get("outlier_policy"),
            detrender=r.get("detrender"),
        )
        labels.append(label)
        statuses.append(status)

        score_val = r.get("score")
        scores.append(float(score_val) if score_val is not None else float("nan"))

        depth_val = r.get("depth_hat_ppm")
        depths_ppm.append(float(depth_val) if depth_val is not None else None)

    y = np.arange(len(labels))

    with style_context(style):
        fig, ax = ensure_ax(ax)

        # Use depth coloring only if we have at least one depth value.
        use_depth_color = show_colorbar and any(d is not None for d in depths_ppm)

        default_scatter_kwargs: dict[str, Any] = {
            "s": 50,
            "alpha": 0.9,
            "edgecolors": "black",
            "linewidths": 0.5,
        }
        default_scatter_kwargs.update(scatter_kwargs)

        cbar = None
        if use_depth_color:
            depth_arr = np.array([d if d is not None else np.nan for d in depths_ppm], dtype=float)
            sc = ax.scatter(scores, y, c=depth_arr, cmap="viridis", **default_scatter_kwargs)
            cbar = add_colorbar(sc, ax, label=colorbar_label)
        else:
            ok_mask = np.array([s == "ok" for s in statuses], dtype=bool)
            if np.any(ok_mask):
                ax.scatter(
                    np.array(scores)[ok_mask],
                    y[ok_mask],
                    c=ok_color,
                    **default_scatter_kwargs,
                )
            if show_failed and np.any(~ok_mask):
                ax.scatter(
                    np.array(scores)[~ok_mask],
                    y[~ok_mask],
                    c=failed_color,
                    marker="x",
                    s=60,
                    alpha=0.9,
                    linewidths=2,
                    label="Failed",
                )

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Score")
        ax.set_title("Sensitivity Sweep")
        ax.grid(True, axis="x", alpha=0.2)

        # Summary annotation (top-right, away from y labels)
        stable = data.get("stable")
        n_total = data.get("n_variants_total")
        n_ok = data.get("n_variants_ok")
        txt = []
        if stable is not None:
            txt.append(f"stable: {bool(stable)}")
        if n_total is not None and n_ok is not None:
            txt.append(f"ok: {int(n_ok)}/{int(n_total)}")
        if txt:
            ax.text(
                0.98,
                0.02,
                "\n".join(txt),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

        if not use_depth_color and show_failed and any(s != "ok" for s in statuses):
            ax.legend(loc="lower right", fontsize="small")

        # Silence unused variable warnings while keeping cbar creation explicit.
        _ = cbar

    return ax


def plot_alias_diagnostics(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    bar_color: str | None = None,
    best_color: str | None = None,
    show_legend: bool = True,
    annotate_best: bool = True,
    style: str = "default",
    **bar_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot harmonic/alias detection scores as a bar chart.

    Creates a bar chart showing detection significance at various harmonic
    periods (P, P/2, 2P, P/3, 3P) to identify potential period aliases.

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V19 (alias diagnostics) check. Must contain
        plot_data with harmonic labels, periods, and scores.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    bar_color : str, optional
        Color for bars. Defaults to COLORS["transit"] (blue).
    best_color : str, optional
        Color for the best (highest score) bar. Defaults to COLORS["model"] (orange).
    show_legend : bool, default=True
        Whether to show the legend.
    annotate_best : bool, default=True
        Whether to annotate the best harmonic.
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
            "harmonic_labels": [...],    # e.g., ["P", "P/2", "2P", "P/3", "3P"]
            "harmonic_periods": [...],   # Actual period values in days
            "harmonic_scores": [...],    # Detection scores/significance
        }

    Examples
    --------
    >>> from bittr_tess_vetter.plotting import plot_alias_diagnostics
    >>> ax = plot_alias_diagnostics(result)  # Basic plot
    """
    import numpy as np

    from ._core import ensure_ax, extract_plot_data, style_context
    from ._styles import COLORS

    # Required keys for V19 alias diagnostics plot
    required_keys = [
        "harmonic_labels",
        "harmonic_periods",
        "harmonic_scores",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    # Get colors from styles if not provided
    if bar_color is None:
        bar_color = COLORS["transit"]
    if best_color is None:
        best_color = COLORS["model"]

    with style_context(style):
        fig, ax = ensure_ax(ax)

        labels = data["harmonic_labels"]
        scores = data["harmonic_scores"]

        n_harmonics = len(labels)
        x_positions = list(range(n_harmonics))

        # Set default bar kwargs
        bar_defaults: dict[str, Any] = {
            "width": 0.6,
            "edgecolor": "black",
            "linewidth": 1,
        }
        bar_defaults.update(bar_kwargs)
        bar_kw: Any = bar_defaults

        # Find best harmonic
        if len(scores) > 0:
            best_idx = int(np.argmax(scores))
            colors = [bar_color] * n_harmonics
            colors[best_idx] = best_color
        else:
            colors = [bar_color] * n_harmonics
            best_idx = 0

        # Plot bars
        if n_harmonics > 0:
            ax.bar(
                x_positions,
                scores,
                color=colors,
                **bar_kw,
            )

        # Annotate best harmonic if requested
        if annotate_best and len(labels) > 0:
            ax.text(
                0.95,
                0.95,
                f"Best: {labels[best_idx]}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

        # Set labels
        ax.set_xlabel("Harmonic")
        ax.set_ylabel("Detection Score")
        ax.set_title("Alias/Harmonic Diagnostics")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels)

        # Add legend if requested
        if show_legend:
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor=bar_color, label="Other harmonics"),
                Patch(facecolor=best_color, label="Best harmonic"),
            ]
            ax.legend(handles=legend_elements, loc="upper right")

    return ax


def plot_ghost_features(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    show_colorbar: bool = True,
    cbar_label: str = "Flux Difference",
    cbar_kwargs: dict[str, Any] | None = None,
    cmap: str | None = None,
    show_aperture: bool = True,
    aperture_color: str = "red",
    annotate_depths: bool = True,
    style: str = "default",
    **imshow_kwargs: Any,
) -> tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar | None]:
    """Plot difference image with aperture mask overlay.

    Creates an image showing the out-of-transit minus in-transit difference
    with the photometric aperture overlaid. This visualization helps identify
    ghost or scattered light artifacts.

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V20 (ghost features) check. Must contain
        plot_data with difference_image and aperture_mask arrays.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    show_colorbar : bool, default=True
        Whether to show the colorbar.
    cbar_label : str, default="Flux Difference"
        Label for the colorbar.
    cbar_kwargs : dict, optional
        Additional keyword arguments passed to add_colorbar().
    cmap : str, optional
        Colormap name. Defaults to COLORMAPS["difference"] ("RdBu_r").
    show_aperture : bool, default=True
        Whether to overlay the aperture mask.
    aperture_color : str, default="red"
        Color for aperture mask contour.
    annotate_depths : bool, default=True
        Whether to annotate in/out aperture depths.
    style : str, default="default"
        Style preset: "default", "paper", or "presentation".
    **imshow_kwargs : Any
        Additional keyword arguments passed to ax.imshow().

    Returns
    -------
    tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar | None]
        The axes and colorbar (None if show_colorbar=False).

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
            "difference_image": [[...]],    # 2D array [row][col]
            "aperture_mask": [[...]],        # 2D boolean array
            "in_aperture_depth": float,
            "out_aperture_depth": float,
        }

    Examples
    --------
    >>> from bittr_tess_vetter.plotting import plot_ghost_features
    >>> ax, cbar = plot_ghost_features(result)  # Basic plot
    """
    import numpy as np

    from ._core import add_colorbar, ensure_ax, extract_plot_data, style_context
    from ._styles import COLORMAPS

    # Required keys for V20 ghost features plot
    required_keys = [
        "difference_image",
        "aperture_mask",
        "in_aperture_depth",
        "out_aperture_depth",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    # Get colormap from styles if not provided
    if cmap is None:
        cmap = COLORMAPS["difference"]

    with style_context(style):
        fig, ax = ensure_ax(ax)

        diff_image = np.array(data["difference_image"])
        aperture_mask = np.array(data["aperture_mask"], dtype=bool)
        in_depth = data["in_aperture_depth"]
        out_depth = data["out_aperture_depth"]

        # Set default imshow kwargs
        imshow_defaults: dict[str, Any] = {
            "origin": "lower",
            "aspect": "equal",
        }
        imshow_defaults.update(imshow_kwargs)
        imshow_kw: Any = imshow_defaults

        # Plot difference image
        im = ax.imshow(diff_image, cmap=cmap, **imshow_kw)

        # Overlay aperture mask if requested
        if show_aperture and aperture_mask.shape == diff_image.shape:
            ax.contour(
                aperture_mask,
                levels=[0.5],
                colors=[aperture_color],
                linewidths=1.5,
                origin="lower",
            )

        # Add colorbar if requested
        cbar = None
        if show_colorbar:
            cbar_kw = cbar_kwargs or {}
            cbar = add_colorbar(im, ax, label=cbar_label, **cbar_kw)

        # Annotate depths if requested
        if annotate_depths:
            ax.text(
                0.05,
                0.95,
                f"In-aperture: {in_depth:.4f}\nOut-aperture: {out_depth:.4f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

        # Set labels
        ax.set_xlabel("Column (pixels)")
        ax.set_ylabel("Row (pixels)")
        ax.set_title("Ghost Feature Analysis")

    return ax, cbar


def plot_sector_consistency(
    result: CheckResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    bar_color: str | None = None,
    outlier_color: str | None = None,
    mean_color: str | None = None,
    show_legend: bool = True,
    show_mean: bool = True,
    show_error_bars: bool = True,
    annotate_chi2: bool = True,
    style: str = "default",
    **bar_kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot per-sector transit depths with error bars.

    Creates a bar chart showing transit depth measurements from individual
    sectors with error bars and a horizontal line for the weighted mean.
    Outlier sectors are highlighted.

    Parameters
    ----------
    result : CheckResult
        A CheckResult from the V21 (sector consistency) check. Must contain
        plot_data with sectors, depths, and errors.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    bar_color : str, optional
        Color for normal sector bars. Defaults to COLORS["transit"] (blue).
    outlier_color : str, optional
        Color for outlier sector bars. Defaults to COLORS["outlier"] (red).
    mean_color : str, optional
        Color for weighted mean line. Defaults to COLORS["model"] (orange).
    show_legend : bool, default=True
        Whether to show the legend.
    show_mean : bool, default=True
        Whether to show weighted mean horizontal line.
    show_error_bars : bool, default=True
        Whether to show error bars on the depth measurements.
    annotate_chi2 : bool, default=True
        Whether to annotate the chi-squared p-value.
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
            "sectors": [...],
            "depths_ppm": [...],
            "depth_errs_ppm": [...],
            "weighted_mean_ppm": float,
        }

    Examples
    --------
    >>> from bittr_tess_vetter.plotting import plot_sector_consistency
    >>> ax = plot_sector_consistency(result)  # Basic plot
    """
    from ._core import ensure_ax, extract_plot_data, style_context
    from ._styles import COLORS

    # Required keys for V21 sector consistency plot
    required_keys = [
        "sectors",
        "depths_ppm",
        "depth_errs_ppm",
        "weighted_mean_ppm",
    ]

    # Extract and validate plot data
    data = extract_plot_data(result, required_keys)

    # Get colors from styles if not provided
    if bar_color is None:
        bar_color = COLORS["transit"]
    if outlier_color is None:
        outlier_color = COLORS["outlier"]
    if mean_color is None:
        mean_color = COLORS["model"]

    with style_context(style):
        fig, ax = ensure_ax(ax)

        sectors = data["sectors"]
        depths = data["depths_ppm"]
        depth_errs = data["depth_errs_ppm"]
        weighted_mean = data["weighted_mean_ppm"]

        # Get outlier sectors if available
        outlier_sectors = data.get("outlier_sectors", [])

        n_sectors = len(sectors)
        x_positions = list(range(n_sectors))

        # Set default bar kwargs
        bar_defaults: dict[str, Any] = {
            "width": 0.6,
            "edgecolor": "black",
            "linewidth": 1,
        }
        bar_defaults.update(bar_kwargs)
        bar_kw: Any = bar_defaults

        # Determine bar colors (highlight outliers)
        colors = []
        for sector in sectors:
            if sector in outlier_sectors:
                colors.append(outlier_color)
            else:
                colors.append(bar_color)

        # Plot bars
        if n_sectors > 0:
            ax.bar(
                x_positions,
                depths,
                color=colors,
                **bar_kw,
            )

            # Add error bars if requested
            if show_error_bars:
                ax.errorbar(
                    x_positions,
                    depths,
                    yerr=depth_errs,
                    fmt="none",
                    color="black",
                    capsize=4,
                    capthick=1,
                )

        # Add weighted mean line if requested
        if show_mean and weighted_mean > 0:
            ax.axhline(
                weighted_mean,
                color=mean_color,
                linestyle="--",
                linewidth=1.5,
                label=f"Weighted mean: {weighted_mean:.0f} ppm",
            )

        # Annotate chi2 p-value if available and requested
        # Place at upper-left to avoid overlap with legend at upper-right
        if annotate_chi2 and result.metrics:
            chi2_p = result.metrics.get("chi2_p_value")
            if chi2_p is not None:
                ax.text(
                    0.05,
                    0.95,
                    f"$\\chi^2$ p-value: {chi2_p:.3f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                )

        # Set labels
        ax.set_xlabel("Sector")
        ax.set_ylabel("Depth (ppm)")
        ax.set_title("Sector-to-Sector Consistency")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(s) for s in sectors])

        # Add legend if requested
        if show_legend:
            from matplotlib.patches import Patch

            legend_elements: list[Any] = [
                Patch(facecolor=bar_color, label="Normal"),
                Patch(facecolor=outlier_color, label="Outlier"),
            ]
            if show_mean and weighted_mean > 0:
                from matplotlib.lines import Line2D

                legend_elements.append(
                    Line2D([0], [0], color=mean_color, linestyle="--", label="Weighted mean")
                )
            ax.legend(handles=legend_elements, loc="upper right")

    return ax


__all__ = [
    "plot_model_comparison",
    "plot_ephemeris_reliability",
    "plot_sensitivity_sweep",
    "plot_alias_diagnostics",
    "plot_ghost_features",
    "plot_sector_consistency",
]
