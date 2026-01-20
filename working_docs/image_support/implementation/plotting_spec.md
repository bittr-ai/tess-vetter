# bittr-tess-vetter Plotting Feature Specification

**Version:** 1.1.0
**Date:** 2026-01-20
**Status:** Implementation Ready
**Authors:** Engineering Team

**Revision Notes (v1.1):** Incorporated feedback on coordinate conventions, style context management, multi-sector handling, testing approach, and corrected module paths to match actual codebase structure.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [API Surface](#3-api-surface)
4. [Data Contracts](#4-data-contracts)
5. [Function Specifications](#5-function-specifications)
6. [Shared Utilities](#6-shared-utilities)
7. [Style System](#7-style-system)
8. [DVR Summary Report](#8-dvr-summary-report)
9. [CheckResult Modifications](#9-checkresult-modifications)
10. [Testing Strategy](#10-testing-strategy)
11. [Documentation Plan](#11-documentation-plan)
12. [Implementation Phases](#12-implementation-phases)
13. [Appendix D: Feedback Decisions](#appendix-d-feedback-decisions)

---

## 1. Overview

### 1.1 Background

Dr. Darin Ragozzine's feedback highlighted a critical gap in bittr-tess-vetter: "scientists use visual scans of data to validate calculations." The lack of diagnostic plots in vetting output is "jarring" compared to established standards like Kepler Data Validation Reports (DVRs).

This specification defines a comprehensive plotting feature that:
- Follows Kepler DVR/DVS visual standards adapted for TESS
- Integrates seamlessly with the existing array-in/array-out API philosophy
- Provides publication-quality diagnostic visualizations for all vetting checks

### 1.2 Goals

1. **Enable visual validation** - Scientists can generate diagnostic plots for any vetting check with a single function call
2. **DVR-style summaries** - Multi-panel summary reports matching Kepler DVS one-page format
3. **Publication quality** - Plots ready for papers with minimal customization
4. **Optional dependency** - matplotlib remains optional; core functionality works without it
5. **Data preservation** - Plot data stored in CheckResult for reproducibility

### 1.3 Non-Goals

1. **Interactive backends** - No Bokeh/Plotly support in Phase 1 (matplotlib only)
2. **Real-time streaming** - Plots are for post-hoc analysis, not live monitoring
3. **GUI applications** - CLI/notebook workflow only
4. **Custom themes** - Three presets (default, paper, presentation) sufficient initially

### 1.4 Success Criteria

1. Every check (V01-V21) has a corresponding plot function
2. `plot_vetting_summary()` produces DVR-style one-page report
3. Tutorial 10 updated with plotting cells demonstrating visual validation
4. All plots render without matplotlib warnings
5. 95%+ test coverage on plotting module

---

## 2. Architecture

### 2.1 Module Structure

```
src/bittr_tess_vetter/
  plotting/
    __init__.py          # MATPLOTLIB_AVAILABLE guard + public exports
    _core.py             # Shared utilities (ax handling, styles, colorbars)
    _styles.py           # Style presets and color definitions
    checks.py            # V01-V05 LC-only check plots
    catalog.py           # V06-V07 catalog check plots
    pixel.py             # V08-V10 pixel-level check plots
    exovetter.py         # V11-V12 exovetter check plots
    false_alarm.py       # V13, V15 false alarm check plots
    extended.py          # V16-V21 extended check plots
    transit.py           # Phase-folded, transit model overlay
    lightcurve.py        # Full light curve visualization
    report.py            # Multi-panel DVR-style summary
```

### 2.2 Dependency Management

#### 2.2.1 Optional Extra in pyproject.toml

```toml
[project.optional-dependencies]
plotting = ["matplotlib>=3.5.1"]
all = ["bittr-tess-vetter[tls,fit,wotan,batman,mlx,exovetter,ldtk,triceratops,plotting]"]
```

#### 2.2.2 Runtime Guard Pattern

```python
# bittr_tess_vetter/plotting/__init__.py
import importlib.util

MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None

if MATPLOTLIB_AVAILABLE:
    from bittr_tess_vetter.plotting.checks import (
        plot_odd_even,
        plot_secondary_eclipse,
        plot_duration_consistency,
        plot_depth_stability,
        plot_v_shape,
    )
    from bittr_tess_vetter.plotting.pixel import (
        plot_centroid_shift,
        plot_difference_image,
        plot_aperture_curve,
    )
    # ... additional imports ...

    __all__ = [
        "plot_odd_even",
        "plot_secondary_eclipse",
        # ... all exported functions ...
    ]
else:
    __all__ = []

def __getattr__(name: str):
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            f"Plotting requires matplotlib. Install with: "
            f"pip install 'bittr-tess-vetter[plotting]'"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

### 2.3 Integration Points

#### 2.3.1 API Surface Export (in api/__init__.py)

```python
# Guarded plotting exports
MATPLOTLIB_AVAILABLE = _importlib_util.find_spec("matplotlib") is not None

if MATPLOTLIB_AVAILABLE:
    __all__.extend([
        # Check-specific plots
        "plot_odd_even",
        "plot_secondary_eclipse",
        "plot_depth_stability",
        "plot_v_shape",
        "plot_nearby_ebs",
        "plot_centroid_shift",
        "plot_difference_image",
        "plot_aperture_curve",
        "plot_modshift",
        "plot_sweet",
        "plot_data_gaps",
        "plot_asymmetry",
        "plot_model_comparison",
        "plot_ephemeris_reliability",
        "plot_alias_diagnostics",
        "plot_ghost_features",
        "plot_sector_consistency",
        # Transit plots
        "plot_phase_folded",
        "plot_transit_fit",
        "plot_full_lightcurve",
        # Summary
        "plot_vetting_summary",
        "save_vetting_report",
    ])
```

#### 2.3.2 Data Flow

```
User Data (arrays)
    |
    v
vet_candidate() / run_check()
    |
    v
CheckResult (metrics + raw["plot_data"])
    |
    v
[Optional] plot_*(result) -> matplotlib Figure/Axes
    |
    v
[Optional] save_vetting_report(bundle) -> PDF/PNG
```

---

## 3. API Surface

### 3.1 Complete Function List

#### 3.1.1 Light Curve Check Plots (V01-V05)

| Function | Check | Returns |
|----------|-------|---------|
| `plot_odd_even(result, *, ax=None, ...)` | V01 | `Axes` |
| `plot_secondary_eclipse(result, *, ax=None, ...)` | V02 | `Axes` |
| `plot_duration_consistency(result, *, ax=None, ...)` | V03 | `Axes` |
| `plot_depth_stability(result, *, ax=None, ...)` | V04 | `Axes` |
| `plot_v_shape(result, *, ax=None, ...)` | V05 | `Axes` |

#### 3.1.2 Catalog Check Plots (V06-V07)

| Function | Check | Returns |
|----------|-------|---------|
| `plot_nearby_ebs(result, *, ax=None, ...)` | V06 | `Axes` |
| `plot_exofop_card(result, *, ax=None, ...)` | V07 | `Axes` |

#### 3.1.3 Pixel-Level Check Plots (V08-V10)

| Function | Check | Returns |
|----------|-------|---------|
| `plot_centroid_shift(result, *, ax=None, tpf=None, ...)` | V08 | `(Axes, Colorbar|None)` |
| `plot_difference_image(result, *, ax=None, ...)` | V09 | `(Axes, Colorbar|None)` |
| `plot_aperture_curve(result, *, ax=None, ...)` | V10 | `Axes` |

#### 3.1.4 Exovetter Check Plots (V11-V12)

| Function | Check | Returns |
|----------|-------|---------|
| `plot_modshift(result, *, ax=None, ...)` | V11 | `Axes` |
| `plot_sweet(result, *, ax=None, ...)` | V12 | `Axes` |

#### 3.1.5 False Alarm Check Plots (V13, V15)

| Function | Check | Returns |
|----------|-------|---------|
| `plot_data_gaps(result, *, ax=None, ...)` | V13 | `Axes` |
| `plot_asymmetry(result, *, ax=None, ...)` | V15 | `Axes` |

#### 3.1.6 Extended Check Plots (V16-V21)

| Function | Check | Returns |
|----------|-------|---------|
| `plot_model_comparison(result, *, ax=None, ...)` | V16 | `Axes` |
| `plot_ephemeris_reliability(result, *, ax=None, ...)` | V17 | `Axes` |
| `plot_sensitivity_sweep(result, *, ax=None, ...)` | V18 | `Axes` |
| `plot_alias_diagnostics(result, *, ax=None, ...)` | V19 | `Axes` |
| `plot_ghost_features(result, *, ax=None, tpf=None, ...)` | V20 | `(Axes, Colorbar|None)` |
| `plot_sector_consistency(result, *, ax=None, ...)` | V21 | `Axes` |

#### 3.1.7 Transit Visualization

| Function | Purpose | Returns |
|----------|---------|---------|
| `plot_phase_folded(lc, candidate, *, ax=None, ...)` | Phase-folded light curve | `Axes` |
| `plot_transit_fit(fit_result, *, ax=None, ...)` | Transit model overlay | `Axes` |
| `plot_full_lightcurve(lc, *, ax=None, transits=None, ...)` | Full time series | `Axes` |

#### 3.1.8 Summary Reports

| Function | Purpose | Returns |
|----------|---------|---------|
| `plot_vetting_summary(bundle, lc, candidate, *, ...)` | DVR-style multi-panel | `Figure` |
| `save_vetting_report(bundle, lc, candidate, path, *, ...)` | Export to file | `Path` |

### 3.2 Common Signature Pattern

All plotting functions follow this convention:

```python
def plot_<name>(
    result: CheckResult,
    *,
    ax: "matplotlib.axes.Axes | None" = None,
    # Check-specific options (varies by function)
    show_legend: bool = True,
    annotate: bool = True,
    style: str = "default",  # "default", "paper", "presentation"
    **mpl_kwargs,
) -> "matplotlib.axes.Axes":
```

For image plots requiring colorbars:

```python
def plot_<image_name>(
    result: CheckResult,
    *,
    ax: "matplotlib.axes.Axes | None" = None,
    show_colorbar: bool = True,
    cbar_label: str = "...",
    cbar_kwargs: dict[str, Any] | None = None,
    cmap: str = "...",
    **imshow_kwargs,
) -> tuple["matplotlib.axes.Axes", "matplotlib.colorbar.Colorbar | None"]:
```

---

## 4. Data Contracts

### 4.1 Design Decision: Always Include plot_data

Per research Q1, `plot_data` is **always included** in `CheckResult.raw` - not opt-in. Rationale:

1. **Small size**: Total ~7-10 KB per VettingBundleResult (negligible)
2. **UX priority**: Plotting works immediately without configuration
3. **Industry pattern**: scikit-learn Display objects store data unconditionally
4. **Reproducibility**: Results are self-contained

### 4.2 plot_data Schema Versioning

All `plot_data` dictionaries include a version marker for forward compatibility:

```python
raw["plot_data"] = {
    "version": 1,  # Schema version - increment on breaking changes
    # ... check-specific fields
}
```

### 4.3 JSON Serializability Requirements

All `plot_data` values MUST be JSON-serializable. Check implementations must explicitly convert:

- `np.float32/np.float64` -> `float`
- `np.int*` -> `int`
- `np.ndarray` (1D) -> `list` via `.tolist()`
- `np.ndarray` (2D stamps) -> nested `list[list[float]]` via `.astype(np.float32).tolist()`

### 4.4 Key Naming Conventions

All keys follow consistent naming with explicit units:

| Suffix | Meaning | Example |
|--------|---------|---------|
| `_ppm` | Parts per million | `odd_depths_ppm` |
| `_hours` | Duration in hours | `t_total_hours` |
| `_btjd` | Barycentric TESS Julian Date | `epoch_times_btjd` |
| `_arcsec` | Angular separation | `sep_arcsec` |
| `_px` / `_pixels` | Pixel units | `aperture_radii_px` |
| `_idx` / `_indices` | Array indices | `worst_epoch_indices` |
| `_x`, `_y` | Pixel coordinates (column, row) | `centroid_x`, `centroid_y` |
| `_row`, `_col` | Explicit array indices | `target_row`, `target_col` |

### 4.5 Coordinate Conventions for Pixel/Image Data

All pixel-level plots (V08, V09, V10, V20) follow these conventions:

1. **Image storage**: `reference_image[row][col]` (standard numpy/FITS convention)
2. **Coordinate system**: `(x, y)` maps to `(column, row)` in pixel space
3. **imshow origin**: Always use `origin="lower"` for astronomical convention (increasing row = up)
4. **Centroid coordinates**: `centroid_x` = column, `centroid_y` = row
5. **Target pixel**: Stored as `target_pixel_x` (column) and `target_pixel_y` (row)

This matches TESS TPF conventions and ensures overlay consistency.

### 4.6 Array Size Caps

To ensure predictable serialization size:

| Array Type | Maximum Size | Rationale |
|------------|--------------|-----------|
| Per-epoch arrays | 50 epochs | Covers typical multi-sector targets |
| Phase/modshift bins | 200 bins | Standard binning resolution |
| Image stamps | 21x21 pixels | Largest expected TPF cutout |

### 4.7 plot_data Schema by Check

#### V01 - Odd/Even Depth

```python
raw["plot_data"] = {
    "version": 1,
    "odd_epochs": list[int],         # Epoch indices (capped at 50)
    "odd_depths_ppm": list[float],   # Depths per epoch
    "odd_errs_ppm": list[float],     # Uncertainties
    "even_epochs": list[int],
    "even_depths_ppm": list[float],
    "even_errs_ppm": list[float],
    "mean_odd_ppm": float,
    "mean_even_ppm": float,
}
```

#### V02 - Secondary Eclipse

```python
raw["plot_data"] = {
    "phase": list[float],            # Phase array (0 to 1)
    "flux": list[float],             # Normalized flux
    "flux_err": list[float],         # Uncertainties
    "secondary_window": [float, float],  # Phase bounds for search
    "primary_window": [float, float],    # Phase bounds for primary
    "secondary_depth_ppm": float | None,
}
```

#### V03 - Duration Consistency

```python
raw["plot_data"] = {
    "observed_hours": float,
    "expected_hours": float,
    "expected_hours_err": float,
    "duration_ratio": float,
}
```

#### V04 - Depth Stability

```python
raw["plot_data"] = {
    "epoch_times_btjd": list[float],  # Mid-transit times
    "depths_ppm": list[float],        # Per-epoch depths
    "depth_errs_ppm": list[float],    # Uncertainties
    "mean_depth_ppm": float,
    "expected_scatter_ppm": float,
    "dominating_epoch_idx": int | None,
}
```

#### V05 - V-Shape / Transit Shape

```python
raw["plot_data"] = {
    "binned_phase": list[float],      # Phase bins (typically 20)
    "binned_flux": list[float],       # Binned flux values
    "binned_flux_err": list[float],   # Binned uncertainties
    "trapezoid_phase": list[float],   # Model phase points
    "trapezoid_flux": list[float],    # Model flux values
    "t_flat_hours": float,
    "t_total_hours": float,
}
```

#### V06 - Nearby EB Search

```python
raw["plot_data"] = {
    "target_ra": float,
    "target_dec": float,
    "matches": list[{
        "ra": float,
        "dec": float,
        "sep_arcsec": float,
        "period_days": float,
        "depth_ppm": float,
        "source_id": str,
    }],
    "search_radius_arcsec": float,
}
```

#### V08 - Centroid Shift

```python
raw["plot_data"] = {
    "reference_image": list[list[float]],  # 2D out-of-transit median (float32)
    "in_centroid_x": float,
    "in_centroid_y": float,
    "out_centroid_x": float,
    "out_centroid_y": float,
    "shift_vector_x": float,
    "shift_vector_y": float,
    "target_pixel_x": int,
    "target_pixel_y": int,
}
```

#### V09 - Difference Image

```python
raw["plot_data"] = {
    "difference_image": list[list[float]],   # 2D diff (float32)
    "depth_map_ppm": list[list[float]],      # 2D per-pixel depths
    "in_transit_median": list[list[float]],  # Optional 2D
    "out_transit_median": list[list[float]], # Optional 2D
    "target_pixel": [int, int],
    "max_depth_pixel": [int, int],
}
```

#### V10 - Aperture Dependence

```python
raw["plot_data"] = {
    "aperture_radii_px": list[float],  # Radii tested
    "depths_ppm": list[float],         # Depth at each radius
    "depth_errs_ppm": list[float],     # Uncertainties
    "reference_image": list[list[float]] | None,  # Optional 2D
}
```

#### V11 - ModShift

```python
raw["plot_data"] = {
    "phase_bins": list[float],         # ~200 phase bins
    "periodogram": list[float],        # ModShift signal
    "primary_phase": float,
    "secondary_phase": float | None,
    "tertiary_phase": float | None,
    "primary_signal": float,
    "secondary_signal": float | None,
    "tertiary_signal": float | None,
}
```

#### V12 - SWEET

```python
raw["plot_data"] = {
    "phase": list[float],
    "flux": list[float],
    "half_period_fit": list[float] | None,
    "at_period_fit": list[float] | None,
    "double_period_fit": list[float] | None,
    "snr_half": float,
    "snr_at": float,
    "snr_double": float,
}
```

#### V13 - Data Gaps

```python
raw["plot_data"] = {
    "epoch_centers_btjd": list[float],
    "coverage_fractions": list[float],  # 0-1 per epoch
    "worst_epoch_indices": list[int],
    "transit_window_hours": float,
}
```

#### V15 - Transit Asymmetry

```python
raw["plot_data"] = {
    "phase": list[float],
    "flux": list[float],
    "left_bin_mean": float,
    "right_bin_mean": float,
    "left_bin_phase_range": [float, float],
    "right_bin_phase_range": [float, float],
}
```

#### V16 - Model Competition

```python
raw["plot_data"] = {
    "phase": list[float],
    "flux": list[float],
    "transit_model": list[float],
    "eb_like_model": list[float] | None,
    "sinusoid_model": list[float] | None,
    "bic_transit": float,
    "bic_eb_like": float | None,
    "bic_sinusoid": float | None,
}
```

#### V17 - Ephemeris Reliability

```python
raw["plot_data"] = {
    "phase_shifts": list[float],       # Tested phase offsets
    "null_scores": list[float],        # Scores at each shift
    "period_neighborhood": list[float],# Periods tested
    "neighborhood_scores": list[float],# Scores at each period
    "best_period": float,
    "best_score": float,
}
```

#### V19 - Alias Diagnostics

```python
raw["plot_data"] = {
    "harmonic_labels": list[str],      # "P", "P/2", "2P", etc.
    "harmonic_periods": list[float],
    "harmonic_scores": list[float],
    "base_period": float,
    "best_score_period": float,
}
```

#### V20 - Ghost Features

```python
raw["plot_data"] = {
    "difference_image": list[list[float]],
    "aperture_mask": list[list[bool]],
    "in_aperture_depth": float,
    "out_aperture_depth": float,
    "gradient_magnitude": list[list[float]] | None,
}
```

#### V21 - Sector Consistency

```python
raw["plot_data"] = {
    "sectors": list[int],
    "depths_ppm": list[float],
    "depth_errs_ppm": list[float],
    "outlier_sectors": list[int],
    "weighted_mean_ppm": float,
}
```

---

## 5. Function Specifications

### 5.1 V01: plot_odd_even

```python
def plot_odd_even(
    result: CheckResult,
    *,
    ax: "matplotlib.axes.Axes | None" = None,
    odd_color: str = "#2ca02c",      # Green
    even_color: str = "#d62728",     # Red
    show_legend: bool = True,
    show_means: bool = True,
    annotate_sigma: bool = True,
    style: str = "default",
    **errorbar_kwargs,
) -> "matplotlib.axes.Axes":
    """Plot odd vs even transit depth comparison.

    Displays per-epoch depths for odd and even transits with error bars,
    highlighting any systematic difference that would indicate an eclipsing
    binary at twice the reported period.

    Parameters
    ----------
    result : CheckResult
        Result from V01 odd-even depth check. Must have raw["plot_data"].
    ax : Axes, optional
        Axes to plot on. Creates new figure if None.
    odd_color : str
        Color for odd transit markers.
    even_color : str
        Color for even transit markers.
    show_legend : bool
        Display legend differentiating odd/even.
    show_means : bool
        Show horizontal lines at mean depths.
    annotate_sigma : bool
        Annotate the sigma difference in the title.
    style : str
        Style preset: "default", "paper", "presentation".
    **errorbar_kwargs
        Additional kwargs passed to ax.errorbar().

    Returns
    -------
    Axes
        The matplotlib axes containing the plot.

    Raises
    ------
    ValueError
        If result does not contain plot_data.

    Examples
    --------
    >>> result = run_check("V01", lc, candidate)
    >>> ax = plot_odd_even(result)
    >>> plt.savefig("odd_even.png")

    >>> # Custom colors for publication
    >>> fig, ax = plt.subplots(figsize=(4, 3))
    >>> plot_odd_even(result, ax=ax, odd_color="blue", even_color="orange")
    """
```

### 5.2 V02: plot_secondary_eclipse

```python
def plot_secondary_eclipse(
    result: CheckResult,
    *,
    ax: "matplotlib.axes.Axes | None" = None,
    primary_color: str = "#1f77b4",
    secondary_color: str = "#9467bd",
    show_windows: bool = True,
    zoom_secondary: bool = False,
    style: str = "default",
    **scatter_kwargs,
) -> "matplotlib.axes.Axes":
    """Plot phase-folded light curve highlighting secondary eclipse region.

    Shows the full orbital phase with primary and secondary eclipse windows
    marked, enabling visual assessment of potential secondary eclipse depth.

    Parameters
    ----------
    result : CheckResult
        Result from V02 secondary eclipse check.
    ax : Axes, optional
        Axes to plot on. Creates new figure if None.
    primary_color : str
        Color for primary transit region shading.
    secondary_color : str
        Color for secondary eclipse region shading.
    show_windows : bool
        Show shaded regions for transit windows.
    zoom_secondary : bool
        If True, zoom axes to secondary window (phase 0.35-0.65).
    style : str
        Style preset.
    **scatter_kwargs
        Additional kwargs passed to ax.scatter().

    Returns
    -------
    Axes
        The matplotlib axes containing the plot.
    """
```

### 5.3 V05: plot_v_shape

```python
def plot_v_shape(
    result: CheckResult,
    *,
    ax: "matplotlib.axes.Axes | None" = None,
    data_color: str = "#1f77b4",
    model_color: str = "#ff7f0e",
    show_trapezoid: bool = True,
    show_ratio: bool = True,
    style: str = "default",
    **scatter_kwargs,
) -> "matplotlib.axes.Axes":
    """Plot binned phase-folded transit with trapezoid model overlay.

    Visualizes transit shape to distinguish planetary (U-shaped) from
    grazing eclipsing binary (V-shaped) transits.

    Parameters
    ----------
    result : CheckResult
        Result from V05 V-shape check.
    ax : Axes, optional
        Axes to plot on. Creates new figure if None.
    data_color : str
        Color for binned flux data points.
    model_color : str
        Color for trapezoid model line.
    show_trapezoid : bool
        Overlay best-fit trapezoid model.
    show_ratio : bool
        Annotate tF/tT ratio on plot.
    style : str
        Style preset.
    **scatter_kwargs
        Additional kwargs passed to ax.scatter() for data.

    Returns
    -------
    Axes
        The matplotlib axes containing the plot.
    """
```

### 5.4 V08: plot_centroid_shift

```python
def plot_centroid_shift(
    result: CheckResult,
    *,
    ax: "matplotlib.axes.Axes | None" = None,
    tpf: "TPFStamp | None" = None,
    show_colorbar: bool = True,
    cbar_label: str = "Flux (e-/s)",
    cbar_kwargs: dict[str, Any] | None = None,
    in_color: str = "#e377c2",       # Pink
    out_color: str = "#17becf",      # Cyan
    target_color: str = "white",
    show_vector: bool = True,
    style: str = "default",
    **imshow_kwargs,
) -> tuple["matplotlib.axes.Axes", "matplotlib.colorbar.Colorbar | None"]:
    """Plot centroid shift diagnostic on TPF thumbnail.

    Shows in-transit and out-of-transit centroid positions overlaid on
    a reference TPF image, with shift vector indicating potential
    off-target transit source.

    Parameters
    ----------
    result : CheckResult
        Result from V08 centroid shift check.
    ax : Axes, optional
        Axes to plot on. Creates new figure if None.
    tpf : TPFStamp, optional
        If provided and result lacks reference_image, compute from TPF.
    show_colorbar : bool
        Display colorbar for flux scale.
    cbar_label : str
        Label for colorbar.
    cbar_kwargs : dict, optional
        Additional kwargs for fig.colorbar().
    in_color : str
        Marker color for in-transit centroid.
    out_color : str
        Marker color for out-of-transit centroid.
    target_color : str
        Marker color for target pixel.
    show_vector : bool
        Draw arrow showing centroid shift vector.
    style : str
        Style preset.
    **imshow_kwargs
        Additional kwargs passed to ax.imshow().

    Returns
    -------
    ax : Axes
        The matplotlib axes containing the plot.
    cbar : Colorbar or None
        The colorbar object, or None if show_colorbar=False.

    Raises
    ------
    ValueError
        If result lacks plot_data and tpf not provided.
    """
```

### 5.5 V09: plot_difference_image

```python
def plot_difference_image(
    result: CheckResult,
    *,
    ax: "matplotlib.axes.Axes | None" = None,
    show_colorbar: bool = True,
    cbar_label: str = "Depth per pixel (ppm)",
    cbar_kwargs: dict[str, Any] | None = None,
    cmap: str = "RdBu_r",
    show_target: bool = True,
    show_max_depth: bool = True,
    style: str = "default",
    **imshow_kwargs,
) -> tuple["matplotlib.axes.Axes", "matplotlib.colorbar.Colorbar | None"]:
    """Plot difference image (out-of-transit minus in-transit).

    Visualizes per-pixel transit depth to localize the transit source.
    Blue indicates flux decrease (transit), red indicates flux increase.

    Parameters
    ----------
    result : CheckResult
        Result from V09 difference image check.
    ax : Axes, optional
        Axes to plot on. Creates new figure if None.
    show_colorbar : bool
        Display colorbar for depth scale.
    cbar_label : str
        Label for colorbar.
    cbar_kwargs : dict, optional
        Additional kwargs for fig.colorbar().
    cmap : str
        Colormap (diverging recommended, centered at zero).
    show_target : bool
        Mark target pixel location.
    show_max_depth : bool
        Mark pixel with maximum depth.
    style : str
        Style preset.
    **imshow_kwargs
        Additional kwargs passed to ax.imshow().

    Returns
    -------
    ax : Axes
        The matplotlib axes containing the plot.
    cbar : Colorbar or None
        The colorbar object, or None if show_colorbar=False.
    """
```

### 5.6 V21: plot_sector_consistency

```python
def plot_sector_consistency(
    result: CheckResult,
    *,
    ax: "matplotlib.axes.Axes | None" = None,
    bar_color: str = "#1f77b4",
    outlier_color: str = "#d62728",
    show_mean: bool = True,
    show_chi2: bool = True,
    style: str = "default",
    **bar_kwargs,
) -> "matplotlib.axes.Axes":
    """Plot per-sector depth comparison for multi-sector targets.

    Bar chart showing transit depth measurements across sectors with
    error bars, highlighting any outlier sectors and displaying
    chi-squared consistency metric.

    Parameters
    ----------
    result : CheckResult
        Result from V21 sector consistency check.
    ax : Axes, optional
        Axes to plot on. Creates new figure if None.
    bar_color : str
        Color for consistent sector bars.
    outlier_color : str
        Color for outlier sector bars.
    show_mean : bool
        Show horizontal line at weighted mean depth.
    show_chi2 : bool
        Annotate chi-squared p-value.
    style : str
        Style preset.
    **bar_kwargs
        Additional kwargs passed to ax.bar().

    Returns
    -------
    Axes
        The matplotlib axes containing the plot.
    """
```

### 5.7 plot_phase_folded (Transit)

```python
def plot_phase_folded(
    lc: LightCurve,
    candidate: Candidate,
    *,
    ax: "matplotlib.axes.Axes | None" = None,
    fit_result: "TransitFitResult | None" = None,
    bin_minutes: float | None = 30.0,
    data_color: str = "#7f7f7f",
    binned_color: str = "#1f77b4",
    model_color: str = "#ff7f0e",
    show_model: bool = True,
    show_binned: bool = True,
    phase_range: tuple[float, float] = (-0.15, 0.15),
    style: str = "default",
    **scatter_kwargs,
) -> "matplotlib.axes.Axes":
    """Plot phase-folded light curve centered on transit.

    The most fundamental vetting visualization: shows the transit signal
    phase-folded at the candidate period with optional binning and
    model overlay.

    Parameters
    ----------
    lc : LightCurve
        Light curve data.
    candidate : Candidate
        Transit candidate with ephemeris.
    ax : Axes, optional
        Axes to plot on. Creates new figure if None.
    fit_result : TransitFitResult, optional
        If provided, overlay fitted transit model.
    bin_minutes : float, optional
        Bin width for time-binning. None disables binning.
    data_color : str
        Color for individual data points.
    binned_color : str
        Color for binned data points.
    model_color : str
        Color for transit model line.
    show_model : bool
        Overlay transit model if fit_result provided.
    show_binned : bool
        Show binned data points in addition to raw.
    phase_range : tuple
        Phase range to display (centered on 0).
    style : str
        Style preset.
    **scatter_kwargs
        Additional kwargs for data scatter.

    Returns
    -------
    Axes
        The matplotlib axes containing the plot.
    """
```

### 5.8 plot_full_lightcurve

```python
def plot_full_lightcurve(
    lc: LightCurve | StitchedLC,
    *,
    ax: "matplotlib.axes.Axes | None" = None,
    candidate: Candidate | None = None,
    color_by_sector: bool = True,
    mark_transits: bool = True,
    transit_color: str = "#d62728",
    separate_panels: bool = False,
    style: str = "default",
    **scatter_kwargs,
) -> "matplotlib.axes.Axes | list[matplotlib.axes.Axes]":
    """Plot full time-series light curve.

    Shows the complete light curve across all sectors with optional
    transit markers and sector coloring.

    Parameters
    ----------
    lc : LightCurve or StitchedLC
        Light curve data. StitchedLC enables sector-aware coloring.
    ax : Axes, optional
        Axes to plot on. Creates new figure if None.
        Ignored if separate_panels=True.
    candidate : Candidate, optional
        If provided, mark transit times.
    color_by_sector : bool
        Use different colors for each sector (StitchedLC only).
    mark_transits : bool
        Add vertical spans at transit times.
    transit_color : str
        Color for transit markers.
    separate_panels : bool
        If True, create vertically stacked subplots per sector.
    style : str
        Style preset.
    **scatter_kwargs
        Additional kwargs for data scatter.

    Returns
    -------
    Axes or list[Axes]
        Single axes if separate_panels=False, list if True.
    """
```

### 5.9 Multi-Sector Handling for Pixel Plots

For pixel-level checks (V08, V09, V10, V20), results are inherently per-sector. The plotting API supports two patterns:

#### Pattern A: Single Result (Default)

```python
# Plot single sector result
ax, cbar = plot_centroid_shift(v08_sector_55)
```

#### Pattern B: Multi-Sector Grid

```python
# Plot multiple sectors in a grid layout
fig = plot_centroid_shift_grid(
    results=[v08_sector_55, v08_sector_56, v08_sector_57],
    ncols=3,
    share_colorbar=True,  # Single colorbar for all panels
)
```

The `*_grid` variants use `compute_subplot_grid()` and apply these conventions:
- Panels sorted by sector number
- Panel titles: "Sector N"
- Shared colorbar when `share_colorbar=True` (uses `cax` positioning)
- Consistent vmin/vmax across panels when sharing colorbar

**Decision**: Rather than overloading single functions to accept `result | list[result]`, provide explicit `*_grid` variants for multi-sector layouts. This keeps the single-result API simple and makes multi-sector intent explicit.

---

## 6. Shared Utilities

### 6.1 _core.py Module

```python
"""bittr_tess_vetter/plotting/_core.py

Shared utilities for plotting functions.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.colorbar
    import matplotlib.figure
    import matplotlib.image


def ensure_ax(
    ax: "matplotlib.axes.Axes | None" = None,
) -> tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Ensure we have valid axes, creating figure if needed.

    Parameters
    ----------
    ax : Axes, optional
        Existing axes to use.

    Returns
    -------
    fig : Figure
        The figure containing the axes.
    ax : Axes
        The axes to plot on.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    return fig, ax


def add_colorbar(
    mappable: "matplotlib.image.AxesImage",
    ax: "matplotlib.axes.Axes",
    *,
    label: str = "",
    **kwargs: Any,
) -> "matplotlib.colorbar.Colorbar":
    """Add colorbar with astronomy-friendly defaults.

    Disables minor ticks following lightkurve convention.

    Parameters
    ----------
    mappable : AxesImage
        The image returned by imshow().
    ax : Axes
        The axes containing the image.
    label : str
        Colorbar label.
    **kwargs
        Additional arguments to fig.colorbar().
        Useful: 'cax', 'shrink', 'pad', 'orientation'.

    Returns
    -------
    Colorbar
        The matplotlib colorbar object.
    """
    fig = ax.get_figure()
    cbar = fig.colorbar(mappable, ax=ax, label=label, **kwargs)
    cbar.ax.yaxis.set_tick_params(tick1On=False, tick2On=False)
    cbar.ax.minorticks_off()
    return cbar


def style_context(style: str):
    """Context manager for applying style without global mutations.

    This is the preferred approach for applying styles within plot functions,
    as it avoids sticky global state in notebooks and tests.

    Parameters
    ----------
    style : str
        One of "default", "paper", "presentation".

    Returns
    -------
    contextmanager
        A matplotlib rc_context with the specified style.

    Examples
    --------
    >>> with style_context("paper"):
    ...     fig, ax = plt.subplots()
    ...     ax.plot(x, y)
    """
    import matplotlib.pyplot as plt
    from ._styles import STYLES

    if style not in STYLES:
        raise ValueError(f"Unknown style '{style}'. Choose from: {list(STYLES.keys())}")

    return plt.rc_context(STYLES[style])


def apply_style(style: str) -> None:
    """Apply a style preset globally to matplotlib.

    WARNING: This mutates global rcParams. Prefer `style_context()` for
    non-sticky style application within plot functions.

    Parameters
    ----------
    style : str
        One of "default", "paper", "presentation".
    """
    import matplotlib.pyplot as plt
    from ._styles import STYLES

    if style not in STYLES:
        raise ValueError(f"Unknown style '{style}'. Choose from: {list(STYLES.keys())}")

    plt.rcParams.update(STYLES[style])


def extract_plot_data(result: "CheckResult", required_keys: list[str]) -> dict[str, Any]:
    """Extract and validate plot_data from CheckResult.

    Parameters
    ----------
    result : CheckResult
        The check result.
    required_keys : list[str]
        Keys that must be present in plot_data.

    Returns
    -------
    dict
        The plot_data dictionary.

    Raises
    ------
    ValueError
        If result lacks plot_data or required keys.
    """
    if result.raw is None or "plot_data" not in result.raw:
        raise ValueError(
            f"Result {result.id} does not contain plot_data. "
            "Re-run the check or ensure check implementation populates raw['plot_data']."
        )

    plot_data = result.raw["plot_data"]
    missing = [k for k in required_keys if k not in plot_data]
    if missing:
        raise ValueError(
            f"Result {result.id} plot_data missing required keys: {missing}"
        )

    return plot_data


def compute_subplot_grid(n: int) -> tuple[int, int]:
    """Compute optimal subplot grid dimensions.

    Parameters
    ----------
    n : int
        Number of subplots needed.

    Returns
    -------
    nrows, ncols : tuple[int, int]
        Grid dimensions.
    """
    import math

    if n <= 2:
        return (1, n)
    elif n <= 4:
        return (2, 2)
    elif n <= 6:
        return (2, 3)
    else:
        return (3, math.ceil(n / 3))


def get_sector_color(sector: int, all_sectors: list[int]) -> str:
    """Get consistent color for a sector.

    Parameters
    ----------
    sector : int
        Sector number.
    all_sectors : list[int]
        All sectors for this target (for index assignment).

    Returns
    -------
    str
        Hex color string (e.g., "#1f77b4").
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    sorted_sectors = sorted(all_sectors)
    idx = sorted_sectors.index(sector)
    # Use tab10 colormap, cycling if > 10 sectors
    rgba = plt.cm.tab10(idx % 10)
    return mcolors.to_hex(rgba)
```

---

## 7. Style System

### 7.1 Style Presets (_styles.py)

```python
"""bittr_tess_vetter/plotting/_styles.py

Style presets and color definitions.
"""

# =============================================================================
# Style Presets
# =============================================================================

STYLES = {
    "default": {
        "figure.figsize": (8, 5),
        "figure.dpi": 100,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
    },
    "paper": {
        # Single-column figure for journals
        "figure.figsize": (3.5, 2.5),
        "figure.dpi": 300,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 0.8,
        "lines.markersize": 3,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    },
    "presentation": {
        # Large figures for slides
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    },
}

# =============================================================================
# Color Scheme
# =============================================================================

COLORS = {
    # Transit states
    "transit": "#1f77b4",           # Blue - in-transit data
    "out_of_transit": "#7f7f7f",    # Gray - out-of-transit data
    "binned": "#1f77b4",            # Blue - binned data

    # Odd/even comparison
    "odd": "#2ca02c",               # Green - odd transits
    "even": "#d62728",              # Red - even transits

    # Models
    "model": "#ff7f0e",             # Orange - fitted model
    "trapezoid": "#ff7f0e",         # Orange - trapezoid model

    # Secondary eclipse
    "secondary": "#9467bd",         # Purple - secondary window
    "primary": "#1f77b4",           # Blue - primary window

    # Centroid
    "centroid_in": "#e377c2",       # Pink - in-transit centroid
    "centroid_out": "#17becf",      # Cyan - out-of-transit centroid
    "target": "#ffffff",            # White - target position

    # Status indicators
    "outlier": "#d62728",           # Red - outlier/flag
    "consistent": "#1f77b4",        # Blue - consistent data
    "warning": "#ff7f0e",           # Orange - warning

    # Multi-sector (accessed via get_sector_color)
    "sector_cmap": "tab10",
}

# =============================================================================
# Colormaps for Images
# =============================================================================

COLORMAPS = {
    "difference": "RdBu_r",         # Diverging for diff images (blue=decrease)
    "flux": "viridis",              # Sequential for flux images
    "depth": "Blues",               # Sequential for depth maps
    "ghost": "RdBu_r",              # Diverging for ghost features
}

# =============================================================================
# Default Labels
# =============================================================================

LABELS = {
    "time": "Time (BTJD)",
    "phase": "Orbital Phase",
    "flux": "Normalized Flux",
    "flux_raw": "Flux (e-/s)",
    "depth_ppm": "Depth (ppm)",
    "epoch": "Epoch",
    "sector": "Sector",
    "pixel_col": "Column (pixels)",
    "pixel_row": "Row (pixels)",
}
```

### 7.2 Using Styles

```python
from bittr_tess_vetter.plotting import plot_odd_even

# Default style
plot_odd_even(result)

# Publication style
plot_odd_even(result, style="paper")

# Presentation style
plot_odd_even(result, style="presentation")

# Or apply globally
from bittr_tess_vetter.plotting._core import apply_style
apply_style("paper")
```

---

## 8. DVR Summary Report

### 8.1 Layout Specification

Based on Kepler DVS one-page summaries, the DVR summary uses an 8-panel layout:

```
+------------------------------------------------------------------+
|  TIC XXXXXXXXX / TOI-XXXX.XX Vetting Summary             [date]  |
+------------------------------------------------------------------+
|  +------------------+  +------------------+  +------------------+ |
|  | A: Full LC       |  | B: Phase-Folded  |  | C: Secondary     | |
|  |    (time series) |  |    (primary)     |  |    Eclipse       | |
|  +------------------+  +------------------+  +------------------+ |
|  +------------------+  +------------------+  +------------------+ |
|  | D: Odd-Even      |  | E: V-Shape       |  | F: Centroid      | |
|  |    Comparison    |  |    (transit)     |  |    Shift         | |
|  +------------------+  +------------------+  +------------------+ |
|  +------------------+  +------------------------------------+    |
|  | G: Depth         |  | H: Metrics Summary Table           |    |
|  |    Stability     |  |    Key results, FPP, dispositions  |    |
|  +------------------+  +------------------------------------+    |
+------------------------------------------------------------------+
```

### 8.2 Function Signature

```python
def plot_vetting_summary(
    bundle: VettingBundleResult,
    lc: LightCurve | StitchedLC,
    candidate: Candidate,
    *,
    figsize: tuple[float, float] = (11, 8.5),
    include_panels: list[str] | None = None,
    title: str | None = None,
    style: str = "default",
    fit_result: "TransitFitResult | None" = None,
    stellar: "StellarParams | None" = None,
) -> "matplotlib.figure.Figure":
    """Generate DVR-style one-page vetting summary.

    Creates a multi-panel figure with key diagnostic plots following
    the Kepler Data Validation Summary (DVS) format.

    Parameters
    ----------
    bundle : VettingBundleResult
        Complete vetting results from vet_candidate().
    lc : LightCurve or StitchedLC
        Light curve data.
    candidate : Candidate
        Transit candidate.
    figsize : tuple
        Figure size in inches (default: letter landscape).
    include_panels : list[str], optional
        Panel IDs to include. Default: all 8 panels (A-H).
        Options: "A" (full_lc), "B" (phase_folded), "C" (secondary),
                 "D" (odd_even), "E" (v_shape), "F" (centroid),
                 "G" (depth_stability), "H" (metrics_table).
    title : str, optional
        Override title. Default: "TIC X / TOI-X.XX Vetting Summary".
    style : str
        Style preset.
    fit_result : TransitFitResult, optional
        Transit fit for model overlay.
    stellar : StellarParams, optional
        Stellar parameters for derived values.

    Returns
    -------
    Figure
        The matplotlib figure containing all panels.

    Examples
    --------
    >>> result = vet_candidate(lc, candidate)
    >>> fig = plot_vetting_summary(result, lc, candidate)
    >>> fig.savefig("TOI-5807_summary.pdf")

    >>> # Select specific panels
    >>> fig = plot_vetting_summary(
    ...     result, lc, candidate,
    ...     include_panels=["B", "D", "E", "H"]
    ... )
    """
```

### 8.3 Panel Mapping

| Panel | Function | Check ID | Fallback |
|-------|----------|----------|----------|
| A | `plot_full_lightcurve()` | N/A | Always available |
| B | `plot_phase_folded()` | N/A | Always available |
| C | `plot_secondary_eclipse()` | V02 | Skip if check not run |
| D | `plot_odd_even()` | V01 | Skip if check not run |
| E | `plot_v_shape()` | V05 | Skip if check not run |
| F | `plot_centroid_shift()` | V08 | Skip if no pixel data |
| G | `plot_depth_stability()` | V04 | Skip if check not run |
| H | `render_metrics_table()` | All | Summary table |

### 8.4 Metrics Table Panel (H)

The metrics table panel displays key results in a formatted table:

```python
def _render_metrics_table(
    ax: "matplotlib.axes.Axes",
    bundle: VettingBundleResult,
    candidate: Candidate,
    stellar: "StellarParams | None" = None,
) -> None:
    """Render summary metrics table in axes.

    Displays:
    - Ephemeris: Period, T0, Duration
    - Depth: ppm, implied Rp/R*
    - FPP/NFPP: If available
    - Key check flags: Any warnings/failures
    - Disposition: If available from V07
    """
```

### 8.5 Export Function

```python
def save_vetting_report(
    bundle: VettingBundleResult,
    lc: LightCurve | StitchedLC,
    candidate: Candidate,
    path: str | Path,
    *,
    format: str = "pdf",
    dpi: int = 300,
    **summary_kwargs,
) -> Path:
    """Save vetting summary to file.

    Parameters
    ----------
    bundle : VettingBundleResult
        Complete vetting results.
    lc : LightCurve or StitchedLC
        Light curve data.
    candidate : Candidate
        Transit candidate.
    path : str or Path
        Output file path (extension determines format if format=None).
    format : str
        Output format: "pdf", "png", "svg".
    dpi : int
        Resolution for raster formats.
    **summary_kwargs
        Additional kwargs passed to plot_vetting_summary().

    Returns
    -------
    Path
        Path to saved file.
    """
```

---

## 9. CheckResult Modifications

### 9.1 Required Changes to Validation Code

Each check implementation must be updated to populate `raw["plot_data"]`.

#### 9.1.1 Example: V01 odd_even_depth

```python
# In src/bittr_tess_vetter/validation/lc_checks.py

def check_odd_even_depth(...) -> CheckResult:
    # ... existing calculation logic ...

    # Prepare plot data (capped arrays for reasonable size)
    MAX_EPOCHS = 50
    odd_epochs = odd_epoch_indices[:MAX_EPOCHS]
    even_epochs = even_epoch_indices[:MAX_EPOCHS]

    plot_data = {
        "odd_epochs": [int(e) for e in odd_epochs],
        "odd_depths_ppm": [float(d) for d in odd_depths[:MAX_EPOCHS]],
        "odd_errs_ppm": [float(e) for e in odd_errs[:MAX_EPOCHS]],
        "even_epochs": [int(e) for e in even_epochs],
        "even_depths_ppm": [float(d) for d in even_depths[:MAX_EPOCHS]],
        "even_errs_ppm": [float(e) for e in even_errs[:MAX_EPOCHS]],
        "mean_odd_ppm": float(mean_odd),
        "mean_even_ppm": float(mean_even),
    }

    return ok_result(
        id="V01",
        name="Odd-Even Depth",
        metrics={...},
        raw={"plot_data": plot_data},
    )
```

### 9.2 Checks Requiring Updates

Module paths match actual codebase structure under `src/bittr_tess_vetter/validation/`:

| Check ID | Module | plot_data Keys |
|----------|--------|----------------|
| V01 | lc_checks.py | odd/even epochs, depths, errors |
| V02 | lc_checks.py | phase, flux, window bounds |
| V03 | lc_checks.py | observed/expected duration |
| V04 | lc_checks.py | epoch times, depths, errors |
| V05 | lc_checks.py | binned phase/flux, trapezoid model |
| V06 | checks_catalog.py | coords, nearby matches |
| V08 | checks_pixel.py | reference_image, centroids |
| V09 | checks_pixel.py | difference_image, depth_map |
| V10 | checks_pixel.py | radii, depths |
| V11 | exovetter_checks.py | phase bins, periodogram |
| V12 | exovetter_checks.py | phase, flux, sinusoid fits |
| V13 | lc_false_alarm_checks.py | epoch coverage |
| V15 | lc_false_alarm_checks.py | phase, left/right bins |
| V16 | (extended checks - TBD) | phase, flux, model fits |
| V17 | ephemeris_reliability.py | phase shifts, scores |
| V19 | alias_diagnostics.py | harmonic periods, scores |
| V20 | ghost_features.py | difference_image, aperture |
| V21 | sector_consistency.py | sectors, depths, errors |

### 9.3 Backward Compatibility

The `raw` field already exists on CheckResult. Adding `plot_data` key is backward compatible:

```python
class CheckResult(BaseModel):
    # ... existing fields ...
    raw: dict[str, Any] | None = None  # Already optional
```

Existing code that doesn't access `raw["plot_data"]` is unaffected.

---

## 10. Testing Strategy

### 10.1 Testing Philosophy

**Prefer structure tests over image baselines.** Image regression tests are valuable but brittle across matplotlib versions, fonts, and platforms. The recommended approach:

1. **Structure tests (primary)**: Verify function returns expected objects, labels exist, no warnings raised, errors on invalid input
2. **Image baselines (selective)**: Reserve for high-risk plots where visual correctness is critical (centroid overlays, difference images, DVR layout)

All tests must:
- Use `pytest.importorskip("matplotlib")` for graceful skipping
- Use `matplotlib.use("Agg")` backend (set in conftest.py)
- Close figures after assertions to prevent memory leaks

### 10.2 Unit Tests

Each plotting function requires unit tests covering:

```python
# tests/test_plotting/test_checks.py

import pytest
from unittest.mock import MagicMock

pytest.importorskip("matplotlib")

from bittr_tess_vetter.plotting import plot_odd_even
from bittr_tess_vetter.validation.result_schema import ok_result


class TestPlotOddEven:
    """Tests for plot_odd_even function."""

    @pytest.fixture
    def mock_result(self):
        """Create mock V01 result with plot_data."""
        return ok_result(
            id="V01",
            name="Odd-Even Depth",
            metrics={"sigma_diff": 1.5},
            raw={
                "plot_data": {
                    "odd_epochs": [1, 3, 5],
                    "odd_depths_ppm": [250.0, 252.0, 248.0],
                    "odd_errs_ppm": [15.0, 14.0, 16.0],
                    "even_epochs": [2, 4, 6],
                    "even_depths_ppm": [245.0, 247.0, 243.0],
                    "even_errs_ppm": [14.0, 15.0, 14.0],
                    "mean_odd_ppm": 250.0,
                    "mean_even_ppm": 245.0,
                }
            },
        )

    def test_creates_figure_when_ax_none(self, mock_result):
        """Test that new figure is created when ax not provided."""
        ax = plot_odd_even(mock_result)
        assert ax is not None
        assert ax.get_figure() is not None

    def test_uses_provided_ax(self, mock_result):
        """Test that provided axes are used."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        returned_ax = plot_odd_even(mock_result, ax=ax)
        assert returned_ax is ax
        plt.close(fig)

    def test_has_correct_labels(self, mock_result):
        """Test axes labels are set correctly."""
        ax = plot_odd_even(mock_result)
        assert "Epoch" in ax.get_xlabel()
        assert "Depth" in ax.get_ylabel()
        plt.close(ax.get_figure())

    def test_has_legend_by_default(self, mock_result):
        """Test legend is present by default."""
        ax = plot_odd_even(mock_result)
        legend = ax.get_legend()
        assert legend is not None
        plt.close(ax.get_figure())

    def test_legend_disabled_when_requested(self, mock_result):
        """Test legend can be disabled."""
        ax = plot_odd_even(mock_result, show_legend=False)
        legend = ax.get_legend()
        assert legend is None
        plt.close(ax.get_figure())

    def test_raises_on_missing_plot_data(self):
        """Test error when plot_data missing."""
        result = ok_result(
            id="V01", name="Odd-Even Depth",
            metrics={"sigma_diff": 1.5},
            raw={},  # No plot_data
        )
        with pytest.raises(ValueError, match="plot_data"):
            plot_odd_even(result)

    def test_custom_colors(self, mock_result):
        """Test custom colors are applied."""
        ax = plot_odd_even(mock_result, odd_color="blue", even_color="red")
        # Verify colors were used (check line colors)
        assert ax is not None
        plt.close(ax.get_figure())
```

### 10.3 Visual Regression Tests (Selective)

Use pytest-mpl image baselines **only** for these high-risk plots:

| Plot | Risk | Rationale |
|------|------|-----------|
| `plot_centroid_shift` | High | Overlay position must be correct |
| `plot_difference_image` | High | Colormap centering, overlay alignment |
| `plot_vetting_summary` | High | Multi-panel layout must not regress |

```python
# tests/test_plotting/test_visual_regression.py

import pytest

pytest.importorskip("matplotlib")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
from bittr_tess_vetter.plotting import plot_centroid_shift, plot_difference_image


@pytest.mark.mpl_image_compare(baseline_dir="baseline_images", tolerance=5)
def test_centroid_shift_visual(mock_v08_result):
    """Visual regression test for centroid shift overlay."""
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_centroid_shift(mock_v08_result, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="baseline_images", tolerance=5)
def test_difference_image_visual(mock_v09_result):
    """Visual regression test for difference image plot."""
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_difference_image(mock_v09_result, ax=ax)
    return fig
```

For all other plots, structure tests are sufficient.

### 10.4 Integration Tests

```python
# tests/test_plotting/test_integration.py

import pytest

pytest.importorskip("matplotlib")

from bittr_tess_vetter.api import (
    vet_candidate, LightCurve, Candidate, Ephemeris
)
from bittr_tess_vetter.plotting import (
    plot_odd_even,
    plot_vetting_summary,
)


class TestPlottingIntegration:
    """Integration tests for plotting with real vetting results."""

    @pytest.fixture
    def vetting_result(self, sample_lc, sample_candidate):
        """Run vetting to get real results."""
        return vet_candidate(sample_lc, sample_candidate)

    def test_plot_from_vetting_result(self, vetting_result):
        """Test plotting from actual vetting output."""
        v01 = vetting_result.get_result("V01")
        if v01 is not None and v01.status == "ok":
            ax = plot_odd_even(v01)
            assert ax is not None
            plt.close(ax.get_figure())

    def test_summary_plot_generation(self, vetting_result, sample_lc, sample_candidate):
        """Test DVR summary generation."""
        fig = plot_vetting_summary(vetting_result, sample_lc, sample_candidate)
        assert fig is not None
        assert len(fig.axes) >= 4  # At least some panels
        plt.close(fig)

    def test_no_matplotlib_warnings(self, vetting_result, caplog):
        """Test that plotting doesn't generate warnings."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v01 = vetting_result.get_result("V01")
            if v01 is not None:
                plot_odd_even(v01)

            mpl_warnings = [x for x in w if "matplotlib" in str(x.category)]
            assert len(mpl_warnings) == 0
```

### 10.5 Test Coverage Requirements

- **Unit tests**: 95%+ line coverage on plotting module
- **Visual regression**: Baseline images for high-risk plots only (V08, V09, DVR summary)
- **Integration**: At least one test per public function with real data
- **Edge cases**: Empty data, single point, missing optional fields

---

## 11. Documentation Plan

### 11.1 API Reference

All public functions documented with NumPy-style docstrings:

```python
def plot_odd_even(...):
    """Plot odd vs even transit depth comparison.

    [Description paragraph]

    Parameters
    ----------
    result : CheckResult
        [Parameter description]
    ax : matplotlib.axes.Axes, optional
        [Parameter description]

    Returns
    -------
    matplotlib.axes.Axes
        [Return description]

    Raises
    ------
    ValueError
        [Exception condition]

    See Also
    --------
    plot_secondary_eclipse : Related function
    plot_depth_stability : Related function

    Notes
    -----
    [Implementation notes]

    Examples
    --------
    >>> result = run_check("V01", lc, candidate)
    >>> ax = plot_odd_even(result)
    """
```

### 11.2 Tutorial Updates

#### Tutorial 10 (TOI-5807) Additions

Add plotting cells after each check execution:

```python
# Cell after V01
from bittr_tess_vetter.plotting import plot_odd_even

v01 = bundle.get_result("V01")
if v01:
    ax = plot_odd_even(v01)
    plt.title(f"V01: Odd-Even Depth (sigma={v01.metrics.get('sigma_diff', 'N/A'):.2f})")
    plt.show()
```

#### New Tutorial 11: Visualizing Vetting Results

```markdown
# Tutorial 11: Visualizing Vetting Results

This tutorial demonstrates:
1. Generating individual diagnostic plots
2. Customizing plot appearance for publications
3. Creating DVR-style summary reports
4. Exporting plots for papers and presentations

## Contents
1. Quick Start: One-Line Plots
2. Customizing Individual Plots
3. Multi-Panel Layouts
4. DVR Summary Reports
5. Publication-Quality Export
```

### 11.3 Example Gallery

Create `docs/gallery/` with PNG examples:

```
docs/
  gallery/
    README.md
    plot_odd_even.png
    plot_secondary_eclipse.png
    plot_v_shape.png
    plot_centroid_shift.png
    plot_difference_image.png
    plot_aperture_curve.png
    plot_vetting_summary.png
    ...
```

Each image accompanied by code snippet:

```markdown
## plot_odd_even

![Odd-Even Comparison](plot_odd_even.png)

```python
from bittr_tess_vetter.plotting import plot_odd_even
ax = plot_odd_even(result)
```
```

---

## 12. Implementation Phases

### 12.1 Phase 1: Foundation (Week 1)

**Objective**: Establish module structure and core utilities.

| Task | Files | Deliverables |
|------|-------|--------------|
| Create plotting subpackage | `plotting/__init__.py` | Package skeleton with guards |
| Add optional dependency | `pyproject.toml` | `[plotting]` extra |
| Implement core utilities | `plotting/_core.py` | `ensure_ax()`, `add_colorbar()`, `extract_plot_data()` |
| Define style system | `plotting/_styles.py` | STYLES, COLORS, COLORMAPS |
| Implement phase-folded | `plotting/transit.py` | `plot_phase_folded()` |
| Update V01 for plot_data | `validation/lc_checks.py` | V01 populates `raw["plot_data"]` |
| Implement plot_odd_even | `plotting/checks.py` | `plot_odd_even()` |
| Basic tests | `tests/test_plotting/` | Unit tests for core + V01 |

**Exit Criteria**:
- `from bittr_tess_vetter.plotting import plot_odd_even, plot_phase_folded` works
- Tests pass, 95%+ coverage on new code

### 12.2 Phase 2: Essential LC Checks (Week 2)

**Objective**: Complete light curve check plots.

| Task | Files | Deliverables |
|------|-------|--------------|
| V02 plot_data | `validation/lc_checks.py` | Update secondary_eclipse |
| plot_secondary_eclipse | `plotting/checks.py` | Function implementation |
| V04 plot_data | `validation/lc_checks.py` | Update depth_stability |
| plot_depth_stability | `plotting/checks.py` | Function implementation |
| V05 plot_data | `validation/lc_checks.py` | Update v_shape |
| plot_v_shape | `plotting/checks.py` | Function implementation |
| V03 plot_data | `validation/lc_checks.py` | Update duration_consistency |
| plot_duration_consistency | `plotting/checks.py` | Function implementation |
| Tests | `tests/test_plotting/` | Unit + visual regression |

**Exit Criteria**: All V01-V05 have plot functions and tests.

### 12.3 Phase 3: Pixel-Level Checks (Week 3)

**Objective**: Complete pixel diagnostic plots.

| Task | Files | Deliverables |
|------|-------|--------------|
| V08 plot_data | `validation/checks_pixel.py` | Update centroid_shift |
| plot_centroid_shift | `plotting/pixel.py` | Function with colorbar |
| V09 plot_data | `validation/checks_pixel.py` | Update difference_image |
| plot_difference_image | `plotting/pixel.py` | Function with colorbar |
| V10 plot_data | `validation/checks_pixel.py` | Update aperture_dependence |
| plot_aperture_curve | `plotting/pixel.py` | Function implementation |
| V06 plot_data | `validation/checks_catalog.py` | Update nearby_eb_search |
| plot_nearby_ebs | `plotting/catalog.py` | Sky map visualization |
| Tests | `tests/test_plotting/` | Unit + visual + integration |

**Exit Criteria**: Pixel plots work with stored plot_data and optional TPF parameter.

### 12.4 Phase 4: Exovetter and False Alarm (Week 4)

**Objective**: Complete V11-V15 plots.

| Task | Files | Deliverables |
|------|-------|--------------|
| V11 plot_data | `validation/exovetter_checks.py` | Update modshift |
| plot_modshift | `plotting/exovetter.py` | Periodogram visualization |
| V12 plot_data | `validation/exovetter_checks.py` | Update sweet |
| plot_sweet | `plotting/exovetter.py` | Sinusoid overlay |
| V13 plot_data | `validation/lc_false_alarm_checks.py` | Update data_gaps |
| plot_data_gaps | `plotting/false_alarm.py` | Epoch coverage heatmap |
| V15 plot_data | `validation/lc_false_alarm_checks.py` | Update asymmetry |
| plot_asymmetry | `plotting/false_alarm.py` | Left/right comparison |
| Tests | `tests/test_plotting/` | Full test coverage |

**Exit Criteria**: V11-V15 plots complete with tests.

### 12.5 Phase 5: Extended Checks (Week 5)

**Objective**: Complete V16-V21 plots.

| Task | Files | Deliverables |
|------|-------|--------------|
| V16 plot_data | `validation/checks_extended.py` | Update model_competition |
| plot_model_comparison | `plotting/extended.py` | Multi-model overlay |
| V17 plot_data | `validation/checks_extended.py` | Update ephemeris_reliability |
| plot_ephemeris_reliability | `plotting/extended.py` | Score distribution |
| V19 plot_data | `validation/checks_extended.py` | Update alias_diagnostics |
| plot_alias_diagnostics | `plotting/extended.py` | Harmonic bar chart |
| V20 plot_data | `validation/ghost_features.py` | Update ghost_features |
| plot_ghost_features | `plotting/extended.py` | Ghost diagnostic |
| V21 plot_data | `validation/sector_consistency.py` | Update sector_consistency |
| plot_sector_consistency | `plotting/extended.py` | Per-sector bar chart |
| Tests | `tests/test_plotting/` | Full test coverage |

**Exit Criteria**: All V01-V21 have corresponding plot functions.

### 12.6 Phase 6: DVR Summary Report (Week 6)

**Objective**: Multi-panel summary and export.

| Task | Files | Deliverables |
|------|-------|--------------|
| Full LC plot | `plotting/lightcurve.py` | `plot_full_lightcurve()` |
| Metrics table | `plotting/report.py` | `_render_metrics_table()` |
| Summary layout | `plotting/report.py` | `plot_vetting_summary()` |
| Export function | `plotting/report.py` | `save_vetting_report()` |
| Integration tests | `tests/test_plotting/` | Full pipeline tests |
| Tutorial 10 update | `docs/tutorials/` | Add plotting cells |

**Exit Criteria**: DVR summary generates complete 8-panel figure.

### 12.7 Phase 7: Documentation and Polish (Week 7)

**Objective**: Complete documentation and quality.

| Task | Files | Deliverables |
|------|-------|--------------|
| API docstrings | All plotting modules | Complete NumPy-style docs |
| Tutorial 11 | `docs/tutorials/` | New plotting tutorial |
| Example gallery | `docs/gallery/` | PNG examples + code |
| Visual regression baselines | `tests/baseline_images/` | Reference images |
| Code review | All plotting modules | Final cleanup |
| Performance check | N/A | Verify no memory leaks |

**Exit Criteria**: Documentation complete, all tests pass, ready for release.

### 12.8 Future Phases (Post-MVP)

#### Phase 8: ipywidgets Integration (Optional)

- Interactive zoom sliders for phase-folded plots
- Parameter adjustment widgets for transit fitting

#### Phase 9: Enhanced Reports (Optional)

- Multi-page PDF reports with additional diagnostics
- HTML report generation with embedded images
- LaTeX table export for papers

#### Phase 10: Multi-Backend Support (Deferred)

- Bokeh backend for interactive notebooks
- Plotly backend for web dashboards
- Reserved `backend=` parameter already in API

---

## Appendix A: Research Document Summary

This specification synthesizes findings from:

| Document | Key Decisions |
|----------|---------------|
| `dr_darin_feedback.md` | Motivation for visual diagnostics |
| `research_kepler_dvr_plots.md` | DVS layout, plot priorities |
| `research_check_specific_plots.md` | Per-check plot recommendations |
| `research_api_integration.md` | Architecture (Option C+D hybrid) |
| `research_astronomy_plotting_patterns.md` | Function-based API, ax handling |
| `consolidated_plotting_implementation_plan.md` | Initial implementation plan |
| `research_q1_plot_data_opt_in.md` | Always include plot_data |
| `research_q2_tpf_retention.md` | Store derived 2D stamps |
| `research_q3_interactive_backends.md` | Matplotlib-only Phase 1 |
| `research_q4_colorbar_pattern.md` | show_colorbar=True default, cbar_kwargs |
| `research_q5_multi_sector_plots.md` | Context-dependent multi-sector |

---

## Appendix B: Naming Conventions

Following existing codebase patterns from `api/__init__.py` and `api/types.py`:

| Pattern | Example | Notes |
|---------|---------|-------|
| Function names | `plot_odd_even()` | snake_case, verb_noun |
| Check plot names | `plot_<check_name>()` | Match check function names |
| Parameter names | `show_colorbar`, `cbar_kwargs` | snake_case |
| Style presets | `"default"`, `"paper"` | lowercase strings |
| Color constants | `COLORS["transit"]` | SCREAMING_SNAKE dict keys |

---

## Appendix C: Dependencies

### Required (with [plotting] extra)

```
matplotlib>=3.5.0
```

### Recommended (already in core)

```
numpy>=1.20.0
astropy>=5.0  # For ImageNormalize, PercentileInterval
```

### Development

```
pytest-mpl>=0.16.0  # Visual regression testing
```

---

## Appendix D: Feedback Decisions

This appendix documents decisions made in response to the consolidated feedback review (2026-01-20).

### Accepted Feedback (Incorporated Above)

| Feedback Item | Decision | Location in Spec |
|---------------|----------|------------------|
| Fix module paths to match codebase | **Accepted** - Updated all references | Section 9.2 |
| Use `raw["plot_data"]` consistently | **Accepted** - Already canonical | Sections 4.1, 4.7 |
| Fix `get_sector_color()` return type | **Accepted** - Now returns hex string | Section 6.1 |
| Prefer `rc_context` over global rcParams | **Accepted** - Added `style_context()` | Section 6.1 |
| Add coordinate conventions for pixel data | **Accepted** - Full subsection added | Section 4.5 |
| Standardize key naming with unit suffixes | **Accepted** - Convention table added | Section 4.4 |
| Add plot_data version marker | **Accepted** - `version: 1` required | Section 4.2 |
| Enforce JSON serializability | **Accepted** - Conversion requirements documented | Section 4.3 |
| Define array size caps | **Accepted** - Max sizes documented | Section 4.6 |
| Prefer structure tests over image baselines | **Accepted** - Testing philosophy updated | Section 10.1 |
| Selective pytest-mpl usage | **Accepted** - Only for V08, V09, DVR summary | Section 10.3 |
| Multi-sector pixel plot support | **Accepted** - `*_grid` variant pattern | Section 5.9 |

### Deferred Feedback (Valid but Out of Scope)

| Feedback Item | Rationale for Deferral |
|---------------|------------------------|
| Bump matplotlib baseline to modern version | Requires broader compatibility testing; current >=3.5.0 is fine for Phase 1 |
| cax support in add_colorbar for multi-panel | Already documented via `**kwargs`; explicit `cax` param can be added in Phase 3 if needed |

### Rejected Feedback (With Rationale)

| Feedback Item | Decision | Rationale |
|---------------|----------|-----------|
| Keep plotting under `bittr_tess_vetter.plotting` only (Option A) | **Rejected** - Keep current re-export from `api` | The conditional re-export from `api` is intentional and consistent with how other optional features (MLX, exovetter) are handled. Error messaging is already consistent. Power users can import directly from `plotting` if preferred. |
| Provide `*_multi_sector` wrappers instead of `*_grid` naming | **Rejected** - Use `*_grid` suffix | "grid" more accurately describes the output (subplot grid), while "multi_sector" implies semantic meaning that may not apply to all use cases. The `_grid` suffix is also more concise. |

### Implementation Notes

1. **Style context usage**: Plot functions should wrap their rendering logic in `with style_context(style):` rather than calling `apply_style()` globally. This prevents style leakage between plots in notebooks.

2. **Coordinate consistency**: All pixel plots must use `origin="lower"` in `imshow()` calls. The `_core.py` module should provide an `imshow_astronomical()` helper that enforces this.

3. **Version bumping**: When `plot_data` schema changes for a check, increment the `version` field and add migration notes to the check's docstring.
