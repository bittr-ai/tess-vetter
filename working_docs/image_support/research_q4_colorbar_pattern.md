# Research Q4: Colorbar Handling Pattern for Astronomy Plotting

**Date:** 2026-01-20
**Status:** Research Complete
**Addresses:** Open Question #4 from `consolidated_plotting_implementation_plan.md`

---

## 1. Executive Summary

Colorbars are essential for difference images (V09), ghost features (V20), river plots, and 2D heatmaps. After analyzing lightkurve, astropy.visualization, and matplotlib best practices, the recommended pattern is:

**Recommendation:** Use an optional `show_colorbar=True` default with a `cbar_kwargs` dictionary for customization. Always return the colorbar object (or None) alongside the axes.

---

## 2. How Lightkurve Handles Colorbars

### 2.1 TPF.plot() - Image Data

From `lightkurve/targetpixelfile.py` (lines 1058-1203):

```python
def plot(
    self,
    ax=None,
    frame=0,
    show_colorbar=True,  # Default: ON for images
    ...
):
    ax = plot_image(
        data_to_plot,
        ax=ax,
        show_colorbar=show_colorbar,
        clabel=clabels.get(column, column),
        **kwargs,
    )
    return ax
```

**Key pattern:** `show_colorbar=True` by default for 2D image data.

### 2.2 plot_image() Utility

From `lightkurve/utils.py` (lines 443-537):

```python
def plot_image(
    image,
    ax=None,
    scale="linear",
    clabel="Flux ($e^{-}s^{-1}$)",
    show_colorbar=True,
    vmin=None,
    vmax=None,
    **kwargs
):
    # ... normalization setup ...
    cax = ax.imshow(image, origin=origin, norm=norm, **kwargs)

    if show_colorbar:
        cbar = plt.colorbar(cax, ax=ax, label=clabel)
        cbar.ax.yaxis.set_tick_params(tick1On=False, tick2On=False)
        cbar.ax.minorticks_off()

    return ax  # Only returns axes, not colorbar
```

**Observations:**
- Colorbar label (`clabel`) is a dedicated parameter
- Minor ticks disabled for cleaner appearance
- Uses `plt.colorbar()` which finds figure from axes
- Does NOT return the colorbar object

### 2.3 LightCurve.scatter() - Color Dimension

From `lightkurve/lightcurve.py` (lines 1851-2027):

```python
def _create_plot(
    self,
    show_colorbar=True,
    colorbar_label="",
    **kwargs,
):
    if method == "scatter":
        sc = ax.scatter(time.value, flux, **kwargs)
        # Only show colorbar if 'c' argument provided and is array-like
        if (
            show_colorbar
            and ("c" in kwargs)
            and (not isinstance(kwargs["c"], str))
            and hasattr(kwargs["c"], "__iter__")
        ):
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(colorbar_label)
            cbar.ax.yaxis.set_tick_params(tick1On=False, tick2On=False)
            cbar.ax.minorticks_off()
```

**Key insight:** For scatter plots, colorbar is conditional on having array-like color data.

---

## 3. Astropy.visualization Pattern

### 3.1 Normalization for Colorbars

From astropy documentation, the critical pattern for colorbar accuracy:

```python
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch

# CORRECT: Pass norm to imshow, colorbar shows original values
norm = ImageNormalize(image, interval=PercentileInterval(95), stretch=SqrtStretch())
im = ax.imshow(image, origin='lower', norm=norm)
fig.colorbar(im)  # Ticks show original data values

# WRONG: Apply norm to data, colorbar shows transformed values
ax.imshow(norm(image))  # DO NOT DO THIS
```

### 3.2 Lightkurve's Usage of Astropy

Lightkurve uses astropy's normalization classes:

```python
from astropy.visualization import PercentileInterval, ImageNormalize, SqrtStretch, LinearStretch

if scale == "sqrt":
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch(), clip=False)
```

---

## 4. Standard Matplotlib Pattern for Optional Colorbars

### 4.1 Figure vs Axes Colorbar

```python
# Method 1: Figure-level (simpler, works with plt.subplots)
fig, ax = plt.subplots()
im = ax.imshow(data)
fig.colorbar(im, ax=ax)

# Method 2: Axes-level (more control)
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)
```

### 4.2 When User Provides Axes

The challenge: `fig.colorbar()` requires the figure. When user provides `ax`, we must get the figure:

```python
def plot_something(data, *, ax=None, show_colorbar=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()  # Get figure from provided axes

    im = ax.imshow(data)

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax)

    return ax
```

---

## 5. Should Colorbar Be On by Default?

### 5.1 Analysis by Plot Type

| Plot Type | Default | Rationale |
|-----------|---------|-----------|
| Difference Image (V09) | `show_colorbar=True` | Depth values are critical for interpretation |
| Ghost Features (V20) | `show_colorbar=True` | Uniformity values need quantitative reference |
| River Plot | `show_colorbar=True` | Time dimension needs colorbar for context |
| 2D Heatmap | `show_colorbar=True` | Any 2D visualization needs scale reference |
| Scatter with color | `show_colorbar=True` (if `c` provided) | Follows lightkurve pattern |

### 5.2 Recommendation

**Default ON** for all image/heatmap plots. Scientists need the quantitative reference. Users can disable with `show_colorbar=False` when creating multi-panel figures where a shared colorbar is preferred.

---

## 6. Handling User-Provided Axes

### 6.1 The Problem

When user provides their own axes (e.g., for multi-panel figures), `fig.colorbar(im, ax=ax)` will "steal" space from the provided axes, which may not be desired.

### 6.2 Solutions

**Option A: Automatic behavior (lightkurve approach)**
```python
def plot_difference_image(result, *, ax=None, show_colorbar=True, ...):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if show_colorbar:
        fig.colorbar(im, ax=ax)  # May resize the provided axes
```

**Option B: cax parameter for precise control**
```python
def plot_difference_image(result, *, ax=None, cax=None, show_colorbar=True, ...):
    """
    Parameters
    ----------
    ax : Axes, optional
        Axes for the image
    cax : Axes, optional
        Axes for colorbar. If None and show_colorbar=True,
        colorbar axes created automatically.
    """
    if show_colorbar:
        if cax is not None:
            fig.colorbar(im, cax=cax)
        else:
            fig.colorbar(im, ax=ax)
```

**Option C: cbar_kwargs for full customization**
```python
def plot_difference_image(result, *, ax=None, show_colorbar=True, cbar_kwargs=None, ...):
    """
    Parameters
    ----------
    cbar_kwargs : dict, optional
        Keyword arguments passed to fig.colorbar().
        Useful keys: 'cax', 'shrink', 'pad', 'label', 'orientation'
    """
    if show_colorbar:
        cbar_kw = cbar_kwargs or {}
        fig.colorbar(im, ax=ax, **cbar_kw)
```

### 6.3 Recommendation

**Use Option C** (`cbar_kwargs`). It provides maximum flexibility while keeping the simple case simple:

```python
# Simple case - automatic colorbar
plot_difference_image(result)

# Multi-panel with shared colorbar - disable automatic
fig, axes = plt.subplots(1, 3)
for i, ax in enumerate(axes[:-1]):
    plot_difference_image(result, ax=ax, show_colorbar=False)
cax = axes[-1]  # Use last axes for colorbar
# Add colorbar manually
```

---

## 7. Concrete Code Pattern Recommendation

### 7.1 For `_core.py` - Shared Colorbar Helper

```python
"""bittr_tess_vetter/plotting/_core.py"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.colorbar
    import matplotlib.image


def add_colorbar(
    mappable: "matplotlib.image.AxesImage",
    ax: "matplotlib.axes.Axes",
    *,
    label: str = "",
    **kwargs: Any,
) -> "matplotlib.colorbar.Colorbar":
    """Add colorbar to image plot with astronomy-friendly defaults.

    Parameters
    ----------
    mappable : AxesImage
        The image returned by imshow()
    ax : Axes
        The axes containing the image
    label : str
        Colorbar label
    **kwargs
        Additional arguments to fig.colorbar()
        Useful: 'cax', 'shrink', 'pad', 'orientation'

    Returns
    -------
    Colorbar
        The matplotlib colorbar object
    """
    fig = ax.get_figure()
    cbar = fig.colorbar(mappable, ax=ax, label=label, **kwargs)
    # Astronomy convention: clean tick appearance
    cbar.ax.yaxis.set_tick_params(tick1On=False, tick2On=False)
    cbar.ax.minorticks_off()
    return cbar
```

### 7.2 For Image Plotting Functions

```python
"""Example: plot_difference_image in plotting/pixel.py"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.colorbar
    from bittr_tess_vetter.validation.result_schema import CheckResult

from ._core import add_colorbar


def plot_difference_image(
    result: "CheckResult",
    *,
    ax: "matplotlib.axes.Axes | None" = None,
    show_colorbar: bool = True,
    cbar_label: str = "Depth per pixel (ppm)",
    cbar_kwargs: dict[str, Any] | None = None,
    cmap: str = "RdBu_r",
    **imshow_kwargs: Any,
) -> tuple["matplotlib.axes.Axes", "matplotlib.colorbar.Colorbar | None"]:
    """Plot difference image from V09 check result.

    Parameters
    ----------
    result : CheckResult
        Result from V09 difference image check
    ax : Axes, optional
        Axes to plot on. Creates new figure if None.
    show_colorbar : bool
        Whether to display colorbar. Default True.
    cbar_label : str
        Label for colorbar
    cbar_kwargs : dict, optional
        Additional kwargs for colorbar (e.g., 'shrink', 'pad', 'cax')
    cmap : str
        Colormap for image
    **imshow_kwargs
        Additional kwargs for imshow()

    Returns
    -------
    ax : Axes
        The axes containing the plot
    cbar : Colorbar or None
        The colorbar object, or None if show_colorbar=False

    Examples
    --------
    # Simple case
    ax, cbar = plot_difference_image(result)

    # Custom colorbar position
    fig, (ax_img, ax_cbar) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 0.05]})
    ax, cbar = plot_difference_image(result, ax=ax_img, cbar_kwargs={'cax': ax_cbar})

    # Multi-panel, shared colorbar
    fig, axes = plt.subplots(1, 3)
    for ax in axes:
        plot_difference_image(result, ax=ax, show_colorbar=False)
    # Add shared colorbar manually
    """
    import matplotlib.pyplot as plt
    from astropy.visualization import ImageNormalize, PercentileInterval, LinearStretch

    if ax is None:
        fig, ax = plt.subplots()

    # Extract data
    plot_data = result.raw.get("plot_data", {})
    depth_map = plot_data.get("depth_map")
    if depth_map is None:
        raise ValueError(f"Result {result.id} missing plot_data['depth_map']")

    # Setup normalization (symmetric around zero for difference)
    vmax = max(abs(depth_map.min()), abs(depth_map.max()))
    norm = ImageNormalize(
        vmin=-vmax, vmax=vmax,
        stretch=LinearStretch()
    )

    # Plot image
    im = ax.imshow(
        depth_map,
        origin='lower',
        norm=norm,
        cmap=cmap,
        **imshow_kwargs
    )

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(f"V09: Difference Image")

    # Add colorbar
    cbar = None
    if show_colorbar:
        cbar_kw = cbar_kwargs or {}
        cbar = add_colorbar(im, ax, label=cbar_label, **cbar_kw)

    return ax, cbar
```

### 7.3 For Scatter Plots with Color Dimension

```python
def plot_depth_vs_epoch(
    result: "CheckResult",
    *,
    ax: "matplotlib.axes.Axes | None" = None,
    c: "ArrayLike | None" = None,  # Color array (e.g., SNR values)
    show_colorbar: bool = True,
    cbar_label: str = "",
    cbar_kwargs: dict[str, Any] | None = None,
    **scatter_kwargs: Any,
) -> tuple["matplotlib.axes.Axes", "matplotlib.colorbar.Colorbar | None"]:
    """
    Colorbar only shown if c is provided and is array-like.
    Follows lightkurve scatter() convention.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    # ... extract data ...

    sc = ax.scatter(epochs, depths, c=c, **scatter_kwargs)

    # Colorbar only if color array provided
    cbar = None
    if (
        show_colorbar
        and c is not None
        and not isinstance(c, str)
        and hasattr(c, "__iter__")
    ):
        cbar_kw = cbar_kwargs or {}
        cbar = add_colorbar(sc, ax, label=cbar_label, **cbar_kw)

    return ax, cbar
```

---

## 8. Return Value Convention

### 8.1 The Question

Should plotting functions return:
- Just `ax` (lightkurve pattern)
- Tuple of `(ax, cbar)`
- A structured object

### 8.2 Recommendation

**Return `(ax, cbar)`** for image plots. Rationale:

1. Users often need to customize the colorbar after creation
2. `cbar` is `None` when `show_colorbar=False`, so tuple unpacking still works
3. Consistent with the pattern of returning "what was created"

For line plots without colorbars, return just `ax` (no tuple needed).

```python
# Image plots
ax, cbar = plot_difference_image(result)
if cbar:
    cbar.set_label("Custom label")

# Line plots
ax = plot_odd_even(result)
```

---

## 9. Summary of Recommendations

| Aspect | Recommendation |
|--------|----------------|
| Default state | `show_colorbar=True` for all image/heatmap plots |
| Customization | `cbar_kwargs` dict for flexibility |
| Label parameter | Dedicated `cbar_label` parameter with sensible default |
| Return value | `(ax, cbar)` tuple for image plots; `cbar` is None if disabled |
| User-provided axes | Works via `ax.get_figure()` |
| User-provided cax | Via `cbar_kwargs={'cax': custom_axes}` |
| Appearance | Disable minor ticks (astronomy convention via `add_colorbar` helper) |
| Normalization | Use astropy's `ImageNormalize` with norm passed to `imshow()` |

---

## 10. Affected Functions

Based on the implementation plan, these plotting functions will need colorbar support:

| Function | Colorbar Label | Notes |
|----------|---------------|-------|
| `plot_difference_image()` | "Depth per pixel (ppm)" | V09, symmetric colormap |
| `plot_ghost_uniformity()` | "Depth ratio" | V20 |
| `plot_river()` | "Time (BTJD)" | Color = time dimension |
| `plot_aperture_heatmap()` | "Flux (e-/s)" | If implemented as 2D |
| `plot_centroid_shift()` | "Flux (e-/s)" | TPF background image |

---

## 11. References

- Lightkurve source: `targetpixelfile.py`, `lightcurve.py`, `utils.py`
- Astropy visualization: https://docs.astropy.org/en/stable/visualization/normalization.html
- Matplotlib colorbar: https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.colorbar.html
- TRICERATOPS vendor code: `triceratops.py` lines 424, 451 (existing colorbar usage in project)
