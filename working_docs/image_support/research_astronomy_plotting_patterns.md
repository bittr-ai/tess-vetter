# Astronomy Plotting Patterns Research

**Date:** 2026-01-20
**Purpose:** Document common plotting patterns in astronomy Python libraries for adoption in bittr-tess-vetter

---

## 1. Lightkurve Patterns (NASA/TESS Standard)

### 1.1 Method-on-Object Pattern
Lightkurve uses methods directly on data objects:
```python
lc = lk.search_lightcurve('TIC 123456789').download()
lc.plot()          # Returns matplotlib Axes
lc.scatter(c=lc.time)  # Scatter with color dimension
lc.plot_river(period=3.5)  # Specialized river plot
```

### 1.2 Shared Internal Implementation
Both `plot()` and `scatter()` delegate to a shared `_create_plot()` method:
```python
def plot(self, **kwargs):
    return self._create_plot(method='plot', **kwargs)

def scatter(self, colorbar_label='', show_colorbar=True, **kwargs):
    return self._create_plot(method='scatter', ...)
```

### 1.3 Axes Handling Convention
```python
def _create_plot(self, ax=None, ...):
    if ax is None:
        fig, ax = plt.subplots(1)
    # ... plotting code ...
    return ax
```

### 1.4 Key Patterns
| Pattern | Lightkurve Approach |
|---------|---------------------|
| Return type | Always returns `matplotlib.axes.Axes` |
| ax parameter | Optional; creates figure if None |
| Style | Uses `plt.style.context(style)` with custom stylesheet |
| Normalization | Built-in `.normalize()` before plotting |
| kwargs pass-through | Forwards unknown kwargs to matplotlib |
| Matplotlib import | Module-level (treated as required) |

### 1.5 Specialized Plot Types
- `plot()` - Standard line plot
- `scatter()` - Scatter with optional colorbar
- `plot_river()` - Period-folded river diagram (TTV visualization)
- `fold().plot()` - Phase-folded light curve

---

## 2. Astropy Visualization Patterns

### 2.1 Module Organization
```
astropy.visualization/
    __init__.py
    units.py          # Quantity support for axes
    time.py           # Time support for axes
    wcsaxes/          # WCS coordinate plotting
    stretch.py        # Image stretching
    interval.py       # Image interval (ZScale, etc.)
    mpl_style.py      # astropy_mpl_style
```

### 2.2 Opt-In Feature Activation
```python
from astropy.visualization import quantity_support, time_support

quantity_support()  # Enable unit-aware axes
time_support()      # Enable Time-aware axes
```

### 2.3 Style Support
```python
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
```

### 2.4 WCSAxes for Coordinate Plotting
```python
from astropy.visualization import wcsaxes
ax = plt.subplot(projection=wcs)
ax.imshow(data)
ax.coords.grid(color='white')
```

---

## 3. ArviZ Multi-Backend Pattern

### 3.1 Backend-Agnostic Design
ArviZ supports multiple plotting backends (matplotlib, bokeh, plotly):
```python
import arviz as az
az.rcParams["plot.backend"] = "matplotlib"  # or "bokeh", "plotly"
az.plot_trace(trace)
```

### 3.2 Optional Dependency Handling
ArviZ-plots has **no plotting library as a hard dependency**:
- Core (`arviz-base`): No I/O library required
- Plots (`arviz-plots`): matplotlib/bokeh/plotly all optional

```python
# Internal pattern
def get_plotting_function(backend, plot_name):
    """Dynamically load backend-specific function."""
    if backend == "matplotlib":
        from arviz.plots.backends.matplotlib import plot_name
    ...
```

### 3.3 Function-Based API
ArviZ uses standalone functions, not methods on objects:
```python
az.plot_trace(trace)           # Not trace.plot()
az.plot_posterior(trace)
az.summary(trace)
```

---

## 4. Pattern Comparison Summary

| Aspect | Lightkurve | Astropy | ArviZ |
|--------|------------|---------|-------|
| **API Style** | Methods on objects | Standalone functions + integrations | Standalone functions |
| **matplotlib** | Required | Optional (for some features) | Optional (multi-backend) |
| **Return type** | Axes | Varies | Axes or Figure |
| **ax parameter** | Optional, creates if None | Standard matplotlib | Optional |
| **Styling** | Custom stylesheet | `astropy_mpl_style` | rcParams |
| **Organization** | Single module | Subpackage hierarchy | Backend subpackages |

---

## 5. Recommended Patterns for bittr-tess-vetter

### 5.1 Function-Based API (Like ArviZ)
Given that results are Pydantic models (data-focused), standalone functions are more appropriate:
```python
from bittr_tess_vetter.api import plot_odd_even

result = run_check_v01(lc, candidate)
ax = plot_odd_even(result)  # Standalone function
```

### 5.2 Optional Dependency Guard (Like ArviZ)
```python
# plotting/__init__.py
import importlib.util

MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None

if not MATPLOTLIB_AVAILABLE:
    def __getattr__(name):
        raise ImportError(
            f"Plotting requires matplotlib. "
            f"Install with: pip install 'bittr-tess-vetter[plotting]'"
        )
```

### 5.3 Axes Handling Convention (Like Lightkurve)
```python
def plot_odd_even(result, *, ax=None, **kwargs):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    # ... plotting ...
    return ax
```

### 5.4 Style System
```python
# plotting/_styles.py
BITTR_STYLE = {
    'figure.figsize': (8, 5),
    'axes.labelsize': 12,
    'lines.linewidth': 1.5,
    # ... astronomy-friendly defaults
}

def use_bittr_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update(BITTR_STYLE)
```

### 5.5 kwargs Pass-Through
```python
def plot_phase_folded(result, *, ax=None, color='C0', **mpl_kwargs):
    # ... setup ...
    ax.plot(phase, flux, color=color, **mpl_kwargs)
    return ax
```

---

## 6. Concrete Implementation Template

```python
"""bittr_tess_vetter/plotting/checks.py"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.axes
    from bittr_tess_vetter.validation.result_schema import CheckResult


def plot_odd_even(
    result: "CheckResult",
    *,
    ax: "matplotlib.axes.Axes | None" = None,
    odd_color: str = "C0",
    even_color: str = "C1",
    show_legend: bool = True,
    **mpl_kwargs,
) -> "matplotlib.axes.Axes":
    """Plot odd vs even transit depth comparison.

    Parameters
    ----------
    result : CheckResult
        Result from V01 odd-even depth check
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates new figure if None.
    odd_color : str
        Color for odd transits
    even_color : str
        Color for even transits
    show_legend : bool
        Whether to show legend
    **mpl_kwargs
        Additional kwargs passed to matplotlib

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    # Extract plot data from result
    plot_data = result.raw.get("plot_data", {})
    if not plot_data:
        raise ValueError(f"Result {result.id} does not contain plot_data")

    # Plot odd transits
    ax.errorbar(
        plot_data["odd_epochs"],
        plot_data["odd_depths"],
        yerr=plot_data["odd_errs"],
        fmt='o',
        color=odd_color,
        label="Odd transits",
        **mpl_kwargs
    )

    # Plot even transits
    ax.errorbar(
        plot_data["even_epochs"],
        plot_data["even_depths"],
        yerr=plot_data["even_errs"],
        fmt='s',
        color=even_color,
        label="Even transits",
        **mpl_kwargs
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Depth (ppm)")
    ax.set_title(f"V01: Odd-Even Depth Comparison (σ={result.metrics.get('sigma_diff', 'N/A'):.2f})")

    if show_legend:
        ax.legend()

    return ax
```

---

## 7. Sources

- [Lightkurve Documentation](https://docs.lightkurve.org/)
- [Lightkurve GitHub - lightcurve.py](https://github.com/nasa/Lightkurve/blob/master/lightkurve/lightcurve.py)
- [Astropy Visualization Docs](https://docs.astropy.org/en/stable/visualization/index.html)
- [Astropy Matplotlib Integration](https://docs.astropy.org/en/stable/visualization/matplotlib_integration.html)
- [ArviZ Documentation](https://python.arviz.org/en/stable/)
- [ArviZ Plotting Backends](https://python.arviz.org/en/stable/contributing/plotting_backends.html)
