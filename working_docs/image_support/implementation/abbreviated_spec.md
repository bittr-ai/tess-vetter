# Plotting Feature - Abbreviated Implementation Spec

**Reference**: Full spec at `plotting_spec.md` (consult for edge cases)

---

## 1. Module Structure

```
src/bittr_tess_vetter/
  plotting/
    __init__.py          # MATPLOTLIB_AVAILABLE guard + public exports
    _core.py             # ensure_ax(), add_colorbar(), style_context(), extract_plot_data()
    _styles.py           # STYLES dict, COLORS dict, COLORMAPS dict, LABELS dict
    checks.py            # V01-V05 plots
    catalog.py           # V06-V07 plots
    pixel.py             # V08-V10 plots
    exovetter.py         # V11-V12 plots
    false_alarm.py       # V13, V15 plots
    extended.py          # V16-V21 plots
    transit.py           # plot_phase_folded, plot_transit_fit
    lightcurve.py        # plot_full_lightcurve
    report.py            # plot_vetting_summary, save_vetting_report
```

---

## 2. Function Signature Convention

### Standard plot function:
```python
def plot_<name>(
    result: CheckResult,
    *,
    ax: "matplotlib.axes.Axes | None" = None,
    show_legend: bool = True,
    annotate: bool = True,
    style: str = "default",  # "default", "paper", "presentation"
    **mpl_kwargs,
) -> "matplotlib.axes.Axes":
```

### Image plot (returns colorbar):
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

## 3. plot_data Contracts

### Required structure:
```python
raw["plot_data"] = {
    "version": 1,  # REQUIRED - increment on breaking changes
    # ... check-specific fields (JSON-serializable)
}
```

### JSON serializability rules:
- `np.float*` -> `float`
- `np.int*` -> `int`
- `np.ndarray` (1D) -> `list` via `.tolist()`
- `np.ndarray` (2D) -> `list[list[float]]` via `.astype(np.float32).tolist()`

### Key naming suffixes:
| Suffix | Meaning |
|--------|---------|
| `_ppm` | Parts per million |
| `_hours` | Duration in hours |
| `_btjd` | Barycentric TESS Julian Date |
| `_arcsec` | Angular separation |
| `_px` / `_pixels` | Pixel units |
| `_x`, `_y` | Pixel coordinates (column, row) |

### Array size caps:
- Per-epoch arrays: 50 epochs max
- Phase/modshift bins: 200 bins max
- Image stamps: 21x21 pixels max

---

## 4. Coordinate Conventions (CRITICAL for pixel plots)

1. **Image storage**: `image[row][col]` (numpy/FITS convention)
2. **Coordinate system**: `(x, y)` = `(column, row)`
3. **imshow**: ALWAYS use `origin="lower"`
4. **Centroid**: `centroid_x` = column, `centroid_y` = row

---

## 5. Style System

### Using styles (non-sticky context manager):
```python
from ._core import style_context

def plot_foo(result, *, style="default", ...):
    with style_context(style):
        fig, ax = ensure_ax(ax)
        # ... plotting code
    return ax
```

### Style presets in `_styles.py`:
- `"default"`: 8x5 inches, 100 dpi, balanced
- `"paper"`: 3.5x2.5 inches, 300 dpi, minimal
- `"presentation"`: 10x6 inches, 150 dpi, large fonts

---

## 6. Core Utilities (_core.py)

```python
def ensure_ax(ax=None) -> tuple[Figure, Axes]:
    """Return (fig, ax), creating if needed."""

def add_colorbar(mappable, ax, *, label="", **kwargs) -> Colorbar:
    """Add colorbar with astronomy defaults (no minor ticks)."""

def style_context(style: str):
    """Context manager for style preset."""

def extract_plot_data(result: CheckResult, required_keys: list[str]) -> dict:
    """Extract and validate plot_data, raise ValueError if missing keys."""

def compute_subplot_grid(n: int) -> tuple[int, int]:
    """Optimal (nrows, ncols) for n subplots."""

def get_sector_color(sector: int, all_sectors: list[int]) -> str:
    """Consistent hex color for sector (tab10 cycling)."""
```

---

## 7. Check-to-Plot Mapping

| Check ID | Plot Function | Module | Returns |
|----------|---------------|--------|---------|
| V01 | `plot_odd_even` | checks.py | Axes |
| V02 | `plot_secondary_eclipse` | checks.py | Axes |
| V03 | `plot_duration_consistency` | checks.py | Axes |
| V04 | `plot_depth_stability` | checks.py | Axes |
| V05 | `plot_v_shape` | checks.py | Axes |
| V06 | `plot_nearby_ebs` | catalog.py | Axes |
| V07 | `plot_exofop_card` | catalog.py | Axes |
| V08 | `plot_centroid_shift` | pixel.py | (Axes, Colorbar\|None) |
| V09 | `plot_difference_image` | pixel.py | (Axes, Colorbar\|None) |
| V10 | `plot_aperture_curve` | pixel.py | Axes |
| V11 | `plot_modshift` | exovetter.py | Axes |
| V12 | `plot_sweet` | exovetter.py | Axes |
| V13 | `plot_data_gaps` | false_alarm.py | Axes |
| V15 | `plot_asymmetry` | false_alarm.py | Axes |
| V16 | `plot_model_comparison` | extended.py | Axes |
| V17 | `plot_ephemeris_reliability` | extended.py | Axes |
| V18 | `plot_sensitivity_sweep` | extended.py | Axes |
| V19 | `plot_alias_diagnostics` | extended.py | Axes |
| V20 | `plot_ghost_features` | extended.py | (Axes, Colorbar\|None) |
| V21 | `plot_sector_consistency` | extended.py | Axes |

---

## 8. plot_data Schemas (Key Checks)

### V01 - Odd/Even:
```python
{
    "version": 1,
    "odd_epochs": list[int],
    "odd_depths_ppm": list[float],
    "odd_errs_ppm": list[float],
    "even_epochs": list[int],
    "even_depths_ppm": list[float],
    "even_errs_ppm": list[float],
    "mean_odd_ppm": float,
    "mean_even_ppm": float,
}
```

### V08 - Centroid Shift:
```python
{
    "version": 1,
    "reference_image": list[list[float]],  # 2D [row][col]
    "in_centroid_x": float,   # column
    "in_centroid_y": float,   # row
    "out_centroid_x": float,
    "out_centroid_y": float,
    "shift_vector_x": float,
    "shift_vector_y": float,
    "target_pixel_x": int,
    "target_pixel_y": int,
}
```

### V21 - Sector Consistency:
```python
{
    "version": 1,
    "sectors": list[int],
    "depths_ppm": list[float],
    "depth_errs_ppm": list[float],
    "outlier_sectors": list[int],
    "weighted_mean_ppm": float,
}
```

---

## 9. Testing Requirements

### All tests must:
1. Use `pytest.importorskip("matplotlib")`
2. Use `matplotlib.use("Agg")` backend
3. Close figures after assertions (`plt.close(fig)`)

### Unit test coverage (per function):
- Creates figure when ax=None
- Uses provided ax when given
- Has correct labels
- Handles legend toggle
- Raises ValueError on missing plot_data
- Custom parameters apply

### Visual regression (pytest-mpl) - ONLY for:
- `plot_centroid_shift`
- `plot_difference_image`
- `plot_vetting_summary`

---

## 10. Quick Reference: Implementation Checklist

For each plot function:
- [ ] Uses `with style_context(style):` wrapper
- [ ] Calls `fig, ax = ensure_ax(ax)` first
- [ ] Calls `extract_plot_data(result, [...])` with required keys
- [ ] Image plots use `origin="lower"`
- [ ] Returns `ax` (or `(ax, cbar)` for images)
- [ ] Has docstring with Parameters, Returns, Raises, Examples

For each check update (adding plot_data):
- [ ] Includes `"version": 1` key
- [ ] All numpy types converted to Python types
- [ ] Arrays capped at documented limits
- [ ] Keys use proper unit suffixes

---

## 11. API Export Pattern

In `plotting/__init__.py`:
```python
import importlib.util

MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None

if MATPLOTLIB_AVAILABLE:
    from .checks import plot_odd_even, plot_secondary_eclipse, ...
    __all__ = ["plot_odd_even", ...]
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

In `api/__init__.py` (add to TYPE_CHECKING block):
```python
if MATPLOTLIB_AVAILABLE:
    from bittr_tess_vetter.plotting import (
        plot_odd_even, plot_secondary_eclipse, ...
    )
```

---

## 12. pyproject.toml Changes

Already present:
```toml
plotting = ["matplotlib>=3.5.1"]
all = ["bittr-tess-vetter[...,plotting]"]
```

No additional changes needed.

---

## 13. DVR Summary Layout

8-panel layout (11x8.5 inches landscape):
```
+------------------------------------------------------------------+
|  TIC XXXXXXXXX Vetting Summary                             [date] |
+------------------------------------------------------------------+
| A: Full LC      | B: Phase-Folded  | C: Secondary Eclipse        |
| D: Odd-Even     | E: V-Shape       | F: Centroid Shift           |
| G: Depth Stab.  | H: Metrics Table                               |
+------------------------------------------------------------------+
```

Panel mapping:
- A: `plot_full_lightcurve()`
- B: `plot_phase_folded()`
- C: `plot_secondary_eclipse()` (V02)
- D: `plot_odd_even()` (V01)
- E: `plot_v_shape()` (V05)
- F: `plot_centroid_shift()` (V08)
- G: `plot_depth_stability()` (V04)
- H: `_render_metrics_table()` (internal)
