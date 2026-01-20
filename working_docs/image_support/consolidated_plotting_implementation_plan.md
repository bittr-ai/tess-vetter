# Consolidated Plotting Implementation Plan

**Date:** 2026-01-20
**Status:** Research Complete - Ready for Implementation Planning
**Input Documents:**
- `research_kepler_dvr_plots.md` - Kepler DVR/DVS standards
- `research_check_specific_plots.md` - Per-check plot recommendations
- `research_api_integration.md` - Architecture analysis
- `research_astronomy_plotting_patterns.md` - Industry patterns

---

## Executive Summary

Dr. Darin's feedback highlights a critical gap: scientists rely on visual validation of calculations, but bittr-tess-vetter currently produces only numeric outputs. This document consolidates research into a concrete implementation plan for first-class plotting support.

**Key Decision:** Adopt a **hybrid architecture** with:
1. A dedicated `plotting/` subpackage (scalable organization)
2. External functions (not methods on results) to maintain data purity
3. `raw["plot_data"]` in CheckResult for plottable arrays
4. matplotlib as an optional `[plotting]` extra

---

## 1. Architecture Design

### 1.1 Module Structure
```
src/bittr_tess_vetter/
  plotting/
    __init__.py          # MATPLOTLIB_AVAILABLE guard + public exports
    _core.py             # Shared utilities (ax handling, styles, annotations)
    checks.py            # V01-V12 check-specific plots
    false_alarm.py       # V13, V15 false alarm check plots
    extended.py          # V16-V21 extended check plots
    transit.py           # Phase-folded, transit model overlay
    pixel.py             # Difference image, centroid, aperture
    report.py            # Multi-panel DVR-style summary
```

### 1.2 API Surface Changes
```python
# In api/__init__.py (guarded by MATPLOTLIB_AVAILABLE)
from bittr_tess_vetter.plotting import (
    # Core plots
    plot_phase_folded,
    plot_transit_fit,

    # Check-specific
    plot_odd_even,           # V01
    plot_secondary_eclipse,  # V02
    plot_depth_stability,    # V04
    plot_v_shape,            # V05
    plot_nearby_ebs,         # V06
    plot_centroid_shift,     # V08
    plot_difference_image,   # V09
    plot_aperture_curve,     # V10
    plot_modshift,           # V11
    ...

    # Summary
    plot_vetting_summary,    # Multi-panel DVR-style
    save_vetting_report,     # PDF/PNG export
)
```

### 1.3 Optional Dependency
```toml
# pyproject.toml
[project.optional-dependencies]
plotting = ["matplotlib>=3.5.0"]
all = ["bittr-tess-vetter[tls,fit,...,plotting]"]
```

---

## 2. Function Signature Convention

All plotting functions follow this pattern:

```python
def plot_<name>(
    result: CheckResult,
    *,
    ax: "matplotlib.axes.Axes | None" = None,
    # Check-specific options
    show_legend: bool = True,
    annotate: bool = True,
    style: str = "default",
    **mpl_kwargs,
) -> "matplotlib.axes.Axes":
    """
    Parameters
    ----------
    result : CheckResult
        Result from the relevant check (must have raw["plot_data"])
    ax : Axes, optional
        Axes to plot on. Creates new figure if None.
    show_legend : bool
        Display legend
    annotate : bool
        Add metric annotations to plot
    style : str
        Style preset ("default", "paper", "presentation")
    **mpl_kwargs
        Passed to underlying matplotlib calls

    Returns
    -------
    matplotlib.axes.Axes
    """
```

---

## 3. Data Requirements (CheckResult Modifications)

Each check needs to populate `raw["plot_data"]` with arrays for plotting.

### 3.1 Example: V01 Odd-Even Depth
```python
CheckResult(
    id="V01",
    metrics={
        "sigma_diff": 1.2,
        "depth_odd_ppm": 250.0,
        "depth_even_ppm": 245.0,
    },
    raw={
        "plot_data": {
            "odd_epochs": [1, 3, 5, 7],
            "odd_depths": [248.0, 252.0, 249.0, 251.0],
            "odd_errs": [15.0, 14.0, 16.0, 15.0],
            "even_epochs": [2, 4, 6, 8],
            "even_depths": [244.0, 246.0, 245.0, 245.0],
            "even_errs": [14.0, 15.0, 14.0, 15.0],
        }
    }
)
```

### 3.2 Checks Requiring plot_data Updates

| Check | Required plot_data Fields |
|-------|---------------------------|
| V01 | odd/even epochs, depths, errors |
| V02 | phase array, flux array, secondary window bounds |
| V04 | epoch times, per-epoch depths with errors |
| V05 | binned phase, binned flux, trapezoid model |
| V06 | target coords, nearby EB coords + metadata |
| V08 | TPF stamp, in/out centroid positions |
| V09 | depth map 2D array, target pixel location |
| V10 | aperture radii, depths per aperture |
| V11 | ModShift periodogram data |
| V13 | epoch coverage arrays |
| V15 | pre/post transit flux bins |
| V16 | model fit arrays for comparison |
| V17 | period neighborhood scores |
| V19 | harmonic period scores |
| V20 | in/out aperture depth maps |
| V21 | per-sector depth measurements |

---

## 4. Implementation Priority

### Phase 1: Foundation (Week 1)
1. Create `plotting/` subpackage skeleton
2. Implement `_core.py` with ax handling, style system
3. Add `[plotting]` extra to pyproject.toml
4. Implement `plot_phase_folded()` - most universally useful
5. Add `plot_data` to V01 (odd-even) as proof of concept
6. Implement `plot_odd_even()`

### Phase 2: Essential Checks (Week 2-3)
7. `plot_secondary_eclipse()` - V02
8. `plot_depth_stability()` - V04
9. `plot_v_shape()` - V05
10. `plot_centroid_shift()` - V08
11. `plot_difference_image()` - V09
12. `plot_aperture_curve()` - V10

### Phase 3: Advanced Checks (Week 4)
13. `plot_modshift()` - V11
14. `plot_data_gaps()` - V13
15. `plot_asymmetry()` - V15
16. `plot_model_comparison()` - V16
17. `plot_sector_consistency()` - V21

### Phase 4: Reports & Extended (Week 5)
18. `plot_vetting_summary()` - Multi-panel DVR-style
19. `save_vetting_report()` - PDF export
20. Remaining extended checks (V17-V20)

---

## 5. DVR-Style Summary Plot Design

Based on Kepler DVS one-page summaries:

```
┌─────────────────────────────────────────────────────────────────┐
│  TOI-XXXX.XX Vetting Summary                          [LOGO]   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ A: Full LC      │  │ B: Phase-Folded │  │ C: Secondary    │ │
│  │    (time series)│  │    (primary)    │  │    Eclipse      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ D: Odd-Even     │  │ E: V-Shape      │  │ F: Centroid     │ │
│  │    Comparison   │  │    (transit)    │  │    Shift        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│  ┌─────────────────┐  ┌─────────────────────────────────────┐  │
│  │ G: Depth        │  │ H: Metrics Summary Table            │  │
│  │    Stability    │  │    FPP, NFPP, key check results     │  │
│  └─────────────────┘  └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.1 Function Signature
```python
def plot_vetting_summary(
    bundle: VettingBundleResult,
    lc: LightCurve,
    candidate: Candidate,
    *,
    figsize: tuple[float, float] = (11, 8.5),
    include_panels: list[str] | None = None,
    title: str | None = None,
) -> "matplotlib.figure.Figure":
    """Generate DVR-style one-page vetting summary."""
```

---

## 6. Style System

### 6.1 Built-in Presets
```python
STYLES = {
    "default": {
        "figure.figsize": (8, 5),
        "font.size": 10,
        "axes.titlesize": 12,
        "lines.linewidth": 1.0,
    },
    "paper": {
        "figure.figsize": (3.5, 2.5),  # Single-column
        "font.size": 8,
        "axes.titlesize": 9,
    },
    "presentation": {
        "figure.figsize": (10, 6),
        "font.size": 14,
        "axes.titlesize": 16,
    },
}
```

### 6.2 Color Scheme
```python
COLORS = {
    "transit": "#1f77b4",      # Blue - in-transit
    "out_of_transit": "#7f7f7f",  # Gray
    "odd": "#2ca02c",          # Green
    "even": "#d62728",         # Red
    "model": "#ff7f0e",        # Orange
    "secondary": "#9467bd",    # Purple
    "centroid_in": "#e377c2",  # Pink
    "centroid_out": "#17becf", # Cyan
}
```

---

## 7. Testing Strategy

### 7.1 Unit Tests
- Each plot function tested with mock CheckResult
- Verify axes labels, title, legend presence
- Test ax=None creates figure
- Test ax provided reuses axes

### 7.2 Visual Regression
- Generate reference images for key plots
- Use pytest-mpl for visual comparison

### 7.3 Integration Tests
- Run full vetting pipeline
- Generate summary plot
- Verify no matplotlib warnings

---

## 8. Documentation Plan

### 8.1 API Reference
- Docstrings for all public functions
- Gallery of example plots

### 8.2 Tutorial Updates
- Add plotting cells to Tutorial 10 (TOI-5807)
- New Tutorial 11: "Visualizing Vetting Results"

### 8.3 Example Gallery
```
docs/
  gallery/
    plot_phase_folded.png
    plot_odd_even.png
    plot_vetting_summary.png
    ...
```

---

## 9. Open Questions for Implementation

1. **Should `plot_data` be opt-in?** Some checks produce large arrays. Should there be a `include_plot_data=True` parameter?

2. **TPF data retention:** Pixel plots (V08-V10) need original TPF data. Should we store minimal TPF stamps in results, or require re-loading?

3. **Interactive backends:** Should we support bokeh/plotly for notebook interactivity? (ArviZ pattern)

4. **Colorbar handling:** Several plots benefit from colorbars (difference images). Standard pattern needed.

5. **Multi-sector plots:** How to handle per-sector subplots vs combined plots?

---

## 10. Success Criteria

1. Scientists can generate diagnostic plots with one function call per check
2. DVR-style summary can be generated from `VettingBundleResult`
3. Plots are publication-quality with minimal customization
4. matplotlib remains optional - core functionality works without it
5. Tutorial 10 updated with plotting cells showing visual validation

---

## 11. References

- Dr. Darin Feedback: "scientists use visual scans of data to validate calculations"
- Kepler DVR Example: KIC 4852528
- Lightkurve API: https://docs.lightkurve.org/
- ArviZ Plotting: https://python.arviz.org/
