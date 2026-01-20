# Research Q3: Interactive Plotting Backends

**Date:** 2026-01-20
**Question:** Should bittr-tess-vetter support interactive plotting backends (bokeh, plotly) in addition to matplotlib?

---

## Executive Summary

**Recommendation: Matplotlib-only for Phase 1; defer multi-backend support to "maybe never"**

Interactive backends add significant implementation and maintenance burden without clear value for the primary vetting workflow. Astronomers predominantly use matplotlib, and the specific cases where interactivity adds value (zooming on transits) can be achieved with matplotlib + ipywidgets at a fraction of the complexity.

---

## 1. How Astronomers Use Plotting in Jupyter Notebooks

### Research Findings

Based on web research and astronomy library analysis:

1. **Matplotlib dominance**: Matplotlib remains the default plotting library in astronomy. All major astronomy packages (astropy, lightkurve, exoplanet, PyTransit, pylightcurve) use matplotlib as their primary or only backend.

2. **Interactive exploration valued, but...**: Astronomers do value zooming and panning during exploratory analysis. However, this is typically achieved through:
   - Native Jupyter matplotlib widget backend (`%matplotlib widget`)
   - ipywidgets sliders for parameter exploration
   - Dedicated Bokeh widgets (like lightkurve's `interact_bls`) for specific workflows

3. **Publication workflow**: For papers and reports, static matplotlib plots are universally required. Multi-backend libraries still need matplotlib for publication output.

### Key Insight

Interactive visualization is valuable for **exploration**, but vetting is fundamentally a **validation** workflow. Scientists want to see diagnostic plots that confirm or reject hypotheses, not freely explore data.

---

## 2. Lightkurve's Approach

### Architecture

Lightkurve uses a **hybrid strategy**, not true multi-backend support:

| Feature | Backend | Notes |
|---------|---------|-------|
| `lc.plot()` | matplotlib | Standard static plots |
| `lc.scatter()` | matplotlib | Static with colorbar |
| `lc.interact()` | Bokeh | Interactive light curve viewer |
| `lc.interact_bls()` | Bokeh | Interactive BLS periodogram |

### Key Implementation Details

From [lightkurve/interact_bls.py](https://github.com/nasa/Lightkurve/blob/master/lightkurve/interact_bls.py):
- Bokeh is an **optional dependency** (not installed by default)
- Interactive widgets are **separate functions**, not backend-swappable versions
- Error message: "The interact_bls() tool requires the `bokeh` package"
- Google Colab note: "Some features are not supported by Google Colab - most notably interactive plots made with the Bokeh package"

### Implication for bittr-tess-vetter

Lightkurve does NOT implement multi-backend support in the ArviZ sense. They have:
- matplotlib for static plots (99% of use cases)
- Bokeh for a few specific interactive widgets

This is a much simpler architecture than true backend abstraction.

---

## 3. Implementation Cost: Multi-Backend vs Matplotlib-Only

### ArviZ Multi-Backend Architecture

ArviZ supports matplotlib, bokeh, and plotly through:
```
arviz/plots/
    backends/
        matplotlib/
            plot_trace.py
            plot_posterior.py
            ...
        bokeh/
            plot_trace.py
            plot_posterior.py
            ...
        plotly/
            ...
```

**Estimated code multiplication**: ~2.5-3x for each additional backend

### Maintenance Burden Analysis

| Aspect | Matplotlib-Only | Multi-Backend (2 backends) |
|--------|-----------------|---------------------------|
| Code to write | ~20 plot functions | ~40-60 implementations |
| Testing | Simple pytest | Visual regression per backend |
| Dependencies | matplotlib only | matplotlib + bokeh + plotly |
| API surface | Unified | Backend-specific kwargs |
| Color handling | Native | Color conversion between systems |
| Documentation | Single gallery | Multiple galleries |
| Bug surface | N | 2-3N |

### SymPy's Warning

From [SymPy plotting documentation](https://docs.sympy.org/latest/modules/plotting.html):
> "The current implementation of the *Series classes is 'matplotlib-centric': the numerical data returned... is meant to be used directly by Matplotlib. Therefore, the new backend will have to pre-process the numerical data... if you code a new backend you have the responsibility to check if its working on each SymPy release."

### Cost Estimate

| Approach | Development Time | Ongoing Maintenance |
|----------|------------------|---------------------|
| Matplotlib-only | ~2-3 weeks | Low |
| +Bokeh backend | +3-4 weeks | +50% testing/debugging |
| +Plotly backend | +3-4 weeks | +50% testing/debugging |
| Full multi-backend | ~10-12 weeks | 2-3x ongoing effort |

---

## 4. Where Interactivity Adds Value in Vetting

### High-Value Interactive Scenarios

| Plot Type | Interactivity Benefit | Alternative |
|-----------|----------------------|-------------|
| Phase-folded transit | Zoom to see ingress/egress detail | Provide two plots: full + zoomed |
| Depth stability | Hover to see epoch metadata | Annotate outlier points |
| Difference image | Zoom/pan pixel grid | Provide multiple zoom levels |
| Full light curve | Select transit events | Highlight transit times with spans |
| Centroid shift | Hover for coordinates | Print coordinates in figure |

### Analysis

For most vetting plots, **static alternatives exist** that provide equivalent information:
- Pre-zoomed panels (ingress/egress detail)
- Annotation labels on points
- Multiple subplot zoom levels
- Printed coordinate tables

### The "interact_bls" Exception

Lightkurve's `interact_bls()` is valuable because:
- BLS requires iterative period refinement
- Users need to test many period values interactively
- This is **exploration**, not **validation**

bittr-tess-vetter operates downstream of BLS detection. By the time vetting runs, the candidate period is already determined. Interactive period exploration is not part of the vetting workflow.

---

## 5. Alternative: Matplotlib + ipywidgets

If interactivity is desired, a lightweight approach uses ipywidgets with matplotlib:

```python
from ipywidgets import interact, FloatRangeSlider
import matplotlib.pyplot as plt

@interact(xlim=FloatRangeSlider(min=-0.1, max=0.1, step=0.01))
def plot_transit_zoom(xlim):
    fig, ax = plt.subplots()
    ax.plot(phase, flux)
    ax.set_xlim(xlim)
    plt.show()
```

**Advantages:**
- No new plotting dependency
- Works in Jupyter and JupyterLab
- Reuses existing matplotlib plotting code
- Lower complexity than Bokeh/Plotly integration

**Disadvantages:**
- Requires ipywidgets
- Less smooth than native Bokeh
- Not suitable for standalone HTML export

---

## 6. Recommendation

### Phase 1: Matplotlib-Only

**Strong recommendation**: Implement plotting with matplotlib only.

**Rationale:**
1. **User expectations**: Astronomers expect matplotlib. It's the universal standard.
2. **Development velocity**: Ship plotting features 3-4x faster.
3. **Maintenance reality**: Small team cannot maintain multiple backends.
4. **Lightkurve precedent**: Even NASA's flagship TESS library uses matplotlib for 95% of plots.
5. **Publication workflow**: Scientists need matplotlib output regardless.

### Future Consideration: Targeted Interactive Widgets

If user feedback demands interactivity, consider:

1. **ipywidgets layer** (low cost): Add zoom sliders to key plots using ipywidgets + matplotlib.

2. **Single Bokeh widget** (medium cost): One interactive transit viewer (like lightkurve's approach), not full backend parity.

3. **Full multi-backend** (high cost): Only if funded with dedicated engineering time.

### Decision Matrix

| Scenario | Recommendation |
|----------|----------------|
| Phase 1 plotting implementation | Matplotlib only |
| User requests "zoom on transit" | ipywidgets slider |
| User requests "interactive exploration" | Point to lightkurve.interact() |
| User requests "Bokeh support" | Evaluate demand, likely defer |
| Grant proposal requires interactivity | Add 1-2 targeted Bokeh widgets |

---

## 7. API Design Consideration

Design the API to **allow future extension** without committing to multi-backend now:

```python
# Current (Phase 1)
def plot_phase_folded(result, *, ax=None, **kwargs):
    """Plot using matplotlib."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    ...
    return ax

# Future-compatible signature
def plot_phase_folded(
    result,
    *,
    ax=None,  # matplotlib axes
    backend="matplotlib",  # reserved for future
    **kwargs
):
    if backend != "matplotlib":
        raise NotImplementedError(
            f"Backend '{backend}' not supported. "
            "Only 'matplotlib' is currently implemented."
        )
    ...
```

This allows adding backends later without breaking existing code.

---

## 8. Sources

### Interactive Plotting Research
- [Bokeh and Plotly for Jupyter Notebooks](https://www.quantconnect.com/forum/discussion/5028/bokeh-and-plotly-for-the-jupyter-notebooks/)
- [Plotly vs Bokeh Comparison](https://pauliacomi.com/2020/06/07/plotly-v-bokeh.html)
- [Interactive Visualization with Bokeh](https://thedatafrog.com/en/articles/interactive-visualization-bokeh-jupyter/)
- [JupyterLab Interactive Plots - GeeksforGeeks](https://www.geeksforgeeks.org/data-visualization/how-to-use-jupyterlab-inline-interactive-plots/)

### Lightkurve
- [Lightkurve interact_bls.py Source](https://github.com/nasa/Lightkurve/blob/master/lightkurve/interact_bls.py)
- [Lightkurve interact_bls Documentation](https://docs.lightkurve.org/reference/api/lightkurve.LightCurve.interact_bls.html)

### ArviZ Multi-Backend
- [ArviZ Plotting with Bokeh](https://python.arviz.org/en/stable/user_guide/plotting_with_bokeh.html)
- [ArviZ-plots Package](https://arviz-plots.readthedocs.io/en/latest/)
- [ArviZ Bokeh Backend Documentation](https://github.com/arviz-devs/arviz/blob/main/doc/source/user_guide/plotting_with_bokeh.md)

### Implementation Complexity
- [Matplotlib Backends Documentation](https://matplotlib.org/stable/users/explain/backends.html)
- [SymPy Plotting Documentation](https://docs.sympy.org/latest/modules/plotting.html)
- [Pandas Plotting Backend](https://plotly.com/python/pandas-backend/)

### Astronomy Plotting
- [HoloViz Background](https://holoviz.org/background.html)
- [Exoplanet Package Transit Fitting](https://docs.exoplanet.codes/en/v0.4.3/tutorials/transit/)
- [PyTransit GitHub](https://github.com/hpparvi/PyTransit)

---

## 9. Conclusion

Multi-backend support is a significant engineering investment that does not align with bittr-tess-vetter's primary value proposition (rigorous vetting science). The astronomy community has settled on matplotlib as the standard, and deviating from this adds complexity without clear user benefit.

**Final recommendation**: Implement matplotlib-only plotting in Phase 1. Reserve `backend=` parameter in the API for future compatibility. Revisit only if funded development time is available AND user demand is demonstrated.
