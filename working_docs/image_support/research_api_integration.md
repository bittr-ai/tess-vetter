# Plotting API Integration Research

**Date:** 2026-01-20
**Status:** Research Complete
**Scope:** `bittr_tess_vetter.api` plotting integration

## 1. Current Architecture Summary

### 1.1 API Design Principles (from `api/__init__.py` and specs)
- **Pure functions**: Accept arrays/dataclasses, return dataclasses
- **Array-in/array-out**: No I/O, no caching, no plotting by default
- **Lazy loading**: PEP 562 `__getattr__` for deferred imports
- **Stable import surface**: All public imports from `bittr_tess_vetter.api`
- **Metrics-only results**: `CheckResult` returns metrics, not pass/fail policies
- **Optional dependencies**: Core is BSD-3, optional extras for specific features

### 1.2 Result Type Structure
```python
# CheckResult (Pydantic model)
- id: str                    # "V01", "V02", etc.
- name: str                  # Human-readable
- status: "ok"|"skipped"|"error"
- confidence: float | None
- metrics: dict[str, scalar] # JSON-serializable
- flags: list[str]
- raw: dict[str, Any] | None # Non-scalar data (arrays, etc.)

# VettingBundleResult
- results: list[CheckResult]
- provenance: dict
- inputs_summary: dict
```

### 1.3 Existing Patterns for Optional Features
- **MLX (GPU)**: `MLX_AVAILABLE` flag, guarded exports, graceful fallback
- **batman/emcee**: Runtime import checks, error results when missing
- **TRICERATOPS**: Separate optional extra with GPL-licensed deps
- **caps.py**: `DEFAULT_PLOTS_CAP` already exists (for capping plot counts)

### 1.4 Precedent: `TransitFitResult` Data-for-Plotting Pattern
```python
@dataclass(frozen=True)
class TransitFitResult:
    # ... fit parameters ...
    phase: list[float]      # "for plotting"
    flux_model: list[float] # "for plotting"
    flux_data: list[float]  # "for plotting"
```
This pattern stores plottable data in result objects, leaving visualization to downstream.

---

## 2. Architectural Options Analysis

### Option A: Methods on Result Objects (`result.plot()`)

**Implementation:**
```python
class CheckResult(BaseModel):
    def plot(self, ax=None, **kwargs) -> "matplotlib.axes.Axes":
        """Plot this check's diagnostic visualization."""
        from bittr_tess_vetter.plotting import plot_check
        return plot_check(self, ax=ax, **kwargs)
```

**Pros:**
- Intuitive discoverability (`result.plot()`)
- Matplotlib-like API (familiar to astronomers)
- Self-documenting: completion shows `.plot()` method

**Cons:**
- Couples result objects to matplotlib (even as lazy import)
- Pydantic models are data-focused; methods feel out of place
- Each result type needs its own plot method (proliferation)
- Hard to test result types without matplotlib
- Breaks "pure data" philosophy of current result types

**Verdict:** Not recommended. Violates array-in/array-out and data purity.

---

### Option B: Dedicated `api/plotting.py` Module

**Implementation:**
```python
# bittr_tess_vetter/api/plotting.py
from bittr_tess_vetter.plotting.core import (
    plot_odd_even,
    plot_phase_folded,
    plot_transit_fit,
    ...
)

__all__ = ["plot_odd_even", "plot_phase_folded", ...]
```

**Pros:**
- Clean separation: results are data, plotting is rendering
- Consistent with existing API structure (`api/lc_only.py`, etc.)
- Single import point for all plotting functions
- Easy to guard with `MATPLOTLIB_AVAILABLE` flag

**Cons:**
- Less discoverable than methods on results
- User must know function names exist

**Verdict:** Good option. Consistent with current patterns.

---

### Option C: Separate `plotting/` Subpackage

**Implementation:**
```
src/bittr_tess_vetter/
  plotting/
    __init__.py          # Exports all plot functions
    checks.py            # plot_odd_even, plot_v_shape, etc.
    transit.py           # plot_phase_folded, plot_fit
    pixel.py             # plot_difference_image, plot_centroid
    report.py            # multi-panel summary plots
```

Then in `api/__init__.py`:
```python
from bittr_tess_vetter.plotting import (
    plot_odd_even,
    plot_phase_folded,
    ...
)
```

**Pros:**
- Clear organization for complex plotting code
- Keeps `api/` thin (just re-exports)
- Easier to maintain as plot count grows
- Natural home for styling, themes, multi-panel layouts

**Cons:**
- More files/directories to navigate
- Slightly deeper import for internal development

**Verdict:** Recommended for long-term scalability.

---

### Option D: Hybrid - Data in Results + External Plotting

**Implementation:**
```python
# Results store plottable data
class OddEvenResult:
    odd_phase: list[float]
    odd_flux: list[float]
    even_phase: list[float]
    even_flux: list[float]

# Separate plotting module
def plot_odd_even(result: CheckResult, ax=None):
    """Plot odd/even depth comparison."""
    data = result.raw["odd_even_data"]
    # ... plotting code ...
```

**Pros:**
- Results remain pure data (JSON-serializable)
- Data can be plotted by any tool (matplotlib, bokeh, etc.)
- Matches `TransitFitResult` precedent
- Host apps can implement their own plotting

**Cons:**
- Larger result objects (storing arrays)
- Need to define what data each check should expose

**Verdict:** Recommended. Extends existing pattern.

---

## 3. Handling Matplotlib as Optional Dependency

### 3.1 Recommended Approach: Optional Extra

```toml
# pyproject.toml
[project.optional-dependencies]
plotting = ["matplotlib>=3.5.0"]
all = ["bittr-tess-vetter[tls,fit,wotan,batman,mlx,exovetter,ldtk,triceratops,plotting]"]
```

### 3.2 Runtime Guard Pattern

```python
# bittr_tess_vetter/plotting/__init__.py
import importlib.util

MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None

if MATPLOTLIB_AVAILABLE:
    from bittr_tess_vetter.plotting.checks import (
        plot_odd_even,
        plot_secondary_eclipse,
        ...
    )
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

### 3.3 API Surface Export

```python
# bittr_tess_vetter/api/__init__.py
MATPLOTLIB_AVAILABLE = _importlib_util.find_spec("matplotlib") is not None

if MATPLOTLIB_AVAILABLE:
    __all__.extend([
        "plot_odd_even",
        "plot_phase_folded",
        ...
    ])
```

---

## 4. Maintaining Array-In/Array-Out Philosophy

### 4.1 Core Principle
Plotting should be **additive**, not **required**. The core workflow remains:
```python
result = vet_candidate(lc, candidate)  # Array-in, struct-out
metrics = result.results[0].metrics     # Pure data extraction
```

### 4.2 Recommended Data Flow
```
User Data (arrays)
    |
    v
vet_candidate() / run_check()
    |
    v
CheckResult (metrics + raw plottable data)
    |
    v
[Optional] plot_*(result) -> matplotlib Figure
```

### 4.3 Plottable Data Convention

Each check that supports plotting should include plot data in `raw`:
```python
CheckResult(
    id="V01",
    name="Odd-Even Depth",
    metrics={"sigma_diff": 1.2, ...},
    raw={
        "plot_data": {
            "odd_phase": [...],
            "odd_flux": [...],
            "even_phase": [...],
            "even_flux": [...],
        }
    }
)
```

This keeps results JSON-serializable (arrays become lists) while providing
all data needed for reconstruction.

---

## 5. Concrete Recommendation

### 5.1 Architecture: Option C + D Hybrid

1. **Create `src/bittr_tess_vetter/plotting/` subpackage**
   ```
   plotting/
     __init__.py      # MATPLOTLIB_AVAILABLE guard + exports
     _core.py         # Shared utilities (ax handling, styles)
     checks.py        # Check-specific plots (V01-V12)
     transit.py       # Phase-fold, fit, model overlays
     pixel.py         # Difference images, centroids, PRF
     report.py        # Multi-panel summary (DVR-style)
   ```

2. **Re-export from `api/__init__.py`** (guarded)
   ```python
   if MATPLOTLIB_AVAILABLE:
       from bittr_tess_vetter.plotting import (
           plot_odd_even,
           plot_phase_folded,
           plot_vetting_summary,
           ...
       )
   ```

3. **Extend result types to include plot data**
   - Add `raw["plot_data"]` dict to checks that support visualization
   - Keep metrics separate for policy/guardrails use

4. **Add `[plotting]` optional extra**
   ```toml
   plotting = ["matplotlib>=3.5.0"]
   ```

### 5.2 Function Signature Convention

```python
def plot_odd_even(
    result: CheckResult,
    *,
    ax: "matplotlib.axes.Axes | None" = None,
    show_legend: bool = True,
    style: str = "default",
) -> "matplotlib.axes.Axes":
    """Plot odd vs even transit depth comparison.

    Args:
        result: CheckResult from V01 odd-even check (must have raw["plot_data"])
        ax: Matplotlib axes to plot on (creates new figure if None)
        show_legend: Whether to display legend
        style: Plot style preset ("default", "paper", "presentation")

    Returns:
        Matplotlib axes with the plot

    Raises:
        ValueError: If result does not contain plot_data
        ImportError: If matplotlib is not installed
    """
```

### 5.3 Summary Table

| Aspect | Recommendation |
|--------|----------------|
| Module location | `src/bittr_tess_vetter/plotting/` |
| API exposure | Re-export in `api/__init__.py` (guarded) |
| Dependency | `[plotting]` optional extra |
| Result modification | Add `raw["plot_data"]` to checks |
| Methods on results | No (keep results as pure data) |
| DVR-style reports | `plot_vetting_summary(bundle)` function |

---

## 6. Implementation Priority

### Phase 1: Foundation
1. Create `plotting/` subpackage with guards
2. Add `[plotting]` extra to pyproject.toml
3. Implement `plot_phase_folded()` (most common)
4. Add plot_data to `odd_even_depth` check

### Phase 2: Core Checks
5. `plot_odd_even()` - V01
6. `plot_secondary_eclipse()` - V02
7. `plot_v_shape()` - V05
8. `plot_depth_stability()` - V04

### Phase 3: Pixel/Advanced
9. `plot_difference_image()` - V09
10. `plot_centroid_shift()` - V08
11. `plot_aperture_family()` - V10

### Phase 4: Reports
12. `plot_vetting_summary()` - Multi-panel DVR-style
13. `save_vetting_report_pdf()` - PDF export

---

## 7. References

- Dr. Darin feedback: "Adding more graphs... scientists use visual scans"
- Kepler DVR/DVS examples: https://exoplanetarchive.ipac.caltech.edu/
- API Facade Spec v0.1: `working_docs/_archive/api_facade_spec.md`
- API Surface v0.2: `working_docs/api/api_surface_additions_v0_2.md`
- Result schema: `src/bittr_tess_vetter/validation/result_schema.py`
