# Research Q5: Multi-Sector Data Visualization Strategy

**Date:** 2026-01-20
**Question:** How should multi-sector data be visualized - combined plots or per-sector subplots?
**Context:** TOI-5807 has sectors 55, 75, 82, 83 - representative of TESS multi-sector targets

---

## Executive Summary

**Recommendation: Option C (Provide Both) with Intelligent Defaults**

The optimal strategy is to provide both combined and per-sector views, with smart defaults based on the diagnostic purpose. Different plot types have different visualization requirements - forcing a single approach degrades scientific utility.

---

## 1. Investigation Findings

### 1.1 Current bittr-tess-vetter Handling (Tutorial 10)

The tutorial demonstrates a clear **dual-path approach**:

1. **Stitched light curve** for LC-only checks (V01-V05, V11-V19)
   - Sectors are combined via `stitch_lightcurves()` with per-sector normalization
   - Stitched LC used for phase-folding, transit fitting, periodogram analysis

2. **Per-sector execution** for pixel-level checks (V08-V10)
   - Checks run separately per sector with per-sector TPFs
   - Results reported as list of per-sector dictionaries
   - Example from Tutorial 10:
   ```python
   for sector in sorted(ds.lc_by_sector.keys()):
       sec_lc = ds.lc_by_sector[sector]
       sec_tpf = tpf_by_sector.get(sector)
       # ... run check per sector
       out.append({'sector': sector, 'status': r.status, ...})
   ```

3. **Per-sector detrending** for robustness validation
   - Tutorial Section 10: "Per-sector detrending/normalization (robustness follow-up)"
   - Each sector detrended independently, then re-stitched

### 1.2 Kepler DVR/DVS Precedent

From `research_kepler_dvr_plots.md`:

| Plot Type | Kepler Approach | Notes |
|-----------|-----------------|-------|
| **Full Time-Series (A)** | Combined, all quarters | Triangles mark transits across full baseline |
| **Centroid Offset (G)** | **Per-quarter** with 3-sigma circle | Separate measurement per quarter |
| **Difference Images** | **Per-quarter** | TESS equivalent is per-sector FFI-based |

Key insight: Kepler DVRs show **combined plots for temporal diagnostics** but **per-quarter panels for spatial/centroid diagnostics**. This is because:
- Temporal signals (transits) benefit from phase-folding across full baseline
- Spatial signals (centroids, pixel behavior) can vary per observing epoch due to detector position, scattered light

### 1.3 Lightkurve Community Patterns

From web search results:

1. **Default behavior:** `LightCurveCollection.stitch()` combines into single LC
2. **Visualization:** Typically shows stitched LC as single time series
3. **Important caveat from lightkurve docs:**
   > "The TPFs from different sectors may have slightly different PSF shapes, due to being on different parts of the detector, at different seasons, with different scattered light. If you plot the TPFs you will see that you cannot simply reuse the same aperture for different sectors."

This directly supports per-sector visualization for pixel-level diagnostics.

### 1.4 V21 Sector Consistency Check Structure

From `sector_consistency.py`:
```python
@dataclass
class SectorMeasurement:
    sector: int
    depth_ppm: float
    depth_err_ppm: float
    # ... per-sector quantities
```

The check returns:
- `measurements`: List of per-sector data
- `outlier_sectors`: Sectors flagged as inconsistent
- `chi2_pval`: Consistency p-value

This structure naturally maps to **per-sector subplot visualization** (sector on x-axis, depth with error bars on y-axis).

---

## 2. Plot-by-Plot Recommendations

### 2.1 COMBINED View Preferred (Stitched Data)

| Check/Plot | Rationale |
|------------|-----------|
| **Full light curve (time series)** | Context across full baseline; color-code by sector but single panel |
| **Phase-folded transit** | Transit depth comparison requires folding all epochs together |
| **V01 (Odd/Even)** | Compares folded odd vs even epochs across all sectors |
| **V02 (Secondary Eclipse)** | Phase-folded full-orbit view needs all data |
| **V05 (V-Shape)** | Transit shape analysis uses folded, binned data |
| **V11 (ModShift)** | Periodogram computed on full stitched LC |
| **V12 (SWEET)** | Sinusoidal variability detection on full LC |
| **V13 (Data Gaps)** | Coverage shown across full baseline (but epoch-level detail per transit) |
| **V16 (Model Competition)** | Model fits on full stitched data |

### 2.2 PER-SECTOR Subplots Preferred

| Check/Plot | Rationale |
|------------|-----------|
| **V08 (Centroid Shift)** | Centroid position varies per sector (different detector location, PSF) |
| **V09 (Difference Image)** | Pixel-level depth map is sector-specific |
| **V10 (Aperture Dependence)** | Aperture effects depend on per-sector contamination |
| **V20 (Ghost Features)** | Scattered light patterns are epoch-specific |
| **V21 (Sector Consistency)** | The purpose is comparing sectors - must show per-sector |

### 2.3 HYBRID Approach Beneficial

| Check/Plot | Recommendation |
|------------|----------------|
| **V04 (Depth Stability)** | Combined: per-epoch depth scatter plot with sector indicated by color/marker. Per-sector option: faceted by sector for detail |
| **Per-sector detrending comparison** | Show before/after per sector, plus stitched result |

---

## 3. Implementation Recommendations

### 3.1 API Design

```python
def plot_sector_consistency(
    result: CheckResult,
    *,
    ax: "Axes | None" = None,
    style: str = "default",
) -> "Axes":
    """Per-sector depth comparison - always uses per-sector view."""
    # Bar chart with sector on x-axis, depth +/- error on y-axis
    # Highlight outlier_sectors

def plot_full_lightcurve(
    lc: LightCurve,
    sectors: list[int] | None = None,
    *,
    ax: "Axes | None" = None,
    color_by_sector: bool = True,
    separate_panels: bool = False,  # NEW OPTION
) -> "Axes | list[Axes]":
    """
    Full time-series light curve visualization.

    Parameters
    ----------
    color_by_sector : bool
        If True, use different colors per sector (combined view)
    separate_panels : bool
        If True, create vertically stacked subplots per sector
    """

def plot_centroid_shift(
    results: list[CheckResult] | CheckResult,
    *,
    per_sector: bool = True,  # Default to per-sector
    figsize: tuple[float, float] | None = None,
) -> "Figure":
    """
    Centroid shift visualization.

    If results is a list (per-sector), creates subplot grid.
    If single result and per_sector=False, shows single panel.
    """
```

### 3.2 Per-Sector Subplot Layout

For checks that produce per-sector results, use a consistent layout:

```
n_sectors = len(sectors)  # e.g., 4 for TOI-5807

# Layout options based on count:
# 1-2 sectors: 1 row, n cols
# 3-4 sectors: 2 rows, 2 cols
# 5-6 sectors: 2 rows, 3 cols
# 7+  sectors: 3 rows, ceil(n/3) cols

def _compute_subplot_grid(n: int) -> tuple[int, int]:
    if n <= 2:
        return (1, n)
    elif n <= 4:
        return (2, 2)
    elif n <= 6:
        return (2, 3)
    else:
        return (3, math.ceil(n / 3))
```

### 3.3 Color-Coding Convention

For combined plots where sectors are distinguished:

```python
SECTOR_COLORS = {
    # Use a qualitative colormap that handles up to 13 sectors (CVZ max)
    # Tab10 or Paired work well
}

def get_sector_color(sector: int, all_sectors: list[int]) -> str:
    """Return consistent color for sector within a target's sector list."""
    idx = sorted(all_sectors).index(sector)
    return plt.cm.tab10(idx / 10)
```

### 3.4 StitchedLC Awareness

The `StitchedLC` container already includes `sector` array:
```python
@dataclass
class StitchedLC:
    time: NDArray
    flux: NDArray
    sector: NDArray  # <- Sector label per cadence
    per_sector_diagnostics: list[SectorDiagnostics]
```

This enables seamless sector-based coloring in combined plots:
```python
for s in np.unique(stitched.sector):
    mask = stitched.sector == s
    ax.scatter(stitched.time[mask], stitched.flux[mask],
               c=get_sector_color(s, sectors), label=f"S{s}", s=1)
```

---

## 4. Summary Table: Plot Type Defaults

| Plot Category | Default View | Per-Sector Option | Notes |
|---------------|--------------|-------------------|-------|
| Full LC (time series) | Combined, color by sector | `separate_panels=True` | Sector gaps naturally visible |
| Phase-folded transit | Combined | Color by sector in scatter | Binned fit ignores sector |
| Per-epoch depths (V04) | Combined scatter, color by sector | Facet option | |
| Centroid (V08) | **Per-sector subplots** | N/A | Must be per-sector |
| Difference image (V09) | **Per-sector subplots** | N/A | Must be per-sector |
| Aperture curve (V10) | **Per-sector subplots** | Combined overlay option | |
| Ghost features (V20) | **Per-sector subplots** | N/A | Must be per-sector |
| Sector consistency (V21) | Per-sector bar chart | N/A | Single panel, x=sector |

---

## 5. Implementation Priority

1. **Phase 1:** Implement sector-aware coloring in combined plots (`StitchedLC.sector` array)
2. **Phase 2:** Implement per-sector subplot grid for V08, V09, V10, V20
3. **Phase 3:** Add `separate_panels` option for full LC plot
4. **Phase 4:** Implement V21 sector comparison bar chart

---

## 6. References

- Tutorial 10: `docs/tutorials/10-toi-5807-check-by-check.ipynb`
- Kepler DVR research: `working_docs/image_support/research_kepler_dvr_plots.md`
- Sector consistency: `src/bittr_tess_vetter/validation/sector_consistency.py`
- Stitch module: `src/bittr_tess_vetter/api/stitch.py`
- [Lightkurve LightCurveCollection.stitch()](https://lightkurve.github.io/lightkurve/reference/api/lightkurve.LightCurveCollection.stitch.html)
- [TESS Workshop Tutorials - Multi-sector exercise](https://github.com/spacetelescope/tessworkshop_tutorials/blob/master/lightkurve/workshop/Exercise_2-Solutions.ipynb)
