# Research: Should `plot_data` in CheckResult.raw be Opt-In?

**Date:** 2026-01-20
**Question:** Should storing plottable arrays in `result.raw["plot_data"]` require an explicit `include_plot_data=True` parameter?

---

## Recommendation

**NO** - `plot_data` should be **always included by default**, not opt-in.

---

## Rationale

### 1. Array Size Analysis by Check Type

After analyzing all check implementations in `src/bittr_tess_vetter/validation/`, the plot_data arrays are consistently small:

| Check | plot_data Contents | Typical Size | Memory |
|-------|-------------------|--------------|--------|
| **V01 Odd-Even** | 2 lists of per-epoch depths (capped at 20 each) | ~40 floats | ~320 bytes |
| **V02 Secondary** | Search window bounds, phase coverage bins | ~25 floats | ~200 bytes |
| **V04 Depth Stability** | Per-epoch depths + errors (capped at 20) | ~40 floats | ~320 bytes |
| **V05 V-Shape** | Trapezoid model parameters, binned flux (20 bins) | ~50 floats | ~400 bytes |
| **V06 Nearby EBs** | Coordinates + metadata for ~5-10 matches | ~100 values | ~1 KB |
| **V08 Centroid** | Two (x,y) positions + uncertainties | ~10 floats | ~80 bytes |
| **V09 Pixel Depths** | 2D depth map (11x11 typical TPF) | ~121 floats | ~1 KB |
| **V10 Aperture** | Depths at 5-8 aperture radii | ~16 floats | ~128 bytes |
| **V11 ModShift** | ~200 phase bins for periodogram | ~400 floats | ~3 KB |
| **V13 Data Gaps** | Epoch coverage fractions (~20-50 epochs) | ~100 floats | ~800 bytes |
| **V21 Sector Consistency** | Per-sector depths (~5-10 sectors) | ~30 floats | ~240 bytes |

**Total per VettingBundleResult:** ~7-10 KB (negligible)

The only potentially large array is V09's pixel depth map, which is bounded by TPF size (typically 11x11 = 121 pixels). Even a 21x21 stamp would only be ~3.5 KB.

### 2. Memory/Serialization Impact

- **JSON serialization:** Adding ~10 KB per bundle has negligible impact on disk I/O or network transfer
- **In-memory:** Python object overhead dominates; array data is trivial
- **Multi-candidate workflows:** Processing 1000 candidates adds ~10 MB total - well within typical scientific computing budgets
- **Comparison:** The input light curve itself is typically 10,000-50,000 cadences x 8 bytes = 80-400 KB per sector

### 3. Industry Patterns

**scikit-learn Display objects always compute and store plotting data:**
- `RocCurveDisplay` stores `fpr`, `tpr`, `roc_auc` unconditionally
- `ConfusionMatrixDisplay` stores the full confusion matrix
- No opt-in mechanism exists; the philosophy is "compute once, plot anywhere"
- The `from_predictions()` pattern exists to avoid re-running the model, but stored data is not optional

**ArviZ (Bayesian visualization):**
- `InferenceData` containers store full MCMC chains by default
- No opt-in for summary statistics; everything is pre-computed

**lightkurve:**
- `Periodogram` objects store full frequency/power arrays
- `LightCurve.fold()` returns full phase-folded arrays, not summaries

**Pattern:** Scientific libraries prioritize reproducibility and exploration over minor memory savings.

### 4. User Experience Trade-offs

**Always-on (recommended):**
- Plotting works immediately: `plot_odd_even(result)` just works
- No "oops, I forgot to enable plot_data" errors
- Consistent API: every CheckResult has the same structure
- Enables downstream tools (report generators, dashboards) to assume data exists

**Opt-in disadvantages:**
- Two code paths to test and document
- Users must plan ahead ("will I want to plot this?")
- Breaks "run once, explore interactively" workflow
- API friction: `run_vetting(include_plot_data=True)` is one more thing to remember

### 5. When Opt-In Makes Sense (and why it doesn't here)

Opt-in is appropriate when:
- Data is genuinely large (e.g., full MCMC chains, video frames)
- Computation is expensive (separate from the primary calculation)
- Data has security/privacy implications

None of these apply to bittr-tess-vetter plot_data:
- Arrays are small (<10 KB total)
- Data is already computed as part of the check logic
- Light curves and TPFs are already public TESS data

### 6. Handling Large TPF Data (V08-V10)

The pixel-level checks (V08 centroid, V09 depth map, V10 aperture) process TPF cubes internally but should NOT store full TPF cubes in `plot_data`. Instead:

```python
# V09 stores only the derived 2D depth map, not the 3D TPF cube
raw["plot_data"] = {
    "depth_map_ppm": depth_map,      # 11x11 array, ~1 KB
    "target_pixel": (5, 5),
    "max_depth_pixel": (6, 5),
    # NOT: "tpf_cube": tpf_data       # Would be ~1 MB
}
```

For plots requiring original pixel data (e.g., TPF thumbnails with centroid overlay), the plotting function should accept optional `tpf` parameter:

```python
def plot_centroid_shift(result, *, tpf=None, ax=None):
    """
    If tpf provided, shows centroid positions on TPF thumbnail.
    Otherwise, shows schematic centroid diagram.
    """
```

This pattern (small derived data always included, large source data optional) is common in astronomy pipelines.

---

## Implementation Recommendation

1. **Always populate `raw["plot_data"]`** in all check implementations
2. **Cap array sizes** where appropriate (e.g., epoch depths capped at 20 per parity)
3. **Store derived summaries**, not raw inputs (depth map, not TPF cube)
4. **Document the schema** for each check's plot_data in docstrings
5. **Plotting functions accept optional source data** for rich visualizations

---

## Alternative Considered: Lazy Computation

One could defer plot_data computation to first access:

```python
@property
def plot_data(self):
    if self._plot_data is None:
        self._plot_data = self._compute_plot_data()
    return self._plot_data
```

**Rejected because:**
- Requires storing intermediate state or re-accessing input data
- Breaks JSON serialization (can't serialize lazy properties)
- Adds complexity for negligible benefit
- Against "metrics-only" design philosophy (CheckResult should be a pure data container)

---

## Conclusion

The memory/serialization cost of always including plot_data is negligible (<10 KB per bundle), while the user experience cost of opt-in is significant (API friction, error-prone, inconsistent structure).

Following scikit-learn's successful pattern of "Display objects store computed values unconditionally," bittr-tess-vetter should always include plot_data in CheckResult.raw.

---

## References

- [scikit-learn Visualizations Documentation](https://scikit-learn.org/stable/visualizations.html)
- bittr-tess-vetter check implementations: `src/bittr_tess_vetter/validation/*.py`
- `consolidated_plotting_implementation_plan.md` (this repository)
