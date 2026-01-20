# Research Q2: TPF Data Retention for Pixel-Level Plots

**Date:** 2026-01-20
**Question:** How should TPF (Target Pixel File) data be handled for pixel-level plots (V08-V10, V20)?

---

## Executive Summary

**Recommendation: Hybrid Approach (Option C) - Store derived 2D stamps, require TPF reload for time-series plots**

Store minimal derived products (~1-4 KB) in `CheckResult.details["plot_data"]` for static diagnostic plots, but require users to re-load full TPF data (~12-45 MB) for advanced visualizations like animations or multi-cadence analysis.

---

## 1. Analysis of Pixel Check Data Requirements

### 1.1 V08: Centroid Shift

**Source:** `src/bittr_tess_vetter/validation/checks_pixel.py` (lines 53-113)

**Input consumed:**
- `tpf_data`: 3D array (time, rows, cols)
- `time`: 1D array

**Output metrics stored:**
- Centroid positions: `in_transit_centroid`, `out_of_transit_centroid` (4 floats)
- Shift magnitude and uncertainty (5 floats)
- `tpf_shape` (metadata only)

**What's needed for plotting:**
1. **Static centroid plot** (showing in/out centroids on stamp): Need single 2D reference image
2. **Centroid motion animation**: Would need full TPF time series

**Verdict:** Store single out-of-transit median image for static plot

### 1.2 V09: Difference Image / Pixel-Level Depth Map

**Source:** `src/bittr_tess_vetter/validation/checks_pixel.py` (lines 117-330)
**Supporting:** `src/bittr_tess_vetter/pixel/difference.py`

**Input consumed:**
- `tpf_data`: 3D array (time, rows, cols)
- `time`: 1D array

**Output metrics stored:**
- Pixel coordinates: `max_depth_pixel`, `target_pixel` (4 ints)
- Depth values (floats)
- Shape metadata

**Key insight from `localization.py` (line 205-210):**
```python
images: dict[str, NDArray[np.floating]] = {
    "in_transit_median": in_img.astype(np.float32),
    "out_of_transit_median": out_img.astype(np.float32),
    "difference_image": diff.astype(np.float32),
}
```
The codebase already computes and returns these derived images!

**What's needed for plotting:**
1. **Difference image heatmap**: 2D array (already computed internally)
2. **In/out transit comparison**: Two 2D arrays
3. **Per-pixel light curves**: Would need full TPF

**Verdict:** Store `difference_image` 2D array (already computed)

### 1.3 V10: Aperture Dependence

**Source:** `src/bittr_tess_vetter/pixel/aperture.py` (lines 304-603)

**Output metrics stored:**
- `depths_by_aperture_ppm`: Dict mapping radius -> depth
- Per-aperture statistics

**What's needed for plotting:**
1. **Depth vs aperture curve**: Just the dict (already stored)
2. **Aperture mask overlays**: Can regenerate from shape + radii
3. **Per-aperture light curves**: Would need full TPF

**Verdict:** Existing details sufficient; optionally add aperture mask images (~1 KB each)

### 1.4 V20: Ghost Features

**Source:** `src/bittr_tess_vetter/validation/ghost_features.py` (lines 309-372)

**Input consumed:**
- `tpf_data`: 3D array
- `aperture_mask`: 2D boolean array

**Internal computation (line 336):**
```python
diff_image = compute_difference_image(tpf_data, time, period, t0, duration_hours)
```

**What's needed for plotting:**
1. **Ghost feature overlay**: difference_image + aperture_mask
2. **Gradient visualization**: Edge gradient maps (computed internally)

**Verdict:** Store `diff_image` and `aperture_mask` (~2 KB total)

---

## 2. Size Analysis

### 2.1 Full TPF Data Sizes (DO NOT STORE)

| Cadence Mode | 11x11 Stamp | 13x13 Stamp | 21x21 Stamp |
|--------------|-------------|-------------|-------------|
| 2-min full sector (13,500 cad) | **12.5 MB** | 17.4 MB | 45.4 MB |
| 30-min FFI full sector (1,350 cad) | 1.2 MB | 1.7 MB | 4.5 MB |
| Transit window (500 cad) | 0.46 MB | 0.65 MB | 1.7 MB |

These sizes are prohibitive for storing in JSON-serializable `CheckResult.details`.

### 2.2 Derived 2D Frame Sizes (SAFE TO STORE)

| Stamp Size | float64 | float32 | Compressed (gzip) |
|------------|---------|---------|-------------------|
| 11x11 | 968 B | 484 B | ~300 B |
| 13x13 | 1.35 KB | 0.68 KB | ~400 B |
| 15x15 | 1.8 KB | 0.9 KB | ~500 B |
| 21x21 | 3.5 KB | 1.75 KB | ~1 KB |

**Storage per check (assuming 3 images stored):**
- V08: ~1.5 KB (1 reference image)
- V09: ~4 KB (3 images: in, out, diff)
- V10: ~1 KB (aperture masks, optional)
- V20: ~3 KB (diff + aperture mask)

**Total for all pixel checks: ~10 KB per target-sector**

---

## 3. Lightkurve Comparison

Based on [Lightkurve documentation](https://docs.lightkurve.org/reference/api/lightkurve.TessTargetPixelFile):

- **Full data retention:** Lightkurve keeps the entire FITS HDU in memory
- **Lazy loading:** Data arrays are accessed on-demand from the HDU
- **No derived caching:** Each plot call recomputes from raw data

**Key difference:** Lightkurve users always have the full TPF object available because they loaded it explicitly. bittr-tess-vetter computes metrics from TPF data that may not be retained by the caller.

---

## 4. Recommendation: Hybrid Approach

### 4.1 Store in `details["plot_data"]` (Minimal Stamps)

```python
# V08 Centroid Shift
"plot_data": {
    "reference_image": np.ndarray,  # float32, 2D (out-of-transit median)
    "in_centroid": (float, float),  # Already in details
    "out_centroid": (float, float), # Already in details
}

# V09 Difference Image
"plot_data": {
    "difference_image": np.ndarray,  # float32, 2D
    "in_transit_median": np.ndarray, # float32, 2D (optional)
    "out_transit_median": np.ndarray,# float32, 2D (optional)
    "depth_map_ppm": np.ndarray,     # float32, 2D (per-pixel depths)
}

# V10 Aperture Dependence
"plot_data": {
    "reference_image": np.ndarray,   # float32, 2D (optional)
    # depths_by_aperture already in details
}

# V20 Ghost Features
"plot_data": {
    "difference_image": np.ndarray,  # float32, 2D
    "aperture_mask": np.ndarray,     # bool, 2D
    "gradient_magnitude": np.ndarray,# float32, 2D (optional)
}
```

### 4.2 Require TPF Reload For (Advanced Plots)

- Per-cadence centroid motion animations
- Per-pixel light curve extraction
- Time-resolved difference imaging
- Custom aperture experimentation

### 4.3 Implementation Pattern

```python
def check_pixel_level_lc_with_tpf(..., store_plot_data: bool = True) -> VetterCheckResult:
    """
    Parameters
    ----------
    store_plot_data : bool
        If True, include derived 2D arrays in details["plot_data"] for
        downstream visualization (~3-5 KB). Set False for minimal memory usage.
    """
    # ... existing computation ...

    details = {...}

    if store_plot_data:
        details["plot_data"] = {
            "difference_image": diff_image.astype(np.float32).tolist(),  # JSON-safe
            "in_transit_median": in_img.astype(np.float32).tolist(),
            "out_transit_median": out_img.astype(np.float32).tolist(),
        }

    return _metrics_result(...)
```

### 4.4 Serialization Strategy

For JSON compatibility, convert to nested lists:
```python
# Store
details["plot_data"]["difference_image"] = diff_image.astype(np.float32).tolist()

# Retrieve for plotting
diff_image = np.array(details["plot_data"]["difference_image"], dtype=np.float32)
```

For more efficient storage (if CheckResult supports bytes), use:
```python
# Store as base64-encoded float32
import base64
details["plot_data"]["difference_image_b64"] = base64.b64encode(
    diff_image.astype(np.float32).tobytes()
).decode('ascii')
details["plot_data"]["difference_image_shape"] = list(diff_image.shape)
```

---

## 5. Specific Recommendations by Check

| Check | Store | Do Not Store | Notes |
|-------|-------|--------------|-------|
| **V08** | `reference_image` (OOT median) | Full time series | Centroids already in details |
| **V09** | `difference_image`, `depth_map_ppm` | In/out medians (optional) | Core diagnostic image |
| **V10** | None required | Aperture masks | `depths_by_aperture` sufficient |
| **V20** | `difference_image`, `aperture_mask` | Gradient maps | Ghost overlay visualization |

---

## 6. API Design for Plotting Functions

```python
# In plotting/pixel.py

def plot_centroid_shift(
    result: VetterCheckResult,
    *,
    tpf_data: NDArray | None = None,  # Optional for reference image
    ax: "Axes | None" = None,
) -> "Axes":
    """
    Plot centroid shift diagnostic.

    If result.details contains plot_data["reference_image"], uses that.
    Otherwise, requires tpf_data parameter.
    """
    if "plot_data" in result.details and "reference_image" in result.details["plot_data"]:
        ref_img = np.array(result.details["plot_data"]["reference_image"])
    elif tpf_data is not None:
        ref_img = np.nanmedian(tpf_data, axis=0)
    else:
        raise ValueError("Either store plot_data in result or provide tpf_data")
    # ... plotting logic ...


def plot_difference_image(
    result: VetterCheckResult,
    *,
    ax: "Axes | None" = None,
    cmap: str = "RdBu_r",
) -> "Axes":
    """
    Plot difference image from V09 result.

    Requires result.details["plot_data"]["difference_image"].
    """
    if "plot_data" not in result.details:
        raise ValueError("Result does not contain plot_data. Re-run check with store_plot_data=True")
    diff = np.array(result.details["plot_data"]["difference_image"])
    # ... plotting logic ...
```

---

## 7. Open Questions

1. **Should `store_plot_data` be opt-in or default?**
   - Recommendation: Default to True for pixel checks (small overhead)
   - Allow `store_plot_data=False` for batch processing

2. **Should we store float32 or float64?**
   - Recommendation: float32 (half the size, sufficient precision for visualization)

3. **Should we compress images in CheckResult?**
   - Recommendation: No compression for now; ~10 KB overhead is acceptable
   - Revisit if storing hundreds of results in memory

---

## 8. Summary

| Option | Pros | Cons | Data Size |
|--------|------|------|-----------|
| **A: Store full TPF** | Complete flexibility | Massive memory/serialization cost | 12-45 MB |
| **B: Always reload** | Minimal storage | Poor UX, requires TPF availability | 0 KB |
| **C: Hybrid (recommended)** | Best of both | Slightly larger CheckResult | ~10 KB |

**Final Recommendation:** Implement Option C with:
1. Store derived 2D stamps (float32) in `details["plot_data"]`
2. Add `store_plot_data: bool = True` parameter to pixel checks
3. Plotting functions prefer stored data, fall back to TPF parameter
4. Document that advanced plots (animations, per-pixel LCs) require TPF reload
