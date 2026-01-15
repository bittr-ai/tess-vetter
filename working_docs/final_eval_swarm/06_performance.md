# Performance & Scalability Analysis

**Module:** `src/bittr_tess_vetter/compute/`
**Reviewer:** Performance & Scalability Lens
**Date:** 2026-01-14

---

## Executive Summary

The compute module demonstrates generally sound algorithmic choices with O(n) or O(n log n) complexity for most operations. However, several scalability concerns exist around multi-sector data handling and nested loop structures in BLS-like searches. The codebase leverages NumPy vectorization well in most places but has opportunities for improvement in pixel-level time-series fitting.

**Overall Assessment:** Good foundation with specific hotspots requiring attention for multi-sector scalability.

---

## 1. Algorithmic Complexity Analysis

### 1.1 BLS-Like Search: O(P * D * R * N)

**Location:** `bls_like_search.py`

The BLS search has concerning nested loop structure:

```python
# Lines 103-151 in bls_like_search.py
for period in period_grid:                    # O(P) periods
    bmeans, bcounts = _phase_bin_means(...)   # O(N) data points
    for dur_h in duration_hours_grid:         # O(D) durations
        score, min_bin = _bls_score_from_binned_flux(...)  # O(nbins)
        scan_phases = np.linspace(...)        # O(R) refinement steps
        for dphi in scan_phases:              # O(R) inner loop
            phase = ((time_btjd - t0_try) / period) % 1.0  # O(N) per iteration!
```

**Complexity:** O(P * D * R * N) where:
- P = number of periods (typically 1000-10000)
- D = number of durations (typically 3-10)
- R = local_refine_steps (default 11)
- N = number of data points

**Scalability Concern:** With 10,000 periods, 5 durations, 11 refinement steps, and 50,000 data points (multi-sector), this becomes ~27.5 billion operations.

**Recommendation:** The inner refinement loop recalculates `phase` for all N points on each iteration. Pre-compute base phases once per period:
```python
# Could vectorize: compute all refinement scores in one pass
```

### 1.2 Periodogram Module: Defers to TLS

**Location:** `periodogram.py`

The module primarily wraps `transitleastsquares` which has its own internal optimizations. The wrapper code is O(N) for data handling.

**Multi-sector handling (lines 297-486):** Iterates through sectors sequentially rather than in parallel. For S sectors:
- Total complexity: O(S * TLS_complexity)
- Memory: Creates copies per sector (S * N_sector)

### 1.3 Model Competition: O(N)

**Location:** `model_competition.py`

All model fitting operations are O(N):
- `fit_transit_only`: Single pass weighted least squares O(N)
- `fit_transit_sinusoid`: Design matrix O(N * k) where k = 2*n_harmonics + 1
- `fit_eb_like`: Design matrix O(N * 3)

The `np.linalg.lstsq` calls are O(N * k^2) which is effectively O(N) for small k.

**No scalability concerns here.**

### 1.4 Pixel Time-Series Fitting: O(W * H * N_pix * T)

**Location:** `pixel_timeseries.py`

The WLS fitting has potential scalability issues:

```python
# Lines 419-466 in pixel_timeseries.py
# For baseline_order=0: n_params = n_pixels + 1
# Design matrix: (n_cadences * n_pixels, n_params)
design_mat = np.zeros((n_cadences * n_pixels, n_params), dtype=np.float64)

for p in range(n_pixels):                     # O(n_pixels)
    design_mat[p::n_pixels, p] = 1.0          # O(n_cadences)

for t in range(n_cadences):                   # O(n_cadences)
    if in_transit[t]:
        design_mat[t * n_pixels : (t + 1) * n_pixels, -1] = prf_weights_flat
```

**Memory:** For 11x11 TPF (121 pixels) with 100 cadences:
- Design matrix: 12100 * 122 * 8 bytes = ~11.8 MB per window per hypothesis

**For baseline_order=1:** n_params doubles, memory quadruples.

**Scalability Concern:** With W=20 windows, H=10 hypotheses, this could require ~2.4 GB just for design matrices.

---

## 2. Memory Usage Patterns

### 2.1 Array Copies and Allocations

**Identified patterns:**

| Location | Pattern | Memory Impact |
|----------|---------|---------------|
| `periodogram.py:538-542` | Copies arrays for downsampling | Low (slicing) |
| `detrend.py:85-99` | `flux.copy()` + interpolation | 2x flux size |
| `bls_like_search.py:36-38` | Concatenation for circular rolling | 2x array |
| `pixel_timeseries.py:432-456` | Dense design matrices | High (see 1.4) |
| `model_competition.py:337-344` | Design matrix per model | Moderate |

### 2.2 In-Place vs Copy Operations

**Good patterns observed:**
```python
# detrend.py - Modifies copy, not original
flux_clean = flux.copy()
flux_clean[nan_mask] = median_val
```

**Potential improvements:**
```python
# pixel_hypothesis_prf.py:262-270 - Creates list then converts
other_prfs: list[tuple[NDArray[np.float64], float]] = []
for src_row, src_col, flux_ratio in other_sources:
    src_prf = prf_model.evaluate(...)  # Allocates new array each time
    other_prfs.append((src_prf, flux_ratio))
```

### 2.3 Multi-Sector Memory Scaling

**Pattern:** `split_by_sectors` (periodogram.py:65-110) creates separate array copies per sector.

For 10 sectors with 20,000 points each at 8 bytes/float:
- Time: 10 * 20,000 * 8 = 1.6 MB
- Flux: 10 * 20,000 * 8 = 1.6 MB
- Flux_err: 10 * 20,000 * 8 = 1.6 MB
- Total per split: ~4.8 MB

This is acceptable but could be avoided with index-based views.

---

## 3. Vectorization Opportunities

### 3.1 Already Well-Vectorized

| Function | Approach | Status |
|----------|----------|--------|
| `primitives.periodogram()` | scipy.signal.lombscargle | Excellent |
| `primitives.fold()` | Pure NumPy operations | Excellent |
| `primitives.detrend()` | scipy.ndimage.median_filter | Excellent |
| `detrend.sigma_clip()` | Vectorized MAD calculation | Excellent |
| `model_competition._box_transit_template()` | Vectorized phase calc | Good |

### 3.2 Missing Vectorization Opportunities

**3.2.1 BLS Score Refinement Loop**

Current (bls_like_search.py:125-145):
```python
for dphi in scan_phases:
    t0_try = float(t0_guess + dphi * float(period))
    phase = ((time_btjd - t0_try) / float(period)) % 1.0
    # ... compute score
```

Vectorizable approach:
```python
# Compute all phases at once: (n_refinements, n_points)
t0_tries = t0_guess + scan_phases * period  # (R,)
phases_all = ((time_btjd[None, :] - t0_tries[:, None]) / period) % 1.0
# Vectorize template creation and scoring
```

**3.2.2 Pixel Time-Series Design Matrix Construction**

Current (pixel_timeseries.py:430-465):
```python
for p in range(n_pixels):
    design_mat[p::n_pixels, p] = 1.0

for t in range(n_cadences):
    if in_transit[t]:
        design_mat[t * n_pixels : (t + 1) * n_pixels, -1] = prf_weights_flat
```

Better approach using sparse matrices or index assignment:
```python
# Use scipy.sparse.diags for baseline columns
# Use advanced indexing for amplitude column
in_transit_indices = np.where(in_transit)[0]
for t in in_transit_indices:
    design_mat[t * n_pixels : (t + 1) * n_pixels, -1] = prf_weights_flat
```

**3.2.3 Phase Bin Means**

The `_phase_bin_means` function (bls_like_search.py:42-54) is well-vectorized using `np.bincount`, which is optimal.

### 3.3 MLX Detection Module

**Location:** `mlx_detection.py`

This module provides GPU-accelerated alternatives for Apple Silicon. The design is sound:
- Uses `mx.vmap` for batched scoring
- Smooth (differentiable) templates enable gradient computation
- Lazy evaluation with explicit `mx.eval()`

**Note:** The integrated_gradients function (lines 236-253) uses a Python loop over alpha steps. This could be further vectorized but is typically called with small step counts (50).

---

## 4. Multi-Sector Data Scaling

### 4.1 Current Architecture

The codebase handles multi-sector data through:

1. **Split-then-process:** `split_by_sectors()` creates independent arrays
2. **Sequential TLS:** Each sector processed independently
3. **Merge results:** `merge_candidates()` deduplicates by period

### 4.2 Scaling Characteristics

| Sectors | Expected Behavior | Notes |
|---------|-------------------|-------|
| 1 | Baseline | Single TLS search |
| 5 | ~5x runtime | Sequential sector processing |
| 10+ | ~10x+ runtime | May hit memory limits on 16GB |
| 27 (full mission) | Untested | Likely requires batching |

### 4.3 Specific Bottlenecks

**4.3.1 Per-Sector TLS Overhead**

Each sector invokes a fresh TLS object:
```python
# periodogram.py:370-403
for i, (sector_time, sector_flux, sector_err) in enumerate(sectors):
    result = tls_search(...)  # New TLS model per sector
```

**Mitigation:** TLS internally caches period grids, but object creation overhead accumulates.

**4.3.2 Cross-Sector Candidate Clustering**

`cluster_cross_sector_candidates` (periodogram.py:159-294):
```python
for item in items:                    # O(C) candidates
    for fam in families:              # O(F) families (worst: O(C))
        if abs(p - pref) / max(pref, 1e-9) <= period_tol_frac:
```

**Worst case:** O(C^2) when all candidates form separate families.
**Typical case:** O(C * F) where F << C.

**4.3.3 Joint Likelihood Computation**

`compute_all_hypotheses_joint` (joint_likelihood.py:330-394):
```python
for hypothesis_id in hypotheses:           # O(H) hypotheses
    for evidence in sector_evidence:       # O(S) sectors
        # O(H_sector) to find hypothesis
```

**Complexity:** O(H * S * H_sector) which is manageable.

---

## 5. Performance Hotspots Summary

### Critical (Should Address Before Release)

| Issue | Location | Impact | Recommendation |
|-------|----------|--------|----------------|
| Quadratic refinement in BLS | `bls_like_search.py:125-151` | O(R*N) per period | Vectorize t0 refinement |
| Dense design matrices | `pixel_timeseries.py:428-466` | Memory explosion | Use sparse matrices |

### Moderate (Consider for Future)

| Issue | Location | Impact | Recommendation |
|-------|----------|--------|----------------|
| Sequential sector processing | `periodogram.py:369-416` | Linear in sectors | Add parallel option |
| Per-hypothesis PRF evaluation | `aperture_prediction.py:268-354` | O(H * A) evaluations | Cache common PRF grids |
| Multiple array copies in split | `periodogram.py:92-109` | Memory overhead | Use view-based splitting |

### Low Priority

| Issue | Location | Impact | Recommendation |
|-------|----------|--------|----------------|
| Python loop in IG | `mlx_detection.py:248-252` | Small step counts | Acceptable |
| Candidate clustering | `periodogram.py:226-242` | Rare worst case | Add early termination |

---

## 6. Recommendations for Open-Source Release

### 6.1 Immediate Actions

1. **Add benchmarks** for multi-sector scenarios (5, 10, 27 sectors)
2. **Document memory requirements** in README (e.g., "16GB recommended for 10+ sectors")
3. **Add progress callbacks** to long-running functions for user feedback

### 6.2 Performance Documentation

Add a section in documentation covering:
- Expected runtime scaling with sector count
- Memory requirements per sector
- When to use downsampling (`downsample_factor` parameter)
- MLX acceleration benefits and requirements

### 6.3 Future Optimization Paths

1. **Parallel sector processing:** Add `use_processes` parameter to `tls_search_per_sector`
2. **Streaming BLS:** Process period grid in chunks to reduce peak memory
3. **Sparse pixel fitting:** Replace dense design matrices with scipy.sparse

---

## 7. Conclusion

The `compute/` module is architecturally sound for single-sector and moderate multi-sector use cases. The primary scalability concerns are:

1. **BLS refinement loops** - Currently O(R*N) per period, easily vectorizable
2. **Pixel time-series fitting** - Dense matrices limit scalability
3. **Sequential sector processing** - Linear scaling, parallelizable

For typical community use (1-5 sectors, <100k data points), performance should be acceptable. Heavy users (10+ sectors, full mission data) may encounter bottlenecks that require the optimizations outlined above.

The MLX integration provides a forward-looking acceleration path for Apple Silicon users, demonstrating good architectural foresight.
