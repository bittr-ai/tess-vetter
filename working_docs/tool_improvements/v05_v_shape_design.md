# V05 Transit Shape (V-Shape) Check - Design Document

**Check ID:** V05
**Check Name:** `v_shape`
**Status:** Draft
**Author:** Claude Code
**Date:** 2026-01-08

---

## 1. Current State

### 1.1 Implementation Summary

The current `check_v_shape` function (lines 717-812 of `lc_checks.py`) distinguishes U-shaped (planetary) from V-shaped (grazing EB) transits using a simple depth ratio metric.

**Current Algorithm:**
1. Phase-fold the light curve centered on transit (phase 0 = mid-transit)
2. Define three regions based on `half_dur = duration_days / period / 2`:
   - **Ingress:** `-half_dur < phase < -half_dur/2`
   - **Bottom:** `-half_dur/4 < phase < half_dur/4`
   - **Egress:** `half_dur/2 < phase < half_dur`
3. Compute `depth_bottom` and `depth_edge` (median of ingress+egress)
4. Calculate `shape_ratio = depth_bottom / depth_edge`
5. Pass if `shape_ratio > 1.3`

**Current Output Fields:**
- `depth_bottom`, `depth_edge`, `shape_ratio`
- `shape` ("U-shaped" or "V-shaped")
- `n_bottom_points`, `n_edge_points`

### 1.2 Limitations

1. **No uncertainty quantification:** The `shape_ratio` has no associated error bar, making it impossible to assess significance.

2. **Fixed region boundaries:** The ingress/egress regions are defined as fractions of the input duration, which may be inaccurate for short-period or poorly-characterized candidates.

3. **No cadence awareness:** A 30-minute cadence TESS light curve may have only 1-2 points in each region for a 2-hour transit, yet the check treats this the same as 2-minute cadence data.

4. **Conflation of grazing planets and EBs:** A high-impact-parameter planet (b > 0.8) can show a V-shaped transit due to limb darkening, but is still a valid planet. The current check cannot distinguish this from a grazing EB.

5. **No tF/tT ratio:** The check does not compute the physically meaningful flat-bottom to total duration ratio used in the literature.

---

## 2. Problems Identified (from Spec)

### 2.1 Duration/Cadence Sensitivity

The current fixed-fraction region definitions fail when:
- **Duration is overestimated:** Bottom region extends into ingress/egress, diluting the shape signal
- **Duration is underestimated:** Edge regions miss the actual ingress/egress slopes
- **Low cadence (30-min):** A 2-hour transit has ~4 points total; fixed regions may have 0-1 points each

**Impact:** False passes (calling EBs "U-shaped") or false fails (calling planets "V-shaped") depending on duration error direction.

### 2.2 Limb Darkening Effects

For high-impact-parameter transits (b > 0.7), limb darkening causes:
- Curved transit floor (not truly flat)
- Asymmetric ingress/egress depths
- Overall V-like appearance even for bonafide planets

The current check uses a single threshold (1.3) regardless of stellar type or expected limb darkening.

### 2.3 Grazing Planet vs EB Conflation

The current binary "U-shaped"/"V-shaped" output conflates:
- **Grazing planet (b ~ 0.9-1.0):** Valid planet, V-shaped transit, smaller apparent depth
- **Grazing EB:** False positive, V-shaped transit, diluted depth from similar-size companion

These require different follow-up actions but are currently indistinguishable.

---

## 3. Proposed Improvements

### 3.1 Trapezoid Model Fit

Replace the simple median comparison with a trapezoid model fit that directly estimates shape parameters.

**Trapezoid Parameters:**
- `t_total` (T): Total transit duration (first to fourth contact)
- `t_flat` (F): Flat-bottom duration (second to third contact)
- `depth`: Transit depth at flat bottom
- `t0`: Mid-transit time (fixed from input)

**Algorithm:**
1. Phase-fold and bin the light curve (adaptive binning based on cadence)
2. Fit a trapezoid model using weighted least squares or scipy.optimize
3. Extract `t_flat` and `t_total` from the fit
4. Compute `tF_tT_ratio = t_flat / t_total`

**Rationale:** The tF/tT ratio is the standard metric in the literature (Seager & Mallen-Ornelas 2003). It directly encodes the transit geometry:
- tF/tT ~ 0: Pure V-shape (grazing)
- tF/tT ~ 1: Pure box (central transit, no limb darkening)
- tF/tT ~ 0.5-0.8: Typical planet with moderate impact parameter

### 3.2 Uncertainty Estimation

Add bootstrap or analytic uncertainty for the shape metric.

**Bootstrap Method (preferred):**
1. Resample in-transit points with replacement (N=100-500 iterations)
2. Refit trapezoid model for each resample
3. Report `tflat_ttotal_ratio_err` as the 16th-84th percentile range

**Analytic Method (fallback):**
For low-N cases, propagate photometric uncertainties through the fit using the Fisher information matrix.

### 3.3 Minimum Sampling Requirements

Gate the check based on data quality.

**Proposed Thresholds:**
- `min_points_in_transit`: 10 (total in-transit points across all epochs)
- `min_transit_coverage`: 0.6 (fraction of transit phases with data)
- `min_snr_per_point`: 3.0 (depth / point scatter)

When thresholds are not met:
- Return `passed=True` with `confidence < 0.3`
- Add warning: `insufficient_sampling` or `low_snr`

### 3.4 Grazing Planet Indicator

Add a separate flag to distinguish grazing planet candidates from likely EBs.

**Decision Logic:**
1. If `tflat_ttotal_ratio > 0.3`: "U-shaped" (normal transit)
2. If `0.1 < tflat_ttotal_ratio <= 0.3` AND `depth < grazing_depth_threshold`:
   - "grazing_planet_candidate" (V-shaped but consistent with high-b planet)
3. If `tflat_ttotal_ratio <= 0.1` OR `depth > grazing_depth_threshold`:
   - "likely_eb" (pure V-shape or too deep for grazing planet)

**grazing_depth_threshold:** 5% (50,000 ppm) - a grazing Jupiter-on-Sun transit is ~0.5%, while a grazing EB is typically >1%.

### 3.5 Limb Darkening Awareness

For improved accuracy, accept optional limb darkening coefficients.

**Implementation:**
- If stellar parameters provided, lookup TESS limb darkening from Claret (2018)
- Adjust the tF/tT threshold based on expected LD curvature
- Flag `limb_darkening_applied` in details

This is optional for v1 and can be deferred to v2.

---

## 4. Recommended Defaults

### 4.1 Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tflat_ttotal_threshold` | 0.15 | Below this, flag as V-shaped |
| `grazing_depth_ppm` | 50000 | Max depth for grazing planet |
| `min_points_in_transit` | 10 | Minimum total in-transit points |
| `min_transit_coverage` | 0.6 | Minimum phase coverage fraction |
| `shape_ratio_legacy_threshold` | 1.3 | Legacy threshold (kept for compatibility) |

### 4.2 Minimum Data Requirements

| Regime | Points | Coverage | Confidence | Action |
|--------|--------|----------|------------|--------|
| Good | >= 30 | >= 0.8 | 0.7-0.9 | Full analysis |
| Marginal | 10-30 | 0.6-0.8 | 0.4-0.7 | Analysis with warning |
| Insufficient | < 10 | < 0.6 | 0.2-0.3 | Skip, return low-confidence pass |

### 4.3 Configuration Dataclass

```python
@dataclass
class VShapeConfig:
    """Configuration for V05 transit shape check."""

    # Primary thresholds
    tflat_ttotal_threshold: float = 0.15
    grazing_depth_ppm: float = 50000.0

    # Minimum data requirements
    min_points_in_transit: int = 10
    min_transit_coverage: float = 0.6

    # Bootstrap settings
    n_bootstrap: int = 100
    bootstrap_ci: float = 0.68  # 1-sigma

    # Legacy compatibility
    shape_ratio_threshold: float = 1.3
```

---

## 5. Required Output Fields (Additive)

### 5.1 New Fields

| Field | Type | Description |
|-------|------|-------------|
| `t_flat_hours` | float | Flat-bottom duration from trapezoid fit |
| `t_total_hours` | float | Total transit duration from trapezoid fit |
| `tflat_ttotal_ratio` | float | Ratio of flat to total duration |
| `tflat_ttotal_ratio_err` | float | Bootstrap uncertainty on ratio |
| `shape_metric_uncertainty` | float | Alias for `tflat_ttotal_ratio_err` |
| `grazing_indicator` | str | One of: "normal", "grazing_planet_candidate", "likely_eb" |
| `transit_coverage` | float | Fraction of transit phases with data |
| `warnings` | list[str] | List of quality warnings |

### 5.2 Preserved Fields (Legacy Compatibility)

| Field | Type | Description |
|-------|------|-------------|
| `depth_bottom` | float | Median depth at transit bottom |
| `depth_edge` | float | Median depth at ingress/egress |
| `shape_ratio` | float | Legacy ratio (depth_bottom/depth_edge) |
| `shape` | str | Legacy label ("U-shaped" or "V-shaped") |
| `n_bottom_points` | int | Points in bottom region |
| `n_edge_points` | int | Points in edge regions |

### 5.3 Example Output

```python
{
    # New fields (v2)
    "t_flat_hours": 1.2,
    "t_total_hours": 2.8,
    "tflat_ttotal_ratio": 0.43,
    "tflat_ttotal_ratio_err": 0.08,
    "shape_metric_uncertainty": 0.08,
    "grazing_indicator": "normal",
    "transit_coverage": 0.85,
    "warnings": [],

    # Legacy fields (preserved)
    "depth_bottom": 0.0012,
    "depth_edge": 0.0006,
    "shape_ratio": 2.0,
    "shape": "U-shaped",
    "n_bottom_points": 45,
    "n_edge_points": 32,
}
```

---

## 6. Test Matrix

### 6.1 Synthetic Test Cases

| Case | tF/tT | Depth (ppm) | Cadence | N_transits | Expected Outcome |
|------|-------|-------------|---------|------------|------------------|
| U-shaped planet | 0.6 | 1000 | 2-min | 10 | PASS, "normal" |
| V-shaped EB | 0.05 | 80000 | 2-min | 5 | FAIL, "likely_eb" |
| Grazing planet | 0.15 | 500 | 2-min | 8 | PASS, "grazing_planet_candidate" |
| Grazing EB | 0.08 | 60000 | 2-min | 6 | FAIL, "likely_eb" |
| Low-N U-shaped | 0.5 | 2000 | 30-min | 3 | PASS, low confidence, warning |
| Insufficient sampling | 0.5 | 1000 | 30-min | 1 | PASS, conf=0.2, "insufficient_sampling" |
| High-b planet (b=0.9) | 0.25 | 300 | 2-min | 12 | PASS, "grazing_planet_candidate" |
| Duration error (+50%) | 0.4 | 1500 | 2-min | 7 | PASS, warning: "duration_uncertain" |

### 6.2 Test Scenarios

**Scenario 1: Clear U-shaped planet (baseline)**
```python
def test_u_shaped_planet():
    # Generate synthetic U-shaped transit (tF/tT = 0.6)
    lc = generate_trapezoid_transit(t_flat=1.5, t_total=2.5, depth=1000e-6)
    result = check_v_shape(lc, period=3.5, t0=100.0, duration_hours=2.5)

    assert result.passed is True
    assert result.details["tflat_ttotal_ratio"] > 0.4
    assert result.details["grazing_indicator"] == "normal"
    assert result.confidence >= 0.7
```

**Scenario 2: Clear V-shaped EB**
```python
def test_v_shaped_eb():
    # Generate synthetic V-shaped transit (tF/tT ~ 0)
    lc = generate_v_transit(depth=0.08)  # 8% depth
    result = check_v_shape(lc, period=2.0, t0=100.0, duration_hours=3.0)

    assert result.passed is False
    assert result.details["tflat_ttotal_ratio"] < 0.1
    assert result.details["grazing_indicator"] == "likely_eb"
```

**Scenario 3: Grazing planet (ambiguous)**
```python
def test_grazing_planet():
    # Generate grazing transit (b ~ 0.95, small depth, some V-shape)
    lc = generate_trapezoid_transit(t_flat=0.3, t_total=2.0, depth=500e-6)
    result = check_v_shape(lc, period=5.0, t0=100.0, duration_hours=2.0)

    assert result.passed is True  # Don't reject grazing planets
    assert result.details["tflat_ttotal_ratio"] < 0.3
    assert result.details["grazing_indicator"] == "grazing_planet_candidate"
```

**Scenario 4: Insufficient sampling**
```python
def test_insufficient_sampling():
    # Only 5 in-transit points
    lc = generate_sparse_transit(n_in_transit=5)
    result = check_v_shape(lc, period=10.0, t0=100.0, duration_hours=4.0)

    assert result.passed is True  # Cannot reject with insufficient data
    assert result.confidence <= 0.3
    assert "insufficient_sampling" in result.details["warnings"]
```

**Scenario 5: 30-minute cadence**
```python
def test_30min_cadence():
    # 30-min cadence, 2-hour transit, 3 transits
    lc = generate_transit_at_cadence(cadence_min=30, duration_hours=2.0, n_transits=3)
    result = check_v_shape(lc, period=4.0, t0=100.0, duration_hours=2.0)

    # Should have degraded confidence due to sparse sampling
    assert result.confidence < 0.6
    assert "low_sampling" in result.details["warnings"] or result.confidence < 0.5
```

**Scenario 6: Duration mismatch**
```python
def test_duration_mismatch():
    # True duration 2h, but input says 3h (50% overestimate)
    lc = generate_trapezoid_transit(t_flat=1.0, t_total=2.0, depth=1000e-6)
    result = check_v_shape(lc, period=5.0, t0=100.0, duration_hours=3.0)  # Wrong!

    # Should still work reasonably or flag uncertainty
    # The fit should recover closer to true duration
    assert result.details.get("duration_uncertain", False) or \
           abs(result.details["t_total_hours"] - 2.0) < 0.5
```

### 6.3 Regression Tests

Maintain golden outputs for known targets:
- Pi Mensae c (confirmed planet, U-shaped)
- TOI-XXX (known EB, V-shaped) - select appropriate example
- High-impact-parameter planets from literature

---

## 7. Backward Compatibility

### 7.1 Changes Summary

| Aspect | Change Type | Details |
|--------|-------------|---------|
| Function signature | Unchanged | Same inputs: `(lightcurve, period, t0, duration_hours)` |
| Return type | Unchanged | Still returns `VetterCheckResult` |
| `passed` semantics | Unchanged | True = consistent with planet, False = likely EB |
| Legacy fields | Preserved | `shape_ratio`, `shape`, `depth_*`, `n_*_points` |
| New fields | Additive | `t_flat_hours`, `tflat_ttotal_ratio`, etc. |
| Default thresholds | Gated | New thresholds behind config; legacy behavior default |

### 7.2 Migration Path

**Phase 1 (v1.1):** Add new fields alongside legacy fields. Default behavior unchanged.

**Phase 2 (v1.2):** Add `VShapeConfig` parameter. Default config matches legacy behavior.

**Phase 3 (v2.0):** New config becomes default. Legacy `shape_ratio` threshold deprecated but still computed.

### 7.3 API Compatibility

The public API wrapper in `bittr_tess_vetter.api.lc_only.v_shape()` remains unchanged:

```python
def v_shape(lc: LightCurve, ephemeris: Ephemeris) -> CheckResult:
    # Signature unchanged
    # New fields appear in result.details
```

---

## 8. Citations

### 8.1 Primary References

| Reference | Bibcode | Relevance |
|-----------|---------|-----------|
| Seager & Mallen-Ornelas 2003 | 2003ApJ...585.1038S | tF/tT ratio definition, transit geometry |
| Thompson et al. 2018 | 2018ApJS..235...38T | Kepler DR25 V-shape metric (Not Transit-Like) |
| Prsa et al. 2011 | 2011AJ....141...83P | EB morphology classification |

### 8.2 Secondary References

| Reference | Bibcode | Relevance |
|-----------|---------|-----------|
| Coughlin et al. 2016 | 2016ApJS..224...12C | Robovetter methodology |
| Claret 2018 | 2018A&A...618A..20C | TESS limb darkening coefficients |
| Mandel & Agol 2002 | 2002ApJ...580L.171M | Analytic transit model |

### 8.3 Docstring Citation Block

```python
"""V05: Distinguish U-shaped (planet) vs V-shaped (grazing EB) transits.

References:
    [1] Seager & Mallen-Ornelas 2003, ApJ 585, 1038 (2003ApJ...585.1038S)
        Section 3: Transit shape parameters tF/tT and impact parameter b
    [2] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T)
        Section 3.1: Not Transit-Like (V-shape) metric in DR25 Robovetter
    [3] Prsa et al. 2011, AJ 141, 83 (2011AJ....141...83P)
        Section 3.2: Morphology classification of eclipsing binaries
"""
```

---

## 9. Implementation Notes

### 9.1 Trapezoid Fitting

Use `scipy.optimize.curve_fit` or `lmfit` for the trapezoid model:

```python
def trapezoid_model(phase, t_flat, t_total, depth):
    """Symmetric trapezoid transit model."""
    half_flat = t_flat / 2
    half_total = t_total / 2

    # Regions: OOT, ingress, flat, egress
    flux = np.ones_like(phase)

    # Flat bottom
    flat_mask = np.abs(phase) < half_flat
    flux[flat_mask] = 1 - depth

    # Ingress/egress slopes
    ingress_mask = (phase < -half_flat) & (phase > -half_total)
    egress_mask = (phase > half_flat) & (phase < half_total)

    if half_total > half_flat:
        slope = depth / (half_total - half_flat)
        flux[ingress_mask] = 1 - depth + slope * (-phase[ingress_mask] - half_flat)
        flux[egress_mask] = 1 - depth + slope * (phase[egress_mask] - half_flat)

    return flux
```

### 9.2 Performance Considerations

- Bootstrap with N=100 adds ~50ms per check (acceptable)
- Trapezoid fit is faster than MCMC limb-darkened model
- Pre-binning reduces fit complexity for long light curves

### 9.3 Edge Cases

1. **Zero flat duration:** When t_flat ~ 0, set floor at 0.01 * t_total
2. **Negative ratio:** Clamp to [0, 1] and add warning
3. **Fit non-convergence:** Fall back to legacy shape_ratio method

---

## 10. Summary

This design improves V05 by:

1. **Replacing ad-hoc depth ratio with physics-based tF/tT metric**
2. **Adding uncertainty quantification via bootstrap**
3. **Distinguishing grazing planets from grazing EBs**
4. **Gating analysis on data quality with explicit warnings**
5. **Maintaining full backward compatibility**

The changes align with Kepler Robovetter methodology while addressing TESS-specific challenges (shorter baselines, mixed cadences, lower SNR per transit).
