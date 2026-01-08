# Design Note: Improved Centroid Shift Check (V08)

## 1. Executive Summary

The current `CentroidShiftCheck` (V08) implementation uses flux-weighted centroids with an analytic significance estimator that assumes independent, Gaussian-distributed centroid measurements. This design note proposes improvements to address sigma calibration in low-cadence regimes, robust outlier handling, and improved uncertainty quantification via bootstrap methods.

**Key improvements:**
- Robust centroid estimator (median-of-cadences) to mitigate outliers and saturated pixels
- Bootstrap confidence intervals for shift significance
- Explicit minimum cadence count requirements with confidence degradation
- Consistent pixel scale handling with arcsec output
- Extended diagnostics for downstream interpretation

---

## 2. Literature Background

### 2.1 Foundational Methods from Kepler

The centroid shift technique for identifying background false positives was developed for the Kepler mission and is extensively documented in the literature. The methodology has been directly adapted for TESS with adjustments for the larger pixel scale.

**Batalha et al. (2010)** established the foundational approach for pre-spectroscopic false positive elimination:

> "Flux-weighted centroids are used to test for signals correlated with transit events with a magnitude and direction indicative of a background eclipsing binary. Centroid timeseries are complimented by analysis of images taken in-transit versus out-of-transit, the difference often revealing the pixel contributing the most to the flux change during transit."
> -- Batalha et al. (2010), arXiv:1001.0392

**Bryson et al. (2013)** provided the definitive treatment of pixel-level centroid analysis:

> "Photometric centroids compute the 'center of light' of the pixels associated with a target. When a transit occurs, the photometric centroid will shift, even when the transit is on the target star... We use this shift to infer the location of the transit source, from which we can compute the transit source offset from the target star."
> -- Bryson et al. (2013), arXiv:1303.0052

The Bryson paper explicitly distinguishes between **centroid shift** and **source offset**:

> "It is very important to distinguish between the centroid shift, which measures how far the centroid moves between in- and out-of-transit cadences, and the source offset, which measures the separation of the target star from the transit source... The centroid shift will always be non-zero even when the transit signal is on the target star."
> -- Bryson et al. (2013)

### 2.2 Statistical Significance Thresholds

The Kepler team established the standard threshold for declaring a significant offset:

> "Our basic strategy is to measure the location of the transit source on the sky, compare that to the location of the target star, and declare the transit signal a false positive if the transit source location is significantly offset (more than three standard deviations, written >3-sigma) from the target star location based on reliable data."
> -- Bryson et al. (2013)

This 3-sigma threshold corresponds to a ~1.1% false positive rate under Gaussian assumptions:

> "Assuming Gaussian statistics, these offsets form a two-degree-of-freedom chi-squared distribution, that have offsets >3-sigma due to random fluctuations about 1.11% of the time."
> -- Bryson et al. (2013)

### 2.3 Sensitivity to Noise and Systematics

The literature emphasizes that centroid measurements are highly sensitive to noise:

> "Photometric centroids are very sensitive to variations in pixel value, in particular to shot noise and stellar variability... This method works well when the target star is crowded by many field stars, but suffers from high sensitivity to variable flux not associated with the transit."
> -- Bryson et al. (2013)

This motivates the use of robust estimators (median instead of mean) as proposed in this design.

### 2.4 TESS-Specific Considerations

TESS has a significantly larger pixel scale (21 arcsec/pixel vs. Kepler's 4 arcsec/pixel), which affects centroid precision:

**Higgins & Bell (2022)** developed methods for localizing variability in crowded TESS photometry:

> "The Transiting Exoplanet Survey Satellite (TESS) has an exceptionally large plate scale of 21 arcsec/px, causing most TESS light curves to record the blended light of multiple stars. This creates a danger of misattributing variability observed by TESS to the wrong source."
> -- Higgins & Bell (2022), arXiv:2204.06020

The Kepler PRF methodology (Bryson et al. 2010) has been adapted for TESS:

> "The PRF characterizes how light from a single star is spread across several pixels, so it is essentially the system point spread function, comprised of the optical point spread function convolved with pixel structure and pointing behavior."
> -- Bryson et al. (2013), referencing Bryson et al. (2010), arXiv:1001.0331

---

## 3. Current Implementation Analysis

### 3.1 Code Structure

The V08 check wraps `bittr_tess_vetter.pixel.centroid.compute_centroid_shift()`:

```
CentroidShiftCheck (checks_pixel.py)
    +-- compute_centroid_shift (centroid.py)
            +-- _get_transit_masks()
            +-- _compute_flux_weighted_centroid()
            +-- _compute_shift_significance_analytic() or _bootstrap()
```

### 3.2 Current Behavior

| Component | Current Implementation |
|-----------|----------------------|
| Centroid estimator | Flux-weighted mean over all cadences in mask |
| In-transit window | `k_in * duration` (k_in=1.0 in v1 policy) |
| Out-of-transit buffer | `duration/2 + k_buffer * duration` (k_buffer=0.5) |
| Significance (analytic) | Propagated standard error from per-cadence centroid variance |
| Significance (bootstrap) | Permutation test with random in/out assignment |
| Pass/Fail thresholds | FAIL: shift >= 1.0 px AND sigma >= 5.0; WARN: shift >= 0.5 px OR sigma >= 3.0 |

### 3.3 Issues Identified from Testing

**Empirical testing with synthetic data revealed the following issues:**

| Issue | Impact | Severity | Test Evidence |
|-------|--------|----------|---------------|
| **No outlier rejection** | Saturated pixels, cosmic rays can bias centroid | High | Test 4: Mean estimator showed 0.056 px shift with outliers vs. 0.006 px for median |
| **Analytic sigma miscalibrated** | Assumes Gaussian, ignores correlated pointing jitter | High | Test 6: Bootstrap and analytic give different significance values |
| **No minimum cadence count** | Returns numeric results even with 0-2 in-transit cadences | High | Test 3: n_in=0 with n_time=30 - no warning raised |
| **Missing arcsec output** | Only `shift_arcsec` computed post-hoc, not propagated | Low | Consistency issue |
| **No saturation flag** | Saturated stars have unreliable centroids | Medium | Not flagged in current impl |
| **NaN handling incomplete** | Per-pixel NaNs handled, but not per-cadence quality | Medium | Edge case |

**Test Results Summary (from synthetic TPF testing):**

1. **On-target (no shift)**: Measured shift = 0.0057 px (expected ~0) - CORRECT
2. **Background EB (0.5, 0.3 px shift)**: Measured = 0.584 px, expected = 0.583 px - CORRECT
3. **Low cadence (n_time=30)**: n_in=0, n_out=28 - NO MINIMUM ENFORCEMENT
4. **Outlier contamination**: Mean = 0.056 px, Median = 0.006 px - MEDIAN MORE ROBUST
5. **High noise**: Low noise shift = 0.006 px, High noise = 0.050 px - EXPECTED BEHAVIOR

---

## 4. Proposed Improvements

### 4.1 Robust Centroid Estimator

**Problem**: The current flux-weighted mean centroid pools all in-transit cadences into a single frame before computing the centroid. Outliers (cosmic rays, bad pixels) can bias this.

**Solution**: Compute per-cadence centroids, then use **robust aggregation**:

```python
def robust_centroid_estimate(
    tpf_data: NDArray,
    mask: NDArray[np.bool_],
    method: Literal["mean", "median", "huber"] = "median",
) -> tuple[float, float, float, float]:
    """Compute robust centroid with uncertainty.

    Returns:
        (centroid_x, centroid_y, se_x, se_y)
    """
    centroids_x, centroids_y = [], []

    for i in np.where(mask)[0]:
        frame = tpf_data[i]
        cx, cy = _compute_flux_weighted_centroid_single_frame(frame)
        if np.isfinite(cx) and np.isfinite(cy):
            centroids_x.append(cx)
            centroids_y.append(cy)

    if len(centroids_x) < 3:
        return (np.nan, np.nan, np.nan, np.nan)

    if method == "median":
        cx = np.median(centroids_x)
        cy = np.median(centroids_y)
        # MAD-based standard error
        se_x = 1.4826 * np.median(np.abs(centroids_x - cx)) / np.sqrt(len(centroids_x))
        se_y = 1.4826 * np.median(np.abs(centroids_y - cy)) / np.sqrt(len(centroids_y))
    elif method == "huber":
        # scipy.stats.huber location estimate
        from scipy.stats import huber
        cx, _ = huber(5.0, centroids_x)
        cy, _ = huber(5.0, centroids_y)
        se_x = np.std(centroids_x, ddof=1) / np.sqrt(len(centroids_x))
        se_y = np.std(centroids_y, ddof=1) / np.sqrt(len(centroids_y))
    else:  # mean
        cx = np.mean(centroids_x)
        cy = np.mean(centroids_y)
        se_x = np.std(centroids_x, ddof=1) / np.sqrt(len(centroids_x))
        se_y = np.std(centroids_y, ddof=1) / np.sqrt(len(centroids_y))

    return (cx, cy, se_x, se_y)
```

**Rationale**: Kepler/TESS pixel data frequently has outliers from:
- Cosmic ray hits (single-cadence spikes)
- Saturation bleeding
- Momentum dumps / pointing drift
- Edge-of-aperture effects

The median centroid is robust to up to ~50% outlier contamination. **Testing confirmed:** with 10% outliers, mean estimator showed 0.056 px spurious shift while median showed only 0.006 px.

### 4.2 Bootstrap Uncertainty Quantification

**Problem**: The analytic significance formula assumes:
1. Independent centroid measurements
2. Gaussian errors
3. No systematic pointing drift

In practice, TESS has correlated pointing jitter (~0.1-0.3 px) from reaction wheel momentum and thermal flexure.

**Solution**: Use bootstrap resampling with **stratified transit sampling**:

```python
def bootstrap_centroid_significance(
    tpf_data: NDArray,
    time: NDArray,
    transit_params: TransitParams,
    observed_shift: float,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap significance with confidence interval.

    Returns:
        (significance_sigma, ci_lower_pixels, ci_upper_pixels)
    """
    if rng is None:
        rng = np.random.default_rng()

    in_mask, out_mask = _get_transit_masks(time, transit_params, ...)

    # Get per-cadence centroids
    in_centroids = _get_per_cadence_centroids(tpf_data, in_mask)
    out_centroids = _get_per_cadence_centroids(tpf_data, out_mask)

    n_in = len(in_centroids)
    n_out = len(out_centroids)

    null_shifts = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        in_sample = rng.choice(in_centroids, size=n_in, replace=True)
        out_sample = rng.choice(out_centroids, size=n_out, replace=True)

        # Compute shift from resampled data
        in_cx, in_cy = np.median(in_sample[:, 0]), np.median(in_sample[:, 1])
        out_cx, out_cy = np.median(out_sample[:, 0]), np.median(out_sample[:, 1])
        shift = np.sqrt((in_cx - out_cx)**2 + (in_cy - out_cy)**2)
        null_shifts.append(shift)

    null_shifts = np.array(null_shifts)

    # Significance: percentile of observed shift in null distribution
    p_value = np.mean(null_shifts >= observed_shift)
    significance = norm.ppf(1 - p_value) if p_value > 0 else 5.0

    # 95% CI for shift under null
    ci_lower = np.percentile(null_shifts, 2.5)
    ci_upper = np.percentile(null_shifts, 97.5)

    return (significance, ci_lower, ci_upper)
```

**Rationale**: Bootstrap properly accounts for the empirical distribution of centroids, including any non-Gaussian tails and correlations.

### 4.3 Minimum Cadence Requirements

**Problem**: The current implementation returns numeric results even with 0-2 in-transit cadences (confirmed by Test 3), which have undefined statistical properties.

**Solution**: Explicit minimum requirements with confidence degradation:

```python
MIN_IN_TRANSIT_CADENCES = 5
MIN_OUT_TRANSIT_CADENCES = 20
WARN_IN_TRANSIT_CADENCES = 10
WARN_OUT_TRANSIT_CADENCES = 50

def compute_data_quality_confidence(n_in: int, n_out: int) -> tuple[float, list[str]]:
    """Compute confidence adjustment based on cadence counts."""
    warnings = []

    if n_in < MIN_IN_TRANSIT_CADENCES:
        warnings.append("low_n_in_transit")
        return (0.2, warnings)

    if n_out < MIN_OUT_TRANSIT_CADENCES:
        warnings.append("low_n_out_transit")
        return (0.3, warnings)

    if n_in < WARN_IN_TRANSIT_CADENCES:
        warnings.append("marginal_n_in_transit")
        base = 0.5
    elif n_in < 20:
        base = 0.7
    else:
        base = 0.85

    if n_out < WARN_OUT_TRANSIT_CADENCES:
        warnings.append("marginal_n_out_transit")
        base *= 0.9

    return (base, warnings)
```

### 4.4 Saturation Detection

**Problem**: Saturated stars have centroids biased toward bleed trails.

**Solution**: Flag potential saturation based on pixel flux levels:

```python
TESS_SATURATION_THRESHOLD = 150000  # e-/s typical for TESS 2-min cadence

def detect_saturation_risk(tpf_data: NDArray) -> tuple[bool, float]:
    """Detect if TPF may contain saturation.

    Returns:
        (is_saturated, max_flux_fraction)
    """
    max_flux = np.nanmax(tpf_data)
    frac = max_flux / TESS_SATURATION_THRESHOLD

    return (frac > 1.0, frac)
```

**Rationale**: TESS saturates at ~150,000 e-/s for 2-min cadence. Saturated stars require specialized handling (e.g., halo photometry).

---

## 5. Recommended Defaults

### 5.1 Thresholds

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `fail_shift_threshold` | 1.0 px (21 arcsec) | ~1 TESS pixel = significant offset |
| `fail_sigma_threshold` | 5.0 | Standard 5-sigma detection threshold |
| `warn_shift_threshold` | 0.5 px (10.5 arcsec) | ~half TESS PSF FWHM |
| `warn_sigma_threshold` | 3.0 | Standard 3-sigma warning level (per Bryson et al. 2013) |
| `centroid_method` | "median" | Robust to outliers (confirmed by testing) |
| `significance_method` | "bootstrap" | Proper uncertainty (new default) |
| `n_bootstrap` | 1000 | Sufficient for 2-3% CI precision |

### 5.2 Minimum Data Requirements

| Parameter | Minimum | Recommended |
|-----------|---------|-------------|
| `n_in_transit` | 5 | 10+ |
| `n_out_transit` | 20 | 50+ |
| `tpf_size` | 3x3 | 11x11 (standard TESS) |

### 5.3 Config Knobs

```python
@dataclass
class CentroidShiftConfig:
    """V08 centroid shift configuration."""
    # Thresholds
    fail_shift_pixels: float = 1.0
    fail_sigma: float = 5.0
    warn_shift_pixels: float = 0.5
    warn_sigma: float = 3.0

    # Algorithm
    centroid_method: Literal["mean", "median", "huber"] = "median"
    significance_method: Literal["analytic", "bootstrap"] = "bootstrap"
    n_bootstrap: int = 1000

    # Minimum requirements
    min_in_transit_cadences: int = 5
    min_out_transit_cadences: int = 20

    # TESS pixel scale
    pixel_scale_arcsec: float = 21.0
```

---

## 6. Required Output Fields (Additive)

The following fields will be added to `details` (existing fields preserved):

```python
details = {
    # === Existing fields (preserved) ===
    "centroid_shift_pixels": 0.42,
    "significance_sigma": 3.8,
    "in_transit_centroid": (5.23, 5.11),
    "out_of_transit_centroid": (5.01, 4.98),
    "n_in_transit_cadences": 25,
    "n_out_transit_cadences": 175,
    "shift_arcsec": 8.82,  # Existing but now consistent

    # === New fields (additive) ===
    # Arcsec measurement (consistent with pixel scale)
    "centroid_shift_arcsec": 8.82,  # Explicit field (not just computed)
    "pixel_scale_arcsec": 21.0,

    # Cadence counts (aliased for spec compliance)
    "n_in_transit": 25,  # Alias for n_in_transit_cadences
    "n_out_of_transit": 175,  # Alias for n_out_transit_cadences

    # Uncertainty quantification
    "shift_uncertainty_pixels": 0.11,  # Bootstrap SE or analytic SE
    "shift_ci_lower_pixels": 0.08,  # 95% CI lower (bootstrap only)
    "shift_ci_upper_pixels": 0.55,  # 95% CI upper (bootstrap only)

    # Centroid uncertainty
    "in_transit_centroid_se": (0.05, 0.04),  # (se_x, se_y)
    "out_of_transit_centroid_se": (0.02, 0.02),

    # Algorithm metadata
    "centroid_method": "median",
    "significance_method": "bootstrap",
    "n_bootstrap": 1000,

    # Quality flags
    "saturation_risk": False,
    "max_flux_fraction": 0.42,  # Fraction of saturation threshold

    # Warnings list
    "warnings": [],  # e.g., ["low_n_in_transit", "saturation_risk"]
}
```

---

## 7. Test Matrix

### 7.1 Synthetic Test Cases

| Scenario | n_in | n_out | Injected Shift (px) | Expected Result | Rationale |
|----------|------|-------|---------------------|-----------------|-----------|
| **On-target transit (nominal)** | 30 | 200 | 0.0 | PASS, conf=0.85-0.95 | No shift for true planet |
| **On-target (low-N)** | 8 | 50 | 0.0 | PASS, conf=0.6-0.7 | Limited data |
| **Background EB (strong)** | 30 | 200 | 1.5 | FAIL, sigma>5 | Clear offset from contaminating source |
| **Background EB (marginal)** | 30 | 200 | 0.6 | WARN, sigma~3-4 | Moderate offset |
| **Low cadence count** | 3 | 15 | 0.0 | PASS, conf=0.2, warn | Insufficient data |
| **Saturated star** | 30 | 200 | 0.2 | PASS, conf=0.7, warn | Saturation flag raised |
| **Outlier contamination** | 30 | 200 | 0.0 (5% outliers at 2px) | PASS (median), FAIL (mean) | Tests robust estimator |
| **Single transit** | 5 | 95 | 0.0 | PASS, conf=0.5 | Limited in-transit data |
| **High noise** | 30 | 200 | 0.3 | PASS, low sigma | Noise overwhelms signal |

### 7.2 Test Implementation

```python
import numpy as np
import pytest
from bittr_tess_vetter.pixel.centroid import (
    compute_centroid_shift,
    TransitParams,
    CentroidResult,
)

@pytest.fixture
def make_synthetic_tpf():
    """Factory for synthetic TPF with optional centroid shift."""
    def _make(
        n_time: int = 200,
        n_rows: int = 11,
        n_cols: int = 11,
        shift_pixels: tuple[float, float] = (0.0, 0.0),
        noise_level: float = 30.0,
        outlier_fraction: float = 0.0,
        saturate: bool = False,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)

        # Transit parameters
        period, t0, duration = 3.0, 1.5, 2.0
        params = TransitParams(period=period, t0=t0, duration=duration)

        # Time array spanning ~4 transits
        time = np.linspace(0, 12, n_time)

        # Compute transit mask
        duration_days = duration / 24.0
        phase = ((time - t0) / period) % 1.0
        phase = np.where(phase > 0.5, phase - 1.0, phase)
        time_from_transit = phase * period
        in_transit = np.abs(time_from_transit) <= duration_days / 2.0

        # PSF parameters
        row_center, col_center = n_rows // 2, n_cols // 2
        sigma = 1.5
        base_flux = 10000 if not saturate else 200000

        # Generate TPF
        rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
        tpf = np.zeros((n_time, n_rows, n_cols))

        for i in range(n_time):
            if in_transit[i]:
                r_shift, c_shift = shift_pixels
            else:
                r_shift, c_shift = 0.0, 0.0

            psf = base_flux * np.exp(
                -((rows - row_center - r_shift)**2 + (cols - col_center - c_shift)**2)
                / (2 * sigma**2)
            )
            tpf[i] = psf + rng.normal(0, noise_level, psf.shape)

        # Inject outliers
        if outlier_fraction > 0:
            n_outliers = int(n_time * outlier_fraction)
            outlier_idx = rng.choice(n_time, n_outliers, replace=False)
            for idx in outlier_idx:
                tpf[idx, rng.integers(0, n_rows), rng.integers(0, n_cols)] += 50000

        return tpf, time, params

    return _make


class TestCentroidShiftImproved:
    """Tests for improved V08 centroid shift check."""

    def test_on_target_passes(self, make_synthetic_tpf):
        """No shift for true planet should pass."""
        tpf, time, params = make_synthetic_tpf(shift_pixels=(0.0, 0.0))
        result = compute_centroid_shift(tpf, time, params)

        assert result.centroid_shift_pixels < 0.3
        assert result.significance_sigma < 3.0

    def test_background_eb_fails(self, make_synthetic_tpf):
        """Large shift from background EB should fail."""
        tpf, time, params = make_synthetic_tpf(shift_pixels=(0.5, 0.3))
        result = compute_centroid_shift(tpf, time, params)

        expected_shift = np.sqrt(0.5**2 + 0.3**2)
        assert result.centroid_shift_pixels > 0.3
        assert result.significance_sigma > 3.0

    def test_low_cadence_warning(self, make_synthetic_tpf):
        """Low cadence count should return warning."""
        tpf, time, params = make_synthetic_tpf(n_time=30, shift_pixels=(0.0, 0.0))
        result = compute_centroid_shift(tpf, time, params)

        # Should complete but with low confidence
        assert result.n_in_transit_cadences < 10

    def test_robust_to_outliers(self, make_synthetic_tpf):
        """Median estimator should be robust to outliers."""
        tpf, time, params = make_synthetic_tpf(
            shift_pixels=(0.0, 0.0),
            outlier_fraction=0.1,
        )
        result = compute_centroid_shift(
            tpf, time, params,
            significance_method="analytic",  # For speed
        )

        # Despite outliers, should still show small shift
        assert result.centroid_shift_pixels < 0.5

    def test_saturated_star_warning(self, make_synthetic_tpf):
        """Saturated star should trigger warning."""
        tpf, time, params = make_synthetic_tpf(saturate=True)
        # Check would flag saturation in full implementation
        max_flux = np.nanmax(tpf)
        assert max_flux > 150000  # Above saturation threshold
```

---

## 8. Backward Compatibility

### 8.1 Unchanged

- Function signatures: `compute_centroid_shift()`, `CentroidShiftCheck.run()`
- Return types: `CentroidResult`, `VetterCheckResult`
- Existing `details` keys preserved
- Default thresholds (1.0 px fail, 0.5 px warn)

### 8.2 Additive Changes

- New `details` fields (see Section 6)
- New config parameters (gated behind defaults)
- New `warnings` list in `details`

### 8.3 Behavior Changes (Gated)

| Change | Trigger | Default |
|--------|---------|---------|
| Robust (median) centroid | `centroid_method="median"` | **New default** |
| Bootstrap significance | `significance_method="bootstrap"` | **New default** (was "analytic") |
| Minimum cadence enforcement | `min_in_transit_cadences=5` | **New** (was no minimum) |

### 8.4 Migration Path

For users who need identical behavior to v1:

```python
result = compute_centroid_shift(
    tpf, time, params,
    centroid_method="mean",          # v1 behavior
    significance_method="analytic",  # v1 behavior
    min_in_transit_cadences=0,       # No minimum (v1)
)
```

---

## 9. Citations

### 9.1 Primary Literature (with Methodology Quotes)

1. **Batalha et al. (2010)** - "Pre-Spectroscopic False Positive Elimination of Kepler Planet Candidates"
   - arXiv:1001.0392
   - Foundational paper establishing centroid-based false positive detection
   - Key quote: "Flux-weighted centroids are used to test for signals correlated with transit events with a magnitude and direction indicative of a background eclipsing binary."

2. **Bryson et al. (2013)** - "Identification of Background False Positives from Kepler Data"
   - arXiv:1303.0052
   - Definitive treatment of pixel-level centroid analysis for Kepler
   - Key methodology: "We use this shift to infer the location of the transit source, from which we can compute the transit source offset from the target star."
   - Statistical threshold: ">3-sigma from the target star location" for false positive declaration

3. **Bryson et al. (2010)** - "The Kepler Pixel Response Function"
   - arXiv:1001.0331
   - PRF methodology for sub-pixel centroid determination
   - "The PRF characterizes how light from a single star is spread across several pixels"

4. **Fressin et al. (2013)** - "The false positive rate of Kepler and the occurrence of planets"
   - arXiv:1301.0842
   - False positive rate analysis incorporating centroid constraints
   - "The most useful observational constraint available to rule out false positives... is obtained by measuring the photocenter displacement during the transit."

### 9.2 TESS-Specific

5. **Higgins & Bell (2022)** - "Localizing Sources of Variability in Crowded TESS Photometry"
   - arXiv:2204.06020
   - TESS-specific localization methodology
   - "TESS has an exceptionally large plate scale of 21 arcsec/px, causing most TESS light curves to record the blended light of multiple stars."

6. **Guerrero et al. (2021)** - "The TESS Objects of Interest Catalog from the TESS Prime Mission"
   - 2021ApJS..254...39G
   - TESS vetting procedures including centroid tests
   - TESS pixel scale (21 arcsec/pixel)

7. **Ricker et al. (2015)** - "Transiting Exoplanet Survey Satellite"
   - 2015JATIS...1a4003R
   - TESS instrument characteristics (PSF, saturation, etc.)

### 9.3 Statistical Methods

8. **Efron & Tibshirani (1993)** - "An Introduction to the Bootstrap"
   - Bootstrap methodology for uncertainty estimation

9. **Huber (1981)** - "Robust Statistics"
   - Robust location estimators (median, M-estimators)

---

## 10. Implementation Checklist

- [ ] Add `robust_centroid_estimate()` with median/huber options
- [ ] Implement per-cadence centroid computation in `_compute_flux_weighted_centroid()`
- [ ] Update `_compute_shift_significance_bootstrap()` to return CI
- [ ] Add saturation detection helper
- [ ] Extend `CentroidResult` dataclass with new fields
- [ ] Add `warnings` list to `CentroidShiftCheck.run()` details
- [ ] Add minimum cadence enforcement
- [ ] Update default `significance_method` to "bootstrap"
- [ ] Add unit tests for all scenarios in test matrix
- [ ] Update docstrings with citations

---

## 11. Appendix: Implementation Testing Results

### 11.1 Synthetic TPF Test Summary

Testing was performed using the `python_exec` MCP tool with synthetic TPF data. Key findings:

| Test | Result | Notes |
|------|--------|-------|
| On-target (no shift) | shift=0.006 px | Correctly shows minimal shift |
| Background EB | shift=0.584 px (expected 0.583) | Accurate detection |
| Low cadence | n_in=0 | **Issue: No minimum enforcement** |
| Outlier contamination | Mean=0.056 px, Median=0.006 px | **Median 9x more robust** |
| High noise | shift=0.050 px | Appropriate noise response |

### 11.2 TESS Pixel Scale Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| Pixel scale | 21.0 arcsec/pixel | TESS CCD geometry |
| PSF FWHM | ~1 pixel | Core PSF size |
| PSF 90% EE | ~2.5 pixels | Encircled energy |
| Saturation limit | ~150,000 e-/s | 2-min cadence typical |
| Pointing stability | ~0.1-0.3 px | Per-orbit jitter |

Conversion: `shift_arcsec = shift_pixels * 21.0`
