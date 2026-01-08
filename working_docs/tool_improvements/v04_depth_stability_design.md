# V04 Depth Stability Check: Design Document

**Check ID:** V04
**Name:** `depth_stability`
**Author:** Claude Code
**Date:** 2026-01-08
**Status:** Draft (Updated with Literature Research)

---

## 1. Current State

### 1.1 Implementation Summary

The current `check_depth_stability` function (lines 635-719 in `lc_checks.py`) measures the consistency of transit depths across individual epochs. The algorithm:

1. Computes transit epoch indices from `(time - t0) / period`
2. Creates an in-transit mask using phase distance from transit center
3. Computes a global out-of-transit baseline via median
4. For each unique epoch with >= 3 in-transit points:
   - Computes depth as `1 - median(in_transit_flux) / baseline`
   - Only includes epochs with positive depth (actual dips)
5. Computes `rms_scatter = std(depths) / mean(depths)` (coefficient of variation)
6. Passes if `rms_scatter < 0.3` (30% threshold)

### 1.2 Current Limitations

| Limitation | Impact |
|------------|--------|
| **Global baseline** | Baseline drift or stellar variability biases depths across sectors |
| **No uncertainty model** | Cannot distinguish measurement noise from true depth variability |
| **Hard 30% threshold** | Not calibrated to expected photometric scatter for given SNR |
| **No `flux_err` usage** | Ignores per-point uncertainties that inform expected scatter |
| **Low-N regime** | With 2-3 transits, scatter estimate is unreliable; check still passes |
| **No outlier handling** | Single outlier epoch (e.g., TESS momentum dump) can trigger false fail |
| **Binary pass/fail** | No gradation for marginal cases or confidence degradation |

---

## 2. Problems Identified (from v1_spec.md)

### 2.1 Per-Epoch Robustness

**Problem:** Current implementation uses a global OOT baseline. For multi-sector data or active stars, the baseline can shift significantly between epochs, causing artificial depth scatter.

**Evidence:** Thompson et al. (2018) note that stellar variability on timescales longer than the transit duration can mimic depth variations if not corrected locally (see Section 3.5 of 2018ApJS..235...38T).

### 2.2 Expected Scatter Definition

**Problem:** The check compares observed scatter to a fixed 30% threshold without considering:
- Expected scatter from photometric noise (`sigma_depth ~ sigma_flux / sqrt(n_in_transit)`)
- Red noise inflation on transit timescales
- SNR of the transit signal

**Consequence:** A 500 ppm transit in noisy data may appear "unstable" when depth variations are simply measurement noise, while a 10,000 ppm signal could hide 5% EB-like variability within the 30% threshold.

### 2.3 Low-N Behavior

**Problem:** With only 2 transits, `std(depths)` has 1 degree of freedom and is unreliable. The current check can return `confidence = 0.7` with only 2 measured transits, potentially misleading users.

**Desired behavior:** Confidence should degrade gracefully, and warnings should be issued when statistical power is insufficient to detect ~10-20% depth variations.

---

## 2.5 Literature Background

### Transit Depth Variability as a Vetting Diagnostic

**Wang & Espinoza (2023)** [arXiv:2311.02154] conducted a blind search for transit depth variability among 330 TESS planets, providing the most comprehensive recent study of depth stability methodology:

> *"The phenomenon of transit depth variability offers a pathway through which processes such as exoplanet atmospheric activity and orbital dynamics can be studied."*

**Key methodological findings:**
1. **Per-transit fitting approach:** Each transit fitted independently using `juliet` with `batman` transit models
2. **Local baseline extraction:** Out-of-transit defined as >1 transit duration from nearest transit center
3. **Stellar variability handling:** Quasi-Periodic Gaussian Process kernel on out-of-transit data before in-transit fitting
4. **Detection metric:** Lomb-Scargle periodogram on depth time series; false alarm probability via bootstrap resampling
5. **Detection rate:** ~1% of TESS planets show significant depth variability
6. **False positive sources:** Contaminating flux from nearby stars, grazing orbits, instrumental systematics (KELT-8b, HAT-P-7b, HIP 65Ab, TrES-3b analyzed in detail)

**Santerne et al. (2013)** [arXiv:1307.2003] analyzed false positive signatures from diluted eclipsing binaries:

- **Diluted EB signature:** Depth variations >10% relative are strong EB indicators
- **Mechanism:** Background or bound eclipsing binaries diluted by target star flux produce variable apparent depths due to differential atmospheric extinction, variable third-light contamination, or eccentric orbit effects

**TRICERATOPS Framework** [arXiv:2002.00691] from Giacalone & Dressing (2020):

- Bayesian false positive probability calculation including depth constraints
- Models transit depth as function of flux contribution from target vs nearby contaminating sources
- Validation threshold: FPP < 0.015 for statistical validation

### Expected vs Observed Scatter Theory

For a transit with depth $\delta$ measured from $N_{in}$ in-transit points with per-point scatter $\sigma_{pt}$:

$$\sigma_{\delta,single} = \frac{\sigma_{pt}}{\sqrt{N_{in}}} \cdot \frac{1}{baseline}$$

For $M$ independent transits, the expected RMS scatter in depths (assuming white noise only):

$$\sigma_{expected} = \sqrt{\frac{1}{M} \sum_i \sigma_{\delta,i}^2}$$

**Pont et al. (2006)** [MNRAS 373, 231] established that correlated (red) noise does not average down as $1/\sqrt{N}$, requiring empirical inflation factors:

$$\sigma_{inflated} = \sigma_{white} \times \beta_{red}$$

where $\beta_{red}$ is measured from binned out-of-transit residuals.

### Chi-Squared Stability Metric

The appropriate statistical test is chi-squared goodness-of-fit:

$$\chi^2 = \sum_{i=1}^{M} \frac{(d_i - \bar{d}_{weighted})^2}{\sigma_i^2}$$

where:
- $d_i$ = measured depth for transit $i$
- $\sigma_i$ = uncertainty on $d_i$ (including red noise inflation)
- $\bar{d}_{weighted}$ = inverse-variance weighted mean depth

**Interpretation:**
- $\chi^2_{red} = \chi^2 / (M-1) \approx 1$: Scatter consistent with measurement noise (planet candidate)
- $\chi^2_{red} \gg 1$: Excess scatter beyond noise (possible EB or systematic)

---

## 3. Proposed Improvements

### 3.1 Local Baseline per Epoch

Replace global baseline with per-epoch local baseline:

```
For each epoch k:
    epoch_center = t0 + k * period
    local_window = [epoch_center - 6*duration, epoch_center + 6*duration]
    local_oot = points in local_window AND NOT in_transit
    baseline_k = median(local_oot_flux)
    depth_k = 1 - median(in_transit_flux) / baseline_k
```

**Rationale:** Same approach used in V01 (odd/even) which already implements local baselines. Maintains consistency across checks and handles baseline drift.

### 3.2 Per-Epoch Uncertainty Estimation

Compute expected depth uncertainty for each epoch:

```
sigma_depth_k = robust_std(local_oot_flux) / sqrt(n_in_transit_k) / baseline_k
```

Optionally apply red noise inflation (as in V01):
```
inflation = max(1.0, observed_binned_scatter / expected_white_noise_scatter)
sigma_depth_k *= inflation
```

**Rationale:** Enables comparison of observed scatter to expected scatter, not just an arbitrary threshold.

### 3.3 Stability Metric: Chi-Squared Ratio

Define a proper stability statistic:

```
depth_mean = weighted_mean(depths, weights=1/sigma^2)
chi2_observed = sum((depth_k - depth_mean)^2 / sigma_k^2)
dof = n_transits - 1
chi2_reduced = chi2_observed / dof
```

**Interpretation:**
- `chi2_reduced ~ 1`: Scatter consistent with measurement noise (stable)
- `chi2_reduced >> 1`: Excess scatter beyond noise (variable/suspicious)
- `chi2_reduced << 1`: Suspiciously low scatter (possible overfitting or correlated errors)

**Decision rule:**
- PASS if `chi2_reduced < 2.0` (excess scatter < 2x expected)
- WARN if `2.0 <= chi2_reduced < 4.0`
- FAIL if `chi2_reduced >= 4.0`

### 3.4 Robust Aggregation with Outlier Flagging

Use robust statistics to handle outlier epochs:

1. Compute initial median depth and MAD-based scatter
2. Flag epochs where `|depth_k - median| > 4 * MAD`
3. Report flagged epochs in warnings but do not automatically exclude
4. Recompute statistics with and without outliers; report both

**Rationale:** Automatic exclusion can hide real variability (which is the point of this check), but flagging helps users identify problematic epochs.

### 3.5 Confidence Model

Confidence should reflect statistical power to detect depth variations:

```python
def compute_v04_confidence(n_transits: int, chi2_reduced: float, has_warnings: bool) -> float:
    # Base confidence from transit count
    if n_transits <= 2:
        base = 0.2  # Cannot reliably detect variability
    elif n_transits <= 4:
        base = 0.4  # Marginal detection power
    elif n_transits <= 7:
        base = 0.6
    elif n_transits <= 12:
        base = 0.75
    else:
        base = 0.85

    # Adjust for proximity to threshold
    if 1.5 < chi2_reduced < 2.5:
        base *= 0.85  # Near threshold

    # Degrade for warnings
    if has_warnings:
        base *= 0.9

    return min(0.95, base)
```

---

## 4. Recommended Defaults

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `min_transits_for_confidence` | 3 | Need >= 3 for meaningful scatter |
| `min_points_per_epoch` | 5 | Match V01 default |
| `baseline_window_mult` | 6.0 | 6x duration for local baseline |
| `chi2_threshold_pass` | 2.0 | Standard chi2 excess threshold |
| `chi2_threshold_fail` | 4.0 | Strong evidence of variability |
| `outlier_sigma` | 4.0 | MAD-based outlier flagging |
| `use_red_noise_inflation` | True | Conservative uncertainty estimation |
| `rms_scatter_threshold` | 0.3 | Legacy threshold (backward compat) |

### Configuration Dataclass

```python
@dataclass
class DepthStabilityConfig:
    min_transits_for_confidence: int = 3
    min_points_per_epoch: int = 5
    baseline_window_mult: float = 6.0
    chi2_threshold_pass: float = 2.0
    chi2_threshold_fail: float = 4.0
    outlier_sigma: float = 4.0
    use_red_noise_inflation: bool = True
    # Legacy mode uses rms_scatter < 0.3 for backward compatibility
    legacy_mode: bool = False
    rms_scatter_threshold: float = 0.3
```

---

## 5. Required Output Fields

All new fields are **additive**. Existing fields (`mean_depth`, `std_depth`, `rms_scatter`, `n_transits_measured`, `individual_depths`) are preserved for backward compatibility.

### New Fields

| Field | Type | Description |
|-------|------|-------------|
| `n_transits_measured` | int | Number of epochs with sufficient data (exists, clarified) |
| `depths_ppm` | list[float] | Per-epoch depths in ppm (first 20 epochs) |
| `depth_scatter_ppm` | float | Observed scatter in ppm |
| `expected_scatter_ppm` | float | Expected scatter from noise model |
| `rms_scatter` | float | Coefficient of variation (exists, definition clarified) |
| `chi2_reduced` | float | Chi-squared per DOF for depth consistency |
| `chi2_dof` | int | Degrees of freedom (n_transits - 1) |
| `depth_mean_ppm` | float | Weighted mean depth in ppm |
| `depth_err_mean_ppm` | float | Uncertainty on mean depth |
| `outlier_epochs` | list[int] | Epoch indices flagged as outliers |
| `n_outliers_flagged` | int | Number of flagged outlier epochs |
| `warnings` | list[str] | Warning messages (low_n_transits, outlier_epochs_flagged, etc.) |
| `method` | str | Algorithm variant used ("chi2_local_baseline" or "legacy_rms") |

### Example Output

```json
{
  "id": "V04",
  "name": "depth_stability",
  "passed": true,
  "confidence": 0.72,
  "details": {
    "mean_depth": 0.00234,
    "std_depth": 0.00031,
    "rms_scatter": 0.132,
    "n_transits_measured": 8,
    "individual_depths": [0.00241, 0.00228, ...],
    "depths_ppm": [2410, 2280, 2350, 2290, 2380, 2310, 2420, 2260],
    "depth_scatter_ppm": 58.2,
    "expected_scatter_ppm": 45.1,
    "depth_mean_ppm": 2337.5,
    "depth_err_mean_ppm": 16.0,
    "chi2_reduced": 1.67,
    "chi2_dof": 7,
    "outlier_epochs": [],
    "n_outliers_flagged": 0,
    "warnings": [],
    "method": "chi2_local_baseline"
  }
}
```

---

## 6. Test Matrix

### 6.1 Synthetic Test Cases

| Test Case | N_transits | Depth (ppm) | Injected Scatter | Expected Result |
|-----------|------------|-------------|------------------|-----------------|
| **stable_high_snr** | 10 | 5000 | 0% | PASS, conf >= 0.8 |
| **stable_low_snr** | 10 | 500 | noise-only | PASS, conf >= 0.7 |
| **variable_eb_like** | 8 | 5000 | +/- 30% alternating | FAIL, chi2 >> 4 |
| **marginal_variability** | 6 | 3000 | +/- 15% random | WARN region, conf ~ 0.5 |
| **low_n_stable** | 2 | 2000 | 0% | PASS, conf = 0.2, warning |
| **low_n_variable** | 3 | 2000 | +/- 40% | FAIL or low-conf pass with warning |
| **single_outlier** | 10 | 3000 | 1 epoch at 2x depth | PASS with outlier flag |
| **baseline_drift** | 8 | 2000 | 10% linear trend | PASS (local baseline corrects) |
| **active_star** | 12 | 1500 | 5% stellar + noise | PASS with red noise inflation |

### 6.2 Expected Outcomes Detail

**stable_high_snr:**
- Input: 10 transits, depth = 5000 ppm, no injected variation
- Expected: chi2_reduced ~ 1.0, rms_scatter < 0.1, PASS
- Validates: Basic functionality with clean signal

**variable_eb_like:**
- Input: 8 transits with depths alternating [3500, 6500] ppm
- Expected: chi2_reduced > 10, rms_scatter > 0.35, FAIL
- Validates: Detection of EB-like odd/even pattern (complementary to V01)

**low_n_stable:**
- Input: 2 transits, both at 2000 ppm
- Expected: PASS (cannot reject), confidence = 0.2, warning "low_n_transits"
- Validates: Appropriate confidence degradation

**single_outlier:**
- Input: 10 transits, 9 at 3000 ppm, 1 at 6000 ppm
- Expected: PASS, outlier_epochs = [outlier_idx], warning "outlier_epochs_flagged"
- Validates: Robust handling without auto-exclusion

---

## 7. Backward Compatibility

### 7.1 Preserved (No Change)

- Check ID: `V04`
- Check name: `depth_stability`
- Legacy fields: `mean_depth`, `std_depth`, `rms_scatter`, `n_transits_measured`, `individual_depths`
- Return type: `VetterCheckResult`

### 7.2 Additive (New Fields)

All new fields listed in Section 5 are additive. Consumers ignoring unknown fields will continue to work.

### 7.3 Gated Behavior

- **Default behavior:** New chi2-based algorithm with local baselines
- **Legacy mode:** Controlled by `DepthStabilityConfig(legacy_mode=True)`, uses original global baseline and 30% RMS threshold
- **Method field:** `details["method"]` indicates which algorithm was used

### 7.4 Pass/Fail Semantics

The new algorithm may produce different pass/fail outcomes than the legacy version:
- Cases where noise explains scatter: More likely to PASS (correctly)
- Cases with true variability hidden in noise: Better detection (may FAIL where legacy passed)

For backward compatibility in downstream systems, recommend:
1. Initially deploy with `legacy_mode=True` for existing pipelines
2. Validate new algorithm on test set before switching
3. Monitor `chi2_reduced` and `rms_scatter` correlation during transition

---

## 8. Citations

### Primary References (Literature Research Findings)

| arXiv ID | Citation | Relevance |
|----------|----------|-----------|
| **arXiv:2311.02154** | Wang & Espinoza (2023), "A Blind Search for Transit Depth Variability with TESS" | **Core methodology paper.** Per-transit depth fitting using juliet + batman models, GP-based stellar variability removal. Found ~1% detection rate of significant depth variability in 330 TESS planets. Key insight: local baseline per epoch is essential for precision depth measurements. |
| **arXiv:1507.08285** | Kreidberg (2015), "batman: BAsic Transit Model cAlculatioN in Python" | Standard transit model package. Supports quadratic limb darkening for per-transit box/trapezoid fitting. Can compute 1M models in 30 seconds with single core. |
| **arXiv:2002.00691** | Giacalone & Dressing (2020), "TRICERATOPS: Vetting and Validation of TESS Objects of Interest" | Bayesian false positive framework. Models transit depth as function of flux from target vs contaminant stars. FPP < 0.015 threshold for validation. Shows importance of depth consistency in distinguishing planets from EBs. |
| **arXiv:1307.2003** | Santerne et al. (2013), "Secondary eclipses as astrophysical false positives" | Diluted EB detection methodology. Depth variations >10% relative are strong EB indicators. Provides theoretical basis for depth variability thresholds. |

### Supporting References

| arXiv ID | Citation | Relevance |
|----------|----------|-----------|
| **arXiv:2506.05631** | Kostov et al. (2025), "TESS Ten Thousand Catalog: 10,001 uniformly-vetted Eclipsing Binary Stars" | Large-scale EB vetting pipeline using ML + citizen science. Depth stability among features for EB identification. Validates detection of EBs via depth inconsistency. |
| **arXiv:2202.03401** | Ivshina & Winn (2022), "TESS Transit Timing of Hundreds of Hot Jupiters" | Per-transit ephemeris measurement methodology. Database of 8,667 transit times across 382 systems. Shows expected scatter levels for well-characterized planets. |
| **arXiv:1612.02432** | Dalba et al. (2016), "Kepler Transit Depths Contaminated by a Phantom Star" | Case study: Kepler-445c depth was 50% shallower in ground-based obs due to flux contamination. Demonstrates importance of depth stability checks for blend detection. |
| **arXiv:2203.15826** | Cacciapuoti et al. (2022), "TESS Triple-9 Catalog: 999 uniformly vetted exoplanet candidates" | Uniform TESS vetting methodology. Includes depth stability as part of candidate assessment. |
| **arXiv:1901.07459** | Kostov et al. (2019), "Discovery and Vetting of Exoplanets I: Benchmarking K2 Vetting Tools" | K2 vetting benchmarks. Marks 676/772 as planet candidates based on vetting including depth consistency. 60 newly identified false positives detected via vetting metrics. |

### Classic References (Methodology Foundation)

| Bibcode | Citation | Relevance |
|---------|----------|-----------|
| 2018ApJS..235...38T | Thompson et al. 2018, ApJS 235, 38 | Kepler DR25 Robovetter; individual transit metrics and depth consistency tests. |
| 2006MNRAS.373..231P | Pont et al. 2006, MNRAS 373, 231 | Red/correlated noise in transit photometry; inflation factors. Foundation for uncertainty estimation. |

### Implementation Notes

Per the v1_spec.md requirements, citations should be added to code via:
1. Module-level docstring with bibcodes
2. `@cites()` decorator on the API wrapper function (already present in `lc_only.py`)
3. Inline comments for specific algorithmic choices

### Key Methodological Insights from Literature

From **Wang & Espinoza (2023)** - the most directly relevant paper:
- Uses juliet with batman for per-transit fitting
- Out-of-transit Quasi-Periodic GP kernel to model stellar variability
- Defines out-of-transit as >1 transit duration from nearest transit center
- Uniform priors: depth [0,1], impact parameter [0,1], limb darkening coefficients [0,1]
- Chi-squared goodness of fit for depth consistency assessment
- Lomb-Scargle periodogram on depth time series to detect periodic variations
- False alarm probability threshold: 0.1% (99.9th percentile of bootstrap distribution)

From **Santerne et al. (2013)**:
- Diluted EBs produce systematic depth variations
- Detection threshold: >10% relative depth variation indicates non-planetary origin
- Eccentric orbit EBs can show phase-dependent depth variations

---

## Appendix A: Algorithm Pseudocode

```python
def check_depth_stability_v2(lc, period, t0, duration_hours, config):
    """V04 with chi2-based stability metric and local baselines."""

    duration_days = duration_hours / 24.0
    warnings = []

    # Phase and epoch calculation
    phase = ((time - t0) / period) % 1
    phase_dist = min(phase, 1 - phase)
    in_transit = phase_dist < 0.5 * duration_days / period
    epoch = floor((time - t0) / period)

    # Per-epoch depth extraction
    epoch_data = {}
    for ep in unique(epoch):
        # Get local OOT baseline
        ep_center = t0 + ep * period
        local_window = |time - ep_center| < config.baseline_window_mult * duration_days
        local_oot = local_window & ~in_transit

        if count(local_oot) < 10:
            continue  # Skip epoch

        baseline = median(flux[local_oot])
        in_flux = flux[epoch == ep & in_transit]

        if count(in_flux) < config.min_points_per_epoch:
            continue

        depth = 1 - median(in_flux) / baseline
        sigma = robust_std(flux[local_oot]) / sqrt(count(in_flux)) / baseline

        # Optional red noise inflation
        if config.use_red_noise_inflation:
            sigma *= compute_red_noise_factor(flux[local_oot], time[local_oot])

        epoch_data[ep] = (depth, sigma)

    n_transits = len(epoch_data)

    if n_transits < 2:
        return low_confidence_pass("Insufficient transits")

    # Compute chi2 stability metric
    depths = [d for d, s in epoch_data.values()]
    sigmas = [s for d, s in epoch_data.values()]

    weights = 1 / array(sigmas)**2
    depth_mean = sum(depths * weights) / sum(weights)

    chi2 = sum(((depths - depth_mean) / sigmas)**2)
    dof = n_transits - 1
    chi2_reduced = chi2 / dof

    # Outlier detection
    mad = median(|depths - median(depths)|)
    outliers = |depths - median(depths)| > config.outlier_sigma * mad * 1.4826

    # Decision
    if chi2_reduced < config.chi2_threshold_pass:
        passed = True
    elif chi2_reduced < config.chi2_threshold_fail:
        passed = True  # Marginal - pass with reduced confidence
        warnings.append("marginal_chi2")
    else:
        passed = False

    # Confidence
    confidence = compute_v04_confidence(n_transits, chi2_reduced, len(warnings) > 0)

    return VetterCheckResult(
        id="V04", name="depth_stability",
        passed=passed, confidence=confidence,
        details={...}
    )
```

---

## Appendix B: Migration Path

1. **Phase 1 (Immediate):** Add new output fields alongside existing ones
2. **Phase 2 (Testing):** Run both algorithms in parallel on TOI test set; compare outcomes
3. **Phase 3 (Default switch):** Change default to new algorithm; preserve legacy_mode
4. **Phase 4 (Deprecation):** Document legacy_mode as deprecated after 2 release cycles
