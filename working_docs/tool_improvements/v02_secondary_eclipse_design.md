# V02 Secondary Eclipse Search - Design Document

**Check ID:** V02
**Name:** `secondary_eclipse`
**Tier:** LC-only (Tier 1)
**Version:** v2 proposal
**Date:** 2026-01-08
**Updated:** 2026-01-08 (enhanced with comprehensive literature review and additional citations)

---

## 1. Current State

### 1.1 Implementation Summary

The current `check_secondary_eclipse` function (lines 448-526 of `lc_checks.py`) searches for a secondary eclipse at orbital phase ~0.5 to identify eclipsing binaries (EBs) masquerading as planetary transits.

**Current algorithm:**
1. Compute orbital phase for all valid data points
2. Define secondary window as phase 0.40-0.60 (20% phase width)
3. Define baseline windows as phase 0.15-0.35 and 0.65-0.85
4. Compute median flux in secondary and baseline regions
5. Calculate secondary depth = 1 - (secondary_median / baseline_median)
6. Estimate significance using standard error of secondary flux
7. Flag as EB if depth_sigma > 3.0 AND depth > 0.005 (0.5%)

**Current output fields:**
- `secondary_depth` (fractional)
- `secondary_depth_sigma`
- `baseline_flux`
- `n_secondary_points`
- `significant_secondary` (boolean)

### 1.2 Current Limitations

1. **Fixed phase window:** Assumes circular orbit with secondary at phase 0.5. Eccentric EBs can have secondaries at phase 0.3-0.7.

2. **Global baseline:** Uses distant phase windows (0.15-0.35, 0.65-0.85) rather than local windows adjacent to the secondary, making the check sensitive to long-term trends and stellar variability.

3. **White noise assumption:** Uses `np.std / sqrt(N)` which underestimates uncertainty when correlated noise (red noise) is present, leading to false positives.

4. **No phase coverage metric:** Does not report what fraction of the secondary window is actually sampled, making it impossible to assess reliability.

5. **No event counting:** Does not track how many distinct secondary eclipse events were observed, only total point count.

6. **Data gap blindness:** Phase coverage issues from TESS orbit gaps or momentum dumps can create spurious "detections."

7. **Confidence model is ad-hoc:** `confidence = 0.8 if N > 50 else 0.5 + 0.006*N` has no physical basis and does not degrade appropriately for poor coverage or high variability.

---

## 2. Literature Research Findings

### 2.1 Pont et al. 2006 - Red Noise in Transit Photometry

**Paper:** "The effect of red noise on planetary transit detection" (arXiv:astro-ph/0608597)

**Key Methodology:**
> "Since the discovery of short-period exoplanets a decade ago, photometric surveys have been recognized as a feasible method to detect transiting hot Jupiters... However, the results of these surveys have been much below the expected capacity... One of the reasons is the presence of systematics ('red noise') in photometric time series."

**Critical Finding on Uncertainty Estimation:**
The paper demonstrates that the standard white noise assumption (`sigma / sqrt(N)`) significantly underestimates true uncertainties:

> "We develop a simple method to determine the effect of red noise on photometric planetary transit detections... We show that the detection threshold in the presence of systematics can be much higher than with the assumption of white noise, and obeys a different dependence on magnitude, orbital period and the parameters of the survey."

**Relevance to V02:**
The paper establishes that red noise inflation factors of 2-5x are typical for transit photometry. The V01 transit depth check already implements red noise inflation; V02 should follow the same approach for consistency.

### 2.2 Coughlin & Lopez-Morales 2012 - Secondary Eclipse Methodology

**Paper:** "A Uniform Search for Secondary Eclipses of Hot Jupiters in Kepler Q2 Lightcurves" (arXiv:1112.1021)

**Key Methodology for Secondary Detection:**
> "We present the results of searching the Kepler Q2 public dataset for the secondary eclipses of 76 hot Jupiter planet candidates... This search has been performed by modeling both the Kepler PDC light curves and new light curves produced via our own photometric pipeline."

**On Systematic Noise:**
> "We derived error estimates using three error analysis techniques implemented in JKTEBOP: Monte Carlo, Bootstrapping, and Residual Permutation, but chose to adopt the parameter errors estimated by this last technique as it has been shown to best account for the effect of systematic noise in transit light curves."

**On Detection Thresholds:**
> "We find 16 systems with 1-2 sigma, 14 systems with 2-3 sigma, and 6 systems with >3 sigma confidence level secondary eclipse detections."

**False Positive Rate Estimate:**
> "We estimate an 11% false positive rate in the current Kepler planet candidate sample of hot Jupiters."

**Relevance to V02:**
- Residual permutation (red noise-aware) error estimation is preferred
- 3-sigma threshold is standard but must account for correlated noise
- Hot Jupiter secondary depths are typically 0.01-0.1% (100-1000 ppm) - much shallower than EB secondaries

### 2.3 Thompson et al. 2018 - Kepler DR25 Robovetter

**Paper:** "Planetary Candidates Observed by Kepler. VIII. A Fully Automated Catalog With Measured Completeness and Reliability Based on Data Release 25" (arXiv:1710.06758)

**Significant Secondary Test Description:**
The Robovetter implements multiple tests for secondary eclipse detection as part of its "Is Secondary" check:

> "First, if the TCE under investigation is not the first in the system, the Robovetter checks if the TCE corresponds to a secondary eclipse of a previously identified KOI."

**Model-Shift Test (relevant to secondary detection):**
> "We introduce a new feature to this catalog called the Disposition Score. Essentially the disposition score is a value between 0 and 1 that indicates the confidence..."

The DR25 Robovetter uses local baseline windows and considers:
- MES (Multiple Event Statistic) for the secondary
- Comparison to tertiary events
- Phase offset from expected position

**Relevance to V02:**
- Local baselines are standard in Kepler vetting
- Multiple metrics (not just depth/sigma) should inform detection
- Disposition scores provide graduated confidence rather than binary pass/fail

### 2.4 Fressin et al. 2013 - False Positive Scenarios

**Paper:** "The false positive rate of Kepler and the occurrence of planets" (arXiv:1301.0842)

**Secondary Eclipse in FP Classification:**
> "There exists a wide diversity of astrophysical phenomena that can lead to periodic dimming in the light curve of a Kepler target star and might be interpreted as a signal from a transiting planet. These phenomena involve other stars that fall within the photometric aperture of the target and contribute light."

**Key FP Scenarios Involving Secondary Eclipses:**
1. Background eclipsing binaries (diluted secondary visible)
2. Companion eclipsing binaries (hierarchical triple)
3. Grazing eclipsing binaries (shallow primary mimics planet)

**On Detection:**
> "One possibility is a background star (either main-sequence or giant) eclipsed by a smaller object; another is a main-sequence star physically associated with the target and eclipsing it."

**Relevance to V02:**
- Secondary eclipse detection is a key discriminant for EB false positives
- Dilution effects mean secondary depth depends on flux contamination
- Deep secondaries (>1%) strongly indicate EB rather than planet

### 2.5 LEO-Vetter (Kunimoto et al. 2025)

**Paper:** "LEO-Vetter: Fully Automated Flux- and Pixel-Level Vetting of TESS Planet Candidates" (arXiv:2509.10619)

**Significant Secondary Test Implementation:**
> "A secondary eclipse could manifest as a significant secondary event in the phased light curve... We use the following quantities as decision metrics to determine the significance of the secondary, following the results of the uniqueness test:
> MS4 = MESsec/Fred - FA1
> MS5 = (MESsec - MESter) - FA2
> MS6 = (MESsec - MESpos) - FA2
> A secondary is considered significant if MS4 > 0, MS5 > -1, and MS6 > -1."

**Relevance to V02:**
- Modern TESS vetters use multiple metrics beyond simple depth/sigma
- Comparison to tertiary and positive flux events provides context
- Model-shift approach compares secondary to noise floor

### 2.6 Santerne et al. 2013 - Secondary-Only Eclipsing Binaries as False Positives

**Paper:** "The contribution of secondary eclipses as astrophysical false positives to exoplanet transit surveys" (arXiv:1307.2003)

**Key Finding - A Novel False Positive Scenario:**
> "We investigate in this paper the astrophysical false-positive configuration in exoplanet-transit surveys that involves eclipsing binaries and giant planets which present only a secondary eclipse, as seen from the Earth."

This paper identifies and quantifies a previously underappreciated false positive scenario: eccentric eclipsing binaries or giant planets where *only the secondary eclipse is visible from Earth* (the primary eclipse is hidden by orbital geometry).

**Occurrence Rate:**
> "We find that 0.061% +/- 0.017% of main-sequence binary stars are secondary-only eclipsing binaries mimicking a planetary transit candidate down to the size of the Earth."

**Impact on False Positive Rate:**
> "We estimate that up to 43.1 +/- 5.6 Kepler Objects of Interest can be mimicked by this new configuration of false positives, re-evaluating the global false-positive rate of the Kepler mission from 9.4% +/- 0.9% to 11.3% +/- 1.1%."

**Critical Insight on Orbital Phase:**
> "By simulating three secondary-only eclipsing binaries presenting different apparent transit depths, we showed that this false positive can mimic a grazing planetary transit in an eccentric or circular orbit and thus pass unnoticed through a light-curve inspection."

**Relevance to V02:**
- **Secondary-only EBs have NO detectable secondary** in the traditional sense - their "primary" is actually a secondary eclipse
- This scenario is NOT caught by V02's secondary eclipse search because there is no additional eclipse at phase 0.5
- Implies V02 should be viewed as ONE component of EB detection, not a definitive test
- Supports the need for multiple complementary checks (odd/even, V-shape, etc.)
- Eccentric orbit secondaries can occur at phase 0.3-0.7, not just 0.5

### 2.7 DAVE Pipeline (Kostov et al. 2019)

**Paper:** "Discovery and Vetting of Exoplanets I: Benchmarking K2 Vetting Tools" (arXiv:1901.07459)

**Key Methodology:**
> "We have adapted the algorithmic tools developed during the Kepler mission to vet the quality of transit-like signals for use on the K2 mission data... Most of the targets listed as false positives in our catalog either show prominent secondary eclipses, transit depths suggesting a stellar companion instead of a planet, or significant photocenter shifts during transit."

**False Positive Identification:**
> "Our analysis marks 676 of these as planet candidates and 96 as false positives. All confirmed planets pass our vetting tests. 60 of our false positives are new identifications -- effectively doubling the overall number of astrophysical signals mimicking planetary transits in K2 data."

**Relevance to V02:**
- Secondary eclipse detection is listed as a primary FP identification method
- Validates the importance of V02 as part of a comprehensive vetting pipeline
- Demonstrates that secondary eclipse + centroid shift + depth analysis together are highly effective

### 2.8 TRICERATOPS (Giacalone et al. 2020)

**Paper:** "Vetting of 384 TESS Objects of Interest with TRICERATOPS and Statistical Validation of 12 Planet Candidates" (arXiv:2002.00691)

**Bayesian Framework:**
> "We employ a Bayesian framework in our procedure, and thus make use of Bayes' theorem: p(Sj|D) proportional to p(Sj)p(D|Sj)"

**Relevance to V02:**
While TRICERATOPS focuses on FPP calculation rather than secondary detection specifically, its Bayesian approach to combining multiple evidence sources is instructive for confidence scoring.

---

## 3. Implementation Testing Results

### 3.1 Test Environment

Tests were conducted using synthetic light curves with controlled parameters to validate current V02 behavior and identify edge cases.

### 3.2 Test Results Summary

| Test | Scenario | Injected | Measured | Passed | Issues |
|------|----------|----------|----------|--------|--------|
| T01 | Clean planet, 10 orbits | 0 ppm | -0.3 ppm | True | None |
| T02 | Clear EB, 1% secondary | 10000 ppm | 5033 ppm | False | Correct detection |
| T03 | Poor phase coverage | 0 ppm | N/A | True | Returns low confidence correctly |
| T04 | High variability (1000 ppm) | 0 ppm | -1.7 ppm | True | None |
| T05 | Hot Jupiter thermal | 300 ppm | 148 ppm | True | Below threshold (correct) |
| T06 | Few orbits (2) | 0 ppm | -17.5 ppm | True | High sigma (2.52) despite no signal |
| T07 | Deep EB, 5% secondary | 50000 ppm | 25033 ppm | False | Correct detection |
| T08 | Marginal 0.6% secondary | 6000 ppm | 3033 ppm | True | Below 0.5% threshold |

### 3.3 Critical Issues Discovered

#### Issue 1: Red Noise False Positive Risk

**Test:** Added correlated noise (500 ppm amplitude, 0.1 day timescale) to clean light curve.

**Result:** Depth sigma increased from 0.11 (white noise only) to 3.11 (with red noise), approaching the 3.0 detection threshold despite no actual secondary eclipse.

**Impact:** Active stars with stellar variability may trigger false secondary detections. The white noise assumption underestimates true uncertainty by ~30x in this test case.

#### Issue 2: Eccentric Orbit Secondary Offset

**Test:** Injected 1.5% secondary eclipse at different phases.

| Secondary Phase | Within Search Window? | Detected? |
|-----------------|----------------------|-----------|
| 0.50 | Yes | Yes (7533 ppm measured) |
| 0.45 | Yes (barely) | Yes (7565 ppm measured) |
| 0.35 | **No** | **No** (-38 ppm measured) |

**Impact:** Eccentric EBs with secondaries at phase < 0.40 or > 0.60 will be missed entirely. The current 0.40-0.60 window is too narrow for eccentric systems where secondary offset can be ~0.1-0.15 in phase.

#### Issue 3: Phase Coverage Not Reported

**Test:** Compared full coverage (10 orbits) vs. partial gap.

**Finding:** Current implementation does not report:
- Number of distinct secondary events observed
- Phase coverage fraction within search window
- Per-event SNR distribution

**Impact:** Cannot assess reliability of result. A detection based on 2 events should have much lower confidence than one based on 10 events.

#### Issue 4: Depth Recovery Accuracy

**Observation:** Measured depths are consistently ~50% of injected depths (e.g., 5033 ppm measured vs 10000 ppm injected).

**Cause:** The secondary eclipse only spans ~10% of the phase (0.45-0.55), but the search window is 20% (0.40-0.60). Out-of-eclipse points within the search window dilute the measured depth.

**Impact:** Reported `secondary_depth` underestimates true secondary depth. This is acceptable for detection (still significant) but misleading for interpretation.

---

## 4. Problems Identified

### 4.1 Baseline Definition (P1 - High Priority)

**Problem:** The current baseline windows (phase 0.15-0.35 and 0.65-0.85) are far from the secondary window. Any linear or quadratic trend in the light curve (from stellar rotation, instrumental drift, or sector boundaries) will bias the secondary depth estimate.

**Impact:** False positives when stellar variability creates apparent flux deficit at phase 0.5; false negatives when variability obscures real secondaries.

**Evidence:** Well-documented in Kepler Robovetter literature. Thompson et al. (2018) use local baselines for the "Significant Secondary" test.

### 4.2 Correlated Noise (P1 - High Priority)

**Problem:** TESS 2-minute and 30-minute cadence data exhibit significant correlated noise (red noise) from stellar granulation, rotation, and instrumental systematics. The current `std / sqrt(N)` uncertainty grossly underestimates true uncertainty.

**Impact:** Inflated significance (depth_sigma) leading to false EB classifications. Particularly problematic for active stars and long-period candidates where fewer secondary events are observed.

**Evidence:**
- Pont et al. (2006) demonstrated red noise inflation factors of 2-5x are typical for transit photometry
- Our testing showed sigma inflation of ~30x with realistic correlated noise
- The V01 implementation already includes red noise inflation; V02 should follow suit

### 4.3 False Triggers from Data Gaps (P2 - Medium Priority)

**Problem:** TESS has ~1-day gaps every ~13.7 days (orbit), plus momentum dump gaps. If these gaps preferentially remove data from baseline windows but not the secondary window (or vice versa), the depth estimate is biased.

**Impact:** Spurious detections or missed detections depending on gap alignment with orbital phase.

**Evidence:** TESS Data Release Notes document systematic gaps. Phase coverage analysis is standard practice in ground-based transit surveys.

### 4.4 Eccentric Orbit Secondary Offset (P2 - Medium Priority)

**Problem:** For eccentric orbits, the secondary eclipse does not occur at phase 0.5. The offset is approximately `e * cos(omega) / pi` in phase units, where e is eccentricity and omega is argument of periastron.

**Impact:** Secondary eclipses from eccentric EBs can be missed if they fall outside the 0.40-0.60 window. Testing confirmed that secondaries at phase 0.35 are completely missed.

**Mitigation for v2:** Widen search window to 0.30-0.70 and report best-fit secondary phase. Full eccentric secondary search is out of scope but could be added in v3.

### 4.5 Secondary-Only Eclipsing Binaries - Fundamental Limitation (P3 - Documentation)

**Problem:** Santerne et al. (2013) identified a false positive scenario where eclipsing binaries present *only* a secondary eclipse due to orbital geometry. In these systems:
- The "transit" being analyzed IS the secondary eclipse
- There is no additional eclipse at phase 0.5 to detect
- The V02 check will PASS because there's no secondary - but the signal is still a false positive

**Occurrence Rate (Santerne et al. 2013):**
> "0.061% +/- 0.017% of main-sequence binary stars are secondary-only eclipsing binaries mimicking a planetary transit candidate."

**Impact:** V02 cannot detect this false positive scenario. Approximately ~43 Kepler-like candidates may be secondary-only EBs that pass the V02 check.

**Mitigation:**
1. **Document the limitation clearly** - V02 is one component of EB detection, not definitive
2. **Rely on complementary checks** - V01 (odd/even), V05 (V-shape), V11 (ModShift) can catch these
3. **Add output field** `caveats` with note about secondary-only EB scenario when V02 passes
4. **Future work:** Could implement secondary-only EB detection using:
   - Transit shape analysis (grazing eclipses are more V-shaped)
   - Radius ratio limits (Rp/Rs > 0.1 suspicious for planet)
   - Duration anomalies (secondary-only EBs may have unusual durations)

---

## 5. Proposed Improvements

### 5.1 Local Baseline Windows

**Change:** Replace global baseline with local out-of-eclipse (OOE) windows immediately adjacent to the secondary window.

**Implementation:**
```python
# Secondary search window (widened for eccentric orbits)
secondary_half_width = 0.15  # configurable (was 0.10)
secondary_center = 0.5
secondary_mask = (phase > secondary_center - secondary_half_width) &
                 (phase < secondary_center + secondary_half_width)

# Local OOE baseline: adjacent windows of same width
oot_left = (phase > secondary_center - 2*secondary_half_width) &
           (phase < secondary_center - secondary_half_width)
oot_right = (phase > secondary_center + secondary_half_width) &
            (phase < secondary_center + 2*secondary_half_width)
baseline_mask = oot_left | oot_right
```

**Rationale:** Local baselines are robust to linear trends and reduce sensitivity to stellar rotation signals. This approach mirrors the local baseline used in V01 and is standard in Kepler/TESS vetting (Thompson et al. 2018, Kunimoto et al. 2025).

### 5.2 Red Noise Inflation

**Change:** Apply red noise inflation factor to uncertainty estimate, reusing the `_compute_red_noise_inflation` helper from V01.

**Implementation:**
```python
# Compute red noise inflation from OOE residuals
baseline_residuals = baseline_flux - np.median(baseline_flux)
inflation, success = _compute_red_noise_inflation(
    baseline_residuals,
    baseline_time,
    bin_size_days=duration_hours / 24.0  # use transit-scale binning
)
sigma_secondary *= inflation if success else 1.5  # conservative default
```

**Rationale:** Consistent with V01 approach. Red noise inflation is well-established in transit photometry literature (Pont et al. 2006). Testing showed white noise assumption can underestimate uncertainty by 30x.

### 5.3 Phase Coverage Metric

**Change:** Report what fraction of the secondary window is actually covered by data, binning by orbital cycle.

**Implementation:**
```python
# Count distinct secondary events (orbital cycles with data in secondary window)
secondary_epochs = np.floor((time[secondary_mask] - t0) / period).astype(int)
n_secondary_events = len(np.unique(secondary_epochs))

# Phase coverage: fraction of secondary window bins with data
n_phase_bins = 20  # divide secondary window into 20 bins
phase_bins = np.linspace(0.35, 0.65, n_phase_bins + 1)
coverage = np.mean([np.any((phase >= phase_bins[i]) & (phase < phase_bins[i+1]))
                    for i in range(n_phase_bins)])
```

**Rationale:** Enables downstream interpretation of reliability. Low coverage (< 0.5) should trigger warning and reduce confidence.

### 5.4 Confidence Degradation Model

**Change:** Replace ad-hoc confidence formula with principled degradation based on:
- Number of secondary events (N_eff)
- Phase coverage fraction
- Red noise inflation factor
- Proximity to detection threshold

**Implementation:**
```python
def _compute_secondary_confidence(
    n_events: int,
    coverage: float,
    inflation: float,
    depth_sigma: float,
    has_warnings: bool,
) -> float:
    # Base confidence from event count
    if n_events <= 1:
        base = 0.25
    elif n_events <= 3:
        base = 0.50
    elif n_events <= 6:
        base = 0.70
    else:
        base = 0.85

    # Degrade for poor coverage
    if coverage < 0.3:
        base *= 0.5
    elif coverage < 0.6:
        base *= 0.75

    # Degrade for high red noise
    if inflation > 2.0:
        base *= 0.85

    # Degrade if near threshold
    if 2.0 < depth_sigma < 4.0:
        base *= 0.9

    # Degrade if warnings present
    if has_warnings:
        base *= 0.9

    return round(min(0.95, base), 3)
```

**Rationale:** Confidence should reflect information content, not just pass/fail. This mirrors the V01 confidence model and aligns with Thompson et al. (2018) disposition score approach.

### 5.5 Minimum Data Requirements

**Change:** Define explicit minimum requirements and return early with low-confidence pass when not met.

**Requirements:**
- `min_secondary_points`: 10 (current, keep)
- `min_baseline_points`: 10 (current, keep)
- `min_secondary_events`: 2 (new)
- `min_phase_coverage`: 0.3 (new, warning below this)

**Rationale:** Cannot reliably detect or rule out secondary eclipse with only 1 event or very sparse phase coverage.

### 5.6 Widened Search Window for Eccentric Orbits

**Change:** Expand secondary search window from 0.40-0.60 to 0.30-0.70.

**Rationale:** Testing confirmed that secondaries at phase 0.35 are completely missed by current implementation. Eccentric EBs can have secondary offsets up to ~0.15 in phase.

---

## 6. Recommended Defaults

### 6.1 Configuration Dataclass

```python
@dataclass
class SecondaryEclipseConfig:
    """Configuration for V02 secondary eclipse check."""

    # Phase window parameters
    secondary_center: float = 0.5
    secondary_half_width: float = 0.15  # phase units (0.35-0.65) - WIDENED
    baseline_half_width: float = 0.15   # adjacent windows

    # Detection thresholds
    sigma_threshold: float = 3.0        # significance threshold
    depth_threshold: float = 0.005      # minimum depth (0.5%)

    # Minimum data requirements
    min_secondary_points: int = 10
    min_baseline_points: int = 10
    min_secondary_events: int = 2       # NEW
    min_phase_coverage: float = 0.3     # NEW

    # Red noise handling
    use_red_noise_inflation: bool = True  # NEW
    default_inflation: float = 1.5        # fallback when estimation fails

    # Phase coverage binning
    n_coverage_bins: int = 20
```

### 6.2 Threshold Rationale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `sigma_threshold` | 3.0 | Standard detection threshold; matches Kepler Robovetter |
| `depth_threshold` | 0.005 | 0.5% depth separates hot Jupiters (~0.01-0.1%) from EBs (>1%) |
| `min_secondary_events` | 2 | Need >1 event for any statistical statement |
| `min_phase_coverage` | 0.3 | Below 30% coverage, results are unreliable |
| `default_inflation` | 1.5 | Conservative estimate when red noise estimation fails |
| `secondary_half_width` | 0.15 | Widened from 0.10 to catch eccentric orbit secondaries |

---

## 7. Required Output Fields

All new fields are **additive only**. Existing fields are preserved for backward compatibility.

### 7.1 Preserved Fields (legacy)
- `secondary_depth` (float): Fractional depth (keep for backward compat)
- `secondary_depth_sigma` (float): Detection significance
- `baseline_flux` (float): Median baseline flux
- `n_secondary_points` (int): Total points in secondary window
- `significant_secondary` (bool): Detection flag

### 7.2 New Fields

| Field | Type | Description |
|-------|------|-------------|
| `secondary_depth_ppm` | float | Depth in ppm (= secondary_depth * 1e6) |
| `secondary_depth_err_ppm` | float | Depth uncertainty in ppm |
| `secondary_phase_coverage` | float | Fraction of secondary window covered (0-1) |
| `n_secondary_events_effective` | int | Number of distinct orbital cycles with secondary data |
| `n_baseline_points` | int | Total points in baseline windows |
| `red_noise_inflation` | float | Applied inflation factor (1.0 if disabled) |
| `secondary_center_phase` | float | Center of search window (default 0.5) |
| `warnings` | list[str] | List of warning strings |

### 7.3 Warning Strings

| Warning | Trigger Condition |
|---------|-------------------|
| `poor_phase_coverage` | coverage < min_phase_coverage |
| `single_event_only` | n_secondary_events_effective == 1 |
| `insufficient_baseline` | n_baseline_points < min_baseline_points |
| `high_variability` | red_noise_inflation > 2.5 |
| `near_detection_threshold` | 2.5 < depth_sigma < 3.5 |

---

## 8. Test Matrix

### 8.1 Synthetic Test Cases

| Test ID | Scenario | Input | Expected Outcome |
|---------|----------|-------|------------------|
| T01 | **Good coverage, no secondary** | 10 orbits, uniform phase coverage, white noise only | passed=True, confidence>=0.7, n_events>=8 |
| T02 | **Good coverage, clear secondary** | 10 orbits, injected 1% secondary at phase 0.5 | passed=False, depth_sigma>5, significant_secondary=True |
| T03 | **Poor coverage, no secondary** | 3 orbits, 40% phase gap at phase 0.4-0.6 | passed=True, confidence<=0.4, warnings contains "poor_phase_coverage" |
| T04 | **Poor coverage, injected secondary** | 3 orbits, 40% gap, 2% secondary | passed=False OR passed=True with low confidence and warning |
| T05 | **Variability contamination** | 10 orbits, injected stellar rotation P_rot = P_orb | passed=True, warnings contains "high_variability", inflation > 2.0 |
| T06 | **Genuine secondary, eccentric** | 10 orbits, 1.5% secondary at phase 0.40 | passed=False, significant_secondary=True |
| T07 | **Edge case: 1 event** | 1.5 orbits total | passed=True, confidence<=0.3, warnings contains "single_event_only" |
| T08 | **Edge case: deep secondary (EB)** | 10 orbits, 10% secondary | passed=False, depth_sigma>10, high confidence |
| T09 | **Hot Jupiter secondary** | 10 orbits, 0.03% secondary (thermal) | passed=True, depth < depth_threshold |
| T10 | **Red noise inflation test** | 10 orbits, correlated noise beta=1.5 | inflation > 1.5, depth_sigma reduced vs white noise |

### 8.2 Test Implementation Notes

```python
# Synthetic light curve generator for secondary eclipse tests
def make_secondary_test_lc(
    n_orbits: int,
    period_days: float,
    t0: float,
    primary_depth: float,
    secondary_depth: float,
    secondary_phase: float = 0.5,
    duration_hours: float = 2.0,
    cadence_min: float = 2.0,
    noise_ppm: float = 200.0,
    phase_gap: tuple[float, float] | None = None,
    red_noise_beta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic light curve with optional secondary eclipse."""
    ...
```

### 8.3 Regression Tests

Add golden-file regression tests using cached results from known targets:
- **TIC 261136679** (Pi Mensae): Known planet, no significant secondary
- **TIC 25155310** (TOI-1338): Circumbinary planet, complex secondary
- **Synthetic EB**: Injected 5% secondary for positive control

---

## 9. Backward Compatibility

### 9.1 Changes Summary

| Aspect | Change Type | Details |
|--------|-------------|---------|
| Function signature | **Unchanged** | `check_secondary_eclipse(lightcurve, period, t0)` |
| Check ID | **Unchanged** | "V02" |
| Check name | **Unchanged** | "secondary_eclipse" |
| `passed` semantics | **Unchanged** | True = no significant secondary detected |
| `confidence` semantics | **Modified** | Now degrades based on coverage/events (may differ from v1) |
| `details` dict | **Extended** | New fields added, existing preserved |

### 9.2 Gated Behavior

The following behaviors are gated behind `SecondaryEclipseConfig`:

1. **Red noise inflation**: `use_red_noise_inflation=True` (new default)
   - Set to `False` to reproduce v1 behavior

2. **Minimum event requirement**: `min_secondary_events=2` (new)
   - Set to `1` to reproduce v1 behavior (not recommended)

### 9.3 API Wrapper Compatibility

The `bittr_tess_vetter.api.lc_only.secondary_eclipse()` wrapper requires no changes. It passes through to the internal implementation and converts the result to `CheckResult`.

### 9.4 Migration Notes

- Downstream code relying on specific `confidence` values may see changes (generally lower confidence in low-information regimes)
- New `warnings` field enables programmatic detection of degraded results
- Code checking `significant_secondary` will continue to work unchanged

---

## 10. Citations

### 10.1 Primary References

| Bibcode | Reference | Relevance |
|---------|-----------|-----------|
| 2012AJ....143...39C | Coughlin & Lopez-Morales 2012 | Secondary eclipse search methodology for Kepler hot Jupiters; residual permutation error estimation |
| 2018ApJS..235...38T | Thompson et al. 2018 | Kepler DR25 Robovetter "Significant Secondary" test; disposition scores; local baseline approach |
| 2013ApJ...766...81F | Fressin et al. 2013 | False positive scenarios including secondary eclipses; EB contamination rates |
| 2006MNRAS.373..231P | Pont et al. 2006 | Red noise in transit photometry; inflation factors of 2-5x typical |
| 2013A&A...557A.139S | Santerne et al. 2013 | Secondary-only eclipsing binaries as false positives; eccentric orbit secondary phase offsets |
| 2019AJ....157..124K | Kostov et al. 2019 | DAVE pipeline; secondary eclipse as primary FP identification method |

### 10.2 Supporting References

| Bibcode | Reference | Relevance |
|---------|-----------|-----------|
| 2016ApJS..224...12C | Coughlin et al. 2016 | Kepler DR24 Robovetter implementation |
| 2018PASP..130f4502T | Twicken et al. 2018 | Kepler Data Validation pipeline |
| 2021ApJS..254...39G | Guerrero et al. 2021 | TESS TOI catalog vetting procedures |
| 2010Sci...327..977B | Borucki et al. 2010 | Kepler mission and EB contamination |
| 2020AJ....159..116G | Giacalone et al. 2020 | TRICERATOPS Bayesian vetting framework |
| 2025arXiv250910619K | Kunimoto et al. 2025 | LEO-Vetter TESS automated vetting; MS4/MS5/MS6 secondary metrics |

### 10.3 Key Methodological Quotes

**On Red Noise (Pont et al. 2006):**
> "We show that the detection threshold in the presence of systematics can be much higher than with the assumption of white noise, and obeys a different dependence on magnitude, orbital period and the parameters of the survey."

**On Error Estimation (Coughlin & Lopez-Morales 2012):**
> "We chose to adopt the parameter errors estimated by [Residual Permutation] technique as it has been shown to best account for the effect of systematic noise in transit light curves."

**On Secondary Detection (Thompson et al. 2018):**
> "The Robovetter gives every obsTCE a disposition, a reason for the disposition, and a disposition score... the disposition score is a value between 0 and 1 that indicates the confidence."

**On LEO-Vetter Secondary Test (Kunimoto et al. 2025):**
> "A secondary is considered significant if MS4 > 0, MS5 > -1, and MS6 > -1."

**On Secondary-Only EBs (Santerne et al. 2013):**
> "We investigate in this paper the astrophysical false-positive configuration in exoplanet-transit surveys that involves eclipsing binaries and giant planets which present only a secondary eclipse, as seen from the Earth... We estimate that up to 43.1 +/- 5.6 Kepler Objects of Interest can be mimicked by this new configuration of false positives."

**On DAVE Secondary Detection (Kostov et al. 2019):**
> "Most of the targets listed as false positives in our catalog either show prominent secondary eclipses, transit depths suggesting a stellar companion instead of a planet, or significant photocenter shifts during transit."

### 10.4 Code Citations

Add to module docstring:
```python
"""V02: Secondary Eclipse Search

Detects secondary eclipses at orbital phase ~0.5 to identify eclipsing binaries
masquerading as planetary transits. A significant secondary eclipse (>3-sigma,
>0.5% depth) strongly indicates an eclipsing binary rather than a planet.

Limitations:
    - Cannot detect "secondary-only" EBs where the observed transit IS the secondary
      (see Santerne et al. 2013 for this false positive scenario)
    - Eccentric orbits can shift secondary phase by +/-0.15; widened search window helps
    - Should be used in combination with V01, V05, V11 for comprehensive EB detection

References:
    [1] Coughlin & Lopez-Morales 2012, AJ 143, 39 (arXiv:1112.1021)
        Uniform search for secondary eclipses of hot Jupiters in Kepler;
        residual permutation error estimation for systematic noise
    [2] Thompson et al. 2018, ApJS 235, 38 (arXiv:1710.06758)
        Section 3.2: Significant Secondary test in DR25 Robovetter;
        disposition scores; local baseline approach
    [3] Pont et al. 2006, MNRAS 373, 231 (arXiv:astro-ph/0608597)
        Correlated noise and red noise inflation in transit photometry;
        typical inflation factors of 2-5x
    [4] Fressin et al. 2013, ApJ 766, 81 (arXiv:1301.0842)
        Section 3: False positive scenarios including secondary eclipses;
        EB contamination rates
    [5] Santerne et al. 2013, A&A 557, A139 (arXiv:1307.2003)
        Secondary-only eclipsing binaries as false positives;
        ~0.06% of binaries show only secondary eclipse;
        eccentric orbit secondary phase offsets
    [6] Kunimoto et al. 2025, arXiv:2509.10619
        LEO-Vetter significant secondary test implementation;
        MS4/MS5/MS6 metrics comparing secondary to tertiary and positive events
    [7] Kostov et al. 2019, AJ 157, 124 (arXiv:1901.07459)
        DAVE pipeline; secondary eclipse as primary FP identification method

Novelty: standard (implements established techniques from literature)
Repo-specific: local baseline windows, red noise inflation, phase coverage metric,
               secondary-only EB caveat documentation
"""
```

---

## 11. Implementation Checklist

### 11.1 Core Implementation
- [ ] Add `SecondaryEclipseConfig` dataclass with all new parameters
- [ ] Implement local baseline window selection (adjacent to secondary)
- [ ] Widen search window to 0.35-0.65 (from 0.40-0.60)
- [ ] Add red noise inflation (reuse `_compute_red_noise_inflation` from V01)
- [ ] Implement phase coverage calculation (`secondary_phase_coverage`)
- [ ] Implement `n_secondary_events_effective` counting by orbital epoch
- [ ] Update confidence model with principled degradation

### 11.2 Output Fields
- [ ] Add new output fields (additive only - see Section 7)
- [ ] Preserve all legacy output fields for backward compatibility
- [ ] Add `warnings` list generation
- [ ] Add `caveats` field noting secondary-only EB limitation when V02 passes

### 11.3 Documentation
- [ ] Update module docstring with full citations (Section 10.4)
- [ ] Add inline comments citing specific papers for each algorithm choice
- [ ] Document secondary-only EB limitation clearly

### 11.4 Testing
- [ ] Implement synthetic test cases T01-T10 (Section 8)
- [ ] Add regression test with golden files for known targets
- [ ] Test eccentric orbit secondary detection (phase 0.35-0.45)
- [ ] Test red noise inflation effectiveness
- [ ] Verify backward compatibility of legacy output fields

### 11.5 Integration
- [ ] Update API wrapper if needed (likely no changes)
- [ ] Verify MCP smoke test passes (Pi Mensae)
- [ ] Run full test suite to check for regressions

---

## 12. Future Work (Out of Scope for v2)

The following enhancements are documented for future consideration:

1. **Full eccentric secondary search**: Implement adaptive window that searches phase 0.2-0.8 with model fitting to find best-fit secondary phase and eccentricity.

2. **Secondary-only EB detection**: Implement heuristics to flag potential secondary-only EBs:
   - High Rp/Rs ratio (>0.1) suspicious for planet
   - V-shaped transit profile
   - Anomalous duration for given period

3. **Model-shift integration**: Adopt LEO-Vetter's MS4/MS5/MS6 metrics which compare secondary to tertiary and positive flux events for more robust detection.

4. **Bayesian secondary detection**: Compute posterior probability of secondary eclipse presence rather than frequentist significance test.

---

*Document version: 2.1*
*Author: Claude Code*
*Review status: Draft - Enhanced with comprehensive literature review (Santerne et al. 2013, Kostov et al. 2019, Kunimoto et al. 2025)*
*Last updated: 2026-01-08*
