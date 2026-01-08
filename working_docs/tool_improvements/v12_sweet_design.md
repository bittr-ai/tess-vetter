# V12 SWEET Check: Design Document

**Version**: 2.0 (Revised with literature research)
**Date**: 2026-01-08
**Author**: Claude (AI Assistant)

---

## 1. Current State - Implementation Analysis

### 1.1 Code Location

The V12 SWEET (Sine Wave Evaluation for Ephemeris Transits) check is implemented in two files:

- **Internal implementation**: `/src/bittr_tess_vetter/validation/exovetter_checks.py` (`SWEETCheck` class)
- **Public API**: `/src/bittr_tess_vetter/api/exovetter.py` (`sweet()` function)

### 1.2 Current Behavior

The check wraps the `exovetter` library's `Sweet` vetter class. It fits sinusoidal models at three periods to detect stellar variability that could masquerade as planetary transits:

| Period | Description | Purpose |
|--------|-------------|---------|
| P/2 | Half transit period | Detects even harmonics of stellar rotation |
| P | Transit period | Direct variability at the candidate period |
| 2P | Double transit period | Detects subharmonic variability |

For each period, the check computes an **amplitude-to-uncertainty ratio** (`amp_ratio = amplitude / uncertainty`).

### 1.3 Current Thresholds

```python
threshold = 3.0              # amplitude-to-uncertainty ratio for period P
half_period_threshold = 4.0  # for P/2
double_period_threshold = 4.0  # for 2P
```

### 1.4 Current Pass/Fail Logic

```python
# From exovetter_checks.py lines 453-460
fails_at_period = period_ratio > threshold
fails_at_half = half_p_ratio > half_p_thresh
fails_at_double = double_p_ratio > double_p_thresh

# Primary decision: only fails if variability at transit period
passed = not fails_at_period
```

**Key observation**: The current implementation only **fails** on variability at the transit period P. Variability at P/2 or 2P triggers warnings but does not cause failure, even when such variability could fully explain the observed transit signal.

### 1.5 Current Output Fields

```python
details = {
    "period_amplitude_ratio": float,
    "half_period_amplitude_ratio": float,
    "double_period_amplitude_ratio": float,
    "threshold": float,
    "amplitude_details": dict,
    "fails_at_period": bool,
    "fails_at_half_period": bool,
    "fails_at_double_period": bool,
    "exovetter_message": str,
    "interpretation": str,
}
```

### 1.6 Current Limitations

1. **No input validation**: Does not check for minimum data requirements (baseline length, number of transits, cadence)
2. **No SNR context**: Confidence is assigned without knowledge of transit SNR
3. **Harmonic aliasing unhandled**: Stellar variability at P/2 can produce transit-like dips when phase-folded at P, but the check only warns rather than failing
4. **No structured `warnings` list**: Messages are free-text only
5. **No `inputs_summary`**: Downstream tools cannot assess whether the check had sufficient data
6. **Threshold interpretation unclear**: The 3-sigma threshold is appropriate for well-sampled data but may be too aggressive for sparse baselines

---

## 2. Literature Background

### 2.1 SWEET Test Origins

The SWEET test was developed as part of the Kepler Robovetter pipeline for automated candidate vetting. The definitive references are:

**Thompson et al. 2018** (arXiv:1710.06758)
"Planetary Candidates Observed by Kepler. VIII. A Fully Automated Catalog With Measured Completeness and Reliability Based on Data Release 25"

> The SWEET NTL (Not Transit-Like) test detects sinusoidal variability at the transit period. From Section 3.2.4 and Table 3: "The TCE is sinusoidal" - categorized under the Not Transit-Like major flag with minor flag `SWEET_NTL`.

Key insights from Thompson et al.:
- SWEET is one of several "Not Transit-Like" (NTL) tests in the Robovetter
- The test is designed to catch stellar variability (rotation, pulsation) mimicking transits
- It operates on phase-folded light curves at P, P/2, and 2P

**Coughlin et al. 2016** (2016ApJS..224...12C)
"Planetary Candidates Observed by Kepler. VII. The First Fully Uniform Catalog Based on the Entire 48-month Data Set (Q1-Q17 DR24)"

> Section 4.4 describes the original SWEET implementation in the DR24 Robovetter.

### 2.2 Stellar Variability Context

Understanding the expected amplitudes and periods of stellar variability is crucial for SWEET calibration.

**McQuillan et al. 2014** (arXiv:1402.5694)
"Rotation Periods of 34,030 Kepler Main-Sequence Stars: The Full Autocorrelation Sample"

Key findings:
- Rotation periods range from 0.2 to 70 days for main-sequence stars below 6500 K
- 25.6% of Kepler targets have detectable rotation periods
- The period-temperature relation shows distinct populations for fast vs. slow rotators
- Young, active stars have shorter rotation periods (1-10 days) and higher variability amplitudes

**McQuillan et al. 2013** (arXiv:1308.1845)
"Stellar Rotation Periods of the Kepler Objects of Interest: A Dearth of Close-in Planets around Fast Rotators"

> Demonstrates that 737 of 1919 main-sequence planet hosts have detectable rotation periods. This means ~40% of planet host stars exhibit photometric modulation that could interfere with transit detection.

**Basri et al. 2013** (arXiv:1304.0136)
"Photometric Variability in Kepler Target Stars. III. Comparison with the Sun on Different Timescales"

> Establishes typical variability amplitudes:
> - Solar-type stars: 0.1-1 mmag RMS on rotation timescales
> - Active stars: up to 10+ mmag peak-to-peak
> - This translates to 100-10000 ppm, overlapping significantly with transit depths

### 2.3 Period Aliasing in Transit Detection

**Cooke et al. 2021** (ads:2021MNRAS.500.5088C)
"Resolving period aliases for TESS monotransits recovered during the extended mission"

> While focused on monotransits, this paper establishes the general problem of period aliasing in TESS data. Key insight: when the true period is an integer multiple of the observing baseline, harmonic confusion becomes severe.

**Coughlin et al. 2014** (arXiv:1401.1240)
"Contamination in the Kepler Field. Identification of 685 KOIs as False Positives Via Ephemeris Matching Based On Q1-Q12 Data"

> Documents that 12% of KOIs were false positives due to contamination, including ephemeris matching at period aliases. This establishes the scale of the aliasing problem.

### 2.4 False Positive Mechanisms

**Morton et al. 2016** (arXiv:1605.02825)
"False Positive Probabilities for all Kepler Objects of Interest"

> Presents FPP calculations for all KOIs, demonstrating that stellar variability is a significant source of false positives, particularly for small planets with shallow transits.

---

## 3. Problems Identified

From the spec (`v1_spec.md`) and literature analysis:

### 3.1 P/2 Aliasing: Stellar Rotation at Twice the Transit Period

**Physical scenario**: A star rotates with period P_rot = 2 * P_transit. Starspots create sinusoidal brightness variations. When the light curve is phase-folded at P_transit:
- The sinusoidal pattern folds coherently
- Flux minima appear at consistent phases, mimicking transits
- The expected "transit depth" is approximately 2*A (peak-to-trough of the sinusoid)

**Current problem**: The check flags P/2 variability but does not:
1. Compute the expected transit-like depth from the P/2 variability
2. Compare the observed transit depth to the expected variability amplitude
3. Fail the check when P/2 variability could explain the transit

**Impact**: A significant fraction of false positives from active stars go undetected.

### 3.2 2P Aliasing: Stellar Rotation at Half the Transit Period

**Physical scenario**: A star rotates with period P_rot = P_transit / 2. The variability:
- Creates phase-coherent modulation that reinforces every other transit
- Produces odd/even depth differences that mimic an eclipsing binary

**Current problem**: The implementation warns but does not quantify the impact on transit-like signals.

### 3.3 Low-Information Regime

The check produces confident results even when:
- Baseline < 2 * period (cannot reliably fit a sinusoid at 2P)
- N_transits < 3 (odd/even aliasing becomes problematic)
- Cadence is too coarse relative to the period

From Christiansen et al. 2020 (arXiv:2010.04796):
> Detection efficiency depends strongly on the number of transits and the correlated noise properties. Fewer transits lead to less reliable period determination.

### 3.4 Threshold Calibration

The current 3-sigma threshold for amp_ratio was derived from Kepler data with:
- 4-year baselines
- 30-minute cadence
- Well-characterized noise properties

For TESS with:
- 27-day sector baselines (or stitched multi-sector)
- 2-minute cadence
- Different systematics

The threshold may need recalibration. Per Thompson et al. 2018, the threshold was tuned for "completeness and effectiveness" on Kepler data specifically.

---

## 4. Proposed Improvements

### 4.1 Add Minimum Data Requirements

Before running the sinusoid fit, validate that the data can support reliable detection:

```python
def _validate_inputs(self, lightcurve, candidate):
    """Validate input data quality for SWEET analysis."""
    baseline_days = lightcurve.time.max() - lightcurve.time.min()
    n_cycles = baseline_days / candidate.period

    min_cycles = self.config.additional.get("min_cycles_for_fit", 2.0)

    if n_cycles < min_cycles:
        return False, "insufficient_baseline"

    # Check for 2P detection (need baseline > 2 * 2P = 4P)
    can_detect_2p = n_cycles >= 4.0

    return True, None
```

### 4.2 Harmonic Analysis: Compute Transit-Mimicking Amplitude

For each harmonic, compute whether the variability could explain the observed transit:

```python
def _analyze_harmonics(self, metrics, transit_depth_ppm, period):
    """Compute whether variability at harmonics could explain the transit."""

    # Variability at P/2 with amplitude A creates transit-like depth of ~2*A
    # when phase-folded at period P
    amp_at_half_p = metrics.get("half_period", {}).get("amplitude", 0) * 1e6
    variability_induced_depth_at_half_p = 2.0 * amp_at_half_p

    # Variability at P directly maps to depth
    amp_at_p = metrics.get("period", {}).get("amplitude", 0) * 1e6
    variability_induced_depth_at_p = 2.0 * amp_at_p

    # Fraction of transit depth explainable by variability
    max_variability_depth = max(
        variability_induced_depth_at_p,
        variability_induced_depth_at_half_p
    )
    variability_explains_fraction = max_variability_depth / transit_depth_ppm

    return {
        "variability_induced_depth_at_P_ppm": variability_induced_depth_at_p,
        "variability_induced_depth_at_half_P_ppm": variability_induced_depth_at_half_p,
        "variability_explains_depth_fraction": min(variability_explains_fraction, 1.0),
        "dominant_variability_period": self._identify_dominant(metrics),
    }
```

### 4.3 Revised Pass/Fail Logic

The pass/fail logic should consider whether variability at any harmonic could explain the transit:

```python
def _evaluate_pass_fail(self, metrics, harmonic_analysis, config):
    """Evaluate pass/fail with harmonic-aware logic."""
    threshold = config.threshold or 3.5
    half_p_thresh = config.additional.get("half_period_threshold", 3.5)
    depth_thresh = config.additional.get("variability_depth_threshold", 0.5)

    period_ratio = metrics["period"]["ratio"]
    half_p_ratio = metrics["half_period"]["ratio"]
    explains_fraction = harmonic_analysis["variability_explains_depth_fraction"]

    # Fail if variability at P is significant AND explains significant depth
    fails_at_period = (
        period_ratio > threshold
        and explains_fraction > depth_thresh
    )

    # Fail if variability at P/2 is significant AND explains significant depth
    # This is the key improvement for harmonic aliasing
    fails_at_half_period = (
        half_p_ratio > half_p_thresh
        and harmonic_analysis["variability_induced_depth_at_half_P_ppm"]
            > (depth_thresh * self._get_transit_depth())
    )

    passed = not (fails_at_period or fails_at_half_period)

    return passed, {
        "fails_at_period": fails_at_period,
        "fails_at_half_period": fails_at_half_period,
        "fails_at_double_period": False,  # Warning only, per original design
    }
```

### 4.4 Confidence Scaling

Confidence should degrade based on data quality and warning flags:

| Condition | Confidence Adjustment |
|-----------|----------------------|
| n_cycles_observed < 5 | -0.15 |
| n_transits < 5 | -0.10 |
| SNR < 10 | -0.10 |
| Variability at P/2 > 2-sigma | -0.10 |
| Variability at 2P > 2-sigma | -0.05 |
| Low phase coverage | -0.10 |

```python
def _compute_confidence(self, passed, warnings, inputs_summary, metrics):
    """Compute confidence with data-quality scaling."""
    # Base confidence
    base = 0.95 if passed else 0.90

    # Apply penalties
    penalties = 0.0

    if inputs_summary["n_cycles_observed"] < 5:
        penalties += 0.15
    if inputs_summary.get("n_transits", 10) < 5:
        penalties += 0.10
    if inputs_summary.get("snr") and inputs_summary["snr"] < 10:
        penalties += 0.10
    if "harmonic_variability_detected" in warnings:
        penalties += 0.10

    # Floor at 0.30 for degraded checks
    confidence_floor = self.config.additional.get("confidence_floor", 0.30)
    return max(base - penalties, confidence_floor)
```

---

## 5. Recommended Defaults

| Parameter | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| `threshold` (P) | 3.0 | 3.5 | Slightly more permissive; calibrated for shorter TESS baselines |
| `half_period_threshold` | 4.0 | 3.5 | Now causes failure if depth explained |
| `double_period_threshold` | 4.0 | 4.0 | Keep as warning-only trigger (per Thompson et al.) |
| `min_cycles_for_fit` | N/A | 2.0 | Minimum baseline/period ratio for reliable sinusoid fit |
| `variability_depth_threshold` | N/A | 0.5 | Fraction of depth explainable by variability before failure |
| `confidence_floor` | N/A | 0.30 | Minimum confidence for degraded checks |
| `include_harmonic_analysis` | N/A | False | Gate for new failure logic (backward compatible) |

### Configuration Structure (Additive)

```python
CheckConfig(
    enabled=True,
    threshold=3.5,
    additional={
        "half_period_threshold": 3.5,
        "double_period_threshold": 4.0,
        "min_cycles_for_fit": 2.0,
        "variability_depth_threshold": 0.5,
        "confidence_floor": 0.30,
        "include_harmonic_analysis": True,  # Opt-in for v1.x
    },
)
```

---

## 6. Required Output Fields (Additive)

All new fields are **additive only**. Existing fields are preserved for backward compatibility.

### 6.1 Existing Fields (Preserved)

```python
details = {
    "period_amplitude_ratio": float,
    "half_period_amplitude_ratio": float,
    "double_period_amplitude_ratio": float,
    "threshold": float,
    "amplitude_details": dict,
    "fails_at_period": bool,
    "fails_at_half_period": bool,
    "fails_at_double_period": bool,
    "exovetter_message": str,
    "interpretation": str,
}
```

### 6.2 New Fields

```python
details = {
    # ... existing fields ...

    # NEW: Structured warnings list
    "warnings": [
        "low_baseline_cycles",              # baseline < min_cycles * period
        "harmonic_variability_detected",    # P/2 or 2P signal > 2-sigma
        "variability_may_explain_transit",  # variability explains >30% of depth
        "insufficient_transits",            # n_transits < 3
        "low_snr",                          # SNR < 7
        "cannot_detect_2p",                 # baseline too short for 2P analysis
    ],

    # NEW: Input quality summary
    "inputs_summary": {
        "n_transits": int,              # Number of observed transits
        "n_cycles_observed": float,     # Baseline / period
        "baseline_days": float,         # Total time span
        "cadence_minutes": float,       # Median cadence
        "snr": float | None,            # Transit SNR if available
        "n_points": int,                # Total valid data points
        "can_detect_2p": bool,          # Whether 2P analysis is reliable
    },

    # NEW: Harmonic analysis (gated by include_harmonic_analysis)
    "harmonic_analysis": {
        "variability_induced_depth_at_P_ppm": float,
        "variability_induced_depth_at_half_P_ppm": float,
        "variability_explains_depth_fraction": float,
        "dominant_variability_period": str,  # "P", "P/2", "2P", or "none"
    },

    # NEW: Aliasing flags
    "aliasing_flags": {
        "half_period_alias_risk": bool,   # P/2 variability could mimic transit
        "double_period_alias_risk": bool, # 2P variability could affect odd/even
        "dominant_alias": str | None,     # Most likely alias if any
    },
}
```

---

## 7. Test Matrix

### 7.1 Synthetic Test Cases

| ID | Test Case | Configuration | Expected Outcome |
|----|-----------|---------------|------------------|
| T1 | Clean planet | No variability, 27d baseline, depth=1000ppm | `passed=True`, confidence > 0.90, no warnings |
| T2 | Variability at P | amp_ratio=5.0 at transit period | `passed=False`, `fails_at_period=True` |
| T3 | Variability at P/2, explains depth | P/2 amp=500ppm, transit depth=800ppm | `passed=False`, `fails_at_half_period=True`, fraction > 0.5 |
| T4 | Variability at P/2, weak | P/2 amp=100ppm, transit depth=1000ppm | `passed=True`, warning `harmonic_variability_detected` |
| T5 | Variability at 2P | amp_ratio=5.0 at 2*period | `passed=True`, warning only |
| T6 | Low baseline (3 days, P=5d) | n_cycles < 1 | `passed=True`, confidence=0.30, warning `low_baseline_cycles` |
| T7 | Weak variability below threshold | amp_ratio=2.0 at P | `passed=True`, confidence > 0.85, no warnings |
| T8 | Strong variability, deep transit | amp_ratio=4.0 at P, depth=5000ppm, explains 15% | `passed=True`, confidence reduced |
| T9 | P/2 alias exact match | P/2 amp exactly matches transit depth | `passed=False`, fraction ~1.0 |
| T10 | High noise regime | noise_ppm > depth_ppm | `passed=True`, low confidence, warning `low_snr` |

### 7.2 Test Implementation

```python
import numpy as np
import pytest
from bittr_tess_vetter.api.exovetter import sweet
from bittr_tess_vetter.api.types import LightCurve, Ephemeris, Candidate


def make_synthetic_lc(
    baseline_days=27.0,
    period=5.0,
    depth_ppm=1000,
    noise_ppm=200,
    variability_period=None,
    variability_amp_ppm=0,
    n_points=5000,
    seed=42,
):
    """Create synthetic light curve for testing."""
    np.random.seed(seed)
    t0 = 1850.0
    duration_hours = 2.5

    time = np.linspace(t0, t0 + baseline_days, n_points)
    flux = np.ones_like(time)

    # Add transit
    phase = ((time - t0) % period) / period
    transit_width = (duration_hours / 24.0) / period / 2.0
    in_transit = np.abs(phase) < transit_width
    in_transit |= np.abs(phase - 1.0) < transit_width
    flux[in_transit] -= depth_ppm / 1e6

    # Add variability
    if variability_period and variability_amp_ppm > 0:
        var_phase = 2 * np.pi * time / variability_period
        flux += (variability_amp_ppm / 1e6) * np.sin(var_phase)

    # Add noise
    flux += np.random.normal(0, noise_ppm / 1e6, len(flux))
    flux_err = np.full_like(flux, noise_ppm / 1e6)

    return LightCurve(time=time, flux=flux, flux_err=flux_err)


class TestV12SWEETCheck:
    """Test suite for V12 SWEET check improvements."""

    def test_t1_clean_planet(self):
        """T1: Transit signal with no stellar variability passes."""
        lc = make_synthetic_lc(
            baseline_days=27.0, period=5.0, depth_ppm=1000,
            noise_ppm=200, variability_period=None
        )
        eph = Ephemeris(period_days=5.0, t0_btjd=1850.0, duration_hours=2.5)
        cand = Candidate(ephemeris=eph, depth_ppm=1000)

        result = sweet(lc, cand, config={"include_harmonic_analysis": True})

        assert result.passed is True
        assert result.confidence > 0.90
        assert "variability_may_explain_transit" not in result.details.get("warnings", [])

    def test_t2_variability_at_period(self):
        """T2: Strong variability at transit period should fail."""
        lc = make_synthetic_lc(
            baseline_days=60.0, period=5.0, depth_ppm=1000,
            noise_ppm=200, variability_period=5.0, variability_amp_ppm=2000
        )
        eph = Ephemeris(period_days=5.0, t0_btjd=1850.0, duration_hours=2.5)
        cand = Candidate(ephemeris=eph, depth_ppm=1000)

        result = sweet(lc, cand, config={"include_harmonic_analysis": True})

        assert result.passed is False
        assert result.details["fails_at_period"] is True

    def test_t3_half_period_alias_explains_depth(self):
        """T3: P/2 variability that explains transit depth should fail."""
        # Variability at 2*P (which is P/2 when folded at P)
        lc = make_synthetic_lc(
            baseline_days=60.0, period=5.0, depth_ppm=800,
            noise_ppm=200, variability_period=10.0, variability_amp_ppm=500
        )
        eph = Ephemeris(period_days=5.0, t0_btjd=1850.0, duration_hours=2.5)
        cand = Candidate(ephemeris=eph, depth_ppm=800)

        result = sweet(lc, cand, config={"include_harmonic_analysis": True})

        assert result.passed is False
        assert result.details["fails_at_half_period"] is True
        assert result.details["harmonic_analysis"]["variability_explains_depth_fraction"] > 0.5

    def test_t5_low_baseline(self):
        """T5: Insufficient baseline should return low confidence."""
        lc = make_synthetic_lc(
            baseline_days=3.0, period=5.0, depth_ppm=1000,  # < 1 cycle
            noise_ppm=200
        )
        eph = Ephemeris(period_days=5.0, t0_btjd=1850.0, duration_hours=2.5)
        cand = Candidate(ephemeris=eph, depth_ppm=1000)

        result = sweet(lc, cand, config={"include_harmonic_analysis": True})

        assert result.passed is True  # Default pass when data insufficient
        assert result.confidence <= 0.50
        assert "low_baseline_cycles" in result.details.get("warnings", [])
```

---

## 8. Backward Compatibility

### 8.1 What Changes

| Aspect | Before | After | Breaking? |
|--------|--------|-------|-----------|
| Pass/fail logic | Only fails on P | Also fails on P/2 if depth explained | **Gated** (opt-in) |
| Confidence values | Fixed tiers (0.75-0.95) | Scaled by data quality (0.30-0.95) | No |
| `details` keys | 11 keys | 14 keys | No (additive) |
| Error handling | Generic exceptions | Structured error details | No |

### 8.2 Gating Strategy

New harmonic failure logic is gated behind `config.additional["include_harmonic_analysis"]`:

```python
# Default: False (preserves existing behavior)
# Opt-in: True (enables new P/2 failure logic)

if additional.get("include_harmonic_analysis", False):
    # Run new harmonic analysis
    # Apply new failure logic for P/2
    harmonic_analysis = self._analyze_harmonics(...)
else:
    # Existing behavior only
    harmonic_analysis = {}
```

### 8.3 Migration Path

1. **v1.x**: Add new fields, gate harmonic failure logic behind flag (default: off)
2. **v2.0**: Change default to `include_harmonic_analysis=True`
3. Document the change in release notes with examples of affected cases

### 8.4 API Compatibility

The public API signature remains unchanged:

```python
def sweet(
    lc: LightCurve,
    candidate: Candidate,
    *,
    enabled: bool = True,
    config: dict[str, Any] | None = None,
) -> CheckResult:
```

New config options are passed via the `config` dict and default to backward-compatible values.

---

## 9. Citations

### 9.1 Primary References

1. **Thompson et al. 2018** (arXiv:1710.06758)
   - "Planetary Candidates Observed by Kepler. VIII. A Fully Automated Catalog With Measured Completeness and Reliability Based on Data Release 25"
   - Section 3.2.4 / Table 3: SWEET test definition in the DR25 Robovetter
   - Defines the amplitude-to-uncertainty ratio metric and NTL categorization
   - ADS: https://ui.adsabs.harvard.edu/abs/2018ApJS..235...38T

2. **Coughlin et al. 2016** (2016ApJS..224...12C)
   - "Planetary Candidates Observed by Kepler. VII. The First Fully Uniform Catalog Based on the Entire 48-month Data Set"
   - Section 4.4: Original SWEET implementation in the DR24 Robovetter
   - ADS: https://ui.adsabs.harvard.edu/abs/2016ApJS..224...12C

### 9.2 Stellar Variability Context

3. **McQuillan et al. 2014** (arXiv:1402.5694)
   - "Rotation Periods of 34,030 Kepler Main-Sequence Stars: The Full Autocorrelation Sample"
   - Largest stellar rotation period catalog; establishes expected variability timescales
   - ADS: https://ui.adsabs.harvard.edu/abs/2014ApJS..211...24M

4. **McQuillan et al. 2013** (arXiv:1308.1845)
   - "Stellar Rotation Periods of the Kepler Objects of Interest: A Dearth of Close-in Planets around Fast Rotators"
   - Documents rotation period detection in 40% of planet host stars
   - ADS: https://ui.adsabs.harvard.edu/abs/2013ApJ...775...11M

5. **Basri et al. 2013** (arXiv:1304.0136)
   - "Photometric Variability in Kepler Target Stars. III. Comparison with the Sun on Different Timescales"
   - Establishes typical variability amplitudes (0.1-10 mmag)
   - ADS: https://ui.adsabs.harvard.edu/abs/2013ApJ...769...37B

### 9.3 Period Aliasing and False Positives

6. **Cooke et al. 2021** (2021MNRAS.500.5088C)
   - "Resolving period aliases for TESS monotransits recovered during the extended mission"
   - Establishes period aliasing challenges in TESS data
   - ADS: https://ui.adsabs.harvard.edu/abs/2021MNRAS.500.5088C

7. **Coughlin et al. 2014** (arXiv:1401.1240)
   - "Contamination in the Kepler Field. Identification of 685 KOIs as False Positives Via Ephemeris Matching"
   - Documents 12% FP rate from contamination, including period aliases
   - ADS: https://ui.adsabs.harvard.edu/abs/2014AJ....147..119C

8. **Morton et al. 2016** (arXiv:1605.02825)
   - "False Positive Probabilities for all Kepler Objects of Interest"
   - Comprehensive FPP calculations; context for stellar variability FP rates
   - ADS: https://ui.adsabs.harvard.edu/abs/2016ApJ...822...86M

### 9.4 Pipeline Detection Efficiency

9. **Christiansen et al. 2020** (arXiv:2010.04796)
   - "Measuring Transit Signal Recovery in the Kepler Pipeline IV: Completeness of the DR25 Planet Candidate catalog"
   - Documents detection efficiency dependence on number of transits and noise properties
   - ADS: https://ui.adsabs.harvard.edu/abs/2020AJ....160..159C

### 9.5 Code Citation Format

In module docstrings:
```python
"""V12: SWEET test for stellar variability masquerading as transits.

References:
    [1] Thompson et al. 2018, ApJS 235, 38 (arXiv:1710.06758)
        Section 3.2.4: SWEET test for stellar variability detection
    [2] Coughlin et al. 2016, ApJS 224, 12 (2016ApJS..224...12C)
        Section 4.4: Original SWEET implementation methodology
    [3] McQuillan et al. 2014, ApJS 211, 24 (arXiv:1402.5694)
        Stellar rotation periods establishing expected variability timescales
"""
```

In function decorators:
```python
@cites(
    cite(THOMPSON_2018, "Section 3.2.4 SWEET stellar variability test"),
    cite(COUGHLIN_2016, "Section 4.4 original SWEET implementation"),
    cite(MCQUILLAN_2014, "Stellar rotation context"),
)
def sweet(...):
    ...
```

---

## 10. Summary

This design document proposes improvements to the V12 SWEET check focused on:

1. **Harmonic aliasing detection**: The primary enhancement addresses the P/2 aliasing problem where stellar rotation at twice the transit period can produce transit-like signals. The new logic fails candidates when variability at P/2 could explain >50% of the observed transit depth.

2. **Data quality awareness**: New input validation and confidence scaling ensure that the check degrades gracefully when data is insufficient for reliable analysis.

3. **Structured output**: New `warnings`, `inputs_summary`, and `harmonic_analysis` fields provide machine-readable diagnostics for downstream tools.

4. **Backward compatibility**: All changes are additive, with the new failure logic gated behind an opt-in configuration flag.

5. **Literature grounding**: The improvements are based on established methods from the Kepler Robovetter (Thompson et al. 2018) and informed by stellar variability studies (McQuillan et al. 2014, Basri et al. 2013).
