# Design Note: Improved ModShift Check (V11)

**Version:** 1.1
**Last Updated:** 2026-01-08
**Status:** Draft
**Author:** Claude Opus 4.5

## 1. Executive Summary

The current `ModshiftCheck` (V11) wraps the exovetter library's ModShift implementation to detect secondary eclipses at arbitrary phases, identifying eccentric eclipsing binaries that would be missed by the standard phase-0.5 secondary search. This design note proposes improvements to standardize output semantics, add structured diagnostics, and improve confidence mapping.

**Key improvements:**
- Standardize `passed` meaning: "no strong evidence of EB/systematic" (not "validated planet")
- Confidence mapping driven by SNR, number of transits, Fred (red noise), and exovetter flags
- Structured `warnings` list and `inputs_summary` for downstream interpretation
- Explicit handling of folded/pre-processed inputs
- Unified interpretation text across all result paths
- Fred-gated reliability regime classification

---

## 2. Current Implementation Analysis

### 2.1 Current Architecture

The ModShift check is implemented in two layers:
- **Core check**: `bittr_tess_vetter.validation.exovetter_checks.ModshiftCheck`
- **API wrapper**: `bittr_tess_vetter.api.exovetter.modshift()`

The implementation creates a `_LightkurveLike` wrapper around the domain `LightCurveData` and passes it to exovetter's `ModShift` vetter along with a TCE (Threshold Crossing Event) object.

### 2.2 Current Issues

| Issue | Impact | Severity |
|-------|--------|----------|
| `passed` semantics unclear | Users may interpret "passed" as "validated" rather than "no EB evidence" | High |
| No handling of folded/phase-folded inputs | exovetter may produce invalid results on pre-processed data | Medium |
| No `inputs_summary` | Downstream tools cannot assess data quality used for the check | Medium |
| Confidence mapping is ad-hoc | Confidence values are hardcoded without clear regime definitions | Medium |
| No structured `warnings` | Text-based interpretation harder to parse programmatically | Low |
| Fred threshold (2.0) may be too permissive | High red noise can invalidate ModShift results | Medium |

### 2.3 Current Output Fields

```python
details = {
    "primary_signal": float,      # pri from exovetter
    "secondary_signal": float,    # sec from exovetter
    "tertiary_signal": float,     # ter from exovetter
    "positive_signal": float,     # pos from exovetter
    "fred": float,                # Red noise factor
    "false_alarm_threshold": float,
    "secondary_primary_ratio": float,
    "tertiary_primary_ratio": float,
    "threshold": float,           # Config threshold (default 0.5)
    "significant_secondary": bool,
    "secondary_above_fa": bool,
    "interpretation": str,        # Human-readable text
}
```

---

## 2.5 Literature Background: ModShift Algorithm

### 2.5.1 Origin and Purpose

The Model-Shift test was developed for the Kepler Data Validation (DV) pipeline to identify eclipsing binaries (EBs) that might masquerade as planet candidates. The fundamental insight is that true transiting planets should only produce a single periodic dip (the transit), while EBs typically show both primary and secondary eclipses.

**Key problem addressed:** Standard secondary eclipse searches look only at orbital phase 0.5 (opposite the primary). However, eccentric orbits can place the secondary eclipse at other phases. The ModShift algorithm searches ALL phases for secondary signals.

### 2.5.2 Algorithm Description (from Thompson et al. 2018)

The ModShift algorithm works as follows:

1. **Phase-fold the light curve** at the candidate orbital period
2. **Create a matched filter** (transit-shaped kernel based on reported duration)
3. **Cross-correlate (convolve)** the folded light curve with the transit template
4. **Identify signal peaks** in the convolution result:
   - **pri (primary):** Most negative value (the detected transit/eclipse)
   - **sec (secondary):** Second most negative value at any phase
   - **ter (tertiary):** Third most negative value
   - **pos (positive):** Most positive value (brightening events)
5. **Compute the Fred metric:** std(convolution) / std(lightcurve)

### 2.5.3 The Fred Metric

Fred ("Fraction of Red noise") quantifies correlated noise impact:

```
Fred = std(convolution_result) / std(phase_folded_flux)
```

**Interpretation:**
- **Fred ~ 1.0:** Light curve dominated by white noise; convolution effectively suppresses noise, results reliable
- **Fred ~ 1.5-2.0:** Some red noise present; results should be interpreted with caution
- **Fred > 2.0:** Significant correlated noise; convolution may produce spurious signals
- **Fred > 3.5:** Severe red noise; ModShift results unreliable

From Thompson et al. 2018, Section 3.2.3: "High Fred values indicate that correlated noise in the light curve could produce false secondary eclipse detections. The Robovetter uses Fred as a major factor in assessing ModShift reliability."

### 2.5.4 Secondary Eclipse Phase Offset

For eccentric orbits, the secondary eclipse phase depends on orbital elements. From Santerne et al. 2013 (arXiv:1307.2003):

```
phi_secondary = 0.5 + (2 * e * cos(omega)) / pi    [for small e]
```

Where:
- e = orbital eccentricity
- omega = argument of periastron

This means secondaries can appear at phases ranging from ~0.2 to ~0.8 for moderate eccentricities, making phase-agnostic searches essential.

### 2.5.5 False Positive Rates (Santerne et al. 2013)

Key findings on secondary-only eclipsing binary false positives:
- **0.061% +/- 0.017%** of main-sequence binary stars present only secondary eclipses mimicking planetary transits
- **0.009% +/- 0.002%** of stars harbor giant planets showing only occultations (thermal emission)
- These rates are significant for large surveys: ~43 Kepler KOIs could be secondary-only false positives

---

## 3. Problems Identified (from spec)

### 3.1 Output Mapping to `passed`/`confidence`

**Current logic:**
```python
significant_secondary = sec_pri_ratio > threshold  # threshold=0.5
sec_above_fa = sec > fa_thresh and sec > 0
passed = not (significant_secondary and sec_above_fa)
```

**Issues:**
1. Requires BOTH conditions to fail - this is conservative but may miss marginal EBs
2. The false alarm threshold from exovetter varies with data quality; using it as a hard gate can be inconsistent
3. No intermediate "warn" state for borderline cases

### 3.2 Threshold Defaults

**Current defaults:**
- `threshold`: 0.5 (sec/pri ratio)
- `fred_warning_threshold`: 2.0
- `tertiary_warning_threshold`: 0.3

**Issues:**
- The 0.5 ratio threshold is reasonable but should be contextualized by SNR
- Fred > 2.0 already indicates substantial red noise; confidence should degrade earlier
- Thompson et al. (2018) uses Fred as a major factor in their disposition logic

### 3.3 Handling Folded Inputs

**Current behavior:** No explicit check; exovetter may produce invalid results.

**Issue:** If the input light curve is already phase-folded (e.g., from `recover_transit`), the ModShift algorithm cannot properly identify secondary eclipse phases.

---

## 4. Proposed Improvements

### 4.1 Standardized `passed` Semantics

**Definition:**
- `passed=True`: "No strong evidence of eclipsing binary or systematic signal at secondary phase"
- `passed=False`: "Strong evidence of secondary eclipse suggesting EB or blend"

**NOT:**
- "Planet validated" (requires additional evidence)
- "Safe to publish" (other checks still needed)

### 4.2 Improved Pass/Fail Logic

```python
# Primary condition: secondary eclipse significance
significant_secondary = (
    sec_pri_ratio > threshold and
    sec > fa_thresh * 0.8  # Slightly below FA for margin
)

# Secondary condition: Fred-gated reliability
reliable_result = fred < fred_critical  # fred_critical = 3.0

# Disposition
if not reliable_result:
    passed = True  # Cannot reliably assess - default to pass with low confidence
    confidence = degraded_confidence(fred)
elif significant_secondary:
    passed = False
    confidence = high_confidence_fail(sec_pri_ratio, fred)
else:
    passed = True
    confidence = standard_confidence(sec_pri_ratio, fred, n_transits)
```

### 4.3 Confidence Mapping

**Regime definitions:**

| Regime | Condition | Base Confidence |
|--------|-----------|-----------------|
| High reliability | fred < 1.5, sec_pri < 0.2 | 0.90-0.95 |
| Standard | fred < 2.5, sec_pri < threshold | 0.75-0.85 |
| Marginal | fred < 3.5 OR sec_pri near threshold | 0.50-0.70 |
| Degraded | fred >= 3.5 OR insufficient data | 0.20-0.40 |
| Invalid | folded input, exovetter error | 0.0-0.15 |

**Confidence modifiers:**
- SNR available: +0.05 if SNR > 10
- n_transits > 5: +0.05
- tertiary_signal significant: -0.10 (multiple eclipse-like features)
- positive_signal > primary: -0.10 (potential systematics)

### 4.4 Structured Warnings

Replace ad-hoc interpretation text with structured warnings:

```python
warnings: list[str] = [
    "HIGH_RED_NOISE",           # fred > fred_warning_threshold
    "TERTIARY_SIGNAL",          # ter/pri > tertiary_warning_threshold
    "POSITIVE_SIGNAL_HIGH",     # pos > pri (brightening exceeds transit depth)
    "LOW_PRIMARY_SNR",          # pri < 5 * fa_thresh
    "FOLDED_INPUT_DETECTED",    # Input appears to be phase-folded
    "INSUFFICIENT_BASELINE",    # Not enough out-of-transit data
    "MARGINAL_SECONDARY",       # sec/pri between 0.3 and threshold
]
```

### 4.5 Input Summary

Add `inputs_summary` to details for transparency:

```python
inputs_summary = {
    "n_points": int,
    "n_transits_expected": int,  # floor(baseline / period)
    "cadence_median_min": float,
    "baseline_days": float,
    "snr": float | None,  # From candidate if available
    "is_folded": bool,
    "flux_err_available": bool,
}
```

### 4.6 Folded Input Detection

Detect and reject phase-folded inputs:

```python
def is_likely_folded(time: np.ndarray, period: float) -> bool:
    """Detect if input appears to be phase-folded."""
    baseline = time.max() - time.min()
    # If baseline < 1.5 periods, likely folded
    if baseline < 1.5 * period:
        return True
    # If time is normalized to [0, period], definitely folded
    if time.min() >= 0 and time.max() <= period * 1.1:
        return True
    return False
```

If folded input detected:
- Return `passed=True`, `confidence=0.10`
- Add warning: `"FOLDED_INPUT_DETECTED"`
- Add to interpretation: "ModShift requires unfolded time series; result is invalid."

---

## 5. Recommended Defaults

### 5.1 Thresholds

| Parameter | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| `threshold` (sec/pri) | 0.5 | 0.5 | Thompson et al. 2018 uses similar |
| `fred_warning_threshold` | 2.0 | 2.0 | Warn at this level |
| `fred_critical_threshold` | N/A | 3.5 | Above this, result unreliable |
| `tertiary_warning_threshold` | 0.3 | 0.25 | Slightly more sensitive |
| `marginal_secondary_threshold` | N/A | 0.3 | Trigger warning below main threshold |

### 5.2 Confidence Mapping Table

```python
def compute_confidence(
    passed: bool,
    sec_pri_ratio: float,
    fred: float,
    has_tertiary_warning: bool,
    n_transits: int | None,
    snr: float | None,
) -> float:
    """Compute confidence based on result quality indicators."""

    if not passed:
        # Failed check - confidence in the EB detection
        if sec_pri_ratio > 0.8:
            base = 0.95
        elif sec_pri_ratio > 0.6:
            base = 0.90
        else:
            base = 0.85
    else:
        # Passed check - confidence that it's NOT an EB
        if fred > 3.0:
            base = 0.35  # High red noise undermines result
        elif fred > 2.0:
            base = 0.60
        elif sec_pri_ratio < 0.2:
            base = 0.90  # Well below threshold
        elif sec_pri_ratio < 0.35:
            base = 0.80
        else:
            base = 0.70  # Near threshold

    # Modifiers
    if has_tertiary_warning:
        base -= 0.10
    if n_transits is not None and n_transits > 5:
        base += 0.05
    if snr is not None and snr > 10:
        base += 0.05

    return round(max(0.1, min(0.98, base)), 3)
```

---

## 6. Required Output Fields (Additive Only)

### 6.1 New Fields

```python
# Add to details dict:
{
    # Structured warnings (NEW)
    "warnings": ["HIGH_RED_NOISE", "MARGINAL_SECONDARY"],

    # Input summary (NEW)
    "inputs_summary": {
        "n_points": 15234,
        "n_transits_expected": 8,
        "cadence_median_min": 2.0,
        "baseline_days": 27.3,
        "snr": 12.5,
        "is_folded": False,
        "flux_err_available": True,
    },

    # Fred regime indicator (NEW)
    "fred_regime": "standard",  # "low", "standard", "high", "critical"

    # Explicit pass semantics (NEW)
    "passed_meaning": "no_strong_eb_evidence",
}
```

### 6.2 Preserved Fields (No Changes)

All existing fields are preserved for backward compatibility:
- `primary_signal`, `secondary_signal`, `tertiary_signal`, `positive_signal`
- `fred`, `false_alarm_threshold`
- `secondary_primary_ratio`, `tertiary_primary_ratio`
- `threshold`, `significant_secondary`, `secondary_above_fa`
- `interpretation`

---

## 7. Test Matrix

### 7.1 Synthetic Test Cases

| Case ID | Scenario | Primary (ppm) | Secondary (ppm) | Sec Phase | Fred | N_transits | Expected `passed` | Expected `confidence` |
|---------|----------|---------------|-----------------|-----------|------|------------|-------------------|----------------------|
| T01 | Clean planet | 1000 | 0 | - | ~1.0 | 10 | True | 0.90+ |
| T02 | EB at phase 0.5 | 10000 | 8000 | 0.50 | ~1.0 | 10 | False | 0.90+ |
| T03 | Eccentric EB | 10000 | 6000 | 0.30 | ~1.0 | 10 | False | 0.85+ |
| T04 | Shallow secondary | 5000 | 1500 | 0.50 | ~1.0 | 10 | True* | 0.75+ |
| T05 | High red noise | 1000 | 0 | - | 4.0 | 10 | True | <0.50 |
| T06 | Low SNR planet | 300 | 0 | - | ~2.0 | 10 | True | 0.65-0.80 |
| T07 | Few transits | 1000 | 0 | - | ~1.0 | 2 | True | 0.70-0.80 |
| T08 | Tertiary present | 5000 | 2000 | 0.50 | ~1.0 | 10 | False | 0.80+ |
| T09 | Positive event | 1000 | 0 | - | ~1.0 | 10 | True | 0.85 |
| T10 | Grazing EB | 2000 | 1800 | 0.50 | ~1.0 | 10 | False | 0.95 |
| T11 | Folded input | - | - | - | - | - | True | 0.10 |

\* T04: sec/pri = 0.30 is below threshold (0.5) but warns as marginal

### 7.2 Exovetter Metric Values (Mocked)

| Case ID | pri | sec | ter | pos | fred | fa_thresh |
|---------|-----|-----|-----|-----|------|-----------|
| T01 | 100 | 5 | 3 | 2 | 1.2 | 10 |
| T02 | 100 | 80 | 5 | 3 | 1.1 | 10 |
| T03 | 100 | 60 | 8 | 4 | 1.3 | 10 |
| T04 | 100 | 30 | 5 | 3 | 1.2 | 10 |
| T05 | 100 | 45 | 30 | 20 | 4.5 | 10 |
| T06 | 20 | 3 | 2 | 2 | 2.2 | 5 |
| T07 | 100 | 8 | 5 | 3 | 1.0 | 10 |
| T08 | 100 | 40 | 35 | 5 | 1.3 | 10 |
| T09 | 100 | 5 | 3 | 120 | 1.1 | 10 |
| T10 | 100 | 90 | 5 | 3 | 1.0 | 10 |

### 7.3 Expected Warnings per Test Case

| Case ID | Expected Warnings |
|---------|-------------------|
| T01 | [] (none) |
| T02 | [] (fail is clear) |
| T03 | [] (fail is clear) |
| T04 | ["MARGINAL_SECONDARY"] |
| T05 | ["HIGH_RED_NOISE", "FRED_UNRELIABLE"] |
| T06 | ["LOW_PRIMARY_SNR"] |
| T07 | ["LOW_TRANSIT_COUNT"] |
| T08 | ["TERTIARY_SIGNAL"] |
| T09 | ["POSITIVE_SIGNAL_HIGH"] |
| T10 | [] (fail is clear) |
| T11 | ["FOLDED_INPUT_DETECTED"] |

### 7.4 Test Implementation

```python
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_exovetter_metrics():
    """Factory for mocking exovetter ModShift results."""
    def _make(pri=100, sec=5, ter=3, pos=2, fred=1.2, fa_thresh=10):
        return {
            "pri": pri,
            "sec": sec,
            "ter": ter,
            "pos": pos,
            "Fred": fred,
            "false_alarm_threshold": fa_thresh,
        }
    return _make


class TestModshiftCleanPlanet:
    """Test clean planet scenario - should pass with high confidence."""

    def test_clean_planet_passes(self, mock_exovetter_metrics):
        metrics = mock_exovetter_metrics(pri=100, sec=5, ter=3, fred=1.2)
        # ... run check with mocked exovetter
        assert result.passed is True
        assert result.confidence >= 0.85
        assert "HIGH_RED_NOISE" not in result.details.get("warnings", [])


class TestModshiftSecondaryAt05:
    """Test EB with secondary at phase 0.5 - clear fail."""

    def test_secondary_at_phase_05_fails(self, mock_exovetter_metrics):
        metrics = mock_exovetter_metrics(pri=100, sec=60, ter=5, fred=1.3)
        # sec/pri = 0.60 > threshold (0.5)
        assert result.passed is False
        assert result.confidence >= 0.85
        assert result.details["significant_secondary"] is True


class TestModshiftEccentricEB:
    """Test eccentric EB with secondary at other phase."""

    def test_eccentric_eb_fails(self, mock_exovetter_metrics):
        metrics = mock_exovetter_metrics(pri=100, sec=55, ter=8, fred=1.5)
        assert result.passed is False
        assert 0.80 <= result.confidence <= 0.90


class TestModshiftRedNoise:
    """Test red noise dominated - should pass but with very low confidence."""

    def test_high_red_noise_degrades_confidence(self, mock_exovetter_metrics):
        metrics = mock_exovetter_metrics(pri=100, sec=45, ter=30, fred=4.5)
        assert result.passed is True  # Cannot reliably reject
        assert result.confidence < 0.50
        assert "HIGH_RED_NOISE" in result.details.get("warnings", [])
        assert result.details.get("fred_regime") == "critical"


class TestModshiftFoldedInput:
    """Test folded input detection and handling."""

    def test_folded_input_detected(self):
        # Create light curve with time spanning < 1.5 periods
        time = np.linspace(0, 1.2, 1000)  # 1.2 days, period=1.0
        # ... setup lightcurve and candidate
        assert result.passed is True
        assert result.confidence <= 0.15
        assert "FOLDED_INPUT_DETECTED" in result.details.get("warnings", [])


class TestModshiftInputsSummary:
    """Test that inputs_summary is populated correctly."""

    def test_inputs_summary_present(self):
        # ... run check
        assert "inputs_summary" in result.details
        summary = result.details["inputs_summary"]
        assert "n_points" in summary
        assert "n_transits_expected" in summary
        assert "is_folded" in summary
```

---

## 8. Backward Compatibility

### 8.1 Additive Changes (Safe)

These are purely additive and do not break existing consumers:

| Field | Type | Status |
|-------|------|--------|
| `warnings` | `list[str]` | NEW |
| `inputs_summary` | `dict` | NEW |
| `fred_regime` | `str` | NEW |
| `passed_meaning` | `str` | NEW |

### 8.2 Behavioral Changes (Gated)

Confidence values may change slightly due to improved mapping. This is gated by:
- No change to `passed` logic for existing threshold values
- Old confidence values were ad-hoc; new values are more principled

### 8.3 No Breaking Changes

- Function signatures unchanged
- Return type unchanged (`VetterCheckResult` / `CheckResult`)
- All existing `details` keys preserved
- Threshold defaults unchanged

### 8.4 Deprecation Notes

None required - all changes are additive.

---

## 9. Citations

### 9.1 Primary References

**Thompson et al. 2018** (DR25 Robovetter - Definitive Reference)
- arXiv: [1710.06758](https://arxiv.org/abs/1710.06758)
- ADS Bibcode: `2018ApJS..235...38T`
- Title: "Planetary Candidates Observed by Kepler. VIII. A Fully Automated Catalog With Measured Completeness and Reliability Based on Data Release 25"
- **Section 3.2.3:** ModShift implementation in DR25 Robovetter
- **Table 3:** Summary of Robovetter tests including ModShift metrics
- Relevance: Describes operational thresholds and Fred usage for 4034 planet candidates

**Coughlin et al. 2016** (DR24 Robovetter)
- arXiv: [1601.05413](https://arxiv.org/abs/1601.05413)
- ADS Bibcode: `2016ApJS..224...12C`
- Title: "Planetary Candidates Observed by Kepler. VII. The First Fully Uniform Catalog Based on The Entire 48 Month Data Set (Q1-Q17 DR24)"
- **Section 4.3:** ModShift false positive identification methodology
- Relevance: Earlier threshold calibration and decision logic

**Twicken et al. 2018** (Kepler DV Pipeline)
- arXiv: [1803.04526](https://arxiv.org/abs/1803.04526)
- ADS Bibcode: `2018PASP..130f4502T`
- Title: "Kepler Data Validation I - Architecture, Diagnostic Tests, and Data Products for Vetting Transiting Planet Candidates"
- Relevance: DV pipeline architecture; diagnostic test suite context

### 9.2 Secondary Eclipse False Positive Reference

**Santerne et al. 2013** (Secondary Eclipse False Positives)
- arXiv: [1307.2003](https://arxiv.org/abs/1307.2003)
- ADS Bibcode: `2013A&A...557A.139S`
- Title: "The contribution of secondary eclipses as astrophysical false positives to exoplanet transit surveys"
- **Key results:**
  - 0.061% +/- 0.017% of MS binaries present secondary-only eclipses mimicking planets
  - Secondary phase offset formula for eccentric orbits
  - Re-estimated Kepler false positive rate from 9.4% to 11.3%
- Relevance: Theoretical basis for why ModShift phase-agnostic search is necessary

### 9.3 Supporting References

**Mullally et al. 2016** (False Alarm Identification)
- arXiv: [1602.03204](https://arxiv.org/abs/1602.03204)
- ADS Bibcode: `2016PASP..128g4502M`
- Title: "Identifying False Alarms in the Kepler Planet Candidate Catalog"
- Relevance: Individual transit event vetting; complements ModShift with per-event analysis

**Coughlin et al. 2014** (Contamination Study)
- ADS Bibcode: `2014ApJS..212...25C`
- Title: "Contamination in the Kepler Field: Identification of 685 KOIs as False Positives"
- Relevance: Early systematic use of Model-Shift for contamination identification

### 9.4 Software References

**exovetter library**
- Repository: [spacetelescope/exovetter](https://github.com/spacetelescope/exovetter)
- Documentation: [exovetter.readthedocs.io](https://exovetter.readthedocs.io/)
- Relevance: Implementation used by V11; `ModShift` class wraps original algorithms
- ModShift API: [exovetter.vetters.ModShift](https://exovetter.readthedocs.io/en/latest/api/exovetter.vetters.ModShift.html)

### 9.5 Code Citation Block

```python
"""V11: ModShift test for secondary eclipse detection at arbitrary phase.

Detects eccentric eclipsing binaries where the secondary eclipse occurs at
an unexpected phase (not 0.5). This catches EBs that would be missed by
the standard secondary eclipse search at phase 0.5.

Algorithm: Phase-folds data, convolves with transit template, measures
primary/secondary/tertiary signal strengths and red noise (Fred).

Key Metrics:
- pri: Primary (transit) signal strength
- sec: Secondary eclipse signal at any phase
- ter: Tertiary signal (third strongest dip)
- pos: Positive (brightening) signal
- Fred: Red noise level = std(convolution) / std(lightcurve)
- false_alarm_threshold: 1-sigma noise floor

Decision Logic:
- passed=False if sec/pri > threshold AND sec > false_alarm_threshold
- confidence degraded when Fred > 2.0 (red noise)
- confidence degraded when n_transits < 5

References:
    [1] Thompson et al. 2018, ApJS 235, 38 (arXiv:1710.06758)
        Section 3.2.3: ModShift in DR25 Robovetter; Table 3 metric summary
    [2] Coughlin et al. 2016, ApJS 224, 12 (arXiv:1601.05413)
        Section 4.3: ModShift false positive identification in DR24
    [3] Santerne et al. 2013, A&A 557, A139 (arXiv:1307.2003)
        Secondary eclipse phase offset for eccentric orbits
    [4] Twicken et al. 2018, PASP 130, 064502 (arXiv:1803.04526)
        DV pipeline architecture context

Novelty: standard (implements established Kepler vetting technique via exovetter)
"""
```

---

## 10. Implementation Checklist

- [ ] Add folded input detection with `is_likely_folded()`
- [ ] Add `inputs_summary` computation
- [ ] Add structured `warnings` list
- [ ] Update confidence mapping with regime-based logic
- [ ] Add `fred_regime` categorization
- [ ] Add `passed_meaning` field
- [ ] Update interpretation text for consistency
- [ ] Add unit tests for all scenarios in test matrix
- [ ] Update module docstrings with citations
- [ ] Verify backward compatibility with existing API consumers
