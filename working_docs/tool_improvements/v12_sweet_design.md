# V12 SWEET Check: Design Document

## 1. Current State

The V12 SWEET (Sine Wave Evaluation for Ephemeris Transits) check detects stellar variability that could masquerade as planetary transits. The current implementation in `bittr_tess_vetter/validation/exovetter_checks.py` wraps the `exovetter` library's `Sweet` vetter class.

### Current Behavior

The check fits sinusoids at three periods:
- **Half-period (P/2)**: Even harmonics of stellar rotation
- **Transit period (P)**: Direct variability at the candidate period
- **Double-period (2P)**: Subharmonic variability

For each period, it computes an **amplitude-to-uncertainty ratio** (`amp_ratio = amplitude / uncertainty`).

### Current Thresholds
- `threshold = 3.0` for the transit period P
- `half_period_threshold = 4.0` for P/2
- `double_period_threshold = 4.0` for 2P

### Pass/Fail Logic
- **Fails** if `period_ratio > threshold` (variability at transit period detected)
- Variability at P/2 or 2P triggers warnings but does **not** cause failure

### Limitations

1. **No input validation**: Does not check for minimum data requirements (baseline, number of transits, cadence)
2. **No SNR context**: Confidence is assigned without knowledge of transit SNR
3. **Harmonic aliasing unhandled**: Stellar variability at P/2 can produce transit-like dips when phase-folded at P, but the check only warns rather than investigating the alias scenario
4. **No `warnings` structure**: Messages are free-text; no machine-readable warnings list
5. **No `inputs_summary`**: Downstream tools cannot assess whether the check had sufficient data
6. **Threshold interpretation unclear**: The 3-sigma threshold is appropriate for well-sampled data but may be too aggressive for sparse baselines


## 2. Problems Identified

From the spec (`v1_spec.md`) and analysis of the current implementation:

### 2.1 False Positives from Stellar Variability at P/2

When a star rotates with period P_rot = 2 * P_transit, the sinusoidal variability folds coherently at P_transit. The current check flags this at P/2 but does not:
- Compute the expected transit-like depth from the P/2 variability
- Compare the observed transit depth to the expected variability amplitude
- Downgrade confidence or fail the check when P/2 variability could explain the transit

### 2.2 False Positives from Variability at 2P

Similarly, variability at 2P (stellar rotation with P_rot = P_transit / 2) can produce:
- Phase-coherent modulation that reinforces every other transit
- Odd/even depth differences that mimic an eclipsing binary

The current implementation warns but does not quantify the impact.

### 2.3 Amplitude Ratio Interpretation

The amp_ratio metric is compared to fixed thresholds without considering:
- **Baseline length**: Short baselines have higher noise floors
- **Number of cycles observed**: Fewer cycles = less reliable fit
- **Transit depth**: Deep transits are harder to mimic with variability

### 2.4 Low-Information Regime

The check produces confident results even when:
- Baseline < 2 * period (cannot reliably fit a sinusoid)
- N_transits < 3 (odd/even aliasing possible)
- Cadence is too coarse relative to the period


## 3. Proposed Improvements

### 3.1 Add Minimum Data Requirements

Before running the sinusoid fit, validate:

```python
baseline_days = time.max() - time.min()
n_cycles_observed = baseline_days / period

if n_cycles_observed < 2.0:
    # Cannot reliably detect variability at this period
    return low_confidence_result(
        passed=True,
        confidence=0.30,
        warnings=["insufficient_baseline"]
    )
```

### 3.2 Compute Transit-Mimicking Amplitude

For each harmonic (P, P/2, 2P), compute the expected transit-like depth if the signal were stellar variability:

```python
# If variability at P/2 with amplitude A folds at period P:
# The expected "transit depth" is approximately 2*A (peak-to-trough)
variability_induced_depth_ppm = 2 * amplitude_at_half_period * 1e6
```

Add a new metric: `variability_explains_depth_fraction`:
```python
variability_explains_depth_fraction = variability_induced_depth_ppm / transit_depth_ppm
```

If this fraction > 0.5, the variability could substantially explain the transit signal.

### 3.3 Revised Pass/Fail Logic

```python
# Fail if:
# 1. Variability at P is significant AND could explain >50% of transit depth
# 2. Variability at P/2 is significant AND folded amplitude explains >50% of depth
# 3. Variability at 2P is significant AND creates odd/even-like modulation

fails_at_period = (
    period_ratio > threshold
    and variability_explains_depth_fraction_at_P > 0.5
)

fails_at_half_period = (
    half_p_ratio > half_p_thresh
    and variability_explains_depth_fraction_at_half_P > 0.5
)

passed = not (fails_at_period or fails_at_half_period)
```

### 3.4 Confidence Scaling

Confidence should degrade based on:

| Condition | Confidence Penalty |
|-----------|-------------------|
| n_cycles_observed < 5 | -0.15 |
| n_transits < 5 | -0.10 |
| SNR < 10 | -0.10 |
| Variability at P/2 > 2-sigma | -0.10 |
| Variability at 2P > 2-sigma | -0.05 |

Base confidence starts at 0.95 for passed checks and 0.90 for failed checks.

### 3.5 Structured Warnings

Add a `warnings` list to the output:

```python
warnings = []
if n_cycles_observed < 3:
    warnings.append("low_baseline_cycles")
if half_p_ratio > 2.0:
    warnings.append("harmonic_variability_detected")
if variability_explains_depth_fraction > 0.3:
    warnings.append("variability_may_explain_transit")
```


## 4. Recommended Defaults

| Parameter | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| `threshold` (P) | 3.0 | 3.5 | Slightly more permissive to reduce false failures |
| `half_period_threshold` | 4.0 | 3.5 | Now causes failure if depth explained |
| `double_period_threshold` | 4.0 | 4.0 | Keep as warning-only trigger |
| `min_cycles_for_fit` | N/A | 2.0 | Minimum baseline/period ratio |
| `variability_depth_threshold` | N/A | 0.5 | Fraction of depth explainable by variability |
| `confidence_floor` | N/A | 0.30 | Minimum confidence for degraded checks |

### Config Structure (Additive)

```python
additional={
    "half_period_threshold": 3.5,
    "double_period_threshold": 4.0,
    "min_cycles_for_fit": 2.0,
    "variability_depth_threshold": 0.5,
    "include_harmonic_analysis": True,  # Gate for new behavior
}
```


## 5. Required Output Fields

All new fields are **additive only**. Existing fields are preserved.

### New Fields in `details`

```python
details = {
    # Existing fields (preserved)
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

    # NEW: Structured warnings list
    "warnings": [
        "low_baseline_cycles",      # baseline < 3 * period
        "harmonic_variability_detected",  # P/2 or 2P signal > 2-sigma
        "variability_may_explain_transit",  # variability could cause >30% of depth
        "insufficient_transits",    # n_transits < 3
        "low_snr",                  # SNR < 7
    ],

    # NEW: Input quality summary
    "inputs_summary": {
        "n_transits": int,          # Number of observed transits
        "n_cycles_observed": float, # Baseline / period
        "baseline_days": float,     # Total time span
        "cadence_minutes": float,   # Median cadence
        "snr": float | None,        # Transit SNR if available
        "n_points": int,            # Total valid data points
    },

    # NEW: Harmonic analysis (gated by include_harmonic_analysis)
    "harmonic_analysis": {
        "variability_induced_depth_at_P_ppm": float,
        "variability_induced_depth_at_half_P_ppm": float,
        "variability_explains_depth_fraction": float,
        "dominant_variability_period": str,  # "P", "P/2", "2P", or "none"
    },
}
```


## 6. Test Matrix

### Synthetic Test Cases

| Test Case | Description | Expected Outcome |
|-----------|-------------|------------------|
| **T1: Clean Planet** | Transit with no stellar variability | `passed=True`, confidence > 0.90 |
| **T2: Variability at P** | Sinusoidal signal at transit period with amp_ratio=5.0 | `passed=False`, `fails_at_period=True` |
| **T3: Variability at P/2** | Rotation at 2*P_transit, amp explains 60% of depth | `passed=False`, `fails_at_half_period=True` |
| **T4: Variability at 2P** | Rotation at P/2, amp_ratio=5.0 | `passed=True`, warning `harmonic_variability_detected` |
| **T5: Low Baseline** | Baseline < 2*period | `passed=True`, confidence=0.30, warning `low_baseline_cycles` |
| **T6: Weak Variability** | amp_ratio=2.5 at P (below threshold) | `passed=True`, no warnings |
| **T7: Strong Variability but Deep Transit** | amp_ratio=4.0 at P, but variability explains only 20% of depth | `passed=True`, confidence reduced |
| **T8: P/2 Alias Scenario** | Variability at P/2 exactly matches transit depth | `passed=False`, `variability_explains_depth_fraction > 0.9` |

### Implementation Approach

```python
def test_v12_clean_planet():
    """T1: Transit signal with no stellar variability passes."""
    time = np.linspace(0, 30, 5000)  # 30 days, 2-min cadence
    flux = inject_transit(time, period=5.0, depth=1000e-6)  # No variability
    # ... run check
    assert result.passed is True
    assert result.confidence > 0.90
    assert "variability_may_explain_transit" not in result.details.get("warnings", [])

def test_v12_variability_at_half_period():
    """T3: Stellar rotation at 2*P_transit flagged when it explains depth."""
    time = np.linspace(0, 60, 10000)
    flux = inject_transit(time, period=5.0, depth=500e-6)
    flux += inject_sinusoid(time, period=10.0, amplitude=400e-6)  # P/2 of folded
    # ... run check
    assert result.passed is False
    assert result.details["fails_at_half_period"] is True
    assert result.details["harmonic_analysis"]["variability_explains_depth_fraction"] > 0.5
```


## 7. Backward Compatibility

### What Changes

| Aspect | Before | After | Breaking? |
|--------|--------|-------|-----------|
| Pass/fail logic | Only fails on P | Also fails on P/2 if depth explained | **Potentially** (gated) |
| Confidence values | Fixed tiers | Scaled by data quality | No (values still 0-1) |
| `details` keys | 11 keys | 14 keys | No (additive) |

### Gating Strategy

New behavior is gated behind `config.additional["include_harmonic_analysis"]`:

```python
# Default: False (preserves existing behavior)
# Opt-in: True (enables new P/2 failure logic)

if additional.get("include_harmonic_analysis", False):
    # Run new harmonic analysis
    # Apply new failure logic for P/2
else:
    # Existing behavior only
```

### Migration Path

1. **v1.x**: Add new fields, gate new failure logic behind flag (default off)
2. **v2.0**: Change default to `include_harmonic_analysis=True`
3. Document the change in release notes with examples of affected cases


## 8. Citations

### Primary References

1. **Thompson et al. 2018** (2018ApJS..235...38T)
   - Section 3.2.4: SWEET test for stellar variability in the DR25 Robovetter
   - Defines the amplitude-to-uncertainty ratio metric
   - ADS: https://ui.adsabs.harvard.edu/abs/2018ApJS..235...38T

2. **Coughlin et al. 2016** (2016ApJS..224...12C)
   - Section 4.4: Original SWEET implementation in the DR24 Robovetter
   - Established the sinusoid fitting approach at P, P/2, and 2P
   - ADS: https://ui.adsabs.harvard.edu/abs/2016ApJS..224...12C

### Supporting References

3. **McQuillan et al. 2014** (2014ApJS..211...24M)
   - Stellar rotation period measurements for Kepler stars
   - Context for expected variability timescales
   - ADS: https://ui.adsabs.harvard.edu/abs/2014ApJS..211...24M

4. **Basri et al. 2013** (2013ApJ...769...37B)
   - Photometric variability amplitudes in Kepler targets
   - Establishes typical variability amplitudes for different stellar types
   - ADS: https://ui.adsabs.harvard.edu/abs/2013ApJ...769...37B

### Code References

```python
# In module docstring:
"""
References:
    [1] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T)
        Section 3.2.4: SWEET test for stellar variability detection
    [2] Coughlin et al. 2016, ApJS 224, 12 (2016ApJS..224...12C)
        Section 4.4: Original SWEET implementation methodology
"""

# In function decorator:
@cites(
    cite(THOMPSON_2018, "Section 3.2.4 SWEET stellar variability test"),
    cite(COUGHLIN_2016, "Section 4.4 original SWEET implementation"),
)
def sweet(...):
    ...
```


## Appendix: Implementation Sketch

```python
class SWEETCheck(VetterCheck):
    """V12: SWEET test for stellar variability masquerading as transits."""

    def run(self, candidate, lightcurve, stellar=None):
        # Validate inputs
        warnings = []
        baseline_days = lightcurve.time.max() - lightcurve.time.min()
        n_cycles = baseline_days / candidate.period

        if n_cycles < self.config.additional.get("min_cycles_for_fit", 2.0):
            warnings.append("low_baseline_cycles")
            return self._low_confidence_result(warnings)

        # Run exovetter SWEET
        metrics = self._run_exovetter_sweet(candidate, lightcurve)

        # Extract amplitude ratios
        period_ratio = metrics["period"]["ratio"]
        half_p_ratio = metrics["half_period"]["ratio"]
        double_p_ratio = metrics["double_period"]["ratio"]

        # Compute harmonic analysis (if enabled)
        harmonic_analysis = {}
        if self.config.additional.get("include_harmonic_analysis", False):
            harmonic_analysis = self._analyze_harmonics(
                metrics, candidate.depth, candidate.period
            )
            if harmonic_analysis["variability_explains_depth_fraction"] > 0.3:
                warnings.append("variability_may_explain_transit")

        # Determine pass/fail with new logic
        passed, fails_at = self._evaluate_pass_fail(
            period_ratio, half_p_ratio, double_p_ratio,
            harmonic_analysis, self.config
        )

        # Build inputs summary
        inputs_summary = {
            "n_transits": self._count_transits(lightcurve, candidate),
            "n_cycles_observed": n_cycles,
            "baseline_days": baseline_days,
            "cadence_minutes": self._median_cadence(lightcurve),
            "snr": candidate.snr if candidate.snr else None,
            "n_points": len(lightcurve.time),
        }

        # Scale confidence
        confidence = self._compute_confidence(passed, warnings, inputs_summary)

        return VetterCheckResult(
            id=self.id,
            name=self.name,
            passed=passed,
            confidence=confidence,
            details={
                # ... existing fields ...
                "warnings": warnings,
                "inputs_summary": inputs_summary,
                "harmonic_analysis": harmonic_analysis,
            },
        )
```
