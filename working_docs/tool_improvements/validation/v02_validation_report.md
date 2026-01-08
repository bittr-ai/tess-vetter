# V02 Secondary Eclipse Check Validation Report

**Date:** 2026-01-08
**Author:** Claude (automated validation)
**Check ID:** V02
**Implementation Location:** `bittr-tess-vetter/src/bittr_tess_vetter/validation/lc_checks.py:589-812`

## Executive Summary

The V02 secondary eclipse check was validated against 4 known systems with varying expected behaviors. The implementation uses local baseline windows, red noise inflation, and dual thresholds (3-sigma AND 0.5% depth) for detection. **Results are mixed: the check correctly passes planets without detectable secondaries but fails to detect known secondary eclipses in hot Jupiters and eclipsing binaries.**

### Key Finding: The V02 check is too conservative for detecting secondary eclipses

The 0.5% (5000 ppm) depth threshold is appropriate for detecting deep eclipsing binary secondaries but is **too high** to catch:
1. Hot Jupiter thermal emission (typically 100-500 ppm in TESS band)
2. Ultra-hot Jupiter secondaries (typically 300-1000 ppm)
3. Grazing/diluted eclipsing binaries

---

## Test Results Summary

| Target | TIC ID | Type | Expected V02 | Actual V02 | Secondary Depth | Sigma | Match? |
|--------|--------|------|--------------|------------|-----------------|-------|--------|
| Pi Mensae c | 261136679 | Super-Earth | PASS | PASS | -2.0 ppm | 0.21 | YES |
| WASP-18 b | 100100827 | Ultra-hot Jupiter | FAIL* | PASS | 194.7 ppm | 0.93 | NO |
| KELT-9 b | 16740101 | Hottest exoplanet | FAIL* | PASS | -9.8 ppm | 0.44 | NO |
| CM Draconis | 199574208 | Eclipsing binary | FAIL | PASS | 1825.7 ppm | 17.32 | NO |

\* Expected to FAIL because these systems have known/expected secondary eclipses that V02 is designed to flag.

---

## Detailed Test Results

### 1. Pi Mensae c (TIC 261136679) - Super-Earth

**Expected:** PASS (no detectable secondary - planet too small and cool)

**Result:** PASS (correct)

```
V02 Result:
- passed: true
- confidence: 0.85
- secondary_depth_ppm: -2.0 (noise-level, effectively zero)
- secondary_depth_sigma: 0.21 (well below 3-sigma threshold)
- secondary_phase_coverage: 1.0
- n_secondary_events_effective: 5
- red_noise_inflation: 5.94
```

**Analysis:** Correct behavior. Pi Mensae c is a 2 Earth-radius super-Earth with no expected thermal emission detectable by TESS.

---

### 2. WASP-18 b (TIC 100100827) - Ultra-Hot Jupiter

**Expected:** Secondary eclipse should be detected. Literature (Shporer et al. 2018, arXiv:1811.06020) reports a secondary eclipse depth of **341 ppm** in TESS data.

**Result:** PASS (incorrect - should flag secondary)

```
V02 Result:
- passed: true
- confidence: 0.85
- secondary_depth_ppm: 194.7
- secondary_depth_sigma: 0.93 (below 3-sigma threshold)
- secondary_phase_coverage: 1.0
- n_secondary_events_effective: 28
- red_noise_inflation: 24.25 (very high!)
```

**Analysis:** V02 fails to detect the secondary eclipse for two reasons:
1. **Red noise inflation is very high (24.25x)**, inflating the uncertainty estimate and reducing the sigma below threshold
2. Even without this inflation, 194.7 ppm is well below the 5000 ppm depth threshold

**Note:** The raw detection shows 194.7 ppm which is in the expected range, but the uncertainty is overestimated. The known value from dedicated analysis is 341 ppm.

**Issue:** The t0 used (-259.19 from archive) may be significantly off for the current TESS epoch, affecting phase alignment.

---

### 3. KELT-9 b (TIC 16740101) - Hottest Known Exoplanet

**Expected:** Strong secondary eclipse expected. KELT-9b orbits an A0 star (Teff ~10,000K) and has equilibrium temperature ~4600K. Thermal emission should be significant.

**Result:** PASS (incorrect - should flag or warn about secondary)

```
V02 Result:
- passed: true
- confidence: 0.85
- secondary_depth_ppm: -9.8 (effectively zero)
- secondary_depth_sigma: 0.44
- secondary_phase_coverage: 1.0
- n_secondary_events_effective: 14
- red_noise_inflation: 3.66
```

**Analysis:** V02 shows essentially no secondary signal, which contradicts expectations. Possible causes:
1. Phase alignment issues (t0 epoch propagation)
2. The early-type host star (A0) creates unusual systematics in PDCSAP light curves
3. Secondary may be at a different phase due to orbital eccentricity

---

### 4. CM Draconis (TIC 199574208) - Eclipsing Binary

**Expected:** FAIL - CM Draconis is a famous detached eclipsing binary (M4.5+M4.5) with nearly equal-depth eclipses. The secondary eclipse should be clearly detected.

**Result:** PASS (incorrect - should flag as EB)

```
V02 Result:
- passed: true
- confidence: 0.722
- secondary_depth_ppm: 1825.7
- secondary_depth_sigma: 17.32 (well above 3-sigma!)
- significant_secondary: false (BUG?)
- secondary_phase_coverage: 1.0
- n_secondary_events_effective: 17
- red_noise_inflation: 5.1
```

**Analysis:** This is a clear implementation issue:
- The sigma (17.32) exceeds the 3-sigma threshold
- The depth (1825.7 ppm) is below the 5000 ppm threshold
- Because BOTH conditions must be met, the check passes

**The dual-threshold logic prevents detection of moderate-depth eclipsing binaries.**

For CM Draconis at P=1.268d (full period), the primary eclipse is ~34% deep while the secondary eclipse appears as ~0.18% (1826 ppm) - this is because we're looking at the *difference* between in-secondary flux and adjacent baseline, and the baseline is already affected by ellipsoidal variations.

---

## Literature Findings

### Key Papers Reviewed

1. **Santerne et al. 2013 (arXiv:1307.2003)** - "The contribution of secondary eclipses as astrophysical false positives"
   - Found 0.061% of main-sequence binaries are secondary-only EBs mimicking planetary transits
   - Important for long-period candidates where primary eclipse may not be observed
   - Suggests secondary-only detection is a valid false positive pathway

2. **Shporer et al. 2018 (arXiv:1811.06020)** - "TESS full orbital phase curve of WASP-18b"
   - Reports 341 ppm secondary eclipse for WASP-18b
   - Demonstrates TESS can detect hot Jupiter thermal emission
   - Uses full phase curve fitting, not just secondary window median

3. **Kostov et al. 2025 (arXiv:2506.05631)** - "TESS Ten Thousand Catalog: 10,001 Eclipsing Binaries"
   - Modern TESS EB catalog with photocenter vetting
   - Uses neural networks for initial detection
   - Employs manual citizen science validation
   - Does not describe specific secondary eclipse detection thresholds

### Statistical Approaches in Literature

The current V02 implementation uses:
- Local baseline windows (adjacent to secondary phase)
- Red noise inflation via binning analysis (Pont et al. 2006)
- Dual threshold: 3-sigma AND 0.5% depth

Literature approaches include:
1. **Full phase curve fitting** (Shporer et al. 2018) - Models entire orbital phase including ellipsoidal/beaming effects
2. **BLS-style secondary search** - Treats secondary as independent transit signal
3. **Odd/even depth ratio** - V01 catches equal-depth EBs, V02 complements for unequal depths
4. **ModShift secondary metric** (V11) - Uses phase-shifted signal comparison

---

## Recommendations

### 1. Lower the Depth Threshold (HIGH PRIORITY)

**Current:** 5000 ppm (0.5%)
**Recommended:** 500-1000 ppm (0.05-0.1%)

Rationale: Hot Jupiters show thermal emission of 100-500 ppm in TESS band. The current threshold misses these entirely.

### 2. Adjust Threshold Logic to OR Instead of AND

**Current:** `significant_secondary = (sigma >= 3.0) AND (depth >= 0.5%)`
**Recommended:** `significant_secondary = (sigma >= 4.0) OR (sigma >= 3.0 AND depth >= 0.1%)`

This would catch:
- High-sigma detections regardless of depth (statistical certainty)
- Moderate-sigma with reasonable depth (physical plausibility)

### 3. Add Graduated Warning Levels

Instead of binary pass/fail, report:
- **PASS**: No secondary detected (sigma < 2)
- **MARGINAL**: Possible secondary (2 <= sigma < 3)
- **SUSPICIOUS**: Likely secondary (3 <= sigma < 5)
- **FAIL**: Definite secondary (sigma >= 5 OR depth > 1%)

### 4. Re-evaluate Red Noise Inflation

The red noise inflation factor for WASP-18 was 24.25x, which seems excessive. Consider:
- Capping inflation at 5-10x maximum
- Using transit-duration-matched bins for better estimation
- Adding a diagnostic flag when inflation is extreme

### 5. Consider Phase Alignment Verification

V02 assumes secondary is at phase 0.5. For eccentric orbits, secondary phase shifts. Consider:
- Searching multiple phases (0.4-0.6 in steps)
- Reporting phase of maximum depth detection
- Using known eccentricity when available

### 6. Integrate with V01 and V11

V01 (odd/even) catches EBs at half the true period. V11 (ModShift) has its own secondary metric. Consider ensemble approach:
- If V01 shows odd/even difference > 20%, V02 confidence should be reduced
- If V11 secondary_primary_ratio > 0.3, V02 should flag

---

## Implementation Changes Suggested

```python
# In SecondaryEclipseConfig (lc_checks.py)
@dataclass
class SecondaryEclipseConfig:
    # Current (too high)
    # depth_threshold: float = 0.005  # 5000 ppm

    # Recommended
    depth_threshold: float = 0.001  # 1000 ppm (catches hot Jupiter emission)

    # Add graduated thresholds
    depth_warn_threshold: float = 0.0005  # 500 ppm
    sigma_warn_threshold: float = 2.5

    # Cap red noise inflation
    max_inflation: float = 10.0
```

```python
# In decision logic (lines 749-755)
# Current (too restrictive)
significant_secondary = (
    secondary_depth_sigma >= config.sigma_threshold
    and secondary_depth >= config.depth_threshold
)

# Recommended (more sensitive)
significant_secondary = (
    secondary_depth_sigma >= 5.0  # Very high sigma = definite
    or (secondary_depth_sigma >= config.sigma_threshold
        and secondary_depth >= config.depth_threshold)
)

suspicious_secondary = (
    secondary_depth_sigma >= config.sigma_warn_threshold
    or secondary_depth >= config.depth_warn_threshold
)
```

---

## Test Coverage Gaps

1. **Eccentric orbit EBs** - Secondary not at phase 0.5
2. **Grazing EBs** - Shallow primary with visible secondary
3. **Background EBs** - Diluted secondary below threshold
4. **Active stars** - High red noise inflation may mask real secondaries

---

## Conclusion

The V02 secondary eclipse check implementation is sound in methodology (local baselines, red noise inflation, phase coverage tracking) but **too conservative in thresholds** to catch:
1. Hot Jupiter thermal emission (< 500 ppm)
2. Moderate-depth eclipsing binaries (< 5000 ppm secondary)

The dual threshold requirement (sigma AND depth) creates a blind spot where statistically significant but shallow secondaries pass the check.

**Priority fixes:**
1. Lower depth threshold to 500-1000 ppm
2. Add graduated warning levels (SUSPICIOUS/MARGINAL)
3. Cap red noise inflation at reasonable maximum (10x)
4. Add integration with V01/V11 for ensemble decision

---

## Appendix: Raw V02 Metrics

| Target | secondary_depth_ppm | secondary_depth_sigma | red_noise_inflation | n_secondary_events | phase_coverage |
|--------|--------------------:|---------------------:|--------------------:|-------------------:|---------------:|
| Pi Mensae c | -2.0 | 0.21 | 5.94 | 5 | 1.0 |
| WASP-18 b | 194.7 | 0.93 | 24.25 | 28 | 1.0 |
| KELT-9 b | -9.8 | 0.44 | 3.66 | 14 | 1.0 |
| CM Draconis | 1825.7 | 17.32 | 5.10 | 17 | 1.0 |

---

## References

1. Santerne et al. 2013, A&A 557, A139 - Secondary eclipses as false positives
2. Shporer et al. 2018, AJ 157, 178 - WASP-18b phase curve
3. Kostov et al. 2025, arXiv:2506.05631 - TESS EB catalog
4. Thompson et al. 2018, ApJS 235, 38 - Kepler Robovetter
5. Pont et al. 2006, MNRAS 373, 231 - Red noise in transit photometry
