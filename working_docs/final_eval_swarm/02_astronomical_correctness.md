# Astronomical Correctness Review

**Reviewer:** Claude Opus 4.5
**Date:** 2026-01-14
**Scope:** `src/bittr_tess_vetter/compute/`, `transit/`, `validation/`

---

## Executive Summary

The bittr-tess-vetter codebase demonstrates **strong scientific rigor** in its astronomical algorithms. The implementation properly handles BTJD time formats, ppm depth units, and correctly implements standard transit detection and vetting techniques from the literature. Physical constants are appropriately sourced, and the vetting checks align well with established methodologies (Kepler Robovetter, SPOC DV).

**Overall Assessment: PASS with minor recommendations**

---

## 1. Time Systems and Units

### 1.1 BTJD Handling

**Status: CORRECT**

The codebase consistently uses BTJD (Barycentric TESS Julian Date) as documented:
- All time arrays are expected in BTJD (days)
- Phase folding correctly uses `((time - t0) / period) % 1.0`
- Transit mask calculations properly convert duration from hours to days

```python
# From compute/transit.py - correct phase calculation
phase = ((time - t0) / period + 0.5) % 1.0 - 0.5
```

### 1.2 Depth Units (ppm)

**Status: CORRECT**

Transit depths are consistently handled in fractional units internally and converted to ppm for output:
- Internal calculations use fractional depth (e.g., 0.001 = 1000 ppm)
- Output always includes `_ppm` suffix for clarity
- Conversion factor `1e6` applied correctly

```python
# From validation/lc_checks.py
depth_odd_ppm = median_odd * 1e6
depth_even_ppm = median_even * 1e6
```

### 1.3 Duration Units

**Status: CORRECT**

Duration consistently expressed in hours for API, converted internally:
```python
# Standard pattern throughout
duration_days = duration_hours / 24.0
```

---

## 2. Transit Detection Algorithms

### 2.1 Box Least Squares (BLS-like Search)

**Status: SCIENTIFICALLY SOUND**

The `compute/bls_like_search.py` implements a phase-binning BLS-like algorithm:

**Strengths:**
- Proper phase binning with circular rolling mean for transit detection
- Weighted least squares fitting when flux errors provided
- Local refinement around peak for improved precision
- Chi-squared based scoring normalized by sqrt(N_transits)

**Literature alignment:** Follows Kovacs et al. (2002) BLS methodology with phase-binning approach.

### 2.2 Transit Least Squares (TLS) Integration

**Status: EXCELLENT**

The `compute/periodogram.py` properly integrates the `transitleastsquares` library (Hippke & Heller 2019):

**Strengths:**
- Uses physical transit model instead of box
- Proper stellar parameter integration for better sensitivity
- Multi-sector handling with per-sector search to avoid period aliases
- Built-in odd/even depth comparison and FAP estimation

```python
# From compute/periodogram.py - correct TLS configuration
power_kwargs = {
    "period_min": period_min,
    "period_max": period_max,
    "R_star": r_star,
    "M_star": m_star,
    ...
}
```

### 2.3 SNR Calculation

**Status: CORRECT**

SNR is calculated as:
```python
snr = depth * np.sqrt(total_in_transit) / scatter
```

This follows the standard formula where SNR scales with sqrt(N_in_transit), correctly accounting for averaging of in-transit points.

---

## 3. Physical Transit Modeling

### 3.1 Batman Model Integration

**Status: EXCELLENT**

The `transit/batman_model.py` correctly implements:

**Limb darkening:**
- Uses quadratic limb darkening law via ldtk (Parviainen & Aigrain 2015)
- TESS bandpass correctly applied
- Fallback coefficients from Claret 2017 for solar-like stars

**Exposure time handling:**
- Auto-detection of cadence from time array
- Supersampling for long-cadence data (>1 minute)

```python
# Correct supersampling decision
cadence_minutes = exp_time * 24 * 60
supersample = 3 if cadence_minutes > 1.0 else 1
```

### 3.2 Quick Parameter Estimation

**Status: SCIENTIFICALLY SOUND**

Based on Seager & Mallen-Ornelas (2003) analytic relations:

```python
# From transit/batman_model.py
rp_rs = np.sqrt(depth_ppm / 1e6)  # Correct for zero limb darkening
a_rs_from_density = (G * rho_star * P^2 / (3*pi))^(1/3)  # Kepler's 3rd law
```

### 3.3 Derived Parameters

**Status: CORRECT**

Stellar density calculation from a/Rs and period:
```python
# Kepler's 3rd law inversion - correct formula
stellar_density_gcc = (3 * np.pi / (G * P^2)) * (a/Rs)^3
```

Impact parameter calculation:
```python
impact_parameter = a_rs * np.cos(inc_rad)  # Correct definition
```

---

## 4. Vetting Checks Alignment with Literature

### 4.1 V01: Odd/Even Depth Check

**Status: EXCELLENT**

**Purpose:** Detect eclipsing binaries at 2x the true period

**Implementation:**
- Per-epoch depth measurement with local baselines
- Red noise inflation following Pont et al. (2006)
- MAD-based robust uncertainty estimation

**Literature alignment:** Matches Kepler Robovetter Thompson et al. (2018) methodology.

```python
# Correct epoch definition to avoid boundary issues
epoch = np.floor((time - t0 + period / 2) / period).astype(int)
```

### 4.2 V02: Secondary Eclipse Search

**Status: EXCELLENT**

**Implementation:**
- Search window 0.35-0.65 phase (wider than 0.5 to catch eccentric orbits)
- Local baseline adjacent to secondary window
- Red noise inflation applied

**Literature alignment:**
- Follows Coughlin & Lopez-Morales (2012) methodology
- Widened window per Santerne et al. (2013) for eccentric EBs

### 4.3 V03: Duration Consistency

**Status: CORRECT**

Uses standard duration scaling:
```python
expected_duration_solar = 13.0 * (period_years ** (1.0 / 3.0))
# With stellar density correction:
expected_duration_hours = expected_duration_solar * (rho_star ** (-1.0 / 3.0))
```

This correctly captures T_dur ~ P^(1/3) / rho_star^(1/3) scaling.

### 4.4 V04: Depth Stability

**Status: CORRECT**

- Per-transit depth extraction with local baselines
- Chi-squared consistency test
- MAD-based outlier detection (4-sigma threshold)

**Literature alignment:** Follows Wang & Espinoza (2023) per-transit fitting approach.

### 4.5 V05: V-Shape (Transit Shape) Check

**Status: EXCELLENT**

Uses tF/tT ratio (flat-bottom to total duration):

**Literature references correctly cited:**
- Seager & Mallen-Ornelas (2003) - tF/tT shape parameter
- Kipping (2010) - T14/T23 definitions
- Thompson et al. (2018) - Robovetter V-shape metric
- Prsa et al. (2011) - EB morphology classification

```python
# Grid search over tF/tT ratios from 0 (V-shape) to 1 (box)
tflat_ttotal_ratios = np.linspace(0, 1, n_grid)
```

---

## 5. Physical Constants

### 5.1 Radius Conversions

**Status: CORRECT**

From `validation/stellar_dilution.py`:
```python
R_EARTH_TO_RSUN = 0.009167  # Earth radius in solar radii
R_JUP_TO_RSUN = 0.10045     # Jupiter radius in solar radii
```

**Verification:**
- R_Earth/R_Sun = 6378 km / 695700 km = 0.00917 (correct)
- R_Jup/R_Sun = 69911 km / 695700 km = 0.1005 (correct)

### 5.2 Gravitational Constant

Used in CGS units consistently:
```python
grav_const = 6.674e-8  # cm^3 g^-1 s^-2 (CGS)
```

### 5.3 MAD-to-Sigma Conversion

**Status: CORRECT**
```python
# MAD * 1.4826 = sigma for normal distribution
sigma = mad * 1.4826
```

The factor 1.4826 = 1/Phi^-1(3/4) is the correct scaling.

---

## 6. Model Competition

### 6.1 Known Artifact Periods

**Status: CORRECT**

```python
KNOWN_ARTIFACT_PERIODS = [
    13.7,  # TESS orbital period
    27.4,  # ~2x orbital
    1.0,   # daily systematics
    0.5,   # half-day
    6.85,  # ~0.5x orbital
    41.1,  # ~3x orbital
]
```

These match documented TESS systematic periods.

### 6.2 Information Criteria

**Status: CORRECT**

```python
aic = -2 * log_likelihood + 2 * n_params
bic = -2 * log_likelihood + n_params * np.log(n_points)
```

Standard AIC/BIC formulas correctly implemented.

---

## 7. TTV Analysis

### 7.1 Transit Timing Measurement

**Status: SCIENTIFICALLY SOUND**

- Trapezoid model fitting per transit
- Chi-squared curvature for timing uncertainty
- O-C residual calculation correct

### 7.2 TTV Periodicity Detection

Uses Lomb-Scargle on O-C residuals with linear trend removal:
```python
# Remove linear trend first (period drift)
coeffs = np.polyfit(epochs, o_minus_c, 1)
detrended = o_minus_c - np.polyval(coeffs, epochs)
# Then search for periodic signal
```

---

## 8. Pixel-Level Checks

### 8.1 Centroid Shift (V08)

**Status: CORRECT**

- Flux-weighted centroid calculation
- Bootstrap uncertainty estimation
- Proper arcsec conversion (21 arcsec/pixel for TESS)

### 8.2 Aperture Dependence (V10)

**Status: CORRECT**

- Multi-aperture depth measurement
- Background annulus estimation
- Stability metric for true on-target signals

---

## 9. Dilution and Companion Physics

### 9.1 Flux Fraction Calculation

**Status: CORRECT**

```python
# Magnitude to flux conversion
target_flux = 10 ** ((ref_mag - target_mag) / 2.5)
```

### 9.2 Planet Radius Limits

**Status: APPROPRIATE**

```python
PLANET_MAX_RADIUS_RJUP = 2.0   # Above this = likely not a planet
STELLAR_MIN_RADIUS_RSUN = 0.2  # Above this = clearly stellar
```

These thresholds align with known exoplanet population limits.

---

## 10. Recommendations

### 10.1 Minor Improvements

1. **Document BTJD epoch:** Add explicit note that BTJD = BJD - 2457000 for reference.

2. **Eccentricity handling:** The batman model fixes `ecc=0.0` and `w=90.0`. Consider documenting this circular orbit assumption or adding eccentric orbit support.

3. **Limb darkening uncertainty propagation:** Currently LD coefficients are fixed after initial estimation. Consider propagating LD uncertainty into depth errors.

### 10.2 Consider Adding

1. **Grazing transit detection:** Explicit flag for b > 1 - Rp/Rs (grazing geometry).

2. **Stellar density prior:** Compare derived density from transit fit with TIC catalog value as consistency check.

---

## 11. Conclusion

The astronomical algorithms in bittr-tess-vetter are **scientifically sound and well-implemented**:

- **Time/unit handling:** Consistent BTJD and ppm conventions
- **Physical modeling:** Correct batman integration with limb darkening
- **Vetting checks:** Align with Kepler Robovetter and SPOC DV literature
- **Constants:** All physical constants verified correct
- **Statistical methods:** Appropriate use of robust estimators (MAD, bootstrap)

The codebase is suitable for open-source release from an astronomical correctness perspective.

---

## References

1. Kovacs, G., Zucker, S., & Mazeh, T. (2002). BLS algorithm. A&A, 391, 369
2. Hippke, M., & Heller, R. (2019). TLS. A&A, 623, A39
3. Kreidberg, L. (2015). batman. PASP, 127, 1161
4. Parviainen, H., & Aigrain, S. (2015). ldtk. MNRAS, 453, 3821
5. Thompson, S. E., et al. (2018). Kepler DR25 Robovetter. ApJS, 235, 38
6. Seager, S., & Mallen-Ornelas, G. (2003). Transit parameters. ApJ, 585, 1038
7. Pont, F., et al. (2006). Red noise in transit photometry. MNRAS, 373, 231
8. Claret, A. (2017). Limb darkening coefficients. A&A, 600, A30
9. Coughlin, J. L., & Lopez-Morales, M. (2012). Secondary eclipse. AJ, 143, 39
10. Kipping, D. M. (2010). Transit duration. MNRAS, 407, 301
