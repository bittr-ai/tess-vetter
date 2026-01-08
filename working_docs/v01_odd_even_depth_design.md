# Design Note: Improved Odd/Even Depth Check (V01)

## 1. Executive Summary

The current `check_odd_even_depth()` implementation uses point-based median estimates with a simplified error model. This design note proposes improvements aligned with established vetting practices from the Kepler Robovetter (Thompson et al. 2018) and LEO-Vetter (Kunimoto et al. 2025).

**Key improvements:**
- Per-epoch depth fitting instead of point-based medians
- Multiple complementary statistics (box, trapezoid, transit model)
- Robust uncertainty estimation with pink noise model
- Minimum transit count requirements
- Extended diagnostics for downstream interpretation

---

## 2. Current Implementation Issues

| Issue | Impact | Severity |
|-------|--------|----------|
| Point-based median depth | Sensitive to outliers, doesn't account for transit shape | Medium |
| Simplified error model (`std/sqrt(N)`) | Underestimates uncertainty for correlated noise | High |
| Single 3σ threshold | No handling of low-N edge cases | Medium |
| No minimum transit count per epoch | Undefined behavior for sparse data | High |
| Missing epoch-level diagnostics | Downstream tools can't assess reliability | Medium |

---

## 3. Recommended Statistics

### 3.1 Primary Statistic: Per-Epoch Weighted Mean Depth (OEbox)

Following LEO-Vetter §4.4 (Kunimoto et al. 2025):

```
OE_box = |δ_odd - δ_even| / sqrt(σ²_odd + σ²_even)
```

Where:
- `δ_odd`, `δ_even` = weighted mean depth of odd/even transits
- `σ_odd`, `σ_even` = pink noise uncertainty (see §3.4)

**Rationale**: The weighted mean properly handles mixed cadences and non-uniform uncertainties common in TESS data.

### 3.2 Secondary Statistic: Relative Depth Difference

```
rel_diff = |δ_odd - δ_even| / max(δ_odd, δ_even)
```

**Rationale**: Catches cases where both depths are small but significantly different (low-SNR edge case where sigma test is unreliable).

### 3.3 Tertiary Statistic (Optional): Per-Transit Fitted Depths

For higher-fidelity checks when compute budget allows:

- Fit trapezoid model to odd transits (fix period, vary depth + t0)
- Fit trapezoid model to even transits (fix period, vary depth + t0)
- Compare fitted depths with propagated uncertainties

**Rationale**: LEO-Vetter uses this as `OE_trap` to handle non-box-shaped transits.

### 3.4 Uncertainty Estimation: Pink Noise Model

Following Pont et al. (2006) and LEO-Vetter §2.1:

```python
σ_tr = sqrt(σ²_w / n + σ²_r / N_tr)
```

Where:
- `σ_w` = white noise (weighted std of out-of-transit flux)
- `σ_r` = red noise (from binning analysis, set to 0 if negative)
- `n` = number of in-transit points
- `N_tr` = number of transits

**Implementation**:
```python
def estimate_pink_noise_depth_error(
    flux: np.ndarray,
    flux_err: np.ndarray,
    in_transit_mask: np.ndarray,
    duration_days: float,
    n_transits: int,
) -> float:
    """Estimate depth uncertainty accounting for red noise."""
    oot_flux = flux[~in_transit_mask]

    # White noise from weighted std
    sigma_w = np.sqrt(np.average((oot_flux - np.average(oot_flux))**2))

    # Red noise from binning (Hartman & Bakos 2016)
    bin_size = int(duration_days * 24 * 2)  # ~2 points per duration
    if bin_size > 1 and len(oot_flux) > bin_size:
        binned = np.array([oot_flux[i:i+bin_size].mean()
                          for i in range(0, len(oot_flux)-bin_size, bin_size)])
        sigma_bin = np.std(binned)
        sigma_bin_expected = sigma_w / np.sqrt(bin_size)
        sigma_r_sq = sigma_bin**2 - sigma_bin_expected**2
        sigma_r = np.sqrt(max(0, sigma_r_sq))
    else:
        sigma_r = 0.0

    n_in_transit = in_transit_mask.sum()
    return np.sqrt(sigma_w**2 / n_in_transit + sigma_r**2 / n_transits)
```

---

## 4. Proposed Thresholds

### 4.1 Pass/Fail Thresholds

| Condition | Threshold | Source |
|-----------|-----------|--------|
| Sigma threshold | `OE_box < 3.0` | LEO-Vetter §4.4 |
| Relative depth threshold | `rel_diff < 0.5` | Conservative: 50% difference is EB-like |
| Minimum transits per epoch | `N_odd >= 2 AND N_even >= 2` | Required for meaningful comparison |
| Minimum points per epoch | `n_odd >= 5 AND n_even >= 5` | Current implementation (reasonable) |

### 4.2 Confidence Scaling with N

```python
def compute_confidence(n_odd: int, n_even: int, oe_sigma: float) -> float:
    """Confidence scales with data quantity and margin from threshold."""
    n_min = min(n_odd, n_even)

    # Base confidence from number of transits
    if n_min < 2:
        base = 0.2  # Very low - insufficient data
    elif n_min < 4:
        base = 0.5  # Low - marginal data
    elif n_min < 8:
        base = 0.7  # Moderate
    else:
        base = 0.85  # Good

    # Adjust for proximity to threshold
    if oe_sigma > 2.5:
        base *= 0.7  # Marginal - close to threshold
    elif oe_sigma < 1.0:
        base = min(0.95, base + 0.1)  # Strong pass

    return round(base, 3)
```

### 4.3 Low-N Handling

When `N_odd < 2` or `N_even < 2`:
- Return `passed=True` (cannot reject with insufficient data)
- Set `confidence=0.2` (very low)
- Add warning: `"insufficient_transits_for_odd_even"`

---

## 5. Handling Edge Cases

### 5.1 Missing Epochs

If all transits are odd or all are even (can happen with short baselines):
- Return `passed=True`, `confidence=0.1`
- Add `"single_parity_epochs"` warning

### 5.2 Outliers / Quality Flags

**Before computing depths:**
1. Apply `lightcurve.valid_mask` (already done)
2. Optionally: 3σ clip in-transit points per epoch
3. Weight by inverse variance if `flux_err` available

### 5.3 Grazing Transits

Grazing transits have shallower, V-shaped profiles. The box-based method may underestimate depth.

**Mitigation**: Use depth from overall transit fit if available, not raw median. The `rel_diff` threshold catches cases where depths are shallow but significantly different.

### 5.4 Strong Systematics

If out-of-transit baseline varies significantly:
- Per-transit baseline estimation (local OOT window)
- Use `depth = 1 - in_transit_median / local_baseline`

---

## 6. Synthetic Test Matrix

| Scenario | N_odd | N_even | depth_odd (ppm) | depth_even (ppm) | Expected Result | Rationale |
|----------|-------|--------|-----------------|------------------|-----------------|-----------|
| True planet (nominal) | 5 | 5 | 1000 ± 50 | 1000 ± 50 | PASS, conf=0.85 | Consistent depths |
| True planet (low-N) | 2 | 2 | 1000 ± 100 | 1000 ± 100 | PASS, conf=0.5 | Consistent but uncertain |
| EB at 2× period | 5 | 5 | 5000 ± 100 | 2000 ± 100 | FAIL, OE>10σ | Primary/secondary different |
| EB at 2× period (shallow) | 5 | 5 | 1500 ± 100 | 800 ± 100 | FAIL, OE~5σ | Still detectable |
| Grazing transit | 3 | 3 | 200 ± 50 | 200 ± 50 | PASS, conf=0.6 | Shallow but consistent |
| Strong systematics | 4 | 4 | 1000 ± 200 | 1200 ± 200 | PASS (marginal) | Within noise |
| Sparse data | 1 | 2 | 1000 | 1000 | PASS, conf=0.2, warn | Insufficient for test |
| Single parity | 5 | 0 | 1000 | — | PASS, conf=0.1, warn | Cannot test |

### Test Implementation (pytest)

```python
@pytest.fixture
def make_synthetic_lc():
    """Factory for synthetic light curves."""
    def _make(n_transits, depth_ppm, period=5.0, noise_ppm=100, odd_even_ratio=1.0):
        # Generate time array spanning multiple periods
        t = np.linspace(0, period * n_transits * 1.5, 10000)
        flux = np.ones_like(t)
        flux_err = np.full_like(t, noise_ppm * 1e-6)

        t0 = period / 2
        duration_hours = 3.0
        duration_days = duration_hours / 24

        for i in range(n_transits):
            transit_center = t0 + i * period
            in_transit = np.abs(t - transit_center) < duration_days / 2

            # Apply odd/even depth difference
            if i % 2 == 0:
                depth = depth_ppm * 1e-6
            else:
                depth = depth_ppm * 1e-6 * odd_even_ratio

            flux[in_transit] -= depth

        # Add noise
        flux += np.random.normal(0, noise_ppm * 1e-6, len(flux))

        return LightCurveData(time=t, flux=flux, flux_err=flux_err)
    return _make

def test_true_planet_passes(make_synthetic_lc):
    """Consistent depths should pass."""
    lc = make_synthetic_lc(n_transits=10, depth_ppm=1000, odd_even_ratio=1.0)
    result = check_odd_even_depth(lc, period=5.0, t0=2.5, duration_hours=3.0)
    assert result.passed
    assert result.confidence >= 0.7

def test_eb_at_2x_period_fails(make_synthetic_lc):
    """Primary/secondary depth difference should fail."""
    lc = make_synthetic_lc(n_transits=10, depth_ppm=5000, odd_even_ratio=0.4)
    result = check_odd_even_depth(lc, period=5.0, t0=2.5, duration_hours=3.0)
    assert not result.passed
    assert result.details["depth_diff_sigma"] > 3.0

def test_sparse_data_returns_warning(make_synthetic_lc):
    """Insufficient transits should pass with warning."""
    lc = make_synthetic_lc(n_transits=3, depth_ppm=1000, odd_even_ratio=1.0)
    result = check_odd_even_depth(lc, period=5.0, t0=2.5, duration_hours=3.0)
    assert result.passed
    assert result.confidence < 0.5
    assert "insufficient" in result.details.get("note", "").lower()
```

---

## 7. Output Schema (Extended Diagnostics)

```python
@dataclass
class OddEvenDetails:
    # Core metrics
    n_transits_odd: int
    n_transits_even: int
    n_points_odd: int
    n_points_even: int

    # Depth measurements (ppm)
    depth_odd_ppm: float
    depth_even_ppm: float
    depth_err_odd_ppm: float
    depth_err_even_ppm: float

    # Primary statistic
    delta_ppm: float  # |depth_odd - depth_even|
    delta_sigma: float  # significance of difference

    # Secondary statistic
    rel_diff: float  # delta_ppm / max(depth_odd, depth_even)

    # Diagnostics
    baseline_flux: float
    sigma_white: float | None
    sigma_red: float | None

    # Warnings
    warnings: list[str]  # e.g., ["insufficient_transits", "high_red_noise"]
```

**VetterCheckResult.details mapping:**
```python
details = {
    "n_transits_odd": 5,
    "n_transits_even": 5,
    "n_points_odd": 47,
    "n_points_even": 52,
    "depth_odd_ppm": 1023.4,
    "depth_even_ppm": 987.2,
    "depth_err_odd_ppm": 45.2,
    "depth_err_even_ppm": 42.8,
    "delta_ppm": 36.2,
    "delta_sigma": 0.58,
    "rel_diff": 0.035,
    "baseline_flux": 1.0,
    "sigma_white_ppm": 89.3,
    "sigma_red_ppm": 12.1,
    "warnings": [],
}
```

---

## 8. Code Patch Suggestion

```python
def check_odd_even_depth(
    lightcurve: LightCurveData,
    period: float,
    t0: float,
    duration_hours: float,
) -> VetterCheckResult:
    """V01: Compare depth of odd vs even transits.

    Detects eclipsing binaries masquerading as planets at 2x the true period.
    If odd and even depths differ significantly, likely an EB.

    Uses the OE_box statistic from LEO-Vetter (Kunimoto et al. 2025):
        OE_box = |δ_odd - δ_even| / sqrt(σ²_odd + σ²_even)

    References:
        - Thompson et al. 2018, ApJS 235, 38 (Kepler DR25 Robovetter)
        - Kunimoto et al. 2025, arXiv:2509.10619 (LEO-Vetter §4.4)

    Args:
        lightcurve: Light curve data with time, flux, valid_mask
        period: Orbital period in days
        t0: Reference epoch (BTJD)
        duration_hours: Transit duration in hours

    Returns:
        VetterCheckResult with pass if depths are consistent (OE_box < 3)
    """
    time = lightcurve.time[lightcurve.valid_mask]
    flux = lightcurve.flux[lightcurve.valid_mask]
    flux_err = getattr(lightcurve, 'flux_err', None)
    if flux_err is not None:
        flux_err = flux_err[lightcurve.valid_mask]

    duration_days = duration_hours / 24.0
    half_dur = duration_days / 2

    # Calculate transit number for each point
    transit_num = np.floor((time - t0 + half_dur) / period).astype(int)
    phase = ((time - t0) / period) % 1

    # In-transit mask (handle phase wrapping)
    phase_from_center = np.abs(phase - 0.5) if np.median(phase) > 0.5 else phase
    # More robust: distance from nearest transit center
    phase_dist = np.minimum(phase, 1 - phase)
    in_transit = phase_dist < (duration_days / period / 2)

    # Separate odd and even transits
    odd_mask = in_transit & (transit_num % 2 == 1)
    even_mask = in_transit & (transit_num % 2 == 0)

    # Count unique transits per parity
    n_transits_odd = len(np.unique(transit_num[odd_mask]))
    n_transits_even = len(np.unique(transit_num[even_mask]))
    n_points_odd = odd_mask.sum()
    n_points_even = even_mask.sum()

    warnings = []

    # Minimum requirements
    MIN_TRANSITS = 2
    MIN_POINTS = 5

    if n_transits_odd < MIN_TRANSITS or n_transits_even < MIN_TRANSITS:
        warnings.append("insufficient_transits_for_odd_even")
        return VetterCheckResult(
            id="V01",
            name="odd_even_depth",
            passed=True,
            confidence=0.2,
            details={
                "n_transits_odd": n_transits_odd,
                "n_transits_even": n_transits_even,
                "n_points_odd": n_points_odd,
                "n_points_even": n_points_even,
                "note": "Insufficient transits for odd/even comparison (need ≥2 each)",
                "warnings": warnings,
            },
        )

    if n_points_odd < MIN_POINTS or n_points_even < MIN_POINTS:
        warnings.append("insufficient_points_for_odd_even")
        return VetterCheckResult(
            id="V01",
            name="odd_even_depth",
            passed=True,
            confidence=0.3,
            details={
                "n_transits_odd": n_transits_odd,
                "n_transits_even": n_transits_even,
                "n_points_odd": n_points_odd,
                "n_points_even": n_points_even,
                "note": "Insufficient in-transit points for odd/even comparison",
                "warnings": warnings,
            },
        )

    # Compute baseline from out-of-transit flux
    out_of_transit = ~in_transit
    if out_of_transit.sum() < 10:
        baseline = 1.0
        warnings.append("sparse_out_of_transit_baseline")
    else:
        baseline = np.median(flux[out_of_transit])

    # Compute weighted mean depths
    odd_flux = flux[odd_mask]
    even_flux = flux[even_mask]

    depth_odd = 1.0 - np.mean(odd_flux) / baseline
    depth_even = 1.0 - np.mean(even_flux) / baseline

    # Estimate uncertainties using pink noise model
    def estimate_depth_error(in_mask: np.ndarray, n_tr: int) -> float:
        """Estimate depth uncertainty with pink noise."""
        oot_flux = flux[out_of_transit]
        n_in = in_mask.sum()

        if len(oot_flux) < 20:
            # Fallback to simple std/sqrt(n)
            return np.std(flux[in_mask]) / np.sqrt(n_in) / baseline

        sigma_w = np.std(oot_flux)

        # Red noise from binning
        bin_size = max(2, int(duration_days / (np.median(np.diff(time)) + 1e-6)))
        if bin_size > 1 and len(oot_flux) > 2 * bin_size:
            n_bins = len(oot_flux) // bin_size
            binned = np.array([oot_flux[i*bin_size:(i+1)*bin_size].mean()
                              for i in range(n_bins)])
            sigma_bin = np.std(binned)
            sigma_bin_xpt = sigma_w / np.sqrt(bin_size)
            sigma_r_sq = sigma_bin**2 - sigma_bin_xpt**2
            sigma_r = np.sqrt(max(0, sigma_r_sq))
        else:
            sigma_r = 0.0

        # Pink noise formula (Pont et al. 2006)
        sigma_depth = np.sqrt(sigma_w**2 / n_in + sigma_r**2 / max(1, n_tr))
        return sigma_depth / baseline

    err_odd = estimate_depth_error(odd_mask, n_transits_odd)
    err_even = estimate_depth_error(even_mask, n_transits_even)

    # Primary statistic: OE_box (Kunimoto et al. 2025)
    combined_err = np.sqrt(err_odd**2 + err_even**2)
    delta_depth = abs(depth_odd - depth_even)
    oe_sigma = delta_depth / combined_err if combined_err > 0 else 0.0

    # Secondary statistic: relative difference
    max_depth = max(abs(depth_odd), abs(depth_even))
    rel_diff = delta_depth / max_depth if max_depth > 0 else 0.0

    # Thresholds (LEO-Vetter: OE_box > 3)
    SIGMA_THRESHOLD = 3.0
    REL_DIFF_THRESHOLD = 0.5  # 50% relative difference is suspicious

    passed = oe_sigma < SIGMA_THRESHOLD and rel_diff < REL_DIFF_THRESHOLD

    # Confidence based on data quality
    n_min = min(n_transits_odd, n_transits_even)
    if n_min >= 8:
        confidence = 0.85
    elif n_min >= 4:
        confidence = 0.7
    elif n_min >= 2:
        confidence = 0.5
    else:
        confidence = 0.3

    # Adjust for proximity to threshold
    if oe_sigma > 2.5:
        confidence *= 0.7
    elif oe_sigma < 1.0 and passed:
        confidence = min(0.95, confidence + 0.1)

    # Convert to ppm for human readability
    depth_odd_ppm = depth_odd * 1e6
    depth_even_ppm = depth_even * 1e6
    err_odd_ppm = err_odd * 1e6
    err_even_ppm = err_even * 1e6
    delta_ppm = delta_depth * 1e6

    return VetterCheckResult(
        id="V01",
        name="odd_even_depth",
        passed=passed,
        confidence=round(confidence, 3),
        details={
            # Transit counts
            "n_transits_odd": n_transits_odd,
            "n_transits_even": n_transits_even,
            "n_points_odd": n_points_odd,
            "n_points_even": n_points_even,
            # Depth measurements
            "depth_odd_ppm": round(depth_odd_ppm, 2),
            "depth_even_ppm": round(depth_even_ppm, 2),
            "depth_err_odd_ppm": round(err_odd_ppm, 2),
            "depth_err_even_ppm": round(err_even_ppm, 2),
            # Primary/secondary statistics
            "delta_ppm": round(delta_ppm, 2),
            "delta_sigma": round(oe_sigma, 2),
            "rel_diff": round(rel_diff, 4),
            # Diagnostics
            "baseline_flux": round(baseline, 6),
            # Warnings
            "warnings": warnings,
        },
    )
```

---

## 9. Migration Notes

### API Stability
- Function signature unchanged
- Return type unchanged (`VetterCheckResult`)
- `details` dict extended (additive, backwards compatible)

### Breaking Changes
- `odd_depth` → `depth_odd_ppm` (renamed + units)
- `even_depth` → `depth_even_ppm` (renamed + units)
- `depth_diff_sigma` → `delta_sigma` (renamed for consistency)

**Deprecation path**: Keep old keys in v1, remove in v2.

### Runtime Cost
- Pink noise estimation adds ~5-10% overhead
- Still O(N) in light curve length
- No external dependencies added

---

## 10. References

1. **Thompson et al. 2018**, ApJS 235, 38 — Kepler DR25 Robovetter, KOI catalog
2. **Kunimoto et al. 2025**, arXiv:2509.10619 — LEO-Vetter (§4.4 Odd-Even Test)
3. **Pont et al. 2006**, MNRAS 373, 231 — Pink noise model for transit photometry
4. **Hartman & Bakos 2016**, A&C 17, 1 — VARTOOLS, red noise estimation
