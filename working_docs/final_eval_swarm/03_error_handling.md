# Error Handling & Edge Cases Review

**Package**: bittr-tess-vetter
**Reviewer Focus**: Robustness for open-source release
**Date**: 2026-01-14

## Executive Summary

The codebase demonstrates **strong defensive programming** with consistent patterns for handling NaN/Inf values, empty arrays, and edge cases. Key strengths include explicit finite-value filtering, graceful degradation with informative warnings, and well-structured error taxonomy. A few areas could benefit from additional guards, documented below.

---

## 1. NaN/Inf Handling

### Strengths

The codebase employs consistent finite-value filtering across computational modules:

```python
# Common pattern (activity/primitives.py, lines 54, 313)
finite = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)

# MAST client data cleaning (mast_client.py, lines 205, 744)
finite_time = np.isfinite(time)
if median_flux > 0 and np.isfinite(median_flux):
    ...
```

**Key patterns observed:**
- Input arrays are consistently filtered for finite values before computation
- `np.nanmedian()` and `np.nanmean()` used where NaN values are expected
- `np.nan_to_num()` used in periodogram and Lomb-Scargle outputs (line 844)
- `safe_float()` helpers convert NaN to defaults in TLS result extraction

### Coverage

| Module | NaN/Inf Handling |
|--------|------------------|
| `compute/primitives.py` | Filters via `np.isfinite()` before phase folding |
| `compute/periodogram.py` | Returns `np.nan_to_num(power_arr, nan=0.0, posinf=0.0)` |
| `validation/base.py` | Checks `np.isfinite()` on phase calculations |
| `recovery/primitives.py` | Uses `np.isfinite()` in SNR estimation |
| `transit/vetting.py` | Guards ephemeris params with `np.isfinite()` |
| `domain/lightcurve.py` | Returns `float("nan")` for empty valid sets |

### Potential Gap

In `bls_like_search.py` line 71, `np.nan_to_num()` silently replaces NaN with overall mean, which could mask data quality issues. Consider logging when this occurs.

---

## 2. Empty Array Handling

### Strengths

Functions consistently check for insufficient data before proceeding:

```python
# count_transits (validation/base.py, line 229)
if time.size == 0:
    return 0

# bin_phase_curve (validation/base.py, lines 121-127)
if np.sum(mask) > 0:
    bin_means[i] = np.nanmean(flux[mask])
else:
    bin_means[i] = np.nan

# detect_sector_gaps (compute/periodogram.py, line 57)
if len(time) < 2:
    return np.array([], dtype=np.intp)
```

### Pattern: Return Safe Defaults

When data is insufficient, functions return structured "empty" results rather than raising exceptions:

```python
# transit/vetting.py - _empty_odd_even_result()
def _empty_odd_even_result(*, n_odd: int = 0, n_even: int = 0) -> OddEvenResult:
    return OddEvenResult(
        depth_odd_ppm=0.0,
        depth_even_ppm=0.0,
        ...
        interpretation="INSUFFICIENT_DATA",
    )

# compute/periodogram.py - auto_periodogram() with < 3 points
if len(time) < 3:
    return PeriodogramResult(
        peaks=[],
        best_period=float(min_period),
        ...
    )
```

### Coverage Summary

| Scenario | Handling |
|----------|----------|
| Empty time array | Returns 0 transits / empty result |
| Single-point data | Robust std returns 0.0 |
| No valid in-transit points | Returns depth=0, error=inf |
| Empty phase bins | Sets bin values to NaN |
| Zero-length bootstrap | Falls back to default uncertainty |

---

## 3. Division-by-Zero Protection

### Explicit Guards

```python
# recovery/primitives.py, line 257 - Variability removal
model = np.clip(model, 0.5, 2.0)  # Prevents division by near-zero

# recovery/primitives.py, line 317 - Transit stacking
safe_flux_err = np.maximum(flux_err, 1e-10)  # Prevents 1/0 in weights

# validation/ephemeris_specificity.py, line 121
sigma2 = np.maximum(flux_err * flux_err, eps)  # eps guard

# compute/bls_like_search.py, line 101
w = 1.0 / np.maximum(flux_err * flux_err, 1e-12)
```

### Pattern: Conditional Division

```python
# SNR calculation pattern used throughout
snr = depth / depth_err if depth_err > 0 else 0.0

# Relative difference with epsilon
max_depth = max(abs(median_odd), abs(median_even), eps)
rel_diff = abs(delta) / max_depth

# Duration ratio (validation/lc_checks.py)
ratio = duration_hours / expected_duration_hours if expected_duration_hours > 0 else float("inf")
```

### Coverage Summary

| Location | Protection |
|----------|------------|
| Flux error weights | `np.maximum(flux_err, 1e-10)` or `1e-12` |
| Model division | `np.clip(model, 0.5, 2.0)` |
| SNR calculation | Conditional `if err > 0` |
| Depth ratios | `max(..., eps)` denominator |
| Chi-squared | `if sigma > 0` guard |

---

## 4. Graceful Failure Patterns

### Structured Error Returns

The codebase uses a clean error taxonomy (`errors.py`):

```python
class ErrorType(str, Enum):
    CACHE_MISS = "CACHE_MISS"
    INVALID_REF = "INVALID_REF"
    INVALID_DATA = "INVALID_DATA"
    INTERNAL_ERROR = "INTERNAL_ERROR"
```

### Validation Guardrails

Validation checks return metrics-only results with `passed=None` rather than failing:

```python
# validation/lc_checks.py - Duration too long guard
if duration_days >= 0.5 * period:
    warnings.append("duration_too_long_relative_to_period")
    return VetterCheckResult(
        passed=None,
        confidence=0.2,
        details={"warnings": warnings, ...}
    )
```

### Network Operation Protection

```python
# platform/network/timeout.py
class NetworkTimeoutError(Exception):
    def __init__(self, operation: str, timeout_seconds: float) -> None:
        super().__init__(
            f"{operation} timed out after {timeout_seconds:.1f}s. "
            "The external service may be slow or unavailable."
        )

@contextmanager
def network_timeout(seconds: float, operation: str) -> Generator[...]:
    if seconds <= 0:
        raise ValueError(f"Timeout must be positive, got {seconds}")
    ...
```

### Cache Layer Robustness

The `TPFFitsCache` gracefully handles corrupted data:

```python
# pixel/tpf_fits.py
except (ValueError, KeyError, OSError, fits.VerifyError) as e:
    logger.warning("Failed to load cached TPF FITS for %s: %s", ref.to_string(), e)
    return None
```

### Recovery Pipeline Guards

```python
# recovery/pipeline.py
if len(time) != len(flux) or len(time) != len(flux_err):
    raise ValueError("time/flux/flux_err must have the same length")
if len(time) < 10:
    raise ValueError("Insufficient data points for recovery")
if period <= 0 or duration_hours <= 0:
    raise ValueError("period and duration_hours must be positive")
if not fit.converged:
    raise RuntimeError("Transit fit failed to converge. Check ephemeris parameters.")
```

---

## 5. Input Validation

### Domain Model Validation

`LightCurveData` validates all inputs in `__post_init__`:

```python
# domain/lightcurve.py
def __post_init__(self) -> None:
    # Type checks
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(arr).__name__}")

    # Dtype checks
    if self.time.dtype != np.float64:
        raise ValueError(f"time must be float64, got {self.time.dtype}")

    # Shape consistency
    if len(self.flux) != n:
        raise ValueError(f"flux length {len(self.flux)} != time length {n}")

    # Immutability
    arr.flags.writeable = False
```

### Reference String Parsing

TPFFitsRef validates all components:

```python
# pixel/tpf_fits.py
def __post_init__(self) -> None:
    if self.tic_id < 1:
        raise ValueError(f"tic_id must be positive, got {self.tic_id}")
    if self.sector < 1:
        raise ValueError(f"sector must be positive, got {self.sector}")
    if normalized_author not in VALID_AUTHORS:
        raise ValueError(f"author must be one of {sorted(VALID_AUTHORS)}")
```

---

## 6. Warning System

### Graduated Warning Collection

Validation functions collect warnings without failing:

```python
# validation/lc_checks.py - check_odd_even_depth
warnings: list[str] = []

if n_odd_transits < config.min_transits_per_parity:
    warnings.append(f"Only {n_odd_transits} odd transit(s), need {config.min_transits_per_parity}")

if global_oot_fallback_count >= epochs_processed * 0.5:
    warnings.append(f"odd_even_baseline_fallback_global_oot: ...")
```

### Confidence Degradation

Warnings reduce confidence scores rather than causing failures:

```python
if has_warnings:
    base_confidence *= 0.9
```

---

## 7. Recommendations

### High Priority

1. **Consistent epsilon values**: Standardize `1e-10` vs `1e-12` across modules for weight calculations.

2. **Log NaN replacement**: Add debug logging when `np.nan_to_num()` replaces values in production code paths.

### Medium Priority

3. **Add bounds checking**: For array indexing in bootstrap resampling when source array could be very small.

4. **Document failure modes**: Add docstring sections describing what happens with edge-case inputs.

### Low Priority

5. **Sentinel values**: Consider using `Optional[float]` instead of `float("inf")` for "no measurement" cases to improve type safety.

6. **Error context**: Enhance `ErrorEnvelope.context` usage to include stack traces for debugging.

---

## 8. Test Coverage Assessment

The test suite includes edge case coverage:

```
tests/test_support/test_errors_and_timeout.py  # Timeout/error handling
tests/validation/test_odd_even_depth.py        # Empty/insufficient data
tests/test_compute/test_bls_like_search.py     # Edge period ranges
tests/pixel/test_centroid.py                   # Empty frame handling
```

**Suggested additions:**
- Test with all-NaN flux arrays
- Test with single-point time series
- Test with alternating valid/invalid points
- Test exact boundary conditions (e.g., duration == 0.5 * period)

---

## 9. Summary Table

| Category | Rating | Notes |
|----------|--------|-------|
| NaN/Inf filtering | Excellent | Consistent `np.isfinite()` usage |
| Empty array handling | Excellent | Returns safe defaults |
| Division-by-zero | Good | Uses `np.maximum()` and conditionals |
| Graceful failure | Excellent | Metrics-only returns, no crashes |
| Input validation | Good | Strong in domain models |
| Warning system | Excellent | Graduated degradation |
| Error messages | Good | Informative but could add context |

**Overall Assessment**: The codebase is **well-prepared for open-source release** from an error handling perspective. The defensive programming patterns are mature and consistent across modules.
