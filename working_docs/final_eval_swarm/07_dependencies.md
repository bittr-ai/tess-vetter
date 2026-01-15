# Dependencies & Optional Features Evaluation

**Reviewer:** Claude Code (Dependencies & Optional Features Lens)
**Date:** 2026-01-14
**Package:** bittr-tess-vetter v0.1.0

---

## Executive Summary

The bittr-tess-vetter package demonstrates **excellent dependency hygiene** with a well-structured
approach to optional dependencies. The core package maintains BSD-3-Clause licensing with minimal
required dependencies, while correctly isolating GPL-licensed optional features. Import guards are
consistently applied across all optional dependency modules.

| Aspect | Status | Notes |
|--------|--------|-------|
| Core deps minimal | PASS | Only numpy, scipy, pydantic, astropy, requests |
| Optional deps isolated | PASS | 8 distinct optional extras defined |
| Import guards | PASS | All optional imports properly guarded |
| License isolation | PASS | GPL deps clearly marked as optional |
| Error messages | PASS | Actionable installation instructions |

---

## 1. Core Dependencies Analysis

### pyproject.toml Dependencies (Lines 29-35)

```toml
dependencies = [
  "numpy>=1.24.0,<2.4.0",
  "scipy>=1.10.0",
  "pydantic>=2.4.0",  # CVE-2024-3772 fix
  "astropy>=5.0.0",
  "requests>=2.32.4",  # CVE-2024-47081 fix
]
```

**Assessment:** Minimal and appropriate core dependencies.

| Dependency | Purpose | License | Notes |
|------------|---------|---------|-------|
| numpy | Array operations | BSD-3-Clause | Version pin avoids NumPy 2.x breaking changes |
| scipy | Scientific computing | BSD-3-Clause | |
| pydantic | Input validation | MIT | CVE fix noted in comment |
| astropy | Astronomy utilities | BSD-3-Clause | |
| requests | HTTP client | Apache-2.0 | CVE fix noted in comment |

All core dependencies are BSD/MIT/Apache compatible with the package's BSD-3-Clause license.

---

## 2. Optional Dependencies Structure

### Extras Defined in pyproject.toml (Lines 37-88)

| Extra | Dependencies | Primary License(s) | Purpose |
|-------|--------------|-------------------|---------|
| `tls` | transitleastsquares, numba | MIT, BSD-2-Clause | Transit detection |
| `fit` | emcee, arviz | MIT, Apache-2.0 | MCMC fitting |
| `wotan` | wotan | MIT | Transit-aware detrending |
| `batman` | batman-package | MIT | Physical transit models |
| `mlx` | mlx | Apache-2.0 | Apple Silicon GPU (arm64 only) |
| `exovetter` | exovetter | BSD-3-Clause | ModShift/SWEET checks |
| `ldtk` | ldtk | **GPL-2.0** | Limb darkening coeffs |
| `triceratops` | lightkurve, pytransit, etc. | **GPL-2.0** (pytransit) | False positive probability |

### License Considerations

The package correctly documents GPL-licensed dependencies:

```toml
# ldtk is GPL-2.0 licensed; kept optional to maintain BSD-3-Clause compatibility
ldtk = ["ldtk>=1.8.5"]

# Note: triceratops extra includes pytransit (GPL-2.0). Installing changes
# the effective license of your environment from BSD-3-Clause.
triceratops = [...]
```

**Verdict:** Excellent handling of GPL isolation. Users who install `ldtk` or `triceratops` extras
are clearly warned that their environment's effective license changes.

---

## 3. Import Guard Patterns

### Pattern 1: Exception-Based Guard (Recommended)

**File:** `src/bittr_tess_vetter/compute/periodogram.py` (Lines 529-535)

```python
try:
    from transitleastsquares import transitleastsquares
except ImportError as e:
    raise ImportError(
        "Transit detection requires the 'tls' extra. "
        "Install with: pip install 'bittr-tess-vetter[tls]'"
    ) from e
```

**Assessment:** Actionable error message with exact installation command.

### Pattern 2: Module-Level Availability Flag

**File:** `src/bittr_tess_vetter/compute/detrend.py` (Lines 26-32)

```python
try:
    from wotan import flatten as _wotan_flatten
    WOTAN_AVAILABLE = True
except ImportError:
    _wotan_flatten = None
    WOTAN_AVAILABLE = False
```

**Assessment:** Allows graceful degradation with fallback behavior.

### Pattern 3: Lazy Import with Helper Function

**File:** `src/bittr_tess_vetter/compute/mlx_detection.py` (Lines 24-31)

```python
def _require_mlx() -> Any:
    try:
        import mlx.core as mx
    except ImportError as e:
        raise ImportError(
            "MLX is not installed. Install it (Apple Silicon only), e.g.: `pip install mlx`."
        ) from e
    return mx
```

**Assessment:** Defers import until runtime, clean error message.

### Pattern 4: API-Level Availability Check

**File:** `src/bittr_tess_vetter/api/__init__.py` (Lines 86-96)

```python
MLX_AVAILABLE = _importlib_util.find_spec("mlx") is not None

_MLX_GUARDED_EXPORTS: set[str] = {
    "MlxTopKScoreResult",
    "MlxT0RefinementResult",
    "smooth_box_template",
    ...
}
```

**Assessment:** Prevents import errors at module import time; exports only available when MLX installed.

### Pattern 5: Function-Level Late Import

**File:** `src/bittr_tess_vetter/transit/batman_model.py` (Lines 343, 543-549)

```python
def compute_batman_model(...):
    import batman  # Late import inside function
    ...

def fit_mcmc(...):
    try:
        import arviz as az
        import emcee
    except ImportError as e:
        raise ImportError(
            "MCMC fitting requires the 'fit' extra. "
            "Install with: pip install 'bittr-tess-vetter[fit]'"
        ) from e
```

**Assessment:** Late imports for heavy dependencies, actionable error messages.

---

## 4. Optional Feature Summary

### batman (Transit Model Fitting)

| Location | Guard Type | Coverage |
|----------|-----------|----------|
| `transit/batman_model.py:343` | Late import | compute_batman_model() |
| `api/transit_fit.py:314-318` | try/except with error result | fit_transit() |

**Fallback behavior:** Returns error result with status="error" and message explaining missing dep.

### TLS (Transit Least Squares)

| Location | Guard Type | Coverage |
|----------|-----------|----------|
| `compute/periodogram.py:529-535` | ImportError with hint | tls_search() |
| `compute/periodogram.py:449-453` | ImportError with hint | tls_search_per_sector() |
| `compute/periodogram.py:729-735` | ImportError with hint | search_planets() |

**Fallback behavior:** No fallback; raises ImportError with installation instructions.

### MLX (Apple Silicon GPU)

| Location | Guard Type | Coverage |
|----------|-----------|----------|
| `compute/mlx_detection.py:24-31` | Helper function | All MLX functions |
| `api/__init__.py:86-96` | Conditional __all__ export | API exposure |
| `api/mlx.py:28-33` | find_spec check | MLX_AVAILABLE flag |

**Fallback behavior:** MLX exports hidden from API when unavailable; AttributeError if accessed.

### wotan (Transit-Aware Detrending)

| Location | Guard Type | Coverage |
|----------|-----------|----------|
| `compute/detrend.py:26-32` | Module-level flag | WOTAN_AVAILABLE |
| `compute/detrend.py:428-429` | Raise ImportError | wotan_flatten() |
| `compute/detrend.py:521-542` | Graceful fallback | flatten_with_wotan() |

**Fallback behavior:** `flatten_with_wotan()` falls back to median filter with warning.

### emcee/arviz (MCMC Fitting)

| Location | Guard Type | Coverage |
|----------|-----------|----------|
| `transit/batman_model.py:543-549` | ImportError with hint | fit_mcmc() |
| `api/transit_fit.py:322-330` | Warning + fallback | fit_transit() |

**Fallback behavior:** Falls back to "optimize" method with warning.

### ldtk (Limb Darkening)

| Location | Guard Type | Coverage |
|----------|-----------|----------|
| `transit/batman_model.py:275-305` | try/except with fallback | get_ld_coefficients() |

**Fallback behavior:** Returns empirical LD coefficients with warning.

### TRICERATOPS (False Positive Probability)

| Location | Guard Type | Coverage |
|----------|-----------|----------|
| `validation/triceratops_fpp.py:958-971` | ImportError with error dict | calculate_fpp_handler() |
| `ext/triceratops_plus_vendor/__init__.py:18-26` | Lazy __getattr__ | Module loading |

**Fallback behavior:** Returns error dict with error_type="internal_error".

### exovetter (ModShift/SWEET)

| Location | Guard Type | Coverage |
|----------|-----------|----------|
| `validation/exovetter_checks.py:143-159` | try/except with error result | run_modshift() |
| `validation/exovetter_checks.py:226-242` | try/except with error result | run_sweet() |

**Fallback behavior:** Returns VetterCheckResult with status="error".

---

## 5. License Compatibility Matrix

| Component | License | Compatible with BSD-3-Clause? |
|-----------|---------|------------------------------|
| Core package | BSD-3-Clause | N/A (is BSD) |
| numpy | BSD-3-Clause | Yes |
| scipy | BSD-3-Clause | Yes |
| pydantic | MIT | Yes |
| astropy | BSD-3-Clause | Yes |
| requests | Apache-2.0 | Yes |
| transitleastsquares | MIT | Yes |
| numba | BSD-2-Clause | Yes |
| emcee | MIT | Yes |
| arviz | Apache-2.0 | Yes |
| wotan | MIT | Yes |
| batman-package | MIT | Yes |
| mlx | Apache-2.0 | Yes |
| exovetter | BSD-3-Clause | Yes |
| **ldtk** | **GPL-2.0** | **Optional only** |
| **pytransit** | **GPL-2.0** | **Optional only** |
| lightkurve | MIT | Yes |
| TRICERATOPS+ (vendored) | MIT | Yes |

### Vendored Code

The package vendors TRICERATOPS+ under MIT license:
- Location: `src/bittr_tess_vetter/ext/triceratops_plus_vendor/`
- License file: `LICENSE` (MIT)
- Third-party notice: `THIRD_PARTY_NOTICES.md`

---

## 6. Recommendations

### Strengths

1. **Clear GPL isolation** - GPL-licensed deps (ldtk, pytransit) are clearly documented and optional
2. **Consistent import guards** - All optional deps have proper try/except guards
3. **Actionable error messages** - All ImportError messages include exact pip install commands
4. **Graceful degradation** - Several features have fallback behaviors (wotan, ldtk, emcee)
5. **CVE awareness** - Security fixes noted for pydantic and requests

### Minor Suggestions

1. **Consider adding version constraints for numba** - Currently unconstrained upper bound
2. **Document platform constraint for MLX** - Already in pyproject.toml but could be more prominent in docs
3. **Add `all-non-gpl` extra** - For users who want all features except GPL-licensed ones

### Proposed `all-non-gpl` Extra

```toml
# All optional features without GPL dependencies
all-non-gpl = ["bittr-tess-vetter[tls,fit,wotan,batman,mlx,exovetter]"]
```

---

## 7. Import Guard Coverage Summary

| Optional Dependency | Files with Guards | Guard Quality |
|--------------------|-------------------|---------------|
| batman | 2 | Excellent |
| transitleastsquares | 3 | Excellent |
| mlx | 4 | Excellent |
| wotan | 1 | Excellent |
| emcee/arviz | 2 | Excellent |
| ldtk | 1 | Excellent |
| triceratops | 2 | Excellent |
| exovetter | 1 | Excellent |

**Total files with optional import guards:** 20 files identified via grep

---

## Conclusion

The bittr-tess-vetter package demonstrates exemplary dependency management practices:

1. **Minimal core footprint** - Only 5 required dependencies, all permissively licensed
2. **Clean optional architecture** - 8 well-defined extras for different use cases
3. **Proper GPL isolation** - GPL dependencies are optional with clear documentation
4. **Robust import guards** - Consistent patterns with actionable error messages
5. **Graceful degradation** - Fallback behaviors where scientifically appropriate

The package is **ready for open-source release** from a dependency hygiene perspective.
