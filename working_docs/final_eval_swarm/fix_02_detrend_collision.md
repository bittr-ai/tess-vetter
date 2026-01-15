# Fix 02: Detrend Name Collision

## Issue Summary

Two functions named `detrend` existed in the codebase, causing potential import collisions and API confusion:

1. **`api/recovery.py:detrend()`** - High-level API function for transit recovery workflows
   - Takes `LightCurve` and `Candidate` objects
   - Creates transit masks to protect transit signals
   - Supports wotan/harmonic detrending methods
   - **This is the public API version that should remain as `detrend`**

2. **`compute/primitives.py:detrend()`** - Low-level sandbox primitive
   - Takes only a flux array and window size
   - Uses scipy's `ndimage.median_filter` for simple median detrending
   - Designed for pure-compute sandbox environments

## Solution Applied

Renamed the `compute/primitives.py` function from `detrend` to `median_detrend`. This name:
- Accurately describes the implementation (median filter detrending)
- Follows the naming convention already used in `compute/detrend.py`
- Avoids collision with the public API `detrend` in `api/recovery.py`

## Files Modified

### Source Files

1. **`src/bittr_tess_vetter/compute/primitives.py`**
   - Renamed `detrend()` to `median_detrend()`
   - Updated `astro` namespace: `astro.detrend` -> `astro.median_detrend`
   - Updated `AstroPrimitives` class: `AstroPrimitives.detrend` -> `AstroPrimitives.median_detrend`
   - Updated `__all__` exports

2. **`src/bittr_tess_vetter/compute/__init__.py`**
   - Updated `PRIMITIVES_CATALOG` entry: `"astro.detrend"` -> `"astro.median_detrend"`

3. **`src/bittr_tess_vetter/api/sandbox_primitives.py`**
   - Updated import from `detrend as detrend_median` to direct `median_detrend` import
   - Removed backward-compatibility alias

4. **`src/bittr_tess_vetter/api/primitives.py`**
   - Updated import and `__all__` exports
   - Updated docstring example

5. **`src/bittr_tess_vetter/api/__init__.py`**
   - Updated import and docstring example

### Test Files

1. **`tests/test_compute/test_primitives.py`**
   - Renamed `TestDetrend` class to `TestMedianDetrend`
   - Updated all `detrend()` calls to `median_detrend()`
   - Updated `astro.detrend` calls to `astro.median_detrend`
   - Updated `AstroPrimitives.detrend` references

2. **`tests/test_api/test_api_top_level_exports.py`**
   - Updated import in `test_primitives_submodule_exists()`

3. **`tests/test_integration/test_pipeline_e2e.py`**
   - Updated import in `test_primitives_imports()`

## API Changes

### Breaking Changes

- `from bittr_tess_vetter.compute.primitives import detrend` -> `from bittr_tess_vetter.compute.primitives import median_detrend`
- `astro.detrend(flux, window)` -> `astro.median_detrend(flux, window)`
- `AstroPrimitives.detrend(flux)` -> `AstroPrimitives.median_detrend(flux)`

### Non-Breaking

- `from bittr_tess_vetter.api.recovery import detrend` - **Unchanged** (public API)
- `from bittr_tess_vetter.api.detrend import median_detrend` - **Unchanged** (different implementation)

## Impact on astro-arc-tess

**No changes required.**

Searched `/Users/collier/projects/apps/astro-arc-tess/src/astro_arc` for:
- `astro.detrend` - Not found
- `sandbox.*detrend` - Not found
- `primitives.*detrend` - Not found
- `from bittr_tess_vetter.*import.*detrend` - Not found

The only detrend-related import in astro-arc-tess is:
```python
from bittr_tess_vetter.api.detrend import median_detrend
```

This imports from `api/detrend.py` which has its own `median_detrend` implementation (re-exported from `compute/detrend.py`), which is unaffected by this change.

## Test Results

All primitives-related tests pass:
```
tests/test_compute/test_primitives.py .......................... [ 38%]
tests/test_api/test_api_top_level_exports.py ................... [ 87%]
tests/test_api/test_primitives_api.py .......................... [ 97%]
tests/test_integration/test_pipeline_e2e.py .................... [100%]

======================== 89 passed, 1 warning =========================
```

## Notes

1. There are two `median_detrend` functions in the codebase, which is intentional:
   - `compute/primitives.py:median_detrend` - Sandbox-safe, auto-converts even windows to odd
   - `compute/detrend.py:median_detrend` - Production-quality, requires odd windows

2. The public API `detrend` in `api/recovery.py` serves a different purpose (transit-aware detrending with masking) and remains unchanged.
