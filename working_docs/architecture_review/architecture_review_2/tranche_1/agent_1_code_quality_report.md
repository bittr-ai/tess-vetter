# Agent 1: Code Quality Report

**Date:** 2026-01-14
**Agent:** Claude Opus 4.5
**Focus:** CI-Readiness - All tests pass, all linting passes

## Summary

Successfully completed all 5 tasks to make the bittr-tess-vetter codebase CI-ready. Final verification shows:

- **Tests:** 1882 passed, 48 skipped, 1 warning
- **Ruff:** All checks passed (0 errors)

---

## Task 1: P0.7 - Fix Failing Test

### Problem
The test `tests/pixel/test_tpf_fits.py::TestTPFFitsRefSerialization::test_from_string_invalid` failed due to error message mismatch after recent `exptime_seconds` addition to the TPF FITS reference format.

The test expected "Invalid TPF FITS reference format" but the code now raises "Invalid exptime_seconds" because the 5-part format `tpf_fits:123456789:15:spoc:extra` is now valid (with the new exptime_seconds field), and the error occurs when parsing `extra` as an integer.

### Fix
Updated the test expectation in `/Users/collier/projects/apps/bittr-tess-vetter/tests/pixel/test_tpf_fits.py`:

```python
# Before:
("tpf_fits:123456789:15:spoc:extra", "Invalid TPF FITS reference format"),

# After:
("tpf_fits:123456789:15:spoc:extra", "Invalid exptime_seconds"),
```

### Verification
```bash
uv run pytest tests/pixel/test_tpf_fits.py  # All passed
```

---

## Task 2: P0.8 - Fix All Ruff Errors

### Problem
115 ruff errors across the codebase, including:
- I001: Unsorted/unformatted import blocks
- F401: Unused imports
- F811: Redefinitions of unused names
- SIM102: Nested if statements that can be combined
- SIM105: try-except-pass patterns that should use contextlib.suppress
- SIM108: If-else blocks that should be ternary operators
- SIM117: Nested with statements that can be combined
- B017: Asserting blind exceptions
- E402: Module level imports not at top of file
- F821: Undefined names

### Fixes Applied

1. **Import organization** - Used `uv run ruff check . --fix` to auto-fix 69 import-related issues

2. **F811 redefinitions in `/Users/collier/projects/apps/bittr-tess-vetter/src/bittr_tess_vetter/api/__init__.py`**:
   - Removed duplicate imports of `PeriodogramPeak`, `PeriodogramResult`, `compute_transit_model`, and `detrend`

3. **F821 undefined `types` in `/Users/collier/projects/apps/bittr-tess-vetter/src/bittr_tess_vetter/compute/primitives.py`**:
   - Added `import types` to imports

4. **E402 in `/Users/collier/projects/apps/bittr-tess-vetter/src/bittr_tess_vetter/domain/target.py`**:
   - Moved `from pydantic import Field` to top with other imports

5. **SIM102, SIM105 in `/Users/collier/projects/apps/bittr-tess-vetter/src/bittr_tess_vetter/platform/catalogs/exofop_toi_table.py`**:
   - Combined nested if statements
   - Replaced try-except-pass with `contextlib.suppress(Exception)`

6. **SIM108 in `/Users/collier/projects/apps/bittr-tess-vetter/src/bittr_tess_vetter/platform/io/mast_client.py`**:
   - Converted if-else block to ternary operator

7. **SIM117 in `/Users/collier/projects/apps/bittr-tess-vetter/tests/io/test_mast_client.py`**:
   - Combined nested with statements using parenthesized context managers

8. **B017 in `/Users/collier/projects/apps/bittr-tess-vetter/tests/pixel/test_aperture.py` and `test_centroid.py`**:
   - Changed `pytest.raises(Exception)` to `pytest.raises((AttributeError, TypeError))` for immutability tests

### Verification
```bash
uv run ruff check .  # All checks passed!
```

---

## Task 3: P0.4 - Create CLI Smoke Tests

### Implementation
Created `/Users/collier/projects/apps/bittr-tess-vetter/tests/cli/test_cli_smoke.py` with comprehensive smoke tests for 5 CLI modules:

- `mlx_bls_search_cli.py`
- `mlx_bls_search_range_cli.py`
- `mlx_quick_vet_cli.py`
- `mlx_refine_candidates_cli.py`
- `mlx_tls_calibration_cli.py`

### Test Coverage

1. **Import Smoke Tests** (`TestCLIImports`):
   - Verifies all CLI modules can be imported
   - Checks that each module has a `main()` function

2. **Help Invocation Tests** (`TestCLIHelp`):
   - Tests `--help` produces output and exits cleanly
   - Uses subprocess to invoke modules with `python -m`

3. **Entry Point Tests** (`TestCLIEntryPoints`):
   - Verifies modules can be run with `python -m`
   - Checks appropriate exit codes for missing arguments

4. **Module Content Verification** (`TestCLIModuleContent`):
   - Validates expected dataclasses exist in each module

5. **MLX Availability Tests** (`TestMLXAvailability`, `TestMLXUnavailable`):
   - Conditional tests based on MLX installation
   - Tests MLX import helper behavior

### Verification
```bash
uv run pytest tests/cli/test_cli_smoke.py  # All passed
```

---

## Task 4: P1.1 - Add Exhaustive Export Resolution Test

### Implementation
Added to `/Users/collier/projects/apps/bittr-tess-vetter/tests/test_api/test_api_top_level_exports.py`:

```python
@pytest.mark.parametrize("name", _get_api_all())
def test_all_exports_resolve(name: str) -> None:
    """Verify every symbol in __all__ is actually accessible on the api module."""
    from bittr_tess_vetter import api

    assert hasattr(api, name), f"Export {name!r} declared in __all__ but not accessible on api module"
    obj = getattr(api, name)
    assert obj is not None or name.endswith("_AVAILABLE") or name.startswith("WOTAN"), (
        f"Export {name!r} resolved to None - check import path"
    )
```

### Issue Discovered
This test uncovered a missing export: `recover_transit_timeseries` was declared in `__all__` but not actually imported in the API module.

### Fix
Added re-export in `/Users/collier/projects/apps/bittr-tess-vetter/src/bittr_tess_vetter/api/recovery.py`:
```python
from bittr_tess_vetter.recovery import pipeline as _pipeline
recover_transit_timeseries = _pipeline.recover_transit_timeseries
```

### Additional Fix
Also fixed missing exports in triceratops_cache.py that were causing `test_triceratops_cache_facade_exports` to fail:
- `prefetch_trilegal_csv`
- `load_cached_triceratops_target`
- `save_cached_triceratops_target`
- `estimate_transit_duration`

These were private functions (prefixed with `_`) that needed to be re-exported with public names.

### Verification
```bash
uv run pytest tests/test_api/test_api_top_level_exports.py  # All 229 parametrized tests passed
```

---

## Task 5: P1.2 - Add vet_candidate Integration Tests

### Implementation
Created `/Users/collier/projects/apps/bittr-tess-vetter/tests/test_integration/test_vet_candidate_full.py` with comprehensive tests:

### Test Classes

1. **TestVetCandidateBasicWorkflow** (3 tests):
   - Full workflow with synthetic transit
   - Minimal ephemeris passthrough
   - VettingBundleResult structure verification

2. **TestVetCandidateErrorPropagation** (4 tests):
   - Invalid ephemeris period raises ValueError
   - Invalid duration raises ValueError
   - Zero period raises ValueError
   - Unknown check IDs produce warnings

3. **TestVetCandidateConfigPassthrough** (3 tests):
   - `network=False` skips catalog checks
   - Enabled subset runs only specified checks
   - Stellar params enhance duration consistency check

4. **TestVetCandidateMultiCheck** (3 tests):
   - LC-only checks always run
   - Result aggregation counts
   - Warnings list populated on issues

5. **TestVetCandidateEdgeCases** (3 tests):
   - Short lightcurve handled gracefully
   - Very long period handled
   - Deep transit does not crash

### Helper Functions
- `_make_synthetic_lightcurve()`: Generates synthetic light curve with Gaussian noise
- `_inject_box_transit()`: Injects box-shaped transits
- `_make_lc_with_transit()`: Creates LightCurve with injected transit and matching ephemeris

### Verification
```bash
uv run pytest tests/test_integration/test_vet_candidate_full.py  # All 16 tests passed
```

---

## Files Modified

1. `/Users/collier/projects/apps/bittr-tess-vetter/tests/pixel/test_tpf_fits.py` - Updated error expectation
2. `/Users/collier/projects/apps/bittr-tess-vetter/src/bittr_tess_vetter/api/__init__.py` - Fixed duplicate imports
3. `/Users/collier/projects/apps/bittr-tess-vetter/src/bittr_tess_vetter/api/pixel_localize.py` - Simplified return statement
4. `/Users/collier/projects/apps/bittr-tess-vetter/src/bittr_tess_vetter/compute/__init__.py` - Added F401 noqa
5. `/Users/collier/projects/apps/bittr-tess-vetter/src/bittr_tess_vetter/compute/primitives.py` - Added types import
6. `/Users/collier/projects/apps/bittr-tess-vetter/src/bittr_tess_vetter/domain/target.py` - Fixed import order
7. `/Users/collier/projects/apps/bittr-tess-vetter/src/bittr_tess_vetter/platform/catalogs/exofop_toi_table.py` - Simplified conditionals
8. `/Users/collier/projects/apps/bittr-tess-vetter/src/bittr_tess_vetter/platform/io/mast_client.py` - Converted to ternary
9. `/Users/collier/projects/apps/bittr-tess-vetter/tests/io/test_mast_client.py` - Combined with statements
10. `/Users/collier/projects/apps/bittr-tess-vetter/tests/pixel/test_aperture.py` - Fixed exception assertions
11. `/Users/collier/projects/apps/bittr-tess-vetter/tests/pixel/test_centroid.py` - Fixed exception assertions
12. `/Users/collier/projects/apps/bittr-tess-vetter/src/bittr_tess_vetter/api/recovery.py` - Added recover_transit_timeseries export
13. `/Users/collier/projects/apps/bittr-tess-vetter/src/bittr_tess_vetter/api/triceratops_cache.py` - Added cache helper exports

## Files Created

1. `/Users/collier/projects/apps/bittr-tess-vetter/tests/cli/__init__.py`
2. `/Users/collier/projects/apps/bittr-tess-vetter/tests/cli/test_cli_smoke.py`
3. `/Users/collier/projects/apps/bittr-tess-vetter/tests/test_integration/test_vet_candidate_full.py`

---

## Final Status

```bash
$ uv run pytest
1882 passed, 48 skipped, 1 warning in 51.77s

$ uv run ruff check .
All checks passed!
```

The codebase is now CI-ready with all tests passing and all linting issues resolved.
