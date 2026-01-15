# Fix 01: Consolidate Dual CheckResult/VettingBundleResult Types

## Problem Statement

Two distinct definitions existed for `CheckResult` and `VettingBundleResult`:

1. **api/types.py** - dataclass-based with:
   - `passed: bool | None`
   - `confidence: float`
   - `details: dict[str, Any]`

2. **validation/result_schema.py** - Pydantic-based with:
   - `status: Literal["ok", "skipped", "error"]`
   - `metrics: dict[str, float | int | str | bool | None]`
   - `flags: list[str]`
   - `notes: list[str]`
   - `provenance: dict[str, ...]`
   - `raw: dict[str, Any] | None`

This duplication caused confusion and inconsistent APIs between the validation module and public API facade.

## Solution

Consolidated to a SINGLE definition in `validation/result_schema.py` (Pydantic) as the source of truth.

### Changes Made

#### 1. Enhanced validation/result_schema.py

Added backward-compatible properties to `CheckResult`:

```python
@property
def passed(self) -> bool | None:
    """Backward-compatible property: True if status='ok', False if 'error', None if 'skipped'."""
    if self.status == "ok":
        return True
    if self.status == "error":
        return False
    return None  # skipped

@property
def details(self) -> dict[str, Any]:
    """Backward-compatible property: combined metrics/flags/notes/raw as details dict."""
    result: dict[str, Any] = dict(self.metrics)
    result["status"] = self.status
    if self.flags:
        result["flags"] = self.flags
    if self.notes:
        result["notes"] = self.notes
    if self.raw:
        result.update(self.raw)
    result["_metrics_only"] = True
    return result
```

Added missing properties to `VettingBundleResult`:
- `n_failed`
- `n_unknown`
- `all_passed`
- `failed_check_ids`
- `unknown_check_ids`

#### 2. Updated api/types.py

Changed from defining its own dataclasses to re-exporting from `validation/result_schema.py`:

```python
from bittr_tess_vetter.validation.result_schema import (
    CheckResult,
    CheckStatus,
    VettingBundleResult,
    error_result,
    ok_result,
    skipped_result,
)
```

Removed duplicate dataclass definitions of `CheckResult` and `VettingBundleResult`.

#### 3. Updated API modules

Updated `_convert_result()` functions in:
- `api/exovetter.py`
- `api/catalog.py`
- `api/pixel.py`
- `api/lc_only.py`
- `api/evidence.py`

Changed from constructing old dataclass to using helper functions:
```python
# Before
return CheckResult(
    id=result.id,
    name=result.name,
    passed=None,
    confidence=result.confidence,
    details=details,
)

# After
return ok_result(
    id=result.id,
    name=result.name,
    metrics=metrics,
    confidence=result.confidence,
    raw=raw_data if raw_data else None,
)
```

#### 4. Updated test files

Updated tests to use new schema:
- `tests/test_api/test_evidence_api.py`
- `tests/test_api/test_exovetter_api.py`
- `tests/test_api/test_lc_only.py`
- `tests/test_api/test_policy_mode_deprecated.py`
- `tests/test_api/test_types.py`

Key changes in test assertions:
```python
# Before
assert r.passed is None
assert r.details.get("status") == "skipped"

# After
assert r.status == "skipped"
assert r.passed is None  # via backward-compat property
```

## Semantic Changes

| Old Schema | New Schema | Meaning |
|------------|------------|---------|
| `passed=True` | `status="ok"` | Check completed successfully with metrics |
| `passed=False` | `status="error"` | Check encountered an error |
| `passed=None` | `status="skipped"` | Check was skipped (missing data, disabled) |

Note: The old `passed=None` was overloaded to mean both "metrics-only" and "skipped". The new schema distinguishes these:
- `status="ok"` with metrics = successful metrics-only result (now `passed=True`)
- `status="skipped"` = explicitly skipped (still `passed=None`)

## API Surface Changes

### Constructing Results

```python
# Old way (no longer works)
CheckResult(id="V01", name="check", passed=True, confidence=0.9, details={...})

# New way - use helper functions
ok_result(id="V01", name="check", metrics={...}, confidence=0.9)
skipped_result(id="V01", name="check", reason_flag="MISSING_DATA")
error_result(id="V01", name="check", error="VALIDATION_ERROR")
```

### Accessing Results

```python
result.status      # "ok" | "skipped" | "error"
result.passed      # True | False | None (backward-compat property)
result.metrics     # Structured metrics dict
result.flags       # Machine-readable flags list
result.notes       # Human-readable notes list
result.details     # Backward-compat combined dict
```

## astro-arc-tess Impact Assessment

**No changes required.**

astro-arc-tess imports from `bittr_tess_vetter.api.types`:
- `Candidate`
- `Ephemeris`
- `LightCurve`
- `StellarParams`
- `TPFStamp`

It does NOT import `CheckResult` or `VettingBundleResult` from api/types.py.

astro-arc-tess uses `VetterCheckResult` from `bittr_tess_vetter.api.detection`, which is a separate internal type (not affected by this change).

The robustness matrix and other astro-arc-tess components use their own Pydantic models with `.passed` and `.confidence` attributes.

## Test Results

All tests pass after changes:
```
uv run pytest tests/ -x -q --tb=short
# 100% pass (with expected skips for optional dependencies)
```

## Files Modified

### bittr-tess-vetter

**Source files:**
- `src/bittr_tess_vetter/validation/result_schema.py` - Added backward-compat properties
- `src/bittr_tess_vetter/api/types.py` - Re-export instead of define
- `src/bittr_tess_vetter/api/exovetter.py` - Use new result helpers
- `src/bittr_tess_vetter/api/catalog.py` - Use new result helpers
- `src/bittr_tess_vetter/api/pixel.py` - Use new result helpers
- `src/bittr_tess_vetter/api/lc_only.py` - Use new result helpers
- `src/bittr_tess_vetter/api/evidence.py` - Update docstring

**Test files:**
- `tests/test_api/test_evidence_api.py`
- `tests/test_api/test_exovetter_api.py`
- `tests/test_api/test_lc_only.py`
- `tests/test_api/test_policy_mode_deprecated.py`
- `tests/test_api/test_types.py`

## Summary

The consolidation eliminates type duplication and establishes `validation/result_schema.py` as the single source of truth for vetting result types. Backward compatibility is maintained through properties on the Pydantic model, allowing existing code to continue using `.passed` and `.details` while new code can use the more structured `.status`, `.metrics`, `.flags`, and `.notes` fields.
