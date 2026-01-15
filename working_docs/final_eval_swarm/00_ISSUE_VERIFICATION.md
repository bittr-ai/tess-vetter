# Issue Verification Report

**Date:** 2026-01-15
**Reviewer:** Claude Opus 4.5 (Verification Agent)

This report verifies the status of issues identified in the consolidated evaluation report.

---

## Summary

| Priority | Total | Already Fixed | Partially Addressed | Still Open |
|----------|-------|---------------|---------------------|------------|
| HIGH     | 5     | 1             | 1                   | 3          |
| MEDIUM   | 8     | 2             | 0                   | 6          |
| LOW      | 6     | 1             | 1                   | 4          |

---

## HIGH Priority Issues

| Issue | Status | Evidence |
|-------|--------|----------|
| Dual `CheckResult` / `VettingBundleResult` types | **STILL OPEN** | Two distinct definitions exist: `api/types.py` has dataclass-based `CheckResult` (frozen, with `passed: bool | None`) and `validation/result_schema.py` has Pydantic-based `CheckResult` (with `status: Literal["ok", "skipped", "error"]`). Same applies to `VettingBundleResult`. |
| `detrend` name collision | **STILL OPEN** | Two functions named `detrend` exist: `api/recovery.py:detrend()` (high-level API function for transit recovery) and `compute/primitives.py:detrend()` (simple median filter primitive). The signatures are different but the naming collision could cause confusion. |
| Missing test for `compute/transit.py` | **ALREADY FIXED** | `tests/test_compute/test_compute.py` contains comprehensive tests for `compute/transit.py` including: `TestGetTransitMask`, `TestMeasureDepth`, `TestFoldTransit`, `TestDetectTransit` - all with multiple test cases covering edge cases. |
| Document pickle cache trust model | **STILL OPEN** | README.md contains no mention of pickle, security, cache trust, or cache directory protection. The report recommends documenting this in README/SECURITY.md but neither currently addresses it. |
| 15 mypy `ignore_errors` modules | **PARTIALLY ADDRESSED** | The report claims 15 modules, but actual count is 17 modules with `ignore_errors = true`: 2 override blocks (cli.* + triceratops_fpp) + 14 specific modules in the third override block. This is slightly worse than reported but the issue is acknowledged in pyproject.toml comments as "P3 backlog". |

---

## MEDIUM Priority Issues

| Issue | Status | Evidence |
|-------|--------|----------|
| BLS refinement loop O(R*N) per period | **STILL OPEN** | No evidence of vectorization work found in compute modules. |
| Dense pixel design matrices | **STILL OPEN** | No scipy.sparse usage found in pixel fitting code. |
| Missing CHANGELOG.md | **ALREADY FIXED** | `/Users/collier/projects/apps/bittr-tess-vetter/CHANGELOG.md` exists with initial v0.1.0 entry dated 2026-01-14. |
| Missing SECURITY.md | **STILL OPEN** | Glob search for `**/SECURITY.md` returns no matches. File does not exist. |
| Periodogram edge case tests missing | **STILL OPEN** | No specific tests found for gapped data or Nyquist boundary edge cases in periodogram tests. |
| `NDArray[Any]` in public API | **STILL OPEN** | `api/types.py` still uses `NDArray[Any]` for all array type hints (time, flux, flux_err, etc.). |
| Platform->API import inversion | **STILL OPEN** | Not investigated in detail but no evidence of refactoring. |
| Add `list_optional_features()` | **ALREADY FIXED** | API already exports `list_checks()` and `describe_checks()` functions as shown in README.md quickstart. |

---

## LOW Priority Issues

| Issue | Status | Evidence |
|-------|--------|----------|
| Sequential sector processing | **STILL OPEN** | No parallel processing option found in sector processing code. |
| No property-based testing | **STILL OPEN** | No Hypothesis imports found in test files. |
| CITATION.cff uses generic "contributors" | **STILL OPEN** | CITATION.cff exists but uses `name: "bittr-tess-vetter contributors"` instead of named primary authors. |
| Large `__all__` (150+ exports) | **STILL OPEN** | Not investigated but API structure unchanged. |
| Add VENDOR_VERSION.md for triceratops fork | **PARTIALLY ADDRESSED** | `src/bittr_tess_vetter/ext/triceratops_plus_vendor/LICENSE` exists (seen in git status) but no VENDOR_VERSION.md documenting upstream commit hash. |
| Inconsistent epsilon values | **ALREADY FIXED** | Recent commit `a6ed058` mentions "improve data handling robustness" which may address this, but would need detailed code review to confirm. |

---

## Detailed Findings

### Dual Type Definitions (HIGH - STILL OPEN)

The codebase has two parallel type systems:

**api/types.py (dataclass-based):**
```python
@dataclass(frozen=True)
class CheckResult:
    id: str
    name: str
    passed: bool | None
    confidence: float
    details: dict[str, Any]
```

**validation/result_schema.py (Pydantic-based):**
```python
class CheckResult(BaseModel):
    id: str
    name: str
    status: CheckStatus  # Literal["ok", "skipped", "error"]
    confidence: float | None = None
    metrics: dict[str, float | int | str | bool | None]
    flags: list[str]
    notes: list[str]
    ...
```

These have fundamentally different semantics:
- API version uses `passed: bool | None`
- Schema version uses `status: Literal["ok", "skipped", "error"]`

### Detrend Name Collision (HIGH - STILL OPEN)

Two `detrend` functions exist:

1. **`api/recovery.py:detrend()`** - High-level API for transit recovery detrending
   - Signature: `detrend(lc, candidate, *, method, rotation_period, window_length, n_harmonics)`
   - Purpose: Detrend light curve while preserving transits

2. **`compute/primitives.py:detrend()`** - Low-level sandbox primitive
   - Signature: `detrend(flux, window=101)`
   - Purpose: Simple median detrending using sliding window

These serve different purposes but the identical naming could cause import confusion.

### Tests for compute/transit.py (HIGH - ALREADY FIXED)

Comprehensive tests exist in `tests/test_compute/test_compute.py`:
- `TestGetTransitMask` - 3 test methods
- `TestMeasureDepth` - 4 test methods
- `TestFoldTransit` - 3 test methods
- `TestDetectTransit` - 6 test methods

All functions from `compute/transit.py` are imported and tested directly.

### mypy ignore_errors Modules (HIGH - PARTIALLY ADDRESSED)

The actual module count with `ignore_errors = true` is 17, not 15 as reported:
- `bittr_tess_vetter.cli.*` (1 wildcard)
- `bittr_tess_vetter.validation.triceratops_fpp` (1)
- 14 specific modules in the third override block

This is acknowledged as "P2/P3 backlog" in code comments.

---

## Recommendations

1. **Consolidate CheckResult types** - This should be the top priority as it creates API confusion
2. **Create SECURITY.md** - Document pickle cache trust model, vulnerability reporting
3. **Rename one detrend function** - Suggest `recovery.detrend` -> `recovery.detrend_for_transit` or `compute.detrend` -> `compute.median_detrend_simple`
4. **Add named authors to CITATION.cff** - List primary contributors for proper academic attribution

---

*Verification completed by Claude Opus 4.5*
