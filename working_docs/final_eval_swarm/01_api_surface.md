# API Surface & Usability Review

**Package:** bittr-tess-vetter
**Review Date:** 2026-01-14
**Reviewer:** API Surface Evaluation
**Scope:** `src/bittr_tess_vetter/api/` - Public API surface

---

## Executive Summary

The API surface is **well-designed for open-source release** with a clear "Golden Path" approach, lazy loading for performance, and comprehensive type coverage. However, there are several areas requiring attention before release:

- **70+ modules** in `api/` directory with **229+ exports** in `__all__`
- Clean import story via lazy loading (PEP 562)
- Good separation of golden path vs advanced APIs
- Some naming inconsistencies and redundant exports to address

**Overall Grade: B+** - Ready with minor polish needed

---

## 1. Import Story

### Strengths

1. **Clean Golden Path**: The docstring in `__init__.py` clearly documents recommended imports:
   ```python
   from bittr_tess_vetter.api import (
       LightCurve, Ephemeris, Candidate, CheckResult, VettingBundleResult,
       vet_candidate, VettingPipeline, run_periodogram,
       list_checks, describe_checks,
   )
   ```

2. **Lazy Loading**: PEP 562 implementation via `__getattr__` prevents import-time cost:
   - Parses TYPE_CHECKING block to build export map
   - Only loads modules when accessed
   - Caches resolved symbols in `globals()`

3. **Single Import Point**: Users can import everything from `bittr_tess_vetter.api` without deep imports

4. **Submodule Access**: Advanced users can import from submodules:
   ```python
   from bittr_tess_vetter.api.primitives import fold, detrend
   from bittr_tess_vetter.api.experimental import ...
   ```

### Issues

1. **Large Surface Area**: 70+ modules and 229+ exports may overwhelm users
   - **Recommendation**: Consider trimming `__all__` to golden path only; advanced exports available but not advertised

2. **TYPE_CHECKING Block Complexity**: Export resolution parses AST at runtime
   - Works but adds cognitive overhead for maintainers
   - **Recommendation**: Add comments explaining this pattern for future contributors

---

## 2. Naming Consistency

### Consistent Patterns (Good)

| Pattern | Examples |
|---------|----------|
| `check_*` for check functions | `check_odd_even_depth`, `check_secondary_eclipse` |
| `compute_*` for calculations | `compute_centroid_shift`, `compute_ghost_features` |
| `*Result` for return types | `CheckResult`, `LocalizationResult`, `RecoveryResult` |
| `*Config` for configuration | `PipelineConfig`, `OddEvenConfig` |

### Inconsistencies (Fix Before Release)

| Issue | Examples | Recommendation |
|-------|----------|----------------|
| Mixed verb forms | `vet_candidate` vs `calculate_fpp` vs `fit_transit` | Standardize: all should use imperative (`vet`, `calculate`, `fit`) |
| Redundant prefixes | `run_periodogram` vs `auto_periodogram` | Keep `run_periodogram` as public, `auto_periodogram` as internal |
| Alias confusion | `vet` -> `vet_candidate`, `periodogram` -> `run_periodogram` | Document aliases clearly; consider deprecating duplicates |
| Localization naming | `localize_transit_source` vs `compute_localization_diagnostics` | Inconsistent: one uses `localize_*`, other uses `compute_*` |

### Short Aliases

The module provides short aliases for convenience:
```python
_ALIASES = {
    "vet": "vet_candidate",
    "periodogram": "run_periodogram",
    "localize": "localize_transit_source",
    "aperture_family_depth_curve": "compute_aperture_family_depth_curve",
}
```

**Assessment**: Aliases are helpful but create discoverability issues. Users may not realize `vet` and `vet_candidate` are identical.

**Recommendation**:
- Keep aliases for interactive use
- Document in README that both forms exist
- IDE autocomplete will show both; this is acceptable

---

## 3. Required vs Optional Parameters

### Clear Parameter Patterns (Good)

1. **Keyword-only parameters** after `*`:
   ```python
   def vet_candidate(
       lc: LightCurve,
       candidate: Candidate,
       *,  # Everything after is keyword-only
       stellar: StellarParams | None = None,
       tpf: TPFStamp | None = None,
       network: bool = False,
       ...
   )
   ```

2. **Sensible defaults**:
   - `network=False` - Safe default (no external calls)
   - `checks=None` - Runs all checks
   - `preset="fast"` for FPP - Interactive-friendly default

3. **Type hints** throughout with `| None` for optional params

### Parameter Clarity Issues

| Function | Issue | Recommendation |
|----------|-------|----------------|
| `fit_transit` | `stellar` required but type is `StellarParams` | Add docstring note that minimal params (teff, logg) needed |
| `calculate_fpp` | `cache: PersistentCache` first param | Unclear what cache type is needed; add type alias docs |
| `recover_transit` | `activity` optional but affects `rotation_period` | Document interaction in docstring |
| `run_periodogram` | Many optional params | Consider splitting into preset-based vs custom modes |

### Example of Good Parameter Documentation

From `centroid_shift()`:
```python
def centroid_shift(
    tpf: TPFStamp,
    candidate: Candidate,
    *,
    config: dict[str, Any] | None = None,
) -> CheckResult:
    """V08: Detect centroid motion during transit.

    Args:
        tpf: Target Pixel File data (TPFStamp)
        candidate: Transit candidate with ephemeris
        config: Optional algorithm configuration overrides:
            - centroid_method: {"mean","median","huber"} (default "median")
            - significance_method: {"analytic","bootstrap","permutation"}
            ...
    """
```

This pattern of documenting config dict keys is excellent and should be applied consistently.

---

## 4. Discoverability

### What Works Well

1. **`list_checks()` and `describe_checks()`**: Introspection for available checks
2. **`PRIMITIVES_CATALOG`**: Discoverable list of low-level primitives
3. **Module docstrings**: Each module has clear purpose documentation
4. **`__all__` exports**: Explicit about what's public

### Discoverability Gaps

1. **No single entry point for "what can I do?"**
   - `list_checks()` shows vetting checks only
   - No equivalent for periodogram methods, fitting options, etc.
   - **Recommendation**: Add `list_capabilities()` or expand documentation

2. **Optional dependency discovery**:
   ```python
   MLX_AVAILABLE = _importlib_util.find_spec("mlx") is not None
   ```
   - Good for MLX, but no equivalent for other optionals (wotan, batman, etc.)
   - **Recommendation**: Add `list_optional_features()` showing what's available

3. **Check IDs are opaque**:
   - `V01`, `V02`, etc. require looking up documentation
   - **Recommendation**: Consider human-readable IDs like `lc.odd_even` alongside codes

---

## 5. Redundant/Confusing Exports

### Duplicate Exports

| Export 1 | Export 2 | Issue |
|----------|----------|-------|
| `CheckResult` (api.types) | `CheckResult` (validation.result_schema) | Different types! Pydantic vs dataclass |
| `VettingBundleResult` (api.types) | `VettingBundleResult` (validation.result_schema) | Same issue |
| `detrend` (recovery) | `detrend` (sandbox_primitives) | Different functions! |
| `TransitParams` (localization) | `TransitParams` (primitives) | May be same, needs verification |

**Critical Issue**: The `CheckResult` confusion is serious:
- `api.types.CheckResult` is a frozen dataclass for users
- `validation.result_schema.CheckResult` is Pydantic for pipeline
- Both are exported, causing type confusion

**Recommendation**:
1. Rename one or clearly document the distinction
2. Ensure public API only exposes the user-facing version
3. Add deprecation warning if wrong one is used

### Over-Exposed Internal Details

The following appear in `__all__` but seem like internal implementation details:

```python
# Questionable exports
"_ALIASES",  # Private
"HARMONIC_RATIOS",  # Implementation detail
"FLIP_RATE_MIXED_THRESHOLD",  # Internal constant
"MARGIN_RESOLVE_THRESHOLD",  # Internal constant
```

**Recommendation**: Audit `__all__` and move implementation constants to a `constants` submodule if needed externally.

---

## 6. Type Safety & API Contracts

### Strengths

1. **Frozen dataclasses** for immutable results:
   ```python
   @dataclass(frozen=True)
   class Ephemeris:
       period_days: float
       t0_btjd: float
       duration_hours: float
   ```

2. **Validation in `__post_init__`**:
   ```python
   def __post_init__(self) -> None:
       if self.period_days <= 0:
           raise ValueError(f"period_days must be positive, got {self.period_days}")
   ```

3. **Type conversions** in `LightCurve.to_internal()`:
   - Normalizes dtypes to float64
   - Handles NaN/Inf masking automatically

### Areas for Improvement

1. **Inconsistent return types**:
   - `vet_candidate` returns `VettingBundleResult`
   - `vet_lc_only` returns `list[CheckResult]`
   - **Recommendation**: `vet_lc_only` should also return a bundle type

2. **Raw dict returns**:
   - `calculate_fpp` returns `dict[str, Any]`
   - **Recommendation**: Create `FppResult` dataclass

3. **Loose `config` parameters**:
   - Many functions accept `config: dict[str, Any] | None`
   - No validation of config keys at runtime
   - **Recommendation**: Create typed config dataclasses per function

---

## 7. Optional Dependencies Handling

### Current Approach

```python
MLX_AVAILABLE = _importlib_util.find_spec("mlx") is not None

_MLX_GUARDED_EXPORTS: set[str] = {
    "MlxTopKScoreResult",
    "score_fixed_period",
    ...
}

def __getattr__(name: str) -> Any:
    if name in _MLX_GUARDED_EXPORTS and not MLX_AVAILABLE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

### Assessment

- **Good**: Clear error when accessing unavailable optional features
- **Issue**: Inconsistent across optionals (MLX has guards, wotan doesn't)
- **Issue**: Error message doesn't tell user how to install

### Recommendation

```python
def __getattr__(name: str) -> Any:
    if name in _MLX_GUARDED_EXPORTS and not MLX_AVAILABLE:
        raise ImportError(
            f"{name} requires MLX. Install with: pip install 'bittr-tess-vetter[mlx]'"
        )
```

Apply consistently for: `mlx`, `wotan`, `tls`, `fit`, `batman`, `triceratops`, `exovetter`, `ldtk`

---

## 8. Deprecation Strategy

### Current Deprecations

```python
def _apply_policy_mode(check: CheckResult, *, policy_mode: str) -> CheckResult:
    if policy_mode != "metrics_only":
        warnings.warn(
            "policy_mode is deprecated and ignored...",
            category=FutureWarning,
            stacklevel=2,
        )
```

### Assessment

- **Good**: Uses `FutureWarning` (correct category)
- **Good**: Includes actionable migration path
- **Issue**: Legacy `_vet_candidate_legacy` function exists but isn't connected

### Recommendation

Add deprecation schedule to documentation:
- v0.1.x: Warnings issued
- v0.2.0: Deprecated APIs removed

---

## 9. Actionable Recommendations

### Critical (Block Release)

1. **Resolve CheckResult/VettingBundleResult dual exports** - Confusing for users
2. **Fix detrend name collision** - `recovery.detrend` vs `sandbox_primitives.detrend`

### High Priority

3. Standardize verb forms in function names
4. Add `ImportError` with install instructions for all optional deps
5. Create typed result classes for `calculate_fpp` and similar dict-returning functions
6. Document the lazy loading pattern for maintainers

### Medium Priority

7. Trim `__all__` to golden path; move advanced exports to submodules
8. Add `list_optional_features()` for discoverability
9. Create human-readable check ID aliases (`lc.odd_even` alongside `V01`)
10. Standardize config parameter handling with typed dataclasses

### Low Priority

11. Add more examples in module docstrings
12. Consider namespace packages for cleaner organization
13. Add `__version__` to api module

---

## 10. Summary Checklist

| Criteria | Status | Notes |
|----------|--------|-------|
| Golden path documented | PASS | Clear in `__init__.py` docstring |
| Import story clean | PASS | Lazy loading works well |
| Naming consistent | PARTIAL | Some inconsistencies to fix |
| Required/optional clear | PASS | Good use of keyword-only params |
| Types well-defined | PARTIAL | Dual CheckResult issue |
| Discoverable | PARTIAL | Could use `list_capabilities()` |
| Optional deps handled | PARTIAL | Need consistent guards |
| Deprecations documented | PASS | FutureWarnings in place |

---

## Files Reviewed

- `/src/bittr_tess_vetter/api/__init__.py` (738 lines, lazy loading hub)
- `/src/bittr_tess_vetter/api/types.py` (core types)
- `/src/bittr_tess_vetter/api/vet.py` (main orchestrator)
- `/src/bittr_tess_vetter/api/periodogram.py` (detection API)
- `/src/bittr_tess_vetter/api/primitives.py` (low-level ops)
- `/src/bittr_tess_vetter/api/lc_only.py` (LC checks V01-V05)
- `/src/bittr_tess_vetter/api/pixel.py` (pixel checks V08-V10)
- `/src/bittr_tess_vetter/api/transit_fit.py` (batman fitting)
- `/src/bittr_tess_vetter/api/recovery.py` (active star recovery)
- `/src/bittr_tess_vetter/api/catalogs.py` (catalog clients)
- `/src/bittr_tess_vetter/api/localization.py` (pixel localization)
- `/src/bittr_tess_vetter/api/wcs_localization.py` (WCS-aware localization)
- `/src/bittr_tess_vetter/api/pipeline.py` (VettingPipeline)
- `/src/bittr_tess_vetter/api/mlx.py` (MLX GPU acceleration)
- `/src/bittr_tess_vetter/api/fpp.py` (TRICERATOPS+ FPP)
- `/src/bittr_tess_vetter/api/experimental.py` (unstable APIs)
- `/tests/test_api/` (38 test files)
