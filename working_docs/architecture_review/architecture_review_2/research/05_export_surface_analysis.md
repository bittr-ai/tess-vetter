# Export Surface Analysis

## Executive Summary

The `bittr_tess_vetter.api` module exports **229 symbols** via `__all__`, with an additional **62 undocumented but accessible exports** through the lazy loader. The lazy loading implementation is robust but the first access pulls in heavy dependencies. There are clear deprecation patterns in place but no formal versioning strategy.

---

## 1. Export Size and Categories

**Total documented exports (`__all__`):** 229

| Category | Count | Examples |
|----------|-------|----------|
| Data Types | 54 | `Ephemeris`, `LightCurve`, `Candidate`, `TPFStamp` |
| Compute Functions | 34 | `compute_*`, `run_*` functions |
| Result Types | 18 | `CheckResult`, `VettingBundleResult`, `TransitFitResult` |
| Analysis Functions | 8 | `fit_transit`, `measure_transit_times`, `analyze_ttvs` |
| Orchestrators | 6 | `vet_candidate`, `vet_lc_only`, `vet_pixel`, `vet_catalog` |
| Config/Presets | 4 | `PerformancePreset`, `TTVSearchBudget`, `TriceratopsFppPreset` |
| Cap Utilities | 4 | `cap_top_k`, `cap_neighbors`, `cap_plots` |
| Constants | 7 | `DEFAULT_TOP_K_CAP`, `HARMONIC_RATIOS`, `MARGIN_RESOLVE_THRESHOLD` |
| Other Functions | 100 | Remaining helper functions |

**MLX-guarded exports:** 8 additional symbols when MLX is available:
- `MlxTopKScoreResult`, `MlxT0RefinementResult`
- `smooth_box_template`, `score_fixed_period`, `score_fixed_period_refine_t0`
- `score_top_k_periods`, `integrated_gradients`

---

## 2. Export Resolution Testing

### Documented Test Coverage

**File:** `/tests/test_api/test_api_top_level_exports.py`

Tests that key exports are importable from top-level `bittr_tess_vetter.api`:

```python
def test_api_top_level_exports_import() -> None:
    from bittr_tess_vetter.api import (
        ConsistencyClass, ControlType, EphemerisEntry, EvidenceEnvelope,
        GhostFeatures, PhaseShiftEvent, TransitTime, analyze_ttvs, ...
    )
```

**Coverage:** Validates ~47 representative exports resolve correctly.

**Finding:** No comprehensive test that iterates over `__all__` to validate ALL exports.

**Recommendation:** Add a parametrized test:
```python
@pytest.mark.parametrize("name", __all__)
def test_all_exports_resolve(name):
    assert hasattr(api, name)
```

---

## 3. Alias Testing

**File:** `/tests/test_api/test_api_aliases.py`

Tests the short alias system:

```python
def test_top_level_aliases_import_and_resolve() -> None:
    from bittr_tess_vetter.api import (
        aperture_family_depth_curve, localize, periodogram, vet, vet_candidate
    )
    assert callable(vet)
    assert vet is vet_candidate  # Identity check
    assert callable(periodogram)
    assert callable(localize)
    assert callable(aperture_family_depth_curve)
```

**Aliases defined in `_ALIASES`:**
| Alias | Target |
|-------|--------|
| `vet` | `vet_candidate` |
| `periodogram` | `run_periodogram` |
| `localize` | `localize_transit_source` |
| `aperture_family_depth_curve` | `compute_aperture_family_depth_curve` |

**Finding:** Alias identity is verified (`vet is vet_candidate`).

---

## 4. Lazy Loading Implementation

### Architecture

The module uses PEP 562 (`__getattr__`) for lazy loading:

```python
def __getattr__(name: str) -> Any:
    # 1. Handle MLX_AVAILABLE constant
    if name == "MLX_AVAILABLE":
        return MLX_AVAILABLE

    # 2. Guard MLX exports when MLX unavailable
    if name in _MLX_GUARDED_EXPORTS and not MLX_AVAILABLE:
        raise AttributeError(...)

    # 3. Resolve aliases first
    alias_target = _ALIASES.get(name)
    if alias_target is not None:
        value = getattr(sys.modules[__name__], alias_target)
        globals()[name] = value  # Cache for future access
        return value

    # 4. Parse TYPE_CHECKING block for import locations
    exports = _get_export_map()
    target = exports.get(name)
    if target is None:
        raise AttributeError(...)

    # 5. Import and cache
    module, attr = target
    mod = importlib.import_module(module)
    value = getattr(mod, attr)
    globals()[name] = value
    return value
```

### `_APIModule` Wrapper

A custom module class ensures aliases always resolve to callables even when submodules share the same name:

```python
class _APIModule(_types.ModuleType):
    def __getattribute__(self, name: str) -> Any:
        if name in _ALIASES:
            return getattr(self, _ALIASES[name])
        return super().__getattribute__(name)
```

### Performance

| Operation | Time |
|-----------|------|
| Initial `import bittr_tess_vetter.api` | 6ms |
| First access to `Ephemeris` (type) | +935 modules loaded |
| First access to `run_periodogram` | +3 modules loaded |
| Access to alias `vet` | +1 module loaded |

**Finding:** Types trigger heavy dependency loading (numpy, astropy chains). Compute functions are more isolated.

### Robustness Assessment

| Aspect | Status | Notes |
|--------|--------|-------|
| AST parsing for export map | Robust | Handles nested `if TYPE_CHECKING` blocks |
| Caching (`globals()[name] = value`) | Robust | Prevents repeat parsing |
| Alias collision handling | Robust | `_APIModule` wrapper protects aliases |
| MLX availability guard | Robust | Clean error when MLX unavailable |
| Error messages | Good | Clear `AttributeError` with module context |

---

## 5. Undocumented Exports

**62 symbols accessible via lazy loader but NOT in `__all__`:**

These are in `TYPE_CHECKING` imports but excluded from `__all__`:

| Category | Examples |
|----------|----------|
| Internal types | `LightCurveData`, `StellarParameters`, `Target`, `Detection` |
| Cache types | `TPFCache`, `TPFRef`, `TPFFitsCache`, `TPFFitsRef` |
| Refinement API | `EphemerisRefinementCandidate`, `EphemerisRefinementConfig` |
| FPP internals | `CalculateFppInput`, `FppResult`, `prefetch_trilegal_csv` |
| Detrending | `flatten`, `normalize_flux`, `sigma_clip`, `wotan_flatten` |
| Dilution | `DilutionScenario`, `HostHypothesis`, `PhysicsFlags` |
| Sandbox | `AstroPrimitives`, `astro`, `box_model`, `fold` |
| Stitching | `stitch_lightcurves`, `StitchedLC`, `SectorDiagnostics` |

**Risk Assessment:**

These are intentionally semi-private (available for power users but not advertised):
- Good: Types like `LightCurveData` are useful for host integration
- Risky: No stability guarantee; could break without notice
- Missing: No documentation on which are stable vs experimental

---

## 6. API Versioning and Deprecation Strategy

### Current Deprecation Pattern

**File:** `src/bittr_tess_vetter/api/vet.py` and `src/bittr_tess_vetter/api/lc_only.py`

```python
if policy_mode != "metrics_only":
    warnings.warn(
        "`policy_mode` is deprecated and ignored; bittr-tess-vetter always returns "
        "metrics-only results. Move interpretation/policy decisions to astro-arc-tess "
        "validation (tess-validate).",
        category=FutureWarning,
        stacklevel=2,
    )
```

**Test coverage:** `/tests/test_api/test_policy_mode_deprecated.py`

### Versioning

**Package version:** `__version__ = "0.0.1"` in `src/bittr_tess_vetter/__init__.py`

**Provenance tracking:** `vet_candidate` records version in output:
```python
def _get_package_version() -> str:
    return importlib.metadata.version("bittr-tess-vetter")
```

### Missing Versioning Elements

| Missing | Impact |
|---------|--------|
| API version separate from package version | Cannot evolve API independently |
| Semantic versioning on exports | No stability promises |
| `@deprecated` decorator | Ad-hoc warning patterns |
| Changelog/migration guides | Users cannot plan upgrades |

---

## 7. Internal Helpers Exposure Risk

The `api/__init__.py` module properly protects internal helpers:

```python
# These remain private (underscore prefix):
def _iter_stmts_in_order(stmts: list[ast.stmt]) -> list[ast.stmt]: ...
def _get_export_map() -> dict[str, tuple[str, str]]: ...
_EXPORT_MAP: dict[str, tuple[str, str]] | None = None
_ALIASES: dict[str, str] = { ... }
_MLX_GUARDED_EXPORTS: set[str] = { ... }
class _APIModule(_types.ModuleType): ...
```

**No internal helpers are exposed in `__all__`.**

However, some undocumented exports (see section 5) may be considered internal:
- `prepare_recovery_inputs` - internal prep function
- `prefetch_trilegal_csv` - cache warmup utility
- `load_cached_triceratops_target` - cache internals

---

## 8. Recommendations

### High Priority

1. **Add exhaustive export resolution test:**
   ```python
   @pytest.mark.parametrize("name", api.__all__)
   def test_export_resolves(name):
       obj = getattr(api, name)
       assert obj is not None
   ```

2. **Document the undocumented exports:**
   - Create a `STABLE.md` file listing guaranteed stable exports
   - Mark experimental exports with `@experimental` decorator
   - Consider moving internal utilities to a separate `api._internal` namespace

3. **Formalize deprecation infrastructure:**
   ```python
   from bittr_tess_vetter.api._deprecation import deprecated

   @deprecated("Use vet_candidate() instead", removal="1.0.0")
   def legacy_function(): ...
   ```

### Medium Priority

4. **Reduce first-access latency:**
   - The first access to types loads 935 modules
   - Consider splitting `types.py` to avoid transitive imports

5. **Add API versioning:**
   ```python
   __api_version__ = "2.0"  # Separate from package version
   ```

6. **Create stability tiers:**
   - Tier 1: Core (Ephemeris, LightCurve, vet_candidate) - never break
   - Tier 2: Extended (compute_*, Result types) - deprecation cycle
   - Tier 3: Advanced (cache internals, refinement) - may change

### Low Priority

7. **Consider `__all__` automation:**
   - Generate `__all__` from TYPE_CHECKING imports with explicit exclusions
   - Reduce drift between documented and accessible exports

8. **Add IDE hints:**
   - Ensure `.pyi` stub files exist for the public API
   - Improve autocomplete experience for users

---

## Summary Table

| Metric | Value | Assessment |
|--------|-------|------------|
| Documented exports | 229 | Large but organized |
| Undocumented exports | 62 | Risk: no stability guarantee |
| Export resolution tests | Partial | Missing comprehensive test |
| Alias tests | Present | Identity verified |
| Lazy loading | Robust | AST-based, cached, guarded |
| First access latency | High | 935 modules for types |
| Deprecation pattern | Present | Ad-hoc warnings.warn |
| API versioning | None | Package version only |
| Internal helper exposure | None | Properly underscore-prefixed |
