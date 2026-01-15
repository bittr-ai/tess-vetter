# Architecture and Maintainability Review

**Date**: 2026-01-14
**Reviewer**: Claude Opus 4.5
**Package**: bittr-tess-vetter v0.1.0

## Executive Summary

The bittr-tess-vetter codebase demonstrates **strong architectural design** with clear
separation of concerns, a well-implemented registry pattern, and thoughtful layering.
The package follows domain-driven design principles with distinct boundaries between
computational logic and I/O operations. Minor areas for improvement exist around
code duplication in result types and the complexity of the lazy API facade.

**Overall Assessment**: Ready for open-source release with minor recommendations.

---

## 1. Package Structure Overview

```
src/bittr_tess_vetter/
    __init__.py           # Minimal, exports __version__ only
    api/                  # Public facade (user-facing entry points)
    compute/              # Pure computation (periodograms, detection)
    domain/               # Core domain models (LightCurve, Target)
    pixel/                # Pixel-level analysis (TPF, WCS, centroid)
    platform/             # I/O and external services
        catalogs/         # Gaia, SIMBAD, ExoFOP clients
        io/               # MAST client, caching
        network/          # Network utilities
    transit/              # Transit-specific analysis (timing, TTV)
    recovery/             # Transit recovery from active stars
    activity/             # Stellar activity characterization
    validation/           # Vetting checks (registry, implementations)
    utils/                # Shared utilities
    cli/                  # CLI tools (MLX-specific)
    ext/                  # Vendored external code (TRICERATOPS)
```

**Strengths:**
- Clear modular organization by domain responsibility
- Explicit separation of `api/` (public) from internal modules
- `platform/` cleanly isolates I/O from pure computation
- `ext/` properly isolates vendored code with its own LICENSE

---

## 2. Separation of Concerns

### 2.1 Domain Logic vs I/O

**Rating: Excellent**

The codebase maintains exemplary separation:

| Layer | Responsibility | I/O? | Examples |
|-------|---------------|------|----------|
| `domain/` | Pure data models | None | `LightCurveData`, `TransitCandidate` |
| `compute/` | Array-in/array-out math | None | `periodogram.py`, `primitives.py` |
| `validation/` | Vetting checks (metrics) | None | `lc_checks.py`, `base.py` |
| `platform/` | External services | Yes | `mast_client.py`, `gaia_client.py` |
| `pixel/` | TPF handling | Mixed | WCS/aperture (pure) + FITS I/O |

**Evidence of good design:**

```python
# domain/lightcurve.py - Pure data container
@dataclass
class LightCurveData:
    time: NDArray[np.float64]
    flux: NDArray[np.float64]
    # ... immutable arrays, no I/O methods
```

```python
# platform/__init__.py - Clear I/O boundary
"""I/O-adjacent and platform-facing modules.
This package is intentionally separate from the core, array-in/array-out domain logic.
"""
```

### 2.2 API Facade Pattern

**Rating: Good (with caveats)**

The `api/__init__.py` implements a sophisticated lazy-loading facade using PEP 562
`__getattr__`. This reduces import-time cost but adds complexity:

**Positives:**
- 700+ lines of well-documented export mapping
- Lazy loading prevents circular import issues
- Clean `__all__` defines the golden path exports
- Aliases (`vet` -> `vet_candidate`) handled cleanly

**Concerns:**
- AST parsing of its own source at runtime is clever but fragile
- Module replacement (`sys.modules[__name__].__class__ = _APIModule`) is non-standard
- 150+ exports may be overwhelming for new users

**Recommendation:** Consider splitting into sub-facades (e.g., `api.pipeline`, `api.checks`,
`api.localization`) for more focused imports in v1.0.

---

## 3. Registry/Pipeline Pattern

### 3.1 VettingCheck Registry

**Rating: Excellent**

The check registry pattern (`validation/registry.py`) is a textbook implementation:

```python
@runtime_checkable
class VettingCheck(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def tier(self) -> CheckTier: ...
    @property
    def requirements(self) -> CheckRequirements: ...
    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult: ...

class CheckRegistry:
    def register(self, check: VettingCheck) -> None: ...
    def get(self, id: str) -> VettingCheck: ...
    def list(self) -> list[VettingCheck]: ...
    def list_by_tier(self, tier: CheckTier) -> list[VettingCheck]: ...
```

**Highlights:**
- Protocol-based (duck typing friendly, no inheritance required)
- `CheckTier` enum enables filtering (LC_ONLY, CATALOG, PIXEL, EXOVETTER)
- `CheckRequirements` dataclass makes dependencies explicit
- Lazy registration via `get_default_registry()` avoids startup cost
- Thread-safe: immutable frozen dataclasses for configs

### 3.2 Pipeline Orchestration

The `VettingPipeline` class (`api/pipeline.py`) cleanly orchestrates check execution:

```python
class VettingPipeline:
    def __init__(self, checks: list[str] | None = None, registry: CheckRegistry | None = None): ...
    def run(self, lc, candidate, ...) -> VettingBundleResult: ...
    def run_many(self, lc, candidates, ...) -> tuple[list[VettingBundleResult], list[dict]]: ...
    def describe(self, ...) -> dict[str, Any]: ...
```

**Best practices observed:**
- Checks can be selected by ID list or run all defaults
- Requirement checking with skip-on-missing (not fail-on-missing)
- Provenance tracking (`duration_ms`, `checks_run`, `pipeline_version`)
- Error isolation: one check failure doesn't stop the pipeline

---

## 4. Circular Import Analysis

### 4.1 Import Graph Health

**Rating: Good**

Cross-referenced imports across 128 files (425 total imports). Key observations:

| Import Direction | Count | Assessment |
|-----------------|-------|------------|
| `domain/` -> other | 0 | Clean dependency boundary |
| `validation/` -> `domain/` | Yes | Correct direction |
| `api/` -> `validation/` | Yes | Facade delegates correctly |
| `platform/` -> `api/` | 3 | Minor concern (see below) |

**Potential issue:**
```python
# platform/io/mast_client.py
from bittr_tess_vetter.api.lightcurve import LightCurveData, LightCurveProvenance
from bittr_tess_vetter.api.target import Target
```

The `mast_client` imports from `api/` rather than `domain/`. This creates a subtle
dependency where the platform layer depends on the API layer instead of the domain.

**Recommendation:** Consider moving `LightCurveData` to `domain/lightcurve.py` (already
exists there) and `Target` to `domain/target.py` (also exists), then have `api/` re-export
them. The current structure works but is slightly inverted.

### 4.2 TYPE_CHECKING Guards

**Rating: Excellent**

Proper use of `TYPE_CHECKING` guards throughout:

```python
# validation/registry.py
if TYPE_CHECKING:
    from bittr_tess_vetter.domain.detection import TransitCandidate
    from bittr_tess_vetter.domain.lightcurve import LightCurveData
    from bittr_tess_vetter.validation.result_schema import CheckResult
```

This pattern prevents runtime circular imports while maintaining type safety.

---

## 5. Code Duplication Analysis

### 5.1 Result Type Duplication

**Rating: Needs Attention**

Two `VettingBundleResult` classes exist:

1. `api/types.py:VettingBundleResult` - dataclass (user-facing)
2. `validation/result_schema.py:VettingBundleResult` - Pydantic BaseModel (internal)

These have overlapping but different fields:

```python
# api/types.py (dataclass)
@dataclass
class VettingBundleResult:
    results: list[CheckResult]
    provenance: dict[str, Any]
    warnings: list[str]

# validation/result_schema.py (Pydantic)
class VettingBundleResult(BaseModel):
    results: list[CheckResult]
    warnings: list[str]
    provenance: dict[str, Any]
    inputs_summary: dict[str, Any]  # Additional field
```

**Impact:** Potential confusion for users about which type is returned from which API.

**Recommendation:** Consolidate to single type. The Pydantic version is more feature-complete.

### 5.2 Shared Utility Functions

Some helper functions are duplicated across modules:

```python
# validation/lc_checks.py
def _robust_std(arr: np.ndarray) -> float:
    mad = np.median(np.abs(arr - np.median(arr)))
    return float(mad * 1.4826)

# Similar patterns in compute/primitives.py
```

**Recommendation:** Extract common statistical utilities to `compute/stats.py` or similar.

### 5.3 Check Implementation Pattern

The check wrapper pattern (`checks_lc_wrapped.py`) shows good DRY design:

```python
def _convert_legacy_result(legacy, check_id, check_name) -> CheckResult:
    # Single conversion function for all V01-V05 checks
    ...

class OddEvenDepthCheck:
    def run(self, inputs, config) -> CheckResult:
        result = check_odd_even_depth(...)
        return _convert_legacy_result(result, self.id, self.name)
```

All five LC-only checks share the same wrapper pattern without code duplication.

---

## 6. Optional Dependencies Handling

**Rating: Excellent**

The codebase handles optional dependencies gracefully:

```python
# api/__init__.py
MLX_AVAILABLE = _importlib_util.find_spec("mlx") is not None

_MLX_GUARDED_EXPORTS: set[str] = {
    "MlxTopKScoreResult", "score_fixed_period", ...
}

def __getattr__(name: str) -> Any:
    if name in _MLX_GUARDED_EXPORTS and not MLX_AVAILABLE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

Optional extras in `pyproject.toml`:
- `tls` - Transit Least Squares
- `fit` - MCMC fitting (emcee)
- `mlx` - Apple Silicon GPU
- `triceratops` - False positive probability
- `ldtk` - Limb darkening (GPL, kept optional)

This design allows:
1. Minimal install for basic use
2. Graceful degradation when extras missing
3. Clear error messages guiding users to install extras

---

## 7. Vendored Code Management

**Rating: Good**

The `ext/triceratops_plus_vendor/` directory contains a vendored TRICERATOPS fork:

```
ext/
    __init__.py
    triceratops_plus_vendor/
        LICENSE              # Proper license retention
        __init__.py
        triceratops/
            funcs.py
            likelihoods.py
            ...
```

**Positives:**
- Separate LICENSE file maintained
- Excluded from linting (`pyproject.toml` exclude)
- Excluded from mypy checking
- Clear naming indicates vendored status

**Concerns:**
- No documented versioning (which upstream commit?)
- Modifications should be tracked

**Recommendation:** Add `VENDOR_VERSION.md` documenting source commit hash and any patches.

---

## 8. Type Safety and Validation

### 8.1 Pydantic Usage

**Rating: Good**

Pydantic is used appropriately for:
- Catalog client responses (`GaiaSourceRecord`, `GaiaQueryResult`)
- Check results (`CheckResult`, `VettingBundleResult`)
- API input validation

Frozen configs prevent accidental mutation:
```python
model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
```

### 8.2 NumPy Array Immutability

**Rating: Excellent**

Domain data containers enforce immutability:
```python
# domain/lightcurve.py
def __post_init__(self) -> None:
    # Make all arrays read-only to prevent accidental mutation
    for arr in arrays.values():
        arr.flags.writeable = False
```

This prevents subtle bugs from shared mutable state in cached data.

---

## 9. Maintainability Metrics

### 9.1 File Size Distribution

| Category | Count | Avg Lines | Assessment |
|----------|-------|-----------|------------|
| < 200 lines | 85 | 120 | Good |
| 200-500 lines | 35 | 320 | Acceptable |
| 500-1000 lines | 8 | 700 | Review candidates |
| > 1000 lines | 3 | 1200+ | Consider splitting |

Large files (>1000 lines):
- `validation/lc_checks.py` (1556 lines) - Could split by check
- `platform/io/mast_client.py` (1217 lines) - Consider separating clients
- `api/__init__.py` (738 lines) - Necessary for lazy facade

### 9.2 Cyclomatic Complexity Hotspots

Based on structure analysis, complex functions exist in:
- `lc_checks.py:check_odd_even_depth()` - Deep nesting for epoch handling
- `mast_client.py:download_lightcurve()` - Many conditional paths
- `api/__init__.py:__getattr__()` - Multi-path lazy resolution

These are justified by their functionality but warrant extra test coverage.

---

## 10. Recommendations Summary

### Critical (P0)
*None identified*

### High Priority (P1)
1. **Consolidate VettingBundleResult types** - Single canonical definition
2. **Fix platform->api import direction** - Platform should import from domain

### Medium Priority (P2)
3. Extract common statistical utilities to shared module
4. Add VENDOR_VERSION.md for triceratops fork
5. Consider API sub-modules for focused imports

### Low Priority (P3)
6. Split large files (>1000 lines) where natural boundaries exist
7. Add architectural decision records (ADRs) for lazy loading pattern

---

## 11. Conclusion

The bittr-tess-vetter architecture is **well-designed for maintainability and extensibility**.
The separation between computational logic and I/O is exemplary. The registry pattern
enables easy addition of new vetting checks without modifying core code. The lazy API
facade, while complex, effectively manages a large public surface area.

The codebase is ready for open-source release. The identified issues are minor and can
be addressed incrementally post-release without breaking changes.

**Architectural Quality Score: 8.5/10**

| Criterion | Score | Notes |
|-----------|-------|-------|
| Separation of Concerns | 9/10 | Excellent layering |
| Import Hygiene | 8/10 | Minor platform->api inversion |
| Registry Pattern | 10/10 | Textbook implementation |
| Code Duplication | 7/10 | Result type duplication |
| Type Safety | 9/10 | Good Pydantic/immutability use |
| Extensibility | 9/10 | Easy to add checks/primitives |
| Documentation | 8/10 | Good docstrings, missing ADRs |

---

*Report generated by Claude Opus 4.5 for bittr-tess-vetter open-source release review.*
