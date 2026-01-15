# Type Safety & Schemas Evaluation

**Evaluation Date:** 2026-01-14
**Reviewer Focus:** Type hints coverage, Pydantic models, `Any` usage, mypy compliance
**Verdict:** STRONG - Production-ready type safety with documented gaps

---

## Executive Summary

The `bittr-tess-vetter` codebase demonstrates **excellent type safety practices** for an open-source astronomy library. The project passes mypy with zero errors, ships with a `py.typed` marker for PEP 561 compliance, and uses Pydantic v2 models extensively for runtime validation. While `Any` types appear in expected contexts (external library interfaces, JSON serialization), the core domain models are rigorously typed.

---

## 1. Overall Type Hints Coverage

### 1.1 Coverage Statistics

| Metric | Status |
|--------|--------|
| **mypy Result** | `Success: no issues found in 174 source files` |
| **py.typed Marker** | Present at `src/bittr_tess_vetter/py.typed` |
| **Python Version** | 3.11+ (uses modern union syntax `X | None`) |
| **pyproject.toml Classifier** | `"Typing :: Typed"` declared |

### 1.2 Type Annotation Style

The codebase consistently uses modern Python 3.11+ typing conventions:

```python
# Modern union syntax (not Optional[X])
stellar: StellarParameters | None = None

# Generic numpy arrays with dtype constraints
time: NDArray[np.float64]
flux: NDArray[np.float64]
quality: NDArray[np.int32]

# Annotated types for domain validation
SNR = Annotated[float, Field(ge=0, description="Signal-to-noise ratio")]
PeriodDays = Annotated[float, Field(gt=0, description="Period, in days")]
```

### 1.3 Function Signatures

All public API functions have complete type annotations:

```python
# Example from api/transit_fit.py
def fit_transit(
    lc: LightCurve,
    candidate: Candidate,
    stellar: StellarParams,
    *,
    method: Literal["optimize", "mcmc"] = "optimize",
    fit_limb_darkening: bool = False,
    mcmc_samples: int = 2000,
    mcmc_burn: int = 500,
) -> TransitFitResult:
```

---

## 2. Pydantic Models Analysis

### 2.1 Model Inventory

The codebase defines **30+ Pydantic models** across these categories:

| Category | Models | Location |
|----------|--------|----------|
| **Domain Core** | `StellarParameters`, `Target`, `TransitCandidate`, `Detection` | `domain/` |
| **Periodogram** | `PeriodogramPeak`, `PeriodogramResult` | `domain/detection.py` |
| **Validation Results** | `CheckResult`, `VettingBundleResult` | `validation/result_schema.py` |
| **Catalog Clients** | `GaiaSourceRecord`, `SimbadQueryResult`, `SourceRecord` | `platform/catalogs/` |
| **Error Handling** | `ErrorEnvelope` | `errors.py` |
| **Pixel Analysis** | `PixelVetReport` | `pixel/report.py` |

### 2.2 Pydantic Configuration Patterns

**Frozen immutable models (recommended pattern):**
```python
class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

class StellarParameters(FrozenModel):
    teff: float | None = Field(default=None, ge=0, description="Effective temp (K)")
    logg: float | None = Field(default=None, description="Surface gravity")
    radius: float | None = Field(default=None, ge=0, description="Solar radii")
```

**Validation constraints using `Field`:**
```python
class TransitCandidate(FrozenModel):
    period: PeriodDays  # Annotated type with gt=0 constraint
    depth: float = Field(gt=0, le=1, description="Transit depth (fractional)")
    snr: SNR  # Annotated type with ge=0 constraint
```

**Model validators:**
```python
@model_validator(mode="after")
def _validate_n_periods_searched(self) -> PeriodogramResult:
    if self.method != "tls" and self.n_periods_searched < 1:
        raise ValueError("n_periods_searched must be >= 1 for non-TLS methods")
    return self
```

### 2.3 Strengths

1. **Consistent `frozen=True, extra="forbid"`** prevents accidental mutation and catches typos
2. **Domain-specific annotated types** (`SNR`, `FAP`, `PeriodDays`) improve self-documentation
3. **Validation constraints on construction** catch invalid data early
4. **Clean separation** between Pydantic models (validation) and dataclasses (pure data)

---

## 3. `Any` Type Usage Analysis

### 3.1 Usage Categories

The codebase contains ~300 uses of `Any`. Analysis by category:

| Category | Count | Justification | Risk Level |
|----------|-------|---------------|------------|
| **External library interfaces** | ~80 | lightkurve, MLX, astropy WCS | Low |
| **JSON serialization** | ~100 | `dict[str, Any]` for to_dict() | Low |
| **Cache interfaces** | ~15 | Generic pickle storage | Low |
| **Config dicts** | ~40 | User-provided config | Medium |
| **NDArray[Any]** | ~30 | API boundary types | Medium |
| **CLI/MLX code** | ~50 | Platform-specific, excluded from mypy | N/A |

### 3.2 Justified `Any` Usage

**External library return types (unavoidable):**
```python
# lightkurve objects are untyped
def _ensure_lightkurve(self) -> Any:
    ...

# MLX arrays when mlx is optional
def score_fixed_period(...) -> Any:
    ...
```

**JSON serialization (intentional flexibility):**
```python
def to_dict(self) -> dict[str, Any]:
    """JSON-serializable output for downstream consumers."""
```

### 3.3 Areas for Improvement

**1. Public API `NDArray[Any]` types:**
```python
# Current (api/types.py)
class LightCurve:
    time: NDArray[Any]
    flux: NDArray[Any]

# Improvement: Could use NDArray[np.floating[Any]] for more precision
```

**2. Config dictionaries:**
```python
# Current
context: dict[str, Any] | None = None

# Improvement: Could define TypedDict for known config keys
class VettingContext(TypedDict, total=False):
    stellar: StellarParameters
    tpf: TPFData
    ...
```

**3. TypedDict in compute layer:**
The `HypothesisScore` TypedDict uses `total=False` with many optional fields. This is correct but could benefit from splitting into required vs optional protocols.

---

## 4. mypy Configuration

### 4.1 Configuration (pyproject.toml)

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = false
warn_unused_configs = true
check_untyped_defs = true
disallow_untyped_decorators = false
exclude = ["src/bittr_tess_vetter/ext/triceratops_plus_vendor/"]
```

**Analysis:**
- `check_untyped_defs = true` - Good, catches errors in untyped functions
- `warn_return_any = false` - Relaxed, appropriate for library with external deps
- Vendor code properly excluded

### 4.2 Module-Level Overrides

```toml
# Modules with type issues deferred to P3 backlog
[[tool.mypy.overrides]]
module = [
    "bittr_tess_vetter.api.pixel_localize",
    "bittr_tess_vetter.api.transit_fit",
    "bittr_tess_vetter.compute.periodogram",
    ...
]
ignore_errors = true
```

**15 modules have `ignore_errors = true`** - these are documented as P3 backlog items. This is acceptable for initial open-source release but should be tracked for resolution.

### 4.3 External Library Stubs

All major external dependencies have `ignore_missing_imports = true`:
- numpy, scipy, astropy, requests (well-typed, stubs available)
- lightkurve, transitleastsquares, batman (no stubs, `Any` expected)
- MLX (Apple-specific, conditional import)

---

## 5. TypedDict Usage

The codebase uses `TypedDict` for structured dictionary interfaces:

```python
class HypothesisScore(TypedDict, total=False):
    """Per-hypothesis scoring result from PRF-based fitting."""
    source_id: str
    source_name: str
    fit_loss: float
    delta_loss: float
    rank: int
    # Extended fields...

class ReferenceSource(TypedDict, total=False):
    """Nearby source for localization."""
    source_id: str
    ra_deg: float
    dec_deg: float
    tmag: float
```

**Good practice:** Using `total=False` for optional fields with clear documentation.

---

## 6. Dataclass vs Pydantic Strategy

The codebase has a clear separation:

| Use Case | Type | Example |
|----------|------|---------|
| **Domain models with validation** | Pydantic | `StellarParameters`, `TransitCandidate` |
| **Pure data containers (frozen)** | dataclass | `LightCurveData`, `PRFParams`, `TransitTime` |
| **Result types (immutable)** | dataclass | `TTVResult`, `OddEvenResult`, `ActivityResult` |
| **API facade types** | dataclass | `LightCurve`, `Ephemeris`, `Candidate` |

This is a good architectural choice:
- Pydantic for validation at boundaries
- Dataclasses for internal performance-critical data
- `frozen=True` on both for immutability

---

## 7. Recommendations

### 7.1 Pre-Release (Required)

| Issue | Status |
|-------|--------|
| mypy passes with zero errors | DONE |
| py.typed marker present | DONE |
| Pydantic v2 models validated | DONE |
| Core domain types fully annotated | DONE |

### 7.2 Post-Release Improvements (P2-P3)

1. **Resolve 15 modules with `ignore_errors = true`** in mypy config
2. **Tighten `NDArray[Any]` to `NDArray[np.floating[Any]]`** in public API types
3. **Add TypedDict for common config patterns** (VettingContext, PipelineConfig)
4. **Consider Protocol types** for VettingCheck interface (already using `@runtime_checkable`)
5. **Add return type annotations to `to_dict()` methods** (currently `dict[str, Any]`)

### 7.3 Documentation

Consider adding a typing guide to docs:
- Explain the dataclass vs Pydantic strategy
- Document expected types for callback/plugin interfaces
- List external libraries that return `Any`

---

## 8. Files Examined

Key files reviewed for this analysis:

- `/src/bittr_tess_vetter/api/types.py` - Public API types
- `/src/bittr_tess_vetter/domain/detection.py` - Core domain models
- `/src/bittr_tess_vetter/domain/lightcurve.py` - LightCurveData with strict dtype checks
- `/src/bittr_tess_vetter/domain/target.py` - StellarParameters, Target
- `/src/bittr_tess_vetter/validation/result_schema.py` - CheckResult, VettingBundleResult
- `/src/bittr_tess_vetter/validation/registry.py` - VettingCheck Protocol
- `/src/bittr_tess_vetter/compute/primitives.py` - Pure compute functions
- `/src/bittr_tess_vetter/compute/prf_schemas.py` - PRF parameter dataclasses
- `/src/bittr_tess_vetter/compute/pixel_host_hypotheses.py` - TypedDict usage
- `/src/bittr_tess_vetter/api/__init__.py` - Lazy loading with type safety
- `/src/bittr_tess_vetter/errors.py` - ErrorEnvelope model
- `/pyproject.toml` - mypy configuration

---

## 9. Conclusion

**Verdict: READY FOR OPEN-SOURCE RELEASE**

The `bittr-tess-vetter` codebase demonstrates strong type safety practices:

1. **Zero mypy errors** with reasonable configuration
2. **PEP 561 compliant** with py.typed marker
3. **Well-structured Pydantic models** with validation constraints
4. **Clear dataclass/Pydantic separation** following best practices
5. **Justified `Any` usage** for external library boundaries
6. **Modern Python 3.11+ typing syntax** throughout

The 15 modules with `ignore_errors = true` are documented as P3 backlog items and do not block release. The codebase provides a solid foundation for type-safe integrations with downstream applications.
