# Docstring Coverage Audit Report

**Date:** 2026-01-14
**Scope:** `src/bittr_tess_vetter/api/` (68 Python files, ~270 public functions)

---

## Executive Summary

The `api/` module demonstrates **excellent docstring coverage** with consistent Google-style formatting and comprehensive academic citations. An estimated **92-95%** of public functions have docstrings. The package is in strong shape for open-source release, with a few targeted gaps to address.

---

## 1. Docstring Coverage Statistics

### Overall Coverage in `api/`

| Category | Count | With Docstring | Coverage |
|----------|-------|----------------|----------|
| Public functions (`def func_name`) | ~110 | ~102 | **93%** |
| Private functions (`def _func_name`) | ~45 | ~30 | 67% |
| Classes/Dataclasses | ~35 | ~34 | **97%** |
| Modules (module-level docstrings) | 68 | 68 | **100%** |

### Files with Exemplary Documentation

- `vet.py` - Full docstring with Args, Returns, Example, References
- `lc_only.py` - All 6 public functions fully documented with Novelty/References
- `pixel.py` - Comprehensive docstrings with algorithm config details
- `transit_fit.py` - Complete parameter documentation, 15+ references
- `timing.py` - Academic-quality with equation citations
- `recovery.py` - Detailed workflow documentation
- `fpp.py` - TRICERATOPS+ multi-band feature fully documented
- `references.py` - Thorough module-level usage examples

### Files with Thin/Missing Documentation

| File | Issue | Priority |
|------|-------|----------|
| `caps.py` | Only module docstring, delegates to utils | Low |
| `ephemeris_match.py` | Module docstring only, re-exports | Low |
| `stitch.py` | Private helpers lack docstrings | Medium |
| `mlx.py` | MLX functions missing Args/Returns | Medium |
| `canonical.py` | Private helpers undocumented | Low |
| `ttv_track_search.py` | Functions missing full docstrings | High |
| `evidence_contracts.py` | `compute_code_hash` sparse | Medium |

---

## 2. Docstring Format Analysis

### Observed Format: Google Style (Consistent)

The codebase consistently uses **Google-style docstrings** with the following structure:

```python
def function_name(param1: type, param2: type) -> ReturnType:
    """Short summary line.

    Extended description with context about the algorithm,
    workflow, or scientific motivation.

    Args:
        param1: Description of parameter.
        param2: Description with units/defaults noted.

    Returns:
        Description of return value(s).

    Raises:
        ValueError: When condition is not met.
        ImportError: When optional dependency is missing.

    Example:
        >>> from bittr_tess_vetter.api import ...
        >>> result = function_name(...)
        >>> print(result)

    Novelty: standard | new

    References:
        [1] Author et al. YYYY, Journal Vol, Page (bibcode)
            Section X: Specific algorithm or equation
    """
```

### Format Compliance Rate

| Element | Presence Rate | Notes |
|---------|---------------|-------|
| Summary line | 100% | Always present |
| Args section | 95% | Rare omissions in thin facades |
| Returns section | 90% | Some facades just document behavior |
| Raises section | 70% | Only where exceptions are raised |
| Example section | 40% | Present in major entry points |
| Novelty tag | 85% | Consistently used for research tracking |
| References | 80% | Excellent for scientific functions |

### Recommendation

No format change needed. Continue with Google style. Consider adding examples to the 8-10 most-used entry points for open-source users.

---

## 3. Critical Functions Missing Docstrings

### High Priority (Public API Entry Points)

| Function | File | Risk |
|----------|------|------|
| `run_ttv_track_search` | `ttv_track_search.py` | Complex multi-step search; needs workflow doc |
| `run_ttv_track_search_for_candidate` | `ttv_track_search.py` | Higher-level wrapper, no docstring |
| `localize_transit_source` | `wcs_localization.py` | Has docstring but very terse |
| `compute_difference_image_centroid_diagnostics` | `wcs_localization.py` | Complex return, needs more detail |

### Medium Priority (Internal but Host-Facing)

| Function | File | Issue |
|----------|------|-------|
| `prepare_recovery_inputs` | `recovery.py` | Missing Args description |
| `stitch_lightcurves` | `stitch.py` | Good docstring but missing Returns detail |
| `compute_code_hash` | `evidence_contracts.py` | Very sparse docstring |
| `_compute_tpf_cadence_summary` | `pixel_localize.py` | Private but complex return |

### Low Priority (Private/Utility)

- `_normalize`, `_quantize_float` in `canonical.py`
- `_iter_stmts_in_order`, `_get_export_map` in `__init__.py`
- `_infer_cadence_seconds`, `_compute_mad` in `stitch.py`

---

## 4. Type Hints vs. Prose Documentation Analysis

### Type Hints Are Comprehensive

The codebase has excellent type hint coverage. All public functions use:
- Parameter type annotations
- Return type annotations
- `Optional` and `Literal` types appropriately
- Generic types (`list[float]`, `dict[str, Any]`)

### Where Prose Would Help

| Type | Location | Suggested Enhancement |
|------|----------|----------------------|
| `MlxTopKScoreResult` | `mlx.py` | Explain score interpretation |
| `LocalizationResult` | `wcs_localization.py` | Explain verdict confidence |
| `PRFFitResult` | `pixel_prf.py` (re-export) | Link to usage guide |
| `EvidenceEnvelope` | `evidence_contracts.py` | Explain immutability constraints |
| `PreparedRecoveryInputs` | `recovery.py` | Explain when to use vs raw arrays |

### Dataclass Attribute Documentation

Excellent pattern already in use:

```python
@dataclass(frozen=True)
class TransitFitResult:
    """Result of physical transit model fit.

    Attributes:
        fit_method: Method used ("optimize" or "mcmc")
        rp_rs: Planet-to-star radius ratio
        ...
    """
```

All major result types (`RecoveryResult`, `ActivityResult`, `TTVResult`, etc.) follow this pattern.

---

## 5. Documentation Infrastructure

### Sphinx Configuration: **NOT PRESENT**

There is no `docs/` folder or `conf.py`. This is a gap for open-source release.

### Current Documentation

| Asset | Location | Status |
|-------|----------|--------|
| README.md | Root | Good overview, needs API section |
| THIRD_PARTY_NOTICES.md | Root | License attributions |
| REFERENCES.md | Not found | Does not exist |
| Module docstrings | All `api/` files | Complete |
| Inline docstrings | Most functions | 93% coverage |

### Recommendation: Create Minimal Sphinx Setup

For open-source release, add:

1. `docs/conf.py` with autodoc
2. `docs/api.rst` with autosummary for `bittr_tess_vetter.api`
3. `docs/index.rst` linking to README
4. GitHub Actions to build and publish to Read the Docs

---

## 6. REFERENCES.md Analysis

**File does not exist.** However, the `api/references.py` module provides a comprehensive programmatic reference registry.

### Existing Reference Infrastructure

```python
# 52+ academic references defined as typed constants
THOMPSON_2018 = reference(Reference(
    id="thompson_2018",
    bibcode="2018ApJS..235...38T",
    title="Planetary Candidates Observed by Kepler. VIII...",
    authors=("Thompson, S.E.", "Coughlin, J.L.", ...),
    journal="ApJS 235, 38",
    year=2018,
    doi="10.3847/1538-4365/aab4f9",
    note="DR25 Robovetter: odd/even, secondary eclipse...",
))
```

### Available Utilities

- `generate_bibtex(refs)` - Generate BibTeX for export
- `generate_bibliography_markdown()` - Generate markdown bibliography
- `collect_module_citations(module)` - Introspect function references
- `@cites(cite(REF, context))` - Decorator to attach refs to functions

### Recommendation: Generate REFERENCES.md

Add a build step or script to generate `REFERENCES.md` from `api/references.py`:

```python
from bittr_tess_vetter.api.references import generate_bibliography_markdown
with open("REFERENCES.md", "w") as f:
    f.write("# References\n\n")
    f.write(generate_bibliography_markdown())
```

---

## 7. Recommendations for Open-Source Release

### Critical (Must Fix)

1. **Document `ttv_track_search.py` functions** - These are exposed in `__all__` but lack docstrings
2. **Add brief Returns section to MLX functions** - Users need to understand score semantics
3. **Create `REFERENCES.md`** - Auto-generate from the references registry

### High Priority

4. **Add Examples to entry points:**
   - `run_periodogram()`
   - `calculate_fpp()`
   - `localize_transit_source()`
   - `fit_transit()`

5. **Create minimal Sphinx docs:**
   - `docs/conf.py` with autodoc + napoleon
   - API reference auto-generated from docstrings
   - Deploy to Read the Docs

### Medium Priority

6. **Enhance type documentation:**
   - Add "See Also" sections linking related functions
   - Cross-reference dataclass types in Returns sections

7. **Add docstrings to internal helpers** in:
   - `stitch.py` private functions
   - `canonical.py` normalization helpers

### Low Priority

8. **Standardize Novelty tags:**
   - Some functions use "standard", others omit it
   - Could be enforced with a linter

9. **Add Changelog section to README**
   - Track breaking changes for API consumers

---

## 8. Sample Documentation Gaps

### Gap 1: `ttv_track_search.py:run_ttv_track_search`

**Current:**
```python
def run_ttv_track_search(...) -> TTVTrackSearchResult:
    # No docstring
```

**Suggested:**
```python
def run_ttv_track_search(
    *,
    time: np.ndarray,
    flux: np.ndarray,
    ...
) -> TTVTrackSearchResult:
    """Search for TTV-induced period variations across observing windows.

    This function implements a grid search over period perturbations
    within identified observing windows to detect transit timing
    variations that might indicate additional planets.

    Args:
        time: Time array in BTJD
        flux: Normalized flux values
        ...

    Returns:
        TTVTrackSearchResult with best-fit period track and significance

    Novelty: new

    References:
        [1] Holman & Murray 2005 - TTV theory
    """
```

### Gap 2: `mlx.py:score_fixed_period`

**Current:** Has `@cites` but no Args/Returns.

**Suggested addition:**
```python
"""Score a fixed period using MLX-accelerated template matching.

Args:
    time: Time array (MLX array or numpy)
    flux: Normalized flux values
    flux_err: Optional flux uncertainties
    period_days: Orbital period to score
    t0_btjd: Reference transit epoch
    duration_hours: Transit duration

Returns:
    Score result with depth, SNR, and chi-squared metrics.
    Higher scores indicate better match to transit template.
"""
```

---

## 9. Conclusion

The `bittr-tess-vetter` API has **industry-leading docstring coverage** for a scientific Python package. The combination of:

- Consistent Google-style formatting
- Comprehensive type hints
- Academic citation infrastructure (`@cites` decorator)
- Module-level docstrings

...puts it well above average for astronomy software.

**Before open-source release**, address:
1. 5-6 critical functions missing docstrings
2. Create auto-generated REFERENCES.md
3. Set up minimal Sphinx documentation

Estimated effort: **1-2 days** of documentation work.
