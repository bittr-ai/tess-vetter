# Test Coverage & Quality Evaluation

**Repository:** bittr-tess-vetter
**Review Date:** 2026-01-14
**Focus:** Test coverage gaps for open-source release readiness

---

## Executive Summary

The test suite is comprehensive for a scientific computing package, with strong coverage of:
- Core computational primitives (BLS search, periodogram, centroid analysis)
- Vetting pipeline orchestration and check execution
- API contract stability and export resolution
- Edge case handling for numerical edge cases

**Key Coverage Gaps Identified:**
1. Limited integration tests with real TESS data (network-isolated)
2. No performance/regression benchmarks
3. Some API facade modules lack dedicated test files
4. TRICERATOPS vendor code has minimal test coverage

---

## Test Suite Overview

### Test Organization

```
tests/
  activity/          - Stellar activity detection
  api/               - API pixel localization tests
  catalogs/          - Catalog client tests (Gaia, SIMBAD, etc.)
  cli/               - CLI smoke tests
  io/                - MAST client, caching tests
  pixel/             - Centroid, aperture, TPF, difference imaging
  recovery/          - Transit recovery primitives
  test_api/          - API facade layer tests (40+ files)
  test_compute/      - Core computation tests (15+ files)
  test_integration/  - End-to-end pipeline tests
  test_support/      - Error handling, timeout tests
  test_validation/   - Vetting check tests
  transit/           - Transit timing, batman model
  utils/             - Utility function tests
  validation/        - Validation check implementations
```

### Test Count by Category

| Category | Files | Purpose |
|----------|-------|---------|
| test_api/ | ~42 | API contract, facade layer |
| test_compute/ | ~15 | Core algorithms |
| test_validation/ | ~15 | Vetting checks |
| pixel/ | ~12 | TPF/pixel analysis |
| catalogs/ | ~7 | Catalog integration |
| test_integration/ | 3 | E2E pipeline tests |

---

## Critical Path Coverage Analysis

### 1. Vetting Pipeline (GOOD)

**Tested in:** `tests/test_integration/test_pipeline_e2e.py`, `test_vet_candidate_full.py`

Coverage:
- VettingPipeline class instantiation and execution
- CheckRegistry and default check registration
- vet_candidate convenience function
- JSON serialization of results
- Check filtering via `checks=[]` parameter

```python
# Well-tested patterns:
- pipeline.run() with network=False
- result.get_result("V01") access pattern
- VettingBundleResult serialization
```

**Gap:** No tests for concurrent check execution or timeout handling in pipeline.

---

### 2. Periodogram Search (GOOD)

**Tested in:** `tests/test_api/test_periodogram_api.py`, `test_periodogram_wrappers.py`

Coverage:
- auto_periodogram() with Lomb-Scargle method
- TLS integration (conditionally skipped if not installed)
- PeriodogramResult dataclass fields

```python
# Example from test_periodogram_api.py
result = auto_periodogram(time, flux, flux_err, method="ls", min_period=1.0, max_period=5.0)
assert result.method == "ls"
assert result.best_period > 0
```

**Gap:**
- Only 1 test in `test_periodogram_api.py` (minimal)
- No tests for period grid configuration
- No edge cases for sparse/gapped data

---

### 3. BLS-Like Search (EXCELLENT)

**Tested in:** `tests/test_compute/test_bls_like_search.py` (807 lines)

Coverage:
- `_rolling_mean_circular` helper
- `_phase_bin_means` binning
- `_bls_score_from_binned_flux` scoring
- `bls_like_search_numpy` main function
- `bls_like_search_numpy_top_k` multi-candidate selection
- Transit injection/recovery tests

**Strength:** Comprehensive edge case testing including:
- Window validation (zero, negative)
- Empty bins handling
- Sparse data detection
- Determinism verification

---

### 4. Transit Fitting (GOOD)

**Tested in:** `tests/test_api/test_transit_fit_api.py`

Coverage:
- Missing batman dependency error handling
- Valid mask application with NaN filtering
- Insufficient points error handling
- MCMC fallback to optimize when emcee missing

```python
# Tests graceful degradation:
def test_fit_transit_missing_batman_returns_error():
    result = fit_transit(lc, cand, stellar, method="optimize")
    assert result.status == "error"
    assert "batman not installed" in result.error_message
```

**Gap:** No tests with actual batman/emcee dependencies (all mocked).

---

### 5. Centroid Analysis (EXCELLENT)

**Tested in:** `tests/pixel/test_centroid.py` (1385 lines)

Coverage:
- CentroidResult and TransitParams dataclasses
- Window policy configuration
- Flux-weighted centroid computation
- Transit mask generation
- Analytic and bootstrap significance methods
- Saturation detection
- Outlier rejection
- Per-cadence centroid computation
- Robust centroid estimation

**Strength:** One of the most thoroughly tested modules with:
- 20+ test classes
- Synthetic TPF generation fixtures
- Shift detection validation
- Bootstrap reproducibility tests

---

### 6. Model Competition (GOOD)

**Tested in:** `tests/test_compute/test_model_competition.py` (879 lines)

Coverage:
- ModelFit and ModelCompetitionResult dataclasses
- fit_transit_only, fit_transit_sinusoid, fit_eb_like
- run_model_competition BIC comparison
- check_period_alias detection
- compute_artifact_prior
- KNOWN_ARTIFACT_PERIODS validation

---

## Coverage Gaps

### Missing Test Files for Source Modules

The following source modules lack dedicated test files:

| Source Module | Gap Severity |
|--------------|--------------|
| `api/canonical.py` | Medium - JSON canonicalization |
| `api/experimental.py` | Low - Experimental features |
| `api/io.py` | Medium - Cache IO facade |
| `api/prefilter.py` | Medium - Prefilter logic |
| `api/sandbox_primitives.py` | Low - Development helpers |
| `api/references.py` | Low - Citation metadata |
| `compute/transit.py` | High - Transit detection primitives |
| `recovery/pipeline.py` | Medium - Recovery orchestration |
| `validation/register_defaults.py` | Low - Check registration |
| `validation/lc_checks.py` | Partial - Some checks tested elsewhere |

### Untested or Partially Tested Areas

1. **Network Integration**
   - All catalog tests mock network calls
   - No actual MAST/Gaia/SIMBAD integration tests
   - `@pytest.mark.network` tests not present

2. **TPF FITS Loading**
   - `tests/pixel/test_tpf_fits.py` exists but relies on fixtures
   - No tests with actual FITS files from MAST

3. **CLI Actual Execution**
   - Smoke tests verify `--help` and imports
   - No functional tests with synthetic inputs

4. **TRICERATOPS Vendor Code**
   - `ext/triceratops_plus_vendor/` has no direct tests
   - Integration tests mock the target creation

5. **Ephemeris Modules**
   - `ephemeris_match.py` - API tested but edge cases missing
   - `ephemeris_refinement.py` - Basic tests, no multi-sector refinement
   - `ephemeris_sensitivity_sweep.py` - Limited coverage

---

## Edge Case Testing

### Well-Covered Edge Cases

| Scenario | Test Location |
|----------|--------------|
| NaN/Inf in flux arrays | `test_transit_fit_api.py`, `test_centroid.py` |
| Empty masks | `test_centroid.py`, `test_bls_like_search.py` |
| Very short light curves | `test_vet_candidate_full.py` |
| Very long periods | `test_vet_candidate_full.py` |
| Deep transits (EB-like) | `test_vet_candidate_full.py` |
| Zero flux errors | `test_model_competition.py` |
| Insufficient in-transit points | `test_transit_fit_api.py` |

### Missing Edge Cases

| Scenario | Impact |
|----------|--------|
| Negative flux values | Could cause issues in centroid |
| Time array not sorted | No validation tested |
| Duplicate timestamps | Undefined behavior |
| Very high noise (SNR < 1) | Transit detection robustness |
| Multi-sector gaps | Partial coverage in `test_high_leverage_integration.py` |

---

## Test Quality Indicators

### Positive Patterns

1. **Fixtures for synthetic data:**
   ```python
   @pytest.fixture
   def centered_star_tpf() -> np.ndarray:
       """Create TPF with a centered Gaussian PSF."""
   ```

2. **Parametrized tests for exhaustive coverage:**
   ```python
   @pytest.mark.parametrize("name", _get_api_all())
   def test_all_exports_resolve(name: str) -> None:
   ```

3. **Reproducibility with seeded RNG:**
   ```python
   rng = np.random.default_rng(42)
   ```

4. **Contract tests for API stability:**
   ```python
   def test_check_result_schema_stable() -> None:
       result = CheckResult(id="V01", name="Test", status="ok", metrics={})
       assert hasattr(result, "id")
   ```

### Areas for Improvement

1. **Missing docstrings in some test modules**
   - `test_periodogram_api.py` has no module docstring

2. **Inconsistent test class organization**
   - Some files use classes, others use functions only

3. **Limited assertion messages**
   - Many `assert x == y` without failure context

4. **No property-based testing**
   - Could benefit from Hypothesis for numerical edge cases

---

## Conditional Skip Patterns

```python
# TLS dependency (17 occurrences across 7 files)
@pytest.mark.skipif(not TLS_AVAILABLE, reason="transitleastsquares not installed")

# MLX dependency
@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")

# TRICERATOPS vendor
@pytest.mark.skipif(not HAS_TRICERATOPS, reason="TRICERATOPS vendor not available")
```

---

## Recommendations

### Critical (Pre-Release)

1. **Add test for `compute/transit.py`**
   - `detect_transit()` and `get_transit_mask()` are heavily used
   - Only tested indirectly via integration tests

2. **Add periodogram edge case tests**
   - Gapped data, single-period data, period at Nyquist limit

3. **Validate input assumptions**
   - Add tests for unsorted time arrays
   - Add tests for negative flux handling

### High Priority

4. **Add mock-free integration test option**
   - Create small cached test fixtures for offline testing
   - Consider `conftest.py` with cached FITS snippets

5. **Test TRICERATOPS error paths**
   - Degenerate posterior handling
   - Network timeout during TRILEGAL fetch

6. **Add performance regression markers**
   - `@pytest.mark.slow` for long-running tests
   - Baseline timing checks for critical paths

### Medium Priority

7. **Property-based tests with Hypothesis**
   - Random ephemeris parameters
   - Random light curve generation with guaranteed properties

8. **Add mutation testing**
   - Verify tests catch intentional bugs

9. **Coverage measurement**
   - Add `pytest-cov` integration
   - Set coverage threshold (suggest 80%)

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Test files | ~105 |
| Test classes | ~200+ |
| Individual tests | ~1000+ |
| Skip markers | 17 |
| Source modules | ~130 |
| Modules with tests | ~100 |
| Coverage gap modules | ~30 |

**Overall Assessment:** B+

The test suite is thorough for core computational paths but has gaps in:
- API facade layer coverage
- Network integration verification
- Performance regression testing

For open-source release, recommend addressing Critical and High Priority items above.
