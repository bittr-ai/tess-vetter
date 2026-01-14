# Test Coverage Audit: bittr-tess-vetter

**Date:** 2026-01-14
**Auditor:** Claude Code
**Scope:** Comprehensive test suite analysis for open-source release readiness

---

## 1. Test Configuration

### pytest Configuration
**Location:** `pyproject.toml` lines 61-66

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
markers = [
  "slow: long-running tests (deselect with '-m \"not slow\"')",
]
```

**Observations:**
- No pytest.ini file; configuration embedded in pyproject.toml
- Single test marker defined (`slow`)
- No coverage configuration or thresholds defined
- Missing markers for `network`, `integration`, or `skip_ci`

### conftest.py Files
- `/tests/activity/conftest.py` - Light curve fixtures (quiet star, spotted rotator, flare star)
- `/tests/transit/conftest.py` - Transit fixtures (multi-transit, TTV, odd/even depth)
- `/tests/recovery/conftest.py` - Recovery fixtures (active star + transit)
- `/tests/pixel/fixtures/conftest.py` - Synthetic TPF fixtures (blended binary, crowded field, saturation)

**Quality Note:** Fixtures are well-designed with realistic scientific scenarios.

---

## 2. Test Structure Mapping

### Source Module Coverage Matrix

| Source Module | Test File(s) | Coverage Status |
|---------------|-------------|-----------------|
| `api/__init__.py` | `test_api/test_api_top_level_exports.py`, `test_api/test_api_aliases.py` | PARTIAL - exports tested, lazy loading not fully exercised |
| `api/vet.py` | `test_api/test_vet_orchestrator_catalog_gating.py` | PARTIAL - only catalog gating tested |
| `api/lc_only.py` | `test_api/test_lc_only.py` | COVERED |
| `api/periodogram.py` | `test_api/test_periodogram_api.py`, `test_api/test_periodogram_wrappers.py` | COVERED |
| `api/timing.py` | `test_api/test_timing_api.py` | COVERED |
| `api/transit_fit.py` | `test_api/test_transit_fit_api.py`, `test_api/test_transit_fit_primitives.py` | COVERED |
| `api/types.py` | `test_api/test_types.py` | COVERED |
| `activity/primitives.py` | `tests/activity/test_primitives.py` | COVERED |
| `compute/*` | `tests/test_compute/` (15 test files) | GOOD COVERAGE |
| `domain/*` | Implicit coverage via API tests | INDIRECT |
| `pixel/*` | `tests/pixel/` (12 test files) | COVERED |
| `platform/catalogs/*` | `tests/catalogs/` (8 test files) | COVERED |
| `platform/io/*` | `tests/io/` (3 test files) | COVERED |
| `platform/network/timeout.py` | `tests/test_support/test_errors_and_timeout.py` | MINIMAL |
| `recovery/*` | `tests/recovery/` | COVERED |
| `transit/*` | `tests/transit/` (4 test files) | COVERED |
| `validation/*` | `tests/validation/` (12 test files) | COVERED |
| `cli/*` | **NO TESTS** | **CRITICAL GAP** |
| `ext/triceratops_plus_vendor/*` | Implicit via validation tests | INDIRECT |
| `errors.py` | `tests/test_support/test_errors_and_timeout.py` | MINIMAL |
| `utils/*` | `tests/utils/` (2 test files) | COVERED |

### Test File Count by Category
- **Total test files:** 95 (excluding fixtures/conftest)
- **API tests:** 36 files
- **Compute tests:** 15 files
- **Pixel tests:** 12 files
- **Validation tests:** 12 files
- **Catalog tests:** 8 files
- **Transit tests:** 4 files
- **Integration tests:** 1 file

---

## 3. Critical Coverage Gaps

### Priority 1 (CRITICAL - Must Fix Before Release)

#### 1.1 CLI Module - Zero Coverage
**Files:** `/src/bittr_tess_vetter/cli/`
- `mlx_bls_search_cli.py` (11KB)
- `mlx_bls_search_range_cli.py` (5KB)
- `mlx_quick_vet_cli.py` (15KB)
- `mlx_refine_candidates_cli.py` (10KB)
- `mlx_tls_calibration_cli.py` (11KB)

**Risk:** User-facing CLI tools with no smoke tests. Argument parsing bugs, import failures, and runtime errors will not be caught.

**Recommendation:** Add at least:
- Import smoke tests for each CLI module
- `--help` invocation tests
- Basic execution tests with synthetic data

#### 1.2 Main Workflow Integration - Limited Coverage
**File:** `/tests/test_api/test_vet_orchestrator_catalog_gating.py`

The main `vet_candidate()` function has only 2 tests focused on catalog gating (missing metadata handling). Missing:
- Full workflow with all tiers enabled
- Error propagation from individual checks
- Config pass-through verification
- Multiple candidate vetting

**Recommendation:** Add comprehensive integration tests covering:
```python
def test_vet_candidate_all_tiers_with_tpf()
def test_vet_candidate_error_in_one_check_continues()
def test_vet_candidate_custom_config_propagates()
def test_vet_candidate_provenance_metadata()
```

### Priority 2 (HIGH - Should Fix Before Release)

#### 2.1 Export Stability Tests - Incomplete
**File:** `/tests/test_api/test_api_top_level_exports.py`

Current test only verifies that 46 specific exports import without error. Missing:
- Verification that ALL `__all__` items resolve
- Detection of stale/removed exports
- Lazy-loading stress test

**Recommendation:**
```python
def test_all_exports_in_dunder_all_resolve():
    """Every item in __all__ must be importable."""
    from bittr_tess_vetter.api import __all__
    for name in __all__:
        assert hasattr(api_module, name), f"{name} in __all__ but not resolvable"
```

#### 2.2 Network/Timeout Edge Cases - Minimal
**File:** `/tests/test_support/test_errors_and_timeout.py`

Only 2 tests:
- Error envelope construction
- Invalid timeout rejection

Missing:
- Actual timeout firing
- Platform compatibility (Windows lacks SIGALRM)
- Cleanup after timeout
- Nested timeout handling

#### 2.3 Pydantic Schema Serialization
No explicit tests for:
- Model round-trip (serialize -> deserialize)
- Schema evolution compatibility
- JSON canonical form

---

## 4. Integration Test Analysis

### Existing Integration Tests
**File:** `/tests/test_integration/test_high_leverage_integration.py`

**Tests present:**
1. `test_stitch_to_period_search_to_fold_to_lc_only_checks_two_sectors_with_gap()` - Full pipeline: stitch -> TLS -> fold -> vet_lc_only
2. `test_odd_even_and_secondary_combined_integration()` - EB detection workflow
3. `test_duration_scaling_changes_in_transit_points_and_snr()` - SNR scaling validation

**Quality:** Good coverage of the primary scientific workflow. Uses synthetic light curves with injected signals.

### Missing Integration Scenarios
1. **Multi-sector pixel localization** - `localize_transit_host_multi_sector()` not integration-tested
2. **TRICERATOPS FPP calculation** - `calculate_fpp()` relies on vendored code; no end-to-end test
3. **Recovery pipeline** - `recover_transit()` with real-ish detrending + activity removal
4. **TTV track search** - `run_ttv_track_search()` workflow

---

## 5. Edge Case Test Coverage

### Empty Arrays / NaN Handling

**Files with NaN/empty tests:** 54 files contain NaN or empty-related assertions

**Example patterns found:**
```python
# In test_compute/test_primitives.py
assert np.isfinite(result.metric)

# In test_pixel/test_centroid.py
assert not np.isnan(centroid.row_mean)
```

**Coverage quality:** MODERATE - Many tests check for NaN absence but few intentionally inject NaN inputs to verify graceful handling.

### Missing Edge Cases
1. **All-NaN input arrays** - No tests verify graceful failure/handling
2. **Zero-length arrays** - Limited testing
3. **Single-point arrays** - Minimal coverage
4. **Extreme values** - No tests with `np.inf`, very large periods, etc.
5. **Invalid ephemeris** - `duration_hours=0`, `period_days<=0`

---

## 6. Prioritized Recommendations

### Before Open-Source Release (P0)

1. **Add CLI smoke tests**
   - Create `/tests/cli/test_cli_smoke.py`
   - Test each CLI module imports and responds to `--help`
   - Estimated effort: 2-3 hours

2. **Add `__all__` exhaustive export test**
   - Verify every export in `api/__init__.py.__all__` resolves
   - Estimated effort: 30 minutes

3. **Add vet_candidate workflow tests**
   - Full pipeline with LC + TPF
   - Error handling verification
   - Estimated effort: 2 hours

### High Priority (P1)

4. **Edge case hardening**
   - Add parametrized tests for NaN/empty/single-element inputs
   - Focus on public API functions
   - Estimated effort: 4 hours

5. **Timeout/network test expansion**
   - Mock-based timeout firing test
   - Platform compatibility notes
   - Estimated effort: 2 hours

6. **TRICERATOPS integration test**
   - End-to-end FPP calculation with known inputs
   - Skip if optional deps missing
   - Estimated effort: 3 hours

### Nice to Have (P2)

7. **Coverage reporting integration**
   - Add pytest-cov to dev dependencies
   - Define coverage threshold (suggest: 70% for initial release)

8. **Property-based testing for compute primitives**
   - Use hypothesis for numerical stability

9. **Regression test fixtures**
   - Capture known-good outputs for stability

---

## 7. Summary Statistics

| Metric | Value |
|--------|-------|
| Total test files | 95 |
| Source modules with direct tests | ~85% |
| Source modules with ZERO tests | CLI (5 files) |
| Integration test coverage | LIMITED |
| Edge case coverage | MODERATE |
| Export stability tests | PARTIAL |

**Overall Assessment:** The test suite has good breadth for scientific/compute code but has critical gaps in CLI testing and orchestration-level integration. The library is NOT ready for open-source release without addressing P0 items.

---

## Appendix: Key File Paths

### Untested Source Files
- `/src/bittr_tess_vetter/cli/mlx_bls_search_cli.py`
- `/src/bittr_tess_vetter/cli/mlx_bls_search_range_cli.py`
- `/src/bittr_tess_vetter/cli/mlx_quick_vet_cli.py`
- `/src/bittr_tess_vetter/cli/mlx_refine_candidates_cli.py`
- `/src/bittr_tess_vetter/cli/mlx_tls_calibration_cli.py`

### Tests to Add
- `/tests/cli/test_cli_smoke.py` (NEW)
- `/tests/test_api/test_all_exports.py` (NEW)
- `/tests/test_integration/test_vet_candidate_full.py` (NEW or extend existing)
- `/tests/test_edge_cases/test_nan_empty_inputs.py` (NEW)
