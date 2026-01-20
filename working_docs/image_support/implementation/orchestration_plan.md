# Plotting Feature - Agent Orchestration Plan

**Version**: 1.0.0
**Date**: 2026-01-20
**Target**: 100% agent-driven implementation

---

## Overview

This plan orchestrates multiple autonomous agents to implement the plotting feature. Agents run with `background=false` (blocking) and communicate through file artifacts and git commits.

### Execution Model
- **Sequential phases** with validation gates
- **Parallel agents** within phases where dependencies allow
- **Atomic commits** after each validated sub-phase
- **Rollback on failure** via git reset

---

## Agent Roles

### 1. Foundation Agent
**Purpose**: Establish module skeleton, core utilities, style system

**Capabilities**:
- Create new Python modules
- Modify pyproject.toml
- Write comprehensive test fixtures

**Constraints**:
- Does NOT implement plot functions (only infrastructure)
- Does NOT modify validation check implementations

---

### 2. Check Update Agent
**Purpose**: Add `plot_data` to CheckResult for specific checks

**Capabilities**:
- Modify validation check implementations
- Add plot_data generation code
- Ensure JSON serializability

**Constraints**:
- Does NOT create plotting functions
- Does NOT modify core check logic (only adds output)
- Must preserve existing test compatibility

---

### 3. Plot Implementation Agent
**Purpose**: Implement specific plot functions

**Capabilities**:
- Create plot functions in plotting modules
- Write unit tests for plot functions
- Follow signature conventions strictly

**Constraints**:
- Does NOT modify check implementations
- Does NOT modify core utilities once established
- Must use `extract_plot_data()` for data access

---

### 4. Validation Agent
**Purpose**: Verify code quality, test passage, no regressions

**Capabilities**:
- Run pytest
- Run ruff lint
- Run mypy type check
- Verify import structure

**Outputs**:
- PASS/FAIL status
- Failure details for rollback

---

### 5. Integration Agent
**Purpose**: Wire up exports, run integration tests

**Capabilities**:
- Modify `plotting/__init__.py` exports
- Modify `api/__init__.py` re-exports
- Write integration tests

**Constraints**:
- Runs AFTER all plot functions implemented
- Requires Validation Agent approval

---

### 6. Commit Agent
**Purpose**: Create atomic, well-documented commits

**Capabilities**:
- Stage specific files
- Write commit messages
- Tag milestones

**Constraints**:
- ONLY commits after Validation Agent PASS
- Never force-pushes
- Never amends commits

---

### 7. Documentation Agent
**Purpose**: Update tutorials, create gallery

**Capabilities**:
- Modify notebook tutorials
- Create example gallery images
- Write docstrings

**Constraints**:
- Runs AFTER Integration Agent completes
- Does NOT modify implementation code

---

## Execution Phases

### Phase 1: Foundation
**Duration**: ~30 minutes
**Parallelization**: None (must be sequential)

#### Step 1.1: Foundation Agent - Module Skeleton
**Files Created**:
- `src/bittr_tess_vetter/plotting/__init__.py`
- `src/bittr_tess_vetter/plotting/_core.py`
- `src/bittr_tess_vetter/plotting/_styles.py`
- `tests/test_plotting/__init__.py`
- `tests/test_plotting/conftest.py`

**Exit Criteria**:
- [ ] `from bittr_tess_vetter.plotting import MATPLOTLIB_AVAILABLE` works
- [ ] `from bittr_tess_vetter.plotting._core import ensure_ax, style_context` works
- [ ] pytest discovers test_plotting directory

#### Step 1.2: Validation Agent
**Command**: `pytest tests/test_plotting/ -v && ruff check src/bittr_tess_vetter/plotting/`

**On FAIL**: Foundation Agent fixes issues
**On PASS**: Proceed to Step 1.3

#### Step 1.3: Commit Agent
**Commit**: `feat(plotting): add module skeleton and core utilities`

**Files**: All from Step 1.1

---

### Phase 2: First Plot (V01 Proof of Concept)
**Duration**: ~45 minutes
**Parallelization**: Check Update and Plot Implementation can start in parallel after Foundation

#### Step 2.1: Check Update Agent - V01 plot_data
**Files Modified**:
- `src/bittr_tess_vetter/validation/lc_checks.py` (check_odd_even_depth)

**Changes**:
```python
# At end of check_odd_even_depth, before return:
plot_data = {
    "version": 1,
    "odd_epochs": [int(e) for e in odd_epoch_indices[:50]],
    "odd_depths_ppm": [float(d) * 1e6 for d in odd_depths[:50]],
    # ... (see abbreviated_spec.md Section 8)
}
return ok_result(..., raw={"plot_data": plot_data})
```

**Exit Criteria**:
- [ ] V01 check returns `raw["plot_data"]` with all required keys
- [ ] Existing V01 tests still pass

#### Step 2.2: Plot Implementation Agent - plot_odd_even
**Files Created**:
- `src/bittr_tess_vetter/plotting/checks.py`
- `tests/test_plotting/test_checks.py`

**Exit Criteria**:
- [ ] `plot_odd_even(result)` renders without error
- [ ] Function signature matches spec
- [ ] Unit tests cover: ax creation, provided ax, labels, legend toggle, missing data error

#### Step 2.3: Validation Agent
**Command**: `pytest tests/ -v -k "odd_even or test_plotting" && ruff check`

#### Step 2.4: Commit Agent
**Commit**: `feat(plotting): implement V01 odd/even depth plot`

**Files**:
- `src/bittr_tess_vetter/validation/lc_checks.py`
- `src/bittr_tess_vetter/plotting/checks.py`
- `tests/test_plotting/test_checks.py`

---

### Phase 3: LC-Only Checks (V02-V05)
**Duration**: ~60 minutes
**Parallelization**: V02, V03, V04, V05 can be implemented in PARALLEL

#### Step 3.1: Check Update Agents (4 parallel instances)
Each updates one check in `lc_checks.py`:

| Agent | Check | plot_data Keys |
|-------|-------|----------------|
| 3.1a | V02 secondary_eclipse | phase, flux, flux_err, windows |
| 3.1b | V03 duration_consistency | observed/expected hours, ratio |
| 3.1c | V04 depth_stability | epoch_times_btjd, depths_ppm, depth_errs_ppm |
| 3.1d | V05 v_shape | binned_phase, binned_flux, trapezoid model |

#### Step 3.2: Plot Implementation Agents (4 parallel instances)
Each implements one function in `checks.py`:

| Agent | Function | Key Visual Elements |
|-------|----------|---------------------|
| 3.2a | plot_secondary_eclipse | Phase-folded with window shading |
| 3.2b | plot_duration_consistency | Bar chart: observed vs expected |
| 3.2c | plot_depth_stability | Per-epoch depths with error bars |
| 3.2d | plot_v_shape | Binned data + trapezoid overlay |

#### Step 3.3: Validation Agent
**Command**: `pytest tests/test_plotting/test_checks.py -v && pytest tests/test_validation/test_lc_checks.py -v`

#### Step 3.4: Commit Agent
**Commit**: `feat(plotting): implement V02-V05 LC check plots`

---

### Phase 4: Pixel-Level Checks (V08-V10)
**Duration**: ~75 minutes
**Parallelization**: V08, V09, V10 in parallel after validation module updates

#### Step 4.1: Check Update Agents (parallel)
**Files Modified**: `src/bittr_tess_vetter/validation/checks_pixel.py`

| Agent | Check | plot_data Keys |
|-------|-------|----------------|
| 4.1a | V08 centroid_shift | reference_image, centroids, target_pixel |
| 4.1b | V09 difference_image | difference_image, depth_map_ppm, target_pixel |
| 4.1c | V10 aperture_dependence | aperture_radii_px, depths_ppm |

#### Step 4.2: Plot Implementation Agents (parallel)
**Files Created**: `src/bittr_tess_vetter/plotting/pixel.py`

| Agent | Function | Special Handling |
|-------|----------|------------------|
| 4.2a | plot_centroid_shift | origin="lower", colorbar, vector overlay |
| 4.2b | plot_difference_image | RdBu_r colormap, centered at 0 |
| 4.2c | plot_aperture_curve | Simple line plot with error bars |

#### Step 4.3: Visual Regression Setup
**Files Created**:
- `tests/test_plotting/baseline_images/` (directory)
- `tests/test_plotting/test_visual_regression.py`

#### Step 4.4: Validation Agent
**Command**: `pytest tests/test_plotting/ -v && pytest tests/test_validation/test_checks_pixel.py -v`

#### Step 4.5: Commit Agent
**Commit**: `feat(plotting): implement V08-V10 pixel check plots with visual regression`

---

### Phase 5: Catalog Checks (V06-V07)
**Duration**: ~45 minutes
**Parallelization**: V06, V07 in parallel

#### Step 5.1: Check Update Agents
**Files Modified**: `src/bittr_tess_vetter/validation/checks_catalog.py`

#### Step 5.2: Plot Implementation Agents
**Files Created**: `src/bittr_tess_vetter/plotting/catalog.py`

| Function | Visual |
|----------|--------|
| plot_nearby_ebs | Sky map with target + neighbors |
| plot_exofop_card | Status card with disposition |

#### Step 5.3: Validation Agent + Commit Agent

---

### Phase 6: Exovetter Checks (V11-V12)
**Duration**: ~45 minutes
**Parallelization**: V11, V12 in parallel

#### Step 6.1: Check Update Agents
**Files Modified**: `src/bittr_tess_vetter/validation/exovetter_checks.py`

#### Step 6.2: Plot Implementation Agents
**Files Created**: `src/bittr_tess_vetter/plotting/exovetter.py`

| Function | Visual |
|----------|--------|
| plot_modshift | Phase-binned periodogram |
| plot_sweet | Sinusoid fits overlay |

#### Step 6.3: Validation Agent + Commit Agent

---

### Phase 7: False Alarm Checks (V13, V15)
**Duration**: ~30 minutes
**Parallelization**: V13, V15 in parallel

#### Step 7.1: Check Update Agents
**Files Modified**: `src/bittr_tess_vetter/validation/lc_false_alarm_checks.py`

#### Step 7.2: Plot Implementation Agents
**Files Created**: `src/bittr_tess_vetter/plotting/false_alarm.py`

| Function | Visual |
|----------|--------|
| plot_data_gaps | Epoch coverage heatmap |
| plot_asymmetry | Phase-folded with left/right bins |

#### Step 7.3: Validation Agent + Commit Agent

---

### Phase 8: Extended Checks (V16-V21)
**Duration**: ~90 minutes
**Parallelization**: All 6 in parallel (resource permitting)

#### Step 8.1: Check Update Agents
**Files Modified**: Various in validation/

| Check | Source File |
|-------|-------------|
| V16 | validation/checks_extended.py (or create) |
| V17 | validation/ephemeris_reliability.py |
| V19 | validation/alias_diagnostics.py |
| V20 | validation/ghost_features.py |
| V21 | validation/sector_consistency.py |

#### Step 8.2: Plot Implementation Agents
**Files Created**: `src/bittr_tess_vetter/plotting/extended.py`

| Function | Visual |
|----------|--------|
| plot_model_comparison | Multi-model overlay |
| plot_ephemeris_reliability | Score vs phase shift |
| plot_sensitivity_sweep | Robustness heatmap |
| plot_alias_diagnostics | Harmonic bar chart |
| plot_ghost_features | Difference image + aperture |
| plot_sector_consistency | Per-sector bar chart |

#### Step 8.3: Validation Agent + Commit Agent

---

### Phase 9: Transit Visualization
**Duration**: ~45 minutes

#### Step 9.1: Plot Implementation Agents
**Files Created**:
- `src/bittr_tess_vetter/plotting/transit.py`
- `src/bittr_tess_vetter/plotting/lightcurve.py`

| Function | Input | Visual |
|----------|-------|--------|
| plot_phase_folded | lc, candidate | Phase-folded with binning |
| plot_transit_fit | fit_result | Model overlay on data |
| plot_full_lightcurve | lc, candidate | Full time series + transit markers |

#### Step 9.2: Validation Agent + Commit Agent

---

### Phase 10: DVR Summary Report
**Duration**: ~60 minutes

#### Step 10.1: Plot Implementation Agent
**Files Created**: `src/bittr_tess_vetter/plotting/report.py`

**Functions**:
- `plot_vetting_summary()` - 8-panel layout
- `save_vetting_report()` - Export to PDF/PNG
- `_render_metrics_table()` - Internal helper

#### Step 10.2: Validation Agent
**Command**: Includes visual regression for full layout

#### Step 10.3: Commit Agent
**Commit**: `feat(plotting): implement DVR summary report`

---

### Phase 11: Integration
**Duration**: ~45 minutes

#### Step 11.1: Integration Agent - Exports
**Files Modified**:
- `src/bittr_tess_vetter/plotting/__init__.py` - Add all exports
- `src/bittr_tess_vetter/api/__init__.py` - Add re-exports under MATPLOTLIB_AVAILABLE guard

#### Step 11.2: Integration Agent - Tests
**Files Created**:
- `tests/test_plotting/test_integration.py`

**Tests**:
- Import from api works
- Plot from real vetting results
- No matplotlib warnings
- DVR summary with real bundle

#### Step 11.3: Validation Agent (Full Suite)
**Command**: `pytest tests/ -v && ruff check && mypy src/bittr_tess_vetter/plotting/`

#### Step 11.4: Commit Agent
**Commit**: `feat(plotting): wire up API exports and integration tests`

---

### Phase 12: Documentation
**Duration**: ~60 minutes

#### Step 12.1: Documentation Agent - Tutorial Updates
**Files Modified**:
- `docs/tutorials/10_vetting_robust_sweep.ipynb` - Add plotting cells

#### Step 12.2: Documentation Agent - Gallery (Optional)
**Files Created**:
- `docs/gallery/README.md`
- `docs/gallery/*.png` - Example images

#### Step 12.3: Commit Agent
**Commit**: `docs: add plotting examples to Tutorial 10`

---

## Agent Instruction Templates

### Foundation Agent Template
```
## Context
Read: abbreviated_spec.md (Sections 1, 5, 6, 11)

## Task
Create the plotting module skeleton:
1. Create plotting/__init__.py with MATPLOTLIB_AVAILABLE guard
2. Create plotting/_core.py with utility functions
3. Create plotting/_styles.py with style presets
4. Create tests/test_plotting/conftest.py with fixtures

## Constraints
- Do NOT implement any plot functions
- All imports must be lazy (inside functions) for matplotlib
- Use TYPE_CHECKING for type hints

## Exit Signal
When complete, run: pytest tests/test_plotting/ -v
Report PASS or FAIL with details.
```

### Check Update Agent Template
```
## Context
Read: abbreviated_spec.md (Sections 3, 8)
Read: Full spec Section 4.7 for exact schema

## Task
Add plot_data to check V{XX} in {module_path}:
1. Before the return statement, construct plot_data dict
2. Include "version": 1
3. Convert all numpy types to Python types
4. Cap arrays at documented limits
5. Add to raw dict: raw={"plot_data": plot_data, ...existing}

## Constraints
- Do NOT modify check logic, only add output
- Existing tests must still pass
- All values must be JSON-serializable

## Exit Signal
Run: pytest tests/test_validation/test_{module}.py -v -k V{XX}
Report PASS or FAIL.
```

### Plot Implementation Agent Template
```
## Context
Read: abbreviated_spec.md (Sections 2, 4, 7, 9)
Read: Full spec Section 5.X for exact function signature

## Task
Implement plot_{name}() in plotting/{module}.py:
1. Follow signature convention exactly
2. Use style_context() wrapper
3. Use ensure_ax() for axes handling
4. Use extract_plot_data() for data access
5. For images: use origin="lower", return (ax, cbar)
6. Write unit tests covering all checklist items

## Constraints
- Do NOT modify _core.py or _styles.py
- Do NOT modify check implementations
- Must handle missing plot_data gracefully (raise ValueError)

## Exit Signal
Run: pytest tests/test_plotting/test_{module}.py::Test{FunctionName} -v
Report PASS or FAIL.
```

### Validation Agent Template
```
## Task
Run validation suite for Phase {N}:
1. pytest {test_paths} -v
2. ruff check {src_paths}
3. mypy {src_paths} (if Phase >= 11)

## Decision
- All commands exit 0: Report PASS
- Any command fails: Report FAIL with stdout/stderr

## On FAIL
Provide:
- Which command failed
- Error output
- Suggested fix if obvious
```

### Commit Agent Template
```
## Context
Validation Agent reported: PASS

## Task
Create atomic commit:
1. Stage only files from this phase
2. Write commit message following format:
   {type}(plotting): {description}

   {body if needed}

   Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

## Constraints
- Never use --amend
- Never force push
- Always verify staged files before commit
```

---

## Dependency Graph

```
Phase 1 (Foundation)
    |
    v
Phase 2 (V01 Proof of Concept)
    |
    +----> Phase 3 (V02-V05) ----+
    |                            |
    +----> Phase 4 (V08-V10) ----+
    |                            |
    +----> Phase 5 (V06-V07) ----+
    |                            |
    +----> Phase 6 (V11-V12) ----+
    |                            |
    +----> Phase 7 (V13, V15) ---+
    |                            |
    +----> Phase 8 (V16-V21) ----+
                                 |
                                 v
                    Phase 9 (Transit Viz) -- requires Phase 3 complete
                                 |
                                 v
                    Phase 10 (DVR Report) -- requires all plots
                                 |
                                 v
                    Phase 11 (Integration) -- requires all implementation
                                 |
                                 v
                    Phase 12 (Documentation)
```

### Parallelization Matrix

| Phase | Can Run After | Parallel With |
|-------|---------------|---------------|
| 1 | START | None |
| 2 | 1 | None |
| 3 | 2 | 4, 5, 6, 7, 8 |
| 4 | 2 | 3, 5, 6, 7, 8 |
| 5 | 2 | 3, 4, 6, 7, 8 |
| 6 | 2 | 3, 4, 5, 7, 8 |
| 7 | 2 | 3, 4, 5, 6, 8 |
| 8 | 2 | 3, 4, 5, 6, 7 |
| 9 | 3 | 4, 5, 6, 7, 8 (if 3 done) |
| 10 | 3-9 | None |
| 11 | 10 | None |
| 12 | 11 | None |

---

## Error Handling

### Validation Failure Protocol
```
1. Validation Agent reports FAIL
2. Record failure details to .agent_failures/{phase}_{timestamp}.md
3. Identify responsible agent (Check Update or Plot Implementation)
4. Responsible agent receives:
   - Original task
   - Error output
   - Instruction: "Fix the issue and re-run validation"
5. Re-run Validation Agent
6. If FAIL again after 3 attempts:
   - Escalate to human review
   - Mark phase as BLOCKED
```

### Git Conflict Protocol
```
1. Commit Agent detects merge conflict
2. Abort commit
3. Run: git status
4. Identify conflicting files
5. If conflict in test files: merge by keeping both
6. If conflict in implementation: escalate to human
7. After resolution: re-run Validation Agent
```

### Import Error Protocol
```
1. Integration Agent detects ImportError
2. Check MATPLOTLIB_AVAILABLE guard
3. Verify __all__ lists match actual exports
4. Verify TYPE_CHECKING imports present
5. Fix and re-run
```

---

## Milestone Tags

| Tag | After Phase | Meaning |
|-----|-------------|---------|
| `plotting-v0.1.0-foundation` | 1 | Module skeleton ready |
| `plotting-v0.2.0-poc` | 2 | First plot working end-to-end |
| `plotting-v0.3.0-lc-checks` | 3 | All LC check plots (V01-V05) |
| `plotting-v0.4.0-pixel-checks` | 4 | All pixel check plots (V08-V10) |
| `plotting-v0.5.0-all-checks` | 8 | All check plots complete |
| `plotting-v0.6.0-transit-viz` | 9 | Transit visualization complete |
| `plotting-v0.7.0-dvr` | 10 | DVR summary complete |
| `plotting-v0.8.0-integrated` | 11 | API exports wired, integration tested |
| `plotting-v1.0.0` | 12 | Feature complete with docs |

---

## Success Criteria (Feature Complete)

- [ ] All 21 checks have corresponding plot functions
- [ ] `from bittr_tess_vetter.api import plot_odd_even` works
- [ ] `plot_vetting_summary()` produces 8-panel DVR
- [ ] 95%+ test coverage on plotting module
- [ ] Visual regression baselines for V08, V09, DVR
- [ ] No matplotlib warnings during tests
- [ ] Tutorial 10 includes plotting cells
- [ ] All commits atomic and well-documented
