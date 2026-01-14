# Orchestration Plan: v0.1.0 Pipeline Refactor

**Branch:** `feat/v0.1.0-pipeline-refactor`
**Base:** `main` @ `68c66aa`
**Estimated effort:** 8-12 hours agent time
**Parallelism:** Limited (sequential dependencies between phases)

---

## Principles

### Commit Strategy
- **Orchestrator commits** — Agents do NOT commit. The orchestrator (Claude) reviews agent output, runs validation, and commits.
- **Atomic commits** — One commit per logical unit of work. Each commit must leave tests passing.
- **Commit size** — Target 100-500 lines changed per commit. Larger refactors split into "add new" then "migrate" then "remove old" commits.

### Testing Requirements
- **Pre-commit gate:** `uv run ruff check . && uv run mypy src/bittr_tess_vetter && uv run pytest -x -q`
- **No regressions:** All existing tests must pass after each commit.
- **New code = new tests:** Each new module/class gets corresponding test file.

### Git Worktrees
Not used for this refactor. Reasoning:
- Phases are sequential (each depends on prior)
- Single feature branch is cleaner for this scope
- Worktrees add complexity without parallelism benefit here

### Agent Deployment
- **1-2 agents per phase** — Opus agents have 200k context; use fewer, larger agents
- **Agent scope** — Each agent gets a complete, self-contained task with clear inputs/outputs
- **Validation before handoff** — Agent must run tests before reporting completion

---

## Phase Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DEPENDENCY GRAPH                                     │
│                                                                             │
│   Phase 1 ──────► Phase 2 ──────► Phase 3 ──────► Phase 4                  │
│   (Types)         (Registry)      (Checks)        (API Surface)            │
│                                                                             │
│   [1 agent]       [1 agent]       [2 agents       [1 agent]                │
│                                    parallel]                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Result Types + Constructors

**Agent count:** 1
**Estimated time:** 2 hours
**Commits:** 2

### Deliverables

1. **New file:** `src/bittr_tess_vetter/validation/result_schema.py`
   ```python
   CheckStatus = Literal["ok", "skipped", "error"]

   class CheckResult(BaseModel):
       id: str
       name: str
       status: CheckStatus
       confidence: float | None = None
       metrics: dict[str, float | int | str | bool | None] = {}
       flags: list[str] = []
       notes: list[str] = []
       provenance: dict[str, float | int | str | bool | None] = {}
       raw: dict[str, Any] | None = None

   class VettingBundleResult(BaseModel):
       results: list[CheckResult]
       warnings: list[str] = []
       provenance: dict[str, Any] = {}
       inputs_summary: dict[str, Any] = {}

   def ok_result(id, name, *, metrics, ...) -> CheckResult: ...
   def skipped_result(id, name, *, reason_flag, ...) -> CheckResult: ...
   def error_result(id, name, *, error, ...) -> CheckResult: ...
   ```

2. **New file:** `src/bittr_tess_vetter/errors.py`
   ```python
   class MissingOptionalDependency(ImportError):
       def __init__(self, extra: str, install_hint: str): ...
   ```

3. **New file:** `tests/validation/test_result_schema.py`
   - Test CheckResult JSON serialization
   - Test ok_result/skipped_result/error_result helpers
   - Test VettingBundleResult construction

### Commit Plan
```
Commit 1.1: "feat(validation): add CheckResult and VettingBundleResult schemas"
  - result_schema.py (types only)
  - test_result_schema.py (type tests)

Commit 1.2: "feat(errors): add MissingOptionalDependency exception"
  - errors.py
  - Update existing lazy import guards to use new exception
```

### Validation Gate
```bash
uv run pytest tests/validation/test_result_schema.py -v
uv run ruff check . && uv run mypy src/bittr_tess_vetter && uv run pytest -x -q
```

---

## Phase 2: Registry + Pipeline Framework

**Agent count:** 1
**Estimated time:** 3 hours
**Commits:** 3
**Depends on:** Phase 1

### Deliverables

1. **New file:** `src/bittr_tess_vetter/validation/registry.py`
   ```python
   class CheckTier(Enum):
       LC_ONLY = "lc_only"
       CATALOG = "catalog"
       PIXEL = "pixel"
       EXOVETTER = "exovetter"
       AUX = "aux"

   @dataclass
   class CheckRequirements:
       needs_tpf: bool = False
       needs_network: bool = False
       needs_ra_dec: bool = False
       needs_tic_id: bool = False
       needs_stellar: bool = False
       optional_deps: list[str] = field(default_factory=list)

   class VettingCheck(Protocol):
       id: str
       name: str
       tier: CheckTier
       requirements: CheckRequirements
       citations: list[str]
       def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult: ...

   class CheckRegistry:
       def register(self, check: VettingCheck) -> None: ...
       def get(self, id: str) -> VettingCheck: ...
       def list(self) -> list[VettingCheck]: ...
       def list_by_tier(self, tier: CheckTier) -> list[VettingCheck]: ...

   DEFAULT_REGISTRY: CheckRegistry
   def register_default_checks() -> None: ...
   ```

2. **New file:** `src/bittr_tess_vetter/api/pipeline.py`
   ```python
   class VettingPipeline:
       def __init__(self, checks: list[str] | None = None, *,
                    registry: CheckRegistry | None = None,
                    config: PipelineConfig | None = None): ...

       def run(self, lc, candidate, *, stellar=None, tpf=None,
               network=False, ...) -> VettingBundleResult: ...

       def describe(self, ...) -> dict: ...
   ```

3. **New file:** `tests/validation/test_registry.py`
4. **New file:** `tests/test_api/test_pipeline.py`

### Commit Plan
```
Commit 2.1: "feat(validation): add CheckRegistry and VettingCheck protocol"
  - registry.py (registry + protocol, no default checks yet)
  - test_registry.py

Commit 2.2: "feat(api): add VettingPipeline class"
  - pipeline.py
  - test_pipeline.py (with mock checks)

Commit 2.3: "feat(validation): register default checks V01-V12"
  - Update registry.py with register_default_checks()
  - Wire existing check implementations as VettingCheck adapters
```

### Validation Gate
```bash
uv run pytest tests/validation/test_registry.py tests/test_api/test_pipeline.py -v
uv run ruff check . && uv run mypy src/bittr_tess_vetter && uv run pytest -x -q
```

---

## Phase 3: Check Normalization

**Agent count:** 2 (parallel)
**Estimated time:** 3 hours
**Commits:** 4
**Depends on:** Phase 2

### Agent 3A: LC-Only + Catalog Checks

**Scope:** V01-V07

1. Update `validation/lc_checks.py`:
   - Convert V01-V05 to use `ok_result()`/`skipped_result()`
   - Implement as `VettingCheck` classes or register existing functions

2. Update `validation/checks_catalog.py`:
   - Convert V06-V07 to new schema
   - Remove any stub implementations
   - Ensure proper `skipped_result()` when network=False

3. Tests: Update existing tests, add schema validation tests

### Agent 3B: Pixel + Exovetter Checks

**Scope:** V08-V12

1. Update `validation/checks_pixel.py`:
   - Convert V08-V10 to new schema
   - Proper `skipped_result()` when no TPF

2. Update `validation/exovetter_checks.py`:
   - Convert V11-V12 to new schema

3. Remove stub exports from `validation/__init__.py`

4. Tests: Update existing tests, add schema validation tests

### Commit Plan
```
Commit 3.1: "refactor(validation): migrate LC-only checks V01-V05 to new schema"
Commit 3.2: "refactor(validation): migrate catalog checks V06-V07 to new schema"
Commit 3.3: "refactor(validation): migrate pixel checks V08-V10 to new schema"
Commit 3.4: "refactor(validation): migrate exovetter checks V11-V12 to new schema"
```

### Validation Gate
```bash
# After each agent completes:
uv run pytest tests/validation/ -v
uv run ruff check . && uv run mypy src/bittr_tess_vetter && uv run pytest -x -q
```

---

## Phase 4: API Surface Curation

**Agent count:** 1
**Estimated time:** 2 hours
**Commits:** 3
**Depends on:** Phase 3

### Deliverables

1. **Refactor:** `src/bittr_tess_vetter/api/__init__.py`
   - Reduce `__all__` to golden-path exports only (~20-30 symbols)
   - Golden path: `LightCurve`, `Ephemeris`, `Candidate`, `TPFStamp`, `CheckResult`, `VettingBundleResult`, `VettingPipeline`, `vet_candidate`, `run_periodogram`, `localize_transit_source`, `recover_transit`, `fit_transit`, `calculate_fpp`, `list_checks`, `describe_checks`

2. **New file:** `src/bittr_tess_vetter/api/primitives.py`
   - Re-export advanced building blocks
   - Document as "supported but not golden path"

3. **New file:** `src/bittr_tess_vetter/api/experimental.py`
   - Re-export unstable/provisional APIs
   - Add module-level warning

4. **Update:** `vet_candidate()` to use `VettingPipeline` internally
   ```python
   def vet_candidate(...) -> VettingBundleResult:
       """Thin wrapper around VettingPipeline for convenience."""
       return VettingPipeline().run(...)
   ```

5. **Update:** `docs/api.rst` to reflect new structure

### Commit Plan
```
Commit 4.1: "refactor(api): migrate vet_candidate to use VettingPipeline"
Commit 4.2: "refactor(api): create primitives and experimental submodules"
Commit 4.3: "refactor(api): curate __all__ to golden-path exports"
```

### Validation Gate
```bash
# Ensure imports still work
python -c "from bittr_tess_vetter.api import vet_candidate, VettingPipeline, CheckResult"
python -c "from bittr_tess_vetter.api.primitives import bin_median_trend"
uv run ruff check . && uv run mypy src/bittr_tess_vetter && uv run pytest -x -q
```

---

## Phase 5: Integration Testing + Final Validation

**Agent count:** 1
**Estimated time:** 1 hour
**Commits:** 1
**Depends on:** Phase 4

### Deliverables

1. **New file:** `tests/test_integration/test_pipeline_e2e.py`
   - End-to-end pipeline test with synthetic data
   - Verify VettingBundleResult schema compliance
   - Test all check tiers with appropriate inputs

2. **Update:** `tests/test_api/test_api_top_level_exports.py`
   - Verify golden-path exports exist
   - Verify primitives/experimental structure

3. **Contract tests:**
   - Every CheckResult is JSON-serializable
   - No check returns "deferred" status
   - All V01-V12 are registered

### Commit Plan
```
Commit 5.1: "test: add pipeline E2E and contract tests"
```

---

## Execution Commands

### Deploy Phase 1
```
Task(prompt="Phase 1: Create CheckResult/VettingBundleResult schemas...")
# Orchestrator reviews, runs validation, commits
```

### Deploy Phase 2
```
Task(prompt="Phase 2: Create CheckRegistry and VettingPipeline...")
# Orchestrator reviews, runs validation, commits
```

### Deploy Phase 3 (parallel)
```
Task(prompt="Phase 3A: Migrate LC-only and catalog checks...")
Task(prompt="Phase 3B: Migrate pixel and exovetter checks...")
# Orchestrator reviews both, runs validation, commits sequentially
```

### Deploy Phase 4
```
Task(prompt="Phase 4: Curate API surface...")
# Orchestrator reviews, runs validation, commits
```

### Deploy Phase 5
```
Task(prompt="Phase 5: Integration tests...")
# Orchestrator reviews, runs validation, commits
```

---

## Rollback Plan

If a phase fails validation:
1. `git reset --hard HEAD~N` to undo commits from that phase
2. Re-deploy agent with additional context about failure
3. If persistent failure, `git checkout main` and reassess spec

---

## Success Criteria

```bash
# All tests pass
uv run pytest -v

# Golden path imports work
python -c "
from bittr_tess_vetter.api import (
    LightCurve, Ephemeris, Candidate,
    CheckResult, VettingBundleResult, VettingPipeline,
    vet_candidate, run_periodogram, list_checks
)
print('Golden path OK')
"

# Pipeline returns structured results
python -c "
from bittr_tess_vetter.api import vet_candidate, Candidate, Ephemeris, LightCurve
import numpy as np
lc = LightCurve(time=np.linspace(0, 27, 1000), flux=np.ones(1000), flux_err=np.ones(1000)*0.001)
c = Candidate(tic_id=123, ephemeris=Ephemeris(epoch=1.0, period=3.5, duration_hours=2.0))
result = vet_candidate(lc, c, network=False)
assert hasattr(result, 'results')
assert all(hasattr(r, 'status') for r in result.results)
print(f'Pipeline OK: {len(result.results)} checks')
"

# No stub/deferred in results
python -c "
# ... run vet_candidate ...
assert not any('deferred' in str(r.flags) for r in result.results)
print('No stubs OK')
"
```

---

## Post-Refactor Checklist

- [ ] All phases complete, all commits on feature branch
- [ ] `uv run pytest` passes (0 failures)
- [ ] `uv run ruff check .` clean
- [ ] `uv run mypy src/bittr_tess_vetter` clean
- [ ] Docs build: `cd docs && make html`
- [ ] Push feature branch: `git push -u origin feat/v0.1.0-pipeline-refactor`
- [ ] Create PR for review
- [ ] Update astro-arc-tess adapter (Phase 5 in spec)
- [ ] Merge to main
- [ ] Tag v0.1.0
