# Tranche 1: Complete P0 + P1 in 4 Parallel Agents

**Goal:** Complete ALL P0 critical blockers and ALL P1 high-priority items in one tranche.

**Success Criteria:** Repository is release-ready after this tranche completes.

---

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRANCHE 1 (4 PARALLEL AGENTS)                       │
├─────────────────────┬───────────────────┬─────────────────┬─────────────────┤
│   Agent 1: CODE     │  Agent 2: DEPS    │  Agent 3: DOCS  │  Agent 4: INFRA │
│   Quality & Tests   │  & Licensing      │  Overhaul       │  & Community    │
├─────────────────────┼───────────────────┼─────────────────┼─────────────────┤
│ P0.7  Fix test      │ P0.1  Vuln deps   │ P0.9  README    │ P0.3  CI/CD     │
│ P0.8  Ruff errors   │ P0.2  ldtk lic    │ P0.10 reason-co │ P0.6  CITATION  │
│ P0.4  CLI tests     │ P0.5  LICENSE     │ P0.11 astro_arc │ P1.3  pre-commit│
│ P1.1  Export test   │ P1.10 Vendor lic  │ P1.5  TTV docs  │ P1.4  Dependabot│
│ P1.2  Integration   │ P2.6  Dup pins    │ P1.6  REFS.md   │ P1.7  CONTRIB   │
│                     │ P2.7  Version flr │ P1.9  Facade rm │ P1.8  Templates │
│                     │ P2.12 Dev deps    │ P1.12 Platform  │                 │
│                     │                   │ P2.5  Import ali│                 │
│                     │                   │ P2.10 Domain cla│                 │
├─────────────────────┼───────────────────┼─────────────────┼─────────────────┤
│ ~2 hours            │ ~1 hour           │ ~1.5 hours      │ ~1.5 hours      │
└─────────────────────┴───────────────────┴─────────────────┴─────────────────┘
```

---

## Agent 1: Code Quality & Tests

**Mission:** Make the codebase CI-ready. All tests pass, all linting passes, coverage improved.

**Tasks:**
1. **P0.7** - Fix failing test in `tests/pixel/test_tpf_fits.py` (error message mismatch)
2. **P0.8** - Fix all 115 ruff errors (`ruff check . --fix` then manual cleanup)
3. **P0.4** - Create CLI smoke tests for all 5 CLI modules:
   - `tests/cli/test_cli_smoke.py`
   - Import tests + `--help` invocation for each CLI
4. **P1.1** - Add exhaustive export resolution test:
   - `@pytest.mark.parametrize("name", api.__all__)` test
5. **P1.2** - Add `vet_candidate()` integration tests:
   - Full workflow test with synthetic data
   - Error propagation test
   - Config pass-through test

**Verification:**
```bash
uv run pytest  # All pass
uv run ruff check .  # Exit 0
uv run mypy src/bittr_tess_vetter  # No errors (if possible)
```

**Output:** Write summary to `tranche_1/agent_1_code_quality_report.md`

---

## Agent 2: Dependencies & Licensing

**Mission:** Secure dependencies, verify license compatibility, clean up pyproject.toml.

**Tasks:**
1. **P0.2** - Research ldtk license:
   - Check PyPI, GitHub (https://github.com/hpparvi/ldtk), LICENSE file
   - Determine if GPL-3.0 or MIT
   - If GPL: recommend making it optional
2. **P0.1** - Update vulnerable dependency floors in pyproject.toml:
   - `setuptools>=78.1.1` (CVE-2025-47273)
   - `requests>=2.32.4` (CVE-2024-47081)
   - `pydantic>=2.4.0` (CVE-2024-3772)
3. **P0.5** - Create `LICENSE` file (MIT, 2024-2026, bittr.ai contributors)
4. **P1.10** - Copy LICENSE to `src/bittr_tess_vetter/ext/triceratops_plus_vendor/`
5. **P2.6** - Remove duplicate pins from `[triceratops]` extras
6. **P2.7** - Modernize legacy version floors:
   - `pandas>=2.0.0`, `seaborn>=0.13.0`, `mechanicalsoup>=1.0.0`
7. **P2.12** - Fix dev dependencies:
   - Consolidate `[project.optional-dependencies].dev` and `[dependency-groups].dev`
   - Ensure mypy is included

**Verification:**
```bash
uv lock  # Succeeds
cat LICENSE  # Exists
grep -r "GPL" .  # Document any GPL deps
```

**Output:** Write summary + ldtk findings to `tranche_1/agent_2_licensing_report.md`

---

## Agent 3: Documentation Overhaul

**Mission:** Make all documentation accurate, complete, and user-friendly.

**Tasks:**
1. **P0.9** - Fix README path drift:
   - `io/` → `platform/io/`
   - `catalogs/` → `platform/catalogs/`
   - Verify all paths in code map exist
2. **P0.10** - Remove all `bittr-reason-core` references
3. **P0.11** - Fix `astro_arc.*` docstring remnants (grep and fix all)
4. **P1.5** - Document TTV track search functions:
   - Add full docstrings to `run_ttv_track_search` and `run_ttv_track_search_for_candidate`
5. **P1.6** - Generate REFERENCES.md:
   - Use `generate_bibliography_markdown()` from `api/references.py`
   - Or manually create from the references registry
6. **P1.9** - Clean up facade remnants:
   - Archive/delete `working_docs/api_roadmap/README.md`
   - Archive/delete `working_docs/api/api_facade_spec.md`
   - Rename `test_facade_imports()` → `test_api_surface_imports()` in test file
7. **P1.12** - Document platform support in README:
   - macOS/Linux: first-class
   - Windows: best-effort (fcntl, SIGALRM limitations)
8. **P2.5** - Add recommended import alias to README:
   - `import bittr_tess_vetter.api as btv`
9. **P2.10** - Clarify domain-only claim:
   - Document that `platform/` contains opt-in I/O infrastructure
   - Pure domain logic is in `api/`, `compute/`, `validation/`, etc.

**Verification:**
```bash
grep -r "bittr-reason-core" . --include="*.md" --include="*.toml"  # Nothing
grep -r "astro_arc" src/  # Nothing
grep -r "facade" working_docs/  # Cleaned up
```

**Output:** Write summary to `tranche_1/agent_3_documentation_report.md`

---

## Agent 4: Infrastructure & Community

**Mission:** Create all CI/CD and community files for a professional open-source project.

**Tasks:**
1. **P0.3** - Create `.github/workflows/ci.yml`:
   - Trigger: push to main, PRs
   - Jobs: lint (ruff), typecheck (mypy), test (pytest)
   - Matrix: Python 3.11, 3.12
   - Use `uv` for fast installs
2. **P0.6** - Create `CITATION.cff`:
   - Use CFF format for GitHub/Zenodo integration
   - Include all relevant metadata
3. **P1.3** - Create `.pre-commit-config.yaml`:
   - ruff (lint + format)
   - trailing-whitespace, end-of-file-fixer
   - check-yaml, check-added-large-files
4. **P1.4** - Create `.github/dependabot.yml`:
   - pip ecosystem, weekly updates
5. **P1.7** - Create community files:
   - `CONTRIBUTING.md`: dev setup, PR workflow, code style
   - `CODE_OF_CONDUCT.md`: Contributor Covenant v2.1
6. **P1.8** - Create GitHub templates:
   - `.github/ISSUE_TEMPLATE/bug_report.md`
   - `.github/ISSUE_TEMPLATE/feature_request.md`
   - `.github/PULL_REQUEST_TEMPLATE.md`

**Verification:**
```bash
ls .github/workflows/ci.yml  # Exists
ls CITATION.cff  # Exists
ls .pre-commit-config.yaml  # Exists
ls CONTRIBUTING.md CODE_OF_CONDUCT.md  # Exist
ls .github/ISSUE_TEMPLATE/  # Has templates
```

**Output:** Write summary to `tranche_1/agent_4_infrastructure_report.md`

---

## Execution Strategy

All 4 agents run **in parallel**. No dependencies between them.

Agent 1 is the longest-running (test writing), others will complete faster.

---

## Post-Tranche Verification

After all agents complete, run comprehensive check:

```bash
# Full test suite
uv run pytest

# Linting
uv run ruff check .

# Type checking
uv run mypy src/bittr_tess_vetter

# Verify files exist
ls LICENSE CITATION.cff CONTRIBUTING.md CODE_OF_CONDUCT.md REFERENCES.md
ls .github/workflows/ci.yml .pre-commit-config.yaml .github/dependabot.yml

# Verify no stale references
grep -r "bittr-reason-core" . --include="*.md" --include="*.toml" || echo "Clean"
grep -r "astro_arc" src/ || echo "Clean"

# Lock deps
uv lock
```

---

## What This Accomplishes

After Tranche 1:
- ✅ All P0 critical blockers resolved (11 items)
- ✅ All P1 high-priority items resolved (12 items)
- ✅ Some P2 items bonus (4 items)
- ✅ Repository is **release-ready**

**Remaining for Tranche 2 (optional polish):**
- P1.11 - Cache path defaults (platformdirs)
- P2.1 - Sphinx documentation
- P2.2 - Edge case tests
- P2.3 - Coverage reporting
- P2.4 - Release automation
- P2.8 - GPL notice for triceratops
- P2.9 - Skipped check warnings
- P3.* - All backlog items
