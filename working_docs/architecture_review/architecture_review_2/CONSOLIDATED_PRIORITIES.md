# Open-Source Release Priorities: bittr-tess-vetter
*Consolidated Architecture Review - 2026-01-14*

---

## Executive Summary

**bittr-tess-vetter** is a well-architected TESS transit vetting library with strong foundations: 95 test files, 93% docstring coverage, comprehensive type hints, and robust lazy-loading API. However, the package is **not ready for open-source release** in its current state due to critical gaps in CI/CD automation, security vulnerabilities in dependency constraints, and missing community infrastructure.

The most significant blockers include: (1) the complete absence of CI/CD with 33,000 lines of tests that never run automatically; (2) **1 failing test** in `tests/pixel/test_tpf_fits.py` due to error message mismatch after recent exptime changes; (3) **115 ruff errors** that must be resolved (README recommends `ruff check .` but it currently fails); (4) three dependency constraints exposing users to known CVEs (setuptools, requests, pydantic); and (5) README/docstring drift from actual code paths. Additionally, the license compatibility of `ldtk` (potentially GPL-3.0) requires verification before public release under MIT.

The good news: the core library is scientifically sound with excellent documentation. The work required is primarily infrastructure and housekeeping rather than architectural changes. A focused 2-3 day sprint addressing P0 items would make the package release-ready, with P1 items completing within a week.

---

## Critical Blockers (P0 - Must Fix Before Release)

### P0.1 - Security: Update Vulnerable Dependency Constraints
**Source:** Report 09 (Dependency Security)
**Effort:** 30 minutes

Three dependencies have known CVEs not covered by current version floors:

| Package | Current | Required | CVE |
|---------|---------|----------|-----|
| `setuptools` | `>=70.0.0` | `>=78.1.1` | CVE-2025-47273 (path traversal) |
| `requests` | `>=2.31.0` | `>=2.32.4` | CVE-2024-47081 (credential leak) |
| `pydantic` | `>=2.0.0` | `>=2.4.0` | CVE-2024-3772 (ReDoS) |

**File:** `/Users/collier/projects/apps/bittr-tess-vetter/pyproject.toml`

### P0.2 - License: Verify ldtk License Compatibility
**Source:** Report 09 (Dependency Security)
**Effort:** 1 hour (investigation + potential remediation)

`ldtk` is a **core dependency** listed as GPL-3.0 in some sources. If confirmed, this creates a license incompatibility with the MIT-licensed project.

**Actions:**
1. Verify actual license (some sources show MIT)
2. If GPL-3.0: Either change project license OR make ldtk optional
3. Document in THIRD_PARTY_NOTICES.md

### P0.3 - CI/CD: Add Basic GitHub Actions Workflow
**Source:** Report 03 (CI/CD Configuration)
**Effort:** 2-3 hours

The package has **zero CI/CD automation**. The existing 33,000 lines of tests never run automatically.

**Create:** `.github/workflows/ci.yml`
- Lint check (ruff)
- Type check (mypy)
- Test matrix (Python 3.11, 3.12)
- Run on push to main and PRs

### P0.4 - CLI: Add Smoke Tests for CLI Modules
**Source:** Report 02 (Test Coverage)
**Effort:** 2-3 hours

Five CLI modules have **zero test coverage**:
- `/src/bittr_tess_vetter/cli/mlx_bls_search_cli.py`
- `/src/bittr_tess_vetter/cli/mlx_bls_search_range_cli.py`
- `/src/bittr_tess_vetter/cli/mlx_quick_vet_cli.py`
- `/src/bittr_tess_vetter/cli/mlx_refine_candidates_cli.py`
- `/src/bittr_tess_vetter/cli/mlx_tls_calibration_cli.py`

**Create:** `/tests/cli/test_cli_smoke.py`
- Import smoke tests
- `--help` invocation tests

### P0.5 - Community: Create LICENSE File
**Source:** Report 08 (OSS Best Practices)
**Effort:** 5 minutes

License is declared in `pyproject.toml` but no `LICENSE` file exists in the repository root. Required for GitHub license detection and legal clarity.

### P0.6 - Community: Create CITATION.cff
**Source:** Report 08 (OSS Best Practices)
**Effort:** 30 minutes

Machine-readable citation file for GitHub, Zenodo, and academic indexing. Use [cffinit](https://citation-file-format.github.io/cff-initializer-javascript/) to generate.

### P0.7 - Fix Failing Test
**Source:** Report 11 (Codex Review)
**Effort:** 15 minutes

`tests/pixel/test_tpf_fits.py::TestTPFFitsRefSerialization::test_from_string_invalid` fails due to error message mismatch after recent exptime changes. The test expects `Invalid TPF FITS reference format` but the code raises `Invalid exptime_seconds: ...`.

**File:** `/Users/collier/projects/apps/bittr-tess-vetter/tests/pixel/test_tpf_fits.py`

### P0.8 - Fix Ruff Errors
**Source:** Report 11 (Codex Review)
**Effort:** 30-60 minutes

115 ruff errors exist but README recommends `ruff check .`. This command must pass before release. Errors include unsorted imports, unused variables, and style warnings.

**Command:** `ruff check .` (or decide on scoped ruleset and document)

### P0.9 - Fix README Path Drift
**Source:** Report 11 (Codex Review)
**Effort:** 15 minutes

README claims `src/bittr_tess_vetter/io/` and `catalogs/` exist but actual paths are:
- `src/bittr_tess_vetter/platform/io/`
- `src/bittr_tess_vetter/platform/catalogs/`

**File:** `/Users/collier/projects/apps/bittr-tess-vetter/README.md`

### P0.10 - Remove bittr-reason-core References
**Source:** Report 11 (Codex Review)
**Effort:** 10 minutes

README still references `bittr-reason-core` as a local dependency, but this was removed from `pyproject.toml` and `uv.lock`. Update documentation to reflect current standalone status.

**File:** `/Users/collier/projects/apps/bittr-tess-vetter/README.md`

### P0.11 - Fix astro_arc Docstring Remnants
**Source:** Report 11 (Codex Review)
**Effort:** 15 minutes

Some docstrings in `platform/catalogs/store.py` reference `astro_arc.*` paths from a previous codebase. These should be updated to current module paths.

**File:** `/Users/collier/projects/apps/bittr-tess-vetter/src/bittr_tess_vetter/platform/catalogs/store.py`

---

## High Priority (P1 - Should Fix Before Release)

### P1.1 - Testing: Add Exhaustive Export Resolution Test
**Source:** Reports 02, 05 (Test Coverage, Export Surface)
**Effort:** 30 minutes

Current test validates ~47 exports but `__all__` contains 229 symbols.

**Add to:** `/tests/test_api/test_api_top_level_exports.py`
```python
@pytest.mark.parametrize("name", api.__all__)
def test_all_exports_resolve(name):
    assert hasattr(api, name)
```

### P1.2 - Testing: Add vet_candidate Integration Tests
**Source:** Report 02 (Test Coverage)
**Effort:** 2 hours

The main `vet_candidate()` orchestrator has only 2 tests (catalog gating only). Missing:
- Full workflow with all tiers
- Error propagation
- Config pass-through

**Add:** `/tests/test_integration/test_vet_candidate_full.py`

### P1.3 - CI: Add Pre-commit Hooks
**Source:** Report 03 (CI/CD Configuration)
**Effort:** 30 minutes

**Create:** `.pre-commit-config.yaml`
- ruff (lint + format)
- trailing whitespace
- check-yaml
- check-added-large-files

### P1.4 - CI: Add Dependabot Configuration
**Source:** Report 03 (CI/CD Configuration)
**Effort:** 15 minutes

**Create:** `.github/dependabot.yml`

### P1.5 - Documentation: Document TTV Track Search Functions
**Source:** Report 06 (Docstring Coverage)
**Effort:** 1 hour

`run_ttv_track_search` and `run_ttv_track_search_for_candidate` are exported in `__all__` but lack docstrings.

**File:** `/src/bittr_tess_vetter/api/ttv_track_search.py`

### P1.6 - Documentation: Generate REFERENCES.md
**Source:** Report 06 (Docstring Coverage)
**Effort:** 30 minutes

Use existing `generate_bibliography_markdown()` from `api/references.py` to create `REFERENCES.md`.

### P1.7 - Community: Add CONTRIBUTING.md and CODE_OF_CONDUCT.md
**Source:** Report 08 (OSS Best Practices)
**Effort:** 1 hour

- CONTRIBUTING.md: PR workflow, dev setup
- CODE_OF_CONDUCT.md: Adopt Astropy or Contributor Covenant

### P1.8 - Community: Add GitHub Issue and PR Templates
**Source:** Report 08 (OSS Best Practices)
**Effort:** 30 minutes

- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`
- `.github/PULL_REQUEST_TEMPLATE.md`

### P1.9 - Cleanup: Remove Stale Facade Documentation
**Source:** Report 04 (Facade Remnants)
**Effort:** 30 minutes

Delete or archive:
- `/working_docs/api_roadmap/README.md` - references removed `api.facade`
- `/working_docs/api/api_facade_spec.md` - outdated design spec

Rename in `/tests/test_api/test_api_aliases.py`:
- `test_facade_imports()` -> `test_api_surface_imports()`

### P1.10 - Vendored Code: Add LICENSE to Vendor Directory
**Source:** Report 09 (Dependency Security)
**Effort:** 5 minutes

Copy MIT license to `/src/bittr_tess_vetter/ext/triceratops_plus_vendor/LICENSE`

### P1.11 - Cache Path Default
**Source:** Report 11 (Codex Review)
**Effort:** 1-2 hours

`platform/io/cache.py` defaults to CWD-relative paths (`Path.cwd() / ".bittr-tess-vetter" / ...`) which surprises Jupyter/SLURM users and can cause issues with read-only working directories.

**Recommendation:** Use `platformdirs` for OS-appropriate user cache directory while preserving env var overrides.

**File:** `/Users/collier/projects/apps/bittr-tess-vetter/src/bittr_tess_vetter/platform/io/cache.py`

### P1.12 - Document Platform Support
**Source:** Report 11 (Codex Review)
**Effort:** 30 minutes

Cache locking uses `fcntl` (Unix-specific) and network timeouts use SIGALRM (best-effort on some platforms). Document platform support explicitly:
- macOS/Linux: first-class support
- Windows: best-effort with graceful degradation

**File:** `/Users/collier/projects/apps/bittr-tess-vetter/README.md` (or dedicated `PLATFORM_SUPPORT.md`)

---

## Medium Priority (P2 - Fix Soon After Release)

### P2.1 - Documentation: Create Sphinx Documentation Structure
**Source:** Reports 06, 07, 08 (Docstring, Competitor Patterns, OSS Practices)
**Effort:** 4-6 hours

Required for JOSS/pyOpenSci submission:
- `docs/conf.py` with autodoc + napoleon
- `docs/index.rst`
- `docs/api.rst` (autosummary)
- `.readthedocs.yaml`

### P2.2 - Testing: Add Edge Case Tests (NaN, Empty, Single-Element)
**Source:** Report 02 (Test Coverage)
**Effort:** 4 hours

Add parametrized tests for boundary conditions across public API functions.

### P2.3 - CI: Add Coverage Reporting
**Source:** Reports 02, 03 (Test Coverage, CI/CD)
**Effort:** 1 hour

Add pytest-cov and Codecov integration to CI workflow.

### P2.4 - CI: Add Release Automation
**Source:** Report 03 (CI/CD Configuration)
**Effort:** 1 hour

**Create:** `.github/workflows/release.yml`
- Trigger on version tags
- Build with hatch
- Publish to PyPI (trusted publishing)

### P2.5 - API: Document Recommended Import Alias
**Source:** Report 07 (Competitor Patterns)
**Effort:** 15 minutes

Add to README and module docstring:
```python
import bittr_tess_vetter.api as btv
```

### P2.6 - Dependencies: Clean Up Duplicate Pins in triceratops Extras
**Source:** Report 09 (Dependency Security)
**Effort:** 15 minutes

Remove duplicates from `[triceratops]` optional deps (emcee, numba, arviz are already in core).

### P2.7 - Dependencies: Modernize Legacy Version Floors
**Source:** Report 09 (Dependency Security)
**Effort:** 15 minutes

Update triceratops extras:
- `pandas>=0.23.4` -> `>=2.0.0`
- `seaborn>=0.11.1` -> `>=0.13.0`
- `mechanicalsoup>=0.12.0` -> `>=1.0.0`

### P2.8 - API: Add GPL Notice for triceratops Extras
**Source:** Report 09 (Dependency Security)
**Effort:** 15 minutes

Document in installation docs that `[triceratops]` includes `pytransit` (GPL-2.0).

### P2.9 - UX: Add Warning When Checks Are Skipped
**Source:** Report 10 (User Workflow)
**Effort:** 2 hours

When `network=False` or metadata missing, include warnings in result explaining which checks were skipped and why.

### P2.10 - Clarify Domain-Only Claim
**Source:** Report 11 (Codex Review)
**Effort:** 30 minutes

Project includes network clients, caching, and disk-backed catalog snapshots under `platform/`, but README claims "domain-only" focus. Document the split explicitly:
- `api/`, `compute/`, `validation/`, etc. = pure domain logic
- `platform/` = opt-in infrastructure (network, caching, disk I/O)

Consider future distribution split (`bittr-tess-vetter` + `bittr-tess-vetter-platform`).

### P2.11 - Define Minimal Install Path
**Source:** Report 11 (Codex Review)
**Effort:** 2-4 hours

Base dependencies include heavy stacks (TLS/numba, emcee, arviz, ldtk). Many research users prefer minimal core plus extras. Consider:
- Define "minimal core" for CPU-only vetting
- Move heavier features into optional extras
- Document install paths for different use cases

### P2.12 - Fix Dev Dependencies
**Source:** Report 11 (Codex Review)
**Effort:** 15 minutes

`pyproject.toml` has both `[project.optional-dependencies].dev` and `[dependency-groups].dev`. This can cause `mypy` to not install correctly with `uv sync`. Consolidate to one pattern and update dev setup docs.

**File:** `/Users/collier/projects/apps/bittr-tess-vetter/pyproject.toml`

---

## Low Priority (P3 - Backlog)

### P3.1 - Naming: Rename catalog.py to catalog_checks.py
**Source:** Report 01 (Naming Conflict)
**Effort:** 2 hours (with deprecation wrapper)

Address `catalog.py` vs `catalogs.py` confusion. Short-term: document distinction (done). Medium-term: rename with deprecation.

### P3.2 - Documentation: Add Examples to Entry Point Docstrings
**Source:** Report 06 (Docstring Coverage)
**Effort:** 2 hours

Add Example sections to:
- `run_periodogram()`
- `calculate_fpp()`
- `localize_transit_source()`
- `fit_transit()`

### P3.3 - Documentation: Add Tutorial Notebooks
**Source:** Reports 06, 07, 10 (Docstring, Competitor, User Workflow)
**Effort:** 8 hours

Create numbered tutorials following lightkurve pattern:
- `01-basic-vetting.ipynb`
- `02-custom-checks.ipynb`
- `03-batch-processing.ipynb`

### P3.4 - UX: Add vet_tic() Convenience Function
**Source:** Report 10 (User Workflow)
**Effort:** 4 hours

One-liner vetting for researchers with known TIC ID and ephemeris:
```python
result = vet_tic(tic_id=261136679, period=3.5, t0=1850.0, duration_hours=2.5)
```

### P3.5 - UX: Make FPP Cache Optional
**Source:** Report 10 (User Workflow)
**Effort:** 4-6 hours

`calculate_fpp()` requires a `cache` object not easily constructed. Provide default in-memory caching or `calculate_fpp_simple()` alternative.

### P3.6 - Performance: Reduce First-Access Latency
**Source:** Report 05 (Export Surface)
**Effort:** 4 hours

First access to types loads 935 modules. Consider splitting `types.py` to reduce transitive imports.

### P3.7 - API: Add Stability Tier Documentation
**Source:** Report 05 (Export Surface)
**Effort:** 2 hours

Document 62 undocumented-but-accessible exports with stability guarantees:
- Tier 1 (Core): Never break
- Tier 2 (Extended): Deprecation cycle
- Tier 3 (Advanced): May change

### P3.8 - Testing: Add TRICERATOPS Integration Test
**Source:** Report 02 (Test Coverage)
**Effort:** 3 hours

End-to-end FPP calculation with known inputs, skip if optional deps missing.

### P3.9 - Community: Register with ASCL
**Source:** Report 08 (OSS Best Practices)
**Effort:** 30 minutes (after first paper uses code)

Submit to Astrophysics Source Code Library for ADS indexing.

### P3.10 - Community: Submit to pyOpenSci/JOSS
**Source:** Report 08 (OSS Best Practices)
**Effort:** Ongoing

After docs and tests complete, submit for peer review.

---

## Cross-Cutting Themes

### Theme 1: Infrastructure Debt
Multiple reports independently identified the absence of CI/CD as the primary gap. Reports 02, 03, and 08 all highlight that existing tests and linting tools are configured but never automated.

### Theme 2: Documentation Structure vs. Content
The codebase has **excellent docstring content** (93% coverage, academic citations) but **missing documentation infrastructure** (no Sphinx, no RTD, no tutorials). The content exists; it just needs to be surfaced.

### Theme 3: Two-System Type Architecture
Reports 01, 05, and 10 all note friction from the facade/internal type split (`LightCurve` vs `LightCurveData`, `catalog.py` vs `catalogs.py`). This is a intentional design choice but requires better documentation for users.

### Theme 4: Dependency Hygiene
Report 09 found security vulnerabilities, license concerns, and outdated version floors. This is invisible to users until it causes problems.

### Theme 5: Research vs. Research Library Gap
Report 10 identifies that the library is designed for integration into larger systems (requiring cache objects, host metadata) rather than standalone research use. Adding convenience functions would broaden adoption.

### Theme 6: Documentation-Reality Drift
Report 11 identified multiple places where documentation (README, docstrings) references paths or dependencies that no longer exist. This erodes trust for new users and contributors.

---

## Recommended Release Sequence

### Phase 1: Security, Legal, and Green CI (Day 1)
1. Update dependency version constraints (P0.1)
2. Verify ldtk license (P0.2)
3. Create LICENSE file (P0.5)
4. Add LICENSE to vendored code (P1.10)
5. **Fix failing test** (P0.7)
6. **Fix ruff errors** (P0.8)

### Phase 2: Documentation Alignment (Day 1)
1. **Fix README path drift** (P0.9)
2. **Remove bittr-reason-core references** (P0.10)
3. **Fix astro_arc docstring remnants** (P0.11)

### Phase 3: CI/CD Foundation (Day 1-2)
1. Create `.github/workflows/ci.yml` (P0.3)
2. Create `.pre-commit-config.yaml` (P1.3)
3. Create `.github/dependabot.yml` (P1.4)
4. Add CLI smoke tests (P0.4)

### Phase 4: Community Infrastructure (Day 2)
1. Create CITATION.cff (P0.6)
2. Create CONTRIBUTING.md + CODE_OF_CONDUCT.md (P1.7)
3. Add GitHub templates (P1.8)
4. Clean up facade remnants (P1.9)

### Phase 5: Test Hardening (Day 2-3)
1. Add exhaustive export test (P1.1)
2. Add vet_candidate integration tests (P1.2)
3. Document ttv_track_search functions (P1.5)

### Phase 6: Documentation Polish (Week 2)
1. Generate REFERENCES.md (P1.6)
2. Create Sphinx skeleton (P2.1)
3. Add coverage reporting (P2.3)
4. Add release automation (P2.4)
5. Document platform support (P1.12)
6. Improve cache path defaults (P1.11)

### Phase 7: Initial PyPI Release
- Tag v0.1.0
- Publish to PyPI
- Enable Zenodo DOI

### Phase 8: Post-Release Improvements (Ongoing)
- Tutorial notebooks (P3.3)
- Convenience functions (P3.4, P3.5)
- Minimal install path (P2.11)
- ASCL registration (P3.9)
- pyOpenSci/JOSS submission (P3.10)

---

## Appendix: Report Sources

| # | Report | Key Contribution |
|---|--------|------------------|
| 01 | Naming Conflict Analysis | Identified `catalog.py`/`catalogs.py` confusion; recommended documentation short-term, rename medium-term |
| 02 | Test Coverage Audit | Found CLI zero-coverage gap; quantified 95 test files, 93% coverage; identified integration test gaps |
| 03 | CI/CD Configuration | Documented complete absence of automation; provided ready-to-use workflow templates |
| 04 | Facade Remnants | Cataloged stale documentation from removed `api.facade`; identified test function renames needed |
| 05 | Export Surface Analysis | Mapped 229 documented + 62 undocumented exports; validated lazy-loading robustness; identified versioning gaps |
| 06 | Docstring Coverage | Confirmed 93% coverage with excellent quality; identified 5-6 critical functions missing docstrings; noted missing Sphinx setup |
| 07 | Competitor API Patterns | Synthesized patterns from lightkurve, astropy, exoplanet, TLS; recommended import alias, method chaining, documentation structure |
| 08 | OSS Best Practices | Mapped requirements for pyOpenSci, JOSS, ASCL; provided CITATION.cff template; outlined community file checklist |
| 09 | Dependency Security | Found 3 CVEs in version constraints; identified GPL license risk with ldtk/pytransit; validated vendored code isolation |
| 10 | User Workflow Analysis | Documented 8-step TIC-to-vetting workflow; identified FPP cache friction; proposed convenience functions |
| 11 | Codex Review | Ran local CI checks (1 test failure, 115 ruff errors); identified README/docstring drift; flagged cache path and platform support issues |

---

## Summary Metrics

| Category | Status | Effort to Fix |
|----------|--------|---------------|
| Security (CVEs) | 3 actionable | 30 min |
| License compliance | Needs verification | 1 hour |
| CI/CD | Missing entirely | 3-4 hours |
| **Test suite** | **1 failing test** | **15 min** |
| **Linting** | **115 ruff errors** | **30-60 min** |
| Test coverage | 95% good, CLI gap | 2-3 hours |
| **Documentation drift** | **README/docstring mismatches** | **40 min** |
| Documentation content | Excellent (93%) | Minimal |
| Documentation infrastructure | Missing | 4-6 hours |
| Community files | Missing | 2 hours |
| API surface | Stable, well-designed | None needed |
| Platform support | Undocumented | 30 min |
| Cache defaults | CWD-relative (surprising) | 1-2 hours |

**Total P0 effort:** ~10-11 hours (was ~8 hours)
**Total P1 effort:** ~12-13 hours (was ~10 hours)
**Total P2 effort:** ~12-14 hours
**Recommended timeline:** 2-3 days focused work for release-ready state
