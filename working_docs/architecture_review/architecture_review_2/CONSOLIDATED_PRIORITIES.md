# Open-Source Release Priorities: bittr-tess-vetter
*Consolidated Architecture Review - 2026-01-14 (v2)*

---

## Executive Summary

**bittr-tess-vetter is now release-ready.** All P0 critical blockers and P1 high-priority items were completed in commit `2c2f2e4` on 2026-01-14.

The repository now has:
- ✅ All tests passing (`uv run pytest` → 1882 passed)
- ✅ All linting clean (`uv run ruff check .` → 0 errors)
- ✅ CI/CD automation (GitHub Actions: lint, typecheck, test matrix)
- ✅ Security vulnerabilities patched (setuptools, requests, pydantic)
- ✅ License clarity (MIT LICENSE file, ldtk made optional due to GPL-2.0)
- ✅ Community infrastructure (CITATION.cff, CONTRIBUTING.md, CODE_OF_CONDUCT.md, templates)
- ✅ Documentation accuracy (README paths fixed, REFERENCES.md generated)

**Remaining work is polish and enhancements** — none of it blocks release.

---

## Current Priorities (P2 - Post-Release Polish)

### P2.1 - Documentation: Create Sphinx Documentation Structure
**Effort:** 4-6 hours

Required for JOSS/pyOpenSci submission:
- `docs/conf.py` with autodoc + napoleon
- `docs/index.rst`
- `docs/api.rst` (autosummary)
- `.readthedocs.yaml`

### P2.2 - Testing: Add Edge Case Tests (NaN, Empty, Single-Element)
**Effort:** 4 hours

Add parametrized tests for boundary conditions across public API functions.

### P2.3 - CI: Add Coverage Reporting
**Effort:** 1 hour

Add pytest-cov and Codecov integration to CI workflow.

### P2.4 - CI: Add Release Automation
**Effort:** 1 hour

Create `.github/workflows/release.yml` with trusted PyPI publishing.

### P2.8 - API: Add GPL Notice for triceratops Extras
**Effort:** 15 minutes

Document in installation docs that `[triceratops]` includes `pytransit` (GPL-2.0).

### P2.9 - UX: Add Warning When Checks Are Skipped
**Effort:** 2 hours

When `network=False` or metadata missing, include warnings explaining skipped checks.

### P2.11 - Define Minimal Install Path
**Effort:** 2-4 hours

Base deps are heavy (numba, emcee, arviz). Consider defining minimal core + extras pattern.

---

## Backlog (P3 - Future Enhancements)

### P3.1 - Naming: Rename catalog.py to catalog_checks.py
**Effort:** 2 hours

Address `catalog.py` vs `catalogs.py` confusion with deprecation wrapper.

### P3.2 - Documentation: Add Examples to Entry Point Docstrings
**Effort:** 2 hours

Add Example sections to `run_periodogram()`, `calculate_fpp()`, `localize_transit_source()`, `fit_transit()`.

### P3.3 - Documentation: Add Tutorial Notebooks
**Effort:** 8 hours

Create numbered tutorials: `01-basic-vetting.ipynb`, `02-custom-checks.ipynb`, `03-batch-processing.ipynb`.

### P3.4 - UX: Add vet_tic() Convenience Function
**Effort:** 4 hours

One-liner vetting: `result = vet_tic(tic_id=261136679, period=3.5, t0=1850.0, duration_hours=2.5)`

### P3.5 - UX: Make FPP Cache Optional
**Effort:** 4-6 hours

Provide default in-memory caching or `calculate_fpp_simple()` alternative.

### P3.6 - Performance: Reduce First-Access Latency
**Effort:** 4 hours

First access loads 935 modules. Consider splitting `types.py`.

### P3.7 - API: Add Stability Tier Documentation
**Effort:** 2 hours

Document 62 undocumented-but-accessible exports with stability tiers.

### P3.8 - Testing: Add TRICERATOPS Integration Test
**Effort:** 3 hours

End-to-end FPP calculation with known inputs.

### P3.9 - Community: Register with ASCL
**Effort:** 30 minutes

Submit to Astrophysics Source Code Library after first paper.

### P3.10 - Community: Submit to pyOpenSci/JOSS
**Effort:** Ongoing

Submit for peer review after docs complete.

### P3.11 - Cache Path Default (was P1.11)
**Effort:** 1-2 hours

Use `platformdirs` for OS-appropriate cache instead of CWD-relative paths.

---

## Cross-Cutting Themes

1. **Documentation Structure vs. Content** — Excellent docstrings exist but need Sphinx/RTD infrastructure
2. **Two-System Type Architecture** — `LightCurve` vs `LightCurveData`, `catalog.py` vs `catalogs.py` requires documentation
3. **Research vs. Research Library Gap** — Designed for integration; convenience functions would broaden adoption

---

## Recommended Next Steps

1. **Tag v0.1.0** and publish to PyPI
2. **Enable Zenodo DOI** for citation
3. **Create Sphinx docs** (P2.1) for pyOpenSci eligibility
4. **Add tutorial notebooks** (P3.3) for researcher onboarding

---

## Completed Items (Commit 2c2f2e4)

All P0 and P1 items were completed in a single commit. Summary:

### Code Quality (Agent 1)
- ✅ P0.7 - Fixed failing test in `test_tpf_fits.py`
- ✅ P0.8 - Fixed all 115 ruff errors
- ✅ P0.4 - Added CLI smoke tests (`tests/cli/test_cli_smoke.py`)
- ✅ P1.1 - Added exhaustive export test for all 229 API symbols
- ✅ P1.2 - Added `vet_candidate` integration tests (`tests/test_integration/test_vet_candidate_full.py`)

### Dependencies & Licensing (Agent 2)
- ✅ P0.1 - Updated CVE-affected deps: `setuptools>=78.1.1`, `requests>=2.32.4`, `pydantic>=2.4.0`
- ✅ P0.2 - Verified ldtk is GPL-2.0 → made optional to preserve MIT core
- ✅ P0.5 - Created `LICENSE` (MIT)
- ✅ P1.10 - Added LICENSE to vendored triceratops directory
- ✅ P2.6 - Removed duplicate pins from triceratops extras
- ✅ P2.7 - Modernized version floors (pandas, seaborn, mechanicalsoup)
- ✅ P2.12 - Consolidated dev dependencies

### Documentation (Agent 3)
- ✅ P0.9 - Fixed README path drift (`io/` → `platform/io/`)
- ✅ P0.10 - Removed bittr-reason-core references
- ✅ P0.11 - Fixed astro_arc docstring remnants
- ✅ P1.5 - Documented TTV track search functions
- ✅ P1.6 - Generated `REFERENCES.md` from citations registry
- ✅ P1.9 - Archived stale facade documentation
- ✅ P1.12 - Documented platform support in README
- ✅ P2.5 - Added recommended import alias (`import ... as btv`)
- ✅ P2.10 - Clarified domain-only vs platform split

### Infrastructure & Community (Agent 4)
- ✅ P0.3 - Created `.github/workflows/ci.yml` (lint, typecheck, test matrix)
- ✅ P0.6 - Created `CITATION.cff`
- ✅ P1.3 - Created `.pre-commit-config.yaml`
- ✅ P1.4 - Created `.github/dependabot.yml`
- ✅ P1.7 - Created `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`
- ✅ P1.8 - Created GitHub issue/PR templates

---

## Appendix: Research Reports

| # | Report | Key Finding |
|---|--------|-------------|
| 01 | Naming Conflict | `catalog.py` vs `catalogs.py` confusion |
| 02 | Test Coverage | CLI modules had zero coverage |
| 03 | CI/CD Config | Complete absence of automation |
| 04 | Facade Remnants | Stale docs from removed facade |
| 05 | Export Surface | 229 documented + 62 undocumented exports |
| 06 | Docstring Coverage | 93% coverage, missing Sphinx |
| 07 | Competitor Patterns | lightkurve, astropy, exoplanet patterns |
| 08 | OSS Best Practices | pyOpenSci, JOSS, ASCL requirements |
| 09 | Dependency Security | 3 CVEs, GPL license concerns |
| 10 | User Workflow | FPP cache friction, convenience gaps |
| 11 | Codex Review | 1 failing test, 115 ruff errors, README drift |

Full reports in `research/` subdirectory.
