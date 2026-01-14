# Open-Source Release Priorities: bittr-tess-vetter
*Consolidated Architecture Review - 2026-01-14 (v3)*

---

## Executive Summary

**bittr-tess-vetter is ready for public release and JOSS/pyOpenSci submission.**

Two tranches of work completed:
- **Tranche 1** (`2c2f2e4`): All P0/P1 blockers — tests, linting, CI, licensing, community files
- **Tranche 2** (`75b3fe3`): Documentation infrastructure — Sphinx, tutorials, release automation

The repository now has:
- ✅ All tests passing, all linting clean
- ✅ CI/CD with coverage reporting and PyPI release automation
- ✅ Sphinx documentation with full API reference (229 exports)
- ✅ 3 tutorial notebooks (basic vetting, periodogram, pixel analysis)
- ✅ ReadTheDocs configuration
- ✅ MIT license with GPL dependencies made optional

**Remaining work is minor polish** — ready to tag v0.1.0.

---

## Recommended Next Steps

1. **Push to GitHub** and enable ReadTheDocs
2. **Tag `v0.1.0`** to trigger PyPI release
3. **Enable Zenodo** for DOI citation
4. **Submit to pyOpenSci/JOSS** for peer review

---

## Remaining Items (P2/P3 - Optional Polish)

### P2.2 - Testing: Add Edge Case Tests
**Effort:** 4 hours

Add parametrized tests for boundary conditions (NaN, empty, single-element arrays).

### P2.8 - API: Add GPL Notice for triceratops Extras
**Effort:** 15 minutes

Document that `[triceratops]` includes `pytransit` (GPL-2.0).

### P2.9 - UX: Add Warning When Checks Are Skipped
**Effort:** 2 hours

When `network=False` or metadata missing, include warnings explaining skipped checks.

### P2.11 - Define Minimal Install Path
**Effort:** 2-4 hours

Consider minimal core + extras pattern for lighter installs.

### P3.1 - Naming: Rename catalog.py to catalog_checks.py
**Effort:** 2 hours

Address `catalog.py` vs `catalogs.py` confusion with deprecation wrapper.

### P3.2 - Documentation: Add Examples to Entry Point Docstrings
**Effort:** 2 hours

Add Example sections to key functions.

### P3.4 - UX: Add vet_tic() Convenience Function
**Effort:** 4 hours

One-liner vetting for researchers with known TIC ID.

### P3.5 - UX: Make FPP Cache Optional
**Effort:** 4-6 hours

Provide default in-memory caching for `calculate_fpp()`.

### P3.6 - Performance: Reduce First-Access Latency
**Effort:** 4 hours

First access loads 935 modules. Consider splitting `types.py`.

### P3.7 - API: Add Stability Tier Documentation
**Effort:** 2 hours

Document stability guarantees for exports.

### P3.8 - Testing: Add TRICERATOPS Integration Test
**Effort:** 3 hours

End-to-end FPP calculation test.

### P3.9 - Community: Register with ASCL
**Effort:** 30 minutes

Submit to Astrophysics Source Code Library after first paper.

### P3.10 - Community: Submit to pyOpenSci/JOSS
**Effort:** Ongoing

Now eligible — Sphinx docs and tutorials complete.

### P3.11 - Cache Path Default
**Effort:** 1-2 hours

Use `platformdirs` for OS-appropriate cache paths.

---

## Completed Items

### Tranche 2 (Commit `75b3fe3`)

Documentation & release infrastructure:

- ✅ P2.1 - Created Sphinx documentation (`docs/`) with furo theme, autosummary for all 229 API exports
- ✅ P2.3 - Added pytest-cov and Codecov integration to CI
- ✅ P2.4 - Created `.github/workflows/release.yml` for PyPI trusted publishing
- ✅ P3.3 - Created 3 tutorial notebooks:
  - `01-basic-vetting.ipynb` — Core vetting workflow
  - `02-periodogram-detection.ipynb` — Transit detection with TLS/BLS
  - `03-pixel-analysis.ipynb` — Pixel-level diagnostics
- ✅ Created `.readthedocs.yaml` for RTD integration
- ✅ Added `docs` optional dependency group

### Tranche 1 (Commit `2c2f2e4`)

All P0 critical blockers and P1 high-priority items:

**Code Quality:**
- ✅ P0.7 - Fixed failing test in `test_tpf_fits.py`
- ✅ P0.8 - Fixed all 115 ruff errors
- ✅ P0.4 - Added CLI smoke tests
- ✅ P1.1 - Added exhaustive export test for all 229 API symbols
- ✅ P1.2 - Added `vet_candidate` integration tests

**Dependencies & Licensing:**
- ✅ P0.1 - Updated CVE-affected deps (setuptools, requests, pydantic)
- ✅ P0.2 - Made ldtk optional (GPL-2.0 incompatible with MIT core)
- ✅ P0.5 - Created `LICENSE` (MIT)
- ✅ P1.10 - Added LICENSE to vendored triceratops
- ✅ P2.6, P2.7, P2.12 - Cleaned up pyproject.toml

**Documentation:**
- ✅ P0.9 - Fixed README path drift
- ✅ P0.10 - Removed bittr-reason-core references
- ✅ P0.11 - Fixed astro_arc docstring remnants
- ✅ P1.5 - Documented TTV track search functions
- ✅ P1.6 - Generated `REFERENCES.md`
- ✅ P1.9 - Archived stale facade documentation
- ✅ P1.12, P2.5, P2.10 - README improvements

**Infrastructure & Community:**
- ✅ P0.3 - Created `.github/workflows/ci.yml`
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
