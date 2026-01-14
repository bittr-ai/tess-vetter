# Tranche 2: Documentation & Release Infrastructure

**Goal:** JOSS/pyOpenSci eligibility — Sphinx docs, tutorial notebooks, CI enhancements, release automation.

---

## Agent Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                 TRANCHE 2 (2 PARALLEL AGENTS)                  │
├────────────────────────────────┬───────────────────────────────┤
│   Agent 1: DOCS INFRASTRUCTURE │   Agent 2: TUTORIALS          │
├────────────────────────────────┼───────────────────────────────┤
│ P2.1  Sphinx docs structure    │ P3.3  Tutorial notebooks      │
│ P2.3  Coverage reporting       │       - 01-basic-vetting      │
│ P2.4  Release automation       │       - 02-periodogram        │
│       .readthedocs.yaml        │       - 03-pixel-analysis     │
├────────────────────────────────┼───────────────────────────────┤
│ ~3 hours                       │ ~4 hours                      │
└────────────────────────────────┴───────────────────────────────┘
```

---

## Agent 1: Documentation Infrastructure

**Tasks:**
1. Create `docs/` directory with Sphinx structure:
   - `docs/conf.py` (autodoc + napoleon + intersphinx)
   - `docs/index.rst` (landing page)
   - `docs/installation.rst`
   - `docs/quickstart.rst`
   - `docs/api.rst` (autosummary for all public API)
   - `docs/references.rst` (link to REFERENCES.md)
   - `docs/Makefile` and `docs/make.bat`

2. Create `.readthedocs.yaml` for RTD integration

3. Add coverage reporting to CI:
   - Add pytest-cov to dev dependencies
   - Update `.github/workflows/ci.yml` with coverage
   - Add Codecov integration (or coverage badge)

4. Create `.github/workflows/release.yml`:
   - Trigger on version tags (v*)
   - Build with hatch
   - Publish to PyPI via trusted publishing

---

## Agent 2: Tutorial Notebooks

**Tasks:**
Create `docs/tutorials/` with executable notebooks:

1. `01-basic-vetting.ipynb`
   - Load a light curve (synthetic or small fixture)
   - Create Ephemeris and Candidate
   - Run vet_candidate() with network=False
   - Interpret results

2. `02-periodogram-detection.ipynb`
   - Generate synthetic transit signal
   - Run run_periodogram()
   - Extract best period, visualize

3. `03-pixel-analysis.ipynb`
   - Load TPF data (or mock)
   - Run centroid shift analysis
   - Demonstrate localization

**Guidelines:**
- Use small synthetic data (no large downloads)
- Clear markdown explanations
- Show expected output inline
- Keep execution time < 30 seconds per notebook

---

## Success Criteria

```bash
# Docs build
cd docs && make html  # No errors

# CI passes with coverage
uv run pytest --cov=bittr_tess_vetter

# Notebooks execute
uv run jupyter execute docs/tutorials/*.ipynb

# RTD config valid
cat .readthedocs.yaml  # Properly configured
```
