# Documentation and Release Infrastructure Report

**Date:** 2025-01-14
**Agent:** Documentation Infrastructure Setup
**Status:** Complete

## Summary

This report documents the setup of Sphinx documentation infrastructure, ReadTheDocs configuration, CI/CD coverage reporting, and PyPI release automation for the bittr-tess-vetter package.

## Changes Made

### 1. Sphinx Documentation Structure

Created a complete documentation setup in `docs/`:

| File | Purpose |
|------|---------|
| `conf.py` | Sphinx configuration with autodoc, napoleon, intersphinx |
| `index.rst` | Landing page with quick example |
| `installation.rst` | Installation guide for pip, uv, and extras |
| `quickstart.rst` | User guide with code examples |
| `api.rst` | Comprehensive API reference with autosummary |
| `Makefile` | Unix build commands |
| `make.bat` | Windows build commands |

**conf.py Features:**
- `sphinx.ext.autodoc` - Automatic API documentation
- `sphinx.ext.autosummary` - Summary tables with links
- `sphinx.ext.napoleon` - Google-style docstring support
- `sphinx.ext.intersphinx` - Cross-referencing to numpy, scipy, astropy
- `sphinx.ext.viewcode` - Source code links
- `sphinx-autodoc-typehints` - Type hint rendering
- `myst-parser` - Markdown support
- `furo` theme - Clean, modern appearance
- Metadata extraction from `pyproject.toml`
- Mock imports for optional dependencies (MLX, wotan, ldtk, etc.)

**api.rst Organization:**
- Core Types (Ephemeris, LightCurve, Candidate, etc.)
- Transit Fitting Types
- Activity Types
- Main Entry Point (vet_candidate)
- Periodogram and Detection
- Vetting Checks (LC-only, Catalog, Pixel, Exovetter)
- Pixel Localization
- Transit Fitting, Timing, Activity, Recovery (v3)
- PRF and Pixel Modeling
- False Positive Probability
- Ephemeris Matching
- Utilities and Constants
- MLX Acceleration (optional)

### 2. ReadTheDocs Configuration

Created `.readthedocs.yaml`:
- Python 3.12
- Ubuntu 22.04 build environment
- Uses uv for fast dependency installation
- Sphinx HTML builder
- PDF generation enabled
- Installs package with `[docs]` extras

### 3. CI/CD Coverage Reporting

Updated `.github/workflows/ci.yml`:

**Test Job Enhancements:**
- Added pytest-cov installation
- Coverage reporting with XML and terminal output
- Coverage upload to Codecov (Python 3.12 only)
- Uses `--cov=bittr_tess_vetter --cov-report=xml --cov-report=term-missing`

**New Docs Job:**
- Builds documentation on every push/PR
- Uploads built docs as artifact
- Validates documentation integrity

### 4. Release Workflow

Created `.github/workflows/release.yml`:

**Trigger:** Tags matching `v*`

**Jobs:**
1. **build** - Build package with hatch
2. **test-build** - Verify wheel installs and imports work (Python 3.11, 3.12)
3. **publish-pypi** - Publish to PyPI using trusted publishing (OIDC)
4. **github-release** - Create GitHub release with auto-generated notes
5. **publish-testpypi** - Pre-release versions go to TestPyPI

**Security:**
- Uses OIDC trusted publishing (no API tokens needed)
- Requires `pypi` environment approval
- Pre-release detection via tag pattern matching

### 5. pyproject.toml Updates

Added `docs` optional dependency group:
```toml
docs = [
  "sphinx>=7.0",
  "furo",
  "sphinx-autodoc-typehints",
  "myst-parser",
]
```

## Build Verification

**Documentation Build:**
```bash
cd docs && uv run sphinx-build -b html . _build/html
```

**Results:**
- Build completes successfully
- 144 autosummary pages generated
- HTML output in `docs/_build/html/`
- Cross-references to numpy, scipy, astropy enabled
- Warnings are primarily cross-reference issues (expected for complex APIs)

**Generated Documentation Includes:**
- Full API reference with 144+ documented symbols
- Source code links via viewcode
- Search functionality
- Mobile-responsive design (furo theme)

## Files Created/Modified

### Created Files
| Path | Description |
|------|-------------|
| `docs/conf.py` | Sphinx configuration |
| `docs/index.rst` | Documentation landing page |
| `docs/installation.rst` | Installation guide |
| `docs/quickstart.rst` | Quick start guide |
| `docs/api.rst` | API reference |
| `docs/Makefile` | Unix build commands |
| `docs/make.bat` | Windows build commands |
| `docs/_static/` | Static assets directory |
| `docs/_templates/` | Custom templates directory |
| `.readthedocs.yaml` | RTD configuration |
| `.github/workflows/release.yml` | Release automation |

### Modified Files
| Path | Description |
|------|-------------|
| `.github/workflows/ci.yml` | Added coverage + docs build |
| `pyproject.toml` | Added docs dependencies |

## JOSS/pyOpenSci Eligibility Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| Sphinx docs | Done | Full API reference with autosummary |
| Installation guide | Done | pip, uv, extras documented |
| Quick start | Done | Code examples included |
| API reference | Done | 144+ symbols documented |
| CI testing | Done | Coverage reporting added |
| Coverage tracking | Done | Codecov integration |
| Release automation | Done | PyPI trusted publishing |
| ReadTheDocs | Done | Configuration ready |

## Next Steps

1. **Enable ReadTheDocs:** Connect repository at readthedocs.org
2. **Configure Codecov:** Add `CODECOV_TOKEN` secret if using private repos
3. **Configure PyPI trusted publishing:** Set up at pypi.org/manage/project/.../settings/publishing/
4. **Add coverage badge:** Add badge to README once Codecov is configured
5. **Fix warnings (optional):** Address cross-reference warnings for cleaner builds

## Usage

**Build docs locally:**
```bash
uv sync --extra docs
cd docs && make html
# Open _build/html/index.html
```

**Create a release:**
```bash
git tag v0.1.0
git push origin v0.1.0
# Workflow automatically builds and publishes
```

**Run tests with coverage:**
```bash
uv pip install pytest-cov
uv run pytest --cov=bittr_tess_vetter --cov-report=html
# Open htmlcov/index.html
```
